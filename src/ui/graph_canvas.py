"""
Interactive Vector Graph Canvas for GemGraph using pyqtgraph.
Derived from BridgeCLR GraphController contracts (REA-CLR-VM-102).
Includes:
- Publication-grade axis frames (bold baselines, outward ticks, clean labels)
- Superimposed plots (bars + lines/scatter points)
- Live Results Embedding (linked analysis summary tables on canvas)
- High-contrast grayscale journal palette defaults and palette switching
- Constant Hooking rendering (tick lines e.g. Mean ± 2SD, dynamic text labels)
"""

from __future__ import annotations
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QComboBox,
    QLabel,
    QTableWidget,
    QGraphicsProxyWidget,
)
import pyqtgraph as pg
import pyqtgraph.exporters
from src.core.table import DataTable, TableType
from src.core.sheet import GraphSheet, GraphType, ResultSheet
from src.core.hooking import GLOBAL_REGISTRY, HookedConstant


class GraphCanvasWidget(QWidget):
    """
    High-performance scientific plot widget rendering publication-grade vector graphs.
    """

    # Distinct scientific palette matching Prism defaults
    COLOR_PALETTE = [
        QColor(0, 114, 178),    # Blue
        QColor(213, 94, 0),     # Vermilion
        QColor(0, 158, 115),    # Bluish green
        QColor(204, 121, 167),  # Reddish purple
        QColor(230, 159, 0),    # Orange
        QColor(86, 180, 233),   # Sky blue
    ]

    # High-contrast grayscale journal palette
    GRAYSCALE_PALETTE = [
        QColor(0, 0, 0),        # Solid Black
        QColor(70, 70, 70),     # Dark Gray
        QColor(130, 130, 130),  # Medium Gray
        QColor(190, 190, 190),  # Light Gray
    ]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.plot_widget = pg.PlotWidget()
        self.current_palette = self.COLOR_PALETTE
        self.hooked_lines: List[Dict[str, Any]] = []
        self.hooked_labels: List[Dict[str, Any]] = []
        self.embedded_tables: List[Dict[str, Any]] = []

        self._setup_plot_style()

        # Canvas toolbar
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("Graph Type:"))
        self.type_combo = QComboBox()
        self.type_combo.addItems([
            GraphType.XY_SCATTER_CURVE.value,
            GraphType.COLUMN_BAR_ERROR.value,
            "Superimposed Bars & Points",
        ])
        self.type_combo.currentIndexChanged.connect(self._on_type_changed)
        top_bar.addWidget(self.type_combo)

        top_bar.addWidget(QLabel("Palette:"))
        self.palette_combo = QComboBox()
        self.palette_combo.addItems(["Prism Standard Color", "Journal Grayscale"])
        self.palette_combo.currentIndexChanged.connect(self._on_palette_changed)
        top_bar.addWidget(self.palette_combo)

        top_bar.addStretch()

        self.btn_autoscale = QPushButton("Auto Range")
        self.btn_autoscale.clicked.connect(self.plot_widget.autoRange)
        top_bar.addWidget(self.btn_autoscale)

        self.layout.addLayout(top_bar)
        self.layout.addWidget(self.plot_widget)

        self.current_table: Optional[DataTable] = None
        self.current_result: Optional[ResultSheet] = None

    def _setup_plot_style(self) -> None:
        """Configures publication-grade white-background scientific styling with outward ticks."""
        self.plot_widget.setBackground("w")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.15)

        # Bold black baselines (width=2.0)
        axis_pen = pg.mkPen(color="k", width=2.0)
        font = QFont("Arial", 10)

        for axis_name in ("left", "bottom", "top", "right"):
            ax = self.plot_widget.getAxis(axis_name)
            ax.setPen(axis_pen)
            ax.setTextPen("k")
            ax.setStyle(tickFont=font, tickLength=-6)  # Outward ticks

        self.plot_widget.addLegend(offset=(30, 30))

    def _on_palette_changed(self) -> None:
        if self.palette_combo.currentIndex() == 1:
            self.current_palette = self.GRAYSCALE_PALETTE
        else:
            self.current_palette = self.COLOR_PALETTE
        self.refresh_plot()

    def _on_type_changed(self) -> None:
        self.refresh_plot()

    def add_hooked_line(
        self,
        constant_id: str,
        orientation: str = "horizontal",
        pen: Optional[Any] = None,
        label: Optional[str] = None,
    ) -> None:
        """Binds a dynamic tick line to a registered constant (e.g. Mean ± 2SD or EC50)."""
        self.hooked_lines.append({
            "constant_id": constant_id,
            "orientation": orientation,
            "pen": pen or pg.mkPen(color=QColor(220, 50, 50), width=1.5, style=Qt.PenStyle.DashLine),
            "label": label,
        })
        self.refresh_plot()

    def add_hooked_label(
        self,
        constant_id: str,
        template: str = "{name} = {val}",
        x: float = 0.05,
        y: float = 0.95,
    ) -> None:
        """Binds a dynamic text label on plot scene to a constant."""
        self.hooked_labels.append({
            "constant_id": constant_id,
            "template": template,
            "x": x,
            "y": y,
        })
        self.refresh_plot()

    def embed_summary_table(self, title: str, headers: List[str], rows: List[List[Any]]) -> None:
        """Embeds a live-linked summary table onto the plot canvas."""
        self.embedded_tables.append({
            "title": title,
            "headers": headers,
            "rows": rows,
        })
        self.refresh_plot()

    def update_data(
        self,
        table: Optional[DataTable],
        result: Optional[ResultSheet] = None,
        graph_type: Optional[GraphType] = None,
    ) -> None:
        """Updates referenced table and result sheet, then repaints."""
        self.current_table = table
        self.current_result = result
        if graph_type:
            idx = self.type_combo.findText(graph_type.value)
            if idx >= 0:
                self.type_combo.setCurrentIndex(idx)
        self.refresh_plot()

    def refresh_plot(self) -> None:
        """Clears and redraws graph according to current data, hooked elements, and settings."""
        self.plot_widget.clear()
        if self.plot_widget.plotItem.legend is not None:
            self.plot_widget.plotItem.legend.clear()

        if self.current_table is None:
            return

        selected_type = self.type_combo.currentText()
        if selected_type == "Superimposed Bars & Points":
            self._render_superimposed_graph()
        elif selected_type == GraphType.COLUMN_BAR_ERROR.value or self.current_table.table_type == TableType.COLUMN:
            self._render_column_graph()
        else:
            self._render_xy_graph()

        # Render Hooked Lines
        self._render_hooked_elements()

        # Render Embedded Tables
        self._render_embedded_tables()

    def _render_xy_graph(self) -> None:
        """Renders XY points and fitted curve overlay."""
        table = self.current_table
        if table is None or table.x_data is None:
            return

        self.plot_widget.setLabel("bottom", table.x_title or "X", color="black")
        self.plot_widget.setLabel("left", "Y Value", color="black")

        for g_idx, g_title in enumerate(table.y_group_titles):
            color = self.current_palette[g_idx % len(self.current_palette)]
            xs, ys = table.get_clean_xy(g_idx)
            if len(xs) == 0:
                continue

            scatter = pg.ScatterPlotItem(
                x=xs,
                y=ys,
                pen=pg.mkPen(color, width=1),
                brush=pg.mkBrush(color),
                size=8,
                name=g_title,
            )
            self.plot_widget.addItem(scatter)

        # Fitted curve overlay
        if self.current_result is not None:
            raw = self.current_result.raw_result_data
            if "x_dense" in raw and "y_dense" in raw:
                x_dense = np.array(raw["x_dense"])
                y_dense = np.array(raw["y_dense"])
                fit_pen = pg.mkPen(color=QColor(20, 20, 20), width=2.5)
                self.plot_widget.plot(
                    x_dense,
                    y_dense,
                    pen=fit_pen,
                    name=raw.get("model_name", "Fitted Curve"),
                )

        self.plot_widget.autoRange()

    def _render_column_graph(self) -> None:
        """Renders column bars with Mean ± SEM/SD and overlaid scatter points."""
        table = self.current_table
        if table is None:
            return

        error_label = "SD" if getattr(self, "error_style", "SEM") == "SD" else "SEM"
        self.plot_widget.setLabel("bottom", "Groups", color="black")
        self.plot_widget.setLabel("left", f"Value (Mean ± {error_label})", color="black")

        group_data = table.get_all_group_values()
        names = list(group_data.keys())
        x_positions = np.arange(len(names))

        ticks = [(pos, name) for pos, name in zip(x_positions, names)]
        ax = self.plot_widget.getAxis("bottom")
        ax.setTicks([ticks])

        for pos, name in zip(x_positions, names):
            vals = np.array(group_data[name], dtype=np.float64)
            color = self.current_palette[pos % len(self.current_palette)]
            if len(vals) == 0:
                continue

            mean = float(np.mean(vals))
            if len(vals) > 1:
                sd = float(np.std(vals, ddof=1))
                err_val = sd if error_label == "SD" else sd / np.sqrt(len(vals))
            else:
                err_val = 0.0

            # Bar item
            bar = pg.BarGraphItem(
                x=[pos],
                height=[mean],
                width=0.6,
                brush=pg.mkBrush(color.red(), color.green(), color.blue(), 140),
                pen=pg.mkPen(color, width=1.5),
            )
            self.plot_widget.addItem(bar)

            # Error bar
            if err_val > 0:
                err = pg.ErrorBarItem(
                    x=np.array([pos]),
                    y=np.array([mean]),
                    top=np.array([err_val]),
                    bottom=np.array([err_val]),
                    beam=0.2,
                    pen=pg.mkPen("k", width=1.5),
                )
                self.plot_widget.addItem(err)

            # Scatter points
            jitter = (np.random.RandomState(42).rand(len(vals)) - 0.5) * 0.25
            scatter = pg.ScatterPlotItem(
                x=pos + jitter,
                y=vals,
                pen=pg.mkPen("k", width=1),
                brush=pg.mkBrush("w"),
                size=6,
            )
            self.plot_widget.addItem(scatter)

        self.plot_widget.autoRange()

    def _render_superimposed_graph(self) -> None:
        """Combines bars and scatter/line plots on the same canvas."""
        self._render_column_graph()
        # Connect means with trend line
        table = self.current_table
        if table is None:
            return
        group_data = table.get_all_group_values()
        names = list(group_data.keys())
        x_positions = np.arange(len(names))
        means = [float(np.mean(group_data[n])) if group_data[n] else np.nan for n in names]

        valid_mask = ~np.isnan(means)
        if np.sum(valid_mask) >= 2:
            trend_pen = pg.mkPen(color=QColor(0, 0, 0), width=2, style=Qt.PenStyle.DashLine)
            self.plot_widget.plot(
                x_positions[valid_mask],
                np.array(means)[valid_mask],
                pen=trend_pen,
                name="Group Mean Trend",
            )

    def _render_hooked_elements(self) -> None:
        """Renders dynamic tick lines and dynamic text equations from hooked constants."""
        for line in self.hooked_lines:
            cid = line["constant_id"]
            const = GLOBAL_REGISTRY.get(cid)
            if const is not None:
                is_vert = line.get("orientation") == "vertical"
                inf_line = pg.InfiniteLine(
                    pos=const.value,
                    angle=90 if is_vert else 0,
                    pen=line["pen"],
                    label=line.get("label") or f"{const.name}: {const.formatted_value}",
                    labelOpts={"position": 0.85, "color": (50, 50, 50)},
                )
                self.plot_widget.addItem(inf_line)

        # Dynamic text labels
        for idx, lbl in enumerate(self.hooked_labels):
            cid = lbl["constant_id"]
            const = GLOBAL_REGISTRY.get(cid)
            if const is not None:
                text_content = lbl["template"].format(
                    name=const.name,
                    val=const.formatted_value,
                    value=const.formatted_value,
                )
                text_item = pg.TextItem(text_content, color=(20, 20, 20), anchor=(0, 0))
                font = QFont("Arial", 10, QFont.Weight.Bold)
                text_item.setFont(font)
                # Place in view
                view_range = self.plot_widget.viewRange()
                x_range = view_range[0]
                y_range = view_range[1]
                px = x_range[0] + lbl["x"] * (x_range[1] - x_range[0])
                py = y_range[0] + (lbl["y"] - idx * 0.08) * (y_range[1] - y_range[0])
                text_item.setPos(px, py)
                self.plot_widget.addItem(text_item)

    def _render_embedded_tables(self) -> None:
        """Renders embedded summary table overlay on canvas."""
        for tbl in self.embedded_tables:
            title = tbl.get("title", "Results Summary")
            headers = tbl.get("headers", [])
            rows = tbl.get("rows", [])
            # Build text representation
            lines = [f"<b>{title}</b>"]
            for row in rows[:6]:
                row_str = " | ".join(str(c) for c in row)
                lines.append(row_str)
            box_text = "<br>".join(lines)
            text_item = pg.TextItem(html=box_text, anchor=(1, 0), border="k", fill=(255, 255, 255, 220))
            view_range = self.plot_widget.viewRange()
            x_pos = view_range[0][1] - 0.05 * (view_range[0][1] - view_range[0][0])
            y_pos = view_range[1][1] - 0.05 * (view_range[1][1] - view_range[1][0])
            text_item.setPos(x_pos, y_pos)
            self.plot_widget.addItem(text_item)

    def export_png(self, file_path: str) -> None:
        """Exports plot to PNG image."""
        exporter = pg.exporters.ImageExporter(self.plot_widget.plotItem)
        exporter.export(file_path)

    def export_svg(self, file_path: str) -> None:
        """Exports plot to vector SVG file using direct QtSvg generator."""
        from PyQt6.QtSvg import QSvgGenerator
        from PyQt6.QtGui import QPainter

        generator = QSvgGenerator()
        generator.setFileName(file_path)
        generator.setSize(self.plot_widget.size())
        generator.setViewBox(self.plot_widget.rect())
        generator.setTitle(self.current_table.title if self.current_table else "GemGraph Plot")
        generator.setDescription("Generated by GemGraph Clean-Room Edition")

        painter = QPainter(generator)
        try:
            self.plot_widget.render(painter)
        finally:
            painter.end()
