"""
Interactive Vector Graph Canvas for GemGraph using pyqtgraph.
Derived from BridgeCLR GraphController contracts (REA-CLR-VM-102).
Includes:
- Publication-grade axis frames (bold baselines, outward ticks, clean labels)
- Superimposed plots (bars + lines/scatter points)
- Live Results Embedding (linked analysis summary tables on canvas)
- High-contrast grayscale journal palette defaults and palette switching
- Constant Hooking rendering (tick lines e.g. Mean ± 2SD, dynamic text labels)
- Active Analysis Overlay Registry & Layer Clearing UI
- Standardized journal publishing marker shapes with alternating solid/open fills
"""

from __future__ import annotations
from typing import Optional, List, Dict, Any, Tuple, Set
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QToolButton,
    QMenu,
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
from src.ui.mediator import WorkspaceMediator


class ScientificPlotCanvas(QWidget):
    """
    High-performance scientific plot widget rendering publication-grade vector graphs.
    Includes active analysis overlay registry, toolbar layer clearing UI,
    and journal publishing grayscale symbology.
    """

    # Expanded distinct scientific palette
    COLOR_PALETTE = [
        QColor(0, 0, 0),         # Black
        QColor(0, 114, 178),     # Blue
        QColor(220, 20, 60),     # Crimson
        QColor(34, 139, 34),     # Forest Green
        QColor(255, 140, 0),     # Dark Orange
        QColor(128, 0, 128),     # Purple
        QColor(0, 158, 115),     # Bluish green
        QColor(204, 121, 167),   # Reddish purple
        QColor(86, 180, 233),    # Sky blue
    ]

    # Standardized publishing marker shapes with alternating solid/open fills
    GRAYSCALE_SYMBOLS = [
        {
            "symbol": "o",
            "brush": pg.mkBrush(0, 0, 0),
            "pen": pg.mkPen(color=QColor(0, 0, 0), width=1.0),
            "name": "Solid Circle",
        },
        {
            "symbol": "s",
            "brush": pg.mkBrush(255, 255, 255),
            "pen": pg.mkPen(color=QColor(0, 0, 0), width=1.5),
            "name": "Open Square",
        },
        {
            "symbol": "t",
            "brush": pg.mkBrush(0, 0, 0),
            "pen": pg.mkPen(color=QColor(0, 0, 0), width=1.0),
            "name": "Solid Triangle",
        },
        {
            "symbol": "d",
            "brush": pg.mkBrush(255, 255, 255),
            "pen": pg.mkPen(color=QColor(0, 0, 0), width=1.5),
            "name": "Open Diamond",
        },
        {
            "symbol": "t1",
            "brush": pg.mkBrush(0, 0, 0),
            "pen": pg.mkPen(color=QColor(0, 0, 0), width=1.0),
            "name": "Solid Inverted Triangle",
        },
        {
            "symbol": "o",
            "brush": pg.mkBrush(255, 255, 255),
            "pen": pg.mkPen(color=QColor(0, 0, 0), width=1.5),
            "name": "Open Circle",
        },
    ]

    # Backward-compatible grayscale color list
    GRAYSCALE_PALETTE = [
        QColor(0, 0, 0),        # Solid Black
        QColor(70, 70, 70),     # Dark Gray
        QColor(130, 130, 130),  # Medium Gray
        QColor(190, 190, 190),  # Light Gray
    ]

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        mediator: Optional[WorkspaceMediator] = None,
    ) -> None:
        super().__init__(parent)
        self.mediator = mediator or WorkspaceMediator.get_instance()
        self.plot_widget = pg.PlotWidget()
        self.current_palette = self.COLOR_PALETTE

        # Active overlay registry: analysis_id -> { 'name': str, 'plot_items': list }
        self.analysis_layers: Dict[str, Dict[str, Any]] = {}
        self.custom_analysis_layers: Dict[str, Dict[str, Any]] = {}
        self.removed_layer_ids: Set[str] = set()

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
        self.palette_combo.addItems([
            "🎨 Palette: Color",
            "📄 Palette: Journal Grayscale",
        ])
        self.palette_combo.currentIndexChanged.connect(self._on_palette_changed)
        top_bar.addWidget(self.palette_combo)

        top_bar.addStretch()

        # Toolbar menu button: Clear Analysis
        self.btn_clear_analysis = QToolButton()
        self.btn_clear_analysis.setText("🧹 Clear Analysis")
        self.analysis_menu = QMenu(self)
        self.btn_clear_analysis.setMenu(self.analysis_menu)
        self.btn_clear_analysis.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.analysis_menu.aboutToShow.connect(self._rebuild_analysis_menu)
        top_bar.addWidget(self.btn_clear_analysis)

        self.btn_autoscale = QPushButton("Auto Range")
        self.btn_autoscale.clicked.connect(self.plot_widget.autoRange)
        top_bar.addWidget(self.btn_autoscale)

        self.layout.addLayout(top_bar)
        self.layout.addWidget(self.plot_widget)

        self.current_table: Optional[DataTable] = None
        self.current_result: Optional[ResultSheet] = None
        self._rebuild_analysis_menu()

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

    @property
    def is_grayscale_mode(self) -> bool:
        """Returns True if the canvas is currently in Journal Grayscale palette mode."""
        return (
            self.palette_combo.currentIndex() == 1
            or "Grayscale" in self.palette_combo.currentText()
        )

    def set_grayscale_mode(self, enabled: bool = True) -> None:
        """Selects between Color and Journal Grayscale mode."""
        idx = 1 if enabled else 0
        self.palette_combo.setCurrentIndex(idx)

    def _on_palette_changed(self) -> None:
        if self.is_grayscale_mode:
            self.current_palette = self.GRAYSCALE_PALETTE
        else:
            self.current_palette = self.COLOR_PALETTE
        self.refresh_plot()

    def _on_type_changed(self) -> None:
        self.refresh_plot()

    def _rebuild_analysis_menu(self) -> None:
        """Rebuilds the Clear Analysis toolbar popup menu."""
        self.analysis_menu.clear()

        clear_all_act = self.analysis_menu.addAction("Clear All Fitted Curves")
        clear_all_act.triggered.connect(self.clear_all_fitted_curves)

        if not self.analysis_layers:
            clear_all_act.setEnabled(False)
            no_act = self.analysis_menu.addAction("No Active Overlays")
            no_act.setEnabled(False)
            return

        clear_all_act.setEnabled(True)
        self.analysis_menu.addSeparator()

        for layer_id, layer_info in list(self.analysis_layers.items()):
            name = layer_info.get("name", layer_id)
            act_text = name if name.startswith("Remove ") else f"Remove {name}"
            act = self.analysis_menu.addAction(act_text)
            act.triggered.connect(
                lambda checked=False, lid=layer_id: self.remove_analysis_layer(lid)
            )

    def add_analysis_layer(self, layer_id: str, name: str, plot_items: list) -> None:
        """Registers active plot items into the analysis overlay registry."""
        self.analysis_layers[layer_id] = {
            "name": name,
            "plot_items": list(plot_items),
        }
        self.removed_layer_ids.discard(layer_id)
        self._rebuild_analysis_menu()

    def add_fitted_curve(
        self,
        analysis_id: str,
        name: str,
        x_dense: np.ndarray,
        y_dense: np.ndarray,
        pen: Optional[Any] = None,
    ) -> pg.PlotDataItem:
        """Adds a continuous fitted curve to the canvas and registers in analysis_layers."""
        if pen is None:
            pen = pg.mkPen(color=QColor(20, 20, 20), width=2.5)

        self.custom_analysis_layers[analysis_id] = {
            "name": name,
            "x_dense": np.array(x_dense),
            "y_dense": np.array(y_dense),
            "pen": pen,
        }
        self.removed_layer_ids.discard(analysis_id)

        curve_item = self.plot_widget.plot(
            x_dense,
            y_dense,
            pen=pen,
            name=name,
        )
        self.add_analysis_layer(analysis_id, name, [curve_item])
        return curve_item

    def remove_analysis_layer(self, layer_id: str) -> bool:
        """
        Removes the corresponding items from self.plot_widget.plotItem via removeItem().
        Deletes the key from self.analysis_layers.
        Restores the raw dataset scatter view without resetting user zoom or axis ranges.
        """
        if layer_id not in self.analysis_layers:
            return False

        layer = self.analysis_layers.pop(layer_id)
        self.removed_layer_ids.add(layer_id)
        self.custom_analysis_layers.pop(layer_id, None)

        plot_item = self.plot_widget.plotItem
        for item in layer.get("plot_items", []):
            try:
                plot_item.removeItem(item)
            except Exception:
                try:
                    self.plot_widget.removeItem(item)
                except Exception:
                    pass
            if plot_item.legend is not None:
                try:
                    plot_item.legend.removeItem(item)
                except Exception:
                    pass

        self._rebuild_analysis_menu()

        if self.mediator is not None:
            self.mediator.notify_analysis_layer_removed(layer_id)

        return True

    def clear_all_analysis_layers(self) -> None:
        """Removes all active fitted curves and statistical overlays."""
        for layer_id in list(self.analysis_layers.keys()):
            self.remove_analysis_layer(layer_id)
        if self.mediator is not None:
            self.mediator.notify_clear_all_analysis_layers()

    def clear_all_fitted_curves(self) -> None:
        """Top-level action: clears all fitted curves."""
        self.clear_all_analysis_layers()

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
        if result is not None and (
            self.current_result is None
            or getattr(result, "sheet_id", None) != getattr(self.current_result, "sheet_id", None)
        ):
            self.removed_layer_ids.discard(getattr(result, "sheet_id", ""))

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

        # Reset active overlay item references before redrawing
        self.analysis_layers.clear()

        if self.current_table is None:
            self._rebuild_analysis_menu()
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
        """Renders XY points and fitted curve overlay with discrete colors or journal grayscale symbology."""
        table = self.current_table
        if table is None or table.x_data is None:
            return

        self.plot_widget.setLabel("bottom", table.x_title or "X", color="black")
        self.plot_widget.setLabel("left", "Y Value", color="black")

        y_datasets_dict = getattr(table, "y_datasets_dict", None)
        if y_datasets_dict is None:
            y_datasets_dict = {}
            for g_idx, g_title in enumerate(table.y_group_titles):
                y_datasets_dict[g_title] = table.get_clean_xy(g_idx)

        is_grayscale = self.is_grayscale_mode

        for col_idx, (g_title, (xs, ys)) in enumerate(y_datasets_dict.items()):
            if len(xs) == 0:
                continue

            if is_grayscale:
                sym_spec = self.GRAYSCALE_SYMBOLS[col_idx % len(self.GRAYSCALE_SYMBOLS)]
                symbol = sym_spec["symbol"]
                pen = sym_spec["pen"]
                brush = sym_spec["brush"]
            else:
                color = self.COLOR_PALETTE[col_idx % len(self.COLOR_PALETTE)]
                symbol = "o"
                pen = pg.mkPen(color, width=1.0)
                brush = pg.mkBrush(color)

            scatter = pg.ScatterPlotItem(
                x=xs,
                y=ys,
                pen=pen,
                brush=brush,
                symbol=symbol,
                size=8,
                name=g_title,
            )
            scatter.opts["pen"] = pen
            scatter.opts["brush"] = brush
            scatter.opts["symbol"] = symbol
            scatter.opts["size"] = 8
            scatter.opts["name"] = g_title

            self.plot_widget.addItem(scatter)

        # Custom registered continuous analysis curves
        for layer_id, layer_data in list(self.custom_analysis_layers.items()):
            if layer_id not in self.removed_layer_ids:
                x_d = layer_data["x_dense"]
                y_d = layer_data["y_dense"]
                p_item = self.plot_widget.plot(
                    x_d,
                    y_d,
                    pen=layer_data.get("pen", pg.mkPen(color=QColor(20, 20, 20), width=2.5)),
                    name=layer_data.get("name", "Fitted Curve"),
                )
                self.analysis_layers[layer_id] = {
                    "name": layer_data.get("name", "Fitted Curve"),
                    "plot_items": [p_item],
                }

        # Result sheet fitted curve overlay
        if self.current_result is not None:
            res_id = getattr(self.current_result, "sheet_id", "active_fit")
            if res_id not in self.removed_layer_ids:
                raw = self.current_result.raw_result_data
                if "x_dense" in raw and "y_dense" in raw:
                    x_dense = np.array(raw["x_dense"])
                    y_dense = np.array(raw["y_dense"])
                    fit_pen = pg.mkPen(color=QColor(20, 20, 20), width=2.5)
                    name = raw.get("model_name", getattr(self.current_result, "title", "Fitted Curve"))
                    curve_item = self.plot_widget.plot(
                        x_dense,
                        y_dense,
                        pen=fit_pen,
                        name=name,
                    )
                    self.analysis_layers[res_id] = {
                        "name": name,
                        "plot_items": [curve_item],
                    }

        self._rebuild_analysis_menu()
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

        is_grayscale = self.is_grayscale_mode

        for pos, name in zip(x_positions, names):
            vals = np.array(group_data[name], dtype=np.float64)
            if len(vals) == 0:
                continue

            if is_grayscale:
                sym_spec = self.GRAYSCALE_SYMBOLS[pos % len(self.GRAYSCALE_SYMBOLS)]
                bar_pen = pg.mkPen("k", width=1.5)
                bar_brush = pg.mkBrush(230, 230, 230, 160)
                pt_symbol = sym_spec["symbol"]
                pt_pen = sym_spec["pen"]
                pt_brush = sym_spec["brush"]
            else:
                color = self.COLOR_PALETTE[pos % len(self.COLOR_PALETTE)]
                bar_pen = pg.mkPen(color, width=1.5)
                bar_brush = pg.mkBrush(color.red(), color.green(), color.blue(), 140)
                pt_symbol = "o"
                pt_pen = pg.mkPen("k", width=1)
                pt_brush = pg.mkBrush("w")

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
                brush=bar_brush,
                pen=bar_pen,
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
                pen=pt_pen,
                brush=pt_brush,
                symbol=pt_symbol,
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
        trend_id = "group_mean_trend"
        if np.sum(valid_mask) >= 2 and trend_id not in self.removed_layer_ids:
            trend_pen = pg.mkPen(color=QColor(0, 0, 0), width=2, style=Qt.PenStyle.DashLine)
            trend_item = self.plot_widget.plot(
                x_positions[valid_mask],
                np.array(means)[valid_mask],
                pen=trend_pen,
                name="Group Mean Trend",
            )
            self.analysis_layers[trend_id] = {
                "name": "Group Mean Trend",
                "plot_items": [trend_item],
            }
            self._rebuild_analysis_menu()

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


# Alias for backward compatibility
GraphCanvasWidget = ScientificPlotCanvas
