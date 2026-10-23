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
from PyQt6.QtCore import Qt, pyqtSignal, QPointF, QRectF
from PyQt6.QtGui import QColor, QFont, QBrush, QPolygonF, QPen
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
    QGraphicsPolygonItem,
)
import pyqtgraph as pg
import pyqtgraph.exporters
from src.core.table import DataTable, TableType
from src.core.sheet import BaseSheet, GraphSheet, GraphType, ResultSheet
from src.core.hooking import GLOBAL_REGISTRY, HookedConstant
from src.ui.mediator import WorkspaceMediator
from src.ui.dialogs import FormatSeriesDialog


class ScientificPlotCanvas(QWidget):
    """
    High-performance scientific plot widget rendering publication-grade vector graphs.
    Includes active analysis overlay registry, toolbar layer clearing UI,
    and journal publishing grayscale symbology.
    """

    pinGraphRequested = pyqtSignal()
    freezeToggled = pyqtSignal(bool)
    graphModified = pyqtSignal()

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
        self.series_styles: Dict[Any, str] = {}
        self.error_style: str = "SEM"
        self.custom_series_colors: Dict[str, QColor] = {}
        self.custom_series_opacities: Dict[str, float] = {}

        # Active overlay registry: analysis_id -> { 'name': str, 'plot_items': list }
        self.analysis_layers: Dict[str, Dict[str, Any]] = {}
        self.custom_analysis_layers: Dict[str, Dict[str, Any]] = {}
        self.removed_layer_ids: Set[str] = set()

        self.hooked_lines: List[Dict[str, Any]] = []
        self.hooked_labels: List[Dict[str, Any]] = []
        self.embedded_tables: List[Dict[str, Any]] = []

        self.show_bar_scatter_points: bool = True
        self.current_graph_sheet: Optional[GraphSheet] = None
        self.active_sheet: Optional[BaseSheet] = None

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
            GraphType.BARS_AND_POINTS.value,
            "Superimposed Bars & Points",
            GraphType.GROUPED_BARS_POINTS.value,
            GraphType.XY_SPIKES_CURVE.value,
            GraphType.DEPTH_GRAPH.value,
            GraphType.FLOATING_BARS.value,
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

        top_bar.addWidget(QLabel("Whiskers:"))
        self.whisker_dir_combo = QComboBox()
        self.whisker_dir_combo.addItems([
            "Both (±)",
            "Upper (+) Only",
        ])
        self.whisker_dir_combo.currentIndexChanged.connect(self._on_whisker_dir_changed)
        top_bar.addWidget(self.whisker_dir_combo)

        # Per-graph scatter points toggle on bar charts
        self.btn_toggle_scatter = QToolButton()
        self.btn_toggle_scatter.setText("⚪ Show Scatter Points")
        self.btn_toggle_scatter.setCheckable(True)
        self.btn_toggle_scatter.setChecked(True)
        self.btn_toggle_scatter.toggled.connect(self._on_toggle_scatter_points)
        top_bar.addWidget(self.btn_toggle_scatter)

        # Pinned & Frozen Graph Controls
        self.btn_pin = QToolButton()
        self.btn_pin.setText("📌 Pin Graph")
        self.btn_pin.setToolTip("Pin snapshot as an independent Graph Sheet")
        self.btn_pin.clicked.connect(self._on_pin_clicked)
        top_bar.addWidget(self.btn_pin)

        self.btn_freeze = QToolButton()
        self.btn_freeze.setText("❄️ Freeze Graph")
        self.btn_freeze.setToolTip("Freeze graph to disconnect from real-time data table modification signals")
        self.btn_freeze.setCheckable(True)
        self.btn_freeze.setVisible(False)
        self.btn_freeze.toggled.connect(self._on_freeze_toggled)
        top_bar.addWidget(self.btn_freeze)

        # Specification-compliant control aliases
        self.plot_type_combo = self.type_combo
        self.show_scatter_check = self.btn_toggle_scatter
        self.btn_pin_graph = self.btn_pin
        self.btn_freeze_graph = self.btn_freeze

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

        # Context menu integration on plot ViewBox
        if hasattr(self.plot_widget.plotItem, "vb") and hasattr(self.plot_widget.plotItem.vb, "menu"):
            try:
                self.plot_widget.plotItem.vb.menu.addSeparator()
                self.plot_widget.plotItem.vb.menu.addMenu(self.analysis_menu)
            except Exception:
                pass

    def contextMenuEvent(self, event) -> None:
        """Shows contextual formatting and analysis options."""
        menu = QMenu(self)

        self._rebuild_analysis_menu()
        menu.addMenu(self.analysis_menu)

        menu.addSeparator()

        if self._is_bar_graph_type():
            act_scatter_bars = menu.addAction("Show Individual Scatter Points on Bars")
            act_scatter_bars.setCheckable(True)
            act_scatter_bars.setChecked(self.show_bar_scatter_points)
            act_scatter_bars.toggled.connect(self.set_show_bar_scatter_points)
            menu.addSeparator()

        if self.current_table and self.current_table.y_group_titles:
            series_menu = menu.addMenu("Series Representation")
            for g_title in self.current_table.y_group_titles:
                sub = series_menu.addMenu(f"{g_title}")
                act_bars = sub.addAction("📊 Spikes / Columns")
                act_scatter = sub.addAction("⚪ Scatter Points")
                act_line = sub.addAction("📈 Connected Line")
                act_bars.triggered.connect(lambda chk=False, t=g_title: self.set_series_representation(t, "spikes"))
                act_scatter.triggered.connect(lambda chk=False, t=g_title: self.set_series_representation(t, "scatter"))
                act_line.triggered.connect(lambda chk=False, t=g_title: self.set_series_representation(t, "line"))

        err_menu = menu.addMenu("Error Whisker Style")
        for style_name in ("SEM", "SD", "95% CI"):
            act = err_menu.addAction(f"Mean ± {style_name}")
            act.triggered.connect(lambda chk=False, s=style_name: self.set_error_style(s))

        menu.addSeparator()
        act_format = menu.addAction("🎨 Format Series / Bars...")
        act_format.triggered.connect(self.action_format_series)

        menu.addSeparator()
        if self.is_grayscale_mode:
            pal_act = menu.addAction("🎨 Switch to Color Palette")
            pal_act.triggered.connect(lambda: self.set_grayscale_mode(False))
        else:
            pal_act = menu.addAction("📄 Switch to Journal Grayscale")
            pal_act.triggered.connect(lambda: self.set_grayscale_mode(True))

        menu.exec(event.globalPos())

    @property
    def whisker_direction(self) -> str:
        """Returns the active error whisker direction: 'Both (±)' or 'Upper (+) Only'."""
        if hasattr(self, "whisker_dir_combo"):
            return self.whisker_dir_combo.currentText()
        return "Both (±)"

    def set_whisker_direction(self, direction: str) -> None:
        """Sets error whisker direction: 'Both (±)' or 'Upper (+) Only'."""
        if hasattr(self, "whisker_dir_combo"):
            idx = self.whisker_dir_combo.findText(direction)
            if idx >= 0:
                self.whisker_dir_combo.setCurrentIndex(idx)
                return
        self.refresh_plot(preserve_range=True)

    def action_format_series(self) -> None:
        """Opens FormatSeriesDialog to configure custom color and opacity for series."""
        names = []
        if self.current_table is not None:
            if self.current_table.table_type == TableType.COLUMN:
                names = list(self.current_table.get_all_group_values().keys())
            else:
                names = list(self.current_table.y_group_titles)
        if not names:
            return

        dlg = FormatSeriesDialog(
            series_names=names,
            series_colors=self.custom_series_colors,
            series_opacities=self.custom_series_opacities,
            is_grayscale=self.is_grayscale_mode,
            parent=self,
        )
        if dlg.exec():
            colors, opacities = dlg.get_formatting()
            self.custom_series_colors.update(colors)
            self.custom_series_opacities.update(opacities)
            self._sync_plot_config_to_active_sheet()
            self.refresh_plot(preserve_range=True)

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
        self._sync_plot_config_to_active_sheet()
        self.refresh_plot(preserve_range=True)

    def _on_type_changed(self) -> None:
        self._update_scatter_button_visibility()
        self._sync_plot_config_to_active_sheet()
        self.refresh_plot()

    def _on_whisker_dir_changed(self) -> None:
        self._sync_plot_config_to_active_sheet()
        self.refresh_plot(preserve_range=True)

    def _on_pin_clicked(self) -> None:
        self.pinGraphRequested.emit()

    def pin_graph(self) -> None:
        """Triggers pinning of current graph snapshot."""
        self._on_pin_clicked()

    def _on_freeze_toggled(self, checked: bool) -> None:
        sheet = self.active_sheet or self.current_graph_sheet
        if isinstance(sheet, GraphSheet):
            sheet.is_frozen = checked
            if not hasattr(sheet, "plot_config") or sheet.plot_config is None:
                sheet.plot_config = {}
            sheet.plot_config["is_frozen"] = checked
        self.freezeToggled.emit(checked)

    def _sync_plot_config_to_active_sheet(self) -> None:
        sheet = self.active_sheet or self.current_graph_sheet
        if sheet is None:
            return
        if not hasattr(sheet, "plot_config") or sheet.plot_config is None:
            sheet.plot_config = {}
        sheet.plot_config["plot_type"] = self.type_combo.currentText()
        sheet.plot_config["graph_type"] = self.type_combo.currentText()
        sheet.plot_config["palette"] = self.palette_combo.currentText()
        sheet.plot_config["palette_index"] = self.palette_combo.currentIndex()
        sheet.plot_config["is_grayscale"] = self.is_grayscale_mode
        sheet.plot_config["whisker_dir"] = self.whisker_dir_combo.currentText()
        sheet.plot_config["show_bar_scatter_points"] = self.show_bar_scatter_points
        sheet.plot_config["show_scatter"] = self.show_bar_scatter_points
        sheet.plot_config["error_style"] = self.error_style
        sheet.plot_config["series_styles"] = dict(self.series_styles)
        sheet.plot_config["series_colors"] = {
            k: (v.name() if hasattr(v, "name") else str(v))
            for k, v in self.custom_series_colors.items()
        }
        sheet.plot_config["series_opacities"] = dict(self.custom_series_opacities)
        if isinstance(sheet, GraphSheet):
            sheet.plot_config["is_frozen"] = sheet.is_frozen
            sheet.plot_config["is_pinned"] = sheet.is_pinned
            for gt in GraphType:
                if gt.value == self.type_combo.currentText():
                    sheet.graph_type = gt
                    break
        self.graphModified.emit()

    def load_plot_config(
        self,
        plot_config: Optional[Dict[str, Any]] = None,
        default_graph_type: Optional[str] = None,
    ) -> None:
        """Loads saved plot_config onto toolbar controls and repaints."""
        controls = [
            self.type_combo,
            self.palette_combo,
            self.whisker_dir_combo,
            self.btn_toggle_scatter,
        ]
        for ctrl in controls:
            ctrl.blockSignals(True)

        cfg = dict(plot_config or {})
        try:
            ptype = cfg.get("plot_type") or cfg.get("graph_type") or default_graph_type
            if ptype:
                val = ptype.value if hasattr(ptype, "value") else str(ptype)
                idx = self.type_combo.findText(val)
                if idx < 0:
                    val_lower = val.lower().replace("_", " ").replace("-", " ")
                    if "depth" in val_lower or "isometric" in val_lower or "2.5d" in val_lower:
                        idx = self.type_combo.findText(GraphType.DEPTH_GRAPH.value)
                    elif "bar" in val_lower and "point" not in val_lower:
                        idx = self.type_combo.findText(GraphType.COLUMN_BAR_ERROR.value)
                    elif "xy" in val_lower or "scatter" in val_lower:
                        idx = self.type_combo.findText(GraphType.XY_SCATTER_CURVE.value)
                    elif "grouped" in val_lower:
                        idx = self.type_combo.findText(GraphType.GROUPED_BARS_POINTS.value)
                    elif "spike" in val_lower or "survival" in val_lower:
                        idx = self.type_combo.findText(GraphType.XY_SPIKES_CURVE.value)
                if idx >= 0:
                    self.type_combo.setCurrentIndex(idx)

            pal_idx = cfg.get("palette_index")
            pal_text = cfg.get("palette")
            is_gray = cfg.get("is_grayscale")
            if is_gray is True or pal_text in ("📄 Palette: Journal Grayscale", "grayscale", "gray") or pal_idx == 1:
                self.palette_combo.setCurrentIndex(1)
            elif pal_text and pal_text not in ("🎨 Palette: Color", "color"):
                idx = self.palette_combo.findText(pal_text)
                if idx < 0:
                    self.palette_combo.addItem(pal_text)
                    idx = self.palette_combo.findText(pal_text)
                if idx >= 0:
                    self.palette_combo.setCurrentIndex(idx)
            else:
                self.palette_combo.setCurrentIndex(0)

            whisker_dir = cfg.get("whisker_dir")
            if whisker_dir:
                idx = self.whisker_dir_combo.findText(whisker_dir)
                if idx >= 0:
                    self.whisker_dir_combo.setCurrentIndex(idx)
            else:
                self.whisker_dir_combo.setCurrentIndex(0)

            val = bool(cfg.get("show_bar_scatter_points", cfg.get("show_scatter", True)))
            self.show_bar_scatter_points = val
            self.btn_toggle_scatter.setChecked(val)

            self.error_style = cfg.get("error_style", "SEM")
            self.series_styles = dict(cfg.get("series_styles", {}))
            if "series_colors" in cfg:
                self.custom_series_colors = {
                    k: QColor(v) for k, v in cfg["series_colors"].items()
                }
            else:
                self.custom_series_colors = {}
            self.custom_series_opacities = dict(cfg.get("series_opacities", {}))
        finally:
            for ctrl in controls:
                ctrl.blockSignals(False)

        self.refresh_plot()

    def _is_bar_graph_type(self) -> bool:
        t = self.type_combo.currentText()
        if t in (
            GraphType.COLUMN_BAR_ERROR.value,
            GraphType.BARS_AND_POINTS.value,
            "Superimposed Bars & Points",
            GraphType.GROUPED_BARS_POINTS.value,
            "Grouped Bars + Replicate Points",
            GraphType.DEPTH_GRAPH.value,
            "Depth Profile",
            GraphType.FLOATING_BARS.value,
            "Floating Bars (Min to Max)",
        ):
            return True
        if self.current_table and self.current_table.table_type in (TableType.GROUPED, TableType.COLUMN):
            return True
        return False

    def _update_scatter_button_visibility(self) -> None:
        is_bar = self._is_bar_graph_type()
        self.btn_toggle_scatter.setVisible(is_bar)
        self.btn_toggle_scatter.setEnabled(is_bar)

    def _on_toggle_scatter_points(self, checked: bool) -> None:
        self.set_show_bar_scatter_points(checked)

    def set_show_bar_scatter_points(self, show: bool) -> None:
        self.show_bar_scatter_points = bool(show)
        self.btn_toggle_scatter.blockSignals(True)
        self.btn_toggle_scatter.setChecked(self.show_bar_scatter_points)
        self.btn_toggle_scatter.blockSignals(False)
        self._sync_plot_config_to_active_sheet()
        self.refresh_plot(preserve_range=True)

    def set_series_representation(self, series: Any, rep_type: str) -> None:
        """Sets presentation style for a series (e.g. 'bars', 'spikes', 'scatter', 'line')."""
        self.series_styles[series] = rep_type.lower()
        self._sync_plot_config_to_active_sheet()
        self.refresh_plot()

    def set_error_style(self, style: str) -> None:
        """Sets error bar calculation style: 'SEM', 'SD', or '95% CI'."""
        self.error_style = style
        self._sync_plot_config_to_active_sheet()
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

    def remove_analysis_layer(self, layer_id: str, notify: bool = True) -> bool:
        """
        Removes the corresponding items from self.plot_widget.plotItem via removeItem().
        Deletes the key from self.analysis_layers.
        Restores the raw dataset scatter view without resetting user zoom or axis ranges.
        """
        if layer_id not in self.analysis_layers and layer_id not in self.custom_analysis_layers:
            return False

        self.removed_layer_ids.add(layer_id)
        self.custom_analysis_layers.pop(layer_id, None)
        layer = self.analysis_layers.pop(layer_id, {})

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

        # Ensure raw dataset scatter items remain visible and restored
        for item in plot_item.items:
            if isinstance(item, pg.ScatterPlotItem):
                item.setVisible(True)

        self._rebuild_analysis_menu()

        if notify and self.mediator is not None:
            self.mediator.notify_analysis_layer_removed(layer_id)

        return True

    def clear_all_analysis_layers(self, notify: bool = True) -> None:
        """Removes all active fitted curves and statistical overlays."""
        all_ids = set(self.analysis_layers.keys()) | set(self.custom_analysis_layers.keys())
        for layer_id in list(all_ids):
            self.remove_analysis_layer(layer_id, notify=False)
        if notify and self.mediator is not None:
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
        sheet: Optional[Any] = None,
        table_name: Optional[str] = None,
    ) -> None:
        """Updates referenced table, result sheet, and optional graph sheet, then repaints."""
        if result is not None and (
            self.current_result is None
            or getattr(result, "sheet_id", None) != getattr(self.current_result, "sheet_id", None)
        ):
            self.removed_layer_ids.discard(getattr(result, "sheet_id", ""))

        self.current_table = table
        self.current_result = result
        self.active_sheet = sheet
        self.current_graph_sheet = sheet if isinstance(sheet, GraphSheet) else None

        if isinstance(sheet, GraphSheet):
            self.btn_freeze.setVisible(True)
            self.btn_freeze.blockSignals(True)
            self.btn_freeze.setChecked(bool(sheet.is_frozen))
            self.btn_freeze.blockSignals(False)
        else:
            self.btn_freeze.setVisible(False)

        if sheet is not None:
            # Determine default graph type if none is saved in plot_config
            def_gt = None
            if graph_type:
                def_gt = graph_type.value
            elif isinstance(sheet, GraphSheet):
                def_gt = sheet.graph_type.value
            elif self.current_table:
                if self.current_table.table_type == TableType.COLUMN:
                    def_gt = GraphType.COLUMN_BAR_ERROR.value
                elif self.current_table.table_type == TableType.GROUPED:
                    def_gt = GraphType.GROUPED_BARS_POINTS.value
                else:
                    def_gt = GraphType.XY_SCATTER_CURVE.value

            plot_cfg = getattr(sheet, "plot_config", None) or {}
            self.load_plot_config(plot_cfg, default_graph_type=def_gt)
        else:
            if graph_type:
                idx = self.type_combo.findText(graph_type.value)
                if idx >= 0:
                    self.type_combo.setCurrentIndex(idx)
            self.refresh_plot()

    def refresh_plot(self, preserve_range: bool = False) -> None:
        """Clears and redraws graph according to current data, hooked elements, and settings."""
        self._update_scatter_button_visibility()
        range_before = None
        if preserve_range:
            try:
                range_before = self.plot_widget.viewRange()
            except Exception:
                range_before = None

        self.plot_widget.clear()
        if self.plot_widget.plotItem.legend is not None:
            self.plot_widget.plotItem.legend.clear()

        # Reset active overlay item references before redrawing
        self.analysis_layers.clear()

        if self.current_table is None:
            self._rebuild_analysis_menu()
            return

        selected_type = self.type_combo.currentText()
        if selected_type not in (GraphType.DEPTH_GRAPH.value, "Depth Graph", "Depth Profile"):
            self.plot_widget.getViewBox().invertY(False)

        if selected_type in (GraphType.DEPTH_GRAPH.value, "Depth Graph", "Depth Profile"):
            self._render_depth_graph()
        elif selected_type in (GraphType.FLOATING_BARS.value, "Floating Bars (Min to Max)"):
            self._render_floating_bars()
        elif (
            selected_type in (GraphType.GROUPED_BARS_POINTS.value, "Grouped Bars + Replicate Points")
            or (self.current_table.table_type == TableType.GROUPED and selected_type not in (GraphType.XY_SPIKES_CURVE.value, GraphType.XY_SCATTER_CURVE.value))
        ):
            self._render_grouped_bars_points()
        elif selected_type in (GraphType.XY_SPIKES_CURVE.value, "Columns/Spikes + Curve Fit"):
            self._render_xy_spikes_and_curve()
        elif selected_type == "Superimposed Bars & Points":
            self._render_superimposed_graph()
        elif (
            selected_type in (GraphType.BARS_AND_POINTS.value, GraphType.COLUMN_BAR_ERROR.value)
            or self.current_table.table_type == TableType.COLUMN
        ):
            self._render_column_graph()
        else:
            self._render_xy_graph()

        # Render Hooked Lines
        self._render_hooked_elements()

        # Render Embedded Tables
        self._render_embedded_tables()

        if preserve_range and range_before is not None:
            self.plot_widget.setRange(xRange=range_before[0], yRange=range_before[1], padding=0)
        else:
            self.plot_widget.autoRange()

    def _render_analysis_overlays(self) -> None:
        """Renders custom and result sheet analysis fitted curves."""
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

            has_custom_color = g_title in self.custom_series_colors
            custom_color = self.custom_series_colors.get(g_title)
            custom_opacity = self.custom_series_opacities.get(g_title, 1.0 if is_grayscale else 0.8)

            if is_grayscale:
                sym_spec = self.GRAYSCALE_SYMBOLS[col_idx % len(self.GRAYSCALE_SYMBOLS)]
                symbol = sym_spec["symbol"]
                pen = sym_spec["pen"]
                brush = sym_spec["brush"]
            else:
                color = custom_color if (has_custom_color and custom_color is not None) else self.COLOR_PALETTE[col_idx % len(self.COLOR_PALETTE)]
                symbol = "o"
                pen = pg.mkPen(color, width=1.0)
                alpha = int(round(custom_opacity * 255))
                brush = pg.mkBrush(color.red(), color.green(), color.blue(), alpha)

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

        self._render_analysis_overlays()

    def _render_column_graph(self) -> None:
        """Renders column bars with Mean ± SEM/SD/95% CI and overlaid scatter points."""
        table = self.current_table
        if table is None:
            return

        err_style = getattr(self, "error_style", "SEM")
        if err_style == "SD":
            error_label = "SD"
        elif err_style == "95% CI":
            error_label = "95% CI"
        else:
            error_label = "SEM"

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

            has_custom_color = name in self.custom_series_colors
            custom_color = self.custom_series_colors.get(name)
            custom_opacity = self.custom_series_opacities.get(name, 1.0 if is_grayscale else 0.8)

            sym_spec = self.GRAYSCALE_SYMBOLS[pos % len(self.GRAYSCALE_SYMBOLS)]

            if is_grayscale and not has_custom_color:
                pattern_idx = pos % 5
                bar_pen = pg.mkPen("k", width=1.5)
                if pattern_idx == 0:
                    bar_brush = QBrush(QColor("#000000"))
                    pt_brush = pg.mkBrush("w")
                    pt_pen = pg.mkPen("k", width=1.5)
                elif pattern_idx == 1:
                    bar_brush = QBrush(QColor("#ffffff"))
                    bar_pen = pg.mkPen(QColor(0, 0, 0), width=1.5)
                    pt_brush = pg.mkBrush("k")
                    pt_pen = pg.mkPen("k", width=1.5)
                elif pattern_idx == 2:
                    bar_brush = QBrush(QColor("#808080"))
                    pt_brush = pg.mkBrush("w")
                    pt_pen = pg.mkPen("k", width=1.5)
                elif pattern_idx == 3:
                    bar_brush = QBrush(QColor("#c0c0c0"))
                    pt_brush = pg.mkBrush("k")
                    pt_pen = pg.mkPen("k", width=1.5)
                else:  # pattern_idx == 4
                    bar_brush = QBrush(QColor("#000000"), Qt.BrushStyle.DiagCrossPattern)
                    pt_brush = pg.mkBrush("w")
                    pt_pen = pg.mkPen("k", width=1.5)

                pt_symbol = sym_spec["symbol"]
                err_pen = pg.mkPen("k", width=1.5)
            else:
                if has_custom_color and custom_color is not None:
                    color = custom_color
                else:
                    color = self.COLOR_PALETTE[pos % len(self.COLOR_PALETTE)]
                alpha = int(round(custom_opacity * 255))
                bar_pen = pg.mkPen(color, width=1.5)
                bar_brush = pg.mkBrush(color.red(), color.green(), color.blue(), alpha)
                pt_symbol = "o"
                pt_pen = pg.mkPen(color.darker(140), width=1)
                pt_brush = pg.mkBrush(color)
                err_pen = pg.mkPen(color.darker(160), width=1.5)

            mean = float(np.mean(vals))
            if len(vals) > 1:
                sd = float(np.std(vals, ddof=1))
                sem = sd / np.sqrt(len(vals))
                if error_label == "SD":
                    err_val = sd
                elif error_label == "95% CI":
                    try:
                        from scipy import stats
                        t_crit = float(stats.t.ppf(0.975, df=len(vals) - 1))
                        err_val = sem * t_crit
                    except Exception:
                        err_val = sem * 1.96
                else:
                    err_val = sem
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
                top_err = np.array([err_val])
                bot_err = np.zeros_like(top_err) if self.whisker_direction == "Upper (+) Only" else top_err
                err = pg.ErrorBarItem(
                    x=np.array([pos]),
                    y=np.array([mean]),
                    top=top_err,
                    bottom=bot_err,
                    beam=0.2,
                    pen=err_pen,
                )
                self.plot_widget.addItem(err)

            # Scatter points
            if self.show_bar_scatter_points:
                if len(vals) > 1:
                    jitter = (np.random.RandomState(42 + pos).rand(len(vals)) - 0.5) * 0.25
                else:
                    jitter = np.zeros(len(vals))
                scatter = pg.ScatterPlotItem(
                    x=pos + jitter,
                    y=vals,
                    pen=pt_pen,
                    brush=pt_brush,
                    symbol=pt_symbol,
                    size=6,
                )
                self.plot_widget.addItem(scatter)

    def _render_grouped_bars_points(self) -> None:
        """
        Renders two-way grouped data (Factor A rows x Factor B columns)
        displaying clustered bars with matched individual replicate points nested inside each category group.
        Matches docs/graph_specs/1661_Points_and_grouped_bars.pzfx.
        """
        table = self.current_table
        if table is None:
            return

        err_style = getattr(self, "error_style", "SEM")
        if err_style == "SD":
            error_label = "SD"
        elif err_style == "95% CI":
            error_label = "95% CI"
        else:
            error_label = "SEM"

        self.plot_widget.setLabel("bottom", "Conditions", color="black")
        self.plot_widget.setLabel("left", f"Value (Mean ± {error_label})", color="black")

        num_groups = len(table.y_group_titles)
        if num_groups == 0:
            return

        is_grayscale = self.is_grayscale_mode

        # Determine active rows (Factor A)
        active_rows = []
        for r in range(table.num_rows):
            has_data = False
            for g in range(num_groups):
                for s in range(table.num_replicates):
                    if not table.is_excluded(r, "Y", g, s):
                        val = table.y_data[g, r, s]
                        if not np.isnan(val):
                            has_data = True
                            break
                if has_data:
                    break
            if has_data or (r < len(table.row_titles) and table.row_titles[r].strip()):
                active_rows.append(r)

        if not active_rows:
            active_rows = list(range(min(table.num_rows, 10)))

        ticks = []
        for idx, r in enumerate(active_rows):
            title = table.row_titles[r] if r < len(table.row_titles) and table.row_titles[r].strip() else f"Row {r+1}"
            ticks.append((idx, title))
        self.plot_widget.getAxis("bottom").setTicks([ticks])

        cluster_w = 0.8
        sub_w = cluster_w / max(1, num_groups)

        for g_idx, g_title in enumerate(table.y_group_titles):
            has_custom_color = g_title in self.custom_series_colors
            custom_color = self.custom_series_colors.get(g_title)
            custom_opacity = self.custom_series_opacities.get(g_title, 1.0 if is_grayscale else 0.8)

            sym_spec = self.GRAYSCALE_SYMBOLS[g_idx % len(self.GRAYSCALE_SYMBOLS)]

            if is_grayscale and not has_custom_color:
                pattern_idx = g_idx % 5
                bar_pen = pg.mkPen("k", width=1.5)
                if pattern_idx == 0:
                    bar_brush = QBrush(QColor("#000000"))
                    pt_brush = pg.mkBrush("w")
                    pt_pen = pg.mkPen("k", width=1.5)
                elif pattern_idx == 1:
                    bar_brush = QBrush(QColor("#ffffff"))
                    bar_pen = pg.mkPen(QColor(0, 0, 0), width=1.5)
                    pt_brush = pg.mkBrush("k")
                    pt_pen = pg.mkPen("k", width=1.5)
                elif pattern_idx == 2:
                    bar_brush = QBrush(QColor("#808080"))
                    pt_brush = pg.mkBrush("w")
                    pt_pen = pg.mkPen("k", width=1.5)
                elif pattern_idx == 3:
                    bar_brush = QBrush(QColor("#c0c0c0"))
                    pt_brush = pg.mkBrush("k")
                    pt_pen = pg.mkPen("k", width=1.5)
                else:  # pattern_idx == 4
                    bar_brush = QBrush(QColor("#000000"), Qt.BrushStyle.DiagCrossPattern)
                    pt_brush = pg.mkBrush("w")
                    pt_pen = pg.mkPen("k", width=1.5)

                pt_symbol = sym_spec["symbol"]
                err_pen = pg.mkPen("k", width=1.5)
            else:
                if has_custom_color and custom_color is not None:
                    color = custom_color
                else:
                    color = self.COLOR_PALETTE[g_idx % len(self.COLOR_PALETTE)]
                alpha = int(round(custom_opacity * 255))
                bar_pen = pg.mkPen(color, width=1.5)
                bar_brush = pg.mkBrush(color.red(), color.green(), color.blue(), alpha)
                pt_symbol = "o"
                pt_pen = pg.mkPen(color.darker(140), width=1.0)
                pt_brush = pg.mkBrush(color)
                err_pen = pg.mkPen(color.darker(160), width=1.5)

            if self.plot_widget.plotItem.legend is not None:
                dummy = pg.PlotDataItem(pen=bar_pen, brush=bar_brush, symbol=pt_symbol, name=g_title)
                self.plot_widget.plotItem.legend.addItem(dummy, g_title)

            for x_idx, r in enumerate(active_rows):
                x_pos = x_idx + (g_idx - (num_groups - 1) / 2.0) * sub_w

                valid_vals = []
                for s in range(table.num_replicates):
                    if not table.is_excluded(r, "Y", g_idx, s):
                        val = table.y_data[g_idx, r, s]
                        if not np.isnan(val):
                            valid_vals.append(float(val))

                if not valid_vals:
                    continue

                mean = float(np.mean(valid_vals))
                if len(valid_vals) > 1:
                    sd = float(np.std(valid_vals, ddof=1))
                    sem = sd / np.sqrt(len(valid_vals))
                    if error_label == "SD":
                        err_val = sd
                    elif error_label == "95% CI":
                        try:
                            from scipy import stats
                            t_crit = float(stats.t.ppf(0.975, df=len(valid_vals) - 1))
                            err_val = sem * t_crit
                        except Exception:
                            err_val = sem * 1.96
                    else:
                        err_val = sem
                else:
                    err_val = 0.0

                bar = pg.BarGraphItem(
                    x=[x_pos],
                    height=[mean],
                    width=sub_w * 0.85,
                    brush=bar_brush,
                    pen=bar_pen,
                )
                self.plot_widget.addItem(bar)

                if err_val > 0:
                    top_err = np.array([err_val])
                    bot_err = np.zeros_like(top_err) if self.whisker_direction == "Upper (+) Only" else top_err
                    err = pg.ErrorBarItem(
                        x=np.array([x_pos]),
                        y=np.array([mean]),
                        top=top_err,
                        bottom=bot_err,
                        beam=sub_w * 0.3,
                        pen=err_pen,
                    )
                    self.plot_widget.addItem(err)

                if self.show_bar_scatter_points:
                    if len(valid_vals) > 1:
                        rng = np.random.RandomState(42 + r * 13 + g_idx * 7)
                        jitter = (rng.rand(len(valid_vals)) - 0.5) * (sub_w * 0.35)
                    else:
                        jitter = np.zeros(len(valid_vals))

                    scatter = pg.ScatterPlotItem(
                        x=x_pos + jitter,
                        y=valid_vals,
                        pen=pt_pen,
                        brush=pt_brush,
                        symbol=pt_symbol,
                        size=6,
                    )
                    self.plot_widget.addItem(scatter)

    def get_plot_snapshot(self, width: int = 1200, height: int = 800) -> bytes:
        """Renders high-DPI snapshot of the plot with generous axis margins to prevent label truncation."""
        from src.core.office_exporter import render_graph_to_png_bytes
        sheet = self.current_graph_sheet or self.active_sheet
        doc = (
            getattr(sheet, "document", None)
            or getattr(self, "document", None)
            or (getattr(self.mediator, "document", None) if hasattr(self, "mediator") else None)
        )
        return render_graph_to_png_bytes(
            sheet if sheet is not None else self.current_table,
            document=doc,
            graph_canvas=self,
            width=width,
            height=height,
        )

    def _render_depth_graph(self) -> None:
        """
        Renders authentic GraphPad Prism Depth graph / profile.
        Plots measurements on the horizontal axis and depth/categories on the vertical axis,
        with depth 0 at top and increasing downwards (inverted vertical axis),
        with horizontal error whiskers and high-contrast styling matching Prism specifications.
        """
        table = self.current_table
        if table is None:
            return

        sheet = self.current_graph_sheet or self.active_sheet
        plot_cfg = getattr(sheet, "plot_config", {}) or {}
        custom_y_title = plot_cfg.get("y_title")

        def _get_depth_vertical_label() -> str:
            if custom_y_title and str(custom_y_title).strip():
                return str(custom_y_title).strip()
            if table.table_type in (TableType.COLUMN, TableType.GROUPED):
                factor_title = (
                    getattr(table, "factor_title", None)
                    or getattr(table, "grouping_title", None)
                    or getattr(table, "group_title", None)
                )
                if factor_title and str(factor_title).strip():
                    return str(factor_title).strip()
                sheet_name = getattr(sheet, "name", None) or getattr(sheet, "title", None)
                if sheet_name and str(sheet_name).strip() and str(sheet_name).strip() not in ("Graph 1", "Graph"):
                    return str(sheet_name).strip()
                table_title = getattr(table, "title", None)
                if table_title and str(table_title).strip() and str(table_title).strip() not in ("Data 1", "Table 1", "Untitled"):
                    return str(table_title).strip()
                return "Groups"
            else:
                # XY / multi-column datasets: Use active Y-column header names
                y_cols = [t for t in getattr(table, "y_group_titles", []) if t and str(t).strip()]
                if y_cols:
                    return ", ".join(y_cols)
                if getattr(table, "x_title", None) and str(table.x_title).strip():
                    return str(table.x_title).strip()
                sheet_name = getattr(sheet, "name", None) or getattr(sheet, "title", None)
                if sheet_name and str(sheet_name).strip():
                    return str(sheet_name).strip()
                return "Depth"

        err_style = getattr(self, "error_style", "SEM")
        if err_style == "SD":
            error_label = "SD"
        elif err_style == "95% CI":
            error_label = "95% CI"
        else:
            error_label = "SEM"

        is_grayscale = self.is_grayscale_mode
        vb = self.plot_widget.getViewBox()
        vb.invertY(True)

        vert_lbl = _get_depth_vertical_label()

        if table.table_type == TableType.GROUPED:
            num_groups = len(table.y_group_titles)
            if num_groups == 0:
                return

            self.plot_widget.setLabel("bottom", f"Value (Mean ± {error_label})", color="black")
            self.plot_widget.setLabel("left", vert_lbl, color="black")

            active_rows = []
            for r in range(table.num_rows):
                has_data = False
                for g in range(num_groups):
                    for s in range(table.num_replicates):
                        if not table.is_excluded(r, "Y", g, s):
                            val = table.y_data[g, r, s]
                            if not np.isnan(val):
                                has_data = True
                                break
                    if has_data:
                        break
                if has_data or (r < len(table.row_titles) and table.row_titles[r].strip()):
                    active_rows.append(r)

            if not active_rows:
                active_rows = list(range(min(table.num_rows, 10)))

            ticks = []
            for idx, r in enumerate(active_rows):
                title = table.row_titles[r] if r < len(table.row_titles) and table.row_titles[r].strip() else f"Depth {r}"
                ticks.append((idx, title))
            self.plot_widget.getAxis("left").setTicks([ticks])

            for g_idx in range(num_groups):
                g_title = table.y_group_titles[g_idx]
                has_custom_color = g_title in self.custom_series_colors
                custom_color = self.custom_series_colors.get(g_title)
                custom_opacity = self.custom_series_opacities.get(g_title, 1.0 if is_grayscale else 0.8)

                if is_grayscale and not has_custom_color:
                    base_color = QColor(0, 0, 0)
                else:
                    base_color = custom_color or self.COLOR_PALETTE[g_idx % len(self.COLOR_PALETTE)]

                alpha = int(round(custom_opacity * 255))
                pt_color = QColor(base_color)
                pt_color.setAlpha(alpha)

                pt_brush = pg.mkBrush(pt_color)
                pt_pen = pg.mkPen("k" if is_grayscale else base_color.darker(140), width=1.5)
                line_pen = pg.mkPen(color=base_color, width=2.0)
                err_pen = pg.mkPen("k" if is_grayscale else base_color.darker(150), width=1.5)

                if self.plot_widget.plotItem.legend is not None:
                    dummy = pg.PlotDataItem(pen=line_pen, brush=pt_brush, name=g_title)
                    self.plot_widget.plotItem.legend.addItem(dummy, g_title)

                series_means = []
                series_positions = []

                for y_idx, r in enumerate(active_rows):
                    valid_vals = []
                    for s in range(table.num_replicates):
                        if not table.is_excluded(r, "Y", g_idx, s):
                            val = table.y_data[g_idx, r, s]
                            if not np.isnan(val):
                                valid_vals.append(float(val))
                    if not valid_vals:
                        continue

                    mean = float(np.mean(valid_vals))
                    if len(valid_vals) > 1:
                        sd = float(np.std(valid_vals, ddof=1))
                        sem = sd / np.sqrt(len(valid_vals))
                        if error_label == "SD":
                            err_val = sd
                        elif error_label == "95% CI":
                            try:
                                from scipy import stats
                                t_crit = float(stats.t.ppf(0.975, df=len(valid_vals) - 1))
                                err_val = sem * t_crit
                            except Exception:
                                err_val = sem * 1.96
                        else:
                            err_val = sem
                    else:
                        err_val = 0.0

                    series_means.append(mean)
                    series_positions.append(y_idx)

                    # Horizontal error whiskers
                    if err_val > 0:
                        right_err = np.array([err_val])
                        left_err = np.zeros_like(right_err) if self.whisker_direction == "Upper (+) Only" else right_err
                        err = pg.ErrorBarItem(
                            x=np.array([mean]),
                            y=np.array([y_idx]),
                            left=left_err,
                            right=right_err,
                            pen=err_pen,
                            beam=0.2,
                        )
                        self.plot_widget.addItem(err)

                if series_means:
                    line_item = pg.PlotDataItem(
                        x=np.array(series_means),
                        y=np.array(series_positions),
                        pen=line_pen,
                    )
                    self.plot_widget.addItem(line_item)

                    sym = self.GRAYSCALE_SYMBOLS[g_idx % len(self.GRAYSCALE_SYMBOLS)]["symbol"] if is_grayscale else "o"
                    scatter = pg.ScatterPlotItem(
                        x=np.array(series_means),
                        y=np.array(series_positions),
                        symbol=sym,
                        size=8,
                        brush=pt_brush,
                        pen=pt_pen,
                    )
                    self.plot_widget.addItem(scatter)

        elif table.table_type == TableType.COLUMN:
            group_data = table.get_all_group_values()
            names = list(group_data.keys())
            num_cols = len(names)
            if num_cols == 0:
                return

            self.plot_widget.setLabel("bottom", f"Value (Mean ± {error_label})", color="black")
            self.plot_widget.setLabel("left", vert_lbl, color="black")

            ticks = [(pos, name) for pos, name in enumerate(names)]
            self.plot_widget.getAxis("left").setTicks([ticks])

            all_means = []
            all_positions = []

            for pos, name in enumerate(names):
                vals = np.array(group_data[name], dtype=np.float64)
                if len(vals) == 0 or np.all(np.isnan(vals)):
                    continue

                valid_vals = vals[~np.isnan(vals)]
                if len(valid_vals) == 0:
                    continue

                mean = float(np.mean(valid_vals))
                if len(valid_vals) > 1:
                    sd = float(np.std(valid_vals, ddof=1))
                    sem = sd / np.sqrt(len(valid_vals))
                    if error_label == "SD":
                        err_val = sd
                    elif error_label == "95% CI":
                        try:
                            from scipy import stats
                            t_crit = float(stats.t.ppf(0.975, df=len(valid_vals) - 1))
                            err_val = sem * t_crit
                        except Exception:
                            err_val = sem * 1.96
                    else:
                        err_val = sem
                else:
                    err_val = 0.0

                all_means.append(mean)
                all_positions.append(pos)

                has_custom_color = name in self.custom_series_colors
                custom_color = self.custom_series_colors.get(name)
                custom_opacity = self.custom_series_opacities.get(name, 1.0 if is_grayscale else 0.8)

                if is_grayscale and not has_custom_color:
                    base_color = QColor(0, 0, 0)
                else:
                    base_color = custom_color or self.COLOR_PALETTE[pos % len(self.COLOR_PALETTE)]

                pt_pen = pg.mkPen("k", width=1.5)
                pt_brush = pg.mkBrush("k" if is_grayscale else base_color)
                err_pen = pg.mkPen("k", width=1.5)

                if err_val > 0:
                    right_err = np.array([err_val])
                    left_err = np.zeros_like(right_err) if self.whisker_direction == "Upper (+) Only" else right_err
                    err = pg.ErrorBarItem(
                        x=np.array([mean]),
                        y=np.array([pos]),
                        left=left_err,
                        right=right_err,
                        pen=err_pen,
                        beam=0.2,
                    )
                    self.plot_widget.addItem(err)

            if all_means:
                line_item = pg.PlotDataItem(
                    x=np.array(all_means),
                    y=np.array(all_positions),
                    pen=pg.mkPen("k" if is_grayscale else "b", width=2.0),
                )
                self.plot_widget.addItem(line_item)

                scatter = pg.ScatterPlotItem(
                    x=np.array(all_means),
                    y=np.array(all_positions),
                    symbol="o",
                    size=9,
                    brush=pg.mkBrush("k"),
                    pen=pg.mkPen("k", width=1.5),
                )
                self.plot_widget.addItem(scatter)

        elif table.table_type == TableType.XY:
            # XY table: X column is Depth, Y columns are measurements
            num_groups = len(table.y_group_titles)
            if num_groups == 0:
                return

            self.plot_widget.setLabel("left", vert_lbl, color="black")
            if num_groups == 1 and table.y_group_titles[0]:
                x_lbl = f"{table.y_group_titles[0]} (Mean ± {error_label})"
            else:
                x_lbl = f"Measurement (Mean ± {error_label})"
            self.plot_widget.setLabel("bottom", x_lbl, color="black")

            x_vals = table.x_data

            for g_idx in range(num_groups):
                g_title = table.y_group_titles[g_idx]
                has_custom_color = g_title in self.custom_series_colors
                custom_color = self.custom_series_colors.get(g_title)
                custom_opacity = self.custom_series_opacities.get(g_title, 1.0 if is_grayscale else 0.8)

                if is_grayscale and not has_custom_color:
                    base_color = QColor(0, 0, 0)
                else:
                    base_color = custom_color or self.COLOR_PALETTE[g_idx % len(self.COLOR_PALETTE)]

                alpha = int(round(custom_opacity * 255))
                pt_color = QColor(base_color)
                pt_color.setAlpha(alpha)

                pt_brush = pg.mkBrush(pt_color)
                pt_pen = pg.mkPen("k" if is_grayscale else base_color.darker(140), width=1.5)
                line_pen = pg.mkPen(color=base_color, width=2.0)
                err_pen = pg.mkPen("k" if is_grayscale else base_color.darker(150), width=1.5)

                if num_groups > 1 and self.plot_widget.plotItem.legend is not None:
                    dummy = pg.PlotDataItem(pen=line_pen, brush=pt_brush, name=g_title)
                    self.plot_widget.plotItem.legend.addItem(dummy, g_title)

                series_means = []
                series_depths = []

                for r, x in enumerate(x_vals):
                    if np.isnan(x):
                        continue
                    valid_y = []
                    for s in range(table.num_replicates):
                        if not table.is_excluded(r, "Y", g_idx, s):
                            v = table.y_data[g_idx, r, s]
                            if not np.isnan(v):
                                valid_y.append(float(v))
                    if not valid_y:
                        continue

                    mean = float(np.mean(valid_y))
                    if len(valid_y) > 1:
                        sd = float(np.std(valid_y, ddof=1))
                        sem = sd / np.sqrt(len(valid_y))
                        if error_label == "SD":
                            err_val = sd
                        elif error_label == "95% CI":
                            try:
                                from scipy import stats
                                t_crit = float(stats.t.ppf(0.975, df=len(valid_y) - 1))
                                err_val = sem * t_crit
                            except Exception:
                                err_val = sem * 1.96
                        else:
                            err_val = sem
                    else:
                        err_val = 0.0

                    series_means.append(mean)
                    series_depths.append(float(x))

                    if err_val > 0:
                        right_err = np.array([err_val])
                        left_err = np.zeros_like(right_err) if self.whisker_direction == "Upper (+) Only" else right_err
                        err = pg.ErrorBarItem(
                            x=np.array([mean]),
                            y=np.array([float(x)]),
                            left=left_err,
                            right=right_err,
                            pen=err_pen,
                            beam=0.2,
                        )
                        self.plot_widget.addItem(err)

                if series_means:
                    line_item = pg.PlotDataItem(
                        x=np.array(series_means),
                        y=np.array(series_depths),
                        pen=line_pen,
                    )
                    self.plot_widget.addItem(line_item)

                    sym = self.GRAYSCALE_SYMBOLS[g_idx % len(self.GRAYSCALE_SYMBOLS)]["symbol"] if is_grayscale else "o"
                    scatter = pg.ScatterPlotItem(
                        x=np.array(series_means),
                        y=np.array(series_depths),
                        symbol=sym,
                        size=9,
                        brush=pt_brush,
                        pen=pt_pen,
                    )
                    self.plot_widget.addItem(scatter)

        else:
            self._render_xy_graph()
            return

        self._render_analysis_overlays()

    def _render_floating_bars(self) -> None:
        """
        Renders Floating Bars (Min to Max) for Column tables.
        Each bar spans from min(vals) to max(vals) for the column.
        """
        table = self.current_table
        if table is None:
            return
        group_data = table.get_all_group_values() if hasattr(table, "get_all_group_values") else {}
        names = list(group_data.keys())
        if not names:
            return

        self.plot_widget.setLabel("bottom", "Groups", color="black")
        self.plot_widget.setLabel("left", "Value (Min to Max)", color="black")

        ticks = [(pos, name) for pos, name in enumerate(names)]
        self.plot_widget.getAxis("bottom").setTicks([ticks])

        is_grayscale = self.is_grayscale_mode

        for pos, name in enumerate(names):
            vals = np.array(group_data[name], dtype=np.float64)
            if len(vals) == 0:
                continue

            has_custom_color = name in self.custom_series_colors
            custom_color = self.custom_series_colors.get(name)
            custom_opacity = self.custom_series_opacities.get(name, 1.0 if is_grayscale else 0.8)

            if is_grayscale and not has_custom_color:
                base_color = QColor(60 + (pos * 40) % 160, 60 + (pos * 40) % 160, 60 + (pos * 40) % 160)
            else:
                base_color = custom_color or self.COLOR_PALETTE[pos % len(self.COLOR_PALETTE)]

            alpha = int(round(custom_opacity * 255))
            bar_color = QColor(base_color)
            bar_color.setAlpha(alpha)
            bar_brush = QBrush(bar_color)
            bar_pen = pg.mkPen(base_color.darker(140), width=1.5)

            min_val = float(np.min(vals))
            max_val = float(np.max(vals))
            height = max(1e-6, max_val - min_val)

            bar = pg.BarGraphItem(
                x=[pos],
                y0=[min_val],
                height=[height],
                width=0.5,
                brush=bar_brush,
                pen=bar_pen,
            )
            self.plot_widget.addItem(bar)

            # Draw median line inside the floating bar
            med_val = float(np.median(vals))
            med_line = pg.PlotCurveItem(
                x=[pos - 0.25, pos + 0.25],
                y=[med_val, med_val],
                pen=pg.mkPen(base_color.darker(160), width=2.0),
            )
            self.plot_widget.addItem(med_line)

            # Scatter points
            if self.show_bar_scatter_points:
                if len(vals) > 1:
                    jitter = (np.random.RandomState(42 + pos).rand(len(vals)) - 0.5) * 0.2
                else:
                    jitter = np.zeros(len(vals))
                scatter = pg.ScatterPlotItem(
                    x=np.full(len(vals), pos) + jitter,
                    y=vals,
                    size=7,
                    pen=pg.mkPen(base_color.darker(140), width=1.0),
                    brush=pg.mkBrush(base_color),
                )
                self.plot_widget.addItem(scatter)

        self._render_analysis_overlays()

    def _render_xy_spikes_and_curve(self) -> None:
        """
        Renders hybrid XY graphs where one Y dataset renders as vertical bar spikes/columns
        while an adjacent Y dataset renders as scatter points and a continuous non-linear regression curve.
        Matches docs/graph_specs/bars_and_points_2.pzf & BarsandPointsStacked.pzf.
        """
        table = self.current_table
        if table is None:
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

            # Determine representation style for this series
            rep_style = self.series_styles.get(g_title)
            if rep_style is None:
                rep_style = self.series_styles.get(col_idx)
            if rep_style is None:
                # Default for XY_SPIKES_CURVE: first dataset is bars/spikes, subsequent are scatter points
                rep_style = "bars" if col_idx == 0 else "scatter"
            else:
                rep_style = rep_style.lower()

            if is_grayscale:
                sym_spec = self.GRAYSCALE_SYMBOLS[col_idx % len(self.GRAYSCALE_SYMBOLS)]
                symbol = sym_spec["symbol"]
                pen = sym_spec["pen"]
                brush = sym_spec["brush"]
                bar_pen = pg.mkPen("k", width=1.5)
                bar_brush = pg.mkBrush(180, 180, 180, 160)
            else:
                color = self.COLOR_PALETTE[col_idx % len(self.COLOR_PALETTE)]
                symbol = "o"
                pen = pg.mkPen(color, width=1.0)
                brush = pg.mkBrush(color)
                bar_pen = pg.mkPen(color, width=1.5)
                bar_brush = pg.mkBrush(color.red(), color.green(), color.blue(), 140)

            if rep_style in ("bars", "columns", "spikes"):
                unique_xs = np.unique(xs)
                bar_ys = []
                for ux in unique_xs:
                    matching_ys = ys[xs == ux]
                    bar_ys.append(float(np.mean(matching_ys)))
                bar_ys = np.array(bar_ys)

                if len(unique_xs) > 1:
                    diffs = np.diff(np.sort(unique_xs))
                    bar_w = float(np.min(diffs)) * 0.4 if len(diffs) > 0 else 0.5
                else:
                    bar_w = 0.5
                if bar_w <= 0:
                    bar_w = 0.5

                bar_item = pg.BarGraphItem(
                    x=unique_xs,
                    height=bar_ys,
                    width=bar_w,
                    brush=bar_brush,
                    pen=bar_pen,
                )
                self.plot_widget.addItem(bar_item)
                if self.plot_widget.plotItem.legend is not None:
                    dummy = pg.PlotDataItem(pen=bar_pen, brush=bar_brush, name=g_title)
                    self.plot_widget.plotItem.legend.addItem(dummy, g_title)
            else:
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

                # Continuous regression curve overlay for scatter dataset in XY_SPIKES_CURVE
                curve_id = f"auto_curve_{col_idx}"
                if (
                    curve_id not in self.removed_layer_ids
                    and len(xs) >= 3
                    and not self.custom_analysis_layers
                    and self.current_result is None
                ):
                    try:
                        x_dense = np.linspace(np.min(xs), np.max(xs), 200)
                        poly_deg = min(2, len(np.unique(xs)) - 1)
                        if poly_deg >= 1:
                            coeffs = np.polyfit(xs, ys, deg=poly_deg)
                            y_dense = np.polyval(coeffs, x_dense)
                            curve_pen = pg.mkPen(color=QColor(20, 20, 20) if is_grayscale else color.darker(160), width=2.0)
                            curve_item = self.plot_widget.plot(
                                x_dense, y_dense, pen=curve_pen, name=f"{g_title} Fit"
                            )
                            self.analysis_layers[curve_id] = {
                                "name": f"{g_title} Fit",
                                "plot_items": [curve_item],
                            }
                    except Exception:
                        pass

        # Continuous fitted curve overlays
        self._render_analysis_overlays()

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
            if not painter.isActive():
                return
            scene = self.plot_widget.scene() if hasattr(self.plot_widget, "scene") else None
            if scene is not None:
                plot_item = getattr(self.plot_widget, "plotItem", None)
                if plot_item is None and hasattr(self.plot_widget, "getPlotItem"):
                    plot_item = self.plot_widget.getPlotItem()
                if plot_item is not None and hasattr(plot_item, "mapToScene") and hasattr(plot_item, "boundingRect"):
                    source_rect = plot_item.mapToScene(plot_item.boundingRect()).boundingRect()
                else:
                    source_rect = QRectF(self.plot_widget.rect())
                if source_rect.isEmpty() or not source_rect.isValid() or source_rect.width() < 10 or source_rect.height() < 10:
                    source_rect = QRectF(self.plot_widget.rect())
                scene.render(painter, QRectF(self.plot_widget.rect()), source_rect)
            else:
                self.plot_widget.render(painter)
        finally:
            if painter.isActive():
                painter.end()


GraphCanvasWidget = ScientificPlotCanvas
