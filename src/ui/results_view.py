"""
Multi-Tab Results Architecture and Green/Red Grid Semantics for GemGraph.
Implements:
- Sub-tabs above results tables ([Table of results], [Curve], [Residual plot], [Outliers])
- Tab visibility drop-down menu on right edge of tab bar
- Green Grid (#a3d9a5) vs. Red Grid (#f5c6cb) visual and functional semantics
- Freeze Sheet (snowflake icon) and Orphaned status indicators
- Cell color coding
"""

from __future__ import annotations
from typing import Optional, List, Dict, Any
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QBrush, QAction
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QPushButton,
    QMenu,
    QHeaderView,
)
from src.core.sheet import ResultSheet


class ResultsTableWidget(QTableWidget):
    """Table widget with Green/Red grid semantics and cell highlight menu."""

    def __init__(self, grid_type: str = "red", parent=None) -> None:
        super().__init__(parent)
        self.grid_type = grid_type
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.apply_grid_style(grid_type)

    def apply_grid_style(self, grid_type: str) -> None:
        self.grid_type = grid_type
        if grid_type == "green":
            gridline_color = "#a3d9a5"  # Light green gridlines
            header_border = "#81c784"
            selection_color = "#2e7d32"
        else:
            gridline_color = "#f5c6cb"  # Light red/coral gridlines
            header_border = "#e57373"
            selection_color = "#c62828"

        self.setStyleSheet(f"""
            QTableWidget {{
                background-color: #ffffff;
                alternate-background-color: #fdfdfd;
                color: #000000;
                gridline-color: {gridline_color};
                selection-background-color: {selection_color};
                selection-color: #ffffff;
            }}
            QTableWidget::item {{
                color: #000000;
                padding: 4px 6px;
            }}
            QHeaderView {{
                background-color: #f8f9fa;
                color: #000000;
            }}
            QHeaderView::section {{
                background-color: #f8f9fa;
                color: #000000;
                border: 1px solid {header_border};
                padding: 4px 8px;
                font-weight: bold;
            }}
        """)

    def contextMenuEvent(self, event) -> None:
        menu = QMenu(self)
        color_menu = menu.addMenu("Highlight Cell(s)")
        colors = [
            ("Yellow", "#fff3cd"),
            ("Red", "#f8d7da"),
            ("Blue", "#cce5ff"),
            ("Green", "#d4edda"),
            ("Purple", "#e2d9f3"),
            ("Orange", "#ffe8cc"),
            ("Tan", "#f5ebe0"),
            ("Gray", "#e9ecef"),
            ("Clear Highlight", None),
        ]
        for c_name, c_hex in colors:
            act = color_menu.addAction(c_name)
            act.triggered.connect(lambda checked, col=c_hex: self._set_selected_cells_color(col))

        menu.exec(event.globalPos())

    def _set_selected_cells_color(self, color_hex: Optional[str]) -> None:
        selected_items = self.selectedItems()
        for item in selected_items:
            if color_hex:
                item.setBackground(QBrush(QColor(color_hex)))
            else:
                item.setBackground(QBrush(QColor(255, 255, 255)))


class ResultTabsWidget(QWidget):
    """
    Complete Multi-Tab Results presentation widget with:
    - Sub-tabs for results
    - Tab visibility drop-down menu on right edge
    - Green vs Red grid indicator badge & button controls
    - Freeze Sheet toggle
    - Orphaned indicator
    """

    freezeToggled = pyqtSignal(bool)
    analyzeRequested = pyqtSignal()
    newGraphRequested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.current_result: Optional[ResultSheet] = None
        self.table_widgets: List[ResultsTableWidget] = []

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(6, 6, 6, 6)
        self.layout.setSpacing(6)

        # Header toolbar
        top_bar = QHBoxLayout()

        self.title_label = QLabel("Analysis Results")
        font = self.title_label.font()
        font.setBold(True)
        font.setPointSize(12)
        self.title_label.setFont(font)
        top_bar.addWidget(self.title_label)

        # Orphaned badge
        self.orphan_badge = QLabel("⚠️ Orphaned (Parent Deleted)")
        self.orphan_badge.setStyleSheet(
            "background-color: #fff3cd; color: #856404; font-weight: bold; padding: 2px 8px; border-radius: 4px; border: 1px solid #ffeeba;"
        )
        self.orphan_badge.setVisible(False)
        top_bar.addWidget(self.orphan_badge)

        # Grid type badge
        self.grid_badge = QLabel("Red Grid (Summary Only)")
        self.grid_badge.setStyleSheet(
            "background-color: #f8d7da; color: #721c24; font-weight: bold; padding: 2px 8px; border-radius: 4px; border: 1px solid #f5c6cb;"
        )
        top_bar.addWidget(self.grid_badge)

        top_bar.addStretch()

        # Downstream action buttons (Active only for Green Grid)
        self.btn_analyze = QPushButton("Analyze...")
        self.btn_analyze.clicked.connect(self.analyzeRequested.emit)
        top_bar.addWidget(self.btn_analyze)

        self.btn_new_graph = QPushButton("New Graph")
        self.btn_new_graph.clicked.connect(self.newGraphRequested.emit)
        top_bar.addWidget(self.btn_new_graph)

        # Freeze Sheet button
        self.btn_freeze = QPushButton("❄️ Freeze Sheet")
        self.btn_freeze.setCheckable(True)
        self.btn_freeze.clicked.connect(self._on_freeze_clicked)
        top_bar.addWidget(self.btn_freeze)

        self.layout.addLayout(top_bar)

        # Tab widget and right-edge visibility button
        self.tab_widget = QTabWidget()
        self.tab_widget.setDocumentMode(True)

        # Visibility tool button in the corner
        self.btn_tab_visibility = QToolButton()
        self.btn_tab_visibility.setText("▼ Tabs")
        self.btn_tab_visibility.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.tab_visibility_menu = QMenu(self.btn_tab_visibility)
        self.btn_tab_visibility.setMenu(self.tab_visibility_menu)
        self.tab_widget.setCornerWidget(self.btn_tab_visibility, Qt.Corner.TopRightCorner)

        self.layout.addWidget(self.tab_widget)
        self.tab_widget.currentChanged.connect(self._on_tab_changed)

    def display_result(self, result: ResultSheet) -> None:
        self.current_result = result
        self.title_label.setText(f"{result.analysis_type} — {result.title}")

        # Update Freeze state
        self.btn_freeze.setChecked(result.is_frozen)
        self._update_freeze_button_style(result.is_frozen)

        # Update Orphan state
        self.orphan_badge.setVisible(result.is_orphaned)

        # Build sub-tabs
        self.tab_widget.clear()
        self.table_widgets.clear()
        self.tab_visibility_menu.clear()

        # Gather tabs: if result.tabs present use them, else generate standard tabs
        tabs_to_show = list(result.tabs)
        if not tabs_to_show:
            if result.summary_tables:
                for tbl in result.summary_tables:
                    tabs_to_show.append({
                        "name": tbl.get("title", "Table of results"),
                        "grid_type": result.grid_type,
                        "headers": tbl.get("headers", []),
                        "rows": tbl.get("rows", []),
                        "visible": True,
                    })
            elif result.raw_result_data:
                raw = result.raw_result_data
                if "x_dense" in raw and "y_dense" in raw:
                    tabs_to_show.append({
                        "name": "Curve Coordinates",
                        "grid_type": "green",
                        "headers": ["X (Curve)", "Y (Fit)"],
                        "rows": [
                            [round(x, 4), round(y, 4)]
                            for x, y in zip(raw["x_dense"][:50], raw["y_dense"][:50])
                        ],
                        "visible": True,
                    })

        # Populate tabs
        for idx, t in enumerate(tabs_to_show):
            t_name = t.get("name", f"Tab {idx + 1}")
            g_type = t.get("grid_type", result.grid_type)
            tbl_widget = ResultsTableWidget(grid_type=g_type)

            headers = t.get("headers", [])
            rows = t.get("rows", [])
            tbl_widget.setRowCount(len(rows))
            tbl_widget.setColumnCount(len(headers))
            tbl_widget.setHorizontalHeaderLabels(headers)

            for r_idx, row in enumerate(rows):
                for c_idx, cell in enumerate(row):
                    item = QTableWidgetItem(str(cell) if cell is not None else "")
                    item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                    tbl_widget.setItem(r_idx, c_idx, item)

            self.table_widgets.append(tbl_widget)
            tab_index = self.tab_widget.addTab(tbl_widget, t_name)

            # Add checkable visibility action in menu
            act = QAction(t_name, self.tab_visibility_menu)
            act.setCheckable(True)
            act.setChecked(t.get("visible", True))
            act.toggled.connect(
                lambda checked, w=tbl_widget: self._toggle_tab_visibility(w, checked)
            )
            self.tab_visibility_menu.addAction(act)

            if not t.get("visible", True):
                self.tab_widget.removeTab(self.tab_widget.indexOf(tbl_widget))

        self._update_grid_semantics()

    def _toggle_tab_visibility(self, widget: ResultsTableWidget, visible: bool) -> None:
        idx = self.tab_widget.indexOf(widget)
        if visible and idx < 0:
            # Re-add tab
            orig_idx = self.table_widgets.index(widget)
            tab_title = f"Tab {orig_idx + 1}"
            if self.current_result and orig_idx < len(self.current_result.tabs):
                tab_title = self.current_result.tabs[orig_idx].get("name", tab_title)
            self.tab_widget.insertTab(orig_idx, widget, tab_title)
        elif not visible and idx >= 0:
            self.tab_widget.removeTab(idx)

    def _on_tab_changed(self) -> None:
        self._update_grid_semantics()

    def _update_grid_semantics(self) -> None:
        curr_widget = self.tab_widget.currentWidget()
        if isinstance(curr_widget, ResultsTableWidget):
            is_green = curr_widget.grid_type == "green"
        elif self.current_result:
            is_green = self.current_result.grid_type == "green"
        else:
            is_green = False

        if is_green:
            self.grid_badge.setText("Green Grid (Graphable / Analyzable)")
            self.grid_badge.setStyleSheet(
                "background-color: #d4edda; color: #155724; font-weight: bold; padding: 2px 8px; border-radius: 4px; border: 1px solid #a3d9a5;"
            )
            self.btn_analyze.setEnabled(True)
            self.btn_new_graph.setEnabled(True)
        else:
            self.grid_badge.setText("Red Grid (Formatted Summary)")
            self.grid_badge.setStyleSheet(
                "background-color: #f8d7da; color: #721c24; font-weight: bold; padding: 2px 8px; border-radius: 4px; border: 1px solid #f5c6cb;"
            )
            self.btn_analyze.setEnabled(False)
            self.btn_new_graph.setEnabled(False)

    def _on_freeze_clicked(self, checked: bool) -> None:
        if self.current_result:
            self.current_result.is_frozen = checked
            self._update_freeze_button_style(checked)
            self.freezeToggled.emit(checked)

    def _update_freeze_button_style(self, frozen: bool) -> None:
        if frozen:
            self.btn_freeze.setText("❄️ Frozen")
            self.btn_freeze.setStyleSheet(
                "background-color: #cce5ff; color: #004085; font-weight: bold; border: 1px solid #b8daff;"
            )
        else:
            self.btn_freeze.setText("❄️ Freeze Sheet")
            self.btn_freeze.setStyleSheet("")
