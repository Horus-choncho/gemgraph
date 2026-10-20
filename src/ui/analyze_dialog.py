"""
Analyze Data Dialog for GemGraph.
Two-pane selector: Search bar + hierarchical analysis tree on the left;
source table and dataset selector checklist ([x] A:Control, [x] B:Treated) on the right.
Includes the Interpret Analysis Checklist.
"""

from __future__ import annotations
from typing import Optional, List, Dict, Any, Tuple
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QWidget,
    QLabel,
    QLineEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QComboBox,
    QCheckBox,
    QPushButton,
    QDialogButtonBox,
    QGroupBox,
    QScrollArea,
    QMessageBox,
)
from src.core.document import PrismDocument
from src.core.sheet import DataSheet, SheetType
from src.core.table import TableType


class AnalysisChecklistDialog(QDialog):
    """Prism-style Analysis Checklist to assist experimental interpretation."""

    CHECKLISTS = {
        "General": [
            "Are the observations independent (separate animals/subjects)?",
            "Are values entered on appropriate arithmetic or logarithmic scale?",
            "Were missing values or outliers inspected prior to model fitting?",
            "Is the sample size sufficient (N ≥ 3 per group)?",
        ],
        "Nonlinear Regression": [
            "Did the fit converge cleanly without hitting parameter boundaries?",
            "Is the R² value and Sy.x consistent with experimental noise?",
            "Are residuals randomly distributed around zero across all X values?",
            "Are parameter standard errors and confidence intervals reasonably tight?",
            "Are parameter dependencies < 0.99 (no severe parameter collinearity)?",
        ],
        "Group Comparisons (ANOVA)": [
            "Do data within groups follow approximately Gaussian distributions?",
            "Are variances across groups approximately homogeneous?",
            "Was an appropriate post-hoc multiple comparisons test selected (Dunnett / Tukey)?",
            "For repeated measures, was the Geisser-Greenhouse sphericity correction applied?",
        ],
    }

    def __init__(self, analysis_name: str = "General", parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Analysis Checklist — {analysis_name}")
        self.resize(500, 360)

        layout = QVBoxLayout(self)

        title = QLabel(f"Interpretation Checklist: {analysis_name}")
        font = title.font()
        font.setBold(True)
        font.setPointSize(11)
        title.setFont(font)
        layout.addWidget(title)

        items = self.CHECKLISTS.get(analysis_name, self.CHECKLISTS["General"])
        for text in items:
            cb = QCheckBox(text)
            cb.setStyleSheet("padding: 4px; font-size: 11px;")
            layout.addWidget(cb)

        layout.addStretch()

        btn = QPushButton("Close")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)


class AnalyzeDataDialog(QDialog):
    """
    Complete Analyze Data Dialog matching GraphPad Prism 10:
    - Left Pane: Dynamic search bar + hierarchical tree organized by objective
    - Right Pane: "Analyze which data sets?" panel with table source dropdown,
      column selection checkboxes ([x] A:Group A, [x] B:Group B), and Select/Deselect All buttons.
    """

    ANALYSIS_TREE: Dict[str, List[str]] = {
        "Data Processing": [
            "Transform...",
            "Normalize...",
            "Prune Rows...",
            "Remove Baseline & Column Math...",
            "Fraction of Total...",
            "Transpose...",
            "Extract and Rearrange...",
        ],
        "Data Exploration and Summary": [
            "Identify Outliers (ROUT / Grubbs)...",
            "Descriptive Statistics / Summary...",
            "Correlation Matrix...",
            "Frequency Distribution...",
        ],
        "Regression and Curves": [
            "Nonlinear Regression (Curve Fit)...",
            "Linear Regression...",
            "Multiple Regression...",
            "Logistic Regression...",
            "Interpolate from Standard Curve...",
            "Principal Component Analysis (PCA)...",
        ],
        "Group Comparison": [
            "t-test (and Nonparametric)...",
            "One-Way ANOVA & Multiple Comparisons...",
            "Two-Way ANOVA (Grouped Data)...",
            "Multiple t-tests...",
            "Nested t-test / Nested One-Way ANOVA...",
            "Chi-Square / Fisher's Exact (Contingency)...",
            "Chi-Square Goodness of Fit...",
        ],
        "Survival Analysis": [
            "Kaplan-Meier Survival Analysis...",
        ],
        "Simulation": [
            "Simulate XY Data...",
            "Simulate Column Data...",
        ],
    }

    def __init__(self, document: PrismDocument, parent=None) -> None:
        super().__init__(parent)
        self.document = document
        self.setWindowTitle("Analyze Data")
        self.resize(780, 520)

        main_layout = QVBoxLayout(self)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ---------------- Left Pane: Tree & Search ----------------
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        left_layout.addWidget(QLabel("Select an analysis:"))

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search built-in analyses...")
        self.search_edit.textChanged.connect(self._filter_tree)
        left_layout.addWidget(self.search_edit)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.itemSelectionChanged.connect(self._on_tree_selection_changed)
        self.tree.itemDoubleClicked.connect(self._on_tree_double_clicked)
        left_layout.addWidget(self.tree)

        self._populate_tree()
        splitter.addWidget(left_widget)

        # ---------------- Right Pane: Data Set Selector ----------------
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        dataset_group = QGroupBox("Analyze which data sets?")
        dataset_layout = QVBoxLayout(dataset_group)

        # Source Table dropdown
        dataset_layout.addWidget(QLabel("Data table:"))
        self.table_combo = QComboBox()
        self._populate_table_combo()
        self.table_combo.currentIndexChanged.connect(self._on_table_changed)
        dataset_layout.addWidget(self.table_combo)

        dataset_layout.addWidget(QLabel("Select columns / treatment groups to include:"))

        # Checkboxes inside scroll area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.checkbox_container = QWidget()
        self.checkbox_layout = QVBoxLayout(self.checkbox_container)
        self.checkbox_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll_area.setWidget(self.checkbox_container)
        dataset_layout.addWidget(self.scroll_area)

        # Select / Deselect All buttons
        btn_sel_layout = QHBoxLayout()
        self.btn_select_all = QPushButton("Select All")
        self.btn_select_all.clicked.connect(self._select_all_columns)
        btn_sel_layout.addWidget(self.btn_select_all)

        self.btn_deselect_all = QPushButton("Deselect All")
        self.btn_deselect_all.clicked.connect(self._deselect_all_columns)
        btn_sel_layout.addWidget(self.btn_deselect_all)
        dataset_layout.addLayout(btn_sel_layout)

        right_layout.addWidget(dataset_group)

        # Interpret Section: Analysis Checklist Button
        interpret_group = QGroupBox("Interpret & Quality Control")
        interpret_layout = QHBoxLayout(interpret_group)
        self.btn_checklist = QPushButton("📋 Analysis Checklist...")
        self.btn_checklist.clicked.connect(self._open_checklist)
        interpret_layout.addWidget(self.btn_checklist)
        right_layout.addWidget(interpret_group)

        splitter.addWidget(right_widget)
        splitter.setSizes([380, 400])
        main_layout.addWidget(splitter)

        # Dialog Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)

        # Selection state
        self.column_checkboxes: List[QCheckBox] = []
        self._populate_column_checkboxes()

        self.selected_analysis: Optional[str] = None
        self.selected_table_id: Optional[str] = None
        self.selected_columns: List[int] = []

    def _populate_tree(self) -> None:
        self.tree.clear()
        for category, analyses in self.ANALYSIS_TREE.items():
            cat_item = QTreeWidgetItem([f"📁 {category}"])
            font = cat_item.font(0)
            font.setBold(True)
            cat_item.setFont(0, font)

            for a in analyses:
                leaf = QTreeWidgetItem([a])
                leaf.setData(0, Qt.ItemDataRole.UserRole, a)
                cat_item.addChild(leaf)

            self.tree.addTopLevelItem(cat_item)
        self.tree.expandAll()

    def _filter_tree(self, text: str) -> None:
        query = text.strip().lower()
        for i in range(self.tree.topLevelItemCount()):
            cat_item = self.tree.topLevelItem(i)
            cat_visible = False
            for j in range(cat_item.childCount()):
                leaf = cat_item.child(j)
                matches = query in leaf.text(0).lower()
                leaf.setHidden(not matches)
                if matches:
                    cat_visible = True
            cat_item.setHidden(not cat_visible)

    def _populate_table_combo(self) -> None:
        self.table_combo.clear()
        data_sheets = self.document.get_sheets_by_type(SheetType.DATA)
        for s in data_sheets:
            self.table_combo.addItem(s.title, s.sheet_id)
        if self.document.active_sheet_id:
            idx = self.table_combo.findData(self.document.active_sheet_id)
            if idx >= 0:
                self.table_combo.setCurrentIndex(idx)

    def _on_table_changed(self) -> None:
        self._populate_column_checkboxes()

    def _populate_column_checkboxes(self) -> None:
        # Clear existing checkboxes
        for cb in self.column_checkboxes:
            self.checkbox_layout.removeWidget(cb)
            cb.deleteLater()
        self.column_checkboxes.clear()

        sheet_id = self.table_combo.currentData()
        sheet = self.document.get_sheet(sheet_id) if sheet_id else None
        if not isinstance(sheet, DataSheet):
            return

        table = sheet.table
        titles = table.y_group_titles
        if table.table_type == TableType.MULTIPLE_VARIABLES and table.variables:
            titles = [v.name for v in table.variables]

        for idx, title in enumerate(titles):
            letter = chr(ord("A") + idx) if idx < 26 else f"Col{idx + 1}"
            cb = QCheckBox(f"{letter}: {title}")
            cb.setChecked(True)
            self.checkbox_layout.addWidget(cb)
            self.column_checkboxes.append(cb)

    def _select_all_columns(self) -> None:
        for cb in self.column_checkboxes:
            cb.setChecked(True)

    def _deselect_all_columns(self) -> None:
        for cb in self.column_checkboxes:
            cb.setChecked(False)

    def _on_tree_selection_changed(self) -> None:
        pass

    def _on_tree_double_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        val = item.data(0, Qt.ItemDataRole.UserRole)
        if val:
            self._on_accept()

    def _open_checklist(self) -> None:
        current = self.tree.currentItem()
        analysis_name = current.data(0, Qt.ItemDataRole.UserRole) if current else "General"
        # Determine category
        cat = "General"
        if analysis_name:
            if "Nonlinear" in analysis_name or "Curve" in analysis_name:
                cat = "Nonlinear Regression"
            elif "ANOVA" in analysis_name or "Comparison" in analysis_name or "t-test" in analysis_name:
                cat = "Group Comparisons (ANOVA)"
        dlg = AnalysisChecklistDialog(cat, self)
        dlg.exec()

    def _on_accept(self) -> None:
        current = self.tree.currentItem()
        if not current or not current.data(0, Qt.ItemDataRole.UserRole):
            QMessageBox.warning(self, "Select Analysis", "Please select an analysis from the tree.")
            return

        self.selected_analysis = current.data(0, Qt.ItemDataRole.UserRole)
        self.selected_table_id = self.table_combo.currentData()
        self.selected_columns = [
            idx for idx, cb in enumerate(self.column_checkboxes) if cb.isChecked()
        ]

        if not self.selected_columns:
            QMessageBox.warning(self, "No Columns Selected", "Please select at least one column to analyze.")
            return

        self.accept()

    def get_selection(self) -> Tuple[Optional[str], Optional[str], List[int]]:
        """Returns (selected_analysis, selected_table_id, selected_columns)."""
        return self.selected_analysis, self.selected_table_id, self.selected_columns
