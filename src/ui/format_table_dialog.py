"""
Format Data Table Dialog for GemGraph.
Allows configuring the 8 scientific table formats, column/variable titles,
and variable types (Categorical, Continuous, Label) with level reordering.
"""

from __future__ import annotations
from typing import Optional, List, Dict
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QTabWidget,
    QWidget,
    QLabel,
    QComboBox,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QDialogButtonBox,
    QGroupBox,
    QFormLayout,
    QHeaderView,
    QAbstractItemView,
)
from src.core.table import DataTable, TableType, VariableType, VariableMetadata, XFormat, YFormat


class FormatDataTableDialog(QDialog):
    """
    Three-tab dialog for configuring:
    1. Table Format (8 scientific table paradigms)
    2. Variable Names (Column / group titles)
    3. Variable Types (Categorical, Continuous, Label; level breakdown & reordering)
    """

    FORMAT_DESCRIPTIONS = {
        TableType.XY.value: "XY: Single X column (numbers/time) and multiple Y groups with replicates or Mean/SD/SEM.",
        TableType.COLUMN.value: "Column: Unpaired or paired treatment groups without X column. Excluded points in blue italics.",
        TableType.GROUPED.value: "Grouped: Two grouping factors (rows = Factor 1, columns = Factor 2) with replicate subcolumns.",
        TableType.CONTINGENCY.value: "Contingency: Rows = treatments, columns = outcomes. Non-negative integer counts only.",
        TableType.SURVIVAL.value: "Survival: X = elapsed time (days), Y = outcome code (1 = death/event, 0 = censored).",
        TableType.PARTS_OF_WHOLE.value: "Parts of Whole: Single column representing slices/fractions of a whole (pie/donut charts).",
        TableType.MULTIPLE_VARIABLES.value: "Multiple Variables: Each row = case/observation; columns have typed metadata (Categorical, Continuous, Label).",
        TableType.NESTED.value: "Nested: Hierarchical replication where subcolumns represent nested physical units.",
    }

    def __init__(self, table: DataTable, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.table = table
        self.setWindowTitle(f"Format Data Table — {table.title}")
        self.resize(550, 480)

        main_layout = QVBoxLayout(self)

        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        self._init_format_tab()
        self._init_names_tab()
        self._init_types_tab()

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)

    def _init_format_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        form_group = QGroupBox("Scientific Table Paradigm")
        form_layout = QFormLayout(form_group)

        self.format_combo = QComboBox()
        for tt in TableType:
            self.format_combo.addItem(tt.value, tt)
        idx = self.format_combo.findData(self.table.table_type)
        if idx >= 0:
            self.format_combo.setCurrentIndex(idx)
        self.format_combo.currentIndexChanged.connect(self._on_format_changed)
        form_layout.addRow("Table Format:", self.format_combo)

        self.reps_spin = QSpinBox()
        self.reps_spin.setRange(1, 64)
        self.reps_spin.setValue(self.table.num_replicates)
        form_layout.addRow("Replicates per Group:", self.reps_spin)

        self.desc_label = QLabel()
        self.desc_label.setWordWrap(True)
        self.desc_label.setStyleSheet("color: #444444; font-style: italic; padding: 4px;")
        form_layout.addRow("Description:", self.desc_label)

        layout.addWidget(form_group)
        layout.addStretch()
        self.tab_widget.addTab(tab, "Table Format")
        self._update_format_description()

    def _update_format_description(self) -> None:
        cur_type = self.format_combo.currentData()
        if cur_type:
            self.desc_label.setText(self.FORMAT_DESCRIPTIONS.get(cur_type.value, ""))

    def _on_format_changed(self) -> None:
        self._update_format_description()

    def _init_names_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        layout.addWidget(QLabel("Column and Treatment Group Titles:"))
        self.names_table = QTableWidget()
        self.names_table.setColumnCount(2)
        self.names_table.setHorizontalHeaderLabels(["Group / Column", "Title"])
        self.names_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)

        titles = self.table.y_group_titles
        if self.table.table_type == TableType.MULTIPLE_VARIABLES and self.table.variables:
            titles = [v.name for v in self.table.variables]

        self.names_table.setRowCount(len(titles))
        for r, title in enumerate(titles):
            idx_item = QTableWidgetItem(f"Column {r + 1}")
            idx_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.names_table.setItem(r, 0, idx_item)
            title_item = QTableWidgetItem(title)
            self.names_table.setItem(r, 1, title_item)

        layout.addWidget(self.names_table)
        self.tab_widget.addTab(tab, "Variable Names")

    def _init_types_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        layout.addWidget(QLabel("Variable Types (Multiple Variables Tables):"))

        # Split into upper table of variables + lower level reordering
        self.types_table = QTableWidget()
        self.types_table.setColumnCount(3)
        self.types_table.setHorizontalHeaderLabels(["Variable", "Type", "Levels"])
        self.types_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.types_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.types_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

        variables = self.table.variables
        if not variables:
            # Fallback if not yet initialized
            variables = [
                VariableMetadata(t, VariableType.CONTINUOUS)
                for t in self.table.y_group_titles
            ]

        self.type_combos: List[QComboBox] = []
        self.types_table.setRowCount(len(variables))
        for r, var in enumerate(variables):
            name_item = QTableWidgetItem(var.name)
            name_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self.types_table.setItem(r, 0, name_item)

            combo = QComboBox()
            for vt in VariableType:
                combo.addItem(vt.value, vt)
            idx = combo.findData(var.var_type)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            combo.currentIndexChanged.connect(self._on_type_selection_changed)
            self.types_table.setCellWidget(r, 1, combo)
            self.type_combos.append(combo)

            count_str = str(len(var.levels)) if var.var_type == VariableType.CATEGORICAL else "N/A"
            lvl_item = QTableWidgetItem(count_str)
            lvl_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.types_table.setItem(r, 2, lvl_item)

        self.types_table.itemSelectionChanged.connect(self._on_type_row_selected)
        layout.addWidget(self.types_table)

        # Level management box
        self.level_box = QGroupBox("Categorical Level Breakdown & Order")
        lvl_layout = QHBoxLayout(self.level_box)

        self.level_list = QListWidget()
        self.level_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        lvl_layout.addWidget(self.level_list)

        btn_layout = QVBoxLayout()
        self.btn_up = QPushButton("Move Up")
        self.btn_up.clicked.connect(self._move_level_up)
        btn_layout.addWidget(self.btn_up)

        self.btn_down = QPushButton("Move Down")
        self.btn_down.clicked.connect(self._move_level_down)
        btn_layout.addWidget(self.btn_down)
        btn_layout.addStretch()
        lvl_layout.addLayout(btn_layout)

        layout.addWidget(self.level_box)
        self.tab_widget.addTab(tab, "Variable Types")

        if len(variables) > 0:
            self.types_table.selectRow(0)

    def _on_type_selection_changed(self) -> None:
        self._on_type_row_selected()

    def _on_type_row_selected(self) -> None:
        selected_rows = self.types_table.selectionModel().selectedRows()
        if not selected_rows:
            return
        row = selected_rows[0].row()
        if row < 0 or row >= len(self.type_combos):
            return

        cur_type = self.type_combos[row].currentData()
        self.level_list.clear()

        if cur_type == VariableType.CATEGORICAL:
            self.level_box.setEnabled(True)
            var_levels = []
            if row < len(self.table.variables):
                var_levels = self.table.get_levels(row)
                counts = self.table.get_variable_level_counts(row)
            else:
                counts = {}

            for lvl in var_levels:
                cnt = counts.get(lvl, 0)
                item = QListWidgetItem(f"{lvl} (n = {cnt})")
                item.setData(Qt.ItemDataRole.UserRole, lvl)
                self.level_list.addItem(item)
        else:
            self.level_box.setEnabled(False)

    def _move_level_up(self) -> None:
        cur_row = self.level_list.currentRow()
        if cur_row > 0:
            item = self.level_list.takeItem(cur_row)
            self.level_list.insertItem(cur_row - 1, item)
            self.level_list.setCurrentRow(cur_row - 1)

    def _move_level_down(self) -> None:
        cur_row = self.level_list.currentRow()
        if 0 <= cur_row < self.level_list.count() - 1:
            item = self.level_list.takeItem(cur_row)
            self.level_list.insertItem(cur_row + 1, item)
            self.level_list.setCurrentRow(cur_row + 1)

    def _on_accept(self) -> None:
        # Apply table type and replicates
        new_type = self.format_combo.currentData()
        if new_type and new_type != self.table.table_type:
            self.table.table_type = new_type
            if new_type in (TableType.XY, TableType.SURVIVAL) and self.table.x_data is None:
                import numpy as np
                self.table.x_title = "Elapsed Time (days)" if new_type == TableType.SURVIVAL else "X"
                self.table.x_data = np.full(self.table.num_rows, np.nan, dtype=np.float64)
            elif new_type not in (TableType.XY, TableType.SURVIVAL):
                self.table.x_data = None
                self.table.x_title = ""

        new_reps = self.reps_spin.value()
        if new_reps != self.table.num_replicates and self.table.table_type != TableType.MULTIPLE_VARIABLES:
            self.table.set_replicates(new_reps)

        # Apply names
        for r in range(self.names_table.rowCount()):
            item = self.names_table.item(r, 1)
            if item:
                val = item.text().strip()
                if val:
                    if self.table.table_type == TableType.MULTIPLE_VARIABLES and r < len(self.table.variables):
                        self.table.variables[r].name = val
                    elif r < len(self.table.y_group_titles):
                        self.table.y_group_titles[r] = val

        # Apply variable types and level order
        for r, combo in enumerate(self.type_combos):
            if r < len(self.table.variables):
                self.table.variables[r].var_type = combo.currentData()

        # Update level order for selected categorical variable
        selected_rows = self.types_table.selectionModel().selectedRows()
        if selected_rows:
            sel_r = selected_rows[0].row()
            if 0 <= sel_r < len(self.table.variables) and self.table.variables[sel_r].var_type == VariableType.CATEGORICAL:
                new_order = []
                for i in range(self.level_list.count()):
                    lvl = self.level_list.item(i).data(Qt.ItemDataRole.UserRole)
                    if lvl:
                        new_order.append(lvl)
                if new_order:
                    self.table.reorder_levels(sel_r, new_order)

        self.accept()
