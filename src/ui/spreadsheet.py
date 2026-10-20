"""
Scientific Spreadsheet Model and View for GemGraph.
Derived from BridgeCLR DataGridView presentation contracts (REA-CLR-VM-102).
"""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from PyQt6.QtCore import Qt, QAbstractTableModel, QModelIndex, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QBrush
from PyQt6.QtWidgets import QTableView, QMenu, QInputDialog, QMessageBox
from src.core.table import DataTable, TableType
from src.core.undo import UndoStack, CellEditCommand, CellExcludeCommand


class ScientificSpreadsheetModel(QAbstractTableModel):
    """
    Table model that maps a DataTable (with X column and replicated Y subcolumns)
    into a two-dimensional grid.
    """

    dataChangedSignal = pyqtSignal()

    def __init__(self, table: DataTable, undo_stack: Optional[UndoStack] = None) -> None:
        super().__init__()
        self.table = table
        self.undo_stack = undo_stack or UndoStack()

    def set_table(self, table: DataTable) -> None:
        self.beginResetModel()
        self.table = table
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return self.table.num_rows

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if self.table.table_type == TableType.MULTIPLE_VARIABLES:
            return len(self.table.variables)
        x_cols = 1 if self.table.x_data is not None else 0
        y_cols = len(self.table.y_group_titles) * self.table.num_replicates
        return x_cols + y_cols

    def _col_to_cell_coords(self, col: int) -> Tuple[str, int, int]:
        """
        Translates a view column index into (col_type, group_idx, subcol_idx).
        """
        has_x = self.table.x_data is not None
        if has_x:
            if col == 0:
                return ("X", 0, 0)
            col -= 1

        group_idx = col // self.table.num_replicates
        subcol_idx = col % self.table.num_replicates
        return ("Y", group_idx, subcol_idx)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.ForegroundRole:
            return QBrush(QColor(0, 0, 0))
        elif role == Qt.ItemDataRole.BackgroundRole:
            return QBrush(QColor(240, 240, 240))

        if orientation == Qt.Orientation.Vertical:
            if role == Qt.ItemDataRole.DisplayRole:
                rt = self.table.row_titles[section] if section < len(self.table.row_titles) else ""
                return rt if rt else str(section + 1)
        elif orientation == Qt.Orientation.Horizontal:
            if role == Qt.ItemDataRole.DisplayRole:
                if self.table.table_type == TableType.MULTIPLE_VARIABLES:
                    if 0 <= section < len(self.table.variables):
                        var = self.table.variables[section]
                        badge = "📊" if var.var_type.value == "Categorical" else ("🔔" if var.var_type.value == "Continuous" else "🏷️")
                        return f"{var.name} [{badge} {var.var_type.value}]"
                    return f"Var {section + 1}"
                col_type, g_idx, s_idx = self._col_to_cell_coords(section)
                if col_type == "X":
                    return self.table.x_title or "X"
                if 0 <= g_idx < len(self.table.y_group_titles):
                    g_title = self.table.y_group_titles[g_idx]
                else:
                    g_title = f"Group {g_idx + 1}"
                if self.table.table_type == TableType.NESTED:
                    nested_title = self.table.get_nested_subcolumn_title(g_idx, s_idx)
                    return f"{g_title}: {nested_title}"
                if self.table.num_replicates > 1:
                    return f"{g_title} [Y{s_idx + 1}]"
                elif g_title.startswith("Group "):
                    return f"{g_title} [Y{s_idx + 1}]"
                elif g_title in ("Y", ""):
                    return f"Y{s_idx + 1}"
                return g_title
        return None

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        return (
            Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemIsSelectable
            | Qt.ItemFlag.ItemIsEditable
        )

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None

        row = index.row()
        col = index.column()

        if self.table.table_type == TableType.MULTIPLE_VARIABLES:
            val = self.table.get_variable_value(row, col)
            if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    return ""
                if isinstance(val, float):
                    return f"{val:.4g}" if (abs(val) < 1e-4 or abs(val) >= 1e6) and abs(val) > 0 else f"{val:g}"
                return str(val)
            elif role == Qt.ItemDataRole.ForegroundRole:
                return QBrush(QColor(0, 0, 0))
            elif role == Qt.ItemDataRole.BackgroundRole:
                color = self.table.get_cell_color(row, "VAR", col, 0)
                if color:
                    return QBrush(QColor(color))
                if row % 2 == 1:
                    return QBrush(QColor(248, 249, 250))
                return QBrush(QColor(255, 255, 255))
            return None

        col_type, g_idx, s_idx = self._col_to_cell_coords(col)
        val = self.table.get_cell_value(row, col_type, g_idx, s_idx)
        is_ex = self.table.is_excluded(row, col_type, g_idx, s_idx)

        if role == Qt.ItemDataRole.DisplayRole:
            if np.isnan(val):
                return ""
            return f"{val:.4g}" if (abs(val) < 1e-4 or abs(val) >= 1e6) and abs(val) > 0 else f"{val:g}"

        elif role == Qt.ItemDataRole.EditRole:
            if np.isnan(val):
                return ""
            return str(val)

        elif role == Qt.ItemDataRole.FontRole:
            font = QFont()
            if is_ex:
                font.setStrikeOut(True)
                font.setItalic(True)
            return font

        elif role == Qt.ItemDataRole.ForegroundRole:
            if is_ex:
                return QBrush(QColor("#0055ff"))  # Blue italic for excluded values
            return QBrush(QColor(0, 0, 0))

        elif role == Qt.ItemDataRole.BackgroundRole:
            color = self.table.get_cell_color(row, col_type, g_idx, s_idx)
            if color:
                return QBrush(QColor(color))
            if row % 2 == 1:
                return QBrush(QColor(248, 249, 250))
            return QBrush(QColor(255, 255, 255))

        elif role == Qt.ItemDataRole.ToolTipRole:
            status = " [EXCLUDED]" if is_ex else ""
            return f"Row {row+1}, {self.headerData(col, Qt.Orientation.Horizontal)}{status}: {val}"

        return None

    def setData(self, index: QModelIndex, value, role: int = Qt.ItemDataRole.EditRole) -> bool:
        if not index.isValid() or role != Qt.ItemDataRole.EditRole:
            return False

        row = index.row()
        col = index.column()

        if self.table.table_type == TableType.MULTIPLE_VARIABLES:
            self.table.set_variable_value(row, col, value)
            self.dataChanged.emit(index, index, [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole])
            self.dataChangedSignal.emit()
            return True

        col_type, g_idx, s_idx = self._col_to_cell_coords(col)
        old_val = self.table.get_cell_value(row, col_type, g_idx, s_idx)

        new_val = np.nan
        if value is not None and str(value).strip() != "":
            try:
                new_val = float(str(value).strip())
            except ValueError:
                return False

        if (np.isnan(old_val) and np.isnan(new_val)) or old_val == new_val:
            return False

        try:
            cmd = CellEditCommand(
                table=self.table,
                row=row,
                col_type=col_type,
                group_idx=g_idx,
                subcol_idx=s_idx,
                old_val=old_val,
                new_val=new_val,
            )
            self.undo_stack.push(cmd)
            self.dataChanged.emit(index, index, [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole])
            self.dataChangedSignal.emit()
            return True
        except ValueError:
            return False


class ScientificSpreadsheetView(QTableView):
    """
    QTableView with contextual actions: exclude/include cells,
    subcolumn replicate manipulation, and undo/redo support.
    """

    cellExclusionToggled = pyqtSignal()

    def __init__(self, model: ScientificSpreadsheetModel, parent=None) -> None:
        super().__init__(parent)
        self.setModel(model)
        self.setAlternatingRowColors(True)
        self.horizontalHeader().setStretchLastSection(False)
        self.setSelectionMode(QTableView.SelectionMode.ExtendedSelection)

        # Explicit scientific spreadsheet styling (GraphPad standard) to guarantee
        # high-contrast readability under both light and dark OS desktop themes.
        self.setStyleSheet("""
            QTableView {
                background-color: #ffffff;
                alternate-background-color: #f8f9fa;
                color: #000000;
                gridline-color: #d0d0d0;
                selection-background-color: #0078d7;
                selection-color: #ffffff;
            }
            QTableView::item {
                padding: 4px 6px;
                color: #000000;
            }
            QTableView::item:selected {
                background-color: #0078d7;
                color: #ffffff;
            }
            QTableView QLineEdit {
                background-color: #ffffff;
                color: #000000;
                selection-background-color: #0078d7;
                selection-color: #ffffff;
                border: 1px solid #0078d7;
            }
            QHeaderView {
                background-color: #f0f0f0;
                color: #000000;
            }
            QHeaderView::section {
                background-color: #f0f0f0;
                color: #000000;
                border: 1px solid #d0d0d0;
                padding: 4px 8px;
                font-weight: bold;
                font-size: 11px;
            }
            QTableCornerButton::section {
                background-color: #f0f0f0;
                border: 1px solid #d0d0d0;
            }
        """)


        # Ensure column headers clearly display replicate groupings with proper margins
        self.horizontalHeader().setDefaultSectionSize(110)
        self.horizontalHeader().setMinimumSectionSize(80)
        self.verticalHeader().setDefaultSectionSize(24)

    def get_model(self) -> ScientificSpreadsheetModel:
        return self.model()  # type: ignore

    def contextMenuEvent(self, event) -> None:
        menu = QMenu(self)
        color_menu = menu.addMenu("Highlight Cells")
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
        color_actions = {}
        for c_name, c_hex in colors:
            color_actions[color_menu.addAction(c_name)] = c_hex

        menu.addSeparator()
        exclude_action = menu.addAction("Exclude Selected Values")
        include_action = menu.addAction("Include Selected Values")

        menu.addSeparator()
        add_rows_action = menu.addAction("Add 10 Rows")
        add_group_action = menu.addAction("Add Y Column Group")
        set_reps_action = menu.addAction("Set Replicate Count...")

        action = menu.exec(event.globalPos())
        if action == exclude_action:
            self._set_selected_exclusion(True)
        elif action == include_action:
            self._set_selected_exclusion(False)
        elif action in color_actions:
            self._set_selected_color(color_actions[action])
        elif action == add_rows_action:
            self._add_rows(10)
        elif action == add_group_action:
            self._add_group()
        elif action == set_reps_action:
            self._prompt_set_replicates()

    def _set_selected_color(self, color_hex: Optional[str]) -> None:
        model = self.get_model()
        indexes = self.selectedIndexes()
        if not indexes:
            return
        for idx in indexes:
            row = idx.row()
            col = idx.column()
            if model.table.table_type == TableType.MULTIPLE_VARIABLES:
                model.table.set_cell_color(row, "VAR", col, 0, color_hex)
            else:
                col_type, g_idx, s_idx = model._col_to_cell_coords(col)
                model.table.set_cell_color(row, col_type, g_idx, s_idx, color_hex)
        model.dataChanged.emit(
            indexes[0], indexes[-1], [Qt.ItemDataRole.BackgroundRole]
        )
        model.dataChangedSignal.emit()

    def _set_selected_exclusion(self, excluded: bool) -> None:
        model = self.get_model()
        indexes = self.selectedIndexes()
        if not indexes:
            return

        for idx in indexes:
            row = idx.row()
            col = idx.column()
            col_type, g_idx, s_idx = model._col_to_cell_coords(col)
            curr_ex = model.table.is_excluded(row, col_type, g_idx, s_idx)
            if curr_ex != excluded:
                cmd = CellExcludeCommand(
                    table=model.table,
                    row=row,
                    col_type=col_type,
                    group_idx=g_idx,
                    subcol_idx=s_idx,
                    old_excluded=curr_ex,
                    new_excluded=excluded,
                )
                model.undo_stack.push(cmd)

        model.dataChanged.emit(
            indexes[0], indexes[-1],
            [Qt.ItemDataRole.FontRole, Qt.ItemDataRole.ForegroundRole]
        )
        model.dataChangedSignal.emit()
        self.cellExclusionToggled.emit()

    def _add_rows(self, count: int) -> None:
        model = self.get_model()
        model.beginResetModel()
        model.table.add_row(count)
        model.endResetModel()
        model.dataChangedSignal.emit()

    def _add_group(self) -> None:
        model = self.get_model()
        model.beginResetModel()
        model.table.add_y_group()
        model.endResetModel()
        model.dataChangedSignal.emit()

    def _prompt_set_replicates(self) -> None:
        model = self.get_model()
        reps, ok = QInputDialog.getInt(
            self,
            "Replicate Count",
            "Number of subcolumn replicates per group:",
            model.table.num_replicates,
            1,
            64,
        )
        if ok and reps != model.table.num_replicates:
            model.beginResetModel()
            model.table.set_replicates(reps)
            model.endResetModel()
            model.dataChangedSignal.emit()


# Alias for explicit ReplicateSpreadsheet references
ReplicateSpreadsheet = ScientificSpreadsheetView

