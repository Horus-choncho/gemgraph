"""
Scientific Spreadsheet Model and View for GemGraph.
Derived from BridgeCLR DataGridView presentation contracts (REA-CLR-VM-102).
"""

from __future__ import annotations
from typing import Optional, Tuple, Any, List
import numpy as np
import pandas as pd
from PyQt6.QtCore import Qt, QAbstractTableModel, QModelIndex, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QBrush, QKeyEvent, QUndoCommand
from PyQt6.QtWidgets import QAbstractItemView, QTableView, QMenu, QInputDialog
import re
from src.core.table import DataTable, TableType
from src.core.undo import UndoStack, CellEditCommand, CellExcludeCommand, VariableCellEditCommand
from src.ui.mediator import WorkspaceMediator


class ScientificSpreadsheetModel(QAbstractTableModel):
    """
    Table model that maps a DataTable (with X column and replicated Y subcolumns)
    into a two-dimensional grid.
    """

    dataChangedSignal = pyqtSignal()

    def __init__(
        self,
        table: DataTable,
        undo_stack: Optional[UndoStack] = None,
        mediator: Optional[WorkspaceMediator] = None,
        is_frozen: bool = False,
    ) -> None:
        super().__init__()
        self.table = table
        self.undo_stack = undo_stack if undo_stack is not None else UndoStack()
        self.mediator = mediator or WorkspaceMediator.get_instance()
        self.is_frozen: bool = bool(is_frozen or getattr(table, "is_frozen", False) or getattr(table, "is_snapshot", False))
        self.is_frozen_snapshot: bool = self.is_frozen

        # Multi-X column mapping tracking Y-column index/name -> designated X-column index/name
        if hasattr(table, "xy_mapping") and isinstance(table.xy_mapping, dict):
            self.xy_mapping = table.xy_mapping
        else:
            self.xy_mapping = {}
            if table is not None:
                table.xy_mapping = self.xy_mapping

    @property
    def dataframe(self) -> pd.DataFrame:
        """Exposes the underlying pandas DataFrame representation."""
        return self.table.to_dataframe()

    @dataframe.setter
    def dataframe(self, df: pd.DataFrame) -> None:
        """Updates the table from a pandas DataFrame."""
        self.beginResetModel()
        if isinstance(df, pd.DataFrame) and self.table is not None:
            cols = self.table.get_columns()
            n_rows = min(len(df), self.table.num_rows)
            for c_idx in range(min(len(df.columns), len(cols))):
                col_info = cols[c_idx]
                for r_idx in range(n_rows):
                    val = df.iloc[r_idx, c_idx]
                    try:
                        fval = float(val) if pd.notna(val) else np.nan
                    except (ValueError, TypeError):
                        fval = np.nan
                    self.table.set_cell_value(r_idx, col_info.role, col_info.group_idx, col_info.subcol_idx, fval)
        self.endResetModel()
        self.layoutChanged.emit()
        self.dataChangedSignal.emit()
        if self.mediator:
            self.mediator.notify_table_modified()

    def to_dataframe(self) -> pd.DataFrame:
        """Exports data to a pandas DataFrame."""
        return self.table.to_dataframe()

    def get_column_title(self, col: int) -> str:
        """Returns the editable column title for a view column index."""
        return self.table.get_column_title(col)

    def rename_column(self, col: int, new_title: str) -> bool:
        """Renames column and notifies WorkspaceMediator."""
        if getattr(self, "is_frozen", False) or getattr(self, "is_frozen_snapshot", False) or getattr(getattr(self, "table", None), "is_frozen", False) or getattr(getattr(self, "table", None), "is_snapshot", False):
            return False
        if self.table.rename_column(col, new_title):
            self.headerDataChanged.emit(Qt.Orientation.Horizontal, col, col)
            self.layoutChanged.emit()
            self.dataChangedSignal.emit()
            if self.mediator:
                self.mediator.notify_column_renamed(col, new_title)
            return True
        return False

    def insert_column(self, col: int, position: str = "right", title: Optional[str] = None) -> bool:
        """Inserts a new column left or right of col and notifies WorkspaceMediator."""
        if getattr(self, "is_frozen", False) or getattr(self, "is_frozen_snapshot", False) or getattr(getattr(self, "table", None), "is_frozen", False) or getattr(getattr(self, "table", None), "is_snapshot", False):
            return False
        self.beginResetModel()
        self.table.insert_column(col, position=position, title=title)
        self.endResetModel()
        self.layoutChanged.emit()
        self.dataChangedSignal.emit()
        actual_title = title or self.get_column_title(col + (1 if position == "right" else 0))
        if self.mediator:
            self.mediator.notify_column_inserted(col, actual_title)
        return True

    def insert_x_column(self, col: int, position: str = "left", title: Optional[str] = None) -> bool:
        """Inserts a new X column left or right of col and notifies WorkspaceMediator."""
        if getattr(self, "is_frozen", False) or getattr(self, "is_frozen_snapshot", False) or getattr(getattr(self, "table", None), "is_frozen", False) or getattr(getattr(self, "table", None), "is_snapshot", False):
            return False
        self.beginResetModel()
        self.table.insert_x_column_at(col, position=position, title=title)
        self.endResetModel()
        self.layoutChanged.emit()
        self.dataChangedSignal.emit()
        actual_title = title or ("X" if len(self.table.x_columns) <= 1 else f"X{len(self.table.x_columns)}")
        if self.mediator:
            self.mediator.notify_column_inserted(col, actual_title)
        return True

    def delete_column(self, col: int) -> bool:
        """Deletes column at col and notifies WorkspaceMediator."""
        if getattr(self, "is_frozen", False) or getattr(self, "is_frozen_snapshot", False) or getattr(getattr(self, "table", None), "is_frozen", False) or getattr(getattr(self, "table", None), "is_snapshot", False):
            return False
        self.beginResetModel()
        res = self.table.delete_column(col)
        self.endResetModel()
        if res:
            self.layoutChanged.emit()
            self.dataChangedSignal.emit()
            if self.mediator:
                self.mediator.notify_column_deleted(col)
        return res

    def set_table(self, table: DataTable, is_frozen: Optional[bool] = None) -> None:
        self.beginResetModel()
        self.table = table
        if is_frozen is not None:
            self.is_frozen = bool(is_frozen)
        else:
            self.is_frozen = bool(getattr(table, "is_frozen", False) or getattr(table, "is_snapshot", False))
        self.is_frozen_snapshot = self.is_frozen
        if hasattr(table, "xy_mapping") and isinstance(table.xy_mapping, dict):
            self.xy_mapping = table.xy_mapping
        else:
            self.xy_mapping = {}
            if table is not None:
                table.xy_mapping = self.xy_mapping
        self.endResetModel()

    def to_dict(self) -> Dict[str, Any]:
        """Serializes model and table state to JSON-compatible dictionary."""
        d = self.table.to_dict() if self.table else {}
        d["xy_mapping"] = {str(k): v for k, v in self.xy_mapping.items()}
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScientificSpreadsheetModel":
        """Deserializes ScientificSpreadsheetModel from dictionary."""
        tbl = DataTable.from_dict(data)
        model = cls(tbl)
        model.xy_mapping = {
            int(k) if isinstance(k, str) and k.isdigit() else k: v
            for k, v in data.get("xy_mapping", {}).items()
        }
        tbl.xy_mapping = model.xy_mapping
        return model

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return self.table.num_rows

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if self.table.table_type == TableType.MULTIPLE_VARIABLES:
            return len(self.table.variables)
        return len(self.table.get_columns())

    def _col_to_cell_coords(self, col: int) -> Tuple[str, int, int]:
        """
        Translates a view column index into (col_type, group_idx, subcol_idx).
        """
        cols = self.table.get_columns()
        if 0 <= col < len(cols):
            c = cols[col]
            return (c.role, c.group_idx, c.subcol_idx)
        return ("Y", 0, 0)

    @staticmethod
    def _parse_x_value(val_str: str) -> Optional[float]:
        """Parses numeric, date, or elapsed time inputs for X columns."""
        s = val_str.strip()
        if not s:
            return np.nan
        # 1. Plain numeric
        try:
            return float(s)
        except ValueError:
            pass

        # 2. Elapsed time: e.g. "HH:MM:SS" or "MM:SS" or "-01:30"
        if ":" in s:
            sign = -1.0 if s.startswith("-") else 1.0
            clean_s = s.lstrip("+-")
            parts = clean_s.split(":")
            try:
                if len(parts) == 2:  # MM:SS
                    return sign * (float(parts[0]) * 60.0 + float(parts[1]))
                elif len(parts) == 3:  # HH:MM:SS
                    return sign * (float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2]))
            except ValueError:
                pass

        # 3. Time with unit: e.g. "10h", "30m", "45s", "5d", "5 days"
        m = re.match(r"^([+-]?\d+(?:\.\d+)?)\s*(d|day|days|h|hr|hours?|m|min|mins?|s|sec|secs?)$", s, re.IGNORECASE)
        if m:
            num = float(m.group(1))
            unit = m.group(2).lower()
            if unit.startswith("d"):
                return num * 86400.0
            elif unit.startswith("h"):
                return num * 3600.0
            elif unit.startswith("m"):
                return num * 60.0
            elif unit.startswith("s"):
                return num

        # 4. Date string e.g. YYYY-MM-DD
        try:
            ts = pd.to_datetime(s)
            return float(ts.timestamp() / 86400.0)
        except Exception:
            pass

        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Vertical:
            if role == Qt.ItemDataRole.ForegroundRole:
                return QBrush(QColor(0, 0, 0))
            elif role == Qt.ItemDataRole.BackgroundRole:
                return QBrush(QColor(240, 240, 240))
            elif role == Qt.ItemDataRole.DisplayRole:
                rt = self.table.row_titles[section] if section < len(self.table.row_titles) else ""
                return rt if rt else str(section + 1)
            return None

        if orientation == Qt.Orientation.Horizontal:
            if role == Qt.ItemDataRole.ForegroundRole:
                return QBrush(QColor(0, 0, 0))
            elif role == Qt.ItemDataRole.BackgroundRole:
                return QBrush(QColor(240, 240, 240))
            elif role == Qt.ItemDataRole.FontRole:
                font = QFont()
                if self.table.table_type != TableType.MULTIPLE_VARIABLES:
                    cols = self.table.get_columns()
                    if 0 <= section < len(cols) and cols[section].role == "X":
                        font.setBold(True)
                return font
            elif role == Qt.ItemDataRole.DisplayRole:
                if self.table.table_type == TableType.MULTIPLE_VARIABLES:
                    if 0 <= section < len(self.table.variables):
                        var = self.table.variables[section]
                        badge = "📊" if var.var_type.value == "Categorical" else ("🔔" if var.var_type.value == "Continuous" else "🏷️")
                        return f"{var.name} [{badge} {var.var_type.value}]"
                    return f"Var {section + 1}"

                cols = self.table.get_columns()
                if 0 <= section < len(cols):
                    col_info = cols[section]
                    if col_info.role == "X":
                        if len(self.table.x_columns) > 1 and col_info.title in ("", "X"):
                            return f"X{col_info.group_idx + 1}"
                        return col_info.title or "X"

                    g_idx = col_info.group_idx
                    s_idx = col_info.subcol_idx
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
        if getattr(self, "is_frozen", False) or getattr(self, "is_frozen_snapshot", False) or getattr(self.table, "is_frozen", False) or getattr(self.table, "is_snapshot", False):
            return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
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
        if getattr(self, "is_frozen", False) or getattr(self, "is_frozen_snapshot", False) or getattr(self.table, "is_frozen", False) or getattr(self.table, "is_snapshot", False):
            return False

        row = index.row()
        col = index.column()

        if self.table.table_type == TableType.MULTIPLE_VARIABLES:
            old_val = self.table.get_variable_value(row, col)
            if old_val == value or (value == "" and (old_val is None or (isinstance(old_val, float) and np.isnan(old_val)))):
                return False
            cmd = VariableCellEditCommand(self.table, row, col, old_val, value)
            self.undo_stack.push(cmd)
            self.dataChanged.emit(index, index, [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole])
            self.dataChangedSignal.emit()
            return True

        col_type, g_idx, s_idx = self._col_to_cell_coords(col)
        old_val = self.table.get_cell_value(row, col_type, g_idx, s_idx)

        new_val = np.nan
        if value is not None and str(value).strip() != "":
            raw_s = str(value).strip()
            if col_type == "X":
                parsed = self._parse_x_value(raw_s)
                if parsed is None:
                    return False
                new_val = parsed
            else:
                try:
                    new_val = float(raw_s)
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


class BatchClearCellsCommand(QUndoCommand):
    """
    Undoable batch command for clearing selected cells via Delete/Backspace.
    Preserves prior cell values and restores them upon undo().
    """

    def __init__(
        self,
        model: ScientificSpreadsheetModel,
        cells: Any = None,
        text: str = "Clear Cells",
        description: Optional[str] = None,
    ) -> None:
        cmd_text = description if description is not None else text
        super().__init__(cmd_text)
        self.model = model
        self.entries: List[Tuple[int, int, Any]] = []

        if cells:
            seen = set()
            if isinstance(cells[0], QModelIndex):
                for idx in cells:
                    r = idx.row()
                    c = idx.column()
                    if (r, c) in seen:
                        continue
                    seen.add((r, c))
                    if model.table.table_type == TableType.MULTIPLE_VARIABLES:
                        old_val = model.table.get_variable_value(r, c)
                    else:
                        col_type, g_idx, s_idx = model._col_to_cell_coords(c)
                        old_val = model.table.get_cell_value(r, col_type, g_idx, s_idx)
                    self.entries.append((r, c, old_val))
            elif isinstance(cells[0], (tuple, list)):
                for item in cells:
                    r = int(item[0])
                    c = int(item[1])
                    if (r, c) in seen:
                        continue
                    seen.add((r, c))
                    if len(item) >= 3:
                        old_val = item[2]
                    else:
                        if model.table.table_type == TableType.MULTIPLE_VARIABLES:
                            old_val = model.table.get_variable_value(r, c)
                        else:
                            col_type, g_idx, s_idx = model._col_to_cell_coords(c)
                            old_val = model.table.get_cell_value(r, col_type, g_idx, s_idx)
                    self.entries.append((r, c, old_val))

        self.cells = self.entries

    def redo(self) -> None:
        """Sets each cell to NaN (or empty string for string/multi-var tables) and emits changes."""
        if not self.entries:
            return
        if getattr(self.model, "is_frozen", False) or getattr(self.model, "is_frozen_snapshot", False) or getattr(getattr(self.model, "table", None), "is_frozen", False):
            return
        is_multivar = self.model.table.table_type == TableType.MULTIPLE_VARIABLES
        for r, c, _ in self.entries:
            if is_multivar:
                self.model.table.set_variable_value(r, c, "")
            else:
                col_type, g_idx, s_idx = self.model._col_to_cell_coords(c)
                self.model.table.set_cell_value(r, col_type, g_idx, s_idx, np.nan)
        self._notify_model_changed()

    def undo(self) -> None:
        """Restores each cell to its exact old value and emits changes."""
        if not self.entries:
            return
        if getattr(self.model, "is_frozen", False) or getattr(self.model, "is_frozen_snapshot", False) or getattr(getattr(self.model, "table", None), "is_frozen", False):
            return
        is_multivar = self.model.table.table_type == TableType.MULTIPLE_VARIABLES
        for r, c, old_val in self.entries:
            if is_multivar:
                self.model.table.set_variable_value(r, c, old_val)
            else:
                col_type, g_idx, s_idx = self.model._col_to_cell_coords(c)
                self.model.table.set_cell_value(r, col_type, g_idx, s_idx, old_val)
        self._notify_model_changed()

    def execute(self) -> None:
        self.redo()

    def _notify_model_changed(self) -> None:
        if not self.entries:
            return
        min_r = min(r for r, c, _ in self.entries)
        max_r = max(r for r, c, _ in self.entries)
        min_c = min(c for r, c, _ in self.entries)
        max_c = max(c for r, c, _ in self.entries)
        top_left = self.model.index(min_r, min_c)
        bottom_right = self.model.index(max_r, max_c)
        self.model.dataChanged.emit(
            top_left, bottom_right, [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole]
        )
        self.model.dataChangedSignal.emit()
        mediator = self.model.mediator or WorkspaceMediator.get_instance()
        if mediator:
            mediator.notify_table_modified()


class ScientificSpreadsheetView(QTableView):
    """
    QTableView with contextual actions: exclude/include cells,
    subcolumn replicate manipulation, and undo/redo support.
    """

    cellExclusionToggled = pyqtSignal()

    def __init__(self, model: ScientificSpreadsheetModel, parent=None) -> None:
        super().__init__(parent)
        self.table_view = self
        self.setModel(model)
        self.undo_stack = model.undo_stack
        self.setAlternatingRowColors(True)
        self.horizontalHeader().setStretchLastSection(False)
        self.setSelectionMode(QTableView.SelectionMode.ExtendedSelection)
        self.update_edit_triggers()

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

        # Header double-click renaming and context menu
        self.horizontalHeader().sectionDoubleClicked.connect(self._on_header_double_clicked)
        self.horizontalHeader().setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.horizontalHeader().customContextMenuRequested.connect(self._on_header_context_menu)

    def setModel(self, model) -> None:
        super().setModel(model)
        if hasattr(self, "table_view"):
            self.update_edit_triggers()

    def update_edit_triggers(self) -> None:
        """Sets edit triggers: NoEditTriggers for frozen snapshot tables, standard triggers otherwise."""
        model = self.model()
        is_frozen = False
        if model is not None:
            is_frozen = (
                getattr(model, "is_frozen", False)
                or getattr(model, "is_frozen_snapshot", False)
                or getattr(getattr(model, "table", None), "is_frozen", False)
                or getattr(getattr(model, "table", None), "is_snapshot", False)
            )
        if is_frozen:
            self.table_view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        else:
            self.table_view.setEditTriggers(
                QAbstractItemView.EditTrigger.DoubleClicked
                | QAbstractItemView.EditTrigger.EditKeyPressed
                | QAbstractItemView.EditTrigger.AnyKeyPressed
            )

    def set_frozen(self, frozen: bool) -> None:
        """Explicitly sets frozen status on view and model, updating edit triggers."""
        model = self.model()
        if model is not None:
            setattr(model, "is_frozen", bool(frozen))
            setattr(model, "is_frozen_snapshot", bool(frozen))
            if getattr(model, "table", None) is not None:
                setattr(model.table, "is_frozen", bool(frozen))
        self.update_edit_triggers()

    def load_sheet(self, sheet: Any) -> None:
        """Loads a DataSheet or GraphSheet snapshot into the view, applying frozen isolation if needed."""
        is_frozen = getattr(sheet, "is_frozen", False)
        table = getattr(sheet, "snapshot_table", None) if is_frozen else getattr(sheet, "table", None)
        if table is None and hasattr(sheet, "table"):
            table = sheet.table
        if table is not None:
            model = self.get_model()
            if model is not None:
                model.set_table(table, is_frozen=is_frozen)
        self.update_edit_triggers()

    def get_model(self) -> ScientificSpreadsheetModel:
        return self.model()  # type: ignore

    def _prompt_rename_column(self, col_idx: int) -> None:
        model = self.get_model()
        if getattr(model, "is_frozen", False) or getattr(model, "is_frozen_snapshot", False) or getattr(getattr(model, "table", None), "is_frozen", False):
            return
        current_title = model.get_column_title(col_idx)
        new_title, ok = QInputDialog.getText(
            self,
            "Rename Column",
            "Column Name:",
            text=current_title,
        )
        if ok and new_title.strip():
            model.rename_column(col_idx, new_title.strip())

    def _on_header_double_clicked(self, logicalIndex: int) -> None:
        model = self.get_model()
        if getattr(model, "is_frozen", False) or getattr(model, "is_frozen_snapshot", False) or getattr(getattr(model, "table", None), "is_frozen", False):
            return
        self._prompt_rename_column(logicalIndex)

    def _on_header_context_menu(self, pos) -> None:
        model = self.get_model()
        if getattr(model, "is_frozen", False) or getattr(model, "is_frozen_snapshot", False) or getattr(getattr(model, "table", None), "is_frozen", False):
            return
        col_idx = self.horizontalHeader().logicalIndexAt(pos)
        if col_idx < 0:
            return
        menu = QMenu(self)
        self._populate_column_actions(menu, col_idx)
        menu.exec(self.horizontalHeader().mapToGlobal(pos))

    def _populate_column_actions(self, menu: QMenu, col_idx: int) -> None:
        act_add_x_left = menu.addAction("➕ Add X Column to Left")
        act_add_x_right = menu.addAction("➕ Add X Column to Right")
        act_add_y = menu.addAction("➕ Add Y Column / Replicate")
        act_ins_right = menu.addAction("➕ Insert Column to Right")
        act_ins_left = menu.addAction("➕ Insert Column to Left")
        act_rename = menu.addAction("✏️ Rename Column...")
        act_delete = menu.addAction("🗑️ Delete Column")

        model = self.get_model()
        is_y_column = False
        x_columns = []
        if model and getattr(model, "table", None):
            cols = model.table.get_columns()
            if 0 <= col_idx < len(cols):
                is_y_column = (cols[col_idx].role == "Y")
            for idx, c in enumerate(cols):
                if c.role == "X":
                    title = c.title or f"X{c.group_idx + 1}"
                    x_columns.append((idx, title))

        assign_x_actions = {}
        if is_y_column:
            assign_menu = menu.addMenu("Assign to X Column ➔")
            if not x_columns:
                x_columns.append((0, "X (Column 1)"))
            for x_idx, x_title in x_columns:
                act_assign = assign_menu.addAction(f"{x_title} (Column {x_idx + 1})")
                assign_x_actions[act_assign] = (x_idx, x_title)

        def on_action(act):
            m = self.get_model()
            if getattr(m, "is_frozen", False) or getattr(m, "is_frozen_snapshot", False) or getattr(getattr(m, "table", None), "is_frozen", False):
                return
            if act == act_add_x_left:
                m.insert_x_column(col_idx, position="left")
            elif act == act_add_x_right:
                m.insert_x_column(col_idx, position="right")
            elif act in (act_add_y, act_ins_right):
                m.insert_column(col_idx, position="right")
            elif act == act_ins_left:
                m.insert_column(col_idx, position="left")
            elif act == act_rename:
                self._prompt_rename_column(col_idx)
            elif act == act_delete:
                m.delete_column(col_idx)
            elif act in assign_x_actions:
                x_i, x_t = assign_x_actions[act]
                m.xy_mapping[col_idx] = x_i
                m.xy_mapping[str(col_idx)] = x_i
                col_title = m.get_column_title(col_idx)
                if col_title:
                    m.xy_mapping[col_title] = x_t
                if hasattr(m, "table") and m.table is not None:
                    m.table.xy_mapping[col_idx] = x_i
                m.layoutChanged.emit()
                m.dataChangedSignal.emit()
                mediator = getattr(m, "mediator", None) or WorkspaceMediator.get_instance()
                if mediator:
                    mediator.notify_data_changed()
                    if hasattr(mediator, "sync_data_to_visualization"):
                        mediator.sync_data_to_visualization(model=m)

        act_add_x_left.triggered.connect(lambda: on_action(act_add_x_left))
        act_add_x_right.triggered.connect(lambda: on_action(act_add_x_right))
        act_add_y.triggered.connect(lambda: on_action(act_add_y))
        act_ins_right.triggered.connect(lambda: on_action(act_ins_right))
        act_ins_left.triggered.connect(lambda: on_action(act_ins_left))
        act_rename.triggered.connect(lambda: on_action(act_rename))
        act_delete.triggered.connect(lambda: on_action(act_delete))
        for a_act in assign_x_actions:
            a_act.triggered.connect(lambda checked=False, act_ref=a_act: on_action(act_ref))

    def contextMenuEvent(self, event) -> None:
        model = self.get_model()
        if getattr(model, "is_frozen", False) or getattr(model, "is_frozen_snapshot", False) or getattr(getattr(model, "table", None), "is_frozen", False):
            menu = QMenu(self)
            action = menu.addAction("🔒 Frozen Snapshot (Read-Only)")
            action.setEnabled(False)
            menu.exec(event.globalPos())
            return
        menu = QMenu(self)

        col_idx = self.columnAt(event.pos().x())
        if col_idx < 0 and self.selectedIndexes():
            col_idx = self.selectedIndexes()[0].column()
        if col_idx < 0:
            col_idx = 0

        self._populate_column_actions(menu, col_idx)
        menu.addSeparator()

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
        if getattr(model, "is_frozen", False) or getattr(model, "is_frozen_snapshot", False) or getattr(getattr(model, "table", None), "is_frozen", False) or getattr(getattr(model, "table", None), "is_snapshot", False):
            return
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
        self.cellExclusionToggled.emit()

    def _add_rows(self, count: int) -> None:
        model = self.get_model()
        if getattr(model, "is_frozen", False) or getattr(model, "is_frozen_snapshot", False) or getattr(getattr(model, "table", None), "is_frozen", False) or getattr(getattr(model, "table", None), "is_snapshot", False):
            return
        model.beginResetModel()
        model.table.add_row(count)
        model.endResetModel()
        model.dataChangedSignal.emit()

    def _add_group(self) -> None:
        model = self.get_model()
        if getattr(model, "is_frozen", False) or getattr(model, "is_frozen_snapshot", False) or getattr(getattr(model, "table", None), "is_frozen", False) or getattr(getattr(model, "table", None), "is_snapshot", False):
            return
        model.beginResetModel()
        model.table.add_y_group()
        model.endResetModel()
        model.dataChangedSignal.emit()

    def _prompt_set_replicates(self) -> None:
        model = self.get_model()
        if getattr(model, "is_frozen", False) or getattr(model, "is_frozen_snapshot", False) or getattr(getattr(model, "table", None), "is_frozen", False) or getattr(getattr(model, "table", None), "is_snapshot", False):
            return
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

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if self.state() != self.State.EditingState and event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            model = self.get_model()
            if getattr(model, "is_frozen", False) or getattr(model, "is_frozen_snapshot", False) or getattr(getattr(model, "table", None), "is_frozen", False) or getattr(getattr(model, "table", None), "is_snapshot", False):
                event.accept()
                return
            selection_model = self.selectionModel()
            if selection_model is not None:
                indexes = selection_model.selectedIndexes()
                if indexes:
                    cmd = BatchClearCellsCommand(model, indexes)
                    self.undo_stack.push(cmd)
                    event.accept()
                    return
        super().keyPressEvent(event)


# Aliases for explicit ReplicateSpreadsheet and ScientificTableWidget references
ReplicateSpreadsheet = ScientificSpreadsheetView
ScientificTableWidget = ScientificSpreadsheetView
ScientificTableModel = ScientificSpreadsheetModel

__all__ = [
    "ScientificSpreadsheetModel",
    "ScientificSpreadsheetView",
    "ScientificTableModel",
    "ScientificTableWidget",
    "ReplicateSpreadsheet",
    "BatchClearCellsCommand",
]


