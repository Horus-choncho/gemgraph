"""
Core Data Table and Column Definitions for GemGraph.
Derived from Core.dll schema specifications (REA-CORE-SCHEMA-208).
"""

from __future__ import annotations
import enum
from typing import List, Dict, Optional, Tuple, Set, Any
import numpy as np
import pandas as pd


class TableType(enum.Enum):
    XY = "XY"
    COLUMN = "Column"          # One-way / Column groups
    GROUPED = "Grouped"        # Two-way / Grouped
    CONTINGENCY = "Contingency"
    SURVIVAL = "Survival"
    PARTS_OF_WHOLE = "Parts of Whole"
    MULTIPLE_VARIABLES = "Multiple Variables"
    NESTED = "Nested"


class VariableType(enum.Enum):
    CATEGORICAL = "Categorical"
    CONTINUOUS = "Continuous"
    LABEL = "Label"


class VariableMetadata:
    """Metadata for a variable column in a Multiple Variables table."""

    def __init__(
        self,
        name: str,
        var_type: VariableType = VariableType.CONTINUOUS,
        levels: Optional[List[str]] = None,
    ) -> None:
        self.name = name
        self.var_type = var_type
        self.levels: List[str] = list(levels) if levels is not None else []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "var_type": self.var_type.value,
            "levels": self.levels,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VariableMetadata:
        return cls(
            name=data.get("name", ""),
            var_type=VariableType(data.get("var_type", "Continuous")),
            levels=data.get("levels", []),
        )


class XFormat(enum.Enum):
    NUMBERS = "numbers"
    SERIES = "series"
    TEXT = "text"
    NONE = "none"


class YFormat(enum.Enum):
    REPLICATES = "replicates"
    MEAN_SD = "SD"
    MEAN_SEM = "SE"
    TEXT = "text"


class ColumnInfo:
    """Descriptor for an individual visual column in a DataTable."""

    def __init__(
        self,
        role: str,
        group_idx: int,
        subcol_idx: int = 0,
        title: str = "",
        associated_x_idx: int = 0,
    ) -> None:
        self.role = role  # "X" or "Y"
        self.group_idx = group_idx
        self.subcol_idx = subcol_idx
        self.title = title
        self.associated_x_idx = associated_x_idx

    @property
    def display_header(self) -> str:
        if self.role == "X":
            return self.title if self.title else f"X{self.group_idx + 1}"
        if self.subcol_idx > 0 or self.title.startswith("Group "):
            return f"{self.title} [Y{self.subcol_idx + 1}]"
        return self.title

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "group_idx": self.group_idx,
            "subcol_idx": self.subcol_idx,
            "title": self.title,
            "associated_x_idx": self.associated_x_idx,
        }

    def __repr__(self) -> str:
        return f"ColumnInfo(role={self.role}, group={self.group_idx}, subcol={self.subcol_idx}, title='{self.title}')"


class DataTable:
    """
    Scientific data table supporting replicates, subcolumns, missing values,
    and cell exclusion masking across 8 standard scientific paradigms.
    """

    def __init__(
        self,
        table_id: str,
        title: str = "Data 1",
        table_type: TableType = TableType.XY,
        num_rows: int = 20,
        x_format: XFormat = XFormat.NUMBERS,
        y_format: YFormat = YFormat.REPLICATES,
        num_replicates: int = 3,
        y_group_titles: Optional[List[str]] = None,
    ) -> None:
        self.table_id = table_id
        self.title = title
        self.table_type = table_type
        self.x_format = x_format
        self.y_format = y_format
        self.num_replicates = max(1, num_replicates)
        self.num_rows = max(1, num_rows)

        # Row titles (e.g., sample names or categories)
        self.row_titles: List[str] = ["" for _ in range(self.num_rows)]

        self._legacy_x_title: str = ""
        # Multi-X column storage: list of dicts {"title": str, "data": np.ndarray}
        self.x_columns: List[Dict[str, Any]] = []
        if self.table_type in (TableType.XY, TableType.SURVIVAL):
            init_x_title = "Elapsed Time (days)" if self.table_type == TableType.SURVIVAL else "X"
            self._legacy_x_title = init_x_title
            self.x_columns.append({
                "title": init_x_title,
                "data": np.full(self.num_rows, np.nan, dtype=np.float64),
            })

        # Y column groups: list of group titles, each group has num_replicates subcolumns
        if y_group_titles is not None:
            self.y_group_titles = list(y_group_titles)
        elif self.table_type == TableType.PARTS_OF_WHOLE:
            self.y_group_titles = ["Fraction"]
            self.num_replicates = 1
        elif self.table_type == TableType.CONTINGENCY:
            self.y_group_titles = ["Outcome A", "Outcome B"]
            self.num_replicates = 1
        else:
            self.y_group_titles = ["Group A", "Group B"]

        self.y_data = np.full(
            (len(self.y_group_titles), self.num_rows, self.num_replicates),
            np.nan,
            dtype=np.float64,
        )

        # Column sequence order: list of (block_type, block_idx) e.g. [("X", 0), ("Y", 0), ("Y", 1)]
        self.column_order: List[Tuple[str, int]] = []
        if self.x_columns:
            self.column_order.append(("X", 0))
        for g_idx in range(len(self.y_group_titles)):
            self.column_order.append(("Y", g_idx))

        # Excluded points bitmask: set of (row, col_type, group_idx, subcol_idx)
        # col_type: 'X' or 'Y'
        self.excluded_points: Set[Tuple[int, str, int, int]] = set()

        # Cell background colors: map (row, col_type, group_idx, subcol_idx) -> hex color
        self.cell_colors: Dict[Tuple[int, str, int, int], str] = {}

        # Nested tables: map group_idx -> list of nested physical unit titles (e.g. ["Room 1", "Room 2"])
        self.nested_subgroup_titles: Dict[int, List[str]] = {}

        # Multiple Variables table support
        self.variables: List[VariableMetadata] = []
        self.multi_var_data: List[List[Any]] = [[] for _ in range(self.num_rows)]
        if self.table_type == TableType.MULTIPLE_VARIABLES:
            self._init_default_variables()

    @property
    def x_title(self) -> str:
        if self.x_columns:
            return self.x_columns[0]["title"]
        return self._legacy_x_title

    @x_title.setter
    def x_title(self, val: str) -> None:
        self._legacy_x_title = val
        if self.x_columns:
            self.x_columns[0]["title"] = val
        elif val and self.table_type in (TableType.XY, TableType.SURVIVAL):
            self.x_columns.append({"title": val, "data": np.full(self.num_rows, np.nan, dtype=np.float64)})
            if ("X", 0) not in self.column_order:
                self.column_order.insert(0, ("X", 0))

    @property
    def x_data(self) -> Optional[np.ndarray]:
        if self.x_columns:
            return self.x_columns[0]["data"]
        return None

    @x_data.setter
    def x_data(self, val: Optional[np.ndarray]) -> None:
        if val is None:
            self.x_columns = []
            self.column_order = [b for b in self.column_order if b[0] != "X"]
        else:
            arr = np.asarray(val, dtype=np.float64)
            if not self.x_columns:
                title = self._legacy_x_title or "X"
                self.x_columns = [{"title": title, "data": arr}]
                if ("X", 0) not in self.column_order:
                    self.column_order.insert(0, ("X", 0))
            else:
                self.x_columns[0]["data"] = arr

    def get_columns(self) -> List[ColumnInfo]:
        """Returns flat list of ColumnInfo descriptors for all visual columns."""
        cols: List[ColumnInfo] = []
        current_x_idx = 0 if self.x_columns else -1
        for block_type, idx in self.column_order:
            if block_type == "X":
                if 0 <= idx < len(self.x_columns):
                    current_x_idx = idx
                    title = self.x_columns[idx]["title"]
                    cols.append(ColumnInfo(
                        role="X",
                        group_idx=idx,
                        subcol_idx=0,
                        title=title,
                        associated_x_idx=idx,
                    ))
            elif block_type == "Y":
                if 0 <= idx < len(self.y_group_titles):
                    g_title = self.y_group_titles[idx]
                    assoc_x = current_x_idx if current_x_idx >= 0 else 0
                    for s_idx in range(self.num_replicates):
                        cols.append(ColumnInfo(
                            role="Y",
                            group_idx=idx,
                            subcol_idx=s_idx,
                            title=g_title,
                            associated_x_idx=assoc_x,
                        ))
        return cols

    @property
    def columns(self) -> List[ColumnInfo]:
        return self.get_columns()

    def get_column_role(self, col_idx: int) -> str:
        """Returns 'X' or 'Y' for visual column index."""
        cols = self.get_columns()
        if 0 <= col_idx < len(cols):
            return cols[col_idx].role
        return "Y"

    def get_column_replicate_index(self, col_idx: int) -> int:
        """Returns subcolumn replicate index (0-based) for visual column."""
        cols = self.get_columns()
        if 0 <= col_idx < len(cols):
            return cols[col_idx].subcol_idx
        return 0

    def get_x_column_count(self) -> int:
        """Returns number of distinct X columns."""
        return len(self.x_columns)

    def get_x_data(self, x_idx: int = 0) -> Optional[np.ndarray]:
        """Retrieves raw numpy array for a specific X column."""
        if 0 <= x_idx < len(self.x_columns):
            return self.x_columns[x_idx]["data"]
        return None

    def get_y_associated_x_idx(self, group_idx: int) -> int:
        """Finds preceding X column for a given Y group index, falling back to 0."""
        current_x = 0
        for block_type, idx in self.column_order:
            if block_type == "X":
                current_x = idx
            elif block_type == "Y" and idx == group_idx:
                return current_x
        return 0

    def add_row(self, count: int = 1) -> None:
        """Appends empty rows to the table."""
        if count <= 0:
            return
        self.num_rows += count
        self.row_titles.extend(["" for _ in range(count)])
        for x_col in self.x_columns:
            new_x = np.full(count, np.nan, dtype=np.float64)
            x_col["data"] = np.concatenate([x_col["data"], new_x])

        new_y = np.full(
            (len(self.y_group_titles), count, self.num_replicates),
            np.nan,
            dtype=np.float64,
        )
        self.y_data = np.concatenate([self.y_data, new_y], axis=1)

        for _ in range(count):
            self.multi_var_data.append([
                np.nan if v.var_type == VariableType.CONTINUOUS else ""
                for v in self.variables
            ])

    def add_y_group(self, title: Optional[str] = None) -> int:
        """Adds a new Y column group with the current replicate count."""
        group_idx = len(self.y_group_titles)
        if not title:
            # Generate next letter e.g. Group C, D...
            letter = chr(ord("A") + group_idx) if group_idx < 26 else f"G{group_idx + 1}"
            title = f"Group {letter}"
        self.y_group_titles.append(title)
        self.column_order.append(("Y", group_idx))
        new_group_data = np.full(
            (1, self.num_rows, self.num_replicates), np.nan, dtype=np.float64
        )
        if self.y_data.size == 0 or self.y_data.shape[0] == 0:
            self.y_data = new_group_data
        else:
            self.y_data = np.concatenate([self.y_data, new_group_data], axis=0)
        return group_idx

    def remove_row(self, row_idx: int) -> bool:
        """Removes a row at the specified index, updating exclusion masks and colors."""
        if row_idx < 0 or row_idx >= self.num_rows:
            return False
        if self.num_rows <= 1:
            return False
        self.num_rows -= 1
        del self.row_titles[row_idx]
        for x_col in self.x_columns:
            x_col["data"] = np.delete(x_col["data"], row_idx)
        self.y_data = np.delete(self.y_data, row_idx, axis=1)
        if row_idx < len(self.multi_var_data):
            del self.multi_var_data[row_idx]

        # Update excluded points: discard points in this row, shift points > row_idx
        new_excluded = set()
        for r, c_type, g_idx, s_idx in self.excluded_points:
            if r == row_idx:
                continue
            elif r > row_idx:
                new_excluded.add((r - 1, c_type, g_idx, s_idx))
            else:
                new_excluded.add((r, c_type, g_idx, s_idx))
        self.excluded_points = new_excluded

        # Update cell colors: shift row indices
        new_colors = {}
        for (r, c_type, g_idx, s_idx), col in self.cell_colors.items():
            if r == row_idx:
                continue
            elif r > row_idx:
                new_colors[(r - 1, c_type, g_idx, s_idx)] = col
            else:
                new_colors[(r, c_type, g_idx, s_idx)] = col
        self.cell_colors = new_colors
        return True

    def remove_y_group(self, group_idx: int) -> bool:
        """Removes a Y column group, updating exclusion masks, colors, and nested titles."""
        if group_idx < 0 or group_idx >= len(self.y_group_titles):
            return False
        if len(self.y_group_titles) <= 1:
            return False
        del self.y_group_titles[group_idx]
        self.y_data = np.delete(self.y_data, group_idx, axis=0)

        # Update column_order
        new_order = []
        for b_type, b_idx in self.column_order:
            if b_type == "Y":
                if b_idx == group_idx:
                    continue
                elif b_idx > group_idx:
                    new_order.append(("Y", b_idx - 1))
                else:
                    new_order.append(("Y", b_idx))
            else:
                new_order.append((b_type, b_idx))
        self.column_order = new_order

        # Update excluded points
        new_excluded = set()
        for r, c_type, g_idx, s_idx in self.excluded_points:
            if c_type == "Y":
                if g_idx == group_idx:
                    continue
                elif g_idx > group_idx:
                    new_excluded.add((r, c_type, g_idx - 1, s_idx))
                else:
                    new_excluded.add((r, c_type, g_idx, s_idx))
            else:
                new_excluded.add((r, c_type, g_idx, s_idx))
        self.excluded_points = new_excluded

        # Update cell colors
        new_colors = {}
        for (r, c_type, g_idx, s_idx), col in self.cell_colors.items():
            if c_type == "Y":
                if g_idx == group_idx:
                    continue
                elif g_idx > group_idx:
                    new_colors[(r, c_type, g_idx - 1, s_idx)] = col
                else:
                    new_colors[(r, c_type, g_idx, s_idx)] = col
            else:
                new_colors[(r, c_type, g_idx, s_idx)] = col
        self.cell_colors = new_colors

        # Update nested titles
        new_nested = {}
        for g_i, titles in self.nested_subgroup_titles.items():
            if g_i == group_idx:
                continue
            elif g_i > group_idx:
                new_nested[g_i - 1] = titles
            else:
                new_nested[g_i] = titles
        self.nested_subgroup_titles = new_nested
        return True

    def insert_y_group(self, group_idx: int, title: Optional[str] = None) -> int:
        """Inserts a new Y column group at group_idx, shifting subsequent groups and masks."""
        group_idx = max(0, min(group_idx, len(self.y_group_titles)))
        if not title:
            i = 0
            while True:
                candidate = f"Group {chr(ord('A') + i)}" if i < 26 else f"Group {i + 1}"
                if candidate not in self.y_group_titles:
                    title = candidate
                    break
                i += 1
        self.y_group_titles.insert(group_idx, title)

        new_group_data = np.full(
            (1, self.num_rows, self.num_replicates), np.nan, dtype=np.float64
        )
        if self.y_data.size == 0 or self.y_data.shape[0] == 0:
            self.y_data = new_group_data
        else:
            self.y_data = np.insert(self.y_data, group_idx, new_group_data[0], axis=0)

        # Update column_order: find where to insert
        target_pos = len(self.column_order)
        for i, (b_type, b_idx) in enumerate(self.column_order):
            if b_type == "Y" and b_idx == group_idx:
                target_pos = i
                break
        new_order = []
        for b_type, b_idx in self.column_order:
            if b_type == "Y" and b_idx >= group_idx:
                new_order.append(("Y", b_idx + 1))
            else:
                new_order.append((b_type, b_idx))
        new_order.insert(target_pos, ("Y", group_idx))
        self.column_order = new_order

        # Shift excluded points for g_idx >= group_idx
        new_excluded = set()
        for r, c_type, g_idx, s_idx in self.excluded_points:
            if c_type == "Y" and g_idx >= group_idx:
                new_excluded.add((r, c_type, g_idx + 1, s_idx))
            else:
                new_excluded.add((r, c_type, g_idx, s_idx))
        self.excluded_points = new_excluded

        # Shift cell colors for g_idx >= group_idx
        new_colors = {}
        for (r, c_type, g_idx, s_idx), col in self.cell_colors.items():
            if c_type == "Y" and g_idx >= group_idx:
                new_colors[(r, c_type, g_idx + 1, s_idx)] = col
            else:
                new_colors[(r, c_type, g_idx, s_idx)] = col
        self.cell_colors = new_colors

        # Shift nested titles
        new_nested = {}
        for g_i, titles in self.nested_subgroup_titles.items():
            if g_i >= group_idx:
                new_nested[g_i + 1] = titles
            else:
                new_nested[g_i] = titles
        self.nested_subgroup_titles = new_nested

        return group_idx

    def add_x_column(self, title: Optional[str] = None, position_in_order: Optional[int] = None) -> int:
        """Adds a new X column to the table."""
        x_idx = len(self.x_columns)
        if not title:
            title = f"X{x_idx + 1}" if x_idx > 0 else "X"
        col_data = np.full(self.num_rows, np.nan, dtype=np.float64)
        self.x_columns.append({"title": title, "data": col_data})
        if position_in_order is not None and 0 <= position_in_order <= len(self.column_order):
            self.column_order.insert(position_in_order, ("X", x_idx))
        else:
            self.column_order.append(("X", x_idx))
        return x_idx

    def insert_x_column_at(self, col_idx: int, position: str = "left", title: Optional[str] = None) -> int:
        """Inserts an X column to the left or right of visual column col_idx."""
        cols = self.get_columns()
        if not cols:
            return self.add_x_column(title=title)

        col_idx = max(0, min(col_idx, len(cols) - 1))
        target_col = cols[col_idx]

        block_pos = 0
        for i, (b_type, b_idx) in enumerate(self.column_order):
            if b_type == target_col.role and b_idx == target_col.group_idx:
                block_pos = i
                break

        insert_pos = block_pos + 1 if position == "right" else block_pos

        x_idx = len(self.x_columns)
        if not title:
            title = f"X{x_idx + 1}"
        col_data = np.full(self.num_rows, np.nan, dtype=np.float64)
        self.x_columns.append({"title": title, "data": col_data})
        self.column_order.insert(insert_pos, ("X", x_idx))
        return x_idx

    def remove_x_column(self, x_idx: int) -> bool:
        """Removes X column at x_idx, updating column_order, masks, and colors."""
        if x_idx < 0 or x_idx >= len(self.x_columns):
            return False
        if len(self.x_columns) <= 1:
            return False

        del self.x_columns[x_idx]

        new_order = []
        for b_type, b_idx in self.column_order:
            if b_type == "X":
                if b_idx == x_idx:
                    continue
                elif b_idx > x_idx:
                    new_order.append(("X", b_idx - 1))
                else:
                    new_order.append(("X", b_idx))
            else:
                new_order.append((b_type, b_idx))
        self.column_order = new_order

        # Shift excluded points
        new_excluded = set()
        for r, c_type, g_idx, s_idx in self.excluded_points:
            if c_type == "X":
                if g_idx == x_idx:
                    continue
                elif g_idx > x_idx:
                    new_excluded.add((r, c_type, g_idx - 1, s_idx))
                else:
                    new_excluded.add((r, c_type, g_idx, s_idx))
            else:
                new_excluded.add((r, c_type, g_idx, s_idx))
        self.excluded_points = new_excluded

        # Shift cell colors
        new_colors = {}
        for (r, c_type, g_idx, s_idx), col in self.cell_colors.items():
            if c_type == "X":
                if g_idx == x_idx:
                    continue
                elif g_idx > x_idx:
                    new_colors[(r, c_type, g_idx - 1, s_idx)] = col
                else:
                    new_colors[(r, c_type, g_idx, s_idx)] = col
            else:
                new_colors[(r, c_type, g_idx, s_idx)] = col
        self.cell_colors = new_colors

        return True

    def rename_group(self, group_idx: int, new_title: str) -> None:
        """Renames a Y group."""
        if 0 <= group_idx < len(self.y_group_titles):
            self.y_group_titles[group_idx] = new_title

    def insert_variable(
        self,
        var_idx: int,
        name: Optional[str] = None,
        var_type: VariableType = VariableType.CONTINUOUS,
        levels: Optional[List[str]] = None,
    ) -> int:
        """Inserts a new variable column at var_idx in a Multiple Variables table."""
        var_idx = max(0, min(var_idx, len(self.variables)))
        if not name:
            name = f"Var {len(self.variables) + 1}"
        var = VariableMetadata(name, var_type, levels)
        self.variables.insert(var_idx, var)
        for row in self.multi_var_data:
            row.insert(var_idx, "" if var_type != VariableType.CONTINUOUS else np.nan)

        # Shift cell colors for VAR columns >= var_idx
        new_colors = {}
        for (r, c_type, g_idx, s_idx), col in self.cell_colors.items():
            if c_type == "VAR" and g_idx >= var_idx:
                new_colors[(r, c_type, g_idx + 1, s_idx)] = col
            else:
                new_colors[(r, c_type, g_idx, s_idx)] = col
        self.cell_colors = new_colors

        return var_idx

    def remove_variable(self, var_idx: int) -> bool:
        """Removes a variable column from a Multiple Variables table."""
        if var_idx < 0 or var_idx >= len(self.variables):
            return False
        if len(self.variables) <= 1:
            return False
        del self.variables[var_idx]
        for row in self.multi_var_data:
            if var_idx < len(row):
                del row[var_idx]

        # Shift and prune cell colors for VAR columns
        new_colors = {}
        for (r, c_type, g_idx, s_idx), col in self.cell_colors.items():
            if c_type == "VAR":
                if g_idx == var_idx:
                    continue
                elif g_idx > var_idx:
                    new_colors[(r, c_type, g_idx - 1, s_idx)] = col
                else:
                    new_colors[(r, c_type, g_idx, s_idx)] = col
            else:
                new_colors[(r, c_type, g_idx, s_idx)] = col
        self.cell_colors = new_colors

        return True

    def rename_variable(self, var_idx: int, new_name: str) -> None:
        """Renames a variable in a Multiple Variables table."""
        if 0 <= var_idx < len(self.variables):
            self.variables[var_idx].name = new_name

    def get_column_title(self, col_idx: int) -> str:
        """Returns the editable column title for a view column index."""
        if self.table_type == TableType.MULTIPLE_VARIABLES:
            if 0 <= col_idx < len(self.variables):
                return self.variables[col_idx].name
            return f"Var {col_idx + 1}"

        cols = self.get_columns()
        if 0 <= col_idx < len(cols):
            return cols[col_idx].title
        return f"Col {col_idx + 1}"

    def rename_column(self, col_idx: int, new_title: str) -> bool:
        """Renames a column by view column index."""
        if not new_title or not new_title.strip():
            return False
        new_title = new_title.strip()
        if self.table_type == TableType.MULTIPLE_VARIABLES:
            if 0 <= col_idx < len(self.variables):
                self.rename_variable(col_idx, new_title)
                return True
            return False

        cols = self.get_columns()
        if 0 <= col_idx < len(cols):
            target_col = cols[col_idx]
            if target_col.role == "X":
                if 0 <= target_col.group_idx < len(self.x_columns):
                    self.x_columns[target_col.group_idx]["title"] = new_title
                    return True
            else:
                if 0 <= target_col.group_idx < len(self.y_group_titles):
                    self.rename_group(target_col.group_idx, new_title)
                    return True
        return False

    def insert_column(self, col_idx: int, position: str = "right", title: Optional[str] = None) -> int:
        """Inserts a Y column group to the left or right of col_idx."""
        if self.table_type == TableType.MULTIPLE_VARIABLES:
            target = col_idx + 1 if position == "right" else col_idx
            return self.insert_variable(target, name=title)

        cols = self.get_columns()
        if not cols:
            return self.add_y_group(title=title)

        col_idx = max(0, min(col_idx, len(cols) - 1))
        target_col = cols[col_idx]

        if target_col.role == "Y":
            target_g = target_col.group_idx + 1 if position == "right" else target_col.group_idx
        else:
            # If target is X column
            target_g = 0
            for b_type, b_idx in self.column_order:
                if b_type == "Y":
                    target_g = b_idx
                    break
            if position == "right":
                target_g = 0
        return self.insert_y_group(target_g, title=title)

    def delete_column(self, col_idx: int) -> bool:
        """Deletes column at col_idx. Returns False if operation disallowed."""
        if self.table_type == TableType.MULTIPLE_VARIABLES:
            return self.remove_variable(col_idx)

        cols = self.get_columns()
        if col_idx < 0 or col_idx >= len(cols):
            return False

        target_col = cols[col_idx]
        if target_col.role == "X":
            if len(self.x_columns) <= 1:
                return False
            return self.remove_x_column(target_col.group_idx)
        return self.remove_y_group(target_col.group_idx)

    def to_dataframe(self) -> pd.DataFrame:
        """Converts DataTable representation into a pandas DataFrame."""
        if self.table_type == TableType.MULTIPLE_VARIABLES:
            cols = [v.name for v in self.variables]
            return pd.DataFrame(self.multi_var_data, columns=cols)

        col_infos = self.get_columns()
        cols = []
        for c in col_infos:
            if c.role == "X":
                cols.append(c.title or f"X{c.group_idx + 1}")
            else:
                if self.num_replicates == 1:
                    cols.append(c.title)
                else:
                    cols.append(f"{c.title}:Y{c.subcol_idx + 1}")

        rows = []
        for r in range(self.num_rows):
            row_vals = []
            for c in col_infos:
                if c.role == "X":
                    row_vals.append(self.x_columns[c.group_idx]["data"][r])
                else:
                    row_vals.append(self.y_data[c.group_idx, r, c.subcol_idx])
            rows.append(row_vals)

        return pd.DataFrame(rows, columns=cols)


    def update_from_dataframe(self, df: pd.DataFrame) -> None:
        """Updates DataTable content, column names, and dimensions from a pandas DataFrame."""
        if df is None or not isinstance(df, pd.DataFrame):
            return

        if self.table_type == TableType.MULTIPLE_VARIABLES:
            self.num_rows = len(df)
            self.row_titles = ["" for _ in range(self.num_rows)]
            self.variables = []
            for col_name in df.columns:
                series = df[col_name]
                if pd.api.types.is_numeric_dtype(series):
                    v_type = VariableType.CONTINUOUS
                    levels = []
                else:
                    v_type = VariableType.CATEGORICAL
                    levels = [str(x) for x in series.dropna().unique().tolist()]
                self.variables.append(VariableMetadata(str(col_name), v_type, levels))
            self.multi_var_data = []
            for _, row in df.iterrows():
                row_vals = []
                for v_idx, col_name in enumerate(df.columns):
                    val = row[col_name]
                    if self.variables[v_idx].var_type == VariableType.CONTINUOUS:
                        row_vals.append(np.nan if pd.isna(val) else float(val))
                    else:
                        row_vals.append("" if pd.isna(val) else str(val))
                self.multi_var_data.append(row_vals)
            return

        cols = list(df.columns)
        num_rows = len(df)
        if num_rows > 0:
            if num_rows > self.num_rows:
                self.add_row(num_rows - self.num_rows)
            elif num_rows < self.num_rows:
                self.num_rows = num_rows
                self.row_titles = self.row_titles[:num_rows]
                if self.x_data is not None:
                    self.x_data = self.x_data[:num_rows]
                self.y_data = self.y_data[:, :num_rows, :]

        col_start = 0
        if self.x_data is not None and len(cols) > 0:
            self.x_title = str(cols[0])
            self.x_data[:num_rows] = pd.to_numeric(df.iloc[:num_rows, 0], errors="coerce").to_numpy()
            col_start = 1

        y_cols = cols[col_start:]
        if y_cols:
            import re
            rep_matches = [re.match(r"^(.*?):Y(\d+)$", str(c)) for c in y_cols]
            if any(m is not None for m in rep_matches):
                groups_dict: Dict[str, List[Tuple[int, str]]] = {}
                for c in y_cols:
                    m = re.match(r"^(.*?):Y(\d+)$", str(c))
                    if m:
                        g_name, rep_idx = m.group(1), int(m.group(2))
                        groups_dict.setdefault(g_name, []).append((rep_idx, c))
                    else:
                        groups_dict.setdefault(str(c), []).append((1, str(c)))
                new_groups = list(groups_dict.keys())
                max_reps = max(len(v) for v in groups_dict.values())
                self.y_group_titles = new_groups
                self.num_replicates = max(1, max_reps)
                self.y_data = np.full(
                    (len(new_groups), self.num_rows, self.num_replicates),
                    np.nan,
                    dtype=np.float64,
                )
                for g_idx, g_name in enumerate(new_groups):
                    for s_idx, (_, c_name) in enumerate(groups_dict[g_name]):
                        if s_idx < self.num_replicates:
                            self.y_data[g_idx, :num_rows, s_idx] = pd.to_numeric(
                                df[c_name].iloc[:num_rows], errors="coerce"
                            ).to_numpy()
            else:
                self.y_group_titles = [str(c) for c in y_cols]
                self.num_replicates = 1
                self.y_data = np.full(
                    (len(self.y_group_titles), self.num_rows, 1),
                    np.nan,
                    dtype=np.float64,
                )
                for g_idx, c_name in enumerate(y_cols):
                    self.y_data[g_idx, :num_rows, 0] = pd.to_numeric(
                        df[c_name].iloc[:num_rows], errors="coerce"
                    ).to_numpy()

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        table_id: str = "data_df",
        title: str = "DataFrame Table",
        table_type: TableType = TableType.XY,
    ) -> DataTable:
        """Creates a DataTable from a pandas DataFrame."""
        tbl = cls(
            table_id=table_id,
            title=title,
            table_type=table_type,
            num_rows=max(1, len(df)),
            num_replicates=1,
            y_group_titles=["Group A"],
        )
        tbl.update_from_dataframe(df)
        return tbl

    def set_replicates(self, new_replicates: int) -> None:
        """Expands or trims the number of subcolumn replicates."""
        if new_replicates <= 0:
            raise ValueError("Replicate count must be at least 1")
        if new_replicates == self.num_replicates:
            return

        old_replicates = self.num_replicates
        self.num_replicates = new_replicates

        new_y_data = np.full(
            (len(self.y_group_titles), self.num_rows, new_replicates),
            np.nan,
            dtype=np.float64,
        )
        min_reps = min(old_replicates, new_replicates)
        new_y_data[:, :, :min_reps] = self.y_data[:, :, :min_reps]
        self.y_data = new_y_data

        # Filter out excluded points and colors that are now out of bounds
        self.excluded_points = {
            pt for pt in self.excluded_points if pt[1] == "X" or pt[3] < new_replicates
        }
        self.cell_colors = {
            k: v for k, v in self.cell_colors.items() if k[1] == "X" or k[3] < new_replicates
        }

    def _init_default_variables(self) -> None:
        """Initializes default columns for a Multiple Variables table."""
        self.variables = [
            VariableMetadata("Treatment", VariableType.CATEGORICAL, ["Control", "Treated"]),
            VariableMetadata("Dose_uM", VariableType.CONTINUOUS),
            VariableMetadata("Subject_ID", VariableType.LABEL),
        ]
        self.multi_var_data = [["" for _ in self.variables] for _ in range(self.num_rows)]

    def add_variable(
        self,
        name: str,
        var_type: VariableType = VariableType.CONTINUOUS,
        levels: Optional[List[str]] = None,
    ) -> int:
        """Adds a new variable column to a Multiple Variables table."""
        var = VariableMetadata(name, var_type, levels)
        idx = len(self.variables)
        self.variables.append(var)
        for row in self.multi_var_data:
            row.append("" if var_type != VariableType.CONTINUOUS else np.nan)
        return idx

    def set_variable_value(self, row: int, col: int, value: Any) -> None:
        """Sets a cell value in a Multiple Variables table."""
        if row < 0 or row >= self.num_rows or col < 0 or col >= len(self.variables):
            return
        var = self.variables[col]
        if var.var_type == VariableType.CONTINUOUS:
            try:
                val = np.nan if (value is None or str(value).strip() == "") else float(value)
            except (ValueError, TypeError):
                val = np.nan
            self.multi_var_data[row][col] = val
        else:
            val_str = "" if value is None else str(value).strip()
            self.multi_var_data[row][col] = val_str
            if var.var_type == VariableType.CATEGORICAL and val_str:
                if val_str not in var.levels:
                    var.levels.append(val_str)

    def get_variable_value(self, row: int, col: int) -> Any:
        """Gets a cell value from a Multiple Variables table."""
        if row < 0 or row >= self.num_rows or col < 0 or col >= len(self.variables):
            return None
        return self.multi_var_data[row][col]

    def get_levels(self, col: int) -> List[str]:
        """Returns the distinct levels in order for a categorical variable."""
        if 0 <= col < len(self.variables):
            return list(self.variables[col].levels)
        return []

    def reorder_levels(self, col: int, new_order: List[str]) -> None:
        """Reorders the categorical levels for a variable."""
        if 0 <= col < len(self.variables):
            self.variables[col].levels = list(new_order)

    def get_variable_level_counts(self, col: int) -> Dict[str, int]:
        """Returns frequency breakdown of levels for a categorical variable."""
        if col < 0 or col >= len(self.variables):
            return {}
        counts: Dict[str, int] = {lvl: 0 for lvl in self.variables[col].levels}
        for r in range(self.num_rows):
            val = str(self.multi_var_data[r][col]).strip()
            if val:
                counts[val] = counts.get(val, 0) + 1
        return counts

    def set_nested_subcolumn_title(self, group_idx: int, subcol_idx: int, title: str) -> None:
        """Sets title for a nested sub-replicate (e.g. 'Room 1')."""
        if group_idx not in self.nested_subgroup_titles:
            self.nested_subgroup_titles[group_idx] = ["" for _ in range(self.num_replicates)]
        while len(self.nested_subgroup_titles[group_idx]) <= subcol_idx:
            self.nested_subgroup_titles[group_idx].append("")
        self.nested_subgroup_titles[group_idx][subcol_idx] = title

    def get_nested_subcolumn_title(self, group_idx: int, subcol_idx: int) -> str:
        """Gets title for a nested sub-replicate."""
        if group_idx in self.nested_subgroup_titles:
            titles = self.nested_subgroup_titles[group_idx]
            if subcol_idx < len(titles) and titles[subcol_idx]:
                return titles[subcol_idx]
        return f"Sub-unit {subcol_idx + 1}"

    def set_cell_color(
        self,
        row: int,
        col_type: str,
        group_idx: int = 0,
        subcol_idx: int = 0,
        color: Optional[str] = None,
    ) -> None:
        """Assigns background highlight color (hex string) to a cell."""
        key = (row, col_type, group_idx, subcol_idx)
        if color:
            self.cell_colors[key] = color
        else:
            self.cell_colors.pop(key, None)

    def get_cell_color(
        self,
        row: int,
        col_type: str,
        group_idx: int = 0,
        subcol_idx: int = 0,
    ) -> Optional[str]:
        """Returns background highlight color for a cell, or None."""
        return self.cell_colors.get((row, col_type, group_idx, subcol_idx))

    def set_cell_value(
        self,
        row: int,
        col_type: str,
        group_idx: int,
        subcol_idx: int,
        value: Optional[float],
    ) -> None:
        """Sets the numeric value for a specific cell."""
        if row < 0 or row >= self.num_rows:
            return
        val = np.nan if value is None else float(value)

        # Contingency tables: non-negative integer counts only
        if self.table_type == TableType.CONTINGENCY and not np.isnan(val):
            if val < 0 or not float(val).is_integer():
                raise ValueError(
                    f"Contingency tables only support non-negative integer counts. Got {val}"
                )

        if col_type == "X":
            if 0 <= group_idx < len(self.x_columns):
                self.x_columns[group_idx]["data"][row] = val
            elif self.x_data is not None:
                self.x_data[row] = val
        elif col_type == "Y":
            if 0 <= group_idx < len(self.y_group_titles) and 0 <= subcol_idx < self.num_replicates:
                self.y_data[group_idx, row, subcol_idx] = val

    def get_cell_value(
        self, row: int, col_type: str, group_idx: int = 0, subcol_idx: int = 0
    ) -> float:
        """Retrieves numeric value for a cell, returning np.nan if empty."""
        if row < 0 or row >= self.num_rows:
            return np.nan
        if col_type == "X":
            if 0 <= group_idx < len(self.x_columns):
                return float(self.x_columns[group_idx]["data"][row])
            elif self.x_data is not None:
                return float(self.x_data[row])
            return np.nan
        elif col_type == "Y":
            if 0 <= group_idx < len(self.y_group_titles) and 0 <= subcol_idx < self.num_replicates:
                return float(self.y_data[group_idx, row, subcol_idx])
            return np.nan
        return np.nan

    def is_excluded(
        self, row: int, col_type: str, group_idx: int = 0, subcol_idx: int = 0
    ) -> bool:
        """Checks if a cell is flagged as excluded from calculation."""
        return (row, col_type, group_idx, subcol_idx) in self.excluded_points

    def set_excluded(
        self,
        row: int,
        col_type: str,
        group_idx: int = 0,
        subcol_idx: int = 0,
        excluded: bool = True,
    ) -> None:
        """Toggles the exclusion bitmask for a cell."""
        pt = (row, col_type, group_idx, subcol_idx)
        if excluded:
            self.excluded_points.add(pt)
        else:
            self.excluded_points.discard(pt)

    def get_clean_xy(self, group_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts clean (X, Y) pairs for regression/plotting.
        Expands all valid replicates into flat (X, Y) arrays,
        filtering out missing values (NaN) and excluded points.
        Binds to the associated preceding X column.
        """
        if not self.x_columns or group_idx < 0 or group_idx >= len(self.y_group_titles):
            return np.array([]), np.array([])

        x_idx = self.get_y_associated_x_idx(group_idx)
        if x_idx < 0 or x_idx >= len(self.x_columns):
            x_idx = 0
        x_data = self.x_columns[x_idx]["data"]

        xs: List[float] = []
        ys: List[float] = []

        max_reps = min(self.num_replicates, self.y_data.shape[2]) if self.y_data.ndim >= 3 else self.num_replicates
        for r in range(min(self.num_rows, len(x_data))):
            x_val = x_data[r]
            if np.isnan(x_val) or self.is_excluded(r, "X", x_idx, 0):
                continue
            for s in range(max_reps):
                if self.is_excluded(r, "Y", group_idx, s):
                    continue
                y_val = self.y_data[group_idx, r, s]
                if not np.isnan(y_val):
                    xs.append(x_val)
                    ys.append(y_val)

        return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

    @property
    def y_datasets_dict(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Returns mapping of column/group title to (xs, ys) clean coordinates."""
        result = {}
        for g_idx, title in enumerate(self.y_group_titles):
            key = title if title else f"Group {chr(65 + g_idx) if g_idx < 26 else g_idx + 1}"
            if key in result:
                key = f"{key} ({g_idx + 1})"
            xs, ys = self.get_clean_xy(g_idx)
            result[key] = (xs, ys)
        return result

    def get_group_replicate_summary(self, group_idx: int = 0) -> Dict[str, np.ndarray]:
        """
        Returns Mean, SD, SEM, and N per row for a given group,
        skipping excluded or NaN entries.
        """
        means = np.full(self.num_rows, np.nan)
        sds = np.full(self.num_rows, np.nan)
        sems = np.full(self.num_rows, np.nan)
        ns = np.zeros(self.num_rows, dtype=int)

        if group_idx < 0 or group_idx >= len(self.y_group_titles):
            return {"mean": means, "sd": sds, "sem": sems, "n": ns}

        max_reps = min(self.num_replicates, self.y_data.shape[2]) if self.y_data.ndim >= 3 else self.num_replicates
        for r in range(self.num_rows):
            valid_vals = []
            for s in range(max_reps):
                if not self.is_excluded(r, "Y", group_idx, s):
                    val = self.y_data[group_idx, r, s]
                    if not np.isnan(val):
                        valid_vals.append(val)
            if valid_vals:
                arr = np.array(valid_vals, dtype=np.float64)
                n = len(arr)
                means[r] = np.mean(arr)
                ns[r] = n
                if n > 1:
                    sd = float(np.std(arr, ddof=1))
                    sds[r] = sd
                    sems[r] = sd / np.sqrt(n)
                else:
                    sds[r] = 0.0
                    sems[r] = 0.0

        return {"mean": means, "sd": sds, "sem": sems, "n": ns}

    def get_all_group_values(self) -> Dict[str, List[float]]:
        """
        Returns all valid non-excluded values per group (for One-Way ANOVA).
        """
        data: Dict[str, List[float]] = {}
        max_reps = min(self.num_replicates, self.y_data.shape[2]) if self.y_data.ndim >= 3 else self.num_replicates
        for g_idx, title in enumerate(self.y_group_titles):
            vals: List[float] = []
            for r in range(self.num_rows):
                for s in range(max_reps):
                    if not self.is_excluded(r, "Y", g_idx, s):
                        val = self.y_data[g_idx, r, s]
                        if not np.isnan(val):
                            vals.append(float(val))
            data[title] = vals
        return data

    def to_dict(self) -> Dict[str, Any]:
        """Serializes DataTable state to JSON-compatible dictionary."""
        return {
            "table_id": self.table_id,
            "title": self.title,
            "table_type": self.table_type.value,
            "x_format": self.x_format.value,
            "y_format": self.y_format.value,
            "num_rows": self.num_rows,
            "num_replicates": self.num_replicates,
            "row_titles": self.row_titles,
            "x_title": self.x_title,
            "x_data": self.x_data.tolist() if self.x_data is not None else None,
            "x_columns": [
                {"title": c["title"], "data": c["data"].tolist()}
                for c in self.x_columns
            ],
            "column_order": [list(item) for item in self.column_order],
            "y_group_titles": self.y_group_titles,
            "y_data": self.y_data.tolist(),
            "excluded_points": list(self.excluded_points),
            "cell_colors": [
                [r, c_type, g_idx, s_idx, color]
                for (r, c_type, g_idx, s_idx), color in self.cell_colors.items()
            ],
            "nested_subgroup_titles": {
                str(k): v for k, v in self.nested_subgroup_titles.items()
            },
            "variables": [v.to_dict() for v in self.variables],
            "multi_var_data": self.multi_var_data,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DataTable:
        """Deserializes DataTable from JSON-compatible dictionary."""
        tbl = cls(
            table_id=data.get("table_id", "data_1"),
            title=data.get("title", "Data Table"),
            table_type=TableType(data.get("table_type", "XY")),
            num_rows=data.get("num_rows", 20),
            x_format=XFormat(data.get("x_format", "numbers")),
            y_format=YFormat(data.get("y_format", "replicates")),
            num_replicates=data.get("num_replicates", 3),
            y_group_titles=data.get("y_group_titles"),
        )
        tbl.row_titles = data.get("row_titles", ["" for _ in range(tbl.num_rows)])
        tbl.x_title = data.get("x_title", "X")
        if "x_columns" in data and data["x_columns"]:
            tbl.x_columns = [
                {"title": c.get("title", "X"), "data": np.array(c["data"], dtype=np.float64)}
                for c in data["x_columns"]
            ]
        elif data.get("x_data") is not None:
            tbl.x_columns = [{
                "title": data.get("x_title", "X"),
                "data": np.array(data["x_data"], dtype=np.float64),
            }]
        else:
            tbl.x_columns = []

        tbl.y_group_titles = data.get("y_group_titles", ["Group A", "Group B"])
        if data.get("y_data") is not None:
            tbl.y_data = np.array(data["y_data"], dtype=np.float64)

        if "column_order" in data and data["column_order"]:
            tbl.column_order = [(item[0], int(item[1])) for item in data["column_order"]]
        else:
            tbl.column_order = []
            if tbl.x_columns:
                tbl.column_order.append(("X", 0))
            for g_idx in range(len(tbl.y_group_titles)):
                tbl.column_order.append(("Y", g_idx))

        tbl.excluded_points = {
            (pt[0], pt[1], pt[2], pt[3]) for pt in data.get("excluded_points", [])
        }
        # Restore cell colors
        for item in data.get("cell_colors", []):
            if len(item) == 5:
                tbl.cell_colors[(item[0], item[1], item[2], item[3])] = item[4]
        # Restore nested titles
        if "nested_subgroup_titles" in data:
            tbl.nested_subgroup_titles = {
                int(k): v for k, v in data["nested_subgroup_titles"].items()
            }
        # Restore variables
        if "variables" in data and data["variables"]:
            tbl.variables = [VariableMetadata.from_dict(v) for v in data["variables"]]
        if "multi_var_data" in data and data["multi_var_data"]:
            tbl.multi_var_data = data["multi_var_data"]
        return tbl
