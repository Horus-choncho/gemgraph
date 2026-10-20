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

        # X column data
        if self.table_type in (TableType.XY, TableType.SURVIVAL):
            self.x_title = "Elapsed Time (days)" if self.table_type == TableType.SURVIVAL else "X"
            self.x_data = np.full(self.num_rows, np.nan, dtype=np.float64)
        else:
            self.x_title = ""
            self.x_data = None

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

    def add_row(self, count: int = 1) -> None:
        """Appends empty rows to the table."""
        if count <= 0:
            return
        self.num_rows += count
        self.row_titles.extend(["" for _ in range(count)])
        if self.x_data is not None:
            new_x = np.full(count, np.nan, dtype=np.float64)
            self.x_data = np.concatenate([self.x_data, new_x])

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
        if self.x_data is not None:
            self.x_data = np.delete(self.x_data, row_idx)
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
            if self.x_data is not None:
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
            if self.x_data is not None:
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
        """
        if self.x_data is None or group_idx < 0 or group_idx >= len(self.y_group_titles):
            return np.array([]), np.array([])

        xs: List[float] = []
        ys: List[float] = []

        for r in range(min(self.num_rows, len(self.x_data))):
            x_val = self.x_data[r]
            if np.isnan(x_val) or self.is_excluded(r, "X", 0, 0):
                continue
            for s in range(self.num_replicates):
                if self.is_excluded(r, "Y", group_idx, s):
                    continue
                y_val = self.y_data[group_idx, r, s]
                if not np.isnan(y_val):
                    xs.append(x_val)
                    ys.append(y_val)

        return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

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

        for r in range(self.num_rows):
            valid_vals = []
            for s in range(self.num_replicates):
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
        for g_idx, title in enumerate(self.y_group_titles):
            vals: List[float] = []
            for r in range(self.num_rows):
                for s in range(self.num_replicates):
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
        if data.get("x_data") is not None:
            tbl.x_data = np.array(data["x_data"], dtype=np.float64)
        tbl.y_group_titles = data.get("y_group_titles", ["Group A", "Group B"])
        if data.get("y_data") is not None:
            tbl.y_data = np.array(data["y_data"], dtype=np.float64)
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
