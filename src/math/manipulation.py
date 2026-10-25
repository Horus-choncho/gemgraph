"""
Scientific Table Manipulations for GemGraph.
Implements Normalize, Remove Baseline, Prune Rows, Fraction of Total with CIs,
Transpose, and Extract/Rearrange Multiple Variables.
"""

from __future__ import annotations
import enum
from typing import Optional, Tuple, List, Dict, Any, Union
import numpy as np
import scipy.stats
from src.core.table import DataTable, TableType, VariableType


class Baseline0Method(enum.Enum):
    SMALLEST = "smallest"
    FIRST_ROW = "first_row"
    MANUAL = "manual"


class Baseline100Method(enum.Enum):
    LARGEST = "largest"
    LAST_ROW = "last_row"
    MANUAL = "manual"
    SUM_COLUMN = "sum_column"


class NormalizeScale(enum.Enum):
    FRACTION = 1.0
    PERCENT = 100.0


def normalize_array(
    y: np.ndarray,
    b0_method: Baseline0Method = Baseline0Method.SMALLEST,
    b0_val: float = 0.0,
    b100_method: Baseline100Method = Baseline100Method.LARGEST,
    b100_val: float = 100.0,
    scale: NormalizeScale = NormalizeScale.PERCENT,
) -> np.ndarray:
    """
    Normalizes array according to defined 0% and 100% baselines.
    """
    out = np.full(y.shape, np.nan, dtype=np.float64)
    valid = ~np.isnan(y)
    vals = y[valid]
    if len(vals) == 0:
        return out

    # Determine 0% baseline
    if b0_method == Baseline0Method.SMALLEST:
        b0 = float(np.min(vals))
    elif b0_method == Baseline0Method.FIRST_ROW:
        first_valid = y[~np.isnan(y)]
        b0 = float(first_valid[0]) if len(first_valid) > 0 else 0.0
    elif b0_method == Baseline0Method.MANUAL:
        b0 = float(b0_val)
    else:
        b0 = 0.0

    # Determine 100% baseline
    if b100_method == Baseline100Method.LARGEST:
        b100 = float(np.max(vals))
    elif b100_method == Baseline100Method.LAST_ROW:
        last_valid = y[~np.isnan(y)]
        b100 = float(last_valid[-1]) if len(last_valid) > 0 else 100.0
    elif b100_method == Baseline100Method.MANUAL:
        b100 = float(b100_val)
    elif b100_method == Baseline100Method.SUM_COLUMN:
        b100 = float(np.sum(vals))
    else:
        b100 = 100.0

    denom = b100 - b0
    if denom == 0:
        out[valid] = 0.0
    else:
        out[valid] = ((y[valid] - b0) / denom) * scale.value

    return out


def normalize_table(
    table: DataTable,
    b0_method: Baseline0Method = Baseline0Method.SMALLEST,
    b0_val: float = 0.0,
    b100_method: Baseline100Method = Baseline100Method.LARGEST,
    b100_val: float = 100.0,
    scale: NormalizeScale = NormalizeScale.PERCENT,
) -> DataTable:
    """
    Applies normalization across all treatment groups and replicate subcolumns.
    """
    new_tbl = DataTable(
        table_id=f"{table.table_id}_norm",
        title=f"Normalized of {table.title}",
        table_type=table.table_type,
        num_rows=table.num_rows,
        num_replicates=table.num_replicates,
        y_group_titles=table.y_group_titles,
    )
    new_tbl.row_titles = list(table.row_titles)
    if table.x_data is not None:
        new_tbl.x_data = np.copy(table.x_data)
        new_tbl.x_title = table.x_title

    for g_idx in range(len(table.y_group_titles)):
        for s_idx in range(table.num_replicates):
            col = table.y_data[g_idx, :, s_idx]
            new_tbl.y_data[g_idx, :, s_idx] = normalize_array(
                col, b0_method, b0_val, b100_method, b100_val, scale
            )

    return new_tbl


class BaselineMethod(enum.Enum):
    FIRST_ROW = "first_row"
    LAST_ROW = "last_row"
    MEAN_FIRST_K = "mean_first_k"
    MEAN_LAST_K = "mean_last_k"
    LINEAR_REGRESSION = "linear_regression"


class BaselineOperation(enum.Enum):
    SUBTRACT = "subtract"
    DIVIDE = "divide"


def remove_baseline(
    y: np.ndarray,
    x: Optional[np.ndarray] = None,
    method: BaselineMethod = BaselineMethod.FIRST_ROW,
    operation: BaselineOperation = BaselineOperation.SUBTRACT,
    k_rows: int = 1,
) -> np.ndarray:
    """
    Removes baseline by subtracting or dividing baseline defined by
    first row, last row, mean of first/last K rows, or linear regression against X.
    """
    out = np.full(y.shape, np.nan, dtype=np.float64)
    valid = ~np.isnan(y)
    vals = y[valid]
    if len(vals) == 0:
        return out

    if method == BaselineMethod.FIRST_ROW:
        first_valid = y[~np.isnan(y)]
        b_val = float(first_valid[0]) if len(first_valid) > 0 else 0.0
        baseline = np.full(y.shape, b_val)

    elif method == BaselineMethod.LAST_ROW:
        last_valid = y[~np.isnan(y)]
        b_val = float(last_valid[-1]) if len(last_valid) > 0 else 0.0
        baseline = np.full(y.shape, b_val)

    elif method == BaselineMethod.MEAN_FIRST_K:
        k = max(1, min(k_rows, len(y)))
        k_vals = [v for v in y[:k] if not np.isnan(v)]
        b_val = float(np.mean(k_vals)) if k_vals else 0.0
        baseline = np.full(y.shape, b_val)

    elif method == BaselineMethod.MEAN_LAST_K:
        k = max(1, min(k_rows, len(y)))
        k_vals = [v for v in y[-k:] if not np.isnan(v)]
        b_val = float(np.mean(k_vals)) if k_vals else 0.0
        baseline = np.full(y.shape, b_val)

    elif method == BaselineMethod.LINEAR_REGRESSION:
        if x is not None:
            xy_valid = (~np.isnan(x)) & (~np.isnan(y))
            if np.sum(xy_valid) >= 2 and np.ptp(x[xy_valid]) > 0:
                slope, intercept, _, _, _ = scipy.stats.linregress(x[xy_valid], y[xy_valid])
                baseline = slope * x + intercept
            else:
                baseline = np.full(y.shape, np.mean(vals) if len(vals) > 0 else 0.0)
        else:
            baseline = np.full(y.shape, np.mean(vals) if len(vals) > 0 else 0.0)
    else:
        baseline = np.zeros(y.shape)

    if operation == BaselineOperation.SUBTRACT:
        out[valid] = y[valid] - baseline[valid]
    elif operation == BaselineOperation.DIVIDE:
        denom = baseline[valid]
        res = np.full(len(denom), np.nan, dtype=np.float64)
        nz = denom != 0
        res[nz] = y[valid][nz] / denom[nz]
        out[valid] = res

    return out


def prune_rows(
    table: DataTable,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    decimate_k: Optional[int] = None,
    average_k: Optional[int] = None,
) -> DataTable:
    """
    Prunes rows by range [x_min, x_max], decimation (every K rows),
    or row block averaging (every K rows).
    """
    if table.x_data is not None and (x_min is not None or x_max is not None):
        mask = np.ones(table.num_rows, dtype=bool)
        if x_min is not None:
            mask = mask & (~np.isnan(table.x_data)) & (table.x_data >= x_min)
        if x_max is not None:
            mask = mask & (~np.isnan(table.x_data)) & (table.x_data <= x_max)

        keep_indices = np.where(mask)[0]
        new_rows = len(keep_indices)
        new_tbl = DataTable(
            table_id=f"{table.table_id}_pruned",
            title=f"Pruned {table.title}",
            table_type=table.table_type,
            num_rows=max(1, new_rows),
            num_replicates=table.num_replicates,
            y_group_titles=table.y_group_titles,
        )
        if new_rows > 0:
            new_tbl.x_data = table.x_data[keep_indices]
            new_tbl.y_data = table.y_data[:, keep_indices, :]
            new_tbl.row_titles = [table.row_titles[i] for i in keep_indices]
        return new_tbl

    if decimate_k is not None and decimate_k > 1:
        keep_indices = np.arange(0, table.num_rows, decimate_k)
        new_rows = len(keep_indices)
        new_tbl = DataTable(
            table_id=f"{table.table_id}_decimated",
            title=f"Decimated {table.title}",
            table_type=table.table_type,
            num_rows=max(1, new_rows),
            num_replicates=table.num_replicates,
            y_group_titles=table.y_group_titles,
        )
        if table.x_data is not None:
            new_tbl.x_data = table.x_data[keep_indices]
        new_tbl.y_data = table.y_data[:, keep_indices, :]
        new_tbl.row_titles = [table.row_titles[i] for i in keep_indices]
        return new_tbl

    if average_k is not None and average_k > 1:
        num_blocks = int(np.ceil(table.num_rows / average_k))
        new_tbl = DataTable(
            table_id=f"{table.table_id}_averaged",
            title=f"Averaged {table.title}",
            table_type=table.table_type,
            num_rows=num_blocks,
            num_replicates=table.num_replicates,
            y_group_titles=table.y_group_titles,
        )
        if table.x_data is not None:
            new_x = np.full(num_blocks, np.nan)
            for b in range(num_blocks):
                sub = table.x_data[b * average_k : (b + 1) * average_k]
                valid_sub = sub[~np.isnan(sub)]
                if len(valid_sub) > 0:
                    new_x[b] = np.mean(valid_sub)
            new_tbl.x_data = new_x

        for g in range(len(table.y_group_titles)):
            for s in range(table.num_replicates):
                for b in range(num_blocks):
                    sub = table.y_data[g, b * average_k : (b + 1) * average_k, s]
                    valid_sub = sub[~np.isnan(sub)]
                    if len(valid_sub) > 0:
                        new_tbl.y_data[g, b, s] = np.mean(valid_sub)
        return new_tbl

    return table


def wilson_score_interval(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Computes Wilson/Brown score confidence interval for binomial proportion."""
    if n <= 0 or k < 0 or k > n:
        return 0.0, 0.0
    p = k / n
    alpha = 1.0 - confidence
    z = scipy.stats.norm.ppf(1.0 - alpha / 2.0)
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2.0 * n)) / denom
    half = (z * np.sqrt(max(0.0, (p * (1.0 - p) + (z**2) / (4.0 * n)) / n))) / denom
    lower = max(0.0, float(center - half))
    upper = min(1.0, float(center + half))
    return lower, upper


def clopper_pearson_interval(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Computes exact Clopper-Pearson confidence interval for binomial proportion."""
    if n <= 0 or k < 0 or k > n:
        return 0.0, 0.0
    alpha = 1.0 - confidence
    lower = 0.0 if k == 0 else float(scipy.stats.beta.ppf(alpha / 2.0, k, n - k + 1))
    upper = 1.0 if k == n else float(scipy.stats.beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lower, upper


def fraction_of_total(
    table: DataTable,
    total_type: str = "column",
    scale: NormalizeScale = NormalizeScale.PERCENT,
) -> Dict[str, Any]:
    """
    Computes fraction of total across:
    - 'column': fraction of each column's total
    - 'row': fraction of each row's total across groups
    - 'grand': fraction of grand sum of all cells
    Also computes 95% Wilson and Clopper-Pearson confidence intervals.
    """
    data = table.y_data  # shape (num_groups, num_rows, num_replicates)
    num_groups, num_rows, num_reps = data.shape

    fractions = np.full(data.shape, np.nan, dtype=np.float64)
    ci_wilson = np.full((num_groups, num_rows, num_reps, 2), np.nan, dtype=np.float64)
    ci_clopper = np.full((num_groups, num_rows, num_reps, 2), np.nan, dtype=np.float64)

    if total_type == "column":
        for g in range(num_groups):
            for s in range(num_reps):
                col = data[g, :, s]
                valid = ~np.isnan(col)
                tot = np.sum(col[valid])
                if tot > 0:
                    fractions[g, valid, s] = (col[valid] / tot) * scale.value
                    # Compute CIs if counts/integers
                    for r in np.where(valid)[0]:
                        k = int(round(col[r]))
                        n = int(round(tot))
                        if 0 <= k <= n and n > 0:
                            w_lo, w_hi = wilson_score_interval(k, n)
                            c_lo, c_hi = clopper_pearson_interval(k, n)
                            ci_wilson[g, r, s] = [w_lo * scale.value, w_hi * scale.value]
                            ci_clopper[g, r, s] = [c_lo * scale.value, c_hi * scale.value]

    elif total_type == "row":
        for r in range(num_rows):
            for s in range(num_reps):
                row_vals = data[:, r, s]
                valid = ~np.isnan(row_vals)
                tot = np.sum(row_vals[valid])
                if tot > 0:
                    fractions[valid, r, s] = (row_vals[valid] / tot) * scale.value
                    for g in np.where(valid)[0]:
                        k = int(round(row_vals[g]))
                        n = int(round(tot))
                        if 0 <= k <= n and n > 0:
                            w_lo, w_hi = wilson_score_interval(k, n)
                            c_lo, c_hi = clopper_pearson_interval(k, n)
                            ci_wilson[g, r, s] = [w_lo * scale.value, w_hi * scale.value]
                            ci_clopper[g, r, s] = [c_lo * scale.value, c_hi * scale.value]

    elif total_type == "grand":
        valid = ~np.isnan(data)
        tot = np.sum(data[valid])
        if tot > 0:
            fractions[valid] = (data[valid] / tot) * scale.value
            indices = np.argwhere(valid)
            n = int(round(tot))
            for g, r, s in indices:
                k = int(round(data[g, r, s]))
                if 0 <= k <= n and n > 0:
                    w_lo, w_hi = wilson_score_interval(k, n)
                    c_lo, c_hi = clopper_pearson_interval(k, n)
                    ci_wilson[g, r, s] = [w_lo * scale.value, w_hi * scale.value]
                    ci_clopper[g, r, s] = [c_lo * scale.value, c_hi * scale.value]

    return {
        "fractions": fractions,
        "ci_wilson": ci_wilson,
        "ci_clopper": ci_clopper,
        "scale": scale.value,
    }


def transpose_table(table: DataTable) -> DataTable:
    """
    Transposes table: interchanges rows and columns,
    mapping row titles to column headers and vice versa.
    """
    # Assuming single replicate for clean transposition
    num_cols = len(table.y_group_titles)
    new_num_rows = num_cols
    new_group_titles = [
        table.row_titles[r] if (r < len(table.row_titles) and table.row_titles[r]) else f"Row {r+1}"
        for r in range(table.num_rows)
    ]

    new_tbl = DataTable(
        table_id=f"{table.table_id}_transposed",
        title=f"Transposed {table.title}",
        table_type=TableType.COLUMN,
        num_rows=new_num_rows,
        num_replicates=1,
        y_group_titles=new_group_titles,
    )
    new_tbl.row_titles = list(table.y_group_titles)

    for g_idx in range(num_cols):
        for r_idx in range(table.num_rows):
            val = table.get_cell_value(r_idx, "Y", g_idx, 0)
            new_tbl.set_cell_value(g_idx, "Y", r_idx, 0, val)

    return new_tbl


def extract_and_rearrange(
    table: DataTable,
    target_type: TableType,
    mapping: Dict[str, Any],
) -> DataTable:
    """
    Restructures a Multiple Variables table into XY, Column, Grouped, or Contingency:
    - XY: mapping = {'x_var': col_idx, 'y_vars': [col_idx, ...]}
    - Column: mapping = {'val_var': col_idx, 'group_var': col_idx}
    - Grouped: mapping = {'val_var': col_idx, 'row_factor': col_idx, 'col_factor': col_idx}
    - Contingency: mapping = {'row_var': col_idx, 'col_var': col_idx}
    """
    if table.table_type != TableType.MULTIPLE_VARIABLES:
        raise ValueError("Extract and rearrange requires a Multiple Variables source table")

    if target_type == TableType.XY:
        x_idx = mapping["x_var"]
        y_indices = mapping["y_vars"]
        y_titles = [table.variables[i].name for i in y_indices]
        out_tbl = DataTable(
            table_id=f"{table.table_id}_xy",
            title=f"Extracted XY from {table.title}",
            table_type=TableType.XY,
            num_rows=table.num_rows,
            num_replicates=1,
            y_group_titles=y_titles,
        )
        out_tbl.x_title = table.variables[x_idx].name
        for r in range(table.num_rows):
            x_val = table.get_variable_value(r, x_idx)
            try:
                out_tbl.set_cell_value(r, "X", 0, 0, float(x_val))
            except (ValueError, TypeError):
                pass
            for out_g, y_i in enumerate(y_indices):
                y_val = table.get_variable_value(r, y_i)
                try:
                    out_tbl.set_cell_value(r, "Y", out_g, 0, float(y_val))
                except (ValueError, TypeError):
                    pass
        return out_tbl

    elif target_type == TableType.COLUMN:
        val_idx = mapping["val_var"]
        grp_idx = mapping["group_var"]
        levels = table.get_levels(grp_idx)
        if not levels:
            levels = list({str(table.get_variable_value(r, grp_idx)) for r in range(table.num_rows)})

        # Separate values into groups
        grouped_vals: Dict[str, List[float]] = {lvl: [] for lvl in levels}
        for r in range(table.num_rows):
            grp = str(table.get_variable_value(r, grp_idx)).strip()
            v = table.get_variable_value(r, val_idx)
            try:
                f_val = float(v)
                if grp in grouped_vals:
                    grouped_vals[grp].append(f_val)
            except (ValueError, TypeError):
                pass

        max_len = max(len(v) for v in grouped_vals.values()) if grouped_vals else 10
        out_tbl = DataTable(
            table_id=f"{table.table_id}_col",
            title=f"Extracted Column from {table.title}",
            table_type=TableType.COLUMN,
            num_rows=max(1, max_len),
            num_replicates=1,
            y_group_titles=levels,
        )
        for g_idx, lvl in enumerate(levels):
            vals = grouped_vals[lvl]
            for r_idx, val in enumerate(vals):
                out_tbl.set_cell_value(r_idx, "Y", g_idx, 0, val)
        return out_tbl

    elif target_type == TableType.CONTINGENCY:
        row_var = mapping["row_var"]
        col_var = mapping["col_var"]
        row_levels = table.get_levels(row_var) or list({str(table.get_variable_value(r, row_var)) for r in range(table.num_rows)})
        col_levels = table.get_levels(col_var) or list({str(table.get_variable_value(r, col_var)) for r in range(table.num_rows)})

        out_tbl = DataTable(
            table_id=f"{table.table_id}_cnt",
            title=f"Contingency from {table.title}",
            table_type=TableType.CONTINGENCY,
            num_rows=len(row_levels),
            num_replicates=1,
            y_group_titles=col_levels,
        )
        out_tbl.row_titles = row_levels

        # Tally counts
        counts = {r_l: {c_l: 0 for c_l in col_levels} for r_l in row_levels}
        for r in range(table.num_rows):
            r_val = str(table.get_variable_value(r, row_var)).strip()
            c_val = str(table.get_variable_value(r, col_var)).strip()
            if r_val in counts and c_val in counts[r_val]:
                counts[r_val][c_val] += 1

        for r_idx, r_l in enumerate(row_levels):
            for c_idx, c_l in enumerate(col_levels):
                out_tbl.set_cell_value(r_idx, "Y", c_idx, 0, counts[r_l][c_l])

        return out_tbl

    elif target_type == TableType.GROUPED:
        val_idx = mapping["val_var"]
        row_idx = mapping["row_factor"]
        col_idx = mapping["col_factor"]

        row_levels = table.get_levels(row_idx)
        if not row_levels:
            seen_r = []
            for r in range(table.num_rows):
                v = str(table.get_variable_value(r, row_idx)).strip()
                if v and v not in seen_r:
                    seen_r.append(v)
            row_levels = seen_r or ["Row 1"]

        col_levels = table.get_levels(col_idx)
        if not col_levels:
            seen_c = []
            for r in range(table.num_rows):
                v = str(table.get_variable_value(r, col_idx)).strip()
                if v and v not in seen_c:
                    seen_c.append(v)
            col_levels = seen_c or ["Group A"]

        buckets: Dict[Tuple[str, str], List[float]] = {
            (r_l, c_l): [] for r_l in row_levels for c_l in col_levels
        }
        for r in range(table.num_rows):
            r_val = str(table.get_variable_value(r, row_idx)).strip()
            c_val = str(table.get_variable_value(r, col_idx)).strip()
            val = table.get_variable_value(r, val_idx)
            try:
                f_val = float(val)
                if (r_val, c_val) in buckets:
                    buckets[(r_val, c_val)].append(f_val)
            except (ValueError, TypeError):
                pass

        max_reps = max((len(v) for v in buckets.values()), default=1)
        max_reps = max(1, max_reps)

        out_tbl = DataTable(
            table_id=f"{table.table_id}_grouped",
            title=f"Extracted Grouped from {table.title}",
            table_type=TableType.GROUPED,
            num_rows=len(row_levels),
            num_replicates=max_reps,
            y_group_titles=col_levels,
        )
        out_tbl.row_titles = row_levels

        for r_i, r_l in enumerate(row_levels):
            for c_i, c_l in enumerate(col_levels):
                vals = buckets[(r_l, c_l)]
                for s_i, v in enumerate(vals):
                    out_tbl.set_cell_value(r_i, "Y", c_i, s_i, v)

        return out_tbl

    else:
        raise NotImplementedError(f"Extraction to {target_type} not yet implemented")
