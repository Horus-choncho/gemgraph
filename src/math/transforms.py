"""
Scientific Data Transformations for GemGraph.
Derived from GraphPad Prism standard and pharmacological transformation libraries.
"""

from __future__ import annotations
import ast
from typing import Optional, Tuple, List, Dict, Any, Union
import numpy as np
import scipy.stats
from src.core.table import DataTable, TableType, XFormat, YFormat


def arithmetic_transform(y: np.ndarray, op: str, k: float = 1.0) -> np.ndarray:
    """
    Performs standard arithmetic transformation on Y values.
    Preserves NaNs.
    """
    out = np.copy(y).astype(np.float64)
    valid = ~np.isnan(out)

    if op == "Y*K":
        out[valid] = out[valid] * k
    elif op == "Y+K":
        out[valid] = out[valid] + k
    elif op == "Y-K":
        out[valid] = out[valid] - k
    elif op == "Y/K":
        if k == 0:
            out[valid] = np.nan
        else:
            out[valid] = out[valid] / k
    elif op == "Y^2":
        out[valid] = np.power(out[valid], 2)
    elif op == "Y^K":
        out[valid] = np.where(out[valid] >= 0, np.power(out[valid], k), np.nan)
    elif op == "log10":
        out[valid] = np.where(out[valid] > 0, np.log10(out[valid]), np.nan)
    elif op == "-log10":
        out[valid] = np.where(out[valid] > 0, -np.log10(out[valid]), np.nan)
    elif op == "ln":
        out[valid] = np.where(out[valid] > 0, np.log(out[valid]), np.nan)
    elif op == "10^Y":
        out[valid] = np.power(10.0, out[valid])
    elif op == "exp":
        out[valid] = np.exp(out[valid])
    elif op == "1/Y":
        out[valid] = np.where(out[valid] != 0, 1.0 / out[valid], np.nan)
    elif op == "sqrt":
        out[valid] = np.where(out[valid] >= 0, np.sqrt(out[valid]), np.nan)
    elif op == "logit":
        p = out[valid]
        valid_p = (p > 0) & (p < 1)
        res = np.full(len(p), np.nan)
        res[valid_p] = np.log(p[valid_p] / (1.0 - p[valid_p]))
        out[valid] = res
    else:
        raise ValueError(f"Unknown arithmetic transform operation: {op}")

    return out


def probit_transform(y: np.ndarray) -> np.ndarray:
    """
    Computes probit transform with Prism's legacy +5.0 offset:
    probit(Y) = norm.ppf(p) + 5.0
    For p = 0.5 -> 5.0; p = 0.025 -> 3.04 (approx).
    Accepts fractions (0 < p < 1) or percentages (0 < p < 100).
    """
    out = np.copy(y).astype(np.float64)
    valid = ~np.isnan(out)
    vals = out[valid]

    # Convert percentages to fractions if needed
    if np.any(vals > 1.0) and np.all(vals[vals > 0] <= 100.0):
        p = vals / 100.0
    else:
        p = np.copy(vals)

    res = np.full(len(p), np.nan)
    within_bounds = (p > 0) & (p < 1)
    res[within_bounds] = scipy.stats.norm.ppf(p[within_bounds]) + 5.0
    out[valid] = res
    return out


def rank_transform(y: np.ndarray) -> np.ndarray:
    """Computes fractional ranks for Y values, preserving NaNs."""
    out = np.copy(y).astype(np.float64)
    valid = ~np.isnan(out)
    if not np.any(valid):
        return out
    ranks = scipy.stats.rankdata(out[valid], method="average")
    out[valid] = ranks
    return out


def zscore_transform(y: np.ndarray) -> np.ndarray:
    """Computes sample Z-scores: (Y - mean) / sd."""
    out = np.copy(y).astype(np.float64)
    valid = ~np.isnan(out)
    vals = out[valid]
    if len(vals) < 2:
        out[valid] = 0.0
        return out
    mean = np.mean(vals)
    sd = np.std(vals, ddof=1)
    if sd == 0:
        out[valid] = 0.0
    else:
        out[valid] = (vals - mean) / sd
    return out


def trig_transform(y: np.ndarray, func: str) -> np.ndarray:
    """Computes trigonometric functions in radians."""
    out = np.copy(y).astype(np.float64)
    valid = ~np.isnan(out)
    v = out[valid]
    func_l = func.lower()
    if func_l == "sin":
        out[valid] = np.sin(v)
    elif func_l == "cos":
        out[valid] = np.cos(v)
    elif func_l == "tan":
        out[valid] = np.tan(v)
    elif func_l == "arcsin":
        out[valid] = np.where((v >= -1) & (v <= 1), np.arcsin(v), np.nan)
    elif func_l == "arccos":
        out[valid] = np.where((v >= -1) & (v <= 1), np.arccos(v), np.nan)
    elif func_l == "arctan":
        out[valid] = np.arctan(v)
    else:
        raise ValueError(f"Unknown trigonometric function: {func}")
    return out


def abs_transform(y: np.ndarray) -> np.ndarray:
    """Computes absolute value |Y|."""
    return np.abs(y)


def gaussian_noise(y: np.ndarray, sigma: float = 1.0, seed: Optional[int] = None) -> np.ndarray:
    """Adds Gaussian noise Y + N(0, sigma^2)."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, size=y.shape)
    out = np.copy(y).astype(np.float64)
    valid = ~np.isnan(out)
    out[valid] = out[valid] + noise[valid]
    return out


def cross_transform(x: np.ndarray, y: np.ndarray, op: str, k: float = 1.0) -> np.ndarray:
    """
    Computes coordinate cross-transforms between X and Y arrays.
    """
    out = np.full(y.shape, np.nan, dtype=np.float64)
    valid = (~np.isnan(x)) & (~np.isnan(y))

    if op == "X/Y":
        out[valid] = np.where(y[valid] != 0, x[valid] / y[valid], np.nan)
    elif op == "Y/X":
        out[valid] = np.where(x[valid] != 0, y[valid] / x[valid], np.nan)
    elif op == "Y-X":
        out[valid] = y[valid] - x[valid]
    elif op == "Y+X":
        out[valid] = y[valid] + x[valid]
    elif op == "Y*X":
        out[valid] = y[valid] * x[valid]
    elif op == "X-Y":
        out[valid] = x[valid] - y[valid]
    elif op == "K-Y":
        out[valid] = k - y[valid]
    elif op == "K/Y":
        out[valid] = np.where(y[valid] != 0, k / y[valid], np.nan)
    elif op == "log2":
        out[valid] = np.where(y[valid] > 0, np.log2(y[valid]), np.nan)
    elif op == "2^Y":
        out[valid] = np.power(2.0, y[valid])
    elif op == "round":
        out[valid] = np.round(y[valid], int(k))
    else:
        raise ValueError(f"Unknown cross transform: {op}")

    return out


def pharmacological_transform(
    x: np.ndarray,
    y: np.ndarray,
    p_type: str,
    y_max: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pharmacological & biochemical linearized transforms:
    - Lineweaver-Burk: 1/X, 1/Y
    - Eadie-Hofstee: Y/X, Y
    - Hanes-Woolf: X, X/Y
    - Scatchard: Y, Y/X
    - Hill plot: log10(X), log10(Y / (Ymax - Y))
    """
    p_type_l = p_type.lower().replace("-", "_").replace(" ", "_")
    x_out = np.full(x.shape, np.nan, dtype=np.float64)
    y_out = np.full(y.shape, np.nan, dtype=np.float64)
    valid = (~np.isnan(x)) & (~np.isnan(y))

    if p_type_l == "lineweaver_burk":
        valid_nz = valid & (x != 0) & (y != 0)
        x_out[valid_nz] = 1.0 / x[valid_nz]
        y_out[valid_nz] = 1.0 / y[valid_nz]

    elif p_type_l == "eadie_hofstee":
        valid_nz = valid & (x != 0)
        x_out[valid_nz] = y[valid_nz] / x[valid_nz]
        y_out[valid_nz] = y[valid_nz]

    elif p_type_l == "hanes_woolf":
        valid_nz = valid & (y != 0)
        x_out[valid_nz] = x[valid_nz]
        y_out[valid_nz] = x[valid_nz] / y[valid_nz]

    elif p_type_l == "scatchard":
        valid_nz = valid & (x != 0)
        x_out[valid_nz] = y[valid_nz]
        y_out[valid_nz] = y[valid_nz] / x[valid_nz]

    elif p_type_l == "hill_plot":
        valid_y = y[valid]
        actual_ymax = y_max if y_max is not None else (np.max(valid_y) * 1.05 if len(valid_y) > 0 else 100.0)
        valid_hp = valid & (x > 0) & (y > 0) & (y < actual_ymax)
        x_out[valid_hp] = np.log10(x[valid_hp])
        y_out[valid_hp] = np.log10(y[valid_hp] / (actual_ymax - y[valid_hp]))

    else:
        raise ValueError(f"Unknown pharmacological transform: {p_type}")

    return x_out, y_out


def concentration_x_transform(
    x: np.ndarray,
    zero_baseline: Optional[float] = None,
    scale_multiplier: float = 1.0,
    log_base: str = "log10",
) -> np.ndarray:
    """
    Transforms concentration X column:
    - Replaces X=0 with small positive baseline (default: 2 log units below min positive X)
    - Multiplies by unit scaling multiplier
    - Computes log10 or ln
    """
    out = np.copy(x).astype(np.float64)
    valid = ~np.isnan(out)

    pos_x = out[valid & (out > 0)]
    if zero_baseline is None:
        if len(pos_x) > 0:
            min_pos = np.min(pos_x) * scale_multiplier
            zero_baseline = 10.0 ** (np.floor(np.log10(min_pos)) - 2)
        else:
            zero_baseline = 1e-12

    scaled = out * scale_multiplier
    scaled[valid & (out == 0)] = zero_baseline

    if log_base == "log10":
        return np.where(scaled > 0, np.log10(scaled), np.nan)
    elif log_base == "ln":
        return np.where(scaled > 0, np.log(scaled), np.nan)
    return scaled


class SafeTransformEvaluator(ast.NodeVisitor):
    """Safely evaluates mathematical expressions with IF(cond, true_val, false_val) branching."""

    ALLOWED_NODES = {
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Call,
        ast.Name,
        ast.Constant,
        ast.Compare,
        ast.IfExp,
        ast.BoolOp,
        ast.And,
        ast.Or,
        ast.Not,
        ast.Invert,
        ast.Load,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
    }

    def __init__(self, x: np.ndarray, y: np.ndarray, k: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.k = k
        self.math_funcs = {
            "log": np.log,
            "ln": np.log,
            "log10": np.log10,
            "log2": np.log2,
            "exp": np.exp,
            "sqrt": np.sqrt,
            "abs": np.abs,
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
        }

    def evaluate(self, expr: str) -> np.ndarray:
        parsed = ast.parse(expr.strip(), mode="eval")
        self._validate_ast(parsed)
        return self._eval_node(parsed.body)

    def _validate_ast(self, node: ast.AST) -> None:
        if type(node) not in self.ALLOWED_NODES:
            raise ValueError(f"Disallowed AST element in expression: {type(node).__name__}")
        for child in ast.iter_child_nodes(node):
            self._validate_ast(child)

    def _eval_node(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            return node.value

        elif isinstance(node, ast.Name):
            name = node.id.upper()
            if name == "X":
                return self.x
            elif name == "Y":
                return self.y
            elif name == "K":
                return self.k
            elif name == "PI":
                return np.pi
            elif name == "E":
                return np.e
            raise ValueError(f"Unknown variable name: {node.id}")

        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            elif isinstance(node.op, ast.Sub):
                return left - right
            elif isinstance(node.op, ast.Mult):
                return left * right
            elif isinstance(node.op, ast.Div):
                return np.where(right != 0, left / right, np.nan)
            elif isinstance(node.op, ast.Pow):
                return np.power(left, right)

        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            if isinstance(node.op, ast.USub):
                return -operand
            elif isinstance(node.op, ast.UAdd):
                return operand
            elif isinstance(node.op, (ast.Not, ast.Invert)):
                return ~np.asarray(operand, dtype=bool)

        elif isinstance(node, ast.IfExp):
            cond = self._eval_node(node.test)
            val_true = self._eval_node(node.body)
            val_false = self._eval_node(node.orelse)
            return np.where(cond, val_true, val_false)

        elif isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                res = True
                for v_node in node.values:
                    res = res & np.asarray(self._eval_node(v_node), dtype=bool)
                return res
            elif isinstance(node.op, ast.Or):
                res = False
                for v_node in node.values:
                    res = res | np.asarray(self._eval_node(v_node), dtype=bool)
                return res

        elif isinstance(node, ast.Compare):
            left = self._eval_node(node.left)
            res = True
            for op, comparator in zip(node.ops, node.comparators):
                right = self._eval_node(comparator)
                if isinstance(op, ast.Eq):
                    cmp_res = left == right
                elif isinstance(op, ast.NotEq):
                    cmp_res = left != right
                elif isinstance(op, ast.Lt):
                    cmp_res = left < right
                elif isinstance(op, ast.LtE):
                    cmp_res = left <= right
                elif isinstance(op, ast.Gt):
                    cmp_res = left > right
                elif isinstance(op, ast.GtE):
                    cmp_res = left >= right
                else:
                    raise ValueError(f"Unsupported comparator: {type(op).__name__}")
                res = res & cmp_res
                left = right
            return res

        elif isinstance(node, ast.Call):
            func_name = getattr(node.func, "id", "").upper()
            if func_name == "IF":
                if len(node.args) != 3:
                    raise ValueError("IF(condition, value_if_true, value_if_false) requires exactly 3 arguments")
                cond = self._eval_node(node.args[0])
                val_true = self._eval_node(node.args[1])
                val_false = self._eval_node(node.args[2])
                return np.where(cond, val_true, val_false)

            func_l = func_name.lower()
            if func_l in self.math_funcs:
                if len(node.args) != 1:
                    raise ValueError(f"{func_name} expects 1 argument")
                arg = self._eval_node(node.args[0])
                return self.math_funcs[func_l](arg)

            raise ValueError(f"Unsupported function call: {func_name}")

        raise ValueError(f"Cannot evaluate AST node: {type(node).__name__}")


def eval_user_defined_transform(
    expr: str, x: np.ndarray, y: np.ndarray, k: float = 0.0
) -> np.ndarray:
    """Safely evaluates user-defined transform with IF branching logic."""
    evaluator = SafeTransformEvaluator(x, y, k)
    res = evaluator.evaluate(expr)
    if isinstance(res, (int, float)):
        return np.full(y.shape, float(res), dtype=np.float64)
    return np.asarray(res, dtype=np.float64)


def transform_data_table(
    table: DataTable,
    transform_name: str,
    k_constant: float = 1.0,
    user_expr: Optional[str] = None,
    pharma_type: Optional[str] = None,
) -> DataTable:
    """
    Applies transformation across all groups/replicates of a DataTable,
    returning a new linked DataTable.
    """
    new_tbl = DataTable(
        table_id=f"{table.table_id}_tf",
        title=f"Transform of {table.title}",
        table_type=table.table_type,
        num_rows=table.num_rows,
        num_replicates=table.num_replicates,
        y_group_titles=table.y_group_titles,
    )
    new_tbl.row_titles = list(table.row_titles)
    if table.x_data is not None:
        new_tbl.x_data = np.copy(table.x_data)
        new_tbl.x_title = table.x_title

    # Pharmacological transforms alter both X and Y
    if pharma_type is not None:
        if table.x_data is not None and len(table.y_group_titles) > 0:
            for g_idx in range(len(table.y_group_titles)):
                for s_idx in range(table.num_replicates):
                    x_t, y_t = pharmacological_transform(
                        table.x_data, table.y_data[g_idx, :, s_idx], pharma_type
                    )
                    if g_idx == 0 and s_idx == 0:
                        new_tbl.x_data = x_t
                    new_tbl.y_data[g_idx, :, s_idx] = y_t
        return new_tbl

    # User-defined transform
    if user_expr:
        x_vals = table.x_data if table.x_data is not None else np.zeros(table.num_rows)
        for g_idx in range(len(table.y_group_titles)):
            for s_idx in range(table.num_replicates):
                new_tbl.y_data[g_idx, :, s_idx] = eval_user_defined_transform(
                    user_expr, x_vals, table.y_data[g_idx, :, s_idx], k_constant
                )
        return new_tbl

    # Standard arithmetic / probit / rank / zscore
    for g_idx in range(len(table.y_group_titles)):
        for s_idx in range(table.num_replicates):
            y_col = table.y_data[g_idx, :, s_idx]
            if transform_name == "probit":
                new_tbl.y_data[g_idx, :, s_idx] = probit_transform(y_col)
            elif transform_name == "rank":
                new_tbl.y_data[g_idx, :, s_idx] = rank_transform(y_col)
            elif transform_name == "zscore":
                new_tbl.y_data[g_idx, :, s_idx] = zscore_transform(y_col)
            elif transform_name == "abs":
                new_tbl.y_data[g_idx, :, s_idx] = abs_transform(y_col)
            elif transform_name in ("sin", "cos", "tan", "arcsin", "arccos", "arctan"):
                new_tbl.y_data[g_idx, :, s_idx] = trig_transform(y_col, transform_name)
            elif transform_name.startswith("cross_") and table.x_data is not None:
                op = transform_name[6:]
                new_tbl.y_data[g_idx, :, s_idx] = cross_transform(
                    table.x_data, y_col, op, k_constant
                )
            else:
                new_tbl.y_data[g_idx, :, s_idx] = arithmetic_transform(
                    y_col, transform_name, k_constant
                )

    return new_tbl
