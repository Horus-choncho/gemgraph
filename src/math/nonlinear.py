"""
Nonlinear Regression Engine for GemGraph.
Derived from prism.exe Levenberg-Marquardt solver (REA-PRISM-NUM-301).
"""

from __future__ import annotations
import enum
from typing import Callable, Dict, List, Optional, Tuple, Any
import numpy as np
from scipy import optimize
from scipy import stats


class ConvergencePreset(enum.Enum):
    STANDARD = "Standard"  # Delta SS / SS < 1e-4
    STRICT = "Strict"      # Delta SS / SS < 1e-6


class RegressionModel:
    """Represents a mathematical model with parameter names, function, and initial guess heuristic."""

    def __init__(
        self,
        name: str,
        param_names: List[str],
        func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        heuristic: Callable[[np.ndarray, np.ndarray], np.ndarray],
        description: str = "",
    ) -> None:
        self.name = name
        self.param_names = param_names
        self.func = func
        self.heuristic = heuristic
        self.description = description

    def evaluate(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        return self.func(x, params)


def _four_param_logistic(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    # p = [Bottom, Top, LogEC50, HillSlope]
    bottom, top, logec50, hillslope = p
    power = np.clip((logec50 - x) * hillslope, -100.0, 100.0)
    return bottom + (top - bottom) / (1.0 + 10.0 ** power)


def _four_param_heuristic(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    bottom = float(np.min(y))
    top = float(np.max(y))
    logec50 = float(np.median(x))
    hillslope = 1.0 if y[-1] >= y[0] else -1.0
    return np.array([bottom, top, logec50, hillslope], dtype=np.float64)


def _one_site_binding(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    # p = [Bmax, Kd]
    bmax, kd = p
    return (bmax * x) / (np.maximum(kd + x, 1e-12))


def _one_site_heuristic(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    bmax = float(np.max(y)) * 1.1
    half_y = np.max(y) / 2.0
    idx = int(np.argmin(np.abs(y - half_y)))
    kd = float(np.maximum(x[idx], 1e-6))
    return np.array([bmax, kd], dtype=np.float64)


def _exponential_decay(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    # p = [Y0, Plateau, K]
    y0, plateau, k = p
    arg = np.clip(-k * x, -100.0, 100.0)
    return (y0 - plateau) * np.exp(arg) + plateau


def _exponential_decay_heuristic(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    y0 = float(y[0])
    plateau = float(y[-1])
    span_x = np.ptp(x) if np.ptp(x) > 0 else 1.0
    k = 1.0 / span_x
    return np.array([y0, plateau, k], dtype=np.float64)


def _linear_model(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    # p = [Slope, Intercept]
    slope, intercept = p
    return slope * x + intercept


def _linear_heuristic(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if len(x) > 1 and np.ptp(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        return np.array([float(slope), float(intercept)], dtype=np.float64)
    return np.array([1.0, 0.0], dtype=np.float64)


STANDARD_MODELS: Dict[str, RegressionModel] = {
    "Sigmoidal Dose-Response (4PL)": RegressionModel(
        name="Sigmoidal Dose-Response (4PL)",
        param_names=["Bottom", "Top", "LogEC50", "HillSlope"],
        func=_four_param_logistic,
        heuristic=_four_param_heuristic,
        description="Y = Bottom + (Top - Bottom) / (1 + 10^((LogEC50 - X) * HillSlope))",
    ),
    "One Site Specific Binding": RegressionModel(
        name="One Site Specific Binding",
        param_names=["Bmax", "Kd"],
        func=_one_site_binding,
        heuristic=_one_site_heuristic,
        description="Y = Bmax * X / (Kd + X)",
    ),
    "One Phase Exponential Decay": RegressionModel(
        name="One Phase Exponential Decay",
        param_names=["Y0", "Plateau", "K"],
        func=_exponential_decay,
        heuristic=_exponential_decay_heuristic,
        description="Y = (Y0 - Plateau) * exp(-K * X) + Plateau",
    ),
    "Straight Line (Linear)": RegressionModel(
        name="Straight Line (Linear)",
        param_names=["Slope", "Intercept"],
        func=_linear_model,
        heuristic=_linear_heuristic,
        description="Y = Slope * X + Intercept",
    ),
}


class NonlinearEngine:
    """
    Production-grade Levenberg-Marquardt solver adhering to GraphPad Prism convergence
    rules, iteration constraints, parameter dependency, and goodness-of-fit metrics.
    """

    MAX_ITERATIONS = 1000

    @classmethod
    def fit(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        model: RegressionModel,
        preset: ConvergencePreset = ConvergencePreset.STANDARD,
        initial_params: Optional[np.ndarray] = None,
        confidence_level: float = 0.95,
    ) -> Dict[str, Any]:
        """
        Executes Levenberg-Marquardt curve fitting.
        Tolerances:
          Standard: ftol = 1e-4
          Strict: ftol = 1e-6
        """
        x_clean = np.asarray(x, dtype=np.float64)
        y_clean = np.asarray(y, dtype=np.float64)

        # Filter out any non-finite values
        mask = np.isfinite(x_clean) & np.isfinite(y_clean)
        x_valid = x_clean[mask]
        y_valid = y_clean[mask]

        n_pts = len(x_valid)
        n_params = len(model.param_names)

        if n_pts <= n_params:
            raise ValueError(
                f"Too few points ({n_pts}) to fit model with {n_params} parameters"
            )

        if initial_params is None:
            p0 = model.heuristic(x_valid, y_valid)
        else:
            p0 = np.asarray(initial_params, dtype=np.float64)

        # Set tolerance according to recovered Prism presets
        ftol = 1e-4 if preset == ConvergencePreset.STANDARD else 1e-6

        def residuals(p: np.ndarray) -> np.ndarray:
            y_pred = model.evaluate(x_valid, p)
            return y_valid - y_pred

        # Execute LM least squares
        result = optimize.least_squares(
            residuals,
            p0,
            method="lm",
            ftol=ftol,
            xtol=ftol,
            gtol=ftol,
            max_nfev=cls.MAX_ITERATIONS,
        )

        p_opt = result.x
        res = result.fun
        ss_res = float(np.sum(res ** 2))
        ss_tot = float(np.sum((y_valid - np.mean(y_valid)) ** 2))
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        df = n_pts - n_params
        syx = np.sqrt(ss_res / df) if df > 0 else np.nan

        # AICc calculation
        # AICc = N * ln(SS/N) + 2K + 2K(K+1)/(N - K - 1)
        if ss_res > 0 and n_pts > n_params + 1:
            aicc = float(
                n_pts * np.log(ss_res / n_pts)
                + 2 * n_params
                + (2 * n_params * (n_params + 1)) / (n_pts - n_params - 1)
            )
        else:
            aicc = np.nan

        # Parameter Covariance & Standard Errors
        # J^T J
        jac = result.jac
        j_t_j = jac.T @ jac

        try:
            # Covariance matrix unscaled: C = (J^T J)^-1
            cov_unscaled = np.linalg.inv(j_t_j)
            cov_matrix = cov_unscaled * (ss_res / df)
            std_errors = np.sqrt(np.maximum(np.diag(cov_matrix), 0.0))

            # Dependency calculation:
            # Dependency_i = 1 - 1 / ( (C)_ii * (C^-1)_ii )
            # Here C = cov_unscaled, C^-1 = j_t_j
            dependencies = np.zeros(n_params, dtype=np.float64)
            for i in range(n_params):
                c_ii = cov_unscaled[i, i]
                c_inv_ii = j_t_j[i, i]
                product = c_ii * c_inv_ii
                if product >= 1.0:
                    dependencies[i] = max(0.0, min(1.0, 1.0 - (1.0 / product)))
                else:
                    dependencies[i] = 0.0
        except np.linalg.LinAlgError:
            cov_matrix = np.full((n_params, n_params), np.nan)
            std_errors = np.full(n_params, np.nan)
            dependencies = np.full(n_params, np.nan)

        # 95% Confidence intervals using Student-t distribution
        t_crit = stats.t.ppf((1.0 + confidence_level) / 2.0, df=df) if df > 0 else np.nan
        ci_lower = p_opt - t_crit * std_errors
        ci_upper = p_opt + t_crit * std_errors

        # Dense curve generation for smooth plot rendering
        x_dense = np.linspace(np.min(x_valid), np.max(x_valid), 250)
        y_dense = model.evaluate(x_dense, p_opt)

        param_results = []
        for i, name in enumerate(model.param_names):
            param_results.append({
                "name": name,
                "best_fit": float(p_opt[i]),
                "std_error": float(std_errors[i]),
                "ci_lower": float(ci_lower[i]),
                "ci_upper": float(ci_upper[i]),
                "dependency": float(dependencies[i]),
            })

        return {
            "converged": bool(result.success),
            "status_message": str(result.message),
            "iterations": int(result.nfev),
            "model_name": model.name,
            "preset": preset.value,
            "n_points": n_pts,
            "degrees_of_freedom": df,
            "ss_res": ss_res,
            "ss_tot": ss_tot,
            "r_squared": float(r_squared),
            "syx": float(syx),
            "aicc": float(aicc),
            "parameters": param_results,
            "x_dense": x_dense,
            "y_dense": y_dense,
            "p_opt": p_opt,
        }
