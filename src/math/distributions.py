"""
Distribution Statistics and Profile Engine for GemGraph.
Provides five-number summary (Tukey, min-max, percentiles) and
Gaussian KDE profiles for Box and Violin plots.
"""

from __future__ import annotations
from typing import Dict, Any
import numpy as np
from scipy import stats


def five_number_summary(
    values: Any,
    whisker_rule: str = "tukey",
) -> Dict[str, Any]:
    """
    Computes five-number summary and outliers for a numeric distribution.
    whisker_rule:
      - 'tukey': Whiskers extend to the furthest observations within 1.5 * IQR
                 from Q1 and Q3; values outside are marked as outliers.
      - 'minmax': Whiskers extend to minimum and maximum values; no outliers.
      - 'percentile_5_95': Whiskers extend to 5th and 95th percentiles; values
                           outside are marked as outliers.
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]

    if len(v) == 0:
        return {
            "q1": np.nan,
            "median": np.nan,
            "q3": np.nan,
            "iqr": np.nan,
            "whisker_low": np.nan,
            "whisker_high": np.nan,
            "outliers": np.array([], dtype=np.float64),
        }

    q1 = float(np.percentile(v, 25))
    median = float(np.percentile(v, 50))
    q3 = float(np.percentile(v, 75))
    iqr = float(q3 - q1)

    rule = whisker_rule.lower().strip()
    if rule == "tukey":
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr

        in_bounds = v[(v >= lower_fence) & (v <= upper_fence)]
        whisker_low = float(np.min(in_bounds)) if len(in_bounds) > 0 else q1
        whisker_high = float(np.max(in_bounds)) if len(in_bounds) > 0 else q3

        outliers = v[(v < lower_fence) | (v > upper_fence)]

    elif rule == "minmax":
        whisker_low = float(np.min(v))
        whisker_high = float(np.max(v))
        outliers = np.array([], dtype=np.float64)

    elif rule in ("percentile_5_95", "5_95", "percentiles"):
        whisker_low = float(np.percentile(v, 5))
        whisker_high = float(np.percentile(v, 95))
        outliers = v[(v < whisker_low) | (v > whisker_high)]

    else:
        raise ValueError(f"Unknown whisker_rule: {whisker_rule}")

    return {
        "q1": q1,
        "median": median,
        "q3": q3,
        "iqr": iqr,
        "whisker_low": whisker_low,
        "whisker_high": whisker_high,
        "outliers": np.array(outliers, dtype=np.float64),
    }


def gaussian_kde_profile(
    values: Any,
    n_points: int = 128,
    bw_method: str = "scott",
    clip: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Computes a mirrored Gaussian Kernel Density Estimation (KDE) profile.
    Normalized so the maximum half-width is 1.0.
    Falls back safely (flat line) when N < 3 or variance == 0.
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]

    # Safe fallback if N < 3 or zero variance
    if len(v) < 3 or np.isclose(float(np.var(v)), 0.0):
        if len(v) == 0:
            grid = np.linspace(0.0, 1.0, n_points)
        else:
            val = float(v[0])
            grid = np.linspace(val - 1.0, val + 1.0, n_points)
        density = np.zeros(n_points, dtype=np.float64)
        half_width = np.zeros(n_points, dtype=np.float64)
        return {
            "grid": grid,
            "density": density,
            "half_width": half_width,
        }

    kde = stats.gaussian_kde(v, bw_method=bw_method)
    v_min = float(np.min(v))
    v_max = float(np.max(v))

    if clip:
        grid = np.linspace(v_min, v_max, n_points)
    else:
        span = v_max - v_min
        pad = span * 0.1 if span > 0 else 1.0
        grid = np.linspace(v_min - pad, v_max + pad, n_points)

    density = kde(grid)
    max_d = float(np.max(density)) if len(density) > 0 else 0.0
    half_width = (density / max_d) if max_d > 0.0 else np.zeros_like(density)

    return {
        "grid": grid,
        "density": density,
        "half_width": half_width,
    }
