"""
Mathematical engines for GemGraph: Nonlinear regression, outlier detection, and ANOVA.
"""

from src.math.nonlinear import (
    NonlinearEngine,
    ConvergencePreset,
    RegressionModel,
    STANDARD_MODELS,
)
from src.math.outlier import OutlierEngine
from src.math.anova import AnovaEngine

__all__ = [
    "NonlinearEngine",
    "ConvergencePreset",
    "RegressionModel",
    "STANDARD_MODELS",
    "OutlierEngine",
    "AnovaEngine",
]
