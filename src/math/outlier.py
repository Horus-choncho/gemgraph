"""
Outlier Detection Engine for GemGraph (ROUT and Grubbs ESD).
Derived from prism.exe outlier routines (REA-PRISM-ALGO-314).
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
import numpy as np
from scipy import stats


class OutlierEngine:
    """
    Implements Prism's ROUT (Robust regression and Outlier removal)
    and Grubbs ESD test procedures.
    """

    # ROUT standard scaling factor for Median Absolute Deviation
    MAD_SCALE_FACTOR = 1.4826

    @classmethod
    def rout_1d(
        cls,
        data: np.ndarray,
        q_fdr: float = 0.01,  # Default Q = 1.0% = 0.01
    ) -> Dict[str, Any]:
        """
        Executes 1D ROUT outlier detection on a series of values.
        Constraint: N >= 3, 0 < Q <= 1.0.
        """
        vals = np.asarray(data, dtype=np.float64)
        valid_mask = np.isfinite(vals)
        vals_clean = vals[valid_mask]
        n = len(vals_clean)

        if not np.isfinite(q_fdr) or not (0.0 < q_fdr <= 1.0):
            raise ValueError(f"q_fdr must satisfy 0 < Q <= 1, got {q_fdr}")

        if n < 3:
            return {
                "method": "ROUT",
                "n_samples": n,
                "outlier_indices": [],
                "q_fdr": q_fdr,
                "rsdr": np.nan,
                "message": f"Sample size N={n} < 3, cannot perform ROUT outlier detection.",
            }

        # 1. Robust center = median
        center = float(np.median(vals_clean))
        residuals = np.abs(vals_clean - center)

        # 2. RSDR = 1.4826 * MAD, where MAD = median(|residuals|)
        mad = float(np.median(residuals))
        rsdr = cls.MAD_SCALE_FACTOR * mad

        if rsdr <= 1e-12:
            # Fallback if MAD is zero due to multiple identical values
            mean_dev = float(np.mean(residuals))
            rsdr = 1.2533 * mean_dev if mean_dev > 1e-12 else 1e-12

        # 3. Student-t test statistics & p-values
        df = n - 1  # K = 1 for center
        t_stats = residuals / rsdr
        p_values = 2.0 * (1.0 - stats.t.cdf(t_stats, df=df))

        # 4. Benjamini-Hochberg False Discovery Rate (FDR) procedure with threshold Q
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]

        # Critical thresholds: (rank / n) * Q
        ranks = np.arange(1, n + 1)
        fdr_thresholds = (ranks / n) * q_fdr

        # Find largest index k where p_(k) <= (k/n)*Q
        is_significant = sorted_p <= fdr_thresholds
        outlier_clean_indices: List[int] = []

        if np.any(is_significant):
            max_k = np.max(np.where(is_significant)[0])
            outlier_clean_indices = sorted_indices[: max_k + 1].tolist()

        # Map back to original array indices
        original_valid_indices = np.where(valid_mask)[0]
        outlier_indices = [int(original_valid_indices[idx]) for idx in outlier_clean_indices]

        return {
            "method": "ROUT",
            "n_samples": n,
            "q_fdr": q_fdr,
            "rsdr": rsdr,
            "center": center,
            "outlier_indices": outlier_indices,
            "outlier_values": [float(vals[i]) for i in outlier_indices],
            "p_values": p_values.tolist(),
        }

    @classmethod
    def rout_residuals(
        cls,
        residuals: np.ndarray,
        degrees_of_freedom: Optional[int] = None,
        q_fdr: float = 0.01,
        df: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Executes ROUT on regression residuals with degrees of freedom (df = N - K).
        Filters out missing or non-finite residual entries.
        """
        eff_df = degrees_of_freedom if degrees_of_freedom is not None else df
        res_arr = np.asarray(residuals, dtype=np.float64)
        valid_mask = np.isfinite(res_arr)
        clean_res = res_arr[valid_mask]
        n = len(clean_res)

        if eff_df is None or eff_df < 1 or n < 3:
            return {
                "method": "ROUT (Residuals)",
                "n_samples": n,
                "outlier_indices": [],
                "q_fdr": q_fdr,
                "rsdr": np.nan,
                "message": "Too few valid samples or degrees of freedom for ROUT.",
            }

        mad = float(np.median(np.abs(clean_res)))
        rsdr = cls.MAD_SCALE_FACTOR * mad
        if rsdr <= 1e-12:
            mean_dev = float(np.mean(np.abs(clean_res)))
            rsdr = 1.2533 * mean_dev if mean_dev > 1e-12 else 1e-12

        t_stats = np.abs(clean_res) / rsdr
        p_values = 2.0 * (1.0 - stats.t.cdf(t_stats, df=eff_df))

        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        ranks = np.arange(1, n + 1)
        fdr_thresholds = (ranks / n) * q_fdr

        is_sig = sorted_p <= fdr_thresholds
        outlier_clean_indices: List[int] = []
        if np.any(is_sig):
            max_k = np.max(np.where(is_sig)[0])
            outlier_clean_indices = sorted_indices[: max_k + 1].tolist()

        orig_valid_indices = np.where(valid_mask)[0]
        outlier_indices = [int(orig_valid_indices[idx]) for idx in outlier_clean_indices]

        # Reconstruct full p_values array with NaNs for invalid entries
        full_p_values = np.full(len(res_arr), np.nan, dtype=np.float64)
        full_p_values[valid_mask] = p_values

        return {
            "method": "ROUT (Residuals)",
            "n_samples": n,
            "q_fdr": q_fdr,
            "rsdr": rsdr,
            "outlier_indices": outlier_indices,
            "p_values": full_p_values.tolist(),
        }

    @classmethod
    def grubbs_esd(
        cls,
        data: np.ndarray,
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Executes Grubbs Extreme Studentized Deviate (ESD) test.
        Detects significant outliers one at a time for N >= 3.
        """
        vals = np.asarray(data, dtype=np.float64)
        valid_mask = np.isfinite(vals)
        current_data = vals[valid_mask].copy()
        current_indices = np.where(valid_mask)[0].tolist()

        outlier_indices: List[int] = []
        outlier_values: List[float] = []

        while len(current_data) >= 3:
            n = len(current_data)
            mean = np.mean(current_data)
            std = np.std(current_data, ddof=1)

            if std <= 1e-12:
                break

            diff = np.abs(current_data - mean)
            max_idx = int(np.argmax(diff))
            g_stat = diff[max_idx] / std

            # Critical G value:
            # G_crit = (N - 1) / sqrt(N) * sqrt( t^2 / (N - 2 + t^2) )
            # where t = t_ppf( 1 - alpha / (2 * N), df = N - 2 )
            t_crit = stats.t.ppf(1.0 - (alpha / (2.0 * n)), df=n - 2)
            t_sq = t_crit ** 2
            g_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(t_sq / (n - 2 + t_sq))

            if g_stat > g_crit:
                orig_idx = current_indices[max_idx]
                outlier_indices.append(orig_idx)
                outlier_values.append(float(vals[orig_idx]))

                # Remove outlier from working set
                current_data = np.delete(current_data, max_idx)
                del current_indices[max_idx]
            else:
                break

        return {
            "method": "Grubbs ESD",
            "alpha": alpha,
            "outlier_indices": outlier_indices,
            "outlier_values": outlier_values,
            "n_outliers": len(outlier_indices),
        }
