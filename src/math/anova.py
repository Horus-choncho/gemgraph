"""
ANOVA and Multiple Comparisons Engine for GemGraph.
Derived from prism.exe group comparison routines (REA-PRISM-STAT-325).
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy import stats
from scipy import integrate


class AnovaEngine:
    """
    Computes One-Way, Two-Way, and Repeated Measures ANOVA,
    and post-hoc comparisons (Dunnett's test vs control and Tukey's HSD test).
    """

    @classmethod
    def one_way(cls, groups: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Executes Ordinary One-Way ANOVA across multiple groups.
        """
        group_names = list(groups.keys())
        cleaned_groups: List[np.ndarray] = []
        names_used: List[str] = []

        for name in group_names:
            vals = np.array(groups[name], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if len(vals) > 0:
                cleaned_groups.append(vals)
                names_used.append(name)

        k = len(cleaned_groups)
        if k < 2:
            raise ValueError(f"One-Way ANOVA requires at least 2 groups with valid data (found {k}).")

        # Sample sizes and totals
        ns = np.array([len(g) for g in cleaned_groups], dtype=int)
        total_n = int(np.sum(ns))
        all_vals = np.concatenate(cleaned_groups)
        grand_mean = float(np.mean(all_vals))

        group_means = np.array([np.mean(g) for g in cleaned_groups], dtype=np.float64)
        group_sds = np.array([np.std(g, ddof=1) if len(g) > 1 else 0.0 for g in cleaned_groups], dtype=np.float64)
        group_sems = group_sds / np.sqrt(ns)

        # Sum of squares
        ss_total = float(np.sum((all_vals - grand_mean) ** 2))
        ss_between = float(np.sum(ns * (group_means - grand_mean) ** 2))
        ss_within = float(np.sum([np.sum((g - m) ** 2) for g, m in zip(cleaned_groups, group_means)]))

        df_between = k - 1
        df_within = total_n - k
        df_total = total_n - 1

        ms_between = ss_between / df_between if df_between > 0 else np.nan
        ms_within = ss_within / df_within if df_within > 0 else np.nan

        f_stat = ms_between / ms_within if ms_within > 0 else np.nan
        p_val = float(1.0 - stats.f.cdf(f_stat, df_between, df_within)) if not np.isnan(f_stat) else np.nan

        # Format summary table
        group_summaries = []
        for i, name in enumerate(names_used):
            group_summaries.append({
                "group": name,
                "n": int(ns[i]),
                "mean": float(group_means[i]),
                "sd": float(group_sds[i]),
                "sem": float(group_sems[i]),
            })

        anova_table = [
            {"source": "Between Groups", "ss": ss_between, "df": df_between, "ms": ms_between, "f": f_stat, "p_value": p_val},
            {"source": "Within Groups (Error)", "ss": ss_within, "df": df_within, "ms": ms_within, "f": None, "p_value": None},
            {"source": "Total", "ss": ss_total, "df": df_total, "ms": None, "f": None, "p_value": None},
        ]

        return {
            "type": "One-Way Ordinary ANOVA",
            "k_groups": k,
            "total_n": total_n,
            "f_stat": float(f_stat),
            "p_value": float(p_val),
            "r_squared": float(ss_between / ss_total) if ss_total > 0 else 0.0,
            "ms_within": float(ms_within),
            "df_within": int(df_within),
            "anova_table": anova_table,
            "group_summaries": group_summaries,
            "cleaned_groups": cleaned_groups,
            "names_used": names_used,
        }

    @classmethod
    def two_way(
        cls,
        data_matrix: np.ndarray,
        row_labels: Optional[List[str]] = None,
        col_labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Executes Ordinary Two-Way ANOVA.
        data_matrix shape: (num_rows, num_cols, num_replicates)
        """
        r_dim, c_dim, rep_dim = data_matrix.shape
        if r_dim < 2 or c_dim < 2 or rep_dim < 1:
            raise ValueError("Two-Way ANOVA requires at least 2 rows, 2 columns, and 1 replicate.")

        row_names = row_labels or [f"Row {i+1}" for i in range(r_dim)]
        col_names = col_labels or [f"Col {j+1}" for j in range(c_dim)]

        total_n = r_dim * c_dim * rep_dim
        grand_mean = float(np.nanmean(data_matrix))

        row_means = np.nanmean(data_matrix, axis=(1, 2))
        col_means = np.nanmean(data_matrix, axis=(0, 2))
        cell_means = np.nanmean(data_matrix, axis=2)

        # Sum of squares
        ss_row = float(c_dim * rep_dim * np.sum((row_means - grand_mean) ** 2))
        ss_col = float(r_dim * rep_dim * np.sum((col_means - grand_mean) ** 2))

        # Interaction SS
        ss_interaction = 0.0
        for i in range(r_dim):
            for j in range(c_dim):
                interaction_term = cell_means[i, j] - row_means[i] - col_means[j] + grand_mean
                ss_interaction += rep_dim * (interaction_term ** 2)

        # Residual SS
        ss_residual = 0.0
        for i in range(r_dim):
            for j in range(c_dim):
                for k in range(rep_dim):
                    val = data_matrix[i, j, k]
                    if np.isfinite(val):
                        ss_residual += (val - cell_means[i, j]) ** 2

        ss_total = ss_row + ss_col + ss_interaction + ss_residual

        df_row = r_dim - 1
        df_col = c_dim - 1
        df_total = total_n - 1

        ms_row = ss_row / df_row
        ms_col = ss_col / df_col

        if rep_dim == 1:
            # Without replicates: Interaction is the error term
            df_residual = df_row * df_col
            ms_residual = ss_interaction / df_residual if df_residual > 0 else np.nan
            f_row = ms_row / ms_residual if ms_residual > 0 else np.nan
            f_col = ms_col / ms_residual if ms_residual > 0 else np.nan
            p_row = float(1.0 - stats.f.cdf(f_row, df_row, df_residual)) if not np.isnan(f_row) else np.nan
            p_col = float(1.0 - stats.f.cdf(f_col, df_col, df_residual)) if not np.isnan(f_col) else np.nan

            anova_table = [
                {"source": "Row Factor", "ss": ss_row, "df": df_row, "ms": ms_row, "f": f_row, "p_value": p_row},
                {"source": "Column Factor", "ss": ss_col, "df": df_col, "ms": ms_col, "f": f_col, "p_value": p_col},
                {"source": "Residual (Interaction)", "ss": ss_interaction, "df": df_residual, "ms": ms_residual, "f": None, "p_value": None},
                {"source": "Total", "ss": ss_total, "df": df_total, "ms": None, "f": None, "p_value": None},
            ]
        else:
            df_interaction = df_row * df_col
            df_residual = r_dim * c_dim * (rep_dim - 1)
            ms_interaction = ss_interaction / df_interaction if df_interaction > 0 else np.nan
            ms_residual = ss_residual / df_residual if df_residual > 0 else np.nan

            f_row = ms_row / ms_residual if ms_residual > 0 else np.nan
            f_col = ms_col / ms_residual if ms_residual > 0 else np.nan
            f_interaction = ms_interaction / ms_residual if ms_residual > 0 else np.nan

            p_row = float(1.0 - stats.f.cdf(f_row, df_row, df_residual)) if not np.isnan(f_row) else np.nan
            p_col = float(1.0 - stats.f.cdf(f_col, df_col, df_residual)) if not np.isnan(f_col) else np.nan
            p_interaction = (
                float(1.0 - stats.f.cdf(f_interaction, df_interaction, df_residual))
                if not np.isnan(f_interaction)
                else np.nan
            )

            anova_table = [
                {"source": "Row Factor", "ss": ss_row, "df": df_row, "ms": ms_row, "f": f_row, "p_value": p_row},
                {"source": "Column Factor", "ss": ss_col, "df": df_col, "ms": ms_col, "f": f_col, "p_value": p_col},
                {"source": "Interaction", "ss": ss_interaction, "df": df_interaction, "ms": ms_interaction, "f": f_interaction, "p_value": p_interaction},
                {"source": "Residual (Error)", "ss": ss_residual, "df": df_residual, "ms": ms_residual, "f": None, "p_value": None},
                {"source": "Total", "ss": ss_total, "df": df_total, "ms": None, "f": None, "p_value": None},
            ]

        return {
            "type": "Two-Way ANOVA",
            "anova_table": anova_table,
            "row_names": row_names,
            "col_names": col_names,
            "r_squared": float((ss_row + ss_col + ss_interaction) / ss_total) if ss_total > 0 else 0.0,
            "ms_residual": ms_residual,
            "df_residual": df_residual,
        }

    @classmethod
    def repeated_measures(cls, data_matrix: np.ndarray) -> Dict[str, Any]:
        """
        Executes One-Way Repeated Measures ANOVA across treatments (columns)
        matched across subjects (rows).
        data_matrix shape: (n_subjects, n_treatments)
        """
        mat = np.asarray(data_matrix, dtype=np.float64)
        if mat.ndim != 2:
            raise ValueError(f"Repeated Measures ANOVA requires a 2D matrix (subjects x treatments), got shape {mat.shape}.")
        valid_row_mask = np.all(np.isfinite(mat), axis=1)
        clean_mat = mat[valid_row_mask]
        n_subj, n_treat = clean_mat.shape
        if n_subj < 2 or n_treat < 2:
            raise ValueError("Repeated Measures ANOVA requires at least 2 complete subjects with no missing values and 2 treatments.")

        grand_mean = float(np.mean(clean_mat))
        subj_means = np.mean(clean_mat, axis=1)
        treat_means = np.mean(clean_mat, axis=0)

        ss_subj = float(n_treat * np.sum((subj_means - grand_mean) ** 2))
        ss_treat = float(n_subj * np.sum((treat_means - grand_mean) ** 2))
        ss_total = float(np.sum((clean_mat - grand_mean) ** 2))
        ss_error = ss_total - ss_subj - ss_treat

        df_subj = n_subj - 1
        df_treat = n_treat - 1
        df_error = df_subj * df_treat
        df_total = (n_subj * n_treat) - 1

        ms_treat = ss_treat / df_treat
        ms_error = ss_error / df_error if df_error > 0 else np.nan

        f_treat = ms_treat / ms_error if ms_error > 0 else np.nan
        p_treat = float(1.0 - stats.f.cdf(f_treat, df_treat, df_error)) if not np.isnan(f_treat) else np.nan

        # Geisser-Greenhouse epsilon calculation for sphericity
        # S = covariance matrix of treatments
        cov_s = np.cov(clean_mat, rowvar=False)
        k = n_treat
        mean_diag = np.mean(np.diag(cov_s))
        mean_all = np.mean(cov_s)
        mean_row = np.mean(cov_s, axis=1)

        num = (k * (mean_diag - mean_all)) ** 2
        den = (k - 1) * (
            np.sum(cov_s ** 2)
            - 2 * k * np.sum(mean_row ** 2)
            + (k ** 2) * (mean_all ** 2)
        )
        epsilon = float(np.clip(num / den, 1.0 / (k - 1), 1.0)) if den > 0 else 1.0

        p_treat_corrected = float(1.0 - stats.f.cdf(f_treat, df_treat * epsilon, df_error * epsilon))

        anova_table = [
            {"source": "Treatments (Between groups)", "ss": ss_treat, "df": df_treat, "ms": ms_treat, "f": f_treat, "p_value": p_treat},
            {"source": "Subjects (Matching)", "ss": ss_subj, "df": df_subj, "ms": ss_subj / df_subj, "f": None, "p_value": None},
            {"source": "Error", "ss": ss_error, "df": df_error, "ms": ms_error, "f": None, "p_value": None},
            {"source": "Total", "ss": ss_total, "df": df_total, "ms": None, "f": None, "p_value": None},
        ]

        return {
            "type": "Repeated Measures One-Way ANOVA",
            "anova_table": anova_table,
            "f_stat": float(f_treat),
            "p_value": float(p_treat),
            "epsilon_geisser_greenhouse": epsilon,
            "p_value_corrected": float(p_treat_corrected),
            "ms_error": ms_error,
            "df_error": df_error,
            "n_subjects": n_subj,
            "n_treatments": n_treat,
        }

    @classmethod
    def dunnett_test(
        cls,
        anova_result: Dict[str, Any],
        control_idx: int = 0,
        alpha: float = 0.05,
    ) -> List[Dict[str, Any]]:
        """
        Executes Dunnett's multiple comparisons test against a single control group.
        Uses exact multivariate-t distribution quadrature via scipy.stats.dunnett.
        """
        cleaned_groups = anova_result["cleaned_groups"]
        names = anova_result["names_used"]
        ms_within = anova_result["ms_within"]
        df_within = anova_result["df_within"]

        k = len(cleaned_groups)
        ctrl_group = cleaned_groups[control_idx]
        ctrl_name = names[control_idx]
        n0 = len(ctrl_group)
        mean0 = float(np.mean(ctrl_group))

        treat_indices = [j for j in range(k) if j != control_idx]
        if not treat_indices:
            return []

        comparisons: List[Dict[str, Any]] = []

        try:
            treat_samples = [cleaned_groups[j] for j in treat_indices]
            dunnett_out = stats.dunnett(*treat_samples, control=ctrl_group)
            ci_out = dunnett_out.confidence_interval(confidence_level=1.0 - alpha)

            stat_arr = np.atleast_1d(dunnett_out.statistic)
            pval_arr = np.atleast_1d(dunnett_out.pvalue)
            ci_l_arr = np.atleast_1d(ci_out.low)
            ci_u_arr = np.atleast_1d(ci_out.high)

            for idx, j in enumerate(treat_indices):
                nj = len(cleaned_groups[j])
                mean_j = float(np.mean(cleaned_groups[j]))
                diff = mean_j - mean0
                se_diff = np.sqrt(ms_within * (1.0 / nj + 1.0 / n0))

                q_stat = float(stat_arr[idx])
                p_val = float(pval_arr[idx])
                ci_l = float(ci_l_arr[idx])
                ci_u = float(ci_u_arr[idx])

                comparisons.append({
                    "comparison": f"{names[j]} vs. {ctrl_name}",
                    "mean_diff": float(diff),
                    "se_diff": float(se_diff),
                    "q_stat": float(q_stat),
                    "p_value": p_val,
                    "significant": bool(p_val < alpha),
                    "ci_lower": ci_l,
                    "ci_upper": ci_u,
                })
        except Exception:
            num_treatments = len(treat_indices)
            for j in treat_indices:
                nj = len(cleaned_groups[j])
                mean_j = float(np.mean(cleaned_groups[j]))
                diff = mean_j - mean0
                se_diff = np.sqrt(ms_within * (1.0 / nj + 1.0 / n0))
                q_stat = diff / se_diff
                p_raw = 2.0 * (1.0 - stats.t.cdf(abs(q_stat), df=df_within))
                p_val = float(min(1.0, p_raw * num_treatments))
                ci_half = stats.t.ppf(1.0 - (alpha / (2.0 * num_treatments)), df=df_within) * se_diff

                comparisons.append({
                    "comparison": f"{names[j]} vs. {ctrl_name}",
                    "mean_diff": float(diff),
                    "se_diff": float(se_diff),
                    "q_stat": float(q_stat),
                    "p_value": p_val,
                    "significant": bool(p_val < alpha),
                    "ci_lower": float(diff - ci_half),
                    "ci_upper": float(diff + ci_half),
                })

        return comparisons

    @classmethod
    def tukey_hsd(
        cls,
        anova_result: Dict[str, Any],
        alpha: float = 0.05,
    ) -> List[Dict[str, Any]]:
        """
        Executes Tukey's Honest Significant Difference (HSD) test
        for all pairwise group comparisons.
        """
        cleaned_groups = anova_result["cleaned_groups"]
        names = anova_result["names_used"]
        ms_within = anova_result["ms_within"]
        df_within = anova_result["df_within"]
        k = len(cleaned_groups)

        comparisons: List[Dict[str, Any]] = []

        for i in range(k):
            for j in range(i + 1, k):
                g1 = cleaned_groups[i]
                g2 = cleaned_groups[j]
                n1 = len(g1)
                n2 = len(g2)
                mean1 = float(np.mean(g1))
                mean2 = float(np.mean(g2))
                diff = mean1 - mean2

                # Standard error for studentized range q
                # q = (mean1 - mean2) / sqrt( (ms_within / 2) * (1/n1 + 1/n2) )
                se_q = np.sqrt(0.5 * ms_within * (1.0 / n1 + 1.0 / n2))
                q_stat = abs(diff) / se_q

                # p-value using studentized range distribution
                try:
                    p_val = float(stats.studentized_range.sf(q_stat, k, df_within))
                except Exception:
                    # Fallback student-t with Bonferroni correction
                    n_comp = (k * (k - 1)) / 2
                    p_t = 2.0 * (1.0 - stats.t.cdf(abs(diff) / np.sqrt(ms_within * (1/n1 + 1/n2)), df=df_within))
                    p_val = float(min(1.0, p_t * n_comp))

                q_crit = stats.studentized_range.ppf(1.0 - alpha, k, df_within)
                ci_half = (q_crit / np.sqrt(2)) * np.sqrt(ms_within * (1.0 / n1 + 1.0 / n2))

                comparisons.append({
                    "comparison": f"{names[i]} vs. {names[j]}",
                    "mean_diff": float(diff),
                    "q_stat": float(q_stat),
                    "p_value": p_val,
                    "significant": p_val < alpha,
                    "ci_lower": float(diff - ci_half),
                    "ci_upper": float(diff + ci_half),
                })

        return comparisons
