"""
Survival Analysis Engine for GemGraph.
Includes Kaplan-Meier estimation with Greenwood/log-log confidence bands,
log-rank and Gehan-Breslow-Wilcoxon tests, Mantel-Haenszel hazard ratio,
and Cox proportional hazards regression via Newton-Raphson with Efron tie handling.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from scipy import stats


class SurvivalEngine:
    """
    Pure NumPy/SciPy survival statistics and modeling engine.
    """

    @classmethod
    def kaplan_meier(
        cls,
        times: Any,
        events: Any,
        confidence_level: float = 0.95,
    ) -> Dict[str, Any]:
        """
        Computes Kaplan-Meier survival curve with Greenwood SE,
        log-log transform 95% confidence intervals, number at risk,
        censor events, and median survival.
        """
        t_arr = np.asarray(times, dtype=np.float64)
        e_arr = np.asarray(events, dtype=np.float64)
        valid = ~np.isnan(t_arr) & ~np.isnan(e_arr)
        t_clean = t_arr[valid]
        e_clean = e_arr[valid]

        if len(t_clean) == 0:
            return {
                "times": np.array([0.0]),
                "survival": np.array([1.0]),
                "se": np.array([0.0]),
                "ci_lower": np.array([1.0]),
                "ci_upper": np.array([1.0]),
                "n_at_risk": np.array([0]),
                "n_events": np.array([0]),
                "n_censored": np.array([0]),
                "median_survival": np.nan,
                "censor_times": np.array([]),
                "censor_survival": np.array([]),
            }

        order = np.argsort(t_clean)
        t_sorted = t_clean[order]
        e_sorted = e_clean[order]

        unique_times = np.unique(t_sorted)
        n_at_risk = len(t_sorted)
        s_cur = 1.0
        greenwood_sum = 0.0
        z_crit = float(stats.norm.ppf((1.0 + confidence_level) / 2.0))

        res_times = [0.0]
        res_surv = [1.0]
        res_se = [0.0]
        res_ci_low = [1.0]
        res_ci_high = [1.0]
        res_risk = [n_at_risk]
        res_events = [0]
        res_cens = [0]

        censor_times: List[float] = []
        censor_surv: List[float] = []

        for ut in unique_times:
            mask = t_sorted == ut
            d = int(np.sum(e_sorted[mask] == 1))
            c = int(np.sum(e_sorted[mask] == 0))

            if n_at_risk > 0:
                if d > 0:
                    s_cur *= (1.0 - d / n_at_risk)
                    if n_at_risk > d:
                        greenwood_sum += d / (n_at_risk * (n_at_risk - d))

                se = float(s_cur * np.sqrt(greenwood_sum))

                # Kalbfleisch-Prentice log-log transform 95% CI
                if 0.0 < s_cur < 1.0 and greenwood_sum > 0:
                    sigma_w = np.sqrt(greenwood_sum) / np.abs(np.log(s_cur))
                    factor = np.exp(z_crit * sigma_w)
                    ci_low = float(s_cur ** factor)
                    ci_high = float(s_cur ** (1.0 / factor))
                elif s_cur >= 1.0:
                    ci_low = 1.0
                    ci_high = 1.0
                else:
                    ci_low = 0.0
                    ci_high = 0.0

                ci_low = max(0.0, min(1.0, ci_low))
                ci_high = max(0.0, min(1.0, ci_high))

                if c > 0:
                    censor_times.extend([float(ut)] * c)
                    censor_surv.extend([float(s_cur)] * c)

                if ut == 0.0 and len(res_times) == 1 and res_times[0] == 0.0:
                    res_surv[0] = float(s_cur)
                    res_se[0] = float(se)
                    res_ci_low[0] = float(ci_low)
                    res_ci_high[0] = float(ci_high)
                    res_risk[0] = int(n_at_risk)
                    res_events[0] = int(d)
                    res_cens[0] = int(c)
                else:
                    res_times.append(float(ut))
                    res_surv.append(float(s_cur))
                    res_se.append(float(se))
                    res_ci_low.append(float(ci_low))
                    res_ci_high.append(float(ci_high))
                    res_risk.append(int(n_at_risk))
                    res_events.append(int(d))
                    res_cens.append(int(c))

                n_at_risk -= (d + c)

        # Median survival: time at which survival drops to <= 0.5
        median_survival = np.nan
        for t_val, s_val in zip(res_times[1:], res_surv[1:]):
            if s_val <= 0.5:
                median_survival = float(t_val)
                break

        return {
            "times": np.array(res_times, dtype=np.float64),
            "survival": np.array(res_surv, dtype=np.float64),
            "se": np.array(res_se, dtype=np.float64),
            "ci_lower": np.array(res_ci_low, dtype=np.float64),
            "ci_upper": np.array(res_ci_high, dtype=np.float64),
            "n_at_risk": np.array(res_risk, dtype=np.int64),
            "n_events": np.array(res_events, dtype=np.int64),
            "n_censored": np.array(res_cens, dtype=np.int64),
            "median_survival": median_survival,
            "censor_times": np.array(censor_times, dtype=np.float64),
            "censor_survival": np.array(censor_surv, dtype=np.float64),
        }

    @classmethod
    def log_rank_test(
        cls,
        groups: List[Tuple[Any, Any]],
    ) -> Dict[str, Any]:
        """
        Calculates log-rank test (Mantel-Cox) and Gehan-Breslow-Wilcoxon test
        across k groups.
        groups: list of (times, events) pairs.
        """
        k = len(groups)
        if k < 2:
            return {
                "chi2": 0.0,
                "df": 0,
                "p_value": 1.0,
                "gehan_wilcoxon_chi2": 0.0,
                "gehan_wilcoxon_p": 1.0,
                "observed": [0.0] * k,
                "expected": [0.0] * k,
            }

        cleaned_groups: List[Tuple[np.ndarray, np.ndarray]] = []
        for g_times, g_events in groups:
            t = np.asarray(g_times, dtype=np.float64)
            e = np.asarray(g_events, dtype=np.float64)
            valid = ~np.isnan(t) & ~np.isnan(e)
            cleaned_groups.append((t[valid], e[valid]))

        # Pool all distinct event times where at least one death occurred
        death_times: List[float] = []
        for t, e in cleaned_groups:
            death_times.extend(t[e == 1].tolist())
        unique_death_times = np.sort(np.unique(death_times))

        if len(unique_death_times) == 0:
            return {
                "chi2": 0.0,
                "df": k - 1,
                "p_value": 1.0,
                "gehan_wilcoxon_chi2": 0.0,
                "gehan_wilcoxon_p": 1.0,
                "observed": [0.0] * k,
                "expected": [0.0] * k,
            }

        # Tabulate for each failure time
        O_lr = np.zeros(k, dtype=np.float64)
        E_lr = np.zeros(k, dtype=np.float64)
        V_lr = np.zeros((k, k), dtype=np.float64)

        O_gw = np.zeros(k, dtype=np.float64)
        E_gw = np.zeros(k, dtype=np.float64)
        V_gw = np.zeros((k, k), dtype=np.float64)

        for dt in unique_death_times:
            # At risk and events per group at dt
            n_ig = np.zeros(k, dtype=np.float64)
            d_ig = np.zeros(k, dtype=np.float64)
            for i, (gt, ge) in enumerate(cleaned_groups):
                n_ig[i] = np.sum(gt >= dt)
                d_ig[i] = np.sum((gt == dt) & (ge == 1))

            n_total = float(np.sum(n_ig))
            d_total = float(np.sum(d_ig))

            if n_total <= 1 or d_total == 0:
                continue

            w_gw = n_total  # Gehan-Breslow weight

            for i in range(k):
                e_i = n_ig[i] * (d_total / n_total)
                O_lr[i] += d_ig[i]
                E_lr[i] += e_i

                O_gw[i] += w_gw * d_ig[i]
                E_gw[i] += w_gw * e_i

                for j in range(k):
                    if i == j:
                        v = (n_ig[i] * (n_total - n_ig[i]) * d_total * (n_total - d_total)) / (
                            (n_total ** 2) * (n_total - 1.0)
                        )
                    else:
                        v = -(n_ig[i] * n_ig[j] * d_total * (n_total - d_total)) / (
                            (n_total ** 2) * (n_total - 1.0)
                        )
                    V_lr[i, j] += v
                    V_gw[i, j] += (w_gw ** 2) * v

        df = k - 1
        # Invert (k-1) x (k-1) submatrix
        diff_lr = (O_lr - E_lr)[:df]
        sub_v_lr = V_lr[:df, :df]

        diff_gw = (O_gw - E_gw)[:df]
        sub_v_gw = V_gw[:df, :df]

        try:
            inv_v_lr = np.linalg.pinv(sub_v_lr)
            chi2_lr = float(diff_lr @ inv_v_lr @ diff_lr)
            p_lr = float(stats.chi2.sf(max(0.0, chi2_lr), df))
        except Exception:
            chi2_lr = 0.0
            p_lr = 1.0

        try:
            inv_v_gw = np.linalg.pinv(sub_v_gw)
            chi2_gw = float(diff_gw @ inv_v_gw @ diff_gw)
            p_gw = float(stats.chi2.sf(max(0.0, chi2_gw), df))
        except Exception:
            chi2_gw = 0.0
            p_gw = 1.0

        return {
            "chi2": float(max(0.0, chi2_lr)),
            "df": int(df),
            "p_value": float(p_lr),
            "gehan_wilcoxon_chi2": float(max(0.0, chi2_gw)),
            "gehan_wilcoxon_p": float(p_gw),
            "observed": [float(x) for x in O_lr],
            "expected": [float(x) for x in E_lr],
            "variance_matrix": V_lr,
        }

    @classmethod
    def hazard_ratio_logrank(
        cls,
        group_a: Tuple[Any, Any],
        group_b: Tuple[Any, Any],
    ) -> Dict[str, Any]:
        """
        Computes the hazard ratio and 95% CI between two groups using the
        Mantel-Haenszel / log-rank method:
        HR = (O_a / E_a) / (O_b / E_b)
        ln(HR) SE = sqrt(1 / V)
        """
        lr = cls.log_rank_test([group_a, group_b])
        o_a, o_b = lr["observed"][0], lr["observed"][1]
        e_a, e_b = lr["expected"][0], lr["expected"][1]

        v = float(lr["variance_matrix"][0, 0]) if "variance_matrix" in lr else 0.0

        if e_a <= 0 or e_b <= 0 or o_a <= 0 or o_b <= 0 or v <= 0:
            return {
                "hazard_ratio": 1.0,
                "ci_lower": 1.0,
                "ci_upper": 1.0,
                "log_hr": 0.0,
                "se": 0.0,
                "o_a": float(o_a),
                "e_a": float(e_a),
                "o_b": float(o_b),
                "e_b": float(e_b),
            }

        hr = float((o_a / e_a) / (o_b / e_b))
        log_hr = float(np.log(hr))
        se = float(np.sqrt(1.0 / v))

        z_crit = 1.959963984540054
        ci_lower = float(np.exp(log_hr - z_crit * se))
        ci_upper = float(np.exp(log_hr + z_crit * se))

        return {
            "hazard_ratio": hr,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "log_hr": log_hr,
            "se": se,
            "o_a": float(o_a),
            "e_a": float(e_a),
            "o_b": float(o_b),
            "e_b": float(e_b),
        }

    @classmethod
    def cox_ph(
        cls,
        times: Any,
        events: Any,
        covariates: Any,
        names: Optional[List[str]] = None,
        max_iter: int = 50,
        tol: float = 1e-6,
    ) -> Dict[str, Any]:
        """
        Fits Cox proportional hazards regression via Newton-Raphson
        with Efron tie handling and step-halving.
        """
        t = np.asarray(times, dtype=np.float64)
        e = np.asarray(events, dtype=np.float64)
        x = np.asarray(covariates, dtype=np.float64)

        if x.ndim == 1:
            x = x.reshape(-1, 1)

        n_samples, n_covariates = x.shape

        if names is None:
            names = [f"Covariate {i+1}" for i in range(n_covariates)]

        # Clean NaNs
        valid = ~np.isnan(t) & ~np.isnan(e) & ~np.isnan(x).any(axis=1)
        t_clean = t[valid]
        e_clean = e[valid]
        x_clean = x[valid]

        n_clean = len(t_clean)
        if n_clean <= n_covariates:
            raise ValueError(
                f"Sample size ({n_clean}) must exceed number of covariates ({n_covariates})."
            )

        n_events = int(np.sum(e_clean == 1))
        if n_events == 0:
            raise ValueError("No events observed; cannot fit Cox proportional hazards model.")
        if n_events < n_covariates:
            raise ValueError(
                f"Number of events ({n_events}) must be at least the number of covariates ({n_covariates})."
            )

        # Check for collinearity / singular matrix in covariate matrix
        rank = np.linalg.matrix_rank(x_clean)
        if rank < n_covariates:
            raise ValueError(
                f"Singular covariate matrix (rank {rank} < {n_covariates}). Covariates are collinear."
            )

        # Sort descending or find unique event times
        unique_death_times = np.sort(np.unique(t_clean[e_clean == 1]))

        def compute_loglik_score_hessian(beta: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
            # Exp(X * beta)
            xb = x_clean @ beta
            # Prevent overflow
            if np.any(xb > 50.0) or np.any(xb < -50.0):
                xb = np.clip(xb, -50.0, 50.0)
            theta = np.exp(xb)

            loglik = 0.0
            score = np.zeros(n_covariates, dtype=np.float64)
            hessian = np.zeros((n_covariates, n_covariates), dtype=np.float64)

            for dt in unique_death_times:
                r_mask = t_clean >= dt
                d_mask = (t_clean == dt) & (e_clean == 1)
                d_m = int(np.sum(d_mask))
                if d_m == 0:
                    continue

                theta_r = theta[r_mask]
                x_r = x_clean[r_mask]

                theta_d = theta[d_mask]
                x_d = x_clean[d_mask]

                sum_theta_r = np.sum(theta_r)
                sum_theta_x_r = theta_r @ x_r
                sum_theta_xx_r = (theta_r[:, None] * x_r).T @ x_r

                sum_theta_d = np.sum(theta_d)
                sum_theta_x_d = theta_d @ x_d
                sum_theta_xx_d = (theta_d[:, None] * x_d).T @ x_d

                # Sum of covariates for dying individuals
                s_d = np.sum(x_d, axis=0)
                loglik += float(s_d @ beta)
                score += s_d

                for k_step in range(d_m):
                    alpha = k_step / float(d_m)
                    a0 = sum_theta_r - alpha * sum_theta_d
                    if a0 <= 0:
                        raise ValueError("Numerical instability: risk score sum <= 0.")
                    a1 = sum_theta_x_r - alpha * sum_theta_x_d
                    a2 = sum_theta_xx_r - alpha * sum_theta_xx_d

                    loglik -= float(np.log(a0))
                    a1_norm = a1 / a0
                    score -= a1_norm
                    hessian -= (a2 / a0) - np.outer(a1_norm, a1_norm)

            return loglik, score, hessian

        # Null model log-likelihood (beta = 0)
        beta_null = np.zeros(n_covariates, dtype=np.float64)
        loglik_null, _, _ = compute_loglik_score_hessian(beta_null)

        beta = np.zeros(n_covariates, dtype=np.float64)
        loglik_curr, score_curr, hessian_curr = compute_loglik_score_hessian(beta)

        converged = False
        num_iterations = 0
        for it in range(max_iter):
            num_iterations = it + 1
            # Check for separation / divergence
            if np.any(np.abs(beta) >= 15.0):
                raise ValueError("Perfect separation or non-convergence detected in Cox regression.")

            # Observed Information = -Hessian
            info = -hessian_curr
            cond = np.linalg.cond(info)
            info_eigs = np.linalg.eigvalsh(info)
            if np.isnan(cond) or cond > 1e12 or np.any(np.isnan(info_eigs)) or np.min(info_eigs) <= 1e-5:
                raise ValueError("Singular information matrix or perfect separation detected in Cox regression.")

            try:
                step_direction = np.linalg.solve(info, score_curr)
            except np.linalg.LinAlgError:
                raise ValueError("Singular matrix encountered in Cox regression.")

            # Step-halving line search
            alpha_step = 1.0
            step_halved = False
            for _ in range(12):
                beta_new = beta + alpha_step * step_direction
                try:
                    loglik_new, score_new, hessian_new = compute_loglik_score_hessian(beta_new)
                    if loglik_new >= loglik_curr - 1e-8:
                        step_halved = True
                        break
                except Exception:
                    pass
                alpha_step *= 0.5

            if not step_halved:
                raise ValueError("Step halving failed to improve likelihood; likely non-convergence.")

            delta_loglik = abs(loglik_new - loglik_curr)
            beta = beta_new
            loglik_curr = loglik_new
            score_curr = score_new
            hessian_curr = hessian_new

            if np.max(np.abs(score_curr)) < tol and delta_loglik < 1e-7:
                converged = True
                break

        if not converged and np.max(np.abs(score_curr)) >= 1e-3:
            raise ValueError("Cox proportional hazards failed to converge within maximum iterations.")

        # Variance-covariance matrix
        info_final = -hessian_curr
        info_eigs = np.linalg.eigvalsh(info_final)
        if np.any(np.isnan(info_eigs)) or np.min(info_eigs) <= 1e-5:
            raise ValueError("Singular information matrix or perfect separation detected in Cox regression.")
        try:
            cov_matrix = np.linalg.inv(info_final)
        except np.linalg.LinAlgError:
            raise ValueError("Singular matrix encountered in Cox regression.")

        se = np.sqrt(np.maximum(np.diag(cov_matrix), 0.0))
        if np.any(np.isnan(se)) or np.any(se > 30.0) or np.any(np.abs(beta) >= 15.0):
            raise ValueError("Perfect separation or non-convergence detected in Cox regression.")

        z_scores = np.where(se > 0, beta / se, 0.0)
        p_values = float(2.0) * stats.norm.sf(np.abs(z_scores))
        hazard_ratios = np.exp(beta)

        z_crit = 1.959963984540054
        hr_ci_lower = np.exp(beta - z_crit * se)
        hr_ci_upper = np.exp(beta + z_crit * se)

        lrt_stat = float(2.0 * max(0.0, loglik_curr - loglik_null))
        lrt_p = float(stats.chi2.sf(lrt_stat, n_covariates))

        results_list = []
        for i in range(n_covariates):
            results_list.append({
                "name": names[i],
                "coef": float(beta[i]),
                "hazard_ratio": float(hazard_ratios[i]),
                "se": float(se[i]),
                "std_error": float(se[i]),
                "z": float(z_scores[i]),
                "wald_z": float(z_scores[i]),
                "p_value": float(p_values[i]),
                "hr_ci_lower": float(hr_ci_lower[i]),
                "hr_ci_upper": float(hr_ci_upper[i]),
            })

        return {
            "converged": bool(converged),
            "n_samples": n_clean,
            "n_events": n_events,
            "log_likelihood": float(loglik_curr),
            "log_likelihood_null": float(loglik_null),
            "lrt_stat": lrt_stat,
            "lrt_df": int(n_covariates),
            "lrt_p": lrt_p,
            "aic": float(2 * n_covariates - 2 * loglik_curr),
            "ties": "Efron",
            "iterations": num_iterations,
            "covariates": results_list,
            "beta": beta,
            "covariance_matrix": cov_matrix,
        }
