"""
Unit and Integration Tests for Survival Analysis Suite.
Validates Kaplan-Meier estimator with Greenwood standard errors and log-log 95% CIs,
Log-Rank and Gehan-Breslow-Wilcoxon tests, Hazard Ratios, and Cox Proportional Hazards regression.
"""

import unittest
import numpy as np

from src.math.survival import SurvivalEngine


class TestSurvivalAnalysis(unittest.TestCase):
    def test_kaplan_meier_hand_computed_reference_with_censoring_and_ties(self) -> None:
        """
        Validates KM against hand-calculated textbook example with ties and censoring:
        Times: [6, 6, 6, 7, 10, 13, 16, 22, 23]
        Events: [1, 1, 0, 1, 1, 1, 1, 1, 0] (Total N=9, 2 ties at t=6 with 1 censor)
        t=0: S=1.0
        t=6: N_risk=9, d=2, c=1. S = 1.0 * (1 - 2/9) = 7/9 ≈ 0.7778. N_risk after = 9 - 3 = 6.
        t=7: N_risk=6, d=1, c=0. S = (7/9) * (1 - 1/6) = (7/9)*(5/6) = 35/54 ≈ 0.6481. N_risk after = 5.
        t=10: N_risk=5, d=1, c=0. S = (35/54) * (1 - 1/5) = (35/54)*(4/5) = 28/54 ≈ 0.5185. N_risk after = 4.
        t=13: N_risk=4, d=1, c=0. S = (28/54) * (1 - 1/4) = (28/54)*(3/4) = 21/54 ≈ 0.3889.
        """
        times = [6, 6, 6, 7, 10, 13, 16, 22, 23]
        events = [1, 1, 0, 1, 1, 1, 1, 1, 0]

        res = SurvivalEngine.kaplan_meier(times, events)

        # Check unique times (starts at 0.0)
        np.testing.assert_array_equal(res["times"], [0.0, 6.0, 7.0, 10.0, 13.0, 16.0, 22.0, 23.0])

        # Check survival probabilities
        self.assertAlmostEqual(res["survival"][0], 1.0, places=4)
        self.assertAlmostEqual(res["survival"][1], 7.0 / 9.0, places=4)   # t=6
        self.assertAlmostEqual(res["survival"][2], 35.0 / 54.0, places=4) # t=7
        self.assertAlmostEqual(res["survival"][3], 28.0 / 54.0, places=4) # t=10
        self.assertAlmostEqual(res["survival"][4], 21.0 / 54.0, places=4) # t=13

        # Median survival: drops below 0.5 at t=13
        self.assertEqual(res["median_survival"], 13.0)

        # Censor times should capture t=6 and t=23
        self.assertTrue(np.isin(6.0, res["censor_times"]))
        self.assertTrue(np.isin(23.0, res["censor_times"]))

    def test_greenwood_confidence_intervals_bounded_in_unit_interval(self) -> None:
        """Verifies log-log transformed confidence intervals are strictly within [0.0, 1.0]."""
        rng = np.random.default_rng(42)
        times = rng.exponential(scale=20.0, size=80)
        events = rng.binomial(n=1, p=0.75, size=80)

        res = SurvivalEngine.kaplan_meier(times, events)

        ci_low = res["ci_lower"]
        ci_high = res["ci_upper"]

        self.assertTrue(np.all(ci_low >= 0.0), f"Lower CI < 0: {ci_low}")
        self.assertTrue(np.all(ci_low <= 1.0), f"Lower CI > 1: {ci_low}")
        self.assertTrue(np.all(ci_high >= 0.0), f"Upper CI < 0: {ci_high}")
        self.assertTrue(np.all(ci_high <= 1.0), f"Upper CI > 1: {ci_high}")
        self.assertTrue(np.all(ci_low <= ci_high), "Lower CI greater than upper CI")

    def test_log_rank_test_statistic_against_known_reference(self) -> None:
        """
        Validates log-rank test against a reference two-group cohort:
        Group A: [2, 3, 5, 8, 12], all events (1)
        Group B: [4, 6, 10, 14, 18], all events (1)
        Group A has faster events, so Group A observed events exceed expected events.
        """
        group_a = (np.array([2, 3, 5, 8, 12]), np.array([1, 1, 1, 1, 1]))
        group_b = (np.array([4, 6, 10, 14, 18]), np.array([1, 1, 1, 1, 1]))

        res = SurvivalEngine.log_rank_test([group_a, group_b])

        self.assertEqual(res["df"], 1)
        # Closed-form reference:
        # E_A = 1/2 + 4/9 + 3/8 + 3/7 + 2/6 + 2/5 + 1/4 + 1/3 = 3.06468...
        # V = 1/4 + 20/81 + 15/64 + 12/49 + 8/36 + 6/25 + 3/16 + 2/9 = 1.84813...
        # chi2 = (5 - 3.06468)^2 / 1.84813 ≈ 2.0266
        self.assertAlmostEqual(res["chi2"], 2.0266, places=4)
        self.assertAlmostEqual(res["expected"][0], 3.0647, places=4)
        self.assertAlmostEqual(res["observed"][0], 5.0, places=4)
        self.assertAlmostEqual(res["p_value"], 0.1546, places=4)

        # Hazard ratio: group A vs group B > 1.0
        hr_res = SurvivalEngine.hazard_ratio_logrank(group_a, group_b)
        self.assertGreater(hr_res["hazard_ratio"], 1.0)
        self.assertGreater(hr_res["ci_upper"], hr_res["hazard_ratio"])
        self.assertLess(hr_res["ci_lower"], hr_res["hazard_ratio"])

    def test_cox_proportional_hazards_recovers_known_hazard_ratio(self) -> None:
        """
        Generates a synthetic survival cohort with Treatment (0=Control, 1=Treated)
        where true HR ≈ 0.5 (beta = ln(0.5) ≈ -0.6931).
        Verifies Cox regression recovers HR ≈ 0.5 and matches log-rank direction.
        """
        rng = np.random.default_rng(999)
        n = 150
        treatment = rng.binomial(1, 0.5, size=n)
        age = rng.normal(55, 10, size=n)

        # Baseline hazard lambda0 = 0.02
        # True beta_treat = -0.6931 (HR = 0.5), beta_age = 0.03
        linear_predictor = -0.6931 * treatment + 0.03 * (age - 55)
        # Inversion method for Cox model with exponential baseline: T = -ln(U) / (lambda0 * exp(LP))
        u = rng.uniform(0.001, 0.999, size=n)
        times = -np.log(u) / (0.02 * np.exp(linear_predictor))

        # Administrative censoring at t=60
        events = np.where(times < 60.0, 1, 0)
        times = np.minimum(times, 60.0)

        covariates = np.column_stack([treatment, age])
        cov_names = ["Treatment", "Age"]

        res = SurvivalEngine.cox_ph(times, events, covariates, names=cov_names)

        self.assertTrue(res["converged"])
        self.assertGreater(res["n_events"], 30)

        # Check Treatment coefficient and HR
        treat_cov = res["covariates"][0]
        self.assertEqual(treat_cov["name"], "Treatment")
        self.assertLess(treat_cov["coef"], 0.0, "Treatment coefficient should be negative (protective)")
        # HR should be near 0.5 (within [0.30, 0.75])
        self.assertAlmostEqual(treat_cov["hazard_ratio"], 0.5, delta=0.25)
        self.assertLess(treat_cov["p_value"], 0.05, "Treatment should be statistically significant")

        # Global Likelihood Ratio Test should be significant
        self.assertLess(res["lrt_p"], 0.01)

        # Verify that Cox PH matches log-rank direction on this cohort
        grp_ctrl = (times[treatment == 0], events[treatment == 0])
        grp_trt = (times[treatment == 1], events[treatment == 1])
        hr_lr = SurvivalEngine.hazard_ratio_logrank(grp_trt, grp_ctrl)
        # Both methods must agree on protective effect (HR < 1.0)
        self.assertLess(hr_lr["hazard_ratio"], 1.0)
        self.assertLess(treat_cov["hazard_ratio"], 1.0)

    def test_cox_ph_rejects_singular_or_separated_inputs_with_clean_error(self) -> None:
        """Verifies Cox PH raises ValueError with a clean error on collinear/singular covariates and perfect separation."""
        times = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        events = np.array([1, 1, 1, 1, 1, 1, 1, 1])

        # 1. Collinear covariates: X2 = 2 * X1
        x1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        x2 = 2.0 * x1
        covariates = np.column_stack([x1, x2])

        with self.assertRaises(ValueError) as ctx:
            SurvivalEngine.cox_ph(times, events, covariates)
        self.assertIn("singular", str(ctx.exception).lower())

        # 2. Perfect quasi-complete separation (events only in one group)
        t_sep = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        e_sep = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=float)
        cov_sep = np.array([10, 10, 10, 10, 10, -10, -10, -10, -10, -10], dtype=float)

        with self.assertRaises(ValueError) as ctx_sep:
            SurvivalEngine.cox_ph(t_sep, e_sep, cov_sep)
        self.assertTrue(
            any(k in str(ctx_sep.exception).lower() for k in ("separation", "singular", "non-convergence")),
            f"Unexpected exception message: {ctx_sep.exception}",
        )

        # 3. Kaplan-Meier edge case: all subjects censored at time 0
        t_c0 = np.array([0.0, 0.0, 0.0], dtype=float)
        e_c0 = np.array([0.0, 0.0, 0.0], dtype=float)
        km_c0 = SurvivalEngine.kaplan_meier(t_c0, e_c0)
        self.assertEqual(len(km_c0["times"]), 1)
        self.assertEqual(km_c0["times"][0], 0.0)
        self.assertEqual(km_c0["survival"][0], 1.0)
        self.assertEqual(len(km_c0["censor_times"]), 3)


if __name__ == "__main__":
    unittest.main()
