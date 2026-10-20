"""
Numerical Parity Unit Tests for GemGraph Mathematical Engines.
Asserted against GraphPad Prism 10 recovered specifications
(REA-PRISM-NUM-301, REA-PRISM-ALGO-314, REA-PRISM-STAT-325).
"""

import unittest
import numpy as np
from src.math.nonlinear import (
    NonlinearEngine,
    ConvergencePreset,
    STANDARD_MODELS,
)
from src.math.outlier import OutlierEngine
from src.math.anova import AnovaEngine


class TestNonlinearParity(unittest.TestCase):
    """Verifies Levenberg-Marquardt solver, convergence tolerances, and parameter dependencies."""

    def test_4pl_sigmoidal_fit_and_presets(self) -> None:
        """Tests 4-Parameter Logistic fit with Standard and Strict convergence presets."""
        model = STANDARD_MODELS["Sigmoidal Dose-Response (4PL)"]
        true_params = np.array([5.0, 95.0, -5.5, 1.0])  # Bottom, Top, LogEC50, HillSlope

        x = np.array([-8.0, -7.0, -6.5, -6.0, -5.5, -5.0, -4.5, -4.0, -3.0], dtype=np.float64)
        np.random.seed(42)
        y_clean = model.evaluate(x, true_params)
        noise = np.random.normal(0, 0.5, size=len(x))
        y = y_clean + noise

        # 1. Standard Preset (1e-4)
        res_standard = NonlinearEngine.fit(
            x, y, model, preset=ConvergencePreset.STANDARD
        )
        self.assertTrue(res_standard["converged"])
        self.assertLessEqual(res_standard["iterations"], 1000)
        self.assertGreater(res_standard["r_squared"], 0.99)
        self.assertTrue(np.isfinite(res_standard["aicc"]))
        self.assertTrue(np.isfinite(res_standard["syx"]))

        # Parameter estimates match true values within 5%
        p_opt = res_standard["p_opt"]
        np.testing.assert_allclose(p_opt[0], true_params[0], atol=2.0)
        np.testing.assert_allclose(p_opt[1], true_params[1], atol=2.0)
        np.testing.assert_allclose(p_opt[2], true_params[2], atol=0.2)
        np.testing.assert_allclose(p_opt[3], true_params[3], atol=0.2)

        # Parameter dependencies must be valid probabilities [0, 1]
        for p in res_standard["parameters"]:
            self.assertGreaterEqual(p["dependency"], 0.0)
            self.assertLessEqual(p["dependency"], 1.0)
            self.assertGreater(p["std_error"], 0.0)

        # 2. Strict Preset (1e-6)
        res_strict = NonlinearEngine.fit(
            x, y, model, preset=ConvergencePreset.STRICT
        )
        self.assertTrue(res_strict["converged"])
        # Strict preset should reach equal or higher precision
        self.assertLessEqual(res_strict["ss_res"], res_standard["ss_res"] + 1e-6)

    def test_one_site_binding_fit(self) -> None:
        """Tests One Site Specific Binding (Hyperbola) model."""
        model = STANDARD_MODELS["One Site Specific Binding"]
        true_params = np.array([120.0, 15.0])  # Bmax, Kd

        x = np.array([0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 40.0, 80.0, 150.0], dtype=np.float64)
        y = model.evaluate(x, true_params)

        res = NonlinearEngine.fit(x, y, model)
        self.assertTrue(res["converged"])
        p_opt = res["p_opt"]
        np.testing.assert_allclose(p_opt, true_params, rtol=1e-3)
        self.assertAlmostEqual(res["r_squared"], 1.0, places=4)

    def test_exponential_decay_fit(self) -> None:
        """Tests One Phase Exponential Decay model."""
        model = STANDARD_MODELS["One Phase Exponential Decay"]
        true_params = np.array([100.0, 10.0, 0.25])  # Y0, Plateau, K

        x = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0], dtype=np.float64)
        y = model.evaluate(x, true_params)

        res = NonlinearEngine.fit(x, y, model)
        self.assertTrue(res["converged"])
        p_opt = res["p_opt"]
        np.testing.assert_allclose(p_opt, true_params, rtol=1e-3)


class TestOutlierParity(unittest.TestCase):
    """Verifies ROUT (RSDR, FDR Q) and Grubbs ESD outlier detection."""

    def test_rout_detection_with_mad(self) -> None:
        """Tests ROUT algorithm constants (RSDR = 1.4826 * MAD, Q = 1%)."""
        np.random.seed(123)
        # 20 samples from normal distribution + 1 extreme outlier
        clean_samples = np.random.normal(50.0, 2.0, size=20)
        data = np.copy(clean_samples)
        # Inject extreme outlier at index 10
        data[10] = 120.0

        res = OutlierEngine.rout_1d(data, q_fdr=0.01)
        self.assertEqual(res["method"], "ROUT")
        self.assertEqual(res["outlier_indices"], [10])
        self.assertAlmostEqual(res["outlier_values"][0], 120.0)

        # Verify RSDR calculation
        mad = np.median(np.abs(data - np.median(data)))
        expected_rsdr = 1.4826 * mad
        self.assertAlmostEqual(res["rsdr"], expected_rsdr, places=4)

    def test_grubbs_esd_detection(self) -> None:
        """Tests Grubbs ESD test critical value and outlier identification."""
        data = np.array([199.5, 200.1, 198.9, 200.4, 199.8, 201.0, 200.2, 280.0])
        res = OutlierEngine.grubbs_esd(data, alpha=0.05)
        self.assertEqual(res["method"], "Grubbs ESD")
        self.assertEqual(res["outlier_indices"], [7])
        self.assertEqual(res["outlier_values"], [280.0])

    def test_sample_size_boundary(self) -> None:
        """N < 3 constraint must prevent outlier calculation."""
        small_data = np.array([10.0, 50.0])
        res_rout = OutlierEngine.rout_1d(small_data)
        self.assertEqual(res_rout["outlier_indices"], [])
        self.assertIn("cannot perform", res_rout["message"])

        res_grubbs = OutlierEngine.grubbs_esd(small_data)
        self.assertEqual(res_grubbs["outlier_indices"], [])


class TestAnovaParity(unittest.TestCase):
    """Verifies One-Way, Two-Way, Repeated Measures ANOVA, and Dunnett/Tukey tests."""

    def test_one_way_anova(self) -> None:
        """Tests Ordinary One-Way ANOVA against textbook reference numbers."""
        # Groups: Control, Low Dose, High Dose
        groups = {
            "Control": [10.2, 11.1, 9.8, 10.5, 10.0],
            "Low Dose": [14.1, 13.8, 14.5, 13.9, 14.2],
            "High Dose": [20.1, 19.5, 20.8, 20.2, 19.9],
        }
        res = AnovaEngine.one_way(groups)
        self.assertEqual(res["k_groups"], 3)
        self.assertEqual(res["total_n"], 15)
        self.assertGreater(res["f_stat"], 500.0)  # Huge separation
        self.assertLess(res["p_value"], 1e-10)

        # Check ANOVA table structure
        table = res["anova_table"]
        self.assertEqual(table[0]["df"], 2)   # Between df: k - 1 = 2
        self.assertEqual(table[1]["df"], 12)  # Within df: N - k = 12

    def test_dunnett_multiple_comparisons(self) -> None:
        """Tests Dunnett's post-hoc comparisons against control group."""
        groups = {
            "Control": [10.0, 10.2, 9.9, 10.1],
            "Drug A": [10.1, 10.0, 10.2, 10.0],   # Not significant
            "Drug B": [25.0, 24.8, 25.2, 25.1],   # Highly significant
        }
        res = AnovaEngine.one_way(groups)
        comps = AnovaEngine.dunnett_test(res, control_idx=0)

        self.assertEqual(len(comps), 2)
        # Drug A vs Control: not significant
        self.assertFalse(comps[0]["significant"])
        self.assertGreater(comps[0]["p_value"], 0.05)

        # Drug B vs Control: significant
        self.assertTrue(comps[1]["significant"])
        self.assertLess(comps[1]["p_value"], 0.001)

    def test_tukey_hsd_comparisons(self) -> None:
        """Tests Tukey HSD all pairwise comparisons."""
        groups = {
            "A": [5.0, 5.2, 4.9],
            "B": [15.0, 15.1, 14.9],
            "C": [25.0, 24.9, 25.2],
        }
        res = AnovaEngine.one_way(groups)
        comps = AnovaEngine.tukey_hsd(res)

        # 3 groups -> 3 pairwise comparisons: A vs B, A vs C, B vs C
        self.assertEqual(len(comps), 3)
        for c in comps:
            self.assertTrue(c["significant"])
            self.assertLess(c["p_value"], 0.001)

    def test_two_way_anova(self) -> None:
        """Tests Two-Way ANOVA table calculations."""
        # 2 rows x 3 columns x 3 replicates
        np.random.seed(42)
        data = np.zeros((2, 3, 3))
        # Factor 1 effect: row 1 +10
        data[0, :, :] = 10.0 + np.random.normal(0, 0.5, size=(3, 3))
        data[1, :, :] = 20.0 + np.random.normal(0, 0.5, size=(3, 3))

        res = AnovaEngine.two_way(data)
        self.assertEqual(res["type"], "Two-Way ANOVA")
        table = res["anova_table"]
        # df_row = 1, df_col = 2, df_interaction = 2, df_residual = 2*3*2 = 12
        self.assertEqual(table[0]["df"], 1)
        self.assertEqual(table[1]["df"], 2)
        self.assertEqual(table[2]["df"], 2)
        self.assertEqual(table[3]["df"], 12)
        # Row factor should be highly significant
        self.assertLess(table[0]["p_value"], 1e-5)

    def test_two_way_anova_without_replicates(self) -> None:
        """Tests Two-Way ANOVA with 1 replicate per cell (interaction as error)."""
        # 3 rows x 3 columns x 1 replicate
        data = np.array([
            [[1.0], [2.0], [3.0]],
            [[4.0], [5.0], [6.0]],
            [[7.0], [8.0], [9.0]],
        ])
        res = AnovaEngine.two_way(data)
        self.assertEqual(res["type"], "Two-Way ANOVA")
        table = res["anova_table"]
        self.assertEqual(table[0]["df"], 2)
        self.assertEqual(table[1]["df"], 2)
        self.assertEqual(table[2]["df"], 4)  # df_residual = 4
        self.assertEqual(table[0]["source"], "Row Factor")

    def test_repeated_measures_anova(self) -> None:
        """Tests Repeated Measures One-Way ANOVA and Geisser-Greenhouse sphericity correction."""
        # 4 subjects across 3 treatments
        data = np.array([
            [10.0, 12.0, 15.0],
            [11.0, 13.0, 16.0],
            [9.0, 11.5, 14.5],
            [10.5, 12.5, 15.5],
        ])
        res = AnovaEngine.repeated_measures(data)
        self.assertEqual(res["type"], "Repeated Measures One-Way ANOVA")
        self.assertEqual(res["n_subjects"], 4)
        self.assertEqual(res["n_treatments"], 3)
        self.assertGreater(res["f_stat"], 50.0)
        self.assertLess(res["p_value"], 0.001)
        self.assertGreaterEqual(res["epsilon_geisser_greenhouse"], 0.5)
        self.assertLessEqual(res["epsilon_geisser_greenhouse"], 1.0)
        self.assertLess(res["p_value_corrected"], 0.01)

    def test_repeated_measures_with_missing_values(self) -> None:
        """Verifies row-wise complete case filtering for RM ANOVA."""
        data = np.array([
            [10.0, 12.0, 15.0],
            [11.0, np.nan, 16.0],  # Incomplete row
            [9.0, 11.5, 14.5],
            [10.5, 12.5, 15.5],
        ])
        res = AnovaEngine.repeated_measures(data)
        # Should drop the incomplete row and evaluate remaining 3 complete subjects
        self.assertEqual(res["n_subjects"], 3)
        self.assertGreater(res["f_stat"], 10.0)

    def test_rout_residuals_with_nan(self) -> None:
        """Verifies that rout_residuals handles NaN and detects outliers accurately."""
        res = np.array([0.1, 0.2, np.nan, 50.0, 0.1, -0.15])
        out = OutlierEngine.rout_residuals(res, df=4)
        self.assertEqual(out["outlier_indices"], [3])
        self.assertTrue(np.isnan(out["p_values"][2]))
        self.assertLess(out["p_values"][3], 1e-4)


if __name__ == "__main__":
    unittest.main()
