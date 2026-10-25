"""
Unit and Integration Tests for Biomedical Models and Enzyme Kinetics Suite.
Validates Michaelis-Menten, Substrate Inhibition, Hill Kinetics, Competitive Inhibition,
Extra-Sum-of-Squares F-tests, parameter confidence intervals, and diagnostic transforms.
"""

import unittest
import numpy as np

from src.math.nonlinear import NonlinearEngine, STANDARD_MODELS, ConvergencePreset
from src.math.transforms import pharmacological_transform


class TestBiomedicalModels(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(12345)

    def test_michaelis_menten_parameter_recovery_and_confidence_intervals(self) -> None:
        """Verifies Michaelis-Menten fit recovers Vmax and Km within 5% and 95% CIs cover true values."""
        true_vmax = 120.0
        true_km = 15.0

        # Substrate concentrations
        x = np.array([0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 40.0, 80.0, 120.0, 200.0], dtype=np.float64)
        y_true = (true_vmax * x) / (true_km + x)

        # 3% Gaussian noise
        noise = self.rng.normal(0, 0.03 * y_true)
        y = y_true + noise

        model = STANDARD_MODELS["Michaelis-Menten"]
        res = NonlinearEngine.fit(x, y, model, preset=ConvergencePreset.STRICT)

        params = {p["name"]: p for p in res["parameters"]}
        fit_vmax = params["Vmax"]["best_fit"]
        fit_km = params["Km"]["best_fit"]

        # Parameter recovery within 5%
        self.assertAlmostEqual(fit_vmax, true_vmax, delta=true_vmax * 0.05)
        self.assertAlmostEqual(fit_km, true_km, delta=true_km * 0.05)

        # 95% CIs contain true values
        self.assertLessEqual(params["Vmax"]["ci_lower"], true_vmax)
        self.assertGreaterEqual(params["Vmax"]["ci_upper"], true_vmax)
        self.assertLessEqual(params["Km"]["ci_lower"], true_km)
        self.assertGreaterEqual(params["Km"]["ci_upper"], true_km)

        # R^2 should be very high (> 0.98)
        self.assertGreater(res["r_squared"], 0.98)

    def test_substrate_inhibition_fit_convergence(self) -> None:
        """Verifies Substrate Inhibition model converges on data with velocity drop at high [S]."""
        vmax = 100.0
        km = 10.0
        ki = 50.0

        x = np.array([1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 400.0], dtype=np.float64)
        y_true = (vmax * x) / (km + x * (1.0 + x / ki))
        noise = self.rng.normal(0, 0.02 * y_true)
        y = y_true + noise

        model = STANDARD_MODELS["Substrate Inhibition"]
        res = NonlinearEngine.fit(x, y, model, preset=ConvergencePreset.STRICT)

        params = {p["name"]: p["best_fit"] for p in res["parameters"]}
        self.assertAlmostEqual(params["Vmax"], vmax, delta=vmax * 0.15)
        self.assertAlmostEqual(params["Km"], km, delta=km * 0.20)
        self.assertAlmostEqual(params["Ki"], ki, delta=ki * 0.25)
        self.assertGreater(res["r_squared"], 0.98)

    def test_hill_kinetics_allosteric_fit_convergence(self) -> None:
        """Verifies Allosteric Sigmoidal (Hill) fit converges and recovers sigmoidal cooperativity."""
        vmax = 150.0
        khalf = 25.0
        hill_coeff = 2.5

        x = np.array([2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 60.0, 100.0], dtype=np.float64)
        y_true = (vmax * (x ** hill_coeff)) / ((khalf ** hill_coeff) + (x ** hill_coeff))
        y = y_true + self.rng.normal(0, 0.02 * y_true)

        model = STANDARD_MODELS["Allosteric Sigmoidal (Hill Kinetics)"]
        res = NonlinearEngine.fit(x, y, model, preset=ConvergencePreset.STRICT)

        params = {p["name"]: p["best_fit"] for p in res["parameters"]}
        self.assertAlmostEqual(params["Vmax"], vmax, delta=vmax * 0.10)
        self.assertAlmostEqual(params["Khalf"], khalf, delta=khalf * 0.15)
        self.assertAlmostEqual(params["h"], hill_coeff, delta=0.3)
        self.assertGreater(res["r_squared"], 0.98)

    def test_extra_sum_of_squares_f_test_model_selection(self) -> None:
        """
        Verifies compare_models selects Michaelis-Menten on pure MM data (p > 0.05)
        and selects Substrate Inhibition when strong inhibition is present (p < 0.01).
        """
        mm_model = STANDARD_MODELS["Michaelis-Menten"]
        inh_model = STANDARD_MODELS["Substrate Inhibition"]

        x = np.array([1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 150.0, 300.0], dtype=np.float64)

        # 1. Pure MM data with minor noise
        y_mm = (100.0 * x) / (15.0 + x) + self.rng.normal(0, 1.0, size=len(x))
        comp_mm = NonlinearEngine.compare_models(x, y_mm, mm_model, inh_model)
        self.assertEqual(comp_mm["simpler_model"], "Michaelis-Menten")
        self.assertEqual(comp_mm["complex_model"], "Substrate Inhibition")
        self.assertGreater(comp_mm["p_value"], 0.05)
        self.assertEqual(comp_mm["preferred_model"], "Michaelis-Menten")

        # 2. Strong substrate inhibition data (Ki = 30)
        y_inh = (100.0 * x) / (15.0 + x * (1.0 + x / 30.0)) + self.rng.normal(0, 0.5, size=len(x))
        comp_inh = NonlinearEngine.compare_models(x, y_inh, mm_model, inh_model)
        self.assertLess(comp_inh["p_value"], 0.01)
        self.assertEqual(comp_inh["preferred_model"], "Substrate Inhibition")
        self.assertLess(comp_inh["aicc_complex"], comp_inh["aicc_simpler"])

    def test_model_comparison_rejects_non_nested_models(self) -> None:
        """Verifies compare_models raises ValueError if both models have equal number of parameters."""
        model_a = STANDARD_MODELS["Straight Line (Linear)"]  # 2 params: Slope, Intercept
        model_b = STANDARD_MODELS["Michaelis-Menten"]        # 2 params: Vmax, Km

        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        with self.assertRaises(ValueError) as ctx:
            NonlinearEngine.compare_models(x, y, model_a, model_b)
        self.assertIn("different parameter counts", str(ctx.exception))

    def test_lineweaver_burk_transform_linearity_on_noiseless_data(self) -> None:
        """Verifies Lineweaver-Burk transform (1/V vs 1/[S]) is perfectly linear for noiseless MM data."""
        vmax = 100.0
        km = 20.0
        x = np.array([1.0, 2.0, 5.0, 10.0, 20.0, 50.0], dtype=np.float64)
        y = (vmax * x) / (km + x)

        trans_x, trans_y = pharmacological_transform(x, y, "lineweaver_burk")
        valid = ~np.isnan(trans_x) & ~np.isnan(trans_y)
        trans_x, trans_y = trans_x[valid], trans_y[valid]

        # Expected: 1/V = (Km/Vmax) * (1/[S]) + (1/Vmax)
        expected_slope = km / vmax  # 0.2
        expected_intercept = 1.0 / vmax  # 0.01

        poly = np.polyfit(trans_x, trans_y, deg=1)
        fit_slope, fit_intercept = poly[0], poly[1]

        self.assertAlmostEqual(fit_slope, expected_slope, places=6)
        self.assertAlmostEqual(fit_intercept, expected_intercept, places=6)


if __name__ == "__main__":
    unittest.main()
