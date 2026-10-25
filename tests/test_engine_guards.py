"""Regression tests for input guards in nonlinear and outlier engines."""

import unittest
import numpy as np

from src.math.nonlinear import NonlinearEngine, STANDARD_MODELS
from src.math.outlier import OutlierEngine


class TestEngineGuards(unittest.TestCase):
    def setUp(self):
        self.model = STANDARD_MODELS["Straight Line (Linear)"]
        self.x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.y = 2.0 * self.x + 1.0 + np.array([0.1, -0.1, 0.05, -0.05, 0.0])

    def test_fit_rejects_wrong_length_initial_params(self):
        with self.assertRaises(ValueError):
            NonlinearEngine.fit(self.x, self.y, self.model, initial_params=np.array([1.0]))

    def test_fit_rejects_nonfinite_initial_params(self):
        with self.assertRaises(ValueError):
            NonlinearEngine.fit(self.x, self.y, self.model, initial_params=np.array([np.nan, 0.0]))

    def test_fit_rejects_bad_confidence_level(self):
        with self.assertRaises(ValueError):
            NonlinearEngine.fit(self.x, self.y, self.model, confidence_level=1.5)

    def test_fit_still_works(self):
        res = NonlinearEngine.fit(self.x, self.y, self.model)
        self.assertAlmostEqual(res["parameters"][0]["best_fit"], 2.0, places=1)

    def test_rout_rejects_bad_q(self):
        for q in (0.0, -0.1, 1.5, float("nan")):
            with self.assertRaises(ValueError):
                OutlierEngine.rout_1d(np.array([1.0, 2.0, 3.0, 4.0]), q_fdr=q)

    def test_rout_constant_data_no_outliers(self):
        res = OutlierEngine.rout_1d(np.array([5.0, 5.0, 5.0, 5.0]))
        self.assertEqual(res["outlier_indices"], [])


if __name__ == "__main__":
    unittest.main()
