"""
Unit tests for standard and pharmacological data transforms in GemGraph.
Verifies all mathematical formulations and edge cases against GraphPad Prism 10 specs.
"""

import unittest
import numpy as np
from src.math.transforms import (
    arithmetic_transform,
    probit_transform,
    rank_transform,
    zscore_transform,
    trig_transform,
    abs_transform,
    gaussian_noise,
    cross_transform,
    pharmacological_transform,
    concentration_x_transform,
    eval_user_defined_transform,
)


class TestDataTransforms(unittest.TestCase):
    """Verifies all mathematical transforms."""

    def test_arithmetic_transforms(self) -> None:
        y = np.array([1.0, 2.0, 4.0, 10.0, np.nan])

        # Y * K
        res = arithmetic_transform(y, "Y*K", k=2.5)
        np.testing.assert_allclose(res[:4], [2.5, 5.0, 10.0, 25.0])
        self.assertTrue(np.isnan(res[4]))

        # Y + K
        res = arithmetic_transform(y, "Y+K", k=10.0)
        np.testing.assert_allclose(res[:4], [11.0, 12.0, 14.0, 20.0])

        # Y - K
        res = arithmetic_transform(y, "Y-K", k=1.0)
        np.testing.assert_allclose(res[:4], [0.0, 1.0, 3.0, 9.0])

        # Y / K
        res = arithmetic_transform(y, "Y/K", k=2.0)
        np.testing.assert_allclose(res[:4], [0.5, 1.0, 2.0, 5.0])

        # Y^2
        res = arithmetic_transform(y, "Y^2")
        np.testing.assert_allclose(res[:4], [1.0, 4.0, 16.0, 100.0])

        # Y^K
        res = arithmetic_transform(y, "Y^K", k=3.0)
        np.testing.assert_allclose(res[:4], [1.0, 8.0, 64.0, 1000.0])

        # log10
        res = arithmetic_transform(y, "log10")
        np.testing.assert_allclose(res[:4], [0.0, np.log10(2.0), np.log10(4.0), 1.0])

        # -log10
        res = arithmetic_transform(y, "-log10")
        np.testing.assert_allclose(res[:4], [0.0, -np.log10(2.0), -np.log10(4.0), -1.0])

        # ln
        res = arithmetic_transform(y, "ln")
        np.testing.assert_allclose(res[:4], np.log([1.0, 2.0, 4.0, 10.0]))

        # 10^Y
        y_small = np.array([0.0, 1.0, 2.0])
        res = arithmetic_transform(y_small, "10^Y")
        np.testing.assert_allclose(res, [1.0, 10.0, 100.0])

        # exp
        res = arithmetic_transform(y_small, "exp")
        np.testing.assert_allclose(res, [1.0, np.e, np.e**2])

        # 1/Y
        res = arithmetic_transform(y, "1/Y")
        np.testing.assert_allclose(res[:4], [1.0, 0.5, 0.25, 0.1])

        # sqrt
        res = arithmetic_transform(y, "sqrt")
        np.testing.assert_allclose(res[:4], [1.0, np.sqrt(2.0), 2.0, np.sqrt(10.0)])

        # logit: ln(p / (1 - p))
        p_vals = np.array([0.1, 0.5, 0.9])
        res = arithmetic_transform(p_vals, "logit")
        np.testing.assert_allclose(res[1], 0.0)  # ln(0.5 / 0.5) = 0
        np.testing.assert_allclose(res[0], np.log(0.1 / 0.9))

    def test_probit_transform_legacy_offset(self) -> None:
        """Tests probit transform with Prism's legacy +5.0 offset."""
        # probit(0.5) = 5.0
        # probit(0.025) ≈ 3.04 (standard normal 2.5% quantile is -1.95996 -> -1.96 + 5.0 = 3.04)
        p = np.array([0.5, 0.025])
        res = probit_transform(p)
        self.assertAlmostEqual(res[0], 5.0, places=4)
        self.assertAlmostEqual(res[1], 3.04, places=2)

        # Percentage input (50%, 2.5%)
        p_pct = np.array([50.0, 2.5])
        res_pct = probit_transform(p_pct)
        self.assertAlmostEqual(res_pct[0], 5.0, places=4)
        self.assertAlmostEqual(res_pct[1], 3.04, places=2)

    def test_rank_and_zscore(self) -> None:
        y = np.array([10.0, 50.0, 20.0, 40.0, 30.0])
        ranks = rank_transform(y)
        np.testing.assert_array_equal(ranks, [1.0, 5.0, 2.0, 4.0, 3.0])

        z = zscore_transform(y)
        self.assertAlmostEqual(np.mean(z), 0.0, places=5)
        self.assertAlmostEqual(np.std(z, ddof=1), 1.0, places=5)

    def test_trigonometric_and_abs(self) -> None:
        angles = np.array([0.0, np.pi / 2, np.pi])
        sin_res = trig_transform(angles, "sin")
        np.testing.assert_allclose(sin_res, [0.0, 1.0, 0.0], atol=1e-7)

        cos_res = trig_transform(angles, "cos")
        np.testing.assert_allclose(cos_res, [1.0, 0.0, -1.0], atol=1e-7)

        neg = np.array([-5.0, 3.0, -10.0])
        np.testing.assert_allclose(abs_transform(neg), [5.0, 3.0, 10.0])

    def test_gaussian_noise(self) -> None:
        y = np.full(100, 10.0)
        noisy = gaussian_noise(y, sigma=2.0, seed=42)
        self.assertNotEqual(y[0], noisy[0])
        self.assertAlmostEqual(float(np.mean(noisy)), 10.0, delta=0.5)

    def test_coordinate_cross_transforms(self) -> None:
        x = np.array([2.0, 4.0, 10.0])
        y = np.array([4.0, 2.0, 5.0])

        np.testing.assert_allclose(cross_transform(x, y, "X/Y"), [0.5, 2.0, 2.0])
        np.testing.assert_allclose(cross_transform(x, y, "Y/X"), [2.0, 0.5, 0.5])
        np.testing.assert_allclose(cross_transform(x, y, "Y-X"), [2.0, -2.0, -5.0])
        np.testing.assert_allclose(cross_transform(x, y, "Y+X"), [6.0, 6.0, 15.0])
        np.testing.assert_allclose(cross_transform(x, y, "Y*X"), [8.0, 8.0, 50.0])
        np.testing.assert_allclose(cross_transform(x, y, "X-Y"), [-2.0, 2.0, 5.0])
        np.testing.assert_allclose(cross_transform(x, y, "K-Y", k=10.0), [6.0, 8.0, 5.0])
        np.testing.assert_allclose(cross_transform(x, y, "K/Y", k=20.0), [5.0, 10.0, 4.0])
        np.testing.assert_allclose(cross_transform(x, y, "log2"), [2.0, 1.0, np.log2(5.0)])
        np.testing.assert_allclose(cross_transform(x, y, "2^Y"), [16.0, 4.0, 32.0])

    def test_pharmacological_transforms(self) -> None:
        # [S] concentrations and V reaction velocities
        s = np.array([1.0, 2.0, 5.0, 10.0])
        v = np.array([5.0, 8.0, 12.0, 15.0])

        # Lineweaver-Burk: 1/X, 1/Y
        lb_x, lb_y = pharmacological_transform(s, v, "lineweaver_burk")
        np.testing.assert_allclose(lb_x, [1.0, 0.5, 0.2, 0.1])
        np.testing.assert_allclose(lb_y, [1.0 / 5.0, 1.0 / 8.0, 1.0 / 12.0, 1.0 / 15.0])

        # Eadie-Hofstee: Y/X, Y
        eh_x, eh_y = pharmacological_transform(s, v, "eadie_hofstee")
        np.testing.assert_allclose(eh_x, [5.0, 4.0, 2.4, 1.5])
        np.testing.assert_allclose(eh_y, v)

        # Hanes-Woolf: X, X/Y
        hw_x, hw_y = pharmacological_transform(s, v, "hanes_woolf")
        np.testing.assert_allclose(hw_x, s)
        np.testing.assert_allclose(hw_y, [1.0 / 5.0, 2.0 / 8.0, 5.0 / 12.0, 10.0 / 15.0])

        # Scatchard: Y, Y/X
        sc_x, sc_y = pharmacological_transform(s, v, "scatchard")
        np.testing.assert_allclose(sc_x, v)
        np.testing.assert_allclose(sc_y, [5.0, 4.0, 2.4, 1.5])

        # Hill plot: log10(X), log10(Y / (Ymax - Y))
        hp_x, hp_y = pharmacological_transform(s, v, "hill_plot", y_max=20.0)
        np.testing.assert_allclose(hp_x, np.log10(s))
        np.testing.assert_allclose(hp_y, np.log10(v / (20.0 - v)))

    def test_concentration_x_transform(self) -> None:
        # X with 0 baseline concentration
        x = np.array([0.0, 1e-8, 1e-7, 1e-6])
        x_trans = concentration_x_transform(x, scale_multiplier=1.0, log_base="log10")

        # Zero value should be replaced with baseline (2 log units below 1e-8 = 1e-10 -> log10 = -10.0)
        self.assertAlmostEqual(x_trans[0], -10.0, places=4)
        self.assertAlmostEqual(x_trans[1], -8.0, places=4)
        self.assertAlmostEqual(x_trans[2], -7.0, places=4)
        self.assertAlmostEqual(x_trans[3], -6.0, places=4)

    def test_user_defined_transform_with_if_branching(self) -> None:
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([-5.0, 0.0, 5.0, 10.0])

        # IF(Y > 0, log(Y), 0)
        expr = "IF(Y > 0, log(Y), 0)"
        res = eval_user_defined_transform(expr, x, y)
        self.assertEqual(res[0], 0.0)
        self.assertEqual(res[1], 0.0)
        self.assertAlmostEqual(res[2], np.log(5.0))
        self.assertAlmostEqual(res[3], np.log(10.0))

        # Nested IF: IF(Y < 0, -1, IF(Y == 0, 0, 1))
        expr_nested = "IF(Y < 0, -1, IF(Y == 0, 0, 1))"
        res_nested = eval_user_defined_transform(expr_nested, x, y)
        np.testing.assert_array_equal(res_nested, [-1.0, 0.0, 1.0, 1.0])

    def test_user_defined_transform_boolean_and_ternary(self) -> None:
        """Verifies AND, OR, NOT, and Python ternary (IfExp) in user-defined AST evaluator."""
        x = np.array([0.0, 1.0, 2.0, -1.0])
        y = np.array([10.0, 20.0, -5.0, 30.0])

        # Test boolean AND
        expr_and = "IF(X > 0 and Y > 0, 100.0, 0.0)"
        res_and = eval_user_defined_transform(expr_and, x, y)
        np.testing.assert_array_equal(res_and, [0.0, 100.0, 0.0, 0.0])

        # Test boolean OR
        expr_or = "IF(X > 0 or Y < 0, 50.0, 0.0)"
        res_or = eval_user_defined_transform(expr_or, x, y)
        np.testing.assert_array_equal(res_or, [0.0, 50.0, 50.0, 0.0])

        # Test Python ternary: Y * 2 if X > 0 else 0
        expr_ternary = "Y * 2.0 if X > 0 else 0.0"
        res_ternary = eval_user_defined_transform(expr_ternary, x, y)
        np.testing.assert_array_equal(res_ternary, [0.0, 40.0, -10.0, 0.0])


if __name__ == "__main__":
    unittest.main()
