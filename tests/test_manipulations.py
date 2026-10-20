"""
Unit tests for scientific table manipulations in GemGraph.
Verifies normalization, baseline removal, pruning, fraction of total with CIs,
transposition, and Multiple Variables extraction.
"""

import unittest
import numpy as np
from src.core.table import DataTable, TableType, VariableType
from src.math.manipulation import (
    normalize_array,
    normalize_table,
    remove_baseline,
    prune_rows,
    fraction_of_total,
    transpose_table,
    extract_and_rearrange,
    Baseline0Method,
    Baseline100Method,
    BaselineMethod,
    BaselineOperation,
    NormalizeScale,
    wilson_score_interval,
    clopper_pearson_interval,
)


class TestTableManipulations(unittest.TestCase):
    """Verifies all manipulation engines."""

    def test_normalize_array_and_table(self) -> None:
        y = np.array([10.0, 30.0, 50.0])

        # Min = 0%, Max = 100%, percent scale
        norm_pct = normalize_array(y, Baseline0Method.SMALLEST, b0_val=0, b100_method=Baseline100Method.LARGEST, scale=NormalizeScale.PERCENT)
        np.testing.assert_allclose(norm_pct, [0.0, 50.0, 100.0])

        # Fraction scale (0.0 to 1.0)
        norm_frac = normalize_array(y, Baseline0Method.SMALLEST, b0_val=0, b100_method=Baseline100Method.LARGEST, scale=NormalizeScale.FRACTION)
        np.testing.assert_allclose(norm_frac, [0.0, 0.5, 1.0])

        # Manual baselines: 0% at 0.0, 100% at 100.0
        norm_man = normalize_array(y, Baseline0Method.MANUAL, b0_val=0.0, b100_method=Baseline100Method.MANUAL, b100_val=100.0, scale=NormalizeScale.PERCENT)
        np.testing.assert_allclose(norm_man, [10.0, 30.0, 50.0])

        # Normalize whole table with replicates
        tbl = DataTable("tbl_n", table_type=TableType.XY, num_rows=3, num_replicates=2)
        tbl.set_cell_value(0, "Y", 0, 0, 10.0)
        tbl.set_cell_value(1, "Y", 0, 0, 30.0)
        tbl.set_cell_value(2, "Y", 0, 0, 50.0)

        tbl.set_cell_value(0, "Y", 0, 1, 20.0)
        tbl.set_cell_value(1, "Y", 0, 1, 60.0)
        tbl.set_cell_value(2, "Y", 0, 1, 100.0)

        tbl_norm = normalize_table(tbl, Baseline0Method.SMALLEST, b0_val=0, b100_method=Baseline100Method.LARGEST, scale=NormalizeScale.PERCENT)
        self.assertEqual(tbl_norm.get_cell_value(0, "Y", 0, 0), 0.0)
        self.assertEqual(tbl_norm.get_cell_value(2, "Y", 0, 0), 100.0)
        self.assertEqual(tbl_norm.get_cell_value(0, "Y", 0, 1), 0.0)
        self.assertEqual(tbl_norm.get_cell_value(2, "Y", 0, 1), 100.0)

    def test_remove_baseline(self) -> None:
        y = np.array([5.0, 15.0, 25.0, 35.0])

        # Subtract first row
        sub_first = remove_baseline(y, method=BaselineMethod.FIRST_ROW, operation=BaselineOperation.SUBTRACT)
        np.testing.assert_allclose(sub_first, [0.0, 10.0, 20.0, 30.0])

        # Divide by first row
        div_first = remove_baseline(y, method=BaselineMethod.FIRST_ROW, operation=BaselineOperation.DIVIDE)
        np.testing.assert_allclose(div_first, [1.0, 3.0, 5.0, 7.0])

        # Mean of first 2 rows: (5 + 15)/2 = 10.0
        sub_mean2 = remove_baseline(y, method=BaselineMethod.MEAN_FIRST_K, operation=BaselineOperation.SUBTRACT, k_rows=2)
        np.testing.assert_allclose(sub_mean2, [-5.0, 5.0, 15.0, 25.0])

        # Linear regression baseline
        x = np.array([1.0, 2.0, 3.0, 4.0])
        # Perfect line: baseline = y itself -> residual is 0
        sub_lin = remove_baseline(y, x=x, method=BaselineMethod.LINEAR_REGRESSION, operation=BaselineOperation.SUBTRACT)
        np.testing.assert_allclose(sub_lin, [0.0, 0.0, 0.0, 0.0], atol=1e-10)

    def test_prune_rows(self) -> None:
        tbl = DataTable("tbl_p", table_type=TableType.XY, num_rows=10, num_replicates=1)
        for r in range(10):
            tbl.set_cell_value(r, "X", 0, 0, float(r + 1))
            tbl.set_cell_value(r, "Y", 0, 0, float(10 * (r + 1)))

        # Prune range [3.0, 7.0]
        pruned_range = prune_rows(tbl, x_min=3.0, x_max=7.0)
        self.assertEqual(pruned_range.num_rows, 5)
        self.assertEqual(pruned_range.get_cell_value(0, "X", 0, 0), 3.0)
        self.assertEqual(pruned_range.get_cell_value(4, "X", 0, 0), 7.0)

        # Decimate every 3 rows (indices 0, 3, 6, 9)
        decimated = prune_rows(tbl, decimate_k=3)
        self.assertEqual(decimated.num_rows, 4)
        self.assertEqual(decimated.get_cell_value(0, "X", 0, 0), 1.0)
        self.assertEqual(decimated.get_cell_value(1, "X", 0, 0), 4.0)

        # Average every 2 rows
        averaged = prune_rows(tbl, average_k=2)
        self.assertEqual(averaged.num_rows, 5)
        self.assertAlmostEqual(averaged.get_cell_value(0, "X", 0, 0), 1.5)
        self.assertAlmostEqual(averaged.get_cell_value(0, "Y", 0, 0), 15.0)

    def test_fraction_of_total_and_binomial_cis(self) -> None:
        # Confidence interval functions
        w_lo, w_hi = wilson_score_interval(50, 100, 0.95)
        self.assertAlmostEqual(w_lo, 0.4038, places=3)
        self.assertAlmostEqual(w_hi, 0.5962, places=3)

        c_lo, c_hi = clopper_pearson_interval(50, 100, 0.95)
        self.assertAlmostEqual(c_lo, 0.3983, places=3)
        self.assertAlmostEqual(c_hi, 0.6017, places=3)

        # Table test
        tbl = DataTable("tbl_fot", table_type=TableType.PARTS_OF_WHOLE, num_rows=3, num_replicates=1)
        tbl.set_cell_value(0, "Y", 0, 0, 25.0)
        tbl.set_cell_value(1, "Y", 0, 0, 25.0)
        tbl.set_cell_value(2, "Y", 0, 0, 50.0)

        res = fraction_of_total(tbl, total_type="column", scale=NormalizeScale.PERCENT)
        fractions = res["fractions"]
        np.testing.assert_allclose(fractions[0, :, 0], [25.0, 25.0, 50.0])
        self.assertFalse(np.isnan(res["ci_wilson"][0, 0, 0, 0]))
        self.assertFalse(np.isnan(res["ci_clopper"][0, 0, 0, 0]))

    def test_transpose_table(self) -> None:
        tbl = DataTable("tbl_t", table_type=TableType.COLUMN, num_rows=3, num_replicates=1, y_group_titles=["Control", "Treated"])
        tbl.row_titles = ["Sample 1", "Sample 2", "Sample 3"]
        tbl.set_cell_value(0, "Y", 0, 0, 1.0)
        tbl.set_cell_value(1, "Y", 0, 0, 2.0)
        tbl.set_cell_value(2, "Y", 0, 0, 3.0)

        tbl.set_cell_value(0, "Y", 1, 0, 10.0)
        tbl.set_cell_value(1, "Y", 1, 0, 20.0)
        tbl.set_cell_value(2, "Y", 1, 0, 30.0)

        trans = transpose_table(tbl)
        self.assertEqual(trans.num_rows, 2)
        self.assertEqual(len(trans.y_group_titles), 3)
        self.assertEqual(trans.y_group_titles, ["Sample 1", "Sample 2", "Sample 3"])
        self.assertEqual(trans.row_titles, ["Control", "Treated"])

        self.assertEqual(trans.get_cell_value(0, "Y", 0, 0), 1.0)
        self.assertEqual(trans.get_cell_value(0, "Y", 2, 0), 3.0)
        self.assertEqual(trans.get_cell_value(1, "Y", 1, 0), 20.0)

    def test_extract_and_rearrange_multiple_variables(self) -> None:
        tbl = DataTable("mv_extract", table_type=TableType.MULTIPLE_VARIABLES, num_rows=4)
        tbl.variables[0].name = "Group"
        tbl.variables[0].var_type = VariableType.CATEGORICAL
        tbl.variables[1].name = "Concentration"
        tbl.variables[1].var_type = VariableType.CONTINUOUS
        tbl.variables[2].name = "Response"
        tbl.variables[2].var_type = VariableType.CONTINUOUS

        data = [
            ("DrugA", 1.0, 10.5),
            ("DrugA", 2.0, 20.5),
            ("DrugB", 1.0, 15.0),
            ("DrugB", 2.0, 30.0),
        ]
        for r, (g, c, resp) in enumerate(data):
            tbl.set_variable_value(r, 0, g)
            tbl.set_variable_value(r, 1, c)
            tbl.set_variable_value(r, 2, resp)

        # Extract to XY: X = Concentration (1), Y = Response (2)
        xy_tbl = extract_and_rearrange(tbl, TableType.XY, {"x_var": 1, "y_vars": [2]})
        self.assertEqual(xy_tbl.table_type, TableType.XY)
        self.assertEqual(xy_tbl.get_cell_value(0, "X", 0, 0), 1.0)
        self.assertEqual(xy_tbl.get_cell_value(0, "Y", 0, 0), 10.5)

        # Extract to Column: Group = Group (0), Val = Response (2)
        col_tbl = extract_and_rearrange(tbl, TableType.COLUMN, {"group_var": 0, "val_var": 2})
        self.assertEqual(col_tbl.table_type, TableType.COLUMN)
        self.assertIn("DrugA", col_tbl.y_group_titles)
        self.assertIn("DrugB", col_tbl.y_group_titles)

    def test_extract_and_rearrange_grouped(self) -> None:
        """Verifies restructuring Multiple Variables into a Grouped table with replicate subcolumns."""
        tbl = DataTable("mv_grp", table_type=TableType.MULTIPLE_VARIABLES, num_rows=8)
        tbl.variables[0].name = "Genotype"
        tbl.variables[0].var_type = VariableType.CATEGORICAL
        tbl.variables[1].name = "Treatment"
        tbl.variables[1].var_type = VariableType.CATEGORICAL
        tbl.variables[2].name = "BloodPressure"
        tbl.variables[2].var_type = VariableType.CONTINUOUS

        records = [
            ("WT", "Vehicle", 120.0),
            ("WT", "Vehicle", 122.0),
            ("WT", "Drug", 110.0),
            ("WT", "Drug", 112.0),
            ("KO", "Vehicle", 140.0),
            ("KO", "Vehicle", 145.0),
            ("KO", "Drug", 125.0),
            ("KO", "Drug", 128.0),
        ]
        for r, (geno, treat, bp) in enumerate(records):
            tbl.set_variable_value(r, 0, geno)
            tbl.set_variable_value(r, 1, treat)
            tbl.set_variable_value(r, 2, bp)

        grouped_tbl = extract_and_rearrange(
            tbl,
            TableType.GROUPED,
            {"row_factor": 0, "col_factor": 1, "val_var": 2},
        )
        self.assertEqual(grouped_tbl.table_type, TableType.GROUPED)
        self.assertEqual(grouped_tbl.num_replicates, 2)
        self.assertIn("WT", grouped_tbl.row_titles)
        self.assertIn("KO", grouped_tbl.row_titles)
        self.assertIn("Vehicle", grouped_tbl.y_group_titles)
        self.assertIn("Drug", grouped_tbl.y_group_titles)

        wt_idx = grouped_tbl.row_titles.index("WT")
        veh_idx = grouped_tbl.y_group_titles.index("Vehicle")
        drug_idx = grouped_tbl.y_group_titles.index("Drug")

        # WT Vehicle replicates: 120, 122
        self.assertEqual(grouped_tbl.get_cell_value(wt_idx, "Y", veh_idx, 0), 120.0)
        self.assertEqual(grouped_tbl.get_cell_value(wt_idx, "Y", veh_idx, 1), 122.0)
        # WT Drug replicates: 110, 112
        self.assertEqual(grouped_tbl.get_cell_value(wt_idx, "Y", drug_idx, 0), 110.0)
        self.assertEqual(grouped_tbl.get_cell_value(wt_idx, "Y", drug_idx, 1), 112.0)


if __name__ == "__main__":
    unittest.main()
