"""
Unit tests for Analysis Chaining, Family Tree Dependencies, and Constant Hooking in GemGraph.
Verifies DAG tracking, downstream live recalculation, frozen and orphaned sheet states,
and parameter registry binding.
"""

import unittest
from src.core.document import PrismDocument
from src.core.table import DataTable, TableType
from src.core.sheet import ResultSheet
from src.core.hooking import ConstantRegistry
from src.math.nonlinear import ConvergencePreset


class TestHookingAndAnalysisChain(unittest.TestCase):
    """Verifies dependency DAG, recalculation, frozen/orphaned semantics, and hooking."""

    def test_constant_registry_and_extraction(self) -> None:
        registry = ConstantRegistry()

        # 1. Extract from DataTable
        tbl = DataTable("tbl_qc", "QC Table", table_type=TableType.XY, num_rows=4, num_replicates=2)
        # Group A values: 10, 12, 14, 16, 18, 20, 22, 24 -> mean = 17, sd ≈ 4.899
        vals = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0]
        for i, v in enumerate(vals):
            r = i // 2
            s = i % 2
            tbl.set_cell_value(r, "Y", 0, s, v)

        extracted = registry.extract_from_table(tbl)
        self.assertGreaterEqual(len(extracted), 4)

        mean_c = registry.get(f"{tbl.table_id}_0_mean")
        self.assertIsNotNone(mean_c)
        self.assertEqual(mean_c.value, 17.0)

        qc_up = registry.get(f"{tbl.table_id}_0_mean_plus_2sd")
        self.assertIsNotNone(qc_up)
        sd_c = registry.get(f"{tbl.table_id}_0_sd")
        self.assertAlmostEqual(qc_up.value, 17.0 + 2 * sd_c.value)

        # 2. Extract from ResultSheet
        res = ResultSheet(
            sheet_id="res_fit",
            title="Nonlin Fit",
            source_sheet_id=tbl.table_id,
            raw_result_data={
                "params": {"Bottom": 5.0, "Top": 95.0, "LogEC50": -6.0},
                "goodness_of_fit": {"R2": 0.995, "Sy.x": 1.25},
            },
        )
        res_constants = registry.extract_from_result(res)
        self.assertGreaterEqual(len(res_constants), 4)

        # Check antilog EC50 registration
        ec50_c = registry.get("res_fit_ec50")
        self.assertIsNotNone(ec50_c)
        self.assertAlmostEqual(ec50_c.value, 1e-6)

        # Check R2
        r2_c = registry.get("res_fit_r2")
        self.assertIsNotNone(r2_c)
        self.assertEqual(r2_c.value, 0.995)

    def test_document_dag_and_recalculation(self) -> None:
        doc = PrismDocument("DAG Test")
        data_sheet = doc.create_data_sheet("Dose Response", table_type=TableType.XY, num_rows=6, num_replicates=1)
        tbl = data_sheet.table

        # Populate sigmoidal data
        xs = [-8.0, -7.0, -6.0, -5.0, -4.0, -3.0]
        for r, x in enumerate(xs):
            tbl.set_cell_value(r, "X", 0, 0, x)
            # 4PL with Bottom=0, Top=100, LogEC50=-5.5
            y_val = 100.0 / (1.0 + 10.0 ** (-5.5 - x))
            tbl.set_cell_value(r, "Y", 0, 0, y_val)

        # Create child ResultSheet linked via calc_spec
        res_sheet = doc.create_result_sheet(
            title="Nonlin fit 1",
            analysis_type="Nonlinear Regression",
            source_sheet_id=data_sheet.sheet_id,
            grid_type="green",
            calc_spec={
                "type": "nonlinear",
                "model_name": "Sigmoidal Dose-Response (Variable Slope)",
                "group_idx": 0,
                "preset": ConvergencePreset.STANDARD,
            },
        )

        # Verify dependency registration
        self.assertIn(res_sheet.sheet_id, data_sheet.child_sheet_ids)
        self.assertIn(data_sheet.sheet_id, res_sheet.parent_sheet_ids)

        # Run initial recalculation
        recalculated = doc.recalculate_downstream(data_sheet.sheet_id)
        self.assertIn(res_sheet.sheet_id, recalculated)
        self.assertIn("params", res_sheet.raw_result_data)

    def test_frozen_and_orphaned_sheet_states(self) -> None:
        doc = PrismDocument("Freeze & Orphan Test")
        d_sheet = doc.create_data_sheet("Parent Data")
        r_sheet = doc.create_result_sheet("Child Result", source_sheet_id=d_sheet.sheet_id)

        # 1. Test Freeze
        self.assertFalse(r_sheet.is_frozen)
        doc.freeze_sheet(r_sheet.sheet_id, True)
        self.assertTrue(r_sheet.is_frozen)

        # When frozen, downstream recalculation skips it
        recalculated = doc.recalculate_downstream(d_sheet.sheet_id)
        self.assertNotIn(r_sheet.sheet_id, recalculated)

        # Unfreeze
        doc.freeze_sheet(r_sheet.sheet_id, False)
        self.assertFalse(r_sheet.is_frozen)

        # 2. Test Orphan handling on parent sheet removal
        self.assertFalse(r_sheet.is_orphaned)
        doc.remove_sheet(d_sheet.sheet_id)
        self.assertIsNone(doc.get_sheet(d_sheet.sheet_id))

        # Child sheet remains in document, but marked as orphaned and source_sheet_id cleared
        self.assertIsNotNone(doc.get_sheet(r_sheet.sheet_id))
        self.assertTrue(r_sheet.is_orphaned)
        self.assertEqual(r_sheet.source_sheet_id, "")

    def test_multi_tier_dag_chain_recalculation(self) -> None:
        """
        Verifies a 4-tier chained pipeline:
        Raw Data -> Transform (log10) -> Normalize (0-100%) -> Linear Regression
        Recalculating from Raw Data updates all downstream tables and fit results.
        """
        doc = PrismDocument("Chained Pipeline")

        # Tier 1: Raw Data
        raw_sheet = doc.create_data_sheet("Raw Data", table_type=TableType.XY, num_rows=4, num_replicates=1)
        raw_tbl = raw_sheet.table
        xs = [1.0, 2.0, 3.0, 4.0]
        ys = [10.0, 100.0, 1000.0, 10000.0]
        for r in range(4):
            raw_tbl.set_cell_value(r, "X", 0, 0, xs[r])
            raw_tbl.set_cell_value(r, "Y", 0, 0, ys[r])

        # Tier 2: Transform (log10)
        from src.math.transforms import transform_data_table
        tf_tbl = transform_data_table(raw_tbl, transform_name="log10")
        tf_sheet = doc.create_data_sheet(
            "Transform of Raw Data",
            table_type=TableType.XY,
            num_rows=4,
            num_replicates=1,
            calc_spec={"type": "transform", "transform_name": "log10", "k_constant": 1.0},
        )
        tf_sheet.table = tf_tbl
        doc.register_dependency(raw_sheet.sheet_id, tf_sheet.sheet_id)

        # Tier 3: Normalize
        from src.math.manipulation import normalize_table, Baseline0Method, Baseline100Method, NormalizeScale
        norm_tbl = normalize_table(
            tf_tbl,
            b0_method=Baseline0Method.SMALLEST,
            b100_method=Baseline100Method.LARGEST,
            scale=NormalizeScale.PERCENT,
        )
        norm_sheet = doc.create_data_sheet(
            "Normalized of Transform",
            table_type=TableType.XY,
            num_rows=4,
            num_replicates=1,
            calc_spec={
                "type": "normalize",
                "b0_method": Baseline0Method.SMALLEST.value,
                "b100_method": Baseline100Method.LARGEST.value,
                "scale": NormalizeScale.PERCENT.value,
            },
        )
        norm_sheet.table = norm_tbl
        doc.register_dependency(tf_sheet.sheet_id, norm_sheet.sheet_id)

        # Tier 4: Linear Fit
        lin_res = doc.create_result_sheet(
            "Linear Fit of Normalized",
            analysis_type="Linear Regression",
            source_sheet_id=norm_sheet.sheet_id,
            grid_type="red",
            calc_spec={"type": "linear", "group_idx": 0},
        )
        doc.register_dependency(norm_sheet.sheet_id, lin_res.sheet_id)

        # Initial fit
        doc.recalculate_downstream(raw_sheet.sheet_id)
        self.assertIn("params", lin_res.raw_result_data)
        # Log values are [1, 2, 3, 4] -> normalized [0, 33.33, 66.67, 100]
        # X is [1, 2, 3, 4], so slope should be ~33.33
        initial_slope = lin_res.raw_result_data["params"]["Slope"]
        self.assertAlmostEqual(initial_slope, 33.3333, places=2)

        # Modify Raw Data: Change row 3 from 10000 to 100000 (log = 5)
        raw_tbl.set_cell_value(3, "Y", 0, 0, 100000.0)

        # Recalculate downstream
        recalculated = doc.recalculate_downstream(raw_sheet.sheet_id)
        self.assertIn(tf_sheet.sheet_id, recalculated)
        self.assertIn(norm_sheet.sheet_id, recalculated)
        self.assertIn(lin_res.sheet_id, recalculated)

        # Verify Tier 2 updated: log10(100000) = 5.0
        self.assertAlmostEqual(tf_sheet.table.get_cell_value(3, "Y", 0, 0), 5.0)

        # Verify Tier 4 updated: slope changes
        new_slope = lin_res.raw_result_data["params"]["Slope"]
        self.assertNotEqual(initial_slope, new_slope)
        self.assertGreater(len(lin_res.tabs), 0)

    def test_dag_cycle_prevention(self) -> None:
        """Verifies that register_dependency detects and rejects cyclic dependencies."""
        doc = PrismDocument("Cycle Test")
        s1 = doc.create_data_sheet("S1")
        s2 = doc.create_data_sheet("S2")
        s3 = doc.create_data_sheet("S3")

        self.assertTrue(doc.register_dependency(s1.sheet_id, s2.sheet_id))
        self.assertTrue(doc.register_dependency(s2.sheet_id, s3.sheet_id))

        # Attempt to register cycle: S3 -> S1
        self.assertFalse(doc.register_dependency(s3.sheet_id, s1.sheet_id))
        self.assertNotIn(s1.sheet_id, s3.child_sheet_ids)

        # Self cycle: S1 -> S1
        self.assertFalse(doc.register_dependency(s1.sheet_id, s1.sheet_id))


if __name__ == "__main__":
    unittest.main()
