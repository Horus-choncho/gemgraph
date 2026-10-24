"""
Unit and Integration Tests for Box & Whiskers and Violin Plots with Nested Distributions.
Validates five_number_summary, gaussian_kde_profile, canvas rendering in Color and Grayscale,
Column and Grouped table rendering, and plot_config persistence across Freeze, Pin, and .bigo roundtrip.
"""

import os
import tempfile
import unittest
import numpy as np
from PyQt6.QtWidgets import QApplication

from src.core.document import PrismDocument
from src.core.sheet import DataSheet, GraphSheet, SheetType, GraphType
from src.core.table import TableType, DataTable
from src.io.prism_io import PrismIO
from src.math.distributions import five_number_summary, gaussian_kde_profile
from src.ui.graph_canvas import GraphCanvasWidget

app = QApplication.instance()
if app is None:
    app = QApplication([])


class TestDistributionPlots(unittest.TestCase):
    def test_five_number_summary_and_tukey_outliers_on_known_array(self) -> None:
        """Verifies five_number_summary correctly calculates quartiles and Tukey 1.5xIQR outliers."""
        # 11 points: 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 100 (100 is an outlier)
        data = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 100], dtype=np.float64)

        res = five_number_summary(data, whisker_rule="tukey")

        self.assertEqual(res["median"], 15.0)
        self.assertEqual(res["q1"], 12.5)
        self.assertEqual(res["q3"], 17.5)
        # IQR = 17.5 - 12.5 = 5.0. Upper fence = 17.5 + 1.5 * 5.0 = 25.0.
        # Upper whisker should be max value <= 25, which is 19.0.
        self.assertEqual(res["whisker_high"], 19.0)
        self.assertEqual(res["whisker_low"], 10.0)
        self.assertEqual(list(res["outliers"]), [100.0])

        # MinMax rule
        res_minmax = five_number_summary(data, whisker_rule="minmax")
        self.assertEqual(res_minmax["whisker_high"], 100.0)
        self.assertEqual(res_minmax["whisker_low"], 10.0)
        self.assertEqual(len(res_minmax["outliers"]), 0)

    def test_gaussian_kde_profile_edge_cases_n_less_than_3_and_zero_variance(self) -> None:
        """Verifies gaussian_kde_profile safely returns flat fallback for N < 3 and zero variance."""
        # Empty array
        res_empty = gaussian_kde_profile(np.array([]))
        self.assertEqual(len(res_empty["grid"]), 128)
        self.assertTrue(np.all(res_empty["density"] == 0.0))

        # N = 2 points
        res_2 = gaussian_kde_profile(np.array([5.0, 10.0]))
        self.assertEqual(len(res_2["grid"]), 128)
        self.assertTrue(np.all(res_2["density"] == 0.0))

        # Zero variance: N=10 identical points
        res_zero_var = gaussian_kde_profile(np.full(10, 42.0))
        self.assertEqual(len(res_zero_var["grid"]), 128)
        self.assertTrue(np.all(res_zero_var["density"] == 0.0))

        # Valid distribution: bimodal sample
        rng = np.random.default_rng(42)
        bimodal = np.concatenate([rng.normal(10, 1, 50), rng.normal(25, 2, 50)])
        res_kde = gaussian_kde_profile(bimodal, n_points=128)
        self.assertEqual(len(res_kde["grid"]), 128)
        self.assertAlmostEqual(np.max(res_kde["half_width"]), 1.0, places=4)
        self.assertGreaterEqual(np.min(res_kde["density"]), 0.0)

    def test_canvas_rendering_box_and_violin_column_and_grouped(self) -> None:
        """Verifies Box and Violin plot rendering on canvas for both Column and Grouped tables."""
        canvas = GraphCanvasWidget()

        # 1. Column table: Box plot
        col_tbl = DataTable(TableType.COLUMN, num_rows=10, num_replicates=1, y_group_titles=["Ctrl", "Drug"])
        for r in range(10):
            col_tbl.set_cell_value(r, "Y", 0, 0, float(10 + r))
            col_tbl.set_cell_value(r, "Y", 1, 0, float(20 + 2 * r))
        # Exclude one cell to verify exclusion handling
        col_tbl.set_excluded(0, "Y", 0, 0, True)

        canvas.update_data(col_tbl)
        canvas.set_graph_type(GraphType.BOX_WHISKERS.value)
        self.assertEqual(canvas.current_graph_type, GraphType.BOX_WHISKERS.value)

        # Toggle scatter points
        canvas.set_show_scatter(True)
        self.assertTrue(canvas.show_scatter_points)

        # Switch to Violin plot
        canvas.set_graph_type(GraphType.VIOLIN.value)
        self.assertEqual(canvas.current_graph_type, GraphType.VIOLIN.value)

        # Toggle grayscale
        canvas.set_grayscale_mode(True)
        self.assertTrue(canvas.is_grayscale_mode)

        # 2. Grouped table: Nested distributions
        grp_tbl = DataTable(TableType.GROUPED, num_rows=4, num_replicates=5, y_group_titles=["Male", "Female"])
        rng = np.random.default_rng(101)
        for r in range(4):
            for s in range(5):
                grp_tbl.set_cell_value(r, "Y", 0, s, float(rng.normal(50, 5)))
                grp_tbl.set_cell_value(r, "Y", 1, s, float(rng.normal(70, 7)))

        canvas.update_data(grp_tbl)
        canvas.set_graph_type(GraphType.BOX_WHISKERS.value)
        self.assertEqual(canvas.current_graph_type, GraphType.BOX_WHISKERS.value)

        canvas.set_graph_type(GraphType.VIOLIN.value)
        self.assertEqual(canvas.current_graph_type, GraphType.VIOLIN.value)

    def test_plot_config_persistence_and_bigo_roundtrip(self) -> None:
        """Verifies BOX_WHISKERS and VIOLIN survive Freeze, Pin, and .bigo serialization."""
        doc = PrismDocument("Distributions Test")
        sheet = doc.create_data_sheet(
            title="Biomarkers",
            table_type=TableType.COLUMN,
            num_rows=8,
            num_replicates=1,
            y_group_titles=["Baseline", "Post-Tx"],
        )
        for r in range(8):
            sheet.table.set_cell_value(r, "Y", 0, 0, float(100 + 5 * r))
            sheet.table.set_cell_value(r, "Y", 1, 0, float(80 + 3 * r))

        graph1 = doc.create_graph_sheet(title="Box Plot", source_sheet_id=sheet.sheet_id)
        graph1.graph_type = GraphType.BOX_WHISKERS
        graph1.plot_config["graph_type"] = GraphType.BOX_WHISKERS.value
        graph1.is_pinned = True

        graph2 = doc.create_graph_sheet(title="Violin Plot", source_sheet_id=sheet.sheet_id)
        graph2.graph_type = GraphType.VIOLIN
        graph2.plot_config["graph_type"] = GraphType.VIOLIN.value
        graph2.is_frozen = True

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "dist_test.bigo")
            PrismIO.save(doc, file_path)

            loaded_doc = PrismIO.load(file_path)
            loaded_g1 = loaded_doc.get_sheet(graph1.sheet_id)
            loaded_g2 = loaded_doc.get_sheet(graph2.sheet_id)

            self.assertIsInstance(loaded_g1, GraphSheet)
            self.assertEqual(loaded_g1.graph_type, GraphType.BOX_WHISKERS)
            self.assertEqual(loaded_g1.plot_config.get("graph_type"), GraphType.BOX_WHISKERS.value)
            self.assertTrue(loaded_g1.is_pinned)

            self.assertIsInstance(loaded_g2, GraphSheet)
            self.assertEqual(loaded_g2.graph_type, GraphType.VIOLIN)
            self.assertEqual(loaded_g2.plot_config.get("graph_type"), GraphType.VIOLIN.value)
            self.assertTrue(loaded_g2.is_frozen)

    def test_sheet_navigation_preserves_distribution_graph_type(self) -> None:
        """Verifies graph type survives sheet navigation in MainWindow."""
        from src.ui.main_window import MainWindow

        win = MainWindow()
        data_sheet = win.document.create_data_sheet(
            title="NavTest Data",
            table_type=TableType.COLUMN,
            num_rows=6,
            num_replicates=1,
            y_group_titles=["A", "B"],
        )
        for r in range(6):
            data_sheet.table.set_cell_value(r, "Y", 0, 0, float(10 + r))
            data_sheet.table.set_cell_value(r, "Y", 1, 0, float(20 + r))

        g1 = win.document.create_graph_sheet("Box Graph", source_sheet_id=data_sheet.sheet_id)
        g1.graph_type = GraphType.BOX_WHISKERS
        g1.plot_config["graph_type"] = GraphType.BOX_WHISKERS.value

        g2 = win.document.create_graph_sheet("Violin Graph", source_sheet_id=data_sheet.sheet_id)
        g2.graph_type = GraphType.VIOLIN
        g2.plot_config["graph_type"] = GraphType.VIOLIN.value

        win.navigator.populate(win.document)

        # Select g1
        win.on_sheet_selected(g1.sheet_id)
        self.assertEqual(win.graph_canvas.current_graph_type, GraphType.BOX_WHISKERS.value)

        # Select g2
        win.on_sheet_selected(g2.sheet_id)
        self.assertEqual(win.graph_canvas.current_graph_type, GraphType.VIOLIN.value)

        # Navigate back to g1
        win.on_sheet_selected(g1.sheet_id)
        self.assertEqual(win.graph_canvas.current_graph_type, GraphType.BOX_WHISKERS.value)

    def test_grouped_table_rendering_with_excluded_cells(self) -> None:
        """Verifies Box and Violin plot rendering on Grouped tables with excluded cells and scatter toggle."""
        canvas = GraphCanvasWidget()
        grp_tbl = DataTable(TableType.GROUPED, num_rows=5, num_replicates=3, y_group_titles=["Control", "Treated"])
        for r in range(5):
            for s in range(3):
                grp_tbl.set_cell_value(r, "Y", 0, s, float(10 * r + s + 1))
                grp_tbl.set_cell_value(r, "Y", 1, s, float(15 * r + s + 1))

        # Exclude cells in multiple rows and groups
        grp_tbl.set_excluded(0, "Y", 0, 0, True)
        grp_tbl.set_excluded(1, "Y", 1, 2, True)

        canvas.update_data(grp_tbl)
        canvas.set_graph_type(GraphType.BOX_WHISKERS.value)
        canvas.set_show_scatter(True)
        self.assertEqual(canvas.current_graph_type, GraphType.BOX_WHISKERS.value)

        canvas.set_graph_type(GraphType.VIOLIN.value)
        canvas.set_show_scatter(False)
        self.assertEqual(canvas.current_graph_type, GraphType.VIOLIN.value)


if __name__ == "__main__":
    unittest.main()

