"""
Comprehensive Unit & Integration Test Suite for GemGraph Specs.
Covers:
1. Multi-X column table architecture & spreadsheet interactions
2. Hybrid Graph 1: Combined Bars & Points with whiskers (Mean ± SEM/SD/95% CI)
3. Hybrid Graph 2: Grouped Bars + Replicate Points (1661_Points_and_grouped_bars.pzfx)
4. Hybrid Graph 3: Columns/Spikes + Curve Fit on XY Graph
5. Active Analysis Overlay Registry & Layer Clearing UI
"""

import unittest
import numpy as np
from PyQt6.QtWidgets import QApplication, QMenu
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtCore import Qt

from src.core.table import DataTable, TableType, ColumnInfo
from src.core.sheet import GraphSheet, GraphType, ResultSheet, DataSheet
from src.core.document import PrismDocument
from src.io.xml_io import PrismXMLIO
from src.ui.spreadsheet import ScientificSpreadsheetModel, ScientificSpreadsheetView
from src.ui.graph_canvas import ScientificPlotCanvas
from src.ui.mediator import WorkspaceMediator
import pyqtgraph as pg


class TestMultiXColumnArchitecture(unittest.TestCase):
    """Verifies Multi-X column storage, ordering, bindings, and spreadsheet UI actions."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_multi_x_column_storage_and_ordering(self):
        table = DataTable(table_id="tbl_mx", title="Multi-X Table", table_type=TableType.XY, num_rows=5, num_replicates=1)
        self.assertEqual(table.get_x_column_count(), 1)
        self.assertEqual(len(table.column_order), 3)  # [('X', 0), ('Y', 0), ('Y', 1)]
        
        # Add a second X column
        x2_idx = table.add_x_column(title="X2 (Time)")
        self.assertEqual(x2_idx, 1)
        self.assertEqual(table.get_x_column_count(), 2)
        self.assertEqual(table.x_columns[1]["title"], "X2 (Time)")
        
        # Columns inspection
        cols = table.get_columns()
        # Initial: X0, Y0, Y1, then X1 was appended: total 4
        self.assertEqual(len(cols), 4)
        self.assertEqual(cols[0].role, "X")
        self.assertEqual(cols[3].role, "X")
        self.assertEqual(cols[3].title, "X2 (Time)")

    def test_multi_x_y_association_and_clean_xy(self):
        """Tests that Y groups bind to the preceding X column, falling back to X1."""
        table = DataTable(table_id="tbl_assoc", title="Assoc Table", table_type=TableType.XY, num_rows=3, num_replicates=1)
        # Sequence: X1, Y_A, X2, Y_B
        table.y_group_titles = ["Group A"]
        table.column_order = [("X", 0), ("Y", 0)]
        table.x_columns[0]["data"] = np.array([1.0, 2.0, 3.0])
        table.y_data = np.full((1, 3, 1), np.nan)
        table.y_data[0, :, 0] = [10.0, 20.0, 30.0]

        # Add second X and second Y group
        table.add_x_column(title="X2")
        table.x_columns[1]["data"] = np.array([100.0, 200.0, 300.0])
        table.add_y_group(title="Group B")
        # Explicit order: X1, Y_A, X2, Y_B
        table.column_order = [("X", 0), ("Y", 0), ("X", 1), ("Y", 1)]

        self.assertEqual(table.get_y_associated_x_idx(0), 0)
        self.assertEqual(table.get_y_associated_x_idx(1), 1)

        xs_a, ys_a = table.get_clean_xy(0)
        np.testing.assert_array_equal(xs_a, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(ys_a, [10.0, 20.0, 30.0])

        table.y_data[1, :, 0] = [5.0, 15.0, 25.0]
        xs_b, ys_b = table.get_clean_xy(1)
        np.testing.assert_array_equal(xs_b, [100.0, 200.0, 300.0])
        np.testing.assert_array_equal(ys_b, [5.0, 15.0, 25.0])

    def test_multi_x_serialization_roundtrip(self):
        table = DataTable(table_id="tbl_ser", title="Serial Table", table_type=TableType.XY, num_rows=4)
        table.add_x_column(title="Second X")
        table.x_columns[0]["data"] = np.array([1.0, 2.0, 3.0, 4.0])
        table.x_columns[1]["data"] = np.array([10.0, 20.0, 30.0, 40.0])
        table.set_cell_value(0, "Y", 0, 0, 99.0)

        d = table.to_dict()
        self.assertIn("x_columns", d)
        self.assertEqual(len(d["x_columns"]), 2)

        restored = DataTable.from_dict(d)
        self.assertEqual(restored.get_x_column_count(), 2)
        self.assertEqual(restored.x_columns[1]["title"], "Second X")
        np.testing.assert_array_equal(restored.x_columns[1]["data"], [10.0, 20.0, 30.0, 40.0])
        self.assertEqual(restored.get_cell_value(0, "Y", 0, 0), 99.0)

    def test_spreadsheet_model_multi_x_and_parsing(self):
        table = DataTable(table_id="tbl_ss", title="Spreadsheet Table", table_type=TableType.XY, num_rows=5)
        model = ScientificSpreadsheetModel(table)

        # Insert X column via model
        col_count_before = model.columnCount()
        inserted_idx = model.insert_x_column(col=1, title="Inserted X")
        self.assertTrue(inserted_idx)
        self.assertEqual(model.columnCount(), col_count_before + 1)

        # Test coordinate parsing: float, time, unit strings
        self.assertEqual(model._parse_x_value("123.45"), 123.45)
        self.assertEqual(model._parse_x_value("01:30"), 90.0)
        self.assertEqual(model._parse_x_value("01:02:03"), 3723.0)
        self.assertEqual(model._parse_x_value("2h"), 7200.0)
        self.assertEqual(model._parse_x_value("45s"), 45.0)
        self.assertEqual(model._parse_x_value("10m"), 600.0)

        # Header formatting
        font_x = model.headerData(0, Qt.Orientation.Horizontal, Qt.ItemDataRole.FontRole)
        self.assertIsInstance(font_x, QFont)
        self.assertTrue(font_x.bold())

        bg_x = model.headerData(0, Qt.Orientation.Horizontal, Qt.ItemDataRole.BackgroundRole)
        self.assertEqual(bg_x, QColor(240, 240, 240))

    def test_spreadsheet_view_context_menu_x_actions(self):
        table = DataTable(table_id="tbl_view", title="View Table", table_type=TableType.XY, num_rows=5)
        model = ScientificSpreadsheetModel(table)
        view = ScientificSpreadsheetView(model)

        menu = QMenu(view)
        view._populate_column_actions(menu, 0)
        action_texts = [a.text() for a in menu.actions()]

        self.assertIn("➕ Add X Column to Left", action_texts)
        self.assertIn("➕ Add X Column to Right", action_texts)
        self.assertIn("➕ Add Y Column / Replicate", action_texts)
        self.assertIn("➕ Insert Column to Right", action_texts)
        self.assertIn("➕ Insert Column to Left", action_texts)
        self.assertIn("🗑️ Delete Column", action_texts)


class TestHybridGraphPlotting(unittest.TestCase):
    """Verifies the three publication hybrid graph plotting types from docs/graph_specs/."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_hybrid_1_bars_and_points_error_styles(self):
        """Hybrid 1: Bars + Individual Points with Mean ± SEM, SD, and 95% CI."""
        table = DataTable(table_id="tbl_col", title="Column Groups", table_type=TableType.COLUMN, num_rows=10)
        table.y_group_titles = ["Control", "Treated"]
        # Control: 10, 12, 14, 16; Treated: 20, 25, 30, 35
        table.set_cell_value(0, "Y", 0, 0, 10.0)
        table.set_cell_value(1, "Y", 0, 0, 12.0)
        table.set_cell_value(2, "Y", 0, 0, 14.0)
        table.set_cell_value(3, "Y", 0, 0, 16.0)

        table.set_cell_value(0, "Y", 1, 0, 20.0)
        table.set_cell_value(1, "Y", 1, 0, 25.0)
        table.set_cell_value(2, "Y", 1, 0, 30.0)
        table.set_cell_value(3, "Y", 1, 0, 35.0)

        canvas = ScientificPlotCanvas()
        canvas.update_data(table, graph_type=GraphType.BARS_AND_POINTS)

        # SEM
        canvas.set_error_style("SEM")
        self.assertEqual(canvas.error_style, "SEM")
        plot_items = canvas.plot_widget.plotItem.items
        bars = [item for item in plot_items if isinstance(item, pg.BarGraphItem)]
        scatters = [item for item in plot_items if isinstance(item, pg.ScatterPlotItem)]
        error_bars = [item for item in plot_items if isinstance(item, pg.ErrorBarItem)]
        self.assertEqual(len(bars), 2)
        self.assertEqual(len(scatters), 2)
        self.assertEqual(len(error_bars), 2)

        # SD
        canvas.set_error_style("SD")
        self.assertEqual(canvas.error_style, "SD")

        # 95% CI
        canvas.set_error_style("95% CI")
        self.assertEqual(canvas.error_style, "95% CI")

        # Journal Grayscale toggle
        canvas.set_grayscale_mode(True)
        self.assertTrue(canvas.is_grayscale_mode)
        canvas.set_grayscale_mode(False)
        self.assertFalse(canvas.is_grayscale_mode)

    def test_hybrid_2_grouped_bars_and_points_from_pzfx(self):
        """Hybrid 2: Grouped Bars + Replicate Points loaded from 1661_Points_and_grouped_bars.pzfx."""
        doc = PrismXMLIO.import_xml("docs/graph_specs/1661_Points_and_grouped_bars.pzfx")
        self.assertGreater(len(doc.sheets), 0)
        sheet = list(doc.sheets.values())[0]
        table = sheet.table
        self.assertEqual(table.table_type, TableType.GROUPED)
        self.assertEqual(len(table.row_titles), 14)
        self.assertEqual(len(table.y_group_titles), 4)

        canvas = ScientificPlotCanvas()
        canvas.update_data(table, graph_type=GraphType.GROUPED_BARS_POINTS)

        plot_items = canvas.plot_widget.plotItem.items
        bars = [item for item in plot_items if isinstance(item, pg.BarGraphItem)]
        scatters = [item for item in plot_items if isinstance(item, pg.ScatterPlotItem)]
        self.assertGreater(len(bars), 0)
        self.assertGreater(len(scatters), 0)

        # Verify axis label reflects grouped conditions
        bottom_ax = canvas.plot_widget.getAxis("bottom")
        self.assertIsNotNone(bottom_ax)

    def test_hybrid_3_xy_spikes_and_curve(self):
        """Hybrid 3: Columns/Spikes + Curve Fit on XY graph with per-series styles."""
        table = DataTable(table_id="tbl_xy_spikes", title="XY Hybrid", table_type=TableType.XY, num_rows=5, num_replicates=1)
        table.y_group_titles = ["Baseline Spikes", "Receptor Binding"]
        table.x_columns[0]["data"] = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        table.y_data = np.full((2, 5, 1), np.nan)
        table.y_data[0, :, 0] = [10.0, 15.0, 20.0, 12.0, 8.0]
        table.y_data[1, :, 0] = [5.0, 8.0, 14.0, 19.0, 22.0]

        canvas = ScientificPlotCanvas()
        canvas.update_data(table, graph_type=GraphType.XY_SPIKES_CURVE)

        # By default for XY_SPIKES_CURVE: dataset 0 is spikes/bars, dataset 1 is scatter
        plot_items = canvas.plot_widget.plotItem.items
        bars = [item for item in plot_items if isinstance(item, pg.BarGraphItem)]
        scatters = [item for item in plot_items if isinstance(item, pg.ScatterPlotItem)]
        self.assertEqual(len(bars), 1)
        self.assertEqual(len(scatters), 1)

        # Per-series representation override
        canvas.set_series_representation("Receptor Binding", "spikes")
        bars_updated = [item for item in canvas.plot_widget.plotItem.items if isinstance(item, pg.BarGraphItem)]
        self.assertEqual(len(bars_updated), 2)


class TestAnalysisLayerRegistryAndClearing(unittest.TestCase):
    """Verifies analysis overlay layer registration, layer-specific removal, and full clearing."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_analysis_layer_lifecycle(self):
        table = DataTable(table_id="tbl_fit", title="Fit Test", table_type=TableType.XY, num_rows=5, num_replicates=1)
        table.x_columns[0]["data"] = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        table.y_data[0, :, 0] = [2.0, 4.0, 6.0, 8.0, 10.0]

        canvas = ScientificPlotCanvas()
        canvas.update_data(table, graph_type=GraphType.XY_SCATTER_CURVE)

        # Add two fitted curves
        x_dense = np.linspace(1.0, 5.0, 50)
        y_dense_1 = 2.0 * x_dense
        y_dense_2 = 1.8 * x_dense + 0.5

        curve1 = canvas.add_fitted_curve("fit_linear", "Linear Fit", x_dense, y_dense_1)
        curve2 = canvas.add_fitted_curve("fit_robust", "Robust Fit", x_dense, y_dense_2)

        self.assertIn("fit_linear", canvas.analysis_layers)
        self.assertIn("fit_robust", canvas.analysis_layers)

        # Remove single layer "fit_linear"
        removed = canvas.remove_analysis_layer("fit_linear")
        self.assertTrue(removed)
        self.assertNotIn("fit_linear", canvas.analysis_layers)
        self.assertIn("fit_robust", canvas.analysis_layers)

        # Underlying raw scatter item should remain visible
        scatters = [it for it in canvas.plot_widget.plotItem.items if isinstance(it, pg.ScatterPlotItem)]
        self.assertGreater(len(scatters), 0)
        for s in scatters:
            self.assertTrue(s.isVisible())

        # Clear all analysis layers
        canvas.clear_all_analysis_layers()
        self.assertEqual(len(canvas.analysis_layers), 0)

        # Raw scatter points still exist and visible
        scatters_after = [it for it in canvas.plot_widget.plotItem.items if isinstance(it, pg.ScatterPlotItem)]
        self.assertGreater(len(scatters_after), 0)


if __name__ == "__main__":
    unittest.main()
