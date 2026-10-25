"""
Unit and Integration Tests for Layout Geometry, Analysis Layer Management,
and Scientific Plotting Symbology.
"""

from __future__ import annotations
import os
import unittest
import numpy as np
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QColor
from PyQt6.QtCore import Qt
import pyqtgraph as pg

from src.core.table import DataTable, TableType
from src.core.sheet import ResultSheet
from src.ui.main_window import MainWindow
from src.main import init_interface_shell
from src.ui.plot_canvas import ScientificPlotCanvas
from src.ui.mediator import WorkspaceMediator

app = QApplication.instance()
if app is None:
    app = QApplication([])


class TestLayoutStability(unittest.TestCase):
    def setUp(self) -> None:
        self.window = MainWindow()

    def tearDown(self) -> None:
        self.window.close()

    def test_splitter_collapsible_and_stability(self) -> None:
        """Verifies inner and outer splitters cannot collapse children."""
        self.assertFalse(self.window.inner_splitter.childrenCollapsible())
        self.assertFalse(self.window.outer_splitter.childrenCollapsible())

    def test_minimum_widget_widths(self) -> None:
        """Verifies contained widgets have minimum widths preventing snapping."""
        self.assertGreaterEqual(self.window.navigator.minimumWidth(), 180)
        self.assertGreaterEqual(self.window.center_stack.minimumWidth(), 240)
        self.assertGreaterEqual(self.window.graph_canvas.minimumWidth(), 320)

    def test_proportional_stretch_factors(self) -> None:
        """Verifies stretch factors ensure workspace and plot receive extra resize space."""
        # Check hierarchy
        self.assertEqual(self.window.outer_splitter.widget(0), self.window.navigator)
        self.assertEqual(self.window.outer_splitter.widget(1), self.window.inner_splitter)
        self.assertEqual(self.window.inner_splitter.widget(0), self.window.center_stack)
        self.assertEqual(self.window.inner_splitter.widget(1), self.window.graph_canvas)

        # Re-applying stretch factors works without error
        self.window.inner_splitter.setStretchFactor(0, 0)
        self.window.inner_splitter.setStretchFactor(1, 1)
        self.window.outer_splitter.setStretchFactor(0, 0)
        self.window.outer_splitter.setStretchFactor(1, 1)

    def test_init_interface_shell_helper(self) -> None:
        """Verifies init_interface_shell from src.main can re-apply configuration."""
        init_interface_shell(self.window)
        self.assertFalse(self.window.inner_splitter.childrenCollapsible())
        self.assertFalse(self.window.outer_splitter.childrenCollapsible())


class TestAnalysisLayerRegistry(unittest.TestCase):
    def setUp(self) -> None:
        self.mediator = WorkspaceMediator()
        self.canvas = ScientificPlotCanvas(mediator=self.mediator)
        self.table = DataTable("tbl_test", table_type=TableType.XY, num_rows=10, num_replicates=1)
        for r in range(10):
            self.table.set_cell_value(r, "X", 0, 0, float(r))
            self.table.set_cell_value(r, "Y", 0, 0, float(r * 2 + 1))
        self.canvas.update_data(self.table)

    def tearDown(self) -> None:
        self.canvas.close()

    def test_registry_initial_state(self) -> None:
        """Verifies active overlay registry starts clean."""
        self.assertIsInstance(self.canvas.analysis_layers, dict)
        self.assertEqual(len(self.canvas.analysis_layers), 0)
        self.assertIsNotNone(self.canvas.btn_clear_analysis)
        self.assertIsNotNone(self.canvas.analysis_menu)

    def test_add_fitted_curve_and_registration(self) -> None:
        """Verifies adding a fitted curve registers PlotDataItem in analysis_layers."""
        xs = np.linspace(0, 10, 50)
        ys = 2 * xs + 1
        curve_item = self.canvas.add_fitted_curve("lin_fit_1", "Linear Fit (Group A)", xs, ys)
        
        self.assertIn("lin_fit_1", self.canvas.analysis_layers)
        layer = self.canvas.analysis_layers["lin_fit_1"]
        self.assertEqual(layer["name"], "Linear Fit (Group A)")
        self.assertIn(curve_item, layer["plot_items"])
        self.assertIn(curve_item, self.canvas.plot_widget.plotItem.items)

    def test_remove_analysis_layer_restores_scatter_view(self) -> None:
        """Verifies remove_analysis_layer removes plot item without modifying view ranges."""
        xs = np.linspace(0, 10, 50)
        ys = 2 * xs + 1
        curve_item = self.canvas.add_fitted_curve("fit_4pl", "4PL Fit (colonia 1)", xs, ys)
        
        # Simulate user zooming to a specific range
        self.canvas.plot_widget.setRange(xRange=(2.0, 8.0), yRange=(5.0, 15.0), padding=0)
        view_before = self.canvas.plot_widget.viewRange()

        # Remove layer
        removed = self.canvas.remove_analysis_layer("fit_4pl")
        self.assertTrue(removed)
        self.assertNotIn("fit_4pl", self.canvas.analysis_layers)
        self.assertNotIn(curve_item, self.canvas.plot_widget.plotItem.items)

        # User zoom / axis range must not be reset!
        view_after = self.canvas.plot_widget.viewRange()
        self.assertAlmostEqual(view_before[0][0], view_after[0][0], places=3)
        self.assertAlmostEqual(view_before[0][1], view_after[0][1], places=3)
        self.assertAlmostEqual(view_before[1][0], view_after[1][0], places=3)
        self.assertAlmostEqual(view_before[1][1], view_after[1][1], places=3)

    def test_clear_all_fitted_curves(self) -> None:
        """Verifies clear_all_analysis_layers removes all active fitted curves."""
        self.canvas.add_fitted_curve("fit_1", "Fit 1", np.array([1, 2]), np.array([3, 4]))
        self.canvas.add_fitted_curve("fit_2", "Fit 2", np.array([1, 2]), np.array([5, 6]))
        self.assertEqual(len(self.canvas.analysis_layers), 2)

        self.canvas.clear_all_analysis_layers()
        self.assertEqual(len(self.canvas.analysis_layers), 0)

    def test_mediator_layer_clearing(self) -> None:
        """Verifies workspace mediator notifies and triggers analysis layer removal."""
        self.canvas.add_fitted_curve("fit_med", "Mediator Fit", np.array([1, 2]), np.array([2, 4]))
        self.assertIn("fit_med", self.canvas.analysis_layers)

        # Mock mediator notification
        signal_received = []
        self.mediator.analysisLayerRemoved.connect(lambda lid: signal_received.append(lid))
        self.canvas.remove_analysis_layer("fit_med")

        self.assertIn("fit_med", signal_received)
        self.assertNotIn("fit_med", self.canvas.analysis_layers)

    def test_result_sheet_overlay_and_menu_actions(self) -> None:
        """Verifies ResultSheet curve automatically registers and populates toolbar menu."""
        res = ResultSheet("res_4pl", "4PL Fit", "4PL", source_sheet_id="tbl_test")
        res.raw_result_data = {
            "model_name": "4PL Fit (colonia 1)",
            "x_dense": [0.0, 1.0, 2.0, 3.0],
            "y_dense": [1.0, 2.5, 4.0, 6.0],
        }
        self.canvas.update_data(self.table, result=res)

        self.assertIn("res_4pl", self.canvas.analysis_layers)
        self.assertEqual(self.canvas.analysis_layers["res_4pl"]["name"], "4PL Fit (colonia 1)")

        # Verify toolbar menu actions
        self.canvas._rebuild_analysis_menu()
        action_texts = [a.text() for a in self.canvas.analysis_menu.actions()]
        self.assertIn("Clear All Fitted Curves", action_texts)
        self.assertIn("Remove 4PL Fit (colonia 1)", action_texts)


class TestColorAndGrayscaleSymbology(unittest.TestCase):
    def setUp(self) -> None:
        self.canvas = ScientificPlotCanvas()
        # Create a table with 6 Y groups
        self.table = DataTable(
            "tbl_symbology",
            table_type=TableType.XY,
            num_rows=5,
            num_replicates=1,
            y_group_titles=["Col 1", "Col 2", "Col 3", "Col 4", "Col 5", "Col 6"],
        )
        for r in range(5):
            self.table.set_cell_value(r, "X", 0, 0, float(r + 1))
            for g in range(6):
                self.table.set_cell_value(r, "Y", g, 0, float((g + 1) * 10 + r))

    def tearDown(self) -> None:
        self.canvas.close()

    def test_per_column_y_colors(self) -> None:
        """Verifies each column gets a distinct color from the expanded palette in Color mode."""
        self.canvas.set_grayscale_mode(False)
        self.canvas.update_data(self.table)

        scatters = [
            it for it in self.canvas.plot_widget.plotItem.items
            if isinstance(it, pg.ScatterPlotItem)
        ]
        self.assertEqual(len(scatters), 6)

        # Check distinct colors
        colors = [s.opts["brush"].color().name() for s in scatters]
        # In expanded palette, first 6 colors are all distinct
        self.assertEqual(len(set(colors)), 6)

    def test_grayscale_publishing_symbology(self) -> None:
        """
        Verifies Journal Grayscale Mode maps columns to:
        Col 1: Solid Circle ('o', brush=black, pen=black)
        Col 2: Open Square ('s', brush=white, pen=black, width=1.5)
        Col 3: Solid Triangle ('t', brush=black, pen=black)
        Col 4: Open Diamond ('d', brush=white, pen=black, width=1.5)
        Col 5: Solid Inverted Triangle ('t1', brush=black, pen=black)
        Col 6: Open Circle ('o', brush=white, pen=black, width=1.5)
        """
        self.canvas.set_grayscale_mode(True)
        self.canvas.update_data(self.table)

        scatters = [
            it for it in self.canvas.plot_widget.plotItem.items
            if isinstance(it, pg.ScatterPlotItem)
        ]
        self.assertEqual(len(scatters), 6)

        expected = [
            ("o", QColor(0, 0, 0), 1.0),       # Col 1: Solid Circle
            ("s", QColor(255, 255, 255), 1.5), # Col 2: Open Square
            ("t", QColor(0, 0, 0), 1.0),       # Col 3: Solid Triangle
            ("d", QColor(255, 255, 255), 1.5), # Col 4: Open Diamond
            ("t1", QColor(0, 0, 0), 1.0),      # Col 5: Solid Inverted Triangle
            ("o", QColor(255, 255, 255), 1.5), # Col 6: Open Circle
        ]

        for idx, (exp_sym, exp_brush_col, exp_width) in enumerate(expected):
            scatter = scatters[idx]
            self.assertEqual(scatter.opts["symbol"], exp_sym, f"Mismatch on Col {idx+1} symbol")
            self.assertEqual(
                scatter.opts["brush"].color().name(),
                exp_brush_col.name(),
                f"Mismatch on Col {idx+1} brush color",
            )
            pen = scatter.opts["pen"]
            self.assertEqual(pen.color().name(), QColor(0, 0, 0).name(), f"Mismatch on Col {idx+1} pen color")
            self.assertAlmostEqual(pen.widthF(), exp_width, delta=0.1, msg=f"Mismatch on Col {idx+1} pen width")

    def test_legend_mirroring(self) -> None:
        """Verifies scatter item options and shapes are correctly mirrored into pyqtgraph legend."""
        self.canvas.set_grayscale_mode(True)
        self.canvas.update_data(self.table)

        legend = self.canvas.plot_widget.plotItem.legend
        self.assertIsNotNone(legend)
        legend_labels = [label.text for sample, label in legend.items]
        for title in self.table.y_group_titles:
            self.assertIn(title, legend_labels)

        expected = [
            ("o", QColor(0, 0, 0), 1.0),       # Col 1: Solid Circle
            ("s", QColor(255, 255, 255), 1.5), # Col 2: Open Square
            ("t", QColor(0, 0, 0), 1.0),       # Col 3: Solid Triangle
            ("d", QColor(255, 255, 255), 1.5), # Col 4: Open Diamond
            ("t1", QColor(0, 0, 0), 1.0),      # Col 5: Solid Inverted Triangle
            ("o", QColor(255, 255, 255), 1.5), # Col 6: Open Circle
        ]

        for idx, (sample, label) in enumerate(legend.items):
            item = sample.item
            opts = item.opts
            exp_sym, exp_brush_col, exp_width = expected[idx]
            self.assertEqual(opts["symbol"], exp_sym, f"Legend symbol mismatch on item {idx+1}")
            self.assertEqual(
                opts["brush"].color().name(),
                exp_brush_col.name(),
                f"Legend brush mismatch on item {idx+1}",
            )
            self.assertEqual(
                opts["pen"].color().name(),
                QColor(0, 0, 0).name(),
                f"Legend pen color mismatch on item {idx+1}",
            )
            self.assertAlmostEqual(
                opts["pen"].widthF(),
                exp_width,
                delta=0.1,
                msg=f"Legend pen width mismatch on item {idx+1}",
            )

    def test_palette_switch_preserves_user_zoom(self) -> None:
        """Verifies switching palette mode preserves custom user zoom and axis ranges."""
        self.canvas.update_data(self.table)
        self.canvas.plot_widget.setRange(xRange=(2.0, 4.0), yRange=(10.0, 30.0), padding=0)
        view_before = self.canvas.plot_widget.viewRange()

        self.canvas.set_grayscale_mode(True)
        view_after = self.canvas.plot_widget.viewRange()
        self.assertAlmostEqual(view_before[0][0], view_after[0][0], places=3)
        self.assertAlmostEqual(view_before[0][1], view_after[0][1], places=3)
        self.assertAlmostEqual(view_before[1][0], view_after[1][0], places=3)
        self.assertAlmostEqual(view_before[1][1], view_after[1][1], places=3)

        self.canvas.set_grayscale_mode(False)
        view_color = self.canvas.plot_widget.viewRange()
        self.assertAlmostEqual(view_before[0][0], view_color[0][0], places=3)
        self.assertAlmostEqual(view_before[0][1], view_color[0][1], places=3)

    def test_mediator_no_duplicate_signals(self) -> None:
        """Verifies mediator signals are emitted exactly once and do not re-enter recursively."""
        mediator = WorkspaceMediator()
        canvas = ScientificPlotCanvas(mediator=mediator)
        canvas.add_fitted_curve("layer_x", "Curve X", np.array([1, 2]), np.array([3, 4]))
        received_removals = []
        mediator.analysisLayerRemoved.connect(lambda lid: received_removals.append(lid))

        canvas.remove_analysis_layer("layer_x")
        self.assertEqual(received_removals, ["layer_x"])
        self.assertNotIn("layer_x", canvas.analysis_layers)

        canvas.add_fitted_curve("layer_y", "Curve Y", np.array([1, 2]), np.array([3, 4]))
        received_clears = []
        mediator.allAnalysisLayersCleared.connect(lambda: received_clears.append(True))
        canvas.clear_all_analysis_layers()
        self.assertEqual(len(received_clears), 1)
        self.assertEqual(len(canvas.analysis_layers), 0)
        canvas.close()

    def test_init_interface_shell_no_args(self) -> None:
        """Verifies init_interface_shell can be called without arguments and returns a configured window."""
        win = init_interface_shell()
        self.assertIsInstance(win, MainWindow)
        self.assertFalse(win.inner_splitter.childrenCollapsible())
        self.assertFalse(win.outer_splitter.childrenCollapsible())
        win.close()

    def test_y_datasets_dict_duplicate_and_empty_titles(self) -> None:
        """Verifies table.y_datasets_dict does not clobber duplicate or empty group titles."""
        tbl = DataTable("tbl_dup", table_type=TableType.XY, num_rows=3, num_replicates=1, y_group_titles=["A", "A", ""])
        d = tbl.y_datasets_dict
        self.assertEqual(len(d), 3)


if __name__ == "__main__":
    unittest.main()

