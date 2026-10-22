"""
Unit and integration tests for GemGraph usability, plotting refinement, and scientific metadata.
Covers:
1. Spreadsheet cell clearing via Delete and Backspace keys
2. Configurable error whisker directionality ("Both (±)" vs "Upper (+) Only")
3. Journal grayscale monochromatic opaque patterns and high-contrast scatter symbology
4. Per-series custom color and opacity formatting via FormatSeriesDialog
"""

from __future__ import annotations
import unittest
import numpy as np
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QKeyEvent, QColor
from PyQt6.QtCore import Qt, QEvent, QItemSelectionModel, QItemSelection

from src.core.table import DataTable, TableType
from src.core.sheet import GraphType
from src.ui.spreadsheet import ScientificSpreadsheetModel, ScientificSpreadsheetView
from src.ui.graph_canvas import ScientificPlotCanvas
from src.ui.dialogs import FormatSeriesDialog
from src.ui.mediator import WorkspaceMediator
import pyqtgraph as pg

# Headless application
app = QApplication.instance() or QApplication([])


class TestSpreadsheetCellClearing(unittest.TestCase):
    """Verifies that pressing Delete or Backspace clears selected cells to NaN or empty."""

    def setUp(self) -> None:
        self.table = DataTable(table_id="tbl_clear", title="Clearing Test", table_type=TableType.XY, num_rows=5, num_replicates=1)
        self.table.x_columns[0]["data"] = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.table.set_cell_value(0, "Y", 0, 0, 10.0)
        self.table.set_cell_value(1, "Y", 0, 0, 20.0)
        self.table.set_cell_value(2, "Y", 0, 0, 30.0)

        self.model = ScientificSpreadsheetModel(self.table)
        self.view = ScientificSpreadsheetView(self.model)

    def test_delete_key_clears_selected_cells_to_nan(self) -> None:
        # Select row 0 and row 1 in column 1 (Y0:S0)
        sel_model = self.view.selectionModel()
        idx0 = self.model.index(0, 1)
        idx1 = self.model.index(1, 1)
        sel = QItemSelection(idx0, idx1)
        sel_model.select(sel, QItemSelectionModel.SelectionFlag.Select)

        self.assertEqual(self.table.get_cell_value(0, "Y", 0, 0), 10.0)
        self.assertEqual(self.table.get_cell_value(1, "Y", 0, 0), 20.0)

        # Trigger Delete key press
        key_event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Delete, Qt.KeyboardModifier.NoModifier)
        self.view.keyPressEvent(key_event)

        # Values should now be NaN
        self.assertTrue(np.isnan(self.table.get_cell_value(0, "Y", 0, 0)))
        self.assertTrue(np.isnan(self.table.get_cell_value(1, "Y", 0, 0)))
        # Row 2 was not selected, must remain 30.0
        self.assertEqual(self.table.get_cell_value(2, "Y", 0, 0), 30.0)

    def test_backspace_key_clears_selected_cells(self) -> None:
        # Select row 0 in column 0 (X column)
        sel_model = self.view.selectionModel()
        idx_x = self.model.index(0, 0)
        sel = QItemSelection(idx_x, idx_x)
        sel_model.select(sel, QItemSelectionModel.SelectionFlag.Select)

        self.assertEqual(self.table.get_cell_value(0, "X", 0, 0), 1.0)

        # Trigger Backspace key press
        key_event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Backspace, Qt.KeyboardModifier.NoModifier)
        self.view.keyPressEvent(key_event)

        self.assertTrue(np.isnan(self.table.get_cell_value(0, "X", 0, 0)))
        # Other rows in X must remain
        self.assertEqual(self.table.get_cell_value(1, "X", 0, 0), 2.0)

    def test_multiple_variables_table_clears_to_empty_string(self) -> None:
        from src.core.table import VariableType
        mv_table = DataTable(table_id="tbl_mv", title="MV Test", table_type=TableType.MULTIPLE_VARIABLES, num_rows=3)
        mv_table.add_variable(name="Treatment", var_type=VariableType.CATEGORICAL)
        mv_table.set_variable_value(0, 0, "Veh")
        mv_table.set_variable_value(1, 0, "DoseA")

        mv_model = ScientificSpreadsheetModel(mv_table)
        mv_view = ScientificSpreadsheetView(mv_model)

        sel_model = mv_view.selectionModel()
        idx = mv_model.index(0, 0)
        sel_model.select(QItemSelection(idx, idx), QItemSelectionModel.SelectionFlag.Select)

        key_event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Delete, Qt.KeyboardModifier.NoModifier)
        mv_view.keyPressEvent(key_event)

        self.assertEqual(mv_table.get_variable_value(0, 0), "")
        self.assertEqual(mv_table.get_variable_value(1, 0), "DoseA")


class TestWhiskerDirectionAndControls(unittest.TestCase):
    """Verifies configurable whisker direction ('Both (±)' vs 'Upper (+) Only')."""

    def setUp(self) -> None:
        self.table = DataTable(table_id="tbl_whiskers", title="Whisker Test", table_type=TableType.COLUMN, num_rows=4)
        self.table.y_group_titles = ["Group 1"]
        # Set replicate values: 10, 20, 30, 40 (Mean = 25)
        for r, v in enumerate([10.0, 20.0, 30.0, 40.0]):
            self.table.set_cell_value(r, "Y", 0, 0, v)

        self.canvas = ScientificPlotCanvas()
        self.canvas.update_data(self.table, graph_type=GraphType.BARS_AND_POINTS)

    def test_whisker_direction_defaults_to_both(self) -> None:
        self.assertEqual(self.canvas.whisker_direction, "Both (±)")
        error_items = [it for it in self.canvas.plot_widget.plotItem.items if isinstance(it, pg.ErrorBarItem)]
        self.assertEqual(len(error_items), 1)
        eb = error_items[0]
        # In 'Both (±)', top and bottom error lengths are equal and > 0
        self.assertGreater(eb.opts["top"][0], 0)
        self.assertGreater(eb.opts["bottom"][0], 0)
        self.assertAlmostEqual(eb.opts["top"][0], eb.opts["bottom"][0], places=5)

    def test_whisker_direction_upper_only(self) -> None:
        self.canvas.set_whisker_direction("Upper (+) Only")
        self.assertEqual(self.canvas.whisker_direction, "Upper (+) Only")

        error_items = [it for it in self.canvas.plot_widget.plotItem.items if isinstance(it, pg.ErrorBarItem)]
        self.assertEqual(len(error_items), 1)
        eb = error_items[0]
        # In 'Upper (+) Only', top > 0 and bottom == 0
        self.assertGreater(eb.opts["top"][0], 0)
        self.assertEqual(eb.opts["bottom"][0], 0.0)

    def test_switch_back_to_both(self) -> None:
        self.canvas.set_whisker_direction("Upper (+) Only")
        self.canvas.set_whisker_direction("Both (±)")
        self.assertEqual(self.canvas.whisker_direction, "Both (±)")

        error_items = [it for it in self.canvas.plot_widget.plotItem.items if isinstance(it, pg.ErrorBarItem)]
        self.assertEqual(len(error_items), 1)
        eb = error_items[0]
        self.assertAlmostEqual(eb.opts["top"][0], eb.opts["bottom"][0], places=5)


class TestJournalGrayscalePaletteAndFills(unittest.TestCase):
    """Verifies that Journal Grayscale mode applies publication-grade opaque fills and contrast."""

    def setUp(self) -> None:
        self.table = DataTable(table_id="tbl_gray", title="Grayscale Test", table_type=TableType.COLUMN, num_rows=3)
        # Add 3 more groups so we have 5 groups total
        self.table.add_y_group("Series 3")
        self.table.add_y_group("Series 4")
        self.table.add_y_group("Series 5")
        self.table.y_group_titles = ["Series 1", "Series 2", "Series 3", "Series 4", "Series 5"]
        for g_idx in range(5):
            self.table.set_cell_value(0, "Y", g_idx, 0, 10.0 + g_idx * 5.0)
            self.table.set_cell_value(1, "Y", g_idx, 0, 12.0 + g_idx * 5.0)

        self.canvas = ScientificPlotCanvas()
        self.canvas.set_grayscale_mode(True)
        self.canvas.update_data(self.table, graph_type=GraphType.BARS_AND_POINTS)

    def test_journal_grayscale_mode_active(self) -> None:
        self.assertTrue(self.canvas.is_grayscale_mode)

    def test_bars_are_opaque_with_distinct_patterns(self) -> None:
        plot_items = self.canvas.plot_widget.plotItem.items
        bars = [it for it in plot_items if isinstance(it, pg.BarGraphItem)]
        self.assertEqual(len(bars), 5)

        # Series 1 (idx 0): Solid Black
        brush0 = bars[0].opts["brush"]
        self.assertEqual(brush0.color().name(), "#000000")
        self.assertEqual(brush0.color().alpha(), 255)

        # Series 2 (idx 1): Pure White with Black border
        brush1 = bars[1].opts["brush"]
        self.assertEqual(brush1.color().name(), "#ffffff")
        self.assertEqual(brush1.color().alpha(), 255)
        pen1 = bars[1].opts["pen"]
        self.assertEqual(pen1.color().name(), "#000000")

        # Series 3 (idx 2): Medium Gray
        brush2 = bars[2].opts["brush"]
        self.assertEqual(brush2.color().name(), "#808080")
        self.assertEqual(brush2.color().alpha(), 255)

        # Series 4 (idx 3): Light Gray
        brush3 = bars[3].opts["brush"]
        self.assertEqual(brush3.color().name(), "#c0c0c0")
        self.assertEqual(brush3.color().alpha(), 255)

        # Series 5 (idx 4): Diagonal Cross-Hatch Pattern
        brush4 = bars[4].opts["brush"]
        self.assertEqual(brush4.style(), Qt.BrushStyle.DiagCrossPattern)

    def test_scatter_points_over_bars_have_high_contrast(self) -> None:
        plot_items = self.canvas.plot_widget.plotItem.items
        scatters = [it for it in plot_items if isinstance(it, pg.ScatterPlotItem)]
        self.assertEqual(len(scatters), 5)

        # In grayscale mode, scatter points over dark bars (0, 2, 4) should be white fill with black outline
        # Over white/light bars (1, 3), points are black fill
        for idx, sc in enumerate(scatters):
            pen = sc.opts.get("pen")
            brush = sc.opts.get("brush")
            brush_color = pg.mkBrush(brush).color()
            pen_color = pg.mkPen(pen).color()
            self.assertEqual(pen_color.name(), "#000000")
            if idx in (0, 2, 4):
                self.assertEqual(brush_color.name(), "#ffffff")
            else:
                self.assertEqual(brush_color.name(), "#000000")


class TestFormatSeriesDialog(unittest.TestCase):
    """Verifies FormatSeriesDialog interactions and custom color/opacity overrides."""

    def test_dialog_initialization_and_formatting(self) -> None:
        series_names = ["Control", "Treated A", "Treated B"]
        dialog = FormatSeriesDialog(series_names, is_grayscale=False)
        self.assertEqual(dialog.series_combo.count(), 3)
        self.assertEqual(dialog.opacity_slider.value(), 80)

        # Modify opacity to 50%
        dialog.opacity_slider.setValue(50)
        self.assertEqual(dialog.opacities["Control"], 0.5)

        # Set custom color
        custom_red = QColor("#e74c3c")
        dialog.colors["Control"] = custom_red
        colors, opacities = dialog.get_formatting()
        self.assertEqual(colors["Control"].name(), "#e74c3c")
        self.assertEqual(opacities["Control"], 0.5)

    def test_canvas_custom_series_formatting(self) -> None:
        table = DataTable(table_id="tbl_custom_fmt", title="Custom Format", table_type=TableType.COLUMN, num_rows=3)
        table.y_group_titles = ["Sample A", "Sample B"]
        table.set_cell_value(0, "Y", 0, 0, 10.0)
        table.set_cell_value(1, "Y", 0, 0, 15.0)
        table.set_cell_value(0, "Y", 1, 0, 20.0)
        table.set_cell_value(1, "Y", 1, 0, 25.0)

        canvas = ScientificPlotCanvas()
        canvas.custom_series_colors["Sample A"] = QColor("#ff0000")
        canvas.custom_series_opacities["Sample A"] = 0.60
        canvas.update_data(table, graph_type=GraphType.BARS_AND_POINTS)

        plot_items = canvas.plot_widget.plotItem.items
        bars = [it for it in plot_items if isinstance(it, pg.BarGraphItem)]
        self.assertEqual(len(bars), 2)

        # Bar 0 should have the custom red color with 60% opacity (alpha ~ 153)
        brush_a = bars[0].opts["brush"]
        self.assertEqual(brush_a.color().red(), 255)
        self.assertEqual(brush_a.color().green(), 0)
        self.assertEqual(brush_a.color().blue(), 0)
        self.assertAlmostEqual(brush_a.color().alpha(), int(255 * 0.60), delta=2)


if __name__ == "__main__":
    unittest.main()
