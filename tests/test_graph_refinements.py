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
from unittest.mock import MagicMock
import numpy as np
from PyQt6.QtWidgets import QApplication, QAbstractItemView
from PyQt6.QtGui import QKeyEvent, QColor
from PyQt6.QtCore import Qt, QEvent, QItemSelectionModel, QItemSelection

from src.core.table import DataTable, TableType
from src.core.sheet import GraphType, GraphSheet, SheetType
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


class TestDepthGraphParityRefinements(unittest.TestCase):
    """Verifies reverse-engineered Depth Graph parity without 2.5D projection math."""

    def test_depth_graph_rendering_and_whisker_modes(self) -> None:
        table = DataTable(table_id="tbl_depth", title="Depth Test", table_type=TableType.XY, num_rows=4, num_replicates=2)
        table.x_columns[0]["data"] = np.array([0.0, 10.0, 20.0, 30.0])
        for r in range(4):
            table.set_cell_value(r, "Y", 0, 0, 50.0 - r * 5.0)
            table.set_cell_value(r, "Y", 0, 1, 52.0 - r * 5.0)

        canvas = ScientificPlotCanvas()
        canvas.set_error_style("SD")
        canvas.set_whisker_direction("Both (±)")
        canvas.update_data(table, graph_type=GraphType.DEPTH_GRAPH)

        # Inverted vertical axis
        self.assertTrue(canvas.plot_widget.getViewBox().yInverted())

        # Contains horizontal error whiskers and scatter points
        items = canvas.plot_widget.plotItem.items
        err_items = [it for it in items if isinstance(it, pg.ErrorBarItem)]
        scatter_items = [it for it in items if isinstance(it, pg.ScatterPlotItem)]
        self.assertGreater(len(err_items), 0)
        self.assertGreater(len(scatter_items), 0)

        # Toggle whisker direction
        canvas.set_whisker_direction("Upper (+) Only")
        self.assertEqual(canvas.whisker_direction, "Upper (+) Only")

        # Column table depth profile
        col_table = DataTable(
            table_id="tbl_col_d",
            title="Col Depth",
            table_type=TableType.COLUMN,
            num_rows=3,
            num_replicates=1,
            y_group_titles=["0-5m", "5-10m", "10-20m"],
        )
        col_table.set_cell_value(0, "Y", 0, 0, 12.0)
        col_table.set_cell_value(0, "Y", 1, 0, 8.5)
        col_table.set_cell_value(0, "Y", 2, 0, 4.2)
        canvas.update_data(col_table, graph_type=GraphType.DEPTH_GRAPH)
        self.assertTrue(canvas.plot_widget.getViewBox().yInverted())

    def test_depth_graph_vertical_axis_dynamic_binding(self) -> None:
        """Verifies that Depth Graph dynamically binds its vertical axis and eliminates hardcoded 'Depth in meters'."""
        canvas = ScientificPlotCanvas()

        # 1. XY table: vertical axis reflects Y column titles, no "Depth in meters"
        table_xy = DataTable(table_id="tbl_xy", title="Core Depth", table_type=TableType.XY, num_rows=3, y_group_titles=["Oxygen (mg/L)", "Phosphate (uM)"])
        canvas.update_data(table_xy, graph_type=GraphType.DEPTH_GRAPH)
        lbl_xy = canvas.plot_widget.getAxis("left").labelText
        self.assertIn("Oxygen (mg/L)", lbl_xy)
        self.assertIn("Phosphate (uM)", lbl_xy)
        self.assertNotIn("Depth in meters", lbl_xy)

        # 2. Column table: vertical axis reflects table title / factor title, no "Depth in meters"
        col_table = DataTable(table_id="tbl_col", title="Substrate Strata", table_type=TableType.COLUMN, num_rows=3, y_group_titles=["Topsoil", "Subsoil", "Bedrock"])
        canvas.update_data(col_table, graph_type=GraphType.DEPTH_GRAPH)
        lbl_col = canvas.plot_widget.getAxis("left").labelText
        self.assertEqual(lbl_col, "Substrate Strata")
        self.assertNotIn("Depth in meters", lbl_col)


class TestFrozenGraphIsolationAndAxisSpacing(unittest.TestCase):
    """
    Regression unit tests verifying:
    1. Frozen snapshot tables initialize with NoEditTriggers.
    2. Frozen graphs do not invoke refresh_plot upon upstream data changes.
    3. X-Axis label vertical spacing margin for exports.
    """

    def test_frozen_snapshot_table_initializes_with_no_edit_triggers(self) -> None:
        """Verifies that frozen snapshot tables are read-only with NoEditTriggers."""
        from src.core.sheet import GraphSheet

        table = DataTable(table_id="snap_1", title="Snapshot Table", table_type=TableType.XY, num_rows=3, num_replicates=1)
        table.set_cell_value(0, "Y", 0, 0, 100.0)

        # 1. Direct model and view with is_frozen=True
        model = ScientificSpreadsheetModel(table, is_frozen=True)
        view = ScientificSpreadsheetView(model)

        self.assertEqual(view.editTriggers(), QAbstractItemView.EditTrigger.NoEditTriggers)
        self.assertEqual(view.table_view.editTriggers(), QAbstractItemView.EditTrigger.NoEditTriggers)
        self.assertEqual(model.flags(model.index(0, 0)) & Qt.ItemFlag.ItemIsEditable, Qt.ItemFlag.NoItemFlags)
        self.assertFalse(model.setData(model.index(0, 0), 200.0))

        # 2. Editable table has standard triggers
        editable_model = ScientificSpreadsheetModel(table, is_frozen=False)
        editable_view = ScientificSpreadsheetView(editable_model)
        self.assertNotEqual(editable_view.editTriggers(), QAbstractItemView.EditTrigger.NoEditTriggers)
        self.assertTrue(bool(editable_model.flags(editable_model.index(0, 0)) & Qt.ItemFlag.ItemIsEditable))

        # 3. Loading a frozen GraphSheet via load_sheet sets NoEditTriggers
        frozen_graph = GraphSheet(sheet_id="g_froz", title="Frozen Graph", source_sheet_id="src_1", is_frozen=True)
        frozen_graph.snapshot_table = table
        editable_view.load_sheet(frozen_graph)
        self.assertEqual(editable_view.editTriggers(), QAbstractItemView.EditTrigger.NoEditTriggers)

        # 4. Column mutations and key deletion blocked on frozen model and view
        self.assertFalse(model.rename_column(0, "Mutated Title"))
        self.assertFalse(model.insert_column(0))
        self.assertFalse(model.insert_x_column(0))
        self.assertFalse(model.delete_column(0))

        # Test keyPressEvent on frozen table does not clear cell
        idx = view.model().index(0, 0)
        view.setCurrentIndex(idx)
        del_event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Delete, Qt.KeyboardModifier.NoModifier)
        view.keyPressEvent(del_event)
        self.assertEqual(table.get_cell_value(0, "Y", 0, 0), 100.0)

    def test_frozen_graphs_do_not_invoke_refresh_plot_upon_upstream_data_changes(self) -> None:
        """Verifies that upstream table mutations never trigger refresh_plot on frozen graphs."""
        from src.ui.main_window import MainWindow

        win = MainWindow()
        win.resize(1024, 768)

        # Data Sheet & Graph
        data_sheet = win.document.get_sheets_by_type(SheetType.DATA)[0]
        data_sheet.table.set_cell_value(0, "Y", 0, 0, 10.0)

        graph_sheet = win.document.create_graph_sheet("Test Graph", source_sheet_id=data_sheet.sheet_id)
        graph_sheet.snapshot_table = data_sheet.table.copy()
        graph_sheet.is_frozen = True

        # Select frozen graph
        win.on_sheet_selected(graph_sheet.sheet_id)
        app.processEvents()

        # Mock refresh_plot and update_data on graph canvas
        win.graph_canvas.refresh_plot = MagicMock()
        win.graph_canvas.update_data = MagicMock()

        # Modify upstream table data and notify mediator
        data_sheet.table.set_cell_value(0, "Y", 0, 0, 999.0)
        win.mediator.notify_data_changed(data_sheet.table)
        app.processEvents()

        # Severed upstream link must prevent refresh_plot and update_data
        win.graph_canvas.refresh_plot.assert_not_called()
        win.graph_canvas.update_data.assert_not_called()

        win.close()

    def test_x_axis_label_vertical_spacing_margin_for_exports(self) -> None:
        """Verifies that set_axis_titles and bottom axis style inject export margins to prevent tick overlap."""
        canvas = ScientificPlotCanvas()

        bottom_axis = canvas.plot_widget.getAxis("bottom")
        canvas.set_axis_titles(x_title="Concentration (uM)", y_title="Fluorescence")
        self.assertGreaterEqual(max(bottom_axis.height(), bottom_axis.fixedHeight or 0), 60)
        self.assertEqual(bottom_axis.labelStyle.get("margin-top"), "15px")
        self.assertFalse(bottom_axis.labelText.startswith("\n"))

        # Idempotent: repeated calls must not compound height
        h = bottom_axis.fixedHeight
        canvas.set_axis_titles(x_title="Concentration (uM)")
        self.assertEqual(bottom_axis.fixedHeight, h)

        y_label = canvas.plot_widget.getAxis("left").labelText
        self.assertEqual(y_label, "Fluorescence")


if __name__ == "__main__":
    unittest.main()
