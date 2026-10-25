"""
Comprehensive Unit Tests for:
1. Spreadsheet Del / Backspace undoability (BatchClearCellsCommand & QUndoStack).
2. Focus-Aware Global Undo/Redo Routing (Text edit vs Spreadsheet vs Document).
3. Per-sheet graph state persistence across navigation.
4. Office/PDF export isolation (live canvas state restoration).
5. Pinned & Frozen Graphs architecture (GraphSheet in doc.graphs, freeze/unfreeze snapshot).
6. Project Navigator context menus with exact action labels and "➕ Add Info Sheet".
"""

from __future__ import annotations
import os
import tempfile
import unittest
import numpy as np

from PyQt6.QtCore import Qt, QPoint, QItemSelection
from PyQt6.QtGui import QKeyEvent, QTextCursor
from PyQt6.QtWidgets import QApplication, QTextEdit, QTreeWidgetItem

from src.core.sheet import DataSheet, GraphSheet, SheetType, GraphType, format_pinned_graph_name
from src.core.table import TableType, DataTable
from src.core.office_exporter import render_graph_to_png_bytes
from src.io.prism_io import PrismIO
from src.ui.main_window import MainWindow
from src.ui.spreadsheet import ScientificSpreadsheetView, ScientificSpreadsheetModel, BatchClearCellsCommand
from src.ui.graph_canvas import GraphCanvasWidget

app = QApplication.instance() or QApplication([])


class TestSpreadsheetUndo(unittest.TestCase):
    """Tests BatchClearCellsCommand and Del/Backspace undoability."""

    def setUp(self) -> None:
        self.table = DataTable(table_id="tbl_test", title="Test Table", table_type=TableType.COLUMN, num_rows=5, num_replicates=2)
        self.table.set_cell_value(0, "Y", 0, 0, 10.5)
        self.table.set_cell_value(0, "Y", 0, 1, 20.5)
        self.table.set_cell_value(1, "Y", 0, 0, 30.5)
        self.table.set_cell_value(1, "Y", 0, 1, 40.5)

        self.model = ScientificSpreadsheetModel(self.table)
        self.view = ScientificSpreadsheetView(self.model)

    def test_batch_clear_command_direct(self) -> None:
        """BatchClearCellsCommand clears cells and restores them on undo."""
        old_val_00 = self.table.get_cell_value(0, "Y", 0, 0)
        old_val_01 = self.table.get_cell_value(0, "Y", 0, 1)

        cmd = BatchClearCellsCommand(
            self.model,
            [(0, 0, old_val_00), (0, 1, old_val_01)],
            description="Clear 2 cells"
        )
        self.view.undo_stack.push(cmd)

        self.assertTrue(np.isnan(self.table.get_cell_value(0, "Y", 0, 0)))
        self.assertTrue(np.isnan(self.table.get_cell_value(0, "Y", 0, 1)))
        self.assertEqual(self.table.get_cell_value(1, "Y", 0, 0), 30.5)

        # Undo
        self.view.undo_stack.undo()
        self.assertEqual(self.table.get_cell_value(0, "Y", 0, 0), 10.5)
        self.assertEqual(self.table.get_cell_value(0, "Y", 0, 1), 20.5)

        # Redo
        self.view.undo_stack.redo()
        self.assertTrue(np.isnan(self.table.get_cell_value(0, "Y", 0, 0)))
        self.assertTrue(np.isnan(self.table.get_cell_value(0, "Y", 0, 1)))

    def test_key_press_delete_triggers_batch_clear_undoable(self) -> None:
        """Pressing Delete key on selected cells pushes an undoable command."""
        tl = self.model.index(0, 0)
        br = self.model.index(0, 1)
        sel = QItemSelection(tl, br)
        self.view.selectionModel().select(sel, self.view.selectionModel().SelectionFlag.Select)

        initial_undo_count = self.view.undo_stack.count()

        event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Delete, Qt.KeyboardModifier.NoModifier)
        self.view.keyPressEvent(event)

        self.assertGreater(self.view.undo_stack.count(), initial_undo_count)
        self.assertTrue(np.isnan(self.table.get_cell_value(0, "Y", 0, 0)))
        self.assertTrue(np.isnan(self.table.get_cell_value(0, "Y", 0, 1)))

        # Undo restores values
        self.view.undo_stack.undo()
        self.assertEqual(self.table.get_cell_value(0, "Y", 0, 0), 10.5)
        self.assertEqual(self.table.get_cell_value(0, "Y", 0, 1), 20.5)

    def test_key_press_backspace_triggers_batch_clear_undoable(self) -> None:
        """Pressing Backspace key on selected cells pushes an undoable command."""
        tl = self.model.index(1, 0)
        br = self.model.index(1, 1)
        sel = QItemSelection(tl, br)
        self.view.selectionModel().select(sel, self.view.selectionModel().SelectionFlag.Select)

        event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Backspace, Qt.KeyboardModifier.NoModifier)
        self.view.keyPressEvent(event)

        self.assertTrue(np.isnan(self.table.get_cell_value(1, "Y", 0, 0)))
        self.assertTrue(np.isnan(self.table.get_cell_value(1, "Y", 0, 1)))

        self.view.undo_stack.undo()
        self.assertEqual(self.table.get_cell_value(1, "Y", 0, 0), 30.5)
        self.assertEqual(self.table.get_cell_value(1, "Y", 0, 1), 40.5)

    def test_batch_clear_discontinuous_selection_and_tuple_formats(self) -> None:
        """Discontinuous cell clearing using 2-tuple coordinates clears cells and undoes cleanly."""
        self.table.set_cell_value(0, "Y", 0, 0, 111.0)
        self.table.set_cell_value(2, "Y", 0, 1, 222.0)

        # Pass 2-tuples [(0, 0), (2, 1)] with a duplicate (0, 0)
        cmd = BatchClearCellsCommand(self.model, [(0, 0), (2, 1), (0, 0)], "Clear Non-Adjacent")
        self.view.undo_stack.push(cmd)

        self.assertTrue(np.isnan(self.table.get_cell_value(0, "Y", 0, 0)))
        self.assertTrue(np.isnan(self.table.get_cell_value(2, "Y", 0, 1)))

        # Undo restores both cells
        self.view.undo_stack.undo()
        self.assertEqual(self.table.get_cell_value(0, "Y", 0, 0), 111.0)
        self.assertEqual(self.table.get_cell_value(2, "Y", 0, 1), 222.0)


class TestFocusAwareUndoRedo(unittest.TestCase):
    """Tests routing of Ctrl+Z / Ctrl+Y according to QApplication.focusWidget()."""

    def setUp(self) -> None:
        self.window = MainWindow()
        self.window.show()

    def tearDown(self) -> None:
        self.window.close()

    def test_redo_shortcut_is_ctrl_y(self) -> None:
        """Verifies redo action has Ctrl+Y shortcut."""
        shortcuts = [s.toString() for s in self.window.redo_act.shortcuts()]
        self.assertTrue(any("Y" in s for s in shortcuts), f"Redo shortcuts: {shortcuts}")

    def test_focus_aware_undo_on_text_edit(self) -> None:
        """When a QTextEdit has focus, window.action_undo calls widget.undo()."""
        info_sheet = self.window.document.create_info_sheet("Notes")
        self.window.on_sheet_selected(info_sheet.sheet_id)
        text_edit = self.window.info_widget.editor
        text_edit.setPlainText("Initial")
        text_edit.setFocus()
        cursor = text_edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        text_edit.setTextCursor(cursor)
        text_edit.insertPlainText(" Added")
        self.assertEqual(text_edit.toPlainText(), "Initial Added")

        # Global undo should undo the text edit typing
        self.window.action_undo()
        self.assertEqual(text_edit.toPlainText(), "Initial")

        # Global redo should redo the text edit typing
        self.window.action_redo()
        self.assertEqual(text_edit.toPlainText(), "Initial Added")

    def test_focus_aware_undo_on_spreadsheet(self) -> None:
        """When spreadsheet view has focus, window.action_undo delegates to spreadsheet stack."""
        active_sheet = self.window.document.get_sheet(self.window.document.active_sheet_id or "")
        self.assertIsInstance(active_sheet, DataSheet)
        sheet_view = self.window.spreadsheet_view
        sheet_view.setFocus()

        # Mutate table via batch clear command
        old_val = 99.0
        active_sheet.table.set_cell_value(0, "X", 0, 0, old_val)
        cmd = BatchClearCellsCommand(self.window.spreadsheet_model, [(0, 0, old_val)], "Clear")
        sheet_view.undo_stack.push(cmd)
        self.assertTrue(np.isnan(active_sheet.table.get_cell_value(0, "X", 0, 0)))

        self.window.action_undo()
        self.assertEqual(active_sheet.table.get_cell_value(0, "X", 0, 0), 99.0)

        self.window.action_redo()
        self.assertTrue(np.isnan(active_sheet.table.get_cell_value(0, "X", 0, 0)))


class TestPerSheetGraphStatePersistence(unittest.TestCase):
    """Tests per-sheet plot_config persistence and canvas state retention."""

    def setUp(self) -> None:
        self.window = MainWindow()
        self.doc = self.window.document

    def tearDown(self) -> None:
        self.window.close()

    def test_graph_config_persists_across_navigation(self) -> None:
        """Switching between sheets persists individual plot_config and does not reset to defaults."""
        # Create sheet 1 (COLUMN)
        sheet1 = self.doc.create_data_sheet("Sheet 1", TableType.COLUMN)
        sheet1.table.set_cell_value(0, "Y", 0, 0, 10.0)
        sheet1.table.set_cell_value(1, "Y", 0, 0, 20.0)
        sheet1.plot_config = {
            "plot_type": "bar",
            "whiskers": "SEM",
            "palette": "viridis",
            "show_scatter": False,
        }

        # Create sheet 2 (XY)
        sheet2 = self.doc.create_data_sheet("Sheet 2", TableType.XY)
        sheet2.table.set_cell_value(0, "X", 0, 0, 1.0)
        sheet2.table.set_cell_value(0, "Y", 0, 0, 2.0)
        sheet2.plot_config = {
            "plot_type": "xy_scatter",
            "palette": "inferno",
            "show_scatter": True,
        }

        self.window.navigator.populate(self.doc)

        # Select sheet 1
        self.window.on_sheet_selected(sheet1.sheet_id)
        canvas = self.window.graph_canvas
        self.assertEqual(canvas.active_sheet.sheet_id, sheet1.sheet_id)
        self.assertEqual(canvas.plot_type_combo.currentText(), GraphType.COLUMN_BAR_ERROR.value)
        self.assertEqual(canvas.palette_combo.currentText(), "viridis")
        self.assertFalse(canvas.show_scatter_check.isChecked())

        # Select sheet 2
        self.window.on_sheet_selected(sheet2.sheet_id)
        self.assertEqual(canvas.active_sheet.sheet_id, sheet2.sheet_id)
        self.assertEqual(canvas.plot_type_combo.currentText(), GraphType.XY_SCATTER_CURVE.value)
        self.assertEqual(canvas.palette_combo.currentText(), "inferno")

        # Select back sheet 1
        self.window.on_sheet_selected(sheet1.sheet_id)
        self.assertEqual(canvas.active_sheet.sheet_id, sheet1.sheet_id)
        self.assertEqual(canvas.plot_type_combo.currentText(), GraphType.COLUMN_BAR_ERROR.value)
        self.assertEqual(canvas.palette_combo.currentText(), "viridis")
        self.assertFalse(canvas.show_scatter_check.isChecked())

    def test_navigation_switches_defaults_without_contamination(self) -> None:
        """Navigating from a customized Column sheet to a new XY sheet resets to XY defaults without bleed."""
        sheet1 = self.doc.create_data_sheet("Custom Column", TableType.COLUMN)
        self.window.on_sheet_selected(sheet1.sheet_id)
        canvas = self.window.graph_canvas
        canvas.palette_combo.setCurrentIndex(1)  # Journal Grayscale
        self.assertEqual(sheet1.plot_config["palette_index"], 1)

        # Create new unconfigured XY sheet
        sheet2 = self.doc.create_data_sheet("New XY", TableType.XY)
        self.window.on_sheet_selected(sheet2.sheet_id)

        # Regression check: must be XY Scatter & Curve with default color palette, NOT Column/Grayscale!
        self.assertEqual(canvas.plot_type_combo.currentText(), GraphType.XY_SCATTER_CURVE.value)
        self.assertEqual(canvas.palette_combo.currentIndex(), 0)


class TestOfficeExportIsolation(unittest.TestCase):
    """Tests that render_graph_to_png_bytes isolates live canvas state with try...finally."""

    def test_render_graph_preserves_live_canvas_state(self) -> None:
        t1 = DataTable(table_id="t1", title="Active", table_type=TableType.COLUMN, num_rows=5, num_replicates=1)
        t1.set_cell_value(0, "Y", 0, 0, 5.0)
        sheet1 = DataSheet(sheet_id="s1", title="Active", table=t1)
        sheet1.plot_config = {"plot_type": "bar", "palette": "viridis", "show_scatter": False}

        t2 = DataTable(table_id="t2", title="ExportTarget", table_type=TableType.XY, num_rows=5, num_replicates=1)
        t2.set_cell_value(0, "X", 0, 0, 1.0)
        t2.set_cell_value(0, "Y", 0, 0, 2.0)
        sheet2 = DataSheet(sheet_id="s2", title="ExportTarget", table=t2)
        sheet2.plot_config = {"plot_type": "xy_scatter", "palette": "magma"}

        canvas = GraphCanvasWidget()
        canvas.update_data(sheet1.table, sheet=sheet1)

        self.assertEqual(canvas.active_sheet, sheet1)
        self.assertEqual(canvas.plot_type_combo.currentText(), GraphType.COLUMN_BAR_ERROR.value)
        self.assertEqual(canvas.palette_combo.currentText(), "viridis")

        # Render sheet 2 to bytes
        png_bytes = render_graph_to_png_bytes(canvas, sheet2)
        self.assertGreater(len(png_bytes), 100)

        # Verify canvas state is unconditionally restored to sheet1
        self.assertEqual(canvas.active_sheet, sheet1)
        self.assertEqual(canvas.plot_type_combo.currentText(), GraphType.COLUMN_BAR_ERROR.value)
        self.assertEqual(canvas.palette_combo.currentText(), "viridis")
        self.assertFalse(canvas.show_scatter_check.isChecked())


class TestPinnedAndFrozenGraphs(unittest.TestCase):
    """Tests GraphSheet creation, pin button, freeze toggle, and persistence."""

    def setUp(self) -> None:
        self.window = MainWindow()
        self.doc = self.window.document

    def tearDown(self) -> None:
        self.window.close()

    def test_standard_pinned_name_formatter(self) -> None:
        """Tests standard pinned graph name formatting function."""
        name = format_pinned_graph_name("Bioassay 1", "4PL Fit", "XY Scatter & Curve")
        self.assertEqual(name, "Bioassay 1 - 4PL Fit (XY Scatter & Curve)")

        raw_name = format_pinned_graph_name("Cell Counts", None, "Bar Chart")
        self.assertEqual(raw_name, "Cell Counts - Raw (Bar Chart)")

    def test_pin_graph_creates_graph_sheet_and_node(self) -> None:
        """Clicking Pin Graph creates a GraphSheet in doc.graphs and adds child to 📈 Graphs."""
        active_sheet = self.doc.get_sheet(self.doc.active_sheet_id or "")
        self.assertIsNotNone(active_sheet)

        initial_graphs_count = len(self.doc.graphs)
        self.window.action_pin_graph()

        self.assertEqual(len(self.doc.graphs), initial_graphs_count + 1)
        pinned = self.doc.graphs[-1]
        self.assertTrue(pinned.is_pinned)
        self.assertEqual(pinned.sheet_type, SheetType.GRAPH)

        # Check navigator root_graphs has child
        root_graphs = self.window.navigator.root_graphs
        self.assertIsNotNone(root_graphs)
        self.assertEqual(root_graphs.childCount(), len(self.doc.graphs))

    def test_freeze_graph_disables_upstream_updates(self) -> None:
        """Toggling freeze creates a snapshot and disconnects upstream table mutations."""
        data_sheet = self.doc.create_data_sheet("Dosing", TableType.COLUMN)
        data_sheet.table.set_cell_value(0, "Y", 0, 0, 50.0)

        self.window.on_sheet_selected(data_sheet.sheet_id)
        self.window.action_pin_graph()

        pinned_sheet = self.doc.graphs[-1]
        self.window.on_sheet_selected(pinned_sheet.sheet_id)

        # Toggle freeze ON
        canvas = self.window.graph_canvas
        self.assertFalse(canvas.btn_freeze.isHidden())
        canvas.btn_freeze.setChecked(True)  # triggers freezeToggled

        self.assertTrue(pinned_sheet.is_frozen)
        self.assertIsNotNone(pinned_sheet.snapshot_table)
        self.assertEqual(pinned_sheet.snapshot_table.get_cell_value(0, "Y", 0, 0), 50.0)

        # Mutate upstream data sheet
        data_sheet.table.set_cell_value(0, "Y", 0, 0, 999.0)

        # Re-select pinned sheet: graph should still show frozen snapshot data (50.0)
        self.window.on_sheet_selected(pinned_sheet.sheet_id)
        self.assertEqual(canvas.active_sheet.snapshot_table.get_cell_value(0, "Y", 0, 0), 50.0)

        # Toggle freeze OFF
        canvas.btn_freeze.setChecked(False)
        self.assertFalse(pinned_sheet.is_frozen)
        self.assertIsNone(pinned_sheet.snapshot_table)

    def test_pinned_and_frozen_persistence_in_bigo(self) -> None:
        """Pinned and frozen graph sheets are saved and loaded in .bigo project files."""
        data_sheet = self.doc.create_data_sheet("Assay", TableType.COLUMN)
        data_sheet.table.set_cell_value(0, "Y", 0, 0, 42.0)

        pinned_sheet = self.doc.create_graph_sheet("Assay - Raw (Bar Chart)", data_sheet.sheet_id)
        pinned_sheet.is_pinned = True
        pinned_sheet.is_frozen = True
        pinned_sheet.snapshot_table = data_sheet.table.clone()
        pinned_sheet.plot_config = {"plot_type": "bar", "palette": "plasma"}

        with tempfile.NamedTemporaryFile(suffix=".bigo", delete=False) as f:
            temp_path = f.name

        try:
            PrismIO.save(self.doc, temp_path)
            loaded_doc = PrismIO.load(temp_path)

            self.assertEqual(len(loaded_doc.graphs), 1)
            loaded_graph = loaded_doc.graphs[0]
            self.assertTrue(loaded_graph.is_pinned)
            self.assertTrue(loaded_graph.is_frozen)
            self.assertIsNotNone(loaded_graph.snapshot_table)
            self.assertEqual(loaded_graph.snapshot_table.get_cell_value(0, "Y", 0, 0), 42.0)
            self.assertEqual(loaded_graph.plot_config.get("palette"), "plasma")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_freeze_graph_preserves_analysis_curves(self) -> None:
        """Freezing an analysis graph preserves analysis_layers and snapshot_result_data."""
        data_sheet = self.doc.create_data_sheet("Bioassay", TableType.XY, num_rows=5, num_replicates=1)
        for i in range(5):
            data_sheet.table.set_cell_value(i, "X", 0, 0, float(i))
            data_sheet.table.set_cell_value(i, "Y", 0, 0, float(2 * i + 1))

        self.window.on_sheet_selected(data_sheet.sheet_id)
        self.window.action_run_linear()

        res_sheet = [s for s in self.doc.sheets.values() if s.sheet_type == SheetType.RESULTS][0]
        self.window.on_sheet_selected(res_sheet.sheet_id)

        pinned_sheet = self.window.action_pin_graph()
        self.assertIsNotNone(pinned_sheet.result_sheet_id)
        self.assertIsNotNone(pinned_sheet.snapshot_result_data)
        self.assertGreaterEqual(len(self.window.graph_canvas.analysis_layers), 1)

        # Freeze graph
        self.window.graph_canvas.btn_freeze.setChecked(True)
        self.window.on_sheet_selected(pinned_sheet.sheet_id)

        # Regression check: fitted curve MUST remain present on frozen graph!
        self.assertGreaterEqual(len(self.window.graph_canvas.analysis_layers), 1)
        self.assertTrue(pinned_sheet.is_frozen)

    def test_unfreeze_orphaned_graph_preserves_snapshot_fallback(self) -> None:
        """Unfreezing a graph whose upstream parent was deleted retains snapshot data fallback."""
        data_sheet = self.doc.create_data_sheet("Parent", TableType.COLUMN)
        data_sheet.table.set_cell_value(0, "Y", 0, 0, 77.0)

        self.window.on_sheet_selected(data_sheet.sheet_id)
        pinned = self.window.action_pin_graph()
        self.window.graph_canvas.btn_freeze.setChecked(True)

        # Delete upstream parent sheet
        self.doc.remove_sheet(data_sheet.sheet_id)
        pinned.is_orphaned = True

        # Unfreeze
        self.window.graph_canvas.btn_freeze.setChecked(False)
        self.assertIsNotNone(pinned.snapshot_table)
        self.assertEqual(pinned.snapshot_table.get_cell_value(0, "Y", 0, 0), 77.0)


class TestProjectNavigatorContextMenus(unittest.TestCase):
    """Tests exact context menu strings and '➕ Add Info Sheet'."""

    def setUp(self) -> None:
        self.window = MainWindow()
        self.doc = self.window.document
        self.navigator = self.window.navigator

    def tearDown(self) -> None:
        self.window.close()

    def _get_item_center_pos(self, item: QTreeWidgetItem) -> QPoint:
        rect = self.navigator.visualItemRect(item)
        return rect.center()

    def _find_item_by_sheet_id(self, sheet_id: str) -> QTreeWidgetItem | None:
        for folder in self.navigator.folder_items.values():
            for i in range(folder.childCount()):
                child = folder.child(i)
                if child.data(0, Qt.ItemDataRole.UserRole) == sheet_id:
                    return child
        return None

    def test_exact_context_menu_labels_for_each_sheet_type(self) -> None:
        """Checks rename and delete labels match specific types."""
        # Data sheet
        data_sheet = self.doc.create_data_sheet("Table 1", TableType.XY)
        self.navigator.populate(self.doc)
        item_data = self._find_item_by_sheet_id(data_sheet.sheet_id)
        self.assertIsNotNone(item_data)
        menu_data = self.window._create_navigator_context_menu(self._get_item_center_pos(item_data))
        self.assertIsNotNone(menu_data)
        visible_texts = [a.text() for a in menu_data.actions() if a.isVisible()]
        self.assertIn("✏️ Rename Data Table...", visible_texts)
        self.assertIn("🗑️ Delete Data Table", visible_texts)

        # Info sheet
        info_sheet = self.doc.create_info_sheet("Notes 1")
        self.navigator.populate(self.doc)
        item_info = self._find_item_by_sheet_id(info_sheet.sheet_id)
        self.assertIsNotNone(item_info)
        menu_info = self.window._create_navigator_context_menu(self._get_item_center_pos(item_info))
        self.assertIsNotNone(menu_info)
        visible_info_texts = [a.text() for a in menu_info.actions() if a.isVisible()]
        self.assertIn("✏️ Rename Info Sheet...", visible_info_texts)
        self.assertIn("🗑️ Delete Info Sheet", visible_info_texts)

        # Graph sheet
        graph_sheet = self.doc.create_graph_sheet("Graph 1")
        self.navigator.populate(self.doc)
        item_graph = self._find_item_by_sheet_id(graph_sheet.sheet_id)
        self.assertIsNotNone(item_graph)
        menu_graph = self.window._create_navigator_context_menu(self._get_item_center_pos(item_graph))
        self.assertIsNotNone(menu_graph)
        visible_graph_texts = [a.text() for a in menu_graph.actions() if a.isVisible()]
        self.assertIn("✏️ Rename Graph...", visible_graph_texts)
        self.assertIn("🗑️ Delete Graph", visible_graph_texts)

    def test_add_info_sheet_action(self) -> None:
        """Navigator menu contains '➕ Add Info Sheet' and triggers creation."""
        root_info = self.navigator.root_info
        self.assertIsNotNone(root_info)
        menu = self.window._create_navigator_context_menu(self._get_item_center_pos(root_info))
        self.assertIsNotNone(menu)
        action_texts = [a.text() for a in menu.actions() if a.isVisible()]
        self.assertIn("➕ Add Info Sheet", action_texts)

        initial_info_count = len(self.doc.info_sheets)
        self.window.action_add_info_sheet()
        self.assertEqual(len(self.doc.info_sheets), initial_info_count + 1)
        self.assertEqual(self.navigator.root_info.childCount(), len(self.doc.info_sheets))


if __name__ == "__main__":
    unittest.main()
