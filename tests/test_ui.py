"""
UI Unit Tests for GemGraph.
Tests window title unsaved changes indicator, state tracking, and save clearing lifecycle.
"""

from __future__ import annotations
import os
import tempfile
import unittest
import unittest.mock
from PyQt6.QtWidgets import QApplication, QMessageBox, QMenu
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtCore import Qt
import numpy as np

from src.ui.main_window import MainWindow
from src.core.table import DataTable, TableType
from src.ui.spreadsheet import ScientificSpreadsheetModel, ScientificSpreadsheetView
from src.ui.mediator import WorkspaceMediator
from src.data_engine.table_model import ScientificTableModel

app = QApplication.instance()
if app is None:
    app = QApplication([])


class TestUIUnsavedChangesIndicator(unittest.TestCase):
    def setUp(self) -> None:
        self.window = MainWindow()

    def tearDown(self) -> None:
        self.window.close()

    def test_window_title_unsaved_changes_lifecycle(self) -> None:
        """Verifies triggering a data modification updates window title to include '*', and saving clears it."""
        # 1. Fresh launch: not modified, no asterisk
        self.assertFalse(self.window._is_modified)
        self.assertEqual(
            self.window.windowTitle(),
            "Untitled - GemGraph - v0.1.2-nightly - Clean Room Edition",
        )

        # 2. Modify cell data in spreadsheet model
        model = self.window.spreadsheet_model
        index = model.index(0, 0)
        success = model.setData(index, "99.5", Qt.ItemDataRole.EditRole)
        self.assertTrue(success)

        # Title must include asterisk
        self.assertTrue(self.window._is_modified)
        self.assertEqual(
            self.window.windowTitle(),
            "Untitled* - GemGraph - v0.1.2-nightly - Clean Room Edition",
        )

        # 3. Save project clears modification indicator
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_proj.bigo")
            with unittest.mock.patch("PyQt6.QtWidgets.QFileDialog.getSaveFileName", return_value=(save_path, "")):
                self.window.action_save_project()

            self.assertFalse(self.window._is_modified)
            self.assertEqual(
                self.window.windowTitle(),
                "test_proj.bigo - GemGraph - v0.1.2-nightly - Clean Room Edition",
            )

            # 4. Modifying again after save appends asterisk to the saved file name
            model.setData(model.index(1, 0), "123.4", Qt.ItemDataRole.EditRole)
            self.assertTrue(self.window._is_modified)
            self.assertEqual(
                self.window.windowTitle(),
                "test_proj.bigo* - GemGraph - v0.1.2-nightly - Clean Room Edition",
            )

            # 5. Save again directly with current_file_path clears asterisk
            self.window.action_save_project()
            self.assertFalse(self.window._is_modified)
            self.assertEqual(
                self.window.windowTitle(),
                "test_proj.bigo - GemGraph - v0.1.2-nightly - Clean Room Edition",
            )

            # 6. action_save_project_as clears asterisk
            save_as_path = os.path.join(tmpdir, "test_proj_copy.bigo")
            self.window.mark_as_modified()
            self.assertTrue(self.window._is_modified)
            with unittest.mock.patch("PyQt6.QtWidgets.QFileDialog.getSaveFileName", return_value=(save_as_path, "")):
                self.window.action_save_project_as()
            self.assertFalse(self.window._is_modified)
            self.assertEqual(
                self.window.windowTitle(),
                "test_proj_copy.bigo - GemGraph - v0.1.2-nightly - Clean Room Edition",
            )

        # 7. Creating a new project clears modification indicator
        self.window.mark_as_modified()
        self.assertTrue(self.window._is_modified)
        self.window.action_new_project()
        self.assertFalse(self.window._is_modified)
        self.assertEqual(
            self.window.windowTitle(),
            "Untitled - GemGraph - v0.1.2-nightly - Clean Room Edition",
        )
    def test_modification_signals_trigger_indicator(self) -> None:
        """Verifies layout, graph, and info modification signals set _is_modified to True."""
        # 1. Info modified signal
        self.assertFalse(self.window._is_modified)
        self.window.info_widget.infoModified.emit()
        self.assertTrue(self.window._is_modified)
        self.assertIn("*", self.window.windowTitle())

        # Reset
        self.window.action_new_project()
        self.assertFalse(self.window._is_modified)

        # 2. Layout modified signal
        self.window.layout_view.layoutModified.emit()
        self.assertTrue(self.window._is_modified)
        self.assertIn("*", self.window.windowTitle())

        # Reset
        self.window.action_new_project()
        self.assertFalse(self.window._is_modified)

        # 3. Graph modified signal
        self.window.graph_canvas.graphModified.emit()
        self.assertTrue(self.window._is_modified)
        self.assertIn("*", self.window.windowTitle())

    def test_open_project_clears_indicator(self) -> None:
        """Verifies opening an existing file resets _is_modified to False."""
        self.window.mark_as_modified()
        self.assertTrue(self.window._is_modified)

        self.window.action_open_sample_project()
        self.assertFalse(self.window._is_modified)
        self.assertEqual(
            self.window.windowTitle(),
            "textbook_sample.bigo - GemGraph - v0.1.2-nightly - Clean Room Edition",
        )


    def test_cancel_save_preserves_indicator(self) -> None:
        """Verifies canceling the file dialog when modified preserves _is_modified and '*' in window title."""
        self.window.mark_as_modified()
        self.assertTrue(self.window._is_modified)
        self.assertIn("*", self.window.windowTitle())

        with unittest.mock.patch("PyQt6.QtWidgets.QFileDialog.getSaveFileName", return_value=("", "")):
            self.window.action_save_project_as()

        self.assertTrue(self.window._is_modified)
        self.assertIn("*", self.window.windowTitle())

    def test_failed_save_preserves_indicator(self) -> None:
        """Verifies that an exception during save leaves _is_modified as True and preserves '*' in window title."""
        self.window.current_file_path = "/tmp/failing_save_test.bigo"
        self.window.mark_as_modified()
        self.assertTrue(self.window._is_modified)
        self.assertIn("*", self.window.windowTitle())

        with unittest.mock.patch("src.io.prism_io.PrismIO.save", side_effect=IOError("Simulated write failure")), \
             unittest.mock.patch("PyQt6.QtWidgets.QMessageBox.critical"):
            self.window.action_save_project()

        self.assertTrue(self.window._is_modified)
        self.assertIn("*", self.window.windowTitle())

    def test_document_sheet_mutations_trigger_indicator(self) -> None:
        """Verifies sheet addition, sheet deletion, sheet renaming, and freeze toggling mark project as modified."""
        from src.core.table import TableType
        from PyQt6.QtWidgets import QTreeWidgetItem

        # 1. Adding a table marks modified
        self.assertFalse(self.window._is_modified)
        self.window.action_add_table(TableType.COLUMN)
        self.assertTrue(self.window._is_modified)
        self.assertIn("*", self.window.windowTitle())

        # Reset
        self.window.action_new_project()
        self.assertFalse(self.window._is_modified)

        # 2. Deleting a sheet marks modified
        sheet_to_delete = self.window.document.sheet_order[-1]
        self.window.on_sheet_deleted(sheet_to_delete)
        self.assertTrue(self.window._is_modified)
        self.assertIn("*", self.window.windowTitle())

        # Reset
        self.window.action_new_project()
        self.assertFalse(self.window._is_modified)

        # 3. Adding an info sheet marks modified
        self.window.action_add_info()
        self.assertTrue(self.window._is_modified)
        self.assertIn("*", self.window.windowTitle())

        # Reset
        self.window.action_new_project()
        self.assertFalse(self.window._is_modified)

        # 4. Adding a graph sheet marks modified
        self.window.action_add_graph()
        self.assertTrue(self.window._is_modified)
        self.assertIn("*", self.window.windowTitle())

        # Reset
        self.window.action_new_project()
        self.assertFalse(self.window._is_modified)

        # 5. Running simulation / analysis marks modified
        self.window.action_run_simulate_xy()
        self.assertTrue(self.window._is_modified)
        self.assertIn("*", self.window.windowTitle())

        # Reset
        self.window.action_new_project()
        self.assertFalse(self.window._is_modified)

        # 6. Renaming a sheet marks modified
        item = QTreeWidgetItem(["Data 1"])
        sid = self.window.document.sheet_order[0]
        with unittest.mock.patch("PyQt6.QtWidgets.QInputDialog.getText", return_value=("Renamed Data", True)):
            self.window._rename_navigator_sheet(item, sid)
        self.assertTrue(self.window._is_modified)
        self.assertIn("*", self.window.windowTitle())

        # Reset
        self.window.action_new_project()
        self.assertFalse(self.window._is_modified)

        # 7. Toggling graph freeze marks modified
        self.window.action_add_graph()
        self.window._is_modified = False
        self.window._on_freeze_toggled(True)
        self.assertTrue(self.window._is_modified)
        self.assertIn("*", self.window.windowTitle())

        # Reset
        self.window.action_new_project()
        self.assertFalse(self.window._is_modified)

        # 8. Mediator tableModified and dataChanged mark modified
        self.window.mediator.tableModified.emit()
        self.assertTrue(self.window._is_modified)
        self.assertIn("*", self.window.windowTitle())

        self.window.action_new_project()
        self.assertFalse(self.window._is_modified)

        self.window.mediator.dataChanged.emit()
        self.assertTrue(self.window._is_modified)
        self.assertIn("*", self.window.windowTitle())

    def test_close_event_unsaved_changes_lifecycle(self) -> None:
        """Verifies closeEvent accepts if not modified, ignores if cancelled, accepts if discarded, and handles save."""
        # 1. Unmodified project: accepted immediately without dialog
        self.window._is_modified = False
        ev = QCloseEvent()
        self.window.closeEvent(ev)
        self.assertTrue(ev.isAccepted())

        # 2. Modified project, user clicks Discard -> accept
        self.window.mark_as_modified()
        self.assertTrue(self.window._is_modified)
        ev_discard = QCloseEvent()
        with unittest.mock.patch.object(QMessageBox, "exec", return_value=QMessageBox.StandardButton.Discard):
            self.window.closeEvent(ev_discard)
        self.assertTrue(ev_discard.isAccepted())

        # 3. Modified project, user clicks Cancel -> ignore
        self.window.mark_as_modified()
        ev_cancel = QCloseEvent()
        with unittest.mock.patch.object(QMessageBox, "exec", return_value=QMessageBox.StandardButton.Cancel):
            self.window.closeEvent(ev_cancel)
        self.assertFalse(ev_cancel.isAccepted())

        # 4. Modified project, user clicks Save, but save dialog is cancelled -> ignore
        self.window.mark_as_modified()
        ev_save_cancel = QCloseEvent()
        with unittest.mock.patch.object(QMessageBox, "exec", return_value=QMessageBox.StandardButton.Save), \
             unittest.mock.patch.object(self.window, "action_save_project", return_value=False):
            self.window.closeEvent(ev_save_cancel)
        self.assertFalse(ev_save_cancel.isAccepted())

        # 5. Modified project, user clicks Save, and save succeeds -> accept
        self.window.mark_as_modified()
        ev_save_success = QCloseEvent()
        def mock_successful_save():
            self.window._is_modified = False
            return True
        with unittest.mock.patch.object(QMessageBox, "exec", return_value=QMessageBox.StandardButton.Save), \
             unittest.mock.patch.object(self.window, "action_save_project", side_effect=mock_successful_save):
            self.window.closeEvent(ev_save_success)
        self.assertTrue(ev_save_success.isAccepted())

    def test_mediator_multi_x_extraction_with_xy_mapping(self) -> None:
        """Verifies that xy_mapping dictates the correct x_data extraction in WorkspaceMediator."""
        table = DataTable(table_id="multi_x_tbl", title="Multi-X Test", table_type=TableType.XY, num_rows=3, num_replicates=1)
        table.x_columns[0]["title"] = "X1"
        table.x_columns[0]["data"] = np.array([1.0, 2.0, 3.0])
        table.set_cell_value(0, "Y", 0, 0, 10.0)
        table.set_cell_value(1, "Y", 0, 0, 20.0)
        table.set_cell_value(2, "Y", 0, 0, 30.0)
        table.y_group_titles[0] = "Y1"

        x2_idx = table.insert_x_column_at(1, position="right", title="X2")
        table.x_columns[x2_idx]["data"] = np.array([100.0, 200.0, 300.0])
        table.insert_column(2, position="right", title="Y2")
        table.set_cell_value(0, "Y", 1, 0, 40.0)
        table.set_cell_value(1, "Y", 1, 0, 50.0)
        table.set_cell_value(2, "Y", 1, 0, 60.0)

        model = ScientificTableModel(table)
        mediator = WorkspaceMediator.get_instance()
        mediator.model = model

        # Default unassigned: Y2 falls back to Column 0 (X1)
        packaged = mediator.sync_data_to_visualization(model=model)
        self.assertEqual(len(packaged), 2)
        np.testing.assert_array_equal(packaged[0][0], np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(packaged[1][0], np.array([1.0, 2.0, 3.0]))

        # Now assign Y2 (visual col 3) to X2 (visual col 2)
        model.xy_mapping[3] = 2
        packaged_mapped = mediator.sync_data_to_visualization(model=model)
        np.testing.assert_array_equal(packaged_mapped[0][0], np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(packaged_mapped[1][0], np.array([100.0, 200.0, 300.0]))
        np.testing.assert_array_equal(packaged_mapped[1][1], np.array([40.0, 50.0, 60.0]))
        self.assertEqual(packaged_mapped[1][2], "Y2")

        # Verify serialization and deserialization of xy_mapping
        data_dict = model.to_dict()
        self.assertIn("xy_mapping", data_dict)
        restored_model = ScientificTableModel.from_dict(data_dict)
        self.assertIn(3, restored_model.xy_mapping)
        self.assertEqual(restored_model.xy_mapping[3], 2)

    def test_spreadsheet_header_context_menu_assigns_x_column(self) -> None:
        """Verifies right-clicking a Y header presents Assign to X Column and updates model.xy_mapping."""
        table = DataTable(table_id="hdr_test", title="Header Test", table_type=TableType.XY, num_rows=2, num_replicates=1)
        table.x_columns[0]["title"] = "X1"
        table.insert_x_column_at(1, position="right", title="X2")
        table.insert_column(2, position="right", title="Y2")

        model = ScientificTableModel(table)
        view = ScientificSpreadsheetView(model)

        menu = QMenu(view)
        # Column 3 is Y2 (a Y column)
        view._populate_column_actions(menu, 3)

        submenus = [act.menu() for act in menu.actions() if act.menu() is not None]
        assign_menus = [m for m in submenus if "Assign to X Column" in m.title()]
        self.assertEqual(len(assign_menus), 1)

        x2_actions = [act for act in assign_menus[0].actions() if "X2" in act.text()]
        self.assertGreater(len(x2_actions), 0)
        x2_actions[0].trigger()

        self.assertIn(3, model.xy_mapping)
        self.assertEqual(model.xy_mapping[3], 2)


if __name__ == "__main__":
    unittest.main()
