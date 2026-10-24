"""
UI Unit Tests for GemGraph.
Tests window title unsaved changes indicator, state tracking, and save clearing lifecycle.
"""

from __future__ import annotations
import os
import tempfile
import unittest
import unittest.mock
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from src.ui.main_window import MainWindow

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


if __name__ == "__main__":
    unittest.main()
