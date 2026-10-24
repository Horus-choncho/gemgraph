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


if __name__ == "__main__":
    unittest.main()
