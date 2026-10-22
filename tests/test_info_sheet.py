"""
Unit tests for GLP Scientific Info Sheets in GemGraph.
Tests InfoSheet data model, GLP fields, JSON serialization, and interactive InfoSheetView.
"""

from __future__ import annotations
import unittest
import numpy as np
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from src.core.info_sheet import InfoSheet, GLP_FIELD_DEFINITIONS
from src.core.sheet import SheetType
from src.core.document import PrismDocument
from src.ui.info_sheet_view import InfoSheetView
from src.ui.main_window import MainWindow

# Ensure headless QApplication
app = QApplication.instance() or QApplication([])


class TestInfoSheetModel(unittest.TestCase):
    """Verifies InfoSheet GLP fields and archive serialization."""

    def test_default_initialization(self) -> None:
        sheet = InfoSheet(sheet_id="info_test", title="Test Protocol")
        self.assertEqual(sheet.sheet_id, "info_test")
        self.assertEqual(sheet.title, "Test Protocol")
        self.assertEqual(sheet.sheet_type, SheetType.INFO)
        self.assertEqual(sheet.notes, "")
        self.assertEqual(sheet.experiment_title, "")
        self.assertEqual(sheet.project_study_id, "")

    def test_glp_field_access_and_constants_sync(self) -> None:
        sheet = InfoSheet(sheet_id="info_glp", title="Bioassay GLP")
        sheet.set_field_value("experiment_title", "Kinase Inhibition Assay")
        sheet.set_field_value("project_study_id", "PRJ-2026-042")
        sheet.set_field_value("author_scientist", "Dr. Jane Doe")
        sheet.set_field_value("principal_investigator", "Prof. John Smith")
        sheet.set_field_value("experiment_date", "2026-10-22")
        sheet.notes = "Cells incubated for 48h at 37°C."

        self.assertEqual(sheet.get_field_value("experiment_title"), "Kinase Inhibition Assay")
        self.assertEqual(sheet.get_field_value("project_study_id"), "PRJ-2026-042")
        self.assertEqual(sheet.constants.get("Experiment Title"), "Kinase Inhibition Assay")
        self.assertEqual(sheet.constants.get("Project / Study ID"), "PRJ-2026-042")
        self.assertEqual(sheet.notes, "Cells incubated for 48h at 37°C.")

    def test_serialization_round_trip(self) -> None:
        sheet = InfoSheet(
            sheet_id="info_roundtrip",
            title="GLP Report",
            notes="Protocol SOP-402 Rev 3",
            experiment_title="Dose Response Curve",
            project_study_id="STUDY-88",
            author_scientist="Alice Researcher",
            principal_investigator="Bob Director",
            notebook_page_ref="NB-12, p.45",
            experiment_date="2026-10-20",
            organism_cell_line="HEK293T",
            compound_lot_number="LOT-9921",
            protocol_sop_number="SOP-402",
        )
        d = sheet.to_dict()
        self.assertEqual(d["experiment_title"], "Dose Response Curve")
        self.assertEqual(d["organism_cell_line"], "HEK293T")
        self.assertIn("Experiment Title", d["constants"])

        restored = InfoSheet.from_dict(d)
        self.assertEqual(restored.sheet_id, "info_roundtrip")
        self.assertEqual(restored.experiment_title, "Dose Response Curve")
        self.assertEqual(restored.organism_cell_line, "HEK293T")
        self.assertEqual(restored.compound_lot_number, "LOT-9921")
        self.assertEqual(restored.notes, "Protocol SOP-402 Rev 3")


class TestInfoSheetView(unittest.TestCase):
    """Verifies interactive InfoSheetView editing and signal propagation."""

    def setUp(self) -> None:
        self.view = InfoSheetView()
        self.sheet = InfoSheet(
            sheet_id="info_view_test",
            title="Assay Info",
            notes="Observation: slight precipitation in well D4.",
            experiment_title="Solubility Test",
            author_scientist="Lab Tech 1",
        )

    def test_display_info_populates_widgets(self) -> None:
        self.view.display_info(self.sheet)

        # First row is Experiment Title
        item_title = self.view.table.item(0, 1)
        self.assertIsNotNone(item_title)
        self.assertEqual(item_title.text(), "Solubility Test")

        # Second row is Project / Study ID (empty)
        item_study = self.view.table.item(1, 1)
        self.assertIsNotNone(item_study)
        self.assertEqual(item_study.text(), "")

        # Notes
        self.assertEqual(self.view.notes_edit.toPlainText(), "Observation: slight precipitation in well D4.")

    def test_table_cell_editing_updates_model(self) -> None:
        self.view.display_info(self.sheet)

        signals_emitted = []
        self.view.infoModified.connect(lambda: signals_emitted.append(True))

        # Edit author_scientist (row index 2)
        item_author = self.view.table.item(2, 1)
        self.assertIsNotNone(item_author)
        item_author.setText("Dr. Alexander Fleming")

        self.assertEqual(self.sheet.author_scientist, "Dr. Alexander Fleming")
        self.assertEqual(self.sheet.constants.get("Author / Scientist"), "Dr. Alexander Fleming")
        self.assertTrue(len(signals_emitted) > 0)

    def test_notes_editing_updates_model(self) -> None:
        self.view.display_info(self.sheet)

        signals_emitted = []
        self.view.infoModified.connect(lambda: signals_emitted.append(True))

        self.view.notes_edit.setPlainText("Updated observations: culture clean.")
        self.assertEqual(self.sheet.notes, "Updated observations: culture clean.")
        self.assertTrue(len(signals_emitted) > 0)

    def test_main_window_info_sheet_integration(self) -> None:
        window = MainWindow()
        info_sheet = window.document.create_info_sheet(title="Project Specs")
        window.navigator.populate(window.document)

        # Select the info sheet
        window.on_sheet_selected(info_sheet.sheet_id)
        self.assertEqual(window.center_stack.currentWidget(), window.info_widget)

        # Edit through the view and ensure modified
        window.info_widget.notes_edit.setPlainText("Test Integration Notes")
        self.assertEqual(info_sheet.notes, "Test Integration Notes")
        window.close()


if __name__ == "__main__":
    unittest.main()
