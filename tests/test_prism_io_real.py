"""
Unit and regression tests for loading real-world GraphPad Prism 10 (.prism) archives.
Tests samples/test_sample.prism parsing, sheet hierarchy, replicate mapping, and cell values.
"""

import os
import unittest
import numpy as np

from src.core.document import PrismDocument
from src.core.sheet import DataSheet, InfoSheet, GraphSheet, SheetType
from src.core.table import TableType
from src.io.prism_io import PrismIO


class TestPrismIORealSample(unittest.TestCase):
    """Verifies clean loading and data fidelity of real GraphPad Prism 10 archives."""

    SAMPLE_PATH = os.path.join(os.path.dirname(__file__), "..", "samples", "test_sample.prism")

    def setUp(self) -> None:
        self.assertTrue(
            os.path.exists(self.SAMPLE_PATH),
            f"Required sample file not found at {self.SAMPLE_PATH}",
        )

    def test_load_real_sample_archive(self) -> None:
        """Loads samples/test_sample.prism and verifies sheet counts and structure."""
        doc = PrismIO.load(self.SAMPLE_PATH)
        self.assertIsInstance(doc, PrismDocument)

        # The sample file contains 11 data sheets, 2 info sheets, and 11 graph sheets = 24 sheets
        data_sheets = doc.get_sheets_by_type(SheetType.DATA)
        info_sheets = doc.get_sheets_by_type(SheetType.INFO)
        graph_sheets = doc.get_sheets_by_type(SheetType.GRAPH)

        self.assertEqual(len(data_sheets), 11)
        self.assertEqual(len(info_sheets), 2)
        self.assertEqual(len(graph_sheets), 11)
        self.assertEqual(len(doc.sheets), 24)
        self.assertEqual(len(doc.sheet_order), 24)

        # Active sheet must be a valid sheet in the document
        self.assertIsNotNone(doc.active_sheet_id)
        self.assertIn(doc.active_sheet_id, doc.sheets)







    def test_real_sample_data_fidelity_18s_sheet(self) -> None:
        """Verifies table metadata, titles, and numeric values in the '18s' DataSheet."""
        doc = PrismIO.load(self.SAMPLE_PATH)
        sheet_18s = doc.get_sheet("AD380D60-D6CA-4D3D-BF9D-65CC8B04BAC5")

        self.assertIsNotNone(sheet_18s)
        self.assertIsInstance(sheet_18s, DataSheet)
        self.assertEqual(sheet_18s.title, "18s")

        table = sheet_18s.table
        self.assertEqual(table.table_type, TableType.XY)
        self.assertEqual(table.x_title, "nM")
        self.assertEqual(table.num_rows, 4)
        self.assertEqual(
            table.y_group_titles,
            ["Ctrl", "Lipid", "siCtrl", "siDuox1", "siDuox2"],
        )

        # Check X values: [0, 10, 30, 60]
        expected_x = [0.0, 10.0, 30.0, 60.0]
        for r, expected_val in enumerate(expected_x):
            self.assertEqual(table.get_cell_value(r, "X", 0, 0), expected_val)

        # Check Row 0: Ctrl=1.0, Lipid=1.03, others NaN
        self.assertEqual(table.get_cell_value(0, "Y", 0, 0), 1.0)
        self.assertEqual(table.get_cell_value(0, "Y", 1, 0), 1.03)
        self.assertTrue(np.isnan(table.get_cell_value(0, "Y", 2, 0)))
        self.assertTrue(np.isnan(table.get_cell_value(0, "Y", 3, 0)))
        self.assertTrue(np.isnan(table.get_cell_value(0, "Y", 4, 0)))

        # Check Row 1: Ctrl=NaN, Lipid=NaN, siCtrl=1.06, siDuox1=0.97, siDuox2=1.01
        self.assertTrue(np.isnan(table.get_cell_value(1, "Y", 0, 0)))
        self.assertTrue(np.isnan(table.get_cell_value(1, "Y", 1, 0)))
        self.assertEqual(table.get_cell_value(1, "Y", 2, 0), 1.06)
        self.assertEqual(table.get_cell_value(1, "Y", 3, 0), 0.97)
        self.assertEqual(table.get_cell_value(1, "Y", 4, 0), 1.01)

        # Check clean XY extraction for group 0 ("Ctrl")
        xs, ys = table.get_clean_xy(0)
        self.assertEqual(len(xs), 1)
        self.assertEqual(xs[0], 0.0)
        self.assertEqual(ys[0], 1.0)

        # Check clean XY extraction for group 2 ("siCtrl")
        xs2, ys2 = table.get_clean_xy(2)
        self.assertEqual(len(xs2), 3)
        np.testing.assert_array_equal(xs2, [10.0, 30.0, 60.0])
        np.testing.assert_array_equal(ys2, [1.06, 1.0, 1.05])

    def test_real_sample_info_and_graph_metadata(self) -> None:
        """Verifies InfoSheet constants and GraphSheet linkage to DataSheets."""
        doc = PrismIO.load(self.SAMPLE_PATH)

        # Info sheet check
        info = doc.get_sheet("115026B6-27BE-4EEC-B53B-DB0620EB2F2F")
        self.assertIsNotNone(info)
        self.assertIsInstance(info, InfoSheet)
        self.assertEqual(info.title, "Project info 1")
        self.assertEqual(info.constants.get("Experiment Date"), "04/05/2026")

        # Graph sheet check
        graph_18s = doc.get_sheet("C823F5EC-6DDE-4675-8BAB-D3FC18C0C9F7")
        self.assertIsNotNone(graph_18s)
        self.assertIsInstance(graph_18s, GraphSheet)
        self.assertEqual(graph_18s.title, "18s")
        # Linked source sheet must be the 18s DataSheet
        self.assertEqual(graph_18s.source_sheet_id, "AD380D60-D6CA-4D3D-BF9D-65CC8B04BAC5")

    def test_all_data_sheets_fidelity(self) -> None:
        """Verifies table metadata, row counts, and numeric values across all 11 data sheets."""
        doc = PrismIO.load(self.SAMPLE_PATH)
        data_sheets = doc.get_sheets_by_type(SheetType.DATA)
        self.assertEqual(len(data_sheets), 11)

        for sheet in data_sheets:
            self.assertIsInstance(sheet, DataSheet)
            self.assertIsNotNone(sheet.table)
            self.assertEqual(sheet.table.table_type, TableType.XY)
            self.assertGreater(sheet.table.num_rows, 0)
            self.assertGreater(len(sheet.table.y_group_titles), 0)

        # Check Duox1 fold siCtrl sheet
        duox1_fold = doc.get_sheet("5DF9A9DD-D01A-4C82-8A7F-151CA4D50084")
        self.assertIsNotNone(duox1_fold)
        self.assertEqual(duox1_fold.title, "Duox1 fold siCtrl")
        tbl = duox1_fold.table
        self.assertEqual(tbl.num_rows, 3)
        self.assertEqual(tbl.x_title, "nM")
        self.assertEqual(tbl.y_group_titles, ["siDuox1", "siDuox2"])
        # Check first row values: X=10, siDuox1=0.83, siDuox2=1.84
        self.assertEqual(tbl.get_cell_value(0, "X", 0, 0), 10.0)
        self.assertEqual(tbl.get_cell_value(0, "Y", 0, 0), 0.83)
        self.assertEqual(tbl.get_cell_value(0, "Y", 1, 0), 1.84)


        # Check Duox1 sheet
        duox1 = doc.get_sheet("6A0C6BA4-8952-494C-8C56-1C18851C1ECA")
        self.assertIsNotNone(duox1)
        self.assertEqual(duox1.title, "Duox1")
        tbl_duox1 = duox1.table
        self.assertEqual(tbl_duox1.num_rows, 4)
        self.assertEqual(tbl_duox1.get_cell_value(0, "X", 0, 0), 0.0)
        self.assertEqual(tbl_duox1.get_cell_value(0, "Y", 0, 0), 1.0)
        self.assertEqual(tbl_duox1.get_cell_value(0, "Y", 1, 0), 0.52)


    def test_real_sample_round_trip(self) -> None:
        """Verifies save and reload round-trip fidelity of loaded real Prism 10 project."""
        import tempfile
        doc_original = PrismIO.load(self.SAMPLE_PATH)

        with tempfile.NamedTemporaryFile(suffix=".prism", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            PrismIO.save(doc_original, tmp_path)
            doc_reloaded = PrismIO.load(tmp_path)

            self.assertEqual(len(doc_reloaded.sheets), len(doc_original.sheets))
            self.assertEqual(len(doc_reloaded.sheet_order), len(doc_original.sheet_order))

            for sid in doc_original.sheet_order:
                orig_s = doc_original.get_sheet(sid)
                relo_s = doc_reloaded.get_sheet(sid)
                self.assertIsNotNone(relo_s)
                self.assertEqual(relo_s.title, orig_s.title)
                self.assertEqual(relo_s.sheet_type, orig_s.sheet_type)

                if isinstance(orig_s, DataSheet):
                    self.assertEqual(relo_s.table.num_rows, orig_s.table.num_rows)
                    self.assertEqual(relo_s.table.y_group_titles, orig_s.table.y_group_titles)
                    # Check X values
                    if orig_s.table.x_data is not None:
                        np.testing.assert_array_equal(
                            np.nan_to_num(relo_s.table.x_data),
                            np.nan_to_num(orig_s.table.x_data),
                        )
                    # Check Y values
                    np.testing.assert_array_equal(
                        np.nan_to_num(relo_s.table.y_data),
                        np.nan_to_num(orig_s.table.y_data),
                    )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_binary_dt_fallback_parsing(self) -> None:
        """Verifies decoding of binary float arrays from data.dt when data.csv is missing."""
        import struct
        import zipfile
        import json
        import tempfile

        # Create 8 double values: 2 rows of 4 columns [X, Y1, Y2, Y3]
        double_vals = [1.0, 10.0, 20.0, 30.0, 2.0, 15.0, 25.0, 35.0]
        binary_dt = struct.pack(f"<{len(double_vals)}d", *double_vals)

        manifest = {
            "format_version": "10.0",
            "sheets": {"data": ["test_sheet"]},
            "sheetAttributesMap": {"test_sheet": {"title": "Binary Sheet"}},
        }
        sheet_json = {
            "title": "Binary Sheet",
            "table": {
                "uid": "tbl_binary",
                "format": "xy",
                "xDataSet": "ds_x",
                "dataSets": ["ds_y1"],
            },
        }
        ds_x = {"title": "Concentration", "replicates": [{"cellAttributes": []}]}
        ds_y = {"title": "Response", "replicates": [
            {"cellAttributes": []},
            {"cellAttributes": []},
            {"cellAttributes": []},
        ]}
        content_json = {
            "$id": "sheets/data_sheet/data_table/data_table_storage.schema.json",
            "numberOfColumns": 4,
        }

        with tempfile.NamedTemporaryFile(suffix=".prism", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with zipfile.ZipFile(tmp_path, "w") as zf:
                zf.writestr("document.json", json.dumps(manifest))
                zf.writestr("data/sheets/test_sheet/sheet.json", json.dumps(sheet_json))
                zf.writestr("data/sets/ds_x.json", json.dumps(ds_x))
                zf.writestr("data/sets/ds_y1.json", json.dumps(ds_y))
                zf.writestr("data/tables/tbl_binary/content.json", json.dumps(content_json))
                zf.writestr("data/tables/tbl_binary/data.dt", binary_dt)

            doc = PrismIO.load(tmp_path)
            sheet = doc.get_sheet("test_sheet")
            self.assertIsNotNone(sheet)
            tbl = sheet.table
            self.assertEqual(tbl.num_rows, 2)
            self.assertEqual(tbl.x_title, "Concentration")
            self.assertEqual(tbl.get_cell_value(0, "X", 0, 0), 1.0)
            self.assertEqual(tbl.get_cell_value(0, "Y", 0, 0), 10.0)
            self.assertEqual(tbl.get_cell_value(0, "Y", 0, 1), 20.0)
            self.assertEqual(tbl.get_cell_value(0, "Y", 0, 2), 30.0)
            self.assertEqual(tbl.get_cell_value(1, "X", 0, 0), 2.0)
            self.assertEqual(tbl.get_cell_value(1, "Y", 0, 0), 15.0)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_cell_exclusion_parsing_formats(self) -> None:
        """Verifies cell exclusion detection across various GraphPad Prism attribute encodings."""
        # Row ranges
        self.assertEqual(PrismIO._parse_row_range("0~2"), [0, 1, 2])
        self.assertEqual(PrismIO._parse_row_range("3"), [3])
        self.assertEqual(PrismIO._parse_row_range("1-3"), [1, 2, 3])
        self.assertEqual(PrismIO._parse_row_range([4, 5, 6]), [4, 5, 6])
        self.assertEqual(PrismIO._parse_row_range(7), [7])

        # Exclusion attribute detection
        self.assertTrue(PrismIO._is_excluded_attribute({"attributes": ["EXCLUDED"]}))
        self.assertTrue(PrismIO._is_excluded_attribute({"attributes": ["1"]}))
        self.assertTrue(PrismIO._is_excluded_attribute({"Excluded": "1"}))
        self.assertTrue(PrismIO._is_excluded_attribute({"excluded": True}))
        self.assertTrue(PrismIO._is_excluded_attribute({"isExcluded": "true"}))
        self.assertFalse(PrismIO._is_excluded_attribute({"attributes": ["USER", "NUMERIC"]}))
        self.assertFalse(PrismIO._is_excluded_attribute({"attributes": []}))
        self.assertFalse(PrismIO._is_excluded_attribute({}))


class TestBlankCanvasAndSampleData(unittest.TestCase):
    """Verifies that GemGraph starts with a clean canvas and only loads samples on demand."""

    def test_main_window_starts_clean(self) -> None:
        """Ensures MainWindow opens with a pristine blank table (all NaN cells)."""
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtGui import QColor
        from src.ui.main_window import MainWindow
        from src.ui.spreadsheet import ReplicateSpreadsheet

        app = QApplication.instance()
        if app is None:
            app = QApplication([])

        window = MainWindow()
        data_sheets = window.document.get_sheets_by_type(SheetType.DATA)
        self.assertEqual(len(data_sheets), 1)

        table = data_sheets[0].table
        # Default empty XY layout: columns X, Group A [Y1]
        self.assertEqual(table.table_type, TableType.XY)
        self.assertEqual(table.num_replicates, 1)
        self.assertEqual(table.y_group_titles, ["Group A"])

        # Grid cells must be completely blank (all NaN)
        self.assertTrue(np.all(np.isnan(table.x_data)))
        self.assertTrue(np.all(np.isnan(table.y_data)))

        # Header data check
        header_x = window.spreadsheet_model.headerData(0, from_qt_orientation())
        header_y = window.spreadsheet_model.headerData(1, from_qt_orientation())
        self.assertEqual(header_x, "X")
        self.assertEqual(header_y, "Group A [Y1]")

        # High-contrast styling check under simulated dark theme
        idx = window.spreadsheet_model.index(0, 0)
        fg_brush = window.spreadsheet_model.data(idx, from_qt_foreground())
        bg_brush = window.spreadsheet_model.data(idx, from_qt_background())
        self.assertEqual(fg_brush.color(), QColor(0, 0, 0))
        self.assertEqual(bg_brush.color(), QColor(255, 255, 255))

        # Header high-contrast check
        hdr_fg = window.spreadsheet_model.headerData(0, from_qt_orientation(), from_qt_foreground())
        hdr_bg = window.spreadsheet_model.headerData(0, from_qt_orientation(), from_qt_background())
        self.assertEqual(hdr_fg.color(), QColor(0, 0, 0))
        self.assertEqual(hdr_bg.color(), QColor(240, 240, 240))

        # Verify ReplicateSpreadsheet alias exists
        self.assertIs(ReplicateSpreadsheet, type(window.spreadsheet_view))

        # Test action_add_table(TableType.XY) creates standard empty XY layout
        window.action_add_table(TableType.XY)
        active_sheet = window.document.get_sheet(window.document.active_sheet_id)
        self.assertEqual(active_sheet.table.table_type, TableType.XY)
        self.assertEqual(active_sheet.table.num_replicates, 1)
        self.assertEqual(active_sheet.table.y_group_titles, ["Group A"])

        # Test action_new_project maintains pristine blank table
        window.action_new_project()
        table_new = window.document.get_sheets_by_type(SheetType.DATA)[0].table
        self.assertTrue(np.all(np.isnan(table_new.x_data)))
        self.assertTrue(np.all(np.isnan(table_new.y_data)))

        # Test explicit sample data loading via action_load_sample_data
        window.action_load_sample_data()
        table_sample = window.document.get_sheets_by_type(SheetType.DATA)[0].table
        self.assertFalse(np.all(np.isnan(table_sample.x_data)))
        self.assertFalse(np.all(np.isnan(table_sample.y_data)))

        window.close()


def from_qt_orientation():
    from PyQt6.QtCore import Qt
    return Qt.Orientation.Horizontal


def from_qt_foreground():
    from PyQt6.QtCore import Qt
    return Qt.ItemDataRole.ForegroundRole


def from_qt_background():
    from PyQt6.QtCore import Qt
    return Qt.ItemDataRole.BackgroundRole



if __name__ == "__main__":
    unittest.main()
