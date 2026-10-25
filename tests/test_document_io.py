"""
Document Model & I/O Round-Trip Unit Tests for GemGraph.
Tests native .prism PK-ZIP archive serialization, Prism XML schemas,
and core spreadsheet data operations.
"""

import unittest
import tempfile
import os
import numpy as np
from src.core.document import PrismDocument
from src.core.sheet import DataSheet, InfoSheet, GraphSheet, SheetType
from src.core.table import DataTable, TableType
from src.core.undo import UndoStack, CellEditCommand, CellExcludeCommand
from src.io.prism_io import PrismIO
from src.io.xml_io import PrismXMLIO


class TestDataTableCore(unittest.TestCase):
    """Tests DataTable operations, replicates, cell exclusions, and summary extractions."""

    def test_datatable_creation_and_replicates(self) -> None:
        tbl = DataTable("tbl1", "Test Table", table_type=TableType.XY, num_rows=10, num_replicates=3)
        self.assertEqual(tbl.num_rows, 10)
        self.assertEqual(tbl.num_replicates, 3)
        self.assertEqual(len(tbl.y_group_titles), 2)  # Group A, Group B
        self.assertEqual(tbl.y_data.shape, (2, 10, 3))

        # Test setting values
        tbl.set_cell_value(0, "X", 0, 0, 1.5)
        tbl.set_cell_value(0, "Y", 0, 0, 10.2)
        tbl.set_cell_value(0, "Y", 0, 1, 10.5)
        tbl.set_cell_value(0, "Y", 0, 2, 10.3)

        self.assertEqual(tbl.get_cell_value(0, "X", 0, 0), 1.5)
        self.assertEqual(tbl.get_cell_value(0, "Y", 0, 1), 10.5)

        # Test expand replicates
        tbl.set_replicates(5)
        self.assertEqual(tbl.num_replicates, 5)
        self.assertEqual(tbl.y_data.shape, (2, 10, 5))
        self.assertEqual(tbl.get_cell_value(0, "Y", 0, 1), 10.5)

    def test_cell_exclusion_and_clean_xy(self) -> None:
        tbl = DataTable("tbl2", "Exclusion Test", table_type=TableType.XY, num_rows=4, num_replicates=2)
        for r in range(4):
            tbl.set_cell_value(r, "X", 0, 0, float(r + 1))
            tbl.set_cell_value(r, "Y", 0, 0, float(10 * (r + 1)))
            tbl.set_cell_value(r, "Y", 0, 1, float(10 * (r + 1) + 1))

        # Initially 4 rows * 2 replicates = 8 points
        xs, ys = tbl.get_clean_xy(0)
        self.assertEqual(len(xs), 8)

        # Exclude point at row 1, replicate 0
        tbl.set_excluded(1, "Y", 0, 0, True)
        self.assertTrue(tbl.is_excluded(1, "Y", 0, 0))

        xs_clean, ys_clean = tbl.get_clean_xy(0)
        self.assertEqual(len(xs_clean), 7)
        self.assertNotIn(20.0, ys_clean)

        # Exclude X at row 3 -> both replicates for row 3 must be excluded from clean XY
        tbl.set_excluded(3, "X", 0, 0, True)
        xs_clean2, ys_clean2 = tbl.get_clean_xy(0)
        self.assertEqual(len(xs_clean2), 5)


class TestDocumentIO(unittest.TestCase):
    """Tests serialization and deserialization across PK-ZIP (.prism) and XML formats."""

    def test_prism_zip_round_trip(self) -> None:
        """Verifies full round-trip save and load of .prism PK-ZIP archive."""
        doc = PrismDocument("Zip Project")
        sheet1 = doc.create_data_sheet("Dose Response", TableType.XY, num_rows=5, num_replicates=2)
        sheet1.table.set_cell_value(0, "X", 0, 0, 1.0)
        sheet1.table.set_cell_value(0, "Y", 0, 0, 25.5)
        sheet1.table.set_excluded(0, "Y", 0, 0, True)

        info = doc.create_info_sheet("Experiment Info")
        info.notes = "Testing PK-ZIP serialization integrity."

        graph = doc.create_graph_sheet("Graph 1", source_sheet_id=sheet1.sheet_id)

        with tempfile.NamedTemporaryFile(suffix=".prism", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            PrismIO.save(doc, tmp_path)
            self.assertTrue(os.path.exists(tmp_path))
            self.assertGreater(os.path.getsize(tmp_path), 0)

            # Load document back
            loaded_doc = PrismIO.load(tmp_path)
            self.assertEqual(loaded_doc.title, "Zip Project")
            self.assertEqual(len(loaded_doc.sheets), 3)

            loaded_sheet1 = loaded_doc.get_sheet(sheet1.sheet_id)
            self.assertIsInstance(loaded_sheet1, DataSheet)
            self.assertEqual(loaded_sheet1.table.get_cell_value(0, "X", 0, 0), 1.0)
            self.assertEqual(loaded_sheet1.table.get_cell_value(0, "Y", 0, 0), 25.5)
            self.assertTrue(loaded_sheet1.table.is_excluded(0, "Y", 0, 0))

            loaded_info = loaded_doc.get_sheet(info.sheet_id)
            self.assertIsInstance(loaded_info, InfoSheet)
            self.assertEqual(loaded_info.notes, "Testing PK-ZIP serialization integrity.")

            loaded_graph = loaded_doc.get_sheet(graph.sheet_id)
            self.assertIsInstance(loaded_graph, GraphSheet)
            self.assertEqual(loaded_graph.source_sheet_id, sheet1.sheet_id)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_xml_round_trip(self) -> None:
        """Verifies full round-trip export and import of Prism XML schema."""
        doc = PrismDocument("XML Project")
        sheet1 = doc.create_data_sheet("Table A", TableType.XY, num_rows=4, num_replicates=2)
        sheet1.table.set_cell_value(0, "X", 0, 0, 2.0)
        sheet1.table.set_cell_value(0, "Y", 0, 0, 42.0)
        sheet1.table.set_excluded(0, "Y", 0, 0, True)

        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            PrismXMLIO.export_xml(doc, tmp_path)
            self.assertTrue(os.path.exists(tmp_path))

            loaded_doc = PrismXMLIO.import_xml(tmp_path)
            sheets = loaded_doc.get_sheets_by_type(SheetType.DATA)
            self.assertGreaterEqual(len(sheets), 1)

            t = sheets[0].table
            self.assertEqual(t.get_cell_value(0, "X", 0, 0), 2.0)
            self.assertEqual(t.get_cell_value(0, "Y", 0, 0), 42.0)
            self.assertTrue(t.is_excluded(0, "Y", 0, 0))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestUndoRedoCommands(unittest.TestCase):
    """Verifies undo and redo stack operations."""

    def test_cell_edit_command(self) -> None:
        tbl = DataTable("tbl_undo", num_rows=5, num_replicates=1)
        stack = UndoStack()

        cmd = CellEditCommand(tbl, 0, "X", 0, 0, old_val=np.nan, new_val=100.0)
        stack.push(cmd)
        self.assertEqual(tbl.get_cell_value(0, "X", 0, 0), 100.0)

        # Undo
        self.assertTrue(stack.undo())
        self.assertTrue(np.isnan(tbl.get_cell_value(0, "X", 0, 0)))

        # Redo
        self.assertTrue(stack.redo())
        self.assertEqual(tbl.get_cell_value(0, "X", 0, 0), 100.0)

    def test_cell_exclude_command(self) -> None:
        tbl = DataTable("tbl_undo2", num_rows=5, num_replicates=1)
        stack = UndoStack()

        cmd = CellExcludeCommand(tbl, 1, "Y", 0, 0, old_excluded=False, new_excluded=True)
        stack.push(cmd)
        self.assertTrue(tbl.is_excluded(1, "Y", 0, 0))

        stack.undo()
        self.assertFalse(tbl.is_excluded(1, "Y", 0, 0))

        stack.redo()
        self.assertTrue(tbl.is_excluded(1, "Y", 0, 0))


    def test_datatable_remove_row_and_group(self) -> None:
        """Verifies row and group removal and exclusion mask synchronization."""
        tbl = DataTable("tbl_rem", num_rows=4, num_replicates=2)
        tbl.set_cell_value(0, "X", 0, 0, 1.0)
        tbl.set_cell_value(1, "X", 0, 0, 2.0)
        tbl.set_cell_value(2, "X", 0, 0, 3.0)
        tbl.set_cell_value(3, "X", 0, 0, 4.0)
        tbl.set_excluded(2, "X", 0, 0, True)

        # Remove row 1
        self.assertTrue(tbl.remove_row(1))
        self.assertEqual(tbl.num_rows, 3)
        self.assertEqual(tbl.get_cell_value(1, "X", 0, 0), 3.0)
        # Point previously at row 2 shifted to row 1
        self.assertTrue(tbl.is_excluded(1, "X", 0, 0))

        # Add and remove Y group
        g_idx = tbl.add_y_group("Group C")
        self.assertEqual(len(tbl.y_group_titles), 3)
        self.assertTrue(tbl.remove_y_group(g_idx))
        self.assertEqual(len(tbl.y_group_titles), 2)


class TestHeadlessUI(unittest.TestCase):
    """Verifies that the PyQt6 UI and vector graphics render cleanly in offscreen mode."""

    def test_main_window_instantiation(self) -> None:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        from PyQt6.QtWidgets import QApplication
        from src.ui.main_window import MainWindow

        app = QApplication.instance()
        if app is None:
            app = QApplication([])

        window = MainWindow()
        self.assertIsNotNone(window)
        self.assertEqual(window.windowTitle(), "Untitled - GemGraph - v0.1.2-nightly - Clean Room Edition")
        # Ensure 3-pane components are active
        self.assertIsNotNone(window.navigator)
        self.assertIsNotNone(window.center_stack)
        self.assertIsNotNone(window.graph_canvas)

        # Test adding all 5 table types
        window.action_add_table(TableType.COLUMN)
        window.action_add_table(TableType.CONTINGENCY)
        window.action_add_table(TableType.SURVIVAL)
        self.assertEqual(len(window.document.get_sheets_by_type(SheetType.DATA)), 4)
        window.close()

    def test_graph_canvas_svg_and_png_export(self) -> None:
        """Verifies vector SVG and raster PNG export from GraphCanvasWidget."""
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        from PyQt6.QtWidgets import QApplication
        from src.ui.graph_canvas import GraphCanvasWidget

        app = QApplication.instance()
        if app is None:
            app = QApplication([])

        gc = GraphCanvasWidget()
        tbl = DataTable("tbl_export", table_type=TableType.XY, num_rows=5, num_replicates=2)
        for r in range(5):
            tbl.set_cell_value(r, "X", 0, 0, float(r + 1))
            tbl.set_cell_value(r, "Y", 0, 0, float(10 * (r + 1)))
            tbl.set_cell_value(r, "Y", 0, 1, float(10 * (r + 1) + 2))
        gc.update_data(tbl)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_png:
            png_path = tmp_png.name
        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp_svg:
            svg_path = tmp_svg.name

        try:
            gc.export_png(png_path)
            self.assertTrue(os.path.exists(png_path))
            self.assertGreater(os.path.getsize(png_path), 0)

            gc.export_svg(svg_path)
            self.assertTrue(os.path.exists(svg_path))
            self.assertGreater(os.path.getsize(svg_path), 0)

            # Validate SVG content
            with open(svg_path, "r", encoding="utf-8") as f:
                content = f.read()
                self.assertIn("<svg", content)
        finally:
            if os.path.exists(png_path):
                os.remove(png_path)
            if os.path.exists(svg_path):
                os.remove(svg_path)


if __name__ == "__main__":
    unittest.main()
