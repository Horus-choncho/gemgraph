"""
Comprehensive Unit Tests for Rich Text Formatting, Granular Office & PDF Export Engine,
Per-Graph Bar Scatter Point Toggling, and Project Navigator Context Menus.
"""

from __future__ import annotations
import os
import tempfile
import unittest
import zipfile
import xml.etree.ElementTree as ET
import numpy as np

from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QKeyEvent, QTextCursor, QTextListFormat
from PyQt6.QtWidgets import QApplication, QTreeWidgetItem, QMenu

from src.core.document import PrismDocument
from src.core.sheet import DataSheet, GraphSheet, InfoSheet, ResultSheet, SheetType, GraphType
from src.core.table import TableType, DataTable
from src.core.office_exporter import OfficeExporter, ExportFormat, render_graph_to_png_bytes
from src.ui.export_dialog import ExportProjectDialog
from src.ui.rich_text_editor import RichTextEditor
from src.ui.info_sheet_view import InfoSheetView
from src.ui.graph_canvas import GraphCanvasWidget
from src.ui.main_window import MainWindow

app = QApplication.instance() or QApplication([])


class TestRichTextEditorAndPersistence(unittest.TestCase):
    """Tests RichTextEditor auto-formatting (- + Space, Tab/Shift+Tab), toolbars, and persistence."""

    def setUp(self) -> None:
        self.editor_widget = RichTextEditor()
        self.edit = self.editor_widget.editor

    def test_auto_bullet_conversion(self) -> None:
        """Typing '- ' at the start of a block converts to a bulleted list."""
        self.edit.setPlainText("-")
        cursor = self.edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.EndOfBlock)
        self.edit.setTextCursor(cursor)

        space_event = QKeyEvent(
            QKeyEvent.Type.KeyPress,
            Qt.Key.Key_Space,
            Qt.KeyboardModifier.NoModifier,
            " ",
        )
        self.edit.keyPressEvent(space_event)
        self.assertIsNotNone(self.edit.textCursor().currentList())

    def test_tab_indentation_increase_and_decrease(self) -> None:
        """Tab increases indentation level; Shift+Tab decreases it."""
        self.edit.setPlainText("First line of notes")
        cursor = self.edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.Start)
        self.edit.setTextCursor(cursor)

        self.assertEqual(cursor.blockFormat().indent(), 0)

        # Simulate Tab
        tab_event = QKeyEvent(
            QKeyEvent.Type.KeyPress,
            Qt.Key.Key_Tab,
            Qt.KeyboardModifier.NoModifier,
            "\t",
        )
        self.edit.keyPressEvent(tab_event)
        self.assertEqual(self.edit.textCursor().blockFormat().indent(), 1)

        # Simulate second Tab
        self.edit.keyPressEvent(tab_event)
        self.assertEqual(self.edit.textCursor().blockFormat().indent(), 2)

        # Simulate Shift+Tab
        shift_tab_event = QKeyEvent(
            QKeyEvent.Type.KeyPress,
            Qt.Key.Key_Backtab,
            Qt.KeyboardModifier.ShiftModifier,
            "",
        )
        self.edit.keyPressEvent(shift_tab_event)
        self.assertEqual(self.edit.textCursor().blockFormat().indent(), 1)

        # Simulate Shift+Tab down to 0
        self.edit.keyPressEvent(shift_tab_event)
        self.assertEqual(self.edit.textCursor().blockFormat().indent(), 0)

        # Additional Shift+Tab should not go negative
        self.edit.keyPressEvent(shift_tab_event)
        self.assertEqual(self.edit.textCursor().blockFormat().indent(), 0)

    def test_font_family_and_size_switching(self) -> None:
        """Selecting typeface and size changes format on text cursor."""
        self.edit.setPlainText("Sample text for font test")
        cursor = self.edit.textCursor()
        cursor.select(QTextCursor.SelectionType.Document)
        self.edit.setTextCursor(cursor)

        # Switch to Serif
        self.editor_widget.combo_font.setCurrentText("Serif")
        fmt = self.edit.currentCharFormat()
        self.assertIn("serif", fmt.font().family().lower())

        # Switch to Monospace
        self.editor_widget.combo_font.setCurrentText("Monospace")
        fmt = self.edit.currentCharFormat()
        self.assertTrue("mono" in fmt.font().family().lower() or "courier" in fmt.font().family().lower())

        # Switch to Sans-Serif
        self.editor_widget.combo_font.setCurrentText("Sans-Serif")
        fmt = self.edit.currentCharFormat()
        self.assertIn("sans", fmt.font().family().lower())

        # Switch sizes: Small (10pt), Medium (12pt), Large (16pt)
        self.editor_widget.combo_size.setCurrentText("Small (10pt)")
        self.assertEqual(self.edit.currentCharFormat().fontPointSize(), 10.0)

        self.editor_widget.combo_size.setCurrentText("Large (16pt)")
        self.assertEqual(self.edit.currentCharFormat().fontPointSize(), 16.0)

        self.editor_widget.combo_size.setCurrentText("Medium (12pt - Default)")
        self.assertEqual(self.edit.currentCharFormat().fontPointSize(), 12.0)

    def test_style_toggles(self) -> None:
        """Bold, Italic, Underline, and Bullet List buttons toggle formatting."""
        self.edit.setPlainText("Styling target")
        cursor = self.edit.textCursor()
        cursor.select(QTextCursor.SelectionType.Document)
        self.edit.setTextCursor(cursor)

        # Bold
        self.editor_widget.btn_bold.click()
        self.assertTrue(self.editor_widget.btn_bold.isChecked())

        # Italic
        self.editor_widget.btn_italic.click()
        self.assertTrue(self.editor_widget.btn_italic.isChecked())

        # Underline
        self.editor_widget.btn_underline.click()
        self.assertTrue(self.editor_widget.btn_underline.isChecked())

        # Bullet list toggle
        self.editor_widget.btn_bullet.click()
        self.assertIsNotNone(self.edit.textCursor().currentList())

    def test_info_sheet_rich_text_persistence(self) -> None:
        """InfoSheet saves and loads HTML and plain text notes across serialization."""
        sheet = InfoSheet(sheet_id="info_rt", title="Rich Info")
        view = InfoSheetView()
        view.display_info(sheet)

        view.notes_editor.setHtml("<p><b>Important Assay Finding:</b></p><ul><li>Inhibition = 95%</li></ul>")
        self.assertEqual(sheet.notes_html, view.notes_editor.toHtml())
        self.assertIn("Inhibition = 95%", sheet.notes)

        # Dictionary serialization round-trip
        d = sheet.to_dict()
        self.assertIn("notes_html", d)
        self.assertIn("Inhibition = 95%", d["notes_html"])

        restored = InfoSheet.from_dict(d)
        self.assertEqual(restored.sheet_id, "info_rt")
        self.assertEqual(restored.notes_html, sheet.notes_html)
        self.assertEqual(restored.notes, sheet.notes)


class TestOfficeExportEngine(unittest.TestCase):
    """Tests zero-subprocess .docx, .odt, .xlsx, .ods, and .pdf document generation."""

    def setUp(self) -> None:
        self.doc = PrismDocument("Test Export Project")

        # 1. Info Sheet
        self.info = self.doc.create_info_sheet(title="Protocol Specifications")
        self.info.set_field_value("experiment_title", "Kinase Assay Dose Response")
        self.info.set_field_value("author_scientist", "Dr. A. Researcher")
        self.info.notes = "- Incubated 24h\n- Stopped with reagent B"
        self.info.notes_html = "<p>• Incubated 24h</p><p>• Stopped with reagent B</p>"

        # 2. Data Sheet (XY)
        self.data_sheet = self.doc.create_data_sheet(
            title="Dose Response Data",
            table_type=TableType.XY,
            num_rows=5,
            num_replicates=2,
            y_group_titles=["Vehicle", "Compound X"],
        )
        self.data_sheet.table.x_columns[0]["data"] = np.array([0.1, 1.0, 10.0, 100.0, 1000.0])
        self.data_sheet.table.y_data[0, :, 0] = np.array([5.2, 12.1, 45.3, 85.0, 98.2])
        self.data_sheet.table.y_data[0, :, 1] = np.array([4.9, 11.8, 47.0, 83.5, 99.1])
        self.data_sheet.table.y_data[1, :, 0] = np.array([2.1, 5.0, 22.3, 61.2, 91.0])
        self.data_sheet.table.y_data[1, :, 1] = np.array([2.3, 5.2, 21.8, 62.0, 90.5])

        # 3. Result Sheet
        self.res_sheet = self.doc.create_result_sheet(
            title="Nonlinear Fit Results",
            analysis_type="Nonlinear Regression",
            source_sheet_id=self.data_sheet.sheet_id,
        )
        self.res_sheet.add_tab(
            name="Parameters",
            headers=["Parameter", "Vehicle", "Compound X"],
            rows=[
                ["logIC50", "1.12 ± 0.05", "1.65 ± 0.08"],
                ["HillSlope", "1.05", "0.98"],
                ["R squared", "0.998", "0.995"],
            ],
        )

        # 4. Graph Sheet
        self.graph_sheet = self.doc.create_graph_sheet(
            title="Dose Response Curve",
            source_sheet_id=self.data_sheet.sheet_id,
            graph_type=GraphType.COLUMN_BAR_ERROR,
        )

        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_docx_export_validity(self) -> None:
        """Word document (.docx) builds valid ZIP archive with document.xml and styles."""
        out_path = os.path.join(self.tmp_dir, "report.docx")
        exporter = OfficeExporter(self.doc)
        exporter.export(out_path, fmt=ExportFormat.DOCX)

        self.assertTrue(os.path.exists(out_path))
        self.assertGreater(os.path.getsize(out_path), 0)

        with zipfile.ZipFile(out_path, "r") as zf:
            namelist = zf.namelist()
            self.assertIn("[Content_Types].xml", namelist)
            self.assertIn("word/document.xml", namelist)
            self.assertIn("word/styles.xml", namelist)
            self.assertIn("word/_rels/document.xml.rels", namelist)

            # Parse document.xml to verify valid XML
            doc_xml = zf.read("word/document.xml")
            root = ET.fromstring(doc_xml)
            self.assertIsNotNone(root)
            self.assertIn("Dose Response Data", doc_xml.decode("utf-8"))

    def test_odt_export_validity(self) -> None:
        """OpenDocument Text (.odt) builds valid OASIS package with content.xml and manifest."""
        out_path = os.path.join(self.tmp_dir, "report.odt")
        exporter = OfficeExporter(self.doc)
        exporter.export(out_path, fmt=ExportFormat.ODT)

        self.assertTrue(os.path.exists(out_path))
        self.assertGreater(os.path.getsize(out_path), 0)

        with zipfile.ZipFile(out_path, "r") as zf:
            namelist = zf.namelist()
            # Mimetype must be the first file and uncompressed
            self.assertEqual(namelist[0], "mimetype")
            self.assertEqual(zf.read("mimetype").decode("utf-8"), "application/vnd.oasis.opendocument.text")
            self.assertIn("META-INF/manifest.xml", namelist)
            self.assertIn("content.xml", namelist)
            self.assertIn("styles.xml", namelist)

            # Parse content.xml
            content_xml = zf.read("content.xml")
            root = ET.fromstring(content_xml)
            self.assertIsNotNone(root)
            self.assertIn("Dose Response Data", content_xml.decode("utf-8"))

    def test_xlsx_export_validity(self) -> None:
        """Excel workbook (.xlsx) builds separate worksheets for data and results."""
        out_path = os.path.join(self.tmp_dir, "tables.xlsx")
        exporter = OfficeExporter(self.doc)
        exporter.export(out_path, fmt=ExportFormat.XLSX)

        self.assertTrue(os.path.exists(out_path))
        self.assertGreater(os.path.getsize(out_path), 0)

        with zipfile.ZipFile(out_path, "r") as zf:
            namelist = zf.namelist()
            self.assertIn("[Content_Types].xml", namelist)
            self.assertIn("xl/workbook.xml", namelist)
            self.assertIn("xl/worksheets/sheet1.xml", namelist)

            # Parse workbook.xml
            wb_xml = zf.read("xl/workbook.xml")
            root = ET.fromstring(wb_xml)
            self.assertIsNotNone(root)
            self.assertIn("Dose Response Data", wb_xml.decode("utf-8"))

    def test_ods_export_validity(self) -> None:
        """OpenDocument Spreadsheet (.ods) builds valid OASIS spreadsheet package."""
        out_path = os.path.join(self.tmp_dir, "tables.ods")
        exporter = OfficeExporter(self.doc)
        exporter.export(out_path, fmt=ExportFormat.ODS)

        self.assertTrue(os.path.exists(out_path))
        self.assertGreater(os.path.getsize(out_path), 0)

        with zipfile.ZipFile(out_path, "r") as zf:
            namelist = zf.namelist()
            self.assertEqual(namelist[0], "mimetype")
            self.assertEqual(zf.read("mimetype").decode("utf-8"), "application/vnd.oasis.opendocument.spreadsheet")
            self.assertIn("META-INF/manifest.xml", namelist)
            self.assertIn("content.xml", namelist)

            # Parse content.xml
            content_xml = zf.read("content.xml")
            root = ET.fromstring(content_xml)
            self.assertIsNotNone(root)

    def test_pdf_export_validity(self) -> None:
        """PDF report renders publication layout with %PDF- header."""
        out_path = os.path.join(self.tmp_dir, "report.pdf")
        exporter = OfficeExporter(self.doc)
        exporter.export(out_path, fmt=ExportFormat.PDF)

        self.assertTrue(os.path.exists(out_path))
        self.assertGreater(os.path.getsize(out_path), 1000)

        with open(out_path, "rb") as f:
            header = f.read(5)
            self.assertEqual(header, b"%PDF-")

    def test_granular_item_selection(self) -> None:
        """Exporting only selected sheet IDs omits unselected sheets."""
        out_path = os.path.join(self.tmp_dir, "data_only.xlsx")
        exporter = OfficeExporter(self.doc)
        # Select only the DataSheet
        exporter.export(out_path, fmt=ExportFormat.XLSX, selected_sheet_ids=[self.data_sheet.sheet_id])

        with zipfile.ZipFile(out_path, "r") as zf:
            wb_xml = zf.read("xl/workbook.xml").decode("utf-8")
            self.assertIn("Dose Response Data", wb_xml)
            self.assertNotIn("Nonlinear Fit Results", wb_xml)
            self.assertNotIn("Info Sheets", wb_xml)

    def test_export_project_dialog_interaction(self) -> None:
        """ExportProjectDialog scope checklist and selection behavior."""
        dialog = ExportProjectDialog(self.doc)
        # Initially all selected
        selected_initial = dialog.get_selected_sheet_ids()
        self.assertEqual(len(selected_initial), len(self.doc.sheet_order))

        # Deselect all
        dialog._deselect_all()
        self.assertEqual(len(dialog.get_selected_sheet_ids()), 0)

        # Select all again
        dialog._select_all()
        self.assertEqual(len(dialog.get_selected_sheet_ids()), len(self.doc.sheet_order))


class TestBarScatterPointsToggle(unittest.TestCase):
    """Tests per-graph show_bar_scatter_points toggle on column and grouped bar charts."""

    def setUp(self) -> None:
        self.canvas = GraphCanvasWidget()
        self.table = DataTable(
            table_id="tbl_test",
            title="Bar Table",
            table_type=TableType.COLUMN,
            num_rows=4,
            num_replicates=1,
            y_group_titles=["Control", "Treated"],
        )
        self.table.y_data[0, :, 0] = np.array([10.0, 11.5, 9.8, 10.2])
        self.table.y_data[1, :, 0] = np.array([25.0, 27.2, 24.1, 26.5])

    def test_column_bar_scatter_toggle(self) -> None:
        """Toggling show_bar_scatter_points adds/removes ScatterPlotItem on column bar chart."""
        import pyqtgraph as pg

        self.canvas.update_data(self.table, graph_type=GraphType.COLUMN_BAR_ERROR)

        # Scatter points toggle button must be visible for bar chart
        self.assertFalse(self.canvas.btn_toggle_scatter.isHidden())
        self.assertTrue(self.canvas.btn_toggle_scatter.isEnabled())
        self.assertTrue(self.canvas.show_bar_scatter_points)

        # Count ScatterPlotItem in plot items
        scatter_items = [it for it in self.canvas.plot_widget.items() if isinstance(it, pg.ScatterPlotItem)]
        self.assertGreater(len(scatter_items), 0, "ScatterPlotItem missing when show_bar_scatter_points=True")

        # Toggle OFF
        self.canvas.set_show_bar_scatter_points(False)
        self.assertFalse(self.canvas.show_bar_scatter_points)
        self.assertFalse(self.canvas.btn_toggle_scatter.isChecked())

        scatter_items_off = [it for it in self.canvas.plot_widget.items() if isinstance(it, pg.ScatterPlotItem)]
        self.assertEqual(len(scatter_items_off), 0, "ScatterPlotItem still present when show_bar_scatter_points=False")

        # Bars and error items should remain
        bars = [it for it in self.canvas.plot_widget.items() if isinstance(it, pg.BarGraphItem)]
        self.assertGreater(len(bars), 0, "BarGraphItem removed when turning off scatter points")

        # Toggle ON again
        self.canvas.set_show_bar_scatter_points(True)
        scatter_items_on = [it for it in self.canvas.plot_widget.items() if isinstance(it, pg.ScatterPlotItem)]
        self.assertGreater(len(scatter_items_on), 0)

    def test_grouped_bar_scatter_toggle(self) -> None:
        """Toggling show_bar_scatter_points on grouped bar charts."""
        import pyqtgraph as pg

        grp_table = DataTable(
            table_id="grp_tbl",
            title="Grouped Table",
            table_type=TableType.GROUPED,
            num_rows=2,
            num_replicates=3,
            y_group_titles=["Genotype WT", "Genotype KO"],
        )
        grp_table.y_data[0, :, :] = np.array([[1.0, 1.2, 1.1], [2.0, 2.1, 1.9]])
        grp_table.y_data[1, :, :] = np.array([[3.0, 3.2, 3.1], [4.0, 4.1, 3.9]])

        self.canvas.update_data(grp_table, graph_type=GraphType.GROUPED_BARS_POINTS)
        self.canvas.set_show_bar_scatter_points(False)
        scatter_items_off = [it for it in self.canvas.plot_widget.items() if isinstance(it, pg.ScatterPlotItem)]
        self.assertEqual(len(scatter_items_off), 0)

        self.canvas.set_show_bar_scatter_points(True)
        scatter_items_on = [it for it in self.canvas.plot_widget.items() if isinstance(it, pg.ScatterPlotItem)]
        self.assertGreater(len(scatter_items_on), 0)

    def test_graph_sheet_plot_config_persistence(self) -> None:
        """Per-graph show_bar_scatter_points setting persists in plot_config of GraphSheet."""
        graph_sheet = GraphSheet(
            sheet_id="g_test",
            title="Bar Graph Sheet",
            graph_type=GraphType.COLUMN_BAR_ERROR,
            plot_config={"show_bar_scatter_points": False},
        )
        self.canvas.update_data(self.table, graph_type=GraphType.COLUMN_BAR_ERROR, sheet=graph_sheet)

        self.assertFalse(self.canvas.show_bar_scatter_points)
        self.assertFalse(self.canvas.btn_toggle_scatter.isChecked())

        # Toggle on
        self.canvas.set_show_bar_scatter_points(True)
        self.assertTrue(graph_sheet.plot_config["show_bar_scatter_points"])


class TestProjectNavigatorContextMenu(unittest.TestCase):
    """Tests right-click '➕ Add Sheet' context menu on Data Tables in the Project Navigator."""

    def setUp(self) -> None:
        self.window = MainWindow()

    def tearDown(self) -> None:
        self.window.close()

    def test_right_click_data_tables_root_adds_sheet(self) -> None:
        """Right-clicking 'Data Tables' root node displays context menu with '➕ Add Sheet'."""
        nav = self.window.navigator
        data_root = nav.folder_items[SheetType.DATA]
        self.assertIsNotNone(data_root)

        initial_sheet_count = len(self.window.document.get_sheets_by_type(SheetType.DATA))

        # Position of data_root item in tree
        rect = nav.visualItemRect(data_root)
        pos = rect.center()

        # Build context menu via helper
        menu = self.window._create_navigator_context_menu(pos)
        self.assertIsNotNone(menu, "Context menu was not created for Data Tables root")
        action_texts = [a.text() for a in menu.actions()]
        self.assertTrue(any("Add Sheet" in t for t in action_texts), f"'Add Sheet' missing from: {action_texts}")

        # Trigger adding a sheet via action_add_table
        self.window.action_add_table(TableType.XY)

        # Verify new sheet was added and selected
        new_count = len(self.window.document.get_sheets_by_type(SheetType.DATA))
        self.assertEqual(new_count, initial_sheet_count + 1)
        self.assertEqual(self.window.center_stack.currentWidget(), self.window.spreadsheet_view)

    def test_right_click_child_data_sheet_has_actions(self) -> None:
        """Right-clicking an existing data sheet provides Add Sheet, Rename, and Delete."""
        data_sheets = self.window.document.get_sheets_by_type(SheetType.DATA)
        self.assertTrue(len(data_sheets) > 0)
        target_sheet = data_sheets[0]

        # Find item in tree
        nav = self.window.navigator
        data_root = nav.folder_items[SheetType.DATA]
        child_item = None
        for i in range(data_root.childCount()):
            item = data_root.child(i)
            if item.data(0, Qt.ItemDataRole.UserRole) == target_sheet.sheet_id:
                child_item = item
                break

        self.assertIsNotNone(child_item)
        rect = nav.visualItemRect(child_item)
        pos = rect.center()

        menu = self.window._create_navigator_context_menu(pos)
        self.assertIsNotNone(menu, "Context menu was not created for child Data Sheet")
        action_texts = [a.text() for a in menu.actions()]
        self.assertTrue(any("Add Sheet" in t for t in action_texts), f"'Add Sheet' missing from: {action_texts}")
        self.assertTrue(any("Rename Sheet" in t for t in action_texts), f"'Rename Sheet' missing from: {action_texts}")
        self.assertTrue(any("Delete Sheet" in t for t in action_texts), f"'Delete Sheet' missing from: {action_texts}")


if __name__ == "__main__":
    unittest.main()
