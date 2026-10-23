"""
Comprehensive Unit & Integration Test Suite for Page Layouts, Depth Graphs (2.5D Isometric),
and High-DPI Publication Exporter Engine in GemGraph.

Strictly enforces 'No-Sudo-Strict-Local' policy: pure user-space Python, zero external binaries,
and zero subprocess/os.system calls.
"""

from __future__ import annotations
import math
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import zipfile

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, QEvent
from PyQt6.QtGui import QColor, QImage, QMouseEvent, QKeyEvent, QFocusEvent, QPainter
from PyQt6.QtWidgets import QApplication, QGraphicsItem, QWidget

from src.core.document import PrismDocument
from src.core.table import TableType, DataTable
from src.core.layout import (
    PageLayout,
    Placeholder,
    align_placeholders,
    create_template_placeholders,
    equalize_placeholders,
    render_layout_to_image,
    render_layout_to_pdf,
    render_layout_to_png_bytes,
    render_layout_to_svg,
)
from src.core.office_exporter import OfficeExporter, ExportFormat, render_graph_to_png_bytes
from src.core.sheet import DataSheet, GraphSheet, GraphType, SheetType
from src.core.table import TableType
from src.io.prism_io import PrismIO
from src.ui.graph_canvas import ScientificPlotCanvas
from src.ui.layout_canvas import LayoutCanvas, GraphPlaceholderItem
from src.ui.layout_view import LayoutView

app = QApplication.instance()
if app is None:
    app = QApplication([])


class TestPageLayoutsAndDepthGraphs(unittest.TestCase):
    def setUp(self) -> None:
        self.doc = PrismDocument()
        self.doc.create_data_sheet(
            "Test Column Data",
            table_type=TableType.COLUMN,
            y_group_titles=["Control", "Low Dose", "High Dose"],
            num_replicates=1,
        )
        ds = self.doc.sheets[self.doc.sheet_order[0]]
        self.data_sheet = ds
        # Populate column data
        col_vals = [
            [10.5, 20.2, 35.1],
            [12.1, 19.8, 33.4],
            [11.3, 21.0, 34.8],
        ]
        for r, row in enumerate(col_vals):
            for g, val in enumerate(row):
                ds.table.set_cell_value(r, "Y", g, 0, val)

        # Add a graph sheet
        self.graph_sheet = self.doc.create_graph_sheet("Test Graph", source_sheet_id=ds.sheet_id)
        self.graph_sheet.graph_type = GraphType.DEPTH_GRAPH

    def test_placeholder_data_model(self) -> None:
        """Verifies Placeholder initialization, normalization, and round-trip serialization."""
        p = Placeholder(slot_id="slot_1", rect=(0.05, 0.1, 0.4, 0.35), graph_id="graph_123")
        self.assertEqual(p.slot_id, "slot_1")
        self.assertEqual(p.rect, (0.05, 0.1, 0.4, 0.35))
        self.assertEqual(p.graph_id, "graph_123")
        self.assertTrue(p.aspect_ratio_locked)

        # Clamping and normalization check
        p_invalid = Placeholder(slot_id="slot_inv", rect=(-0.5, 1.5, 2.0, 0.0))
        self.assertEqual(p_invalid.rect[0], 0.0)
        self.assertEqual(p_invalid.rect[1], 1.0)
        self.assertEqual(p_invalid.rect[2], 1.0)
        self.assertEqual(p_invalid.rect[3], 0.01)

        # Round-trip dictionary serialization
        data = p.to_dict()
        p_restored = Placeholder.from_dict(data)
        self.assertEqual(p_restored.slot_id, p.slot_id)
        self.assertEqual(p_restored.rect, p.rect)
        self.assertEqual(p_restored.graph_id, p.graph_id)
        self.assertEqual(p_restored.aspect_ratio_locked, p.aspect_ratio_locked)

    def test_page_layout_dimensions(self) -> None:
        """Verifies physical page dimensions in inches, points, and pixels across formats and orientations."""
        # A4 Portrait: 8.27 x 11.69 in
        l_a4_p = PageLayout(name="A4 Portrait", page_format="A4", orientation="portrait")
        in_w, in_h = l_a4_p.get_dimensions_in()
        self.assertAlmostEqual(in_w, 8.27, places=2)
        self.assertAlmostEqual(in_h, 11.69, places=2)

        # A4 Landscape: 11.69 x 8.27 in
        l_a4_l = PageLayout(name="A4 Landscape", page_format="A4", orientation="landscape")
        in_w, in_h = l_a4_l.get_dimensions_in()
        self.assertAlmostEqual(in_w, 11.69, places=2)
        self.assertAlmostEqual(in_h, 8.27, places=2)

        # Points (1 in = 72 pt)
        pt_w, pt_h = l_a4_p.get_dimensions_pt()
        self.assertAlmostEqual(pt_w, 595.28, places=1)
        self.assertAlmostEqual(pt_h, 841.89, places=1)

        # Pixels at 300 DPI
        px_w, px_h = l_a4_p.get_dimensions_px(dpi=300)
        self.assertEqual(px_w, int(round(8.27 * 300)))
        self.assertEqual(px_h, int(round(11.69 * 300)))

        # Letter Landscape: 11.0 x 8.5 in
        l_let = PageLayout(name="Letter", page_format="Letter", orientation="landscape")
        in_w, in_h = l_let.get_dimensions_in()
        self.assertEqual(in_w, 11.0)
        self.assertEqual(in_h, 8.5)

    def test_template_presets_and_coordinate_bounds(self) -> None:
        """Verifies all grid templates generate valid placeholders strictly bounded in [0, 1]."""
        for preset in ["1", "2 horizontal", "2 vertical", "3", "4", "6", "8"]:
            placeholders = create_template_placeholders(preset)
            self.assertGreater(len(placeholders), 0, f"Preset {preset} generated no placeholders")
            for idx, p in enumerate(placeholders):
                x, y, w, h = p.rect
                self.assertGreaterEqual(x, 0.0, f"{preset} p{idx} x < 0")
                self.assertGreaterEqual(y, 0.0, f"{preset} p{idx} y < 0")
                self.assertGreater(w, 0.0, f"{preset} p{idx} w <= 0")
                self.assertGreater(h, 0.0, f"{preset} p{idx} h <= 0")
                self.assertLessEqual(x + w, 1.0001, f"{preset} p{idx} right edge exceeds 1.0")
                self.assertLessEqual(y + h, 1.0001, f"{preset} p{idx} bottom edge exceeds 1.0")

    def test_placeholder_graph_assignment(self) -> None:
        """Verifies assigning, unassigning, and replacing graphs on placeholders."""
        layout = self.doc.create_layout("Test Layout", template_preset="4")
        self.assertEqual(len(layout.placeholders), 4)

        # Initially unassigned
        self.assertIsNone(layout.placeholders[0].graph_id)

        # Assign graph
        layout.placeholders[0].graph_id = self.graph_sheet.sheet_id
        self.assertEqual(layout.placeholders[0].graph_id, self.graph_sheet.sheet_id)

        # Replace graph with another ID
        layout.placeholders[0].graph_id = "new_graph_id"
        self.assertEqual(layout.placeholders[0].graph_id, "new_graph_id")

        # Unassign
        layout.placeholders[0].graph_id = None
        self.assertIsNone(layout.placeholders[0].graph_id)

    def test_alignment_and_equalization_math(self) -> None:
        """Tests mathematical alignment and equalization routines for placeholders."""
        p1 = Placeholder(slot_id="p1", rect=(0.10, 0.20, 0.30, 0.25))
        p2 = Placeholder(slot_id="p2", rect=(0.50, 0.30, 0.20, 0.40))
        p3 = Placeholder(slot_id="p3", rect=(0.60, 0.15, 0.15, 0.20))

        # Align Lefts: all should have x matching p1 (0.10)
        align_placeholders([p1, p2, p3], mode="left")
        self.assertAlmostEqual(p2.rect[0], 0.10)
        self.assertAlmostEqual(p3.rect[0], 0.10)

        # Align Centers: align horizontal centers
        p2.rect = (0.50, 0.30, 0.20, 0.40)
        align_placeholders([p1, p2], mode="center")
        self.assertAlmostEqual(p1.rect[0] + p1.rect[2] / 2.0, p2.rect[0] + p2.rect[2] / 2.0)

        # Align Tops: all should have y matching p1 (0.20)
        p2.rect = (0.50, 0.30, 0.20, 0.40)
        align_placeholders([p1, p2], mode="top")
        self.assertAlmostEqual(p2.rect[1], 0.20)

        # Align Middles: align vertical centers
        p2.rect = (0.50, 0.50, 0.20, 0.40)
        align_placeholders([p1, p2], mode="middle")
        self.assertAlmostEqual(p1.rect[1] + p1.rect[3] / 2.0, p2.rect[1] + p2.rect[3] / 2.0)

        # Equalize Widths: p2 should match p1 width (0.30)
        equalize_placeholders([p1, p2], mode="width")
        self.assertAlmostEqual(p2.rect[2], 0.30)

        # Equalize Heights: p2 should match p1 height (0.25)
        equalize_placeholders([p1, p2], mode="height")
        self.assertAlmostEqual(p2.rect[3], 0.25)

        # Equalize Both
        p3.rect = (0.2, 0.2, 0.1, 0.1)
        equalize_placeholders([p1, p3], mode="both")
        self.assertAlmostEqual(p3.rect[2], 0.30)
        self.assertAlmostEqual(p3.rect[3], 0.25)

    def test_depth_graph_reverse_engineered_parity(self) -> None:
        """Verifies authentic GraphPad Prism Depth Graph profile rendering without 2.5D math."""
        # Create an XY dataset representing depth measurements (depth in meters vs oxygen pO2)
        xy_sheet = self.doc.create_data_sheet("Depth Oxygen", table_type=TableType.XY, num_replicates=2)
        xy_sheet.table.x_title = "Depth in meters"
        xy_sheet.table.y_group_titles = ["pO2 (Control)", "pO2 (Treated)"]
        depths = [0.0, 2.0, 4.0, 6.0]
        xy_sheet.table.x_data = np.array(depths)
        for r, d in enumerate(depths):
            # Control group
            xy_sheet.table.set_cell_value(r, "Y", 0, 0, 100.0 - d * 5.0)
            xy_sheet.table.set_cell_value(r, "Y", 0, 1, 105.0 - d * 5.0)
            # Treated group
            xy_sheet.table.set_cell_value(r, "Y", 1, 0, 90.0 - d * 7.0)
            xy_sheet.table.set_cell_value(r, "Y", 1, 1, 94.0 - d * 7.0)

        canvas = ScientificPlotCanvas(parent=None)
        canvas.set_error_style("SEM")
        canvas.set_whisker_direction("Both (±)")
        canvas.update_data(xy_sheet.table, graph_type=GraphType.DEPTH_GRAPH, sheet=self.graph_sheet)

        # 1. Y-axis must be inverted (depth 0 at top, increasing downwards)
        self.assertTrue(canvas.plot_widget.getViewBox().yInverted())

        # 2. Both series must be plotted with error bars and scatter items
        items = canvas.plot_widget.plotItem.items
        import pyqtgraph as pg
        error_items = [it for it in items if isinstance(it, pg.ErrorBarItem)]
        scatter_items = [it for it in items if isinstance(it, pg.ScatterPlotItem)]
        self.assertGreaterEqual(len(error_items), 2)
        self.assertGreaterEqual(len(scatter_items), 2)

        # 3. Test Upper (+) Only whisker configuration
        canvas.set_whisker_direction("Upper (+) Only")
        canvas.refresh_plot()
        self.assertEqual(canvas.whisker_direction, "Upper (+) Only")

    def test_depth_graph_canvas_rendering(self) -> None:
        """Verifies ScientificPlotCanvas renders authentic Depth Graph without 2.5D math."""
        canvas = ScientificPlotCanvas(parent=None)

        # Render for Column table
        canvas.update_data(self.data_sheet.table, graph_type=GraphType.DEPTH_GRAPH, sheet=self.graph_sheet)
        self.assertGreater(len(canvas.plot_widget.plotItem.items), 0)
        self.assertTrue(canvas.plot_widget.getViewBox().yInverted())

        # Render for Grouped table
        grouped_ds = self.doc.create_data_sheet("Test Grouped", table_type=TableType.GROUPED)
        grouped_ds.table.y_group_titles = ["Group 1", "Group 2"]
        grouped_ds.table.num_replicates = 2
        for r, row in enumerate([[5.0, 6.0, 15.0, 16.0], [7.0, 8.0, 18.0, 19.0]]):
            grouped_ds.table.set_cell_value(r, "Y", 0, 0, row[0])
            grouped_ds.table.set_cell_value(r, "Y", 0, 1, row[1])
            grouped_ds.table.set_cell_value(r, "Y", 1, 0, row[2])
            grouped_ds.table.set_cell_value(r, "Y", 1, 1, row[3])
        canvas.update_data(grouped_ds.table, graph_type=GraphType.DEPTH_GRAPH, sheet=None)
        self.assertGreater(len(canvas.plot_widget.plotItem.items), 0)
        self.assertTrue(canvas.plot_widget.getViewBox().yInverted())

    def test_cleanroom_project_serialization(self) -> None:
        """Verifies cleanroom .bigo archive serialization and deserialization of PageLayout sheets."""
        layout = self.doc.create_layout("Paper Composite", template_preset="2 horizontal")
        layout.placeholders[0].graph_id = self.graph_sheet.sheet_id
        layout.page_format = "Letter"
        layout.orientation = "landscape"

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test_project.bigo")
            PrismIO.save(self.doc, file_path)

            # Check zip contents contains layouts/
            with zipfile.ZipFile(file_path, "r") as zf:
                names = zf.namelist()
                layout_entries = [n for n in names if n.startswith("layouts/") and n.endswith("sheet.json")]
                self.assertGreaterEqual(len(layout_entries), 1)

            # Reload document
            loaded_doc = PrismIO.load(file_path)
            self.assertIn(layout.sheet_id, loaded_doc.sheets)
            loaded_layout = loaded_doc.sheets[layout.sheet_id]
            self.assertIsInstance(loaded_layout, PageLayout)
            self.assertEqual(loaded_layout.name, "Paper Composite")
            self.assertEqual(loaded_layout.page_format, "Letter")
            self.assertEqual(loaded_layout.orientation, "landscape")
            self.assertEqual(len(loaded_layout.placeholders), 2)
            self.assertEqual(loaded_layout.placeholders[0].graph_id, self.graph_sheet.sheet_id)

    def test_offscreen_composite_layout_rendering(self) -> None:
        """Verifies offscreen composite rendering to publication-grade QImage, PNG, PDF, and SVG."""
        layout = self.doc.create_layout("Publication Figure", template_preset="2 horizontal")
        layout.placeholders[0].graph_id = self.graph_sheet.sheet_id

        # 1. QImage at 150 DPI
        img = render_layout_to_image(layout, document=self.doc, dpi=150)
        self.assertFalse(img.isNull())
        self.assertGreater(img.width(), 500)
        self.assertGreater(img.height(), 500)

        # 2. PNG bytes at 300 DPI
        png_bytes = render_layout_to_png_bytes(layout, document=self.doc, dpi=300)
        self.assertTrue(png_bytes.startswith(b"\x89PNG\r\n\x1a\n"))

        # 3. PDF generation
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = os.path.join(tmpdir, "layout_figure.pdf")
            render_layout_to_pdf(layout, document=self.doc, file_path=pdf_path, dpi=300)
            self.assertTrue(os.path.exists(pdf_path))
            self.assertGreater(os.path.getsize(pdf_path), 500)

        # 4. SVG generation
        with tempfile.TemporaryDirectory() as tmpdir:
            svg_path = os.path.join(tmpdir, "layout_figure.svg")
            render_layout_to_svg(layout, document=self.doc, file_path=svg_path)
            self.assertTrue(os.path.exists(svg_path))
            with open(svg_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                self.assertIn("<svg", content)

    def test_office_exporter_with_layouts(self) -> None:
        """Verifies OfficeExporter exports Word (.docx), ODT, and PDF with layout panels."""
        layout = self.doc.create_layout("Composite Panel 1", template_preset="2 vertical")
        layout.placeholders[0].graph_id = self.graph_sheet.sheet_id

        exporter = OfficeExporter(self.doc)

        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. DOCX export
            docx_path = os.path.join(tmpdir, "report.docx")
            exporter.export_docx(docx_path)
            self.assertTrue(os.path.exists(docx_path))
            with zipfile.ZipFile(docx_path, "r") as zf:
                doc_xml = zf.read("word/document.xml").decode("utf-8")
                self.assertIn("Page Layout: Composite Panel 1", doc_xml)
                self.assertIn("Figure 1", doc_xml)

            # 2. ODT export
            odt_path = os.path.join(tmpdir, "report.odt")
            exporter.export_odt(odt_path)
            self.assertTrue(os.path.exists(odt_path))
            with zipfile.ZipFile(odt_path, "r") as zf:
                content_xml = zf.read("content.xml").decode("utf-8")
                self.assertIn("Page Layout: Composite Panel 1", content_xml)

            # 3. PDF export (pure layouts)
            pdf_path = os.path.join(tmpdir, "pure_layout.pdf")
            exporter.export_pdf(pdf_path, selected_sheet_ids=[layout.sheet_id])
            self.assertTrue(os.path.exists(pdf_path))
            self.assertGreater(os.path.getsize(pdf_path), 1000)

            # 4. PDF export (full project with tables, graphs, layouts)
            full_pdf_path = os.path.join(tmpdir, "full_report.pdf")
            exporter.export_pdf(full_pdf_path)
            self.assertTrue(os.path.exists(full_pdf_path))
            self.assertGreater(os.path.getsize(full_pdf_path), 1000)

    def test_layout_ui_and_canvas_interaction(self) -> None:
        """Verifies LayoutView and LayoutCanvas interactive manipulation and selection."""
        layout = self.doc.create_layout("Canvas View Test", template_preset="2 horizontal")
        view = LayoutView(document=self.doc)
        view.set_layout(layout)

        self.assertIsNotNone(view.canvas)
        self.assertEqual(len(view.canvas.placeholder_items), 2)

        # Check placeholder items
        item0 = view.canvas.placeholder_items[0]
        self.assertIsInstance(item0, GraphPlaceholderItem)
        self.assertFalse(item0.isSelected())

        # Select item
        item0.setSelected(True)
        self.assertTrue(item0.isSelected())

        # Change template preset via view
        view.set_template("4")
        self.assertEqual(len(layout.placeholders), 4)
        self.assertEqual(len(view.canvas.placeholder_items), 4)

        # Change orientation via view
        view.set_orientation("landscape")
        self.assertEqual(layout.orientation.lower(), "landscape")

        # Alignment action execution
        view.canvas.placeholder_items[0].setSelected(True)
        view.canvas.placeholder_items[1].setSelected(True)
        view.align_selected("left")
        self.assertAlmostEqual(
            view.canvas.placeholder_items[0].placeholder.rect[0],
            view.canvas.placeholder_items[1].placeholder.rect[0],
        )

        view.close()

    def test_all_8_resize_handles(self) -> None:
        """Verifies all 8 resize handles (including 1, 3, 5) transform item rect properly."""
        layout = self.doc.create_layout("Resize Test", template_preset="1")
        canvas = LayoutCanvas()
        canvas.set_layout(layout, self.doc)
        item = canvas.placeholder_items[0]
        # Initial position: (100, 100) with size 800x800 on a 1000x1000 page
        canvas.page_width = 1000.0
        canvas.page_height = 1000.0
        item.page_width = 1000.0
        item.page_height = 1000.0
        item.setPos(100, 100)
        item.setRect(0, 0, 800, 800)
        item.placeholder.locked_aspect = False

        modified_emissions = []
        canvas.layoutModifiedSignal.connect(lambda: modified_emissions.append(True))

        # Handle 1: Top-Center (TC) -> adjusts top (y and height)
        item.handle_resize(1, QPointF(500, 200), shift_pressed=True)
        self.assertAlmostEqual(item.pos().y(), 200, delta=1.0)
        self.assertAlmostEqual(item.rect().height(), 700, delta=1.0)
        self.assertGreater(len(modified_emissions), 0)

        # Handle 3: Middle-Left (ML) -> adjusts left (x and width)
        item.handle_resize(3, QPointF(250, 500), shift_pressed=True)
        self.assertAlmostEqual(item.pos().x(), 250, delta=1.0)
        self.assertAlmostEqual(item.rect().width(), 650, delta=1.0)

        # Handle 5: Bottom-Left (BL) -> adjusts bottom and left
        item.handle_resize(5, QPointF(200, 850), shift_pressed=True)
        self.assertAlmostEqual(item.pos().x(), 200, delta=1.0)
        self.assertAlmostEqual(item.pos().y() + item.rect().height(), 850, delta=1.0)

        # Handle 0: Top-Left (TL)
        item.handle_resize(0, QPointF(150, 150), shift_pressed=True)
        self.assertAlmostEqual(item.pos().x(), 150, delta=1.0)
        self.assertAlmostEqual(item.pos().y(), 150, delta=1.0)

        # Handle 2: Top-Right (TR)
        item.handle_resize(2, QPointF(800, 100), shift_pressed=True)
        self.assertAlmostEqual(item.pos().x() + item.rect().width(), 800, delta=1.0)
        self.assertAlmostEqual(item.pos().y(), 100, delta=1.0)

        # Handle 4: Middle-Right (MR)
        item.handle_resize(4, QPointF(900, 500), shift_pressed=True)
        self.assertAlmostEqual(item.pos().x() + item.rect().width(), 900, delta=1.0)

        # Handle 6: Bottom-Center (BC)
        item.handle_resize(6, QPointF(500, 950), shift_pressed=True)
        self.assertAlmostEqual(item.pos().y() + item.rect().height(), 950, delta=1.0)

        # Handle 7: Bottom-Right (BR)
        item.handle_resize(7, QPointF(850, 900), shift_pressed=True)
        self.assertAlmostEqual(item.pos().x() + item.rect().width(), 850, delta=1.0)
        self.assertAlmostEqual(item.pos().y() + item.rect().height(), 900, delta=1.0)

    def test_slide_pdf_export(self) -> None:
        """Verifies Slide 16:9 and Slide 4:3 PDF export preserves proper proportions."""
        layout_16_9 = self.doc.create_layout("Slide 16-9 Layout", template_preset="2 horizontal")
        layout_16_9.page_format = "Slide (16:9)"
        layout_16_9.orientation = "landscape"

        layout_4_3 = self.doc.create_layout("Slide 4-3 Layout", template_preset="2 vertical")
        layout_4_3.page_format = "Slide (4:3)"
        layout_4_3.orientation = "landscape"

        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_16_9 = os.path.join(tmpdir, "slide_16_9.pdf")
            render_layout_to_pdf(layout_16_9, document=self.doc, file_path=pdf_16_9)
            self.assertTrue(os.path.exists(pdf_16_9))
            self.assertGreater(os.path.getsize(pdf_16_9), 1000)

            pdf_4_3 = os.path.join(tmpdir, "slide_4_3.pdf")
            render_layout_to_pdf(layout_4_3, document=self.doc, file_path=pdf_4_3)
            self.assertTrue(os.path.exists(pdf_4_3))
            self.assertGreater(os.path.getsize(pdf_4_3), 1000)

    def test_floating_bars_graph_rendering(self) -> None:
        """Verifies Floating Bars (Min to Max) graph type renders properly without exceptions."""
        fb_sheet = self.doc.create_graph_sheet("Floating Bars Graph", source_sheet_id=self.data_sheet.sheet_id)
        fb_sheet.graph_type = GraphType.FLOATING_BARS

        canvas = ScientificPlotCanvas()
        canvas.update_data(self.data_sheet.table, sheet=fb_sheet)
        self.assertEqual(canvas.type_combo.currentText(), GraphType.FLOATING_BARS.value)

        # Render layout containing floating bars
        layout = self.doc.create_layout("Floating Bars Layout", template_preset="1")
        layout.placeholders[0].graph_id = fb_sheet.sheet_id

        img = render_layout_to_image(layout, document=self.doc, dpi=150, graph_canvas=canvas)
        self.assertFalse(img.isNull())
        self.assertGreater(img.width(), 100)
        self.assertGreater(img.height(), 100)

    def test_layout_annotations_rendering(self) -> None:
        """Verifies connector lines and text annotations render to image and SVG."""
        layout = self.doc.create_layout("Annotated Layout", template_preset="1")
        layout.annotations = [
            {"type": "text", "text": "Panel A", "x": 0.05, "y": 0.05, "font_size": 16, "color": "#112233"},
            {"type": "connector", "x1": 0.1, "y1": 0.1, "x2": 0.6, "y2": 0.6, "color": "#FF0000", "width": 2},
            {"type": "line", "x1": 0.2, "y1": 0.3, "x2": 0.8, "y2": 0.3, "color": "#00FF00", "width": 1},
        ]
        img = render_layout_to_image(layout, document=self.doc, dpi=150)
        self.assertFalse(img.isNull())

        with tempfile.TemporaryDirectory() as tmpdir:
            svg_path = os.path.join(tmpdir, "annotated.svg")
            render_layout_to_svg(layout, document=self.doc, file_path=svg_path)
            self.assertTrue(os.path.exists(svg_path))
            with open(svg_path, "r", encoding="utf-8") as f:
                svg_content = f.read()
            self.assertIn("Panel A", svg_content)
            self.assertTrue("<line" in svg_content or "<polyline" in svg_content)

    def test_sheet_deletion_unassigns_layout_placeholders(self) -> None:
        """Verifies that removing a sheet cleanly unassigns any layout placeholders linking to it."""
        layout = self.doc.create_layout("Unassign Test", template_preset="2 vertical")
        layout.placeholders[0].graph_id = self.graph_sheet.sheet_id
        layout.placeholders[0].linked_graph_title = self.graph_sheet.title
        self.assertEqual(layout.placeholders[0].graph_id, self.graph_sheet.sheet_id)

        # Delete the graph sheet
        self.doc.remove_sheet(self.graph_sheet.sheet_id)
        self.assertIsNone(layout.placeholders[0].graph_id)
        self.assertIsNone(layout.placeholders[0].linked_graph_title)

    def test_multi_slot_graph_assignment(self) -> None:
        """Verifies that all slots in multi-slot grids (including slot 0, 1, 2, 3) independently maintain ID and assignment."""
        layout = self.doc.create_layout("Multi-Slot Test", template="4 (2x2 Quadrants)")
        self.assertEqual(len(layout.placeholders), 4)

        # Force slot_ids to 0, 1, 2, 3 to explicitly verify slot_id=0 edge case
        for i, p in enumerate(layout.placeholders):
            p.slot_id = i

        canvas = LayoutCanvas()
        canvas.set_layout(layout, self.doc)
        self.assertEqual(len(canvas.placeholder_items), 4)

        # Create distinct test graphs
        g0 = self.doc.create_graph_sheet("Graph Slot 0", source_sheet_id=self.data_sheet.sheet_id)
        g1 = self.doc.create_graph_sheet("Graph Slot 1", source_sheet_id=self.data_sheet.sheet_id)
        g2 = self.doc.create_graph_sheet("Graph Slot 2", source_sheet_id=self.data_sheet.sheet_id)
        g3 = self.doc.create_graph_sheet("Graph Slot 3", source_sheet_id=self.data_sheet.sheet_id)

        # Assign each slot independently with rotation
        for idx, (p_item, g_sheet) in enumerate(zip(canvas.placeholder_items, [g0, g1, g2, g3])):
            self.assertEqual(p_item.placeholder.slot_id, idx)
            p_item.placeholder.assign_graph(g_sheet.sheet_id, g_sheet.title)
            p_item.placeholder.rotation = 90 if idx == 1 else 0
            self.assertEqual(p_item.placeholder.graph_id, g_sheet.sheet_id)
            self.assertEqual(p_item.placeholder.linked_graph_title, g_sheet.title)

        # Verify all 4 placeholders have their respective graphs and distinct slot_ids
        assigned_ids = [p.graph_id for p in layout.placeholders]
        self.assertEqual(assigned_ids, [g0.sheet_id, g1.sheet_id, g2.sheet_id, g3.sheet_id])
        self.assertEqual([p.slot_id for p in layout.placeholders], [0, 1, 2, 3])
        self.assertEqual(layout.placeholders[1].rotation, 90)

    def test_layout_context_menu_and_add_actions(self) -> None:
        """Verifies right-click context menu actions: Add Graph Placeholder and Add Text Box."""
        from PyQt6.QtCore import QPoint
        layout = self.doc.create_layout("Context Menu Test", template_preset="1")
        canvas = LayoutCanvas()
        canvas.set_layout(layout, self.doc)
        initial_p_count = len(layout.placeholders)
        initial_ann_count = len(getattr(layout, "annotations", []))

        # Add graph placeholder at cursor position (100, 100)
        canvas.action_add_placeholder_at(QPoint(100, 100))
        self.assertEqual(len(layout.placeholders), initial_p_count + 1)
        new_p = layout.placeholders[-1]
        self.assertGreater(new_p.rect[2], 0.0)
        self.assertGreater(new_p.rect[3], 0.0)
        self.assertLessEqual(new_p.rect[0] + new_p.rect[2], 1.0)
        self.assertLessEqual(new_p.rect[1] + new_p.rect[3], 1.0)

        # Add text box at cursor position (150, 150)
        canvas.action_add_text_at(QPoint(150, 150))
        self.assertEqual(len(layout.annotations), initial_ann_count + 1)
        new_ann = layout.annotations[-1]
        self.assertEqual(new_ann["type"], "text")
        self.assertIn("x", new_ann)
        self.assertIn("y", new_ann)
        self.assertIn("text", new_ann)

    def test_normalized_coordinate_stability_across_tabs(self) -> None:
        """Verifies placeholder normalized coordinates never shrink to microscopic size across tab switches."""
        from src.ui.main_window import MainWindow
        win = MainWindow()
        win.resize(1024, 768)

        layout = win.document.create_layout("Stability Test", template="4 (2x2 Quadrants)")
        initial_rects = [p.rect for p in layout.placeholders]

        # Select layout
        win.on_sheet_selected(layout.sheet_id)
        for p, init_r in zip(layout.placeholders, initial_rects):
            self.assertAlmostEqual(p.rect[2], init_r[2], places=3)
            self.assertAlmostEqual(p.rect[3], init_r[3], places=3)

        # Switch between sheets multiple times
        data_sheet = win.document.get_sheets_by_type(SheetType.DATA)[0]
        for _ in range(5):
            win.on_sheet_selected(data_sheet.sheet_id)
            win.on_sheet_selected(layout.sheet_id)

        # Verify zero shrinking
        for p, init_r in zip(layout.placeholders, initial_rects):
            self.assertAlmostEqual(p.rect[2], init_r[2], places=3)
            self.assertAlmostEqual(p.rect[3], init_r[3], places=3)
            self.assertGreaterEqual(p.rect[2], 0.10)
            self.assertGreaterEqual(p.rect[3], 0.10)

        # Convert orientation
        win.layout_view.set_orientation("landscape")
        for p, init_r in zip(layout.placeholders, initial_rects):
            self.assertAlmostEqual(p.rect[2], init_r[2], places=3)
            self.assertAlmostEqual(p.rect[3], init_r[3], places=3)

        win.layout_view.set_orientation("portrait")
        for p, init_r in zip(layout.placeholders, initial_rects):
            self.assertAlmostEqual(p.rect[2], init_r[2], places=3)
            self.assertAlmostEqual(p.rect[3], init_r[3], places=3)

        win.close()

    def test_label_truncation_prevention_on_thumbnail(self) -> None:
        """Verifies generous axis margin padding prevents tick and group label truncation."""
        canvas = ScientificPlotCanvas()
        canvas.update_data(self.data_sheet.table, sheet=self.graph_sheet)

        png_bytes = canvas.get_plot_snapshot(width=800, height=600)
        self.assertTrue(png_bytes.startswith(b"\x89PNG\r\n\x1a\n"))

        # Check that bottom and left axes have generous dimensions
        bottom_ax = canvas.plot_widget.getAxis("bottom")
        left_ax = canvas.plot_widget.getAxis("left")
        self.assertGreaterEqual(bottom_ax.height(), 10.0)
        self.assertGreaterEqual(left_ax.width(), 10.0)

    def test_pinned_graph_dialog_flags_and_navigator(self) -> None:
        """Verifies AssignGraphDialog flags ('❄️' vs '🔥') and left-aligned '❄️' in navigator."""
        from src.ui.dialogs import AssignGraphDialog
        from src.ui.sidebar import PrismNavigator

        # Create active graph, pinned+active graph, and pinned+frozen graph
        g_active = self.graph_sheet

        g_pinned_active = self.doc.create_graph_sheet("Pinned Active Graph", source_sheet_id=self.data_sheet.sheet_id)
        g_pinned_active.is_pinned = True
        g_pinned_active.is_frozen = False

        g_pinned_frozen = self.doc.create_graph_sheet("Pinned Frozen Graph", source_sheet_id=self.data_sheet.sheet_id)
        g_pinned_frozen.is_pinned = True
        g_pinned_frozen.is_frozen = True

        # Test AssignGraphDialog
        dialog = AssignGraphDialog(self.doc, current_graph_id=None, current_rotation=90, current_scale=1.2)
        self.assertEqual(dialog.get_selected_rotation(), 90)
        self.assertAlmostEqual(dialog.get_selected_scale(), 1.2)

        pinned_root = None
        for i in range(dialog.tree.topLevelItemCount()):
            top_item = dialog.tree.topLevelItem(i)
            if "Pinned" in top_item.text(0):
                pinned_root = top_item
                break
        self.assertIsNotNone(pinned_root)

        pinned_titles = [pinned_root.child(c).text(0) for c in range(pinned_root.childCount())]
        # Pinned+frozen must have snowflake prefix
        self.assertTrue(any(t.startswith("❄️  ") and "Pinned Frozen Graph" in t for t in pinned_titles))
        # Pinned+active must have fire prefix
        self.assertTrue(any(t.startswith("🔥  ") and "Pinned Active Graph" in t for t in pinned_titles))

        # Test PrismNavigator left-aligned snowflake
        nav = PrismNavigator()
        nav.populate(self.doc)
        graphs_folder = nav.folder_items[SheetType.GRAPH]
        graph_labels = [graphs_folder.child(c).text(0) for c in range(graphs_folder.childCount())]
        self.assertTrue(any(lbl.startswith("❄️  ") and "Pinned Frozen Graph" in lbl for lbl in graph_labels))
        # Unfrozen graphs must NOT have snowflake
        self.assertTrue(any("Pinned Active Graph" in lbl and not lbl.startswith("❄️  ") for lbl in graph_labels))

    def test_undo_redo_layout_operations(self) -> None:
        """Verifies QUndoStack integration across layout placeholder additions, graph assignments, and text edits."""
        layout = self.doc.create_layout("Undo Test", template_preset="1")
        view = LayoutView(document=self.doc)
        view.set_layout(layout, self.doc)

        initial_count = len(layout.placeholders)
        new_p = Placeholder(slot_id=99, rect=(0.2, 0.2, 0.3, 0.3))

        # 1. Add placeholder command
        view.push_add_placeholder_command(new_p)
        self.assertEqual(len(layout.placeholders), initial_count + 1)

        # Undo add
        view.undo()
        self.assertEqual(len(layout.placeholders), initial_count)

        # Redo add
        view.redo()
        self.assertEqual(len(layout.placeholders), initial_count + 1)

        # 2. Assign graph command
        p = layout.placeholders[0]
        self.assertIsNone(p.graph_id)
        view.push_assign_graph_command(p, self.graph_sheet.sheet_id, self.graph_sheet.title, rotation=90, scale=1.5)
        self.assertEqual(p.graph_id, self.graph_sheet.sheet_id)
        self.assertEqual(p.rotation, 90)
        self.assertEqual(p.custom_scale, 1.5)

        # Undo assign
        view.undo()
        self.assertIsNone(p.graph_id)
        self.assertEqual(p.rotation, 0)
        self.assertEqual(p.custom_scale, 1.0)

        # Redo assign
        view.redo()
        self.assertEqual(p.graph_id, self.graph_sheet.sheet_id)
        self.assertEqual(p.rotation, 90)

        # 3. Add text annotation command
        ann = {"id": "text_undo_test", "type": "text", "text": "Original Title", "html": "Original Title"}
        view.push_add_text_command(ann)
        self.assertIn(ann, layout.annotations)

        # Edit text annotation command
        view.push_edit_text_command(ann, {"text": "Original Title"}, {"text": "Updated Title"})
        self.assertEqual(ann["text"], "Updated Title")

        # Undo edit
        view.undo()
        self.assertEqual(ann["text"], "Original Title")

        # Redo edit
        view.redo()
        self.assertEqual(ann["text"], "Updated Title")

        # Undo add text
        view.undo()  # undos edit
        view.undo()  # undos add
        self.assertNotIn(ann, layout.annotations)

        view.close()

    def test_hot_graph_placeholder_live_rendering(self) -> None:
        """Verifies an unfrozen ('hot') graph placeholder renders non-empty pixmap bytes with live data series."""
        layout = self.doc.create_layout("Hot Graph Layout")
        layout.placeholders.clear()
        p = Placeholder(slot_id="slot_hot", rect=(0.1, 0.1, 0.4, 0.4), graph_id=self.graph_sheet.sheet_id, linked_graph_title=self.graph_sheet.title)
        layout.add_placeholder(p)
        self.graph_sheet.is_pinned = True
        self.graph_sheet.is_frozen = False

        view = LayoutView()
        view.load_layout(layout, self.doc)
        item = view.canvas.placeholder_items[0]

        # Trigger snapshot update
        item.update_snapshot()
        self.assertIsNotNone(item.cached_pixmap)
        self.assertFalse(item.cached_pixmap.isNull())
        self.assertGreater(item.cached_pixmap.width(), 0)
        self.assertGreater(item.cached_pixmap.height(), 0)

        # Pixmap contains rendered content (non-white pixels)
        img = item.cached_pixmap.toImage()
        has_non_white = False
        for y in range(0, img.height(), 15):
            for x in range(0, img.width(), 15):
                if img.pixelColor(x, y).name() != "#ffffff":
                    has_non_white = True
                    break
            if has_non_white:
                break
        self.assertTrue(has_non_white)
        view.close()

    def test_navigating_back_to_layout_view_triggers_snapshot_refresh(self) -> None:
        """Verifies that showEvent on LayoutView triggers snapshot refresh on unfrozen graphs."""
        layout = self.doc.create_layout("Reactive Layout")
        layout.placeholders.clear()
        p = Placeholder(slot_id="slot_react", rect=(0.1, 0.1, 0.4, 0.4), graph_id=self.graph_sheet.sheet_id, linked_graph_title=self.graph_sheet.title)
        layout.add_placeholder(p)
        self.graph_sheet.is_pinned = True
        self.graph_sheet.is_frozen = False

        view = LayoutView()
        view.load_layout(layout, self.doc)
        item = view.canvas.placeholder_items[0]

        # Modify upstream data table
        self.data_sheet.table.set_cell_value(0, "Y", 0, 0, 999.0)

        # Clear cached pixmap to simulate state before navigation
        item.cached_pixmap = None

        from PyQt6.QtGui import QShowEvent
        event = QShowEvent()
        view.showEvent(event)
        QApplication.processEvents()

        # showEvent must have regenerated the cached pixmap
        self.assertIsNotNone(item.cached_pixmap)
        self.assertFalse(item.cached_pixmap.isNull())
        view.close()

    def test_layout_text_item_interaction_states_and_resize(self) -> None:
        """Verifies LayoutTextItem has ItemIsMovable in idle mode, switches to editing on double click, and exits on focusOutEvent/Escape."""
        layout = self.doc.create_layout("Annotated Layout")
        ann = {
            "id": "ann_test_1",
            "type": "text",
            "text": "Figure 1A: Primary Response Curve",
            "x": 0.15,
            "y": 0.25,
            "w": 0.35,
            "font_size": 14,
        }
        layout.annotations = [ann]

        view = LayoutView()
        view.load_layout(layout, self.doc)
        text_item = view.canvas.text_items[0]

        # 1. Idle mode: ItemIsMovable enabled, textInteractionFlags is NoTextInteraction
        self.assertTrue(bool(text_item.flags() & QGraphicsItem.GraphicsItemFlag.ItemIsMovable))
        self.assertTrue(bool(text_item.flags() & QGraphicsItem.GraphicsItemFlag.ItemIsSelectable))
        self.assertEqual(text_item.textInteractionFlags(), Qt.TextInteractionFlag.NoTextInteraction)

        # 2. Double-click switches to TextEditorInteraction
        dbl_click = QMouseEvent(
            QEvent.Type.MouseButtonDblClick,
            QPointF(10, 10),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        text_item.mouseDoubleClickEvent(dbl_click)
        self.assertEqual(text_item.textInteractionFlags(), Qt.TextInteractionFlag.TextEditorInteraction)

        # 3. Escape key exits editing mode back to NoTextInteraction
        esc_event = QKeyEvent(
            QEvent.Type.KeyPress,
            Qt.Key.Key_Escape,
            Qt.KeyboardModifier.NoModifier,
        )
        text_item.keyPressEvent(esc_event)
        self.assertEqual(text_item.textInteractionFlags(), Qt.TextInteractionFlag.NoTextInteraction)

        # 4. Double click again and focusOutEvent exits editing
        text_item.mouseDoubleClickEvent(dbl_click)
        self.assertEqual(text_item.textInteractionFlags(), Qt.TextInteractionFlag.TextEditorInteraction)
        focus_out = QFocusEvent(QEvent.Type.FocusOut)
        text_item.focusOutEvent(focus_out)
        self.assertEqual(text_item.textInteractionFlags(), Qt.TextInteractionFlag.NoTextInteraction)

        # 5. Interactive resize handle adjusts document text width
        initial_w = text_item.boundingRect().width()
        text_item.handle_resize(1, QPointF(500, 100), QPointF(200, 100), initial_w, text_item.pos())
        self.assertGreater(text_item.boundingRect().width(), initial_w)
        text_item.finish_resize()
        text_item = view.canvas.text_items[0]

        # 6. Double click again, click outside exits editing mode
        text_item.mouseDoubleClickEvent(dbl_click)
        self.assertEqual(text_item.textInteractionFlags(), Qt.TextInteractionFlag.TextEditorInteraction)
        outside_event = QMouseEvent(
            QEvent.Type.MouseButtonPress,
            QPointF(500, 500),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        view.canvas.mousePressEvent(outside_event)
        self.assertEqual(text_item.textInteractionFlags(), Qt.TextInteractionFlag.NoTextInteraction)

        # 7. Backspace while editing does NOT delete text item
        text_item.mouseDoubleClickEvent(dbl_click)
        self.assertEqual(text_item.textInteractionFlags(), Qt.TextInteractionFlag.TextEditorInteraction)
        initial_count = len(view.canvas.text_items)
        bksp_event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Backspace, Qt.KeyboardModifier.NoModifier)
        view.canvas.keyPressEvent(bksp_event)
        self.assertEqual(len(view.canvas.text_items), initial_count)

        # 8. Selection preservation across scene rebuild
        text_item.setSelected(True)
        view.canvas.rebuild_scene()
        self.assertTrue(view.canvas.text_items[0].isSelected())

        view.close()

    def test_depth_graph_dynamic_vertical_axis_label(self) -> None:
        """Verifies Depth Graph vertical axis label reflects column/group names instead of 'Depth in meters'."""
        canvas = ScientificPlotCanvas(parent=None)

        # XY table without custom title: uses active Y-column headers
        xy_sheet = self.doc.create_data_sheet("Depth Profile XY", table_type=TableType.XY)
        xy_sheet.table.y_group_titles = ["Salinity (psu)", "Temperature (°C)"]
        canvas.update_data(xy_sheet.table, graph_type=GraphType.DEPTH_GRAPH, sheet=self.graph_sheet)

        left_ax = canvas.plot_widget.getAxis("left")
        left_label_text = left_ax.labelText
        self.assertIn("Salinity (psu)", left_label_text)
        self.assertIn("Temperature (°C)", left_label_text)
        self.assertNotIn("Depth in meters", left_label_text)

        # Column table without custom title: uses factor title, sheet name, or Groups
        col_table = DataTable(table_id="col_t", title="Treatment Conditions", table_type=TableType.COLUMN, num_rows=3, y_group_titles=["Baseline", "Dose 1", "Dose 2"])
        canvas.update_data(col_table, graph_type=GraphType.DEPTH_GRAPH)
        left_label_text2 = canvas.plot_widget.getAxis("left").labelText
        self.assertEqual(left_label_text2, "Treatment Conditions")
        self.assertNotIn("Depth in meters", left_label_text2)

        # Custom y_title in plot_config overrides everything
        self.graph_sheet.plot_config["y_title"] = "Sediment Depth (cm)"
        canvas.update_data(xy_sheet.table, graph_type=GraphType.DEPTH_GRAPH, sheet=self.graph_sheet)
        left_label_text3 = canvas.plot_widget.getAxis("left").labelText
        self.assertEqual(left_label_text3, "Sediment Depth (cm)")
        self.assertNotIn("Depth in meters", left_label_text3)

    def test_freeze_multiple_graphs_no_unended_painters(self) -> None:
        """Verifies creating, assigning, and freezing multiple graphs does not leave un-ended painters."""
        layout = self.doc.create_layout("Multi-Frozen Regression Layout")
        layout.placeholders.clear()

        # Create multiple graphs, assign and freeze them
        graphs = []
        for i in range(3):
            gs = self.doc.create_graph_sheet(f"FrozenGraph_{i}", source_sheet_id=self.data_sheet.sheet_id)
            gs.is_frozen = True
            gs.snapshot_table = self.data_sheet.table
            graphs.append(gs)
            p = Placeholder(slot_id=f"slot_frz_{i}", rect=(0.05 + 0.3 * i, 0.1, 0.25, 0.35), graph_id=gs.sheet_id)
            layout.add_placeholder(p)

        view = LayoutView()
        view.load_layout(layout, self.doc)

        for item in view.canvas.placeholder_items:
            item.update_snapshot()
            self.assertIsNotNone(item.cached_pixmap)
            self.assertFalse(item.cached_pixmap.isNull())

        # Render layout canvas scene to an offscreen image and verify no active painter leaks
        img = QImage(800, 600, QImage.Format.Format_ARGB32_Premultiplied)
        img.fill(Qt.GlobalColor.white)
        test_painter = QPainter(img)
        self.assertTrue(test_painter.isActive())
        view.canvas.scene.render(test_painter)
        test_painter.end()
        self.assertFalse(test_painter.isActive())
        view.close()

    def test_render_graph_to_png_bytes_rapid_consecutive_calls(self) -> None:
        """Verifies render_graph_to_png_bytes safely handles rapid consecutive calls without paint device conflicts."""
        for i in range(12):
            gs = self.doc.create_graph_sheet(f"RapidGraph_{i}", source_sheet_id=self.data_sheet.sheet_id)
            if i % 2 == 0:
                gs.is_frozen = True
                gs.snapshot_table = self.data_sheet.table
            png_bytes = render_graph_to_png_bytes(
                gs,
                document=self.doc,
                width=600 + i * 20,
                height=400 + i * 15,
            )
            self.assertTrue(png_bytes.startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertGreater(len(png_bytes), 500)

    def test_layout_text_item_drag_and_resize_without_grab_mouse(self) -> None:
        """Verifies LayoutTextItem drag and resize functions cleanly without invoking grabMouse()."""
        layout = self.doc.create_layout("Wayland Text Item Test Layout")
        ann = {
            "id": "ann_wayland_spec",
            "type": "text",
            "text": "Figure Legend Wayland Safe",
            "x": 0.15,
            "y": 0.20,
            "w": 0.35,
            "font_size": 13,
        }
        layout.annotations = [ann]

        view = LayoutView()
        view.load_layout(layout, self.doc)
        text_item = view.canvas.text_items[0]

        # Verify item flags rely on ItemIsMovable without manual mouse grab
        self.assertTrue(bool(text_item.flags() & QGraphicsItem.GraphicsItemFlag.ItemIsMovable))

        with patch.object(text_item, "grabMouse", MagicMock()) as mock_grab:
            # 1. Native mouse press & drag event propagation
            ev_press = MagicMock()
            ev_press.button.return_value = Qt.MouseButton.LeftButton
            text_item.mousePressEvent(ev_press)
            text_item.setPos(QPointF(120, 140))
            ev_rel = MagicMock()
            ev_rel.button.return_value = Qt.MouseButton.LeftButton
            text_item.mouseReleaseEvent(ev_rel)
            mock_grab.assert_not_called()

        # Retrieve current text item after move commit
        text_item = view.canvas.text_items[0]
        with patch.object(text_item, "grabMouse", MagicMock()) as mock_grab:
            # 2. Resize handle drag propagation
            mr_handle = text_item.handles[2]
            h_press = MagicMock()
            h_press.button.return_value = Qt.MouseButton.LeftButton
            h_press.scenePos.return_value = QPointF(200, 140)
            mr_handle.mousePressEvent(h_press)

            h_move = MagicMock()
            h_move.scenePos.return_value = QPointF(260, 140)
            mr_handle.mouseMoveEvent(h_move)

            h_rel = MagicMock()
            h_rel.button.return_value = Qt.MouseButton.LeftButton
            mr_handle.mouseReleaseEvent(h_rel)
            mock_grab.assert_not_called()

        self.assertGreater(text_item.annotation.get("x", 0), 0)
        view.close()

    def test_graph_placeholder_item_drag_and_resize_without_grab_mouse(self) -> None:
        """Verifies GraphPlaceholderItem drag and resize functions cleanly without invoking grabMouse()."""
        layout = self.doc.create_layout("Placeholder Wayland Safety Layout")
        layout.placeholders.clear()
        p = Placeholder(slot_id="slot_wl", rect=(0.1, 0.1, 0.35, 0.35), graph_id=self.graph_sheet.sheet_id)
        layout.add_placeholder(p)

        view = LayoutView()
        view.load_layout(layout, self.doc)
        item = view.canvas.placeholder_items[0]

        # Verify ItemIsMovable flag is set for native QGraphicsScene event routing
        self.assertTrue(bool(item.flags() & QGraphicsItem.GraphicsItemFlag.ItemIsMovable))

        with patch.object(item, "grabMouse", MagicMock()) as mock_grab:
            ev_press = MagicMock()
            ev_press.button.return_value = Qt.MouseButton.LeftButton
            item.mousePressEvent(ev_press)
            item.setPos(50, 60)
            ev_rel = MagicMock()
            ev_rel.button.return_value = Qt.MouseButton.LeftButton
            item.mouseReleaseEvent(ev_rel)
            mock_grab.assert_not_called()

        item = view.canvas.placeholder_items[0]
        with patch.object(item, "grabMouse", MagicMock()) as mock_grab:
            handle = item.handles[7]  # BR handle
            h_press = MagicMock()
            h_press.button.return_value = Qt.MouseButton.LeftButton
            h_press.scenePos.return_value = QPointF(200, 200)
            handle.mousePressEvent(h_press)

            h_move = MagicMock()
            h_move.scenePos.return_value = QPointF(250, 250)
            handle.mouseMoveEvent(h_move)

            h_rel = MagicMock()
            h_rel.button.return_value = Qt.MouseButton.LeftButton
            handle.mouseReleaseEvent(h_rel)
            mock_grab.assert_not_called()

        view.close()

    def test_render_graph_to_png_bytes_isolates_live_canvas_completely(self) -> None:
        """Verifies render_graph_to_png_bytes NEVER resizes, mutates axes, or calls render() on live widget."""
        from src.ui.graph_canvas import GraphCanvasWidget

        parent_win = QWidget()
        live_canvas = GraphCanvasWidget(parent=parent_win)
        parent_win.show()
        live_canvas.show()
        live_canvas.resize(700, 500)

        live_canvas.update_data(self.data_sheet.table, sheet=self.graph_sheet)
        orig_size = live_canvas.size()
        orig_plot_geo = live_canvas.plot_widget.geometry()
        orig_bottom_h = live_canvas.plot_widget.getAxis("bottom").height()
        orig_left_w = live_canvas.plot_widget.getAxis("left").width()

        with patch.object(live_canvas.plot_widget, "render", MagicMock()) as mock_render:
            png_bytes = render_graph_to_png_bytes(
                self.graph_sheet,
                document=self.doc,
                graph_canvas=live_canvas,
                width=1400,
                height=900,
            )
            # Must NOT invoke widget-level render() on live widget
            mock_render.assert_not_called()

        self.assertTrue(png_bytes.startswith(b"\x89PNG\r\n\x1a\n"))
        # Live canvas geometry and axis properties must remain completely untouched
        self.assertEqual(live_canvas.size(), orig_size)
        self.assertEqual(live_canvas.plot_widget.geometry(), orig_plot_geo)
        self.assertEqual(live_canvas.plot_widget.getAxis("bottom").height(), orig_bottom_h)
        self.assertEqual(live_canvas.plot_widget.getAxis("left").width(), orig_left_w)

        parent_win.close()

    def test_placeholder_initialization_lock_prevents_corruption(self) -> None:
        """Verifies initializing GraphPlaceholderItem never corrupts placeholder.rect to 0.02."""
        initial_rect = (0.15, 0.20, 0.50, 0.40)
        p = Placeholder("slot_lock_test", rect=initial_rect, graph_id=self.graph_sheet.sheet_id)
        canvas = LayoutCanvas()
        item = GraphPlaceholderItem(p, 800.0, 600.0, self.doc, canvas)

        # 1. Bounds must remain untouched post-initialization
        self.assertEqual(p.rect, initial_rect)
        self.assertNotEqual(p.rect[2], 0.02)
        self.assertNotEqual(p.rect[3], 0.02)

        # 2. Early itemChange or small geometry updates (< 20px) must abort and never corrupt rect
        item.setRect(0, 0, 10, 10)
        item._sync_placeholder_from_geometry()
        self.assertEqual(p.rect, initial_rect)

        # 3. Paint invocation on sub-40px geometry forces automatic re-sync from placeholder
        img = QImage(100, 100, QImage.Format.Format_ARGB32)
        painter = QPainter(img)
        item.paint(painter, None)
        painter.end()
        self.assertEqual(item.rect().width(), 400.0)
        self.assertEqual(item.rect().height(), 240.0)
        self.assertEqual(p.rect, initial_rect)
        canvas.close()

    def test_render_graph_to_png_bytes_offscreen_fills_target_buffer(self) -> None:
        """Verifies calling render_graph_to_png_bytes on offscreen canvas produces tightly bounded snapshot."""
        w, h = 800, 600
        png_bytes = render_graph_to_png_bytes(
            self.graph_sheet,
            document=self.doc,
            width=w,
            height=h,
        )
        self.assertTrue(png_bytes.startswith(b"\x89PNG\r\n\x1a\n"))
        img = QImage.fromData(png_bytes)
        self.assertFalse(img.isNull())
        self.assertEqual(img.width(), w)
        self.assertEqual(img.height(), h)

        # Verify that non-white content spans substantially across the buffer,
        # confirming plotItem fills the target image rather than shrinking to a microscopic speck.
        non_white_xs = []
        non_white_ys = []
        for y in range(0, h, 2):
            for x in range(0, w, 2):
                c = img.pixelColor(x, y)
                if (c.red(), c.green(), c.blue()) != (255, 255, 255):
                    non_white_xs.append(x)
                    non_white_ys.append(y)

        self.assertGreater(len(non_white_xs), 1000)
        min_x, max_x = min(non_white_xs), max(non_white_xs)
        min_y, max_y = min(non_white_ys), max(non_white_ys)
        # Content must span at least 80% of width and 70% of height
        self.assertLess(min_x, w * 0.20)
        self.assertGreater(max_x, w * 0.80)
        self.assertLess(min_y, h * 0.20)
        self.assertGreater(max_y, h * 0.80)

    def test_assign_graphs_slots_renders_snapshots_cleanly(self) -> None:
        """Verifies assigning graphs to slots 0, 1, 2, 3 renders snapshots scaled cleanly to the placeholder rectangle."""
        layout = self.doc.create_layout("Four Slot Layout", template_preset="4")
        self.assertEqual(len(layout.placeholders), 4)

        for p in layout.placeholders:
            p.graph_id = self.graph_sheet.sheet_id

        view = LayoutView()
        view.load_layout(layout, self.doc)
        self.assertEqual(len(view.canvas.placeholder_items), 4)

        for item in view.canvas.placeholder_items:
            item.update_snapshot()
            self.assertIsNotNone(item.cached_pixmap)
            self.assertFalse(item.cached_pixmap.isNull())
            # Cached pixmap should be high-DPI rendered snapshot
            self.assertGreaterEqual(item.cached_pixmap.width(), 600)
            self.assertGreaterEqual(item.cached_pixmap.height(), 450)
            # Placeholder rect must maintain proportional page coordinates (> 0.05)
            rx, ry, rw, rh = item.placeholder.rect
            self.assertGreater(rw, 0.20)
            self.assertGreater(rh, 0.20)

        # Full layout rendering must succeed with non-empty snapshot composition
        png_layout = render_layout_to_png_bytes(layout, self.doc, dpi=150)
        self.assertTrue(png_layout.startswith(b"\x89PNG\r\n\x1a\n"))
        layout_img = QImage.fromData(png_layout)
        self.assertFalse(layout_img.isNull())
        self.assertGreater(layout_img.width(), 800)
        self.assertGreater(layout_img.height(), 800)
        view.close()

    def test_assign_graph_dialog_scale_normalization(self) -> None:
        """Verifies AssignGraphDialog correctly normalizes and clamps scale factors."""
        from src.ui.dialogs import AssignGraphDialog

        # 1. Float 1.0 -> 100%
        dlg1 = AssignGraphDialog(self.doc, current_scale=1.0)
        self.assertEqual(dlg1.spin_scale.value(), 100)
        self.assertEqual(dlg1.get_selected_scale(), 1.0)
        dlg1.close()

        # 2. Percentage 100.0 -> normalized to 100% (1.0)
        dlg2 = AssignGraphDialog(self.doc, current_scale=100.0)
        self.assertEqual(dlg2.spin_scale.value(), 100)
        self.assertEqual(dlg2.get_selected_scale(), 1.0)
        dlg2.close()

        # 3. Microscopic 0.01 -> clamped to minimum 20% (0.2)
        dlg3 = AssignGraphDialog(self.doc, current_scale=0.01)
        self.assertEqual(dlg3.spin_scale.value(), 20)
        self.assertEqual(dlg3.get_selected_scale(), 0.2)
        dlg3.close()

        # 4. Out-of-bounds 10.0 -> clamped to maximum 500% (5.0)
        dlg4 = AssignGraphDialog(self.doc, current_scale=10.0)
        self.assertEqual(dlg4.spin_scale.value(), 500)
        self.assertEqual(dlg4.get_selected_scale(), 5.0)
        dlg4.close()

        # 5. Normal custom scale 2.5 -> 250%
        dlg5 = AssignGraphDialog(self.doc, current_scale=2.5)
        self.assertEqual(dlg5.spin_scale.value(), 250)
        self.assertEqual(dlg5.get_selected_scale(), 2.5)
        # Interactive spinbox edit
        dlg5.spin_scale.setValue(150)
        self.assertEqual(dlg5.get_selected_scale(), 1.5)
        dlg5.close()

    def test_render_graph_to_png_bytes_with_passed_offscreen_canvas(self) -> None:
        """Verifies rendering with a passed offscreen (not live) canvas forces geometry update and fills buffer."""
        passed_canvas = ScientificPlotCanvas()
        w, h = 800, 600
        png_bytes = render_graph_to_png_bytes(
            self.graph_sheet,
            document=self.doc,
            graph_canvas=passed_canvas,
            width=w,
            height=h,
        )
        self.assertTrue(png_bytes.startswith(b"\x89PNG\r\n\x1a\n"))
        img = QImage.fromData(png_bytes)
        self.assertFalse(img.isNull())
        self.assertEqual(img.width(), w)
        self.assertEqual(img.height(), h)

        non_white_xs = []
        for y in range(0, h, 2):
            for x in range(0, w, 2):
                c = img.pixelColor(x, y)
                if (c.red(), c.green(), c.blue()) != (255, 255, 255):
                    non_white_xs.append(x)

        self.assertGreater(len(non_white_xs), 1000)
        min_x, max_x = min(non_white_xs), max(non_white_xs)
        self.assertLess(min_x, w * 0.20)
        self.assertGreater(max_x, w * 0.80)
        passed_canvas.close()

    def test_layout_text_item_deferred_undo_push_on_unfocus(self) -> None:
        """Verifies text editing and un-focusing triggers deferred undo push without RuntimeError or SIGSEGV."""
        layout = self.doc.create_layout("Text Unfocus Regression")
        ann = {
            "id": "ann_text_reg",
            "type": "text",
            "text": "Before Edit",
            "html": "Before Edit",
            "x": 0.1,
            "y": 0.1,
            "w": 0.2,
            "h": 0.1,
        }
        layout.annotations = [ann]

        view = LayoutView()
        view.load_layout(layout, self.doc)
        self.assertEqual(len(view.canvas.text_items), 1)
        text_item = view.canvas.text_items[0]

        # Double click to enter edit mode
        ev_dbl = MagicMock()
        ev_dbl.button.return_value = Qt.MouseButton.LeftButton
        text_item.mouseDoubleClickEvent(ev_dbl)
        self.assertEqual(text_item.textInteractionFlags(), Qt.TextInteractionFlag.TextEditorInteraction)

        # Mutate text
        text_item.setPlainText("After Edit Text")

        # Simulate focus out event
        focus_ev = QFocusEvent(QEvent.Type.FocusOut, Qt.FocusReason.OtherFocusReason)
        # Must not raise RuntimeError: wrapped C/C++ object of type LayoutTextItem has been deleted
        text_item.focusOutEvent(focus_ev)

        # Process pending QTimer.singleShot(0) events
        app.processEvents()

        # Check undo stack has received the command
        self.assertTrue(view.undo_stack.canUndo())
        self.assertEqual(layout.annotations[0]["text"], "After Edit Text")

        # Undo must restore previous text without crashing
        view.undo()
        app.processEvents()
        self.assertEqual(layout.annotations[0]["text"], "Before Edit")
        view.close()

    def test_layout_placeholder_deferred_move_resize_command(self) -> None:
        """Verifies modifying placeholder coordinates triggers deferred command queue safely without mid-event destruction."""
        layout = self.doc.create_layout("Placeholder Deferred Undo Layout")
        layout.placeholders.clear()
        p = Placeholder(slot_id="slot_def_undo", rect=(0.1, 0.1, 0.3, 0.3))
        layout.add_placeholder(p)

        view = LayoutView()
        view.load_layout(layout, self.doc)
        self.assertEqual(len(view.canvas.placeholder_items), 1)
        item = view.canvas.placeholder_items[0]

        # 1. Test mouse move & release on placeholder
        ev_press = MagicMock()
        ev_press.button.return_value = Qt.MouseButton.LeftButton
        item.mousePressEvent(ev_press)
        item.setPos(QPointF(40, 50))
        ev_rel = MagicMock()
        ev_rel.button.return_value = Qt.MouseButton.LeftButton
        # Must execute without RuntimeError
        item.mouseReleaseEvent(ev_rel)

        # Process deferred undo push
        app.processEvents()
        self.assertTrue(view.undo_stack.canUndo())
        self.assertEqual(view.undo_stack.count(), 1)

        # 2. Test resize handle mouse release
        item = view.canvas.placeholder_items[0]
        handle = item.handles[7]  # Bottom-right
        h_press = MagicMock()
        h_press.button.return_value = Qt.MouseButton.LeftButton
        h_press.scenePos.return_value = QPointF(200, 200)
        handle.mousePressEvent(h_press)

        # Simulate resize modification
        item.placeholder.rect = (0.1, 0.1, 0.45, 0.45)
        h_rel = MagicMock()
        h_rel.button.return_value = Qt.MouseButton.LeftButton
        handle.mouseReleaseEvent(h_rel)

        # Process deferred resize command
        app.processEvents()
        self.assertEqual(view.undo_stack.count(), 2)

        # Test undo
        view.undo()
        app.processEvents()
        self.assertEqual(view.undo_stack.count(), 2)
        view.close()


if __name__ == "__main__":
    unittest.main()
