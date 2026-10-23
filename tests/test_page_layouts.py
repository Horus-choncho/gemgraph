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
import zipfile

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QImage
from PyQt6.QtWidgets import QApplication

from src.core.document import PrismDocument
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
from src.core.office_exporter import OfficeExporter, ExportFormat
from src.core.sheet import DataSheet, GraphSheet, GraphType, SheetType
from src.core.table import TableType
from src.io.prism_io import PrismIO
from src.ui.graph_canvas import ScientificPlotCanvas, compute_isometric_projection
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

    def test_depth_graph_isometric_math(self) -> None:
        """Verifies isometric projection coordinate calculations and back-to-front sorting."""
        # Theta = 45 deg, depth_factor = 0.2
        # cos(45) = sin(45) = sqrt(2)/2 ~= 0.70710678
        x, y, z = 10.0, 50.0, 2.0
        theta_rad = math.radians(45.0)
        expected_xp = 10.0 + 2.0 * math.cos(theta_rad) * 0.2
        expected_yp = 50.0 + 2.0 * math.sin(theta_rad) * 0.2

        xp, yp = compute_isometric_projection(x, y, z, theta_deg=45.0, depth_factor=0.2)
        self.assertAlmostEqual(xp, expected_xp, places=5)
        self.assertAlmostEqual(yp, expected_yp, places=5)

        # Static method on ScientificPlotCanvas
        xp_static, yp_static = ScientificPlotCanvas.compute_isometric_projection(x, y, z, theta_deg=45.0, depth_factor=0.2)
        self.assertAlmostEqual(xp_static, expected_xp, places=5)
        self.assertAlmostEqual(yp_static, expected_yp, places=5)

        # Verify back-to-front topological ordering: higher Z rendered before lower Z
        layers = [
            {"series_idx": 0, "z": 0.0, "name": "Front"},
            {"series_idx": 1, "z": 1.0, "name": "Middle"},
            {"series_idx": 2, "z": 2.0, "name": "Back"},
        ]
        # Sorted back-to-front: Z=2.0 -> Z=1.0 -> Z=0.0
        sorted_layers = sorted(layers, key=lambda l: l["z"], reverse=True)
        self.assertEqual(sorted_layers[0]["name"], "Back")
        self.assertEqual(sorted_layers[1]["name"], "Middle")
        self.assertEqual(sorted_layers[2]["name"], "Front")

    def test_depth_graph_canvas_rendering(self) -> None:
        """Verifies ScientificPlotCanvas renders Depth Graph (2.5D Isometric) for Column and Grouped tables."""
        canvas = ScientificPlotCanvas(parent=None)

        # Render for Column table
        canvas.update_data(self.data_sheet.table, graph_type=GraphType.DEPTH_GRAPH, sheet=self.graph_sheet)
        self.assertGreater(len(canvas.plot_widget.plotItem.items), 0)

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


if __name__ == "__main__":
    unittest.main()
