"""
Zero-Subprocess Granular Office & PDF Document Export Engine for GemGraph.
Supports exporting entire projects or granularly selected items (Info Sheets, Data Tables,
Analysis Results, and High-DPI Graph Renders) to .docx, .odt, .xlsx, .ods, and .pdf.

Strictly enforces 'No-Sudo-Strict-Local' policy: pure user-space Python, zero external binaries,
zero subprocess/os.system calls, standard-compliant ZIP+XML and Qt PDF writers.
"""

from __future__ import annotations
import base64
import html
import io
import math
import os
import re
import zipfile
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PyQt6.QtCore import QByteArray, QBuffer, QIODevice, QSizeF, Qt
from PyQt6.QtGui import QColor, QImage, QPainter, QPageSize, QPdfWriter, QTextDocument

from src.core.document import PrismDocument
from src.core.sheet import DataSheet, GraphSheet, InfoSheet, ResultSheet, SheetType
from src.core.table import DataTable


class ExportFormat(str, Enum):
    DOCX = "docx"
    ODT = "odt"
    XLSX = "xlsx"
    ODS = "ods"
    PDF = "pdf"


def col_letter(col_idx: int) -> str:
    """Converts 0-based column index to Excel column name (0 -> A, 27 -> AB)."""
    result = ""
    while col_idx >= 0:
        result = chr(col_idx % 26 + 65) + result
        col_idx = col_idx // 26 - 1
    return result


def sanitize_xml(text: Any) -> str:
    """Escapes string for safe XML text inclusion."""
    if text is None:
        return ""
    s = str(text)
    # Remove control characters invalid in XML
    s = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", s)
    return html.escape(s)


def sanitize_sheet_name(name: str) -> str:
    """Sanitizes sheet title for Excel/ODS tab name constraints."""
    cleaned = re.sub(r"[\\/*?:\[\]]", "_", name)
    return cleaned[:31] if cleaned else "Sheet"


def render_graph_to_png_bytes(
    graph_sheet: GraphSheet,
    document: PrismDocument,
    graph_canvas: Optional[Any] = None,
    width: int = 1200,
    height: int = 800,
) -> bytes:
    """Renders a GraphSheet to high-DPI PNG bytes offscreen with zero subprocesses."""
    canvas = graph_canvas
    created_canvas = False
    if canvas is None:
        from src.ui.graph_canvas import GraphCanvasWidget
        canvas = GraphCanvasWidget()
        canvas.resize(width // 2, height // 2)
        created_canvas = True

    src_sheet = document.get_sheet(graph_sheet.source_sheet_id)
    src_table = src_sheet.table if isinstance(src_sheet, DataSheet) else None
    res_sheet = (
        document.get_sheet(graph_sheet.result_sheet_id)
        if graph_sheet.result_sheet_id
        else None
    )

    canvas.update_data(
        src_table,
        result=res_sheet if isinstance(res_sheet, ResultSheet) else None,
        graph_type=graph_sheet.graph_type,
        sheet=graph_sheet,
    )

    img = QImage(width, height, QImage.Format.Format_ARGB32)
    img.fill(QColor(255, 255, 255))
    painter = QPainter(img)
    try:
        canvas.plot_widget.render(painter)
    finally:
        painter.end()

    ba = QByteArray()
    buf = QBuffer(ba)
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    img.save(buf, "PNG")
    png_bytes = bytes(ba)

    if created_canvas:
        canvas.deleteLater()

    return png_bytes


class OfficeExporter:
    """Granular document exporter producing standard-compliant .docx, .odt, .xlsx, .ods, and .pdf files."""

    def __init__(self, document: PrismDocument, graph_canvas: Optional[Any] = None) -> None:
        self.document = document
        self.graph_canvas = graph_canvas

    def export(
        self,
        file_path: str,
        fmt: Optional[ExportFormat] = None,
        selected_sheet_ids: Optional[List[str]] = None,
    ) -> str:
        """
        Exports the document or selected items to target file path.
        Infers format from file extension if fmt is not explicitly specified.
        """
        if fmt is None:
            ext = os.path.splitext(file_path)[1].lower().lstrip(".")
            try:
                fmt = ExportFormat(ext)
            except ValueError:
                fmt = ExportFormat.DOCX

        if fmt == ExportFormat.DOCX:
            return self.export_docx(file_path, selected_sheet_ids)
        elif fmt == ExportFormat.ODT:
            return self.export_odt(file_path, selected_sheet_ids)
        elif fmt == ExportFormat.XLSX:
            return self.export_xlsx(file_path, selected_sheet_ids)
        elif fmt == ExportFormat.ODS:
            return self.export_ods(file_path, selected_sheet_ids)
        elif fmt == ExportFormat.PDF:
            return self.export_pdf(file_path, selected_sheet_ids)
        else:
            raise ValueError(f"Unsupported export format: {fmt}")

    def _get_export_items(
        self, selected_sheet_ids: Optional[List[str]]
    ) -> Tuple[List[InfoSheet], List[DataSheet], List[ResultSheet], List[GraphSheet]]:
        info_sheets: List[InfoSheet] = []
        data_sheets: List[DataSheet] = []
        result_sheets: List[ResultSheet] = []
        graph_sheets: List[GraphSheet] = []

        target_ids = set(selected_sheet_ids) if selected_sheet_ids is not None else set(self.document.sheet_order)

        for sid in self.document.sheet_order:
            if sid not in target_ids:
                continue
            sheet = self.document.sheets.get(sid)
            if sheet is None:
                continue
            if sheet.sheet_type == SheetType.INFO and isinstance(sheet, InfoSheet):
                info_sheets.append(sheet)
            elif sheet.sheet_type == SheetType.DATA and isinstance(sheet, DataSheet):
                data_sheets.append(sheet)
            elif sheet.sheet_type == SheetType.RESULTS and isinstance(sheet, ResultSheet):
                result_sheets.append(sheet)
            elif sheet.sheet_type == SheetType.GRAPH and isinstance(sheet, GraphSheet):
                graph_sheets.append(sheet)

        return info_sheets, data_sheets, result_sheets, graph_sheets

    # -------------------------------------------------------------------------
    # 1. Word Document (.docx)
    # -------------------------------------------------------------------------
    def export_docx(self, file_path: str, selected_sheet_ids: Optional[List[str]] = None) -> str:
        info_sheets, data_sheets, result_sheets, graph_sheets = self._get_export_items(selected_sheet_ids)

        graph_images: List[Tuple[str, bytes]] = []
        for g in graph_sheets:
            try:
                png_bytes = render_graph_to_png_bytes(g, self.document, self.graph_canvas)
                graph_images.append((g.title, png_bytes))
            except Exception:
                pass

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            # [Content_Types].xml
            ct_xml = [
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
                '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
                '  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
                '  <Default Extension="xml" ContentType="application/xml"/>',
                '  <Default Extension="png" ContentType="image/png"/>',
                '  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>',
                '  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>',
                '</Types>',
            ]
            zf.writestr("[Content_Types].xml", "\n".join(ct_xml))

            # _rels/.rels
            root_rels = [
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
                '  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>',
                '</Relationships>',
            ]
            zf.writestr("_rels/.rels", "\n".join(root_rels))

            # word/_rels/document.xml.rels
            doc_rels = [
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
                '  <Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>',
            ]
            for idx, _ in enumerate(graph_images, 1):
                doc_rels.append(
                    f'  <Relationship Id="rIdImg{idx}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/image{idx}.png"/>'
                )
            doc_rels.append('</Relationships>')
            zf.writestr("word/_rels/document.xml.rels", "\n".join(doc_rels))

            # Write media images
            for idx, (_, png_data) in enumerate(graph_images, 1):
                zf.writestr(f"word/media/image{idx}.png", png_data)

            # word/styles.xml
            styles_xml = [
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
                '<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">',
                '  <w:docDefaults>',
                '    <w:rPrDefault><w:rPr><w:rFonts w:ascii="Liberation Sans" w:hAnsi="Liberation Sans"/><w:sz w:val="24"/></w:rPr></w:rPrDefault>',
                '  </w:docDefaults>',
                '  <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="Heading 1"/><w:rPr><w:b/><w:sz w:val="36"/><w:color w:val="1F497D"/></w:rPr></w:style>',
                '  <w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="Heading 2"/><w:rPr><w:b/><w:sz w:val="28"/><w:color w:val="365F91"/></w:rPr></w:style>',
                '</w:styles>',
            ]
            zf.writestr("word/styles.xml", "\n".join(styles_xml))

            # word/document.xml body builder
            body_elems: List[str] = []

            # Project title
            doc_title = sanitize_xml(self.document.title or "GemGraph Scientific Report")
            body_elems.append(f'<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>{doc_title}</w:t></w:r></w:p>')

            # Info Sheets
            for info in info_sheets:
                body_elems.append(f'<w:p><w:pPr><w:pStyle w:val="Heading2"/></w:pPr><w:r><w:t>Info: {sanitize_xml(info.title)}</w:t></w:r></w:p>')
                # GLP Table
                props = [(k, v) for k, v in info.constants.items() if v]
                if props:
                    body_elems.append('<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/><w:tblW w:w="5000" w:type="pct"/></w:tblPr>')
                    for prop_name, prop_val in props:
                        body_elems.append(
                            f'<w:tr><w:tc><w:tcPr><w:shd w:fill="F0F4F8"/></w:tcPr><w:p><w:r><w:rPr><w:b/></w:rPr><w:t>{sanitize_xml(prop_name)}</w:t></w:r></w:p></w:tc>'
                            f'<w:tc><w:p><w:r><w:t>{sanitize_xml(prop_val)}</w:t></w:r></w:p></w:tc></w:tr>'
                        )
                    body_elems.append('</w:tbl>')

                # Experimental Notes
                if info.notes:
                    body_elems.append('<w:p><w:r><w:rPr><w:b/><w:i/></w:rPr><w:t>Experimental Notes &amp; Observations:</w:t></w:r></w:p>')
                    for line in info.notes.splitlines():
                        s_line = line.strip()
                        if s_line.startswith("- ") or s_line.startswith("* "):
                            body_elems.append(f'<w:p><w:pPr><w:ind w:left="360"/></w:pPr><w:r><w:t>• {sanitize_xml(s_line[2:])}</w:t></w:r></w:p>')
                        elif s_line:
                            body_elems.append(f'<w:p><w:r><w:t>{sanitize_xml(s_line)}</w:t></w:r></w:p>')

            # Data Tables
            for ds in data_sheets:
                body_elems.append(f'<w:p><w:pPr><w:pStyle w:val="Heading2"/></w:pPr><w:r><w:t>Data Table: {sanitize_xml(ds.title)}</w:t></w:r></w:p>')
                df = ds.table.to_dataframe()
                body_elems.append('<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/><w:tblW w:w="5000" w:type="pct"/></w:tblPr>')
                # Header row
                body_elems.append('<w:tr>')
                for col in df.columns:
                    body_elems.append(f'<w:tc><w:tcPr><w:shd w:fill="E9EEF4"/></w:tcPr><w:p><w:r><w:rPr><w:b/></w:rPr><w:t>{sanitize_xml(col)}</w:t></w:r></w:p></w:tc>')
                body_elems.append('</w:tr>')
                # Data rows (max 1000 rows to keep document performant)
                for _, row in df.head(1000).iterrows():
                    body_elems.append('<w:tr>')
                    for v in row:
                        val_str = "" if pd_isna(v) else f"{v:.6g}" if isinstance(v, (int, float, np.floating, np.integer)) else str(v)
                        body_elems.append(f'<w:tc><w:p><w:r><w:t>{sanitize_xml(val_str)}</w:t></w:r></w:p></w:tc>')
                    body_elems.append('</w:tr>')
                body_elems.append('</w:tbl>')

            # Analysis Results
            for res in result_sheets:
                body_elems.append(f'<w:p><w:pPr><w:pStyle w:val="Heading2"/></w:pPr><w:r><w:t>Results: {sanitize_xml(res.title)} ({sanitize_xml(res.analysis_type)})</w:t></w:r></w:p>')
                for tab in res.tabs:
                    tab_name = tab.get("name", "Analysis Ledger")
                    headers = tab.get("headers", [])
                    rows = tab.get("rows", [])
                    body_elems.append(f'<w:p><w:r><w:rPr><w:b/></w:rPr><w:t>{sanitize_xml(tab_name)}</w:t></w:r></w:p>')
                    if headers:
                        body_elems.append('<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/><w:tblW w:w="5000" w:type="pct"/></w:tblPr>')
                        body_elems.append('<w:tr>')
                        for h in headers:
                            body_elems.append(f'<w:tc><w:tcPr><w:shd w:fill="E2EFDA"/></w:tcPr><w:p><w:r><w:rPr><w:b/></w:rPr><w:t>{sanitize_xml(h)}</w:t></w:r></w:p></w:tc>')
                        body_elems.append('</w:tr>')
                        for r in rows:
                            body_elems.append('<w:tr>')
                            for c in r:
                                val_str = "" if c is None else str(c)
                                body_elems.append(f'<w:tc><w:p><w:r><w:t>{sanitize_xml(val_str)}</w:t></w:r></w:p></w:tc>')
                            body_elems.append('</w:tr>')
                        body_elems.append('</w:tbl>')

            # Embedded Graph Images
            for idx, (g_title, _) in enumerate(graph_images, 1):
                body_elems.append(f'<w:p><w:pPr><w:pStyle w:val="Heading2"/></w:pPr><w:r><w:t>Graph: {sanitize_xml(g_title)}</w:t></w:r></w:p>')
                cx = 5486400  # 6 inches in EMU
                cy = 3657600  # 4 inches in EMU
                drawing_xml = (
                    f'<w:p><w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0">'
                    f'<wp:extent cx="{cx}" cy="{cy}"/>'
                    f'<wp:docPr id="{idx}" name="Picture {idx}"/>'
                    f'<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
                    f'<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
                    f'<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
                    f'<pic:nvPicPr><pic:cNvPr id="{idx}" name="Image{idx}.png"/><pic:cNvPicPr/></pic:nvPicPr>'
                    f'<pic:blipFill><a:blip r:embed="rIdImg{idx}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
                    f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
                    f'</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>'
                )
                body_elems.append(drawing_xml)

            document_xml = (
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
                '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
                'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
                'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing">\n'
                '<w:body>\n' + "\n".join(body_elems) + '\n<w:sectPr/></w:body></w:document>'
            )
            zf.writestr("word/document.xml", document_xml)

        with open(file_path, "wb") as f:
            f.write(buf.getvalue())
        return file_path

    # -------------------------------------------------------------------------
    # 2. OpenDocument Text (.odt)
    # -------------------------------------------------------------------------
    def export_odt(self, file_path: str, selected_sheet_ids: Optional[List[str]] = None) -> str:
        info_sheets, data_sheets, result_sheets, graph_sheets = self._get_export_items(selected_sheet_ids)

        graph_images: List[Tuple[str, bytes]] = []
        for g in graph_sheets:
            try:
                png_bytes = render_graph_to_png_bytes(g, self.document, self.graph_canvas)
                graph_images.append((g.title, png_bytes))
            except Exception:
                pass

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            # 1. mimetype (MUST be first and uncompressed)
            zf.writestr("mimetype", "application/vnd.oasis.opendocument.text", compress_type=zipfile.ZIP_STORED)

            # 2. META-INF/manifest.xml
            manifest_entries = [
                '  <manifest:file-entry manifest:full-path="/" manifest:version="1.2" manifest:media-type="application/vnd.oasis.opendocument.text"/>',
                '  <manifest:file-entry manifest:full-path="content.xml" manifest:media-type="text/xml"/>',
                '  <manifest:file-entry manifest:full-path="styles.xml" manifest:media-type="text/xml"/>',
            ]
            for idx, _ in enumerate(graph_images, 1):
                manifest_entries.append(
                    f'  <manifest:file-entry manifest:full-path="Pictures/image{idx}.png" manifest:media-type="image/png"/>'
                )
            manifest_xml = (
                '<?xml version="1.0" encoding="UTF-8"?>\n'
                '<manifest:manifest xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0" manifest:version="1.2">\n'
                + "\n".join(manifest_entries)
                + '\n</manifest:manifest>'
            )
            zf.writestr("META-INF/manifest.xml", manifest_xml)

            # Write Pictures
            for idx, (_, png_data) in enumerate(graph_images, 1):
                zf.writestr(f"Pictures/image{idx}.png", png_data)

            # 3. styles.xml
            styles_xml = (
                '<?xml version="1.0" encoding="UTF-8"?>\n'
                '<office:document-styles xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
                'xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0" '
                'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" office:version="1.2">\n'
                '  <office:styles>\n'
                '    <style:style style:name="Standard" style:family="paragraph"/>\n'
                '  </office:styles>\n'
                '</office:document-styles>'
            )
            zf.writestr("styles.xml", styles_xml)

            # 4. content.xml
            text_body: List[str] = []
            doc_title = sanitize_xml(self.document.title or "GemGraph Scientific Report")
            text_body.append(f'<text:h text:outline-level="1">{doc_title}</text:h>')

            for info in info_sheets:
                text_body.append(f'<text:h text:outline-level="2">Info: {sanitize_xml(info.title)}</text:h>')
                props = [(k, v) for k, v in info.constants.items() if v]
                if props:
                    text_body.append(f'<table:table table:name="Info_{sanitize_xml(info.sheet_id)}">')
                    text_body.append('<table:table-column table:number-columns-repeated="2"/>')
                    for k, v in props:
                        text_body.append(
                            f'<table:table-row>'
                            f'<table:table-cell office:value-type="string"><text:p><text:span text:style-name="Bold">{sanitize_xml(k)}</text:span></text:p></table:table-cell>'
                            f'<table:table-cell office:value-type="string"><text:p>{sanitize_xml(v)}</text:p></table:table-cell>'
                            f'</table:table-row>'
                        )
                    text_body.append('</table:table>')

                if info.notes:
                    text_body.append('<text:p><text:span text:style-name="Bold">Experimental Notes &amp; Observations:</text:span></text:p>')
                    for line in info.notes.splitlines():
                        s_line = line.strip()
                        if s_line.startswith("- ") or s_line.startswith("* "):
                            text_body.append(f'<text:list><text:list-item><text:p>{sanitize_xml(s_line[2:])}</text:p></text:list-item></text:list>')
                        elif s_line:
                            text_body.append(f'<text:p>{sanitize_xml(s_line)}</text:p>')

            for ds in data_sheets:
                text_body.append(f'<text:h text:outline-level="2">Data Table: {sanitize_xml(ds.title)}</text:h>')
                df = ds.table.to_dataframe()
                text_body.append(f'<table:table table:name="Data_{sanitize_xml(ds.sheet_id)}">')
                text_body.append(f'<table:table-column table:number-columns-repeated="{len(df.columns)}"/>')
                # Header row
                text_body.append('<table:table-row>')
                for col in df.columns:
                    text_body.append(f'<table:table-cell office:value-type="string"><text:p>{sanitize_xml(col)}</text:p></table:table-cell>')
                text_body.append('</table:table-row>')
                for _, row in df.head(1000).iterrows():
                    text_body.append('<table:table-row>')
                    for v in row:
                        if pd_isna(v):
                            text_body.append('<table:table-cell/>')
                        elif isinstance(v, (int, float, np.floating, np.integer)):
                            text_body.append(f'<table:table-cell office:value-type="float" office:value="{v}"><text:p>{v:.6g}</text:p></table:table-cell>')
                        else:
                            text_body.append(f'<table:table-cell office:value-type="string"><text:p>{sanitize_xml(str(v))}</text:p></table:table-cell>')
                    text_body.append('</table:table-row>')
                text_body.append('</table:table>')

            for res in result_sheets:
                text_body.append(f'<text:h text:outline-level="2">Results: {sanitize_xml(res.title)} ({sanitize_xml(res.analysis_type)})</text:h>')
                for t_idx, tab in enumerate(res.tabs):
                    t_name = tab.get("name", "Ledger")
                    headers = tab.get("headers", [])
                    rows = tab.get("rows", [])
                    text_body.append(f'<text:p><text:span text:style-name="Bold">{sanitize_xml(t_name)}</text:span></text:p>')
                    if headers:
                        text_body.append(f'<table:table table:name="Res_{sanitize_xml(res.sheet_id)}_{t_idx}">')
                        text_body.append(f'<table:table-column table:number-columns-repeated="{len(headers)}"/>')
                        text_body.append('<table:table-row>')
                        for h in headers:
                            text_body.append(f'<table:table-cell office:value-type="string"><text:p>{sanitize_xml(h)}</text:p></table:table-cell>')
                        text_body.append('</table:table-row>')
                        for r in rows:
                            text_body.append('<table:table-row>')
                            for c in r:
                                text_body.append(f'<table:table-cell office:value-type="string"><text:p>{sanitize_xml("" if c is None else str(c))}</text:p></table:table-cell>')
                            text_body.append('</table:table-row>')
                        text_body.append('</table:table>')

            for idx, (g_title, _) in enumerate(graph_images, 1):
                text_body.append(f'<text:h text:outline-level="2">Graph: {sanitize_xml(g_title)}</text:h>')
                text_body.append(
                    f'<text:p><draw:frame draw:name="Graph_{idx}" text:anchor-type="paragraph" svg:width="15cm" svg:height="10cm">'
                    f'<draw:image xlink:href="Pictures/image{idx}.png" xlink:type="simple" xlink:show="embed" xlink:actuate="onLoad"/>'
                    f'</draw:frame></text:p>'
                )

            content_xml = (
                '<?xml version="1.0" encoding="UTF-8"?>\n'
                '<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
                'xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0" '
                'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" '
                'xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" '
                'xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0" '
                'xmlns:xlink="http://www.w3.org/1999/xlink" '
                'xmlns:svg="urn:oasis:names:tc:opendocument:xmlns:svg-compatible:1.0" office:version="1.2">\n'
                '  <office:body>\n'
                '    <office:text>\n' + "\n".join(text_body) + '\n    </office:text>\n'
                '  </office:body>\n'
                '</office:document-content>'
            )
            zf.writestr("content.xml", content_xml)

        with open(file_path, "wb") as f:
            f.write(buf.getvalue())
        return file_path

    # -------------------------------------------------------------------------
    # 3. Excel Workbook (.xlsx)
    # -------------------------------------------------------------------------
    def export_xlsx(self, file_path: str, selected_sheet_ids: Optional[List[str]] = None) -> str:
        info_sheets, data_sheets, result_sheets, graph_sheets = self._get_export_items(selected_sheet_ids)

        worksheets: List[Tuple[str, List[List[Any]]]] = []

        # Info sheet tab
        if info_sheets:
            info_rows: List[List[Any]] = [["Category", "Property", "Value"]]
            for info in info_sheets:
                info_rows.append([f"Sheet: {info.title}", "", ""])
                for k, v in info.constants.items():
                    if v:
                        info_rows.append(["GLP Specification", k, v])
                if info.notes:
                    info_rows.append(["Experimental Notes", "Protocol Notes", info.notes])
            worksheets.append(("Info Sheets", info_rows))

        # Data tables
        for ds in data_sheets:
            df = ds.table.to_dataframe()
            tab_rows: List[List[Any]] = [list(df.columns)]
            for _, r in df.head(5000).iterrows():
                tab_rows.append(list(r))
            worksheets.append((sanitize_sheet_name(ds.title), tab_rows))

        # Results
        for res in result_sheets:
            for t_idx, tab in enumerate(res.tabs):
                name = sanitize_sheet_name(f"{res.title}_{tab.get('name', 'Tab')}")
                tab_rows = []
                headers = tab.get("headers", [])
                if headers:
                    tab_rows.append(headers)
                for row_data in tab.get("rows", []):
                    tab_rows.append(row_data)
                worksheets.append((name, tab_rows))

        if not worksheets:
            worksheets.append(("GemGraph Data", [["Empty Project"]]))

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            # [Content_Types].xml
            ct_lines = [
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
                '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
                '  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
                '  <Default Extension="xml" ContentType="application/xml"/>',
                '  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
                '  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
            ]
            for idx in range(1, len(worksheets) + 1):
                ct_lines.append(
                    f'  <Override PartName="/xl/worksheets/sheet{idx}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                )
            ct_lines.append('</Types>')
            zf.writestr("[Content_Types].xml", "\n".join(ct_lines))

            # _rels/.rels
            root_rels = [
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
                '  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>',
                '</Relationships>',
            ]
            zf.writestr("_rels/.rels", "\n".join(root_rels))

            # xl/_rels/workbook.xml.rels
            wb_rels = [
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
                '  <Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>',
            ]
            for idx in range(1, len(worksheets) + 1):
                wb_rels.append(
                    f'  <Relationship Id="rId{idx}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{idx}.xml"/>'
                )
            wb_rels.append('</Relationships>')
            zf.writestr("xl/_rels/workbook.xml.rels", "\n".join(wb_rels))

            # xl/styles.xml
            styles_xml = (
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
                '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">\n'
                '  <fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>\n'
                '  <fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills>\n'
                '  <borders count="1"><border><left/><right/><top/><bottom/></border></borders>\n'
                '  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>\n'
                '  <cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>\n'
                '</styleSheet>'
            )
            zf.writestr("xl/styles.xml", styles_xml)

            # xl/workbook.xml
            wb_sheets = []
            used_names = set()
            for idx, (sheet_title, _) in enumerate(worksheets, 1):
                safe_name = sheet_title
                suffix = 1
                while safe_name in used_names:
                    safe_name = f"{sheet_title[:28]}_{suffix}"
                    suffix += 1
                used_names.add(safe_name)
                wb_sheets.append(f'    <sheet name="{sanitize_xml(safe_name)}" sheetId="{idx}" r:id="rId{idx}"/>')

            wb_xml = (
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
                '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
                'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">\n'
                '  <sheets>\n' + "\n".join(wb_sheets) + '\n  </sheets>\n</workbook>'
            )
            zf.writestr("xl/workbook.xml", wb_xml)

            # xl/worksheets/sheetN.xml
            for idx, (_, rows) in enumerate(worksheets, 1):
                sheet_lines = [
                    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
                    '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">',
                    '  <sheetData>',
                ]
                for r_idx, row_vals in enumerate(rows, 1):
                    sheet_lines.append(f'    <row r="{r_idx}">')
                    for c_idx, val in enumerate(row_vals):
                        cref = f"{col_letter(c_idx)}{r_idx}"
                        if pd_isna(val):
                            continue
                        elif isinstance(val, (int, float, np.floating, np.integer)):
                            if math.isinf(val) or math.isnan(val):
                                sheet_lines.append(f'      <c r="{cref}" t="inlineStr"><is><t>{val}</t></is></c>')
                            else:
                                sheet_lines.append(f'      <c r="{cref}"><v>{val}</v></c>')
                        else:
                            sheet_lines.append(f'      <c r="{cref}" t="inlineStr"><is><t>{sanitize_xml(str(val))}</t></is></c>')
                    sheet_lines.append('    </row>')
                sheet_lines.append('  </sheetData>')
                sheet_lines.append('</worksheet>')
                zf.writestr(f"xl/worksheets/sheet{idx}.xml", "\n".join(sheet_lines))

        with open(file_path, "wb") as f:
            f.write(buf.getvalue())
        return file_path

    # -------------------------------------------------------------------------
    # 4. OpenDocument Spreadsheet (.ods)
    # -------------------------------------------------------------------------
    def export_ods(self, file_path: str, selected_sheet_ids: Optional[List[str]] = None) -> str:
        info_sheets, data_sheets, result_sheets, _ = self._get_export_items(selected_sheet_ids)

        tables: List[Tuple[str, List[List[Any]]]] = []

        if info_sheets:
            info_rows: List[List[Any]] = [["Category", "Property", "Value"]]
            for info in info_sheets:
                info_rows.append([f"Sheet: {info.title}", "", ""])
                for k, v in info.constants.items():
                    if v:
                        info_rows.append(["GLP Specification", k, v])
                if info.notes:
                    info_rows.append(["Experimental Notes", "Protocol Notes", info.notes])
            tables.append(("Info Sheets", info_rows))

        for ds in data_sheets:
            df = ds.table.to_dataframe()
            t_rows = [list(df.columns)]
            for _, r in df.head(5000).iterrows():
                t_rows.append(list(r))
            tables.append((sanitize_sheet_name(ds.title), t_rows))

        for res in result_sheets:
            for tab in res.tabs:
                name = sanitize_sheet_name(f"{res.title}_{tab.get('name', 'Tab')}")
                t_rows = []
                headers = tab.get("headers", [])
                if headers:
                    t_rows.append(headers)
                for row_data in tab.get("rows", []):
                    t_rows.append(row_data)
                tables.append((name, t_rows))

        if not tables:
            tables.append(("GemGraph Data", [["Empty Project"]]))

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            # 1. mimetype (first, stored uncompressed)
            zf.writestr("mimetype", "application/vnd.oasis.opendocument.spreadsheet", compress_type=zipfile.ZIP_STORED)

            # 2. META-INF/manifest.xml
            manifest_xml = (
                '<?xml version="1.0" encoding="UTF-8"?>\n'
                '<manifest:manifest xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0" manifest:version="1.2">\n'
                '  <manifest:file-entry manifest:full-path="/" manifest:version="1.2" manifest:media-type="application/vnd.oasis.opendocument.spreadsheet"/>\n'
                '  <manifest:file-entry manifest:full-path="content.xml" manifest:media-type="text/xml"/>\n'
                '  <manifest:file-entry manifest:full-path="styles.xml" manifest:media-type="text/xml"/>\n'
                '</manifest:manifest>'
            )
            zf.writestr("META-INF/manifest.xml", manifest_xml)

            # 3. styles.xml
            styles_xml = (
                '<?xml version="1.0" encoding="UTF-8"?>\n'
                '<office:document-styles xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
                'xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0" office:version="1.2">\n'
                '  <office:styles>\n'
                '    <style:style style:name="Default" style:family="table-cell"/>\n'
                '  </office:styles>\n'
                '</office:document-styles>'
            )
            zf.writestr("styles.xml", styles_xml)

            # 4. content.xml
            content_tables: List[str] = []
            used_names = set()
            for t_title, rows in tables:
                safe_name = t_title
                suffix = 1
                while safe_name in used_names:
                    safe_name = f"{t_title[:28]}_{suffix}"
                    suffix += 1
                used_names.add(safe_name)

                content_tables.append(f'      <table:table table:name="{sanitize_xml(safe_name)}">')
                for row_data in rows:
                    content_tables.append('        <table:table-row>')
                    for val in row_data:
                        if pd_isna(val):
                            content_tables.append('          <table:table-cell/>')
                        elif isinstance(val, (int, float, np.floating, np.integer)):
                            if math.isinf(val) or math.isnan(val):
                                content_tables.append(f'          <table:table-cell office:value-type="string"><text:p>{val}</text:p></table:table-cell>')
                            else:
                                content_tables.append(f'          <table:table-cell office:value-type="float" office:value="{val}"><text:p>{val}</text:p></table:table-cell>')
                        else:
                            content_tables.append(f'          <table:table-cell office:value-type="string"><text:p>{sanitize_xml(str(val))}</text:p></table:table-cell>')
                    content_tables.append('        </table:table-row>')
                content_tables.append('      </table:table>')

            content_xml = (
                '<?xml version="1.0" encoding="UTF-8"?>\n'
                '<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
                'xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" '
                'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" office:version="1.2">\n'
                '  <office:body>\n'
                '    <office:spreadsheet>\n' + "\n".join(content_tables) + '\n    </office:spreadsheet>\n'
                '  </office:body>\n'
                '</office:document-content>'
            )
            zf.writestr("content.xml", content_xml)

        with open(file_path, "wb") as f:
            f.write(buf.getvalue())
        return file_path

    # -------------------------------------------------------------------------
    # 5. Portable Document Format (.pdf)
    # -------------------------------------------------------------------------
    def export_pdf(self, file_path: str, selected_sheet_ids: Optional[List[str]] = None) -> str:
        info_sheets, data_sheets, result_sheets, graph_sheets = self._get_export_items(selected_sheet_ids)

        html_parts: List[str] = [
            """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body { font-family: "Liberation Sans", "DejaVu Sans", sans-serif; font-size: 10pt; color: #1a1a1a; margin: 15px; }
  h1 { font-size: 18pt; color: #1f4e79; border-bottom: 2px solid #1f4e79; padding-bottom: 4px; margin-bottom: 12px; }
  h2 { font-size: 13pt; color: #2f5597; margin-top: 16px; margin-bottom: 8px; }
  h3 { font-size: 11pt; color: #333333; margin-top: 10px; margin-bottom: 4px; }
  table { border-collapse: collapse; width: 100%; margin-bottom: 14px; font-size: 9pt; }
  th { background-color: #f2f4f7; color: #111; font-weight: bold; border: 1px solid #d0d7de; padding: 4px 8px; text-align: left; }
  td { border: 1px solid #d0d7de; padding: 4px 8px; vertical-align: top; }
  tr:nth-child(even) td { background-color: #fafbfc; }
  .notes-box { background-color: #fdfdfd; border: 1px solid #e1e4e8; border-left: 4px solid #2f5597; padding: 8px; margin: 8px 0; }
  .graph-container { text-align: center; margin: 16px 0; }
  .graph-container img { max-width: 100%; height: auto; border: 1px solid #e1e4e8; }
</style>
</head>
<body>
"""
        ]

        doc_title = sanitize_xml(self.document.title or "GemGraph Scientific Report")
        html_parts.append(f"<h1>{doc_title}</h1>")

        # Info Sheets
        for info in info_sheets:
            html_parts.append(f"<h2>Info Sheet: {sanitize_xml(info.title)}</h2>")
            props = [(k, v) for k, v in info.constants.items() if v]
            if props:
                html_parts.append("<table><tr><th style='width: 35%;'>GLP Property</th><th>Specification / Value</th></tr>")
                for k, v in props:
                    html_parts.append(f"<tr><td><b>{sanitize_xml(k)}</b></td><td>{sanitize_xml(v)}</td></tr>")
                html_parts.append("</table>")

            if getattr(info, "notes_html", ""):
                html_parts.append(f"<div class='notes-box'><b>Experimental Notes:</b><br>{info.notes_html}</div>")
            elif info.notes:
                notes_escaped = "<br>".join(sanitize_xml(l) for l in info.notes.splitlines())
                html_parts.append(f"<div class='notes-box'><b>Experimental Notes:</b><br>{notes_escaped}</div>")

        # Data Tables
        for ds in data_sheets:
            html_parts.append(f"<h2>Data Table: {sanitize_xml(ds.title)}</h2>")
            df = ds.table.to_dataframe()
            html_parts.append("<table><tr>")
            for col in df.columns:
                html_parts.append(f"<th>{sanitize_xml(col)}</th>")
            html_parts.append("</tr>")
            for _, r in df.head(100).iterrows():
                html_parts.append("<tr>")
                for v in r:
                    val_str = "" if pd_isna(v) else f"{v:.6g}" if isinstance(v, (int, float, np.floating, np.integer)) else str(v)
                    html_parts.append(f"<td>{sanitize_xml(val_str)}</td>")
                html_parts.append("</tr>")
            html_parts.append("</table>")

        # Result Sheets
        for res in result_sheets:
            html_parts.append(f"<h2>Analysis Results: {sanitize_xml(res.title)} ({sanitize_xml(res.analysis_type)})</h2>")
            for tab in res.tabs:
                t_name = tab.get("name", "Ledger")
                headers = tab.get("headers", [])
                rows = tab.get("rows", [])
                html_parts.append(f"<h3>{sanitize_xml(t_name)}</h3>")
                if headers:
                    html_parts.append("<table><tr>")
                    for h in headers:
                        html_parts.append(f"<th>{sanitize_xml(h)}</th>")
                    html_parts.append("</tr>")
                    for r in rows:
                        html_parts.append("<tr>")
                        for c in r:
                            val_str = "" if c is None else str(c)
                            html_parts.append(f"<td>{sanitize_xml(val_str)}</td>")
                        html_parts.append("</tr>")
                    html_parts.append("</table>")

        # Graphs
        for g in graph_sheets:
            html_parts.append(f"<h2>Graph: {sanitize_xml(g.title)}</h2>")
            try:
                png_bytes = render_graph_to_png_bytes(g, self.document, self.graph_canvas)
                b64_str = base64.b64encode(png_bytes).decode("ascii")
                html_parts.append(f"<div class='graph-container'><img src='data:image/png;base64,{b64_str}' /></div>")
            except Exception as e:
                html_parts.append(f"<p><i>(Graph render unavailable: {sanitize_xml(str(e))})</i></p>")

        html_parts.append("</body></html>")
        full_html = "\n".join(html_parts)

        # Print via QTextDocument + QPdfWriter
        doc = QTextDocument()
        doc.setHtml(full_html)

        writer = QPdfWriter(file_path)
        writer.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
        writer.setResolution(300)
        doc.setPageSize(QSizeF(writer.width(), writer.height()))
        doc.print(writer)

        return file_path


def pd_isna(val: Any) -> bool:
    """Checks if a scalar value is None or NaN without pandas dependency."""
    if val is None:
        return True
    try:
        return bool(np.isnan(val))
    except Exception:
        return False
