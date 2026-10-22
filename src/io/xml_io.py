"""
Prism XML Schema Importer and Exporter for GemGraph.
Derived from PrismXMLSchema.xml analysis (REA-CORE-SCHEMA-208).
"""

from __future__ import annotations
import xml.etree.ElementTree as ET
from typing import Optional
import numpy as np
from src.core.document import PrismDocument
from src.core.sheet import DataSheet, SheetType
from src.core.table import DataTable, TableType, XFormat, YFormat


class PrismXMLIO:
    """
    Parses and serializes legacy GraphPad Prism XML documents (.xml / .pzm).
    """

    @classmethod
    def export_xml(cls, document: PrismDocument, file_path: str) -> None:
        """Exports document data sheets to Prism XML Schema."""
        root = ET.Element("GraphPadPrismFile", PrismXMLVersion="5.00")
        created = ET.SubElement(root, "Created")
        ET.SubElement(
            created,
            "OriginalVersion",
            CreatedByProgram="GemGraph",
            CreatedByVersion="1.0",
        )

        table_seq = ET.SubElement(root, "TableSequence")

        for sheet in document.get_sheets_by_type(SheetType.DATA):
            ET.SubElement(table_seq, "Ref", ID=sheet.sheet_id)

            table: DataTable = sheet.table
            t_elem = ET.SubElement(
                root,
                "Table",
                ID=sheet.sheet_id,
                TableType=table.table_type.value,
                XFormat=table.x_format.value,
                YFormat=table.y_format.value,
                Replicates=str(table.num_replicates),
            )
            title_elem = ET.SubElement(t_elem, "Title")
            title_elem.text = sheet.title

            # Row titles if any
            row_titles_col = ET.SubElement(t_elem, "RowTitlesColumn")
            subcol = ET.SubElement(row_titles_col, "Subcolumn")
            for rt in table.row_titles:
                d = ET.SubElement(subcol, "d")
                d.text = rt

            # X Column
            if table.x_data is not None:
                x_col = ET.SubElement(t_elem, "XColumn")
                x_title = ET.SubElement(x_col, "Title")
                x_title.text = table.x_title or "X"
                x_sub = ET.SubElement(x_col, "Subcolumn")
                for r in range(table.num_rows):
                    val = table.x_data[r]
                    d = ET.SubElement(x_sub, "d")
                    if table.is_excluded(r, "X", 0, 0):
                        d.set("Excluded", "1")
                    if not np.isnan(val):
                        d.text = f"{val:.8g}"

            # Y Columns
            for g_idx, g_title in enumerate(table.y_group_titles):
                y_col = ET.SubElement(t_elem, "YColumn")
                y_title = ET.SubElement(y_col, "Title")
                y_title.text = g_title
                for s_idx in range(table.num_replicates):
                    y_sub = ET.SubElement(y_col, "Subcolumn")
                    for r in range(table.num_rows):
                        val = table.y_data[g_idx, r, s_idx]
                        d = ET.SubElement(y_sub, "d")
                        if table.is_excluded(r, "Y", g_idx, s_idx):
                            d.set("Excluded", "1")
                        if not np.isnan(val):
                            d.text = f"{val:.8g}"

        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(file_path, encoding="utf-8", xml_declaration=True)

    @classmethod
    def import_xml(cls, file_path: str) -> PrismDocument:
        """Parses a GraphPad Prism XML file into a PrismDocument."""
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Strip XML namespaces for seamless tag matching across official Prism and cleanroom files
        for el in root.iter():
            if "}" in el.tag:
                el.tag = el.tag.split("}", 1)[1]

        doc = PrismDocument(title="Imported Prism XML")

        for t_elem in root.findall(".//Table"):
            t_id = t_elem.get("ID", "table_1")
            ttype_str = t_elem.get("TableType", "XY")
            if ttype_str == "TwoWay":
                table_type = TableType.GROUPED
            elif ttype_str == "OneWay":
                table_type = TableType.COLUMN
            else:
                try:
                    table_type = TableType(ttype_str)
                except ValueError:
                    table_type = TableType.XY

            replicates = int(t_elem.get("Replicates", "1"))
            title_node = t_elem.find("Title")
            title = title_node.text if title_node is not None and title_node.text else "Data Table"

            # Parse Row titles if any
            row_titles = []
            rt_col = t_elem.find("RowTitlesColumn")
            if rt_col is not None:
                rt_sub = rt_col.find("Subcolumn")
                if rt_sub is not None:
                    for d in rt_sub.findall("d"):
                        row_titles.append(d.text.strip() if d.text else "")

            # Parse X values (support multiple X columns)
            x_cols = t_elem.findall("XColumn")
            x_cols_data = []
            for xc in x_cols:
                xt_node = xc.find("Title")
                xt = xt_node.text if xt_node is not None and xt_node.text else "X"
                vals = []
                exs = []
                x_sub = xc.find("Subcolumn")
                if x_sub is not None:
                    for d in x_sub.findall("d"):
                        ex = d.get("Excluded", "0") in ("1", "true")
                        exs.append(ex)
                        text = d.text.strip() if d.text else ""
                        try:
                            vals.append(float(text) if text else np.nan)
                        except ValueError:
                            vals.append(np.nan)
                x_cols_data.append((xt, vals, exs))

            # Parse Y columns
            y_cols = t_elem.findall("YColumn")
            y_titles = []
            y_subcol_data = []  # list of list of list of floats: [group][subcol][row]

            for y_col in y_cols:
                yt_node = y_col.find("Title")
                g_title = yt_node.text if yt_node is not None and yt_node.text else f"Group {len(y_titles)+1}"
                y_titles.append(g_title)

                subcols = y_col.findall("Subcolumn")
                group_reps = []
                for sub in subcols:
                    rep_vals = []
                    rep_ex = []
                    for d in sub.findall("d"):
                        ex = d.get("Excluded", "0") in ("1", "true")
                        rep_ex.append(ex)
                        text = d.text.strip() if d.text else ""
                        try:
                            rep_vals.append((float(text) if text else np.nan, ex))
                        except ValueError:
                            rep_vals.append((np.nan, ex))
                    group_reps.append(rep_vals)
                y_subcol_data.append(group_reps)

            # Determine row count
            num_rows = len(row_titles)
            for _, x_vals, _ in x_cols_data:
                num_rows = max(num_rows, len(x_vals))
            for g in y_subcol_data:
                for rep in g:
                    num_rows = max(num_rows, len(rep))
            num_rows = max(num_rows, 10)

            # Create table
            table = DataTable(
                table_id=t_id,
                title=title,
                table_type=table_type,
                num_rows=num_rows,
                num_replicates=max(1, len(y_subcol_data[0]) if y_subcol_data and y_subcol_data[0] else replicates),
            )
            if row_titles:
                padded_rt = list(row_titles) + ["" for _ in range(max(0, num_rows - len(row_titles)))]
                table.row_titles = padded_rt[:num_rows]

            if x_cols_data:
                table.x_columns = []
                for x_idx, (xt, x_vals, x_exs) in enumerate(x_cols_data):
                    col_arr = np.full(num_rows, np.nan, dtype=np.float64)
                    for r, v in enumerate(x_vals):
                        if r < num_rows:
                            col_arr[r] = v
                    for r, ex in enumerate(x_exs):
                        if ex and r < num_rows:
                            table.set_excluded(r, "X", x_idx, 0, True)
                    table.x_columns.append({"title": xt, "data": col_arr})

            if y_titles:
                table.y_group_titles = y_titles
                table.y_data = np.full(
                    (len(y_titles), num_rows, table.num_replicates), np.nan, dtype=np.float64
                )
                for g_idx, g in enumerate(y_subcol_data):
                    for s_idx, rep in enumerate(g):
                        if s_idx >= table.num_replicates:
                            break
                        for r_idx, (val, ex) in enumerate(rep):
                            if r_idx < num_rows:
                                table.y_data[g_idx, r_idx, s_idx] = val
                                if ex:
                                    table.set_excluded(r_idx, "Y", g_idx, s_idx, True)

            sheet = DataSheet(sheet_id=t_id, title=title, table=table)
            doc.add_sheet(sheet)

        return doc
