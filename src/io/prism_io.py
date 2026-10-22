"""
PK-ZIP Container Parser & Serializer for .prism files.
Derived from Core.dll storage format analysis (REA-CORE-STORAGE-204).
Supports both GemGraph cleanroom archives and official GraphPad Prism 10 packages.
"""

from __future__ import annotations
import csv
import io
import json
import os
import re
import struct
import zipfile
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from src.core.document import PrismDocument
from src.core.sheet import BaseSheet, DataSheet, GraphSheet, GraphType, InfoSheet, ResultSheet, SheetType
from src.core.table import DataTable, TableType, XFormat, YFormat


class PrismIO:
    """
    Handles saving and loading of .prism ZIP packages.
    """

    FORMAT_VERSION = "10.0"

    @classmethod
    def save(cls, document: PrismDocument, file_path: str) -> None:
        """Serializes PrismDocument into a .bigo / .prism PK-ZIP archive compatible with GraphPad Prism 10."""
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

        data_uids: List[str] = []
        info_uids: List[str] = []
        graph_uids: List[str] = []
        result_uids: List[str] = []

        for sid in document.sheet_order:
            s = document.sheets.get(sid)
            if s is None:
                continue
            if s.sheet_type == SheetType.DATA:
                data_uids.append(sid)
            elif s.sheet_type == SheetType.INFO:
                info_uids.append(sid)
            elif s.sheet_type == SheetType.GRAPH:
                graph_uids.append(sid)
            elif s.sheet_type == SheetType.RESULTS:
                result_uids.append(sid)

        active_data = None
        if document.active_sheet_id and document.active_sheet_id in document.sheets:
            act_s = document.sheets[document.active_sheet_id]
            if act_s.sheet_type == SheetType.DATA:
                active_data = act_s.sheet_id
        if not active_data and data_uids:
            active_data = data_uids[0]

        manifest = {
            "@class": "Document",
            "$id": "document.schema.json",
            "formatVersion": "1-4-1",
            "minFormatVersion": "1-2-0",
            "minPrismVersion": "10.1.0",
            "generator": "GemGraph Clean-Room 1.0",
            "doc_id": document.doc_id,
            "title": document.title,
            "sheetAttributesMap": {
                s.sheet_id: {
                    "title": s.title,
                    "type": "table" if s.sheet_type == SheetType.DATA else (
                        "info" if s.sheet_type == SheetType.INFO else (
                            "graph" if s.sheet_type == SheetType.GRAPH else "result"
                        )
                    ),
                }
                for s in document.sheets.values()
            },
            "sheets": {
                "data": data_uids,
                "info": info_uids,
                "graphs": graph_uids,
                "results": result_uids,
            },
            "sheetOrder": document.sheet_order,
            "uiSettings": {
                "currentSheets": {
                    "data": active_data or "",
                }
            },
        }

        with zipfile.ZipFile(file_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("document.json", json.dumps(manifest, indent=2))

            for sheet_id in document.sheet_order:
                sheet = document.sheets[sheet_id]
                if isinstance(sheet, DataSheet):
                    table = sheet.table
                    t_uid = table.table_id

                    format_map = {
                        TableType.XY: ("XYDataTable", "xy"),
                        TableType.COLUMN: ("OneWayDataTable", "one_way"),
                        TableType.GROUPED: ("TwoWayDataTable", "two_way"),
                        TableType.CONTINGENCY: ("ContingencyDataTable", "contingency"),
                        TableType.SURVIVAL: ("SurvivalDataTable", "survival"),
                        TableType.PARTS_OF_WHOLE: ("PartsOfWholeDataTable", "parts_of_whole"),
                        TableType.MULTIPLE_VARIABLES: ("MultipleVariablesDataTable", "multiple_variables"),
                        TableType.NESTED: ("NestedDataTable", "nested"),
                    }
                    cls_name, fmt_name = format_map.get(table.table_type, ("XYDataTable", "xy"))

                    x_ds_id = f"X_{t_uid}" if table.x_data is not None else None
                    y_ds_ids = [f"Y_{t_uid}_{g_idx}" for g_idx in range(len(table.y_group_titles))]

                    # 1. data/sheets/<s_uid>/sheet.json
                    sheet_meta: Dict[str, Any] = {
                        "@class": "DataSheet",
                        "$id": "sheets/data_sheet/data_sheet.schema.json",
                        "uid": sheet.sheet_id,
                        "title": sheet.title,
                        "isCustomTitle": True,
                        "table": {
                            "@class": cls_name,
                            "$id": "sheets/data_sheet/data_table/data_table.schema.json",
                            "uid": t_uid,
                            "format": fmt_name,
                            "dataSets": y_ds_ids,
                        },
                    }
                    if x_ds_id:
                        sheet_meta["table"]["xDataSet"] = x_ds_id

                    zf.writestr(
                        f"data/sheets/{sheet.sheet_id}/sheet.json",
                        json.dumps(sheet_meta, indent=2),
                    )

                    # 2. data/sets/<x_ds_id>.json
                    if x_ds_id:
                        x_exclusions = [
                            {"attributes": ["EXCLUDED"], "rows": str(r)}
                            for r in range(table.num_rows)
                            if table.is_excluded(r, "X", 0, 0)
                        ]
                        x_set_meta = {
                            "@class": "DataSet",
                            "$id": "sheets/data_sheet/data_table/data_set.schema.json",
                            "uid": x_ds_id,
                            "format": "y_single",
                            "title": {"string": table.x_title or "X"},
                            "attributes": ["DS_ATTR_X"],
                            "replicates": [
                                {
                                    "@class": "DriverReplicate",
                                    "valueType": "real",
                                    "cellAttributes": x_exclusions,
                                }
                            ],
                        }
                        zf.writestr(f"data/sets/{x_ds_id}.json", json.dumps(x_set_meta, indent=2))

                    # 3. data/sets/<y_ds_id>.json for each group
                    for g_idx, y_ds_id in enumerate(y_ds_ids):
                        g_title = table.y_group_titles[g_idx]
                        reps = []
                        for s_idx in range(table.num_replicates):
                            y_exclusions = [
                                {"attributes": ["EXCLUDED"], "rows": str(r)}
                                for r in range(table.num_rows)
                                if table.is_excluded(r, "Y", g_idx, s_idx)
                            ]
                            reps.append({
                                "@class": "DriverReplicate",
                                "valueType": "real",
                                "cellAttributes": y_exclusions,
                            })
                        y_set_meta = {
                            "@class": "DataSet",
                            "$id": "sheets/data_sheet/data_table/data_set.schema.json",
                            "uid": y_ds_id,
                            "format": "y_single",
                            "title": {"string": g_title},
                            "attributes": ["DS_ATTR_Y"],
                            "replicates": reps,
                        }
                        zf.writestr(f"data/sets/{y_ds_id}.json", json.dumps(y_set_meta, indent=2))

                    # 4. data/tables/<t_uid>/content.json
                    num_cols = (1 if table.x_data is not None else 0) + len(table.y_group_titles) * table.num_replicates
                    if table.table_type == TableType.MULTIPLE_VARIABLES:
                        num_cols = len(table.variables)
                    content_meta = {
                        "$id": "sheets/data_sheet/data_table/data_table_storage.schema.json",
                        "numberOfColumns": num_cols,
                        "numberOfRows": table.num_rows,
                        "version": 1,
                        "cellColors": [
                            [r, c_type, g_idx, s_idx, color]
                            for (r, c_type, g_idx, s_idx), color in table.cell_colors.items()
                        ],
                        "rowTitles": table.row_titles,
                        "variables": [v.to_dict() for v in table.variables] if table.table_type == TableType.MULTIPLE_VARIABLES else None,
                        "multiVarData": table.multi_var_data if table.table_type == TableType.MULTIPLE_VARIABLES else None,
                        "nestedSubgroupTitles": {str(k): v for k, v in table.nested_subgroup_titles.items()},
                    }
                    zf.writestr(
                        f"data/tables/{t_uid}/content.json",
                        json.dumps(content_meta, indent=2),
                    )

                    # 5. data/tables/<t_uid>/data.csv
                    csv_buf = io.StringIO()
                    csv_writer = csv.writer(csv_buf)
                    if table.table_type == TableType.MULTIPLE_VARIABLES:
                        for row in table.multi_var_data:
                            csv_writer.writerow([
                                ("" if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v))
                                for v in row
                            ])
                    else:
                        for r in range(table.num_rows):
                            row_vals = []
                            if table.x_data is not None:
                                xv = table.x_data[r]
                                row_vals.append("" if np.isnan(xv) else f"{xv:.8g}")
                            for g_idx in range(len(table.y_group_titles)):
                                for s_idx in range(table.num_replicates):
                                    yv = table.y_data[g_idx, r, s_idx]
                                    row_vals.append("" if np.isnan(yv) else f"{yv:.8g}")
                            csv_writer.writerow(row_vals)
                    zf.writestr(f"data/tables/{t_uid}/data.csv", csv_buf.getvalue())

                elif isinstance(sheet, InfoSheet):
                    info_meta = {
                        "@class": "InfoSheet",
                        "uid": sheet.sheet_id,
                        "title": sheet.title,
                        "notesText": sheet.notes,
                        "notesHtml": getattr(sheet, "notes_html", ""),
                        "constants": sheet.constants,
                    }
                    zf.writestr(f"info/{sheet.sheet_id}/sheet.json", json.dumps(info_meta, indent=2))

                elif isinstance(sheet, GraphSheet):
                    graph_meta = {
                        "@class": "GraphSheet",
                        "uid": sheet.sheet_id,
                        "title": sheet.title,
                        "graph_type": sheet.graph_type.value,
                        "source_sheet_id": sheet.source_sheet_id,
                        "result_sheet_id": getattr(sheet, "result_sheet_id", None),
                        "inputDataSets": [f"Y_{sheet.source_sheet_id}_0"] if sheet.source_sheet_id else [],
                        "plot_config": getattr(sheet, "plot_config", {}),
                        "is_pinned": getattr(sheet, "is_pinned", False),
                        "is_frozen": getattr(sheet, "is_frozen", False),
                    }
                    if getattr(sheet, "snapshot_table", None) is not None:
                        graph_meta["snapshot_table"] = sheet.snapshot_table.to_dict()
                    if getattr(sheet, "snapshot_result_data", None) is not None:
                        graph_meta["snapshot_result_data"] = sheet.snapshot_result_data
                    zf.writestr(f"graphs/{sheet.sheet_id}/sheet.json", json.dumps(graph_meta, indent=2))

                elif isinstance(sheet, ResultSheet):
                    res_meta = {
                        "@class": "ResultSheet",
                        "uid": sheet.sheet_id,
                        "title": sheet.title,
                        "analysis_type": sheet.analysis_type,
                        "source_sheet_id": sheet.source_sheet_id,
                        "summary_tables": sheet.summary_tables,
                        "raw_result_data": sheet.raw_result_data,
                    }
                    zf.writestr(f"results/{sheet.sheet_id}/sheet.json", json.dumps(res_meta, indent=2))

    @classmethod
    def load(cls, file_path: str) -> PrismDocument:
        """Parses a project archive or data file (.bigo, .prism, .pzfx, .xml, .csv) into a PrismDocument."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".csv":
            return cls.load_csv(file_path)
        if ext in (".pzfx", ".xml", ".pzm"):
            from src.io.xml_io import PrismXMLIO
            return PrismXMLIO.import_xml(file_path)
        if ext == ".pzf":
            return cls.load_pzf(file_path)

        if zipfile.is_zipfile(file_path):
            with zipfile.ZipFile(file_path, mode="r") as zf:
                if "document.json" not in zf.namelist():
                    raise ValueError(f"Invalid project archive ({ext}): missing document.json")

                manifest = json.loads(zf.read("document.json").decode("utf-8"))

                if "sheets_manifest" in manifest:
                    return cls._load_cleanroom(zf, manifest)
                else:
                    return cls._load_prism10(zf, manifest)

        # Fallback: check binary header for pzf
        try:
            with open(file_path, "rb") as f:
                header = f.read(8)
            if header.startswith(b"PCFF"):
                return cls.load_pzf(file_path)
        except Exception:
            pass

        # Fallback: try XML or CSV
        try:
            from src.io.xml_io import PrismXMLIO
            return PrismXMLIO.import_xml(file_path)
        except Exception:
            return cls.load_csv(file_path)

    @classmethod
    def load_pzf(cls, file_path: str) -> PrismDocument:
        """Parses a legacy GraphPad Prism binary project (.pzf) into a PrismDocument."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "rb") as f:
            data = f.read()

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        doc = PrismDocument(title=base_name or "Prism PZF Project")

        pattern = re.compile(rb'\t\x00(..)\x00\x00([A-Za-z0-9_ \-]{1,50})\x00\x10\x00(.)\x00\x00\x00\x00\x00(....)')
        cols = []
        seen = set()
        for m in pattern.finditer(data):
            title = m.group(2).decode("latin1").strip()
            num_rows = struct.unpack("<I", m.group(4))[0]
            key = (title, num_rows)
            if key in seen or num_rows == 0 or num_rows > 10000:
                continue
            seen.add(key)
            p = m.end()
            vals = []
            for _ in range(num_rows):
                if p + 12 > len(data):
                    break
                val = struct.unpack("<f", data[p+8:p+12])[0]
                vals.append(float(val))
                p += 12
            if len(vals) == num_rows:
                cols.append((title, vals))

        if not cols:
            raise ValueError(f"Could not extract table data from {file_path}")

        first_title_lower = cols[0][0].lower()
        is_xy = any(k in first_title_lower for k in ("x", "second", "time", "day", "dose", "conc"))

        if is_xy:
            x_title, x_vals = cols[0]
            num_rows = len(x_vals)

            candidate_y = []
            for title, vals in cols[1:]:
                t_lower = title.lower()
                if any(k in t_lower for k in ("x", "second", "time", "day", "dose", "conc")):
                    break
                if len(vals) != num_rows and (len(vals) % num_rows != 0 or len(vals) == 0):
                    break
                candidate_y.append((title, vals))

            y_cols = candidate_y
            replicates = 1
            if y_cols and len(y_cols[0][1]) > num_rows and len(y_cols[0][1]) % num_rows == 0:
                replicates = len(y_cols[0][1]) // num_rows

            table = DataTable(
                table_id="table_1",
                title=base_name or "Data 1",
                table_type=TableType.XY,
                num_rows=num_rows,
                num_replicates=replicates,
                y_group_titles=[c[0] for c in y_cols] if y_cols else ["Group A"],
            )
            table.x_columns = [{"title": x_title, "data": np.array(x_vals, dtype=np.float64)}]
            table.column_order = [("X", 0)] + [("Y", i) for i in range(len(y_cols))]

            if y_cols:
                table.y_data = np.full((len(y_cols), num_rows, replicates), np.nan, dtype=np.float64)
                for g_idx, (_, vals) in enumerate(y_cols):
                    for r in range(num_rows):
                        for s in range(replicates):
                            val_idx = r * replicates + s
                            if val_idx < len(vals):
                                table.y_data[g_idx, r, s] = vals[val_idx]

            graph_type = GraphType.XY_SPIKES_CURVE
        else:
            first_len = len(cols[0][1])
            candidate_y = []
            for title, vals in cols:
                if len(vals) != first_len:
                    break
                candidate_y.append((title, vals))
            y_cols = candidate_y
            num_rows = first_len
            table = DataTable(
                table_id="table_1",
                title=base_name or "Data 1",
                table_type=TableType.COLUMN,
                num_rows=num_rows,
                num_replicates=1,
                y_group_titles=[c[0] for c in y_cols],
            )
            table.column_order = [("Y", i) for i in range(len(y_cols))]
            table.y_data = np.full((len(y_cols), num_rows, 1), np.nan, dtype=np.float64)
            for g_idx, (_, vals) in enumerate(y_cols):
                for r, v in enumerate(vals):
                    table.y_data[g_idx, r, 0] = v

            graph_type = GraphType.BARS_AND_POINTS

        data_sheet = DataSheet(sheet_id="data_1", title=base_name or "Data 1", table=table)
        doc.add_sheet(data_sheet)

        graph_sheet = GraphSheet(
            sheet_id="graph_1",
            title=f"Graph of {base_name or 'Data 1'}",
            graph_type=graph_type,
            source_sheet_id="data_1",
        )
        doc.add_sheet(graph_sheet)
        doc.active_sheet_id = "data_1"
        return doc

    @classmethod
    def load_csv(cls, file_path: str) -> PrismDocument:
        """Parses a CSV file into a PrismDocument with a populated DataSheet and GraphSheet."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        doc_title = base_name if base_name else "CSV Project"
        doc = PrismDocument(title=doc_title)

        delimiter = ","
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                sample = f.read(4096)
            if not sample.strip():
                df = pd.DataFrame()
            else:
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
                    delimiter = dialect.delimiter
                except Exception:
                    delimiter = ","
                df = pd.read_csv(file_path, sep=delimiter)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame()
        except Exception as e:
            try:
                df = pd.read_csv(file_path)
            except Exception:
                raise ValueError(f"Could not parse CSV file: {e}") from e

        if len(df.columns) == 0:
            sheet = doc.create_data_sheet(
                title=base_name or "Data 1",
                table_type=TableType.XY,
                num_rows=10,
                num_replicates=1,
                y_group_titles=["Group A"],
            )
            doc.create_graph_sheet(title=f"{sheet.title} Graph", source_sheet_id=sheet.sheet_id)
            return doc

        cols = [str(c).strip() for c in df.columns]
        df.columns = cols

        rep_matches = [re.match(r"^(.*?):Y(\d+)$", col) for col in cols]
        has_reps = any(m is not None for m in rep_matches)

        if has_reps:
            first_col = cols[0]
            first_is_x = re.match(r"^(.*?):Y(\d+)$", first_col) is None

            x_title = first_col if first_is_x else "X"
            x_vals = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy() if first_is_x else None
            y_cols = cols[1:] if first_is_x else cols

            groups: Dict[str, List[Tuple[int, str]]] = {}
            for c in y_cols:
                m = re.match(r"^(.*?):Y(\d+)$", c)
                if m:
                    g_name, rep_idx = m.group(1), int(m.group(2))
                    groups.setdefault(g_name, []).append((rep_idx, c))
                else:
                    groups.setdefault(c, []).append((1, c))

            y_group_titles = list(groups.keys())
            max_reps = max(len(v) for v in groups.values()) if groups else 1
            num_rows = max(len(df), 10)

            table_type = TableType.XY if first_is_x else TableType.COLUMN
            sheet = doc.create_data_sheet(
                title=base_name or "Data 1",
                table_type=table_type,
                num_rows=num_rows,
                num_replicates=max_reps,
                y_group_titles=y_group_titles,
            )
            table = sheet.table
            if first_is_x and x_vals is not None:
                table.x_title = x_title
                table.x_data[:len(df)] = x_vals

            for g_idx, g_name in enumerate(y_group_titles):
                for s_idx, (_, c_name) in enumerate(groups[g_name]):
                    if s_idx < max_reps:
                        table.y_data[g_idx, :len(df), s_idx] = pd.to_numeric(
                            df[c_name], errors="coerce"
                        ).to_numpy()

            doc.create_graph_sheet(title=f"{sheet.title} Graph", source_sheet_id=sheet.sheet_id)
            return doc

        first_col = cols[0]
        first_is_x = (
            first_col.upper() == "X"
            or any(term in first_col.lower() for term in ("time", "dose", "conc", "hour", "day", "min", "sec"))
            or (len(cols) >= 2 and pd.to_numeric(df.iloc[:, 0], errors="coerce").notna().sum() > 0)
        )

        num_rows = max(len(df), 10)
        if first_is_x and len(cols) >= 2:
            table_type = TableType.XY
            x_title = first_col
            x_vals = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()
            y_group_titles = cols[1:]
            num_replicates = 1

            sheet = doc.create_data_sheet(
                title=base_name or "Data 1",
                table_type=table_type,
                num_rows=num_rows,
                num_replicates=num_replicates,
                y_group_titles=y_group_titles,
            )
            table = sheet.table
            table.x_title = x_title
            table.x_data[:len(df)] = x_vals
            for g_idx, col_name in enumerate(y_group_titles):
                table.y_data[g_idx, :len(df), 0] = pd.to_numeric(
                    df[col_name], errors="coerce"
                ).to_numpy()
        else:
            table_type = TableType.COLUMN
            y_group_titles = cols
            sheet = doc.create_data_sheet(
                title=base_name or "Data 1",
                table_type=table_type,
                num_rows=num_rows,
                num_replicates=1,
                y_group_titles=y_group_titles,
            )
            table = sheet.table
            for g_idx, col_name in enumerate(y_group_titles):
                table.y_data[g_idx, :len(df), 0] = pd.to_numeric(
                    df[col_name], errors="coerce"
                ).to_numpy()

        doc.create_graph_sheet(title=f"{sheet.title} Graph", source_sheet_id=sheet.sheet_id)
        return doc

    @classmethod
    def _load_cleanroom(cls, zf: zipfile.ZipFile, manifest: Dict[str, Any]) -> PrismDocument:
        """Loads GemGraph native cleanroom .prism archive."""
        doc = PrismDocument(title=manifest.get("title", "Untitled Project"))
        doc.doc_id = manifest.get("doc_id", doc.doc_id)
        doc.active_sheet_id = manifest.get("active_sheet_id")

        for item in manifest.get("sheets_manifest", []):
            sheet_file = item["sheet_file"]
            if sheet_file in zf.namelist():
                s_data = json.loads(zf.read(sheet_file).decode("utf-8"))
                stype = s_data.get("sheet_type")
                if stype == SheetType.DATA.value:
                    table_file = f"data/tables/{item['sheet_id']}.json"
                    if "table" not in s_data and table_file in zf.namelist():
                        t_data = json.loads(zf.read(table_file).decode("utf-8"))
                        s_data["table"] = t_data
                    sheet = DataSheet.from_dict(s_data)
                elif stype == SheetType.INFO.value:
                    sheet = InfoSheet.from_dict(s_data)
                elif stype == SheetType.RESULTS.value:
                    sheet = ResultSheet.from_dict(s_data)
                elif stype == SheetType.GRAPH.value:
                    sheet = GraphSheet.from_dict(s_data)
                else:
                    continue
                doc.sheets[sheet.sheet_id] = sheet

        doc.sheet_order = [
            sid for sid in manifest.get("sheet_order", []) if sid in doc.sheets
        ]
        if not doc.sheet_order and doc.sheets:
            doc.sheet_order = list(doc.sheets.keys())
        if doc.active_sheet_id not in doc.sheets and doc.sheet_order:
            doc.active_sheet_id = doc.sheet_order[0]

        return doc

    @classmethod
    def _parse_row_range(cls, rows_spec: Any) -> List[int]:
        """Parses Prism row range strings like '0~2', '3', '1-5', integers, or lists."""
        if rows_spec is None:
            return []
        if isinstance(rows_spec, (list, tuple, set)):
            rows: List[int] = []
            for item in rows_spec:
                rows.extend(cls._parse_row_range(item))
            return rows
        if isinstance(rows_spec, int):
            return [rows_spec]
        rows = []
        for part in str(rows_spec).split(","):
            part = part.strip()
            if not part:
                continue
            if "~" in part:
                sub = part.split("~")
                try:
                    start, end = int(sub[0].strip()), int(sub[1].strip())
                    rows.extend(range(start, end + 1))
                except ValueError:
                    pass
            elif "-" in part and not part.startswith("-"):
                sub = part.split("-")
                try:
                    start, end = int(sub[0].strip()), int(sub[1].strip())
                    rows.extend(range(start, end + 1))
                except ValueError:
                    pass
            elif part.isdigit():
                rows.append(int(part))
        return rows

    @staticmethod
    def _is_excluded_attribute(ca: Dict[str, Any]) -> bool:
        """Determines if a cellAttribute object flags cell exclusion."""
        attrs = ca.get("attributes", [])
        if isinstance(attrs, list):
            for attr in attrs:
                if str(attr).upper() in ("EXCLUDED", "EXCLUDE", "1", "TRUE"):
                    return True
        for key in ("excluded", "Excluded", "isExcluded", "is_excluded"):
            if key in ca:
                val = ca[key]
                if val is True or str(val).lower() in ("1", "true", "yes"):
                    return True
        return False

    @staticmethod
    def _is_md5_hash(b: bytes) -> bool:
        """Checks if a 32-byte buffer is an ASCII hex MD5 checksum."""
        if len(b) == 32:
            try:
                s = b.decode("ascii")
                return all(c in "0123456789abcdefABCDEF" for c in s)
            except UnicodeDecodeError:
                return False
        return False

    @classmethod
    def _parse_binary_table(
        cls, dt_bytes: bytes, num_cols: int
    ) -> List[List[float]]:
        """Decodes binary float/double array buffer into rows of floats."""
        if not dt_bytes or num_cols <= 0:
            return []
        if cls._is_md5_hash(dt_bytes):
            return []

        # Try double precision (<d, 8 bytes per value)
        if len(dt_bytes) % 8 == 0:
            try:
                count = len(dt_bytes) // 8
                floats = list(struct.unpack(f"<{count}d", dt_bytes))
                return [floats[i : i + num_cols] for i in range(0, len(floats), num_cols)]
            except Exception:
                pass

        # Try single precision (<f, 4 bytes per value)
        if len(dt_bytes) % 4 == 0:
            try:
                count = len(dt_bytes) // 4
                floats = list(struct.unpack(f"<{count}f", dt_bytes))
                return [floats[i : i + num_cols] for i in range(0, len(floats), num_cols)]
            except Exception:
                pass

        return []


    @classmethod
    def _load_prism10(cls, zf: zipfile.ZipFile, manifest: Dict[str, Any]) -> PrismDocument:
        """Parses official GraphPad Prism 10 PK-ZIP archives."""
        title = manifest.get("title", "Prism Project")
        doc = PrismDocument(title=title)

        sheet_attrs = manifest.get("sheetAttributesMap", {})
        sheets_dict = manifest.get("sheets", {})

        data_sheet_uids = sheets_dict.get("data", []) if isinstance(sheets_dict, dict) else []
        info_sheet_uids = sheets_dict.get("info", []) if isinstance(sheets_dict, dict) else []
        graph_sheet_uids = sheets_dict.get("graphs", []) if isinstance(sheets_dict, dict) else []

        dataset_to_sheet: Dict[str, str] = {}

        # 1. Load Data Sheets
        for s_uid in data_sheet_uids:
            sheet_json_path = f"data/sheets/{s_uid}/sheet.json"
            if sheet_json_path not in zf.namelist():
                sheet_json_path = f"data/sheets/{s_uid}.json"
            if sheet_json_path not in zf.namelist():
                continue

            s_data = json.loads(zf.read(sheet_json_path).decode("utf-8"))
            s_title = (
                s_data.get("title")
                or sheet_attrs.get(s_uid, {}).get("title")
                or f"Data {len(doc.sheets) + 1}"
            )

            table_meta = s_data.get("table", {})
            t_uid = table_meta.get("uid", s_uid)

            raw_fmt = str(table_meta.get("format", "xy")).lower()
            if "column" in raw_fmt or "one_way" in raw_fmt:
                t_type = TableType.COLUMN
            elif "grouped" in raw_fmt or "two_way" in raw_fmt:
                t_type = TableType.GROUPED
            elif "contingency" in raw_fmt:
                t_type = TableType.CONTINGENCY
            elif "survival" in raw_fmt:
                t_type = TableType.SURVIVAL
            elif "parts_of_whole" in raw_fmt or "fraction" in raw_fmt:
                t_type = TableType.PARTS_OF_WHOLE
            elif "multiple_variables" in raw_fmt:
                t_type = TableType.MULTIPLE_VARIABLES
            elif "nested" in raw_fmt:
                t_type = TableType.NESTED
            else:
                t_type = TableType.XY

            # X Dataset inspection
            x_dataset_id = table_meta.get("xDataSet")
            x_title = "X"
            x_exclusions: Set[int] = set()

            if x_dataset_id:
                x_ds_path = f"data/sets/{x_dataset_id}.json"
                if x_ds_path in zf.namelist():
                    x_ds_data = json.loads(zf.read(x_ds_path).decode("utf-8"))
                    xt = x_ds_data.get("title")
                    if isinstance(xt, dict):
                        x_title = xt.get("string", "X")
                    elif xt:
                        x_title = str(xt)
                    for r_obj in x_ds_data.get("replicates", []):
                        for ca in r_obj.get("cellAttributes", []):
                            if cls._is_excluded_attribute(ca):
                                for r_idx in cls._parse_row_range(ca.get("rows", "")):
                                    x_exclusions.add(r_idx)

            # Y Datasets inspection
            y_dataset_ids = table_meta.get("dataSets", [])
            y_group_titles: List[str] = []
            group_replicate_counts: List[int] = []
            y_exclusions: Set[Tuple[int, int, int]] = set()  # (row, group_idx, subcol_idx)

            for g_idx, ds_id in enumerate(y_dataset_ids):
                dataset_to_sheet[ds_id] = s_uid
                ds_path = f"data/sets/{ds_id}.json"
                g_title = f"Group {g_idx + 1}"
                num_reps = 1

                if ds_path in zf.namelist():
                    ds_data = json.loads(zf.read(ds_path).decode("utf-8"))
                    yt = ds_data.get("title")
                    if isinstance(yt, dict):
                        g_title = yt.get("string", g_title)
                    elif yt:
                        g_title = str(yt)

                    reps = ds_data.get("replicates", [])
                    if reps:
                        num_reps = len(reps)
                        for s_idx, rep in enumerate(reps):
                            for ca in rep.get("cellAttributes", []):
                                if cls._is_excluded_attribute(ca):
                                    for r_idx in cls._parse_row_range(ca.get("rows", "")):
                                        y_exclusions.add((r_idx, g_idx, s_idx))

                y_group_titles.append(g_title)
                group_replicate_counts.append(num_reps)

            max_reps = max(group_replicate_counts) if group_replicate_counts else 1
            if not y_group_titles:
                y_group_titles = ["Group A"]
                group_replicate_counts = [1]

            # Read CSV table data
            csv_path = f"data/tables/{t_uid}/data.csv"
            if csv_path not in zf.namelist():
                csv_path = f"data/tables/{t_uid}.csv"

            rows_data: List[List[float]] = []
            if csv_path in zf.namelist():
                csv_raw = zf.read(csv_path).decode("utf-8", errors="replace")
                reader = csv.reader(io.StringIO(csv_raw))
                for row_idx, row in enumerate(reader):
                    if not row or not any(cell.strip() for cell in row):
                        continue
                    # Check if first row is header row
                    if row_idx == 0:
                        has_str = False
                        for c in row:
                            c_str = c.strip()
                            if c_str:
                                try:
                                    float(c_str)
                                except ValueError:
                                    has_str = True
                                    break
                        if has_str:
                            continue

                    parsed_row: List[float] = []
                    for c in row:
                        c_str = c.strip()
                        if c_str == "" or c_str.lower() == "nan":
                            parsed_row.append(np.nan)
                        else:
                            try:
                                parsed_row.append(float(c_str))
                            except ValueError:
                                parsed_row.append(np.nan)
                    rows_data.append(parsed_row)

            # Fallback for binary data.dt buffer if CSV is absent
            dt_path = f"data/tables/{t_uid}/data.dt"
            if not rows_data and dt_path in zf.namelist():
                dt_bytes = zf.read(dt_path)
                content_json_path = f"data/tables/{t_uid}/content.json"
                n_cols = len(y_group_titles) + (1 if t_type in (TableType.XY, TableType.SURVIVAL) else 0)
                if content_json_path in zf.namelist():
                    try:
                        cj = json.loads(zf.read(content_json_path).decode("utf-8"))
                        n_cols = cj.get("numberOfColumns", n_cols)
                    except Exception:
                        pass
                rows_data = cls._parse_binary_table(dt_bytes, n_cols)


            num_rows = len(rows_data) if rows_data else 20
            if num_rows == 0:
                num_rows = 20

            table = DataTable(
                table_id=t_uid,
                title=s_title,
                table_type=t_type,
                num_rows=num_rows,
                num_replicates=max_reps,
                y_group_titles=y_group_titles,
            )
            table.x_title = x_title

            # Populate table cells from rows_data
            for r, row in enumerate(rows_data):
                col_idx = 0
                if table.x_data is not None:
                    if col_idx < len(row):
                        table.set_cell_value(r, "X", 0, 0, row[col_idx])
                        col_idx += 1
                for g_idx in range(len(y_group_titles)):
                    g_reps = group_replicate_counts[g_idx] if g_idx < len(group_replicate_counts) else max_reps
                    for s_idx in range(g_reps):
                        if col_idx < len(row):
                            table.set_cell_value(r, "Y", g_idx, s_idx, row[col_idx])
                            col_idx += 1

            # Apply cell exclusions
            for r_idx in x_exclusions:
                table.set_excluded(r_idx, "X", 0, 0, True)
            for r_idx, g_idx, s_idx in y_exclusions:
                table.set_excluded(r_idx, "Y", g_idx, s_idx, True)

            # Restore content metadata (cell colors, row titles, variables, nested titles)
            content_json_path = f"data/tables/{t_uid}/content.json"
            if content_json_path in zf.namelist():
                try:
                    cj = json.loads(zf.read(content_json_path).decode("utf-8"))
                    for item in cj.get("cellColors", []):
                        if len(item) == 5:
                            table.cell_colors[(item[0], item[1], item[2], item[3])] = item[4]
                    if cj.get("rowTitles"):
                        table.row_titles = cj["rowTitles"]
                    if cj.get("variables"):
                        from src.core.table import VariableMetadata
                        table.variables = [VariableMetadata.from_dict(v) for v in cj["variables"]]
                    if cj.get("multiVarData"):
                        table.multi_var_data = cj["multiVarData"]
                    if cj.get("nestedSubgroupTitles"):
                        table.nested_subgroup_titles = {int(k): v for k, v in cj["nestedSubgroupTitles"].items()}
                except Exception:
                    pass

            data_sheet = DataSheet(sheet_id=s_uid, title=s_title, table=table)
            doc.sheets[s_uid] = data_sheet
            doc.sheet_order.append(s_uid)

        # 2. Load Info Sheets
        for i_uid in info_sheet_uids:
            sheet_json_path = f"info/{i_uid}/sheet.json"
            if sheet_json_path not in zf.namelist():
                sheet_json_path = f"data/sheets/{i_uid}.json"
            if sheet_json_path not in zf.namelist():
                continue

            i_data = json.loads(zf.read(sheet_json_path).decode("utf-8"))
            i_title = (
                i_data.get("title")
                or sheet_attrs.get(i_uid, {}).get("title")
                or f"Info {len(doc.sheets) + 1}"
            )
            notes = i_data.get("notesText", "") or i_data.get("notes", "")
            notes_html = i_data.get("notesHtml", "") or i_data.get("notes_html", "")

            raw_consts = i_data.get("constants", {})
            consts_dict: Dict[str, str] = {}
            if isinstance(raw_consts, list):
                for c in raw_consts:
                    if isinstance(c, dict) and c.get("name"):
                        consts_dict[c["name"]] = c.get("value", "")
            elif isinstance(raw_consts, dict):
                consts_dict = raw_consts

            info_sheet = InfoSheet(sheet_id=i_uid, title=i_title, notes=notes, notes_html=notes_html, constants=consts_dict)
            doc.sheets[i_uid] = info_sheet
            doc.sheet_order.append(i_uid)

        # 3. Load Graph Sheets
        for g_uid in graph_sheet_uids:
            sheet_json_path = f"graphs/{g_uid}/sheet.json"
            if sheet_json_path not in zf.namelist():
                sheet_json_path = f"data/sheets/{g_uid}.json"
            if sheet_json_path not in zf.namelist():
                continue

            g_data = json.loads(zf.read(sheet_json_path).decode("utf-8"))
            g_title = (
                g_data.get("title")
                or sheet_attrs.get(g_uid, {}).get("title")
                or f"Graph {len(doc.sheets) + 1}"
            )

            input_datasets = g_data.get("inputDataSets", [])
            source_sheet_id = ""
            for ds_id in input_datasets:
                if ds_id in dataset_to_sheet:
                    source_sheet_id = dataset_to_sheet[ds_id]
                    break
            if not source_sheet_id and data_sheet_uids:
                source_sheet_id = data_sheet_uids[0]

            plot_config = g_data.get("plot_config", {})
            raw_gt = g_data.get("graph_type")
            try:
                g_type = GraphType(raw_gt) if raw_gt else GraphType.XY_SCATTER_CURVE
            except Exception:
                g_type = GraphType.XY_SCATTER_CURVE
            snapshot_table = None
            if "snapshot_table" in g_data and g_data["snapshot_table"]:
                snapshot_table = DataTable.from_dict(g_data["snapshot_table"])

            graph_sheet = GraphSheet(
                sheet_id=g_uid,
                title=g_title,
                graph_type=g_type,
                source_sheet_id=source_sheet_id,
                result_sheet_id=g_data.get("result_sheet_id"),
                plot_config=plot_config,
                is_pinned=g_data.get("is_pinned", False),
                snapshot_table=snapshot_table,
                snapshot_result_data=g_data.get("snapshot_result_data"),
            )
            graph_sheet.is_frozen = g_data.get("is_frozen", False)
            doc.sheets[g_uid] = graph_sheet
            doc.sheet_order.append(g_uid)

        # 4. Load Result Sheets (if present)
        result_sheet_uids = sheets_dict.get("results", []) if isinstance(sheets_dict, dict) else []
        for r_uid in result_sheet_uids:
            sheet_json_path = f"results/{r_uid}/sheet.json"
            if sheet_json_path not in zf.namelist():
                sheet_json_path = f"data/sheets/{r_uid}.json"
            if sheet_json_path not in zf.namelist():
                continue

            r_data = json.loads(zf.read(sheet_json_path).decode("utf-8"))
            r_title = (
                r_data.get("title")
                or sheet_attrs.get(r_uid, {}).get("title")
                or f"Results {len(doc.sheets) + 1}"
            )
            res_sheet = ResultSheet(
                sheet_id=r_uid,
                title=r_title,
                analysis_type=r_data.get("analysis_type", "Analysis"),
                source_sheet_id=r_data.get("source_sheet_id", ""),
                summary_tables=r_data.get("summary_tables", []),
                raw_result_data=r_data.get("raw_result_data", {}),
            )
            doc.sheets[r_uid] = res_sheet
            doc.sheet_order.append(r_uid)

        # Respect explicit sheet order if provided in manifest
        explicit_order = manifest.get("sheetOrder") or manifest.get("sheetsOrder")
        if isinstance(explicit_order, list) and explicit_order:
            ordered = [sid for sid in explicit_order if sid in doc.sheets]
            for sid in doc.sheet_order:
                if sid not in ordered:
                    ordered.append(sid)
            doc.sheet_order = ordered

        # Determine active sheet
        current_data = manifest.get("uiSettings", {}).get("currentSheets", {}).get("data")
        if current_data and current_data in doc.sheets:
            doc.active_sheet_id = current_data
        elif doc.sheet_order:
            doc.active_sheet_id = doc.sheet_order[0]

        return doc

