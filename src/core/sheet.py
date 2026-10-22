"""
Sheet abstraction hierarchy for GemGraph.
Derived from BridgeCLR and Core schemas (REA-CLR-BRIDGE-091 & REA-CORE-SCHEMA-208).
"""

from __future__ import annotations
import enum
from typing import Dict, Any, Optional, List
from src.core.table import DataTable, TableType


class SheetType(enum.Enum):
    DATA = "data"
    INFO = "info"
    RESULTS = "results"
    GRAPH = "graph"
    LAYOUT = "layout"


class BaseSheet:
    """Base class for all sheets in a GemGraph project."""

    def __init__(self, sheet_id: str, title: str, sheet_type: SheetType) -> None:
        self.sheet_id = sheet_id
        self.title = title
        self.sheet_type = sheet_type
        self.is_frozen: bool = False
        self.is_orphaned: bool = False
        self.parent_sheet_ids: List[str] = []
        self.child_sheet_ids: List[str] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sheet_id": self.sheet_id,
            "title": self.title,
            "sheet_type": self.sheet_type.value,
            "is_frozen": self.is_frozen,
            "is_orphaned": self.is_orphaned,
            "parent_sheet_ids": self.parent_sheet_ids,
            "child_sheet_ids": self.child_sheet_ids,
        }


class DataSheet(BaseSheet):
    """Sheet holding an editable scientific DataTable."""

    def __init__(
        self,
        sheet_id: str,
        title: str = "Data 1",
        table: Optional[DataTable] = None,
        table_type: TableType = TableType.XY,
        calc_spec: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(sheet_id, title, SheetType.DATA)
        self.table = table or DataTable(table_id=sheet_id, title=title, table_type=table_type)
        self.calc_spec: Dict[str, Any] = calc_spec or {}

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["table"] = self.table.to_dict()
        d["calc_spec"] = self.calc_spec
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DataSheet:
        table_data = data.get("table", {})
        if not table_data:
            table_data = {
                "table_id": data.get("sheet_id", "data_1"),
                "title": data.get("title", "Data 1"),
            }
        table = DataTable.from_dict(table_data)
        sheet = cls(
            sheet_id=data.get("sheet_id", table.table_id),
            title=data.get("title", table.title),
            table=table,
            calc_spec=data.get("calc_spec", {}),
        )
        sheet.is_frozen = data.get("is_frozen", False)
        sheet.is_orphaned = data.get("is_orphaned", False)
        sheet.parent_sheet_ids = data.get("parent_sheet_ids", [])
        sheet.child_sheet_ids = data.get("child_sheet_ids", [])
        return sheet


class InfoSheet(BaseSheet):
    """Sheet holding project, experiment, and protocol metadata."""

    def __init__(
        self,
        sheet_id: str,
        title: str = "Info 1",
        notes: str = "",
        constants: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(sheet_id, title, SheetType.INFO)
        self.notes = notes
        self.constants: Dict[str, str] = constants or {
            "Experimenter": "",
            "Project": "",
            "Notebook ID": "",
            "Date": "",
        }

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["notes"] = self.notes
        d["constants"] = self.constants
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> InfoSheet:
        sheet = cls(
            sheet_id=data["sheet_id"],
            title=data.get("title", "Info"),
            notes=data.get("notes", ""),
            constants=data.get("constants", {}),
        )
        sheet.is_frozen = data.get("is_frozen", False)
        sheet.is_orphaned = data.get("is_orphaned", False)
        sheet.parent_sheet_ids = data.get("parent_sheet_ids", [])
        sheet.child_sheet_ids = data.get("child_sheet_ids", [])
        return sheet


class ResultSheet(BaseSheet):
    """Sheet holding analytical computation results, multi-tab tables, and green/red grid semantics."""

    def __init__(
        self,
        sheet_id: str,
        title: str = "Nonlin reg",
        analysis_type: str = "Nonlinear Regression",
        source_sheet_id: str = "",
        summary_tables: Optional[List[Dict[str, Any]]] = None,
        raw_result_data: Optional[Dict[str, Any]] = None,
        grid_type: str = "red",
        tabs: Optional[List[Dict[str, Any]]] = None,
        calc_spec: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(sheet_id, title, SheetType.RESULTS)
        self.analysis_type = analysis_type
        self.source_sheet_id = source_sheet_id
        if source_sheet_id and source_sheet_id not in self.parent_sheet_ids:
            self.parent_sheet_ids.append(source_sheet_id)
        self.summary_tables: List[Dict[str, Any]] = summary_tables or []
        self.raw_result_data: Dict[str, Any] = raw_result_data or {}
        self.grid_type = grid_type  # "green" (graphable) or "red" (summary)
        self.tabs: List[Dict[str, Any]] = tabs or []
        self.calc_spec: Dict[str, Any] = calc_spec or {}

        # Auto-populate tabs from summary_tables if tabs not explicitly provided
        if not self.tabs and self.summary_tables:
            for tbl in self.summary_tables:
                self.tabs.append({
                    "name": tbl.get("title", "Table of results"),
                    "grid_type": self.grid_type,
                    "headers": tbl.get("headers", []),
                    "rows": tbl.get("rows", []),
                    "visible": True,
                })

    def add_tab(
        self,
        name: str,
        headers: List[str],
        rows: List[List[Any]],
        grid_type: str = "red",
        visible: bool = True,
    ) -> None:
        """Adds a sub-tab to the multi-tab results view."""
        self.tabs.append({
            "name": name,
            "grid_type": grid_type,
            "headers": headers,
            "rows": rows,
            "visible": visible,
        })

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["analysis_type"] = self.analysis_type
        d["source_sheet_id"] = self.source_sheet_id
        d["summary_tables"] = self.summary_tables
        d["raw_result_data"] = self.raw_result_data
        d["grid_type"] = self.grid_type
        d["tabs"] = self.tabs
        d["calc_spec"] = self.calc_spec
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ResultSheet:
        sheet = cls(
            sheet_id=data["sheet_id"],
            title=data.get("title", "Results"),
            analysis_type=data.get("analysis_type", "General Analysis"),
            source_sheet_id=data.get("source_sheet_id", ""),
            summary_tables=data.get("summary_tables", []),
            raw_result_data=data.get("raw_result_data", {}),
            grid_type=data.get("grid_type", "red"),
            tabs=data.get("tabs", []),
            calc_spec=data.get("calc_spec", {}),
        )
        sheet.is_frozen = data.get("is_frozen", False)
        sheet.is_orphaned = data.get("is_orphaned", False)
        sheet.parent_sheet_ids = data.get("parent_sheet_ids", [])
        sheet.child_sheet_ids = data.get("child_sheet_ids", [])
        return sheet


class GraphType(enum.Enum):
    XY_SCATTER_CURVE = "XY Scatter & Fitted Curve"
    COLUMN_BAR_ERROR = "Column Bars (Mean ± SEM/SD)"
    BARS_AND_POINTS = "Bars + Individual Points"
    GROUPED_BARS_POINTS = "Grouped Bars + Replicate Points"
    XY_SPIKES_CURVE = "Columns/Spikes + Curve Fit"
    BOX_WHISKER = "Box & Whisker"
    SURVIVAL_CURVE = "Kaplan-Meier Survival"



class GraphSheet(BaseSheet):
    """Sheet holding graphical presentation parameters."""

    def __init__(
        self,
        sheet_id: str,
        title: str = "Graph 1",
        graph_type: GraphType = GraphType.XY_SCATTER_CURVE,
        source_sheet_id: str = "",
        result_sheet_id: Optional[str] = None,
        x_label: str = "X",
        y_label: str = "Y",
        show_error_bars: bool = True,
        error_style: str = "SEM",  # "SD" or "SEM"
    ) -> None:
        super().__init__(sheet_id, title, SheetType.GRAPH)
        self.graph_type = graph_type
        self.source_sheet_id = source_sheet_id
        self.result_sheet_id = result_sheet_id
        self.x_label = x_label
        self.y_label = y_label
        self.show_error_bars = show_error_bars
        self.error_style = error_style

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["graph_type"] = self.graph_type.value
        d["source_sheet_id"] = self.source_sheet_id
        d["result_sheet_id"] = self.result_sheet_id
        d["x_label"] = self.x_label
        d["y_label"] = self.y_label
        d["show_error_bars"] = self.show_error_bars
        d["error_style"] = self.error_style
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GraphSheet:
        return cls(
            sheet_id=data["sheet_id"],
            title=data.get("title", "Graph"),
            graph_type=GraphType(data.get("graph_type", GraphType.XY_SCATTER_CURVE.value)),
            source_sheet_id=data.get("source_sheet_id", ""),
            result_sheet_id=data.get("result_sheet_id"),
            x_label=data.get("x_label", "X"),
            y_label=data.get("y_label", "Y"),
            show_error_bars=data.get("show_error_bars", True),
            error_style=data.get("error_style", "SEM"),
        )
