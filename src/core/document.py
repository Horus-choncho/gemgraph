"""
Document model managing sheets, relationships, and project state.
Derived from Core.dll document layout (REA-CORE-STORAGE-204).
"""

from __future__ import annotations
import uuid
from typing import Dict, List, Optional, Any
from src.core.sheet import BaseSheet, DataSheet, InfoSheet, ResultSheet, GraphSheet, SheetType
from src.core.table import TableType, DataTable


class PrismDocument:
    """
    Top-level project document holding all sheets (Data, Info, Results, Graphs).
    """

    def __init__(self, title: str = "Untitled Project") -> None:
        self.doc_id: str = str(uuid.uuid4())
        self.title: str = title
        self.sheets: Dict[str, BaseSheet] = {}
        self.sheet_order: List[str] = []
        self.active_sheet_id: Optional[str] = None

    def add_sheet(self, sheet: BaseSheet) -> None:
        """Adds a sheet to the document and updates tracking order."""
        if sheet.sheet_id in self.sheets:
            raise ValueError(f"Sheet with ID {sheet.sheet_id} already exists")
        self.sheets[sheet.sheet_id] = sheet
        self.sheet_order.append(sheet.sheet_id)
        if self.active_sheet_id is None:
            self.active_sheet_id = sheet.sheet_id

    def get_sheet(self, sheet_id: str) -> Optional[BaseSheet]:
        """Returns the sheet matching sheet_id, or None."""
        return self.sheets.get(sheet_id)

    def get_sheets_by_type(self, sheet_type: Any) -> List[BaseSheet]:
        """Returns all sheets of a specific type in insertion order."""
        if isinstance(sheet_type, SheetType):
            return [
                self.sheets[sid]
                for sid in self.sheet_order
                if self.sheets[sid].sheet_type == sheet_type
            ]
        elif isinstance(sheet_type, type) and issubclass(sheet_type, BaseSheet):
            return [
                self.sheets[sid]
                for sid in self.sheet_order
                if isinstance(self.sheets[sid], sheet_type)
            ]
        return []

    def has_path(self, start_id: str, target_id: str) -> bool:
        """Returns True if there is a directed path from start_id to target_id."""
        visited = set()
        queue = [start_id]
        while queue:
            curr = queue.pop(0)
            if curr == target_id:
                return True
            sheet = self.sheets.get(curr)
            if sheet:
                for c_id in sheet.child_sheet_ids:
                    if c_id not in visited:
                        visited.add(c_id)
                        queue.append(c_id)
        return False

    def register_dependency(self, parent_id: str, child_id: str) -> bool:
        """Records DAG dependency where child_id depends on parent_id. Rejects cycles."""
        if parent_id == child_id:
            return False
        # If child_id already has a path to parent_id, adding parent_id -> child_id would create a cycle
        if self.has_path(child_id, parent_id):
            return False

        parent = self.sheets.get(parent_id)
        child = self.sheets.get(child_id)
        if parent and child:
            if child_id not in parent.child_sheet_ids:
                parent.child_sheet_ids.append(child_id)
            if parent_id not in child.parent_sheet_ids:
                child.parent_sheet_ids.append(parent_id)
            return True
        return False

    def freeze_sheet(self, sheet_id: str, frozen: bool = True) -> bool:
        """Freezes or unfreezes a sheet. When frozen, recalculation is disconnected."""
        sheet = self.sheets.get(sheet_id)
        if not sheet:
            return False
        sheet.is_frozen = frozen
        return True

    def recalculate_downstream(self, sheet_id: str) -> List[str]:
        """
        Traverses downstream dependent sheets in topological order,
        recalculating any non-frozen child analyses.
        """
        recalculated: List[str] = []
        visited = set()
        queue = [sheet_id]

        while queue:
            curr_id = queue.pop(0)
            curr = self.sheets.get(curr_id)
            if not curr:
                continue

            for child_id in curr.child_sheet_ids:
                if child_id in visited:
                    continue
                visited.add(child_id)
                child = self.sheets.get(child_id)
                if child and not child.is_frozen:
                    if isinstance(child, DataSheet) and child.calc_spec:
                        self._recalculate_data_sheet(child)
                    elif isinstance(child, ResultSheet) and child.calc_spec:
                        self._recalculate_result_sheet(child)
                    recalculated.append(child_id)
                    queue.append(child_id)

        return recalculated

    def _recalculate_data_sheet(self, sheet: DataSheet) -> None:
        """Executes recalculation for a DataSheet based on its calc_spec and parent sheet."""
        if not sheet.parent_sheet_ids or not sheet.calc_spec:
            return
        parent = self.sheets.get(sheet.parent_sheet_ids[0])
        if not isinstance(parent, DataSheet):
            return

        spec = sheet.calc_spec
        t = spec.get("type")
        from src.math.transforms import transform_data_table
        from src.math.manipulation import (
            normalize_table, prune_rows, remove_baseline, transpose_table,
            Baseline0Method, Baseline100Method, NormalizeScale, BaselineMethod, BaselineOperation
        )

        if t == "transform":
            sheet.table = transform_data_table(
                parent.table,
                transform_name=spec.get("transform_name", "log10"),
                k_constant=spec.get("k_constant", 1.0),
                user_expr=spec.get("user_expr"),
                pharma_type=spec.get("pharma_type"),
            )
            sheet.table.table_id = sheet.sheet_id
            sheet.table.title = sheet.title
        elif t == "normalize":
            b0_m = Baseline0Method(spec.get("b0_method", "smallest"))
            b100_m = Baseline100Method(spec.get("b100_method", "largest"))
            sc_val = spec.get("scale", 100.0)
            sc = NormalizeScale.FRACTION if sc_val == 1.0 else NormalizeScale.PERCENT
            sheet.table = normalize_table(
                parent.table,
                b0_method=b0_m,
                b0_val=spec.get("b0_val", 0.0),
                b100_method=b100_m,
                b100_val=spec.get("b100_val", 100.0),
                scale=sc,
            )
            sheet.table.table_id = sheet.sheet_id
            sheet.table.title = sheet.title
        elif t == "prune":
            sheet.table = prune_rows(
                parent.table,
                x_min=spec.get("x_min"),
                x_max=spec.get("x_max"),
                decimate_k=spec.get("decimate_k"),
                average_k=spec.get("average_k"),
            )
            sheet.table.table_id = sheet.sheet_id
            sheet.table.title = sheet.title
        elif t == "remove_baseline":
            new_table = DataTable(
                table_id=sheet.sheet_id,
                title=sheet.title,
                table_type=parent.table.table_type,
                num_rows=parent.table.num_rows,
                num_replicates=parent.table.num_replicates,
                y_group_titles=parent.table.y_group_titles,
            )
            new_table.row_titles = list(parent.table.row_titles)
            if parent.table.x_data is not None:
                new_table.x_data = np.copy(parent.table.x_data)
                new_table.x_title = parent.table.x_title
            method = BaselineMethod(spec.get("method", "first_row"))
            op = BaselineOperation(spec.get("operation", "subtract"))
            k_rows = spec.get("k_rows", 1)
            for g in range(len(parent.table.y_group_titles)):
                for s in range(parent.table.num_replicates):
                    new_table.y_data[g, :, s] = remove_baseline(
                        parent.table.y_data[g, :, s],
                        x=parent.table.x_data,
                        method=method,
                        operation=op,
                        k_rows=k_rows,
                    )
            sheet.table = new_table
        elif t == "transpose":
            sheet.table = transpose_table(parent.table)
            sheet.table.table_id = sheet.sheet_id
            sheet.table.title = sheet.title

    def _recalculate_result_sheet(self, result: ResultSheet) -> None:
        """Executes recalculation for a ResultSheet based on its calc_spec."""
        src_sheet = self.sheets.get(result.source_sheet_id)
        if not isinstance(src_sheet, DataSheet):
            return

        spec = result.calc_spec
        analysis_type = spec.get("type")
        table = src_sheet.table

        if analysis_type == "nonlinear":
            from src.math.nonlinear import NonlinearEngine, STANDARD_MODELS, ConvergencePreset
            model_name = spec.get("model_name", "Sigmoidal Dose-Response (4PL)")
            model = STANDARD_MODELS.get(model_name)
            if not model:
                for k, m in STANDARD_MODELS.items():
                    if "sigmoidal" in k.lower() or "4pl" in k.lower():
                        model = m
                        model_name = k
                        break
            if not model:
                model = next(iter(STANDARD_MODELS.values()))

            group_idx = spec.get("group_idx", 0)
            preset = spec.get("preset", ConvergencePreset.STANDARD)
            if model and 0 <= group_idx < len(table.y_group_titles):
                xs, ys = table.get_clean_xy(group_idx)
                if len(xs) >= len(model.param_names):
                    fit_result = NonlinearEngine.fit(xs, ys, model, preset=preset)
                    from src.core.hooking import GLOBAL_REGISTRY
                    params_dict = {p["name"]: p["best_fit"] for p in fit_result.get("parameters", [])}
                    result.raw_result_data = {
                        "model_name": model_name,
                        "params": params_dict,
                        "parameters": fit_result.get("parameters", []),
                        "goodness_of_fit": {
                            "R2": fit_result.get("r_squared", 0.0),
                            "Sy.x": fit_result.get("syx", 0.0),
                            "AICc": fit_result.get("aicc", 0.0),
                        },
                        "x_dense": fit_result.get("x_dense", []),
                        "y_dense": fit_result.get("y_dense", []),
                    }
                    param_rows = [
                        [p["name"], f"{p['best_fit']:.4g}"]
                        for p in fit_result.get("parameters", [])
                    ]
                    result.summary_tables = [
                        {
                            "title": model_name,
                            "headers": ["Parameter", "Best-fit value"],
                            "rows": param_rows,
                        }
                    ]
                    result.tabs = [
                        {
                            "name": "Table of results",
                            "grid_type": "red",
                            "headers": ["Parameter", "Best-fit value"],
                            "rows": param_rows,
                            "visible": True,
                        }
                    ]
                    if "x_dense" in fit_result and "y_dense" in fit_result:
                        result.tabs.append({
                            "name": "Curve Coordinates",
                            "grid_type": "green",
                            "headers": ["X (Curve)", "Y (Fit)"],
                            "rows": [
                                [round(x, 4), round(y, 4)]
                                for x, y in zip(fit_result["x_dense"][:50], fit_result["y_dense"][:50])
                            ],
                            "visible": True,
                        })
                    GLOBAL_REGISTRY.extract_from_result(result)

        elif analysis_type == "linear":
            from src.math.nonlinear import NonlinearEngine, STANDARD_MODELS
            model = STANDARD_MODELS.get("Straight Line (Linear)")
            group_idx = spec.get("group_idx", 0)
            if model and 0 <= group_idx < len(table.y_group_titles):
                xs, ys = table.get_clean_xy(group_idx)
                if len(xs) >= 2:
                    fit_res = NonlinearEngine.fit(xs, ys, model)
                    from src.core.hooking import GLOBAL_REGISTRY
                    params_dict = {p["name"]: p["best_fit"] for p in fit_res.get("parameters", [])}
                    result.raw_result_data = {
                        "model_name": "Straight Line (Linear)",
                        "params": params_dict,
                        "parameters": fit_res.get("parameters", []),
                        "goodness_of_fit": {
                            "R2": fit_res.get("r_squared", 0.0),
                            "Sy.x": fit_res.get("syx", 0.0),
                        },
                        "x_dense": fit_res.get("x_dense", []),
                        "y_dense": fit_res.get("y_dense", []),
                    }
                    param_rows = [
                        [p["name"], f"{p['best_fit']:.4g}"]
                        for p in fit_res.get("parameters", [])
                    ]
                    result.summary_tables = [
                        {
                            "title": "Linear Regression",
                            "headers": ["Parameter", "Best-fit value"],
                            "rows": param_rows,
                        }
                    ]
                    result.tabs = [
                        {
                            "name": "Table of results",
                            "grid_type": "red",
                            "headers": ["Parameter", "Best-fit value"],
                            "rows": param_rows,
                            "visible": True,
                        }
                    ]
                    if "x_dense" in fit_res and "y_dense" in fit_res:
                        result.tabs.append({
                            "name": "Curve Coordinates",
                            "grid_type": "green",
                            "headers": ["X (Curve)", "Y (Fit)"],
                            "rows": [
                                [round(x, 4), round(y, 4)]
                                for x, y in zip(fit_res["x_dense"][:50], fit_res["y_dense"][:50])
                            ],
                            "visible": True,
                        })
                    GLOBAL_REGISTRY.extract_from_result(result)

    def remove_sheet(self, sheet_id: str) -> bool:
        """Removes a sheet; orphaned children have parent links severed and is_orphaned set to True."""
        if sheet_id not in self.sheets:
            return False

        removed_sheet = self.sheets[sheet_id]

        # Handle orphaned children
        for child_id in list(removed_sheet.child_sheet_ids):
            child = self.sheets.get(child_id)
            if child:
                if sheet_id in child.parent_sheet_ids:
                    child.parent_sheet_ids.remove(sheet_id)
                child.is_orphaned = True
                if hasattr(child, "source_sheet_id") and child.source_sheet_id == sheet_id:
                    child.source_sheet_id = ""

        # Remove from parents' child list
        for parent_id in list(removed_sheet.parent_sheet_ids):
            parent = self.sheets.get(parent_id)
            if parent and sheet_id in parent.child_sheet_ids:
                parent.child_sheet_ids.remove(sheet_id)

        del self.sheets[sheet_id]
        self.sheet_order.remove(sheet_id)
        if self.active_sheet_id == sheet_id:
            self.active_sheet_id = self.sheet_order[0] if self.sheet_order else None
        return True

    def create_data_sheet(
        self,
        title: Optional[str] = None,
        table_type: TableType = TableType.XY,
        num_rows: int = 20,
        num_replicates: int = 3,
        y_group_titles: Optional[List[str]] = None,
        calc_spec: Optional[Dict[str, Any]] = None,
    ) -> DataSheet:
        """Creates and registers a new DataSheet."""
        sheet_count = len(self.get_sheets_by_type(SheetType.DATA)) + 1
        name = title or f"Data {sheet_count}"
        sheet_id = f"data_{uuid.uuid4().hex[:8]}"
        table = DataTable(
            table_id=sheet_id,
            title=name,
            table_type=table_type,
            num_rows=num_rows,
            num_replicates=num_replicates,
            y_group_titles=y_group_titles,
        )
        sheet = DataSheet(sheet_id=sheet_id, title=name, table=table, calc_spec=calc_spec)
        self.add_sheet(sheet)
        return sheet

    def create_info_sheet(self, title: Optional[str] = None) -> InfoSheet:
        """Creates and registers a new InfoSheet."""
        sheet_count = len(self.get_sheets_by_type(SheetType.INFO)) + 1
        name = title or f"Info {sheet_count}"
        sheet_id = f"info_{uuid.uuid4().hex[:8]}"
        sheet = InfoSheet(sheet_id=sheet_id, title=name)
        self.add_sheet(sheet)
        return sheet

    def create_result_sheet(
        self,
        title: Optional[str] = None,
        analysis_type: str = "Analysis",
        source_sheet_id: str = "",
        summary_tables: Optional[List[Dict[str, Any]]] = None,
        raw_result_data: Optional[Dict[str, Any]] = None,
        grid_type: str = "red",
        tabs: Optional[List[Dict[str, Any]]] = None,
        calc_spec: Optional[Dict[str, Any]] = None,
    ) -> ResultSheet:
        """Creates and registers a new ResultSheet, linking DAG dependency."""
        sheet_count = len(self.get_sheets_by_type(SheetType.RESULTS)) + 1
        name = title or f"Results {sheet_count}"
        sheet_id = f"result_{uuid.uuid4().hex[:8]}"
        sheet = ResultSheet(
            sheet_id=sheet_id,
            title=name,
            analysis_type=analysis_type,
            source_sheet_id=source_sheet_id,
            summary_tables=summary_tables,
            raw_result_data=raw_result_data,
            grid_type=grid_type,
            tabs=tabs,
            calc_spec=calc_spec,
        )
        self.add_sheet(sheet)
        if source_sheet_id:
            self.register_dependency(source_sheet_id, sheet_id)
        return sheet

    def create_graph_sheet(
        self,
        title: Optional[str] = None,
        source_sheet_id: str = "",
        result_sheet_id: Optional[str] = None,
    ) -> GraphSheet:
        """Creates and registers a new GraphSheet linked to a DataSheet."""
        sheet_count = len(self.get_sheets_by_type(SheetType.GRAPH)) + 1
        name = title or f"Graph {sheet_count}"
        sheet_id = f"graph_{uuid.uuid4().hex[:8]}"
        sheet = GraphSheet(
            sheet_id=sheet_id,
            title=name,
            source_sheet_id=source_sheet_id,
            result_sheet_id=result_sheet_id,
        )
        self.add_sheet(sheet)
        if source_sheet_id:
            self.register_dependency(source_sheet_id, sheet_id)
        if result_sheet_id:
            self.register_dependency(result_sheet_id, sheet_id)
        return sheet

    def to_dict(self) -> Dict[str, Any]:
        """Serializes document to JSON structure."""
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "active_sheet_id": self.active_sheet_id,
            "sheet_order": self.sheet_order,
            "sheets": [self.sheets[sid].to_dict() for sid in self.sheet_order],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PrismDocument:
        """Deserializes document from JSON structure."""
        doc = cls(title=data.get("title", "Untitled Project"))
        doc.doc_id = data.get("doc_id", str(uuid.uuid4()))
        doc.active_sheet_id = data.get("active_sheet_id")

        for s_data in data.get("sheets", []):
            stype = s_data.get("sheet_type")
            if stype == SheetType.DATA.value:
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
            sid for sid in data.get("sheet_order", []) if sid in doc.sheets
        ]
        if not doc.sheet_order and doc.sheets:
            doc.sheet_order = list(doc.sheets.keys())
        if doc.active_sheet_id not in doc.sheets and doc.sheet_order:
            doc.active_sheet_id = doc.sheet_order[0]

        return doc
