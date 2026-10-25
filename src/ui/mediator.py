"""
Workspace Mediator for GemGraph.
Coordinates notifications and state synchronization across the scientific spreadsheet view,
vector graph canvas, project navigator, and document model.
"""

from __future__ import annotations
from typing import Optional, Any, List, Tuple, Dict
import numpy as np
import pandas as pd
from PyQt6.QtCore import QObject, pyqtSignal


class WorkspaceMediator(QObject):
    """
    Mediator coordinating updates across ScientificSpreadsheetView,
    GraphCanvasWidget, Sidebar/Navigator, and Document model.
    """

    columnRenamed = pyqtSignal(int, str)
    columnInserted = pyqtSignal(int, str)
    columnDeleted = pyqtSignal(int)
    tableModified = pyqtSignal()
    dataChanged = pyqtSignal()
    analysisLayerRemoved = pyqtSignal(str)
    allAnalysisLayersCleared = pyqtSignal()

    _instance: Optional[WorkspaceMediator] = None

    def __init__(self, main_window: Optional[Any] = None, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.main_window = main_window
        self._model: Optional[Any] = None
        WorkspaceMediator._instance = self

    @property
    def model(self) -> Optional[Any]:
        if getattr(self, "_model", None) is not None:
            return self._model
        if self.main_window is not None:
            return getattr(self.main_window, "spreadsheet_model", None)
        return None

    @model.setter
    def model(self, value: Optional[Any]) -> None:
        self._model = value

    def sync_data_to_visualization(
        self,
        model: Optional[Any] = None,
        df: Optional[Any] = None,
        canvas: Optional[Any] = None,
    ) -> List[Tuple[np.ndarray, np.ndarray, str]]:
        """
        Synchronizes data from spreadsheet model/DataFrame to visualization.
        Inspects xy_mapping to pair each Y column with its designated X column,
        falling back to Column 0 if unassigned.
        Packages and returns data as a list of (x_data, y_data, series_name) tuples.
        """
        target_model = model if model is not None else self.model
        xy_map: Dict[Any, Any] = {}
        table = None
        if target_model is not None:
            if hasattr(target_model, "table"):
                table = getattr(target_model, "table", None)
            elif hasattr(target_model, "x_columns") and hasattr(target_model, "get_columns"):
                table = target_model

            if table is not None and hasattr(table, "xy_mapping") and isinstance(table.xy_mapping, dict):
                xy_map.update(table.xy_mapping)
            if hasattr(target_model, "xy_mapping") and isinstance(target_model.xy_mapping, dict):
                xy_map.update(target_model.xy_mapping)

        if df is None:
            if target_model is not None:
                if hasattr(target_model, "dataframe"):
                    df = target_model.dataframe
                elif hasattr(target_model, "to_dataframe"):
                    df = target_model.to_dataframe()

        if df is None:
            return []

        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        if df.empty or len(df.columns) == 0:
            return []

        y_col_indices: List[int] = []
        cols_info = table.get_columns() if (table is not None and hasattr(table, "get_columns")) else None
        if cols_info is not None and len(cols_info) == len(df.columns):
            for idx, c in enumerate(cols_info):
                if c.role == "Y":
                    y_col_indices.append(idx)

        if not y_col_indices:
            for i, col_name in enumerate(df.columns):
                if i == 0:
                    continue
                col_str = str(col_name)
                if i in xy_map or col_name in xy_map or col_str in xy_map:
                    y_col_indices.append(i)
                elif col_str.upper().startswith("X") and not col_str.upper().startswith("XY"):
                    pass
                else:
                    y_col_indices.append(i)

            if not y_col_indices and len(df.columns) > 1:
                y_col_indices = list(range(1, len(df.columns)))

        packaged_data: List[Tuple[np.ndarray, np.ndarray, str]] = []

        for y_idx in y_col_indices:
            y_col_name = df.columns[y_idx]
            series_name = str(y_col_name)

            assigned_x = None
            if y_idx in xy_map:
                assigned_x = xy_map[y_idx]
            elif y_col_name in xy_map:
                assigned_x = xy_map[y_col_name]
            elif str(y_idx) in xy_map:
                assigned_x = xy_map[str(y_idx)]
            elif series_name in xy_map:
                assigned_x = xy_map[series_name]

            # Also check group_idx or group title if table info is available
            if assigned_x is None and cols_info is not None and 0 <= y_idx < len(cols_info):
                g_idx = cols_info[y_idx].group_idx
                if g_idx in xy_map:
                    assigned_x = xy_map[g_idx]
                elif str(g_idx) in xy_map:
                    assigned_x = xy_map[str(g_idx)]
                elif hasattr(table, "y_group_titles") and 0 <= g_idx < len(table.y_group_titles):
                    g_title = table.y_group_titles[g_idx]
                    if g_title in xy_map:
                        assigned_x = xy_map[g_title]

            if assigned_x is not None:
                if isinstance(assigned_x, (np.ndarray, list, pd.Series)):
                    x_data = np.asarray(assigned_x, dtype=np.float64)
                elif isinstance(assigned_x, int):
                    # Check if assigned_x is directly a valid visual column index
                    if 0 <= assigned_x < len(df.columns):
                        if cols_info is not None and 0 <= assigned_x < len(cols_info) and cols_info[assigned_x].role == "X":
                            x_data = pd.to_numeric(df.iloc[:, assigned_x], errors="coerce").to_numpy(dtype=np.float64)
                        elif cols_info is not None and hasattr(table, "x_columns") and assigned_x < len(table.x_columns):
                            # It could be an X column index (e.g. 0 for X1, 1 for X2)
                            found_v_idx = None
                            for v_i, c_info in enumerate(cols_info):
                                if c_info.role == "X" and c_info.group_idx == assigned_x:
                                    found_v_idx = v_i
                                    break
                            if found_v_idx is not None:
                                x_data = pd.to_numeric(df.iloc[:, found_v_idx], errors="coerce").to_numpy(dtype=np.float64)
                            else:
                                x_data = pd.to_numeric(df.iloc[:, assigned_x], errors="coerce").to_numpy(dtype=np.float64)
                        else:
                            x_data = pd.to_numeric(df.iloc[:, assigned_x], errors="coerce").to_numpy(dtype=np.float64)
                    elif table is not None and hasattr(table, "x_columns") and 0 <= assigned_x < len(table.x_columns):
                        x_data = table.x_columns[assigned_x]["data"]
                    else:
                        x_data = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(dtype=np.float64)
                elif assigned_x in df.columns:
                    x_data = pd.to_numeric(df[assigned_x], errors="coerce").to_numpy(dtype=np.float64)
                elif str(assigned_x) in df.columns:
                    x_data = pd.to_numeric(df[str(assigned_x)], errors="coerce").to_numpy(dtype=np.float64)
                elif table is not None and hasattr(table, "x_columns"):
                    found_arr = None
                    for x_col in table.x_columns:
                        if x_col.get("title") == str(assigned_x):
                            found_arr = x_col.get("data")
                            break
                    if found_arr is not None:
                        x_data = np.asarray(found_arr, dtype=np.float64)
                    else:
                        x_data = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(dtype=np.float64)
                else:
                    x_data = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(dtype=np.float64)
            else:
                # Unassigned: fallback to preceding X column or Column 0
                preceding_x_idx = None
                if cols_info is not None and len(cols_info) == len(df.columns):
                    for prev_i in range(y_idx - 1, -1, -1):
                        if cols_info[prev_i].role == "X":
                            preceding_x_idx = prev_i
                            break
                if preceding_x_idx is None:
                    for prev_i in range(y_idx - 1, -1, -1):
                        c_str = str(df.columns[prev_i]).upper()
                        if c_str.startswith("X") and not c_str.startswith("XY"):
                            preceding_x_idx = prev_i
                            break

                if preceding_x_idx is not None and 0 <= preceding_x_idx < len(df.columns):
                    x_data = pd.to_numeric(df.iloc[:, preceding_x_idx], errors="coerce").to_numpy(dtype=np.float64)
                else:
                    x_data = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(dtype=np.float64)

            y_data = pd.to_numeric(df.iloc[:, y_idx], errors="coerce").to_numpy(dtype=np.float64)
            packaged_data.append((x_data, y_data, series_name))

        target_canvas = canvas
        if target_canvas is None and self.main_window is not None:
            target_canvas = getattr(self.main_window, "graph_canvas", None)

        if target_canvas is not None and hasattr(target_canvas, "refresh_plot"):
            try:
                target_canvas.refresh_plot(packaged_data)
            except TypeError:
                target_canvas.refresh_plot()

        return packaged_data

    @classmethod
    def get_instance(cls, main_window: Optional[Any] = None) -> WorkspaceMediator:
        from PyQt6 import sip
        if cls._instance is None or sip.isdeleted(cls._instance):
            cls._instance = cls(main_window)
        elif main_window is not None:
            try:
                cls._instance.main_window = main_window
            except RuntimeError:
                cls._instance = cls(main_window)
        return cls._instance

    def notify_data_changed(self, table: Optional[Any] = None) -> None:
        """
        Notifies workspace of dataChanged signals from active table.
        Ensures frozen graphs are completely isolated and ignored.
        """
        self.dataChanged.emit()
        self.notify_table_modified()

    def on_table_data_changed(self, top_left: Optional[Any] = None, bottom_right: Optional[Any] = None, roles: Optional[Any] = None) -> None:
        """Slot handler connecting to QAbstractItemModel.dataChanged signals."""
        self.notify_data_changed()

    def notify_column_renamed(self, col_idx: int, new_name: str) -> None:
        """Notifies workspace of column renaming, updating graph legends and axis labels."""
        self.columnRenamed.emit(col_idx, new_name)
        if self._is_active_graph_frozen():
            return
        if self.main_window and hasattr(self.main_window, "_on_table_data_modified"):
            self.main_window._on_table_data_modified()

    def notify_column_inserted(self, col_idx: int, name: str) -> None:
        """Notifies workspace of column insertion, triggering view and graph refresh."""
        self.columnInserted.emit(col_idx, name)
        if self._is_active_graph_frozen():
            return
        if self.main_window and hasattr(self.main_window, "_on_table_data_modified"):
            self.main_window._on_table_data_modified()

    def notify_column_deleted(self, col_idx: int) -> None:
        """Notifies workspace of column deletion, updating graph series and downstream DAG."""
        self.columnDeleted.emit(col_idx)
        if self._is_active_graph_frozen():
            return
        if self.main_window and hasattr(self.main_window, "_on_table_data_modified"):
            self.main_window._on_table_data_modified()

    def _is_active_graph_frozen(self) -> bool:
        """Returns True if the current graph canvas or document graph is flagged as frozen."""
        if not self.main_window:
            return False
        canvas = getattr(self.main_window, "graph_canvas", None)
        if canvas is not None:
            sheet = getattr(canvas, "current_graph_sheet", None) or getattr(canvas, "active_sheet", None)
            if sheet is not None and getattr(sheet, "is_frozen", False):
                return True
        doc = getattr(self.main_window, "document", None)
        if doc is not None:
            active_id = getattr(doc, "active_sheet_id", None)
            if active_id:
                active_sheet = doc.get_sheet(active_id)
                if active_sheet is not None and getattr(active_sheet, "is_frozen", False):
                    return True
        return False

    def notify_table_modified(self) -> None:
        """Notifies workspace of generic table content mutation, ignoring frozen graphs."""
        self.tableModified.emit()
        if self._is_active_graph_frozen():
            return
        if self.main_window and hasattr(self.main_window, "_on_table_data_modified"):
            self.main_window._on_table_data_modified()

    def notify_analysis_layer_removed(self, layer_id: str) -> None:
        """Notifies workspace of an analysis overlay layer removal."""
        self.analysisLayerRemoved.emit(layer_id)
        if self.main_window and hasattr(self.main_window, "graph_canvas"):
            canvas = getattr(self.main_window, "graph_canvas")
            if hasattr(canvas, "analysis_layers") and layer_id in canvas.analysis_layers:
                canvas.remove_analysis_layer(layer_id, notify=False)

    def notify_clear_all_analysis_layers(self) -> None:
        """Notifies workspace that all analysis overlay layers should be cleared."""
        self.allAnalysisLayersCleared.emit()
        if self.main_window and hasattr(self.main_window, "graph_canvas"):
            canvas = getattr(self.main_window, "graph_canvas")
            if hasattr(canvas, "clear_all_analysis_layers"):
                if hasattr(canvas, "analysis_layers") and canvas.analysis_layers:
                    canvas.clear_all_analysis_layers(notify=False)
