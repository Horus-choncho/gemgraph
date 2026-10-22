"""
Workspace Mediator for GemGraph.
Coordinates notifications and state synchronization across the scientific spreadsheet view,
vector graph canvas, project navigator, and document model.
"""

from __future__ import annotations
from typing import Optional, Any
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
    analysisLayerRemoved = pyqtSignal(str)
    allAnalysisLayersCleared = pyqtSignal()

    _instance: Optional[WorkspaceMediator] = None

    def __init__(self, main_window: Optional[Any] = None, parent: Optional[QObject] = None) -> None:
        super().__init__(parent or (main_window if isinstance(main_window, QObject) else None))
        self.main_window = main_window
        WorkspaceMediator._instance = self

    @classmethod
    def get_instance(cls, main_window: Optional[Any] = None) -> WorkspaceMediator:
        if cls._instance is None:
            cls._instance = cls(main_window)
        elif main_window is not None:
            cls._instance.main_window = main_window
        return cls._instance

    def notify_column_renamed(self, col_idx: int, new_name: str) -> None:
        """Notifies workspace of column renaming, updating graph legends and axis labels."""
        self.columnRenamed.emit(col_idx, new_name)
        if self.main_window and hasattr(self.main_window, "_on_table_data_modified"):
            self.main_window._on_table_data_modified()

    def notify_column_inserted(self, col_idx: int, name: str) -> None:
        """Notifies workspace of column insertion, triggering view and graph refresh."""
        self.columnInserted.emit(col_idx, name)
        if self.main_window and hasattr(self.main_window, "_on_table_data_modified"):
            self.main_window._on_table_data_modified()

    def notify_column_deleted(self, col_idx: int) -> None:
        """Notifies workspace of column deletion, updating graph series and downstream DAG."""
        self.columnDeleted.emit(col_idx)
        if self.main_window and hasattr(self.main_window, "_on_table_data_modified"):
            self.main_window._on_table_data_modified()

    def notify_table_modified(self) -> None:
        """Notifies workspace of generic table content mutation."""
        self.tableModified.emit()
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
