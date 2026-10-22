"""
Scientific Table View and Model module for GemGraph.
Exposes ScientificTableWidget and ScientificTableModel with in-sheet editing,
inline column renaming, context menus, and pandas DataFrame synchronization.
"""

from __future__ import annotations
from src.ui.spreadsheet import (
    ScientificSpreadsheetModel,
    ScientificSpreadsheetView,
    ScientificTableModel,
    ScientificTableWidget,
    ReplicateSpreadsheet,
)

__all__ = [
    "ScientificTableModel",
    "ScientificTableWidget",
    "ScientificSpreadsheetModel",
    "ScientificSpreadsheetView",
    "ReplicateSpreadsheet",
]
