"""
Data Engine module for GemGraph.
Provides high-performance I/O, native .bigo/.prism archive serialization,
and dataset generation contracts.
"""

from __future__ import annotations
from src.io.prism_io import PrismIO
from src.io.xml_io import PrismXMLIO
from src.core.table import DataTable, TableType
from src.core.document import PrismDocument

SUPPORTED_EXTENSIONS = [".bigo", ".prism", ".pzfx", ".xml", ".csv"]
PRIMARY_EXTENSION = ".bigo"
FILE_FILTERS = (
    "GemGraph Projects (*.bigo);;"
    "GraphPad Prism Projects (*.prism *.pzfx);;"
    "All Supported (*.bigo *.prism *.pzfx);;"
    "All Files (*)"
)

__all__ = [
    "PrismIO",
    "PrismXMLIO",
    "DataTable",
    "TableType",
    "PrismDocument",
    "SUPPORTED_EXTENSIONS",
    "PRIMARY_EXTENSION",
    "FILE_FILTERS",
]
