"""
Native archive I/O engine for GemGraph.
Supports dual native .bigo and .prism project archives with full bi-directional compatibility.
"""

from __future__ import annotations
import os
from src.core.document import PrismDocument
from src.io.prism_io import PrismIO
from src.io.xml_io import PrismXMLIO


class DataEngineIO:
    """High-level facade for project loading and saving across formats."""

    @staticmethod
    def save(document: PrismDocument, file_path: str) -> None:
        """Saves project to .bigo, .prism, or .xml."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext in (".pzfx", ".xml", ".pzm"):
            PrismXMLIO.export_xml(document, file_path)
        else:
            PrismIO.save(document, file_path)

    @staticmethod
    def load(file_path: str) -> PrismDocument:
        """Loads project from .bigo, .prism, .pzfx, .xml, or .csv."""
        return PrismIO.load(file_path)
