"""
Main application launcher and interface shell entry point for GemGraph.
Derived from BridgeCLR GraphController contracts and Clean-Room Desktop architecture.
"""

from __future__ import annotations
import sys
from typing import Optional
from PyQt6.QtWidgets import QApplication
from src.ui.main_window import MainWindow
from src.app import create_app, main


def init_interface_shell(window: MainWindow) -> None:
    """Configures or re-applies stable 3-pane layout shell on MainWindow."""
    if hasattr(window, "init_interface_shell"):
        window.init_interface_shell()


__all__ = ["MainWindow", "create_app", "main", "init_interface_shell"]


if __name__ == "__main__":
    sys.exit(main())
