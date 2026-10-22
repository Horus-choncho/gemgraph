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


def init_interface_shell(
    window: Optional[MainWindow] = None,
    self: Optional[MainWindow] = None,
) -> MainWindow:
    """Configures or re-applies stable 3-pane layout shell on MainWindow."""
    target = window or self
    if target is None:
        target = MainWindow()
    if hasattr(target, "init_interface_shell"):
        target.init_interface_shell()
    return target


__all__ = ["MainWindow", "create_app", "main", "init_interface_shell"]


if __name__ == "__main__":
    sys.exit(main())
