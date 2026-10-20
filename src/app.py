"""
Application Entry Point and Lifecycle Manager for GemGraph.
"""

from __future__ import annotations
import sys
import os
from typing import Tuple
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from src.ui.main_window import MainWindow


def create_app() -> Tuple[QApplication, MainWindow]:
    """Creates and configures QApplication and MainWindow."""
    # Ensure offscreen rendering if QT_QPA_PLATFORM is set or no display available
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    app.setApplicationName("GemGraph")
    app.setOrganizationName("GemGraph")
    app.setApplicationVersion("1.0.0")

    window = MainWindow()
    return app, window


def main() -> int:
    """Main execution function."""
    app, window = create_app()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
