"""
Interactive Scientific Info Sheet View for GemGraph.
Two-column property table for GLP metadata and split-pane experimental protocol notes.
"""

from __future__ import annotations
from typing import Optional
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QLabel,
    QHeaderView,
)
from src.core.info_sheet import InfoSheet, GLP_FIELD_DEFINITIONS


class InfoSheetView(QWidget):
    """
    Structured scientific laboratory notebook view displaying GLP experimental
    properties in an editable property table and protocol observations in a notes pane.
    """

    infoModified = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.current_sheet: Optional[InfoSheet] = None
        self._is_updating = False

        self._setup_ui()

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setChildrenCollapsible(False)

        # Top Pane: GLP Experimental Metadata
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)

        header_label = QLabel("GLP Experimental Metadata & Specification")
        font = header_label.font()
        font.setBold(True)
        font.setPointSize(11)
        header_label.setFont(font)
        top_layout.addWidget(header_label)

        self.table = QTableWidget(len(GLP_FIELD_DEFINITIONS), 2)
        self.table.setHorizontalHeaderLabels(["Property Name", "Value"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)

        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #ffffff;
                alternate-background-color: #f8f9fa;
                color: #000000;
                gridline-color: #d0d0d0;
                border: 1px solid #d0d0d0;
            }
            QHeaderView::section {
                background-color: #f0f0f0;
                color: #000000;
                border: 1px solid #d0d0d0;
                padding: 4px 8px;
                font-weight: bold;
            }
        """)

        # Initialize rows
        prop_font = QFont("Arial", 10)
        prop_font.setBold(True)
        for row, (attr, label) in enumerate(GLP_FIELD_DEFINITIONS):
            item_name = QTableWidgetItem(label)
            item_name.setFlags(item_name.flags() & ~Qt.ItemFlag.ItemIsEditable)
            item_name.setFont(prop_font)
            item_name.setForeground(QColor("#333333"))
            self.table.setItem(row, 0, item_name)

            item_val = QTableWidgetItem("")
            self.table.setItem(row, 1, item_val)

        self.table.cellChanged.connect(self._on_cell_changed)
        top_layout.addWidget(self.table)
        splitter.addWidget(top_widget)

        # Bottom Pane: Experimental Notes & Protocol Details
        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0, 4, 0, 0)

        notes_label = QLabel("Experimental Notes & Protocol Details")
        notes_font = notes_label.font()
        notes_font.setBold(True)
        notes_font.setPointSize(11)
        notes_label.setFont(notes_font)
        bottom_layout.addWidget(notes_label)

        self.notes_edit = QTextEdit()
        self.notes_edit.setStyleSheet("""
            QTextEdit {
                background-color: #ffffff;
                color: #000000;
                border: 1px solid #d0d0d0;
                padding: 6px;
                font-family: Arial, sans-serif;
                font-size: 11pt;
            }
        """)
        self.notes_edit.textChanged.connect(self._on_notes_changed)
        bottom_layout.addWidget(self.notes_edit)
        splitter.addWidget(bottom_widget)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter)

    def display_info(self, sheet: InfoSheet) -> None:
        """Populates the view from an InfoSheet instance."""
        self.current_sheet = sheet
        self._is_updating = True

        for row, (attr, label) in enumerate(GLP_FIELD_DEFINITIONS):
            val = sheet.get_field_value(attr)
            item = self.table.item(row, 1)
            if item is not None:
                item.setText(val)

        self.notes_edit.blockSignals(True)
        self.notes_edit.setPlainText(sheet.notes)
        self.notes_edit.blockSignals(False)

        self._is_updating = False

    def _on_cell_changed(self, row: int, col: int) -> None:
        if self._is_updating or col != 1 or self.current_sheet is None:
            return
        attr, label = GLP_FIELD_DEFINITIONS[row]
        item = self.table.item(row, col)
        new_val = item.text().strip() if item else ""
        self.current_sheet.set_field_value(attr, new_val)
        self.infoModified.emit()

    def _on_notes_changed(self) -> None:
        if self._is_updating or self.current_sheet is None:
            return
        self.current_sheet.notes = self.notes_edit.toPlainText()
        self.infoModified.emit()
