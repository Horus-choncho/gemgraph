"""
Prism Navigator Sidebar for GemGraph.
Derived from BridgeCLR SheetNavigator hierarchy (REA-CLR-VM-102).
"""

from __future__ import annotations
from typing import Dict, Optional
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QTreeWidget, QTreeWidgetItem, QMenu, QInputDialog, QMessageBox
from src.core.document import PrismDocument
from src.core.sheet import BaseSheet, SheetType


class PrismNavigator(QTreeWidget):
    """
    Project hierarchy tree organizing sheets into Data, Info, Results, and Graphs folders.
    """

    sheetSelected = pyqtSignal(str)  # Emits sheet_id
    sheetDeleted = pyqtSignal(str)   # Emits sheet_id

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setHeaderLabel("Project Navigator")
        self.setAnimated(True)
        self.itemClicked.connect(self._on_item_clicked)

        # Folder root items
        self.folder_items: Dict[SheetType, QTreeWidgetItem] = {}
        self._init_folders()

        self.current_doc: Optional[PrismDocument] = None

    def _init_folders(self) -> None:
        self.clear()
        self.folder_items[SheetType.DATA] = QTreeWidgetItem(self, ["📁 Data Tables"])
        self.folder_items[SheetType.INFO] = QTreeWidgetItem(self, ["📁 Info Sheets"])
        self.folder_items[SheetType.RESULTS] = QTreeWidgetItem(self, ["📁 Results"])
        self.folder_items[SheetType.GRAPH] = QTreeWidgetItem(self, ["📁 Graphs"])

        for item in self.folder_items.values():
            item.setExpanded(True)
            item.setData(0, Qt.ItemDataRole.UserRole, None)

    def populate(self, document: PrismDocument) -> None:
        """Rebuilds tree items from document sheets, supporting family tree nesting."""
        self.current_doc = document
        self._init_folders()

        items_by_id: Dict[str, QTreeWidgetItem] = {}

        # First pass: Data and Info sheets under their folders
        for sheet_id in document.sheet_order:
            sheet = document.sheets[sheet_id]
            if sheet.sheet_type in (SheetType.DATA, SheetType.INFO, SheetType.GRAPH):
                folder_item = self.folder_items.get(sheet.sheet_type)
                if folder_item is not None:
                    title = sheet.title
                    if getattr(sheet, "is_frozen", False):
                        title += " ❄️"
                    if getattr(sheet, "is_orphaned", False):
                        title += " ⚠️"
                    leaf = QTreeWidgetItem(folder_item, [title])
                    leaf.setData(0, Qt.ItemDataRole.UserRole, sheet.sheet_id)
                    items_by_id[sheet.sheet_id] = leaf
                    if sheet.sheet_id == document.active_sheet_id:
                        self.setCurrentItem(leaf)

        # Second pass: Result sheets, nesting children under their parent result sheet if present
        for sheet_id in document.sheet_order:
            sheet = document.sheets[sheet_id]
            if sheet.sheet_type == SheetType.RESULTS:
                title = sheet.title
                if getattr(sheet, "is_frozen", False):
                    title += " ❄️"
                if getattr(sheet, "is_orphaned", False):
                    title += " ⚠️"

                # Check if parent result sheet exists in items_by_id
                parent_item = None
                for p_id in getattr(sheet, "parent_sheet_ids", []):
                    if p_id in items_by_id and p_id in document.sheets and document.sheets[p_id].sheet_type == SheetType.RESULTS:
                        parent_item = items_by_id[p_id]
                        break

                if parent_item is None:
                    parent_item = self.folder_items.get(SheetType.RESULTS)

                leaf = QTreeWidgetItem(parent_item, [title])
                leaf.setData(0, Qt.ItemDataRole.UserRole, sheet.sheet_id)
                items_by_id[sheet.sheet_id] = leaf
                if sheet.sheet_id == document.active_sheet_id:
                    self.setCurrentItem(leaf)

        self.expandAll()

    def _on_item_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        sheet_id = item.data(0, Qt.ItemDataRole.UserRole)
        if sheet_id:
            self.sheetSelected.emit(sheet_id)

    def contextMenuEvent(self, event) -> None:
        item = self.itemAt(event.pos())
        if not item:
            return
        sheet_id = item.data(0, Qt.ItemDataRole.UserRole)
        if not sheet_id:
            return

        menu = QMenu(self)
        rename_act = menu.addAction("Rename Sheet...")
        delete_act = menu.addAction("Delete Sheet")

        action = menu.exec(event.globalPos())
        if action == rename_act:
            new_title, ok = QInputDialog.getText(self, "Rename Sheet", "New title:", text=item.text(0))
            if ok and new_title.strip() and self.current_doc:
                sheet = self.current_doc.get_sheet(sheet_id)
                if sheet:
                    sheet.title = new_title.strip()
                    item.setText(0, sheet.title)
        elif action == delete_act:
            reply = QMessageBox.question(
                self,
                "Delete Sheet",
                f"Are you sure you want to delete '{item.text(0)}'?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.sheetDeleted.emit(sheet_id)
