"""
Granular Office & PDF Document Export Dialog for GemGraph.
Provides ExportProjectDialog with a hierarchical checklist tree (Select All, Info Sheets,
Data Tables, Analysis Results, Graphs) and target format selector (.docx, .odt, .xlsx, .ods, .pdf).
"""

from __future__ import annotations
import os
from typing import Any, Dict, List, Optional, Set

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.core.document import PrismDocument
from src.core.office_exporter import ExportFormat, OfficeExporter
from src.core.sheet import DataSheet, GraphSheet, InfoSheet, ResultSheet, SheetType


class ExportProjectDialog(QDialog):
    """
    Granular Document Export Dialog allowing selective or full project exports to
    publication-grade Word (.docx), LibreOffice Writer (.odt), Excel (.xlsx),
    LibreOffice Calc (.ods), and PDF (.pdf).
    """

    FORMAT_FILTERS: Dict[str, Tuple[str, ExportFormat]] = {
        "Word Document (.docx)": ("Microsoft Word (*.docx)", ExportFormat.DOCX),
        "OpenDocument Text (.odt)": ("OpenDocument Text (*.odt)", ExportFormat.ODT),
        "Excel Workbook (.xlsx)": ("Microsoft Excel (*.xlsx)", ExportFormat.XLSX),
        "OpenDocument Spreadsheet (.ods)": ("OpenDocument Spreadsheet (*.ods)", ExportFormat.ODS),
        "Portable Document Format (.pdf)": ("PDF Documents (*.pdf)", ExportFormat.PDF),
    }

    def __init__(
        self,
        document: PrismDocument,
        graph_canvas: Optional[Any] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.document = document
        self.graph_canvas = graph_canvas
        self.setWindowTitle("Export Report / Office Document")
        self.resize(520, 560)
        self._updating_checks = False

        self._setup_ui()
        self._populate_tree()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Header description
        desc = QLabel(
            "Select items to export and choose the target document format. "
            "Exports are generated locally with zero external CLI dependencies."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Scope Checklist Tree
        grp_scope = QGroupBox("Export Scope Checklist")
        scope_layout = QVBoxLayout(grp_scope)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setAnimated(True)
        self.tree.itemChanged.connect(self._on_item_changed)
        scope_layout.addWidget(self.tree)

        # Quick selection buttons
        btn_bar = QHBoxLayout()
        btn_select_all = QPushButton("Select All")
        btn_select_all.clicked.connect(self._select_all)
        btn_bar.addWidget(btn_select_all)

        btn_deselect_all = QPushButton("Deselect All")
        btn_deselect_all.clicked.connect(self._deselect_all)
        btn_bar.addWidget(btn_deselect_all)
        btn_bar.addStretch()
        scope_layout.addLayout(btn_bar)

        layout.addWidget(grp_scope)

        # Target Format Selector
        fmt_group = QGroupBox("Target Document Format")
        fmt_layout = QFormLayout(fmt_group)

        self.combo_format = QComboBox()
        for label in self.FORMAT_FILTERS:
            self.combo_format.addItem(label)
        self.combo_format.setCurrentIndex(0)  # Default .docx
        fmt_layout.addRow("Document Format:", self.combo_format)
        layout.addWidget(fmt_group)

        # Dialog Buttons
        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.btn_export = btn_box.button(QDialogButtonBox.StandardButton.Ok)
        if self.btn_export:
            self.btn_export.setText("Export...")
        btn_box.accepted.connect(self._do_export)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _populate_tree(self) -> None:
        self._updating_checks = True
        self.tree.clear()

        # Root Item: Select All / Entire Project
        self.root_item = QTreeWidgetItem(self.tree, ["Select All / Entire Project"])
        self.root_item.setFlags(self.root_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        self.root_item.setCheckState(0, Qt.CheckState.Checked)

        # Sub-folders
        self.info_root = QTreeWidgetItem(self.root_item, ["Info Sheets"])
        self.info_root.setFlags(self.info_root.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        self.info_root.setCheckState(0, Qt.CheckState.Checked)

        self.data_root = QTreeWidgetItem(self.root_item, ["Data Tables"])
        self.data_root.setFlags(self.data_root.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        self.data_root.setCheckState(0, Qt.CheckState.Checked)

        self.results_root = QTreeWidgetItem(self.root_item, ["Analysis Results"])
        self.results_root.setFlags(self.results_root.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        self.results_root.setCheckState(0, Qt.CheckState.Checked)

        self.graphs_root = QTreeWidgetItem(self.root_item, ["Graphs"])
        self.graphs_root.setFlags(self.graphs_root.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        self.graphs_root.setCheckState(0, Qt.CheckState.Checked)

        self.layouts_root = QTreeWidgetItem(self.root_item, ["Page Layouts"])
        self.layouts_root.setFlags(self.layouts_root.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        self.layouts_root.setCheckState(0, Qt.CheckState.Checked)

        # Add leaves
        for sid in self.document.sheet_order:
            sheet = self.document.sheets.get(sid)
            if sheet is None:
                continue

            parent_folder = None
            if sheet.sheet_type == SheetType.INFO:
                parent_folder = self.info_root
            elif sheet.sheet_type == SheetType.DATA:
                parent_folder = self.data_root
            elif sheet.sheet_type == SheetType.RESULTS:
                parent_folder = self.results_root
            elif sheet.sheet_type == SheetType.GRAPH:
                parent_folder = self.graphs_root
            elif sheet.sheet_type == SheetType.LAYOUT:
                parent_folder = self.layouts_root

            if parent_folder is not None:
                leaf = QTreeWidgetItem(parent_folder, [sheet.title])
                leaf.setFlags(leaf.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                leaf.setCheckState(0, Qt.CheckState.Checked)
                leaf.setData(0, Qt.ItemDataRole.UserRole, sheet.sheet_id)

        self.tree.expandAll()
        self._updating_checks = False

    def _set_subtree_checked(self, item: QTreeWidgetItem, state: Qt.CheckState) -> None:
        item.setCheckState(0, state)
        for i in range(item.childCount()):
            self._set_subtree_checked(item.child(i), state)

    def _select_all(self) -> None:
        self._updating_checks = True
        self._set_subtree_checked(self.root_item, Qt.CheckState.Checked)
        self._updating_checks = False

    def _deselect_all(self) -> None:
        self._updating_checks = True
        self._set_subtree_checked(self.root_item, Qt.CheckState.Unchecked)
        self._updating_checks = False

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._updating_checks:
            return
        self._updating_checks = True
        state = item.checkState(0)

        # If user changed root or folder, propagate down to children
        if item.childCount() > 0:
            for i in range(item.childCount()):
                self._set_subtree_checked(item.child(i), state)

        # Update root state if any child changed
        self._update_parent_states(item)
        self._updating_checks = False

    def _update_parent_states(self, item: QTreeWidgetItem) -> None:
        parent = item.parent()
        if parent is None:
            return

        child_states = [parent.child(i).checkState(0) for i in range(parent.childCount())]
        if all(s == Qt.CheckState.Checked for s in child_states):
            parent.setCheckState(0, Qt.CheckState.Checked)
        elif all(s == Qt.CheckState.Unchecked for s in child_states):
            parent.setCheckState(0, Qt.CheckState.Unchecked)
        else:
            parent.setCheckState(0, Qt.CheckState.PartiallyChecked)

        self._update_parent_states(parent)

    def get_selected_sheet_ids(self) -> List[str]:
        """Returns list of sheet IDs that are checked in the scope tree."""
        selected: List[str] = []

        def gather_checked_leaves(item: QTreeWidgetItem) -> None:
            if item.childCount() == 0:
                if item.checkState(0) == Qt.CheckState.Checked:
                    sid = item.data(0, Qt.ItemDataRole.UserRole)
                    if sid:
                        selected.append(sid)
            else:
                for i in range(item.childCount()):
                    gather_checked_leaves(item.child(i))

        gather_checked_leaves(self.root_item)
        return selected

    def _do_export(self) -> None:
        selected_ids = self.get_selected_sheet_ids()
        if not selected_ids:
            QMessageBox.warning(
                self, "Export Scope Empty", "Please select at least one item from the checklist to export."
            )
            return

        fmt_key = self.combo_format.currentText()
        filt_str, fmt_enum = self.FORMAT_FILTERS.get(fmt_key, ("*.docx", ExportFormat.DOCX))

        default_name = f"{self.document.title or 'Export'}.{fmt_enum.value}"
        initial_path = os.path.join(os.getcwd(), default_name)

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            f"Export as {fmt_key}",
            initial_path,
            f"{filt_str};;All Files (*)",
        )

        if not file_path:
            return

        try:
            exporter = OfficeExporter(self.document, self.graph_canvas)
            exporter.export(file_path, fmt=fmt_enum, selected_sheet_ids=selected_ids)
            QMessageBox.information(
                self,
                "Export Complete",
                f"Successfully exported {len(selected_ids)} item(s) to:\n{file_path}",
            )
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"An error occurred during export:\n{str(e)}")
