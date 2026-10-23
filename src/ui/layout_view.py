"""
Presentation Page Layout View Widget for GemGraph.
Integrates LayoutCanvas with manipulation toolbar for template presets, alignment tools,
dimension equalization, page setup controls, and high-DPI export actions.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.core.layout import (
    PageLayout,
    Placeholder,
    align_placeholders,
    create_template_placeholders,
    equalize_placeholders,
    render_layout_to_pdf,
    render_layout_to_png_bytes,
    render_layout_to_svg,
)
from src.ui.layout_canvas import LayoutCanvas


class LayoutView(QWidget):
    """
    Center pane widget hosting presentation layout canvas and manipulation toolbar.
    """

    layoutModified = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None, document: Optional[Any] = None) -> None:
        super().__init__(parent)
        self.current_layout: Optional[PageLayout] = None
        self.current_document: Any = document

        self._setup_ui()

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Manipulation & Alignment Toolbar
        self.toolbar = QToolBar("Layout Tools", self)
        self.toolbar.setMovable(False)

        # 1. Template Presets Menu
        self.btn_templates = QToolButton(self)
        self.btn_templates.setText("📐 Template Presets")
        self.btn_templates.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.template_menu = QMenu(self)

        presets = [
            ("1 Graph", "1 Graph"),
            ("2 Horizontal", "2 Horizontal"),
            ("2 Vertical", "2 Vertical"),
            ("3 (1 Top / 2 Bottom)", "3 (1 Top / 2 Bottom)"),
            ("4 (2x2 Quadrants)", "4 (2x2 Quadrants)"),
            ("6 (2x3)", "6 (2x3)"),
            ("8 (2x4)", "8 (2x4)"),
            ("Custom Grid...", "Custom Grid"),
        ]
        for label, preset_id in presets:
            act = self.template_menu.addAction(label)
            act.triggered.connect(lambda checked, p=preset_id: self._apply_template_preset(p))

        self.btn_templates.setMenu(self.template_menu)
        self.toolbar.addWidget(self.btn_templates)

        self.toolbar.addSeparator()

        # 2. Alignment Actions
        self.btn_align = QToolButton(self)
        self.btn_align.setText("⫸ Align")
        self.btn_align.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.align_menu = QMenu(self)

        alignments = [
            ("Align Lefts", "left"),
            ("Align Centers", "center"),
            ("Align Rights", "right"),
            ("Align Tops", "top"),
            ("Align Middles", "middle"),
            ("Align Bottoms", "bottom"),
        ]
        for label, align_type in alignments:
            act = self.align_menu.addAction(label)
            act.triggered.connect(lambda checked, a=align_type: self._apply_alignment(a))

        self.btn_align.setMenu(self.align_menu)
        self.toolbar.addWidget(self.btn_align)

        # 3. Equalize Actions
        self.btn_equalize = QToolButton(self)
        self.btn_equalize.setText("⚖️ Equalize")
        self.btn_equalize.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.equalize_menu = QMenu(self)

        equalizes = [
            ("Equalize Widths", "width"),
            ("Equalize Heights", "height"),
            ("Equalize Both", "both"),
        ]
        for label, eq_type in equalizes:
            act = self.equalize_menu.addAction(label)
            act.triggered.connect(lambda checked, m=eq_type: self._apply_equalize(m))

        self.btn_equalize.setMenu(self.equalize_menu)
        self.toolbar.addWidget(self.btn_equalize)

        self.toolbar.addSeparator()

        # 4. Page Setup Controls
        self.toolbar.addWidget(QLabel(" Format: "))
        self.combo_format = QComboBox(self)
        self.combo_format.addItems(["A4", "Letter", "Slide 16:9", "Slide 4:3"])
        self.combo_format.currentIndexChanged.connect(self._on_format_changed)
        self.toolbar.addWidget(self.combo_format)

        self.toolbar.addWidget(QLabel(" Orientation: "))
        self.combo_orientation = QComboBox(self)
        self.combo_orientation.addItems(["Portrait", "Landscape"])
        self.combo_orientation.currentIndexChanged.connect(self._on_orientation_changed)
        self.toolbar.addWidget(self.combo_orientation)

        self.toolbar.addSeparator()

        # 5. Fit & Refresh Actions
        self.btn_fit = QPushButton("🔍 Fit Page", self)
        self.btn_fit.clicked.connect(self._fit_page)
        self.toolbar.addWidget(self.btn_fit)

        self.btn_refresh = QPushButton("🔄 Refresh", self)
        self.btn_refresh.clicked.connect(self.refresh_all_snapshots)
        self.toolbar.addWidget(self.btn_refresh)

        self.toolbar.addSeparator()

        # 6. Export Layout Button
        self.btn_export = QPushButton("📤 Export Layout...", self)
        self.btn_export.clicked.connect(self._export_layout_dialog)
        self.toolbar.addWidget(self.btn_export)

        main_layout.addWidget(self.toolbar)

        # Layout Presentation Canvas
        self.canvas = LayoutCanvas(self)
        self.canvas.placeholderAssigned.connect(lambda slot: self.layoutModified.emit())
        self.canvas.layoutModifiedSignal.connect(self.layoutModified.emit)
        main_layout.addWidget(self.canvas)

    def set_layout(self, layout: PageLayout, document: Any = None) -> None:
        """Sets the active PageLayout and updates toolbar controls and canvas."""
        self.current_layout = layout
        if document is not None:
            self.current_document = document

        # Update controls without firing triggers
        self.combo_format.blockSignals(True)
        idx = self.combo_format.findText(layout.page_format)
        if idx >= 0:
            self.combo_format.setCurrentIndex(idx)
        self.combo_format.blockSignals(False)

        self.combo_orientation.blockSignals(True)
        idx = self.combo_orientation.findText(layout.orientation)
        if idx >= 0:
            self.combo_orientation.setCurrentIndex(idx)
        self.combo_orientation.blockSignals(False)

        self.canvas.set_layout(layout, document)

    def refresh_all_snapshots(self) -> None:
        """Refreshes all placeholder snapshots to reflect modified data or plots."""
        self.canvas.refresh_snapshots()

    def _apply_template_preset(self, preset: str) -> None:
        if self.current_layout is None:
            return

        if preset == "Custom Grid":
            rows, ok_r = QInputDialog.getInt(self, "Custom Grid", "Number of Rows:", 2, 1, 10)
            if not ok_r:
                return
            cols, ok_c = QInputDialog.getInt(self, "Custom Grid", "Number of Columns:", 2, 1, 10)
            if not ok_c:
                return
            new_placeholders = create_template_placeholders("Custom Grid", rows=rows, cols=cols)
        else:
            new_placeholders = create_template_placeholders(preset)

        # Transfer existing assigned graph IDs if available
        for idx, new_p in enumerate(new_placeholders):
            if idx < len(self.current_layout.placeholders):
                old_p = self.current_layout.placeholders[idx]
                if old_p.graph_id:
                    new_p.assign_graph(old_p.graph_id, old_p.linked_graph_title)

        self.current_layout.placeholders = new_placeholders
        self.canvas.rebuild_scene()
        self.layoutModified.emit()

    def _apply_alignment(self, align_type: str) -> None:
        selected = self.canvas.get_selected_placeholders()
        if len(selected) < 2:
            QMessageBox.information(
                self,
                "Alignment",
                "Please select 2 or more placeholders on the canvas to align.",
            )
            return

        selected_slot_ids = {p.slot_id for p in selected}
        align_placeholders(selected, align_type)
        self.canvas.rebuild_scene()
        for item in self.canvas.placeholder_items:
            if item.placeholder.slot_id in selected_slot_ids:
                item.setSelected(True)
        self.layoutModified.emit()

    def _apply_equalize(self, mode: str) -> None:
        selected = self.canvas.get_selected_placeholders()
        if len(selected) < 2:
            QMessageBox.information(
                self,
                "Equalize",
                "Please select 2 or more placeholders on the canvas to equalize dimensions.",
            )
            return

        selected_slot_ids = {p.slot_id for p in selected}
        equalize_placeholders(selected, mode)
        self.canvas.rebuild_scene()
        for item in self.canvas.placeholder_items:
            if item.placeholder.slot_id in selected_slot_ids:
                item.setSelected(True)
        self.layoutModified.emit()

    def _on_format_changed(self) -> None:
        if self.current_layout is None:
            return
        self.current_layout.page_format = self.combo_format.currentText()
        self.canvas.rebuild_scene()
        self.layoutModified.emit()

    def _on_orientation_changed(self) -> None:
        if self.current_layout is None:
            return
        self.current_layout.orientation = self.combo_orientation.currentText()
        self.canvas.rebuild_scene()
        self.layoutModified.emit()

    def set_template(self, preset: str) -> None:
        """Applies template preset to current layout."""
        self._apply_template_preset(preset)

    def set_orientation(self, orientation: str) -> None:
        """Sets layout orientation ('portrait' or 'landscape')."""
        if self.current_layout is None:
            return
        self.current_layout.orientation = orientation
        idx = self.combo_orientation.findText(orientation, Qt.MatchFlag.MatchFixedString)
        if idx >= 0:
            self.combo_orientation.setCurrentIndex(idx)
        else:
            self.canvas.rebuild_scene()
            self.layoutModified.emit()

    def align_selected(self, align_type: str) -> None:
        """Aligns selected placeholders."""
        self._apply_alignment(align_type)

    def equalize_selected(self, mode: str) -> None:
        """Equalizes dimensions of selected placeholders."""
        self._apply_equalize(mode)

    def _fit_page(self) -> None:
        self.canvas.fit_page_to_window()

    def _export_layout_dialog(self) -> None:
        """Exports the active page layout to PNG, PDF, or SVG."""
        if self.current_layout is None:
            return

        filters = "PNG Image (*.png);;PDF Document (*.pdf);;Scalable Vector Graphics (*.svg);;TIFF Image (*.tif)"
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Page Layout",
            f"{self.current_layout.name}.png",
            filters,
        )
        if not file_path:
            return

        ext = file_path.split(".")[-1].lower()
        try:
            if ext == "pdf":
                render_layout_to_pdf(self.current_layout, document=self.current_document, file_path=file_path)
            elif ext == "svg":
                render_layout_to_svg(self.current_layout, document=self.current_document, file_path=file_path)
            else:
                dpi = 300
                dpi_choices = ["150 DPI (Screen / Draft)", "300 DPI (Standard Journal Submission)", "600 DPI (High-Resolution Print)"]
                chosen, ok = QInputDialog.getItem(
                    self,
                    "Resolution (DPI)",
                    "Select Output Resolution:",
                    dpi_choices,
                    1,
                    False,
                )
                if ok and chosen:
                    if "150" in chosen:
                        dpi = 150
                    elif "600" in chosen:
                        dpi = 600
                    else:
                        dpi = 300

                from src.core.layout import render_layout_to_image
                img = render_layout_to_image(self.current_layout, document=self.current_document, dpi=dpi)
                fmt_tag = "TIFF" if ext in ("tif", "tiff") else "PNG"
                img.save(file_path, fmt_tag)

            QMessageBox.information(self, "Export Complete", f"Successfully exported layout to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Failed to export layout:\n{e}")
