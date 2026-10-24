"""
Analysis configuration dialogs for GemGraph.
Derived from BridgeCLR analysis parameter dialogs (REA-CLR-VM-102).
"""

from __future__ import annotations
from typing import List, Optional, Tuple, Dict
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QDoubleSpinBox,
    QCheckBox,
    QDialogButtonBox,
    QGroupBox,
    QFormLayout,
    QPushButton,
    QSlider,
    QColorDialog,
    QLineEdit,
    QSpinBox,
    QTreeWidget,
    QTreeWidgetItem,
    QRadioButton,
    QTabWidget,
    QWidget,
)
from src.math.nonlinear import STANDARD_MODELS, ConvergencePreset


class NonlinearRegressionDialog(QDialog):
    """Configuration dialog for Levenberg-Marquardt curve fitting."""

    def __init__(self, group_names: List[str], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Parameters: Nonlinear Regression")
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)

        # Model selection
        model_group = QGroupBox("Model Selection")
        model_form = QFormLayout(model_group)

        self.model_combo = QComboBox()
        for name, model in STANDARD_MODELS.items():
            self.model_combo.addItem(name, model)
        model_form.addRow("Model Equation:", self.model_combo)

        self.group_combo = QComboBox()
        self.group_combo.addItems(group_names)
        model_form.addRow("Data Group:", self.group_combo)

        layout.addWidget(model_group)

        # Convergence settings
        conv_group = QGroupBox("Convergence Criteria (prism.exe Presets)")
        conv_form = QFormLayout(conv_group)

        self.preset_combo = QComboBox()
        self.preset_combo.addItem("Standard (ΔSS/SS < 1e-4)", ConvergencePreset.STANDARD)
        self.preset_combo.addItem("Strict (ΔSS/SS < 1e-6)", ConvergencePreset.STRICT)
        conv_form.addRow("Preset Table:", self.preset_combo)

        self.max_iter_label = QLabel("1000 iterations (Hard limit)")
        conv_form.addRow("Iteration Limit:", self.max_iter_label)

        layout.addWidget(conv_group)

        # Button Box
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_settings(self) -> Tuple[str, int, ConvergencePreset]:
        model_name = self.model_combo.currentText()
        group_idx = self.group_combo.currentIndex()
        preset = self.preset_combo.currentData()
        return model_name, group_idx, preset


class OutlierDialog(QDialog):
    """Configuration dialog for ROUT and Grubbs ESD outlier detection."""

    def __init__(self, group_names: List[str], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Parameters: Identify Outliers")
        self.setMinimumWidth(380)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.group_combo = QComboBox()
        self.group_combo.addItems(group_names)
        form.addRow("Group to Test:", self.group_combo)

        self.method_combo = QComboBox()
        self.method_combo.addItem("ROUT (Robust fit & FDR)", "ROUT")
        self.method_combo.addItem("Grubbs ESD Test", "Grubbs")
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        form.addRow("Method:", self.method_combo)

        self.q_spin = QDoubleSpinBox()
        self.q_spin.setRange(0.01, 50.0)
        self.q_spin.setSingleStep(0.1)
        self.q_spin.setValue(1.0)
        self.q_spin.setSuffix("%")
        self.q_label = QLabel("FDR Threshold (Q):")
        form.addRow(self.q_label, self.q_spin)

        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.001, 0.20)
        self.alpha_spin.setSingleStep(0.01)
        self.alpha_spin.setValue(0.05)
        self.alpha_label = QLabel("Significance Level (α):")
        form.addRow(self.alpha_label, self.alpha_spin)
        self.alpha_spin.hide()
        self.alpha_label.hide()

        self.auto_exclude_cb = QCheckBox("Automatically exclude detected outliers in table")
        self.auto_exclude_cb.setChecked(True)
        form.addRow("", self.auto_exclude_cb)

        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_method_changed(self) -> None:
        is_rout = self.method_combo.currentData() == "ROUT"
        self.q_label.setVisible(is_rout)
        self.q_spin.setVisible(is_rout)
        self.alpha_label.setVisible(not is_rout)
        self.alpha_spin.setVisible(not is_rout)

    def get_settings(self) -> Tuple[int, str, float, bool]:
        group_idx = self.group_combo.currentIndex()
        method = self.method_combo.currentData()
        param_val = self.q_spin.value() / 100.0 if method == "ROUT" else self.alpha_spin.value()
        auto_exclude = self.auto_exclude_cb.isChecked()
        return group_idx, method, param_val, auto_exclude


class AnovaDialog(QDialog):
    """Configuration dialog for ANOVA and Post-Hoc Comparisons."""

    def __init__(self, group_names: List[str], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Parameters: One-Way ANOVA")
        self.setMinimumWidth(380)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.test_combo = QComboBox()
        self.test_combo.addItem("Ordinary One-Way ANOVA", "ordinary")
        self.test_combo.addItem("Repeated Measures ANOVA", "rm")
        form.addRow("Analysis Type:", self.test_combo)

        self.posthoc_combo = QComboBox()
        self.posthoc_combo.addItem("Dunnett (Compare all vs. Control)", "dunnett")
        self.posthoc_combo.addItem("Tukey HSD (All pairwise comparisons)", "tukey")
        self.posthoc_combo.addItem("None (ANOVA table only)", "none")
        self.posthoc_combo.currentIndexChanged.connect(self._on_posthoc_changed)
        form.addRow("Multiple Comparisons:", self.posthoc_combo)

        self.ctrl_combo = QComboBox()
        self.ctrl_combo.addItems(group_names)
        self.ctrl_label = QLabel("Control Group:")
        form.addRow(self.ctrl_label, self.ctrl_combo)

        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_posthoc_changed(self) -> None:
        is_dunnett = self.posthoc_combo.currentData() == "dunnett"
        self.ctrl_label.setVisible(is_dunnett)
        self.ctrl_combo.setVisible(is_dunnett)

    def get_settings(self) -> Tuple[str, str, int]:
        test_type = self.test_combo.currentData()
        posthoc = self.posthoc_combo.currentData()
        ctrl_idx = self.ctrl_combo.currentIndex()
        return test_type, posthoc, ctrl_idx


class FormatSeriesDialog(QDialog):
    """Dialog for customizing per-series color and opacity."""

    def __init__(
        self,
        series_names: List[str],
        series_colors: Optional[Dict[str, QColor]] = None,
        series_opacities: Optional[Dict[str, float]] = None,
        is_grayscale: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Format Series / Bars")
        self.setMinimumWidth(380)

        self.series_names = list(series_names)
        self.colors: Dict[str, QColor] = dict(series_colors or {})
        self.opacities: Dict[str, float] = dict(series_opacities or {})
        self.is_grayscale = is_grayscale

        layout = QVBoxLayout(self)

        form_group = QGroupBox("Series Appearance")
        form = QFormLayout(form_group)

        self.series_combo = QComboBox()
        self.series_combo.addItems(self.series_names)
        self.series_combo.currentIndexChanged.connect(self._on_series_selected)
        form.addRow("Select Series:", self.series_combo)

        # Color picker button with preview swatch
        color_layout = QHBoxLayout()
        self.btn_color = QPushButton("Pick Color...")
        self.btn_color.clicked.connect(self._choose_color)
        self.color_preview = QLabel("        ")
        self.color_preview.setAutoFillBackground(True)
        color_layout.addWidget(self.btn_color)
        color_layout.addWidget(self.color_preview)
        color_layout.addStretch()
        form.addRow("Fill Color:", color_layout)

        # Opacity slider 0 - 100%
        opacity_layout = QHBoxLayout()
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        default_op = 100 if is_grayscale else 80
        self.opacity_slider.setValue(default_op)
        self.opacity_label = QLabel(f"{default_op}%")
        self.opacity_slider.valueChanged.connect(self._on_opacity_slider_changed)
        opacity_layout.addWidget(self.opacity_slider)
        opacity_layout.addWidget(self.opacity_label)
        form.addRow("Opacity:", opacity_layout)

        layout.addWidget(form_group)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        if self.series_names:
            self._on_series_selected(0)

    def _current_series(self) -> str:
        return self.series_combo.currentText()

    def _on_series_selected(self, index: int) -> None:
        name = self._current_series()
        col = self.colors.get(name, QColor("#000000" if self.is_grayscale else "#0072B2"))
        default_op = 1.0 if self.is_grayscale else 0.8
        op_val = self.opacities.get(name, default_op)
        op_pct = int(round(op_val * 100))
        self._update_color_preview(col)
        self.opacity_slider.blockSignals(True)
        self.opacity_slider.setValue(op_pct)
        self.opacity_slider.blockSignals(False)
        self.opacity_label.setText(f"{op_pct}%")

    def _choose_color(self) -> None:
        name = self._current_series()
        initial = self.colors.get(name, QColor("#000000" if self.is_grayscale else "#0072B2"))
        color = QColorDialog.getColor(initial, self, f"Select Color for {name}")
        if color.isValid():
            self.colors[name] = color
            self._update_color_preview(color)

    def _update_color_preview(self, color: QColor) -> None:
        self.color_preview.setStyleSheet(
            f"background-color: {color.name()}; border: 1px solid #333333; min-width: 40px; min-height: 20px;"
        )

    def _on_opacity_slider_changed(self, value: int) -> None:
        name = self._current_series()
        self.opacities[name] = value / 100.0
        self.opacity_label.setText(f"{value}%")

    def get_formatting(self) -> Tuple[Dict[str, QColor], Dict[str, float]]:
        return self.colors, self.opacities


class NewLayoutDialog(QDialog):
    """Configuration dialog for creating a new presentation Page Layout."""

    def __init__(self, parent=None, initial_name: str = "Layout 1") -> None:
        super().__init__(parent)
        self.setWindowTitle("Create New Page Layout")
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)

        form_group = QGroupBox("Layout Setup & Grid Template")
        form = QFormLayout(form_group)

        self.name_edit = QLineEdit(initial_name)
        form.addRow("Layout Name:", self.name_edit)

        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            "4 (2x2 Quadrants)",
            "1 Graph",
            "2 Horizontal",
            "2 Vertical",
            "3 (1 Top / 2 Bottom)",
            "6 (2x3)",
            "8 (2x4)",
            "Custom Grid",
        ])
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        form.addRow("Template Preset:", self.preset_combo)

        # Custom grid row/col spinboxes
        grid_row_layout = QHBoxLayout()
        self.spin_rows = QSpinBox()
        self.spin_rows.setRange(1, 10)
        self.spin_rows.setValue(2)
        self.spin_rows.setEnabled(False)
        grid_row_layout.addWidget(QLabel("Rows:"))
        grid_row_layout.addWidget(self.spin_rows)

        self.spin_cols = QSpinBox()
        self.spin_cols.setRange(1, 10)
        self.spin_cols.setValue(2)
        self.spin_cols.setEnabled(False)
        grid_row_layout.addWidget(QLabel("Columns:"))
        grid_row_layout.addWidget(self.spin_cols)
        form.addRow("Custom Dimensions:", grid_row_layout)

        self.format_combo = QComboBox()
        self.format_combo.addItems(["A4", "Letter", "Slide 16:9", "Slide 4:3"])
        form.addRow("Page Format:", self.format_combo)

        self.orientation_combo = QComboBox()
        self.orientation_combo.addItems(["Portrait", "Landscape"])
        form.addRow("Orientation:", self.orientation_combo)

        layout.addWidget(form_group)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_preset_changed(self, idx: int) -> None:
        is_custom = self.preset_combo.currentText() == "Custom Grid"
        self.spin_rows.setEnabled(is_custom)
        self.spin_cols.setEnabled(is_custom)

    def get_settings(self) -> Dict[str, Any]:
        return {
            "name": self.name_edit.text().strip() or "Layout",
            "template": self.preset_combo.currentText(),
            "page_format": self.format_combo.currentText(),
            "orientation": self.orientation_combo.currentText(),
            "rows": self.spin_rows.value(),
            "cols": self.spin_cols.value(),
        }


class AssignGraphDialog(QDialog):
    """
    Dialog allowing user to link an available project graph or pinned snapshot to a layout placeholder,
    and configure scaling and rotation (Normal, 90° clockwise Depth chart, 90° counter-clockwise).
    Matches GraphPad Prism 'Place Graph on Layout' dialog.
    """

    def __init__(
        self,
        document: Any,
        current_graph_id: Optional[str] = None,
        current_rotation: int = 0,
        current_scale: float = 1.0,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.document = document
        self.selected_graph_id: Optional[str] = current_graph_id
        self.selected_graph_title: Optional[str] = None

        self.setWindowTitle("Place Graph on Layout")
        self.resize(500, 440)

        main_layout = QVBoxLayout(self)

        self.tab_widget = QTabWidget(self)

        # Tab 1: Assign Graph
        tab_assign = QWidget(self)
        assign_layout = QVBoxLayout(tab_assign)
        assign_layout.addWidget(QLabel("Select a graph or snapshot to display in this placeholder slot:"))

        self.tree = QTreeWidget(tab_assign)
        self.tree.setHeaderLabels(["Graph / Snapshot Title", "Type"])
        self.tree.setAnimated(True)
        self.tree.itemSelectionChanged.connect(self._on_selection_changed)
        self.tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        assign_layout.addWidget(self.tree)

        self.tab_widget.addTab(tab_assign, "Assign Graph")

        # Tab 2: Size and Location
        tab_size = QWidget(self)
        size_layout = QVBoxLayout(tab_size)

        grp_scale = QGroupBox("Scaling Factor", tab_size)
        scale_form = QFormLayout(grp_scale)
        self.spin_scale = QSpinBox(grp_scale)
        self.spin_scale.setRange(20, 500)
        self.spin_scale.setSingleStep(5)
        self.spin_scale.setSuffix(" %")
        norm_scale = float(current_scale)
        if norm_scale > 10.0:  # If 100 was passed instead of 1.0
            norm_scale = norm_scale / 100.0
        self.spin_scale.setValue(int(round(max(0.2, min(5.0, norm_scale)) * 100)))
        scale_form.addRow("Scaling Factor:", self.spin_scale)
        size_layout.addWidget(grp_scale)

        grp_rot = QGroupBox("Rotation", tab_size)
        rot_vbox = QVBoxLayout(grp_rot)
        self.radio_rot_0 = QRadioButton("Normal (0°)", grp_rot)
        self.radio_rot_90 = QRadioButton("90 degrees clockwise (depth chart)", grp_rot)
        self.radio_rot_270 = QRadioButton("90 degrees counter clockwise", grp_rot)
        rot_vbox.addWidget(self.radio_rot_0)
        rot_vbox.addWidget(self.radio_rot_90)
        rot_vbox.addWidget(self.radio_rot_270)

        norm_rot = current_rotation % 360
        if norm_rot == 90:
            self.radio_rot_90.setChecked(True)
        elif norm_rot in (270, -90):
            self.radio_rot_270.setChecked(True)
        else:
            self.radio_rot_0.setChecked(True)

        size_layout.addWidget(grp_rot)
        size_layout.addStretch()

        self.tab_widget.addTab(tab_size, "Size and Location")
        main_layout.addWidget(self.tab_widget)

        self._populate_graphs()

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.btn_ok = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.btn_ok.setText("Assign Graph")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        main_layout.addWidget(buttons)

    def _populate_graphs(self) -> None:
        self.tree.clear()

        # Unassign option
        unassign_item = QTreeWidgetItem(self.tree, ["(None - Leave Empty)", "Placeholder"])
        unassign_item.setData(0, Qt.ItemDataRole.UserRole, "")

        graphs_root = QTreeWidgetItem(self.tree, ["📈 Active Graphs", ""])
        pinned_root = QTreeWidgetItem(self.tree, ["📌 Pinned Snapshots", ""])

        for sid in getattr(self.document, "sheet_order", []):
            sheet = self.document.sheets.get(sid)
            if sheet is None:
                continue

            from src.core.sheet import SheetType, GraphSheet
            if sheet.sheet_type == SheetType.GRAPH and isinstance(sheet, GraphSheet):
                gt_val = sheet.graph_type.value if hasattr(sheet.graph_type, "value") else str(sheet.graph_type)
                clean_title = sheet.title.replace("❄️", "").replace("🔥", "").strip()
                if sheet.is_pinned:
                    parent_node = pinned_root
                    if getattr(sheet, "is_frozen", False):
                        display_title = f"❄️  {clean_title}"
                    else:
                        display_title = f"🔥  {clean_title}"
                else:
                    parent_node = graphs_root
                    if getattr(sheet, "is_frozen", False):
                        display_title = f"❄️  {clean_title}"
                    else:
                        display_title = clean_title

                item = QTreeWidgetItem(parent_node, [display_title, gt_val])
                item.setData(0, Qt.ItemDataRole.UserRole, sheet.sheet_id)
                item.setData(0, Qt.ItemDataRole.UserRole + 1, sheet.title)
                if sheet.sheet_id == self.selected_graph_id:
                    self.tree.setCurrentItem(item)

        self.tree.expandAll()

    def _on_item_double_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        sid = item.data(0, Qt.ItemDataRole.UserRole)
        if sid is not None:
            self._on_selection_changed()
            self.accept()

    def _on_selection_changed(self) -> None:
        item = self.tree.currentItem()
        if not item:
            return
        sid = item.data(0, Qt.ItemDataRole.UserRole)
        if sid == "":
            self.selected_graph_id = None
            self.selected_graph_title = None
        elif sid:
            self.selected_graph_id = sid
            # Return sheet title or display title
            raw_title = item.data(0, Qt.ItemDataRole.UserRole + 1)
            self.selected_graph_title = raw_title if raw_title else item.text(0)

    def get_selected_graph(self) -> Tuple[Optional[str], Optional[str]]:
        return self.selected_graph_id, self.selected_graph_title

    def get_selected_rotation(self) -> int:
        if self.radio_rot_90.isChecked():
            return 90
        elif self.radio_rot_270.isChecked():
            return 270
        return 0

    def get_selected_scale(self) -> float:
        val = float(self.spin_scale.value()) / 100.0
        return max(0.2, min(5.0, val))


class PageSetupDialog(QDialog):
    """Dialog allowing user to configure page format and orientation for a PageLayout."""

    def __init__(self, current_format: str = "A4", current_orientation: str = "Portrait", parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Page Setup")
        self.resize(320, 160)
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self.combo_format = QComboBox()
        self.combo_format.addItems(["A4", "Letter", "Slide 16:9", "Slide 4:3"])
        idx = self.combo_format.findText(current_format)
        if idx >= 0:
            self.combo_format.setCurrentIndex(idx)
        form.addRow("Page Format:", self.combo_format)

        self.combo_orientation = QComboBox()
        self.combo_orientation.addItems(["Portrait", "Landscape"])
        idx_o = self.combo_orientation.findText(current_orientation, Qt.MatchFlag.MatchFixedString)
        if idx_o >= 0:
            self.combo_orientation.setCurrentIndex(idx_o)
        form.addRow("Orientation:", self.combo_orientation)
        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_settings(self) -> Tuple[str, str]:
        return self.combo_format.currentText(), self.combo_orientation.currentText()


class EnzymeKineticsDialog(QDialog):
    """Configuration dialog for Enzyme Kinetics nonlinear fitting, F-test model comparison, and diagnostic plots."""

    def __init__(self, group_names: List[str], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Parameters: Enzyme Kinetics")
        self.setMinimumWidth(440)

        layout = QVBoxLayout(self)

        model_group = QGroupBox("Model Equation & Data")
        form = QFormLayout(model_group)

        self.model_combo = QComboBox()
        enzyme_models = [
            "Michaelis-Menten",
            "Substrate Inhibition",
            "Allosteric Sigmoidal (Hill Kinetics)",
            "Competitive Inhibition",
        ]
        self.model_combo.addItems(enzyme_models)
        form.addRow("Kinetic Model:", self.model_combo)

        self.group_combo = QComboBox()
        self.group_combo.addItems(group_names)
        form.addRow("Velocity Group (Y):", self.group_combo)

        self.preset_combo = QComboBox()
        self.preset_combo.addItem("Standard (1e-4)", ConvergencePreset.STANDARD)
        self.preset_combo.addItem("Strict (1e-6)", ConvergencePreset.STRICT)
        form.addRow("Convergence Tolerance:", self.preset_combo)

        layout.addWidget(model_group)

        comp_group = QGroupBox("Hypothesis Testing & Diagnostics")
        comp_layout = QVBoxLayout(comp_group)

        self.cb_compare_mm = QCheckBox("Compare with simple Michaelis-Menten (Extra-SS F-test)")
        self.cb_compare_mm.setChecked(True)
        comp_layout.addWidget(self.cb_compare_mm)

        self.cb_diagnostic = QCheckBox("Show Lineweaver-Burk / Eadie-Hofstee diagnostic plot")
        self.cb_diagnostic.setChecked(True)
        comp_layout.addWidget(self.cb_diagnostic)

        diag_form = QFormLayout()
        self.diag_combo = QComboBox()
        self.diag_combo.addItems(["Lineweaver-Burk (1/X, 1/Y)", "Eadie-Hofstee (Y/X, Y)"])
        diag_form.addRow("Diagnostic Transform:", self.diag_combo)
        comp_layout.addLayout(diag_form)

        layout.addWidget(comp_group)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_settings(self) -> Tuple[str, int, bool, bool, str, ConvergencePreset]:
        return (
            self.model_combo.currentText(),
            self.group_combo.currentIndex(),
            self.cb_compare_mm.isChecked(),
            self.cb_diagnostic.isChecked(),
            self.diag_combo.currentText(),
            self.preset_combo.currentData(),
        )


class CoxRegressionDialog(QDialog):
    """Dialog allowing user to select time, event, and covariate columns for Cox Proportional Hazards regression."""

    def __init__(
        self,
        column_names: List[str],
        default_time_idx: int = 0,
        default_event_idx: int = 1,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Parameters: Cox Proportional Hazards")
        self.setMinimumWidth(440)
        self.column_names = column_names

        layout = QVBoxLayout(self)

        col_group = QGroupBox("Survival Variables")
        form = QFormLayout(col_group)

        self.combo_time = QComboBox()
        self.combo_time.addItems(column_names)
        if 0 <= default_time_idx < len(column_names):
            self.combo_time.setCurrentIndex(default_time_idx)
        form.addRow("Time Variable (X):", self.combo_time)

        self.combo_event = QComboBox()
        self.combo_event.addItems(column_names)
        if 0 <= default_event_idx < len(column_names):
            self.combo_event.setCurrentIndex(default_event_idx)
        form.addRow("Event/Outcome (1=Event, 0=Censored):", self.combo_event)

        layout.addWidget(col_group)

        cov_group = QGroupBox("Covariates (Predictors)")
        cov_layout = QVBoxLayout(cov_group)
        self.cov_checkboxes: List[QCheckBox] = []

        for i, col_name in enumerate(column_names):
            cb = QCheckBox(col_name)
            if i != default_time_idx and i != default_event_idx:
                cb.setChecked(True)
            self.cov_checkboxes.append(cb)
            cov_layout.addWidget(cb)

        layout.addWidget(cov_group)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_selection(self) -> Tuple[int, int, List[int]]:
        time_idx = self.combo_time.currentIndex()
        event_idx = self.combo_event.currentIndex()
        selected_covs = [
            i for i, cb in enumerate(self.cov_checkboxes)
            if cb.isChecked() and i != time_idx and i != event_idx
        ]
        return time_idx, event_idx, selected_covs

