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
