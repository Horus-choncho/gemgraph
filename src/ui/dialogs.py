"""
Analysis configuration dialogs for GemGraph.
Derived from BridgeCLR analysis parameter dialogs (REA-CLR-VM-102).
"""

from __future__ import annotations
from typing import List, Optional, Tuple
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
