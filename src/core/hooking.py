"""
Constant Hooking Engine for GemGraph.
Central parameter registry allowing computed analysis constants
(Bmax, Km, EC50, Slope, R2, Mean, SD, Mean ± 2SD) to dynamically bind
to graph lines, dynamic text labels, and downstream analysis parameters.
"""

from __future__ import annotations
import enum
from typing import Optional, List, Dict, Any, Callable
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QTreeWidget,
    QTreeWidgetItem,
    QLabel,
    QPushButton,
    QDialogButtonBox,
    QLineEdit,
    QHeaderView,
)
from src.core.sheet import ResultSheet
from src.core.table import DataTable


class BindingTarget(enum.Enum):
    TICK_LINE = "tick_line"
    TEXT_LABEL = "text_label"
    VALUE_INPUT = "value_input"


class HookedConstant:
    """Represents a computed constant registered by an analysis or data table."""

    def __init__(
        self,
        constant_id: str,
        sheet_id: str,
        sheet_title: str,
        name: str,
        value: float,
        formatted_value: Optional[str] = None,
        description: str = "",
        category: str = "Parameters",
    ) -> None:
        self.constant_id = constant_id
        self.sheet_id = sheet_id
        self.sheet_title = sheet_title
        self.name = name
        self.value = float(value)
        self.formatted_value = formatted_value or f"{self.value:.4g}"
        self.description = description
        self.category = category

    def to_dict(self) -> Dict[str, Any]:
        return {
            "constant_id": self.constant_id,
            "sheet_id": self.sheet_id,
            "sheet_title": self.sheet_title,
            "name": self.name,
            "value": self.value,
            "formatted_value": self.formatted_value,
            "description": self.description,
            "category": self.category,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> HookedConstant:
        return cls(
            constant_id=data["constant_id"],
            sheet_id=data.get("sheet_id", ""),
            sheet_title=data.get("sheet_title", ""),
            name=data.get("name", ""),
            value=data.get("value", 0.0),
            formatted_value=data.get("formatted_value"),
            description=data.get("description", ""),
            category=data.get("category", "Parameters"),
        )


class ConstantRegistry:
    """Central registry holding all hookable constants in a project."""

    def __init__(self) -> None:
        self._constants: Dict[str, HookedConstant] = {}

    def register(self, constant: HookedConstant) -> None:
        self._constants[constant.constant_id] = constant

    def get(self, constant_id: str) -> Optional[HookedConstant]:
        return self._constants.get(constant_id)

    def get_all(self) -> List[HookedConstant]:
        return list(self._constants.values())

    def get_by_sheet(self, sheet_id: str) -> List[HookedConstant]:
        return [c for c in self._constants.values() if c.sheet_id == sheet_id]

    def remove_by_sheet(self, sheet_id: str) -> None:
        self._constants = {
            cid: c for cid, c in self._constants.items() if c.sheet_id != sheet_id
        }

    def clear(self) -> None:
        self._constants.clear()

    def extract_from_table(self, table: DataTable) -> List[HookedConstant]:
        """Extracts Mean, SD, and Mean ± 2SD QC constants for each group."""
        extracted: List[HookedConstant] = []
        for g_idx, g_title in enumerate(table.y_group_titles):
            summary = table.get_group_replicate_summary(g_idx)
            means = summary["mean"][~np.isnan(summary["mean"])] if "mean" in summary else []
            sds = summary["sd"][~np.isnan(summary["sd"])] if "sd" in summary else []

            # Overall group mean & SD
            vals = []
            for r in range(table.num_rows):
                for s in range(table.num_replicates):
                    if not table.is_excluded(r, "Y", g_idx, s):
                        v = table.get_cell_value(r, "Y", g_idx, s)
                        if not np.isnan(v):
                            vals.append(v)
            if vals:
                mean_val = float(np.mean(vals))
                sd_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

                c_mean = HookedConstant(
                    constant_id=f"{table.table_id}_{g_idx}_mean",
                    sheet_id=table.table_id,
                    sheet_title=table.title,
                    name=f"{g_title} Mean",
                    value=mean_val,
                    description=f"Arithmetic mean of {g_title}",
                    category="Descriptive Statistics",
                )
                self.register(c_mean)
                extracted.append(c_mean)

                c_sd = HookedConstant(
                    constant_id=f"{table.table_id}_{g_idx}_sd",
                    sheet_id=table.table_id,
                    sheet_title=table.title,
                    name=f"{g_title} SD",
                    value=sd_val,
                    description=f"Standard deviation of {g_title}",
                    category="Descriptive Statistics",
                )
                self.register(c_sd)
                extracted.append(c_sd)

                # Mean + 2SD (QC Upper Limit)
                c_qc_up = HookedConstant(
                    constant_id=f"{table.table_id}_{g_idx}_mean_plus_2sd",
                    sheet_id=table.table_id,
                    sheet_title=table.title,
                    name=f"{g_title} Mean + 2SD",
                    value=mean_val + 2 * sd_val,
                    description=f"Upper QC threshold (Mean + 2SD) for {g_title}",
                    category="QC Lines",
                )
                self.register(c_qc_up)
                extracted.append(c_qc_up)

                # Mean - 2SD (QC Lower Limit)
                c_qc_dn = HookedConstant(
                    constant_id=f"{table.table_id}_{g_idx}_mean_minus_2sd",
                    sheet_id=table.table_id,
                    sheet_title=table.title,
                    name=f"{g_title} Mean - 2SD",
                    value=mean_val - 2 * sd_val,
                    description=f"Lower QC threshold (Mean - 2SD) for {g_title}",
                    category="QC Lines",
                )
                self.register(c_qc_dn)
                extracted.append(c_qc_dn)

        return extracted

    def extract_from_result(self, result: ResultSheet) -> List[HookedConstant]:
        """Extracts parameters, R^2, IC50/EC50, Slope, etc. from ResultSheet."""
        extracted: List[HookedConstant] = []
        raw = result.raw_result_data

        # Best-fit parameters
        params = raw.get("params", {})
        for p_name, p_val in params.items():
            if isinstance(p_val, (int, float)):
                cid = f"{result.sheet_id}_{p_name.lower()}"
                hc = HookedConstant(
                    constant_id=cid,
                    sheet_id=result.sheet_id,
                    sheet_title=result.title,
                    name=p_name,
                    value=float(p_val),
                    description=f"Fitted parameter {p_name} from {result.title}",
                    category="Best-fit Values",
                )
                self.register(hc)
                extracted.append(hc)

                # If parameter is LogEC50 or LogIC50, also register antilog EC50
                if "logec50" in p_name.lower() or "logic50" in p_name.lower():
                    ec50_val = 10.0 ** float(p_val)
                    ec50_name = "EC50" if "ec50" in p_name.lower() else "IC50"
                    hc_ec50 = HookedConstant(
                        constant_id=f"{result.sheet_id}_{ec50_name.lower()}",
                        sheet_id=result.sheet_id,
                        sheet_title=result.title,
                        name=ec50_name,
                        value=ec50_val,
                        description=f"Antilog of {p_name} ({ec50_name})",
                        category="Best-fit Values",
                    )
                    self.register(hc_ec50)
                    extracted.append(hc_ec50)

        # Goodness of fit (R2, Sy.x, AICc)
        goodness = raw.get("goodness_of_fit", {})
        for m_name, m_val in goodness.items():
            if isinstance(m_val, (int, float)):
                cid = f"{result.sheet_id}_{m_name.lower()}"
                hc = HookedConstant(
                    constant_id=cid,
                    sheet_id=result.sheet_id,
                    sheet_title=result.title,
                    name=m_name,
                    value=float(m_val),
                    description=f"Goodness of fit metric {m_name}",
                    category="Goodness of Fit",
                )
                self.register(hc)
                extracted.append(hc)

        return extracted


# Global singleton registry instance
GLOBAL_REGISTRY = ConstantRegistry()


class HookConstantDialog(QDialog):
    """
    Hook Constant Dialog triggered by 's' key or Hook button.
    Allows user to select a constant to dynamically link to tick lines,
    text equations, or dialog inputs.
    """

    def __init__(
        self,
        registry: Optional[ConstantRegistry] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.registry = registry or GLOBAL_REGISTRY
        self.setWindowTitle("Hook a Constant (s)")
        self.resize(520, 420)

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Select an analysis parameter or table constant to hook:"))

        # Filter bar
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter constants...")
        self.filter_edit.textChanged.connect(self._filter_tree)
        layout.addWidget(self.filter_edit)

        # Hierarchical tree: Sheet -> Category -> Constant
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Constant", "Value", "Description"])
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.header().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.tree.itemDoubleClicked.connect(self._on_double_click)
        layout.addWidget(self.tree)

        self._populate_tree()

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.selected_constant: Optional[HookedConstant] = None

    def _populate_tree(self) -> None:
        self.tree.clear()
        constants = self.registry.get_all()

        # Group by sheet_title -> category
        grouped: Dict[str, Dict[str, List[HookedConstant]]] = {}
        for c in constants:
            sheet_title = c.sheet_title or "Global Constants"
            if sheet_title not in grouped:
                grouped[sheet_title] = {}
            if c.category not in grouped[sheet_title]:
                grouped[sheet_title][c.category] = []
            grouped[sheet_title][c.category].append(c)

        for sheet_title, cats in grouped.items():
            sheet_item = QTreeWidgetItem([sheet_title, "", ""])
            font = sheet_item.font(0)
            font.setBold(True)
            sheet_item.setFont(0, font)

            for cat, const_list in cats.items():
                cat_item = QTreeWidgetItem([cat, "", ""])
                for c in const_list:
                    c_item = QTreeWidgetItem([c.name, c.formatted_value, c.description])
                    c_item.setData(0, Qt.ItemDataRole.UserRole, c)
                    cat_item.addChild(c_item)
                sheet_item.addChild(cat_item)

            self.tree.addTopLevelItem(sheet_item)

        self.tree.expandAll()

    def _filter_tree(self, text: str) -> None:
        query = text.strip().lower()
        for i in range(self.tree.topLevelItemCount()):
            sheet_item = self.tree.topLevelItem(i)
            sheet_visible = False
            for j in range(sheet_item.childCount()):
                cat_item = sheet_item.child(j)
                cat_visible = False
                for k in range(cat_item.childCount()):
                    c_item = cat_item.child(k)
                    matches = (
                        query in c_item.text(0).lower()
                        or query in c_item.text(1).lower()
                        or query in c_item.text(2).lower()
                    )
                    c_item.setHidden(not matches)
                    if matches:
                        cat_visible = True
                cat_item.setHidden(not cat_visible)
                if cat_visible:
                    sheet_visible = True
            sheet_item.setHidden(not sheet_visible)

    def _on_double_click(self, item: QTreeWidgetItem, column: int) -> None:
        c = item.data(0, Qt.ItemDataRole.UserRole)
        if isinstance(c, HookedConstant):
            self.selected_constant = c
            self.accept()

    def _on_accept(self) -> None:
        current = self.tree.currentItem()
        if current:
            c = current.data(0, Qt.ItemDataRole.UserRole)
            if isinstance(c, HookedConstant):
                self.selected_constant = c
                self.accept()
                return
        self.reject()

    def get_selected_constant(self) -> Optional[HookedConstant]:
        return self.selected_constant
