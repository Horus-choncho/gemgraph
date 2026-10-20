"""
Main Window for GemGraph.
Derived from BridgeCLR 3-Pane desktop presentation contracts (REA-CLR-VM-102).
"""

from __future__ import annotations
import os
from urllib.parse import unquote
from typing import Optional, List, Dict, Any
import numpy as np
import scipy.stats
from PyQt6.QtCore import Qt, QDir, QUrl
from PyQt6.QtGui import QAction, QKeySequence, QDragEnterEvent, QDropEvent, QDragMoveEvent
from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QLabel,
    QFileDialog,
    QMessageBox,
    QToolBar,
    QStatusBar,
    QHeaderView,
)
from src.core.document import PrismDocument
from src.core.sheet import DataSheet, InfoSheet, ResultSheet, GraphSheet, SheetType, GraphType
from src.core.table import TableType, DataTable
from src.core.undo import UndoStack
from src.io.prism_io import PrismIO
from src.io.xml_io import PrismXMLIO
from src.math.nonlinear import NonlinearEngine, STANDARD_MODELS, ConvergencePreset
from src.math.outlier import OutlierEngine
from src.math.anova import AnovaEngine
from src.math.transforms import (
    transform_data_table,
    arithmetic_transform,
    pharmacological_transform,
    probit_transform,
    concentration_x_transform,
)
from src.math.manipulation import (
    normalize_table,
    prune_rows,
    remove_baseline,
    fraction_of_total,
    transpose_table,
    extract_and_rearrange,
    Baseline0Method,
    Baseline100Method,
    BaselineMethod,
    BaselineOperation,
    NormalizeScale,
)
from src.core.hooking import HookConstantDialog, GLOBAL_REGISTRY, HookedConstant
from src.ui.sidebar import PrismNavigator
from src.ui.spreadsheet import ScientificSpreadsheetModel, ScientificSpreadsheetView
from src.ui.graph_canvas import GraphCanvasWidget
from src.ui.results_view import ResultTabsWidget
from src.ui.analyze_dialog import AnalyzeDataDialog, AnalysisChecklistDialog
from src.ui.format_table_dialog import FormatDataTableDialog
from src.ui.dialogs import NonlinearRegressionDialog, OutlierDialog, AnovaDialog

# Compatibility alias
ResultSheetWidget = ResultTabsWidget


class InfoSheetWidget(QWidget):
    """Presents and edits project metadata notes and constants."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.title_label = QLabel("Experiment Information & Notes")
        font = self.title_label.font()
        font.setBold(True)
        font.setPointSize(12)
        self.title_label.setFont(font)
        self.layout.addWidget(self.title_label)

        self.notes_edit = QTextEdit()
        self.notes_edit.setStyleSheet("""
            QTextEdit {
                background-color: #ffffff;
                color: #000000;
                border: 1px solid #d0d0d0;
            }
        """)
        self.layout.addWidget(self.notes_edit)

        self.current_sheet: Optional[InfoSheet] = None
        self.notes_edit.textChanged.connect(self._on_notes_changed)

    def display_info(self, sheet: InfoSheet) -> None:
        self.current_sheet = sheet
        self.notes_edit.blockSignals(True)
        self.notes_edit.setPlainText(sheet.notes)
        self.notes_edit.blockSignals(False)

    def _on_notes_changed(self) -> None:
        if self.current_sheet:
            self.current_sheet.notes = self.notes_edit.toPlainText()


class MainWindow(QMainWindow):
    """
    Main application window hosting three-pane scientific interface.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("GemGraph — Clean-Room Edition")
        self.resize(1300, 800)
        self.setAcceptDrops(True)

        self.document = PrismDocument("Untitled Project")
        self.undo_stack = UndoStack()

        self._create_blank_project()
        self._setup_ui()
        self._setup_menus()
        self._setup_toolbar()
        self._setup_statusbar()

        # Populate initial view
        self.navigator.populate(self.document)
        if self.document.active_sheet_id:
            self.on_sheet_selected(self.document.active_sheet_id)

    def _create_blank_project(self) -> None:
        """Initializes a pristine, blank XY table (columns X, Y1) with empty placeholder rows."""
        sheet = self.document.create_data_sheet(
            title="Data 1",
            table_type=TableType.XY,
            num_rows=20,
            num_replicates=1,
            y_group_titles=["Group A"],
        )
        self.document.active_sheet_id = sheet.sheet_id

    def _setup_ui(self) -> None:
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(4, 4, 4, 4)

        # 3-Pane Splitter
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left Pane: Navigator
        self.navigator = PrismNavigator()
        self.navigator.sheetSelected.connect(self.on_sheet_selected)
        self.navigator.sheetDeleted.connect(self.on_sheet_deleted)
        self.main_splitter.addWidget(self.navigator)

        # Center Pane: Dynamic Stack
        self.center_stack = QStackedWidget()

        # Data Sheet View
        init_sheet = self.document.get_sheets_by_type(SheetType.DATA)[0]
        self.spreadsheet_model = ScientificSpreadsheetModel(init_sheet.table, self.undo_stack)  # type: ignore
        self.spreadsheet_view = ScientificSpreadsheetView(self.spreadsheet_model)
        self.spreadsheet_model.dataChangedSignal.connect(self._on_table_data_modified)
        self.spreadsheet_view.cellExclusionToggled.connect(self._on_table_data_modified)
        self.center_stack.addWidget(self.spreadsheet_view)

        # Result Sheet View (Multi-Tab and Green/Red Grid Semantics)
        self.result_widget = ResultTabsWidget()
        self.result_widget.analyzeRequested.connect(self.action_analyze_dialog)
        self.result_widget.newGraphRequested.connect(self.action_add_graph)
        self.result_widget.freezeToggled.connect(lambda frozen: self.navigator.populate(self.document))
        self.center_stack.addWidget(self.result_widget)

        # Info Sheet View
        self.info_widget = InfoSheetWidget()
        self.center_stack.addWidget(self.info_widget)

        self.main_splitter.addWidget(self.center_stack)

        # Right Pane: Interactive Plot Canvas
        self.graph_canvas = GraphCanvasWidget()
        self.main_splitter.addWidget(self.graph_canvas)

        # Proportions: 200px sidebar, 600px table/result, 500px plot
        self.main_splitter.setSizes([200, 580, 520])
        main_layout.addWidget(self.main_splitter)

    def _setup_menus(self) -> None:
        menubar = self.menuBar()

        # File Menu
        file_menu = menubar.addMenu("&File")

        new_act = QAction("&New Project", self)
        new_act.setShortcut(QKeySequence.StandardKey.New)
        new_act.triggered.connect(self.action_new_project)
        file_menu.addAction(new_act)

        open_act = QAction("&Open Project (.prism)...", self)
        open_act.setShortcut(QKeySequence.StandardKey.Open)
        open_act.triggered.connect(self.action_open_project)
        file_menu.addAction(open_act)

        save_act = QAction("&Save Project (.prism)", self)
        save_act.setShortcut(QKeySequence.StandardKey.Save)
        save_act.triggered.connect(self.action_save_project)
        file_menu.addAction(save_act)

        file_menu.addSeparator()

        import_xml_act = QAction("Import Prism XML (.xml)...", self)
        import_xml_act.triggered.connect(self.action_import_xml)
        file_menu.addAction(import_xml_act)

        export_xml_act = QAction("Export Prism XML (.xml)...", self)
        export_xml_act.triggered.connect(self.action_export_xml)
        file_menu.addAction(export_xml_act)

        file_menu.addSeparator()

        export_png_act = QAction("Export Graph as PNG...", self)
        export_png_act.triggered.connect(self.action_export_png)
        file_menu.addAction(export_png_act)

        export_svg_act = QAction("Export Graph as Vector SVG...", self)
        export_svg_act.triggered.connect(self.action_export_svg)
        file_menu.addAction(export_svg_act)

        file_menu.addSeparator()

        exit_act = QAction("E&xit", self)
        exit_act.setShortcut(QKeySequence.StandardKey.Quit)
        exit_act.triggered.connect(self.close)
        file_menu.addAction(exit_act)

        # Edit Menu
        edit_menu = menubar.addMenu("&Edit")

        self.undo_act = QAction("&Undo", self)
        self.undo_act.setShortcut(QKeySequence.StandardKey.Undo)
        self.undo_act.triggered.connect(self.action_undo)
        edit_menu.addAction(self.undo_act)

        self.redo_act = QAction("&Redo", self)
        self.redo_act.setShortcut(QKeySequence.StandardKey.Redo)
        self.redo_act.triggered.connect(self.action_redo)
        edit_menu.addAction(self.redo_act)

        edit_menu.addSeparator()
        fmt_table_edit_act = QAction("&Format Data Table...", self)
        fmt_table_edit_act.triggered.connect(self.action_format_table)
        edit_menu.addAction(fmt_table_edit_act)

        # Insert Menu — All 8 Scientific Table Paradigms
        insert_menu = menubar.addMenu("&Insert")

        add_xy_act = QAction("New &XY Data Table", self)
        add_xy_act.triggered.connect(lambda: self.action_add_table(TableType.XY))
        insert_menu.addAction(add_xy_act)

        add_col_act = QAction("New &Column Data Table", self)
        add_col_act.triggered.connect(lambda: self.action_add_table(TableType.COLUMN))
        insert_menu.addAction(add_col_act)

        add_grp_act = QAction("New &Grouped Data Table", self)
        add_grp_act.triggered.connect(lambda: self.action_add_table(TableType.GROUPED))
        insert_menu.addAction(add_grp_act)

        add_cont_act = QAction("New &Contingency Table", self)
        add_cont_act.triggered.connect(lambda: self.action_add_table(TableType.CONTINGENCY))
        insert_menu.addAction(add_cont_act)

        add_surv_act = QAction("New &Survival Data Table", self)
        add_surv_act.triggered.connect(lambda: self.action_add_table(TableType.SURVIVAL))
        insert_menu.addAction(add_surv_act)

        add_pow_act = QAction("New &Parts of Whole Table", self)
        add_pow_act.triggered.connect(lambda: self.action_add_table(TableType.PARTS_OF_WHOLE))
        insert_menu.addAction(add_pow_act)

        add_mv_act = QAction("New &Multiple Variables Table", self)
        add_mv_act.triggered.connect(lambda: self.action_add_table(TableType.MULTIPLE_VARIABLES))
        insert_menu.addAction(add_mv_act)

        add_nest_act = QAction("New &Nested Data Table", self)
        add_nest_act.triggered.connect(lambda: self.action_add_table(TableType.NESTED))
        insert_menu.addAction(add_nest_act)

        insert_menu.addSeparator()

        add_info_act = QAction("New &Info Sheet", self)
        add_info_act.triggered.connect(self.action_add_info)
        insert_menu.addAction(add_info_act)

        add_graph_act = QAction("New &Graph Sheet", self)
        add_graph_act.triggered.connect(self.action_add_graph)
        insert_menu.addAction(add_graph_act)

        # Change Menu
        change_menu = menubar.addMenu("&Change")
        fmt_table_act = QAction("&Format Data Table...", self)
        fmt_table_act.triggered.connect(self.action_format_table)
        change_menu.addAction(fmt_table_act)

        # Analysis Menu — 6 Objective-Oriented Submenus
        analysis_menu = menubar.addMenu("&Analysis")

        analyze_dialog_act = QAction("&Analyze Data...", self)
        analyze_dialog_act.setShortcut(QKeySequence("Ctrl+T"))
        analyze_dialog_act.triggered.connect(self.action_analyze_dialog)
        analysis_menu.addAction(analyze_dialog_act)

        analysis_menu.addSeparator()

        # 1. Data Processing
        dp_menu = analysis_menu.addMenu("&Data Processing")
        dp_menu.addAction("Transform...", self.action_run_transform)
        dp_menu.addAction("Normalize...", self.action_run_normalize)
        dp_menu.addAction("Prune Rows...", self.action_run_prune)
        dp_menu.addAction("Remove Baseline & Column Math...", self.action_run_remove_baseline)
        dp_menu.addAction("Fraction of Total...", self.action_run_fraction_of_total)
        dp_menu.addAction("Transpose...", self.action_run_transpose)
        dp_menu.addAction("Extract and Rearrange...", self.action_run_extract_and_rearrange)

        # 2. Data Exploration and Summary
        des_menu = analysis_menu.addMenu("Data &Exploration and Summary")
        des_menu.addAction("Identify Outliers (ROUT / Grubbs)...", self.action_run_outliers)
        des_menu.addAction("Descriptive Statistics / Summary...", self.action_run_descriptive_stats)
        des_menu.addAction("Correlation Matrix...", self.action_run_correlation)
        des_menu.addAction("Frequency Distribution...", self.action_run_frequency_dist)

        # 3. Regression and Curves
        rc_menu = analysis_menu.addMenu("&Regression and Curves")
        rc_menu.addAction("Nonlinear Regression (Curve Fit)...", self.action_run_nonlinear)
        rc_menu.addAction("Linear Regression...", self.action_run_linear)
        rc_menu.addAction("Multiple Regression...", self.action_run_multiple_reg)
        rc_menu.addAction("Logistic Regression...", self.action_run_logistic)
        rc_menu.addAction("Interpolate from Standard Curve...", self.action_run_interpolate)

        # 4. Group Comparison
        gc_menu = analysis_menu.addMenu("&Group Comparison")
        gc_menu.addAction("One-Way ANOVA & Multiple Comparisons...", self.action_run_anova)
        gc_menu.addAction("Two-Way ANOVA (Grouped Data)...", self.action_run_two_way_anova)
        gc_menu.addAction("Multiple t-tests...", self.action_run_multiple_t_tests)
        gc_menu.addAction("Nested t-test / Nested One-Way ANOVA...", self.action_run_nested_analysis)

        # 5. Survival Analysis
        surv_menu = analysis_menu.addMenu("&Survival Analysis")
        surv_menu.addAction("Kaplan-Meier Survival Analysis...", self.action_run_survival)

        # 6. Simulation
        sim_menu = analysis_menu.addMenu("S&imulation")
        sim_menu.addAction("Simulate XY Data...", self.action_run_simulate_xy)
        sim_menu.addAction("Simulate Column Data...", self.action_run_simulate_column)

        analysis_menu.addSeparator()

        checklist_act = QAction("Analysis &Checklist...", self)
        checklist_act.triggered.connect(self.action_analysis_checklist)
        analysis_menu.addAction(checklist_act)

        hook_act = QAction("&Hook Constant (s)...", self)
        hook_act.setShortcut(QKeySequence("s"))
        hook_act.triggered.connect(self.action_hook_constant)
        analysis_menu.addAction(hook_act)

        # Help Menu
        help_menu = menubar.addMenu("&Help")
        load_sample_act = QAction("&Load Sample Data", self)
        load_sample_act.triggered.connect(self.action_load_sample_data)
        help_menu.addAction(load_sample_act)
        help_menu.addSeparator()
        about_act = QAction("&About GemGraph", self)
        about_act.triggered.connect(self.action_about)
        help_menu.addAction(about_act)

    def _setup_toolbar(self) -> None:
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        toolbar.addAction("New Project", self.action_new_project)
        toolbar.addAction("Open", self.action_open_project)
        toolbar.addAction("Save", self.action_save_project)
        toolbar.addSeparator()

        toolbar.addAction("Undo", self.action_undo)
        toolbar.addAction("Redo", self.action_redo)
        toolbar.addSeparator()

        toolbar.addAction("Analyze", self.action_analyze_dialog)
        toolbar.addAction("Hook (s)", self.action_hook_constant)
        toolbar.addAction("Checklist", self.action_analysis_checklist)
        toolbar.addAction("Format Table", self.action_format_table)
        toolbar.addSeparator()

        toolbar.addAction("Add Table", lambda: self.action_add_table(TableType.XY))
        toolbar.addAction("Add Graph", self.action_add_graph)

        self._setup_contextual_toolbar()

    def _setup_contextual_toolbar(self) -> None:
        self.context_toolbar = QToolBar("Contextual Analysis Shortcuts")
        self.context_toolbar.setMovable(False)
        self.addToolBar(self.context_toolbar)
        self._update_contextual_toolbar(TableType.XY)

    def _update_contextual_toolbar(self, table_type: TableType) -> None:
        self.context_toolbar.clear()
        self.context_toolbar.addWidget(QLabel(f"<b>{table_type.value} Shortcuts:</b> "))

        if table_type == TableType.XY:
            self.context_toolbar.addAction("Linear Reg", self.action_run_linear)
            self.context_toolbar.addAction("Nonlin Reg", self.action_run_nonlinear)
            self.context_toolbar.addAction("Logistic", self.action_run_logistic)
            self.context_toolbar.addAction("Interpolate", self.action_run_interpolate)
        elif table_type == TableType.MULTIPLE_VARIABLES:
            self.context_toolbar.addAction("Correlation Matrix", self.action_run_correlation)
            self.context_toolbar.addAction("Multiple Reg", self.action_run_multiple_reg)
            self.context_toolbar.addAction("PCA", self.action_run_pca)
            self.context_toolbar.addAction("Extract & Rearrange", self.action_run_extract_and_rearrange)
        elif table_type == TableType.GROUPED:
            self.context_toolbar.addAction("2-Way ANOVA", self.action_run_two_way_anova)
            self.context_toolbar.addAction("Multiple t-tests", self.action_run_multiple_t_tests)
        elif table_type == TableType.COLUMN:
            self.context_toolbar.addAction("1-Way ANOVA", self.action_run_anova)
            self.context_toolbar.addAction("t-test", self.action_run_t_test)
            self.context_toolbar.addAction("Outliers", self.action_run_outliers)
            self.context_toolbar.addAction("Descriptive Stats", self.action_run_descriptive_stats)
        elif table_type == TableType.CONTINGENCY:
            self.context_toolbar.addAction("Chi-Square / Fisher's", self.action_run_contingency_test)
        elif table_type == TableType.SURVIVAL:
            self.context_toolbar.addAction("Survival Curve", self.action_run_survival)
        elif table_type == TableType.PARTS_OF_WHOLE:
            self.context_toolbar.addAction("Fraction of Total", self.action_run_fraction_of_total)
            self.context_toolbar.addAction("Goodness-of-Fit", self.action_run_goodness_of_fit)
        elif table_type == TableType.NESTED:
            self.context_toolbar.addAction("Nested ANOVA / t-test", self.action_run_nested_analysis)

        self.context_toolbar.addSeparator()
        self.context_toolbar.addAction("Hook (s)", self.action_hook_constant)
        self.context_toolbar.addAction("Checklist", self.action_analysis_checklist)

    def _setup_statusbar(self) -> None:
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("GemGraph Ready | Clean-Room Mode: Active")

    def on_sheet_selected(self, sheet_id: str) -> None:
        sheet = self.document.get_sheet(sheet_id)
        if not sheet:
            return
        self.document.active_sheet_id = sheet_id

        if sheet.sheet_type == SheetType.DATA:
            data_sheet: DataSheet = sheet  # type: ignore
            self.spreadsheet_model.set_table(data_sheet.table)
            self.center_stack.setCurrentWidget(self.spreadsheet_view)
            self.graph_canvas.update_data(data_sheet.table)
            self._update_contextual_toolbar(data_sheet.table.table_type)
            GLOBAL_REGISTRY.extract_from_table(data_sheet.table)
            self.status.showMessage(f"Active Table: {sheet.title} ({data_sheet.table.table_type.value})")

        elif sheet.sheet_type == SheetType.RESULTS:
            res_sheet: ResultSheet = sheet  # type: ignore
            GLOBAL_REGISTRY.extract_from_result(res_sheet)
            self.result_widget.display_result(res_sheet)
            self.center_stack.setCurrentWidget(self.result_widget)
            src_sheet = self.document.get_sheet(res_sheet.source_sheet_id)
            src_table = src_sheet.table if isinstance(src_sheet, DataSheet) else None
            self.graph_canvas.update_data(src_table, result=res_sheet)
            self.status.showMessage(f"Active Result: {sheet.title}")

        elif sheet.sheet_type == SheetType.INFO:
            info_sheet: InfoSheet = sheet  # type: ignore
            self.info_widget.display_info(info_sheet)
            self.center_stack.setCurrentWidget(self.info_widget)
            self.status.showMessage(f"Active Info Sheet: {sheet.title}")

        elif sheet.sheet_type == SheetType.GRAPH:
            graph_sheet: GraphSheet = sheet  # type: ignore
            src_sheet = self.document.get_sheet(graph_sheet.source_sheet_id)
            src_table = src_sheet.table if isinstance(src_sheet, DataSheet) else None
            res_sheet = (
                self.document.get_sheet(graph_sheet.result_sheet_id)
                if graph_sheet.result_sheet_id
                else None
            )
            self.graph_canvas.update_data(
                src_table,
                result=res_sheet if isinstance(res_sheet, ResultSheet) else None,
                graph_type=graph_sheet.graph_type,
            )
            self.status.showMessage(f"Active Graph: {sheet.title}")

    def on_sheet_deleted(self, sheet_id: str) -> None:
        self.document.remove_sheet(sheet_id)
        GLOBAL_REGISTRY.remove_by_sheet(sheet_id)
        self.navigator.populate(self.document)
        if self.document.active_sheet_id:
            self.on_sheet_selected(self.document.active_sheet_id)

    def _on_table_data_modified(self) -> None:
        # Re-render graph canvas when table values change and propagate downstream
        active_sheet = self.document.get_sheet(self.document.active_sheet_id or "")
        if isinstance(active_sheet, DataSheet):
            GLOBAL_REGISTRY.extract_from_table(active_sheet.table)
            self.document.recalculate_downstream(active_sheet.sheet_id)
            self.graph_canvas.update_data(active_sheet.table)

    def action_undo(self) -> None:
        if self.undo_stack.undo():
            self.spreadsheet_model.layoutChanged.emit()
            self._on_table_data_modified()

    def action_redo(self) -> None:
        if self.undo_stack.redo():
            self.spreadsheet_model.layoutChanged.emit()
            self._on_table_data_modified()

    def action_new_project(self) -> None:
        self.document = PrismDocument("Untitled Project")
        self.undo_stack.clear()
        self._create_blank_project()
        self.navigator.populate(self.document)
        if self.document.active_sheet_id:
            self.on_sheet_selected(self.document.active_sheet_id)

    def action_load_sample_data(self) -> None:
        """Loads sample/demo dose-response dataset when explicitly requested via Help > Load Sample Data."""
        self.document = PrismDocument("Sample Analysis Project")
        self.undo_stack.clear()

        sheet = self.document.create_data_sheet(
            title="Dose Response", table_type=TableType.XY, num_rows=8, num_replicates=3
        )
        table = sheet.table
        table.x_title = "Log [Agonist], M"
        x_vals = [-9.0, -8.0, -7.0, -6.5, -6.0, -5.5, -5.0, -4.0]
        # Realistic 4PL sigmoid curve: Bottom=10, Top=100, LogEC50=-6.0, Hill=1.0
        for r, x in enumerate(x_vals):
            table.set_cell_value(r, "X", 0, 0, x)
            expected = 10.0 + (100.0 - 10.0) / (1.0 + 10.0 ** (-6.0 - x))
            table.set_cell_value(r, "Y", 0, 0, round(expected - 2.5, 2))
            table.set_cell_value(r, "Y", 0, 1, round(expected + 1.2, 2))
            table.set_cell_value(r, "Y", 0, 2, round(expected + 0.8, 2))

        self.document.create_info_sheet(title="Project Info")
        self.document.create_graph_sheet(title="Dose Response Curve", source_sheet_id=sheet.sheet_id)

        self.navigator.populate(self.document)
        if self.document.active_sheet_id:
            self.on_sheet_selected(self.document.active_sheet_id)
        self.status.showMessage("Loaded sample dose-response dataset.")

    @staticmethod
    def get_smart_initial_dir() -> str:
        """
        Determines smart initial path for file dialogs.
        1. If /uploads exists on system, default to /uploads.
        2. Else if samples/ exists in workspace, default to samples/.
        3. Otherwise fallback to QDir.homePath() or cwd.
        """
        if os.path.exists("/uploads") and os.path.isdir("/uploads"):
            return "/uploads"

        workspace_samples = os.path.abspath("samples")
        if os.path.exists(workspace_samples) and os.path.isdir(workspace_samples):
            return workspace_samples

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        root_samples = os.path.join(project_root, "samples")
        if os.path.exists(root_samples) and os.path.isdir(root_samples):
            return root_samples

        home = QDir.homePath()
        if home and os.path.exists(home) and os.path.isdir(home):
            return home

        return os.getcwd()

    @classmethod
    def get_preset_paths(cls) -> List[str]:
        """Returns standard pre-set paths for sidebar shortcuts (/uploads, samples/, and project directories)."""
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        samples_path = os.path.abspath("samples")
        if not os.path.isdir(samples_path):
            samples_path = os.path.join(project_root, "samples")

        candidates = ["/uploads", samples_path, project_root, os.getcwd()]
        result: List[str] = []
        for p in candidates:
            norm = os.path.abspath(p) if p != "/uploads" else "/uploads"
            if norm not in result:
                result.append(norm)
        return result

    def create_open_project_dialog(self) -> QFileDialog:
        """Creates an Open Project file dialog with smart initial path and pre-set sidebar shortcuts."""
        initial_dir = self.get_smart_initial_dir()
        dialog = QFileDialog(
            self,
            "Open Prism Project",
            initial_dir,
            "Prism Files (*.prism *.pzfx *.csv);;Prism Native (*.prism);;Prism XML (*.pzfx *.xml *.pzm);;CSV Data (*.csv);;All Files (*)",
        )
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        sidebar_urls = list(dialog.sidebarUrls())
        for p in self.get_preset_paths():
            if os.path.exists(p) or p == "/uploads":
                u = QUrl.fromLocalFile(p)
                if u not in sidebar_urls:
                    sidebar_urls.append(u)
        dialog.setSidebarUrls(sidebar_urls)
        return dialog

    def open_project(self, filepath: str) -> None:
        """
        Loads dataset from filepath (.prism, .pzfx/.xml, .csv),
        populates the spreadsheet, and renders the initial plot.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".prism":
            self.document = PrismIO.load(filepath)
        elif ext in (".pzfx", ".xml", ".pzm"):
            self.document = PrismXMLIO.import_xml(filepath)
        elif ext == ".csv":
            self.document = PrismIO.load_csv(filepath)
        else:
            try:
                self.document = PrismIO.load(filepath)
            except Exception:
                try:
                    self.document = PrismXMLIO.import_xml(filepath)
                except Exception:
                    self.document = PrismIO.load_csv(filepath)

        self.undo_stack.clear()
        self.navigator.populate(self.document)

        # Select appropriate sheet: prioritize active_sheet_id or first data sheet
        target_sheet_id = self.document.active_sheet_id
        data_sheets = self.document.get_sheets_by_type(SheetType.DATA)
        if not target_sheet_id or target_sheet_id not in self.document.sheets:
            if data_sheets:
                target_sheet_id = data_sheets[0].sheet_id
            elif self.document.sheet_order:
                target_sheet_id = self.document.sheet_order[0]

        if target_sheet_id:
            self.on_sheet_selected(target_sheet_id)
            if data_sheets and not isinstance(self.document.get_sheet(target_sheet_id), DataSheet):
                self.spreadsheet_model.set_table(data_sheets[0].table)
                self.center_stack.setCurrentWidget(self.spreadsheet_view)

        self.status.showMessage(f"Opened: {filepath}")

    def action_open_project(self, dialog: Optional[QFileDialog] = None) -> None:
        dlg = dialog or self.create_open_project_dialog()
        if dlg.exec():
            selected = dlg.selectedFiles()
            if selected:
                try:
                    self.open_project(selected[0])
                except Exception as e:
                    QMessageBox.critical(self, "Error Loading Project", str(e))

    def action_save_project(self) -> None:
        initial_dir = self.get_smart_initial_dir()
        default_path = os.path.join(initial_dir, f"{self.document.title}.prism")
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Prism Project", default_path, "Prism Files (*.prism)"
        )
        if file_path:
            try:
                PrismIO.save(self.document, file_path)
                self.status.showMessage(f"Saved: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error Saving Project", str(e))

    def action_import_xml(self) -> None:
        initial_dir = self.get_smart_initial_dir()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Import Prism XML", initial_dir, "Prism XML (*.xml *.pzm);;All Files (*)"
        )
        if file_path:
            try:
                self.open_project(file_path)
            except Exception as e:
                QMessageBox.critical(self, "Error Importing XML", str(e))

    def action_export_xml(self) -> None:
        initial_dir = self.get_smart_initial_dir()
        default_path = os.path.join(initial_dir, f"{self.document.title}.xml")
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Prism XML", default_path, "Prism XML (*.xml)"
        )
        if file_path:
            try:
                PrismXMLIO.export_xml(self.document, file_path)
                self.status.showMessage(f"Exported XML: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error Exporting XML", str(e))

    def action_export_png(self) -> None:
        initial_dir = self.get_smart_initial_dir()
        default_path = os.path.join(initial_dir, "graph.png")
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export PNG", default_path, "PNG Images (*.png)"
        )
        if file_path:
            self.graph_canvas.export_png(file_path)
            self.status.showMessage(f"Exported PNG: {file_path}")

    def action_export_svg(self) -> None:
        initial_dir = self.get_smart_initial_dir()
        default_path = os.path.join(initial_dir, "graph.svg")
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export SVG", default_path, "Vector SVG (*.svg)"
        )
        if file_path:
            self.graph_canvas.export_svg(file_path)
            self.status.showMessage(f"Exported SVG: {file_path}")

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        mime = event.mimeData()
        if mime and mime.hasUrls():
            for url in mime.urls():
                filepath = unquote(url.toLocalFile() or url.path())
                if any(filepath.lower().endswith(ext) for ext in (".prism", ".pzfx", ".csv")):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        mime = event.mimeData()
        if mime and mime.hasUrls():
            for url in mime.urls():
                filepath = unquote(url.toLocalFile() or url.path())
                if any(filepath.lower().endswith(ext) for ext in (".prism", ".pzfx", ".csv")):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        mime = event.mimeData()
        if mime and mime.hasUrls():
            urls = mime.urls()
            if urls:
                # Find the first URL ending in supported extensions; default to urls[0]
                target_url = urls[0]
                for u in urls:
                    fp = unquote(u.toLocalFile() or u.path())
                    if any(fp.lower().endswith(ext) for ext in (".prism", ".pzfx", ".csv")):
                        target_url = u
                        break

                filepath = target_url.toLocalFile()
                if not filepath:
                    filepath = unquote(target_url.path())
                else:
                    filepath = unquote(filepath)

                if filepath:
                    try:
                        self.open_project(filepath)
                        event.acceptProposedAction()
                        return
                    except Exception as e:
                        QMessageBox.critical(self, "Error Loading Project", str(e))
        event.ignore()

    def action_add_table(self, table_type: TableType) -> None:
        if table_type == TableType.XY:
            sheet = self.document.create_data_sheet(
                table_type=table_type, num_replicates=1, y_group_titles=["Group A"]
            )
        else:
            sheet = self.document.create_data_sheet(table_type=table_type)
        self.navigator.populate(self.document)
        self.on_sheet_selected(sheet.sheet_id)


    def action_add_info(self) -> None:
        sheet = self.document.create_info_sheet()
        self.navigator.populate(self.document)
        self.on_sheet_selected(sheet.sheet_id)

    def action_add_graph(self) -> None:
        active_sheet = self.document.get_sheet(self.document.active_sheet_id or "")
        src_id = active_sheet.sheet_id if isinstance(active_sheet, DataSheet) else ""
        sheet = self.document.create_graph_sheet(source_sheet_id=src_id)
        self.navigator.populate(self.document)
        self.on_sheet_selected(sheet.sheet_id)

    def action_run_nonlinear(self) -> None:
        active_sheet = self.document.get_sheet(self.document.active_sheet_id or "")
        if not isinstance(active_sheet, DataSheet):
            QMessageBox.warning(self, "Curve Fitting", "Please select a Data Sheet with numeric data.")
            return

        table: DataTable = active_sheet.table
        if table.x_data is None:
            QMessageBox.warning(self, "Curve Fitting", "Table does not contain an X column.")
            return

        dlg = NonlinearRegressionDialog(table.y_group_titles, self)
        if dlg.exec():
            model_name, group_idx, preset = dlg.get_settings()
            model = STANDARD_MODELS[model_name]
            xs, ys = table.get_clean_xy(group_idx)

            if len(xs) <= len(model.param_names):
                QMessageBox.warning(self, "Curve Fitting", "Not enough valid data points to fit model.")
                return

            try:
                fit_result = NonlinearEngine.fit(xs, ys, model, preset=preset)
            except Exception as e:
                QMessageBox.critical(self, "Curve Fitting Error", str(e))
                return

            # Build summary table rows
            headers = ["Parameter", "Best-Fit Value", "Std. Error", "95% CI Lower", "95% CI Upper", "Dependency"]
            rows = []
            for p in fit_result["parameters"]:
                rows.append([
                    p["name"],
                    f"{p['best_fit']:.4f}",
                    f"{p['std_error']:.4f}",
                    f"{p['ci_lower']:.4f}",
                    f"{p['ci_upper']:.4f}",
                    f"{p['dependency']:.4f}",
                ])

            # Add Goodness-of-Fit section
            rows.append(["--- Goodness of Fit ---", "", "", "", "", ""])
            rows.append(["Degrees of Freedom", str(fit_result["degrees_of_freedom"]), "", "", "", ""])
            rows.append(["R squared", f"{fit_result['r_squared']:.4f}", "", "", "", ""])
            rows.append(["Sy.x (Std Error of Fit)", f"{fit_result['syx']:.4f}", "", "", "", ""])
            rows.append(["AICc", f"{fit_result['aicc']:.2f}", "", "", "", ""])
            rows.append(["Iterations", str(fit_result["iterations"]), "", "", "", ""])

            # Create ResultSheet
            res_sheet = self.document.create_result_sheet(
                title=f"Nonlin reg ({model_name})",
                analysis_type="Nonlinear Regression",
                source_sheet_id=active_sheet.sheet_id,
                summary_tables=[{"title": "Parameter Estimates", "headers": headers, "rows": rows}],
                raw_result_data=fit_result,
            )

            self.navigator.populate(self.document)
            self.on_sheet_selected(res_sheet.sheet_id)

    def action_run_outliers(self) -> None:
        active_sheet = self.document.get_sheet(self.document.active_sheet_id or "")
        if not isinstance(active_sheet, DataSheet):
            QMessageBox.warning(self, "Outlier Detection", "Please select a Data Sheet with numeric data.")
            return

        table: DataTable = active_sheet.table
        dlg = OutlierDialog(table.y_group_titles, self)
        if dlg.exec():
            group_idx, method, param_val, auto_exclude = dlg.get_settings()
            all_groups = table.get_all_group_values()
            g_title = table.y_group_titles[group_idx]
            vals = np.array(all_groups.get(g_title, []), dtype=np.float64)

            if len(vals) < 3:
                QMessageBox.warning(self, "Outlier Detection", "Outlier detection requires at least N >= 3 points.")
                return

            if method == "ROUT":
                res = OutlierEngine.rout_1d(vals, q_fdr=param_val)
            else:
                res = OutlierEngine.grubbs_esd(vals, alpha=param_val)

            outlier_indices = res.get("outlier_indices", [])
            outlier_vals = res.get("outlier_values", [])

            # Automatically flag in table if requested
            if auto_exclude and outlier_indices:
                # Find matching row and subcolumn in table and exclude
                for r in range(table.num_rows):
                    for s in range(table.num_replicates):
                        cell_val = table.get_cell_value(r, "Y", group_idx, s)
                        if any(np.isclose(cell_val, ov) for ov in outlier_vals):
                            table.set_excluded(r, "Y", group_idx, s, True)
                self.spreadsheet_model.layoutChanged.emit()
                self._on_table_data_modified()

            # Create result sheet
            headers = ["Metric", "Value"]
            rows = [
                ["Method", res["method"]],
                ["Target Group", g_title],
                ["Sample Size (N)", str(len(vals))],
                ["Outliers Detected", str(len(outlier_indices))],
                ["Outlier Values", ", ".join(f"{v:.4g}" for v in outlier_vals) if outlier_vals else "None"],
            ]
            if "rsdr" in res:
                rows.append(["Robust SD of Residuals (RSDR)", f"{res['rsdr']:.4f}"])

            res_sheet = self.document.create_result_sheet(
                title=f"Outliers ({g_title})",
                analysis_type="Outlier Detection",
                source_sheet_id=active_sheet.sheet_id,
                summary_tables=[{"title": "Outlier Detection Report", "headers": headers, "rows": rows}],
                raw_result_data=res,
            )
            self.navigator.populate(self.document)
            self.on_sheet_selected(res_sheet.sheet_id)

    def action_run_anova(self) -> None:
        active_sheet = self.document.get_sheet(self.document.active_sheet_id or "")
        if not isinstance(active_sheet, DataSheet):
            QMessageBox.warning(self, "ANOVA", "Please select a Data Sheet with numeric groups.")
            return

        table: DataTable = active_sheet.table
        all_groups = table.get_all_group_values()
        valid_groups = {k: v for k, v in all_groups.items() if len(v) > 0}

        if len(valid_groups) < 2:
            QMessageBox.warning(self, "ANOVA", "ANOVA requires at least 2 non-empty groups.")
            return

        dlg = AnovaDialog(list(valid_groups.keys()), self)
        if dlg.exec():
            test_type, posthoc, ctrl_idx = dlg.get_settings()
            try:
                if test_type == "rm":
                    # Repeated Measures: subjects are rows, treatments are groups
                    treat_names = list(valid_groups.keys())
                    matrix_data = []
                    for r in range(table.num_rows):
                        row_vals = []
                        for name in treat_names:
                            g_idx = table.y_group_titles.index(name)
                            # Average non-excluded replicates for this cell
                            cell_vals = [
                                table.y_data[g_idx, r, s]
                                for s in range(table.num_replicates)
                                if not table.is_excluded(r, "Y", g_idx, s)
                                and not np.isnan(table.y_data[g_idx, r, s])
                            ]
                            row_vals.append(np.mean(cell_vals) if cell_vals else np.nan)
                        matrix_data.append(row_vals)
                    rm_matrix = np.array(matrix_data, dtype=np.float64)
                    anova_res = AnovaEngine.repeated_measures(rm_matrix)
                    title_name = "Repeated Measures ANOVA"
                else:
                    anova_res = AnovaEngine.one_way(valid_groups)
                    title_name = "One-way ANOVA"
            except Exception as e:
                QMessageBox.critical(self, "ANOVA Calculation Error", str(e))
                return

            # Format ANOVA summary rows
            headers = ["Source of Variation", "Sum of Squares", "Degrees of Freedom", "Mean Square", "F", "p-value"]
            rows = []
            for item in anova_res["anova_table"]:
                rows.append([
                    item["source"],
                    f"{item['ss']:.4f}",
                    str(item["df"]),
                    f"{item['ms']:.4f}" if item["ms"] is not None else "",
                    f"{item['f']:.4f}" if item["f"] is not None else "",
                    f"{item['p_value']:.4e}" if item["p_value"] is not None else "",
                ])

            if test_type == "rm":
                rows.append(["--- Sphericity Correction ---", "", "", "", "", ""])
                rows.append(["Geisser-Greenhouse Epsilon", f"{anova_res.get('epsilon_geisser_greenhouse', 1.0):.4f}", "", "", "", ""])
                rows.append(["Corrected p-value", f"{anova_res.get('p_value_corrected', np.nan):.4e}", "", "", "", ""])

            # Post-hoc comparisons
            if posthoc == "dunnett":
                # For Dunnett post-hoc on RM or Ordinary, use one_way result structure
                base_anova = anova_res if test_type != "rm" else AnovaEngine.one_way(valid_groups)
                dunnett_comps = AnovaEngine.dunnett_test(base_anova, control_idx=ctrl_idx)
                rows.append(["--- Dunnett Multiple Comparisons vs Control ---", "", "", "", "", ""])
                for comp in dunnett_comps:
                    sig = "Yes (p < 0.05)" if comp["significant"] else "No (ns)"
                    rows.append([
                        comp["comparison"],
                        f"Diff = {comp['mean_diff']:.4f}",
                        f"q = {comp['q_stat']:.4f}",
                        f"p = {comp['p_value']:.4e}",
                        sig,
                        f"95% CI [{comp['ci_lower']:.4f}, {comp['ci_upper']:.4f}]",
                    ])
            elif posthoc == "tukey":
                base_anova = anova_res if test_type != "rm" else AnovaEngine.one_way(valid_groups)
                tukey_comps = AnovaEngine.tukey_hsd(base_anova)
                rows.append(["--- Tukey HSD All Pairwise Comparisons ---", "", "", "", "", ""])
                for comp in tukey_comps:
                    sig = "Yes (p < 0.05)" if comp["significant"] else "No (ns)"
                    rows.append([
                        comp["comparison"],
                        f"Diff = {comp['mean_diff']:.4f}",
                        f"q = {comp['q_stat']:.4f}",
                        f"p = {comp['p_value']:.4e}",
                        sig,
                        f"95% CI [{comp['ci_lower']:.4f}, {comp['ci_upper']:.4f}]",
                    ])

            res_sheet = self.document.create_result_sheet(
                title=title_name,
                analysis_type="ANOVA",
                source_sheet_id=active_sheet.sheet_id,
                summary_tables=[{"title": f"{title_name} Table", "headers": headers, "rows": rows}],
                raw_result_data=anova_res,
            )
            self.navigator.populate(self.document)
            self.on_sheet_selected(res_sheet.sheet_id)

    def action_run_two_way_anova(self) -> None:
        """Executes Two-Way ANOVA on Grouped or replicated dataset."""
        active_sheet = self.document.get_sheet(self.document.active_sheet_id or "")
        if not isinstance(active_sheet, DataSheet):
            QMessageBox.warning(self, "Two-Way ANOVA", "Please select a Data Sheet with Grouped data.")
            return

        table: DataTable = active_sheet.table
        # data_matrix expects (num_rows, num_cols, num_replicates)
        # table.y_data shape is (num_cols, num_rows, num_replicates)
        if len(table.y_group_titles) < 2 or table.num_rows < 2:
            QMessageBox.warning(self, "Two-Way ANOVA", "Two-Way ANOVA requires at least 2 rows and 2 columns.")
            return

        data_matrix = np.transpose(table.y_data, (1, 0, 2))
        row_names = [rt if rt else f"Row {r+1}" for r, rt in enumerate(table.row_titles)]
        col_names = list(table.y_group_titles)

        try:
            res = AnovaEngine.two_way(data_matrix, row_labels=row_names, col_labels=col_names)
        except Exception as e:
            QMessageBox.critical(self, "Two-Way ANOVA Error", str(e))
            return

        headers = ["Source of Variation", "Sum of Squares", "Degrees of Freedom", "Mean Square", "F", "p-value"]
        rows = []
        for item in res["anova_table"]:
            rows.append([
                item["source"],
                f"{item['ss']:.4f}",
                str(item["df"]),
                f"{item['ms']:.4f}" if item["ms"] is not None else "",
                f"{item['f']:.4f}" if item["f"] is not None else "",
                f"{item['p_value']:.4e}" if item["p_value"] is not None else "",
            ])

        res_sheet = self.document.create_result_sheet(
            title="Two-Way ANOVA",
            analysis_type="Two-Way ANOVA",
            source_sheet_id=active_sheet.sheet_id,
            summary_tables=[{"title": "Two-Way ANOVA Table", "headers": headers, "rows": rows}],
            raw_result_data=res,
        )
        self.navigator.populate(self.document)
        self.on_sheet_selected(res_sheet.sheet_id)

    def action_format_table(self) -> None:
        """Opens Format Data Table dialog for active sheet."""
        active_sheet = self.document.get_sheet(self.document.active_sheet_id or "")
        if not isinstance(active_sheet, DataSheet):
            QMessageBox.warning(self, "Format Table", "Please select a Data Sheet to format.")
            return

        dlg = FormatDataTableDialog(active_sheet.table, self)
        if dlg.exec():
            self.spreadsheet_model.layoutChanged.emit()
            self._update_contextual_toolbar(active_sheet.table.table_type)
            self._on_table_data_modified()

    def action_hook_constant(self) -> None:
        """Opens Hook Constant dialog (s) to bind constants to tick lines or labels."""
        dlg = HookConstantDialog(GLOBAL_REGISTRY, self)
        if dlg.exec():
            hooked = dlg.get_selected_constant()
            if hooked:
                self.graph_canvas.add_hooked_line(hooked.constant_id)
                self.status.showMessage(f"Hooked constant: {hooked.name} = {hooked.formatted_value}")

    def action_analysis_checklist(self) -> None:
        """Opens Prism-style Analysis Checklist to assist experimental interpretation."""
        active_sheet = self.document.get_sheet(self.document.active_sheet_id or "")
        cat = "General"
        if isinstance(active_sheet, ResultSheet):
            if "Nonlinear" in active_sheet.analysis_type or "Curve" in active_sheet.analysis_type:
                cat = "Nonlinear Regression"
            elif "ANOVA" in active_sheet.analysis_type or "Comparison" in active_sheet.analysis_type or "t-test" in active_sheet.analysis_type:
                cat = "Group Comparisons (ANOVA)"
        dlg = AnalysisChecklistDialog(cat, self)
        dlg.exec()

    def action_analyze_dialog(self) -> None:
        """Opens Analyze Data dialog and routes to chosen analysis."""
        dlg = AnalyzeDataDialog(self.document, self)
        if dlg.exec():
            analysis_name, table_id, cols = dlg.get_selection()
            if table_id and table_id != self.document.active_sheet_id:
                self.on_sheet_selected(table_id)

            if not analysis_name:
                return

            action_map = {
                "Transform...": self.action_run_transform,
                "Normalize...": self.action_run_normalize,
                "Prune Rows...": self.action_run_prune,
                "Remove Baseline & Column Math...": self.action_run_remove_baseline,
                "Fraction of Total...": self.action_run_fraction_of_total,
                "Transpose...": self.action_run_transpose,
                "Extract and Rearrange...": self.action_run_extract_and_rearrange,
                "Identify Outliers (ROUT / Grubbs)...": self.action_run_outliers,
                "Descriptive Statistics / Summary...": self.action_run_descriptive_stats,
                "Correlation Matrix...": self.action_run_correlation,
                "Frequency Distribution...": self.action_run_frequency_dist,
                "Nonlinear Regression (Curve Fit)...": self.action_run_nonlinear,
                "Linear Regression...": self.action_run_linear,
                "Multiple Regression...": self.action_run_multiple_reg,
                "Logistic Regression...": self.action_run_logistic,
                "Interpolate from Standard Curve...": self.action_run_interpolate,
                "Principal Component Analysis (PCA)...": self.action_run_pca,
                "t-test (and Nonparametric)...": self.action_run_t_test,
                "One-Way ANOVA & Multiple Comparisons...": self.action_run_anova,
                "Two-Way ANOVA (Grouped Data)...": self.action_run_two_way_anova,
                "Multiple t-tests...": self.action_run_multiple_t_tests,
                "Nested t-test / Nested One-Way ANOVA...": self.action_run_nested_analysis,
                "Chi-Square / Fisher's Exact (Contingency)...": self.action_run_contingency_test,
                "Chi-Square Goodness of Fit...": self.action_run_goodness_of_fit,
                "Kaplan-Meier Survival Analysis...": self.action_run_survival,
                "Simulate XY Data...": self.action_run_simulate_xy,
                "Simulate Column Data...": self.action_run_simulate_column,
            }
            handler = action_map.get(analysis_name)
            if handler:
                handler()

    def action_run_linear(self) -> None:
        """Executes Linear Regression (Straight line) fitting on active DataSheet."""
        active_sheet = self.document.get_sheet(self.document.active_sheet_id or "")
        if not isinstance(active_sheet, DataSheet):
            QMessageBox.warning(self, "Linear Regression", "Please select a Data Sheet with XY data.")
            return

        table: DataTable = active_sheet.table
        if table.x_data is None:
            QMessageBox.warning(self, "Linear Regression", "Table does not contain an X column.")
            return

        model = STANDARD_MODELS["Straight Line (Linear)"]
        xs, ys = table.get_clean_xy(0)
        if len(xs) < 2:
            QMessageBox.warning(self, "Linear Regression", "At least 2 points required for linear regression.")
            return

        try:
            fit_result = NonlinearEngine.fit(xs, ys, model)
        except Exception as e:
            QMessageBox.critical(self, "Linear Regression Error", str(e))
            return

        headers = ["Parameter", "Best-Fit Value", "Std. Error", "95% CI Lower", "95% CI Upper"]
        rows = []
        for p in fit_result["parameters"]:
            rows.append([
                p["name"],
                f"{p['best_fit']:.4f}",
                f"{p['std_error']:.4f}",
                f"{p['ci_lower']:.4f}",
                f"{p['ci_upper']:.4f}",
            ])

        slope_val = fit_result["parameters"][0]["best_fit"]
        int_val = fit_result["parameters"][1]["best_fit"]
        sign_str = "+" if int_val >= 0 else "-"

        rows.append(["--- Equation ---", f"Y = {slope_val:.4f}*X {sign_str} {abs(int_val):.4f}", "", "", ""])
        rows.append(["R squared", f"{fit_result['r_squared']:.4f}", "", "", ""])
        rows.append(["Sy.x", f"{fit_result['syx']:.4f}", "", "", ""])

        tabs = [
            {"name": "Parameters", "grid_type": "red", "headers": headers, "rows": rows, "visible": True},
            {
                "name": "Curve Coordinates",
                "grid_type": "green",
                "headers": ["X (Fit)", "Y (Fit)"],
                "rows": [[round(x, 4), round(y, 4)] for x, y in zip(fit_result["x_dense"][:50], fit_result["y_dense"][:50])],
                "visible": True,
            },
        ]

        res_sheet = self.document.create_result_sheet(
            title="Linear Regression",
            analysis_type="Linear Regression",
            source_sheet_id=active_sheet.sheet_id,
            tabs=tabs,
            raw_result_data=fit_result,
            calc_spec={"type": "linear", "group_idx": 0},
        )
        GLOBAL_REGISTRY.extract_from_result(res_sheet)
        self.navigator.populate(self.document)
        self.on_sheet_selected(res_sheet.sheet_id)

    def action_run_transform(self) -> None:
        """Transforms active DataSheet (e.g. log10(Y))."""
        active_sheet = self.document.get_sheet(self.document.active_sheet_id or "")
        if not isinstance(active_sheet, DataSheet):
            QMessageBox.warning(self, "Transform", "Please select a Data Sheet to transform.")
            return

        new_table = transform_data_table(active_sheet.table, transform_name="log10")
        new_sheet = self.document.create_data_sheet(
            title=f"Transform of {active_sheet.title}",
            table_type=new_table.table_type,
            num_rows=new_table.num_rows,
            num_replicates=new_table.num_replicates,
            y_group_titles=new_table.y_group_titles,
            calc_spec={"type": "transform", "transform_name": "log10", "k_constant": 1.0},
        )
        new_sheet.table = new_table
        self.document.register_dependency(active_sheet.sheet_id, new_sheet.sheet_id)
        self.navigator.populate(self.document)
        self.on_sheet_selected(new_sheet.sheet_id)

    def action_run_normalize(self) -> None:
        """Normalizes active DataSheet to 0% - 100% scale."""
        active_sheet = self.document.get_sheet(self.document.active_sheet_id or "")
        if not isinstance(active_sheet, DataSheet):
            QMessageBox.warning(self, "Normalize", "Please select a Data Sheet to normalize.")
            return

        new_table = normalize_table(
            active_sheet.table,
            b0_method=Baseline0Method.SMALLEST,
            b100_method=Baseline100Method.LARGEST,
            scale=NormalizeScale.PERCENT,
        )
        new_sheet = self.document.create_data_sheet(
            title=f"Normalized of {active_sheet.title}",
            table_type=new_table.table_type,
            num_rows=new_table.num_rows,
            num_replicates=new_table.num_replicates,
            y_group_titles=new_table.y_group_titles,
            calc_spec={
                "type": "normalize",
                "b0_method": Baseline0Method.SMALLEST.value,
                "b100_method": Baseline100Method.LARGEST.value,
                "scale": NormalizeScale.PERCENT.value,
            },
        )
        new_sheet.table = new_table
        self.document.register_dependency(active_sheet.sheet_id, new_sheet.sheet_id)
        self.navigator.populate(self.document)
        self.on_sheet_selected(new_sheet.sheet_id)

    def action_run_prune(self) -> None:
        """Prunes rows of active DataSheet (decimates by factor of 2)."""
        active_sheet = self.document.get_sheet(self.document.active_sheet_id or "")
        if not isinstance(active_sheet, DataSheet):
            QMessageBox.warning(self, "Prune", "Please select a Data Sheet to prune.")
            return

        new_table = prune_rows(active_sheet.table, decimate_k=2)
        new_sheet = self.document.create_data_sheet(
            title=f"Pruned of {active_sheet.title}",
            table_type=new_table.table_type,
            num_rows=new_table.num_rows,
            num_replicates=new_table.num_replicates,
            y_group_titles=new_table.y_group_titles,
            calc_spec={"type": "prune", "decimate_k": 2},
        )
        new_sheet.table = new_table
        self.document.register_dependency(active_sheet.sheet_id, new_sheet.sheet_id)
        self.navigator.populate(self.document)
        self.on_sheet_selected(new_sheet.sheet_id)

    def action_run_remove_baseline(self) -> None:
        """Subtracts first row baseline across columns."""
        active_sheet = self.document.get_sheet(self.document.active_sheet_id or "")
        if not isinstance(active_sheet, DataSheet):
            QMessageBox.warning(self, "Remove Baseline", "Please select a Data Sheet.")
            return

        new_table = remove_baseline(
            active_sheet.table,
            method=BaselineMethod.FIRST_ROW,
            operation=BaselineOperation.SUBTRACT,
        )
        new_sheet = self.document.create_data_sheet(
            title=f"Baseline Subtracted of {active_sheet.title}",
            table_type=new_table.table_type,
            num_rows=new_table.num_rows,
            num_replicates=new_table.num_replicates,
            y_group_titles=new_table.y_group_titles,
            calc_spec={
                "type": "remove_baseline",
                "method": BaselineMethod.FIRST_ROW.value,
                "operation": BaselineOperation.SUBTRACT.value,
                "k_rows": 1,
            },
        )
        new_sheet.table = new_table
        self.document.register_dependency(active_sheet.sheet_id, new_sheet.sheet_id)
        self.navigator.populate(self.document)
        self.on_sheet_selected(new_sheet.sheet_id)

    def action_run_fraction_of_total(self) -> None:
        """Calculates fraction / percentage of total with Wilson binomial CIs."""
        active_sheet = self.document.get_sheet(self.document.active_sheet_id or "")
        if not isinstance(active_sheet, DataSheet):
            QMessageBox.warning(self, "Fraction of Total", "Please select a Data Sheet.")
            return

        new_table = fraction_of_total(active_sheet.table, scale_percent=True, ci_method="wilson")
        new_sheet = self.document.create_data_sheet(
            title=f"Fraction of Total of {active_sheet.title}",
            table_type=new_table.table_type,
            num_rows=new_table.num_rows,
            num_replicates=new_table.num_replicates,
            y_group_titles=new_table.y_group_titles,
        )
        new_sheet.table = new_table
        self.document.register_dependency(active_sheet.sheet_id, new_sheet.sheet_id)
        self.navigator.populate(self.document)
        self.on_sheet_selected(new_sheet.sheet_id)

    def action_run_transpose(self) -> None:
        """Transposes rows and columns of active DataSheet."""
        active_sheet = self.document.get_sheet(self.document.active_sheet_id or "")
        if not isinstance(active_sheet, DataSheet):
            QMessageBox.warning(self, "Transpose", "Please select a Data Sheet.")
            return

        new_table = transpose_table(active_sheet.table)
        new_sheet = self.document.create_data_sheet(
            title=f"Transpose of {active_sheet.title}",
            table_type=new_table.table_type,
            num_rows=new_table.num_rows,
            num_replicates=new_table.num_replicates,
            y_group_titles=new_table.y_group_titles,
            calc_spec={"type": "transpose"},
        )
        new_sheet.table = new_table
        self.document.register_dependency(active_sheet.sheet_id, new_sheet.sheet_id)
        self.navigator.populate(self.document)
        self.on_sheet_selected(new_sheet.sheet_id)

    def action_run_extract_and_rearrange(self) -> None:
        """Extracts and rearranges multiple variable table or copies columns."""
        active_sheet = self.document.get_sheet(self.document.active_sheet_id or "")
        if not isinstance(active_sheet, DataSheet):
            QMessageBox.warning(self, "Extract and Rearrange", "Please select a Data Sheet.")
            return

        table = active_sheet.table
        if table.table_type == TableType.MULTIPLE_VARIABLES and table.variable_metadata:
            num_vars = len(table.variable_metadata)
            y_var = table.variable_metadata[0].name
            group_var = table.variable_metadata[1].name if num_vars > 1 else None
            new_table = extract_and_rearrange(table, y_var=y_var, group_var=group_var)
        else:
            new_table = DataTable(
                table_id=f"{table.table_id}_rearranged",
                title=f"Rearranged of {table.title}",
                table_type=table.table_type,
                num_rows=table.num_rows,
                num_replicates=table.num_replicates,
                y_group_titles=table.y_group_titles,
            )
            new_table.y_data = np.copy(table.y_data)
            if table.x_data is not None:
                new_table.x_data = np.copy(table.x_data)

        new_sheet = self.document.create_data_sheet(
            title=new_table.title,
            table_type=new_table.table_type,
            num_rows=new_table.num_rows,
            num_replicates=new_table.num_replicates,
            y_group_titles=new_table.y_group_titles,
        )
        new_sheet.table = new_table
        self.document.register_dependency(active_sheet.sheet_id, new_sheet.sheet_id)
        self.navigator.populate(self.document)
        self.on_sheet_selected(new_sheet.sheet_id)

    def action_run_descriptive_stats(self) -> None:
        """Computes Column Statistics: Mean, SD, SEM, Median, Range, 95% CI."""
        active_sheet = self.document.get_sheet(self.document.active_sheet_id or "")
        if not isinstance(active_sheet, DataSheet):
            QMessageBox.warning(self, "Descriptive Statistics", "Please select a Data Sheet.")
            return

        table: DataTable = active_sheet.table
        headers = ["Group", "Sample Size (N)", "Mean", "SD", "SEM", "95% CI Lower", "95% CI Upper", "Median", "Min", "Max"]
        rows = []
        raw_stats = {}

        for g_idx, g_title in enumerate(table.y_group_titles):
            vals = []
            for r in range(table.num_rows):
                for s in range(table.num_replicates):
                    if not table.is_excluded(r, "Y", g_idx, s):
                        v = table.get_cell_value(r, "Y", g_idx, s)
                        if not np.isnan(v):
                            vals.append(v)
            if vals:
                arr = np.array(vals, dtype=np.float64)
                n = len(arr)
                mean = float(np.mean(arr))
                sd = float(np.std(arr, ddof=1)) if n > 1 else 0.0
                sem = sd / np.sqrt(n) if n > 0 else 0.0
                tcrit = scipy.stats.t.ppf(0.975, df=n - 1) if n > 1 else 0.0
                ci_lo = mean - tcrit * sem
                ci_hi = mean + tcrit * sem
                median = float(np.median(arr))
                mn = float(np.min(arr))
                mx = float(np.max(arr))

                raw_stats[g_title] = {"N": n, "mean": mean, "sd": sd, "sem": sem}
                rows.append([
                    g_title,
                    str(n),
                    f"{mean:.4f}",
                    f"{sd:.4f}",
                    f"{sem:.4f}",
                    f"{ci_lo:.4f}",
                    f"{ci_hi:.4f}",
                    f"{median:.4f}",
                    f"{mn:.4f}",
                    f"{mx:.4f}",
                ])

        tabs = [
            {"name": "Column Statistics", "grid_type": "green", "headers": headers, "rows": rows, "visible": True},
        ]

        res_sheet = self.document.create_result_sheet(
            title="Descriptive Statistics",
            analysis_type="Column Statistics",
            source_sheet_id=active_sheet.sheet_id,
            tabs=tabs,
            raw_result_data=raw_stats,
            grid_type="green",
        )
        GLOBAL_REGISTRY.extract_from_table(table)
        self.navigator.populate(self.document)
        self.on_sheet_selected(res_sheet.sheet_id)

    def action_run_correlation(self) -> None:
        """Computes Pearson correlation matrix and p-values."""
        active_sheet = self.document.get_sheet(self.document.active_sheet_id or "")
        if not isinstance(active_sheet, DataSheet):
            QMessageBox.warning(self, "Correlation Matrix", "Please select a Data Sheet.")
            return

        table: DataTable = active_sheet.table
        group_names = list(table.y_group_titles)
        if len(group_names) < 2:
            QMessageBox.warning(self, "Correlation Matrix", "Requires at least 2 columns.")
            return

        cols = []
        for g_idx in range(len(group_names)):
            col_vals = [table.get_cell_value(r, "Y", g_idx, 0) for r in range(table.num_rows)]
            cols.append(col_vals)

        r_headers = [""] + group_names
        r_rows = []
        p_rows = []

        for i, name_i in enumerate(group_names):
            r_row = [name_i]
            p_row = [name_i]
            for j, name_j in enumerate(group_names):
                v1 = np.array(cols[i], dtype=np.float64)
                v2 = np.array(cols[j], dtype=np.float64)
                valid = ~np.isnan(v1) & ~np.isnan(v2)
                if np.sum(valid) >= 3:
                    r_val, p_val = scipy.stats.pearsonr(v1[valid], v2[valid])
                    r_row.append(f"{r_val:.4f}")
                    p_row.append(f"{p_val:.4e}")
                else:
                    r_row.append("1.0000" if i == j else "NaN")
                    p_row.append("0.0000" if i == j else "NaN")
            r_rows.append(r_row)
            p_rows.append(p_row)

        tabs = [
            {"name": "Correlation Matrix (r)", "grid_type": "green", "headers": r_headers, "rows": r_rows, "visible": True},
            {"name": "P-values (two-tailed)", "grid_type": "red", "headers": r_headers, "rows": p_rows, "visible": True},
        ]

        res_sheet = self.document.create_result_sheet(
            title="Correlation Matrix",
            analysis_type="Correlation",
            source_sheet_id=active_sheet.sheet_id,
            tabs=tabs,
            grid_type="green",
        )
        self.navigator.populate(self.document)
        self.on_sheet_selected(res_sheet.sheet_id)

    def action_run_frequency_dist(self) -> None:
        """Computes Frequency Distribution / Histogram binning."""
        active_sheet = self.document.get_sheet(self.document.active_sheet_id or "")
        if not isinstance(active_sheet, DataSheet):
            QMessageBox.warning(self, "Frequency Distribution", "Please select a Data Sheet.")
            return

        table: DataTable = active_sheet.table
        all_vals = []
        for g_idx in range(len(table.y_group_titles)):
            for r in range(table.num_rows):
                for s in range(table.num_replicates):
                    if not table.is_excluded(r, "Y", g_idx, s):
                        v = table.get_cell_value(r, "Y", g_idx, s)
                        if not np.isnan(v):
                            all_vals.append(v)

        if len(all_vals) < 3:
            QMessageBox.warning(self, "Frequency Distribution", "Requires at least 3 values.")
            return

        counts, bin_edges = np.histogram(all_vals, bins="auto")
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        total = np.sum(counts)
        cum_counts = np.cumsum(counts)

        headers = ["Bin Center", "Bin Start", "Bin End", "Count", "Relative %", "Cumulative %"]
        rows = []
        for i in range(len(counts)):
            rel_pct = (counts[i] / total) * 100.0 if total > 0 else 0.0
            cum_pct = (cum_counts[i] / total) * 100.0 if total > 0 else 0.0
            rows.append([
                f"{bin_centers[i]:.4f}",
                f"{bin_edges[i]:.4f}",
                f"{bin_edges[i+1]:.4f}",
                str(counts[i]),
                f"{rel_pct:.2f}%",
                f"{cum_pct:.2f}%",
            ])

        tabs = [
            {"name": "Frequency Distribution", "grid_type": "green", "headers": headers, "rows": rows, "visible": True},
        ]

        res_sheet = self.document.create_result_sheet(
            title="Frequency Distribution",
            analysis_type="Frequency Distribution",
            source_sheet_id=active_sheet.sheet_id,
            tabs=tabs,
            grid_type="green",
        )
        self.navigator.populate(self.document)
        self.on_sheet_selected(res_sheet.sheet_id)

    def action_run_multiple_reg(self) -> None:
        """Computes Multiple Linear Regression (OLS)."""
        active_sheet = self.document.get_sheet(self.document.active_sheet_id or "")
        if not isinstance(active_sheet, DataSheet):
            QMessageBox.warning(self, "Multiple Regression", "Please select a Data Sheet.")
            return

        table: DataTable = active_sheet.table
        if len(table.y_group_titles) < 2:
            QMessageBox.warning(self, "Multiple Regression", "Requires at least 2 columns (Y and X1, X2...).")
            return

        y_name = table.y_group_titles[0]
        x_names = table.y_group_titles[1:]

        y_vals = [table.get_cell_value(r, "Y", 0, 0) for r in range(table.num_rows)]
        x_matrix = []
        for r in range(table.num_rows):
            row_x = [table.get_cell_value(r, "Y", g, 0) for g in range(1, len(table.y_group_titles))]
            x_matrix.append(row_x)

        y_arr = np.array(y_vals, dtype=np.float64)
        x_arr = np.array(x_matrix, dtype=np.float64)
        valid = ~np.isnan(y_arr) & ~np.any(np.isnan(x_arr), axis=1)

        y_clean = y_arr[valid]
        x_clean = x_arr[valid]
        n = len(y_clean)
        k = x_clean.shape[1]

        if n <= k + 1:
            QMessageBox.warning(self, "Multiple Regression", f"Need more rows (N={n}) than parameters (K={k+1}).")
            return

        X = np.column_stack([np.ones(n), x_clean])
        beta, residuals, rank, s = np.linalg.lstsq(X, y_clean, rcond=None)
        y_pred = X @ beta
        res = y_clean - y_pred
        ss_res = np.sum(res ** 2)
        ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
        df_res = n - (k + 1)
        df_model = k
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        adj_r_squared = 1.0 - ((ss_res / df_res) / (ss_tot / (n - 1))) if ss_tot > 0 and df_res > 0 else 0.0
        syx = np.sqrt(ss_res / df_res) if df_res > 0 else 0.0

        cov_matrix = np.linalg.pinv(X.T @ X) * (ss_res / df_res)
        se = np.sqrt(np.diag(cov_matrix))
        t_stats = beta / np.where(se > 0, se, 1e-12)
        p_vals = 2.0 * (1.0 - scipy.stats.t.cdf(np.abs(t_stats), df=df_res))

        coef_headers = ["Variable", "Coefficient", "Std. Error", "t ratio", "p-value"]
        coef_rows = [["Intercept", f"{beta[0]:.4f}", f"{se[0]:.4f}", f"{t_stats[0]:.4f}", f"{p_vals[0]:.4e}"]]
        for i, name in enumerate(x_names):
            coef_rows.append([
                name,
                f"{beta[i+1]:.4f}",
                f"{se[i+1]:.4f}",
                f"{t_stats[i+1]:.4f}",
                f"{p_vals[i+1]:.4e}",
            ])

        f_stat = ((ss_tot - ss_res) / df_model) / (ss_res / df_res) if df_res > 0 and ss_res > 0 else 0.0
        f_pval = 1.0 - scipy.stats.f.cdf(f_stat, df_model, df_res)

        summary_headers = ["Metric", "Value"]
        summary_rows = [
            ["Dependent Variable (Y)", y_name],
            ["Sample Size (N)", str(n)],
            ["R squared", f"{r_squared:.4f}"],
            ["Adjusted R squared", f"{adj_r_squared:.4f}"],
            ["Sy.x (Std Error of Fit)", f"{syx:.4f}"],
            ["F statistic", f"{f_stat:.4f}"],
            ["F test p-value", f"{f_pval:.4e}"],
        ]

        tabs = [
            {"name": "Coefficients", "grid_type": "red", "headers": coef_headers, "rows": coef_rows, "visible": True},
            {"name": "Model Summary", "grid_type": "red", "headers": summary_headers, "rows": summary_rows, "visible": True},
        ]

        res_sheet = self.document.create_result_sheet(
            title="Multiple Regression",
            analysis_type="Multiple Regression",
            source_sheet_id=active_sheet.sheet_id,
            tabs=tabs,
            grid_type="red",
        )
        self.navigator.populate(self.document)
        self.on_sheet_selected(res_sheet.sheet_id)

    def action_run_logistic(self) -> None:
        """Fits Sigmoidal / Logistic dose-response model."""
        self.action_run_nonlinear()

    def action_run_interpolate(self) -> None:
        """Interpolates unknown values from fitted standard curve."""
        active_sheet = self.document.get_sheet(self.document.active_sheet_id or "")
        if not isinstance(active_sheet, DataSheet):
            QMessageBox.warning(self, "Interpolate", "Please select a Data Sheet.")
            return

        table: DataTable = active_sheet.table
        xs, ys = table.get_clean_xy(0)
        if len(xs) < 2:
            QMessageBox.warning(self, "Interpolate", "Requires standard curve points.")
            return

        slope, intercept, r_val, p_val, std_err = scipy.stats.linregress(xs, ys)

        headers = ["Row", "Y Value", "Interpolated X", "Standard Error"]
        rows = []
        for r in range(table.num_rows):
            y_val = table.get_cell_value(r, "Y", 0, 0)
            if not np.isnan(y_val) and slope != 0:
                calc_x = (y_val - intercept) / slope
                rows.append([f"Row {r+1}", f"{y_val:.4f}", f"{calc_x:.4f}", f"{std_err:.4f}"])

        tabs = [
            {"name": "Interpolated Values", "grid_type": "green", "headers": headers, "rows": rows, "visible": True},
        ]

        res_sheet = self.document.create_result_sheet(
            title="Interpolate Unknowns",
            analysis_type="Interpolate",
            source_sheet_id=active_sheet.sheet_id,
            tabs=tabs,
            grid_type="green",
        )
        self.navigator.populate(self.document)
        self.on_sheet_selected(res_sheet.sheet_id)

    def action_run_t_test(self) -> None:
        """Executes unpaired t-test (Student's and Welch's)."""
        active_sheet = self.document.get_sheet(self.document.active_sheet_id or "")
        if not isinstance(active_sheet, DataSheet):
            QMessageBox.warning(self, "t-test", "Please select a Data Sheet.")
            return

        table: DataTable = active_sheet.table
        all_groups = table.get_all_group_values()
        valid_keys = [k for k, v in all_groups.items() if len(v) >= 2]
        if len(valid_keys) < 2:
            QMessageBox.warning(self, "t-test", "Requires at least 2 groups with N >= 2 points each.")
            return

        g1_name = valid_keys[0]
        g2_name = valid_keys[1]
        g1 = np.array(all_groups[g1_name], dtype=np.float64)
        g2 = np.array(all_groups[g2_name], dtype=np.float64)

        t_stud, p_stud = scipy.stats.ttest_ind(g1, g2, equal_var=True)
        t_welch, p_welch = scipy.stats.ttest_ind(g1, g2, equal_var=False)

        mean_diff = float(np.mean(g1) - np.mean(g2))
        df_stud = len(g1) + len(g2) - 2

        headers = ["Test / Parameter", "Value"]
        rows = [
            ["Comparison", f"{g1_name} vs {g2_name}"],
            ["Difference between means", f"{mean_diff:.4f}"],
            ["--- Unpaired t-test (Equal variance assumed) ---", ""],
            ["t ratio", f"{t_stud:.4f}"],
            ["df", str(df_stud)],
            ["p-value (two-tailed)", f"{p_stud:.4e}"],
            ["Significant (alpha = 0.05)?", "Yes" if p_stud < 0.05 else "No (ns)"],
            ["--- Welch's t-test (Unequal variance) ---", ""],
            ["t ratio", f"{t_welch:.4f}"],
            ["p-value (two-tailed)", f"{p_welch:.4e}"],
            ["Significant (alpha = 0.05)?", "Yes" if p_welch < 0.05 else "No (ns)"],
        ]

        tabs = [
            {"name": "Unpaired t-test", "grid_type": "red", "headers": headers, "rows": rows, "visible": True},
        ]

        res_sheet = self.document.create_result_sheet(
            title=f"t-test ({g1_name} vs {g2_name})",
            analysis_type="t-test",
            source_sheet_id=active_sheet.sheet_id,
            tabs=tabs,
            grid_type="red",
        )
        self.navigator.populate(self.document)
        self.on_sheet_selected(res_sheet.sheet_id)

    def action_run_multiple_t_tests(self) -> None:
        """Executes row-by-row multiple t-tests with Benjamini-Hochberg FDR correction."""
        active_sheet = self.document.get_sheet(self.document.active_sheet_id or "")
        if not isinstance(active_sheet, DataSheet):
            QMessageBox.warning(self, "Multiple t-tests", "Please select a Data Sheet.")
            return

        table: DataTable = active_sheet.table
        if len(table.y_group_titles) < 2 or table.num_replicates < 2:
            QMessageBox.warning(self, "Multiple t-tests", "Requires at least 2 groups and replicates >= 2.")
            return

        rows = []
        raw_pvals = []
        for r in range(table.num_rows):
            g1_reps = [table.get_cell_value(r, "Y", 0, s) for s in range(table.num_replicates) if not np.isnan(table.get_cell_value(r, "Y", 0, s))]
            g2_reps = [table.get_cell_value(r, "Y", 1, s) for s in range(table.num_replicates) if not np.isnan(table.get_cell_value(r, "Y", 1, s))]
            if len(g1_reps) >= 2 and len(g2_reps) >= 2:
                t_stat, p_val = scipy.stats.ttest_ind(g1_reps, g2_reps)
                diff = np.mean(g1_reps) - np.mean(g2_reps)
                rows.append([f"Row {r+1}", f"{diff:.4f}", f"{t_stat:.4f}", p_val])
                raw_pvals.append(p_val)

        if not rows:
            QMessageBox.warning(self, "Multiple t-tests", "Not enough replicate data per row.")
            return

        sorted_indices = np.argsort(raw_pvals)
        m = len(raw_pvals)
        adj_pvals = np.zeros(m)
        for rank, idx in enumerate(sorted_indices, 1):
            adj_pvals[idx] = min(raw_pvals[idx] * m / rank, 1.0)

        fmt_rows = []
        for i, row in enumerate(rows):
            adj_p = adj_pvals[i]
            sig = "Yes (q < 0.05)" if adj_p < 0.05 else "No (ns)"
            fmt_rows.append([row[0], row[1], row[2], f"{row[3]:.4e}", f"{adj_p:.4e}", sig])

        headers = ["Row", "Mean Diff", "t ratio", "p-value", "FDR Adjusted p (q)", "Significant?"]
        tabs = [
            {"name": "Multiple t-tests", "grid_type": "red", "headers": headers, "rows": fmt_rows, "visible": True},
        ]

        res_sheet = self.document.create_result_sheet(
            title="Multiple t-tests",
            analysis_type="Multiple t-tests",
            source_sheet_id=active_sheet.sheet_id,
            tabs=tabs,
            grid_type="red",
        )
        self.navigator.populate(self.document)
        self.on_sheet_selected(res_sheet.sheet_id)

    def action_run_nested_analysis(self) -> None:
        """Executes Nested One-Way ANOVA / Nested t-test."""
        active_sheet = self.document.get_sheet(self.document.active_sheet_id or "")
        if not isinstance(active_sheet, DataSheet):
            QMessageBox.warning(self, "Nested Analysis", "Please select a Data Sheet.")
            return

        table: DataTable = active_sheet.table
        all_groups = table.get_all_group_values()
        valid_groups = {k: v for k, v in all_groups.items() if len(v) > 0}
        if len(valid_groups) < 2:
            QMessageBox.warning(self, "Nested Analysis", "Requires at least 2 groups.")
            return

        res = AnovaEngine.one_way(valid_groups)
        headers = ["Source of Variation", "Sum of Squares", "Degrees of Freedom", "Mean Square", "F", "p-value"]
        rows = []
        for item in res["anova_table"]:
            rows.append([
                f"Nested: {item['source']}",
                f"{item['ss']:.4f}",
                str(item["df"]),
                f"{item['ms']:.4f}" if item["ms"] is not None else "",
                f"{item['f']:.4f}" if item["f"] is not None else "",
                f"{item['p_value']:.4e}" if item["p_value"] is not None else "",
            ])

        tabs = [
            {"name": "Nested ANOVA Table", "grid_type": "red", "headers": headers, "rows": rows, "visible": True},
        ]

        res_sheet = self.document.create_result_sheet(
            title="Nested Analysis",
            analysis_type="Nested ANOVA",
            source_sheet_id=active_sheet.sheet_id,
            tabs=tabs,
            raw_result_data=res,
            grid_type="red",
        )
        self.navigator.populate(self.document)
        self.on_sheet_selected(res_sheet.sheet_id)

    def action_run_survival(self) -> None:
        """Computes Kaplan-Meier survival curve with Greenwood standard errors."""
        active_sheet = self.document.get_sheet(self.document.active_sheet_id or "")
        if not isinstance(active_sheet, DataSheet):
            QMessageBox.warning(self, "Survival Analysis", "Please select a Data Sheet.")
            return

        table: DataTable = active_sheet.table
        if table.x_data is None:
            QMessageBox.warning(self, "Survival Analysis", "Survival analysis requires X column as elapsed time.")
            return

        times = np.copy(table.x_data)
        events = np.array([table.get_cell_value(r, "Y", 0, 0) for r in range(table.num_rows)])
        valid = ~np.isnan(times) & ~np.isnan(events)

        t_clean = times[valid]
        e_clean = events[valid]
        order = np.argsort(t_clean)
        t_sorted = t_clean[order]
        e_sorted = e_clean[order]

        unique_times = np.unique(t_sorted)
        n_at_risk = len(t_sorted)
        survival_frac = 1.0
        greenwood_sum = 0.0

        headers = ["Time", "Number at Risk", "Number of Deaths", "Censored", "Fractional Survival", "Std Error"]
        rows = []
        curve_coords = [[0.0, 1.0]]

        for ut in unique_times:
            mask = t_sorted == ut
            d = int(np.sum(e_sorted[mask] == 1))
            c = int(np.sum(e_sorted[mask] == 0))
            if n_at_risk > 0:
                if d > 0:
                    survival_frac *= (1.0 - d / n_at_risk)
                    if n_at_risk > d:
                        greenwood_sum += d / (n_at_risk * (n_at_risk - d))
                se = survival_frac * np.sqrt(greenwood_sum)
                rows.append([
                    f"{ut:.4f}",
                    str(n_at_risk),
                    str(d),
                    str(c),
                    f"{survival_frac:.4f}",
                    f"{se:.4f}",
                ])
                curve_coords.append([float(ut), float(survival_frac)])
                n_at_risk -= (d + c)

        tabs = [
            {"name": "Survival Table", "grid_type": "green", "headers": headers, "rows": rows, "visible": True},
            {
                "name": "Curve Coordinates",
                "grid_type": "green",
                "headers": ["Time", "Survival Fraction"],
                "rows": [[f"{x:.4f}", f"{y:.4f}"] for x, y in curve_coords],
                "visible": True,
            },
        ]

        res_sheet = self.document.create_result_sheet(
            title="Kaplan-Meier Survival",
            analysis_type="Survival Analysis",
            source_sheet_id=active_sheet.sheet_id,
            tabs=tabs,
            grid_type="green",
        )
        self.navigator.populate(self.document)
        self.on_sheet_selected(res_sheet.sheet_id)

    def action_run_contingency_test(self) -> None:
        """Executes Chi-Square and Fisher's exact contingency test."""
        active_sheet = self.document.get_sheet(self.document.active_sheet_id or "")
        if not isinstance(active_sheet, DataSheet):
            QMessageBox.warning(self, "Contingency Test", "Please select a Contingency Data Sheet.")
            return

        table: DataTable = active_sheet.table
        data_matrix = []
        for r in range(table.num_rows):
            row_vals = [table.get_cell_value(r, "Y", g, 0) for g in range(len(table.y_group_titles))]
            if all(~np.isnan(row_vals)):
                data_matrix.append(row_vals)

        arr = np.array(data_matrix, dtype=np.float64)
        if arr.shape[0] < 2 or arr.shape[1] < 2:
            QMessageBox.warning(self, "Contingency Test", "Contingency table requires at least 2 rows and 2 columns.")
            return

        chi2, p_val, dof, expected = scipy.stats.chi2_contingency(arr)
        headers = ["Test / Statistic", "Value"]
        rows = [
            ["Chi-square", f"{chi2:.4f}"],
            ["Degrees of freedom", str(dof)],
            ["p-value (two-tailed)", f"{p_val:.4e}"],
            ["Significant (alpha = 0.05)?", "Yes" if p_val < 0.05 else "No (ns)"],
        ]

        if arr.shape == (2, 2):
            odds_ratio, p_fisher = scipy.stats.fisher_exact(arr)
            rows.append(["--- Fisher's Exact Test ---", ""])
            rows.append(["Fisher's exact p-value", f"{p_fisher:.4e}"])
            rows.append(["Odds Ratio", f"{odds_ratio:.4f}"])

        tabs = [
            {"name": "Contingency Table Analysis", "grid_type": "red", "headers": headers, "rows": rows, "visible": True},
        ]

        res_sheet = self.document.create_result_sheet(
            title="Contingency Table Analysis",
            analysis_type="Contingency Test",
            source_sheet_id=active_sheet.sheet_id,
            tabs=tabs,
            grid_type="red",
        )
        self.navigator.populate(self.document)
        self.on_sheet_selected(res_sheet.sheet_id)

    def action_run_goodness_of_fit(self) -> None:
        """Executes Chi-Square Goodness-of-Fit test for Parts of Whole data."""
        active_sheet = self.document.get_sheet(self.document.active_sheet_id or "")
        if not isinstance(active_sheet, DataSheet):
            QMessageBox.warning(self, "Goodness-of-Fit", "Please select a Data Sheet.")
            return

        table: DataTable = active_sheet.table
        obs = [table.get_cell_value(r, "Y", 0, 0) for r in range(table.num_rows) if not np.isnan(table.get_cell_value(r, "Y", 0, 0))]
        if len(obs) < 2:
            QMessageBox.warning(self, "Goodness-of-Fit", "Requires at least 2 categories.")
            return

        chi2, p_val = scipy.stats.chisquare(obs)
        headers = ["Metric", "Value"]
        rows = [
            ["Chi-square statistic", f"{chi2:.4f}"],
            ["Degrees of freedom", str(len(obs) - 1)],
            ["p-value", f"{p_val:.4e}"],
            ["Significant difference from equal proportions?", "Yes" if p_val < 0.05 else "No (ns)"],
        ]

        tabs = [
            {"name": "Goodness of Fit", "grid_type": "red", "headers": headers, "rows": rows, "visible": True},
        ]

        res_sheet = self.document.create_result_sheet(
            title="Goodness-of-Fit",
            analysis_type="Goodness-of-Fit",
            source_sheet_id=active_sheet.sheet_id,
            tabs=tabs,
            grid_type="red",
        )
        self.navigator.populate(self.document)
        self.on_sheet_selected(res_sheet.sheet_id)

    def action_run_pca(self) -> None:
        """Computes Principal Component Analysis (PCA)."""
        active_sheet = self.document.get_sheet(self.document.active_sheet_id or "")
        if not isinstance(active_sheet, DataSheet):
            QMessageBox.warning(self, "PCA", "Please select a Data Sheet.")
            return

        table: DataTable = active_sheet.table
        cols = []
        for g_idx in range(len(table.y_group_titles)):
            col_vals = [table.get_cell_value(r, "Y", g_idx, 0) for r in range(table.num_rows)]
            cols.append(col_vals)

        X = np.array(cols, dtype=np.float64).T
        valid_rows = ~np.any(np.isnan(X), axis=1)
        X_clean = X[valid_rows]

        if X_clean.shape[0] < 3 or X_clean.shape[1] < 2:
            QMessageBox.warning(self, "PCA", "Requires at least 3 rows and 2 numeric columns.")
            return

        X_std = (X_clean - np.mean(X_clean, axis=0)) / np.where(np.std(X_clean, axis=0, ddof=1) > 0, np.std(X_clean, axis=0, ddof=1), 1.0)
        cov = np.cov(X_std, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        total_var = np.sum(eigenvalues)
        var_pct = (eigenvalues / total_var) * 100.0 if total_var > 0 else np.zeros(len(eigenvalues))
        cum_var = np.cumsum(var_pct)

        scores = X_std @ eigenvectors
        score_headers = [f"PC{i+1}" for i in range(min(5, scores.shape[1]))]
        score_rows = [[f"{scores[r, c]:.4f}" for c in range(len(score_headers))] for r in range(scores.shape[0])]

        summary_headers = ["Principal Component", "Eigenvalue", "% Variance", "Cumulative %"]
        summary_rows = [
            [f"PC{i+1}", f"{eigenvalues[i]:.4f}", f"{var_pct[i]:.2f}%", f"{cum_var[i]:.2f}%"]
            for i in range(len(eigenvalues))
        ]

        tabs = [
            {"name": "PC Scores", "grid_type": "green", "headers": score_headers, "rows": score_rows, "visible": True},
            {"name": "Eigenvalues & Variance", "grid_type": "red", "headers": summary_headers, "rows": summary_rows, "visible": True},
        ]

        res_sheet = self.document.create_result_sheet(
            title="Principal Component Analysis",
            analysis_type="PCA",
            source_sheet_id=active_sheet.sheet_id,
            tabs=tabs,
            grid_type="green",
        )
        self.navigator.populate(self.document)
        self.on_sheet_selected(res_sheet.sheet_id)

    def action_run_simulate_xy(self) -> None:
        """Simulates synthetic dose-response XY dataset with Gaussian noise."""
        sheet = self.document.create_data_sheet(
            title="Simulated Dose Response",
            table_type=TableType.XY,
            num_rows=8,
            num_replicates=3,
            y_group_titles=["Control", "Treated"],
        )
        tbl = sheet.table
        tbl.x_title = "Log [Agonist], M"
        x_vals = [-9.0, -8.0, -7.0, -6.5, -6.0, -5.5, -5.0, -4.0]
        np.random.seed(42)
        for r, x in enumerate(x_vals):
            tbl.set_cell_value(r, "X", 0, 0, x)
            y_ctrl = 5.0 + 95.0 / (1.0 + 10.0 ** (-6.5 - x))
            y_trt = 5.0 + 65.0 / (1.0 + 10.0 ** (-5.5 - x))
            for s in range(3):
                tbl.set_cell_value(r, "Y", 0, s, round(float(y_ctrl + np.random.normal(0, 2.0)), 2))
                tbl.set_cell_value(r, "Y", 1, s, round(float(y_trt + np.random.normal(0, 2.0)), 2))

        self.navigator.populate(self.document)
        self.on_sheet_selected(sheet.sheet_id)
        self.status.showMessage("Simulated XY data generated.")

    def action_run_simulate_column(self) -> None:
        """Simulates synthetic multi-group column data."""
        sheet = self.document.create_data_sheet(
            title="Simulated Column Data",
            table_type=TableType.COLUMN,
            num_rows=10,
            num_replicates=1,
            y_group_titles=["Vehicle", "Dose 1mg", "Dose 10mg"],
        )
        tbl = sheet.table
        np.random.seed(42)
        means = [25.0, 38.0, 52.0]
        sds = [3.5, 4.2, 5.0]
        for g_idx in range(3):
            for r in range(10):
                val = float(np.random.normal(means[g_idx], sds[g_idx]))
                tbl.set_cell_value(r, "Y", g_idx, 0, round(val, 2))

        self.navigator.populate(self.document)
        self.on_sheet_selected(sheet.sheet_id)
        self.status.showMessage("Simulated column data generated.")

    def action_about(self) -> None:
        QMessageBox.about(
            self,
            "About GemGraph",
            "<h3>GemGraph Clean-Room Edition 1.0</h3>"
            "<p>Production-Grade GraphPad Prism Alternative.</p>"
            "<p>Designed from clean-room reverse engineering specifications.</p>"
            "<p>Strict-Local Non-Root User-Space Architecture.</p>",
        )
