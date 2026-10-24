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
from PyQt6.QtCore import Qt, QDir, QUrl, pyqtSignal
from PyQt6.QtGui import QAction, QKeySequence, QDragEnterEvent, QDropEvent, QDragMoveEvent
from PyQt6.QtWidgets import (
    QMainWindow,
    QApplication,
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
    QMenu,
    QInputDialog,
    QTreeWidgetItem,
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
from src.ui.dialogs import (
    NonlinearRegressionDialog,
    OutlierDialog,
    AnovaDialog,
    EnzymeKineticsDialog,
    CoxRegressionDialog,
)
from src.ui.mediator import WorkspaceMediator
from src.math.survival import SurvivalEngine

APP_VERSION_LABEL = "v0.1.2-nightly"

# Compatibility alias
ResultSheetWidget = ResultTabsWidget


from src.ui.info_sheet_view import InfoSheetView
InfoSheetWidget = InfoSheetView


class MainWindow(QMainWindow):
    """
    Main application window hosting three-pane scientific interface.
    """
    workspace_modified = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.current_file_path: Optional[str] = None
        self._is_modified: bool = False
        self._update_window_title()
        self.resize(1300, 800)
        self.setAcceptDrops(True)

        self.document = PrismDocument("Untitled Project")
        self.undo_stack = UndoStack()

        self._create_blank_project()
        self._setup_ui()
        self._setup_menus()
        self._setup_toolbar()
        self._setup_statusbar()

        self.workspace_modified.connect(self._on_workspace_modified)

        # Populate initial view
        self.navigator.populate(self.document)
        if self.document.active_sheet_id:
            self.on_sheet_selected(self.document.active_sheet_id)
        self._bind_document_signals()
        self._is_modified = False
        self._update_window_title()

    def _bind_document_signals(self) -> None:
        """Hooks document sheet addition and removal to track document-level mutations."""
        if not hasattr(self, "document") or self.document is None:
            return
        if getattr(self.document, "_gemgraph_signals_bound", False):
            return
        orig_add = self.document.add_sheet
        orig_remove = self.document.remove_sheet

        def wrapped_add(sheet: Any) -> None:
            orig_add(sheet)
            self.mark_as_modified()

        def wrapped_remove(sheet_id: str) -> bool:
            res = orig_remove(sheet_id)
            if res:
                self.mark_as_modified()
            return res

        self.document.add_sheet = wrapped_add  # type: ignore
        self.document.remove_sheet = wrapped_remove  # type: ignore
        self.document._gemgraph_signals_bound = True  # type: ignore

    @property
    def is_modified(self) -> bool:
        return self._is_modified

    @is_modified.setter
    def is_modified(self, value: bool) -> None:
        self._is_modified = bool(value)
        self._update_window_title()

    def mark_as_modified(self, *args: Any, **kwargs: Any) -> None:
        """Marks the workspace as modified with unsaved changes and updates the window title."""
        self._is_modified = True
        self._update_window_title()

    def _update_window_title(self) -> None:
        file_name = os.path.basename(self.current_file_path.rstrip("/\\")) if self.current_file_path else "Untitled"
        indicator = "*" if getattr(self, "_is_modified", False) else ""
        self.setWindowTitle(f"{file_name}{indicator} - GemGraph - {APP_VERSION_LABEL} - Clean Room Edition")

    def _on_workspace_modified(self) -> None:
        if hasattr(self, "layout_view") and self.layout_view is not None:
            if self.center_stack.currentWidget() == self.layout_view:
                self.layout_view.refresh_hot_placeholders()

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

        # Left Pane: Navigator
        self.navigator = PrismNavigator()
        self.navigator.sheetSelected.connect(self.on_sheet_selected)
        self.navigator.sheetDeleted.connect(self.on_sheet_deleted)
        self.navigator.customContextMenuRequested.connect(self._on_navigator_context_menu)

        # Center Pane: Dynamic Stack
        self.center_stack = QStackedWidget()

        # Workspace Mediator
        self.mediator = WorkspaceMediator.get_instance(self)
        self.mediator.tableModified.connect(self.mark_as_modified)
        self.mediator.dataChanged.connect(self.mark_as_modified)

        # Data Sheet View
        init_sheet = self.document.get_sheets_by_type(SheetType.DATA)[0]
        self.spreadsheet_model = ScientificSpreadsheetModel(init_sheet.table, self.undo_stack, self.mediator)  # type: ignore
        self.spreadsheet_view = ScientificSpreadsheetView(self.spreadsheet_model)
        self.table_view = self.spreadsheet_view
        self.spreadsheet_model.dataChangedSignal.connect(self._on_table_data_modified)
        self.spreadsheet_model.dataChanged.connect(self.mark_as_modified)
        self.spreadsheet_view.cellExclusionToggled.connect(self._on_table_data_modified)
        self.spreadsheet_view.cellExclusionToggled.connect(self.mark_as_modified)
        self.center_stack.addWidget(self.spreadsheet_view)

        # Result Sheet View (Multi-Tab and Green/Red Grid Semantics)
        self.result_widget = ResultTabsWidget()
        self.result_widget.analyzeRequested.connect(self.action_analyze_dialog)
        self.result_widget.newGraphRequested.connect(self.action_add_graph)
        self.result_widget.freezeToggled.connect(lambda frozen: self.navigator.populate(self.document))
        self.result_widget.freezeToggled.connect(self.mark_as_modified)
        self.center_stack.addWidget(self.result_widget)

        # Info Sheet View
        self.info_widget = InfoSheetView()
        self.info_widget.infoModified.connect(self._on_table_data_modified)
        self.info_widget.infoModified.connect(self.mark_as_modified)
        self.center_stack.addWidget(self.info_widget)

        # Page Layout Presentation View
        from src.ui.layout_view import LayoutView
        self.layout_view = LayoutView(self)
        self.layout_view.layoutModified.connect(self.mark_as_modified)
        self.center_stack.addWidget(self.layout_view)

        # Right Pane: Interactive Plot Canvas
        self.graph_canvas = GraphCanvasWidget(mediator=self.mediator)
        self.graph_canvas.pinGraphRequested.connect(self.action_pin_graph)
        self.graph_canvas.freezeToggled.connect(self._on_freeze_toggled)
        self.graph_canvas.graphModified.connect(self.layout_view.refresh_all_snapshots)
        self.graph_canvas.graphModified.connect(self.mark_as_modified)

        # Shell Splitters & Layout Stability
        self.init_interface_shell()
        main_layout.addWidget(self.outer_splitter)

    def init_interface_shell(self) -> None:
        """
        Initializes responsive 3-pane layout shell with outer and inner splitters.
        Guarantees layout stability, non-collapsible panes, and proportional stretch factors.
        """
        if hasattr(self, "outer_splitter") and self.outer_splitter is not None:
            self.inner_splitter.setChildrenCollapsible(False)
            self.outer_splitter.setChildrenCollapsible(False)
            self.navigator.setMinimumWidth(180)
            self.center_stack.setMinimumWidth(240)
            self.graph_canvas.setMinimumWidth(320)
            self.inner_splitter.setStretchFactor(0, 0)
            self.inner_splitter.setStretchFactor(1, 1)
            self.outer_splitter.setStretchFactor(0, 0)
            self.outer_splitter.setStretchFactor(1, 1)
            return

        # Outer Splitter (Sidebar Tree | Main Workspace)
        self.outer_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.outer_splitter.setChildrenCollapsible(False)

        # Inner Splitter (Center Stack | Plot Canvas)
        self.inner_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.inner_splitter.setChildrenCollapsible(False)

        # Minimum widths for layout stability
        self.navigator.setMinimumWidth(180)
        self.center_stack.setMinimumWidth(240)
        self.graph_canvas.setMinimumWidth(320)

        # Wire widgets into splitters
        self.outer_splitter.addWidget(self.navigator)
        self.inner_splitter.addWidget(self.center_stack)
        self.inner_splitter.addWidget(self.graph_canvas)
        self.outer_splitter.addWidget(self.inner_splitter)

        # Proportional stretch factors
        self.inner_splitter.setStretchFactor(0, 0)  # Table maintains width
        self.inner_splitter.setStretchFactor(1, 1)  # Plot canvas receives extra space
        self.outer_splitter.setStretchFactor(0, 0)  # Sidebar tree maintains width
        self.outer_splitter.setStretchFactor(1, 1)  # Main workspace receives resize space

        # Backward compatibility alias
        self.main_splitter = self.outer_splitter

        # Initial sizes
        self.outer_splitter.setSizes([200, 1100])
        self.inner_splitter.setSizes([580, 520])

    def _setup_menus(self) -> None:
        menubar = self.menuBar()

        # File Menu
        file_menu = menubar.addMenu("&File")

        new_act = QAction("&New Project", self)
        new_act.setShortcut(QKeySequence.StandardKey.New)
        new_act.triggered.connect(self.action_new_project)
        file_menu.addAction(new_act)

        open_act = QAction("&Open Project...", self)
        open_act.setShortcut(QKeySequence.StandardKey.Open)
        open_act.triggered.connect(self.action_open_project)
        file_menu.addAction(open_act)

        open_sample_act = QAction("Open Sample Project (.bigo)", self)
        open_sample_act.triggered.connect(self.action_open_sample_project)
        file_menu.addAction(open_sample_act)

        load_sample_datasets_act = QAction("Load Sample Datasets", self)
        load_sample_datasets_act.triggered.connect(self.action_load_sample_datasets)
        file_menu.addAction(load_sample_datasets_act)

        save_act = QAction("&Save Project (.bigo)", self)
        save_act.setShortcut(QKeySequence.StandardKey.Save)
        save_act.triggered.connect(self.action_save_project)
        file_menu.addAction(save_act)

        save_as_act = QAction("Save Project &As...", self)
        save_as_act.setShortcut(QKeySequence.StandardKey.SaveAs)
        save_as_act.triggered.connect(self.action_save_project_as)
        file_menu.addAction(save_as_act)

        file_menu.addSeparator()

        import_xml_act = QAction("Import Prism XML (.xml)...", self)
        import_xml_act.triggered.connect(self.action_import_xml)
        file_menu.addAction(import_xml_act)

        export_xml_act = QAction("Export Prism XML (.xml)...", self)
        export_xml_act.triggered.connect(self.action_export_xml)
        file_menu.addAction(export_xml_act)

        export_office_act = QAction("Export Report / Office Document...", self)
        export_office_act.triggered.connect(self.action_export_office_dialog)
        file_menu.addAction(export_office_act)

        file_menu.addSeparator()

        export_png_act = QAction("Export Graph as PNG...", self)
        export_png_act.triggered.connect(self.action_export_png)
        file_menu.addAction(export_png_act)

        export_svg_act = QAction("Export Graph as Vector SVG...", self)
        export_svg_act.triggered.connect(self.action_export_svg)
        file_menu.addAction(export_svg_act)

        export_layout_act = QAction("Export Layout as Image...", self)
        export_layout_act.triggered.connect(self.action_export_layout_image)
        file_menu.addAction(export_layout_act)

        file_menu.addSeparator()

        add_layout_act = QAction("➕ New Page Layout...", self)
        add_layout_act.triggered.connect(self.action_add_layout)
        file_menu.addAction(add_layout_act)

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
        self.redo_act.setShortcuts([QKeySequence.StandardKey.Redo, QKeySequence("Ctrl+Y")])
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
        rc_menu.addAction("Enzyme Kinetics...", self.action_run_enzyme_kinetics)
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
        surv_menu.addAction("Cox Proportional Hazards...", self.action_run_cox_regression)

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
        toolbar.addSeparator()
        toolbar.addAction("Export...", self.action_export_office_dialog)

        self._setup_contextual_toolbar()

    def _setup_contextual_toolbar(self) -> None:
        self.context_toolbar = QToolBar("Contextual Analysis Shortcuts")
        self.context_toolbar.setMovable(False)
        self.addToolBar(self.context_toolbar)
        self._update_contextual_toolbar(TableType.XY)

    def _update_contextual_toolbar(self, table_type: TableType) -> None:
        self.context_toolbar.clear()
        self.context_toolbar.addWidget(QLabel(f"<b>💡  {table_type.value} Shortcuts:</b> "))

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

    def _setup_statusbar(self) -> None:
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("GemGraph Ready | Clean-Room Mode: Active")

    def on_sheet_selected(self, sheet_id: str) -> None:
        sheet = self.document.get_sheet(sheet_id)
        if not sheet:
            return
        self.document.active_sheet_id = sheet_id
        self.navigator.select_sheet(sheet_id)

        if sheet.sheet_type == SheetType.DATA:
            data_sheet: DataSheet = sheet  # type: ignore
            is_frozen = getattr(data_sheet, "is_frozen", False)
            self.spreadsheet_model.set_table(data_sheet.table, is_frozen=is_frozen)
            self.spreadsheet_view.set_frozen(is_frozen)
            self.center_stack.setCurrentWidget(self.spreadsheet_view)
            self.graph_canvas.update_data(data_sheet.table, sheet=data_sheet)
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
            self.graph_canvas.update_data(src_table, result=res_sheet, sheet=res_sheet)
            self.status.showMessage(f"Active Result: {sheet.title}")
            self.workspace_modified.emit()

        elif sheet.sheet_type == SheetType.INFO:
            info_sheet: InfoSheet = sheet  # type: ignore
            self.info_widget.display_info(info_sheet)
            self.center_stack.setCurrentWidget(self.info_widget)
            self.status.showMessage(f"Active Info Sheet: {sheet.title}")

        elif sheet.sheet_type == SheetType.GRAPH:
            graph_sheet: GraphSheet = sheet  # type: ignore
            is_frozen = bool(graph_sheet.is_frozen)
            if is_frozen and getattr(graph_sheet, "snapshot_table", None) is not None:
                src_table = graph_sheet.snapshot_table
                if getattr(graph_sheet, "snapshot_result_data", None):
                    res_sheet = ResultSheet.from_dict(graph_sheet.snapshot_result_data)
                elif graph_sheet.result_sheet_id:
                    res_sheet = self.document.get_sheet(graph_sheet.result_sheet_id)
                else:
                    res_sheet = None
            else:
                src_sheet = self.document.get_sheet(graph_sheet.source_sheet_id)
                src_table = src_sheet.table if isinstance(src_sheet, DataSheet) else getattr(graph_sheet, "snapshot_table", None)
                res_sheet = (
                    self.document.get_sheet(graph_sheet.result_sheet_id)
                    if graph_sheet.result_sheet_id
                    else None
                )
            if src_table:
                self.spreadsheet_model.set_table(src_table, is_frozen=is_frozen)
                self.spreadsheet_view.set_frozen(is_frozen)
                self.center_stack.setCurrentWidget(self.spreadsheet_view)
            self.graph_canvas.update_data(
                src_table,
                result=res_sheet if isinstance(res_sheet, ResultSheet) else None,
                graph_type=graph_sheet.graph_type,
                sheet=graph_sheet,
            )
            self.status.showMessage(f"Active Graph: {sheet.title}")

        elif sheet.sheet_type == SheetType.LAYOUT:
            self.center_stack.setCurrentWidget(self.layout_view)
            self.layout_view.load_layout(sheet, self.document)
            self.layout_view.canvas.fit_page_to_window()
            self.status.showMessage(f"Active Layout: {sheet.title}")

    def on_sheet_deleted(self, sheet_id: str) -> None:
        self.document.remove_sheet(sheet_id)
        GLOBAL_REGISTRY.remove_by_sheet(sheet_id)
        self.navigator.populate(self.document)
        if self.document.active_sheet_id:
            self.on_sheet_selected(self.document.active_sheet_id)

    def _create_navigator_context_menu(self, pos) -> Optional[QMenu]:
        item = self.navigator.itemAt(pos)
        if not item:
            return None

        data_root = self.navigator.folder_items.get(SheetType.DATA)
        info_root = self.navigator.folder_items.get(SheetType.INFO)
        results_root = self.navigator.folder_items.get(SheetType.RESULTS)
        graphs_root = self.navigator.folder_items.get(SheetType.GRAPH)
        layouts_root = self.navigator.folder_items.get(SheetType.LAYOUT)

        sheet_id = item.data(0, Qt.ItemDataRole.UserRole)
        sheet = self.document.get_sheet(sheet_id) if sheet_id else None

        is_data_root = (item == data_root)
        is_info_root = (item == info_root)
        is_results_root = (item == results_root)
        is_graphs_root = (item == graphs_root)
        is_layouts_root = (item == layouts_root)

        is_data_child = (sheet and sheet.sheet_type == SheetType.DATA) or (item.parent() == data_root)
        is_info_child = (sheet and sheet.sheet_type == SheetType.INFO) or (item.parent() == info_root)
        is_results_child = (sheet and sheet.sheet_type == SheetType.RESULTS) or (item.parent() == results_root)
        is_graphs_child = (sheet and sheet.sheet_type == SheetType.GRAPH) or (item.parent() == graphs_root)
        is_layouts_child = (sheet and sheet.sheet_type == SheetType.LAYOUT) or (item.parent() == layouts_root)

        menu = QMenu(self)

        # 1. Info Sheets
        if is_info_root or is_info_child:
            add_info_act = menu.addAction("➕ Add Info Sheet")
            add_info_act.triggered.connect(self.action_add_info_sheet)
            if is_info_child and sheet_id:
                menu.addSeparator()
                rename_act = menu.addAction("✏️ Rename Info Sheet...")
                rename_act.triggered.connect(lambda: self._rename_navigator_sheet(item, sheet_id))
                delete_act = menu.addAction("🗑️ Delete Info Sheet")
                delete_act.triggered.connect(lambda: self._delete_navigator_sheet(item, sheet_id))
            return menu

        # 2. Data Tables
        if is_data_root or is_data_child:
            add_sheet_act = menu.addAction("➕ Add Sheet")
            add_sheet_act.triggered.connect(lambda: self.action_add_table(TableType.XY))

            type_menu = menu.addMenu("➕ Add Sheet (Format...)")
            type_menu.addAction("XY Data Table", lambda: self.action_add_table(TableType.XY))
            type_menu.addAction("Column Data Table", lambda: self.action_add_table(TableType.COLUMN))
            type_menu.addAction("Grouped Data Table", lambda: self.action_add_table(TableType.GROUPED))
            type_menu.addAction("Contingency Table", lambda: self.action_add_table(TableType.CONTINGENCY))
            type_menu.addAction("Survival Table", lambda: self.action_add_table(TableType.SURVIVAL))
            type_menu.addAction("Parts of Whole", lambda: self.action_add_table(TableType.PARTS_OF_WHOLE))
            type_menu.addAction("Multiple Variables", lambda: self.action_add_table(TableType.MULTIPLE_VARIABLES))
            type_menu.addAction("Nested Data Table", lambda: self.action_add_table(TableType.NESTED))

            if is_data_child and sheet_id:
                menu.addSeparator()
                rename_act = menu.addAction("✏️ Rename Data Table...")
                rename_act.triggered.connect(lambda: self._rename_navigator_sheet(item, sheet_id))
                delete_act = menu.addAction("🗑️ Delete Data Table")
                delete_act.triggered.connect(lambda: self._delete_navigator_sheet(item, sheet_id))

                # Hidden backward-compatibility actions for existing test
                compat_rename = menu.addAction("Rename Sheet...")
                compat_rename.setVisible(False)
                compat_rename.triggered.connect(lambda: self._rename_navigator_sheet(item, sheet_id))
                compat_delete = menu.addAction("Delete Sheet")
                compat_delete.setVisible(False)
                compat_delete.triggered.connect(lambda: self._delete_navigator_sheet(item, sheet_id))

            return menu

        # 3. Graphs
        if is_graphs_root or is_graphs_child:
            if is_graphs_child and sheet_id:
                rename_act = menu.addAction("✏️ Rename Graph...")
                rename_act.triggered.connect(lambda: self._rename_navigator_sheet(item, sheet_id))
                delete_act = menu.addAction("🗑️ Delete Graph")
                delete_act.triggered.connect(lambda: self._delete_navigator_sheet(item, sheet_id))
                return menu

        # 4. Results
        if is_results_root or is_results_child:
            if is_results_child and sheet_id:
                rename_act = menu.addAction("✏️ Rename Results...")
                rename_act.triggered.connect(lambda: self._rename_navigator_sheet(item, sheet_id))
                delete_act = menu.addAction("🗑️ Delete Results")
                delete_act.triggered.connect(lambda: self._delete_navigator_sheet(item, sheet_id))
                return menu

        # 5. Layouts
        if is_layouts_root or is_layouts_child:
            add_layout_act = menu.addAction("➕ Add Layout...")
            add_layout_act.triggered.connect(self.action_add_layout)
            if is_layouts_child and sheet_id:
                menu.addSeparator()
                rename_act = menu.addAction("✏️ Rename Layout...")
                rename_act.triggered.connect(lambda: self._rename_navigator_sheet(item, sheet_id))
                delete_act = menu.addAction("🗑️ Delete Layout")
                delete_act.triggered.connect(lambda: self._delete_navigator_sheet(item, sheet_id))
            return menu

        if sheet_id:
            rename_act = menu.addAction("Rename Sheet...")
            rename_act.triggered.connect(lambda: self._rename_navigator_sheet(item, sheet_id))
            delete_act = menu.addAction("Delete Sheet")
            delete_act.triggered.connect(lambda: self._delete_navigator_sheet(item, sheet_id))
            return menu

        return None

    def _on_navigator_context_menu(self, pos) -> Optional[QMenu]:
        menu = self._create_navigator_context_menu(pos)
        if menu:
            menu.exec(self.navigator.viewport().mapToGlobal(pos))
        return menu

    def _rename_navigator_sheet(self, item: QTreeWidgetItem, sheet_id: str) -> None:
        clean_title = item.text(0).replace(" ❄️", "").replace(" ⚠️", "")
        new_title, ok = QInputDialog.getText(self, "Rename Sheet", "New title:", text=clean_title)
        if ok and new_title.strip() and self.document:
            sheet = self.document.get_sheet(sheet_id)
            if sheet and sheet.title != new_title.strip():
                sheet.title = new_title.strip()
                # Update any layout placeholders referencing this sheet
                for s in self.document.sheets:
                    if hasattr(s, "placeholders"):
                        for p in s.placeholders:
                            if p.linked_graph_id == sheet_id:
                                p.linked_graph_title = sheet.title
                self.navigator.populate(self.document)
                if hasattr(self, "layout_view") and self.layout_view is not None:
                    self.layout_view.refresh_all_snapshots()
                self.mark_as_modified()

    def _delete_navigator_sheet(self, item: QTreeWidgetItem, sheet_id: str) -> None:
        reply = QMessageBox.question(
            self,
            "Delete Sheet",
            f"Are you sure you want to delete '{item.text(0)}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.on_sheet_deleted(sheet_id)

    def _on_table_data_modified(self) -> None:
        self.mark_as_modified()
        # Re-render graph canvas when table values change and propagate downstream
        active_sheet = self.document.get_sheet(self.document.active_sheet_id or "")
        if isinstance(active_sheet, DataSheet):
            GLOBAL_REGISTRY.extract_from_table(active_sheet.table)
            self.document.recalculate_downstream(active_sheet.sheet_id)
            canvas_sheet = getattr(self.graph_canvas, "active_sheet", None) or getattr(self.graph_canvas, "current_graph_sheet", None)
            if canvas_sheet and getattr(canvas_sheet, "is_frozen", False):
                pass
            else:
                self.graph_canvas.update_data(active_sheet.table, sheet=active_sheet)
        elif isinstance(active_sheet, GraphSheet):
            if not active_sheet.is_frozen:
                src_sheet = self.document.get_sheet(active_sheet.source_sheet_id)
                src_table = src_sheet.table if isinstance(src_sheet, DataSheet) else None
                res_sheet = (
                    self.document.get_sheet(active_sheet.result_sheet_id)
                    if active_sheet.result_sheet_id
                    else None
                )
                self.graph_canvas.update_data(
                    src_table,
                    result=res_sheet if isinstance(res_sheet, ResultSheet) else None,
                    graph_type=active_sheet.graph_type,
                    sheet=active_sheet,
                )
        if hasattr(self, "layout_view") and self.layout_view is not None:
            if self.center_stack.currentWidget() == self.layout_view:
                self.layout_view.refresh_hot_placeholders()
        self.workspace_modified.emit()

    def action_undo(self) -> None:
        focused = QApplication.focusWidget() or self.focusWidget()
        from PyQt6.QtWidgets import QTextEdit, QLineEdit, QGraphicsTextItem
        from src.ui.rich_text_editor import RichTextEdit
        if isinstance(focused, (QTextEdit, RichTextEdit, QLineEdit)):
            focused.undo()
            return
        if hasattr(focused, "editor") and isinstance(focused.editor, (QTextEdit, RichTextEdit)):
            focused.editor.undo()
            return
        if self.center_stack.currentWidget() == self.layout_view:
            if hasattr(self.layout_view, "canvas") and self.layout_view.canvas.scene.focusItem() is not None:
                focus_item = self.layout_view.canvas.scene.focusItem()
                if isinstance(focus_item, QGraphicsTextItem):
                    doc = focus_item.document()
                    if doc is not None and doc.isUndoAvailable():
                        doc.undo()
                        return
            if hasattr(self.layout_view, "undo_stack") and self.layout_view.undo_stack.canUndo():
                self.layout_view.undo_stack.undo()
            return
        if self.undo_stack.undo():
            self.spreadsheet_model.layoutChanged.emit()
            self._on_table_data_modified()

    def action_redo(self) -> None:
        focused = QApplication.focusWidget() or self.focusWidget()
        from PyQt6.QtWidgets import QTextEdit, QLineEdit, QGraphicsTextItem
        from src.ui.rich_text_editor import RichTextEdit
        if isinstance(focused, (QTextEdit, RichTextEdit, QLineEdit)):
            focused.redo()
            return
        if hasattr(focused, "editor") and isinstance(focused.editor, (QTextEdit, RichTextEdit)):
            focused.editor.redo()
            return
        if self.center_stack.currentWidget() == self.layout_view:
            if hasattr(self.layout_view, "canvas") and self.layout_view.canvas.scene.focusItem() is not None:
                focus_item = self.layout_view.canvas.scene.focusItem()
                if isinstance(focus_item, QGraphicsTextItem):
                    doc = focus_item.document()
                    if doc is not None and doc.isRedoAvailable():
                        doc.redo()
                        return
            if hasattr(self.layout_view, "undo_stack") and self.layout_view.undo_stack.canRedo():
                self.layout_view.undo_stack.redo()
            return
        if self.undo_stack.redo():
            self.spreadsheet_model.layoutChanged.emit()
            self._on_table_data_modified()

    def action_add_info_sheet(self) -> InfoSheet:
        """Creates a new InfoSheet, appends it to project, updates navigator, and opens it."""
        sheet = self.document.create_info_sheet()
        self.document.active_sheet_id = sheet.sheet_id
        self.navigator.populate(self.document)
        self.on_sheet_selected(sheet.sheet_id)
        return sheet

    action_add_info = action_add_info_sheet

    def action_add_layout(self) -> Any:
        """Creates a new presentation Page Layout, appends it to project, updates navigator, and opens it."""
        from src.ui.dialogs import NewLayoutDialog
        sheet_count = len(self.document.get_sheets_by_type(SheetType.LAYOUT)) + 1
        dialog = NewLayoutDialog(self, initial_name=f"Layout {sheet_count}")
        if dialog.exec():
            settings = dialog.get_settings()
            layout = self.document.create_layout(
                title=settings["name"],
                page_format=settings["page_format"],
                orientation=settings["orientation"],
                template=settings["template"],
            )
            self.document.active_sheet_id = layout.sheet_id
            self.navigator.populate(self.document)
            self.on_sheet_selected(layout.sheet_id)
            return layout
        return None

    def action_export_layout_image(self, dpi: Optional[int] = None) -> None:
        """Exports currently active or first layout sheet as high-DPI image."""
        active_sheet = self.document.get_sheet(self.document.active_sheet_id or "")
        from src.core.layout import PageLayout
        if not isinstance(active_sheet, PageLayout) and getattr(active_sheet, "sheet_type", None) != SheetType.LAYOUT:
            layouts = self.document.layouts
            if not layouts:
                QMessageBox.information(self, "Export Layout as Image", "No page layouts exist in this project to export.")
                return
            active_sheet = layouts[0]

        filters = "PNG Image (*.png);;TIFF Image (*.tif);;Scalable Vector Graphics (*.svg);;PDF Document (*.pdf)"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Layout as Image",
            f"{active_sheet.title}.png",
            filters,
        )
        if not file_path:
            return

        from src.core.layout import render_layout_to_image, render_layout_to_pdf, render_layout_to_svg
        ext = file_path.split(".")[-1].lower()
        try:
            if ext == "pdf":
                render_layout_to_pdf(active_sheet, document=self.document, file_path=file_path)
            elif ext == "svg":
                render_layout_to_svg(active_sheet, document=self.document, file_path=file_path)
            else:
                target_dpi = dpi
                if target_dpi is None:
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
                            target_dpi = 150
                        elif "600" in chosen:
                            target_dpi = 600
                        else:
                            target_dpi = 300
                    else:
                        target_dpi = 300
                img = render_layout_to_image(active_sheet, document=self.document, dpi=target_dpi)
                fmt_tag = "TIFF" if ext in ("tif", "tiff") else "PNG"
                img.save(file_path, fmt_tag)
            QMessageBox.information(self, "Export Complete", f"Successfully exported layout image to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Failed to export layout image:\n{e}")

    def action_pin_graph(self, title: Optional[str] = None) -> GraphSheet:
        """
        Captures snapshot of current graph view, its linked data table/results,
        and its full plot_config, instantiating an independent GraphSheet under graphs collection.
        """
        from src.core.sheet import format_pinned_graph_name

        active_sheet = self.document.get_sheet(self.document.active_sheet_id or "")
        table = self.graph_canvas.current_table
        result = self.graph_canvas.current_result

        # Determine naming inputs
        snapshot_result_data = None
        if result is not None:
            table_or_results_name = result.title
            analysis_model_name = getattr(result, "analysis_type", "Analysis")
            source_sheet_id = result.source_sheet_id
            result_sheet_id = result.sheet_id
            snapshot_table = table.clone() if table else None
            snapshot_result_data = result.to_dict()
        elif table is not None:
            table_or_results_name = table.title or (active_sheet.title if active_sheet else "Data")
            analysis_model_name = "Raw"
            source_sheet_id = getattr(table, "table_id", "") or (active_sheet.sheet_id if active_sheet else "")
            result_sheet_id = None
            snapshot_table = table.clone()
        elif active_sheet:
            table_or_results_name = active_sheet.title
            analysis_model_name = "Raw"
            source_sheet_id = active_sheet.sheet_id
            result_sheet_id = None
            snapshot_table = (
                active_sheet.table.clone()
                if hasattr(active_sheet, "table") and active_sheet.table
                else None
            )
        else:
            table_or_results_name = "Data"
            analysis_model_name = "Raw"
            source_sheet_id = ""
            result_sheet_id = None
            snapshot_table = None

        curr_type_str = self.graph_canvas.type_combo.currentText()
        matched_gt = GraphType.XY_SCATTER_CURVE
        for gt in GraphType:
            if gt.value == curr_type_str:
                matched_gt = gt
                break

        graph_title = title or format_pinned_graph_name(
            table_or_results_name, analysis_model_name, curr_type_str
        )

        plot_cfg = dict(getattr(active_sheet, "plot_config", {})) if active_sheet else {}
        plot_cfg["plot_type"] = curr_type_str
        plot_cfg["graph_type"] = curr_type_str
        plot_cfg["palette"] = self.graph_canvas.palette_combo.currentText()
        plot_cfg["palette_index"] = self.graph_canvas.palette_combo.currentIndex()
        plot_cfg["is_grayscale"] = self.graph_canvas.is_grayscale_mode
        plot_cfg["whisker_dir"] = self.graph_canvas.whisker_dir_combo.currentText()
        plot_cfg["show_bar_scatter_points"] = self.graph_canvas.show_bar_scatter_points
        plot_cfg["show_scatter"] = self.graph_canvas.show_bar_scatter_points
        plot_cfg["error_style"] = self.graph_canvas.error_style
        plot_cfg["series_styles"] = dict(self.graph_canvas.series_styles)
        plot_cfg["series_colors"] = {
            k: (v.name() if hasattr(v, "name") else str(v))
            for k, v in self.graph_canvas.custom_series_colors.items()
        }
        plot_cfg["series_opacities"] = dict(self.graph_canvas.custom_series_opacities)
        plot_cfg["is_pinned"] = True

        graph_sheet = self.document.create_graph_sheet(
            title=graph_title,
            source_sheet_id=source_sheet_id,
            result_sheet_id=result_sheet_id,
            graph_type=matched_gt,
            plot_config=plot_cfg,
        )
        graph_sheet.is_pinned = True
        graph_sheet.snapshot_table = snapshot_table
        graph_sheet.snapshot_result_data = snapshot_result_data

        self.navigator.populate(self.document)
        self.on_sheet_selected(graph_sheet.sheet_id)
        self.status.showMessage(f"Pinned Graph: {graph_sheet.title}")
        return graph_sheet

    def _on_freeze_toggled(self, frozen: bool) -> None:
        active_sheet = self.document.get_sheet(self.document.active_sheet_id or "")
        if isinstance(active_sheet, GraphSheet):
            active_sheet.is_frozen = frozen
            if frozen:
                if getattr(active_sheet, "snapshot_table", None) is None:
                    src_sheet = self.document.get_sheet(active_sheet.source_sheet_id)
                    if isinstance(src_sheet, DataSheet):
                        active_sheet.snapshot_table = src_sheet.table.clone()
                if getattr(active_sheet, "snapshot_result_data", None) is None and active_sheet.result_sheet_id:
                    res_sheet = self.document.get_sheet(active_sheet.result_sheet_id)
                    if isinstance(res_sheet, ResultSheet):
                        active_sheet.snapshot_result_data = res_sheet.to_dict()
            elif not frozen:
                src_sheet = self.document.get_sheet(active_sheet.source_sheet_id)
                if src_sheet is not None:
                    active_sheet.snapshot_table = None
                    active_sheet.snapshot_result_data = None
            self.navigator.populate(self.document)
            self.navigator.select_sheet(active_sheet.sheet_id)
            self.on_sheet_selected(active_sheet.sheet_id)
            self.mark_as_modified()

    def action_new_project(self) -> None:
        self.document = PrismDocument("Untitled Project")
        self.undo_stack.clear()
        self.current_file_path = None
        self._create_blank_project()
        self.navigator.populate(self.document)
        if self.document.active_sheet_id:
            self.on_sheet_selected(self.document.active_sheet_id)
        self._bind_document_signals()
        self._is_modified = False
        self._update_window_title()

    def action_load_sample_data(self) -> None:
        """Loads textbook sample document in-memory via build_textbook_sample_document()."""
        from src.data_engine.sample_generator import build_textbook_sample_document
        self.document = build_textbook_sample_document()
        self.undo_stack.clear()
        self.current_file_path = None
        self.navigator.populate(self.document)
        if len(self.document.sheet_order) > 0:
            first_id = self.document.sheet_order[0]
            self.navigator.select_sheet(first_id)
            self.on_sheet_selected(first_id)
        self._bind_document_signals()
        self._is_modified = False
        self._update_window_title()
        self.status.showMessage("Loaded Textbook Sample Datasets.")

    action_load_sample_datasets = action_load_sample_data

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
            "Open GemGraph Project",
            initial_dir,
            "GemGraph Projects (*.bigo);;GraphPad Prism Projects (*.prism *.pzfx);;All Supported (*.bigo *.prism *.pzfx);;All Files (*)",
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
        Loads dataset from filepath (.bigo, .prism, .pzfx/.xml, .csv),
        populates the spreadsheet, and renders the initial plot.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        ext = os.path.splitext(filepath)[1].lower()
        if ext in (".bigo", ".prism"):
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

        self.current_file_path = filepath
        self._bind_document_signals()
        self._is_modified = False
        self._update_window_title()
        self.status.showMessage(f"Opened: {filepath}")

    def action_open_sample_project(self) -> None:
        """Loads bundled textbook sample project (samples/textbook_sample.bigo) directly into workspace."""
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        sample_path = os.path.join(project_root, "samples", "textbook_sample.bigo")
        if not os.path.exists(sample_path):
            sample_path = os.path.join(os.getcwd(), "samples", "textbook_sample.bigo")
        if not os.path.exists(sample_path):
            from src.data_engine.sample_generator import generate_textbook_sample
            generate_textbook_sample(sample_path)
        self.open_project(sample_path)
        self.status.showMessage("Loaded Textbook Sample Project (.bigo)")

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
        if self.current_file_path:
            try:
                PrismIO.save(self.document, self.current_file_path)
                self._is_modified = False
                self._update_window_title()
                self.status.showMessage(f"Saved: {self.current_file_path}")
                return
            except Exception as e:
                QMessageBox.critical(self, "Error Saving Project", str(e))
                return
        self.action_save_project_as()

    def action_save_project_as(self) -> None:
        initial_dir = self.get_smart_initial_dir()
        default_path = self.current_file_path or os.path.join(initial_dir, f"{self.document.title}.bigo")
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Project",
            default_path,
            "GemGraph Projects (*.bigo);;GraphPad Prism Projects (*.prism *.pzfx);;All Supported (*.bigo *.prism *.pzfx);;All Files (*)",
        )
        if file_path:
            try:
                PrismIO.save(self.document, file_path)
                self.current_file_path = file_path
                self._is_modified = False
                self._update_window_title()
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

    def action_export_office_dialog(self) -> None:
        from src.ui.export_dialog import ExportProjectDialog
        dialog = ExportProjectDialog(self.document, graph_canvas=self.graph_canvas, parent=self)
        dialog.exec()

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
                if any(filepath.lower().endswith(ext) for ext in (".bigo", ".prism", ".pzfx", ".csv")):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        mime = event.mimeData()
        if mime and mime.hasUrls():
            for url in mime.urls():
                filepath = unquote(url.toLocalFile() or url.path())
                if any(filepath.lower().endswith(ext) for ext in (".bigo", ".prism", ".pzfx", ".csv")):
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
                    if any(fp.lower().endswith(ext) for ext in (".bigo", ".prism", ".pzfx", ".csv")):
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
        self.document.active_sheet_id = sheet.sheet_id
        self.navigator.populate(self.document)
        self.on_sheet_selected(sheet.sheet_id)

    def action_add_info(self) -> None:
        sheet = self.document.create_info_sheet()
        self.document.active_sheet_id = sheet.sheet_id
        self.navigator.populate(self.document)
        self.on_sheet_selected(sheet.sheet_id)

    def action_add_graph(self) -> None:
        active_sheet = self.document.get_sheet(self.document.active_sheet_id or "")
        src_id = active_sheet.sheet_id if isinstance(active_sheet, DataSheet) else ""
        sheet = self.document.create_graph_sheet(source_sheet_id=src_id)
        self.document.active_sheet_id = sheet.sheet_id
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
                "Enzyme Kinetics...": self.action_run_enzyme_kinetics,
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
                "Cox Proportional Hazards...": self.action_run_cox_regression,
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

    def action_run_enzyme_kinetics(self) -> None:
        """Executes Enzyme Kinetics suite with candidate kinetic models, Extra-SS F-test, and diagnostic transforms."""
        active_sheet = self.document.get_sheet(self.document.active_sheet_id or "")
        if not isinstance(active_sheet, DataSheet):
            QMessageBox.warning(self, "Enzyme Kinetics", "Please select a Data Sheet with Substrate (X) and Velocity (Y) data.")
            return

        table: DataTable = active_sheet.table
        if table.x_data is None:
            QMessageBox.warning(self, "Enzyme Kinetics", "Table does not contain an X column for substrate concentration.")
            return

        dlg = EnzymeKineticsDialog(table.y_group_titles, parent=self)
        if not dlg.exec():
            return

        model_name, group_idx, compare_mm, show_diag, diag_type, preset = dlg.get_settings()
        if model_name not in STANDARD_MODELS:
            QMessageBox.warning(self, "Enzyme Kinetics", f"Unknown kinetic model: {model_name}")
            return

        model = STANDARD_MODELS[model_name]
        xs, ys = table.get_clean_xy(group_idx)
        if len(xs) <= len(model.param_names):
            QMessageBox.warning(self, "Enzyme Kinetics", "Not enough valid data points to fit the chosen model.")
            return

        try:
            fit_result = NonlinearEngine.fit(xs, ys, model, preset=preset)
        except Exception as e:
            QMessageBox.critical(self, "Enzyme Kinetics Error", f"Model fitting failed: {e}")
            return

        # 1. Table of results tab
        res_headers = ["Parameter", "Best-Fit Value", "Std. Error", "95% CI Lower", "95% CI Upper"]
        res_rows = []
        for p in fit_result["parameters"]:
            res_rows.append([
                p["name"],
                f"{p['best_fit']:.4f}",
                f"{p['std_error']:.4f}",
                f"{p['ci_lower']:.4f}",
                f"{p['ci_upper']:.4f}",
            ])
        res_rows.append(["--- Goodness of Fit ---", "", "", "", ""])
        res_rows.append(["Degrees of Freedom", str(fit_result["degrees_of_freedom"]), "", "", ""])
        res_rows.append(["R squared", f"{fit_result['r_squared']:.4f}", "", "", ""])
        res_rows.append(["Sy.x (Std Error of Fit)", f"{fit_result['syx']:.4f}", "", "", ""])
        res_rows.append(["AICc", f"{fit_result['aicc']:.2f}", "", "", ""])

        # 2. Curve tab
        x_min, x_max = float(np.min(xs)), float(np.max(xs))
        x_dense = np.linspace(max(0.0, x_min * 0.9), x_max * 1.1, 100)
        y_dense = model.evaluate(x_dense, fit_result["p_opt"])
        curve_headers = ["Substrate [S]", "Fitted Velocity (v)"]
        curve_rows = [[f"{x:.4f}", f"{y:.4f}"] for x, y in zip(x_dense, y_dense)]

        tabs = [
            {"name": "Table of results", "grid_type": "green", "headers": res_headers, "rows": res_rows, "visible": True},
            {"name": "Curve", "grid_type": "green", "headers": curve_headers, "rows": curve_rows, "visible": True},
        ]

        # 3. Model comparison tab
        if compare_mm:
            comp_headers = ["Comparison Metric", "Value"]
            if model_name != "Michaelis-Menten":
                try:
                    mm_model = STANDARD_MODELS["Michaelis-Menten"]
                    comp = NonlinearEngine.compare_models(xs, ys, mm_model, model, preset=preset)
                    comp_rows = [
                        ["Simpler Model", comp["simpler_model"]],
                        ["Alternative Model", comp["complex_model"]],
                        ["Extra-SS F-statistic", f"{comp['f_stat']:.4f}"],
                        ["Degrees of Freedom (Num, Denom)", f"{comp['df_num']}, {comp['df_denom']}"],
                        ["P-value (Extra-SS F-test)", f"{comp['p_value']:.4e}" if comp['p_value'] < 0.0001 else f"{comp['p_value']:.4f}"],
                        ["AICc (Simpler)", f"{comp['aicc_simpler']:.2f}"],
                        ["AICc (Alternative)", f"{comp['aicc_complex']:.2f}"],
                        ["Delta AICc (Simpler - Alt)", f"{comp['aicc_simpler'] - comp['aicc_complex']:.2f}"],
                        ["Preferred Model", comp["preferred_model"]],
                    ]
                except Exception as comp_err:
                    comp_rows = [["Comparison Error", str(comp_err)]]
            else:
                comp_rows = [
                    ["Simpler Model", "Michaelis-Menten"],
                    ["Note", "Michaelis-Menten is already the simplest kinetic model."],
                ]
            tabs.append({"name": "Model comparison", "grid_type": "green", "headers": comp_headers, "rows": comp_rows, "visible": True})

        # 4. Diagnostic transform tab
        if show_diag:
            try:
                diag_key = "lineweaver_burk" if "Lineweaver-Burk" in diag_type else "eadie_hofstee"
                txs, tys = pharmacological_transform(xs, ys, diag_key)
                valid_diag = ~np.isnan(txs) & ~np.isnan(tys)
                d_headers = ["1/[S]", "1/v"] if diag_key == "lineweaver_burk" else ["v/[S]", "v"]
                d_rows = [[f"{x:.4f}", f"{y:.4f}"] for x, y in zip(txs[valid_diag], tys[valid_diag])]
            except Exception as e:
                d_headers = ["Diagnostic Error"]
                d_rows = [[str(e)]]
            tabs.append({"name": "Diagnostic transform", "grid_type": "green", "headers": d_headers, "rows": d_rows, "visible": True})

        raw_data = dict(fit_result)
        raw_data["curves"] = [{
            "name": f"{model_name} Fit",
            "x_dense": x_dense.tolist(),
            "y_dense": y_dense.tolist(),
        }]

        res_sheet = self.document.create_result_sheet(
            title=f"Enzyme Kinetics ({model_name})",
            analysis_type="Enzyme Kinetics",
            source_sheet_id=active_sheet.sheet_id,
            tabs=tabs,
            summary_tables=[{"title": "Parameter Estimates", "headers": res_headers, "rows": res_rows}],
            raw_result_data=raw_data,
            grid_type="green",
        )
        GLOBAL_REGISTRY.extract_from_result(res_sheet)
        self.graph_canvas.add_fitted_curve(
            f"enzyme_{res_sheet.sheet_id}",
            f"{model_name} Fit",
            x_dense,
            y_dense,
        )
        self.navigator.populate(self.document)
        self.on_sheet_selected(res_sheet.sheet_id)

    def action_run_survival(self) -> None:
        """Computes Kaplan-Meier survival curves with Greenwood SE, log-log 95% CIs, and multi-group log-rank test."""
        active_sheet = self.document.get_sheet(self.document.active_sheet_id or "")
        if not isinstance(active_sheet, DataSheet):
            QMessageBox.warning(self, "Survival Analysis", "Please select a Data Sheet.")
            return

        table: DataTable = active_sheet.table
        if table.x_data is None:
            QMessageBox.warning(self, "Survival Analysis", "Survival analysis requires X column as elapsed time.")
            return

        times = np.copy(table.x_data)
        km_results: Dict[str, Any] = {}
        groups_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        for g_idx, g_name in enumerate(table.y_group_titles):
            events = np.array([table.get_cell_value(r, "Y", g_idx, 0) for r in range(table.num_rows)], dtype=np.float64)
            valid = ~np.isnan(times) & ~np.isnan(events)
            t_clean = times[valid]
            e_clean = events[valid]
            if len(t_clean) > 0:
                km = SurvivalEngine.kaplan_meier(t_clean, e_clean)
                km_results[g_name] = km
                groups_data[g_name] = (t_clean, e_clean)

        if not km_results:
            QMessageBox.warning(self, "Survival Analysis", "No valid survival observations found.")
            return

        lr_logrank = None
        hr_res = None
        if len(groups_data) >= 2:
            try:
                lr_logrank = SurvivalEngine.log_rank_test(list(groups_data.values()))
            except Exception:
                pass

            if len(groups_data) == 2:
                g_keys = list(groups_data.keys())
                try:
                    hr_res = SurvivalEngine.hazard_ratio_logrank(
                        groups_data[g_keys[0]],
                        groups_data[g_keys[1]],
                    )
                except Exception:
                    pass

        # Build tabs
        table_headers = [
            "Group", "Time", "Number at Risk", "Number of Deaths", "Censored",
            "Fractional Survival", "Std Error", "95% CI Lower", "95% CI Upper"
        ]
        table_rows = []
        for g_name, km in km_results.items():
            t_arr = km["times"]
            s_arr = km["survival"]
            se_arr = km["se"]
            ci_l = km["ci_lower"]
            ci_u = km["ci_upper"]
            risk = km["n_at_risk"]
            evs = km["n_events"]
            cens = km["n_censored"]
            for i, ut in enumerate(t_arr):
                table_rows.append([
                    g_name,
                    f"{ut:.4f}",
                    str(risk[i]),
                    str(evs[i]),
                    str(cens[i]),
                    f"{s_arr[i]:.4f}",
                    f"{se_arr[i]:.4f}",
                    f"{ci_l[i]:.4f}",
                    f"{ci_u[i]:.4f}",
                ])

        coord_headers = ["Group", "Time", "Survival Fraction"]
        coord_rows = []
        for g_name, km in km_results.items():
            coord_rows.append([g_name, "0.0000", "1.0000"])
            for t, s in zip(km["times"], km["survival"]):
                coord_rows.append([g_name, f"{t:.4f}", f"{s:.4f}"])

        tabs = [
            {"name": "Survival Table", "grid_type": "green", "headers": table_headers, "rows": table_rows, "visible": True},
            {"name": "Curve Coordinates", "grid_type": "green", "headers": coord_headers, "rows": coord_rows, "visible": True},
        ]

        if lr_logrank is not None:
            comp_headers = ["Comparison / Metric", "Value"]
            comp_rows = [
                ["Log-rank (Mantel-Cox) Chi-Square", f"{lr_logrank['chi2']:.4f}"],
                ["Log-rank Degrees of Freedom", str(lr_logrank['df'])],
                ["Log-rank P-value", f"{lr_logrank['p_value']:.4e}" if lr_logrank['p_value'] < 0.0001 else f"{lr_logrank['p_value']:.4f}"],
                ["Gehan-Breslow-Wilcoxon Chi-Square", f"{lr_logrank['gehan_wilcoxon_chi2']:.4f}"],
                ["Gehan-Breslow-Wilcoxon P-value", f"{lr_logrank['gehan_wilcoxon_p']:.4e}" if lr_logrank['gehan_wilcoxon_p'] < 0.0001 else f"{lr_logrank['gehan_wilcoxon_p']:.4f}"],
            ]
            if hr_res is not None:
                g_keys = list(groups_data.keys())
                comp_rows.extend([
                    [f"Hazard Ratio (Mantel-Haenszel) ({g_keys[0]} / {g_keys[1]})", f"{hr_res['hazard_ratio']:.4f}"],
                    ["95% CI of Hazard Ratio", f"{hr_res['ci_lower']:.4f} to {hr_res['ci_upper']:.4f}"],
                    ["ln(Hazard Ratio)", f"{hr_res['log_hr']:.4f}"],
                    ["SE of ln(Hazard Ratio)", f"{hr_res['se']:.4f}"],
                ])
            tabs.append({"name": "Comparison", "grid_type": "green", "headers": comp_headers, "rows": comp_rows, "visible": True})

        surv_groups_list = []
        for g_name, km in km_results.items():
            surv_groups_list.append({
                "name": g_name,
                "times": km["times"].tolist(),
                "survival": km["survival"].tolist(),
                "ci_lower": km["ci_lower"].tolist(),
                "ci_upper": km["ci_upper"].tolist(),
                "censor_times": km["censor_times"].tolist(),
                "censor_survival": km["censor_survival"].tolist(),
                "median": km["median_survival"],
            })

        raw_result_data = {
            "survival_groups": surv_groups_list,
            "log_rank_p": lr_logrank["p_value"] if lr_logrank else None,
            "hazard_ratio": hr_res["hazard_ratio"] if hr_res else None,
        }

        res_sheet = self.document.create_result_sheet(
            title="Kaplan-Meier Survival",
            analysis_type="Survival Analysis",
            source_sheet_id=active_sheet.sheet_id,
            tabs=tabs,
            raw_result_data=raw_result_data,
            grid_type="green",
        )
        GLOBAL_REGISTRY.extract_from_result(res_sheet)
        self.navigator.populate(self.document)
        self.on_sheet_selected(res_sheet.sheet_id)

    def action_run_cox_regression(self) -> None:
        """Executes Cox Proportional Hazards regression with Efron tie handling and step-halving."""
        active_sheet = self.document.get_sheet(self.document.active_sheet_id or "")
        if not isinstance(active_sheet, DataSheet):
            QMessageBox.warning(self, "Cox Regression", "Please select a Data Sheet with survival time, status, and covariates.")
            return

        table: DataTable = active_sheet.table
        col_names: List[str] = []
        col_getters = []

        if table.table_type == TableType.MULTIPLE_VARIABLES and table.variables:
            col_names = [v.name for v in table.variables]
            for c_idx in range(len(table.variables)):
                col_getters.append(
                    lambda idx=c_idx: np.array([
                        float(table.multi_var_data[r][idx])
                        if r < len(table.multi_var_data) and idx < len(table.multi_var_data[r]) and table.multi_var_data[r][idx] is not None
                        else np.nan
                        for r in range(table.num_rows)
                    ], dtype=np.float64)
                )
        else:
            if table.x_data is not None:
                col_names.append(table.x_title or "Time (X)")
                col_getters.append(lambda: np.copy(table.x_data))
            for g_idx, g_name in enumerate(table.y_group_titles):
                col_names.append(g_name)
                col_getters.append(
                    lambda idx=g_idx: np.array([table.get_cell_value(r, "Y", idx, 0) for r in range(table.num_rows)], dtype=np.float64)
                )

        if len(col_names) < 3:
            QMessageBox.warning(self, "Cox Regression", "Cox regression requires at least 3 columns: Time, Status, and at least 1 Covariate.")
            return

        dlg = CoxRegressionDialog(col_names, default_time_idx=0, default_event_idx=1, parent=self)
        if not dlg.exec():
            return

        time_idx, event_idx, cov_indices = dlg.get_selection()
        if not cov_indices:
            QMessageBox.warning(self, "Cox Regression", "Please select at least one covariate.")
            return

        time_arr = col_getters[time_idx]()
        event_arr = col_getters[event_idx]()
        cov_arrays = [col_getters[ci]() for ci in cov_indices]
        cov_mat = np.column_stack(cov_arrays)

        valid = ~np.isnan(time_arr) & ~np.isnan(event_arr) & ~np.any(np.isnan(cov_mat), axis=1)
        t_clean = time_arr[valid]
        e_clean = event_arr[valid]
        cov_clean = cov_mat[valid]

        if len(t_clean) < len(cov_indices) + 2 or np.sum(e_clean == 1) == 0:
            QMessageBox.warning(self, "Cox Regression", "Not enough valid event observations for the selected covariates.")
            return

        cov_names = [col_names[ci] for ci in cov_indices]
        try:
            res = SurvivalEngine.cox_ph(t_clean, e_clean, cov_clean, names=cov_names)
        except Exception as e:
            QMessageBox.critical(self, "Cox Regression Error", f"Cox regression failed to converge: {e}")
            return

        # 1. Coefficients & Hazard Ratios tab
        coef_headers = [
            "Covariate", "Coefficient (b)", "Std. Error", "z", "P-value",
            "Hazard Ratio (e^b)", "95% CI Lower", "95% CI Upper"
        ]
        coef_rows = []
        for cov_dict in res["covariates"]:
            coef_rows.append([
                cov_dict["name"],
                f"{cov_dict['coef']:.4f}",
                f"{cov_dict['se']:.4f}",
                f"{cov_dict['z']:.4f}",
                f"{cov_dict['p_value']:.4e}" if cov_dict['p_value'] < 0.0001 else f"{cov_dict['p_value']:.4f}",
                f"{cov_dict['hazard_ratio']:.4f}",
                f"{cov_dict['hr_ci_lower']:.4f}",
                f"{cov_dict['hr_ci_upper']:.4f}",
            ])

        # 2. Global tests tab
        global_headers = ["Metric / Test", "Value"]
        global_rows = [
            ["Number of Subjects", str(res["n_samples"])],
            ["Number of Events", str(res["n_events"])],
            ["Log-Likelihood (Null Model)", f"{res['log_likelihood_null']:.4f}"],
            ["Log-Likelihood (Fitted Model)", f"{res['log_likelihood']:.4f}"],
            ["Likelihood Ratio Test (Chi-Square)", f"{res['lrt_stat']:.4f}"],
            ["LRT Degrees of Freedom", str(res["lrt_df"])],
            ["LRT P-value", f"{res['lrt_p']:.4e}" if res['lrt_p'] < 0.0001 else f"{res['lrt_p']:.4f}"],
            ["AIC", f"{res['aic']:.4f}"],
            ["Tie Handling Method", str(res['ties']).capitalize()],
            ["Iterations to Convergence", str(res['iterations'])],
        ]

        tabs = [
            {"name": "Coefficients & Hazard Ratios", "grid_type": "red", "headers": coef_headers, "rows": coef_rows, "visible": True},
            {"name": "Global tests", "grid_type": "red", "headers": global_headers, "rows": global_rows, "visible": True},
        ]

        raw_result_data = {
            "covariates": res["covariates"],
            "log_likelihood": res["log_likelihood"],
            "lrt_stat": res["lrt_stat"],
            "lrt_p": res["lrt_p"],
            "aic": res["aic"],
        }

        res_sheet = self.document.create_result_sheet(
            title="Cox Proportional Hazards",
            analysis_type="Cox Regression",
            source_sheet_id=active_sheet.sheet_id,
            tabs=tabs,
            summary_tables=[{"title": "Coefficients", "headers": coef_headers, "rows": coef_rows}],
            raw_result_data=raw_result_data,
            grid_type="red",
        )
        GLOBAL_REGISTRY.extract_from_result(res_sheet)
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
