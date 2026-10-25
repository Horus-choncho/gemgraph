"""
UI and Analysis Action Integration Tests for GemGraph.
Tests MainWindow navigation, contextual toolbars, analysis actions, and DAG recalculation.
"""

from __future__ import annotations
import os
import tempfile
import unittest
import unittest.mock
import numpy as np
from PyQt6.QtWidgets import QApplication, QFileDialog
from PyQt6.QtCore import Qt, QDir, QUrl, QMimeData, QPointF
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QDragMoveEvent

from src.core.sheet import DataSheet, ResultSheet, SheetType
from src.core.table import TableType
from src.core.hooking import GLOBAL_REGISTRY
from src.ui.main_window import MainWindow

app = QApplication.instance()
if app is None:
    app = QApplication([])


class TestMainWindowAndActions(unittest.TestCase):
    def setUp(self) -> None:
        GLOBAL_REGISTRY.clear()
        self.window = MainWindow()

    def tearDown(self) -> None:
        if hasattr(self, "window") and self.window:
            self.window._is_modified = False
            self.window.close()

    def test_initial_state_and_contextual_toolbar(self) -> None:
        """Verifies initial window state and contextual toolbar configuration."""
        self.assertIsNotNone(self.window.document)
        self.assertGreaterEqual(len(self.window.document.sheets), 1)

        # XY contextual toolbar has linear, nonlin, etc.
        self.window._update_contextual_toolbar(TableType.XY)
        actions = [a.text() for a in self.window.context_toolbar.actions()]
        self.assertIn("Linear Reg", actions)
        self.assertIn("Nonlin Reg", actions)

        # Column toolbar
        self.window._update_contextual_toolbar(TableType.COLUMN)
        actions = [a.text() for a in self.window.context_toolbar.actions()]
        self.assertIn("1-Way ANOVA", actions)
        self.assertIn("t-test", actions)

        # Grouped toolbar
        self.window._update_contextual_toolbar(TableType.GROUPED)
        actions = [a.text() for a in self.window.context_toolbar.actions()]
        self.assertIn("2-Way ANOVA", actions)

    def test_linear_regression_action(self) -> None:
        """Tests action_run_linear generating a valid ResultSheet with parameters and curve tab."""
        active_sheet = self.window.document.get_sheet(self.window.document.active_sheet_id or "")
        self.assertIsInstance(active_sheet, DataSheet)
        table = active_sheet.table
        table.x_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # Y = 2X + 1
        for r, x in enumerate(table.x_data):
            table.set_cell_value(r, "Y", 0, 0, float(2.0 * x + 1.0))

        self.window.action_run_linear()

        current_sheet = self.window.document.get_sheet(self.window.document.active_sheet_id or "")
        self.assertIsInstance(current_sheet, ResultSheet)
        self.assertEqual(current_sheet.analysis_type, "Linear Regression")
        self.assertEqual(len(current_sheet.tabs), 2)
        self.assertEqual(current_sheet.tabs[0]["name"], "Parameters")
        self.assertEqual(current_sheet.tabs[1]["name"], "Curve Coordinates")

        # Slope should be approximately 2.0
        raw = current_sheet.raw_result_data
        slope = raw["parameters"][0]["best_fit"]
        self.assertAlmostEqual(slope, 2.0, places=3)

    def test_descriptive_stats_and_correlation_actions(self) -> None:
        """Tests descriptive statistics and correlation matrix actions."""
        # Create column data sheet
        sheet = self.window.document.create_data_sheet(
            title="ColData",
            table_type=TableType.COLUMN,
            num_rows=5,
            num_replicates=1,
            y_group_titles=["Control", "Treated"],
        )
        self.window.on_sheet_selected(sheet.sheet_id)
        for r in range(5):
            sheet.table.set_cell_value(r, "Y", 0, 0, float(10 + r))
            sheet.table.set_cell_value(r, "Y", 1, 0, float(20 + 2 * r))

        self.window.action_run_descriptive_stats()
        res_sheet = self.window.document.get_sheet(self.window.document.active_sheet_id or "")
        self.assertIsInstance(res_sheet, ResultSheet)
        self.assertEqual(res_sheet.analysis_type, "Column Statistics")
        self.assertEqual(res_sheet.tabs[0]["grid_type"], "green")

        # Test correlation
        self.window.on_sheet_selected(sheet.sheet_id)
        self.window.action_run_correlation()
        corr_sheet = self.window.document.get_sheet(self.window.document.active_sheet_id or "")
        self.assertIsInstance(corr_sheet, ResultSheet)
        self.assertEqual(corr_sheet.analysis_type, "Correlation")
        # Perfect correlation between 10+r and 20+2r -> r = 1.0
        r_rows = corr_sheet.tabs[0]["rows"]
        self.assertAlmostEqual(float(r_rows[0][2]), 1.0, places=3)

    def test_table_transform_and_manipulation_actions(self) -> None:
        """Tests transform, normalize, prune, and transpose actions creating child sheets in DAG."""
        sheet = self.window.document.create_data_sheet(
            title="RawData",
            table_type=TableType.COLUMN,
            num_rows=6,
            num_replicates=1,
            y_group_titles=["Values"],
        )
        self.window.on_sheet_selected(sheet.sheet_id)
        for r in range(6):
            sheet.table.set_cell_value(r, "Y", 0, 0, float((r + 1) * 10))

        # 1. Transform log10
        self.window.action_run_transform()
        tf_sheet = self.window.document.get_sheet(self.window.document.active_sheet_id or "")
        self.assertIsInstance(tf_sheet, DataSheet)
        self.assertIn(tf_sheet.sheet_id, sheet.child_sheet_ids)
        # log10(10) = 1.0, log10(100) = 2.0
        self.assertAlmostEqual(tf_sheet.table.get_cell_value(0, "Y", 0, 0), 1.0, places=3)

        # 2. Normalize
        self.window.on_sheet_selected(sheet.sheet_id)
        self.window.action_run_normalize()
        norm_sheet = self.window.document.get_sheet(self.window.document.active_sheet_id or "")
        self.assertIsInstance(norm_sheet, DataSheet)
        # Min=10 -> 0%, Max=60 -> 100%
        self.assertAlmostEqual(norm_sheet.table.get_cell_value(0, "Y", 0, 0), 0.0, places=3)
        self.assertAlmostEqual(norm_sheet.table.get_cell_value(5, "Y", 0, 0), 100.0, places=3)

        # 3. Prune rows
        self.window.on_sheet_selected(sheet.sheet_id)
        self.window.action_run_prune()
        prune_sheet = self.window.document.get_sheet(self.window.document.active_sheet_id or "")
        self.assertIsInstance(prune_sheet, DataSheet)
        self.assertEqual(prune_sheet.table.num_rows, 3)

        # 4. Transpose
        self.window.on_sheet_selected(sheet.sheet_id)
        self.window.action_run_transpose()
        tp_sheet = self.window.document.get_sheet(self.window.document.active_sheet_id or "")
        self.assertIsInstance(tp_sheet, DataSheet)
        self.assertEqual(len(tp_sheet.table.y_group_titles), 6)

    def test_survival_and_contingency_actions(self) -> None:
        """Tests Kaplan-Meier survival analysis and Contingency Chi-Square actions."""
        # Survival
        surv_sheet = self.window.document.create_data_sheet(
            title="Survival",
            table_type=TableType.SURVIVAL,
            num_rows=4,
            num_replicates=1,
            y_group_titles=["Status"],
        )
        self.window.on_sheet_selected(surv_sheet.sheet_id)
        surv_sheet.table.x_data = np.array([5.0, 10.0, 15.0, 20.0])
        # 3 events, 1 censored
        surv_sheet.table.set_cell_value(0, "Y", 0, 0, 1.0)
        surv_sheet.table.set_cell_value(1, "Y", 0, 0, 1.0)
        surv_sheet.table.set_cell_value(2, "Y", 0, 0, 0.0)
        surv_sheet.table.set_cell_value(3, "Y", 0, 0, 1.0)

        self.window.action_run_survival()
        res_surv = self.window.document.get_sheet(self.window.document.active_sheet_id or "")
        self.assertIsInstance(res_surv, ResultSheet)
        self.assertEqual(res_surv.analysis_type, "Survival Analysis")

        # Contingency (2x2)
        cont_sheet = self.window.document.create_data_sheet(
            title="Contingency",
            table_type=TableType.CONTINGENCY,
            num_rows=2,
            num_replicates=1,
            y_group_titles=["Success", "Failure"],
        )
        self.window.on_sheet_selected(cont_sheet.sheet_id)
        cont_sheet.table.set_cell_value(0, "Y", 0, 0, 20.0)
        cont_sheet.table.set_cell_value(0, "Y", 1, 0, 10.0)
        cont_sheet.table.set_cell_value(1, "Y", 0, 0, 5.0)
        cont_sheet.table.set_cell_value(1, "Y", 1, 0, 25.0)

        self.window.action_run_contingency_test()
        res_cont = self.window.document.get_sheet(self.window.document.active_sheet_id or "")
        self.assertIsInstance(res_cont, ResultSheet)
        self.assertEqual(res_cont.analysis_type, "Contingency Test")

    def test_simulation_actions(self) -> None:
        """Tests simulate XY and simulate column actions."""
        self.window.action_run_simulate_xy()
        xy_sheet = self.window.document.get_sheet(self.window.document.active_sheet_id or "")
        self.assertIsInstance(xy_sheet, DataSheet)
        self.assertEqual(xy_sheet.table.table_type, TableType.XY)
        self.assertEqual(xy_sheet.table.num_rows, 8)

        self.window.action_run_simulate_column()
        col_sheet = self.window.document.get_sheet(self.window.document.active_sheet_id or "")
        self.assertIsInstance(col_sheet, DataSheet)
        self.assertEqual(col_sheet.table.table_type, TableType.COLUMN)
        self.assertEqual(len(col_sheet.table.y_group_titles), 3)

    def test_statistical_testing_suite(self) -> None:
        """Tests t-test, multiple t-tests, nested analysis, and goodness of fit."""
        # 1. t-test
        t_sheet = self.window.document.create_data_sheet(
            title="TTestSheet",
            table_type=TableType.COLUMN,
            num_rows=4,
            num_replicates=1,
            y_group_titles=["Group1", "Group2"],
        )
        self.window.on_sheet_selected(t_sheet.sheet_id)
        for r in range(4):
            t_sheet.table.set_cell_value(r, "Y", 0, 0, float(10 + r))
            t_sheet.table.set_cell_value(r, "Y", 1, 0, float(20 + r))

        self.window.action_run_t_test()
        res_t = self.window.document.get_sheet(self.window.document.active_sheet_id or "")
        self.assertIsInstance(res_t, ResultSheet)
        self.assertEqual(res_t.analysis_type, "t-test")

        # 2. Multiple t-tests
        mt_sheet = self.window.document.create_data_sheet(
            title="MultTTest",
            table_type=TableType.GROUPED,
            num_rows=3,
            num_replicates=2,
            y_group_titles=["Ctrl", "Trt"],
        )
        self.window.on_sheet_selected(mt_sheet.sheet_id)
        for r in range(3):
            mt_sheet.table.set_cell_value(r, "Y", 0, 0, 10.0 + r)
            mt_sheet.table.set_cell_value(r, "Y", 0, 1, 11.0 + r)
            mt_sheet.table.set_cell_value(r, "Y", 1, 0, 20.0 + r)
            mt_sheet.table.set_cell_value(r, "Y", 1, 1, 22.0 + r)

        self.window.action_run_multiple_t_tests()
        res_mt = self.window.document.get_sheet(self.window.document.active_sheet_id or "")
        self.assertIsInstance(res_mt, ResultSheet)
        self.assertEqual(res_mt.analysis_type, "Multiple t-tests")

        # 3. Goodness of Fit
        gof_sheet = self.window.document.create_data_sheet(
            title="GOF",
            table_type=TableType.PARTS_OF_WHOLE,
            num_rows=3,
            num_replicates=1,
            y_group_titles=["Observed"],
        )
        self.window.on_sheet_selected(gof_sheet.sheet_id)
        gof_sheet.table.set_cell_value(0, "Y", 0, 0, 50.0)
        gof_sheet.table.set_cell_value(1, "Y", 0, 0, 30.0)
        gof_sheet.table.set_cell_value(2, "Y", 0, 0, 20.0)

        self.window.action_run_goodness_of_fit()
        res_gof = self.window.document.get_sheet(self.window.document.active_sheet_id or "")
        self.assertIsInstance(res_gof, ResultSheet)
        self.assertEqual(res_gof.analysis_type, "Goodness-of-Fit")

    def test_multivariate_and_regression_suite(self) -> None:
        """Tests multiple linear regression, PCA, and interpolation actions."""
        # 1. Multiple regression
        reg_sheet = self.window.document.create_data_sheet(
            title="MultReg",
            table_type=TableType.COLUMN,
            num_rows=6,
            num_replicates=1,
            y_group_titles=["Y", "X1", "X2"],
        )
        self.window.on_sheet_selected(reg_sheet.sheet_id)
        for r in range(6):
            x1 = float(r)
            x2 = float(r * 2)
            y = 3.0 + 2.0 * x1 + 1.5 * x2
            reg_sheet.table.set_cell_value(r, "Y", 0, 0, y)
            reg_sheet.table.set_cell_value(r, "Y", 1, 0, x1)
            reg_sheet.table.set_cell_value(r, "Y", 2, 0, x2)

        self.window.action_run_multiple_reg()
        res_reg = self.window.document.get_sheet(self.window.document.active_sheet_id or "")
        self.assertIsInstance(res_reg, ResultSheet)
        self.assertEqual(res_reg.analysis_type, "Multiple Regression")

        # 2. PCA
        pca_sheet = self.window.document.create_data_sheet(
            title="PCASheet",
            table_type=TableType.COLUMN,
            num_rows=5,
            num_replicates=1,
            y_group_titles=["Var1", "Var2", "Var3"],
        )
        self.window.on_sheet_selected(pca_sheet.sheet_id)
        for r in range(5):
            pca_sheet.table.set_cell_value(r, "Y", 0, 0, float(r * 3))
            pca_sheet.table.set_cell_value(r, "Y", 1, 0, float(r * 2 + 1))
            pca_sheet.table.set_cell_value(r, "Y", 2, 0, float(r * 5 - 2))

        self.window.action_run_pca()
        res_pca = self.window.document.get_sheet(self.window.document.active_sheet_id or "")
        self.assertIsInstance(res_pca, ResultSheet)
        self.assertEqual(res_pca.analysis_type, "PCA")

        # 3. Interpolate
        interp_sheet = self.window.document.create_data_sheet(
            title="InterpSheet",
            table_type=TableType.XY,
            num_rows=5,
            num_replicates=1,
            y_group_titles=["Standard"],
        )
        self.window.on_sheet_selected(interp_sheet.sheet_id)
        interp_sheet.table.x_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        for r, x in enumerate(interp_sheet.table.x_data):
            interp_sheet.table.set_cell_value(r, "Y", 0, 0, float(x * 10))

        self.window.action_run_interpolate()
        res_interp = self.window.document.get_sheet(self.window.document.active_sheet_id or "")
        self.assertIsInstance(res_interp, ResultSheet)
        self.assertEqual(res_interp.analysis_type, "Interpolate")

    def test_dag_recalculation_on_cell_modification(self) -> None:
        """Verifies modifying a data cell recalculates downstream result sheets."""
        active_sheet = self.window.document.get_sheet(self.window.document.active_sheet_id or "")
        self.assertIsInstance(active_sheet, DataSheet)
        table = active_sheet.table
        table.x_data = np.array([-9.0, -8.0, -7.0, -6.0, -5.0])
        for r, x in enumerate(table.x_data):
            y_val = 10.0 + 90.0 / (1.0 + 10.0 ** (-6.0 - x))
            table.set_cell_value(r, "Y", 0, 0, y_val)

        # Create child result sheet with calc_spec
        res = self.window.document.create_result_sheet(
            title="Dose Response Curve",
            analysis_type="Nonlinear Regression",
            source_sheet_id=active_sheet.sheet_id,
            calc_spec={
                "type": "nonlinear",
                "model_name": "Sigmoidal Dose-Response (4PL)",
                "group_idx": 0,
            },
        )
        self.assertIn(res.sheet_id, active_sheet.child_sheet_ids)

        # Trigger table modification
        table.set_cell_value(0, "Y", 0, 0, 11.5)
        self.window._on_table_data_modified()

        # Recalculated parameters should exist in result sheet
        updated_res = self.window.document.get_sheet(res.sheet_id)
        self.assertIsInstance(updated_res, ResultSheet)
        self.assertIn("params", updated_res.raw_result_data)
        self.assertIn("LogEC50", updated_res.raw_result_data["params"])

    def test_window_accept_drops_enabled(self) -> None:
        """Verifies MainWindow accepts window-level drops."""
        self.assertTrue(self.window.acceptDrops())

    def test_smart_initial_dir_priority(self) -> None:
        """Verifies hierarchy: /uploads -> samples/ -> home/cwd."""
        # 1. /uploads exists
        with unittest.mock.patch("os.path.exists", side_effect=lambda p: p == "/uploads"), \
             unittest.mock.patch("os.path.isdir", side_effect=lambda p: p == "/uploads"):
            self.assertEqual(MainWindow.get_smart_initial_dir(), "/uploads")

        # 2. samples exists in workspace
        with unittest.mock.patch("os.path.exists", side_effect=lambda p: "samples" in str(p)), \
             unittest.mock.patch("os.path.isdir", side_effect=lambda p: "samples" in str(p)):
            res = MainWindow.get_smart_initial_dir()
            self.assertTrue(res.endswith("samples"))

        # 3. Neither exists: falls back to QDir.homePath() if valid
        with unittest.mock.patch("os.path.exists", side_effect=lambda p: p == "/test/home"), \
             unittest.mock.patch("os.path.isdir", side_effect=lambda p: p == "/test/home"), \
             unittest.mock.patch("PyQt6.QtCore.QDir.homePath", return_value="/test/home"):
            self.assertEqual(MainWindow.get_smart_initial_dir(), "/test/home")

        # 4. Neither exists and home does not exist: falls back to cwd
        with unittest.mock.patch("os.path.exists", return_value=False), \
             unittest.mock.patch("os.path.isdir", return_value=False):
            res = MainWindow.get_smart_initial_dir()
            self.assertEqual(res, os.getcwd())

    def test_open_project_dialog_sidebar_shortcuts_and_preset_paths(self) -> None:
        """Verifies open project dialog contains pre-set sidebar shortcuts."""
        dialog = self.window.create_open_project_dialog()
        self.assertEqual(dialog.fileMode(), QFileDialog.FileMode.ExistingFile)
        sidebar_urls = [u.toLocalFile() for u in dialog.sidebarUrls()]
        self.assertTrue(any("/uploads" in u for u in sidebar_urls))
        self.assertTrue(any("samples" in u for u in sidebar_urls))

        preset_paths = MainWindow.get_preset_paths()
        self.assertIn("/uploads", preset_paths)
        self.assertTrue(any("samples" in p for p in preset_paths))

    def test_drag_enter_and_move_events(self) -> None:
        """Verifies dragEnter and dragMove accept .prism, .pzfx, and .csv while rejecting others."""
        for ext in (".prism", ".pzfx", ".csv"):
            mime = QMimeData()
            mime.setUrls([QUrl.fromLocalFile(f"/tmp/test_file{ext}")])
            evt = QDragEnterEvent(
                QPointF(10, 10).toPoint(),
                Qt.DropAction.CopyAction,
                mime,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
            self.window.dragEnterEvent(evt)
            self.assertTrue(evt.isAccepted(), f"Failed to accept {ext}")

            evt_move = QDragMoveEvent(
                QPointF(10, 10).toPoint(),
                Qt.DropAction.CopyAction,
                mime,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
            self.window.dragMoveEvent(evt_move)
            self.assertTrue(evt_move.isAccepted(), f"Failed to accept {ext} on move")

        # Rejections
        for invalid_ext in (".txt", ".exe", ".pdf", ".png"):
            mime = QMimeData()
            mime.setUrls([QUrl.fromLocalFile(f"/tmp/test_file{invalid_ext}")])
            evt = QDragEnterEvent(
                QPointF(10, 10).toPoint(),
                Qt.DropAction.CopyAction,
                mime,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
            self.window.dragEnterEvent(evt)
            self.assertFalse(evt.isAccepted(), f"Should reject {invalid_ext}")

        # Empty mime data
        empty_mime = QMimeData()
        evt_empty = QDragEnterEvent(
            QPointF(10, 10).toPoint(),
            Qt.DropAction.CopyAction,
            empty_mime,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        self.window.dragEnterEvent(evt_empty)
        self.assertFalse(evt_empty.isAccepted())

    def test_drop_event_loads_prism_file(self) -> None:
        """Verifies dropping a .prism project triggers open_project, updates spreadsheet and canvas."""
        sample_path = os.path.join(os.path.dirname(__file__), "..", "samples", "test_sample.prism")
        self.assertTrue(os.path.exists(sample_path))

        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(os.path.abspath(sample_path))])
        drop_evt = QDropEvent(
            QPointF(10, 10),
            Qt.DropAction.CopyAction,
            mime,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        self.window.dropEvent(drop_evt)
        self.assertTrue(drop_evt.isAccepted())

        # Check document, spreadsheet model, and plot canvas
        self.assertGreater(len(self.window.document.sheets), 1)
        self.assertGreater(self.window.spreadsheet_model.rowCount(), 0)
        self.assertGreater(len(self.window.graph_canvas.plot_widget.plotItem.items), 0)

    def test_drop_event_loads_csv_file(self) -> None:
        """Verifies dropping a CSV file parses table, populates spreadsheet, and plots data."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("X,Control,Treated\n1.0,10.0,20.0\n2.0,15.0,25.0\n3.0,30.0,45.0\n")
            csv_path = f.name

        try:
            mime = QMimeData()
            mime.setUrls([QUrl.fromLocalFile(csv_path)])
            drop_evt = QDropEvent(
                QPointF(10, 10),
                Qt.DropAction.CopyAction,
                mime,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
            self.window.dropEvent(drop_evt)
            self.assertTrue(drop_evt.isAccepted())

            self.assertGreaterEqual(len(self.window.document.sheets), 1)
            active_sheet = self.window.document.get_sheet(self.window.document.active_sheet_id or "")
            self.assertIsInstance(active_sheet, DataSheet)
            self.assertEqual(active_sheet.table.table_type, TableType.XY)
            self.assertEqual(active_sheet.table.get_cell_value(0, "X", 0, 0), 1.0)
            self.assertEqual(active_sheet.table.get_cell_value(0, "Y", 0, 0), 10.0)
            self.assertGreater(self.window.spreadsheet_model.rowCount(), 0)
            self.assertGreater(len(self.window.graph_canvas.plot_widget.plotItem.items), 0)
        finally:
            if os.path.exists(csv_path):
                os.remove(csv_path)

    def test_open_project_errors_and_formats(self) -> None:
        """Verifies open_project exception handling and format loading."""
        with self.assertRaises(FileNotFoundError):
            self.window.open_project("/path/to/non_existent_file.prism")

    def test_drop_event_loads_empty_csv(self) -> None:
        """Verifies dropping an empty CSV file does not crash and creates a blank sheet."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            csv_path = f.name

        try:
            mime = QMimeData()
            mime.setUrls([QUrl.fromLocalFile(csv_path)])
            drop_evt = QDropEvent(
                QPointF(10, 10),
                Qt.DropAction.CopyAction,
                mime,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
            self.window.dropEvent(drop_evt)
            self.assertTrue(drop_evt.isAccepted())
            self.assertGreaterEqual(len(self.window.document.sheets), 1)
        finally:
            if os.path.exists(csv_path):
                os.remove(csv_path)

    def test_drop_event_loads_semicolon_csv(self) -> None:
        """Verifies dropping a European semicolon-delimited CSV parses XY table with correct groups."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("X;Control;Treated\n1.0;10.0;20.0\n2.0;15.0;25.0\n3.0;30.0;45.0\n")
            csv_path = f.name

        try:
            mime = QMimeData()
            mime.setUrls([QUrl.fromLocalFile(csv_path)])
            drop_evt = QDropEvent(
                QPointF(10, 10),
                Qt.DropAction.CopyAction,
                mime,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
            self.window.dropEvent(drop_evt)
            self.assertTrue(drop_evt.isAccepted())

            active_sheet = self.window.document.get_sheet(self.window.document.active_sheet_id or "")
            self.assertIsInstance(active_sheet, DataSheet)
            self.assertEqual(active_sheet.table.table_type, TableType.XY)
            self.assertEqual(active_sheet.table.y_group_titles, ["Control", "Treated"])
            self.assertEqual(active_sheet.table.get_cell_value(0, "X", 0, 0), 1.0)
            self.assertEqual(active_sheet.table.get_cell_value(0, "Y", 0, 0), 10.0)
            self.assertEqual(active_sheet.table.get_cell_value(0, "Y", 1, 0), 20.0)
        finally:
            if os.path.exists(csv_path):
                os.remove(csv_path)

    def test_drop_event_loads_tab_delimited_csv(self) -> None:
        """Verifies dropping a tab-delimited CSV parses XY table."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("X\tControl\tTreated\n1.0\t10.0\t20.0\n2.0\t15.0\t25.0\n")
            csv_path = f.name

        try:
            mime = QMimeData()
            mime.setUrls([QUrl.fromLocalFile(csv_path)])
            drop_evt = QDropEvent(
                QPointF(10, 10),
                Qt.DropAction.CopyAction,
                mime,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
            self.window.dropEvent(drop_evt)
            self.assertTrue(drop_evt.isAccepted())

            active_sheet = self.window.document.get_sheet(self.window.document.active_sheet_id or "")
            self.assertIsInstance(active_sheet, DataSheet)
            self.assertEqual(active_sheet.table.table_type, TableType.XY)
            self.assertEqual(active_sheet.table.y_group_titles, ["Control", "Treated"])
        finally:
            if os.path.exists(csv_path):
                os.remove(csv_path)

    def test_drop_event_multi_file_fallback(self) -> None:
        """Verifies dropping multiple files selects the supported file even if urls[0] is unsupported."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("X,GroupA\n1.0,5.0\n2.0,10.0\n")
            csv_path = f.name

        try:
            mime = QMimeData()
            mime.setUrls([
                QUrl.fromLocalFile("/tmp/unsupported_readme.txt"),
                QUrl.fromLocalFile(csv_path),
            ])

            evt_enter = QDragEnterEvent(
                QPointF(10, 10).toPoint(),
                Qt.DropAction.CopyAction,
                mime,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
            self.window.dragEnterEvent(evt_enter)
            self.assertTrue(evt_enter.isAccepted())

            drop_evt = QDropEvent(
                QPointF(10, 10),
                Qt.DropAction.CopyAction,
                mime,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
            self.window.dropEvent(drop_evt)
            self.assertTrue(drop_evt.isAccepted())

            active_sheet = self.window.document.get_sheet(self.window.document.active_sheet_id or "")
            self.assertIsInstance(active_sheet, DataSheet)
            self.assertEqual(active_sheet.table.get_cell_value(0, "X", 0, 0), 1.0)
        finally:
            if os.path.exists(csv_path):
                os.remove(csv_path)

    def test_drop_event_url_with_spaces_and_encoding(self) -> None:
        """Verifies file paths with spaces and percent encoding are resolved and loaded properly."""
        temp_dir = tempfile.mkdtemp()
        spaced_path = os.path.join(temp_dir, "my spaced dataset.csv")
        with open(spaced_path, "w") as f:
            f.write("X,SpacedCol\n10.0,20.0\n")

        try:
            # Test QUrl representation with encoded spaces
            mime = QMimeData()
            mime.setUrls([QUrl(f"file://{spaced_path.replace(' ', '%20')}")])
            evt_enter = QDragEnterEvent(
                QPointF(10, 10).toPoint(),
                Qt.DropAction.CopyAction,
                mime,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
            self.window.dragEnterEvent(evt_enter)
            self.assertTrue(evt_enter.isAccepted())

            drop_evt = QDropEvent(
                QPointF(10, 10),
                Qt.DropAction.CopyAction,
                mime,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
            self.window.dropEvent(drop_evt)
            self.assertTrue(drop_evt.isAccepted())

            active_sheet = self.window.document.get_sheet(self.window.document.active_sheet_id or "")
            self.assertIsInstance(active_sheet, DataSheet)
            self.assertEqual(active_sheet.table.get_cell_value(0, "X", 0, 0), 10.0)
        finally:
            if os.path.exists(spaced_path):
                os.remove(spaced_path)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)

    def test_dynamic_window_title_lifecycle(self) -> None:
        """Verifies window title dynamically updates across new, open, save, load, and drop events."""
        # 1. Initial fresh launch: "Untitled"
        self.assertEqual(
            self.window.windowTitle(),
            "Untitled - GemGraph - v0.1.2-nightly - Clean Room Edition"
        )

        # 2. Open sample project (.bigo)
        self.window.action_open_sample_project()
        self.assertEqual(
            self.window.windowTitle(),
            "textbook_sample.bigo - GemGraph - v0.1.2-nightly - Clean Room Edition"
        )

        # 3. New project resets to "Untitled"
        self.window.action_new_project()
        self.assertEqual(
            self.window.windowTitle(),
            "Untitled - GemGraph - v0.1.2-nightly - Clean Room Edition"
        )

        # 4. Load Sample Datasets (in-memory) marks as "Untitled"
        self.window.action_load_sample_datasets()
        self.assertEqual(
            self.window.windowTitle(),
            "Untitled - GemGraph - v0.1.2-nightly - Clean Room Edition"
        )

        # 5. Saving updates title to file basename
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "my_experiment.bigo")
            with unittest.mock.patch("PyQt6.QtWidgets.QFileDialog.getSaveFileName", return_value=(save_path, "")):
                self.window.action_save_project()
                self.assertEqual(
                    self.window.windowTitle(),
                    "my_experiment.bigo - GemGraph - v0.1.2-nightly - Clean Room Edition"
                )

        # 6. Drag and drop file updates title
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            f.write(b"Time,Conc\n1,10\n2,20\n")
            f_path = f.name
        try:
            mime = QMimeData()
            mime.setUrls([QUrl.fromLocalFile(f_path)])
            drop_evt = QDropEvent(
                QPointF(10, 10),
                Qt.DropAction.CopyAction,
                mime,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
            self.window.dropEvent(drop_evt)
            expected_name = os.path.basename(f_path)
            self.assertEqual(
                self.window.windowTitle(),
                f"{expected_name} - GemGraph - v0.1.2-nightly - Clean Room Edition"
            )
        finally:
            if os.path.exists(f_path):
                os.remove(f_path)

    def test_load_sample_datasets_creates_expanded_tables(self) -> None:
        """Verifies Load Sample Datasets creates all 7 deterministic tables."""
        self.window.action_load_sample_datasets()

        data_sheets = self.window.document.get_sheets_by_type(SheetType.DATA)
        self.assertEqual(len(data_sheets), 7)

        titles = [s.title for s in data_sheets]
        self.assertIn("Enzyme Kinetics (Michaelis-Menten)", titles)
        self.assertIn("Dose-Response 4PL", titles)
        self.assertIn("One-Way ANOVA", titles)
        self.assertIn("Kaplan-Meier Survival", titles)
        self.assertIn("Cox Survival Cohort", titles)
        self.assertIn("Distribution Comparison (Violin/Box)", titles)
        self.assertIn("PCA Multi-Variable Matrix", titles)

        # Verify deterministic values
        ek_sheet = [s for s in data_sheets if s.title == "Enzyme Kinetics (Michaelis-Menten)"][0]
        self.assertEqual(ek_sheet.table.table_type, TableType.XY)
        self.assertEqual(ek_sheet.table.num_rows, 10)
        self.assertEqual(ek_sheet.table.get_cell_value(0, "X", 0, 0), 0.5)

    def test_enzyme_kinetics_action_creates_result_sheet(self) -> None:
        """Verifies action_run_enzyme_kinetics creates ResultSheet with all 4 expected tabs."""
        self.window.action_load_sample_datasets()
        ek_sheet = [s for s in self.window.document.get_sheets_by_type(SheetType.DATA) if "Enzyme" in s.title][0]
        self.window.on_sheet_selected(ek_sheet.sheet_id)

        from src.math.nonlinear import ConvergencePreset
        settings = (
            "Michaelis-Menten",
            0,
            True,
            True,
            "Lineweaver-Burk (1/X, 1/Y)",
            ConvergencePreset.STANDARD,
        )
        with unittest.mock.patch("src.ui.dialogs.EnzymeKineticsDialog.exec", return_value=True), \
             unittest.mock.patch("src.ui.dialogs.EnzymeKineticsDialog.get_settings", return_value=settings):
            self.window.action_run_enzyme_kinetics()

        active_sheet = self.window.document.get_sheet(self.window.document.active_sheet_id or "")
        self.assertIsInstance(active_sheet, ResultSheet)
        self.assertEqual(active_sheet.analysis_type, "Enzyme Kinetics")

        tab_names = [t["name"] for t in active_sheet.tabs]
        self.assertIn("Table of results", tab_names)
        self.assertIn("Curve", tab_names)
        self.assertIn("Model comparison", tab_names)
        self.assertIn("Diagnostic transform", tab_names)

    def test_cox_regression_action_creates_result_sheet(self) -> None:
        """Verifies action_run_cox_regression creates ResultSheet with expected tabs."""
        self.window.action_load_sample_datasets()
        cox_sheet = [s for s in self.window.document.get_sheets_by_type(SheetType.DATA) if "Cox" in s.title][0]
        self.window.on_sheet_selected(cox_sheet.sheet_id)

        # Time is col 0, Status is col 1, Covariates are cols 2, 3
        selection = (0, 1, [2, 3])
        with unittest.mock.patch("src.ui.dialogs.CoxRegressionDialog.exec", return_value=True), \
             unittest.mock.patch("src.ui.dialogs.CoxRegressionDialog.get_selection", return_value=selection):
            self.window.action_run_cox_regression()

        active_sheet = self.window.document.get_sheet(self.window.document.active_sheet_id or "")
        self.assertIsInstance(active_sheet, ResultSheet)
        self.assertEqual(active_sheet.analysis_type, "Cox Regression")
        self.assertEqual(active_sheet.grid_type, "red")

        tab_names = [t["name"] for t in active_sheet.tabs]
        self.assertIn("Coefficients & Hazard Ratios", tab_names)
        self.assertIn("Global tests", tab_names)
        for t in active_sheet.tabs:
            self.assertEqual(t.get("grid_type"), "red")

        # Verify hooked constants extracted
        self.assertIsNotNone(GLOBAL_REGISTRY.get(f"{active_sheet.sheet_id}_treatment_hr"))

    def test_data_processing_actions_and_fdr_monotonicity(self) -> None:
        """Verifies remove_baseline, fraction_of_total, extract_and_rearrange, and BH-FDR monotonicity."""
        sheet = self.window.document.create_data_sheet(
            title="BaselineRaw",
            table_type=TableType.COLUMN,
            num_rows=4,
            num_replicates=1,
            y_group_titles=["G1", "G2"],
        )
        self.window.on_sheet_selected(sheet.sheet_id)
        for r, v in enumerate([10.0, 20.0, 30.0, 40.0]):
            sheet.table.set_cell_value(r, "Y", 0, 0, v)
            sheet.table.set_cell_value(r, "Y", 1, 0, v * 2.0)

        # 1. Remove Baseline
        self.window.action_run_remove_baseline()
        bl_sheet = self.window.document.get_sheet(self.window.document.active_sheet_id or "")
        self.assertIsInstance(bl_sheet, DataSheet)
        self.assertAlmostEqual(bl_sheet.table.get_cell_value(0, "Y", 0, 0), 0.0, places=4)
        self.assertAlmostEqual(bl_sheet.table.get_cell_value(3, "Y", 0, 0), 30.0, places=4)

        # 2. Fraction of Total
        self.window.on_sheet_selected(sheet.sheet_id)
        self.window.action_run_fraction_of_total()
        ft_sheet = self.window.document.get_sheet(self.window.document.active_sheet_id or "")
        self.assertIsInstance(ft_sheet, DataSheet)
        # Sum of [10, 20, 30, 40] = 100 -> percentages equal [10, 20, 30, 40]
        self.assertAlmostEqual(ft_sheet.table.get_cell_value(0, "Y", 0, 0), 10.0, places=4)
        self.assertAlmostEqual(ft_sheet.table.get_cell_value(3, "Y", 0, 0), 40.0, places=4)

        # 3. Extract and Rearrange on Multiple Variables sheet
        self.window.action_load_sample_datasets()
        mv_sheet = [s for s in self.window.document.get_sheets_by_type(SheetType.DATA) if s.table.table_type == TableType.MULTIPLE_VARIABLES][0]
        self.window.on_sheet_selected(mv_sheet.sheet_id)
        self.window.action_run_extract_and_rearrange()
        ext_sheet = self.window.document.get_sheet(self.window.document.active_sheet_id or "")
        self.assertIsInstance(ext_sheet, DataSheet)
        self.assertEqual(ext_sheet.table.table_type, TableType.COLUMN)

        # 4. Multiple t-tests BH-FDR monotonicity & excluded cell filtering
        mt_sheet = self.window.document.create_data_sheet(
            title="FDRTest",
            table_type=TableType.GROUPED,
            num_rows=3,
            num_replicates=3,
            y_group_titles=["Ctrl", "Trt"],
        )
        self.window.on_sheet_selected(mt_sheet.sheet_id)
        # Row 0: p ~ very small, Row 1: p ~ 0.04, Row 2: p ~ 0.045 (where naive p*m/rank could violate monotonicity)
        for s, v in enumerate([10.0, 10.1, 9.9]):
            mt_sheet.table.set_cell_value(0, "Y", 0, s, v)
            mt_sheet.table.set_cell_value(1, "Y", 0, s, v)
            mt_sheet.table.set_cell_value(2, "Y", 0, s, v)
        for s, v in enumerate([20.0, 20.1, 19.9]):
            mt_sheet.table.set_cell_value(0, "Y", 1, s, v)
        for s, v in enumerate([11.0, 12.0, 11.5]):
            mt_sheet.table.set_cell_value(1, "Y", 1, s, v)
        for s, v in enumerate([10.9, 11.8, 11.4]):
            mt_sheet.table.set_cell_value(2, "Y", 1, s, v)
        # Add an outlier in Row 0 and exclude it
        mt_sheet.table.set_cell_value(0, "Y", 1, 2, 1000.0)
        mt_sheet.table.set_excluded(0, "Y", 1, 2, True)

        self.window.action_run_multiple_t_tests()
        res_mt = self.window.document.get_sheet(self.window.document.active_sheet_id or "")
        self.assertIsInstance(res_mt, ResultSheet)
        rows = res_mt.tabs[0]["rows"]
        p_vals = [float(r[3]) for r in rows]
        q_vals = [float(r[4]) for r in rows]
        # Verify q_vals are monotonic with respect to p_vals
        order = np.argsort(p_vals)
        sorted_q = [q_vals[i] for i in order]
        for i in range(len(sorted_q) - 1):
            self.assertLessEqual(sorted_q[i], sorted_q[i + 1] + 1e-12)


if __name__ == "__main__":
    unittest.main()


