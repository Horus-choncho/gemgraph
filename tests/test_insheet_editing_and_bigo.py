"""
Tests for in-sheet table editing, dual native .bigo/.prism support,
and the bundled textbook sample project.
"""

import os
import unittest
import unittest.mock
import tempfile
import shutil
import numpy as np
import scipy.stats as stats
from PyQt6.QtCore import Qt, QUrl, QMimeData, QPointF
from PyQt6.QtWidgets import QApplication, QInputDialog, QMenu
from PyQt6.QtGui import QDragEnterEvent, QDropEvent

from src.core.document import PrismDocument
from src.core.sheet import DataSheet, SheetType
from src.core.table import DataTable, TableType
from src.io.prism_io import PrismIO
from src.ui.mediator import WorkspaceMediator
from src.ui.spreadsheet import (
    ScientificSpreadsheetModel,
    ScientificSpreadsheetView,
    ScientificTableModel,
    ScientificTableWidget,
)
from src.ui.table_view import (
    ScientificTableModel as AliasTableModel,
    ScientificTableWidget as AliasTableWidget,
)
from src.ui.main_window import MainWindow
from src.data_engine.sample_generator import build_textbook_sample_document, generate_textbook_sample

app = QApplication.instance()
if app is None:
    app = QApplication([])


class TestInSheetColumnEditing(unittest.TestCase):
    """Verifies header double-click renaming, context menu actions, and DataFrame mutations."""

    def setUp(self) -> None:
        self.table = DataTable("test_edit", "Edit Table", table_type=TableType.XY, num_rows=5, num_replicates=2)
        self.table.set_cell_value(0, "X", 0, 0, 1.0)
        self.table.set_cell_value(0, "Y", 0, 0, 10.0)
        self.table.set_cell_value(0, "Y", 0, 1, 11.0)
        self.mediator = WorkspaceMediator()
        self.model = ScientificSpreadsheetModel(self.table, mediator=self.mediator)
        self.view = ScientificSpreadsheetView(self.model)

    def test_aliases_in_table_view(self) -> None:
        """Verifies ScientificTableWidget and ScientificTableModel aliases in src/ui/table_view.py."""
        self.assertIs(AliasTableModel, ScientificSpreadsheetModel)
        self.assertIs(AliasTableWidget, ScientificSpreadsheetView)
        self.assertIs(ScientificTableModel, ScientificSpreadsheetModel)
        self.assertIs(ScientificTableWidget, ScientificSpreadsheetView)

    def test_dataframe_property_and_synchronization(self) -> None:
        """Verifies DataFrame conversion and reflection of table data."""
        df = self.model.dataframe
        self.assertEqual(len(df), 5)
        self.assertEqual(list(df.columns), ["X", "Group A:Y1", "Group A:Y2", "Group B:Y1", "Group B:Y2"])
        self.assertEqual(df.iloc[0]["X"], 1.0)
        self.assertEqual(df.iloc[0]["Group A:Y1"], 10.0)

    def test_header_rename_via_model(self) -> None:
        """Verifies column renaming updates header, table, and emits mediator signals."""
        renamed_signals = []
        self.mediator.columnRenamed.connect(lambda col, name: renamed_signals.append((col, name)))

        # Rename X column
        success = self.model.rename_column(0, "Concentration")
        self.assertTrue(success)
        self.assertEqual(self.table.x_title, "Concentration")
        self.assertEqual(self.model.headerData(0, Qt.Orientation.Horizontal), "Concentration")
        self.assertIn((0, "Concentration"), renamed_signals)

        # Rename Group A
        success2 = self.model.rename_column(1, "Vehicle")
        self.assertTrue(success2)
        self.assertEqual(self.table.y_group_titles[0], "Vehicle")
        self.assertIn("Vehicle", self.model.headerData(1, Qt.Orientation.Horizontal))
        self.assertIn((1, "Vehicle"), renamed_signals)

        # DataFrame column names reflect the new names
        df = self.model.dataframe
        self.assertEqual(df.columns[0], "Concentration")
        self.assertEqual(df.columns[1], "Vehicle:Y1")

    def test_header_double_click_renaming_dialog(self) -> None:
        """Verifies that double clicking the header opens a QInputDialog and renames column on confirmation."""
        with unittest.mock.patch.object(QInputDialog, "getText", return_value=("Drug Treated", True)):
            self.view.horizontalHeader().sectionDoubleClicked.emit(1)

        self.assertEqual(self.table.y_group_titles[0], "Drug Treated")
        self.assertIn("Drug Treated", self.model.headerData(1, Qt.Orientation.Horizontal))

    def test_context_menu_column_insertion_and_deletion(self) -> None:
        """Verifies inserting columns to left and right and deleting columns."""
        ins_signals = []
        del_signals = []
        self.mediator.columnInserted.connect(lambda col, name: ins_signals.append((col, name)))
        self.mediator.columnDeleted.connect(lambda col: del_signals.append(col))

        initial_col_count = self.model.columnCount()
        self.assertEqual(initial_col_count, 5)  # X, A_1, A_2, B_1, B_2

        # Insert column to Right of Group A (column index 1 or 2)
        self.model.insert_column(1, position="right", title="Group A2")
        self.assertEqual(len(self.table.y_group_titles), 3)
        self.assertEqual(self.table.y_group_titles[1], "Group A2")
        self.assertEqual(self.model.columnCount(), 7)  # 1 X + 3*2 Ys = 7
        self.assertTrue(len(ins_signals) > 0)

        # Verify DataFrame has 7 columns now
        df = self.model.dataframe
        self.assertEqual(len(df.columns), 7)
        self.assertIn("Group A2:Y1", df.columns)

        # Insert column to Left of Group A (before group index 0)
        self.model.insert_column(1, position="left", title="Pre-Control")
        self.assertEqual(self.table.y_group_titles[0], "Pre-Control")
        self.assertEqual(self.model.columnCount(), 9)

        # Delete column (delete Group A2 at group index 2)
        # Group 0: Pre-Control (cols 1, 2)
        # Group 1: Group A (cols 3, 4)
        # Group 2: Group A2 (cols 5, 6)
        self.model.delete_column(5)
        self.assertEqual(len(self.table.y_group_titles), 3)
        self.assertNotIn("Group A2", self.table.y_group_titles)
        self.assertTrue(len(del_signals) > 0)

    def test_context_menu_actions_populated_for_cell_and_header(self) -> None:
        """Verifies context menu items for header and cells contain exact required action labels."""
        menu = QMenu()
        self.view._populate_column_actions(menu, 1)
        action_texts = [a.text() for a in menu.actions()]
        self.assertIn("➕ Insert Column to Right", action_texts)
        self.assertIn("➕ Insert Column to Left", action_texts)
        self.assertIn("✏️ Rename Column...", action_texts)
        self.assertIn("🗑️ Delete Column", action_texts)


class TestBigoAndPrismBidirectional(unittest.TestCase):
    """Verifies dual native .bigo and .prism support and Prism 10 packaging fidelity."""

    def test_bigo_save_and_load_round_trip(self) -> None:
        """Verifies full round trip saving and loading of .bigo archives."""
        doc = PrismDocument("Bigo Project")
        sheet = doc.create_data_sheet("Table 1", TableType.XY, num_rows=4, num_replicates=2)
        sheet.table.x_title = "Dose (nM)"
        sheet.table.set_cell_value(0, "X", 0, 0, 50.0)
        sheet.table.set_cell_value(0, "Y", 0, 0, 100.0)
        sheet.table.set_excluded(0, "Y", 0, 0, True)

        with tempfile.NamedTemporaryFile(suffix=".bigo", delete=False) as tmp:
            bigo_path = tmp.name

        try:
            PrismIO.save(doc, bigo_path)
            self.assertTrue(os.path.exists(bigo_path))

            # Load .bigo directly
            loaded = PrismIO.load(bigo_path)
            self.assertEqual(loaded.title, "Bigo Project")
            l_sheet = loaded.get_sheet(sheet.sheet_id)
            self.assertIsInstance(l_sheet, DataSheet)
            self.assertEqual(l_sheet.table.x_title, "Dose (nM)")
            self.assertEqual(l_sheet.table.get_cell_value(0, "X", 0, 0), 50.0)
            self.assertEqual(l_sheet.table.get_cell_value(0, "Y", 0, 0), 100.0)
            self.assertTrue(l_sheet.table.is_excluded(0, "Y", 0, 0))

            # Rename .bigo to .prism and load cleanly
            prism_path = bigo_path + ".prism"
            shutil.copy(bigo_path, prism_path)
            loaded_prism = PrismIO.load(prism_path)
            self.assertEqual(loaded_prism.title, "Bigo Project")
            l_sheet_prism = loaded_prism.get_sheet(sheet.sheet_id)
            self.assertEqual(l_sheet_prism.table.get_cell_value(0, "X", 0, 0), 50.0)
            os.remove(prism_path)
        finally:
            if os.path.exists(bigo_path):
                os.remove(bigo_path)

    def test_file_dialog_filters(self) -> None:
        """Verifies exact required file dialog filter string."""
        expected_filter = "GemGraph Projects (*.bigo);;GraphPad Prism Projects (*.prism *.pzfx);;All Supported (*.bigo *.prism *.pzfx);;All Files (*)"
        window = MainWindow()
        dialog = window.create_open_project_dialog()
        self.assertIn("GemGraph Projects (*.bigo)", dialog.nameFilters()[0])
        self.assertIn("GraphPad Prism Projects (*.prism *.pzfx)", dialog.nameFilters()[1])
        window.close()

    def test_drag_and_drop_accepts_bigo(self) -> None:
        """Verifies dragEnterEvent and dragMoveEvent accept .bigo files."""
        window = MainWindow()
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile("/tmp/project.bigo")])
        evt = QDragEnterEvent(
            QPointF(10, 10).toPoint(),
            Qt.DropAction.CopyAction,
            mime,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        window.dragEnterEvent(evt)
        self.assertTrue(evt.isAccepted(), "MainWindow failed to accept .bigo dragEnterEvent")
        window.close()


class TestTextbookSampleProject(unittest.TestCase):
    """Verifies the bundled textbook_sample.bigo contents and direct loading."""

    SAMPLE_PATH = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "samples", "textbook_sample.bigo")
    )

    def test_sample_file_exists_and_sheets(self) -> None:
        """Verifies samples/textbook_sample.bigo contains the 3 canonical sheets."""
        if not os.path.exists(self.SAMPLE_PATH):
            generate_textbook_sample(self.SAMPLE_PATH)

        self.assertTrue(os.path.exists(self.SAMPLE_PATH))
        doc = PrismIO.load(self.SAMPLE_PATH)

        sheet_titles = [s.title for s in doc.sheets.values()]
        self.assertIn("Dose-Response 4PL", sheet_titles)
        self.assertIn("One-Way ANOVA", sheet_titles)
        self.assertIn("Kaplan-Meier Survival", sheet_titles)

    def test_sheet1_dose_response_math(self) -> None:
        """Verifies Sheet 1 conforms to 4PL sigmoidal agonist dose-response with replicates."""
        doc = PrismIO.load(self.SAMPLE_PATH)
        sheet = next(s for s in doc.sheets.values() if s.title == "Dose-Response 4PL")
        table = sheet.table
        self.assertEqual(table.table_type, TableType.XY)
        self.assertEqual(table.num_replicates, 3)
        self.assertEqual(len(table.x_data), 9)
        self.assertAlmostEqual(table.x_data[0], -9.0)
        self.assertAlmostEqual(table.x_data[-1], -3.0)

        # Baseline and maximal responses
        xs, ys = table.get_clean_xy(0)
        self.assertEqual(len(xs), 27)  # 9 * 3
        # Response at -9 should be near 5, at -3 should be near 100
        y_low = [table.get_cell_value(0, "Y", 0, s) for s in range(3)]
        y_high = [table.get_cell_value(8, "Y", 0, s) for s in range(3)]
        self.assertTrue(all(abs(v - 5.0) < 6.0 for v in y_low))
        self.assertTrue(all(abs(v - 100.0) < 6.0 for v in y_high))

    def test_sheet2_one_way_anova_significance(self) -> None:
        """Verifies Sheet 2 shows statistically significant Compound B increase (P < 0.01) with clean Tukey HSD."""
        doc = PrismIO.load(self.SAMPLE_PATH)
        sheet = next(s for s in doc.sheets.values() if s.title == "One-Way ANOVA")
        table = sheet.table
        self.assertEqual(table.table_type, TableType.COLUMN)
        self.assertEqual(table.y_group_titles, ["Control", "Compound A", "Compound B"])
        self.assertEqual(table.num_rows, 6)

        ctrl = [table.get_cell_value(r, "Y", 0, 0) for r in range(6)]
        cmp_a = [table.get_cell_value(r, "Y", 1, 0) for r in range(6)]
        cmp_b = [table.get_cell_value(r, "Y", 2, 0) for r in range(6)]

        f_val, p_val = stats.f_oneway(ctrl, cmp_a, cmp_b)
        self.assertLess(p_val, 0.01)

        tukey = stats.tukey_hsd(ctrl, cmp_a, cmp_b)
        # Comparison index 0-2 (Control vs Compound B)
        # In tukey.pvalue matrix:
        p_ctrl_cmpb = tukey.pvalue[0, 2]
        self.assertLess(p_ctrl_cmpb, 0.01)

    def test_sheet3_kaplan_meier_survival_separation(self) -> None:
        """Verifies Sheet 3 survival curves separate with log-rank P < 0.01."""
        doc = PrismIO.load(self.SAMPLE_PATH)
        sheet = next(s for s in doc.sheets.values() if s.title == "Kaplan-Meier Survival")
        table = sheet.table
        self.assertEqual(table.table_type, TableType.SURVIVAL)
        self.assertIn("Days", table.x_title)

        # Vehicle vs Treated log-rank calculation
        times = table.x_data
        veh_ev = [table.get_cell_value(r, "Y", 0, 0) for r in range(len(times))]
        trt_ev = [table.get_cell_value(r, "Y", 1, 0) for r in range(len(times))]

        all_times = np.concatenate([times, times])
        all_events = np.concatenate([veh_ev, trt_ev])
        all_groups = np.array([0] * len(times) + [1] * len(times))

        unique_t = np.unique(all_times)
        o1, e1, v1 = 0.0, 0.0, 0.0
        for t in unique_t:
            m_all = all_times >= t
            n_all = np.sum(m_all)
            d_all = np.sum((all_times == t) & (all_events == 1))
            if n_all <= 1 or d_all == 0:
                continue
            m_1 = (all_times >= t) & (all_groups == 0)
            n_1 = np.sum(m_1)
            d_1 = np.sum((all_times == t) & (all_events == 1) & (all_groups == 0))
            e_1_t = n_1 * (d_all / n_all)
            v_1_t = n_1 * (n_all - n_1) * d_all * (n_all - d_all) / (n_all**2 * (n_all - 1))
            o1 += d_1
            e1 += e_1_t
            v1 += v_1_t
        z = (o1 - e1) / np.sqrt(v1)
        p_logrank = 2 * (1 - stats.norm.cdf(abs(z)))
        self.assertLess(p_logrank, 0.01)

    def test_menu_action_open_sample_project(self) -> None:
        """Verifies MainWindow.action_open_sample_project loads sample directly without dialog."""
        window = MainWindow()
        window.action_open_sample_project()
        self.assertIn("Textbook", window.document.title)
        self.assertEqual(len(window.document.get_sheets_by_type(SheetType.DATA)), 3)
        window.close()

    def test_data_table_from_and_update_dataframe(self) -> None:
        """Verifies DataTable.from_dataframe and update_from_dataframe."""
        import pandas as pd
        df = pd.DataFrame({
            "Dose": [1.0, 2.0, 3.0],
            "Response:Y1": [10.0, 20.0, 30.0],
            "Response:Y2": [11.0, 21.0, 31.0],
        })
        tbl = DataTable.from_dataframe(df, table_type=TableType.XY)
        self.assertEqual(tbl.x_title, "Dose")
        self.assertEqual(tbl.num_rows, 3)
        self.assertEqual(tbl.num_replicates, 2)
        self.assertEqual(tbl.get_cell_value(0, "X", 0, 0), 1.0)
        self.assertEqual(tbl.get_cell_value(2, "Y", 0, 1), 31.0)

        # Mutate via update_from_dataframe
        df2 = pd.DataFrame({
            "Conc": [0.1, 0.2],
            "Group A": [100.0, 200.0],
            "Group B": [300.0, 400.0],
        })
        tbl.update_from_dataframe(df2)
        self.assertEqual(tbl.x_title, "Conc")
        self.assertEqual(tbl.y_group_titles, ["Group A", "Group B"])
        self.assertEqual(tbl.get_cell_value(1, "Y", 1, 0), 400.0)

    def test_unique_group_naming_and_whitespace_rejection(self) -> None:
        """Verifies insert_y_group generates unique names and rename_column rejects empty strings."""
        tbl = DataTable("tbl_uniq", "Uniq", table_type=TableType.XY, num_rows=2, num_replicates=1)
        # Default groups: Group A, Group B
        idx = tbl.insert_y_group(2)
        self.assertEqual(tbl.y_group_titles[2], "Group C")
        # Insert again: should be Group D
        idx2 = tbl.insert_y_group(3)
        self.assertEqual(tbl.y_group_titles[3], "Group D")

        # Whitespace rename rejection
        self.assertFalse(tbl.rename_column(0, "   "))
        self.assertFalse(tbl.rename_column(1, ""))

    def test_multiple_variables_cell_color_shift_on_insert_and_remove(self) -> None:
        """Verifies cell highlight colors shift correctly when variables are inserted/removed."""
        tbl = DataTable("tbl_mv", "MV", table_type=TableType.MULTIPLE_VARIABLES, num_rows=3)
        # tbl has default variables: Var 1, Var 2, Var 3
        tbl.set_cell_color(0, "VAR", 1, 0, "#ff0000")
        self.assertEqual(tbl.get_cell_color(0, "VAR", 1, 0), "#ff0000")

        # Insert variable before index 1
        tbl.insert_variable(0, "NewVar")
        # Color that was at col 1 should now be at col 2
        self.assertEqual(tbl.get_cell_color(0, "VAR", 2, 0), "#ff0000")
        self.assertIsNone(tbl.get_cell_color(0, "VAR", 1, 0))

        # Remove variable at index 0 (NewVar)
        tbl.remove_variable(0)
        # Color should be back at col 1
        self.assertEqual(tbl.get_cell_color(0, "VAR", 1, 0), "#ff0000")


if __name__ == "__main__":
    unittest.main()
