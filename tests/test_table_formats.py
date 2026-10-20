"""
Unit tests for the 8 Scientific Data Table Formats in GemGraph.
Verifies DataTable schemas, replicate/exclusion indexing, contingency checks,
Multiple Variables typed metadata, level reordering, and cell color coding.
"""

import unittest
import numpy as np
from src.core.table import DataTable, TableType, VariableType, VariableMetadata, XFormat, YFormat


class TestScientificTableFormats(unittest.TestCase):
    """Verifies all 8 scientific table paradigms and their presentation contracts."""

    def test_xy_table_format(self) -> None:
        tbl = DataTable("xy1", "XY Test", table_type=TableType.XY, num_rows=10, num_replicates=2)
        self.assertEqual(tbl.table_type, TableType.XY)
        self.assertIsNotNone(tbl.x_data)
        self.assertEqual(tbl.x_title, "X")
        self.assertEqual(tbl.num_replicates, 2)
        self.assertEqual(tbl.y_data.shape, (2, 10, 2))

        # Set values
        tbl.set_cell_value(0, "X", 0, 0, 1.0)
        tbl.set_cell_value(0, "Y", 0, 0, 10.0)
        tbl.set_cell_value(0, "Y", 0, 1, 12.0)
        self.assertEqual(tbl.get_cell_value(0, "X", 0, 0), 1.0)
        self.assertEqual(tbl.get_cell_value(0, "Y", 0, 1), 12.0)

        # Exclusion test
        tbl.set_excluded(0, "Y", 0, 1, True)
        self.assertTrue(tbl.is_excluded(0, "Y", 0, 1))
        xs, ys = tbl.get_clean_xy(0)
        self.assertEqual(len(xs), 1)
        self.assertEqual(ys[0], 10.0)

    def test_column_table_format(self) -> None:
        tbl = DataTable("col1", "Column Test", table_type=TableType.COLUMN, num_rows=8, num_replicates=1)
        self.assertEqual(tbl.table_type, TableType.COLUMN)
        self.assertIsNone(tbl.x_data)
        self.assertEqual(len(tbl.y_group_titles), 2)

        # Cell exclusion
        tbl.set_cell_value(2, "Y", 0, 0, 42.0)
        tbl.set_excluded(2, "Y", 0, 0, True)
        self.assertTrue(tbl.is_excluded(2, "Y", 0, 0))

    def test_grouped_table_format(self) -> None:
        tbl = DataTable("grp1", "Grouped Test", table_type=TableType.GROUPED, num_rows=4, num_replicates=3)
        self.assertEqual(tbl.table_type, TableType.GROUPED)
        self.assertIsNone(tbl.x_data)
        self.assertEqual(tbl.num_replicates, 3)
        self.assertEqual(tbl.y_data.shape, (2, 4, 3))

    def test_contingency_table_validation(self) -> None:
        tbl = DataTable("cnt1", "Contingency Test", table_type=TableType.CONTINGENCY, num_rows=2)
        self.assertEqual(tbl.table_type, TableType.CONTINGENCY)
        self.assertIsNone(tbl.x_data)

        # Valid non-negative integers
        tbl.set_cell_value(0, "Y", 0, 0, 15)
        tbl.set_cell_value(0, "Y", 1, 0, 20)
        self.assertEqual(tbl.get_cell_value(0, "Y", 0, 0), 15.0)

        # Negative value must raise ValueError
        with self.assertRaises(ValueError):
            tbl.set_cell_value(1, "Y", 0, 0, -5)

        # Non-integer float must raise ValueError
        with self.assertRaises(ValueError):
            tbl.set_cell_value(1, "Y", 0, 0, 3.14)

    def test_survival_table_format(self) -> None:
        tbl = DataTable("surv1", "Survival Test", table_type=TableType.SURVIVAL, num_rows=5, num_replicates=1)
        self.assertEqual(tbl.table_type, TableType.SURVIVAL)
        self.assertIsNotNone(tbl.x_data)
        self.assertIn("Elapsed Time", tbl.x_title)

        # X = elapsed time, Y = 1 (event) or 0 (censored)
        tbl.set_cell_value(0, "X", 0, 0, 12.0)
        tbl.set_cell_value(0, "Y", 0, 0, 1.0)
        tbl.set_cell_value(1, "X", 0, 0, 24.0)
        tbl.set_cell_value(1, "Y", 0, 0, 0.0)

        self.assertEqual(tbl.get_cell_value(0, "X", 0, 0), 12.0)
        self.assertEqual(tbl.get_cell_value(0, "Y", 0, 0), 1.0)
        self.assertEqual(tbl.get_cell_value(1, "Y", 0, 0), 0.0)

    def test_parts_of_whole_table_format(self) -> None:
        tbl = DataTable("pow1", "Parts of Whole Test", table_type=TableType.PARTS_OF_WHOLE, num_rows=4)
        self.assertEqual(tbl.table_type, TableType.PARTS_OF_WHOLE)
        self.assertIsNone(tbl.x_data)
        self.assertEqual(tbl.num_replicates, 1)

        tbl.row_titles = ["Slice A", "Slice B", "Slice C", "Slice D"]
        tbl.set_cell_value(0, "Y", 0, 0, 25.0)
        tbl.set_cell_value(1, "Y", 0, 0, 35.0)
        tbl.set_cell_value(2, "Y", 0, 0, 20.0)
        tbl.set_cell_value(3, "Y", 0, 0, 20.0)

        self.assertEqual(tbl.get_cell_value(1, "Y", 0, 0), 35.0)

    def test_multiple_variables_metadata_and_reordering(self) -> None:
        tbl = DataTable("mv1", "Multivar Test", table_type=TableType.MULTIPLE_VARIABLES, num_rows=5)
        self.assertEqual(tbl.table_type, TableType.MULTIPLE_VARIABLES)
        self.assertEqual(len(tbl.variables), 3)

        # Check types
        self.assertEqual(tbl.variables[0].var_type, VariableType.CATEGORICAL)
        self.assertEqual(tbl.variables[1].var_type, VariableType.CONTINUOUS)
        self.assertEqual(tbl.variables[2].var_type, VariableType.LABEL)

        # Set values
        tbl.set_variable_value(0, 0, "Control")
        tbl.set_variable_value(0, 1, 15.4)
        tbl.set_variable_value(0, 2, "Subject_001")

        tbl.set_variable_value(1, 0, "Treated")
        tbl.set_variable_value(1, 1, 28.9)
        tbl.set_variable_value(1, 2, "Subject_002")

        tbl.set_variable_value(2, 0, "Vehicle")  # Automatically discovers level

        self.assertEqual(tbl.get_variable_value(0, 1), 15.4)
        self.assertEqual(tbl.get_variable_value(1, 0), "Treated")

        # Level tracking
        levels = tbl.get_levels(0)
        self.assertIn("Vehicle", levels)
        self.assertIn("Control", levels)
        self.assertIn("Treated", levels)

        # Level counts
        counts = tbl.get_variable_level_counts(0)
        self.assertEqual(counts["Control"], 1)
        self.assertEqual(counts["Treated"], 1)
        self.assertEqual(counts["Vehicle"], 1)

        # Reorder levels
        new_order = ["Vehicle", "Control", "Treated"]
        tbl.reorder_levels(0, new_order)
        self.assertEqual(tbl.get_levels(0), new_order)

    def test_nested_table_format(self) -> None:
        tbl = DataTable("nst1", "Nested Test", table_type=TableType.NESTED, num_rows=6, num_replicates=4)
        self.assertEqual(tbl.table_type, TableType.NESTED)

        # Set nested physical unit subcolumn titles
        tbl.set_nested_subcolumn_title(0, 0, "Room 1 (Sub 1)")
        tbl.set_nested_subcolumn_title(0, 1, "Room 1 (Sub 2)")
        tbl.set_nested_subcolumn_title(0, 2, "Room 2 (Sub 1)")
        tbl.set_nested_subcolumn_title(0, 3, "Room 2 (Sub 2)")

        self.assertEqual(tbl.get_nested_subcolumn_title(0, 0), "Room 1 (Sub 1)")
        self.assertEqual(tbl.get_nested_subcolumn_title(0, 2), "Room 2 (Sub 1)")
        self.assertEqual(tbl.get_nested_subcolumn_title(1, 0), "Sub-unit 1")  # Default fallback

    def test_cell_color_coding_and_serialization(self) -> None:
        tbl = DataTable("clr1", "Color Test", table_type=TableType.XY, num_rows=5, num_replicates=2)
        tbl.set_cell_color(1, "Y", 0, 1, "#fff3cd")  # Yellow
        tbl.set_cell_color(2, "X", 0, 0, "#f8d7da")  # Red

        self.assertEqual(tbl.get_cell_color(1, "Y", 0, 1), "#fff3cd")
        self.assertEqual(tbl.get_cell_color(2, "X", 0, 0), "#f8d7da")
        self.assertIsNone(tbl.get_cell_color(0, "Y", 0, 0))

        # Test dictionary round-trip
        data = tbl.to_dict()
        restored = DataTable.from_dict(data)
        self.assertEqual(restored.get_cell_color(1, "Y", 0, 1), "#fff3cd")
        self.assertEqual(restored.get_cell_color(2, "X", 0, 0), "#f8d7da")

    def test_multiple_variables_row_operations(self) -> None:
        """Verifies adding and removing rows preserves multi_var_data and cell_colors without IndexError."""
        tbl = DataTable("mv_ops", "Row Ops Test", table_type=TableType.MULTIPLE_VARIABLES, num_rows=2)
        tbl.set_variable_value(0, 0, "Control")
        tbl.set_variable_value(0, 1, 10.0)
        tbl.set_variable_value(1, 0, "Treated")
        tbl.set_variable_value(1, 1, 20.0)
        tbl.set_cell_color(1, "VAR", 1, 0, "#cce5ff")

        # Add 3 rows
        tbl.add_row(3)
        self.assertEqual(tbl.num_rows, 5)
        self.assertEqual(len(tbl.multi_var_data), 5)

        # Set values on newly added rows
        tbl.set_variable_value(4, 0, "Recovery")
        tbl.set_variable_value(4, 1, 30.0)
        self.assertEqual(tbl.get_variable_value(4, 0), "Recovery")
        self.assertEqual(tbl.get_variable_value(4, 1), 30.0)

        # Remove row 0
        self.assertTrue(tbl.remove_row(0))
        self.assertEqual(tbl.num_rows, 4)
        self.assertEqual(len(tbl.multi_var_data), 4)
        # Former row 1 ("Treated") is now row 0, and its color should have shifted from row 1 to row 0
        self.assertEqual(tbl.get_variable_value(0, 0), "Treated")
        self.assertEqual(tbl.get_cell_color(0, "VAR", 1, 0), "#cce5ff")

    def test_spreadsheet_view_context_menu_and_badges(self) -> None:
        """Verifies header icon badges, exclusion actions, and context menu execution."""
        from src.ui.spreadsheet import ScientificSpreadsheetModel, ScientificSpreadsheetView
        from PyQt6.QtCore import Qt

        tbl = DataTable("mv_ui", "UI Test", table_type=TableType.MULTIPLE_VARIABLES, num_rows=3)
        model = ScientificSpreadsheetModel(tbl)

        # Verify badges in horizontal headers
        hdr_0 = model.headerData(0, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole)
        hdr_1 = model.headerData(1, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole)
        hdr_2 = model.headerData(2, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole)

        self.assertIn("📊", hdr_0)
        self.assertIn("Categorical", hdr_0)
        self.assertIn("🔔", hdr_1)
        self.assertIn("Continuous", hdr_1)
        self.assertIn("🏷️", hdr_2)
        self.assertIn("Label", hdr_2)

        # Test view and context menu action methods
        view = ScientificSpreadsheetView(model)
        # Verify _set_selected_color on MULTIPLE_VARIABLES
        view.selectRow(0)
        view._set_selected_color("#d4edda")
        self.assertEqual(tbl.get_cell_color(0, "VAR", 0, 0), "#d4edda")


if __name__ == "__main__":
    unittest.main()
