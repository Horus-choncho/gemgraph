"""Regression: multi-X xy_mapping must stay aligned after column insert/delete,
end-to-end through ScientificSpreadsheetModel -> WorkspaceMediator.sync_data_to_visualization."""

import unittest
import numpy as np
from PyQt6.QtWidgets import QApplication

from src.core.table import DataTable, TableType
from src.ui.spreadsheet import ScientificSpreadsheetModel
from src.ui.mediator import WorkspaceMediator


def _build():
    t = DataTable(table_id="t", title="T", table_type=TableType.XY, num_rows=3, num_replicates=1)
    t.y_group_titles = ["A", "B"]
    t.column_order = [("X", 0), ("Y", 0), ("X", 1), ("Y", 1)]
    t.x_columns[0]["title"] = "X1"
    t.x_columns[0]["data"] = np.array([1.0, 2.0, 3.0])
    t.x_columns.append({"title": "X2", "data": np.array([10.0, 20.0, 30.0])})
    t.y_data = np.zeros((2, 3, 1))
    t.y_data[0, :, 0] = [5, 6, 7]
    t.y_data[1, :, 0] = [50, 60, 70]
    return t


class TestXYMappingAlignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _series(self, model):
        out = WorkspaceMediator().sync_data_to_visualization(model=model)
        return {name: (x.tolist(), y.tolist()) for x, y, name in out}

    def test_explicit_assignment_end_to_end(self):
        m = ScientificSpreadsheetModel(_build())
        # Assign Y "A" (visual col 1) to X2 (visual col 2)
        m.xy_mapping[1] = 2
        s = self._series(m)
        self.assertEqual(s["A"][0], [10.0, 20.0, 30.0])
        self.assertEqual(s["B"][0], [10.0, 20.0, 30.0])

    def test_mapping_survives_insert_left_of_everything(self):
        m = ScientificSpreadsheetModel(_build())
        m.xy_mapping[1] = 2
        m.xy_mapping["1"] = 2
        m.insert_column(0, position="left", title="New")
        # old col1 -> col2, old col2 -> col3
        self.assertEqual(m.xy_mapping.get(2), 3)
        self.assertNotIn(1, m.xy_mapping)
        s = self._series(m)
        self.assertEqual(s["A"][0], [10.0, 20.0, 30.0])

    def test_mapping_dropped_when_target_x_deleted(self):
        m = ScientificSpreadsheetModel(_build())
        m.xy_mapping[1] = 2
        self.assertTrue(m.delete_column(2))  # delete X2
        self.assertNotIn(1, m.xy_mapping)
        s = self._series(m)
        self.assertEqual(s["A"][0], [1.0, 2.0, 3.0])

    def test_mapping_shifts_when_earlier_column_deleted(self):
        t = _build()
        m = ScientificSpreadsheetModel(t)
        m.xy_mapping[3] = 0  # B -> X1
        self.assertTrue(m.delete_column(1))  # delete Y "A"
        self.assertEqual(m.xy_mapping.get(2), 0)
        self.assertNotIn(3, m.xy_mapping)
        s = self._series(m)
        self.assertEqual(s["B"][0], [1.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
