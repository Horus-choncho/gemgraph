"""
Textbook Sample Project Generator for GemGraph.
Generates samples/textbook_sample.bigo with publication-grade datasets for
Dose-Response 4PL, One-Way ANOVA, and Kaplan-Meier Survival.
"""

from __future__ import annotations
import os
import numpy as np
from src.core.document import PrismDocument
from src.core.table import TableType
from src.io.prism_io import PrismIO


def build_textbook_sample_document() -> PrismDocument:
    """Constructs a PrismDocument containing the three canonical textbook sample sheets."""
    doc = PrismDocument("Textbook Scientific Sample Project")

    # Sheet 1: "Dose-Response 4PL" (XY Format)
    # X: Log[Agonist] (µM) from -9.0 to -3.0
    # Y1, Y2, Y3: Triplicate values following 4PL sigmoid curve (Emin ≈ 5, Emax ≈ 100, logEC50 = -6.0, HillSlope = 1.0)
    # with Gaussian noise (sigma = 1.5)
    x_vals = np.array([-9.0, -8.0, -7.0, -6.5, -6.0, -5.5, -5.0, -4.0, -3.0], dtype=np.float64)
    e_min = 5.0
    e_max = 100.0
    log_ec50 = -6.0
    hill_slope = 1.0
    expected_y = e_min + (e_max - e_min) / (1.0 + 10.0 ** ((log_ec50 - x_vals) * hill_slope))

    rng = np.random.RandomState(42)
    y1 = np.round(expected_y + rng.normal(0, 1.5, size=len(x_vals)), 2)
    y2 = np.round(expected_y + rng.normal(0, 1.5, size=len(x_vals)), 2)
    y3 = np.round(expected_y + rng.normal(0, 1.5, size=len(x_vals)), 2)

    sheet1 = doc.create_data_sheet(
        title="Dose-Response 4PL",
        table_type=TableType.XY,
        num_rows=len(x_vals),
        num_replicates=3,
        y_group_titles=["Response"],
    )
    t1 = sheet1.table
    t1.x_title = "Log[Agonist] (µM)"
    for r, x in enumerate(x_vals):
        t1.set_cell_value(r, "X", 0, 0, float(x))
        t1.set_cell_value(r, "Y", 0, 0, float(y1[r]))
        t1.set_cell_value(r, "Y", 0, 1, float(y2[r]))
        t1.set_cell_value(r, "Y", 0, 2, float(y3[r]))

    doc.create_graph_sheet("Dose-Response 4PL Graph", source_sheet_id=sheet1.sheet_id)

    # Sheet 2: "One-Way ANOVA" (Column Format)
    # Columns: Control, Compound A, Compound B (N=6 each)
    # Compound B shows statistically significant increase (P < 0.01) over Control for Tukey HSD
    ctrl = [19.8, 20.4, 19.2, 21.1, 20.0, 19.5]
    cmp_a = [21.2, 22.0, 20.8, 21.5, 22.4, 21.1]
    cmp_b = [34.5, 36.2, 35.0, 37.1, 35.8, 36.4]

    sheet2 = doc.create_data_sheet(
        title="One-Way ANOVA",
        table_type=TableType.COLUMN,
        num_rows=6,
        num_replicates=1,
        y_group_titles=["Control", "Compound A", "Compound B"],
    )
    t2 = sheet2.table
    for r in range(6):
        t2.set_cell_value(r, "Y", 0, 0, ctrl[r])
        t2.set_cell_value(r, "Y", 1, 0, cmp_a[r])
        t2.set_cell_value(r, "Y", 2, 0, cmp_b[r])

    doc.create_graph_sheet("One-Way ANOVA Graph", source_sheet_id=sheet2.sheet_id)

    # Sheet 3: "Kaplan-Meier Survival" (Survival Format)
    # Columns: Days, Vehicle Status, Treated Status (0=censored, 1=event)
    # Showing distinct separation between survival curves (P < 0.01)
    days = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0]
    veh_status = [1.0] * len(days)
    trt_status = [0.0] * (len(days) - 1) + [1.0]

    sheet3 = doc.create_data_sheet(
        title="Kaplan-Meier Survival",
        table_type=TableType.SURVIVAL,
        num_rows=len(days),
        num_replicates=1,
        y_group_titles=["Vehicle Status", "Treated Status"],
    )
    t3 = sheet3.table
    t3.x_title = "Days"
    for r, d in enumerate(days):
        t3.set_cell_value(r, "X", 0, 0, float(d))
        t3.set_cell_value(r, "Y", 0, 0, float(veh_status[r]))
        t3.set_cell_value(r, "Y", 1, 0, float(trt_status[r]))

    doc.create_graph_sheet("Kaplan-Meier Survival Graph", source_sheet_id=sheet3.sheet_id)

    doc.active_sheet_id = sheet1.sheet_id
    return doc


def generate_textbook_sample(output_path: str) -> None:
    """Generates and writes textbook_sample.bigo PK-ZIP archive."""
    doc = build_textbook_sample_document()
    PrismIO.save(doc, output_path)


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "samples/textbook_sample.bigo"
    generate_textbook_sample(out)
    print(f"Generated {out}")
