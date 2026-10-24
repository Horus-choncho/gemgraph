"""
Textbook Sample Project Generator for GemGraph.
Generates samples/textbook_sample.bigo with publication-grade datasets for
Dose-Response 4PL, One-Way ANOVA, Kaplan-Meier Survival, Enzyme Kinetics (Michaelis-Menten),
PCA Multi-Variable Matrix, Cox Survival Cohort, and Distribution Comparison (Violin/Box).
"""

from __future__ import annotations
import os
import numpy as np
from src.core.document import PrismDocument
from src.core.table import TableType, VariableMetadata, VariableType
from src.core.sheet import GraphType
from src.io.prism_io import PrismIO


def build_textbook_sample_document() -> PrismDocument:
    """Constructs a PrismDocument containing the canonical textbook sample sheets."""
    doc = PrismDocument("Textbook Scientific Sample Project")
    rng = np.random.default_rng(42)

    # Sheet 1: "Dose-Response 4PL" (XY Format)
    x_vals = np.array([-9.0, -8.0, -7.0, -6.5, -6.0, -5.5, -5.0, -4.0, -3.0], dtype=np.float64)
    e_min = 5.0
    e_max = 100.0
    log_ec50 = -6.0
    hill_slope = 1.0
    expected_y = e_min + (e_max - e_min) / (1.0 + 10.0 ** ((log_ec50 - x_vals) * hill_slope))

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

    # Sheet 4: "Enzyme Kinetics (Michaelis-Menten)" (XY Format)
    # X = substrate [S] in µM (10 points: 0.5 to 200)
    # Group 1: No Inhibitor (Vmax ≈ 120, Km ≈ 15) with ~4% Gaussian noise
    # Group 2: + Inhibitor (Vmax ≈ 120, Km ≈ 45 competitive) with ~4% Gaussian noise
    x_s = np.array([0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 40.0, 80.0, 120.0, 200.0], dtype=np.float64)
    vmax = 120.0
    km1 = 15.0
    km2 = 45.0
    v1_mean = (vmax * x_s) / (km1 + x_s)
    v2_mean = (vmax * x_s) / (km2 + x_s)

    v1_reps = [np.round(v1_mean + rng.normal(0, 0.04 * v1_mean), 2) for _ in range(3)]
    v2_reps = [np.round(v2_mean + rng.normal(0, 0.04 * v2_mean), 2) for _ in range(3)]

    sheet4 = doc.create_data_sheet(
        title="Enzyme Kinetics (Michaelis-Menten)",
        table_type=TableType.XY,
        num_rows=len(x_s),
        num_replicates=3,
        y_group_titles=["No Inhibitor", "+ Inhibitor"],
    )
    t4 = sheet4.table
    t4.x_title = "Substrate [S] (µM)"
    for r, x in enumerate(x_s):
        t4.set_cell_value(r, "X", 0, 0, float(x))
        for rep in range(3):
            t4.set_cell_value(r, "Y", 0, rep, float(v1_reps[rep][r]))
            t4.set_cell_value(r, "Y", 1, rep, float(v2_reps[rep][r]))

    doc.create_graph_sheet("Enzyme Kinetics Graph", source_sheet_id=sheet4.sheet_id)

    # Sheet 5: "PCA Multi-Variable Matrix" (Multiple Variables Format)
    # >= 30 rows, 5 continuous variables with 2 latent clusters + 1 Categorical (Treatment) + 1 Label (Subject ID)
    n_pca = 40
    sheet5 = doc.create_data_sheet(
        title="PCA Multi-Variable Matrix",
        table_type=TableType.MULTIPLE_VARIABLES,
        num_rows=n_pca,
    )
    t5 = sheet5.table
    t5.variables = [
        VariableMetadata("Subject_ID", VariableType.LABEL),
        VariableMetadata("Treatment", VariableType.CATEGORICAL, ["Vehicle", "Drug"]),
        VariableMetadata("Biomarker_A", VariableType.CONTINUOUS),
        VariableMetadata("Biomarker_B", VariableType.CONTINUOUS),
        VariableMetadata("Biomarker_C", VariableType.CONTINUOUS),
        VariableMetadata("Biomarker_D", VariableType.CONTINUOUS),
        VariableMetadata("Biomarker_E", VariableType.CONTINUOUS),
    ]
    t5.multi_var_data = [["" for _ in t5.variables] for _ in range(n_pca)]

    for r in range(n_pca):
        is_drug = r >= (n_pca // 2)
        subj_id = f"SUBJ_{r+1:03d}"
        trt = "Drug" if is_drug else "Vehicle"
        t5.set_variable_value(r, 0, subj_id)
        t5.set_variable_value(r, 1, trt)

        if not is_drug:
            # Cluster 1: Vehicle
            t5.set_variable_value(r, 2, float(np.round(15.0 + rng.normal(0, 2.5), 2)))
            t5.set_variable_value(r, 3, float(np.round(25.0 + rng.normal(0, 3.0), 2)))
            t5.set_variable_value(r, 4, float(np.round(10.0 + rng.normal(0, 2.0), 2)))
            t5.set_variable_value(r, 5, float(np.round(45.0 + rng.normal(0, 4.0), 2)))
            t5.set_variable_value(r, 6, float(np.round(30.0 + rng.normal(0, 3.5), 2)))
        else:
            # Cluster 2: Drug
            t5.set_variable_value(r, 2, float(np.round(35.0 + rng.normal(0, 2.5), 2)))
            t5.set_variable_value(r, 3, float(np.round(48.0 + rng.normal(0, 3.0), 2)))
            t5.set_variable_value(r, 4, float(np.round(28.0 + rng.normal(0, 2.0), 2)))
            t5.set_variable_value(r, 5, float(np.round(20.0 + rng.normal(0, 4.0), 2)))
            t5.set_variable_value(r, 6, float(np.round(65.0 + rng.normal(0, 3.5), 2)))

    # Sheet 6: "Cox Survival Cohort" (Survival Format)
    # >= 60 subjects with true hazard ratio ≈ 0.5 for Treatment
    n_cox = 70
    trts = np.array([0.0] * (n_cox // 2) + [1.0] * (n_cox // 2), dtype=np.float64)
    ages = np.round(rng.uniform(45.0, 75.0, size=n_cox), 1)
    bios = np.round(rng.normal(0.0, 1.0, size=n_cox), 2)
    beta_trt = np.log(0.5)
    beta_age = 0.02
    beta_bio = 0.15
    lin_pred = beta_trt * trts + beta_age * (ages - 60.0) + beta_bio * bios
    hazard = 0.025 * np.exp(lin_pred)
    u_vals = rng.uniform(0.001, 0.999, size=n_cox)
    true_times = -np.log(u_vals) / hazard
    cens_times = rng.uniform(30.0, 120.0, size=n_cox)
    obs_times = np.round(np.clip(np.minimum(true_times, cens_times), 1.0, 200.0), 1)
    obs_events = np.where(true_times <= cens_times, 1.0, 0.0)

    sheet6 = doc.create_data_sheet(
        title="Cox Survival Cohort",
        table_type=TableType.SURVIVAL,
        num_rows=n_cox,
        num_replicates=1,
        y_group_titles=["Event", "Age", "Treatment", "Biomarker"],
    )
    t6 = sheet6.table
    t6.x_title = "Time (days)"
    for r in range(n_cox):
        t6.set_cell_value(r, "X", 0, 0, float(obs_times[r]))
        t6.set_cell_value(r, "Y", 0, 0, float(obs_events[r]))
        t6.set_cell_value(r, "Y", 1, 0, float(ages[r]))
        t6.set_cell_value(r, "Y", 2, 0, float(trts[r]))
        t6.set_cell_value(r, "Y", 3, 0, float(bios[r]))

    doc.create_graph_sheet("Cox Survival Cohort Graph", source_sheet_id=sheet6.sheet_id)

    # Sheet 7: "Distribution Comparison (Violin/Box)" (Column Format)
    # 3 groups, N >= 40 each, one bimodal, one skewed, one normal
    n_dist = 50
    norm_vals = np.round(rng.normal(50.0, 10.0, size=n_dist), 2)
    bimodal_vals = np.round(
        np.concatenate([rng.normal(30.0, 4.0, size=n_dist // 2), rng.normal(70.0, 4.0, size=n_dist // 2)]),
        2,
    )
    skewed_vals = np.round(rng.gamma(shape=2.0, scale=8.0, size=n_dist) + 15.0, 2)

    sheet7 = doc.create_data_sheet(
        title="Distribution Comparison (Violin/Box)",
        table_type=TableType.COLUMN,
        num_rows=n_dist,
        num_replicates=1,
        y_group_titles=["Normal", "Bimodal", "Skewed"],
    )
    t7 = sheet7.table
    for r in range(n_dist):
        t7.set_cell_value(r, "Y", 0, 0, float(norm_vals[r]))
        t7.set_cell_value(r, "Y", 1, 0, float(bimodal_vals[r]))
        t7.set_cell_value(r, "Y", 2, 0, float(skewed_vals[r]))

    g_sheet7 = doc.create_graph_sheet("Distribution Comparison Graph", source_sheet_id=sheet7.sheet_id)
    g_sheet7.graph_type = GraphType.VIOLIN
    g_sheet7.plot_config["plot_type"] = GraphType.VIOLIN.value
    g_sheet7.plot_config["graph_type"] = GraphType.VIOLIN.value

    # Default active sheet: first data sheet
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
