# GemGraph v0.1.0-alpha Release Notes

Welcome to the first alpha release of **GemGraph**, the clean-room, open-source scientific data analysis and curve fitting application designed as a transparent alternative to GraphPad Prism.

This release brings core statistical algorithms, 8 laboratory table formats, and publication-ready vector graphing directly to your desktop.

---

## 🔬 What Works Right Now

- **8 Scientific Table Paradigms:**
  - **XY Tables** with replicate subcolumns, automatic SD/SEM error bars, and non-destructive exclusion masking.
  - **Column Tables** for comparing groups (Control vs. Treated) with box-whisker or bar-scatter displays.
  - **Grouped (2-Way) Tables** for two-factor experimental layouts with replicate subcolumns.
  - **Contingency Tables** for Chi-Square and Fisher's exact categorical testing.
  - **Survival Tables** for Kaplan-Meier product-limit survival analysis.
  - **Parts of Whole Tables** for fractional distributions.
  - **Multiple Variables Tables** with interactive badges for Categorical (📊), Continuous (🔔), and Label (🏷️) columns.
  - **Nested Tables** for hierarchical laboratory samples.

- **Nonlinear Regression & Curve Fitting:**
  - Levenberg-Marquardt solver with **Standard (10⁻⁴)** and **Strict (10⁻⁶)** convergence presets.
  - Standard models: Sigmoidal 4PL Dose-Response (variable slope), One-site binding (hyperbola), Exponential decay, and Straight line.
  - Goodness-of-fit metrics ($R^2$, $S_{y.x}$, $AIC_c$) and parameter dependency checks.

- **Hypothesis Testing (ANOVA):**
  - Ordinary One-Way, Two-Way (Row, Column, Interaction), and Repeated Measures ANOVA.
  - Exact post-hoc **Dunnett's test** (comparisons against control) and **Tukey-Kramer HSD** (all pairwise comparisons).

- **Outlier Detection:**
  - Automated **ROUT** (Robust Regression and Outlier Removal) with False Discovery Rate $Q = 1\%$.
  - Single-sample **Grubbs ESD** test.

- **Publication Vector Graphing:**
  - Interactive plot canvas with high-contrast grayscale journal styling.
  - Dynamic **Constant Hooking (`s` button)** to link computed parameters directly to graph lines.
  - Live table overlays embedded directly on plot sheets.
  - Direct vector export to **SVG** and high-resolution **PNG**.

- **Multi-Tab Results & Live Recalculation:**
  - Clean distinction between Green Grids (analyzable/plottable data) and Red Grids (summary statistics).
  - DAG-based automatic recalculation when raw data changes.

---

## 🚀 Easy Installation

### 1. Standalone Portable Packages (No Admin Rights Needed)
- **Linux (`.tar.gz`):** Download `GemGraph-Linux-x86_64-Portable.tar.gz`, unpack, and run `./GemGraph`.
- **Windows (`.zip`):** Download `GemGraph-Windows-Portable.zip`, unpack, and run `GemGraph.exe`.
- **macOS (`.tar.gz`):** Download `GemGraph-macOS-Portable.tar.gz`, unpack, and open `GemGraph.app`.

### 2. Debian / Ubuntu Package
```bash
apt install ./gemgraph_0.1.0-alpha_amd64.deb
```
*(On locked institutional workstations without root, extract directly into user space via `dpkg -x gemgraph_0.1.0-alpha_amd64.deb ./gemgraph_app`)*.

### 3. Run from Source (Python 3.10+)
```bash
git clone https://github.com/nicolauda/gemgraph.git
cd gemgraph
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.app
```

---

## 💬 Community Feedback & Call-to-Action

We invite experimental researchers, pharmacologists, and statisticians to put GemGraph to the test:
1. **Test with your real laboratory data:** Load your experimental tables and compare curve fits, IC₅₀ values, and ANOVA results with Prism.
2. **Report edge cases & discrepancies:** If any calculation differs from expected values, open an issue on GitHub using our issue template and include your anonymized test table.
3. **Request models & features:** Tell us which regression equations, post-hoc corrections, or plot styles you need next!
