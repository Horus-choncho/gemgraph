# GemGraph

**Clean-room, open-source scientific data analysis, nonlinear curve fitting, and publication graphing — a modern alternative to GraphPad Prism 10.**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Security Policy: No-Sudo-Strict-Local](https://img.shields.io/badge/Security-No--Sudo--Strict--Local-green.svg)](specs/prism10_cleanroom_spec.md)


## Run from Source !!! ##

git clone https://github.com/Horus-choncho/gemgraph.git
cd gemgraph
pip install -r requirements.txt
python -m src.app


---

## Overview

**GemGraph** is an open-source scientific desktop workbench engineered from scratch under clean-room separation principles. Designed as a direct alternative to GraphPad Prism 10 for biomedical scientists, pharmacologists, and quantitative researchers, GemGraph pairs an intuitive replicate-aware scientific spreadsheet with double-precision numerical solvers and publication-grade vector graphics.

GemGraph operates under the **No-Sudo-Strict-Local** security policy: it executes entirely in unprivileged user space, requires no root/sudo privileges, executes no untrusted foreign binaries, and stores project files in open formats.

---

## Key Scientific Capabilities

### 1. 8 Scientific Table Paradigms
Unlike generic spreadsheets, GemGraph structures data according to laboratory experimental designs:
- **XY Tables:** Continuous or series $X$ with replicate subcolumns ($A:Y_1, A:Y_2, \dots$), automatic mean/SD/SEM calculation, and non-destructive exclusion masking.
- **Column Tables:** One-way grouping (e.g., Control vs. Treated) for unpaired/paired comparisons and one-way ANOVA. Excluded values render with strikethrough and blue italic styling.
- **Grouped Tables (2-Way):** Factor 1 along rows, Factor 2 along columns, with subcolumn replicates for two-way ANOVA and interaction testing.
- **Contingency Tables:** Categorical counts for Chi-square tests and Fisher's Exact test with non-negative integer validation.
- **Survival Tables:** Follow-up duration in $X$, outcome code ($1 = \text{Event}, 0 = \text{Censored}$) across treatment subcolumns for Kaplan-Meier product-limit analysis.
- **Parts of Whole Tables:** Relative proportions, fractional distributions, and percentage breakdowns for pie/donut charts and goodness-of-fit.
- **Multiple Variables Tables:** Tidy-data format where each column is assigned a semantic metadata badge:
  - 📊 **Categorical** (factors/groups with distinct level tracking and reordering)
  - 🔔 **Continuous** (numeric floating-point measurements)
  - 🏷️ **Label** (sample identifiers and barcodes excluded from math)
- **Nested Tables:** Multi-level hierarchical designs (e.g., technical replicates nested within biological replicates within subjects).

### 2. Numerical & Curve Fitting Engines
- **Levenberg-Marquardt Nonlinear Regression:**
  - Robust double-precision optimization solving $(J^T J + \lambda \, \text{diag}(J^T J)) \Delta \theta = J^T (y - \hat{y})$.
  - Two strict convergence presets matching Prism:
    - **Standard Preset:** Stops when $\frac{SS_{\text{prev}} - SS_{\text{curr}}}{SS_{\text{curr}}} < 1.0 \times 10^{-4}$ ($0.0001$).
    - **Strict Preset:** Stops when $\frac{SS_{\text{prev}} - SS_{\text{curr}}}{SS_{\text{curr}}} < 1.0 \times 10^{-6}$ ($0.000001$).
  - Maximum iteration safeguard ($1,000$ iterations).
  - Built-in parameter dependency matrix calculation ($\text{Dependency}_i = 1 - \frac{1}{(C)_{ii} \cdot (C^{-1})_{ii}}$) to flag over-parameterization.
  - Goodness-of-fit reporting: $R^2$, $S_{y.x}$, and second-order Akaike Information Criterion ($AIC_c$).
  - Curated model library: Sigmoidal 4PL Dose-Response (variable Hill slope), One-site specific binding (hyperbola), Two-site binding, Exponential decay, and Linear regression.
- **ROUT Outlier Rejection:**
  - Robust regression with False Discovery Rate threshold $Q$ (default $1.0\%$).
  - Robust Standard Deviation of Residuals: $RSDR = 1.4826 \times \text{MAD}$.
  - Two-tailed Student's $t$ residual testing with Benjamini-Hochberg FDR correction.
  - Outlier coordinates flagged without destructive data loss.
  - Classical Grubbs ESD (Extreme Studentized Deviate) test for single-sample outlier screening.
- **ANOVA Engine & Quadrature Post-Hoc Tests:**
  - Ordinary One-Way, Two-Way (Row, Column, and Interaction factors), and Repeated Measures ANOVA with Geisser-Greenhouse sphericity $\hat{\epsilon}$ correction.
  - Exact post-hoc **Dunnett's test** against control using multivariate normal/t quadrature integration.
  - All-pairwise **Tukey-Kramer HSD** tests using exact Studentized Range distributions.

### 3. Publication Vector Graphing
- **High-Contrast Grayscale Journal Styling:** Instant toggle between full-color scientific palettes and high-contrast grayscale palettes optimized for peer-reviewed print submissions.
- **Dynamic Constant Hooking (`s` Button):** Bind computed parameters (e.g., $IC_{50}$, $B_{\max}$, Slope, $\text{Mean} \pm 2\text{SD}$) directly to graph guide lines, threshold ticks, or dynamic callout labels. When raw data updates, hooked labels update in lockstep.
- **Live Table Overlays:** Embed interactive, linked analysis summary tables directly onto the plot canvas.
- **Vector Graphics Export:** Direct vector export to publication-ready SVG and high-resolution PNG (up to 600 DPI).

### 4. Multi-Tab Results & Grid Semantics
- **Green Grid vs. Red Grid Semantics:**
  - 🟩 **Green Grid (`#a3d9a5`):** Plottable and chainable numerical outputs (fitted curve $X/Y$ coordinates, residual matrices, interpolations).
  - 🟥 **Red Grid (`#f5c6cb`):** Formatted inspection matrices, parameter estimates $\pm$ standard error, confidence intervals, and goodness-of-fit metrics.
- **Directed Acyclic Graph (DAG) Recalculation:** Editing a cell in a data table automatically invalidates and cascades recalculations across dependent analyses and graphs.
- **State Integrity:** Support for frozen sheets (snowflake indicator) to lock historical analyses, and orphaned sheet detection when parent tables are removed.

---

## Installation & Quickstart

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Linux (x86_64, aarch64), macOS (Intel, Apple Silicon), or Windows 10/11

### Running via Virtual Environment
```bash
# 1. Clone repository
git clone https://github.com/nicolauda/gemgraph.git
cd gemgraph

# 2. Create isolated virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install core dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# 4. Launch GemGraph
python -m src.app
```

---

## Packaging & Distribution

Pre-built packages are available on the [Releases](https://github.com/Horus-choncho/gemgraph/releases) page:
- **Portable Linux (`.tar.gz`):** Extract and run `./GemGraph` in unprivileged user-space (no root required).
- **Debian / Ubuntu (`.deb`):** Install via `apt install ./gemgraph_0.1.0-alpha_amd64.deb` or extract in user space without sudo via `dpkg -x gemgraph_0.1.0-alpha_amd64.deb ./gemgraph_local`.
- **Windows Portable (`.zip`):** Unzip and execute `GemGraph.exe`.
- **macOS Portable (`.tar.gz`):** Extract and open `GemGraph.app`.
- **Flatpak & RPM Packages:** Scaffolding recipes provided in `packaging/org.gemgraph.GemGraph.yml` and `packaging/gemgraph.spec`.

---

## Verification & Testing

GemGraph includes a comprehensive automated test suite verifying mathematical parity against validated reference datasets and checking security compliance:

```bash
# Run security boundaries test (No-Sudo-Strict-Local enforcement)
python3 tests/test_security.py

# Run all mathematical, UI, and transform unit tests
python3 -m unittest discover tests
```

---

## License

GemGraph is free and open-source software licensed under the **GNU General Public License v3.0 (GPLv3)**. See [LICENSE](LICENSE) for details.
