# Prism 10 Clean-Room Architecture Specification
**Author:** Lead Architect & Master Orchestrator, GemGraph Project  
**Date:** October 2026  
**Classification:** Clean-Room Reverse Engineering Specification  
**Legal Separation Notice:** This specification documents functional interfaces, data schemas, mathematical formulations, and presentation contracts derived by clean-room reverse engineering of GraphPad Prism 10 binaries (`BridgeCLR.dll`, `Core.dll`, and `prism.exe`). No proprietary binary code, disassembled C++ logic, or copyright-infringing assets are incorporated. All implementations are written from scratch in pure user-space Python.

---

## 1. REA Evidence Registry

| Evidence ID | Source Artifact | Subsystem | Clean-Room Derivation & Description |
| :--- | :--- | :--- | :--- |
| `REA-CLR-BRIDGE-091` | `BridgeCLR.dll` | View/ViewModel Hierarchy | Extracted WPF/CLR ViewModels, command bindings (`RelayCommand`), sheet selection routing, and graph-to-table synchronization contracts. |
| `REA-CLR-VM-102` | `BridgeCLR.dll` | Presentation Contracts | Three-pane desktop layout architecture: Navigation Tree (`PrismNavigator`), Dynamic Center Stack (`SheetGridHost`), and Right Plot Canvas (`VectorGraphHost`). |
| `REA-CORE-STORAGE-204`| `Core.dll` | Container File Format | `.prism` container specification: PK-ZIP archive containing `document.json`, `data/sheets/`, `data/tables/`, and tabular `data.csv`. |
| `REA-CORE-SCHEMA-208` | `Core.dll` / `PrismXMLSchema` | Table Data Models | Sheet schemas supporting 5 foundational table types: `XY`, `Column` (`OneWay`), `Grouped` (`TwoWay`), `Contingency`, and `Survival`, with subcolumn replicate semantics and exclusion masking (`Excluded="1"`). |
| `REA-PRISM-NUM-301`  | `prism.exe` | Nonlinear Solver Presets | Levenberg-Marquardt double-precision convergence tolerance presets: Standard ($\Delta SS/SS < 10^{-4}$), Strict ($\Delta SS/SS < 10^{-6}$), max iterations = 1000, and parameter dependency matrix calculations. |
| `REA-PRISM-ALGO-314` | `prism.exe` | Outlier Detection | ROUT algorithm: $RSDR = 1.4826 \times \text{MAD}$, FDR $Q \in (0, 100]\%$ (default $1.0\%$), sample size constraint $N \ge 3$, and Grubbs ESD critical values. |
| `REA-PRISM-STAT-325` | `prism.exe` | Group Comparisons (ANOVA) | Ordinary One-Way, Two-Way (Row/Column/Interaction), and Repeated Measures ANOVA with post-hoc Dunnett (against control) and Tukey-Kramer HSD tests. |

---

## 2. Presentation & View Model Architecture (`BridgeCLR.dll`)

Derived from mixed-mode CLI contracts (`REA-CLR-BRIDGE-091` & `REA-CLR-VM-102`):

```
+-----------------------------------------------------------------------------------------+
|                                GemGraph Main Window                                     |
+-----------------------------------------------------------------------------------------+
| Menu Bar: File | Edit | View | Insert | Change | Analysis | Graph | Help                |
+-------------------+-------------------------------------+-------------------------------+
| Navigation Tree   | Center Document Stack               | Interactive Plot Canvas       |
| (PrismNavigator)  | (SheetGridHost)                     | (VectorGraphHost)             |
|                   |                                     |                               |
| 📁 Data Tables    | Replicate Scientific Grid           | Vector Graphics (pyqtgraph)   |
|   ├── Table 1     |  - Subcolumns (A:Y1, A:Y2, ...)     |  - Scatter / Points           |
|   └── Table 2     |  - X Values (Numbers/Series)        |  - Fitted Curves              |
| 📁 Info Sheets    |  - Excluded points styling          |  - Column Bars (Mean ± SEM/SD)|
|   └── Info 1      |  - Missing values (NaN)             |  - Error Bars & Whiskers      |
| 📁 Results        |                                     |  - Auto-range & Log Axes      |
|   ├── Nonlin Reg  | Results Tabbed View                 |                               |
|   └── ANOVA       |  - Tabular summary matrices         | Tool Palette:                 |
| 📁 Graphs         |  - Parameter estimates ± SE         |  [Fit Curve] [Detect Outlier] |
|   ├── Graph 1     |  - Goodness-of-fit (R², Sy.x, AICc) |  [Run ANOVA] [Export Vector]  |
|   └── Graph 2     |                                     |                               |
+-------------------+-------------------------------------+-------------------------------+
| Status Bar: Active Sheet | Replicates: 3 | Mode: Clean-Room Strict-Local | Memory: OK   |
+-----------------------------------------------------------------------------------------+
```

### 2.1 View/ViewModel Synchronization Contracts
1. **Selection Model:** The navigation sidebar maintains a tree model of `SheetNode` instances organized into 4 primary folders:
   - `Data Tables` (`SheetType.DATA`)
   - `Info Sheets` (`SheetType.INFO`)
   - `Results` (`SheetType.RESULTS`)
   - `Graphs` (`SheetType.GRAPH`)
2. **Data-to-Graph Reactivity:** When cells in a `DataTable` are modified, edited, or marked as excluded, a signal `table_data_changed` is emitted. The linked `Graph` and dependent `Analysis` nodes receive invalidation events and queue recalculation/redraw on the Qt event loop.
3. **Replicate Grid Architecture:** Each column group (e.g. "Group A") consists of $M \ge 1$ subcolumns representing biological or technical replicates. Cell values support `NaN` (blank/missing) and an exclusion bitmask without deleting raw values.

---

## 3. Document & Storage Model (`Core.dll`)

Derived from container analysis and XML schema (`REA-CORE-STORAGE-204` & `REA-CORE-SCHEMA-208`):

### 3.1 Container Layout (`.prism`)
A `.prism` file is a standard PK-ZIP container containing:
- `document.json`: High-level manifest:
  ```json
  {
    "format_version": "10.0",
    "generator": "GemGraph Clean-Room 1.0",
    "created_at": "2026-10-19T19:00:00Z",
    "sheets": [
      {
        "id": "sheet_data_01",
        "title": "Dose Response",
        "type": "XY",
        "folder": "Data",
        "table_path": "data/tables/table_data_01.json",
        "csv_path": "data/tables/table_data_01.csv"
      }
    ],
    "graphs": [
      {
        "id": "graph_01",
        "title": "Dose Response Curve",
        "source_sheet_id": "sheet_data_01",
        "graph_type": "XY_SCATTER_CURVE"
      }
    ]
  }
  ```
- `data/sheets/`: Sheet metadata JSON files storing subcolumn configurations, column titles, and display formatting.
- `data/tables/`: Data matrix definitions, excluded point coordinates `[row, col, subcol]`, and row titles.
- `data/tables/*.csv`: Fast tabular storage for raw numeric data.

### 3.2 Supported Table Types
1. **XY:** Independent variable $X$ (numbers or series) and dependent variables $Y$ across columns $A, B, \dots$ with optional subcolumn replicates.
2. **Column (OneWay):** Groups arranged in columns (e.g., Control vs. Drug A vs. Drug B). No $X$ values.
3. **Grouped (TwoWay):** Two grouping factors: Rows represent Factor 1, Columns represent Factor 2, with subcolumns for replicates.
4. **Contingency:** Categorical counts in rows and columns for Chi-square or Fisher's exact tests.
5. **Survival:** Follow-up time in $X$, outcome code (Event / Censored: 1 or 0) in $Y$ subcolumns.

---

## 4. Mathematical & Algorithmic Engine Recovery (`prism.exe`)

Derived from numerical routine recovery (`REA-PRISM-NUM-301`, `REA-PRISM-ALGO-314`, `REA-PRISM-STAT-325`):

### 4.1 Nonlinear Regression Engine (`NonlinearEngine`)
1. **Optimization Algorithm:** Levenberg-Marquardt least-squares solver:
   $$(J^T J + \lambda \, \text{diag}(J^T J)) \Delta \theta = J^T (y - \hat{y})$$
2. **Convergence Tolerances Table:**
   - **Standard Preset:** Stop when relative decrease in Sum of Squares satisfies:
     $$\frac{SS_{\text{prev}} - SS_{\text{curr}}}{SS_{\text{curr}}} < 1.0 \times 10^{-4} \quad (0.0001)$$
   - **Strict Preset:** Stop when relative decrease satisfies:
     $$\frac{SS_{\text{prev}} - SS_{\text{curr}}}{SS_{\text{curr}}} < 1.0 \times 10^{-6} \quad (0.000001)$$
   - **Iteration Limit:** Default hard limit = 1000 iterations.
3. **Parameter Dependency Calculation:**
   Parameter dependency quantifies multicollinearity between parameters ($0 \le \text{Dependency} \le 1$):
   $$\text{Dependency}_i = 1 - \frac{1}{(C)_{ii} \cdot (C^{-1})_{ii}}$$
   where $C = (J^T J)^{-1}$ is the unscaled parameter covariance matrix. A value $> 0.99$ indicates severe parameter redundancy.
4. **Goodness-of-Fit Metrics:**
   - Standard Error of the Estimate: $S_{y.x} = \sqrt{\frac{SS}{N - K}}$
   - Coefficient of Determination: $R^2 = 1 - \frac{SS_{\text{reg}}}{SS_{\text{tot}}}$
   - Second-order Akaike Information Criterion:
     $$AIC_c = N \ln\left(\frac{SS}{N}\right) + 2K + \frac{2K(K+1)}{N - K - 1}$$
5. **Standard Model Library:**
   - 4-Parameter Logistic (Sigmoidal Dose-Response):
     $$Y = \text{Bottom} + \frac{\text{Top} - \text{Bottom}}{1 + 10^{(\log IC_{50} - X) \cdot \text{HillSlope}}}$$
   - One-site specific binding (Hyperbola):
     $$Y = \frac{B_{\max} \cdot X}{K_d + X}$$
   - Exponential Decay:
     $$Y = (Y_0 - \text{Plateau}) e^{-k X} + \text{Plateau}$$
   - Straight Line:
     $$Y = \text{Slope} \cdot X + \text{Intercept}$$

### 4.2 Outlier Detection Engine (`OutlierEngine`)
1. **ROUT Algorithm (Robust Regression and Outlier Removal):**
   - Step 1: Calculate robust standard deviation of residuals ($RSDR$):
     $$\text{MAD} = \text{median}(|r_i - \text{median}(r)|)$$
     $$RSDR = 1.4826 \times \text{MAD}$$
   - Step 2: For each point $i$, calculate the studentized residual ratio:
     $$t_i = \frac{|r_i|}{RSDR}$$
   - Step 3: Compute two-tailed $p$-value from Student's $t$-distribution with $df = N - K$:
     $$p_i = 2 \cdot (1 - F_t(t_i, df))$$
   - Step 4: Apply the Benjamini-Hochberg False Discovery Rate procedure with threshold $Q$ (default $Q = 1\% = 0.01$). Any point satisfying $p_{(i)} \le \frac{i}{N} Q$ is flagged as an outlier.
   - Constraint: Minimum sample size $N \ge 3$.
2. **Grubbs ESD (Extreme Studentized Deviate) Test:**
   - Test statistic:
     $$G = \frac{\max |y_i - \bar{y}|}{s}$$
   - Critical value at significance level $\alpha$ (default 0.05):
     $$G_{\text{crit}} = \frac{N - 1}{\sqrt{N}} \sqrt{\frac{t^2_{\alpha/(2N), N-2}}{N - 2 + t^2_{\alpha/(2N), N-2}}}$$
   - Point is an outlier if $G > G_{\text{crit}}$, evaluated for $N \ge 3$.

### 4.3 Group Comparisons & ANOVA Engine (`AnovaEngine`)
1. **One-Way ANOVA:**
   - $SS_{\text{total}} = \sum (y_{ij} - \bar{y}_{\cdot\cdot})^2$, $df = N - 1$
   - $SS_{\text{between}} = \sum n_j (\bar{y}_{\cdot j} - \bar{y}_{\cdot\cdot})^2$, $df = k - 1$
   - $SS_{\text{within}} = \sum \sum (y_{ij} - \bar{y}_{\cdot j})^2$, $df = N - k$
   - $F = \frac{MS_{\text{between}}}{MS_{\text{within}}}$, $p = 1 - F_{F}(F, k-1, N-k)$
2. **Two-Way ANOVA:**
   - Factor A ($r$ levels), Factor B ($c$ levels), Interaction ($A \times B$), Residuals.
3. **Repeated Measures ANOVA:**
   - Total variation partitioned into Between-Subjects, Between-Treatments, and Subject $\times$ Treatment Error.
   - Geisser-Greenhouse correction factor $\hat{\epsilon}$.
4. **Post-Hoc Dunnett's Test:**
   - Comparison of each group $j$ against control group $0$:
     $$q_j = \frac{\bar{y}_j - \bar{y}_0}{\sqrt{MS_{\text{within}} \left(\frac{1}{n_j} + \frac{1}{n_0}\right)}}$$
   - Evaluated against multivariate $t$-distribution / quadrature integration for exact $p$-values.
5. **Post-Hoc Tukey HSD Test:**
   - All pairwise comparisons with Studentized Range distribution critical values.

---

## 5. Security & Operating Gate

Operating Policy: **No-Sudo-Strict-Local**
- Pure local user-space Python execution.
- No root / sudo elevation anywhere in the codebase or build system.
- No direct execution of untrusted Windows PE binaries (`wine`, raw exec, `qemu`).
- File I/O strictly isolated to workspace or explicit user-approved boundaries.
- Continuous automated verification via `tests/test_security.py`.
