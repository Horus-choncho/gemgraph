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
### 4. Fedora / RHEL (.rpm):
- Download `gemgraph-0.1.0-alpha.x86_64.rpm` and install via:
`sudo dnf install ./gemgraph-0.1.0-alpha.x86_64.rpm`

### 5. Flatpak (.flatpak):
- Standard Flatpak installation:
Download `gemgraph_0.1.0-alpha.flatpak` and install:
```
flatpak install --user gemgraph_0.1.0-alpha.flatpak
flatpak run org.gemgraph.GemGraph
```
- Note to Flatpak users: 
You may need to run GemGraph from the terminal after installation if the app does not launch on normal click (should be working, though):
`flatpak run org.gemgraph.GemGraph`

### 6. Windows 
- Windows Portable (.zip):
Download `GemGraph-0.1.0-alpha-Windows-Portable.zip`, unzip to a folder, and double-click `GemGraph.exe` (no administrator privileges required).

### 7. macOS Users 🍎🍎🍏🍏

- macOS 15 Sequoia and later:

1. Double-click `GemGraph-0.1.0-alpha-macOS.dmg` to mount the disk image.
2. Drag GemGraph.app into the Applications folder shortcut.
3. Attempt to open GemGraph.app from Applications. macOS will display a warning that the app cannot be opened because the developer cannot be verified. Click Done (or Cancel).
4. Open System Settings > Privacy & Security.
5. Scroll down to the Security section. You will see: `"GemGraph.app was blocked to protect your Mac."`
6. Click the Open Anyway button next to it, enter your Mac user password or Touch ID, and confirm by clicking Open. macOS will permanently whitelist the app for all subsequent launches.


- macOS 14 Sonoma or older:
1. Drag `GemGraph.app` into Applications.
2. Do not double-click it directly on first launch.
3. Right-click (or Control-click) on `GemGraph.app` and select Open.
4. In the dialog that appears, click the Open button. macOS will whitelist the application for all future double-clicks.


- (Alternative for all macOS via Terminal): Run `xattr -cr /Applications/GemGraph.app` to clear the quarantine flag.

---

## 💬 Community Feedback & Call-to-Action

We invite experimental researchers, pharmacologists, and statisticians to put GemGraph to the test:
1. **Test with your real laboratory data:** Load your experimental tables and compare curve fits, IC₅₀ values, and ANOVA results with Prism.
2. **Report edge cases & discrepancies:** If any calculation differs from expected values, open an issue on GitHub using our issue template and include your anonymized test table.
3. **Request models & features:** Tell us which regression equations, post-hoc corrections, or plot styles you need next!
