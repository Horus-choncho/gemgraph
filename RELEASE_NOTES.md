# 🧬 GemGraph v0.1.1-alpha — Comprehensive Scientific Release

Welcome to **GemGraph v0.1.1-alpha**, a clean-room, free, and open-source scientific desktop platform for biostatistics, non-linear curve fitting, and publication-ready vector graphing built as an alternative to GraphPad Prism 10.

---

### 🚀 What's New in v0.1.1-alpha (October 20–22, 2026 Sprints)

#### 1. Complete Undo/Redo Engine & Spreadsheet Usability
* **Batch Cell Deletion with Undo:** Pressing `Delete` or `Backspace` now clears all highlighted cells wrapped inside an atomic `BatchClearCellsCommand(QUndoCommand)`. Full cell contents and dependent curves are restored instantly with `Ctrl+Z` / `Edit > Undo`.
* **Focus-Aware Undo Routing:** `Ctrl+Z` and `Ctrl+Y` detect active widget focus: text edits in Info Sheets route to native text history without touching spreadsheet cells.
* **In-Sheet Column Renaming & Creation:** Double-clicking any column header opens an inline rename dialog; header context menus support adding X/Y columns left and right.

#### 2. Multi-X Column Architecture
* **Multi-Coordinate Tables:** Tables support multiple distinct X coordinate columns ($X_1, X_2, \dots$) with independent time scales or concentrations.
* **Dynamic Preceding-X Binding:** Consecutive Y replicate groups bind to their immediately preceding X column, allowing staggered kinetics or multi-curve experiments to plot concurrently.

#### 3. Advanced Hybrid Graphing Engine (`docs/graph_specs/`)
* **Bars + Individual Points (Column Format):** Overlays individual biological replicate points directly on summary bars with horizontal scatter jitter (`bars_and_points_overlaying_two_graphs.pzf`).
* **Grouped Bars + Replicate Scatter (Two-Factor Format):** Renders clustered grouped bars with individual replicate points nested over each sub-bar (`1661_Points_and_grouped_bars.pzfx`).
* **Columns / Spikes + Curve Fit (XY Format):** Mixed per-series representations on a shared coordinate plane: one series renders as baseline vertical spikes while an adjacent series renders as scatter points with an overlying regression curve (`bars_and_points_2.pzf`).
* **Per-Graph Scatter Point Toggle:** Toolbar button ("⚪ Show Scatter Points") and canvas context menu option to toggle scatter overlays on bar charts.
* **Binary PCFF (`.pzf`) & XML (`.pzfx`) Engine:** Native binary stream parser for legacy GraphPad Prism 4/5 `.pzf` binary files alongside modern `.pzfx` XML archive compatibility.

#### 4. Pinned & Frozen Graphs Architecture
* **📌 Pin Graph:** Captures an active graph view, linked dataset, and display configuration as an independent `GraphSheet` under `📈 Graphs`.
* **❄ Freeze Graph:** Locks pinned graphs against upstream table edits and serializes snapshot data, preserving regression curve overlays.
* **Per-Sheet Graph Persistence:** Switching between sheets retains custom graph selections, whisker settings, and colors without resetting to default XY scatter.
* **Export State Isolation:** Report exporters render graphs in an isolated pipeline, ensuring document exports never mutate the active on-screen graph view.

#### 5. Publication Grayscale & Visual Formatting Polish
* **Opaque Journal Grayscale Palette:** 5 publication-grade monochromatic fills: Solid Black, Pure White with bold Black border, Medium Gray, Light Gray, and Diagonal Cross-Hatch patterns.
* **High-Contrast Scatter Symbology:** Scatter points plotted over dark bars invert to white fills with black borders (`symbolBrush='w', symbolPen='k'`).
* **Configurable Error Whiskers:** Canvas controls support toggling between bidirectional `Both (±)` whiskers and unilateral `Upper (+) Only` whiskers.
* **Per-Series Formatting Dialog:** Right-clicking any series opens `FormatSeriesDialog` with a custom color picker and a 0–100% opacity slider.
* **Analysis Layer Registry & Clearing:** "🧹 Clear Analysis" toolbar menu allows removing individual fitted curves or clearing all overlays without resetting raw scatter data.

#### 6. Rich Text Editor & Scientific Info Sheets (GLP Metadata)
* **Interactive Scientific Info Sheets:** Structured GLP metadata tracking Experiment Title, Study ID, Scientist, PI, Notebook Page Ref, Date, Organism/Strain, Cell Line/Passage, Compound Batch/Lot Number, and SOP Protocol ID.
* **Rich Text Formatting Toolbar:** Integrated rich text editor for experimental notes with Sans-Serif, Serif, and Monospace fonts, 3 preset sizes (10pt, 12pt, 16pt), Bold, Italic, Underline, Bullet Lists, `- ` + `Space` auto-bullets, and `Tab`/`Shift+Tab` indentation.

#### 7. Granular Office & PDF Export Suite
Built a zero-subprocess, pure-Python export engine (`File > Export Report / Office Document...`):
* **Word Document (`.docx`):** Packages OpenXML ZIP archives with rich text formatting, structured data tables, and embedded high-DPI drawings (`DrawingML`).
* **OpenDocument Text (`.odt`):** OASIS XML package with standard text styles, tables, and embedded graph images in `Pictures/`.
* **Excel Workbook (`.xlsx`) & OpenDocument Spreadsheet (`.ods`):** Multi-sheet workbooks cleanly separating raw data and statistical results tables.
* **Publication PDF (`.pdf`):** Multi-page 300 DPI (A4) publication report rendered via `QTextDocument` + `QPdfWriter`.
* **Granular Scope Selector:** `ExportProjectDialog` allows selecting or deselecting individual Info Sheets, Data Tables, Analysis Ledgers, and Graphs.

#### 8. Native `.bigo` Format & Bundled Textbook Datasets
* **Native `.bigo` Format:** Primary project file format (`.bigo`), matching Prism 10's internal XML/ZIP archive architecture.
* **Quick-Load Textbook Projects:** `File > Open Sample Project (.bigo)` shipping with calibrated benchmark datasets (Dose-Response 4PL, One-Way ANOVA, Kaplan-Meier Survival).
* **Project Navigator Context Menus:** Specific rename/delete labels per category and `"➕ Add Info Sheet"` on the `📁 Info Sheets` root node.

---

### 🔬 Feedback & Incident Reporting

* **Email Feedback & Incident Reports:** Send reports or feature suggestions to **[gemgraph@outlook.com](mailto:gemgraph@outlook.com)**.
* **Bug Report Template:** Refer to `docs/FEEDBACK_TEMPLATE.md` in the repository for structured reporting guidelines.
* **GitHub Issues:** Open an issue at [https://github.com/Horus-choncho/gemgraph/issues](https://github.com/Horus-choncho/gemgraph/issues).

---

### 🔒 Cryptographic Verification (SHA-256)

All binary releases are signed in the `SHA256SUMS.txt` manifest attached to this release:
```bash
sha256sum -c SHA256SUMS.txt
License: Distributed under the GNU General Public License v3.0 (GPLv3).
