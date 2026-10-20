# GemGraph UI/UX Visual Specification Manifest

| Category | Reference Files / Visual Highlights | Target GemGraph Modules |
| :--- | :--- | :--- |
| **8 Scientific Table Formats** | XY, Column, Grouped, Contingency, Survival, Parts of Whole, Multiple Variables, and Nested tables with replicate subcolumns and exclusion rendering (blue italics). | `src/core/table.py`<br>`src/ui/spreadsheet.py` |
| **Multiple Variable Metadata** | Variable types (Categorical, Continuous, Label icons), Level Order drag/sort dialog, and row label column bindings. | `src/ui/format_table_dialog.py`<br>`src/core/table.py` |
| **Analyze Data Dialog & Hierarchy** | Two-pane selector: Search bar + hierarchical tree on the left; source table & dataset selector checklist (`[x] A:Control`) on the right. Contextual toolbar buttons. | `src/ui/analyze_dialog.py`<br>`src/ui/main_window.py` |
| **Data Manipulation & Transforms** | Standard math transforms, pharmacological transforms (Lineweaver-Burk, Hill, Scatchard), concentration X=0 handling, normalization, pruning, baseline subtraction, fraction of total. | `src/math/transforms.py`<br>`src/math/manipulation.py`<br>`src/ui/dialogs.py` |
| **Analysis Tabs & Grid Semantics** | Tabbed results (Table, Curve, Residuals, Outliers); Show/Hide tab dropdown; Green Grid (analyzable/plottable) vs. Red Grid (read-only); Frozen (snowflake) and Orphaned states; Result cell color coding. | `src/ui/results_view.py`<br>`src/core/document.py` |
| **Constant Hooking & Graph Polish** | Hook button (`s`), hierarchical parameter browser, dynamic tick marks/grid lines linked to values (e.g. Mean ± 2SD), dynamic text equations on plot, combined bars + points. | `src/core/hooking.py`<br>`src/ui/graph_canvas.py` |
