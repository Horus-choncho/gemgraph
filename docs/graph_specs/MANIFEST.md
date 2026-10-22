# GemGraph Graph Specifications & Visual Reference Manifest

This manifest maps the sequential screenshot series and GraphPad Prism reference files located in `/home/nicolauda/projects/gemgraph/docs/graph_specs/` to their target GemGraph architecture modules.

## Reference File & Image Mapping

| Category / Prism Feature | Reference Files (.pzf / .pzfx) | Screenshot Sequences (.png) | Target GemGraph Modules | Visual & Statistical Behavior |
| :--- | :--- | :--- | :--- | :--- |
| **Combined Bars & Points (Overlaying Two Graphs)** | `bars_and_points_overlaying_two_graphs.pzf` | Sequential screenshots matching *"Graph tip - Showing both bars and individual points by overlaying two graphs"* | `src/ui/plot_canvas.py`<br>`src/ui/mediator.py`<br>`src/data_engine/table_model.py` | Overlays raw scatter points (individual biological replicates) directly on top of summary column bars with Mean ± SD/SEM error whiskers. Requires transparent/semi-transparent bar fills and jittered or center-aligned scatter markers. |
| **Combined Bars & Points (Grouped Graph)** | `1661_Points_and_grouped_bars.pzfx` (or `.pzf`) | Sequential screenshots matching *"Graph tip - Combining bars and points on a grouped graph"* | `src/ui/plot_canvas.py`<br>`src/ui/grouped_plot.py`<br>`src/ui/dialogs/` | Two-way grouped data (Factor A × Factor B) displaying clustered bars with matched individual replicate points nested inside each category group. Requires discrete symbol/color assignment per condition. |
| **Columns & Points / Curve Fit on XY Graph** | `bars_and_points_2.pzf`<br>`BarsandPointsStacked.pzf` | Sequential screenshots matching *"Graph tip - Combining columns and points or a curve fit on an XY graph"* | `src/ui/plot_canvas.py`<br>`src/analysis/curve_fitting.py`<br>`src/ui/format_dialog.py` | Hybrid XY graphs where one Y dataset renders as vertical bar spikes/columns while an adjacent Y dataset renders as scatter points and a continuous non-linear regression curve. Requires layer-based curve overlay clearing and independent series rendering flags. |

## Implementation Directives for GemGraph

1. **Overlay Layer Management:**
   - Plotted items must be registered in an `analysis_layers` table in `ScientificPlotCanvas`.
   - Each fitted curve or statistical overlay must be individually removable via a context menu or toolbar action without clearing the raw scatter/column data.

2. **Journal-Style Grayscale Formatting:**
   - Scatter plots must support alternating marker shapes and hollow/solid fills for monochrome printing:
     - Series 1: Closed Circle (`'o'`, `#000000`)
     - Series 2: Open Square (`'s'`, Pen: `#000000`, Brush: `#ffffff`)
     - Series 3: Closed Triangle (`'t'`, `#000000`)
     - Series 4: Open Diamond (`'d'`, Pen: `#000000`, Brush: `#ffffff`)

3. **Responsive Resizing Stability:**
   - Eliminate pane snapping in non-maximized resolutions by replacing destructive collapse flags with `setChildrenCollapsible(False)` and balanced splitter stretch factors.
