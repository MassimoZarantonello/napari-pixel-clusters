# napari plugin: pixel-cluster explorer for before/after tissue imaging

## Goal
Make it easy for a biologist to see what the pixel clusters (FlowSOM metaclusters) are,
interpret them, and see how they change between before/after tissue images.

## Starting point
- Existing notebooks (`main.ipynb`, `neuroblastoma.ipynb`): pixel clustering + FlowSOM -> metaclusters.
- Already validated: showing each cluster as a separate napari layer works well.
- Plugin skeleton idea: reuse the structure of the other plugin we already know (to decide tomorrow what to copy).

## Decisions so far (brainstorm day 1)
- Data: MACSima OME-TIFF, whole slides (~10325 x 9424 px, many channels). Different tissues before/after.
  Marker panels are NOT identical across images -> marker list = per-image lists + intersection for shared clustering.
- Plugin loads the WHOLE image lazily (never all channels in RAM; multiscale/dask) for viewing.
  Then the user picks the region to cluster: draw an ROI (napari Shapes layer) = fast mode, because the user's Mac is slow.
  Same ROI is used for the before/after comparison.
  Final design: the clustering region can be set to the WHOLE image (ROI is a speed convenience, not the core concept).
  So "region to cluster" = ROI box or whole image, one switch.
- Clustering runs inside the plugin via harpy (`hp.im.pixel_clustering_preprocess`, `hp.im.flowsom`, which accepts a list of images).
  Reuse napari-spatialdata / harpy napari tools where possible, don't duplicate.
- Training happens ONCE per image (auto on load, ~25 s): random tiles (48 x 256x256 px) sampled across the whole slide,
  normalisation constants fitted on them (own re-implementation of harpy's pixel_clustering_preprocess, same steps),
  FlowSOM via the `flowsom` lib. Clustering an ROI = preprocess + predict only (~1-4 s). Retrain only when the marker set changes.
  Multi-image: pass several stacks to train_model to pool tiles (same function).
  Trade-off seen on ROI1: whole-slide model gives a more dominant big cluster in a given ROI than an ROI-trained model; use higher k.
- Panel = 3 tabs: ROIs (draw, list, cluster) / Clusters (k, hide-all, named metaclusters) / Markers (show+use, retrain).
  Results are cached per ROI id: adding a ROI never reclusters old ones; deleting/editing a ROI deletes its clusters.
  ROIs are clipped to the image; ROIs > 16 Mpx are refused for now (whole-image mode comes later).
- Runs (decided day 2): a run = marker set + trained model + results per ROI. Selector in the Clusters tab (rename/delete),
  Retrain tab makes a NEW run and keeps old ones; marker list stays whole, markers unused by the shown run are greyed.
  ROIs belong to the image and are shared by all runs. No run-vs-run comparison for now (only one run shown at a time).
  Results stored light (label image + per-SOM-node signal sums), so the heatmap works for any k.
- Heatmap + profile in the Clusters tab: markers x metaclusters (z-score per marker, clip +-3, or mean signal), profile bars of the
  selected cluster aligned on the same rows. Select by clicking a pixel in the image, a table row, or a heatmap column.
- k (metaclusters) is a live slider: regroup the 100 SOM node centres, no retraining (see notebook cell 17).
  Changing markers = real re-run.
- One marker list, two checkboxes per marker: show channel in viewer / include in next FlowSOM. Heatmap next to it.
- Heatmap = per-metacluster mean, z-scored per marker, clipped +-3 (notebook cell 21). Click pixel -> profile panel. No hover.
- Comparison (later): default = pool ROIs of both images in one FlowSOM training; maybe "train on first, project second" as advanced option.
- Reusable from notebook: elbow plot (cell 11), variance map (cell 23), per-cluster napari layers (cell 24).
- harpy needs dask.config.set(scheduler="synchronous") (numba crash with threads).

## Phases

### Phase 0 - Foundations (decide tomorrow)
- Input format: what a "sample" is (multi-channel image + marker names + optional mask/ROI).
- Pipeline split: clustering code as a plain Python package (no napari imports), plugin = thin GUI on top.
- Scaffold: `napari-plugin-template` / npe2 manifest, pyproject, dock widgets.
- Data model: Session object holding image, marker list, pixel->cluster labels, metacluster labels, user names.

### Phase 1 - Visualise + annotate (MVP)
- Load image + clustering result (or run FlowSOM from the widget).
- One Labels/Image layer per metacluster, consistent colours, toggle visibility.
- Click a pixel/region -> identify its metacluster -> side panel to give it a name
  (e.g. "T cell-like", "stroma", "artefact") and optional notes/colour.
- Per-metacluster summary: mean marker expression (heatmap / bar plot), pixel count, % area.
- Linked profile panel (no alt-tabbing): click a pixel -> dock widget shows that metacluster's marker profile
  (bar chart, z-score vs other clusters) next to name field + pixel count/% area.
- Heatmap (clusters x markers) as a dock widget, linked both ways: clicked cluster's row highlighted;
  clicking a heatmap row highlights that cluster in the image. Toggle raw / z-scored.
- Names update layer names and legend. Save/load annotations (JSON next to the data).

### Phase 2 - Marker curation + re-clustering (separate tab)
- Marker list with checkboxes (include/exclude), plus quality hints (variance map, per-marker stats; we already have `cluster_variance`/`variance_map` ideas).
- "Re-run FlowSOM" with chosen markers and parameters (grid size, n metaclusters, seed).
- Runs in background thread with progress; result added as a new "run" (don't overwrite).
- Run history so the biologist can switch back/compare runs.
- Open question: carry over names from previous run (match by centroid similarity)?

### Phase 3 - Compare two images (before vs after)
Main design question for the brainstorm. Options:
1. Shared clustering: fit FlowSOM on the pooled pixels of both images, so metaclusters mean the same thing in both. Simplest and most comparable.
2. Cluster separately, then match metaclusters across images by centroid distance (Hungarian matching).
3. Fit on "before", project "after" onto it (reference/query).
Comparison views:
- Side-by-side napari grids (linked cameras) with the same colours/names.
- Table/plot: % area per metacluster before vs after, delta.
- Difference layer: where a cluster appeared/disappeared (needs registration if same tissue; if different tissues, only stats).
- Normalisation/batch concerns between acquisitions.

### Phase 4 - Polish
- Export: figures, tables, annotated label images, session file.
- Tests, docs, example dataset, packaging (napari hub).

## Brainstorm agenda for tomorrow
1. Who is the user and what does a typical session look like (click-through script)?
2. Same tissue before/after, or different tissues? Registration needed?
3. Image size/count: does FlowSOM re-run in seconds or minutes? Need subsampling?
4. Comparison approach (shared vs matched vs reference clustering).
5. How are names stored and reused across runs and images?
6. UI sketch: tabs (Explore / Markers / Compare), what lives in napari vs Qt widgets.
7. Stack: napari version, FlowSOM implementation (which python lib), plotting (matplotlib vs pyqtgraph).

## Risks
- Re-clustering changes metacluster IDs -> annotations get lost (need matching strategy).
- Large images -> memory/speed; need subsample for fitting, assign all pixels afterward.
- Comparison validity across different tissues/acquisitions.
