"""Headless smoke test: python tests/smoke.py  (loads the real slide, trains, exercises ROIs, runs, heatmap)."""
import os, time
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
from napari.components import ViewerModel
from qtpy.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])
import napari_pixel_clusters.widget as W
from napari_pixel_clusters.io import META_KEY
from napari_pixel_clusters.reader import get_reader

PATH = os.path.expanduser("~/Documents/annoloop_demo_images/clustering_images/ROI1_AllMarkers.tif")


def wait(cond, timeout=600):
    t = time.time()
    while not cond() and time.time() - t < timeout:
        app.processEvents(); time.sleep(0.05)
    return time.time() - t


def idle():
    wait(lambda: not w.busy); wait(lambda: not w.busy)


def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name)
    return ok


def run_layers(run):
    return {rid: l for (run_id, rid), l in w.roi_layers.items() if run_id == run.id}


def drawn(layer):  # clusters a labels layer actually draws: colour not transparent
    return {c for c, col in layer.colormap.color_dict.items() if isinstance(c, int) and c > 0 and col[-1] > 0}


def present(layer):
    return {int(v) for v in np.unique(layer.data) if v != 0}


calls = []
_orig = W.cluster_rois
W.cluster_rois = lambda model, stack, rois, report=None: (calls.append(list(rois)), _orig(model, stack, rois, report))[1]

from napari.plugins import _initialize_plugins
_initialize_plugins()  # the real napari app does this at startup: it discovers our reader
viewer = ViewerModel()
w = W.ClusterExplorer(viewer)
w.load_image(PATH)  # = File > Open: napari reader -> channel layers -> the panel attaches
wait(lambda: w.stack is not None)
print("training on load took", round(wait(lambda: not w.busy)), "s |", w.run_label.text())
H, Wd = w.roi_tool.image_hw
check("three tabs", [w.tabs.tabText(i) for i in range(3)] == ["ROIs", "Clusters", "Retrain"])
check("Run 1 exists after load, 29 markers", len(w.runs) == 1 and w.run.name == "Run 1" and len(w.run.model.channels) == 29)
check("run tab bar lists it", w.run_bar.count() == 1 and w.run_bar.tabText(0) == "Run 1" and not w.run_bar.isHidden())
check("no Load button: the image comes from napari's File > Open", not hasattr(w, "load_btn"))
check("the reader gave one lazy layer per channel (33) and nothing else", len(viewer.layers) == 33 and all(l.metadata.get(META_KEY) for l in viewer.layers) and all(l.multiscale for l in viewer.layers))
check("only DAPI is visible after opening", [l.name for l in viewer.layers if l.visible] == ["DAPI C1 (DAPI C1)"])

# the reader only accepts multi-channel CYX tiffs
import tempfile, tifffile
tmp = tempfile.mkdtemp()
tifffile.imwrite(os.path.join(tmp, "flat.tif"), np.zeros((50, 50), np.uint16))
tifffile.imwrite(os.path.join(tmp, "cyx.tif"), np.zeros((3, 40, 40), np.uint16), metadata={"axes": "CYX"})
open(os.path.join(tmp, "notes.txt"), "w").write("x")
check("reader says no to a 2D tiff, a text file and a list of paths", get_reader(os.path.join(tmp, "flat.tif")) is None and get_reader(os.path.join(tmp, "notes.txt")) is None and get_reader([PATH]) is None)
rd = get_reader(os.path.join(tmp, "cyx.tif"))
data, attrs, kind = rd(os.path.join(tmp, "cyx.tif"))[0]
check("reader says yes to a CYX tiff and names the channels", rd is not None and kind == "image" and attrs["channel_axis"] == 0 and len(attrs["name"]) == 3 and len(data) >= 1)
check("marker names not editable", not (w.table.item(0, 1).flags() & W.Qt.ItemFlag.ItemIsEditable))

shapes = lambda: w.roi_tool.layer
w.roi_tool.start_drawing()
shapes().add_rectangles([[[4000, 4000], [4400, 4300]]])
w._run(); idle()
r1 = w.run
layers_run1 = dict(w.roi_layers)
check("ROI 1 clustered in Run 1: ONE layer for the ROI (not one per cluster)", len(layers_run1) == 1 and 1 in r1.results and len(run_layers(r1)) == 1)
check("results are light (no preprocessed image kept)", not hasattr(r1.results[1], "preprocessed") and r1.results[1].node_sum.shape == (101, 29))

# heatmap + profile
cl = sorted(present(run_layers(r1)[1]))
check("heatmap has one column per cluster", w.heatmap._cols == cl)
w._select_cluster(cl[0])
check("selecting a cluster marks it in the table (bold)", any(w.mc_table.item(r, 2).font().bold() for r in range(w.mc_table.rowCount())))
w.selected_cluster = None
w._select_at((4100.5, 4100.5))
check("click on a pixel selects its cluster", w.selected_cluster in cl)


class Ev:
    def __init__(s, pos, typ, position): s.pos, s.type, s.position = pos, typ, position


w.selected_cluster = None
shapes().mode = "add_rectangle"  # while drawing ROIs, clicks must not select clusters
ev = Ev((10, 10), "mouse_press", (4150, 4150)); g = w._on_click(viewer, ev)
try: next(g)
except StopIteration: pass
check("clicks are ignored while drawing an ROI", w.selected_cluster is None)
shapes().mode = "pan_zoom"  # (the real app returns to this mode after every rectangle)
ev = Ev((10, 10), "mouse_press", (4150, 4150)); g = w._on_click(viewer, ev); next(g)
ev.type = "mouse_release"; ev.pos = (11, 10)
try: next(g)
except StopIteration: pass
check("plain click (4 px or less) selects", w.selected_cluster is not None)
w.selected_cluster = None
ev = Ev((10, 10), "mouse_press", (4150, 4150)); g = w._on_click(viewer, ev); next(g)
ev.type = "mouse_move"; ev.pos = (60, 10); next(g)
ev.type = "mouse_release"
try: next(g)
except StopIteration: pass
check("drag (pan) does not select", w.selected_cluster is None)
w.z_box.setChecked(False); w.z_box.setChecked(True)

# second ROI is not reclustered in the same run
shapes().add_rectangles([[[2000, 2000], [2300, 2500]]])
n = len(calls); w._run(); idle()
check("only the new ROI computed; old layers untouched", len(calls) == n + 1 and len(calls[-1]) == 1 and all(w.roi_layers.get(k) is l for k, l in layers_run1.items()))

# master switch
w._set_cluster_visible(cl[0], False)
check("individual off: that cluster is transparent in every layer, the others stay", all(cl[0] not in drawn(l) and set(cl[1:]) <= drawn(l) for l in run_layers(r1).values()))
w.master_box.setChecked(False)
check("master off hides all layers, keeps individual choices", not any(l.visible for l in w.roi_layers.values()) and r1.mc_selected[cl[0]] is False)
w.master_box.setChecked(True)
check("master on restores exactly the previous selection", all(l.visible and cl[0] not in drawn(l) and set(cl[1:]) <= drawn(l) for l in run_layers(r1).values()))
w._set_cluster_visible(cl[0], True)
check("all clusters drawn again", all(set(cl) <= drawn(l) for l in run_layers(r1).values()))

# k change is delayed and regroups without reclustering
regroups = []; _kc = w._k_changed
w._k_changed = lambda k: (regroups.append(k), _kc(k))
w._k_timer.timeout.disconnect(); w._k_timer.timeout.connect(lambda: w._k_changed(w.k_spin.value()))
n = len(calls)
for k in (8, 7, 6, 5):
    w.k_spin.setValue(k); app.processEvents(); time.sleep(0.1)
check("k change waits while still changing", regroups == [])
wait(lambda: regroups); time.sleep(0.3); app.processEvents()
check("one regroup, last value, no reclustering", regroups == [5] and len(calls) == n and r1.k == 5)
w.mc_table.item(0, 2).setText("macrophages")
check("renaming a metacluster is stored and shown in the list", "macrophages" in r1.mc_names.values() and "macrophages" in [w.mc_table.item(r, 2).text() for r in range(w.mc_table.rowCount())])

# names and grouping are remembered per k
lk5 = r1.lookup
w._k_changed(8)
check("k=8 starts without the k=5 names", r1.mc_names == {})
w._k_changed(5)
check("back to k=5: names, the very same grouping and layer data are back", "macrophages" in r1.mc_names.values() and r1.lookup is lk5 and np.array_equal(run_layers(r1)[1].data, lk5[r1.results[1].som]))

# marker lists: search, All/None/Default, Hide all
w.show_boxes[3].setChecked(True); w.show_boxes[7].setChecked(True)
w.show_search.setText("cd")
hidden = [w.table.isRowHidden(r) for r in range(w.table.rowCount())]
check("search hides non-matching markers", any(hidden) and not all(hidden) and all(("cd" in w.stack.channel_names[r].lower()) != h for r, h in enumerate(hidden)))
w.show_search.setText("")
check("clearing the search shows all rows", not any(w.table.isRowHidden(r) for r in range(w.table.rowCount())))
n_use = len(w.selected_markers())
w.use_search.setText("cd"); w._set_use(lambda n: False)
check("None unticks only the markers the search shows", 0 < len(w.selected_markers()) < n_use)
w.use_search.setText(""); w._set_use(lambda n: n in W.default_included(w.stack.channel_names))
check("Default restores the 29 default markers", len(w.selected_markers()) == 29)
w._set_use(lambda n: False); check("None unticks all", w.selected_markers() == [] and not w.train_btn.isEnabled())
w._set_use(lambda n: True); check("All ticks all", len(w.selected_markers()) == 33)
w._set_use(lambda n: n in w.run.model.channels)
for b in w.show_boxes: b.setChecked(False)
check("Hide all switches off every channel", not any(viewer.layers[n].visible for n in w.stack.channel_names))
w.show_boxes[0].setChecked(True)

# ---- runs ----
# new run with fewer markers: Run 1 is kept, the ROIs get clustered in Run 2
drop = w.run.model.channels[0]
box = w.use_boxes[w.stack.channel_names.index(drop)]
box.setChecked(False)
check("retrain tab announces it will be run 2", "run 2" in w.retrain_label.text())
n = len(calls); w._train()
check("busy bar is shown while training", w.busy and not w.progress.isHidden())
idle()
check("busy bar is hidden when done", w.progress.isHidden())
check("Run 2 created and shown, Run 1 kept", len(w.runs) == 2 and w.run.name == "Run 2" and w.runs[0] is r1)
check("Run 2 uses 28 markers", len(w.run.model.channels) == 28 and drop not in w.run.model.channels)
check("both ROIs clustered in Run 2 (one call, two ROIs)", len(calls) == n + 1 and len(calls[-1]) == 2 and len(w.run.results) == 2)
check("unused marker greyed in the list, still listed", w.table.rowCount() == 33 and w.table.item(w.stack.channel_names.index(drop), 1).foreground().color().name() == "#888888")
check("Run 2 has its own (unnamed) metaclusters", "macrophages" not in w.run.mc_names.values() and w.run.k == 5)
check("run tab bar lists 2 runs, Run 2 current", w.run_bar.count() == 2 and w.run_bar.currentIndex() == 1)

# switching back is instant: no clustering, names and k of Run 1 are back, grey is gone
n = len(calls); ids_before = {k: id(l) for k, l in w.roi_layers.items()}; nl = len(viewer.layers)
t_sw = time.time(); w.run_bar.setCurrentIndex(0); app.processEvents(); t_sw = time.time() - t_sw
check("switching creates no new layers and is quick (%.2f s)" % t_sw, ids_before == {k: id(l) for k, l in w.roi_layers.items()} and len(viewer.layers) == nl and t_sw < 1.0)
check("only the shown run's layers are visible", all(l.visible == (k[0] == r1.id) for k, l in w.roi_layers.items()))
check("switch to Run 1: no clustering", len(calls) == n and w.run is r1)
check("Run 1 names are back", "macrophages" in r1.mc_names.values() and "macrophages" in [w.mc_table.item(r, 2).text() for r in range(w.mc_table.rowCount())])
check("Run 1 marker list has no grey markers", w.table.item(w.stack.channel_names.index(drop), 1).foreground().color().name() != "#888888")
check("Use ticks follow the shown run", w.use_boxes[w.stack.channel_names.index(drop)].isChecked())

# training the same marker set again just shows the existing run
w.run_bar.setCurrentIndex(1); app.processEvents()
w.use_boxes[w.stack.channel_names.index(drop)].setChecked(True)
check("ticking Run 1's markers says 'same as Run 1'", "same as Run 1" in w.retrain_label.text())
w._train()
check("no new training: Run 1 shown", len(w.runs) == 2 and w.run is r1)

w.run_bar.setCurrentIndex(1); app.processEvents()
w._set_run_name("without " + drop[:6])
check("rename a run", w.run.name.startswith("without") and "without" in w.run_bar.tabText(1))

W.QInputDialog.getText = staticmethod(lambda *a, **k: ("renamed by double click", True))
w.run_bar.rename_requested.emit(0)
check("double-click rename (tab 1)", w.runs[0].name == "renamed by double click" and w.run_bar.tabText(0) == "renamed by double click" and w.run is w.runs[1])
check("run tabs are at the bottom and closable; main tabs are at the top", w.run_bar.shape() == W.QTabBar.Shape.RoundedSouth and w.run_bar.tabsClosable() and w.tabs.bar.shape() == W.QTabBar.Shape.RoundedNorth)
def _align(widget):
    lay = w.layout()
    return next(lay.itemAt(i).alignment() for i in range(lay.count()) if lay.itemAt(i).widget() is widget)
check("both tab bars are packed to the left", _align(w.run_bar) == W.Qt.AlignmentFlag.AlignLeft and _align(w.tabs.bar) == W.Qt.AlignmentFlag.AlignLeft)

# ROI deleted: its clusters disappear in the shown run, and ROI results are dropped from every run
row = lambda i: w.roi_list.itemWidget(w.roi_list.item(i))
check("no 'Remove selected ROI' button any more", not hasattr(w, "remove_btn"))
check("ROI rows show an x only while hovered", row(0).x.isHidden() and (row(0).enterEvent(None) or True) and not row(0).x.isHidden() and (row(0).leaveEvent(None) or True) and row(0).x.isHidden())
row(0).x.clicked.emit()  # = clicking the x of ROI 1
wait(lambda: 1 not in [i.id for i in w.roi_tool.rois()], 5)
check("delete ROI 1: no layers of ROI 1, result dropped in both runs", not any(rid == 1 for _, rid in w.roi_layers) and 1 not in r1.results and 1 not in w.run.results)
check("ROI 2 kept in both runs", 2 in r1.results and 2 in w.run.results)

# ROIs bigger than the image
shapes().add_rectangles([[[H - 300, Wd - 300], [H + 400, Wd + 400]]])
info = w.roi_tool.rois()[-1]
check("ROI partly outside is clipped", info.box == (H - 300, H, Wd - 300, Wd) and info.clipped and "clipped" in row(w.roi_list.count() - 1).text())
w._run(); idle()
check("clipped ROI clusters fine", w.run.results[info.id].som.shape == (300, 300))
shapes().add_rectangles([[[-500, -500], [H + 500, Wd + 500]]])
big = w.roi_tool.rois()[-1]
check("whole-slide ROI clipped to the slide", big.box == (0, H, 0, Wd))
n = len(calls); w._run(); app.processEvents()
check("whole-slide ROI refused with a message", len(calls) == n and "larger than" in w.status.text())
shapes().add_rectangles([[[H + 100, Wd + 100], [H + 300, Wd + 300]]])
check("ROI entirely outside is ignored", len(w.roi_tool.rois()) == 3)

# delete runs (asks first)
W.QMessageBox.question = staticmethod(lambda *a, **k: W.QMessageBox.StandardButton.No)
w._confirm_delete_run(w.runs.index(w.run))
check("answering No keeps the run", len(w.runs) == 2)
W.QMessageBox.question = staticmethod(lambda *a, **k: W.QMessageBox.StandardButton.Yes)
w._confirm_delete_run(w.runs.index(w.run))
check("delete the shown run: the other one is shown", len(w.runs) == 1 and w.run is r1)
# closing a tab that is not the shown one deletes only that run
w.use_boxes[w.stack.channel_names.index(w.run.model.channels[0])].setChecked(False)
w._train(); idle()
check("a new run was made (r1 + new)", len(w.runs) == 2 and w.run is w.runs[1])
shown = w.run
w.run_bar.tabCloseRequested.emit(0)
check("closing the other run's tab keeps the shown run", len(w.runs) == 1 and w.run is shown and w.run_bar.count() == 1 and w.run_bar.currentIndex() == 0)
w._delete_run()
check("delete the last run: nothing shown, no layers, tab bar hidden", w.run is None and not w.roi_layers and w.mc_table.rowCount() == 0 and w.run_bar.isHidden())

# a panel opened AFTER the image was opened attaches to it as well
trained = []
_orig_train = W.ClusterExplorer._train
W.ClusterExplorer._train = lambda self, *a, **k: trained.append(1)
w2 = W.ClusterExplorer(viewer); wait(lambda: w2.stack is not None, 10)
W.ClusterExplorer._train = _orig_train
check("panel created after the image attaches to it", w2.stack is not None and w2.stack.path.name == "ROI1_AllMarkers.tif" and w2.table.rowCount() == 33 and trained)

# removing the image's channel layers resets the panel
for l in [l for l in list(viewer.layers) if l.metadata.get(META_KEY)]:
    viewer.layers.remove(l)
app.processEvents()
check("removing the image layers detaches the panel", w.stack is None and w2.stack is None and w.table.rowCount() == 0 and "File > Open" in w.image_label.text())
