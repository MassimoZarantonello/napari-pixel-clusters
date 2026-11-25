"""Main dock widget with three tabs: ROIs, Clusters (runs + heatmap), Retrain."""
from __future__ import annotations

import time
import traceback

import numpy as np
from napari.qt.threading import thread_worker
from napari.utils.colormaps import DirectLabelColormap
from qtpy.QtCore import QSize, QTimer, Qt, Signal
from qtpy.QtGui import QBrush, QColor, QFont
from qtpy.QtWidgets import (
    QCheckBox, QFrame, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMessageBox, QProgressBar, QPushButton, QScrollArea, QSizePolicy, QSpinBox, QSplitter, QStackedWidget, QTabBar, QTabWidget, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from .clustering import Run, cluster_rois, default_included, metacluster_stats, short_name, train_model
from .heatmap import HeatmapCanvas
from .io import META_KEY, ImageStack, get_stack
from .roi import RoiTool, size_text

MAX_ROI_MPX = 16  # largest ROI (in megapixels) we try on this machine; bigger ones would exhaust the RAM

_TAB20 = [
    (0.122, 0.467, 0.706), (0.682, 0.78, 0.91), (1.0, 0.498, 0.055), (1.0, 0.733, 0.471),
    (0.173, 0.627, 0.173), (0.596, 0.875, 0.541), (0.839, 0.153, 0.157), (1.0, 0.596, 0.588),
    (0.58, 0.404, 0.741), (0.773, 0.69, 0.835), (0.549, 0.337, 0.294), (0.769, 0.612, 0.58),
    (0.89, 0.467, 0.761), (0.969, 0.714, 0.824), (0.498, 0.498, 0.498), (0.78, 0.78, 0.78),
    (0.737, 0.741, 0.133), (0.859, 0.859, 0.553), (0.09, 0.745, 0.812), (0.62, 0.855, 0.898),
]


def cluster_color(c: int):
    return (*_TAB20[c % 20], 1.0)


def centered_checkbox(checked: bool, on_toggle):
    """A checkbox centred in its table cell (returns the cell widget and the checkbox)."""
    box = QCheckBox()
    box.setChecked(checked)
    box.toggled.connect(on_toggle)
    cell = QWidget()
    row = QHBoxLayout(cell)
    row.setContentsMargins(0, 0, 0, 0)
    row.setAlignment(Qt.AlignmentFlag.AlignCenter)
    row.addWidget(box)
    return cell, box


def tab_layout(widget):
    lay = QVBoxLayout(widget)
    lay.setContentsMargins(6, 6, 6, 6)
    lay.setSpacing(4)
    return lay


def scrollable(widget):
    """Wrap a tab so that it scrolls when the dock is short, instead of squeezing its content."""
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.Shape.NoFrame)
    area.setWidget(widget)
    return area


def make_table(headers, widths, min_height=220):
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.verticalHeader().setVisible(False)
    t.verticalHeader().setDefaultSectionSize(24)
    t.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
    t.setShowGrid(False)
    for i, w in enumerate(widths):
        t.setColumnWidth(i, w)
    t.horizontalHeader().setStretchLastSection(True)
    t.setMinimumHeight(min_height)
    return t


class ClickableLabel(QLabel):
    clicked = Signal()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.rect().contains(event.position().toPoint()):
            self.clicked.emit()
        super().mouseReleaseEvent(event)


class RoiRow(QWidget):
    """One line of the ROI list. A small x shows up while the mouse is over the line; clicking it removes the ROI."""

    remove_clicked = Signal()

    def __init__(self, text: str, tooltip: str):
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 0, 2, 0)
        self.label = QLabel(text)
        self.label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)  # long text is cut, not wrapped
        self.x = ClickableLabel("\u00d7")  # a label, not a button: napari colours labels for both light and dark themes
        self.x.setToolTip("Remove this ROI")
        self.x.setCursor(Qt.CursorShape.PointingHandCursor)
        self.x.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.x.setFixedWidth(22)
        font = self.x.font()
        font.setBold(True)
        font.setPointSize(font.pointSize() + 4)
        self.x.setFont(font)
        keep = self.x.sizePolicy()
        keep.setRetainSizeWhenHidden(True)  # the text does not jump when the x appears
        self.x.setSizePolicy(keep)
        self.x.setVisible(False)
        self.x.clicked.connect(self.remove_clicked)
        lay.addWidget(self.label, 1)
        lay.addWidget(self.x)
        self.setToolTip(tooltip)

    def text(self) -> str:
        return self.label.text()

    def enterEvent(self, event):
        self.x.setVisible(True)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.x.setVisible(False)
        super().leaveEvent(event)


class PageTabs:
    """Real tabs along the TOP: a tab bar and the pages it switches."""

    def __init__(self):
        self.bar = QTabBar()
        self.bar.setShape(QTabBar.Shape.RoundedNorth)
        self.bar.setExpanding(False)
        self.bar.setDrawBase(True)
        self.pages = QStackedWidget()
        self.bar.currentChanged.connect(self.pages.setCurrentIndex)

    def addTab(self, widget, name):
        self.pages.addWidget(widget)
        self.bar.addTab(name)

    def setCurrentIndex(self, i):
        self.bar.setCurrentIndex(i)

    def currentIndex(self):
        return self.bar.currentIndex()

    def tabText(self, i):
        return self.bar.tabText(i)

    def count(self):
        return self.bar.count()

    def widget(self, i):
        return self.pages.widget(i)


class RunTabBar(QTabBar):
    """The runs as real tabs along the bottom (like sheet tabs): click = show, x = delete, double-click = rename."""

    rename_requested = Signal(int)

    def __init__(self):
        super().__init__()
        self.setShape(QTabBar.Shape.RoundedSouth)
        self.setTabsClosable(True)
        self.setExpanding(False)
        self.setMovable(False)
        self.setDrawBase(True)
        self.setUsesScrollButtons(True)

    def mouseDoubleClickEvent(self, event):
        i = self.tabAt(event.position().toPoint())
        if i >= 0:
            self.rename_requested.emit(i)
        else:
            super().mouseDoubleClickEvent(event)


NO_IMAGE_HINT = "No image yet: open a MACSima OME-TIFF with File > Open (or drop it on napari)."


class ClusterExplorer(QWidget):
    status_message = Signal(str)

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.stack: ImageStack | None = None
        self.busy = False
        self._skip_note = ""
        self.runs: list[Run] = []  # every training so far; each keeps its own results on the ROIs
        self.run: Run | None = None  # the run that is shown
        self._next_run_id = 1
        self._scheduled = None  # path of an image whose layers just appeared, waiting to be attached
        self._attaching = False
        self.roi_layers: dict[tuple[int, int], object] = {}  # (run id, ROI id) -> ONE labels layer, kept alive
        self.selected_cluster: int | None = None
        self.roi_tool = RoiTool(self.viewer)
        self.roi_tool.changed.connect(self._rois_changed)
        self.viewer.mouse_drag_callbacks.append(self._on_click)
        self.viewer.layers.events.inserted.connect(self._on_inserted)
        self.viewer.layers.events.removed.connect(self._on_removed)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)
        self.status_message.connect(lambda m: self.status.setText(m))

        # ---- always visible: image ----
        self.image_label = QLabel(NO_IMAGE_HINT)
        self.image_label.setWordWrap(True)
        lay.addWidget(self.image_label)

        self.tabs = PageTabs()
        lay.addWidget(self.tabs.bar, 0, Qt.AlignmentFlag.AlignLeft)  # ROIs | Clusters | Retrain, packed to the left
        lay.addWidget(self.tabs.pages, 1)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # busy indicator: no percentage, FlowSOM does not report one
        self.progress.setTextVisible(False)
        self.progress.setMaximumHeight(8)
        self.progress.setVisible(False)
        lay.addWidget(self.progress)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        self.run_bar = RunTabBar()  # one tab per run, along the bottom: the shown run affects every page
        self.run_bar.currentChanged.connect(self._run_tab_changed)
        self.run_bar.tabCloseRequested.connect(self._confirm_delete_run)
        self.run_bar.rename_requested.connect(self._rename_run)
        self.run_bar.setVisible(False)
        lay.addWidget(self.run_bar, 0, Qt.AlignmentFlag.AlignLeft)  # packed to the left, not centred

        # ---- tab 1: ROIs + the channels to look at ----
        t = QWidget()
        tl = tab_layout(t)
        self.draw_btn = QPushButton("Draw ROI (drag a rectangle)")
        self.draw_btn.clicked.connect(self.roi_tool.start_drawing)
        self.roi_list = QListWidget()
        self.roi_list.setMaximumHeight(96)
        self.roi_list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self.roi_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.roi_list.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.run_btn = QPushButton("Cluster new ROI(s)")
        self.run_btn.clicked.connect(self._run)
        self.run_label = QLabel("No run yet")  # only shown while there is no run (loading / training)
        self.run_label.setWordWrap(True)
        for w in (self.draw_btn, self.roi_list, self.run_btn, self.run_label):
            tl.addWidget(w)  # clustering sits right under the ROI selection
        hint = QLabel("Markers: Show = look at the channel. Grey = not used by the shown run.")
        hint.setWordWrap(True)
        tl.addWidget(hint)
        row = QHBoxLayout()
        self.show_search = QLineEdit()
        self.show_search.setPlaceholderText("search marker...")
        self.show_search.textChanged.connect(lambda text: self._filter_rows(self.table, text))
        hide_btn = QPushButton("Hide all")
        hide_btn.setToolTip("Hide every channel in the viewer")
        hide_btn.clicked.connect(lambda: [b.setChecked(False) for b in self.show_boxes])
        row.addWidget(self.show_search, 1)
        row.addWidget(hide_btn)
        tl.addLayout(row)
        self.table = make_table(["Show", "Marker"], [45])
        self.show_boxes: list[QCheckBox] = []
        tl.addWidget(self.table)
        self.tabs.addTab(scrollable(t), "ROIs")

        # ---- tab 2: clusters ----
        t = QWidget()
        tl = tab_layout(t)
        self.run_info = QLabel("")
        self.run_info.setWordWrap(True)
        tl.addWidget(self.run_info)
        row = QHBoxLayout()
        row.addWidget(QLabel("Metaclusters:"))
        self.k_spin = QSpinBox()
        self.k_spin.setRange(2, 40)
        self.k_spin.setValue(10)
        self.k_spin.setKeyboardTracking(False)  # typing a number applies on Enter, not on every digit
        self.k_spin.valueChanged.connect(self._k_edited)
        self._k_timer = QTimer(self)
        self._k_timer.setSingleShot(True)
        self._k_timer.setInterval(800)
        self._k_timer.timeout.connect(lambda: self._k_changed(self.k_spin.value()))
        row.addWidget(self.k_spin)
        row.addStretch()
        # master switch: hides every cluster without touching the individual choices below
        self.master_box = QCheckBox("Show clusters")
        self.master_box.setToolTip("On/off for all clusters. Your individual ticks below are kept.")
        self.master_box.setChecked(True)
        self.master_box.toggled.connect(self._apply_visibility)
        row.addWidget(self.master_box)
        tl.addLayout(row)
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        tl.addWidget(line)
        self.mc_table = make_table(["Show", "", "Name", "% area"], [45, 24, 110], min_height=110)
        self.mc_table.itemChanged.connect(self._mc_item_changed)
        self.mc_table.cellClicked.connect(self._mc_cell_clicked)
        self.mc_boxes: dict[int, QCheckBox] = {}
        self.z_box = QCheckBox("z-score per marker (uncheck: mean signal)")
        self.z_box.setChecked(True)
        self.z_box.toggled.connect(lambda _: self._refresh_heatmap())
        self.heatmap = HeatmapCanvas()
        self.heatmap.clicked.connect(self._select_cluster)
        bottom = QWidget()
        bl = QVBoxLayout(bottom)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(2)
        bl.addWidget(self.z_box)
        bl.addWidget(self.heatmap, 1)
        split = QSplitter(Qt.Orientation.Vertical)  # drag the handle to give more room to the list or the heatmap
        split.addWidget(self.mc_table)
        split.addWidget(bottom)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 3)
        split.setSizes([120, 340])
        split.setChildrenCollapsible(False)
        tl.addWidget(split, 1)
        t.setMinimumHeight(520)  # below this the tab scrolls
        self.tabs.addTab(scrollable(t), "Clusters")

        # ---- tab 3: retrain FlowSOM with other markers = a NEW run ----
        t = QWidget()
        tl = tab_layout(t)
        info = QLabel("After inspecting the clusters: untick the markers that did not work, then train a new run. "
                      "Earlier runs are kept.")
        info.setWordWrap(True)
        tl.addWidget(info)
        row = QHBoxLayout()
        self.use_search = QLineEdit()
        self.use_search.setPlaceholderText("search marker...")
        self.use_search.textChanged.connect(lambda text: self._filter_rows(self.use_table, text))
        row.addWidget(self.use_search, 1)
        for label, fn, tip in (("All", lambda n: True, "tick the markers shown by the search"),
                               ("None", lambda n: False, "untick the markers shown by the search"),
                               ("Default", lambda n: n in default_included(self.stack.channel_names),
                                "DAPI and empty-cycle channels off, the rest on")):
            b = QPushButton(label)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, fn=fn: self._set_use(fn))
            row.addWidget(b)
        tl.addLayout(row)
        self.use_table = make_table(["Use", "Marker"], [40])
        self.use_boxes: list[QCheckBox] = []
        self.retrain_label = QLabel("")
        self.retrain_label.setWordWrap(True)
        self.train_btn = QPushButton("Train a new run with these markers")
        self.train_btn.setToolTip("Samples tiles across the whole slide with the ticked markers, then clusters the ROIs")
        self.train_btn.clicked.connect(self._train)
        for w in (self.use_table, self.retrain_label, self.train_btn):
            tl.addWidget(w)
        self.tabs.addTab(scrollable(t), "Retrain")
        self._update_buttons()
        for layer in self.viewer.layers:  # the image may have been opened before this panel was
            if layer.metadata.get(META_KEY):
                self._schedule_attach(layer.metadata[META_KEY])
                break

    # ---- image ------------------------------------------------------------------------
    def load_image(self, path):
        """Open an image through napari's own reader system (same as File > Open); the panel then attaches to it."""
        self.viewer.open(str(path), plugin="napari-pixel-clusters")

    def _on_inserted(self, event):
        layer = event.value
        path = getattr(layer, "metadata", {}).get(META_KEY)
        if path and not self._attaching:
            self._schedule_attach(path)

    def _schedule_attach(self, path):
        if self._scheduled == path:
            return  # the reader adds one layer per channel: attach once, after all of them are in
        self._scheduled = path
        QTimer.singleShot(0, lambda: self._attach(path))

    def _on_removed(self, event):
        path = getattr(event.value, "metadata", {}).get(META_KEY)
        if path and not self._attaching and not any(l.metadata.get(META_KEY) == path for l in self.viewer.layers):
            self._detach()  # the image's channel layers were all removed

    def _detach(self):
        self._clear_layers()
        self.roi_tool.reset()
        self.stack, self.runs, self.run, self._next_run_id = None, [], None, 1
        for t in (self.table, self.use_table):
            t.setRowCount(0)
        self.show_boxes, self.use_boxes = [], []
        self.image_label.setText(NO_IMAGE_HINT)
        self._activate(None)

    def _attach(self, path):
        """Start working on the image the reader opened: marker lists, ROI tool, and Run 1 trained once."""
        self._scheduled = None
        try:
            stack = get_stack(path)
        except Exception as e:  # noqa: BLE001
            self.image_label.setText(f"Could not open image: {e}")
            return
        self._attaching = True
        try:
            for layer in list(self.viewer.layers):  # keep only the channel layers of this image
                if layer.metadata.get(META_KEY) != path:
                    self.viewer.layers.remove(layer)
            self._clear_layers()
            self.roi_tool.reset()
            self.runs, self.run, self._next_run_id = [], None, 1
            self.stack = stack
            channel_layers = [l for l in self.viewer.layers if l.metadata.get(META_KEY) == path]
            for lyr in channel_layers:
                lyr.events.visible.connect(self._sync_show_boxes)
        finally:
            self._attaching = False
        c, h, w = stack.shape
        self.roi_tool.image_hw = (h, w)
        self.image_label.setText(f"{stack.path.name}: {c} channels, {w} x {h} px")
        self._fill_tables()
        self.viewer.reset_view()
        self._activate(None)
        self._train()  # Run 1: once per image, tiles sampled over the whole slide

    # ---- marker lists -----------------------------------------------------------------
    def _fill_tables(self):
        names = self.stack.channel_names
        use = set(default_included(names))
        for t in (self.table, self.use_table):
            t.setRowCount(len(names))
        self.show_boxes, self.use_boxes = [], []
        for r, n in enumerate(names):
            cell, box = centered_checkbox(False, lambda on, n=n: self._set_channel_visible(n, on))
            self.table.setCellWidget(r, 0, cell)
            self.show_boxes.append(box)
            cell, box = centered_checkbox(n in use, lambda on: self._update_buttons())
            self.use_table.setCellWidget(r, 0, cell)
            self.use_boxes.append(box)
            for t in (self.table, self.use_table):
                label = QTableWidgetItem(n)
                label.setFlags(Qt.ItemFlag.ItemIsEnabled)  # marker names are not editable
                t.setItem(r, 1, label)
        self._sync_show_boxes()

    @staticmethod
    def _filter_rows(table, text):
        text = text.strip().lower()
        for r in range(table.rowCount()):
            table.setRowHidden(r, bool(text) and text not in table.item(r, 1).text().lower())

    def _set_use(self, wanted):
        """Tick/untick the 'Use' boxes of the rows the search currently shows."""
        if self.stack is None:
            return
        for r, (box, n) in enumerate(zip(self.use_boxes, self.stack.channel_names)):
            if not self.use_table.isRowHidden(r):
                box.blockSignals(True)
                box.setChecked(bool(wanted(n)))
                box.blockSignals(False)
        self._update_buttons()

    def _set_channel_visible(self, name, on):
        if name in self.viewer.layers:
            self.viewer.layers[name].visible = on

    def _sync_show_boxes(self, *_):
        if self.stack is None:
            return
        for box, n in zip(self.show_boxes, self.stack.channel_names):
            box.blockSignals(True)
            box.setChecked(n in self.viewer.layers and self.viewer.layers[n].visible)
            box.blockSignals(False)

    def selected_markers(self) -> list[str]:
        return [n for n, b in zip(self.stack.channel_names, self.use_boxes) if b.isChecked()]

    def _grey_markers(self):
        """The list always shows every marker; the ones the shown run did not use are greyed out."""
        if self.stack is None:
            return
        used = set(self.run.model.channels) if self.run else set(self.stack.channel_names)
        for r, n in enumerate(self.stack.channel_names):
            for t in (self.table, self.use_table):
                item = t.item(r, 1)
                item.setForeground(QBrush(QColor("#888888")) if n not in used else QBrush())
                item.setToolTip("" if n in used else f"Not used by {self.run.name}")

    # ---- ROIs tab ---------------------------------------------------------------------
    def _rois_changed(self):
        infos = self.roi_tool.rois()
        current = {i.id: i.box for i in infos}
        # forget the results of ROIs that were deleted or edited, in every run
        for run in self.runs:
            stale = [rid for rid, res in run.results.items() if current.get(rid) != res.roi]
            for rid in stale:
                del run.results[rid]
                self._remove_layer((run.id, rid))
            if stale and run is self.run:
                self._fill_mc_table()
                self._refresh_heatmap()
        self.roi_list.clear()
        done = self.run.results if self.run else {}
        for info in infos:
            y0, y1, x0, x1 = info.box
            tip = f"ROI {info.id}: {size_text(y1 - y0, x1 - x0)} at x={x0}, y={y0}"
            text = f"ROI {info.id} · {x1 - x0} x {y1 - y0} px · x={x0}, y={y0}"
            if info.clipped:
                text += " · clipped"
                tip += " (clipped to the image)"
            if info.id in done:
                text += " · clustered"
            row = RoiRow(text, tip)
            row.remove_clicked.connect(lambda index=info.index: self._remove_roi_later(index))
            item = QListWidgetItem()
            item.setSizeHint(QSize(100, 26))
            self.roi_list.addItem(item)
            self.roi_list.setItemWidget(item, row)
        self._update_buttons()

    def _remove_roi_later(self, index: int):
        # not right away: removing an ROI rebuilds this list, including the x button that was just clicked
        QTimer.singleShot(0, lambda: self.roi_tool.remove(index))

    # ---- runs -------------------------------------------------------------------------
    def _run_title(self, run: Run) -> str:
        return f"{run.name} · {len(run.model.channels)} markers"

    def _describe(self, run: Run) -> str:
        total = len(self.stack.channel_names) if self.stack else 0
        left = [short_name(n) for n in self.stack.channel_names if n not in run.model.channels] if self.stack else []
        shown = ", ".join(left[:6]) + (f" +{len(left) - 6} more" if len(left) > 6 else "")
        return f"{len(run.model.channels)} of {total} markers" + (f" · not used: {shown}" if left else "")

    def _refresh_run_bar(self):
        self.run_bar.blockSignals(True)
        while self.run_bar.count():
            self.run_bar.removeTab(0)
        for r in self.runs:
            i = self.run_bar.addTab(r.name)
            self.run_bar.setTabToolTip(i, f"{self._describe(r)}\nDouble-click to rename, x to delete")
        if self.run in self.runs:
            self.run_bar.setCurrentIndex(self.runs.index(self.run))
        self.run_bar.blockSignals(False)
        self.run_bar.setVisible(bool(self.runs))

    def _activate(self, run: Run | None):
        """Show a run: its clusters in the viewer, its markers, its heatmap. Nothing is recomputed."""
        self.run = run
        self.selected_cluster = None
        if run is not None:
            self.k_spin.blockSignals(True)
            self.k_spin.setValue(run.k)
            self.k_spin.blockSignals(False)
            if self.stack is not None:  # the 'Use' ticks start from this run's markers ("retrain from here")
                for box, n in zip(self.use_boxes, self.stack.channel_names):
                    box.blockSignals(True)
                    box.setChecked(n in run.model.channels)
                    box.blockSignals(False)
        self._refresh_run_bar()
        self._grey_markers()
        self._fill_mc_table()
        self._refresh_heatmap()
        self._rois_changed()

    def _run_tab_changed(self, i):
        if 0 <= i < len(self.runs) and self.runs[i] is not self.run and not self.busy:
            self._activate(self.runs[i])

    def _rename_run(self, index: int):
        if self.busy or not 0 <= index < len(self.runs):
            return
        run = self.runs[index]
        name, ok = QInputDialog.getText(self, "Rename run", "Name of this run:", text=run.name)
        if ok:
            self._set_run_name(name, run)

    def _set_run_name(self, name: str, run: Run | None = None):
        run = run or self.run
        if run is not None and name.strip():
            run.name = name.strip()
            for (run_id, rid), layer in self.roi_layers.items():
                if run_id == run.id:
                    layer.name = self._layer_name(run, rid)
            self._refresh_run_bar()
            self._grey_markers()
            self._update_buttons()

    def _confirm_delete_run(self, index: int):
        if self.busy or not 0 <= index < len(self.runs):
            return
        run = self.runs[index]
        n = len(run.results)
        answer = QMessageBox.question(
            self, "Delete run", f"Delete {run.name}? Its clusters and names for {n} ROI(s) are lost "
            f"(training it again takes about 20 s).")
        if answer == QMessageBox.StandardButton.Yes:
            self._delete_run(run)

    def _delete_run(self, run: Run | None = None):
        run = run or self.run
        if run is None or self.busy:
            return
        i = self.runs.index(run)
        self.runs.remove(run)
        for key in [k for k in self.roi_layers if k[0] == run.id]:
            self._remove_layer(key)
        if run is self.run:
            self._activate(self.runs[max(0, i - 1)] if self.runs else None)
        else:  # another run was deleted: the shown one stays
            self._refresh_run_bar()
            self._update_buttons()

    # ---- buttons / labels -------------------------------------------------------------
    def _todo_rois(self):
        if self.run is None:
            return []
        return [i for i in self.roi_tool.rois() if i.id not in self.run.results]

    def _update_buttons(self):
        self.progress.setVisible(self.busy)
        have_image = self.stack is not None
        ticked = self.selected_markers() if have_image and self.use_boxes else []
        self.train_btn.setEnabled(have_image and not self.busy and len(ticked) >= 2)
        self.run_btn.setEnabled(self.run is not None and not self.busy and bool(self._todo_rois()))
        self.run_bar.setEnabled(not self.busy)
        self.run_label.setVisible(self.run is None)
        if not have_image:
            self.run_label.setText("Load an image first")
            self.run_info.setText("")
            self.retrain_label.setText("")
            return
        if self.run is None:
            self.run_label.setText("Training on the whole image..." if self.busy else "No run yet")
            self.run_info.setText("")
        else:
            self.run_label.setText("")
            self.run_info.setText(self._describe(self.run))
            self.run_info.setToolTip(f"Trained on {self.run.model.n_train_pixels:,} pixels sampled over the whole image")
        same = next((r for r in self.runs if r.model.channels == ticked), None)
        if same is not None:
            self.retrain_label.setText(f"{len(ticked)} markers ticked: same as {same.name} (no new run needed).")
        else:
            self.retrain_label.setText(f"{len(ticked)} markers ticked. This will be run {self._next_run_id}.")

    # ---- training / clustering in a background thread ---------------------------------
    def _run_in_thread(self, fn, on_done):
        self.busy = True
        self._update_buttons()
        t0 = time.time()

        @thread_worker
        def work():
            return fn(self.status_message.emit)

        worker = work()
        worker.returned.connect(lambda res: self._finished(on_done, res, time.time() - t0))
        worker.errored.connect(self._failed)
        worker.start()

    def _finished(self, on_done, res, seconds):
        self.busy = False
        note, self._skip_note = self._skip_note, ""  # the note belongs to the run that just ended
        on_done(res)  # may start the next step (e.g. clustering the ROIs with the new run)
        if not self.busy:
            self.status.setText(f"Done in {seconds:.0f} s." + note)
        self._update_buttons()

    def _failed(self, exc):
        self.busy = False
        self.status.setText(f"Failed: {exc}")
        traceback.print_exception(exc)
        self._update_buttons()

    def _train(self):
        markers = self.selected_markers()
        if len(markers) < 2:
            self.status.setText("Tick at least two markers in 'Use'.")
            return
        same = next((r for r in self.runs if r.model.channels == markers), None)
        if same is not None:  # do not waste 20 s on a training that already exists
            self._activate(same)
            self.status.setText(f"{same.name} already uses these markers: showing it.")
            return
        stack = self.stack
        cluster_after = bool(self.run and self.run.results)  # the ROIs the user was looking at get this run too

        def new_run(model):
            run = Run(self._next_run_id, model, name=f"Run {self._next_run_id}", k=self.k_spin.value())
            self._next_run_id += 1
            self.runs.append(run)
            self._activate(run)
            if cluster_after:
                self._run()

        self._run_in_thread(lambda report: train_model([stack], markers, report=report), new_run)

    def _run(self):
        run = self.run
        todo = self._todo_rois()  # only ROIs that have no result in this run: finished ones are not recomputed
        if run is None or not todo:
            return
        too_big = [i for i in todo if (i.box[1] - i.box[0]) * (i.box[3] - i.box[2]) > MAX_ROI_MPX * 1e6]
        todo = [i for i in todo if i not in too_big]  # the other ROIs are still clustered
        self._skip_note = (f" Skipped ROI {', '.join(str(i.id) for i in too_big)}: larger than {MAX_ROI_MPX} Mpx, "
                           f"too much memory for this computer.") if too_big else ""
        if not todo:
            self.status.setText(self._skip_note.strip())
            return
        stack, model = self.stack, run.model

        def show(results):
            live = {i.id for i in self.roi_tool.rois()}
            for info, res in zip(todo, results):
                if info.id in live:  # not deleted meanwhile
                    run.results[info.id] = res
            if run is self.run:
                self._fill_mc_table()
                self._refresh_heatmap()
                self._rois_changed()
                self.tabs.setCurrentIndex(1)

        self._run_in_thread(lambda report: cluster_rois(model, stack, [i.box for i in todo], report), show)

    # ---- cluster layers (of the shown run) --------------------------------------------
    def _clear_layers(self):
        for key in list(self.roi_layers):
            self._remove_layer(key)

    def _remove_layer(self, key):
        layer = self.roi_layers.pop(key, None)
        if layer is not None and layer in self.viewer.layers:
            self.viewer.layers.remove(layer)

    def _layer_name(self, run: Run, rid: int) -> str:
        return f"{run.name} · ROI {rid}"

    def _labels(self, run: Run, rid: int) -> np.ndarray:
        """Metacluster of every pixel of the ROI (0 = background / masked). Just a lookup, no clustering."""
        return run.ensure_lookup()[run.results[rid].som]

    def _colormap(self, run: Run) -> DirectLabelColormap:
        """Every cluster in its colour, except the ones the user switched off (transparent)."""
        colors = {None: "transparent", 0: "transparent"}
        for c in range(1, run.k + 1):
            colors[c] = cluster_color(c) if run.mc_selected.get(c, True) else "transparent"
        return DirectLabelColormap(color_dict=colors)

    def _ensure_layer(self, run: Run, rid: int):
        """One labels layer per run and ROI. Created once, then only shown/hidden when switching runs."""
        key = (run.id, rid)
        if key in self.roi_layers:
            return self.roi_layers[key]
        y0, _, x0, _ = run.results[rid].roi
        layer = self.viewer.add_labels(
            self._labels(run, rid), name=self._layer_name(run, rid), colormap=self._colormap(run),
            opacity=1.0, translate=(y0, x0), visible=False,
        )
        self.roi_layers[key] = layer
        return layer

    def _show_run_layers(self):
        """Show the layers of the shown run (if the master switch is on) and hide those of the other runs."""
        run = self.run
        if run is not None:
            for rid in run.results:
                self._ensure_layer(run, rid)
        master = self.master_box.isChecked()
        for (run_id, rid), layer in self.roi_layers.items():
            want = master and run is not None and run_id == run.id
            if layer.visible != want:
                layer.visible = want

    def _refresh_layer_colors(self):
        run = self.run
        if run is None:
            return
        cmap = self._colormap(run)
        for (run_id, rid), layer in self.roi_layers.items():
            if run_id == run.id:
                layer.colormap = cmap

    def _k_edited(self, k):
        """Wait until the user stopped changing the number before regrouping (it redraws all clusters)."""
        if self.run is not None and self.run.results:
            self.status.setText(f"Regrouping into {k} metaclusters...")
        self._k_timer.start()

    def _k_changed(self, k):
        """Regroup the SOM nodes (no retraining): every ROI is redrawn with the new grouping."""
        run = self.run
        if run is None:
            return
        run.switch_k(k)
        self.selected_cluster = None
        for (run_id, rid), layer in self.roi_layers.items():
            if run_id == run.id:
                layer.data = self._labels(run, rid)
                layer.colormap = self._colormap(run)
        self._fill_mc_table()
        self._refresh_heatmap()
        if run.results:
            self.status.setText(f"{k} metaclusters.")

    # ---- clusters tab: list, visibility, selection ------------------------------------
    def _stats(self):
        if self.run is None or not self.run.results:
            return {}
        return metacluster_stats(list(self.run.results.values()), self.run.ensure_lookup())

    def _fill_mc_table(self):
        stats = self._stats()
        counts = {c: n for c, (_, n) in stats.items()}
        total = sum(counts.values()) or 1
        names = self.run.mc_names if self.run else {}
        selected = self.run.mc_selected if self.run else {}
        self.mc_table.blockSignals(True)
        self.mc_table.clearContents()
        self.mc_table.setRowCount(0)  # drops the old cell widgets before new ones are made
        self.mc_table.setRowCount(len(counts))
        self.mc_boxes = {}
        for r, c in enumerate(sorted(counts)):
            cell, box = centered_checkbox(selected.get(c, True), lambda on, c=c: self._set_cluster_visible(c, on))
            self.mc_table.setCellWidget(r, 0, cell)
            self.mc_boxes[c] = box
            swatch = QTableWidgetItem()
            swatch.setBackground(QColor.fromRgbF(*cluster_color(c)))
            swatch.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.mc_table.setItem(r, 1, swatch)
            name = QTableWidgetItem(names.get(c) or f"cluster {c}")
            name.setData(Qt.ItemDataRole.UserRole, c)
            self.mc_table.setItem(r, 2, name)  # the only editable cell: name the metacluster
            share = QTableWidgetItem(f"{100 * counts[c] / total:.1f}")
            share.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.mc_table.setItem(r, 3, share)
        self.mc_table.blockSignals(False)
        self._mark_selected_row()
        self._apply_visibility()

    def _mark_selected_row(self):
        for r in range(self.mc_table.rowCount()):
            name = self.mc_table.item(r, 2)
            on = name.data(Qt.ItemDataRole.UserRole) == self.selected_cluster
            font = QFont(name.font())
            font.setBold(on)
            for col in (2, 3):
                self.mc_table.item(r, col).setFont(font)

    def _apply_visibility(self, *_):
        """Layers of the shown run are visible if the master switch is on; the individual choices only change
        colours (a switched-off cluster is transparent) and are never overwritten by the master switch."""
        self._show_run_layers()
        self._refresh_layer_colors()
        for box in self.mc_boxes.values():
            box.setEnabled(self.master_box.isChecked())  # greyed out, but keeps its state, while the master is off

    def _set_cluster_visible(self, c, on):
        self.run.mc_selected[c] = on
        self._refresh_layer_colors()

    def _mc_item_changed(self, item):
        c = item.data(Qt.ItemDataRole.UserRole)
        if c is None or item.column() != 2 or self.run is None:
            return
        text = item.text().strip()
        if text and text != f"cluster {c}":
            self.run.mc_names[c] = text
        else:
            self.run.mc_names.pop(c, None)  # back to the default name

    def _mc_cell_clicked(self, row, col):
        item = self.mc_table.item(row, 2)
        if item is not None and col != 0:
            self._select_cluster(item.data(Qt.ItemDataRole.UserRole))

    # ---- heatmap + profile, and click in the image ------------------------------------
    def _select_cluster(self, c):
        self.selected_cluster = c
        self._mark_selected_row()
        self._refresh_heatmap()

    def _refresh_heatmap(self):
        stats = self._stats()
        if not stats:
            self.heatmap.show_message("No clusters yet")
            return
        clusters = sorted(stats)
        matrix = np.stack([stats[c][0] for c in clusters], axis=1)
        labels = [short_name(n) for n in self.run.model.channels]
        colors = {c: cluster_color(c)[:3] for c in clusters}
        self.heatmap.plot(clusters, matrix, labels, colors, self.selected_cluster, self.z_box.isChecked())

    def _on_click(self, viewer, event):
        """A plain click (no drag) on a clustered pixel shows that metacluster's profile."""
        if self.run is None or not self.run.results:
            return
        layer = self.roi_tool.layer
        if layer is not None and layer in self.viewer.layers and layer.mode != "pan_zoom":
            return  # drawing / editing ROIs
        start = np.asarray(event.pos, dtype=float)
        yield
        while event.type == "mouse_move":
            yield
        if np.linalg.norm(np.asarray(event.pos, dtype=float) - start) < 4:
            self._select_at(event.position)

    def _select_at(self, position):
        y, x = (int(np.floor(v)) for v in np.asarray(position)[-2:])
        lookup = self.run.ensure_lookup()
        for res in self.run.results.values():
            y0, y1, x0, x1 = res.roi
            if y0 <= y < y1 and x0 <= x < x1:
                c = int(lookup[res.som[y - y0, x - x0]])
                if c:
                    self._select_cluster(c)
                    name = self.run.mc_names.get(c) or f"cluster {c}"
                    self.status.setText(f"Selected {name}")
                return
