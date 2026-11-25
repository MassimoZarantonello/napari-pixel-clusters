"""Rectangle ROIs on a napari Shapes layer, with the size shown next to the cursor while drawing.

Every ROI gets a stable id (stored in the layer's features), so results computed for an ROI can be kept
when other ROIs are added or deleted.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from qtpy.QtCore import QObject, QPoint, QTimer, Signal
from qtpy.QtGui import QCursor
from qtpy.QtWidgets import QToolTip

LAYER_NAME = "ROIs"
MIN_SIZE = 16  # px; anything smaller is ignored


class RoiInfo(NamedTuple):
    id: int  # stable id, never reused
    index: int  # row in the Shapes layer (changes when other ROIs are deleted)
    box: tuple[int, int, int, int]  # y0, y1, x0, x1 clipped to the image
    clipped: bool  # the drawn rectangle reached outside the image


def size_text(h: int, w: int) -> str:
    return f"{w} x {h} px  ({h * w / 1e6:.2f} Mpx)"


def bbox(data: np.ndarray, image_hw: tuple[int, int]):
    """Bounding box (y0, y1, x0, x1) of a shape clipped to the image, and whether it was clipped.
    None if nothing (or less than MIN_SIZE px) of it is inside the image."""
    h, w = image_hw
    ys, xs = data[:, -2], data[:, -1]
    fy0, fy1, fx0, fx1 = math.floor(ys.min()), math.ceil(ys.max()), math.floor(xs.min()), math.ceil(xs.max())
    y0, y1, x0, x1 = max(0, fy0), min(h, fy1), max(0, fx0), min(w, fx1)
    if y1 - y0 < MIN_SIZE or x1 - x0 < MIN_SIZE:
        return None
    return (y0, y1, x0, x1), (y0, y1, x0, x1) != (fy0, fy1, fx0, fx1)


class RoiTool(QObject):
    changed = Signal()

    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.image_hw: tuple[int, int] = (0, 0)
        self.layer = None
        self._busy = False
        self._next_id = 1

    def reset(self):
        """Forget the layer (call after the viewer layers were cleared)."""
        self.layer = None
        self._next_id = 1

    # -- layer ---------------------------------------------------------------------------
    def _ensure_layer(self):
        if self.layer is not None and self.layer in self.viewer.layers:
            return self.layer
        layer = self.viewer.add_shapes(
            name=LAYER_NAME,
            ndim=2,
            edge_color="yellow",
            face_color="transparent",
            edge_width=10,
            features={"size": np.array([], dtype=object), "roi_id": np.array([], dtype=float)},
            text={"string": "ROI {roi_id:.0f}: {size}", "color": "yellow", "size": 9, "anchor": "upper_left"},
        )
        layer.mouse_drag_callbacks.append(self._on_drag)
        layer.events.data.connect(self._on_data)
        self.layer = layer
        return layer

    def start_drawing(self):
        layer = self._ensure_layer()
        self.viewer.layers.selection.active = layer
        layer.mode = "add_rectangle"

    # -- live size next to the cursor ----------------------------------------------------
    def _clip(self, p):
        h, w = self.image_hw
        return np.clip(p, 0, [h, w]) if h and w else p

    def _on_drag(self, layer, event):
        if layer.mode != "add_rectangle":
            return
        start = self._clip(np.asarray(layer.world_to_data(event.position))[-2:])
        yield
        while event.type == "mouse_move":
            cur = self._clip(np.asarray(layer.world_to_data(event.position))[-2:])
            h, w = np.abs(cur - start)  # clipped to the image, like the ROI that will be used
            QToolTip.showText(QCursor.pos() + QPoint(18, 18), size_text(round(h), round(w)))
            yield
        QToolTip.hideText()
        # back to normal pan/zoom once the rectangle exists (after napari has finished adding it)
        QTimer.singleShot(0, lambda: self._finish_drawing(layer))

    def _finish_drawing(self, layer):
        QToolTip.hideText()
        if layer in self.viewer.layers and layer.mode == "add_rectangle":
            layer.mode = "pan_zoom"

    # -- ids and size labels -------------------------------------------------------------
    def _on_data(self, event=None):
        if self._busy or self.layer is None:
            return
        self._busy = True
        try:
            n = len(self.layer.data)
            feats = self.layer.features
            old = list(feats["roi_id"]) if "roi_id" in feats and len(feats) == n else [float("nan")] * n
            ids, seen = [], set()
            for v in old:
                if v is None or np.isnan(v) or int(v) in seen:  # new shape (napari copies the previous id)
                    v = self._next_id
                v = int(v)
                ids.append(v)
                seen.add(v)
                self._next_id = max(self._next_id, v + 1)
            sizes = []
            for d in self.layer.data:
                b = bbox(np.asarray(d), self.image_hw)
                sizes.append(size_text(b[0][1] - b[0][0], b[0][3] - b[0][2]) + (" (clipped)" if b[1] else "") if b else "too small / outside image")
            self.layer.features = {"size": np.array(sizes, dtype=object), "roi_id": np.array(ids, dtype=float)}
        finally:
            self._busy = False
        self.changed.emit()

    # -- query ---------------------------------------------------------------------------
    def rois(self) -> list[RoiInfo]:
        if self.layer is None or self.layer not in self.viewer.layers:
            return []
        ids = list(self.layer.features["roi_id"]) if "roi_id" in self.layer.features else []
        out = []
        for i, d in enumerate(self.layer.data):
            b = bbox(np.asarray(d), self.image_hw)
            if b is not None and i < len(ids) and not np.isnan(ids[i]):
                out.append(RoiInfo(int(ids[i]), i, b[0], b[1]))
        return out

    def remove(self, index: int):
        if self.layer is not None and 0 <= index < len(self.layer.data):
            self.layer.selected_data = {index}
            self.layer.remove_selected()
