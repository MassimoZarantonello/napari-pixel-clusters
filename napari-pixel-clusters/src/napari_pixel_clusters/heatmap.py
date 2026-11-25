"""Heatmap (markers x metaclusters) with the profile of the selected metacluster next to it."""
from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from qtpy.QtCore import QSize, Signal
from qtpy.QtWidgets import QSizePolicy


def zscore(matrix: np.ndarray) -> np.ndarray:
    """z-score every marker (row) across the metaclusters, clipped to +-3 (as in the notebook)."""
    sd = matrix.std(axis=1, keepdims=True)
    return np.clip((matrix - matrix.mean(axis=1, keepdims=True)) / np.where(sd > 0, sd, 1), -3, 3)


class HeatmapCanvas(FigureCanvasQTAgg):
    clicked = Signal(int)  # metacluster id

    def __init__(self):
        self.fig = Figure(figsize=(4.8, 4.6), dpi=100)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._cols: list[int] = []
        self._ax_heat = None
        self.mpl_connect("button_press_event", self._on_press)
        self.show_message("No clusters yet")

    def sizeHint(self):  # do not demand the full figure size: the canvas shrinks to the room it gets
        return QSize(300, 300)

    def minimumSizeHint(self):
        return QSize(200, 240)

    def show_message(self, text: str):
        self.fig.clear()
        self._ax_heat, self._cols = None, []
        self.fig.text(0.5, 0.5, text, ha="center", va="center", fontsize=9, color="#555")
        self.draw_idle()

    def plot(self, clusters, matrix, markers, colors, selected=None, use_zscore=True):
        """clusters: metacluster ids (columns); matrix: (n_markers, n_clusters) mean signal."""
        self.fig.clear()
        self._cols = list(clusters)
        M, K = matrix.shape
        data = zscore(matrix) if use_zscore else matrix
        vmax = 3 if use_zscore else max(float(np.nanmax(data)), 1e-6)
        gs = self.fig.add_gridspec(1, 2, width_ratios=[max(K, 3), 5], wspace=0.06,
                                   left=0.33, right=0.985, top=0.89, bottom=0.07)
        ax = self.fig.add_subplot(gs[0])
        ax.imshow(data, cmap="RdBu_r" if use_zscore else "viridis", vmin=-vmax if use_zscore else 0, vmax=vmax,
                  aspect="auto", interpolation="nearest")
        ax.set_yticks(range(M))
        ax.set_yticklabels(markers, fontsize=6.5)
        ax.set_xticks(range(K))
        ax.set_xticklabels([str(c) for c in clusters], fontsize=7)
        ax.xaxis.tick_top()
        ax.tick_params(length=0, pad=2)
        for j, c in enumerate(clusters):  # colour key = the colour of the cluster in the image
            ax.add_patch(Rectangle((j - 0.5, M - 0.35), 1, 0.9, color=colors[c], clip_on=False, lw=0))
        for sp in ax.spines.values():
            sp.set_visible(False)
        self._ax_heat = ax

        prof = self.fig.add_subplot(gs[1], sharey=ax)
        prof.tick_params(left=False, labelleft=False, labelsize=6.5)
        for sp in ("top", "right"):
            prof.spines[sp].set_visible(False)
        if selected in clusters:
            j = clusters.index(selected)
            ax.add_patch(Rectangle((j - 0.5, -0.5), 1, M, fill=False, ec="black", lw=1.6))
            vals = data[:, j]
            prof.barh(range(M), vals, color=[("#b2182b" if v > 0 else "#2166ac") if use_zscore else colors[selected] for v in vals], height=0.75)
            prof.axvline(0, color="#888", lw=0.6)
            prof.set_title(f"cluster {selected}", fontsize=8, pad=8)
            prof.set_xlim(-vmax, vmax) if use_zscore else prof.set_xlim(0, vmax)
        else:
            prof.set_xticks([])
            prof.text(0.5, 0.5, "click a\ncluster", transform=prof.transAxes, ha="center", va="center", fontsize=8, color="#777")
        self.draw_idle()

    def _on_press(self, event):
        if self._ax_heat is not None and event.inaxes is self._ax_heat and event.xdata is not None:
            j = int(round(event.xdata))
            if 0 <= j < len(self._cols):
                self.clicked.emit(self._cols[j])
