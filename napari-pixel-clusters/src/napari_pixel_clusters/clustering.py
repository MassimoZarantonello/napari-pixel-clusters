"""FlowSOM pixel clustering: train ONCE on tiles sampled from the whole image(s), then label any ROI
with the trained model (no napari imports).

The preprocessing mirrors ``harpy.im.pixel_clustering_preprocess`` (per-channel percentile scaling, background
mask from the channel sum, gaussian blur, per-pixel sum normalisation, post percentile scaling) but the
normalisation constants are fitted once on the training tiles and then reused for every ROI. Otherwise an ROI
would be normalised against itself and would not match the clusters the model learned.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.ndimage import gaussian_filter

Roi = tuple[int, int, int, int]  # y0, y1, x0, x1 (full-resolution pixels, end exclusive)

# DAPI and the empty-cycle channels (FITC C0, PE C0, ...) are not useful for clustering
_NOT_A_MARKER = re.compile(r"^DAPI|\bC0\)")


def default_included(channel_names: list[str]) -> list[str]:
    return [c for c in channel_names if not _NOT_A_MARKER.search(c)]


def short_name(channel: str) -> str:
    """'F4_80 REA126 (PE C2)' -> 'F4_80 REA126' (antibody without the fluorophore/cycle)."""
    return re.sub(r"\s*\([^)]*\)\s*$", "", channel) or channel


def _percentile(values: np.ndarray, q: float) -> float:
    """Percentile of the non-zero, non-NaN values (1.0 if there are none)."""
    v = values[(values > 0) & ~np.isnan(values)]
    return float(np.percentile(v, q)) if v.size else 1.0


@dataclass
class PixelModel:
    """Normalisation constants + trained FlowSOM. Fit once, apply to any region."""

    channels: list[str]
    p_chan: np.ndarray  # (C,) percentile used to scale every channel
    p_sum: float  # pixels whose (blurred) channel sum is below this are background -> masked
    p_post: np.ndarray  # (C,) percentile used after the per-pixel sum normalisation
    fsom: object
    sigma: float = 2.0
    n_train_pixels: int = 0

    # ---- preprocessing ----------------------------------------------------------------
    @staticmethod
    def _scale(raw: np.ndarray, p_chan: np.ndarray) -> np.ndarray:
        return raw.astype(np.float32) / p_chan[:, None, None].astype(np.float32)

    @staticmethod
    def _blur_mask_norm(a: np.ndarray, p_sum: float, sigma: float) -> np.ndarray:
        if sigma:
            for c in range(a.shape[0]):
                a[c] = gaussian_filter(a[c], sigma=sigma)
        total = a.sum(axis=0)
        a[:, ~(total > p_sum)] = np.nan  # background
        s = np.nansum(a, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            a /= np.where(s > 0, s, np.nan)
        return a

    def preprocess(self, raw: np.ndarray) -> np.ndarray:
        """(C, h, w) raw counts -> (C, h, w) float32 FlowSOM input, NaN = masked background."""
        a = self._blur_mask_norm(self._scale(raw, self.p_chan), self.p_sum, self.sigma)
        a /= self.p_post[:, None, None].astype(np.float32)
        return a

    def metacluster_lookup(self, k: int) -> np.ndarray:
        """SOM node (1..100) -> metacluster (1..k); index 0 (masked) stays 0. Computed once per k,
        so that every ROI is grouped identically."""
        self.fsom.model.set_n_clusters(k)
        meta_of_node = self.fsom.model.metacluster_model.fit_predict(self.fsom.model.codes)
        return np.concatenate([[0], meta_of_node + 1]).astype(np.uint16)

    # ---- inference --------------------------------------------------------------------
    def predict_som(self, pre: np.ndarray) -> np.ndarray:
        """(C, h, w) preprocessed -> (h, w) SOM node of every pixel, 0 = masked, 1..100."""
        valid = ~np.isnan(pre).any(axis=0)
        out = np.zeros(pre.shape[1:], dtype=np.int64)
        if valid.any():
            values = pre[:, valid].T
            out[valid] = self.fsom.model.cluster_model.predict(values).astype(np.int64) + 1
        return out


def train_model(
    stacks: list,
    channels: list[str],
    n_clusters: int = 10,
    n_tiles: int = 48,
    tile: int = 256,
    max_pixels: int = 500_000,
    seed: int = 100,
    sigma: float = 2.0,
    report: Callable[[str], None] = lambda msg: None,
) -> PixelModel:
    """Fit the normalisation and train FlowSOM on random tiles taken from the WHOLE image(s)."""
    import flowsom as fs
    import pandas as pd
    from anndata import AnnData

    channels = list(channels)
    rng = np.random.default_rng(seed)

    report(f"Sampling {n_tiles * len(stacks)} tiles of {tile}x{tile} px across the whole image...")
    raw_tiles = []
    for stack in stacks:
        _, H, W = stack.shape
        for _ in range(n_tiles):
            y0 = int(rng.integers(0, max(1, H - tile)))
            x0 = int(rng.integers(0, max(1, W - tile)))
            raw_tiles.append(stack.read_region(y0, min(H, y0 + tile), x0, min(W, x0 + tile), channels))

    report("Fitting normalisation...")
    p_chan = np.array(
        [_percentile(np.concatenate([t[c].ravel() for t in raw_tiles]), 99) for c in range(len(channels))]
    )
    tiles = [PixelModel._scale(t, p_chan) for t in raw_tiles]
    del raw_tiles
    p_sum = _percentile(np.concatenate([t.sum(axis=0).ravel() for t in tiles]), 5)
    tiles = [PixelModel._blur_mask_norm(t, p_sum, sigma) for t in tiles]
    p_post = np.array(
        [_percentile(np.concatenate([t[c].ravel() for t in tiles]), 99.9) for c in range(len(channels))]
    )

    rows = []
    for t in tiles:
        t /= p_post[:, None, None].astype(np.float32)
        valid = ~np.isnan(t).any(axis=0)
        rows.append(t[:, valid].T)
    del tiles
    X = np.concatenate(rows)
    if len(X) > max_pixels:
        X = X[rng.choice(len(X), size=max_pixels, replace=False)]

    report(f"Training FlowSOM on {len(X):,} pixels...")
    var = pd.DataFrame(index=pd.Index(channels, name="channels"))
    fsom = fs.FlowSOM(AnnData(X=X, var=var), n_clusters=n_clusters, cols_to_use=None, seed=seed, xdim=10, ydim=10)
    return PixelModel(channels, p_chan, p_sum, p_post, fsom, sigma, len(X))


N_NODES = 100


@dataclass
class RoiClusters:
    """Result for one ROI. Light on purpose: the label image plus per-SOM-node signal sums, which is all the
    heatmap needs for any number of metaclusters (the preprocessed image itself is not kept)."""

    roi: Roi
    som: np.ndarray  # (h, w) uint16: 0 = masked, 1..100 = SOM node
    node_sum: np.ndarray  # (101, C): summed preprocessed signal of the pixels of every SOM node
    node_count: np.ndarray  # (101,): number of pixels per SOM node


def cluster_rois(
    model: PixelModel,
    stack,
    rois: list[Roi],
    report: Callable[[str], None] = lambda msg: None,
) -> list[RoiClusters]:
    """Label every pixel of every ROI with the already trained model (no training)."""
    out = []
    for i, roi in enumerate(rois):
        report(f"ROI {i + 1}/{len(rois)}: reading and preprocessing...")
        pre = model.preprocess(stack.read_region(*roi, model.channels))
        report(f"ROI {i + 1}/{len(rois)}: assigning clusters...")
        som = model.predict_som(pre)
        valid = som > 0
        nodes = som[valid]
        node_sum = np.stack([np.bincount(nodes, weights=pre[c][valid], minlength=N_NODES + 1) for c in range(pre.shape[0])], axis=1)
        out.append(RoiClusters(roi, som.astype(np.uint16), node_sum, np.bincount(nodes, minlength=N_NODES + 1)))
        del pre
    return out


def metacluster_stats(results: list[RoiClusters], lookup: np.ndarray) -> dict[int, tuple[np.ndarray, int]]:
    """metacluster -> (mean preprocessed signal per marker, number of pixels), over all given ROIs."""
    if not results:
        return {}
    total_sum = sum(r.node_sum for r in results)
    total_n = sum(r.node_count for r in results)
    out = {}
    for c in (int(v) for v in np.unique(lookup) if v != 0):
        nodes = np.flatnonzero(lookup == c)
        n = int(total_n[nodes].sum())
        if n:
            out[c] = (total_sum[nodes].sum(axis=0) / n, n)
    return out


@dataclass
class Run:
    """One FlowSOM training (a marker set) and its results on the ROIs. Several runs can coexist."""

    id: int
    model: PixelModel
    name: str = ""
    k: int = 10
    results: dict[int, RoiClusters] = field(default_factory=dict)  # ROI id -> result
    mc_names: dict[int, str] = field(default_factory=dict)
    mc_selected: dict[int, bool] = field(default_factory=dict)  # individual show/hide per metacluster
    lookup: np.ndarray | None = None
    lookup_k: int | None = None

    _saved: dict = field(default_factory=dict)  # k -> (lookup, mc_names, mc_selected)

    def ensure_lookup(self) -> np.ndarray:
        if self.lookup is None or self.lookup_k != self.k:
            self.lookup = self.model.metacluster_lookup(self.k)
            self.lookup_k = self.k
        return self.lookup

    def switch_k(self, k: int):
        """Change the number of metaclusters. The grouping, names and show/hide choices of the old k are kept,
        so going back to it brings the names back (and the very same grouping, not a re-randomised one)."""
        if self.lookup is not None and self.lookup_k is not None:
            self._saved[self.lookup_k] = (self.lookup, self.mc_names, self.mc_selected)
        self.k = k
        if k in self._saved:
            self.lookup, self.mc_names, self.mc_selected = self._saved[k]
            self.lookup_k = k
        else:
            self.lookup, self.lookup_k, self.mc_names, self.mc_selected = None, None, {}, {}
