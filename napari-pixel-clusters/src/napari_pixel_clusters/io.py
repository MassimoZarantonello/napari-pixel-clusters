"""Lazy access to a whole MACSima OME-TIFF (no napari imports)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import dask.array as da
import numpy as np
import tifffile
import zarr


META_KEY = "pixel_clusters_path"  # set in the metadata of the channel layers made by our reader
_STACKS: dict[str, "ImageStack"] = {}  # resolved path -> the lazily opened image


def _key(path) -> str:
    return str(Path(path).expanduser().resolve())


@dataclass
class ImageStack:
    path: Path
    channel_names: list[str]
    levels: list  # pyramid, each a lazy (C, Y, X) dask array, level 0 = full resolution
    _full: object = field(repr=False)  # memmap or dask array (C, Y, X) at full resolution
    _tif: object = field(repr=False, default=None)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.levels[0].shape)

    def read_region(self, y0: int, y1: int, x0: int, x1: int, channels: list[str]) -> np.ndarray:
        """Read only the requested channels of one region, full resolution -> (C, h, w) array."""
        idx = [self.channel_names.index(c) for c in channels]
        data = self._full[idx, y0:y1, x0:x1]
        return np.asarray(data.compute() if hasattr(data, "compute") else data)

    def close(self):
        if self._tif is not None:
            self._tif.close()


def open_stack(path: str | Path) -> ImageStack:
    path = Path(path).expanduser()
    tf = tifffile.TiffFile(path)
    series = tf.series[0]
    if series.axes != "CYX":
        tf.close()
        raise ValueError(f"Expected a CYX image, got axes {series.axes!r}")

    names = re.findall(r'<Channel [^>]*Name="([^"]*)"', tf.ome_metadata or "")
    if len(names) != series.shape[0]:
        names = [f"channel {i}" for i in range(series.shape[0])]

    levels = []
    for i in range(len(getattr(series, "levels", [series]))):
        z = zarr.open(series.aszarr(level=i), mode="r")
        levels.append(da.from_zarr(z))

    try:  # uncompressed + contiguous: an ROI can be cut straight from disk
        full = tifffile.memmap(path)
        if full.shape != series.shape:
            raise ValueError
    except Exception:
        full = levels[0]
    stack = ImageStack(path, names, levels, full, tf)
    _STACKS[_key(path)] = stack
    return stack


def get_stack(path) -> ImageStack:
    """The image the reader opened (or open it now)."""
    return _STACKS.get(_key(path)) or open_stack(path)
