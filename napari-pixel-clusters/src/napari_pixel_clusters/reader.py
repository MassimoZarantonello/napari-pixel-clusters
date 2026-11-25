"""napari reader: File > Open / drag-and-drop of a MACSima OME-TIFF opens it lazily, one layer per channel."""
from __future__ import annotations

from pathlib import Path

import tifffile

from .io import META_KEY, _key, open_stack


def get_reader(path):
    """Return a reader for multi-channel CYX TIFFs (MACSima OME-TIFF), None for anything else."""
    if isinstance(path, (list, tuple)) or Path(path).suffix.lower() not in {".tif", ".tiff"}:
        return None
    try:
        with tifffile.TiffFile(path) as tf:
            series = tf.series[0]
            if series.axes != "CYX" or series.shape[0] < 2:
                return None
    except Exception:  # noqa: BLE001 - not a readable tiff: let another reader try
        return None
    return read_stack


def read_stack(path):
    stack = open_stack(path)
    first = next((i for i, n in enumerate(stack.channel_names) if n.startswith("DAPI")), 0)
    attrs = {
        "channel_axis": 0,
        "name": stack.channel_names,
        "multiscale": True,
        "blending": "additive",
        "visible": [i == first for i in range(len(stack.channel_names))],
        "metadata": {META_KEY: _key(path)},  # the Pixel Cluster Explorer recognises the image by this
    }
    return [(stack.levels, attrs, "image")]
