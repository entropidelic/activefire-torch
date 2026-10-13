"""Find, parse and read the dataset's GeoTIFF patches.

File names (docs/paper_notes.md §1):
    image: <scene>_p<index>.tif
    mask:  <scene>_<tag>_p<index>.tif
<scene> is the Landsat product ID, e.g. LC08_L1TP_124046_20200810_20200810_01_RT,
and <tag> names the mask source (Schroeder, Murphy, Kumar-Roy, ...).
"""

import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import rasterio

# Order of the bands inside every image file. Band 8 (panchromatic) is not
# stored, so file band 8 is Landsat B9 (docs/paper_notes.md §2).
FILE_BANDS = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9", "B10", "B11")

# File-name tag (lowercased) -> canonical mask source. The small subset names
# Kumar-Roy masks "GOLI_v2"; manual annotations are tagged "v1".
SOURCE_ALIASES = {
    "schroeder": "schroeder",
    "murphy": "murphy",
    "kumar-roy": "kumar-roy",
    "goli_v2": "kumar-roy",
    "intersection": "intersection",
    "voting": "voting",
    "v1": "manual",
}

_NAME_RE = re.compile(
    r"^(?P<scene>LC08_L1[A-Z]{2}_(?P<path>\d{3})(?P<row>\d{3})_(?P<acquired>\d{8})_\d{8}_\d{2}_[A-Z0-9]{2})"
    r"(?:_(?P<tag>.+?))?_p(?P<patch>\d+)\.tif$"
)


@dataclass(frozen=True)
class PatchName:
    scene_id: str
    wrs_path: int
    wrs_row: int
    acquired: date
    patch: int
    tag: str | None  # None for images

    @property
    def pathrow(self) -> str:
        """WRS-2 path/row as in the product ID, e.g. "124046"."""
        return f"{self.wrs_path:03d}{self.wrs_row:03d}"


def parse_name(filename: str) -> PatchName:
    """Parse an image or mask file name. Raises ValueError if it does not match."""
    m = _NAME_RE.match(Path(filename).name)
    if m is None:
        raise ValueError(f"Unrecognized patch file name: {filename!r}")
    return PatchName(
        scene_id=m["scene"],
        wrs_path=int(m["path"]),
        wrs_row=int(m["row"]),
        acquired=datetime.strptime(m["acquired"], "%Y%m%d").date(),
        patch=int(m["patch"]),
        tag=m["tag"],
    )


def canonical_source(tag: str) -> str | None:
    """Canonical mask source for a file-name tag, or None if the tag is unknown."""
    return SOURCE_ALIASES.get(tag.lower())


def scan(root: Path) -> pd.DataFrame:
    """List every .tif under `root`, one row per file.

    Columns: file, bytes, group (first folder below `root`), kind ("image",
    "mask", "unknown_tag" or "unparsed"), source (canonical mask source),
    tag (raw file-name tag), scene_id, pathrow, acquired, patch.
    Unknown tags and unparsable names are kept, not dropped, so they can be
    inspected.
    """
    root = Path(root)
    rows = []
    for file in sorted(root.rglob("*.tif")):
        rel = file.relative_to(root)
        row = {
            "file": str(file),
            "bytes": file.stat().st_size,
            "group": rel.parts[0] if len(rel.parts) > 1 else "",
        }
        try:
            name = parse_name(file.name)
        except ValueError:
            rows.append(row | {"kind": "unparsed"})
            continue
        if name.tag is None:
            kind, source = "image", None
        else:
            source = canonical_source(name.tag)
            kind = "mask" if source else "unknown_tag"
        rows.append(
            row
            | {
                "kind": kind,
                "source": source,
                "tag": name.tag,
                "scene_id": name.scene_id,
                "pathrow": name.pathrow,
                "acquired": name.acquired,
                "patch": name.patch,
            }
        )
    columns = ["file", "bytes", "group", "kind", "source", "tag", "scene_id", "pathrow", "acquired", "patch"]
    return pd.DataFrame(rows, columns=columns)


def band_indices(bands: Sequence[str]) -> list[int]:
    """0-based positions of Landsat bands (e.g. ["B7", "B6", "B2"]) in an image file."""
    unknown = [b for b in bands if b not in FILE_BANDS]
    if unknown:
        raise ValueError(f"Unknown bands {unknown}; expected some of {FILE_BANDS}")
    return [FILE_BANDS.index(b) for b in bands]


def read_image(path: str | Path, bands: Sequence[str] | None = None) -> np.ndarray:
    """Read an image patch as a [C, H, W] array of raw values (no scaling).

    Bands are selected by Landsat name, not by position, so the 1-based
    (rasterio) vs. 0-based (NumPy) mix-up cannot happen at call sites.
    """
    with rasterio.open(path) as src:
        if bands is None:
            return src.read()
        return src.read([i + 1 for i in band_indices(bands)])


def read_mask(path: str | Path) -> np.ndarray:
    """Read a mask patch as an [H, W] array, values as stored in the file."""
    with rasterio.open(path) as src:
        return src.read(1)
