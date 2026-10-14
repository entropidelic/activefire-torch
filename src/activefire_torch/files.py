"""Find, name and read the dataset's image and mask files.

Every file is a 256 x 256 patch cut from one Landsat-8 picture (a "scene").
File names look like this (docs/paper_notes.md §1):

    image: LC08_L1TP_124046_20200810_20200810_01_RT_p00367.tif
    mask:  LC08_L1TP_124046_20200810_20200810_01_RT_Schroeder_p00367.tif

Split at "_", the parts mean:

    LC08                        Landsat 8
    L1TP                        processing level
    124046                      location: WRS path 124, row 046
    20200810                    date the picture was taken
    20200810_01_RT              processing date, version, tier
    Schroeder                   (masks only) who made the mask
    p00367                      patch number inside the scene
"""

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

# The order of the 10 bands inside every image file. Band 8 (panchromatic) is
# not stored, so the 8th band in the file is Landsat B9 (docs/paper_notes.md §2).
BANDS_IN_FILE = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9", "B10", "B11")

# The word in a mask file name -> who made the mask. The small sample calls
# the Kumar-Roy masks "GOLI_v2"; the hand-made masks are tagged "v1".
MASK_SOURCE_BY_TAG = {
    "schroeder": "schroeder",
    "murphy": "murphy",
    "kumar-roy": "kumar-roy",
    "goli_v2": "kumar-roy",
    "intersection": "intersection",
    "voting": "voting",
    "v1": "manual",
}


@dataclass(frozen=True)
class FileName:
    """What a file name says about the file."""

    scene_id: str  # e.g. "LC08_L1TP_124046_20200810_20200810_01_RT"
    pathrow: str  # the location, e.g. "124046"
    date: date  # the day the picture was taken
    patch: int  # e.g. 367
    mask_tag: str | None  # e.g. "Schroeder"; None for images


def read_file_name(filename: str | Path) -> FileName:
    """Read the parts of an image or mask file name.

    Raises ValueError if the name does not look like a dataset file.
    """
    name = Path(filename).name
    parts = name.removesuffix(".tif").split("_")

    # The scene ID is always the first 7 parts and the patch number the last.
    # Anything in between is the mask tag (it can contain "_", e.g. "GOLI_v2").
    looks_right = (
        name.endswith(".tif")
        and len(parts) >= 8
        and parts[0] == "LC08"
        and parts[-1].startswith("p")
        and parts[-1][1:].isdigit()
    )
    if not looks_right:
        raise ValueError(f"Not a dataset file name: {filename!r}")

    scene_parts = parts[:7]
    tag_parts = parts[7:-1]
    return FileName(
        scene_id="_".join(scene_parts),
        pathrow=scene_parts[2],
        date=datetime.strptime(scene_parts[3], "%Y%m%d").date(),
        patch=int(parts[-1][1:]),
        mask_tag="_".join(tag_parts) if tag_parts else None,
    )


def mask_source(mask_tag: str) -> str | None:
    """Who made a mask, from the tag in its file name. None if the tag is unknown."""
    return MASK_SOURCE_BY_TAG.get(mask_tag.lower())


def list_files(data_folder: Path) -> pd.DataFrame:
    """Make a table with one row per .tif file found inside `data_folder`.

    Columns:
        file         path of the file
        bytes        file size
        region       the first folder inside data_folder, e.g. "Oceania"
                     when data_folder is data/raw/regions
        kind         "image", "mask", "unknown_tag" (a mask tag we don't
                     know) or "unparsed" (not a dataset file name)
        mask_source  who made the mask (empty for images)
        mask_tag     the tag exactly as written in the file name
        scene_id, pathrow, date, patch   from the file name

    Files with unknown tags or names are kept in the table, so they can be
    looked at instead of disappearing silently.
    """
    data_folder = Path(data_folder)
    rows = []
    for file in sorted(data_folder.rglob("*.tif")):
        folders = file.relative_to(data_folder).parts
        row = {
            "file": str(file),
            "bytes": file.stat().st_size,
            "region": folders[0] if len(folders) > 1 else "",
        }

        try:
            name = read_file_name(file)
        except ValueError:
            rows.append(row | {"kind": "unparsed"})
            continue

        if name.mask_tag is None:
            kind = "image"
            source = None
        else:
            source = mask_source(name.mask_tag)
            kind = "mask" if source else "unknown_tag"

        row = row | {
            "kind": kind,
            "mask_source": source,
            "mask_tag": name.mask_tag,
            "scene_id": name.scene_id,
            "pathrow": name.pathrow,
            "date": name.date,
            "patch": name.patch,
        }
        rows.append(row)

    columns = ["file", "bytes", "region", "kind", "mask_source", "mask_tag", "scene_id", "pathrow", "date", "patch"]
    return pd.DataFrame(rows, columns=columns)


def band_positions(bands: list[str]) -> list[int]:
    """Where bands (e.g. ["B7", "B6", "B2"]) are inside an image file, counting from 0."""
    unknown = [band for band in bands if band not in BANDS_IN_FILE]
    if unknown:
        raise ValueError(f"Unknown bands {unknown}; the file has {BANDS_IN_FILE}")
    return [BANDS_IN_FILE.index(band) for band in bands]


def read_image(path: str | Path, bands: list[str] | None = None) -> np.ndarray:
    """Read an image patch as an array of shape [bands, height, width].

    Values are returned as stored (no scaling). Pick bands by name, e.g.
    bands=["B7", "B6", "B2"]; None reads all 10.
    """
    with rasterio.open(path) as src:
        if bands is None:
            return src.read()
        # rasterio counts bands from 1, Python from 0.
        return src.read([position + 1 for position in band_positions(bands)])


def read_mask(path: str | Path) -> np.ndarray:
    """Read a mask patch as an array of shape [height, width] (1 = fire, 0 = no fire)."""
    with rasterio.open(path) as src:
        return src.read(1)
