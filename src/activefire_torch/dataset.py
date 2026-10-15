"""Give PyTorch the patches one at a time: an image and its fire mask."""

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset

# The order of the 10 bands inside every image file. Band 8 (panchromatic) is
# not stored, so the 8th band in the file is Landsat B9 (docs/paper_notes.md §2).
BANDS_IN_FILE = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9", "B10", "B11")


def read_image(path, bands=None):
    """Read an image patch as an array of shape [bands, height, width].

    Values are returned as stored (no scaling). Pick bands by name, e.g.
    bands=["B7", "B6", "B2"], so file positions never have to be counted by
    hand. None reads all 10.
    """
    with rasterio.open(path) as src:
        if bands is None:
            return src.read()
        for band in bands:
            if band not in BANDS_IN_FILE:
                raise ValueError(f"Unknown band {band!r}; the file has {BANDS_IN_FILE}")
        # rasterio counts bands from 1, Python from 0.
        return src.read([BANDS_IN_FILE.index(band) + 1 for band in bands])


def read_mask(path):
    """Read a mask patch as an array of shape [height, width] (1 = fire, 0 = no fire)."""
    with rasterio.open(path) as src:
        return src.read(1)


class FireDataset(Dataset):
    """The patches of one split (train, val or test).

    split_file   a CSV made by splits.py, e.g. data/splits/scene.csv
    split        "train", "val" or "test"
    bands        which bands to read, e.g. ["B7", "B6", "B2"]
    mask_source  whose masks to use, e.g. "schroeder" (reads the column mask_schroeder)
    """

    def __init__(self, split_file, split, bands, mask_source):
        table = pd.read_csv(split_file)
        # Keep only the rows of this split, numbered 0, 1, 2, ...
        self.rows = table[table["split"] == split].reset_index(drop=True)
        self.bands = bands
        self.mask_column = f"mask_{mask_source}"

    def __len__(self):
        """How many patches there are. PyTorch uses this to know when an epoch ends."""
        return len(self.rows)

    def __getitem__(self, index):
        """Patch number `index`, as (image, mask).

        image: float tensor [bands, 256, 256], raw values (not scaled yet)
        mask:  float tensor [1, 256, 256], 1 = fire, 0 = no fire
        """
        row = self.rows.iloc[index]

        image = read_image(row["image"], self.bands)

        # No mask file means the mask source found no fire in this patch.
        if pd.isna(row[self.mask_column]):
            mask = np.zeros(image.shape[1:], dtype=np.uint8)
        else:
            mask = read_mask(row[self.mask_column])

        # Convert to PyTorch tensors of decimal numbers (what the model expects).
        image = torch.from_numpy(image.astype(np.float32))
        # The model outputs [1, 256, 256] (one "fire" channel), so the mask
        # gets the same shape: add a first dimension of size 1.
        mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        return image, mask
