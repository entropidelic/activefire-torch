"""Give PyTorch the patches one at a time: an image and its fire mask."""

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset

# The order of the 10 bands inside every image file. Band 8 (panchromatic) is
# not stored, so the 8th band in the file is Landsat B9 (docs/paper_notes.md §2).
BANDS_IN_FILE = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9", "B10", "B11")

# The largest value a pixel can have (the files store 16-bit whole numbers).
MAX_VALUE = 65535


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


def compute_band_stats(split_file, bands):
    """The average and spread (standard deviation) of each band, over every
    pixel of the training patches.

    Only training patches are used: if validation or test patches were
    included, the model would get a small hint about them.

    Returns e.g. {"B7": {"mean": 10061.2, "std": 4052.3}, ...}
    """
    table = pd.read_csv(split_file)
    train_rows = table[table["split"] == "train"]

    # Add up, per band, the values and the squared values of every pixel.
    # The images are read one at a time, so they never all sit in memory.
    total = np.zeros(len(bands))
    total_of_squares = np.zeros(len(bands))
    pixel_count = 0
    for path in train_rows["image"]:
        image = read_image(path, bands).astype(np.float64)
        total += image.sum(axis=(1, 2))
        total_of_squares += (image**2).sum(axis=(1, 2))
        pixel_count += image.shape[1] * image.shape[2]

    mean = total / pixel_count
    # Variance = average of the squares - square of the average.
    std = np.sqrt(total_of_squares / pixel_count - mean**2)
    return {band: {"mean": float(mean[i]), "std": float(std[i])} for i, band in enumerate(bands)}


class FireDataset(Dataset):
    """The patches of one split (train, val or test).

    split_file   a CSV made by splits.py, e.g. data/splits/scene.csv
    split        "train", "val" or "test"
    bands        which bands to read, e.g. ["B7", "B6", "B2"]
    mask_source  whose masks to use, e.g. "schroeder" (reads the column mask_schroeder)
    scaling      how to turn the raw values into small numbers:
                   "max"          divide by 65535, as the original did (values 0 to 1)
                   "standardize"  subtract each band's average and divide by its
                                  spread (needs band_stats)
    band_stats   from compute_band_stats(); always from the training split,
                 also when this dataset is val or test
    """

    def __init__(self, split_file, split, bands, mask_source, scaling="max", band_stats=None):
        if scaling not in ("max", "standardize"):
            raise ValueError(f"Unknown scaling {scaling!r}; use 'max' or 'standardize'")

        table = pd.read_csv(split_file)
        # Keep only the rows of this split, numbered 0, 1, 2, ...
        self.rows = table[table["split"] == split].reset_index(drop=True)
        self.bands = bands
        self.mask_column = f"mask_{mask_source}"
        self.scaling = scaling

        if scaling == "standardize":
            if band_stats is None:
                raise ValueError("scaling='standardize' needs band_stats from compute_band_stats()")
            # Shape [bands, 1, 1], so that each band's number is applied to
            # every pixel of that band of a [bands, 256, 256] image.
            self.mean = np.array([band_stats[band]["mean"] for band in bands], dtype=np.float32).reshape(-1, 1, 1)
            self.std = np.array([band_stats[band]["std"] for band in bands], dtype=np.float32).reshape(-1, 1, 1)

    def __len__(self):
        """How many patches there are. PyTorch uses this to know when an epoch ends."""
        return len(self.rows)

    def __getitem__(self, index):
        """Patch number `index`, as (image, mask).

        image: float tensor [bands, 256, 256], scaled
        mask:  float tensor [1, 256, 256], 1 = fire, 0 = no fire
        """
        row = self.rows.iloc[index]

        # Read the bands and turn them into decimal numbers.
        image = read_image(row["image"], self.bands).astype(np.float32)

        # No mask file means the mask source found no fire in this patch.
        if pd.isna(row[self.mask_column]):
            mask = np.zeros(image.shape[1:], dtype=np.uint8)
        else:
            mask = read_mask(row[self.mask_column])

        # Make the values small (around 0 to 1), which is what the network's
        # starting weights and learning rate are made for.
        if self.scaling == "max":
            image = image / MAX_VALUE
        else:
            image = (image - self.mean) / self.std

        # Convert to PyTorch tensors.
        image = torch.from_numpy(image)
        # The model outputs [1, 256, 256] (one "fire" channel), so the mask
        # gets the same shape: add a first dimension of size 1.
        mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        return image, mask
