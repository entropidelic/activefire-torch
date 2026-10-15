"""PyTorch dataset of (image, fire mask) patches."""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset

# Band order in the image files. B8 (panchromatic) is not stored, so file
# band 8 is B9 (docs/paper_notes.md §2).
BANDS_IN_FILE = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9", "B10", "B11")

# Images are uint16; the original divides by this.
MAX_VALUE = 65535


def read_image(path: str | Path, bands: list[str] | None = None) -> np.ndarray:
    """Raw values as [bands, height, width]. bands: names, e.g. ["B7", "B6", "B2"];
    None reads all 10."""
    with rasterio.open(path) as src:
        if bands is None:
            return src.read()
        for band in bands:
            if band not in BANDS_IN_FILE:
                raise ValueError(f"Unknown band {band!r}; the file has {BANDS_IN_FILE}")
        # rasterio band indices start at 1.
        return src.read([BANDS_IN_FILE.index(band) + 1 for band in bands])


def read_mask(path: str | Path) -> np.ndarray:
    """Mask as [height, width], 1 = fire."""
    with rasterio.open(path) as src:
        return src.read(1)


def select_rows(split_file: str | Path, split: str, mask_source: str, version: str) -> pd.DataFrame:
    """Rows of a split file used by one dataset.

    split        "train", "val" or "test"
    mask_source  e.g. "voting"
    version      "fire_only": only patches with a mask from mask_source, as in
                 the original. "full": all patches; those without a mask get an
                 all-zero mask (patches where only other sources found fire, D16).
    """
    if version not in ("fire_only", "full"):
        raise ValueError(f"Unknown version {version!r}; use 'fire_only' or 'full'")

    table = pd.read_csv(split_file)
    rows = table[table["split"] == split]
    if version == "fire_only":
        rows = rows[rows[f"mask_{mask_source}"].notna()]
    return rows.reset_index(drop=True)


def compute_band_stats(
    split_file: str | Path, bands: list[str], mask_source: str, version: str
) -> dict[str, dict[str, float]]:
    """Per-band mean and std over all pixels of the training patches.

    Uses the same rows as the training dataset (same mask_source and version);
    val and test are excluded to avoid leakage.

    Returns e.g. {"B7": {"mean": 10061.2, "std": 4052.3}, ...}
    """
    train_rows = select_rows(split_file, "train", mask_source, version)

    # Running sums, one image at a time, so the images never all sit in memory.
    total = np.zeros(len(bands))
    total_of_squares = np.zeros(len(bands))
    pixel_count = 0
    for path in train_rows["image"]:
        image = read_image(path, bands).astype(np.float64)
        total += image.sum(axis=(1, 2))
        total_of_squares += (image**2).sum(axis=(1, 2))
        pixel_count += image.shape[1] * image.shape[2]

    mean = total / pixel_count
    std = np.sqrt(total_of_squares / pixel_count - mean**2)  # var = E[x^2] - E[x]^2
    return {band: {"mean": float(mean[i]), "std": float(std[i])} for i, band in enumerate(bands)}


class FireDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Patches of one split.

    split_file   CSV from splits.py, e.g. data/splits/scene.csv
    split        "train", "val" or "test"
    bands        e.g. ["B7", "B6", "B2"]
    mask_source  e.g. "schroeder" (column mask_schroeder)
    version      "fire_only" or "full" (see select_rows)
    scaling      "max": divide by 65535, as the original.
                 "standardize": per band, (x - mean) / std; needs band_stats.
    band_stats   from compute_band_stats() on the training split, also for
                 val and test
    augment      random 90° rotations and mirroring; training only
    """

    def __init__(
        self,
        split_file: str | Path,
        split: str,
        bands: list[str],
        mask_source: str,
        version: str,
        scaling: str = "max",
        band_stats: dict[str, dict[str, float]] | None = None,
        augment: bool = False,
    ) -> None:
        if scaling not in ("max", "standardize"):
            raise ValueError(f"Unknown scaling {scaling!r}; use 'max' or 'standardize'")

        self.rows = select_rows(split_file, split, mask_source, version)
        self.bands = bands
        self.mask_column = f"mask_{mask_source}"
        self.scaling = scaling
        self.augment = augment

        if scaling == "standardize":
            if band_stats is None:
                raise ValueError("scaling='standardize' needs band_stats from compute_band_stats()")
            # Shape [bands, 1, 1] to broadcast over [bands, height, width].
            self.mean = np.array([band_stats[band]["mean"] for band in bands], dtype=np.float32).reshape(-1, 1, 1)
            self.std = np.array([band_stats[band]["std"] for band in bands], dtype=np.float32).reshape(-1, 1, 1)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """(image [bands, H, W] float32 scaled, mask [1, H, W] float32 in {0, 1})."""
        row = self.rows.iloc[index]

        image = read_image(row["image"], self.bands).astype(np.float32)

        # No mask file: this source found no fire in the patch.
        if pd.isna(row[self.mask_column]):
            mask = np.zeros(image.shape[1:], dtype=np.uint8)
        else:
            mask = read_mask(row[self.mask_column])

        if self.scaling == "max":
            image = image / MAX_VALUE
        else:
            image = (image - self.mean) / self.std

        image = torch.from_numpy(image)
        # [1, H, W], the same shape as the model output.
        mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)

        # Same rotation and mirror for image and mask; pixel values unchanged.
        if self.augment:
            quarter_turns = int(torch.randint(0, 4, (1,)))
            image = torch.rot90(image, quarter_turns, dims=(1, 2))
            mask = torch.rot90(mask, quarter_turns, dims=(1, 2))

            if torch.rand(1) < 0.5:
                image = torch.flip(image, dims=(2,))
                mask = torch.flip(mask, dims=(2,))

        return image, mask
