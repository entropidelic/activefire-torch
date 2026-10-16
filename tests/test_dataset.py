import numpy as np
import pandas as pd
import pytest
import rasterio
import torch
from rasterio.transform import from_origin

from activefire_torch.dataset import BANDS_IN_FILE, FireDataset, compute_band_stats, read_image


def write_tif(path, array):
    """Write a [bands, height, width] array as a small GeoTIFF."""
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=array.shape[2],
        height=array.shape[1],
        count=array.shape[0],
        dtype=array.dtype,
        crs="EPSG:32650",
        transform=from_origin(0, 0, 30, 30),
    ) as dst:
        dst.write(array)


def test_read_image_picks_bands_by_name(tmp_path):
    # Band i holds the value i, so the output shows which bands were read.
    image = np.stack([np.full((4, 4), i, dtype=np.uint16) for i in range(len(BANDS_IN_FILE))])
    path = tmp_path / "image.tif"
    write_tif(path, image)

    assert read_image(path).shape == (10, 4, 4)
    selected = read_image(path, ["B7", "B6", "B2"])
    assert selected.dtype == np.uint16
    assert selected[:, 0, 0].tolist() == [6, 5, 1]
    # B9 is the 8th band in the file, because B8 is not stored.
    assert read_image(path, ["B9"])[0, 0, 0] == 7
    with pytest.raises(ValueError):
        read_image(path, ["B8"])


def write_split_file(folder):
    """Patches a (train, all pixels 100), b (train, 300), c (test, 60000).
    Only a has a Schroeder mask (all fire)."""
    rows = []
    for name, value, split in [("a", 100, "train"), ("b", 300, "train"), ("c", 60000, "test")]:
        path = folder / f"{name}.tif"
        write_tif(path, np.full((10, 4, 4), value, dtype=np.uint16))
        rows.append({"image": str(path), "split": split, "mask_schroeder": None})

    mask_path = folder / "a_mask.tif"
    write_tif(mask_path, np.ones((1, 4, 4), dtype=np.uint8))
    rows[0]["mask_schroeder"] = str(mask_path)

    split_file = folder / "split.csv"
    pd.DataFrame(rows).to_csv(split_file, index=False)
    return split_file


def test_fire_only_keeps_patches_with_a_mask_file(tmp_path):
    split_file = write_split_file(tmp_path)

    fire_only = FireDataset(split_file, "train", ["B7"], "schroeder", "fire_only")
    assert len(fire_only) == 1  # only "a"
    _, mask = fire_only[0]
    assert mask.sum() == 16  # all 4 x 4 pixels are fire

    full = FireDataset(split_file, "train", ["B7"], "schroeder", "full")
    assert len(full) == 2  # "a" and "b"
    _, mask = full[1]
    assert mask.shape == (1, 4, 4)
    assert mask.sum() == 0  # "b" has no mask file -> no fire


def test_band_stats_come_from_training_patches_only(tmp_path):
    split_file = write_split_file(tmp_path)
    stats = compute_band_stats(split_file, ["B7"], "schroeder", "full")
    # Only the training patches (100 and 300) count.
    assert stats["B7"]["mean"] == pytest.approx(200)
    assert stats["B7"]["std"] == pytest.approx(100)

    # The test set is scaled with the training numbers.
    test_set = FireDataset(split_file, "test", ["B7"], "schroeder", "full", scaling="standardize", band_stats=stats)
    image, _ = test_set[0]
    assert image[0, 0, 0].item() == pytest.approx((60000 - 200) / 100)


def test_band_stats_use_the_same_patches_as_training(tmp_path):
    split_file = write_split_file(tmp_path)
    # With fire_only, only "a" (100) is a training patch.
    stats = compute_band_stats(split_file, ["B7"], "schroeder", "fire_only")
    assert stats["B7"]["mean"] == pytest.approx(100)


def write_one_fire_patch(folder):
    """One 4 x 4 training patch, bright and on fire only at (0, 1). Each of the
    8 rotations and mirrors moves that pixel to a different place."""
    image = np.zeros((10, 4, 4), dtype=np.uint16)
    image[:, 0, 1] = 1000
    mask = np.zeros((1, 4, 4), dtype=np.uint8)
    mask[0, 0, 1] = 1
    write_tif(folder / "image.tif", image)
    write_tif(folder / "mask.tif", mask)

    row = {"image": str(folder / "image.tif"), "split": "train", "mask_schroeder": str(folder / "mask.tif")}
    split_file = folder / "split.csv"
    pd.DataFrame([row]).to_csv(split_file, index=False)
    return split_file


def test_augmentation_moves_image_and_mask_together(tmp_path):
    split_file = write_one_fire_patch(tmp_path)
    dataset = FireDataset(split_file, "train", ["B7"], "schroeder", "fire_only", augment=True)
    torch.manual_seed(0)

    fire_positions = set()
    for _ in range(100):
        image, mask = dataset[0]
        assert torch.equal(image[0] > 0, mask[0] == 1)  # moved together
        assert image.max().item() == pytest.approx(1000 / 65535)  # values unchanged
        row, column = mask[0].nonzero()[0].tolist()
        fire_positions.add((row, column))

    assert len(fire_positions) == 8  # every rotation and mirror occurred
