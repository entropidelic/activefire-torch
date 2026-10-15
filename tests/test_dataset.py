import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

from activefire_torch.dataset import BANDS_IN_FILE, FireDataset, compute_band_stats, read_image, read_mask


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
    # Fill band number i (counting from 0) with the value i, so the result
    # shows which bands were read.
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


def test_read_mask_returns_2d(tmp_path):
    path = tmp_path / "mask.tif"
    write_tif(path, np.ones((1, 4, 4), dtype=np.uint8))
    mask = read_mask(path)
    assert mask.shape == (4, 4)
    assert mask.dtype == np.uint8


def write_split_file(folder):
    """Three tiny patches, every pixel of every band set to one value:
    two training patches (100 and 300) and one test patch (60000).
    None of them has a mask file."""
    rows = []
    for name, value, split in [("a", 100, "train"), ("b", 300, "train"), ("c", 60000, "test")]:
        path = folder / f"{name}.tif"
        write_tif(path, np.full((10, 4, 4), value, dtype=np.uint16))
        rows.append({"image": str(path), "split": split, "mask_schroeder": None})
    split_file = folder / "split.csv"
    pd.DataFrame(rows).to_csv(split_file, index=False)
    return split_file


def test_max_scaling_divides_by_65535(tmp_path):
    split_file = write_split_file(tmp_path)
    dataset = FireDataset(split_file, "train", ["B7"], "schroeder", scaling="max")
    image, mask = dataset[1]
    assert image.shape == (1, 4, 4)
    assert image[0, 0, 0].item() == pytest.approx(300 / 65535)
    assert mask.shape == (1, 4, 4)
    assert mask.sum() == 0  # no mask file -> no fire


def test_band_stats_come_from_training_patches_only(tmp_path):
    split_file = write_split_file(tmp_path)
    stats = compute_band_stats(split_file, ["B7"])
    # Only 100 and 300 count; the test patch (60000) is ignored.
    assert stats["B7"]["mean"] == pytest.approx(200)
    assert stats["B7"]["std"] == pytest.approx(100)

    # The test set is scaled with the training numbers.
    test_set = FireDataset(split_file, "test", ["B7"], "schroeder", scaling="standardize", band_stats=stats)
    image, _ = test_set[0]
    assert image[0, 0, 0].item() == pytest.approx((60000 - 200) / 100)
