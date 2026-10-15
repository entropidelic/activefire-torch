import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from activefire_torch.dataset import BANDS_IN_FILE, read_image, read_mask


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
