"""Shared by the tests: writing small fake image and mask files."""

import rasterio
from rasterio.transform import from_origin


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
