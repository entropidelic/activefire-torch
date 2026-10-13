from datetime import date

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from activefire_torch.files import (
    FILE_BANDS,
    band_indices,
    canonical_source,
    parse_name,
    read_image,
    read_mask,
    scan,
)

SCENE = "LC08_L1TP_124046_20200810_20200810_01_RT"


def write_tif(path, array):
    """Write a [C, H, W] array as a small GeoTIFF."""
    path.parent.mkdir(parents=True, exist_ok=True)
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


def test_parse_image_name():
    name = parse_name(f"{SCENE}_p00367.tif")
    assert name.scene_id == SCENE
    assert (name.wrs_path, name.wrs_row, name.pathrow) == (124, 46, "124046")
    assert name.acquired == date(2020, 8, 10)
    assert name.patch == 367
    assert name.tag is None


@pytest.mark.parametrize(
    "tag, source",
    [
        ("Schroeder", "schroeder"),
        ("Murphy", "murphy"),
        ("Kumar-Roy", "kumar-roy"),
        ("GOLI_v2", "kumar-roy"),
        ("v1", "manual"),
        ("Voting", "voting"),
        ("intersection", "intersection"),
    ],
)
def test_parse_mask_name(tag, source):
    name = parse_name(f"/some/dir/{SCENE}_{tag}_p00367.tif")
    assert name.scene_id == SCENE
    assert name.patch == 367
    assert name.tag == tag
    assert canonical_source(tag) == source


def test_l1gt_scene_parses():
    assert parse_name("LC08_L1GT_117060_20200825_20200825_01_RT_p00625.tif").pathrow == "117060"


@pytest.mark.parametrize("bad", ["notes.tif", f"{SCENE}.tif", f"{SCENE}_p00367.png"])
def test_unrecognized_name_raises(bad):
    with pytest.raises(ValueError):
        parse_name(bad)


def test_unknown_tag_has_no_source():
    assert canonical_source("Something") is None


def test_band_indices_are_zero_based_and_skip_b8():
    assert band_indices(["B7", "B6", "B2"]) == [6, 5, 1]
    assert band_indices(["B9", "B10", "B11"]) == [7, 8, 9]
    with pytest.raises(ValueError):
        band_indices(["B8"])


def test_read_image_selects_bands_by_name(tmp_path):
    # Fill file band i (0-based) with the value i, so the result shows which
    # file bands were read.
    image = np.stack([np.full((4, 4), i, dtype=np.uint16) for i in range(len(FILE_BANDS))])
    path = tmp_path / f"{SCENE}_p00001.tif"
    write_tif(path, image)

    assert read_image(path).shape == (10, 4, 4)
    selected = read_image(path, ["B7", "B6", "B2"])
    assert selected.dtype == np.uint16
    assert selected[:, 0, 0].tolist() == [6, 5, 1]


def test_read_mask_returns_2d(tmp_path):
    path = tmp_path / f"{SCENE}_Murphy_p00001.tif"
    write_tif(path, np.ones((1, 4, 4), dtype=np.uint8))
    mask = read_mask(path)
    assert mask.shape == (4, 4)
    assert mask.dtype == np.uint8


def test_scan_classifies_files(tmp_path):
    image = np.zeros((10, 2, 2), dtype=np.uint16)
    mask = np.zeros((1, 2, 2), dtype=np.uint8)
    write_tif(tmp_path / "Oceania" / "images" / f"{SCENE}_p00001.tif", image)
    write_tif(tmp_path / "Oceania" / "masks" / f"{SCENE}_Schroeder_p00001.tif", mask)
    write_tif(tmp_path / "Oceania" / "masks" / f"{SCENE}_Mystery_p00001.tif", mask)
    write_tif(tmp_path / "Oceania" / "other.tif", mask)

    df = scan(tmp_path).set_index("tag", drop=False)
    assert set(df["group"]) == {"Oceania"}
    assert sorted(df["kind"]) == ["image", "mask", "unknown_tag", "unparsed"]
    assert df.loc["Schroeder", "source"] == "schroeder"
    assert df.loc["Schroeder", "pathrow"] == "124046"
