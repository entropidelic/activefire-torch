import pandas as pd
import pytest

from activefire_torch.splits import list_patches, read_file_name, split_by_location, split_by_patch

SCENE = "LC08_L1TP_124046_20200810_20200810_01_RT"


def test_read_image_file_name():
    assert read_file_name(f"{SCENE}_p00367.tif") == {
        "scene_id": SCENE,
        "pathrow": "124046",
        "patch": 367,
        "mask_tag": None,
    }


@pytest.mark.parametrize("tag", ["Schroeder", "Murphy", "Kumar-Roy", "GOLI_v2", "v1", "Voting", "intersection"])
def test_read_mask_file_name(tag):
    name = read_file_name(f"/some/folder/{SCENE}_{tag}_p00367.tif")
    assert name is not None
    assert name["scene_id"] == SCENE
    assert name["patch"] == 367
    assert name["mask_tag"] == tag


@pytest.mark.parametrize("bad", ["notes.tif", f"{SCENE}.tif", f"{SCENE}_p00367.png", f"{SCENE}_pXYZ.tif"])
def test_wrong_file_name_gives_none(bad):
    assert read_file_name(bad) is None


def test_list_patches_matches_masks_to_images(tmp_path):
    # list_patches only looks at names, so empty files are enough.
    folder = tmp_path / "Oceania"
    folder.mkdir()
    for name in [
        f"{SCENE}_p00001.tif",  # image with a Schroeder mask
        f"{SCENE}_p00002.tif",  # image without masks (no fire found)
        f"{SCENE}_Schroeder_p00001.tif",
        f"{SCENE}_Mystery_p00001.tif",  # unknown mask source: skipped
        "notes.tif",  # not a dataset file: skipped
    ]:
        (folder / name).touch()

    patches = list_patches(tmp_path).set_index("patch")
    assert list(patches.index) == [1, 2]
    assert set(patches["region"]) == {"Oceania"}
    assert patches.loc[1, "mask_schroeder"] == str(folder / f"{SCENE}_Schroeder_p00001.tif")
    assert pd.isna(patches.loc[2, "mask_schroeder"])


def fake_patches():
    """Two regions with 10 locations each and 5 patches per location."""
    rows = []
    for region in ["Oceania", "North_America2"]:
        for location in range(10):
            pathrow = f"{region[:2]}{location:04d}"
            for patch in range(5):
                rows.append({"region": region, "pathrow": pathrow, "patch": patch})
    return pd.DataFrame(rows)


def test_split_by_patch_is_40_10_50():
    patches = pd.DataFrame({"patch": range(100)})
    split = split_by_patch(patches)
    assert sorted(split["patch"]) == list(range(100))  # every patch exactly once
    assert split["split"].value_counts().to_dict() == {"test": 50, "train": 40, "val": 10}


def test_no_location_in_two_splits():
    split = split_by_location(fake_patches())
    assert len(split) == len(fake_patches())
    assert split.index.is_unique  # no repeated row numbers
    assert split["split"].notna().all()
    assert (split.groupby("pathrow")["split"].nunique() == 1).all()
    # Each region has locations in every split (needed for T1b).
    assert (split.groupby("region")["split"].nunique() == 3).all()


def test_same_seed_gives_same_split():
    patches = fake_patches()
    assert split_by_location(patches).equals(split_by_location(patches))
    assert not split_by_location(patches, seed=1).equals(split_by_location(patches))
