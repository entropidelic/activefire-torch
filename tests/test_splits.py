import pandas as pd

from activefire_torch.splits import split_by_location, split_by_patch


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
    assert split["split"].notna().all()
    assert (split.groupby("pathrow")["split"].nunique() == 1).all()
    # Each region has locations in every split (needed for T1b).
    assert (split.groupby("region")["split"].nunique() == 3).all()


def test_same_seed_gives_same_split():
    patches = fake_patches()
    assert split_by_location(patches).equals(split_by_location(patches))
    assert not split_by_location(patches, seed=1).equals(split_by_location(patches))
