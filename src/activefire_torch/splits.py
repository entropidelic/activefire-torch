"""Decide which patches go to train, validation and test.

Split P (Stage 1) copies the original repo (src/utils/split_dataset.py):
each patch goes to train / val / test at random.

Split S (Stage 2) keeps all patches of one location (WRS path/row) together,
so a location used for testing is never seen in training, not even on
another date.

Run once from the repo root, after downloading the regions:
    uv run python -m activefire_torch.splits
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from activefire_torch.files import list_files

# Where the downloaded regions are. Each region is a subfolder:
#   data/raw/regions/Oceania/...
#   data/raw/regions/North_America2/...
DATA_FOLDER = Path("data/raw/regions")

# Where the split files are written (they are committed to git).
SPLITS_FOLDER = Path("data/splits")

# Same proportions and seed as the original.
TRAIN, VAL, TEST = 0.4, 0.1, 0.5
SEED = 42


def list_patches(data_folder: Path) -> pd.DataFrame:
    """Make a table with one row per image patch.

    Columns:
        image            path of the image file
        scene_id         the satellite picture it was cut from
        pathrow          the location, e.g. "091084"
        region           the region subfolder, e.g. "Oceania"
        patch            the patch number inside the picture
        mask_schroeder   path of the Schroeder mask, or empty if Schroeder found no fire
        mask_murphy      the same for each other mask source found in the folder
        ...
    """
    all_files = list_files(data_folder)

    images = all_files[all_files["kind"] == "image"]
    table = images[["file", "scene_id", "pathrow", "region", "patch"]]
    table = table.rename(columns={"file": "image"})

    # Add one mask column per mask source (schroeder, murphy, voting, ...).
    # A mask belongs to the image with the same scene_id and patch number.
    masks = all_files[all_files["kind"] == "mask"]
    for source in sorted(masks["mask_source"].unique()):
        source_masks = masks[masks["mask_source"] == source]
        source_masks = source_masks[["scene_id", "patch", "file"]]
        source_masks = source_masks.rename(columns={"file": f"mask_{source}"})
        table = table.merge(source_masks, on=["scene_id", "patch"], how="left")

    return table


def split_by_patch(patches: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Split P: send each row to train / val / test at random (40 / 10 / 50).

    Returns the same rows with a new "split" column.
    These are the same two calls as in the original script.
    """
    # Pick the rows (by their row label) that go to each split.
    train_rows, val_and_test_rows = train_test_split(patches.index, test_size=VAL + TEST, random_state=seed)
    val_rows, test_rows = train_test_split(val_and_test_rows, test_size=TEST / (VAL + TEST), random_state=seed)

    # Write the split name on those rows.
    patches = patches.copy()
    patches.loc[train_rows, "split"] = "train"
    patches.loc[val_rows, "split"] = "val"
    patches.loc[test_rows, "split"] = "test"
    return patches


def split_by_location(patches: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Split S: send whole locations to train / val / test (40 / 10 / 50).

    Returns the same rows with a new "split" column.

    Each region is split on its own, so both regions have locations in every
    split (the cross-region experiment T1b needs that). The 40 / 10 / 50 is
    counted in locations, so the share of patches can differ a bit.
    """
    results = []
    for region_name, region_patches in patches.groupby("region"):
        # 1. List this region's locations.
        locations = pd.DataFrame({"pathrow": sorted(region_patches["pathrow"].unique())})
        # 2. Split the list of locations, the same way Split P splits patches.
        locations = split_by_patch(locations, seed)
        # 3. Give each patch the split of its location.
        results.append(region_patches.merge(locations, on="pathrow"))
    return pd.concat(results)


if __name__ == "__main__":
    patches = list_patches(DATA_FOLDER)
    SPLITS_FOLDER.mkdir(parents=True, exist_ok=True)

    # Split P: one file per mask source, with only the patches that have a
    # mask from that source (as in the original).
    mask_columns = [column for column in patches.columns if column.startswith("mask_")]
    for column in mask_columns:
        source = column.removeprefix("mask_")
        patches_with_mask = patches[patches[column].notna()]
        split_p = split_by_patch(patches_with_mask)
        split_p.to_csv(SPLITS_FOLDER / f"patch_{source}.csv", index=False)

    # Split S: one file with every patch.
    split_s = split_by_location(patches)
    split_s.to_csv(SPLITS_FOLDER / "scene.csv", index=False)

    print(f"{len(patches)} patches, {patches['pathrow'].nunique()} locations")
    print("Split S, patches per region and split:")
    print(pd.crosstab(split_s["region"], split_s["split"], margins=True))
