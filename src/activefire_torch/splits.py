"""Assign patches to train / val / test.

Split P (Stage 1) copies the original src/utils/split_dataset.py: patches are
assigned at random. Split S (Stage 2) assigns whole locations (WRS path/row),
so no test location is seen in training, on any date (D15).

Run from the repo root after downloading the regions:
    uv run python -m activefire_torch.splits
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_FOLDER = Path("data/raw/regions")  # one subfolder per region
MANUAL_FOLDER = Path("data/raw/manual_annotations")
SPLITS_FOLDER = Path("data/splits")

# Proportions and seed of the original.
TRAIN, VAL, TEST = 0.4, 0.1, 0.5
SEED = 42

# Mask file tag (lowercased) -> mask source. The 100-patch sample tags Kumar-Roy
# as "GOLI_v2"; the manual annotations are tagged "v1".
MASK_SOURCE_BY_TAG = {
    "schroeder": "schroeder",
    "murphy": "murphy",
    "kumar-roy": "kumar-roy",
    "goli_v2": "kumar-roy",
    "intersection": "intersection",
    "voting": "voting",
    "v1": "manual",
}


def read_file_name(file_name: str | Path) -> dict | None:
    """Parse an image or mask file name; None if it is not a dataset file.

        LC08_L1TP_124046_20200810_20200810_01_RT_Schroeder_p00367.tif

        LC08_L1TP_124046_20200810_20200810_01_RT   scene (first 7 parts)
        124046                                     location: WRS path 124, row 046
        Schroeder                                  mask tag (absent for images)
        p00367                                     patch number within the scene

    -> {"scene_id": "LC08_L1TP_124046_20200810_20200810_01_RT",
        "pathrow": "124046", "patch": 367, "mask_tag": "Schroeder"}
    """
    file_name = Path(file_name)
    parts = file_name.stem.split("_")

    looks_right = (
        file_name.suffix == ".tif"
        and len(parts) >= 8
        and parts[0] == "LC08"
        and parts[-1].startswith("p")
        and parts[-1][1:].isdigit()
    )
    if not looks_right:
        return None

    # The tag may itself contain "_" (e.g. "GOLI_v2").
    tag_parts = parts[7:-1]
    return {
        "scene_id": "_".join(parts[:7]),
        "pathrow": parts[2],
        "patch": int(parts[-1][1:]),
        "mask_tag": "_".join(tag_parts) if tag_parts else None,
    }


def list_patches(data_folder: Path) -> pd.DataFrame:
    """One row per image patch.

    Columns: image (path), scene_id, pathrow, region (first folder under
    data_folder), patch, and one mask_<source> column per mask source found,
    holding the mask path or NaN where that source found no fire.

    Unrecognized files are skipped with a warning, so an unknown mask source
    is not dropped silently.
    """
    images = []
    masks = []
    skipped = []

    for file in sorted(Path(data_folder).rglob("*.tif")):
        name = read_file_name(file)
        if name is None:
            skipped.append(file)
            continue

        if name["mask_tag"] is None:
            images.append(
                {
                    "image": str(file),
                    "scene_id": name["scene_id"],
                    "pathrow": name["pathrow"],
                    "region": file.relative_to(data_folder).parts[0],
                    "patch": name["patch"],
                }
            )
        else:
            source = MASK_SOURCE_BY_TAG.get(name["mask_tag"].lower())
            if source is None:
                skipped.append(file)
                continue
            masks.append({"scene_id": name["scene_id"], "patch": name["patch"], "source": source, "file": str(file)})

    if skipped:
        print(f"Warning: skipped {len(skipped)} files with unknown names, for example {skipped[0]}")

    table = pd.DataFrame(images, columns=["image", "scene_id", "pathrow", "region", "patch"])
    masks = pd.DataFrame(masks, columns=["scene_id", "patch", "source", "file"])

    # A mask belongs to the image with the same scene_id and patch number.
    for source in sorted(masks["source"].unique()):
        source_masks = masks[masks["source"] == source]
        source_masks = source_masks[["scene_id", "patch", "file"]]
        source_masks = source_masks.rename(columns={"file": f"mask_{source}"})
        table = table.merge(source_masks, on=["scene_id", "patch"], how="left")

    return table


def split_by_patch(patches: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Split P: assign each row to train / val / test at random (40 / 10 / 50).

    Same two train_test_split calls as the original. Returns a copy with a
    "split" column.
    """
    train_rows, val_and_test_rows = train_test_split(patches.index, test_size=VAL + TEST, random_state=seed)
    val_rows, test_rows = train_test_split(val_and_test_rows, test_size=TEST / (VAL + TEST), random_state=seed)

    patches = patches.copy()
    patches.loc[train_rows, "split"] = "train"
    patches.loc[val_rows, "split"] = "val"
    patches.loc[test_rows, "split"] = "test"
    return patches


def split_by_location(patches: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Split S: assign whole locations to train / val / test (40 / 10 / 50).

    Each region is split separately, so every region appears in every split
    (needed for T1b). Proportions are counted in locations, so patch shares
    differ. Returns a copy with a "split" column.
    """
    results = []
    for _, region_patches in patches.groupby("region"):
        locations = pd.DataFrame({"pathrow": sorted(region_patches["pathrow"].unique())})
        locations = split_by_patch(locations, seed)
        results.append(region_patches.merge(locations, on="pathrow"))
    # Each region's rows are numbered from 0; renumber to avoid duplicate labels.
    return pd.concat(results, ignore_index=True)


if __name__ == "__main__":
    if not DATA_FOLDER.exists():
        raise SystemExit(f"No regions in {DATA_FOLDER}; run scripts/download_data.sh regions first")
    patches = list_patches(DATA_FOLDER)
    SPLITS_FOLDER.mkdir(parents=True, exist_ok=True)

    # Split P: one file per mask source, with only the patches that have a
    # mask from that source, as in the original.
    mask_columns = [column for column in patches.columns if column.startswith("mask_")]
    for column in mask_columns:
        source = column.removeprefix("mask_")
        patches_with_mask = patches[patches[column].notna()]
        split_p = split_by_patch(patches_with_mask)
        split_p.to_csv(SPLITS_FOLDER / f"patch_{source}.csv", index=False)

    # Split S: one file with every patch.
    split_s = split_by_location(patches)
    split_s.to_csv(SPLITS_FOLDER / "scene.csv", index=False)

    # The manual set is only ever used for testing.
    if MANUAL_FOLDER.exists():
        manual = list_patches(MANUAL_FOLDER)
        manual["split"] = "test"
        manual.to_csv(SPLITS_FOLDER / "manual.csv", index=False)
        print(f"Manual set: {len(manual)} patches, {manual['mask_manual'].notna().sum()} with a manual mask")
    else:
        print(f"No manual set in {MANUAL_FOLDER}; manual.csv not written")

    print(f"{len(patches)} patches, {patches['pathrow'].nunique()} locations")
    print("Split S, patches per region and split:")
    print(pd.crosstab(split_s["region"], split_s["split"], margins=True))
