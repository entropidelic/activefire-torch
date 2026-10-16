"""Evaluate on the manually annotated set, as in paper Table 3: either a model
with the authors' trained weights (R0), or one algorithm's own masks (the "—"
rows of Table 3).

A patch has a manual mask file only if it contains fire; every other patch,
including the 3 fire-free scenes, is evaluated with an all-zero mask.

Run from the repo root, after `uv run python -m activefire_torch.splits`:
    uv run python -m activefire_torch.evaluate --keras-weights <file.h5> [--base-filters 64] [--bands all]
    uv run python -m activefire_torch.evaluate --algorithm schroeder
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from activefire_torch.dataset import BANDS_IN_FILE, FireDataset, read_mask, select_rows
from activefire_torch.device import DEVICE_CHOICES, describe_device, get_device
from activefire_torch.metrics import PixelCounts, ScoreHistogram
from activefire_torch.models import UNet, load_keras_weights
from activefire_torch.train import BATCH_SIZE

MANUAL_SPLIT_FILE = Path("data/splits/manual.csv")


def fire_free_scenes(rows: pd.DataFrame) -> pd.Series:
    """True for patches whose scene has no manual fire at all."""
    scenes_with_fire = set(rows.loc[rows["mask_manual"].notna(), "scene_id"])
    return ~rows["scene_id"].isin(scenes_with_fire)


def evaluate_model(model: nn.Module, dataset: FireDataset, device: torch.device, num_workers: int) -> dict:
    """Run the model over the dataset, in order, and score its probabilities.
    seconds_per_patch includes reading the files."""
    in_fire_free_scene = torch.tensor(fire_free_scenes(dataset.rows).to_numpy())
    counts = PixelCounts()
    fire_free_counts = PixelCounts()
    histogram = ScoreHistogram()

    model.to(device).eval()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
    start_time = time.perf_counter()
    with torch.inference_mode():
        for batch_number, (images, masks) in enumerate(loader):
            probabilities = torch.sigmoid(model(images.to(device))).cpu()
            counts.add(probabilities, masks)
            histogram.add(probabilities, masks)

            first_row = batch_number * BATCH_SIZE
            fire_free = in_fire_free_scene[first_row : first_row + len(images)]
            fire_free_counts.add(probabilities[fire_free], masks[fire_free])
    seconds = time.perf_counter() - start_time

    return {
        **counts.scores(),
        "average_precision": histogram.average_precision(),
        "fire_free_scenes_fp": fire_free_counts.fp,
        "seconds_per_patch": seconds / len(dataset),
    }


def evaluate_algorithm(rows: pd.DataFrame, algorithm: str) -> dict:
    """Score an algorithm's masks against the manual masks. A missing file is an all-zero mask."""
    counts = PixelCounts()
    fire_free_counts = PixelCounts()

    for row, in_fire_free_scene in zip(rows.to_dict("records"), fire_free_scenes(rows)):
        truth_path, predicted_path = row["mask_manual"], row[f"mask_{algorithm}"]
        if pd.isna(truth_path) and pd.isna(predicted_path):
            continue  # nothing to count: no fire and no detection
        truth = read_mask(truth_path) if pd.notna(truth_path) else None
        predicted = read_mask(predicted_path) if pd.notna(predicted_path) else None
        if truth is None:
            truth = np.zeros_like(predicted)
        if predicted is None:
            predicted = np.zeros_like(truth)

        truth = torch.from_numpy(truth.astype(np.float32))
        predicted = torch.from_numpy(predicted.astype(np.float32))
        counts.add(predicted, truth)
        if in_fire_free_scene:
            fire_free_counts.add(predicted, truth)

    return {**counts.scores(), "fire_free_scenes_fp": fire_free_counts.fp}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    what = parser.add_mutually_exclusive_group(required=True)
    what.add_argument("--keras-weights", type=Path, help="the authors' .h5 weights file")
    what.add_argument("--algorithm", choices=["schroeder", "murphy", "kumar-roy"], help="score this algorithm's masks")
    parser.add_argument("--base-filters", type=int, default=16, help="16 for U-Net-Light, 64 for U-Net")
    parser.add_argument("--bands", nargs="+", default=["B7", "B6", "B2"], help='band names, or "all" for the 10 bands')
    parser.add_argument("--split-file", type=Path, default=MANUAL_SPLIT_FILE)
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    # Worker processes each start by importing torch (seconds on macOS), which
    # costs more than they save on these small files.
    parser.add_argument("--num-workers", type=int, default=0, help="extra processes reading files")
    parser.add_argument("--output", type=Path, help="also save the results to this JSON file")
    args = parser.parse_args()

    rows = select_rows(args.split_file, "test", "manual", "full")
    print(f"{len(rows)} patches from {rows['scene_id'].nunique()} scenes, "
          f"{fire_free_scenes(rows).sum()} of them in scenes without fire")

    if args.keras_weights:
        bands: list[str] = list(BANDS_IN_FILE) if args.bands == ["all"] else args.bands
        model = UNet(in_channels=len(bands), base_filters=args.base_filters)
        load_keras_weights(model, args.keras_weights)
        device = get_device(args.device)
        dataset = FireDataset(args.split_file, "test", bands, "manual", "full")
        results = evaluate_model(model, dataset, device, args.num_workers)
        results = {"weights": str(args.keras_weights), "bands": bands, "device": describe_device(device), **results}
    else:
        results = {"algorithm": args.algorithm, **evaluate_algorithm(rows, args.algorithm)}

    for name, value in results.items():
        print(f"  {name:20s} {value:.4f}" if isinstance(value, float) else f"  {name:20s} {value}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
