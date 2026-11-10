"""Evaluate at the paper's threshold (0.25):
- one of our trained runs, on its own test split (paper Table 2), also in its
  full version when that adds patches, and on the manually annotated set
  (Table 3);
- the authors' trained weights (R0) or one algorithm's own masks (the "—" rows
  of Table 3), on the manually annotated set.

A patch has a manual mask file only if it contains fire; every other patch,
including the 3 fire-free scenes, is evaluated with an all-zero mask.

A trained run's results are added to its MLflow run, as test_<score>,
test_full_<score> and manual_<score> metrics.

Run from the repo root, after `uv run splits`:
    uv run evaluate --run <run id>
    uv run evaluate --keras-weights <file.h5> [--base-filters 64] [--bands all]
    uv run evaluate --algorithm schroeder
"""

import argparse
import json
import time
from pathlib import Path

import mlflow
import mlflow.artifacts
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from activefire_torch import tracking
from activefire_torch.dataset import BANDS_IN_FILE, FireDataset, read_mask, select_rows, split_fingerprint
from activefire_torch.device import DEVICE_CHOICES, describe_device, get_device
from activefire_torch.metrics import PixelCounts, ScoreHistogram
from activefire_torch.models import UNet, load_keras_weights

MANUAL_SPLIT_FILE = Path("data/splits/manual.csv")
BATCH_SIZE = 16  # any size gives the same results; 16 as in training


def fire_free_scenes(rows: pd.DataFrame, mask_column: str = "mask_manual") -> pd.Series:
    """True for patches whose scene has no fire at all in mask_column."""
    scenes_with_fire = set(rows.loc[rows[mask_column].notna(), "scene_id"])
    return ~rows["scene_id"].isin(scenes_with_fire)


def evaluate_model(model: nn.Module, dataset: FireDataset, device: torch.device, num_workers: int) -> dict:
    """Run the model over the dataset, in order, and score its probabilities.
    seconds_per_patch includes reading the files."""
    everything = np.ones(len(dataset), dtype=bool)
    return evaluate_subsets(model, dataset, device, num_workers, {"all": everything})["all"]


def evaluate_subsets(
    model: nn.Module, dataset: FireDataset, device: torch.device, num_workers: int, subsets: dict[str, np.ndarray]
) -> dict[str, dict]:
    """evaluate_model for several subsets of the dataset's patches at once,
    running the model over each patch only once.

    subsets  name -> one True / False per row of dataset.rows: the patches
             that subset is scored on.

    Returns name -> that subset's scores. seconds_per_patch is the same for
    all: the time of the one pass, per patch of the dataset.
    """
    scorers = []
    for chosen in subsets.values():
        chosen = np.asarray(chosen, dtype=bool)
        # Fire-free scenes are judged within the subset: a scene whose only
        # fire is outside the subset counts as fire-free here.
        fire_free = np.zeros(len(dataset), dtype=bool)
        fire_free[chosen] = fire_free_scenes(dataset.rows[chosen], dataset.mask_column).to_numpy()
        scorers.append((torch.tensor(chosen), torch.tensor(fire_free), PixelCounts(), PixelCounts(), ScoreHistogram()))

    model.to(device).eval()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
    start_time = time.perf_counter()
    with torch.inference_mode():
        for batch_number, (images, masks) in enumerate(loader):
            logits = model(images.to(device)).cpu()
            probabilities = torch.sigmoid(logits)

            first_row = batch_number * BATCH_SIZE
            batch_rows = slice(first_row, first_row + len(images))
            for chosen, fire_free, counts, fire_free_counts, histogram in scorers:
                in_subset = chosen[batch_rows]
                counts.add(probabilities[in_subset], masks[in_subset])
                histogram.add(logits[in_subset], masks[in_subset])
                in_fire_free_scene = fire_free[batch_rows]
                fire_free_counts.add(probabilities[in_fire_free_scene], masks[in_fire_free_scene])
    seconds = time.perf_counter() - start_time

    return {
        name: {
            **counts.scores(),
            "average_precision": histogram.average_precision(),
            "fire_free_scenes_fp": fire_free_counts.fp,
            "seconds_per_patch": seconds / len(dataset),
        }
        for name, (_, _, counts, fire_free_counts, histogram) in zip(subsets, scorers)
    }


def evaluate_run(
    run_id: str, device: torch.device, num_workers: int, manual_split_file: Path = MANUAL_SPLIT_FILE
) -> dict:
    """A trained run's final weights, on its own test split and on the manual set.

    Bands, model size, scaling and band statistics come from the run's
    config.yaml in MLflow, so the model sees its inputs exactly as in training.
    """
    record = tracking.read_record(run_id)
    if split_fingerprint(record["split_file"]) != record["split_file_sha256"]:
        raise ValueError(f"{record['split_file']} has changed since run {run_id} was trained")
    model = UNet(in_channels=len(record["bands"]), base_filters=record["base_filters"])
    weights = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="weights.pt")
    model.load_state_dict(torch.load(weights, map_location="cpu"))

    manual_set = FireDataset(
        split_file=manual_split_file,
        split="test",
        bands=record["bands"],
        mask_source="manual",
        version="full",
        scaling=record["scaling"],
        band_stats=record["band_stats"],
    )
    # Every test patch, each with the mask of the run's source, or an all-zero
    # mask where that source found no fire.
    full_test_set = FireDataset(
        split_file=record["split_file"],
        split="test",
        bands=record["bands"],
        mask_source=record["mask_source"],
        version="full",
        scaling=record["scaling"],
        band_stats=record["band_stats"],
    )
    # "test" is the run's own version of the test split. For a fire_only run
    # that is the patches with a mask; the full version adds the patches where
    # only other algorithms found fire (Stage 2's primary metric), reported as
    # "test_full". Both come from one pass over the full set. Split P files
    # hold only patches with a mask from their source, so there the two are
    # the same patches and only "test" is reported.
    everything = np.ones(len(full_test_set), dtype=bool)
    has_mask = full_test_set.rows[full_test_set.mask_column].notna().to_numpy()
    if record["version"] == "fire_only" and not has_mask.all():
        subsets = {"test": has_mask, "test_full": everything}
    else:
        subsets = {"test": everything}
    results = evaluate_subsets(model, full_test_set, device, num_workers, subsets)
    return {
        **results,
        "manual": evaluate_model(model, manual_set, device, num_workers),
        "test_split_sha256": record["split_file_sha256"],
        "manual_split_sha256": split_fingerprint(manual_split_file),
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
    what.add_argument("--run", help="MLflow run id of one of our trained models")
    what.add_argument("--keras-weights", type=Path, help="the authors' .h5 weights file")
    what.add_argument("--algorithm", choices=["schroeder", "murphy", "kumar-roy"], help="score this algorithm's masks")
    parser.add_argument("--base-filters", type=int, help="with --keras-weights: 16 (default) for U-Net-Light, 64 for U-Net")
    parser.add_argument("--bands", nargs="+", help='with --keras-weights: band names (default B7 B6 B2), or "all"')
    parser.add_argument("--split-file", type=Path, default=MANUAL_SPLIT_FILE, help="the manual set's split file")
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    # Worker processes each start by importing torch (seconds on macOS), which
    # costs more than they save on these small files.
    parser.add_argument("--num-workers", type=int, default=0, help="extra processes reading files")
    parser.add_argument("--output", type=Path, help="also save the results to this JSON file")
    args = parser.parse_args()
    if not args.keras_weights and (args.base_filters is not None or args.bands is not None):
        parser.error("--base-filters and --bands only apply to --keras-weights (a run reads them from its config)")

    if args.run:
        tracking.setup()
        results = evaluate_run(args.run, get_device(args.device), args.num_workers, args.split_file)
        # Through the client, not mlflow.start_run(run_id=...): closing the run
        # again would overwrite the end time (and so the duration) of its training.
        client = mlflow.MlflowClient()
        for name in ["test", "test_full", "manual"]:
            if name not in results:
                continue
            scores = results[name]
            print(f"{name}: precision {scores['precision']:.4f}  recall {scores['recall']:.4f}  "
                  f"F1 {scores['f1']:.4f}  IoU {scores['iou']:.4f}  AP {scores['average_precision']:.4f}")
            for score, value in scores.items():
                client.log_metric(args.run, f"{name}_{score}", value)
        # A tag, not a parameter: a run can be evaluated again on a newer manual split file.
        client.set_tag(args.run, "manual_split_sha256", results["manual_split_sha256"])
        output = args.output
    else:
        rows = select_rows(args.split_file, "test", "manual", "full")
        print(f"{len(rows)} patches from {rows['scene_id'].nunique()} scenes, "
              f"{fire_free_scenes(rows).sum()} of them in scenes without fire")
        if args.keras_weights:
            bands: list[str] = args.bands or ["B7", "B6", "B2"]
            if bands == ["all"]:
                bands = list(BANDS_IN_FILE)
            model = UNet(in_channels=len(bands), base_filters=args.base_filters or 16)
            load_keras_weights(model, args.keras_weights)
            device = get_device(args.device)
            dataset = FireDataset(args.split_file, "test", bands, "manual", "full")
            results = evaluate_model(model, dataset, device, args.num_workers)
            results = {"weights": str(args.keras_weights), "bands": bands, "device": describe_device(device), **results}
        else:
            results = {"algorithm": args.algorithm, **evaluate_algorithm(rows, args.algorithm)}
        results["manual_split_sha256"] = split_fingerprint(args.split_file)
        for name, value in results.items():
            print(f"  {name:20s} {value:.4f}" if isinstance(value, float) else f"  {name:20s} {value}")
        output = args.output

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
