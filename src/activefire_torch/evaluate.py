"""Evaluate at the paper's threshold (0.25):
- one of our trained runs, on its own test split (paper Table 2), also in its
  full version when that adds patches, and on the manually annotated set
  (Table 3);
- the authors' trained weights (R0) or one algorithm's own masks (the "—" rows
  of Table 3), on the manually annotated set.

A patch has a manual mask file only if it contains fire; every other patch,
including the 3 fire-free scenes, is evaluated with an all-zero mask.

A trained run is also scored at a tuned threshold: the one with the best F1
on its validation split. Several runs that differ only in seed can be
evaluated together, as the average of their probabilities.

A trained run's results are added to its MLflow run, as final_val_<score>,
test_<score>, test_full_<score> and manual_<score> metrics (<score> at 0.25,
tuned_<score> at the tuned threshold) and tuned_threshold. An average gets
an MLflow run of its own, tagged with the runs it averages.

Run from the repo root, after `uv run splits`:
    uv run evaluate --run <run id>
    uv run evaluate --run <run id> <run id> <run id>
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


class AverageOfModels(nn.Module):
    """Several models as one, whose probability is the average of theirs."""

    def __init__(self, models: list[nn.Module]) -> None:
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        probability = torch.stack([torch.sigmoid(model(image)) for model in self.models]).mean(dim=0)
        # Back to a logit, which is what a model returns. A probability of
        # exactly 0 or 1 gives an infinite logit; the scores handle that.
        return torch.logit(probability)


class Scorer:
    """The scores of one subset of a dataset's patches, added batch by batch.

    chosen  one True / False per row of rows: the patches to score.
    """

    def __init__(self, rows: pd.DataFrame, mask_column: str, chosen: np.ndarray) -> None:
        chosen = np.asarray(chosen, dtype=bool)
        # Fire-free scenes are judged within the subset: a scene whose only
        # fire is outside the subset counts as fire-free here.
        fire_free = np.zeros(len(rows), dtype=bool)
        fire_free[chosen] = fire_free_scenes(rows[chosen], mask_column).to_numpy()
        self.chosen = torch.tensor(chosen)
        self.fire_free = torch.tensor(fire_free)
        self.counts = PixelCounts()
        self.fire_free_counts = PixelCounts()
        self.histogram = ScoreHistogram()
        self.fire_free_histogram = ScoreHistogram()  # for false alarms at another threshold
        self.seconds_per_patch = float("nan")  # set when the pass is over

    def add(self, logits: torch.Tensor, masks: torch.Tensor, batch_rows: slice) -> None:
        """A batch's logits and masks; batch_rows: where its patches are in rows."""
        probabilities = torch.sigmoid(logits)
        in_subset = self.chosen[batch_rows]
        self.counts.add(probabilities[in_subset], masks[in_subset])
        self.histogram.add(logits[in_subset], masks[in_subset])
        in_fire_free_scene = self.fire_free[batch_rows]
        self.fire_free_counts.add(probabilities[in_fire_free_scene], masks[in_fire_free_scene])
        self.fire_free_histogram.add(logits[in_fire_free_scene], masks[in_fire_free_scene])

    def scores(self, tuned_threshold: float | None = None) -> dict:
        """Scores at the paper's threshold and average precision. With
        tuned_threshold (from ScoreHistogram.best_threshold() on validation
        data), also the same scores at that threshold, as tuned_<score>."""
        scores = {
            **self.counts.scores(),
            "average_precision": self.histogram.average_precision(),
            "fire_free_scenes_fp": self.fire_free_counts.fp,
            "seconds_per_patch": self.seconds_per_patch,
        }
        if tuned_threshold is not None:
            tuned = self.histogram.scores_at(tuned_threshold)
            tuned["fire_free_scenes_fp"] = self.fire_free_histogram.scores_at(tuned_threshold)["fp"]
            scores.update({f"tuned_{name}": value for name, value in tuned.items()})
        return scores


def score_subsets(
    model: nn.Module, dataset: FireDataset, device: torch.device, num_workers: int, subsets: dict[str, np.ndarray]
) -> dict[str, Scorer]:
    """Run the model over the dataset once, in order, scoring each subset of
    its patches.

    subsets  name -> one True / False per row of dataset.rows: the patches
             that subset is scored on.

    Returns name -> that subset's Scorer. seconds_per_patch is the same for
    all: the time of the one pass, per patch of the dataset, including
    reading the files.
    """
    scorers = {name: Scorer(dataset.rows, dataset.mask_column, chosen) for name, chosen in subsets.items()}

    model.to(device).eval()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
    start_time = time.perf_counter()
    with torch.inference_mode():
        for batch_number, (images, masks) in enumerate(loader):
            logits = model(images.to(device)).cpu()
            first_row = batch_number * BATCH_SIZE
            for scorer in scorers.values():
                scorer.add(logits, masks, slice(first_row, first_row + len(images)))
    seconds = time.perf_counter() - start_time

    for scorer in scorers.values():
        scorer.seconds_per_patch = seconds / len(dataset)
    return scorers


def evaluate_model(model: nn.Module, dataset: FireDataset, device: torch.device, num_workers: int) -> dict:
    """Run the model over the dataset, in order, and score its probabilities.
    seconds_per_patch includes reading the files."""
    everything = np.ones(len(dataset), dtype=bool)
    return score_subsets(model, dataset, device, num_workers, {"all": everything})["all"].scores()


# Settings that runs must share to be averaged: they fix what the models are
# given and what they are scored on. Runs that differ only in seed share them.
SAME_FOR_AVERAGING = ["split_file", "split_file_sha256", "mask_source", "version", "bands", "scaling", "band_stats"]


def evaluate_runs(
    run_ids: list[str], device: torch.device, num_workers: int, manual_split_file: Path = MANUAL_SPLIT_FILE
) -> dict:
    """A trained run's final weights, on its own test split and on the manual
    set. With several runs, the average of their models' probabilities.

    Bands, model size, scaling and band statistics come from the runs'
    config.yaml in MLflow, so the models see their inputs exactly as in
    training.

    Each set is scored at the paper's threshold (0.25) and at a tuned one:
    the threshold with the best F1 on the run's validation split, as
    tuned_<score>. The validation scores themselves are returned as
    "final_val" (for one run, at 0.25 they are its last epoch's).
    """
    records = [tracking.read_record(run_id) for run_id in run_ids]
    record = records[0]
    for run_id, other in zip(run_ids, records):
        different = [name for name in SAME_FOR_AVERAGING if other[name] != record[name]]
        if different:
            raise ValueError(f"Run {run_id} differs from run {run_ids[0]} in {different}; they cannot be averaged")
    if split_fingerprint(record["split_file"]) != record["split_file_sha256"]:
        raise ValueError(f"{record['split_file']} has changed since run {run_ids[0]} was trained")

    models = []
    for run_id, run_record in zip(run_ids, records):
        run_model = UNet(in_channels=len(run_record["bands"]), base_filters=run_record["base_filters"])
        weights = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="weights.pt")
        run_model.load_state_dict(torch.load(weights, map_location="cpu"))
        models.append(run_model)
    model = models[0] if len(models) == 1 else AverageOfModels(models)

    def dataset(split_file: str | Path, split: str, mask_source: str, version: str) -> FireDataset:
        return FireDataset(
            split_file=split_file,
            split=split,
            bands=record["bands"],
            mask_source=mask_source,
            version=version,
            scaling=record["scaling"],
            band_stats=record["band_stats"],
        )

    # The validation split the run stopped on. The threshold is chosen here,
    # never on the test or manual sets it is then applied to.
    val_set = dataset(record["split_file"], "val", record["mask_source"], record["version"])
    everything = np.ones(len(val_set), dtype=bool)
    val_scorer = score_subsets(model, val_set, device, num_workers, {"final_val": everything})["final_val"]
    threshold = val_scorer.histogram.best_threshold()  # None if the validation split has no fire
    results = {"final_val": val_scorer.scores(threshold)}

    # Every test patch, each with the mask of the run's source, or an all-zero
    # mask where that source found no fire.
    full_test_set = dataset(record["split_file"], "test", record["mask_source"], "full")
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
    for name, scorer in score_subsets(model, full_test_set, device, num_workers, subsets).items():
        results[name] = scorer.scores(threshold)

    manual_set = dataset(manual_split_file, "test", "manual", "full")
    everything = np.ones(len(manual_set), dtype=bool)
    manual_scorer = score_subsets(model, manual_set, device, num_workers, {"manual": everything})["manual"]
    results["manual"] = manual_scorer.scores(threshold)

    return {
        **results,
        "tuned_threshold": threshold,
        "runs": run_ids,
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
    what.add_argument("--run", nargs="+", metavar="RUN_ID",
                      help="MLflow run id of one of our trained models; several ids (runs that differ only in "
                           "seed) evaluate the average of their probabilities")
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
        results = evaluate_runs(args.run, get_device(args.device), args.num_workers, args.split_file)
        metrics = {}
        threshold = results["tuned_threshold"]
        if threshold is not None:
            metrics["tuned_threshold"] = threshold
            print(f"Tuned threshold (best F1 on validation): {threshold:.4f}")
        for name in ["final_val", "test", "test_full", "manual"]:
            if name not in results:
                continue
            scores = results[name]
            print(f"{name}: precision {scores['precision']:.4f}  recall {scores['recall']:.4f}  "
                  f"F1 {scores['f1']:.4f}  IoU {scores['iou']:.4f}  AP {scores['average_precision']:.4f}")
            if threshold is not None:
                print(f"{'':{len(name)}}  at the tuned threshold: precision {scores['tuned_precision']:.4f}  "
                      f"recall {scores['tuned_recall']:.4f}  F1 {scores['tuned_f1']:.4f}  "
                      f"IoU {scores['tuned_iou']:.4f}")
            metrics.update({f"{name}_{score}": value for score, value in scores.items()})

        if len(args.run) == 1:
            # Through the client, not mlflow.start_run(run_id=...): closing the run
            # again would overwrite the end time (and so the duration) of its training.
            client = mlflow.MlflowClient()
            for name, value in metrics.items():
                client.log_metric(args.run[0], name, value)
            # A tag, not a parameter: a run can be evaluated again on a newer manual split file.
            client.set_tag(args.run[0], "manual_split_sha256", results["manual_split_sha256"])
        else:
            # An average belongs to none of its runs, so it gets an MLflow run
            # of its own, which names them.
            record = tracking.read_record(args.run[0])
            config_name = Path(record.get("config_file", "runs")).stem
            with mlflow.start_run(run_name=f"{config_name}_average_of_{len(args.run)}") as average_run:
                mlflow.log_params({name: record[name] for name in ["split_file", "mask_source", "version", "bands",
                                                                   "scaling"]})
                mlflow.set_tags({"averaged_runs": " ".join(args.run),
                                 "manual_split_sha256": results["manual_split_sha256"]})
                mlflow.log_metrics(metrics)
                print(f"Recorded as run {average_run.info.run_id} ({average_run.info.run_name})")
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
