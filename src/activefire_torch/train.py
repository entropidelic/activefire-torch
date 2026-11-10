"""Training. Defaults copy the original train.py (docs/paper_notes.md §7).

Run from the repo root:
    uv run train configs/<run>.yaml [--seed 1]
    uv run train --resume <run id>   # after an interruption

Each run is recorded in MLflow (tracking.py). Its working files, the
checkpoint and history.csv, are in runs/<run id>/.
"""

import argparse
import random
import subprocess
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from activefire_torch import tracking
from activefire_torch.dataset import FireDataset, compute_band_stats, split_fingerprint
from activefire_torch.device import describe_device, get_device
from activefire_torch.metrics import PixelCounts, ScoreHistogram
from activefire_torch.models import UNet

BATCH_SIZE = 16  # as the paper states
LEARNING_RATE = 0.001  # Keras Adam default, used by the original
ADAM_EPS = 1e-7  # Keras Adam default (PyTorch's is 1e-8)
MAX_EPOCHS = 50
PATIENCE = 5  # stop after this many epochs without a lower validation loss
RUNS_FOLDER = Path("runs")
EXAMPLES_EVERY = 5  # epochs between example pictures (and on epoch 1)

# A config file must set these; see configs/ for examples.
REQUIRED_SETTINGS = ["split_file", "mask_source", "version", "bands"]
# And may set these; anything else is a typo and an error.
DEFAULT_SETTINGS = {
    "base_filters": 16,
    "scaling": "max",
    "augment": False,
    "batch_size": BATCH_SIZE,
    "max_epochs": MAX_EPOCHS,
    "patience": PATIENCE,
    "seed": 0,
    "device": "auto",
    "num_workers": 0,
}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """One pass over the training data, updating the weights after every batch.
    Returns the average loss per patch."""
    model.train()  # dropout on; BatchNorm uses and learns from each batch
    total_loss = 0.0
    patch_count = 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        loss = loss_function(logits, masks)

        optimizer.zero_grad()  # gradients add up across batches unless cleared
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(images)
        patch_count += len(images)
    return total_loss / patch_count


def validate(model: nn.Module, loader: DataLoader, loss_function: nn.Module, device: torch.device) -> dict[str, float]:
    """Loss and scores on held-out data, without changing the model."""
    model.eval()  # dropout off; BatchNorm uses its running averages
    counts = PixelCounts()
    histogram = ScoreHistogram()
    total_loss = 0.0
    patch_count = 0
    with torch.inference_mode():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            total_loss += loss_function(logits, masks).item() * len(images)
            patch_count += len(images)

            probabilities = torch.sigmoid(logits)
            counts.add(probabilities, masks)
            histogram.add(logits, masks)
    return {"loss": total_loss / patch_count, **counts.scores(), "average_precision": histogram.average_precision()}


def example_picture(
    image: torch.Tensor, mask: torch.Tensor, probability: torch.Tensor, display_channels: list[int]
) -> torch.Tensor:
    """One strip of three pictures [3, H, 3W]: the image's display_channels as
    red, green, blue, the true mask, and the predicted probability (white = fire)."""
    # Stretch each band so its darkest 2% is black and brightest 2% white. Fire is
    # far brighter than anything else, so it saturates instead of darkening the rest.
    color = image[display_channels]
    low = color.flatten(1).quantile(0.02, dim=1).view(-1, 1, 1)
    high = color.flatten(1).quantile(0.98, dim=1).view(-1, 1, 1)
    color = ((color - low) / (high - low + 1e-6)).clamp(0, 1)
    return torch.cat([color, mask.expand(3, -1, -1), probability.expand(3, -1, -1)], dim=2)


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    run_folder: Path,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    examples: tuple[torch.Tensor, torch.Tensor, list[int]] | None = None,
) -> list[dict]:
    """Train as the original does: Adam and binary cross-entropy, validating
    after every epoch, and stopping once the validation loss has not improved
    for `patience` epochs in a row (Keras EarlyStopping). As in the original,
    the model keeps the weights of the last epoch, not of the best one.

    Must be called inside an MLflow run (mlflow.start_run). Records there:
        metrics       train and validation loss and scores, per epoch
        pictures      with examples = (images, masks, display_channels), the
                      model's predictions on them, on epoch 1 and every
                      EXAMPLES_EVERY epochs
        history.csv   and weights.pt (the final weights), when training ends

    Working files in run_folder:
        history.csv   one row per epoch, rewritten every epoch
        checkpoint.pt everything needed to continue, rewritten every epoch;
                      if it exists when fit() starts, training continues
                      from it. Deleted when the run finishes.
        weights.pt    the final weights (state_dict)

    Returns the rows of history.csv: epoch, seconds, train_loss, and
    val_<score> for every score from validate().
    """
    if mlflow.active_run() is None:
        # Otherwise MLflow would silently start a run wherever the code runs from.
        raise RuntimeError("fit() must run inside an MLflow run (mlflow.start_run)")
    if (run_folder / "weights.pt").exists():
        raise ValueError(f"{run_folder} is a finished run")
    run_folder.mkdir(parents=True, exist_ok=True)
    model.to(device)
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, eps=ADAM_EPS)

    history: list[dict] = []
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    first_epoch = 1
    checkpoint_file = run_folder / "checkpoint.pt"
    if checkpoint_file.exists():
        # Continue an interrupted run. The random state (shuffling, dropout)
        # is not restored, so the rest of the run is not bit-for-bit what it
        # would have been.
        checkpoint = torch.load(checkpoint_file, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])  # Adam's running averages
        best_val_loss = checkpoint["best_val_loss"]
        epochs_without_improvement = checkpoint["epochs_without_improvement"]
        first_epoch = checkpoint["epoch"] + 1
        history = pd.read_csv(run_folder / "history.csv").to_dict("records")[: checkpoint["epoch"]]
        print(f"Resuming {run_folder} after epoch {checkpoint['epoch']}")

    for epoch in range(first_epoch, max_epochs + 1):
        # Checked before the epoch, not after it: a run interrupted right after
        # its last epoch's checkpoint then stops here when resumed, instead of
        # training one more epoch.
        if epochs_without_improvement >= patience:
            print(f"Stopping: no lower validation loss in the last {patience} epochs")
            break

        start = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, loss_function, optimizer, device)
        scores = validate(model, val_loader, loss_function, device)
        seconds = time.perf_counter() - start

        row = {"epoch": epoch, "seconds": seconds, "train_loss": train_loss}
        row.update({f"val_{name}": value for name, value in scores.items()})
        history.append(row)
        # Written to a temporary file, then renamed: a rename is all or
        # nothing, so an interruption mid-write cannot leave a broken file.
        pd.DataFrame(history).to_csv(run_folder / "history.tmp", index=False)
        (run_folder / "history.tmp").replace(run_folder / "history.csv")

        print(f"epoch {epoch:2d}  train loss {train_loss:.4f}  val loss {scores['loss']:.4f}  "
              f"val F1 {scores['f1']:.3f}  {seconds:.0f} s")

        if scores["loss"] < best_val_loss:
            best_val_loss = scores["loss"]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "epochs_without_improvement": epochs_without_improvement,
        }
        torch.save(checkpoint, run_folder / "checkpoint.tmp")
        (run_folder / "checkpoint.tmp").replace(checkpoint_file)

        # Recorded after the checkpoint: if training is cut in between, the
        # resumed run does not record this epoch a second time. At worst the
        # epoch is missing from MLflow's charts; history.csv still has it.
        mlflow.log_metrics({name: value for name, value in row.items() if name != "epoch"}, step=epoch)
        if examples is not None and (epoch == 1 or epoch % EXAMPLES_EVERY == 0):
            images, masks, display_channels = examples
            with torch.inference_mode():  # model is in eval mode after validate()
                probabilities = torch.sigmoid(model(images.to(device))).cpu()
            for i in range(len(images)):
                picture = example_picture(images[i], masks[i], probabilities[i], display_channels)
                # MLflow takes pictures as [height, width, channels].
                mlflow.log_image(picture.permute(1, 2, 0).numpy(), key=f"examples/{i}", step=epoch)

    # Temporary file, then rename: an interrupted save cannot leave a partial
    # weights.pt, which would mark the run as finished.
    torch.save(model.state_dict(), run_folder / "weights.tmp")
    (run_folder / "weights.tmp").replace(run_folder / "weights.pt")
    mlflow.log_artifact(str(run_folder / "history.csv"))
    mlflow.log_artifact(str(run_folder / "weights.pt"))
    checkpoint_file.unlink()
    return history


def read_config(path: Path) -> dict:
    """The settings in a YAML config file, with defaults for the ones it leaves out."""
    with open(path) as file:
        config = yaml.safe_load(file)
    missing = [name for name in REQUIRED_SETTINGS if name not in config]
    unknown = [name for name in config if name not in REQUIRED_SETTINGS and name not in DEFAULT_SETTINGS]
    if missing or unknown:
        raise ValueError(f"{path}: missing settings {missing}, unknown settings {unknown}")
    config = DEFAULT_SETTINGS | config
    if not config["bands"]:
        raise ValueError(f"{path}: bands is empty")
    for name in ["batch_size", "max_epochs", "patience"]:
        if config[name] < 1:
            raise ValueError(f"{path}: {name} must be at least 1")
    return config


def git_commit() -> str:
    """The code version, e.g. "9d1a704", with "+changes" if there are uncommitted edits."""
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    changes = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    return (commit or "unknown") + ("+changes" if changes else "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a U-Net from a config file.")
    parser.add_argument("config", type=Path, nargs="?", help="YAML file in configs/")
    parser.add_argument("--seed", type=int, help="overrides the config's seed")
    parser.add_argument("--resume", help="MLflow run id of an interrupted run, to continue it")
    args = parser.parse_args()
    if (args.config is None) == (args.resume is None):
        parser.error("give a config file, or --resume with a run id (one of the two)")

    tracking.setup()
    if args.resume:
        # The settings the run started with, recorded in MLflow.
        run_record = tracking.read_record(args.resume)
        config = {name: run_record[name] for name in REQUIRED_SETTINGS + list(DEFAULT_SETTINGS)}
        if split_fingerprint(config["split_file"]) != run_record["split_file_sha256"]:
            raise SystemExit(f"{config['split_file']} has changed since this run started; it cannot be resumed")
        resumed_run = mlflow.get_run(args.resume)
        if resumed_run.info.status == "FINISHED":
            raise SystemExit(f"Run {args.resume} has finished; there is nothing to resume")
        # Without its checkpoint the run would start again from epoch 1 and
        # record those epochs on top of the ones already in MLflow. A run cut
        # during its first epoch has neither, and simply starts again.
        checkpoint_file = RUNS_FOLDER / args.resume / "checkpoint.pt"
        if resumed_run.data.metrics and not checkpoint_file.exists():
            raise SystemExit(f"Run {args.resume} has recorded epochs, but {checkpoint_file} is missing; "
                             "it cannot be resumed")
    else:
        config = read_config(args.config)
        if args.seed is not None:
            config["seed"] = args.seed
        run_record = {**config, "config_file": str(args.config)}

    # Seeds every random choice: starting weights, dropout, shuffling and
    # augmentation (also in DataLoader workers). Runs on the Apple GPU are
    # still not bit-for-bit repeatable; differences between seeds are larger.
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    device = get_device(config["device"])
    band_stats = None
    if args.resume:
        band_stats = run_record["band_stats"]
    elif config["scaling"] == "standardize":
        band_stats = compute_band_stats(config["split_file"], config["bands"], config["mask_source"], config["version"])

    datasets = {}
    for split in ["train", "val"]:
        datasets[split] = FireDataset(
            split_file=config["split_file"],
            split=split,
            bands=config["bands"],
            mask_source=config["mask_source"],
            version=config["version"],
            scaling=config["scaling"],
            band_stats=band_stats,
            augment=config["augment"] and split == "train",
        )
    if len(datasets["train"]) < config["batch_size"]:
        # The last partial batch is skipped, so there would be no batch at all.
        raise SystemExit(f"{len(datasets['train'])} training patches is less than one batch of {config['batch_size']}")
    if len(datasets["val"]) == 0:
        raise SystemExit(f"{config['split_file']} has no validation patches for this mask source and version")
    workers = config["num_workers"]
    # Training: reshuffled every epoch, and the last partial batch is skipped,
    # as in the original (steps_per_epoch = patches // batch size).
    train_loader = DataLoader(
        datasets["train"], batch_size=config["batch_size"], shuffle=True, drop_last=True,
        num_workers=workers, persistent_workers=workers > 0,
    )
    val_loader = DataLoader(
        datasets["val"], batch_size=config["batch_size"], num_workers=workers, persistent_workers=workers > 0,
    )

    model = UNet(in_channels=len(config["bands"]), base_filters=config["base_filters"])

    # Example pictures: the 4 largest fires among the first 100 validation
    # patches (most fires are 1-2 pixels, invisible in a picture).
    val = datasets["val"]
    fire_pixels = {i: int(val[i][1].sum()) for i in range(min(100, len(val)))}
    largest = sorted(fire_pixels, key=lambda i: fire_pixels[i], reverse=True)[:4]
    chosen = [val[i] for i in largest]  # (image, mask) pairs
    # Shown as B7, B6, B2 (the paper's false color), or the first band in gray
    # if those three are not all used.
    bands = config["bands"]
    display_channels = [0, 0, 0]
    if {"B7", "B6", "B2"} <= set(bands):
        display_channels = [bands.index(band) for band in ["B7", "B6", "B2"]]
    examples = (torch.stack([image for image, _ in chosen]), torch.stack([mask for _, mask in chosen]), display_channels)

    if args.resume:
        run = mlflow.start_run(run_id=args.resume)
    else:
        run = mlflow.start_run(run_name=f"{args.config.stem}_seed{config['seed']}")
        run_record.update({
            # MLflow does not record the code version for these commands.
            "code_version": git_commit(),
            "split_file_sha256": split_fingerprint(config["split_file"]),
            "device_used": describe_device(device),  # "device" stays the setting, for resuming
            "train_patches": len(datasets["train"]),
            "val_patches": len(datasets["val"]),
            "band_stats": band_stats,
        })
        mlflow.log_params(config)  # for searching and comparing runs; stored as text
        mlflow.log_dict(run_record, "config.yaml")  # the exact record; read with tracking.read_record()

    with run:
        run_id = run.info.run_id
        print(f"Run {run_id} ({run.info.run_name}): {len(datasets['train'])} training and "
              f"{len(datasets['val'])} validation patches on {describe_device(device)}")
        fit(model, train_loader, val_loader, device, RUNS_FOLDER / run_id, config["max_epochs"], config["patience"],
            examples)


if __name__ == "__main__":
    main()
