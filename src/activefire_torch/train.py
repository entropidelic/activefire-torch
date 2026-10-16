"""Training. Defaults copy the original train.py (docs/paper_notes.md §7).

Run from the repo root:
    uv run train configs/<run>.yaml [--seed 1]
    uv run train --resume runs/<run folder>   # after an interruption
"""

import argparse
import hashlib
import random
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from activefire_torch.dataset import FireDataset, compute_band_stats
from activefire_torch.device import describe_device, get_device
from activefire_torch.metrics import PixelCounts, ScoreHistogram
from activefire_torch.models import UNet

BATCH_SIZE = 16  # as the paper states (D11)
LEARNING_RATE = 0.001  # Keras Adam default, used by the original
ADAM_EPS = 1e-7  # Keras Adam default (PyTorch's is 1e-8)
MAX_EPOCHS = 50
PATIENCE = 5  # stop after this many epochs without a lower validation loss
RUNS_FOLDER = Path("runs")
WANDB_PROJECT = "activefire-torch"
EXAMPLES_EVERY = 5  # epochs between example pictures in TensorBoard (and on epoch 1)

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
    "wandb": False,  # also send the TensorBoard log to Weights & Biases
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
    patches = 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        loss = loss_function(logits, masks)

        optimizer.zero_grad()  # gradients add up across batches unless cleared
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(images)
        patches += len(images)
    return total_loss / patches


def validate(model: nn.Module, loader: DataLoader, loss_function: nn.Module, device: torch.device) -> dict[str, float]:
    """Loss and scores on held-out data, without changing the model."""
    model.eval()  # dropout off; BatchNorm uses its running averages
    counts = PixelCounts()
    histogram = ScoreHistogram()
    total_loss = 0.0
    patches = 0
    with torch.inference_mode():  # no gradients needed
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            total_loss += loss_function(logits, masks).item() * len(images)
            patches += len(images)

            probabilities = torch.sigmoid(logits)
            counts.add(probabilities, masks)
            histogram.add(probabilities, masks)
    return {"loss": total_loss / patches, **counts.scores(), "average_precision": histogram.average_precision()}


def example_picture(image: torch.Tensor, mask: torch.Tensor, probability: torch.Tensor) -> torch.Tensor:
    """One strip of three pictures [3, H, 3W]: the image's first three bands as
    red, green, blue, the true mask, and the predicted probability (white = fire)."""
    # Stretch each band so its darkest 2% is black and brightest 2% white. Fire is
    # far brighter than anything else, so it saturates instead of darkening the rest.
    color = image[:3]
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
    examples: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> list[dict]:
    """Train as the original does: Adam and binary cross-entropy, validating
    after every epoch, and stopping once the validation loss has not improved
    for `patience` epochs in a row (Keras EarlyStopping). As in the original,
    the model keeps the weights of the last epoch, not of the best one.

    Writes to run_folder:
        history.csv   one row per epoch, rewritten every epoch
        checkpoint.pt everything needed to continue, rewritten every epoch;
                      if it exists when fit() starts, training continues
                      from it. Deleted when the run finishes.
        weights.pt    the final weights (state_dict)
        events.*      TensorBoard log (tensorboard --logdir runs); with
                      examples = (images, masks), also pictures of the
                      model's predictions on them, on epoch 1 and every
                      EXAMPLES_EVERY epochs

    Returns the rows of history.csv: epoch, seconds, train_loss, and
    val_<score> for every score from validate().
    """
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
    # `with` closes the TensorBoard log even if training crashes, so the
    # epochs done so far are kept.
    with SummaryWriter(run_folder) as writer:
        for epoch in range(first_epoch, max_epochs + 1):
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
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", scores["loss"], epoch)
            for name in ["precision", "recall", "f1", "iou", "average_precision"]:
                writer.add_scalar(f"val/{name}", scores[name], epoch)
            writer.add_scalar("seconds_per_epoch", seconds, epoch)
            if examples is not None and (epoch == 1 or epoch % EXAMPLES_EVERY == 0):
                images, masks = examples
                with torch.inference_mode():  # model is in eval mode after validate()
                    probabilities = torch.sigmoid(model(images.to(device))).cpu()
                for i in range(len(images)):
                    writer.add_image(f"examples/{i}", example_picture(images[i], masks[i], probabilities[i]), epoch)
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

            if epochs_without_improvement >= patience:
                print(f"Stopping: no lower validation loss in the last {patience} epochs")
                break

    torch.save(model.state_dict(), run_folder / "weights.pt")
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
    return DEFAULT_SETTINGS | config


def git_commit() -> str:
    """The code version, e.g. "9d1a704", with "+changes" if there are uncommitted edits."""
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    changes = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    return (commit or "unknown") + ("+changes" if changes else "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a U-Net from a config file.")
    parser.add_argument("config", type=Path, nargs="?", help="YAML file in configs/")
    parser.add_argument("--seed", type=int, help="overrides the config's seed")
    parser.add_argument("--resume", type=Path, help="run folder of an interrupted run, to continue it")
    args = parser.parse_args()
    if (args.config is None) == (args.resume is None):
        parser.error("give a config file, or --resume with a run folder (one of the two)")

    if args.resume:
        run_folder = args.resume
        # The settings the run started with, saved in its folder.
        run_record = yaml.safe_load((run_folder / "config.yaml").read_text())
        config = {name: run_record[name] for name in REQUIRED_SETTINGS + list(DEFAULT_SETTINGS)}
    else:
        config = read_config(args.config)
        if args.seed is not None:
            config["seed"] = args.seed
        started = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_folder = RUNS_FOLDER / f"{args.config.stem}_seed{config['seed']}_{started}"
        run_record = {**config, "config_file": str(args.config), "started": started}

    # Seeds every random choice: starting weights, dropout, shuffling and
    # augmentation (also in DataLoader workers). Runs on the Apple GPU are
    # still not bit-for-bit repeatable; differences between seeds are larger.
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    device = get_device(config["device"])
    band_stats = None
    if config["scaling"] == "standardize":
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
    examples = (torch.stack([val[i][0] for i in largest]), torch.stack([val[i][1] for i in largest]))

    if not args.resume:
        run_record.update({
            "code_version": git_commit(),
            # The split files are not in git; this identifies the exact split.
            "split_file_sha256": hashlib.sha256(Path(config["split_file"]).read_bytes()).hexdigest()[:16],
            "device": describe_device(device),
            "train_patches": len(datasets["train"]),
            "val_patches": len(datasets["val"]),
            "band_stats": band_stats,
        })
        run_folder.mkdir(parents=True)
        (run_folder / "config.yaml").write_text(yaml.safe_dump(run_record, sort_keys=False))
    print(f"Run {run_folder}: {len(datasets['train'])} training and {len(datasets['val'])} validation patches "
          f"on {describe_device(device)}")

    if config["wandb"]:
        # Must start before fit() opens the TensorBoard log, so it can copy it.
        wandb.init(project=WANDB_PROJECT, name=run_folder.name, config=run_record, dir=run_folder,
                   sync_tensorboard=True)
    fit(model, train_loader, val_loader, device, run_folder, config["max_epochs"], config["patience"], examples)
    if config["wandb"]:
        wandb.finish()


if __name__ == "__main__":
    main()
