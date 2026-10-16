"""Training. Defaults copy the original train.py (docs/paper_notes.md §7)."""

import torch
from torch import nn
from torch.utils.data import DataLoader

from activefire_torch.metrics import PixelCounts, ScoreHistogram

BATCH_SIZE = 16  # as the paper states (D11)
LEARNING_RATE = 0.001  # Keras Adam default, used by the original
ADAM_EPS = 1e-7  # Keras Adam default (PyTorch's is 1e-8)
MAX_EPOCHS = 50
PATIENCE = 5  # stop after this many epochs without a lower validation loss


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


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
) -> list[dict[str, float]]:
    """Train as the original does: Adam and binary cross-entropy, validating
    after every epoch, and stopping once the validation loss has not improved
    for `patience` epochs in a row (Keras EarlyStopping). As in the original,
    the model keeps the weights of the last epoch, not of the best one.

    Returns one row per epoch: epoch, train_loss, and val_<score> for every
    score from validate().
    """
    model.to(device)
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, eps=ADAM_EPS)

    history = []
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, loss_function, optimizer, device)
        scores = validate(model, val_loader, loss_function, device)
        history.append({"epoch": epoch, "train_loss": train_loss, **{f"val_{name}": value for name, value in scores.items()}})
        print(f"epoch {epoch:2d}  train loss {train_loss:.4f}  val loss {scores['loss']:.4f}  val F1 {scores['f1']:.3f}")

        if scores["loss"] < best_val_loss:
            best_val_loss = scores["loss"]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Stopping: no lower validation loss in the last {patience} epochs")
                break
    return history
