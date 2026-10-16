"""Training. Defaults copy the original train.py (docs/paper_notes.md §7)."""

import torch
from torch import nn
from torch.utils.data import DataLoader

from activefire_torch.metrics import PixelCounts, ScoreHistogram

BATCH_SIZE = 16  # as the paper states (D11)
LEARNING_RATE = 0.001  # Keras Adam default, used by the original
ADAM_EPS = 1e-7  # Keras Adam default (PyTorch's is 1e-8)


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
