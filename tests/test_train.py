import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from activefire_torch.models import UNet
from activefire_torch.train import ADAM_EPS, LEARNING_RATE, fit, train_one_epoch, validate


def fake_batch():
    """4 patches of 3 bands, 32 x 32, with a bright 4 x 4 fire in each."""
    torch.manual_seed(0)
    images = torch.rand(4, 3, 32, 32) * 0.2
    masks = torch.zeros(4, 1, 32, 32)
    images[:, :, 10:14, 10:14] = 0.9
    masks[:, :, 10:14, 10:14] = 1
    return images, masks


def fake_loader():
    images, masks = fake_batch()
    return DataLoader(TensorDataset(images, masks), batch_size=2)


def test_epochs_lower_the_training_loss():
    torch.manual_seed(0)
    model = UNet(in_channels=3, base_filters=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, eps=ADAM_EPS)
    cpu = torch.device("cpu")
    losses = [train_one_epoch(model, fake_loader(), nn.BCEWithLogitsLoss(), optimizer, cpu) for _ in range(10)]
    # Starts near ln 2 = 0.69: the untrained model says 50% fire everywhere.
    assert losses[0] > 0.6
    assert losses[-1] < losses[0] - 0.1


def test_validate_changes_nothing_and_repeats_exactly():
    torch.manual_seed(0)
    model = UNet(in_channels=3, base_filters=8)
    weights_before = {name: value.clone() for name, value in model.state_dict().items()}

    first = validate(model, fake_loader(), nn.BCEWithLogitsLoss(), torch.device("cpu"))
    second = validate(model, fake_loader(), nn.BCEWithLogitsLoss(), torch.device("cpu"))

    # Same numbers twice: dropout is off. state_dict includes the BatchNorm
    # running averages, so this also checks that validation did not update them.
    assert first == second
    for name, value in model.state_dict().items():
        assert torch.equal(value, weights_before[name]), name


def test_fit_stops_early_when_validation_gets_worse():
    # Validation masks are the opposite of the training masks, so the better
    # the model learns, the higher the validation loss.
    images, masks = fake_batch()
    opposite = DataLoader(TensorDataset(images, 1 - masks), batch_size=2)
    torch.manual_seed(0)
    history = fit(UNet(in_channels=3, base_filters=8), fake_loader(), opposite, torch.device("cpu"), patience=2)
    val_losses = [row["val_loss"] for row in history]
    # Epoch 1 sets the best loss; epochs 2 and 3 are worse; stop after epoch 3.
    assert len(history) == 3
    assert val_losses[1] > val_losses[0] and val_losses[2] > val_losses[0]
