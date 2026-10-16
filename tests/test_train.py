import sys

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from activefire_torch import train
from activefire_torch.evaluate import evaluate_run
from activefire_torch.models import UNet
from activefire_torch.train import ADAM_EPS, LEARNING_RATE, fit, read_config, train_one_epoch, validate
from helpers import write_tif


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


def test_fit_stops_early_and_saves_the_last_epoch(tmp_path):
    # Validation masks are the opposite of the training masks, so the better
    # the model learns, the higher the validation loss.
    images, masks = fake_batch()
    opposite = DataLoader(TensorDataset(images, 1 - masks), batch_size=2)
    torch.manual_seed(0)
    model = UNet(in_channels=3, base_filters=8)
    history = fit(model, fake_loader(), opposite, torch.device("cpu"), tmp_path, patience=2)
    val_losses = [row["val_loss"] for row in history]
    # Epoch 1 sets the best loss; epochs 2 and 3 are worse; stop after epoch 3.
    assert len(history) == 3
    assert val_losses[1] > val_losses[0] and val_losses[2] > val_losses[0]

    # The saved weights are the last epoch's, as in the original.
    saved = torch.load(tmp_path / "weights.pt")
    for name, value in model.state_dict().items():
        assert torch.equal(saved[name], value), name
    assert pd.read_csv(tmp_path / "history.csv")["epoch"].tolist() == [1, 2, 3]


def test_config_with_a_typo_is_rejected(tmp_path):
    config = tmp_path / "run.yaml"
    config.write_text("split_file: x.csv\nmask_source: voting\nversion: fire_only\nbands: [B7]\npateince: 10\n")
    with pytest.raises(ValueError, match="pateince"):
        read_config(config)


def write_fake_run(folder):
    """8 patches of 32 x 32 with a 4 x 4 fire at a random place (6 train, 2 val,
    and the same 2 again as test), and a config file for a 2-epoch run on the CPU."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(8):
        image = rng.integers(0, 10000, size=(10, 32, 32), dtype=np.uint16)
        mask = np.zeros((1, 32, 32), dtype=np.uint8)
        row, column = rng.integers(0, 28, size=2)
        image[:, row : row + 4, column : column + 4] = 60000
        mask[:, row : row + 4, column : column + 4] = 1
        write_tif(folder / f"image{i}.tif", image)
        write_tif(folder / f"mask{i}.tif", mask)
        rows.append({"image": str(folder / f"image{i}.tif"), "scene_id": f"scene{i}",
                     "mask_voting": str(folder / f"mask{i}.tif"), "split": "train" if i < 6 else "val"})
    rows += [{**row, "split": "test"} for row in rows[6:]]
    pd.DataFrame(rows).to_csv(folder / "split.csv", index=False)

    config = folder / "run.yaml"
    config.write_text(
        f"split_file: {folder / 'split.csv'}\nmask_source: voting\nversion: fire_only\nbands: [B7, B6, B2]\n"
        "base_filters: 4\nbatch_size: 2\nmax_epochs: 2\ndevice: cpu\n"
    )
    return config


def run_training(config, seed, runs_folder, monkeypatch):
    """Run the training command and return its history.csv."""
    monkeypatch.setattr(train, "RUNS_FOLDER", runs_folder)
    monkeypatch.setattr(sys, "argv", ["train", str(config), "--seed", str(seed)])
    train.main()
    (run_folder,) = runs_folder.iterdir()
    return pd.read_csv(run_folder / "history.csv")


def test_same_seed_gives_the_same_run(tmp_path, monkeypatch):
    config = write_fake_run(tmp_path)
    first = run_training(config, 1, tmp_path / "first", monkeypatch)
    second = run_training(config, 1, tmp_path / "second", monkeypatch)
    other_seed = run_training(config, 2, tmp_path / "other", monkeypatch)

    losses = ["train_loss", "val_loss"]
    assert first[losses].equals(second[losses])
    assert not first[losses].equals(other_seed[losses])


def test_interrupted_run_resumes_where_it_stopped(tmp_path, monkeypatch):
    images, masks = fake_batch()
    opposite = DataLoader(TensorDataset(images, 1 - masks), batch_size=2)  # validation only gets worse
    real_validate = train.validate
    calls = 0

    def validate_until_power_cut(*args):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("power cut during epoch 3")
        return real_validate(*args)

    monkeypatch.setattr(train, "validate", validate_until_power_cut)
    torch.manual_seed(0)
    with pytest.raises(RuntimeError, match="power cut"):
        fit(UNet(in_channels=3, base_filters=8), fake_loader(), opposite, torch.device("cpu"), tmp_path, patience=2)
    monkeypatch.setattr(train, "validate", real_validate)

    history = fit(UNet(in_channels=3, base_filters=8), fake_loader(), opposite, torch.device("cpu"), tmp_path, patience=2)
    # Epochs 1-2 come from before the cut. Epoch 3 is the second worse epoch in
    # a row only if the early-stopping counter was restored, so the run stops there.
    assert [row["epoch"] for row in history] == [1, 2, 3]
    assert pd.read_csv(tmp_path / "history.csv")["epoch"].tolist() == [1, 2, 3]
    assert not (tmp_path / "checkpoint.pt").exists()


def test_evaluating_a_run_uses_its_training_inputs(tmp_path, monkeypatch):
    # The fake run's test split is the same 2 patches as its validation split,
    # so evaluating the final weights must give exactly the last validation
    # counts, unless bands, scaling or weights were loaded differently.
    config = write_fake_run(tmp_path)
    history = run_training(config, 0, tmp_path / "runs", monkeypatch)
    (run_folder,) = (tmp_path / "runs").iterdir()

    split = pd.read_csv(tmp_path / "split.csv")
    manual = split[split["split"] == "test"].rename(columns={"mask_voting": "mask_manual"})
    manual.to_csv(tmp_path / "manual.csv", index=False)

    results = evaluate_run(run_folder, torch.device("cpu"), num_workers=0, manual_split_file=tmp_path / "manual.csv")
    last_epoch = history.iloc[-1]
    for name in ["tp", "fp", "fn"]:
        assert results["test"][name] == last_epoch[f"val_{name}"], name
    assert results["test"]["average_precision"] == pytest.approx(last_epoch["val_average_precision"])
