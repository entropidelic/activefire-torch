import sys

import mlflow
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from activefire_torch import evaluate, tracking, train
from activefire_torch.dataset import FireDataset
from activefire_torch.evaluate import AverageOfModels, evaluate_model, evaluate_runs
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
    with mlflow.start_run():
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


def test_run_cut_after_its_last_epoch_does_not_train_another(tmp_path, monkeypatch):
    images, masks = fake_batch()
    opposite = DataLoader(TensorDataset(images, 1 - masks), batch_size=2)  # validation only gets worse
    real_log_metrics = mlflow.log_metrics
    calls = 0

    def log_metrics_until_power_cut(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("power cut after epoch 3's checkpoint")
        return real_log_metrics(*args, **kwargs)

    # Epoch 3 is the second worse epoch in a row, so the run would stop after
    # it. It is cut between saving that epoch's checkpoint and stopping.
    monkeypatch.setattr(mlflow, "log_metrics", log_metrics_until_power_cut)
    torch.manual_seed(0)
    with pytest.raises(RuntimeError, match="power cut"):
        with mlflow.start_run():
            fit(UNet(in_channels=3, base_filters=8), fake_loader(), opposite, torch.device("cpu"), tmp_path, patience=2)
    monkeypatch.setattr(mlflow, "log_metrics", real_log_metrics)
    after_epoch_3 = torch.load(tmp_path / "checkpoint.pt")["model"]

    with mlflow.start_run():
        history = fit(UNet(in_channels=3, base_filters=8), fake_loader(), opposite, torch.device("cpu"), tmp_path,
                      patience=2)
    assert [row["epoch"] for row in history] == [1, 2, 3]
    saved = torch.load(tmp_path / "weights.pt")
    for name, value in after_epoch_3.items():
        assert torch.equal(saved[name], value), name


@pytest.mark.parametrize(
    "extra, problem",
    [("pateince: 10", "pateince"), ("max_epochs: 0", "max_epochs"), ("bands: []", "bands is empty")],
)
def test_bad_config_is_rejected(tmp_path, extra, problem):
    config = tmp_path / "run.yaml"
    settings = "split_file: x.csv\nmask_source: voting\nversion: fire_only\n"
    if not extra.startswith("bands"):
        settings += "bands: [B7]\n"
    config.write_text(settings + extra + "\n")
    with pytest.raises(ValueError, match=problem):
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
        with mlflow.start_run():
            fit(UNet(in_channels=3, base_filters=8), fake_loader(), opposite, torch.device("cpu"), tmp_path, patience=2)
    monkeypatch.setattr(train, "validate", real_validate)

    with mlflow.start_run():
        history = fit(UNet(in_channels=3, base_filters=8), fake_loader(), opposite, torch.device("cpu"), tmp_path,
                      patience=2)
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

    run_id = run_folder.name  # the working folder is named after the MLflow run
    results = evaluate_runs([run_id], torch.device("cpu"), num_workers=0, manual_split_file=tmp_path / "manual.csv")
    last_epoch = history.iloc[-1]
    for name in ["tp", "fp", "fn"]:
        assert results["test"][name] == last_epoch[f"val_{name}"], name
    assert results["test"]["average_precision"] == pytest.approx(last_epoch["val_average_precision"])


def test_one_pass_scores_test_and_test_full_as_two_passes_would(tmp_path, monkeypatch):
    config = write_fake_run(tmp_path)
    # Add two test patches without a voting mask: the first two training
    # images again. The test split is now 2 patches with a mask and 2 without.
    split = pd.read_csv(tmp_path / "split.csv")
    without_mask = split.iloc[:2].assign(split="test", mask_voting=None)
    # Mixed in between the patches with a mask, to check the rows are matched up.
    split = pd.concat([split.iloc[:-1], without_mask, split.iloc[-1:]], ignore_index=True)
    split.to_csv(tmp_path / "split.csv", index=False)
    run_training(config, 0, tmp_path / "runs", monkeypatch)
    (run_folder,) = (tmp_path / "runs").iterdir()

    split[(split["split"] == "test") & split["mask_voting"].notna()].rename(
        columns={"mask_voting": "mask_manual"}
    ).to_csv(tmp_path / "manual.csv", index=False)
    cpu = torch.device("cpu")
    results = evaluate_runs([run_folder.name], cpu, num_workers=0, manual_split_file=tmp_path / "manual.csv")

    model = UNet(in_channels=3, base_filters=4)
    model.load_state_dict(torch.load(run_folder / "weights.pt"))
    for name, version, patches in [("test", "fire_only", 2), ("test_full", "full", 4)]:
        dataset = FireDataset(tmp_path / "split.csv", "test", ["B7", "B6", "B2"], "voting", version)
        assert len(dataset) == patches
        separate = evaluate_model(model, dataset, cpu, num_workers=0)
        for score in separate:
            if score != "seconds_per_patch":
                assert results[name][score] == pytest.approx(separate[score], nan_ok=True), (name, score)
    # The patches without a mask add no fire, only possible false alarms.
    assert results["test_full"]["tp"] == results["test"]["tp"]
    assert results["test_full"]["fn"] == results["test"]["fn"]
    assert results["test_full"]["fp"] >= results["test"]["fp"]


def test_resume_refuses_a_changed_split_and_keeps_the_device_setting(tmp_path, monkeypatch):
    config = write_fake_run(tmp_path)
    run_training(config, 0, tmp_path / "runs", monkeypatch)
    (run_folder,) = (tmp_path / "runs").iterdir()
    run_id = run_folder.name  # the working folder is named after the MLflow run
    record = tracking.read_record(run_id)
    # "device" must stay a valid setting (e.g. not "cuda (NVIDIA ...)"), so resume can use it.
    assert record["device"] == "cpu"

    # Recreate the split file with one patch moved from train to val.
    split = pd.read_csv(tmp_path / "split.csv")
    split.loc[0, "split"] = "val"
    split.to_csv(tmp_path / "split.csv", index=False)

    monkeypatch.setattr(sys, "argv", ["train", "--resume", run_id])
    with pytest.raises(SystemExit, match="has changed"):
        train.main()
    with pytest.raises(ValueError, match="has changed"):
        evaluate_runs([run_id], torch.device("cpu"), num_workers=0)


def test_resume_refuses_a_started_run_without_its_checkpoint(tmp_path, monkeypatch):
    config = write_fake_run(tmp_path)
    real_validate = train.validate
    calls = 0

    def validate_until_power_cut(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("power cut during epoch 2")
        return real_validate(*args)

    monkeypatch.setattr(train, "validate", validate_until_power_cut)
    with pytest.raises(RuntimeError, match="power cut"):
        run_training(config, 0, tmp_path / "runs", monkeypatch)
    monkeypatch.setattr(train, "validate", real_validate)
    (run_folder,) = (tmp_path / "runs").iterdir()
    run_id = run_folder.name  # the working folder is named after the MLflow run

    # Epoch 1 is recorded in MLflow. Without the checkpoint, resuming would
    # train epoch 1 again and record it a second time.
    monkeypatch.setattr(sys, "argv", ["train", "--resume", run_id])
    (run_folder / "checkpoint.pt").rename(tmp_path / "checkpoint.pt")
    with pytest.raises(SystemExit, match="checkpoint.pt is missing"):
        train.main()

    # With it, the run continues from epoch 2.
    (tmp_path / "checkpoint.pt").rename(run_folder / "checkpoint.pt")
    train.main()
    assert pd.read_csv(run_folder / "history.csv")["epoch"].tolist() == [1, 2]
    assert len(mlflow.MlflowClient().get_metric_history(run_id, "train_loss")) == 2


def write_manual_split(folder):
    """A manual set for the fake run: its test patches, with their masks as manual masks."""
    split = pd.read_csv(folder / "split.csv")
    manual = split[split["split"] == "test"].rename(columns={"mask_voting": "mask_manual"})
    manual.to_csv(folder / "manual.csv", index=False)
    return folder / "manual.csv"


def test_evaluation_tunes_the_threshold_on_validation(tmp_path, monkeypatch):
    config = write_fake_run(tmp_path)
    history = run_training(config, 0, tmp_path / "runs", monkeypatch)
    (run_folder,) = (tmp_path / "runs").iterdir()
    results = evaluate_runs([run_folder.name], torch.device("cpu"), num_workers=0,
                            manual_split_file=write_manual_split(tmp_path))

    assert 0 < results["tuned_threshold"] < 1
    val, test = results["final_val"], results["test"]
    # One run's validation scores at 0.25 are its last epoch's.
    for name in ["tp", "fp", "fn"]:
        assert val[name] == history.iloc[-1][f"val_{name}"], name
    # The best threshold for validation cannot score below 0.25 there.
    assert val["tuned_f1"] >= val["f1"]
    # The fake test split is the validation split again, so the threshold
    # chosen on validation gives the test split the same counts.
    for name in ["tuned_tp", "tuned_fp", "tuned_fn", "tuned_fire_free_scenes_fp"]:
        assert test[name] == val[name], name


class Constant(nn.Module):
    """Predicts the same logit everywhere."""

    def __init__(self, logit):
        super().__init__()
        self.logit = logit

    def forward(self, image):
        return torch.full_like(image[:, :1], self.logit)


def test_average_of_models_averages_probabilities():
    low, high = Constant(-2.0), Constant(3.0)
    average = AverageOfModels([low, high])
    probability = torch.sigmoid(average(torch.zeros(1, 3, 4, 4)))
    expected = (torch.sigmoid(torch.tensor(-2.0)) + torch.sigmoid(torch.tensor(3.0))) / 2
    assert probability.shape == (1, 1, 4, 4)
    assert torch.allclose(probability, expected)


def test_evaluating_several_runs_scores_their_average(tmp_path, monkeypatch):
    config = write_fake_run(tmp_path)
    run_training(config, 0, tmp_path / "seed0", monkeypatch)
    run_training(config, 1, tmp_path / "seed1", monkeypatch)
    (folder0,) = (tmp_path / "seed0").iterdir()
    (folder1,) = (tmp_path / "seed1").iterdir()
    run_ids = [folder0.name, folder1.name]
    manual = write_manual_split(tmp_path)
    cpu = torch.device("cpu")

    results = evaluate_runs(run_ids, cpu, num_workers=0, manual_split_file=manual)

    # The same as averaging the two saved models by hand.
    models = []
    for folder in [folder0, folder1]:
        model = UNet(in_channels=3, base_filters=4)
        model.load_state_dict(torch.load(folder / "weights.pt"))
        models.append(model)
    dataset = FireDataset(tmp_path / "split.csv", "test", ["B7", "B6", "B2"], "voting", "fire_only")
    by_hand = evaluate_model(AverageOfModels(models), dataset, cpu, num_workers=0)
    for name in ["tp", "fp", "fn"]:
        assert results["test"][name] == by_hand[name], name
    assert results["test"]["average_precision"] == pytest.approx(by_hand["average_precision"])

    # A run averaged with itself is that run.
    alone = evaluate_runs(run_ids[:1], cpu, num_workers=0, manual_split_file=manual)
    twice = evaluate_runs(run_ids[:1] * 2, cpu, num_workers=0, manual_split_file=manual)
    for name in ["tp", "fp", "fn"]:
        assert twice["test"][name] == alone["test"][name], name

    # The command records an average as an MLflow run of its own.
    monkeypatch.setattr(sys, "argv", ["evaluate", "--run", *run_ids, "--split-file", str(manual), "--device", "cpu"])
    evaluate.main()
    recorded = mlflow.search_runs(filter_string=f"tags.averaged_runs = '{' '.join(run_ids)}'")
    assert len(recorded) == 1
    assert recorded.iloc[0]["metrics.test_tp"] == results["test"]["tp"]
    assert recorded.iloc[0]["metrics.tuned_threshold"] == pytest.approx(results["tuned_threshold"])


def test_runs_with_different_inputs_cannot_be_averaged(tmp_path, monkeypatch):
    config = write_fake_run(tmp_path)
    other_bands = tmp_path / "other_bands.yaml"
    other_bands.write_text(config.read_text().replace("bands: [B7, B6, B2]", "bands: [B7, B6, B5]"))
    run_training(config, 0, tmp_path / "blue", monkeypatch)
    run_training(other_bands, 0, tmp_path / "nir", monkeypatch)
    (blue,) = (tmp_path / "blue").iterdir()
    (nir,) = (tmp_path / "nir").iterdir()
    with pytest.raises(ValueError, match="bands"):
        evaluate_runs([blue.name, nir.name], torch.device("cpu"), num_workers=0,
                      manual_split_file=write_manual_split(tmp_path))
