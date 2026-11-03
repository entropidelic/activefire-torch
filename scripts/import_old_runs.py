"""One-off: import into MLflow the runs trained before the switch to it.

Each of those runs is a folder runs/<name>/ with config.yaml, history.csv,
weights.pt and, once evaluated, evaluation.json. Each becomes an MLflow run
with what train.py and evaluate.py now record, so `uv run evaluate --run
<run id>` works on it and the MLflow UI compares it with new runs. The
example pictures, inside the TensorBoard files, are left behind.

Run from the repo root, once the old runs have finished:
    uv run python scripts/import_old_runs.py runs/<folder> [runs/<folder> ...]
A folder imported before is skipped, so running it again is harmless.
"""

# First: importing our package turns off MLflow's telemetry, which must
# happen before mlflow is imported.
from activefire_torch import tracking  # isort: skip

import argparse
import json
from datetime import datetime
from pathlib import Path

import mlflow
import pandas as pd
import yaml
from mlflow.entities import Metric, Param

from activefire_torch.train import DEFAULT_SETTINGS, REQUIRED_SETTINGS


def import_run(client: mlflow.MlflowClient, experiment_id: str, folder: Path) -> str:
    """Create the MLflow run for one old run folder; returns its run id."""
    record = yaml.safe_load((folder / "config.yaml").read_text())
    history = pd.read_csv(folder / "history.csv")
    # MLflow keeps times in milliseconds. Each epoch is recorded at the time
    # it ended, so the run's start, duration and charts by time are the real ones.
    started = datetime.strptime(record["started"], "%Y%m%d-%H%M%S").timestamp() * 1000
    epoch_ends = (started + history["seconds"].cumsum() * 1000).astype(int)

    run_id = client.create_run(experiment_id, start_time=int(started), run_name=folder.name).info.run_id
    try:
        settings = {name: record[name] for name in REQUIRED_SETTINGS + list(DEFAULT_SETTINGS)}
        metrics = [
            Metric(name, float(row[name]), int(end), int(row["epoch"]))
            for (_, row), end in zip(history.iterrows(), epoch_ends)
            for name in history.columns
            if name != "epoch"
        ]
        client.log_batch(run_id, metrics=metrics, params=[Param(name, str(value)) for name, value in settings.items()])
        for file in ["config.yaml", "history.csv", "weights.pt"]:
            client.log_artifact(run_id, str(folder / file))

        evaluation_file = folder / "evaluation.json"
        if evaluation_file.exists():
            results = json.loads(evaluation_file.read_text())
            evaluated = int(evaluation_file.stat().st_mtime * 1000)
            client.log_batch(run_id, metrics=[
                Metric(f"{name}_{score}", float(value), evaluated, 0)
                for name in ["test", "manual"]
                for score, value in results[name].items()
            ])
            client.set_tag(run_id, "manual_split_sha256", results["manual_split_sha256"])

        client.set_tag(run_id, "imported_from", folder.name)
        client.set_terminated(run_id, "FINISHED", end_time=int(epoch_ends.iloc[-1]))
    except BaseException:
        client.delete_run(run_id)  # no half-imported run left behind
        raise
    return run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folders", type=Path, nargs="+", help="old run folders, e.g. runs/r1_*")
    args = parser.parse_args()

    tracking.setup()
    client = mlflow.MlflowClient()
    experiment = mlflow.get_experiment_by_name(tracking.EXPERIMENT)
    assert experiment is not None  # tracking.setup() creates it
    for folder in args.folders:
        if not (folder / "config.yaml").exists():
            print(f"{folder}: skipped, not an old run folder (no config.yaml)")
            continue
        if not (folder / "weights.pt").exists():
            print(f"{folder}: skipped, not finished (no weights.pt); finish it with the old code first")
            continue
        imported = client.search_runs([experiment.experiment_id], f"tags.imported_from = '{folder.name}'")
        if imported:
            print(f"{folder}: skipped, already imported as run {imported[0].info.run_id}")
            continue
        run_id = import_run(client, experiment.experiment_id, folder)
        evaluated = "with" if (folder / "evaluation.json").exists() else "without"
        print(f"{folder}: imported as run {run_id}, {evaluated} evaluation")


if __name__ == "__main__":
    main()
