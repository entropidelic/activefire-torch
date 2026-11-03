"""Where runs are recorded: MLflow, with its database and files in mlruns/ at
the repo root, whichever folder a command runs from.

MLflow's own commands, pointed at those runs:
    uv run tracking ui                       # http://localhost:5000
    uv run tracking ui --host 0.0.0.0        # also http://<this machine's IP>:5000
    uv run tracking runs list --experiment-id 1
"""

import os
from pathlib import Path

import mlflow
import mlflow.artifacts
import yaml
from mlflow.cli import cli

# The repo root: uv installs this package in editable mode, so this file stays
# at src/activefire_torch/ in the repo.
TRACKING_FOLDER = Path(__file__).resolve().parents[2] / "mlruns"  # MLflow's usual name; not in git
EXPERIMENT = "activefire"


def database_uri() -> str:
    return f"sqlite:///{TRACKING_FOLDER}/mlflow.db"


def setup() -> None:
    """Point MLflow at TRACKING_FOLDER and select the experiment.

    Set explicitly: by default MLflow writes its files relative to whatever
    folder the command is run from.
    """
    TRACKING_FOLDER.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(database_uri())
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=(TRACKING_FOLDER / "artifacts").as_uri())
    mlflow.set_experiment(EXPERIMENT)


def read_record(run_id: str) -> dict:
    """A run's config.yaml: its settings, code version, split fingerprint, etc.

    Read this, not the run's MLflow parameters: MLflow stores parameters as
    text, so a setting like augment: False would come back as "False", which
    Python treats as true.
    """
    path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="config.yaml")
    return yaml.safe_load(Path(path).read_text())


def mlflow_command() -> None:
    """`uv run tracking <args>`: the `mlflow <args>` command, on our runs.
    `mlflow ui` reads a different variable from every other command."""
    os.environ["MLFLOW_TRACKING_URI"] = database_uri()
    os.environ["MLFLOW_BACKEND_STORE_URI"] = database_uri()
    cli(prog_name="uv run tracking")
