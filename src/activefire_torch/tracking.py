"""Where runs are recorded: MLflow, with its database and files in one local folder.

To browse runs (add --host 0.0.0.0 to open it from another machine):
    uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
"""

from pathlib import Path

import mlflow
import mlflow.artifacts
import yaml

TRACKING_FOLDER = Path("mlruns")  # MLflow's usual name; not in git
EXPERIMENT = "activefire"


def setup() -> None:
    """Point MLflow at TRACKING_FOLDER and select the experiment.

    Set explicitly: by default MLflow writes its files relative to whatever
    folder the command is run from.
    """
    folder = TRACKING_FOLDER.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{folder}/mlflow.db")
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=(folder / "artifacts").as_uri())
    mlflow.set_experiment(EXPERIMENT)


def read_record(run_id: str) -> dict:
    """A run's config.yaml: its settings, code version, split fingerprint, etc.

    Read this, not the run's MLflow parameters: MLflow stores parameters as
    text, so a setting like augment: False would come back as "False", which
    Python treats as true.
    """
    path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="config.yaml")
    return yaml.safe_load(Path(path).read_text())
