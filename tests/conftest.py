import mlflow
import pytest

from activefire_torch import tracking


@pytest.fixture(autouse=True)
def temporary_mlflow(tmp_path, monkeypatch):
    """Every test records to its own MLflow folder, never to the real mlruns/."""
    monkeypatch.setattr(tracking, "TRACKING_FOLDER", tmp_path / "mlruns")
    tracking.setup()
    yield
    while mlflow.active_run():  # a failing test can leave a run open
        mlflow.end_run()
