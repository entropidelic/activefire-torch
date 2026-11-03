import os

# MLflow sends anonymous usage data and prints hints for AI agents unless told
# not to. Set here so it applies before any module imports MLflow.
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
