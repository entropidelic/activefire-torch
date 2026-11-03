# activefire-torch

A PyTorch reimplementation of Pereira et al. (2021), *Active Fire Detection in
Landsat-8 Imagery: a Large-Scale Dataset and a Deep-Learning Study* (ISPRS
Journal of Photogrammetry and Remote Sensing; original Keras code:
[pereira-gha/activefire](https://github.com/pereira-gha/activefire)).

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```
uv sync
scripts/download_data.sh               # or some parts: subset, manual, regions, weights
uv run splits
```

Data, split files, training runs and evaluation outputs are not in git. Each
run records a fingerprint of the split file it used.

## Commands

Run from the repo root.

Train a model with the settings in a config file. Each run is recorded in
MLflow (settings, per-epoch metrics, example predictions, final weights) and
gets an MLflow run id; `--seed` changes the random seed (default 0).

```
uv run train configs/r1_unet_light_voting.yaml [--seed 1]
```

Continue an interrupted run from its last finished epoch.

```
uv run train --resume <run id>
```

Evaluate a trained run on its own test split and on the manually annotated set.
The results are added to the run in MLflow, as `test_*` and `manual_*` metrics.

```
uv run evaluate --run <run id>
```

Evaluate the authors' trained Keras weights on the manually annotated set
(`--base-filters 64` for U-Net instead of U-Net-Light, `--bands all` for the
10-band model).

```
uv run evaluate --keras-weights <file.h5> [--base-filters 64] [--bands all]
```

Evaluate one fire-detection algorithm's own masks (schroeder, murphy or
kumar-roy) against the manually annotated set.

```
uv run evaluate --algorithm schroeder
```

Browse and compare runs, their training curves and example predictions in
MLflow, at http://localhost:5000. To open it from another machine, add `--host
0.0.0.0` and go to http://<this machine's IP>:5000 (a name like `mac.local`
also needs `--allowed-hosts mac.local`). `uv run tracking` is MLflow's own
command (`mlflow`), pointed at this repo's runs; it takes any of its
subcommands, e.g. `uv run tracking runs list --experiment-id 1`.

```
uv run tracking ui
```

Run the tests.

```
uv run pytest
```

## Relevant locations

| Path | What |
|---|---|
| `configs/` | one YAML file per kind of training run |
| `data/raw/` | downloaded images, masks and the authors' weights |
| `data/splits/` | split files written by `uv run splits` |
| `mlruns/` | MLflow's database and files: every run's settings, metrics, example predictions and final weights |
| `runs/<run id>/` | a run's working files: the checkpoint while training, `history.csv`, and the final weights |
| `results/figures/` | figures |
| `results/r0/` | evaluation outputs of the authors' models |
