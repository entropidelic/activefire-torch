# activefire-torch

A PyTorch reimplementation of Pereira et al. (2021), *Active Fire Detection in
Landsat-8 Imagery: a Large-Scale Dataset and a Deep-Learning Study* (ISPRS
Journal of Photogrammetry and Remote Sensing; original Keras code:
[pereira-gha/activefire](https://github.com/pereira-gha/activefire)).
Work in progress.

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

```
uv run train configs/r1_unet_light_voting.yaml [--seed 1]
uv run train --resume runs/<run folder>
uv run evaluate --run runs/<run folder>          # its test split and the manual set
uv run evaluate --keras-weights <file.h5> [--base-filters 64] [--bands all]
uv run evaluate --algorithm schroeder            # an algorithm's own masks
uv run tensorboard --logdir runs
uv run pytest
```

## Relevant locations

| Path | What |
|---|---|
| `configs/` | one YAML file per kind of training run |
| `data/raw/` | downloaded images, masks and the authors' weights |
| `data/splits/` | split files written by `uv run splits` |
| `runs/<run>/` | one folder per training run: settings (`config.yaml`), per-epoch results (`history.csv`), final weights (`weights.pt`), checkpoint while training, TensorBoard logs, and `evaluation.json` after `uv run evaluate --run` |
| `results/figures/` | figures |
| `results/r0/` | evaluation outputs of the authors' models |
