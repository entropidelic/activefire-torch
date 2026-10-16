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

Train a model with the settings in a config file. Each run gets its own folder
in `runs/`; `--seed` changes the random seed (default 0).

```
uv run train configs/r1_unet_light_voting.yaml [--seed 1]
```

Continue an interrupted run from its last finished epoch.

```
uv run train --resume runs/<run folder>
```

Evaluate a trained run on its own test split and on the manually annotated set.
The results are saved to `evaluation.json` in the run folder.

```
uv run evaluate --run runs/<run folder>
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

Watch training curves and example predictions in TensorBoard, at
http://localhost:6006.

```
uv run tensorboard --logdir runs
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
| `runs/<run>/` | one folder per training run: settings (`config.yaml`), per-epoch results (`history.csv`), final weights (`weights.pt`), checkpoint while training, TensorBoard logs, and `evaluation.json` after `uv run evaluate --run` |
| `results/figures/` | figures |
| `results/r0/` | evaluation outputs of the authors' models |
