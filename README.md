# activefire-torch

A PyTorch reimplementation of Pereira et al. (2021), *Active Fire Detection in
Landsat-8 Imagery: a Large-Scale Dataset and a Deep-Learning Study* (ISPRS
Journal of Photogrammetry and Remote Sensing; original Keras code:
[pereira-gha/activefire](https://github.com/pereira-gha/activefire)).
Work in progress.

## Status

- **Done:** data download, inspection and splits (3 of the dataset's 11
  regions); the dataset; the U-Net, matching the original layer by layer and
  in parameter count; metrics; evaluation; training with checkpoints and resume.
- **R0:** the authors' trained models, converted to PyTorch, reproduce the
  paper's Table 3 on the manually annotated set
  ([docs/data_overview.md](docs/data_overview.md), section 7).
- **In progress:** Stage 1, replicating the paper's training (R1: U-Net-Light
  on voting masks).
- **Next:** Stage 2, controlled changes one at a time (location-based splits,
  augmentation, losses, thresholds, …).

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```
uv sync
scripts/download_data.sh               # or some parts: subset, manual, regions, weights
uv run python -m activefire_torch.splits
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

## Where things are

| Path | What |
|---|---|
| `src/activefire_torch/` | `splits.py`, `dataset.py`, `models.py`, `metrics.py`, `train.py`, `evaluate.py`, `device.py` |
| `configs/` | one YAML file per kind of run |
| `docs/data_overview.md` | how the data is handled, in short |
| `docs/data_inspection.md` | measurements of the data |
| `docs/paper_notes.md` | facts from the paper and the original code |
| `docs/DECISIONS.md` | every choice, with its reasons (D1, D2, …) |
| `notebooks/01_data_inspection.ipynb` | a first look at the data |
| `runs/<run>/` | each run's settings (`config.yaml`), per-epoch results (`history.csv`), weights and logs; not in git |
