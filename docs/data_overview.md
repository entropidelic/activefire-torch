# How the data is handled

A short overview of the path from downloaded files to numbers, and of what
we learned about the data on the way. Details and sources are in
`data_inspection.md` (measurements), `DECISIONS.md` (choices, as Dn) and
`paper_notes.md` (the paper and original code).

```
downloaded files ──► one table of patches ──► splits ──► model input ──► metrics
                          │                      │
                          └── characterization ──┘
```

## 1. What the data is

The dataset of Pereira et al. (2021) cuts Landsat-8 pictures ("scenes",
about 185 km wide) into **patches** of 256 x 256 pixels (30 m per pixel, so
7.7 km wide).

- **Image:** 10 bands of light (B1–B7, B9–B11), whole numbers from 0 to 65,535.
- **Masks:** a 0/1 picture per patch marking fire. There is one mask per
  source:
  - three classic fire-detection algorithms: **Schroeder, Murphy, Kumar-Roy**;
  - **voting** (pixels at least 2 of the 3 algorithms agree on) and
    **intersection** (all 3 agree).
- **Manual set:** 9,044 patches from 13 other scenes where the authors drew
  the fire by hand. It is only for testing, and it is where the paper's
  headline result comes from.

The full dataset has 11 regions (~200 GB). We use three (D8, D17):

| Region | Patches | Scenes | Locations | With a voting mask | On disk |
|---|---|---|---|---|---|
| North_America2 | 3,031 | 369 | 314 | 808 | 4.1 GB |
| Oceania | 2,200 | 229 | 190 | 1,162 | 3.1 GB |
| South_America | 11,875 | 354 | 302 | 8,083 | 17 GB |
| **All** | **17,106** | **952** | **806** | **10,053** | **24 GB** |

## 2. From files to one table (`splits.py`)

Everything is read from the **file names**, e.g.
`LC08_L1TP_124046_20200810_20200810_01_RT_Schroeder_p00367.tif`:
scene, location (`124046` = WRS path 124, row 46), mask source (`Schroeder`;
absent for images) and patch number (`367`).

The result is one row per image patch, with its scene, location, region,
and one column per mask source holding the mask's path, or nothing. A file
with an unknown name is reported, never silently dropped.

**No mask file means no fire.** A source only has a mask file where it found
fire, and none of the ~50,000 mask files is empty, so a missing file is read
as an all-zero mask.

## 3. What the data looks like (characterization)

**Only patches with fire were kept.** Every one of the 17,106 patches has a
mask from at least one algorithm (D16). A whole scene would give ~880 patches;
we have 18 per scene on average. So:
- there are no ordinary fire-free landscapes in the training regions;
- a patch without a voting mask is one where the algorithms did not agree
  on any fire pixel: a disagreement case, not a clean "no fire" example;
- false alarms on ordinary landscape can only be measured on the manual set,
  which keeps every patch of its scenes, including 3 fire-free scenes.

**Fire is extremely rare and small.**
- In the voting training data, 0.018% of pixels are fire: about **1 pixel in
  5,500**, or about 12 pixels per patch.
- Half of all mask files mark **2 pixels or fewer**.
- So accuracy is meaningless (always answering "no fire" gives 99.98%). We
  use precision, recall, F1, IoU and average precision. A random model's
  average precision is about the fire share, ~0.0002.

**The labels are noisy.** The masks are made by algorithms, not people.
- They disagree mostly at the edges of fires (on one large fire: 115, 157 and
  156 pixels).
- Some 1–3 pixel detections show no visible hot spot (a coastline, a
  lakeshore; see `results/figures/patches_and_masks.png`).
- Against the manual set, each algorithm has precision 77–88%.

**Locations repeat.** The 952 scenes cover 806 locations: 146 scenes are a
second picture of the same place on another date. This drives the split
design below.

**The manual set.** Only 100 of its 9,044 patches contain fire; 2,091 are in
the fire-free scenes. 2 of its 13 locations also appear in our training
regions (44 patches; `data_inspection.md`).

## 4. Splitting train / validation / test (`splits.py`)

Two schemes, both 40% / 10% / 50% with seed 42, the original's numbers:

- **Split P (Stage 1, replication):** a copy of the original's
  `split_dataset.py`. Each patch goes to a split at random, separately for
  each mask source and using only patches with that source's mask. Patches
  of the same scene, or the same place on another date, can land in both
  training and test, which can make test scores optimistic.
- **Split S (Stage 2):** whole **locations** go to a split (D15), so a test
  location is never seen in training, on any date. Each region is split on
  its own, so every region is in every split.
  - Voting patches: 3,928 train, 1,145 validation, 4,980 test (D19).
  - Proportions are counted in locations, so patch shares drift a little.
  - The 2 manual-set locations both fell in Split S test, so no Stage 2
    model trains on them.

The split files (`data/splits/*.csv`) are fixed once and shared by every run.
The cross-region experiment T1b selects from them: train on two regions, test
on the third (D20).

## 5. From a patch to model input (`dataset.py`)

For each row of a split file, `FireDataset` returns an image and its mask:

1. **Bands:** picked by name. The paper's model uses B7, B6, B2, the bands
   where fire is bright (B7 is ~4x brighter on fire pixels).
2. **Scaling:** divide by 65,535, as the original. Stage 2 can instead
   standardize each band with averages measured on the training patches only.
3. **Mask:** read from the file, or all zeros if there is none.
4. **Version:** `fire_only` (only patches with a mask from the chosen source,
   as the original) or `full` (all patches).
5. **Augmentation** (Stage 2, training only): random 90° turns and mirroring,
   applied identically to image and mask.

## 6. Model and metrics (`models.py`, `metrics.py`)

- **Model:** the original U-Net rebuilt in PyTorch, matching the original's
  Keras code layer by layer, including the BatchNorm settings and the
  upsampling. Its parameter counts equal the paper's Table 1 exactly.
- **Metrics:** fire pixels are counted (TP, FP, FN) over the whole evaluation
  set and the scores computed once, as the original does. A pixel is fire
  when its probability is above 0.25, the paper's threshold. Average
  precision uses a 10,000-bin histogram of the model's logits, because sorting
  billions of pixels does not fit in memory (within 0.0001 of the exact value
  on real predictions; D22).

## 7. Check: the authors' trained models in our pipeline (R0)

The authors' published weights (voting masks; Google Drive file
`1__LJRdpSgPIRm8XFKzSYsCUFpfdntg8T` from their `download_weights.py`), loaded
into our model and run through our data reading and metrics on all 9,044
manual patches, reproduce the paper's Table 3 to its one reported decimal.
So do the algorithms' own masks. In %, measured on the M3 Max, 2026-10-15:

| Mask, model | Paper P / R / IoU / F | Ours P / R / IoU / F |
|---|---|---|
| Voting, U-Net-Light (3c) | 87.2 / 92.4 / 81.4 / 89.7 | 87.21 / 92.41 / 81.38 / 89.73 |
| Voting, U-Net (3c) | 86.4 / 93.0 / 81.1 / 89.6 | 86.36 / 93.03 / 81.12 / 89.57 |
| Voting, U-Net (10c) | 83.6 / 94.0 / 79.3 / 88.5 | 83.59 / 93.98 / 79.34 / 88.48 |
| Schroeder alone | 88.1 / 70.2 / 64.1 / 78.1 | 88.10 / 70.21 / 64.13 / 78.14 |
| Murphy alone | 76.6 / 96.1 / 74.3 / 85.2 | 76.58 / 96.09 / 74.27 / 85.23 |
| Kumar-Roy alone | 82.3 / 68.4 / 59.7 / 74.7 | 82.29 / 68.44 / 59.65 / 74.73 |

Commands (the result files stay on the machine that made them):
`uv run evaluate --keras-weights data/raw/weights/voting/voting/<model folder>/model_unet_Voting_final_weights.h5`
(add `--base-filters 64` for U-Net, and `--bands all` for 10c), and
`uv run evaluate --algorithm <schroeder | murphy | kumar-roy>`.

**What this shows, and what it does not.** The imported models reproduce the
reported metrics on the manual set. That checks the evaluation path: the
model's structure, the weight conversion, how images are read and scaled,
how manual masks are read, and the metrics. It does not check the training
code (loss, optimizer, data loading for training); the sanity check (D21)
and R1 exercise that. A gap between our trained models and the paper can
come from the training data (3 of 11 regions) and split, from randomness, or
from differences in the training code.
