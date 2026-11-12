# Ideas for improving the scores

Small changes that might raise the model's scores, from a literature search
on 2026-11-09. **None of them has been tested here.** Each is a suggestion
with the evidence found for it; the numbers are other people's, on other
data and splits.

Each idea is meant to be tried as an ablation: R1 with only that one thing
changed, same seeds, compared on validation against the seed-to-seed spread
(as `r1_augment.yaml` and `t1_location_split.yaml` already do).

## 1. Near-infrared instead of Blue: `bands: [B7, B6, B5]`

The smallest change with published evidence: one line in a config.

- **Evidence.** Land8Fire (Tran et al., 2025) trained a U-Net with
  cross-entropy on hand-validated labels derived from this dataset, changing
  only the input bands (their Table 5, in %):

  | Bands | F1 | Recall | Precision | IoU |
  |---|---|---|---|---|
  | B5, B6, B7 | 96.99 | 97.24 | 96.73 | 94.15 |
  | B4, B5, B6, B7 | 96.39 | 95.71 | 97.39 | 93.31 |
  | all 10 | 96.17 | 98.89 | 93.59 | 92.62 |
  | B2, B6, B7 (the paper's) | 96.00 | 96.53 | 95.48 | 92.31 |
  | B6, B7 | 94.98 | 93.52 | 96.49 | 90.44 |

- **Why it could help.** Schroeder and Murphy both test the B7 / B5 ratio
  (`paper_notes.md` §4), so the voting masks depend on a band the 3-channel
  model never sees. Blue carries almost nothing about fire: fire pixels are
  1.07x brighter in the visible bands, against 4x in B7 (`data_inspection.md`).
- **Why it might not.**
  - Land8Fire's table is one fold, with no repeated seeds, on patches that
    overlap by 220 pixels, taken from 26 images with mostly large fires.
  - The original paper's 10-band model, which includes B5, was not better
    on the manual set: F 88.5 against 89.7 for U-Net-Light (3c)
    (`paper_notes.md` §10).
- **Expected gain:** between nothing and about one F1 point.
- **Note:** the example pictures fall back to one band in gray when B7, B6
  and B2 are not all used (`train.py`).

## 2. Choose the threshold on validation (no retraining)

Already planned as experiment E6.

- **Why it could help.** The paper's voting models have recall well above
  precision at 0.25: U-Net-Light (3c) has P 90.2 / R 96.5 on the test split
  and P 87.2 / R 92.4 on the manual set (`paper_notes.md` §10). That
  suggests 0.25 is too low for the best F1.
- **How.** `ScoreHistogram` (`metrics.py`) already counts fire and
  background pixels per logit bin, which is enough to read off the
  threshold with the best validation F1.
- **Evidence:** none found for this dataset beyond the numbers above.

## 3. Average the seeds' probabilities (no retraining)

Average the probabilities of the runs that differ only in seed, then
threshold. Usually a small, reliable gain; no evidence found for this
dataset specifically.

## How to run them

The code for all three is in place (2026-11-12); none has been run on real
data yet.

| Idea | Command | Compare |
|---|---|---|
| 1. Band swap | `uv run train configs/r1_nir.yaml --seed <n>`, with R1's seeds | `final_val_f1` against R1's, seed by seed |
| 2. Tuned threshold | `uv run evaluate --run <run id>`, on any trained run | `final_val_tuned_f1` against `final_val_f1` |
| 3. Seed average | `uv run evaluate --run <id> <id> <id>` | the average's `final_val_f1` against the mean of the single runs' |

- Decide on the validation scores (`final_val_*`). Look at `test_*` and
  `manual_*` once, at the end.
- The threshold is chosen on the validation split of whatever is evaluated,
  so each band choice and each average gets its own.
- Tuned scores are counted from the logit bins of the average-precision
  histogram, so the tuned threshold is always a bin edge (`metrics.py`).
- An average is recorded as its own MLflow run, named
  `<config>_average_of_<n>` and tagged `averaged_runs`.

## What the literature says to skip

- **Focal loss.** In Land8Fire (Table 4, one fold) it lowered the U-Net's F1
  from 95.63 with cross-entropy to 92.62 at best (gamma 2), by missing
  pixels of large fires.
- **Transformers.** TransUNet (F 87.66) and Swin-Unet (F 90.24) scored below
  a plain U-Net in "Fighting Fires from Space" (2025). A 2026 comparison
  found U-Net the most robust of U-Net, DeepLabV3+ and SegFormer across fire
  sizes, and SWIR2 the most useful band.
- **The improved U-Net of "Fighting Fires from Space"** (Dice loss,
  augmentation, AdamW with a one-cycle schedule, dilated convolutions). It
  reports a gain but changes everything at once, and the authors state that
  they ran no ablation, so no single change can be credited.
- **AFI (B7 / B2 as a fourth input)**, from Rostami et al. (2022). Reported
  to reduce missed fire pixels, but only together with a different
  architecture; not isolated here either.

## The limit of small changes

We train on 3 of 11 regions. If our runs are far below the authors' 89.7 F
on the manual set, the gap is most likely the amount of training data, which
none of the ideas above replaces.

## Sources

- Tran et al. (2025). *Land8Fire.* Remote Sensing 17(16), 2776.
  doi:10.3390/rs17162776. https://www.mdpi.com/2072-4292/17/16/2776
- *Fighting Fires from Space: Leveraging Vision Transformers for Enhanced
  Wildfire Detection and Characterization* (2025). arXiv:2504.13776.
- Rostami et al. (2022). *Active Fire Detection from Landsat-8 Imagery Using
  Deep Multiple Kernel Learning.* Remote Sensing 14(4), 992.
  doi:10.3390/rs14040992.
- *Scale-based Approach for Active Wildfire Segmentation on Satellite
  Imagery* (2026). arXiv:2609.01392. Read from the abstract only.
