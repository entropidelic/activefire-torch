# Paper and original code notes

Facts extracted from the paper and the original implementation. Every fact has a source. Anything that is our interpretation, not stated in a source, is marked **[interpretation]**. Anything the sources do not say is marked **not stated**, together with the default we use (also logged in `DECISIONS.md`).

**Sources**
- **Paper:** Pereira, Fusioka, Nassu, Minetto (2021). *Active Fire Detection in Landsat-8 Imagery: a Large-Scale Dataset and a Deep-Learning Study.* ISPRS Journal of Photogrammetry and Remote Sensing, doi:10.1016/j.isprsjprs.2021.06.002. We read the preprint arXiv:2101.03409v2 (2 Jul 2021). Page numbers refer to that PDF.
- **Code:** https://github.com/pereira-gha/activefire at commit `16ef990` (2025-06-02). Paths below are relative to that repo. The training code has one folder per (mask source × architecture), e.g. `src/train/kumar-roy/unet_16f_2conv_762/`.

---

## 1. Dataset

| Fact | Value | Source |
|---|---|---|
| Total patches | 146,214 (≈192 GB). The abstract rounds this to "over 150,000 … more than 200 GB". | Sec. 1 p.2, Sec. 2.4 p.6, abstract |
| Patch size | 256 × 256 px, cropped from full scenes **without overlap** | Sec. 2.4 p.6 |
| Scenes | 8,194 Landsat-8 scenes (≈7,600 × 7,600 px), all real-time images from **August 2020**, worldwide except Antarctica | Sec. 2 p.4 |
| Product | Collection 1, Level-1, Real-Time tier. File names look like `LC08_L1TP_046031_20200908_20200908_01_RT` | Fig. 7 p.10; file names in the code CSVs |
| Patch format | "10-band, 16-bit TIFF images" | Sec. 2.4 p.6 |
| Pixel values | **Not stated** whether DN, radiance or reflectance. The original code divides by 65535 (see §7). **[interpretation]** The patches hold raw Level-1 DN. To verify in Phase 2. | |
| Masks | Computed on the full scenes first, then cropped like the images | Sec. 2.4 p.6 |
| Patches with fire, per mask | Schroeder 62,385; Murphy 82,036; Kumar-Roy 143,133. **[interpretation]** These are patches with ≥1 fire pixel. | Fig. 4 caption p.7 |
| Fire-free patches | **No mask file is stored for patches without fire**, to save space. Zero masks can be made with `src/utils/generate_nonfire_masks.py`. | Docstring of that script |

**File naming** (from the CSVs in `src/train/*/*/dataset/`):
- Image: `LC08_L1TP_124046_20200810_20200810_01_RT_p00367.tif`
- Mask: `LC08_L1TP_124046_20200810_20200810_01_RT_Kumar-Roy_p00367.tif`
- Mask folders: `masks/patches/` (Schroeder, Murphy, Kumar-Roy), `masks/intersection/`, `masks/voting/` (README; `src/utils/unzip_patches.py`)
- The scene ID is everything before `_p<index>.tif`. Parsing is finalized in Phase 2.
- In the small subset, Kumar-Roy masks are named `GOLI_v2`. The original unzip script renames them to `Kumar-Roy` (`src/utils/unzip_patches.py`). The paper never mentions "GOLI".

## 2. Spectral bands

The paper says: "The patches are 10-band, 16-bit TIFF images, with channels c1, …, c7 and c9, …, c11. The panchromatic channel c8, with 15 meters of spatial resolution, will not be considered in this work." (Sec. 2.4 p.6)

**[interpretation]** The file band order follows that listing. **To verify in Phase 2.**

| File band (1-based, rasterio) | 0-based index (NumPy/PyTorch) | Landsat-8 band | Name (USGS; not in paper) |
|---|---|---|---|
| 1 | 0 | B1 | Coastal/aerosol |
| 2 | 1 | B2 | Blue |
| 3 | 2 | B3 | Green |
| 4 | 3 | B4 | Red |
| 5 | 4 | B5 | NIR |
| 6 | 5 | B6 | SWIR 1 |
| 7 | 6 | B7 | SWIR 2 |
| 8 | 7 | B9 | Cirrus |
| 9 | 8 | B10 | TIRS 1 (thermal) |
| 10 | 9 | B11 | TIRS 2 (thermal) |

## 3. The 3-channel input ("7, 6, 2")

- The paper: "a 3-channel image containing only bands c7, c6, and c2" (Sec. 3 p.9), described as bands "which have a good response for active fire" (Table 2 caption p.14).
- The stated reason is to see whether fewer bands are enough, which would save bandwidth and storage. AWS lets you download single bands (p.9).
- **The original code reads them with `rasterio.open(path).read((7, 6, 2))`** (`get_img_762bands` in `src/train/*/unet_*_762/generator.py`). rasterio's `read()` uses **1-based** band indexes, so this reads file bands 7, 6 and 2, in that order.
- **Conclusion:** the 3-channel input is Landsat-8 **B7 (SWIR 2), B6 (SWIR 1), B2 (Blue)**, in that order. In a NumPy array loaded with all 10 bands, that is `img[[6, 5, 1]]` (0-based). This depends on the band order in §2, which Phase 2 verifies by looking at the data.
- The same composite is used for display in the paper's figures (Figs. 5 and 9).

## 4. Mask sources

The three handcrafted algorithms use per-channel reflectance ρi and ratios Rij = ρi/ρj. Reflectance comes from the USGS Level-1 conversion. Murphy and Kumar-Roy correct for solar zenith angle; Schroeder does not (Sec. 2 p.4).

| Mask source | Summary | Source |
|---|---|---|
| **Schroeder et al. (2016)** | Uses c1–c7. Unambiguous fire tests on R75, ρ7−ρ5, ρ7, ρ6, ρ1, ρ5. Candidates then go through a contextual test in a 61×61 window, with statistics that exclude fire and water pixels. The original method's multi-temporal part is not used. | Sec. 2.1 p.5 |
| **Murphy et al. (2016)** | Uses c5–c7. Not contextual: unambiguous if R76 ≥ 1.4, R75 ≥ 1.4 and ρ7 ≥ 0.15. Neighbors (3×3) of an unambiguous fire are added if they pass a weaker test or are saturated. Tends to produce thicker clusters and is sensitive to embers. | Sec. 2.2 p.5; pp.6-7 |
| **Kumar & Roy (2018)** | Uses c2–c7. Unambiguous and potential tests on ρ4, ρ6, ρ7. Potential fires go through a contextual test with a growing window (5×5 up to 61×61). **Change made by the authors:** when even 61×61 is mostly excluded pixels, the candidate is treated as non-fire. | Sec. 2.3 pp.5-6 |
| **Intersection** | A pixel is fire if all three algorithms agree. The paper does not define it further; Fig. 14 labels it "Intersection (S, M, K-R)". | Sec. 3 p.10 |
| **Voting** | "best-of-three": a pixel is fire if at least two algorithms agree | Sec. 3 p.10 |

File and folder names: `masks/patches/*_{Schroeder,Murphy,Kumar-Roy}_p*.tif`, `masks/intersection/`, `masks/voting/`. The unzip script finds intersection and voting files by the substrings `_intersection_` and `_voting_` (case-insensitive) (`src/utils/unzip_patches.py`).

## 5. Manually annotated set

- 13 Landsat-8 scenes from **September 2020** (the training data is from August), split into **9,044 patches** (≈12 GB) without overlap (Sec. 2.5 p.7).
- 10 scenes have active fires. 3 have none but are hard cases that produced false positives: Sahara (Schroeder), Greenland (Murphy, Kumar-Roy) and Milan (urban) (Sec. 2.5 p.7; Fig. 17 p.19).
- Annotated "by the authors" (Fig. 6 caption). The tool and procedure are not stated. The authors note the labels are subjective at the border between burnt and still-burning areas, and with smoke (Sec. 4.3 pp.14-15).
- Used only for testing, never for training (p.7).
- **[interpretation]** Because these scenes are from September and training data is from August, the manual scenes should not overlap with training scenes. Phase 2 checks this by scene ID.

## 6. Architectures

The paper evaluates three variants: **U-Net (10c)**, **U-Net (3c)** and **U-Net-Light (3c)** (Sec. 3 p.9; Table 1 p.10). All three are the same U-Net from Ronneberger et al. (2015), "adds batch normalization after each convolutional layer, and contains additional dropout layers" (p.9). U-Net-Light divides the number of filters by 4 (p.9).

Details from the original Keras code (`get_unet` in `src/train/*/*/models.py`; all 15 copies of `models.py` are identical) and Fig. 8:

| Item | Value | Source |
|---|---|---|
| Depth | 4 max-pool steps; encoder 64-128-256-512, bottleneck 1024 (×¼ for Light: 16…256) | Fig. 8; `get_unet` |
| Block | 2 × (3×3 conv, padding "same" → BatchNorm → ReLU) | `conv2d_block` |
| Initialization | `he_normal` | `conv2d_block` |
| Downsampling | MaxPool 2×2 → Dropout | `get_unet` |
| Upsampling | `Conv2DTranspose`, **3×3 kernel**, stride 2 → concatenate skip → Dropout → block | `get_unet` |
| Dropout rate | 0.1 (default argument, never overridden) | `get_model`, `train.py` |
| Output | 1×1 conv to 1 channel, **sigmoid** | `get_unet` |
| Light variant | Same `get_unet` (same depth) with `n_filters=16` | `N_FILTERS = 16`, `MODEL_NAME = 'unet'` in `unet_16f_2conv_762/train.py` |

The paper itself does not state the activation, dropout rate, kernel sizes or output activation; these come from the code.

**Parameter counts, verified.** We rebuilt this architecture in PyTorch: 3×3 convs with bias, BatchNorm, 3×3 transposed convs, depth 4, 1×1 output conv. It gives exactly the trainable parameter counts in Table 1 (p.10):

| Model | Table 1 | Our rebuild |
|---|---|---|
| U-Net (10c) | 34,529,153 | 34,529,153 |
| U-Net (3c) | 34,525,121 | 34,525,121 |
| U-Net-Light (3c) | 2,161,649 | 2,161,649 |

A 2×2 transposed conv, the more common choice, would give 31,047,553 for U-Net (10c), which does not match. The original therefore uses 3×3.

Table 1 also lists model size (132 / 132 / 8.5 MB) and inference time per patch on a Titan Xp (36.8 / 36 / 25.5 ms).

## 7. Training setup

| Item | Paper | Original code | Source |
|---|---|---|---|
| Split | Random 40% / 10% / 50% (train/val/test) | `train_test_split`, `RANDOM_STATE = 42`, **random by patch** (not by scene) | Sec. 4 p.11; `src/utils/split_dataset.py` |
| Which patches | Not stated | **Only patches that have a mask file for that algorithm**, i.e. ≥1 fire pixel. Fire-free patches are never used. | `split_dataset.py` (lists files from masks) |
| Split sizes | Not stated | Kumar-Roy 57,253 / 14,313 / 71,567; Murphy 32,814 / 8,203 / 41,019; Schroeder 24,954 / 6,238 / 31,193; Intersection and Voting 22,097 / 5,524 / 27,622 | Line counts of `src/train/*/*/dataset/*.csv` |
| Optimizer | Adam, lr 0.001 | `Adam()` (Keras defaults: lr 0.001) | p.12; `train.py` |
| Batch size | **16** | **Inconsistent:** 64 for most 16-filter runs (Kumar-Roy, Schroeder, Intersection, Voting), 16 for Murphy 16f and all 64f runs except Voting/U-Net (3c), which uses 12 | p.12; `BATCH_SIZE` in each `train.py` |
| Loss | Binary cross-entropy | `binary_crossentropy` | p.12; `train.py` |
| Epochs | Max 50 | `EPOCHS = 50` | p.12; `train.py` |
| Early stopping | Stop if validation loss doesn't improve for 5 epochs | `EarlyStopping(monitor='val_loss', patience=5)`, **without** `restore_best_weights`. The saved final weights are from the **last** epoch, not the best one. | p.12; `train.py` |
| Checkpoints | Not stated | Every 5 epochs, monitoring **training** loss | `ModelCheckpoint(..., monitor='loss', period=5)` |
| Input scaling | Not stated | `img / 65535` (float32). No mean/std normalization. | `MAX_PIXEL_VALUE` in `generator.py` |
| Augmentation | Not stated | None | `generator.py` |
| Hardware | Titan Xp (12 GB), TensorFlow | `CUDA_DEVICE = 1` | pp.11-12 |
| Training time | Not stated | | |

**Code issue:** `src/train/schroeder/unet_16f_2conv_762/train.py` calls `get_model` without `n_channels`, so the default of 10 applies, but its generator yields 3-band images. As published, this script would fail with a shape mismatch. The Schroeder U-Net-Light weights were therefore produced by a version of the script that is not in the repository.

## 8. Decision threshold

- "any pixel with value above **0.25** is set to 1 (this threshold was empirically defined, after an initial test run on a small fraction of the dataset)" (Sec. 3 p.9).
- The code agrees: `TH_FIRE = 0.25` in all 20 `inference.py` files, as `y_pred > TH_FIRE`. The ground-truth masks also go through `> 0.25`.
- The same threshold is used for every model and was not tuned per model. Experiment E6 tunes it on validation.

## 9. Metrics

- Precision, Recall, F-score ("also known as dice coefficient") and IoU, all for the **fire class**, at the pixel level (Sec. 4.1 pp.12-13, Eqs. 1-3).
- **Aggregation:** "tp, fp and fn values are accumulated for all the images and the metrics are computed only once for the entire dataset (i.e. the reported values are not the average per-patch performance, but the global per-pixel performance)" (p.13).
- The code agrees: `evaluate_v1.py` concatenates all patches and computes P, R, IoU and F once from the totals (`statistics3`). It also computes per-patch averages, but those are not reported.
- Models are ranked by F-score. Pixel accuracy is not reported because more than 99% of pixels are non-fire (p.13).

## 10. Reported results

### Table 2 (p.14): CNN vs. its own training mask, automatic test split (50%), in %

| Mask | Model | P | R | IoU | F |
|---|---|---|---|---|---|
| Schroeder | U-Net (10c) | 86.8 | 89.7 | 78.9 | 88.2 |
| Schroeder | U-Net (3c) | 89.8 | 88.8 | 80.7 | 89.3 |
| Schroeder | U-Net-Light (3c) | 90.8 | 86.1 | 79.2 | 88.4 |
| Murphy | U-Net (10c) | 93.6 | 92.5 | 87.0 | 93.0 |
| Murphy | U-Net (3c) | 89.1 | 97.6 | 87.2 | 93.2 |
| Murphy | U-Net-Light (3c) | 92.6 | 95.1 | 88.4 | 93.8 |
| Kumar-Roy | U-Net (10c) | 84.6 | 94.1 | 80.3 | 89.1 |
| Kumar-Roy | U-Net (3c) | 84.2 | 90.6 | 77.5 | 87.3 |
| Kumar-Roy | U-Net-Light (3c) | 76.8 | 93.2 | 72.7 | 84.2 |
| Intersection | U-Net (10c) | 84.4 | 99.7 | 84.2 | 91.4 |
| Intersection | U-Net (3c) | 93.4 | 92.4 | 86.7 | 92.9 |
| Intersection | U-Net-Light (3c) | 87.4 | 97.3 | 85.4 | 92.1 |
| Voting | U-Net (10c) | 92.9 | 95.5 | 89.0 | 94.2 |
| Voting | U-Net (3c) | 91.9 | 95.3 | 87.9 | 93.6 |
| Voting | U-Net-Light (3c) | 90.2 | 96.5 | 87.3 | 93.2 |

### Table 3 (p.17): against the manual annotations (13 scenes), in %

"—" is the handcrafted algorithm alone, without a CNN.

| Mask | Model | P | R | IoU | F |
|---|---|---|---|---|---|
| Schroeder | — | 88.1 | 70.2 | 64.1 | 78.1 |
| Schroeder | U-Net (10c) | 86.9 | 88.4 | 78.0 | 87.7 |
| Schroeder | U-Net (3c) | 89.5 | 80.5 | 73.6 | 84.8 |
| Schroeder | U-Net-Light (3c) | 89.0 | 78.8 | 71.8 | 83.6 |
| Murphy | — | 76.6 | 96.1 | 74.3 | 85.2 |
| Murphy | U-Net (10c) | 74.8 | 96.9 | 73.0 | 84.4 |
| Murphy | U-Net (3c) | 72.7 | 97.2 | 71.2 | 83.2 |
| Murphy | U-Net-Light (3c) | 75.5 | 96.9 | 73.7 | 84.9 |
| Kumar-Roy | — | 82.3 | 68.4 | 59.7 | 74.7 |
| Kumar-Roy | U-Net (10c) | 82.2 | 94.2 | 78.3 | 87.8 |
| Kumar-Roy | U-Net (3c) | 84.5 | 93.4 | 79.8 | 88.8 |
| Kumar-Roy | U-Net-Light (3c) | 78.8 | 96.9 | 76.9 | 86.9 |
| Intersection | U-Net (10c) | 91.8 | 75.4 | 70.6 | 82.8 |
| Intersection | U-Net (3c) | 90.8 | 73.5 | 68.4 | 81.2 |
| Intersection | U-Net-Light (3c) | 90.8 | 72.8 | 67.8 | 80.8 |
| Voting | U-Net (10c) | 83.6 | 94.0 | 79.3 | 88.5 |
| Voting | U-Net (3c) | 86.4 | 93.0 | 81.1 | 89.6 |
| Voting | U-Net-Light (3c) | 87.2 | 92.4 | 81.4 | 89.7 |

The abstract's "87.2% precision and 92.4% recall" is the **Voting / U-Net-Light (3c)** row of Table 3: U-Net-Light trained on voting masks, evaluated on the manual set. The paper notes there is "no significant difference" with U-Net (3c) (Sec. 4.3 p.17).

### What this project compares against

- **Main comparison mask: Voting.** It is the source of the paper's headline result and the best performer on the manual set. Baseline runs (plan Phase 3.4) use Voting masks, and Table 2/3 Voting rows are the reference numbers.
- Our numbers will not be directly comparable, for these reasons (updated 2026-10-15; an earlier version said 2 regions and scene splits):
  1. We train on 3 of 11 regions (D8, D17).
  2. Stage 1 copies the original's random patch split (Split P). Stage 2 splits by location (Split S, D15), which is stricter: no test location is seen in training.
  3. Our test split is a different set of patches, so comparisons with Table 2 are approximate. The manual set (Table 3) is the same data as the paper's.
  4. 2 of the 13 manual-set locations (44 patches) also appear in our training regions (`data_inspection.md`).

## 11. Limitations and known failure cases stated in the paper

- False positives in large cities. Milan: "bright and/or reflective urban elements"; the CNNs did not fix this (p.7, p.19, Fig. 17d).
- The Sahara (Schroeder) and Greenland (Murphy, Kumar-Roy) false positives were avoided by the CNNs (Fig. 17a-c, p.17).
- Kumar-Roy misses some fire pixels that it classifies as water (p.17, Fig. 16).
- All algorithms struggle with very high-intensity fire cores, which look greenish in the 7-6-2 composite (p.7, p.16).
- Small clusters found by Kumar-Roy and U-Net (10c) were missed by the 3-channel nets (Fig. 13, p.13). This is relevant to our recall-by-size analysis (Phase 5).
- The authors suggest temporal analysis to remove persistent false positives (p.18), and say they "did not focus on finding highly optimized CNN architectures" (p.19).
- Differences between architectures "may be attributed to the initial conditions of the trained networks, sampling, and other random factors" (p.13). This supports our plan to run several seeds.
