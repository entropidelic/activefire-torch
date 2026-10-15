# Decisions log

Every non-trivial decision in this project: what was decided, which alternatives were considered, and why.

---

## D1. uv for all environment and dependency management (2026-10-12, Phase 0)

**Decision.** Use [uv](https://docs.astral.sh/uv/) for everything: Python version, virtual environment, dependencies and running commands (`uv sync`, `uv add`, `uv run ...`). No `pip`, `venv`, `conda` or `requirements.txt`.

**Alternatives.** `pip` + `requirements.txt` (the original repo's approach), conda, Poetry.

**Why.** `uv.lock` pins the exact version of every package, including transitive ones, so anyone who clones the repo and runs `uv sync` gets the same environment. A `requirements.txt` usually pins only direct dependencies. uv also installs the right Python version on its own and is fast. With one tool, the README's "Reproduce" section is just a few commands.

---

## D2. Python 3.13 (2026-10-12, Phase 0)

**Decision.** Pin Python 3.13 in `.python-version`; `requires-python = ">=3.13"`.

**Alternatives.** 3.12 (more conservative), 3.14 (newest).

**Why.** Before choosing, we installed the key binary dependencies on this Mac under both 3.12 and 3.13: `torch` 2.14.1, `rasterio` 1.5.2 (bundles GDAL, the usual pain point), `segmentation-models-pytorch` 0.5.0, `scipy` and `pandas`. Everything installed from prebuilt wheels on both, and MPS was available. 3.13 is the newest version verified to work, so it was chosen. 3.14 was not tested.

---

## D3. Installable package `activefire_torch` in a `src/` layout (2026-10-12, Phase 0)

**Decision.** All core code lives in `src/activefire_torch/`, created with `uv init --package`. Imports read `from activefire_torch.data import ...`. Differences between runs are expressed as YAML configs (`configs/`), not as copies of code.

**Alternatives.**
- A bare `src/` folder imported as `from src.data import ...`. This only works when Python runs from the repo root, and makes `src` look like a package name.
- The original repo's layout: one folder per (mask source × architecture), each with its own copy of `train.py`, `models.py`, `metrics.py`, etc.

**Why.** The package is installed into the environment by `uv sync`, so tests, scripts and notebooks import it the same way from any directory. A single copy of each module plus one config per experiment makes it easy to see that experiments differ in exactly one factor (diff two YAML files instead of two code folders), and a bug fix applies to every run. The name `activefire-torch` refers to the original dataset/project (ActiveFire) and the framework used for this reimplementation.

---

## D4. Device selection: CUDA → MPS → CPU (2026-10-12, Phase 0)

**Decision.** `activefire_torch.device.get_device("auto")` returns CUDA if available, otherwise MPS, otherwise CPU. A config can force a specific device; forcing one that is unavailable raises an error instead of silently falling back.

**Background.** MPS (Metal Performance Shaders) is PyTorch's backend for the GPU in Apple Silicon Macs, the equivalent of CUDA on NVIDIA GPUs.

**Why this order.** CUDA and MPS never exist on the same machine, so the order only decides what happens on a different computer. On the Mac, MPS is used; on a rented NVIDIA machine, CUDA is used, with no code changes. CUDA comes first by convention and because it is the more mature backend. The important part is that CPU comes last, so training never runs on the CPU when a GPU is available. Code never hard-codes `"cuda"`.

---

## D5. Primary hardware and preliminary throughput (2026-10-12, Phase 0)

**Hardware.** Apple M3 Pro: 11 CPU cores, 14-core GPU, 36 GB unified memory (shared by CPU and GPU). Training runs on MPS.

**Preliminary benchmark.** Run before the project existed, so it used a throwaway U-Net, not this repo's model. Setup: plain U-Net (depth 4, two 3×3 convolutions with BatchNorm and ReLU per block), random 256×256 inputs, batch size 16, Adam, BCE loss, PyTorch 2.14.1, float32. Each number is the mean of 15 training steps after 3 warm-up steps. Data loading and validation were not included.

| Model | Params | MPS | CPU |
|---|---|---|---|
| U-Net-Light shape (3 channels, 16 base filters) | 1.94M | 63.4 patches/s | 6.3 patches/s |
| U-Net shape (10 channels, 64 base filters) | 31.05M | 8.6 patches/s | 1.1 patches/s |

**Consequences.**
- CPU training is not viable (about 10× slower than MPS).
- On MPS, U-Net-Light needs about 16 s per epoch per 1,000 training patches, so the light-model experiments are cheap.
- U-Net (10c) needs about 2 min per epoch per 1,000 patches, so it is the bottleneck. Whether it needs a cloud GPU or a capped training budget is decided at Checkpoint A, once the subset size is known.

**Pending.** Re-run the benchmark with `activefire_torch.models` once it exists (Phase 3), including data loading, and test float16 autocast on MPS (see D6).

---

## D6. MPS caveats (2026-10-12, Phase 0)

- **Unsupported operations.** Some PyTorch operations are not implemented on MPS. `PYTORCH_ENABLE_MPS_FALLBACK=1` would silently run them on the CPU, which can slow training a lot without any visible error. It is therefore **not** set globally; any op that fails gets handled case by case and recorded here.
- **Determinism.** MPS does not offer bit-for-bit deterministic training the way CUDA can with deterministic settings. Runs with the same seed will be close but not identical. Comparisons between configurations therefore rely on several seeds (mean ± std) where compute allows, not on single runs.
- **Mixed precision.** Not yet decided. float16 autocast on MPS will be benchmarked against float32 on the real model, and enabled only if it is faster and training stays numerically stable.

---

## D7. No pretrained-encoder library until E4, now the optional T15 (2026-10-12, Phase 0)

**Decision.** Do not add `segmentation-models-pytorch` (or `torchvision`) now. Only experiment E4 (ImageNet-pretrained ResNet-34 encoder vs. random init) needs one; the baselines and every other experiment use the from-scratch U-Net.

**Alternatives.** Add `segmentation-models-pytorch` up front, as the original plan did.

**Why.** It brings in `timm`, `huggingface_hub`, `safetensors` and more, which would be locked and installed long before they are used, and possibly never if E4 changes or is dropped. Adding it later is a single `uv add`. At E4 the choice is between `smp` (less code, but its 10-channel adaptation of the first convolution must be read and explained) and `torchvision` with our own decoder (about 50 more lines, but the adaptation is explicit). See the plan, Phase 4.

---

## D8. Train on two regions of the full dataset, not the "small subset" (2026-10-12, Phase 2; a third region added in D17)

**Finding.** The "small subset" provided by the original repo (`samples.zip`, 242 MB) contains only **100 image patches from 96 different scenes**, with Schroeder, Murphy and Kumar-Roy masks (Kumar-Roy named `GOLI_v2`). It has **no intersection or voting masks**. A 70/15/15 split by scene would leave about 70 training patches, which is too few to train or compare anything.

**Decision.** Train on two regions of the full dataset: **Oceania (~2 GB) and North_America2 (~3 GB)** (sizes from the original `src/utils/download_dataset.py`). They are the two smallest region archives and include all five mask sources. The small subset is kept only for quick tests that the code runs. `scripts/download_data.sh` downloads each part separately (`subset`, `manual`, `regions`).

**Alternatives.**
- A mid-size set of regions (15–25 GB, e.g. Europe or Asia4 + Asia2): more data and more varied scenes, but slower to download and train, and U-Net (10c) would likely need a capped budget.
- Only the provided data: no meaningful training possible.

**Consequences and limitations.**
- This departs from the original plan, which said to use only the small subset. The plan assumed the subset was trainable; it is not.
- Two regions are not a global sample like the paper's. Results may not carry over to other regions (e.g. African savanna fires or Siberian boreal fires), and this must be stated wherever results are reported.
- More regions can be added later with the same script.

---

## D9. Train and test only on patches with fire, as the original does (2026-10-12, Phase 2; Stage 2 test set extended in D13; finding corrected in D16)

**Finding.** The original dataset stores a mask file only for patches where the algorithm detected fire ("in order to save some space", `src/utils/generate_nonfire_masks.py`). The original split script builds its patch list from the mask files (`src/utils/split_dataset.py`). So **the original models never saw a fire-free patch** during training, validation or the automatic test, even though most patches in the dataset have no fire. This is not stated in the paper. *(Correction, D16: the released dataset only contains patches where at least one algorithm found fire, so "fire-free" here means fire-free for that one source.)*

**Decision.** Do the same: for each mask source, use only the patches that have a mask file for that source.

**Alternatives.** Include fire-free patches with all-zero masks. That is more realistic for deployment, where most of the image has no fire, but our numbers would stop being comparable to the paper's.

**Why.** The first goal is replicating the paper, and this setting determines the class balance the model learns from, so it has to match. It means the automatic test metrics say nothing about false alarms on fire-free patches. The manually annotated set does include fire-free scenes (Sahara, Greenland, Milan), so false alarms are still measured there. Adding fire-free patches is a natural extra experiment later, changing only that one factor.

---

## D10. Voting masks as the main mask source for baselines (2026-10-12, Phase 1)

**Decision.** The baseline runs (U-Net-Light (3c) and U-Net (10c)) train on the **voting** masks: fire where at least 2 of the 3 algorithms agree.

**Why.** The paper's headline result (abstract: 87.2% precision, 92.4% recall) is U-Net-Light (3c) trained on voting masks and evaluated on the manual set (Table 3). Voting also gives the best F-score on the manual set, so it is the natural reference.

**Alternatives.** Kumar-Roy has the most training patches (143k patches with fire in the full dataset), but it is the weakest algorithm on the manual set (F 74.7%, Table 3).

---

## D11. Settings where paper, original code and our plan disagree (2026-10-12, Phase 1; resolved in D12)

Reading the paper and code (see `docs/paper_notes.md` §7–8) turned up differences between the paper, the code and this project's plan. Each one will get a decision before the first real training run.

| Setting | Paper | Original code | Our plan | Leaning |
|---|---|---|---|---|
| Decision threshold | 0.25 | 0.25 | 0.5, with E6 asking "is 0.5 right?" | Use **0.25** for the replication baselines; E6 tunes it on validation. |
| Batch size | 16 | 64 for most U-Net-Light runs, 16 for U-Net, one run 12 | Paper's | **16** for all, as the paper states, so architecture is the only difference. |
| Input scaling | not stated | divide by 65535 | per-band mean/std from the training split | Undecided. Mean/std is more standard, but /65535 is what the original did. Decide after inspecting value ranges in Phase 2. |
| Early stopping / weights kept | val loss, patience 5 | val loss, patience 5; keeps **last** epoch | val **F1**; keeps **best** epoch | Undecided. Plan's choice is better practice; replication argues for the original. |
| Split | random by patch, 40/10/50 | same, seed 42 | by scene, 70/15/15 | **By scene** (decided in the plan), to avoid leakage. The gap this causes is discussed in `results/replication.md`. |
| Augmentation | not stated | none | flips and 90° rotations | Undecided. |

---

## D12. Two stages: faithful replication first, then one change at a time (2026-10-12)

**Decision.**
- **Stage 1** copies the original protocol as closely as possible, including choices we consider weak:
  - Random 40/10/50 split by patch, fire-only patches.
  - Inputs divided by 65535, no augmentation.
  - Batch size 16, Adam with lr 0.001, binary cross-entropy.
  - Early stopping on validation loss (patience 5), keeping the last epoch.
  - Threshold 0.25.
- **Stage 2** turns each place where we would do something differently into its own experiment, changing exactly one factor.
- The first Stage 2 run, **T1**, switches to a split by scene. It becomes the reference for all other Stage 2 runs, so no Stage 2 result is inflated by leakage between train and test.

**This resolves D11.** Each open setting takes the original's value in Stage 1, and the alternative becomes a Stage 2 experiment:

| Setting | Stage 1 | Stage 2 experiment |
|---|---|---|
| Split | by patch | by scene (T1, the new reference) |
| Threshold | 0.25 | tuned on validation (T10) |
| Batch size | 16 (as the paper states; the code's 64/12 values are ignored) | — |
| Input scaling | ÷65535 | per-band mean/std (T4) |
| Weights kept | last epoch | best epoch (T9a); then monitor val F1 (T9b) |
| Augmentation | none | flips + rotations (T3) |

Split S keeps the 40/10/50 proportions on purpose, so that T1 changes only how the data is grouped. With 50% held out for testing, training data is limited; that is the price of a clean comparison.

**Alternatives.** Split by scene from the start, as the earlier plan required. That is better practice, but then no run reproduces the paper's protocol. The gap to the paper would mix the effect of the split with everything else, and could not be explained.

**Why.** The person wants a clean replication **and** practice changing every part of the pipeline (data, inputs, model, loss, training, evaluation), each with a measured result. This design gives both, and T1 directly measures how much the paper's split inflated its scores.

**Other changes.**
- ImageNet pretraining (formerly E4) becomes an optional last run (T15). The application asks whether models were trained from scratch or fine-tuned, so one small comparison of our own is worth having if compute allows.
- Error analysis (Phase 5) stays. Recall by fire size is what makes the depth experiment (T6) interpretable.

---

## D13. Stage 2 evaluation rules, fixed before any run (2026-10-12)

These rules are written down before seeing any Stage 2 result, so they cannot be adjusted to fit the results.

1. **Independent changes plus one combined run.** T2–T13 are each compared against T1 only. Changes can interact, especially scaling, loss and threshold, so a final run (T14) combines every change judged an improvement and measures the combination directly.
2. **Selection only on validation.** Thresholds and the winners for T14 are chosen on validation results, never on the test split or the manual set. Otherwise the reported test numbers would be optimistic, because the test set would have helped choose the model.
3. **Fire-free patches in the test set.** In the original protocol, every test patch contains fire, so false alarms on fire-free terrain are only measured on the manual set (most of its 9,044 patches have no fire, and 3 of its 13 scenes have none). Split S therefore has a **full** version of validation and test that includes fire-free patches (as all-zero masks), and the test split is identical for every Stage 2 run. T1 is reported on both versions, to keep the link to Stage 1. This also makes T2 (fire-free training patches) measurable. **Caveat:** "fire-free" means the algorithm found no fire. If the model finds a real fire the algorithm missed, it counts as a false positive. *(Added 2026-10-15, D16: the dataset only has patches where some algorithm found fire, so these "fire-free" patches are cases where the algorithms disagree, not ordinary landscape.)*
4. **Threshold-free metric.** Different losses and normalizations change how confident the model's outputs are, so comparing them at one fixed threshold is unfair. Every run therefore reports:
   - F1 at 0.25 (for comparison with the paper).
   - F1 at the threshold tuned on validation.
   - **Average precision** (area under the precision-recall curve). It is computed from a 1,000-bin histogram of scores accumulated over the whole set, because exact computation over billions of pixels does not fit in memory.
5. **Primary metric:** average precision on the full test split.
6. **Decision rule:**
   - With 3 seeds per configuration, a change counts as an improvement or a degradation only if the difference in mean primary metric is **larger than the standard deviation of both configurations**. Otherwise it is **inconclusive**.
   - With 3 seeds the standard deviation is itself uncertain, so this rule is deliberately simple. The seed variation on MPS also includes non-determinism (D6).
   - A configuration run with only 1 seed is always inconclusive.
7. **Inference cost.** The averaging and ensemble options (T11 seeds, T12 flips, T13 per-mask models) report their inference cost (3×, 8×, 3×) next to their gain. How much that matters depends on the sensor: Landsat revisits a place every 16 days, so data volume matters more than latency, while geostationary sensors deliver images every few minutes.
8. **Manual-set overlap.** The manual scenes are from September and the training scenes from August, so full scene IDs cannot match. The real risk is the same location on a different date, e.g. the same industrial heat source. Overlap is therefore checked by **WRS path/row**. In Stage 1 it is reported next to the comparison with the paper; in Stage 2, overlapping path/rows are excluded from training.

---

## D14. Cross-region experiment T1b (2026-10-14)

**Decision.** Add T1b to Stage 2: train on one region (Oceania or North_America2), test on the other region's Split S test patches, in both directions, with a size-matched control.

**Why.**
- T1 measures the effect of testing on unseen *locations*. T1b measures the next step: an unseen *region* (different continent, land cover, fire types and sun angles). Together with R1 (random patch split) this gives three levels of "how new is the test data", reported in one table.
- A model that only works where it was trained is a practical problem for any fire-monitoring product, so this is worth measuring directly.

**Design choices.**
- **Same test patches for every model.** T1, T1b and the control are all evaluated on the held-out region's Split S test patches, so only the training data differs.
- **Size-matched control.** Training on one region uses fewer patches than T1. The control removes training locations at random (from both regions) until its training set has the same size as T1b's. T1b vs. control isolates the region effect; T1 vs. control shows the data-size effect.
- **Validation from the training region only.** Early stopping and threshold tuning must not see the held-out region.
- **No extra split file.** `data/splits/scene.csv` gets a region column; T1b selects from it.

**Cost.** 12 extra trainings (2 directions × T1b and control × 3 seeds).

## D15. Split S groups by location, not by scene (2026-10-14)

**Decision.** Split S assigns whole WRS path/rows (locations) to train / val / test, 40 / 10 / 50, with seed 42. Each region is split on its own.

**Why.**
- Landsat passes over the same location every 16 days, so one location can have two August scenes. Splitting by scene could put one date in training and the other in test: the same land, roads, factories and often the same fire. That is a smaller version of the leak Split S is meant to remove.
- Splitting each region on its own guarantees that both regions have test locations, which T1b needs.

**Cost.** The 40 / 10 / 50 proportions are of locations, so the share of patches in each split can differ a bit (on the 100-patch sample: 38 / 9 / 53). The real numbers are reported at Checkpoint A.

---

## D16. The dataset only has patches where some algorithm found fire (2026-10-15, Phase 2)

**Finding.** The two regions have 5,231 image patches from 598 scenes: 8.7 per scene on average (1 to 119), where a full scene would give about 880. Every one of the 5,231 images has a mask from at least one of Schroeder, Murphy and Kumar-Roy. So only patches where at least one algorithm found fire were kept. This corrects D9, which assumed that "most patches in the dataset have no fire". Numbers in `docs/data_inspection.md`.

**Consequences.**
- **The "full" version (D13 rule 3) does not add ordinary landscape.** For a given mask source, its extra patches are ones where that source found nothing but another algorithm did: cases where the algorithms disagree. Results on it are described as "false alarms on patches where the algorithms disagree", never as false alarms on fire-free terrain.
- **False alarms on ordinary landscape can only be measured on the manual set.** It has nearly complete scenes (9,044 patches from 13 scenes), three of them with no fire, and hand-drawn masks. Downloading it is needed before any claim about false alarms.
- **T2** tests training with these disagreement patches, and is worded that way.
- **Small training sets.** With voting masks, the fire-only set has 1,970 patches, so about 790 for training with the 40 / 10 / 50 split. Results will vary more between seeds, and more of them may come out inconclusive under the D13 decision rule. Adding a third region (same download script) is an option to decide at Checkpoint A, once per-region counts are known.

**Why keep the full version anyway.** It is still the only automatic test where a patch can have no fire for the chosen source, and it costs nothing extra.

---

## D17. Add South America as a third region (2026-10-15, Phase 2)

**Finding.** With Oceania and North_America2 only, the voting-mask training set has 840 patches and Split S validation has 128, of which 39 are from North_America2 (`docs/data_inspection.md`). Choosing a stopping point and a threshold from a few dozen patches is mostly noise, and one direction of T1b would train on 312 patches.

**Decision.** Add **South_America** (~13 GB, from the original `src/utils/download_dataset.py`) with the same download script. The split code needs no change: each region is split 40 / 10 / 50 on its own (D15).

**Why South America.**
- It is the target market of the job this project is for (fire detection in Latin America), so results on South American fires are the most relevant ones.
- At roughly 1,000 patches per GB in the two current regions, it should about triple the data (an estimate until the notebook counts it).
- It adds a third, very different region (e.g. Amazon and Cerrado fires) to the region comparison.

**Alternatives.**
- Keep two regions and balance the location split by patch count: cheap, but North America's validation would still be about 80 voting patches.
- A smaller region (Asia4, ~8 GB): less data and less relevant.

**Consequences.**
- Training takes roughly 3× longer.
- South America will be most of the data, so pooled results mostly reflect it; per-region results are reported next to them.
- Whether the location split still needs balancing, and whether T1b gets a third direction (e.g. train on two regions, test on South America), are decided once the real counts are in, before any training.
