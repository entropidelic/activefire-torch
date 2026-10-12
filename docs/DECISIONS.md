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

## D7. No pretrained-encoder library until E4 (2026-10-12, Phase 0)

**Decision.** Do not add `segmentation-models-pytorch` (or `torchvision`) now. Only experiment E4 (ImageNet-pretrained ResNet-34 encoder vs. random init) needs one; the baselines and every other experiment use the from-scratch U-Net.

**Alternatives.** Add `segmentation-models-pytorch` up front, as the original plan did.

**Why.** It brings in `timm`, `huggingface_hub`, `safetensors` and more, which would be locked and installed long before they are used, and possibly never if E4 changes or is dropped. Adding it later is a single `uv add`. At E4 the choice is between `smp` (less code, but its 10-channel adaptation of the first convolution must be read and explained) and `torchvision` with our own decoder (about 50 more lines, but the adaptation is explicit). See the plan, Phase 4.

---

## D8. Train on two regions of the full dataset, not the "small subset" (2026-10-12, Phase 2)

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

## D9. Train and test only on patches with fire, as the original does (2026-10-12, Phase 2)

**Finding.** The original dataset stores a mask file only for patches where the algorithm detected fire ("in order to save some space", `src/utils/generate_nonfire_masks.py`). The original split script builds its patch list from the mask files (`src/utils/split_dataset.py`). So **the original models never saw a fire-free patch** during training, validation or the automatic test, even though most patches in the dataset have no fire. This is not stated in the paper.

**Decision.** Do the same: for each mask source, use only the patches that have a mask file for that source.

**Alternatives.** Include fire-free patches with all-zero masks. That is more realistic for deployment, where most of the image has no fire, but our numbers would stop being comparable to the paper's.

**Why.** The first goal is replicating the paper, and this setting determines the class balance the model learns from, so it has to match. It means the automatic test metrics say nothing about false alarms on fire-free patches. The manually annotated set does include fire-free scenes (Sahara, Greenland, Milan), so false alarms are still measured there. Adding fire-free patches is a natural extra experiment later, changing only that one factor.

---

## D10. Voting masks as the main mask source for baselines (2026-10-12, Phase 1)

**Decision.** The baseline runs (U-Net-Light (3c) and U-Net (10c)) train on the **voting** masks: fire where at least 2 of the 3 algorithms agree.

**Why.** The paper's headline result (abstract: 87.2% precision, 92.4% recall) is U-Net-Light (3c) trained on voting masks and evaluated on the manual set (Table 3). Voting also gives the best F-score on the manual set, so it is the natural reference.

**Alternatives.** Kumar-Roy has the most training patches (143k patches with fire in the full dataset), but it is the weakest algorithm on the manual set (F 74.7%, Table 3).

---

## D11. Settings where paper, original code and our plan disagree: open, to decide at Phase 3 (2026-10-12, Phase 1)

Reading the paper and code (see `docs/paper_notes.md` §7–8) turned up differences between the paper, the code and this project's plan. Each one will get a decision before the first real training run.

| Setting | Paper | Original code | Our plan | Leaning |
|---|---|---|---|---|
| Decision threshold | 0.25 | 0.25 | 0.5, with E6 asking "is 0.5 right?" | Use **0.25** for the replication baselines; E6 tunes it on validation. |
| Batch size | 16 | 64 for most U-Net-Light runs, 16 for U-Net, one run 12 | Paper's | **16** for all, as the paper states, so architecture is the only difference. |
| Input scaling | not stated | divide by 65535 | per-band mean/std from the training split | Undecided. Mean/std is more standard, but /65535 is what the original did. Decide after inspecting value ranges in Phase 2. |
| Early stopping / weights kept | val loss, patience 5 | val loss, patience 5; keeps **last** epoch | val **F1**; keeps **best** epoch | Undecided. Plan's choice is better practice; replication argues for the original. |
| Split | random by patch, 40/10/50 | same, seed 42 | by scene, 70/15/15 | **By scene** (decided in the plan), to avoid leakage. The gap this causes is discussed in `results/replication.md`. |
| Augmentation | not stated | none | flips and 90° rotations | Undecided. |
