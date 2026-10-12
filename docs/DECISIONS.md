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
