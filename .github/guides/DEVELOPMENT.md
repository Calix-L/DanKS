# DanKS Developer Guide

[← Back to DanKS](../../README.md)

## Installation reference

The shared engine and V3 support Python 3.10 and newer; V1 and V2 support Python 3.11 and newer. All commands below run from the repository root. A dedicated virtual environment for each generation keeps the `DanKS` namespace aligned with its features and model format.

### Choose a package

| Goal | Install command | Notes |
| --- | --- | --- |
| Shared rules engine and tests | `python -m pip install -e '.[dev]'` | Rules engine and repository test suite. |
| V1 · structural retrieval | `python -m pip install -e versions/v1` | NumPy selector; Python 3.11+. |
| V2 · learned selection | `python -m pip install -e versions/v2` | ONNX selector; Python 3.11+. |
| V3 · PPO policy | `python -m pip install -e versions/v3` | Install one PyTorch build below. |

### Select one V3 PyTorch build

| Target | Command |
| --- | --- |
| Linux / Windows CPU | `python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu` |
| NVIDIA CUDA 12.8 | `python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128` |
| macOS CPU | `python -m pip install torch==2.8.0` |

Use the [official PyTorch installation matrix](https://pytorch.org/get-started/previous-versions/) when your platform requires a different wheel. Check the resulting installation with `python examples/v3_model_smoke.py` and inspect all learner options with `python -m DanKS.training.train_ppo --help`.

<details>
<summary><strong>Ascend NPU setup</strong></summary>

The Ascend runtime works with matching host drivers and CANN releases. Install the matching CANN release, followed by the vendor-provided PyTorch and `torch_npu` wheels. A validated combination is recorded in [`requirements-training-npu.txt`](../../versions/v3/DanKS/environment/requirements-training-npu.txt).

```bash
source /usr/local/Ascend/cann/set_env.sh
python3.10 -m venv --system-site-packages .venv-v3-npu
source .venv-v3-npu/bin/activate
python -m pip install -e versions/v3
python -m pip install --no-deps \
  /path/to/torch-2.7.1+cpu-cp310-cp310-manylinux_2_28_x86_64.whl \
  /path/to/torch_npu-2.7.1.post2-cp310-cp310-manylinux_2_28_x86_64.whl
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
python -m DanKS.training.train_ppo --help
```

Keep this virtual environment dedicated to Ascend NPU. For other driver, CANN, architecture, or Python combinations, select the corresponding vendor wheels.

</details>

<details>
<summary><strong>Validated configurations</strong></summary>

| Target | System | Python | Framework | Key packages |
| --- | --- | --- | --- | --- |
| CI and shared engine | Linux | 3.10, 3.12 | — | pytest 7+ |
| V1 | CPU | 3.11+ | NumPy selector | NumPy 2.4.6 |
| V2 | CPU | 3.11+ | ONNX selector | NumPy 2.4.6, ONNX Runtime 1.27.0 |
| V3 NVIDIA server | H100, driver 575.57.08 | 3.11.14 | PyTorch 2.8.0 + CUDA 12.8 | NumPy 2.4.6, pybind11 3.0.4 |
| V3 Ascend server | Ubuntu 22.04.5, 910B2C, driver 24.1.0, CANN 8.5.0 | 3.10.12 | PyTorch 2.7.1 + torch_npu 2.7.1.post2 | NumPy 1.26.0, pybind11 3.0.4 |

These are known-good reference configurations; DanKS also runs on other compatible environments.

</details>

### Optional V3 C++ acceleration

The optimized retrieval kernels support Linux and macOS with a C++17 compiler, Python development headers, and `pybind11`; Windows automatically selects the Python implementation. Install the platform toolchain once:

```bash
# Ubuntu/Debian
sudo apt-get update && sudo apt-get install -y build-essential python3-dev

# macOS (run once)
xcode-select --install
```

Then build and verify both kernels with one command in the active V3 environment:

```bash
danks-build-native
```

The command locates the installed V3 source tree automatically and finishes with `cover=True, actor=True`. Linux builds enable host-specific compiler optimization; macOS delegates architecture selection to the Python toolchain and supports universal2 builds. Run it again after changing Python versions or CPU architecture. Windows automatically selects the Python implementation.

### Development checks

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

## Run the examples

The examples cover the rules engine, structural retrieval, a full network forward pass, and a PPO update, all directly runnable from source:

```bash
# Shared rules engine; available from the base environment.
python examples/engine_quickstart.py

# Structural retrieval; run inside a matching V1, V2, or V3 environment.
python examples/retrieval_quickstart.py --version v3

# Full V3 network forward pass; run inside a V3 environment with PyTorch.
python examples/v3_model_smoke.py

# One synthetic optimizer update through the V3 PPO learner.
python examples/v3_ppo_smoke.py
```

Each command includes self-checking assertions for a quick confirmation that the environment and code path are working.

## Shared game engine

The engine can be used independently of the AI generations:

```python
from guandan import Environment

game = Environment(first_player=0)
for seat in range(4):
    game.add_player(f"player-{seat}", seat)

messages = game.start()
assert all(len(player.hand_cards) == 27 for player in game.players)
```

The public API also exports `Move` and `Moves` for move representation and legal-action generation.

## Train V3 with PPO

After activating and verifying a V3 environment, select the rollout and checkpoint output paths:

```bash
python -m DanKS.training.train_ppo \
  --rollout /path/to/rollout.npz \
  --output /path/to/checkpoint.pt \
  --device auto
```

The learner expects rollout arrays for state, candidates, masks, history, actions, behavior log-probabilities, advantages, and returns. Run the entry point with `--help` for optimization, evaluation, accelerator, and initialization options.

The V3 training implementation lives in [`versions/v3/DanKS/training`](../../versions/v3/DanKS/training) and includes:

- model and feature definitions;
- PPO objectives and tactical resampling;
- checkpoint and optimizer-state handling;
- persistent learner transport;
- recall and team-belief auxiliary paths;
- CPU, CUDA, and NPU-aware accelerator helpers.
