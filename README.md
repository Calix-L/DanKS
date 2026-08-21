# DanKS

[![CI](https://github.com/Calix-L/DanKS/actions/workflows/ci.yml/badge.svg)](https://github.com/Calix-L/DanKS/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-D22128)](LICENSE)

**Three generations of GuanDan AI in one compact, code-first repository.**

DanKS preserves the evolution of a four-player, partnership-based GuanDan agent: from structural retrieval, through a learned selector, to a memory-aware policy trained with PPO. Each generation is self-contained and uses the same `DanKS` Python namespace, while a shared 108-card rules engine provides the game foundation.

> This repository contains source code only. Model weights, datasets, evaluation records, private documents, and deployment configuration are intentionally excluded.

## Why DanKS?

- **Three readable generations** — compare complete implementations without mixing their feature or checkpoint contracts.
- **End-to-end V3 training** — PPO learner, rollout loading, optimizer state, persistent transport, recall, and team-belief modeling are included.
- **Shared GuanDan engine** — legal moves, table lifecycle, tribute flow, and settlement live in one reusable package.
- **Small public surface** — no duplicated archives, generated manifests, model binaries, or platform-specific evaluation code.

## System at a glance

```text
GuanDan state
     │
     ▼
structural retrieval ──► candidate ranking ──► top-k features
                                                    │
                                                    ▼
                                       learned selector / PPO policy
                                                    │
                                                    ▼
                                               chosen action

shared foundation: 108-card rules · legal moves · table lifecycle
```

## Generations

| Version | Main idea | What it adds | Entry point |
| --- | --- | --- | --- |
| **V1** | Structural retrieval | Candidate scoring and a NumPy selector | [`ranker.py`](versions/v1/DanKS/retrieval/ranker.py) |
| **V2** | Learned selection | Broader action generation and an ONNX selector | [`action_generator.py`](versions/v2/DanKS/retrieval/action_generator.py) |
| **V3** | Memory-aware policy learning | Card memory, candidate coverage, recall, team belief, and PPO | [`model.py`](versions/v3/DanKS/training/model.py) |

The versions are intentionally isolated. Select one version at a time; their feature schemas and checkpoints are not interchangeable.

## Repository layout

```text
DanKS/
├── versions/
│   ├── v1/DanKS/       # retrieval + NumPy selector
│   ├── v2/DanKS/       # retrieval + ONNX selector
│   └── v3/DanKS/       # retrieval + neural policy + PPO
├── guandan/engine/     # shared Python rules engine
├── tests/              # repository and engine checks
├── pyproject.toml
├── LICENSE
└── NOTICE
```

## Environment setup

DanKS supports Python 3.10 and newer. Use a separate virtual environment for each generation so that its dependencies and `DanKS` namespace remain isolated.

### Validated configurations

| Target | System | Python | Framework | Key packages |
| --- | --- | --- | --- | --- |
| CI and shared engine | Linux | 3.10, 3.12 | — | pytest 7+ |
| V1 | CPU | 3.11+ | NumPy selector | NumPy 2.4.6 |
| V2 | CPU | 3.11+ | ONNX selector | NumPy 2.4.6, ONNX Runtime 1.27.0 |
| V3 NVIDIA server | H100, driver 575.57.08 | 3.11.14 | PyTorch 2.8.0 + CUDA 12.8 | NumPy 2.4.6, pybind11 3.0.4 |
| V3 Ascend server | Ubuntu 22.04.5, 910B2C, driver 24.1.0, CANN 8.5.0 | 3.10.12 | PyTorch 2.7.1 + torch_npu 2.7.1.post2 | NumPy 1.26.0, pybind11 3.0.4 |

The two accelerator rows reproduce the project servers; they are reference configurations, not minimum hardware requirements.

### 1. Clone and create a base environment

```bash
git clone https://github.com/Calix-L/DanKS.git
cd DanKS
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pytest -q
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`.

### 2. Select V1 or V2

Create a Python 3.11 environment for one generation and expose only its package root:

```bash
# V1
python3.11 -m venv .venv-v1
source .venv-v1/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pip install -r versions/v1/requirements.txt
export PYTHONPATH="$PWD/versions/v1"
python -c "import DanKS; print(DanKS.__file__)"
```

For V2, use a different environment:

```bash
python3.11 -m venv .venv-v2
source .venv-v2/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pip install -r versions/v2/requirements.txt
export PYTHONPATH="$PWD/versions/v2"
python -c "import DanKS; print(DanKS.__file__)"
```

Do not place multiple generations on the same `PYTHONPATH`; their feature schemas and checkpoints are intentionally independent.

### 3. Build a V3 CPU or NVIDIA environment

Create a fresh Python 3.11 environment and install the shared and V3 dependencies:

```bash
python3.11 -m venv .venv-v3
source .venv-v3/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pip install -r versions/v3/DanKS/environment/requirements-training-core.txt
export PYTHONPATH="$PWD/versions/v3"
```

Install exactly one PyTorch build. These commands match the tested server version; choose the wheel index appropriate for your machine using the [official PyTorch installation matrix](https://pytorch.org/get-started/previous-versions/).

```bash
# CPU
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu

# NVIDIA CUDA 12.8
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

Verify the environment:

```bash
python -c "import numpy, torch; print('numpy', numpy.__version__); print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"
python -m DanKS.training.train_ppo --help
```

### 4. Build a V3 Ascend NPU environment

The Ascend runtime is coupled to the host driver and CANN installation. Install the matching CANN release first, then use the vendor-provided PyTorch and `torch_npu` wheels. The public [`requirements-training-npu.txt`](versions/v3/DanKS/environment/requirements-training-npu.txt) records the validated pair.

```bash
source /usr/local/Ascend/cann/set_env.sh
python3.10 -m venv --system-site-packages .venv-v3-npu
source .venv-v3-npu/bin/activate
python -m pip install -r versions/v3/DanKS/environment/requirements-training-core.txt

# Replace these paths with the matching vendor wheels for your platform.
python -m pip install --no-deps \
  /path/to/torch-2.7.1+cpu-cp310-cp310-manylinux_2_28_x86_64.whl \
  /path/to/torch_npu-2.7.1.post2-cp310-cp310-manylinux_2_28_x86_64.whl

export PYTHONPATH="$PWD/versions/v3"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
```

Verify that PyTorch can see the accelerator:

```bash
python -c "import torch, torch_npu; print('torch', torch.__version__); print('torch_npu', torch_npu.__version__); print('npu', torch.npu.is_available())"
python -m DanKS.training.train_ppo --help
```

Do not install CUDA packages into the NPU environment. If your driver, CANN, architecture, or Python version differs, obtain a matching wheel pair from the Ascend software distribution instead of forcing the versions above.

### 5. Build the optional V3 C++ kernels

The optimized retrieval kernels require a C++17 compiler, Python development headers, and `pybind11`. Build them inside the active V3 environment on the target machine:

```bash
cd versions/v3/DanKS/retrieval/native_cpp
python setup.py build_ext --inplace
cd ../../../../../

PYTHONPATH=versions/v3 python -c "from DanKS.retrieval.native_cover import available; from DanKS.retrieval.native_actor_core import available as actor_available; print('cover', available()); print('actor', actor_available())"
```

Both values should be `True`. The build uses host-specific compiler optimization, so rebuild the extensions after changing Python versions or moving to a different CPU architecture.

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

After activating and verifying a V3 environment, provide your own rollout and output paths:

```bash
PYTHONPATH=versions/v3 python -m DanKS.training.train_ppo \
  --rollout /path/to/rollout.npz \
  --output /path/to/checkpoint.pt \
  --device auto
```

The learner expects rollout arrays for state, candidates, masks, history, actions, behavior log-probabilities, advantages, and returns. Run the entry point with `--help` for optimization, evaluation, accelerator, and initialization options.

The V3 training implementation lives in [`versions/v3/DanKS/training`](versions/v3/DanKS/training) and includes:

- model and feature definitions;
- PPO objectives and tactical resampling;
- checkpoint and optimizer-state handling;
- persistent learner transport;
- recall and team-belief auxiliary paths;
- CPU, CUDA, and NPU-aware accelerator helpers.

## Project boundaries

DanKS does not distribute pretrained checkpoints or the data needed to reproduce private training runs. Bring your own rollout data and keep generated checkpoints outside the repository. The source tree is designed for code inspection, adaptation, and independent experimentation.

## License

DanKS is available under the [Apache License 2.0](LICENSE). Third-party dependencies retain their respective licenses; see [NOTICE](NOTICE).
