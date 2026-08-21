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

## Quick start

Clone the repository and install the shared engine with the development checks:

```bash
git clone https://github.com/Calix-L/DanKS.git
cd DanKS
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
```

Select an AI generation through `PYTHONPATH`:

```bash
python -m pip install -r versions/v1/requirements.txt
PYTHONPATH=versions/v1 python -c "import DanKS; print(DanKS.__file__)"
```

Replace `v1` with `v2` or `v3` as needed. Do not place multiple generations on the same `PYTHONPATH`.

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

Install a hardware-appropriate PyTorch build first, followed by the hardware-neutral dependencies:

```bash
python -m pip install -r versions/v3/DanKS/environment/requirements-training-core.txt
```

Then provide your own rollout and output paths:

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
