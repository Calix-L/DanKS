# DanKS

DanKS is a compact, code-only repository for three generations of GuanDan AI. All versions expose the `DanKS` Python namespace and are intentionally isolated so their retrieval, feature, and model contracts remain readable as complete implementations.

> Model weights, datasets, evaluation records, private documents, and deployment configuration are not included.

## Versions

| Version | Focus | Start here |
| --- | --- | --- |
| V1 | Retrieval and NumPy selector foundations | `versions/v1/DanKS/retrieval/ranker.py` |
| V2 | Expanded action generation and ONNX selector | `versions/v2/DanKS/retrieval/action_generator.py` |
| V3 | Card memory, team belief, recall, and PPO | `versions/v3/DanKS/training/model.py` |

## Layout

```text
DanKS/
├── versions/
│   ├── v1/DanKS/       # retrieval + NumPy selector
│   ├── v2/DanKS/       # retrieval + ONNX selector
│   └── v3/DanKS/       # retrieval + model + PPO training
├── guandan/engine/     # shared 108-card rules engine
├── tests/              # repository and engine checks
├── pyproject.toml
└── LICENSE
```

## Quick start

Install the shared engine and development checks:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

Select exactly one AI version with `PYTHONPATH`:

```bash
PYTHONPATH=versions/v1 python -c "import DanKS; print(DanKS.__file__)"
PYTHONPATH=versions/v2 python -c "import DanKS; print(DanKS.__file__)"
PYTHONPATH=versions/v3 python -c "import DanKS; print(DanKS.__file__)"
```

Do not place multiple versions on the same `PYTHONPATH` or assume checkpoints are compatible across versions.

## V3 PPO

V3 retains the complete PPO learner, persistent learner transport, optimizer-state handling, feature pipeline, recall path, and team-belief network:

```text
versions/v3/DanKS/training/
├── model.py
├── ppo.py
├── train_ppo.py
├── persistent_ppo_server.py
├── persistent_ppo_transport.py
├── training_state.py
├── featurizer.py
└── team_belief.py
```

Install a hardware-appropriate PyTorch build plus the public core requirements before using the training entry point:

```bash
python -m pip install -r versions/v3/DanKS/environment/requirements-training-core.txt
PYTHONPATH=versions/v3 python -m DanKS.training.train_ppo --help
```

Training requires user-supplied rollout data and produces user-owned checkpoints; neither is distributed here.

## Shared game engine

```python
from guandan.engine import Environment, Move, Moves
```

The shared package contains the Python table lifecycle, legal move generation, tribute flow, settlement types, and public environment wrapper.

## License

DanKS is licensed under the [Apache License 2.0](LICENSE). Third-party dependencies retain their own licenses. See [NOTICE](NOTICE).
