# V3

V3 contains the team-belief retrieval and PPO training stack. Its Python package is `DanKS`.

## Import

```bash
python -m pip install -r generations/v3/source/DanKS/environment/requirements-training-core.txt
PYTHONPATH=generations/v3/source python -c "import DanKS; print(DanKS.__file__)"
```

## Read the code

```text
source/DanKS/
├── retrieval/             # action generation, memory, partitioning, ranking
├── training/              # features, team belief, model, PPO, training runtime
└── environment/           # public dependency specifications
```

Recommended first files:

1. `source/DanKS/retrieval/card_memory.py`
2. `source/DanKS/retrieval/ranker.py`
3. `source/DanKS/training/team_belief.py`
4. `source/DanKS/training/model.py`
5. `source/DanKS/training/ppo.py`

The public V3 snapshot keeps the complete PPO learner, persistent learner transport, optimizer-state handling, recall path, and network feature stack. It excludes later experimental networks, operational services, evaluation tooling, generated Cython sources, and data conversion utilities. `manifest.json` is the authoritative inventory.

Model weights and private datasets are not included, so pretrained inference and full training reproduction are outside this repository.

Repository-level tests validate V3's package boundary, internal imports, manifests, and PPO source presence without requiring private weights or training data.
