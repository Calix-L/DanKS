# V3

V3 contains the team-belief retrieval and PPO training stack. Its Python package is `DanRL_retrieval`.

## Import

```bash
python -m pip install -r generations/v3/source/DanRL_retrieval/environment/requirements-training-core.txt
PYTHONPATH=generations/v3/source python -c "import DanRL_retrieval; print(DanRL_retrieval.__file__)"
```

## Read the code

```text
source/DanRL_retrieval/
├── retrieval/             # action generation, memory, partitioning, ranking
├── training/              # features, team belief, model, PPO, training CLI
├── openguandan_adapter/   # state and evaluation protocol adapters
├── plm_adapter/           # PLM request payload adapter
├── environment/           # public dependency specifications
└── tests/                 # core source tests
```

Recommended first files:

1. `source/DanRL_retrieval/retrieval/card_memory.py`
2. `source/DanRL_retrieval/retrieval/ranker.py`
3. `source/DanRL_retrieval/training/team_belief.py`
4. `source/DanRL_retrieval/training/model.py`
5. `source/DanRL_retrieval/training/ppo.py`

The public V3 snapshot excludes later experimental network files, dated operations, private deployment/evaluation tooling, generated Cython sources, and machine-specific environment snapshots. `manifest.json` is the authoritative inventory.

Model weights and private datasets are not included, so pretrained inference and full training reproduction are outside this repository.

## Self-contained tests

After installing `pytest`, NumPy, and a hardware-appropriate PyTorch build, run the
tests that are self-contained inside the public V3 boundary:

```bash
PYTHONPATH=generations/v3/source python -m pytest \
  generations/v3/source/DanRL_retrieval/tests
```

The snapshot intentionally omits tests coupled to private operations and
adapters, later experimental networks, or an already-built optional native
extension. This keeps the shipped test command deterministic on a clean source
checkout.
