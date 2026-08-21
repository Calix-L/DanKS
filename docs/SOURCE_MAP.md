# Source Map

Use this map to reach the important implementation files without reading every support module.

## V1

Base: `generations/v1/source/DanKS`

| Area | Entry point |
| --- | --- |
| Candidate ranking | `retrieval/ranker.py` |
| Hand partitioning | `retrieval/partitioner.py` |
| Retrieval context | `retrieval/context.py` |
| Candidate scoring | `retrieval/scoring.py` |
| Feature construction | `training/featurizer.py` |
| Selector implementation | `training/numpy_selector.py` |
| Model schema | `training/schema.py` |

## V2

Base: `generations/v2/source/DanKS`

| Area | Entry point |
| --- | --- |
| Action generation | `retrieval/action_generator.py` |
| Candidate ranking | `retrieval/ranker.py` |
| Hand partitioning | `retrieval/partitioner.py` |
| Feature construction | `training/featurizer.py` |
| Selector implementation | `training/onnx_phase14_selector.py` |
| Table runtime | `table_runtime/server.py` |
| GDAI adapter | `gdai_adapter/gdai_payload.py` |

## V3

Base: `generations/v3/source/DanRL_retrieval`

| Area | Entry point |
| --- | --- |
| Action generation | `retrieval/action_generator.py` |
| Card memory | `retrieval/card_memory.py` |
| Candidate ranking | `retrieval/ranker.py` |
| Tactical context | `retrieval/context.py` and `retrieval/pressure.py` |
| Feature construction | `training/featurizer.py` |
| Team-belief features | `training/team_belief.py` |
| Selector model | `training/model.py` |
| PPO implementation | `training/ppo.py` |
| PPO command entry | `training/train_ppo.py` |
| OpenGuandan adapter | `openguandan_adapter/state_adapter.py` |
| PLM payload adapter | `plm_adapter/gdai_payload.py` |

## Repository tooling

| Task | Implementation |
| --- | --- |
| Generation catalog | `danks_repo/catalog.py` |
| Unified CLI | `danks_repo/cli.py` |
| Snapshot policy | `danks_repo/repository.py` |
| Integrity audit | `danks_repo/verify.py` |
| Deterministic archives | `danks_repo/archives.py` |
