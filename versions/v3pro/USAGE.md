# V3Pro Integration Guide

[← Back to DanKS](../../README.md)

## V3Pro · refine decisions without retraining

V3Pro is an optional **source-only extension of V3**, not a fourth network or a replacement training pipeline. It installs alongside V3 as `DanKSPro`; the original V3 model, features and PPO implementation remain in `DanKS`.

**Decision order:** retrieval → asset mask → frozen V3 → equivalent-play rules → verified endgame refinement.

| Component | What it does | Deliberate limits |
| --- | --- | --- |
| [Asset Safe Gate](DanKSPro/safety.py) | Masks ordinary candidates when a same-type, same-strength alternative spends fewer wildcards or breaks fewer natural bombs, without worsening the guarded metrics. | Not a wildcard ban. Preserves the original mask for immediate finishes or opponents holding 1–2 cards; it does not blanket-ban matching singles/pairs. |
| [Equivalent-play rules](DanKSPro/rules.py) | Keeps the physical triple and swaps in a smaller pair only when the residual partition is equivalent; conservatively selects a smaller sufficient follow-bomb. | No arbitrary triple replacement, lead-bomb substitution or structural sacrifice. |
| [Endgame search](DanKSPro/endgame/runtime.py) | Combines public hidden-card enumeration, frozen-policy continuation, exact minimax, complete-proof tie handling and isolated candidate recovery. | At most 16 total remaining cards; 11–16 requires at most 128 hidden allocations. Incomplete verification keeps the base action. |

### Try V3Pro

Use Python 3.11+ and a V3 environment with PyTorch installed using the [developer guide](../../.github/guides/DEVELOPMENT.md):

```bash
python -m pip install -e . -e versions/v3 -e versions/v3pro
python examples/v3pro_smoke.py
```

The smoke example uses a **randomly initialized model and synthetic position** to demonstrate wiring and reversible search, not playing strength. Trained main/expert weights, private datasets and internal evaluation reports are **not distributed**.

```python
from DanKSPro import ProPolicy

policy = ProPolicy.from_checkpoints(
    "checkpoints/main.pt",
    specialist="checkpoints/endgame.pt",
    extended_specialist="checkpoints/endgame_extended.pt",
)
action_id, record = policy.act(hand, context, legal_actions, history=public_history)
```

The checkpoint names above are caller-owned paths, not included files. Experts must identify the supplied main checkpoint by SHA-256. For legacy checkpoints requiring Python pickle, pass `trusted=True` **only for files you trust**. Without experts, ordinary gate/rule-enhanced inference remains available, but learned endgame refinement explicitly abstains.

For search, call `policy.refine_endgame(...)` with the record, a reconstructed public root, the actor's hand, all played cards and absolute-seat remaining counts; see the [executable example](../../examples/v3pro_smoke.py). Ordinary retrieval context counts are **relative to the actor**, while history seats and search counts are **absolute**. Supply the complete public round history: it drives both card-memory and sequence features and is copied into the record. The adapter validates the decision context, played-card history and legal-action identity before translating IDs to engine positions. Search may examine non-retrieved legal alternatives, but never resurrects an explicitly masked action.

The search port follows the r13 algorithm: full legal coverage within admitted routes, strict verification, complete exact-tie/continuation checks for 11–16 cards, and isolated recovery after an incomplete joint proof. It does **not** include the later r14 experiment. Enumerated-world minimax is not a strategy-fusion-free imperfect-information solution. Budgets are node limits, **not a wall-clock deadline**; this source release is not a qualified low-latency serving package and does not claim private evaluation results apply to a deployment.
