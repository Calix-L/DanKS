#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from DanKS.gdai_adapter.gdai_payload import (  # noqa: E402
    build_dan_platform_action_list,
    infer_seat_maps,
    latest_greater_action,
    ogd_label_to_plm_tile,
    tiles_to_ogd_labels,
)

from DanKS.retrieval.ranker import StructuralCandidateRanker  # noqa: E402
from DanKS.retrieval.partitioner import FullSearchPartitioner  # noqa: E402
from DanKS.retrieval.context import build_context  # noqa: E402
from DanKS.retrieval.diagnostics import (  # noqa: E402
    action_structure_diagnostic,
    format_action_diagnostic,
    format_group,
    format_partition,
    structure_opportunities,
)
from DanKS.retrieval.plm_rules import normalize_kind  # noqa: E402
from DanKS.training.featurizer import featurize_topk, history_features  # noqa: E402
from DanKS.training.numpy_selector import NumpyTop10Selector  # noqa: E402
from DanKS.training.onnx_phase14_selector import OnnxPhase14Selector  # noqa: E402
from DanKS.training.schema import (  # noqa: E402
    CANDIDATE_DIM,
    FEATURE_VERSION,
    HISTORY_EVENT_SEMANTICS,
    HISTORY_PROTOCOL,
    STATE_DIM,
    TOPK,
)

try:
    import torch  # noqa: E402
    from DanKS.training.accelerator import initialize_device  # noqa: E402
    from DanKS.training.model import build_selector_from_checkpoint  # noqa: E402
except Exception:  # pragma: no cover - health diagnostics remain available.
    torch = None  # type: ignore[assignment]
    initialize_device = None  # type: ignore[assignment]
    build_selector_from_checkpoint = None  # type: ignore[assignment]

CandidateRecallRuntime = None
featurize_with_original_ranks = None


def parse_partition_limit(value: str) -> int | None:
    text = str(value).strip().lower()
    if text in {"all", "none", "full", "unbounded", "infinite"}:
        return None
    parsed = int(text)
    return None if parsed <= 0 else parsed


class Tee:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def clean_card(card: Any) -> str:
    text = str(card).strip().upper().replace("10", "T")
    if text == "SB":
        return "BJ"
    if text == "HR":
        return "RJ"
    return text


def ogd_card(card: str) -> str:
    if card == "BJ":
        return "SB"
    if card == "RJ":
        return "HR"
    return card


def clean_cards(cards: Any) -> list[str]:
    if cards == "PASS" or cards is None:
        return []
    return [clean_card(card) for card in cards]


def clean_rank(rank: Any) -> str | None:
    if rank is None:
        return None
    text = str(rank).strip().upper().replace("10", "T")
    numeric = {
        "1": "A",
        "2": "2",
        "3": "3",
        "4": "4",
        "5": "5",
        "6": "6",
        "7": "7",
        "8": "8",
        "9": "9",
        "10": "T",
        "11": "J",
        "12": "Q",
        "13": "K",
        "14": "BJ",
        "15": "RJ",
    }
    if text in numeric:
        return numeric[text]
    if text in {"B", "SB"}:
        return "BJ"
    if text in {"R", "HR"}:
        return "RJ"
    return text


def clean_kind(kind: Any) -> str:
    text = str(kind or "PASS")
    aliases = {
        "BOOM": "Bomb",
        "Boom": "Bomb",
        "Trips": "Triple",
        "ThreeWithTwo": "TriplePlus",
        "ThreePair": "StraightPair",
        "ThreePairs": "StraightPair",
        "TwoTrips": "StraightTriple",
        "FullHouse": "TriplePlus",
    }
    return aliases.get(text, text)


def clean_action(action: list[Any], index: int) -> dict[str, Any]:
    kind = clean_kind(action[0] if len(action) > 0 else "PASS")
    cards = clean_cards(action[2] if len(action) > 2 else [])
    return {
        "index": index,
        "kind": kind,
        "rank": "PASS" if kind == "PASS" else clean_rank(action[1] if len(action) > 1 else None),
        "cards": cards,
    }


def remaining_counts_by_relative_seat(payload: dict[str, Any]) -> list[int]:
    _self_uid, uid_to_pos = infer_seat_maps(payload)
    out = [0, 0, 0, 0]
    remaining = payload.get("remaining_counts") or {}
    for uid, count in remaining.items():
        try:
            pos = uid_to_pos.get(int(uid), -1)
            if 0 <= pos <= 3:
                out[pos] = int(count)
        except (TypeError, ValueError):
            continue
    hand_count = len(payload.get("self_hand") or [])
    if hand_count:
        out[0] = hand_count
    return out


def played_cards(payload: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for item in payload.get("play_history") or []:
        if str(item.get("action", "")).lower() == "play":
            out.extend(clean_cards(tiles_to_ogd_labels(item.get("cards") or [])))
    return out


def greater_pos(payload: dict[str, Any]) -> int | None:
    if bool(payload.get("must_discard", False)):
        return None
    _self_uid, uid_to_pos = infer_seat_maps(payload)
    for item in reversed(payload.get("play_history") or []):
        if str(item.get("action", "")).lower() == "play":
            try:
                return uid_to_pos.get(int(item.get("uid")), None)
            except (TypeError, ValueError):
                return None
    return None


def normalized_play_history(payload: dict[str, Any]) -> list[dict[str, Any]]:
    _self_uid, uid_to_pos = infer_seat_maps(payload)
    out: list[dict[str, Any]] = []
    for item in payload.get("play_history") or []:
        action = str(item.get("action", "")).lower()
        try:
            pos = uid_to_pos.get(int(item.get("uid")), -1)
        except (TypeError, ValueError):
            pos = -1
        if pos < 0:
            continue
        cards = (
            clean_cards(tiles_to_ogd_labels(item.get("cards") or []))
            if action == "play"
            else []
        )
        out.append({"pos": pos, "cards": cards, "finished": action == "finish"})
    return out


def context_from_payload(payload: dict[str, Any], hand: list[str]) -> dict[str, Any]:
    greater = None if bool(payload.get("must_discard", False)) else latest_greater_action(payload)
    greater_cards = clean_cards(greater[2]) if greater else []
    greater_kind = clean_kind(greater[0]) if greater else "Lead"
    return {
        "game_id": payload.get("game_id"),
        "curRank": clean_rank(payload.get("level_rank") or payload.get("level_value")),
        "my_seat": 0,
        "public_counts": remaining_counts_by_relative_seat(payload),
        "current_kind": greater_kind,
        "current_rank": clean_rank(greater[1]) if greater else None,
        "current_size": len(greater_cards),
        "last_player": greater_pos(payload),
        "known_hand_cards": {"0": hand},
        "played_cards": played_cards(payload),
        "history": normalized_play_history(payload),
        "history_my_seat": 0,
    }


def action_display(action: Any) -> str:
    return f"{action.kind} {action.rank or ''} {' '.join(action.cards)}".strip()


def resolve_post_selector_constraints(selector: Any, requested: bool | None) -> bool:
    if requested is not None:
        return bool(requested)
    return not bool(
        selector is not None and getattr(selector, "requires_history", False)
    )


PLM_PATTERN_BY_KIND = {
    "Single": 1,
    "Pair": 2,
    "StraightPair": 3,
    "Triple": 4,
    "StraightTriple": 5,
    "Straight": 6,
    "TriplePlus": 7,
    "Bomb": 8,
    "StraightFlush": 8,
    "FourKings": 8,
}


def partition_display(row: Any, max_groups: int = 8) -> str:
    groups = list(row.partition.groups)
    shown = groups[:max_groups]
    text = " | ".join(f"{g.kind}:{','.join(g.cards)}" for g in shown)
    if len(groups) > max_groups:
        text += f" | ...(+{len(groups) - max_groups})"
    return text or "EMPTY"


class SelectorBatcher:
    def __init__(self, model: Any, device: Any, *, window_ms: float, max_batch: int, temperature: float) -> None:
        if torch is None:
            raise RuntimeError("torch is required for SelectorBatcher")
        self.model = model
        self.device = device
        self.window_sec = max(0.0, float(window_ms) / 1000.0)
        self.max_batch = max(1, int(max_batch))
        self.temperature = max(1.0e-3, float(temperature))
        self._queue: list[dict[str, Any]] = []
        self._condition = threading.Condition()
        self._worker = threading.Thread(target=self._run, name="selector-batcher", daemon=True)
        self._worker.start()

    def infer(
        self,
        state: np.ndarray,
        candidates: np.ndarray,
        mask: np.ndarray,
        history: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        item = {
            "state": state,
            "candidates": candidates,
            "mask": mask,
            "history": history,
            "event": threading.Event(),
            "result": None,
            "error": None,
        }
        with self._condition:
            self._queue.append(item)
            self._condition.notify()
        item["event"].wait()
        if item["error"] is not None:
            raise item["error"]
        return item["result"]

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._queue:
                    self._condition.wait()
                if self.window_sec > 0.0 and len(self._queue) < self.max_batch:
                    self._condition.wait(timeout=self.window_sec)
                batch = self._queue[: self.max_batch]
                del self._queue[: self.max_batch]
            try:
                states = np.stack([item["state"] for item in batch], axis=0)
                candidates = np.stack([item["candidates"] for item in batch], axis=0)
                masks = np.stack([item["mask"] for item in batch], axis=0)
                with torch.inference_mode():
                    state_t = torch.from_numpy(states).to(self.device)
                    cand_t = torch.from_numpy(candidates).to(self.device)
                    mask_t = torch.from_numpy(masks).to(self.device)
                    logits, value = self.model(state_t, cand_t, mask_t)
                    probs = torch.softmax(logits / self.temperature, dim=1).detach().cpu().numpy()
                    logits_np = logits.detach().cpu().numpy()
                    value_np = value.detach().cpu().numpy()
                for idx, item in enumerate(batch):
                    item["result"] = (logits_np[idx], probs[idx], float(value_np[idx]))
                    item["event"].set()
            except Exception as exc:  # pragma: no cover - defensive worker propagation.
                for item in batch:
                    item["error"] = exc
                    item["event"].set()


class InteractiveRetrievalPolicy:
    def __init__(
        self,
        top_n: int,
        max_partitions: int | None,
        lead_max_partitions: int | None = None,
        follow_max_partitions: int | None = None,
        auto: bool = False,
        selector_checkpoint: str | None = None,
        selector_device: str = "auto",
        sample_policy: bool = False,
        sample_temperature: float = 1.0,
        selector_batch_window_ms: float = 0.0,
        selector_max_batch: int = 16,
        trajectory_jsonl: str | None = None,
        trajectory_flush_every: int = 1,
        trajectory_minimal: bool = False,
        quiet_decisions: bool = False,
        rank_top_k_only: bool = False,
        fast_approx_rank: bool = False,
        post_selector_constraints: bool | None = None,
        exact_best_cache_size: int = 8192,
        partitioner_cache_size: int = 4096,
        recall_checkpoint: str | None = None,
    ) -> None:
        self.max_partitions = max_partitions
        self.lead_max_partitions = lead_max_partitions
        self.follow_max_partitions = follow_max_partitions
        self._thread_local = threading.local()
        self.top_n = top_n
        self.auto = auto
        self.quiet_decisions = quiet_decisions
        self.rank_top_k_only = rank_top_k_only
        self.fast_approx_rank = fast_approx_rank
        self.exact_best_cache_size = max(0, int(exact_best_cache_size))
        self.partitioner_cache_size = max(0, int(partitioner_cache_size))
        self.selector = None
        self.selector_device = None
        self.selector_checkpoint = selector_checkpoint
        self.recall_checkpoint = recall_checkpoint
        if recall_checkpoint and CandidateRecallRuntime is None:
            raise RuntimeError("recall checkpoint requested, but torch is not importable")
        self.recall = CandidateRecallRuntime(recall_checkpoint, device=selector_device) if recall_checkpoint else None
        self.selector_batcher = None
        self.selector_batch_window_ms = max(0.0, float(selector_batch_window_ms))
        self.selector_max_batch = max(1, int(selector_max_batch))
        self.sample_policy = sample_policy
        self.sample_temperature = max(1.0e-3, float(sample_temperature))
        self.trajectory_jsonl = Path(trajectory_jsonl) if trajectory_jsonl else None
        self.trajectory_flush_every = max(1, int(trajectory_flush_every))
        self.trajectory_minimal = trajectory_minimal
        self._trajectory_pending = 0
        self._trajectory_handle = None
        if self.trajectory_jsonl:
            self.trajectory_jsonl.parent.mkdir(parents=True, exist_ok=True)
            buffering = 1 if self.trajectory_flush_every <= 1 else 1024 * 1024
            self._trajectory_handle = self.trajectory_jsonl.open("a", encoding="utf-8", buffering=buffering)
        if selector_checkpoint:
            self.selector, self.selector_device = self._load_selector(selector_checkpoint, selector_device)
            if (
                not isinstance(
                    self.selector, (NumpyTop10Selector, OnnxPhase14Selector)
                )
                and self.selector_batch_window_ms > 0.0
                and self.selector_max_batch > 1
            ):
                self.selector_batcher = SelectorBatcher(
                    self.selector,
                    self.selector_device,
                    window_ms=self.selector_batch_window_ms,
                    max_batch=self.selector_max_batch,
                    temperature=self.sample_temperature,
                )
        # Preserve legacy deployment behavior. Phase1-4 is trained without a
        # post-policy rule override, so its default inference path must execute
        # the PPO choice unchanged.
        self.post_selector_constraints = resolve_post_selector_constraints(
            self.selector, post_selector_constraints
        )
        self._decision_lock = threading.Lock()
        self._trajectory_lock = threading.Lock()
        self._decision_no = 0
        self._invalid_follow_contexts = 0
        self._offer_requests = 0

    def _ranker(self) -> StructuralCandidateRanker:
        ranker = getattr(self._thread_local, "ranker", None)
        if ranker is None:
            ranker = StructuralCandidateRanker(
                partitioner=FullSearchPartitioner(use_native_all=True, cache_size=self.partitioner_cache_size),
                max_partitions=self.max_partitions,
                lead_max_partitions=self.lead_max_partitions,
                follow_max_partitions=self.follow_max_partitions,
                exact_best_cache_size=self.exact_best_cache_size,
            )
            self._thread_local.ranker = ranker
        return ranker

    def _load_selector(self, checkpoint: str, device_arg: str) -> tuple[Any, Any]:
        checkpoint_path = Path(checkpoint)
        onnx_path = checkpoint_path.with_suffix(".onnx")
        if onnx_path.is_file() and device_arg in {"auto", "cpu", "onnx"}:
            selector = OnnxPhase14Selector(onnx_path)
            print(
                f"[selector] loaded checkpoint={checkpoint} model={onnx_path} "
                "device=onnx-cpu",
                flush=True,
            )
            return selector, "onnx-cpu"
        numpy_path = checkpoint_path.with_suffix(".npz")
        if numpy_path.is_file() and device_arg in {"auto", "cpu", "numpy"}:
            selector = NumpyTop10Selector(numpy_path)
            if (
                selector.state_dim != STATE_DIM
                or selector.candidate_dim != CANDIDATE_DIM
                or selector.feature_version != FEATURE_VERSION
            ):
                raise RuntimeError("NumPy selector feature schema mismatch")
            print(
                f"[selector] loaded checkpoint={checkpoint} weights={numpy_path} device=numpy-cpu",
                flush=True,
            )
            return selector, "numpy-cpu"
        if torch is None:
            raise RuntimeError("selector checkpoint requested, but torch is not importable in this environment")
        assert initialize_device is not None
        assert build_selector_from_checkpoint is not None
        requested = "cuda" if device_arg == "auto" and torch.cuda.is_available() else ("cpu" if device_arg == "auto" else device_arg)
        device = initialize_device(requested)
        payload = torch.load(checkpoint, map_location="cpu")
        ckpt_state_dim = int(payload.get("state_dim", -1))
        ckpt_candidate_dim = int(payload.get("candidate_dim", -1))
        if ckpt_state_dim != STATE_DIM or ckpt_candidate_dim != CANDIDATE_DIM:
            raise RuntimeError(
                "selector checkpoint feature shape mismatch: "
                f"checkpoint state_dim={ckpt_state_dim} candidate_dim={ckpt_candidate_dim}, "
                f"current state_dim={STATE_DIM} candidate_dim={CANDIDATE_DIM}. "
                "Use a checkpoint trained with the current retrieval feature schema, "
                f"for example feature_version={FEATURE_VERSION}."
            )
        metadata = payload.get("metadata") or {}
        ckpt_feature_version = payload.get("feature_version") or metadata.get("feature_version")
        if str(ckpt_feature_version) != FEATURE_VERSION:
            raise RuntimeError(
                "selector checkpoint feature_version mismatch: "
                f"checkpoint={ckpt_feature_version!r} current={FEATURE_VERSION!r}. "
                "Retrain the Top10Selector with the current retrieval ranking semantics."
            )
        model = build_selector_from_checkpoint(payload, device=device)
        model.eval()
        print(f"[selector] loaded checkpoint={checkpoint} device={device}", flush=True)
        return model, device

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "service": "DanKS Phase1-4 u400000 PLM evaluation policy",
            "pid": os.getpid(),
            "worker_id": int(os.environ.get("DANKS_WORKER_ID", "0")),
            "policy_workers": int(os.environ.get("DANKS_POLICY_WORKERS", "1")),
            "reuse_port": os.environ.get("DANKS_REUSE_PORT", "0") == "1",
            "top_n": self.top_n,
            "auto": self.auto,
            "decisions": self._decision_no,
            "max_partitions": self.max_partitions,
            "lead_max_partitions": self.lead_max_partitions,
            "follow_max_partitions": self.follow_max_partitions,
            "selector_checkpoint": self.selector_checkpoint,
            "recall_checkpoint": self.recall_checkpoint,
            "selector_device": str(self.selector_device) if self.selector_device is not None else None,
            "selector_model_class": (
                type(self.selector).__name__ if self.selector is not None else None
            ),
            "history_protocol": HISTORY_PROTOCOL,
            "history_event_semantics": HISTORY_EVENT_SEMANTICS,
            "policy_mode": "selector" if self.selector is not None else "retrieval_baseline",
            "selector_batch_window_ms": self.selector_batch_window_ms,
            "selector_max_batch": self.selector_max_batch,
            "sample_policy": self.sample_policy,
            "sample_temperature": self.sample_temperature,
            "trajectory_jsonl": str(self.trajectory_jsonl) if self.trajectory_jsonl else None,
            "trajectory_flush_every": self.trajectory_flush_every,
            "trajectory_minimal": self.trajectory_minimal,
            "quiet_decisions": self.quiet_decisions,
            "rank_top_k_only": self.rank_top_k_only,
            "fast_approx_rank": self.fast_approx_rank,
            "post_selector_constraints": self.post_selector_constraints,
            "exact_best_cache_size": self.exact_best_cache_size,
            "partitioner_cache_size": self.partitioner_cache_size,
            "invalid_follow_contexts": self._invalid_follow_contexts,
            "offer_requests": self._offer_requests,
            "retrieval_profile": os.environ.get("DANRL_BREAK_PROFILE"),
            "break_group_weight": float(os.environ.get("DANRL_BREAK_GROUP_WEIGHT", "0")),
            "approx_action_limit": int(os.environ.get("DANRL_APPROX_ACTION_LIMIT", "0")),
            "partition_hand_count_window": int(
                os.environ.get("DANRL_PARTITION_HAND_COUNT_WINDOW", "-1")
            ),
            "partition_hand_count_window_min_hand": int(
                os.environ.get("DANRL_PARTITION_HAND_COUNT_WINDOW_MIN_HAND", "0")
            ),
            "partition_hand_count_max_covers": int(
                os.environ.get("DANRL_PARTITION_HAND_COUNT_MAX_COVERS", "0")
            ),
        }

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        # Tribute/return selection is not part of the trained Top10 action
        # policy.  An empty result tells the Go client to use its legal,
        # deterministic offer fallback without polluting PPO trajectories.
        if str(payload.get("request_kind", "")).lower() == "offer":
            with self._decision_lock:
                self._offer_requests += 1
            return {
                "action": 2,
                "play_cards": [],
                "pattern": 0,
                "action_kind": "OFFER_FALLBACK",
                "action_rank": "OFFER_FALLBACK",
            }
        with self._decision_lock:
            self._decision_no += 1
            decision_no = self._decision_no
        if not self.quiet_decisions:
            print(f"\n[decision {decision_no}] request received; building legal actions and scoring...", flush=True)
        if not bool(payload.get("must_discard", False)) and latest_greater_action(payload) is None:
            with self._decision_lock:
                self._invalid_follow_contexts += 1
            print(
                f"[invalid-context] decision={decision_no} follow request has no greater action; returning PASS",
                flush=True,
            )
            return {
                "action": 2,
                "play_cards": [],
                "pattern": 0,
                "action_kind": "PASS",
                "action_rank": "PASS",
            }
        raw_action_list = build_dan_platform_action_list(payload)
        if not raw_action_list:
            return {"action": 2, "play_cards": []}

        hand = clean_cards(tiles_to_ogd_labels(payload.get("self_hand") or []))
        actions = [clean_action(action, idx) for idx, action in enumerate(raw_action_list)]
        ctx_payload = context_from_payload(payload, hand)
        ranker = self._ranker()
        rank_top_k = TOPK if self.rank_top_k_only and self.auto and self.quiet_decisions and self.recall is None else None
        ranked = ranker.rank(
            hand,
            actions,
            ctx_payload,
            top_k=rank_top_k,
            approximate_top_k=self.fast_approx_rank and rank_top_k is not None,
        )
        if self.recall is not None:
            ranked, _state, _candidates, _mask, _recall_logits = self.recall.shortlist(
                hand,
                build_context(ctx_payload),
                ranked,
            )
        if self.selector is None:
            self._attach_retrieval_policy_features(hand, ctx_payload, ranked)
        else:
            ranked = self._rerank_with_selector(hand, ctx_payload, ranked)
        chosen = self._choose(decision_no, payload, hand, ctx_payload, ranked, ranker)
        self._record_decision(decision_no, payload, hand, ctx_payload, ranked, chosen)
        if chosen.action.kind == "PASS" or not chosen.action.cards:
            return {"action": 2, "play_cards": [], "pattern": 0, "action_kind": "PASS", "action_rank": "PASS"}
        original_cards = [ogd_card(card) for card in chosen.action.cards]
        kind = normalize_kind(chosen.action.kind)
        return {
            "action": 1,
            "play_cards": [ogd_label_to_plm_tile(card) for card in original_cards],
            "pattern": PLM_PATTERN_BY_KIND.get(kind, 0),
            "action_kind": kind,
            "action_rank": chosen.action.rank,
        }

    def _attach_retrieval_policy_features(self, hand: list[str], ctx_payload: dict[str, Any], ranked: list[Any]) -> None:
        """Attach top10 PPO features for the raw retrieval-baseline policy."""

        if not ranked:
            return
        top_rows = list(ranked[:TOPK])
        if not top_rows:
            return
        ctx = build_context(ctx_payload)
        if self.recall is not None:
            state, candidates, mask = featurize_with_original_ranks(hand, ctx, top_rows)
        else:
            state, candidates, mask = featurize_topk(hand, ctx, top_rows, top_k=TOPK)
        history = history_features(
            ctx_payload.get("history"),
            my_seat=int(ctx_payload.get("history_my_seat", 0)),
        )
        valid = int(sum(1 for value in mask if value > 0))
        raw_logits = np.full((TOPK,), -1.0e9, dtype=np.float32)
        if valid > 0:
            # Retrieval rank is the baseline policy signal: slot 0 is top1,
            # slot 1 is top2, and so on. Temperature controls exploration
            # without depending on score scale across unrelated states.
            slots = np.arange(valid, dtype=np.float32)
            raw_logits[:valid] = -slots / self.sample_temperature
            stable = raw_logits[:valid] - float(np.max(raw_logits[:valid]))
            probs_valid = np.exp(stable)
            probs_valid /= max(1.0e-12, float(probs_valid.sum()))
        else:
            probs_valid = np.zeros((0,), dtype=np.float32)
        state_list = state.tolist() if self.trajectory_jsonl else None
        candidates_list = candidates.tolist() if self.trajectory_jsonl else None
        mask_list = mask.tolist() if self.trajectory_jsonl else None
        history_list = history.tolist() if self.trajectory_jsonl else None
        for i, row in enumerate(top_rows):
            prob = float(probs_valid[i]) if i < len(probs_valid) else 0.0
            row.details["retrieval_rank"] = i + 1
            row.details["retrieval_logit"] = float(raw_logits[i])
            row.details["retrieval_prob"] = prob
            row.details["retrieval_value"] = 0.0
            row.details["feature_slot"] = i
            if self.trajectory_jsonl:
                row.details["feature_state"] = state_list
                row.details["feature_candidates"] = candidates_list
                row.details["feature_mask"] = mask_list
                row.details["feature_history"] = history_list

    def _rerank_with_selector(self, hand: list[str], ctx_payload: dict[str, Any], ranked: list[Any]) -> list[Any]:
        if self.selector is None or not ranked:
            return ranked
        ctx = build_context(ctx_payload)
        if self.recall is not None:
            state, candidates, mask = featurize_with_original_ranks(hand, ctx, ranked[:TOPK])
        else:
            state, candidates, mask = featurize_topk(hand, ctx, ranked[:TOPK], top_k=TOPK)
        history = history_features(
            ctx_payload.get("history"),
            my_seat=int(ctx_payload.get("history_my_seat", 0)),
        )
        if isinstance(self.selector, OnnxPhase14Selector):
            logits_np, probs, value_float = self.selector.infer(
                state,
                candidates,
                mask,
                history,
                temperature=self.sample_temperature,
            )
        elif isinstance(self.selector, NumpyTop10Selector):
            logits_np, probs, value_float = self.selector.infer(
                state,
                candidates,
                mask,
                temperature=self.sample_temperature,
            )
        elif self.selector_batcher is not None:
            logits_np, probs, value_float = self.selector_batcher.infer(
                state, candidates, mask, history
            )
        else:
            if torch is None:
                raise RuntimeError("Torch selector selected without a Torch runtime")
            with torch.inference_mode():
                state_t = torch.from_numpy(state).unsqueeze(0).to(self.selector_device)
                cand_t = torch.from_numpy(candidates).unsqueeze(0).to(self.selector_device)
                mask_t = torch.from_numpy(mask).unsqueeze(0).to(self.selector_device)
                if getattr(self.selector, "requires_history", False):
                    history_t = torch.from_numpy(history).unsqueeze(0).to(
                        self.selector_device
                    )
                    logits, value = self.selector(
                        state_t, cand_t, mask_t, history_t
                    )
                else:
                    logits, value = self.selector(state_t, cand_t, mask_t)
                probs = torch.softmax(logits / self.sample_temperature, dim=1).squeeze(0).detach().cpu().numpy()
                logits_np = logits.squeeze(0).detach().cpu().numpy()
                value_float = float(value.squeeze(0).detach().cpu().item())
        top_rows = list(ranked[:TOPK])
        tail = list(ranked[TOPK:])
        state_list = state.tolist() if self.trajectory_jsonl else None
        candidates_list = candidates.tolist() if self.trajectory_jsonl else None
        mask_list = mask.tolist() if self.trajectory_jsonl else None
        history_list = history.tolist() if self.trajectory_jsonl else None
        for i, row in enumerate(top_rows):
            row.details["retrieval_rank"] = i + 1
            row.details["selector_logit"] = float(logits_np[i])
            row.details["selector_prob"] = float(probs[i])
            row.details["selector_value"] = value_float
            row.details["feature_slot"] = i
            if self.trajectory_jsonl:
                row.details["feature_state"] = state_list
                row.details["feature_candidates"] = candidates_list
                row.details["feature_mask"] = mask_list
                row.details["feature_history"] = history_list
        top_rows.sort(key=lambda row: row.details.get("selector_logit", -1.0e9), reverse=True)
        if self.post_selector_constraints:
            top_rows = self._apply_post_selector_constraints(top_rows)
        return top_rows + tail

    def _apply_post_selector_constraints(self, rows: list[Any]) -> list[Any]:
        if not rows:
            return rows
        chosen = rows[0]
        if normalize_kind(chosen.action.kind) != "PASS":
            return rows

        must_block = float(chosen.details.get("must_block", 0.0))
        opponent_steps = float(chosen.details.get("opponent_min_steps_min", 99.0))
        pass_escape = float(chosen.details.get("escape_risk_penalty", 0.0))
        pass_pressure = float(chosen.details.get("pass_pressure_penalty", 0.0))
        high_pressure = must_block >= 0.75 or (opponent_steps <= 1.0 and (pass_escape > 0.0 or pass_pressure > 0.0))
        if not high_pressure:
            return rows

        blockers = [
            row
            for row in rows[1:]
            if normalize_kind(row.action.kind) != "PASS"
            and float(row.details.get("must_block", must_block)) >= 0.75
            and float(row.details.get("tempo_score", 0.0)) > 0.0
        ]
        if not blockers:
            return rows

        blocker = max(blockers, key=lambda row: (row.score, row.details.get("selector_logit", -1.0e9)))
        blocker.details["selector_hard_override"] = "must_block"
        return [blocker] + [row for row in rows if row is not blocker]

    def _choose(self, decision_no: int, payload: dict[str, Any], hand: list[str], ctx: dict[str, Any], ranked: list[Any], ranker: StructuralCandidateRanker) -> Any:
        if not self.quiet_decisions:
            self._print_decision(decision_no, payload, hand, ctx, ranked, ranker)
        if self.sample_policy:
            forced = next((row for row in ranked[:TOPK] if row.details.get("selector_hard_override")), None)
            if forced is not None:
                forced.details["behavior_prob"] = 1.0
                forced.details["behavior_logp"] = 0.0
                forced.details["behavior_value"] = float(forced.details.get("selector_value", 0.0))
                if not self.quiet_decisions:
                    print(f"[decision {decision_no}] forced override select: {action_display(forced.action)}", flush=True)
                return forced
            prob_key = "selector_prob" if self.selector is not None else "retrieval_prob"
            value_key = "selector_value" if self.selector is not None else "retrieval_value"
            sample_rows = [row for row in ranked[:TOPK] if prob_key in row.details]
            if sample_rows:
                weights = [max(0.0, float(row.details.get(prob_key, 0.0))) for row in sample_rows]
                if sum(weights) <= 0.0:
                    weights = [1.0 for _ in sample_rows]
                chosen = random.choices(sample_rows, weights=weights, k=1)[0]
                total = max(1.0e-12, float(sum(weights)))
                chosen_prob = max(1.0e-12, float(chosen.details.get(prob_key, 0.0)) / total)
                chosen.details["sampled_policy"] = True
                chosen.details["behavior_prob"] = chosen_prob
                chosen.details["behavior_logp"] = math.log(chosen_prob)
                chosen.details["behavior_value"] = float(chosen.details.get(value_key, 0.0))
                if not self.quiet_decisions:
                    print(
                        f"[decision {decision_no}] sampled select: {action_display(chosen.action)} "
                        f"p={chosen_prob:.3f}",
                        flush=True,
                    )
                return chosen
        if self.auto or not sys.stdin.isatty():
            ranked[0].details["behavior_prob"] = 1.0
            ranked[0].details["behavior_logp"] = 0.0
            ranked[0].details["behavior_value"] = float(
                ranked[0].details.get("selector_value", ranked[0].details.get("retrieval_value", 0.0))
            )
            if not self.quiet_decisions:
                print(f"[decision {decision_no}] auto select top1: {action_display(ranked[0].action)}", flush=True)
            return ranked[0]
        while True:
            choice = input("输入 ok 出 top1；输入 1..N 选择排名；输入 pass 过牌：").strip().lower()
            if choice in {"", "ok", "o"}:
                return ranked[0]
            if choice in {"pass", "p"}:
                for row in ranked:
                    if row.action.kind == "PASS":
                        return row
                print("当前 legal action 里没有 PASS。", flush=True)
                continue
            if choice.isdigit():
                idx = int(choice)
                if 1 <= idx <= len(ranked):
                    return ranked[idx - 1]
            print("输入无效。", flush=True)

    def _record_decision(
        self,
        decision_no: int,
        payload: dict[str, Any],
        hand: list[str],
        ctx: dict[str, Any],
        ranked: list[Any],
        chosen: Any,
    ) -> None:
        if not self.trajectory_jsonl:
            return
        slot = chosen.details.get("feature_slot")
        state = chosen.details.get("feature_state")
        candidates = chosen.details.get("feature_candidates")
        mask = chosen.details.get("feature_mask")
        history = chosen.details.get("feature_history")
        if slot is None or state is None or candidates is None or mask is None:
            return
        prob = max(1.0e-12, float(chosen.details.get("behavior_prob", chosen.details.get("selector_prob", chosen.details.get("retrieval_prob", 0.0)))))
        logp = float(chosen.details.get("behavior_logp", math.log(prob)))
        value = float(chosen.details.get("behavior_value", chosen.details.get("selector_value", chosen.details.get("retrieval_value", 0.0))))
        record = {
            "decision_no": decision_no,
            "uid": payload.get("uid"),
            "game_id": payload.get("game_id"),
            "level": ctx.get("curRank"),
            "round_id": payload.get("round_id") or payload.get("round_count") or payload.get("curr_round"),
            "history_len": len(payload.get("play_history") or []),
            "must_discard": bool(payload.get("must_discard", False)),
            "mode": "lead" if ctx.get("current_kind") == "Lead" else "follow",
            "public_counts": ctx.get("public_counts"),
            "current_kind": ctx.get("current_kind"),
            "current_rank": ctx.get("current_rank"),
            "current_size": ctx.get("current_size"),
            "last_player": ctx.get("last_player"),
            "action_slot": int(slot),
            "logp": logp,
            "value": value,
            "forced": bool(chosen.details.get("selector_hard_override")),
            "sample_temperature": self.sample_temperature,
            "policy_mode": "selector" if self.selector is not None else "retrieval_baseline",
            "behavior_prob": prob,
            "selector_prob": float(chosen.details.get("selector_prob", 0.0)),
            "selector_logit": float(chosen.details.get("selector_logit", 0.0)),
            "retrieval_prob": float(chosen.details.get("retrieval_prob", 0.0)),
            "retrieval_logit": float(chosen.details.get("retrieval_logit", 0.0)),
            "retrieval_rank": int(chosen.details.get("retrieval_rank", 0)),
            "state": state,
            "candidates": candidates,
            "mask": mask,
            "history": history,
        }
        if not self.trajectory_minimal:
            record.update(
                {
                    "hand": hand,
                    "chosen_action": {
                        "index": chosen.action.index,
                        "kind": chosen.action.kind,
                        "rank": chosen.action.rank,
                        "cards": list(chosen.action.cards),
                    },
                }
            )
        with self._trajectory_lock:
            handle = self._trajectory_handle
            if handle is None:
                with self.trajectory_jsonl.open("a", encoding="utf-8") as fallback:
                    fallback.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            else:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                self._trajectory_pending += 1
                if self._trajectory_pending >= self.trajectory_flush_every:
                    handle.flush()
                    self._trajectory_pending = 0

    def _print_decision(
        self,
        decision_no: int,
        payload: dict[str, Any],
        hand: list[str],
        ctx: dict[str, Any],
        ranked: list[Any],
        ranker: StructuralCandidateRanker,
    ) -> None:
        print("\n" + "=" * 96, flush=True)
        print(
            f"[decision {decision_no}] uid={payload.get('uid')} game={payload.get('game_id')} "
            f"level={ctx.get('curRank')} mode={'lead' if ctx.get('current_kind') == 'Lead' else 'follow'}",
            flush=True,
        )
        print(f"public_counts={ctx.get('public_counts')} current={ctx.get('current_kind')} {ctx.get('current_rank')} size={ctx.get('current_size')} last_player={ctx.get('last_player')}", flush=True)
        print(f"hand({len(hand)}): {' '.join(hand)}", flush=True)
        ctx_obj = build_context(ctx)
        before_partitions = ranker.partitioner.generate(hand, ctx_obj, ranker.max_partitions)
        opportunities = structure_opportunities(hand, ctx_obj, ranker.partitioner)
        print(f"P_before top{min(5, len(before_partitions))}:", flush=True)
        for p_idx, partition in enumerate(before_partitions[:5], 1):
            print(f"  P{p_idx}. {format_partition(partition)}", flush=True)
        print(f"structure_opportunities={len(opportunities)} top:", flush=True)
        for opp in opportunities[:8]:
            print(f"  {opp.value:6.1f} {format_group(opp.group)}", flush=True)
        print(f"legal_actions={len(ranked)} top{min(self.top_n, len(ranked))}:", flush=True)
        for i, row in enumerate(ranked[: self.top_n], 1):
            detail = row.details
            diagnostic = action_structure_diagnostic(row.action, opportunities)
            selector_text = ""
            if self.selector is not None and "selector_prob" in detail:
                selector_text = (
                    f" sel_p={detail.get('selector_prob', 0.0):5.3f}"
                    f" sel_logit={detail.get('selector_logit', 0.0):6.2f}"
                    f" rrank={int(detail.get('retrieval_rank', i)):02d}"
                )
                if detail.get("selector_hard_override"):
                    selector_text += f" override={detail.get('selector_hard_override')}"
            print(
                f"{i:02d}. idx={row.action.index:03d} {action_display(row.action):36s} "
                f"score={row.score:8.2f} hand={row.hand_count_score:5.1f} "
                f"asset={row.card_value_score:6.1f} retake={row.retake_score:6.1f} "
                f"resid={detail.get('residue_penalty_score', 0.0):4.1f} "
                f"tempo={detail.get('tempo_score', 0.0):6.1f} "
                f"steps={detail.get('my_min_steps', 0.0):3.0f}/{detail.get('opponent_min_steps_min', 0.0):3.0f} "
                f"blk={detail.get('must_block', 0.0):3.1f} race={detail.get('can_race', 0.0):3.1f} "
                f"current={detail.get('current_control_score', 0.0):6.1f} "
                f"lead={detail.get('lead_action_score', 0.0):6.1f} "
                f"spend={detail.get('spend_penalty', 0.0):4.2f} "
                f"break={detail.get('break_group_penalty', 0.0):4.2f} "
                f"pref={detail.get('low_break_preference_penalty', 0.0):4.2f} "
                f"escape={detail.get('escape_risk_penalty', 0.0):4.2f} "
                f"passp={detail.get('pass_pressure_penalty', 0.0):4.2f}"
                f"{selector_text}",
                flush=True,
            )
            print(f"    structure: {format_action_diagnostic(diagnostic)}", flush=True)
            print(f"    after_partition: {partition_display(row)}", flush=True)


class Handler(BaseHTTPRequestHandler):
    policy: InteractiveRetrievalPolicy
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        if self.policy.quiet_decisions:
            return
        print(f"[http] {self.address_string()} {fmt % args}", flush=True)

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/health":
            self._send_json(self.policy.health())
        else:
            self._send_json({"ok": False, "error": "not_found"}, status=404)

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/api/gdai/predict":
            self._send_json({"code": 404, "msg": "not_found", "data": {}}, status=404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0") or "0")
            body = self.rfile.read(length)
            payload = json.loads(body.decode("utf-8")) if body else {}
            result = self.policy.predict(payload)
            self._send_json({"code": 0, "msg": "ok", "data": result})
        except Exception as exc:
            print(f"[error] predict failed: {exc}", flush=True)
            self._send_json({"code": 500, "msg": str(exc), "data": {"action": 2, "play_cards": []}}, status=500)

    def _send_json(self, obj: dict[str, Any], status: int = 200) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DanKS Phase1-4 u400000 policy endpoint for gdai_linux_local."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--max-partitions", type=parse_partition_limit, default=None, help="Partitions per afterstate; use 'all' for full exact-cover search.")
    parser.add_argument("--lead-max-partitions", type=parse_partition_limit, default=None, help="Override partitions per lead afterstate; use 'all' for full exact-cover search.")
    parser.add_argument("--follow-max-partitions", type=parse_partition_limit, default=None, help="Override partitions per follow afterstate; use 'all' for full exact-cover search.")
    parser.add_argument("--log-file", default=str(ROOT / "runs" / "policy_server.log"))
    parser.add_argument("--auto", action="store_true", help="Do not block for stdin; always select top1.")
    parser.add_argument("--selector-checkpoint", default=None, help="Optional Top10Selector checkpoint for reranking retrieval top10.")
    parser.add_argument("--recall-checkpoint", default=None, help="Optional human-recall candidate-pool reranker applied before Top10Selector.")
    parser.add_argument("--selector-device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--sample-policy", action="store_true", help="Sample from selector probabilities instead of taking selector top1.")
    parser.add_argument("--sample-temperature", type=float, default=1.0, help="Temperature for --sample-policy probabilities.")
    parser.add_argument("--seed", type=int, default=20260706, help="Random seed for reproducible policy sampling.")
    parser.add_argument("--exact-best-cache-size", type=int, default=8192)
    parser.add_argument("--partitioner-cache-size", type=int, default=4096)
    parser.add_argument("--selector-batch-window-ms", type=float, default=0.0, help="Optional micro-batch wait window for selector inference; 0 disables batching.")
    parser.add_argument("--selector-max-batch", type=int, default=16, help="Maximum selector micro-batch size.")
    parser.add_argument(
        "--reuse-port",
        action="store_true",
        help="Allow multiple policy workers to share the same TCP port.",
    )
    parser.add_argument("--trajectory-jsonl", default=None, help="Optional JSONL path for PPO trajectory decisions.")
    parser.add_argument("--trajectory-flush-every", type=int, default=1, help="Flush trajectory JSONL every N decisions; higher is faster but risks more data on crash.")
    parser.add_argument("--trajectory-minimal", action="store_true", help="Omit non-training debug fields from trajectory records.")
    parser.add_argument("--quiet-decisions", action="store_true", help="Suppress verbose per-decision diagnostics for high-throughput sampling.")
    parser.add_argument("--rank-top-k-only", action="store_true", help="Only request retrieval topK in auto quiet sampling. Faster, but lead coverage may be less exhaustive.")
    parser.add_argument("--fast-approx-rank", action="store_true", help="Approximate high-throughput ranking: skip low-break/lead coverage postprocessing and take raw topK.")
    constraint_group = parser.add_mutually_exclusive_group()
    constraint_group.add_argument(
        "--disable-post-selector-constraints",
        dest="post_selector_constraints",
        action="store_false",
        help="Use the selector top1 without the deployment-only must-block/PASS override.",
    )
    constraint_group.add_argument(
        "--enable-post-selector-constraints",
        dest="post_selector_constraints",
        action="store_true",
        help="Explicitly enable the deployment-only must-block/PASS override.",
    )
    parser.set_defaults(post_selector_constraints=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch is not None:
        torch_threads = int(os.environ.get("TORCH_NUM_THREADS", "1"))
        if torch_threads > 0:
            torch.set_num_threads(torch_threads)
    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("a", encoding="utf-8")
        sys.stdout = Tee(sys.stdout, log_handle)  # type: ignore[assignment]
        sys.stderr = Tee(sys.stderr, log_handle)  # type: ignore[assignment]
    Handler.policy = InteractiveRetrievalPolicy(
        top_n=args.top_n,
        max_partitions=args.max_partitions,
        lead_max_partitions=args.lead_max_partitions,
        follow_max_partitions=args.follow_max_partitions,
        auto=args.auto,
        selector_checkpoint=args.selector_checkpoint,
        recall_checkpoint=args.recall_checkpoint,
        selector_device=args.selector_device,
        sample_policy=args.sample_policy,
        sample_temperature=args.sample_temperature,
        selector_batch_window_ms=args.selector_batch_window_ms,
        selector_max_batch=args.selector_max_batch,
        trajectory_jsonl=args.trajectory_jsonl,
        trajectory_flush_every=args.trajectory_flush_every,
        trajectory_minimal=args.trajectory_minimal,
        quiet_decisions=args.quiet_decisions,
        rank_top_k_only=args.rank_top_k_only,
        fast_approx_rank=args.fast_approx_rank,
        post_selector_constraints=args.post_selector_constraints,
        exact_best_cache_size=args.exact_best_cache_size,
        partitioner_cache_size=args.partitioner_cache_size,
    )
    class PolicyHTTPServer(ThreadingHTTPServer):
        allow_reuse_address = True
        allow_reuse_port = bool(args.reuse_port)
        daemon_threads = True
        block_on_close = False
        request_queue_size = 256

    server = PolicyHTTPServer((args.host, args.port), Handler)
    print(
        f"DanKS Phase1-4 u400000 policy server listening on "
        f"http://{args.host}:{args.port} "
        f"worker={os.environ.get('DANKS_WORKER_ID', '0')}/"
        f"{os.environ.get('DANKS_POLICY_WORKERS', '1')} "
        f"reuse_port={int(args.reuse_port)}",
        flush=True,
    )
    print("Endpoint: POST /api/gdai/predict. Use /health for a health check.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutdown", flush=True)
    finally:
        server.server_close()
        handle = getattr(Handler.policy, "_trajectory_handle", None)
        if handle is not None:
            handle.close()
        time.sleep(0.1)


if __name__ == "__main__":
    main()
