#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from math import ceil
from pathlib import Path
from typing import Any, Iterator

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from DanRL_retrieval.retrieval.context import build_context  # noqa: E402
from DanRL_retrieval.retrieval.models import ActionCandidate  # noqa: E402
from DanRL_retrieval.retrieval.ranker import StructuralCandidateRanker  # noqa: E402
from DanRL_retrieval.scripts.evaluate_human_data import (  # noqa: E402
    action_key,
    clean_action,
    clean_cards,
    context_from_state,
    iter_eval_games,
    iter_safe_decision_contexts,
)
from DanRL_retrieval.training.featurizer import featurize_topk  # noqa: E402
from DanRL_retrieval.training.schema import CANDIDATE_DIM, FEATURE_VERSION, STATE_DIM, TOPK  # noqa: E402
from DanRL_retrieval.training.data_identity import (  # noqa: E402
    canonical_sample_id,
    replay_group_from_sample_id,
)
from guandan_llm.sft_pipeline.human_data_adapter import (  # noqa: E402
    full_state_for_payload,
    game_level_value,
    initial_hands,
    iter_decision_contexts,
    source_action_from_human,
    tiles_to_ogd_labels,
    uid_to_chair_map,
)


def parse_partition_limit(value: str) -> int | None:
    text = str(value).strip().lower()
    if text in {"all", "none", "full", "unbounded", "infinite"}:
        return None
    parsed = int(text)
    return None if parsed <= 0 else parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build frozen retrieval top10 BC dataset from human_data JSONL.")
    parser.add_argument("--path", nargs="+", default=[str(ROOT / "human_data_10000.json")])
    parser.add_argument("--output", default=str(ROOT / "DanRL_retrieval" / "data" / "top10_bc.npz"))
    parser.add_argument("--games", type=int, default=200, help="Max games to scan per input path.")
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=TOPK)
    parser.add_argument("--max-actions", type=int, default=80)
    parser.add_argument("--max-hand", type=int, default=20)
    parser.add_argument("--max-partitions", type=parse_partition_limit, default=20)
    parser.add_argument("--lead-max-partitions", type=parse_partition_limit, default=50)
    parser.add_argument("--follow-max-partitions", type=parse_partition_limit, default=20)
    parser.add_argument("--keep-misses", action="store_true", help="Store samples even if human is outside top10 with label=-1.")
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument("--line-shard-index", type=int, default=0)
    parser.add_argument("--line-shard-count", type=int, default=1)
    return parser.parse_args()


def replay_split_v1(replay_id: str) -> str:
    value = int.from_bytes(
        hashlib.blake2b(
            f"plm_proxy_split_v1:{replay_id}".encode("utf-8"), digest_size=8
        ).digest(),
        "little",
    ) % 10
    return "train" if value < 8 else ("validation" if value == 8 else "test")


VALUE_TO_RANK = {1: "A", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9", 10: "T", 11: "J", 12: "Q", 13: "K"}


def _rank_from_level(value: int) -> str:
    return VALUE_TO_RANK.get(int(value), "2")


def _seat(chair: int) -> int:
    return max(0, min(3, int(chair) - 1))


def _tile_labels(tiles: list[int]) -> list[str]:
    return clean_cards(tiles_to_ogd_labels(tiles))


def _remove_tiles(hand: list[int], tiles: list[int]) -> None:
    for tile in tiles:
        try:
            hand.remove(int(tile))
        except ValueError:
            pass


def _find_action_by_cards(actions: list[ActionCandidate], cards: list[str]) -> ActionCandidate | None:
    target = ("PASS", ()) if not cards else ("?", tuple(sorted(cards)))
    for action in actions:
        if not cards and action.kind == "PASS":
            return action
        if cards and tuple(sorted(action.cards)) == target[1]:
            return action
    return None


def _current_context_payload(
    game: dict[str, Any],
    uid: int,
    uid_chair: dict[int, int],
    hands: dict[int, list[int]],
    current_action: ActionCandidate | None,
    current_holder_uid: int | None,
    played_cards: list[str],
    level_rank: str,
) -> tuple[list[str], dict[str, Any]]:
    absolute_my_seat = _seat(uid_chair.get(uid, 1))
    absolute_seat_counts = [27, 27, 27, 27]
    hand: list[str] = []
    for player_uid, chair in uid_chair.items():
        seat = _seat(chair)
        labels = _tile_labels(hands.get(player_uid, []))
        absolute_seat_counts[seat] = len(labels)
        if player_uid == uid:
            hand = labels
    public_counts = [
        absolute_seat_counts[(absolute_my_seat + relative_seat) % 4]
        for relative_seat in range(4)
    ]
    absolute_last_player = (
        _seat(uid_chair[current_holder_uid])
        if current_holder_uid is not None and current_holder_uid in uid_chair
        else None
    )
    relative_last_player = (
        (absolute_last_player - absolute_my_seat) % 4
        if absolute_last_player is not None
        else None
    )
    ctx_payload = {
        "game_id": str(game.get("replay_id") or (game.get("_id") or {}).get("$oid", "")),
        "curRank": level_rank,
        "my_seat": 0,
        "public_counts": public_counts,
        "current_kind": current_action.kind if current_action is not None else "Lead",
        "current_rank": current_action.rank if current_action is not None else None,
        "current_size": current_action.size if current_action is not None else 0,
        "last_player": relative_last_player,
        "known_hand_cards": {"0": hand},
        "played_cards": played_cards,
        "history": [],
        "history_my_seat": absolute_my_seat,
    }
    return hand, ctx_payload


def iter_raw_replay_decisions(game: dict[str, Any], line_no: int, ranker: StructuralCandidateRanker) -> Iterator[dict[str, Any]]:
    hands = {uid: list(cards) for uid, cards in initial_hands(game).items()}
    uid_chair = uid_to_chair_map(game)
    if len(hands) != 4 or len(uid_chair) != 4:
        return
    level_rank = _rank_from_level(game_level_value(game))
    current_action: ActionCandidate | None = None
    current_holder_uid: int | None = None
    pass_count = 0
    played_cards: list[str] = []
    replay_id = str(game.get("replay_id") or (game.get("_id") or {}).get("$oid", line_no))
    doc_id = str((game.get("_id") or {}).get("$oid", line_no))

    for ev_idx, ev in enumerate(game.get("events") or []):
        event_id = ev.get("event_id")
        uid = ev.get("uid")
        if uid is None or int(uid) not in hands:
            continue
        uid = int(uid)
        if event_id not in {3, 4, 15}:
            continue

        hand, ctx_payload = _current_context_payload(
            game,
            uid,
            uid_chair,
            hands,
            current_action,
            current_holder_uid,
            played_cards,
            level_rank,
        )
        ctx = build_context(ctx_payload)
        actions = ranker.action_generator.generate(hand, ctx)
        if event_id == 4:
            human = {"index": None, "type": "PASS", "kind": "PASS", "rank": "PASS", "cards": []}
            matched = _find_action_by_cards(actions, [])
        else:
            cards = _tile_labels([int(t) for t in (ev.get("tiles") or [])])
            matched = _find_action_by_cards(actions, cards)
            human = {
                "index": None,
                "type": matched.kind if matched is not None else "UNKNOWN",
                "kind": matched.kind if matched is not None else "UNKNOWN",
                "rank": matched.rank if matched is not None else "UNKNOWN",
                "cards": cards,
            }

        yield {
            "sample_id": f"line{line_no}:{doc_id}:{replay_id}:{ev_idx}:{uid}",
            "hand": hand,
            "ctx_payload": ctx_payload,
            "actions": [
                {"index": action.index, "kind": action.kind, "rank": action.rank, "cards": list(action.cards)}
                for action in actions
            ],
            "human": human,
            "matched": matched is not None,
        }

        if event_id == 4:
            pass_count += 1
            if current_action is not None and pass_count >= 3:
                current_action = None
                current_holder_uid = None
                pass_count = 0
            continue

        tiles = [int(t) for t in (ev.get("tiles") or [])]
        cards = _tile_labels(tiles)
        played_cards.extend(cards)
        _remove_tiles(hands[uid], tiles)
        if matched is not None:
            current_action = matched
        else:
            current_action = ActionCandidate(index=-1, kind=str(ev.get("pattern") or "Unknown"), cards=tuple(cards), rank=None)
        current_holder_uid = uid
        pass_count = 0


def iter_training_items(game: dict[str, Any], line_no: int, ranker: StructuralCandidateRanker) -> Iterator[dict[str, Any]]:
    used_adapter = False
    for item in iter_safe_decision_contexts(game, line_no):
        if "_adapter_error" in item:
            yield {"error": "adapter_errors"}
            continue
        used_adapter = True
        try:
            state, raw_actions = full_state_for_payload(item["payload"])
        except Exception:
            yield {"error": "state_errors"}
            continue
        human = source_action_from_human(raw_actions, item["human_tiles"], item["human_is_pass"])
        hand = clean_cards(state.get("handCards") or [])
        actions = [clean_action(action, idx) for idx, action in enumerate(raw_actions)]
        ctx_payload = context_from_state(state, item["payload"], hand)
        yield {
            "sample_id": f"line{line_no}:{item.get('sample_id')}",
            "hand": hand,
            "ctx_payload": ctx_payload,
            "actions": actions,
            "human": human,
            "matched": human is not None and human.get("type") != "UNKNOWN",
        }
    if not used_adapter:
        yield from iter_raw_replay_decisions(game, line_no, ranker)


def iter_sharded_games(path: str, games: int, random_seed: int | None, shard_index: int, shard_count: int):
    if shard_count <= 1:
        yield from iter_eval_games(path, games, random_seed)
        return
    emitted = 0
    for line_no, game in iter_eval_games(path, games * shard_count if random_seed is None else games, random_seed):
        if line_no % shard_count != shard_index:
            continue
        yield line_no, game
        emitted += 1
        if emitted >= games:
            break


def main() -> None:
    args = parse_args()
    if args.top_k != TOPK:
        raise ValueError(f"schema fixes top-k={TOPK}, got {args.top_k}")

    ranker = StructuralCandidateRanker(
        max_partitions=args.max_partitions,
        lead_max_partitions=args.lead_max_partitions,
        follow_max_partitions=args.follow_max_partitions,
    )

    states: list[np.ndarray] = []
    candidates: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    labels: list[int] = []
    sample_ids: list[str] = []
    retrieval_ranks: list[int] = []
    human_kinds: list[str] = []
    current_kinds: list[str] = []
    replay_ids: list[str] = []
    stats: Counter[str] = Counter()
    start_time = time.perf_counter()

    source_paths = [str(path) for path in args.path]
    if not (0 <= args.line_shard_index < max(1, args.line_shard_count)):
        raise ValueError("--line-shard-index must be in [0, --line-shard-count)")

    per_source_target = max(1, ceil(args.samples / max(1, len(source_paths))))
    for source_idx, source_path in enumerate(source_paths):
        source_start_samples = len(labels)
        seed = None if args.random_seed is None else args.random_seed + source_idx
        for line_no, game in iter_sharded_games(source_path, args.games, seed, args.line_shard_index, args.line_shard_count):
            stats[f"source_{source_idx}_games_seen"] += 1
            for item in iter_training_items(game, line_no, ranker):
                stats["contexts"] += 1
                if item.get("error"):
                    stats[str(item["error"])] += 1
                    continue
                human = item.get("human")
                if not human or human.get("type") == "UNKNOWN" or not item.get("matched", True):
                    stats["human_action_unmatched"] += 1
                    continue

                hand = clean_cards(item["hand"])
                actions = [clean_action(action, idx) for idx, action in enumerate(item["actions"])]
                if len(actions) > args.max_actions:
                    stats["skipped_many_actions"] += 1
                    continue
                if len(hand) > args.max_hand:
                    stats["skipped_large_hand"] += 1
                    continue
                target = action_key(human)
                if target not in {action_key(action) for action in actions}:
                    stats["target_not_in_actions"] += 1
                    continue

                ctx_payload = item["ctx_payload"]
                ctx = build_context(ctx_payload)
                rank_top_k = None if args.keep_misses else TOPK
                ranked = ranker.rank(hand, actions, ctx_payload, top_k=rank_top_k)
                keys = [action_key({"kind": row.action.kind, "cards": row.action.cards}) for row in ranked]
                rank_pos = keys.index(target) + 1 if target in keys else None
                stats["scored"] += 1
                stats[f"source_{source_idx}_scored"] += 1
                bucket = "lead" if ctx.current_kind == "Lead" else "follow"
                stats[f"{bucket}_scored"] += 1
                if rank_pos is not None and rank_pos <= TOPK:
                    stats["top10_hit"] += 1
                    stats[f"source_{source_idx}_top10_hit"] += 1
                    stats[f"{bucket}_top10_hit"] += 1
                    label = rank_pos - 1
                elif args.keep_misses:
                    stats["top10_miss_kept"] += 1
                    label = -1
                else:
                    stats["top10_miss"] += 1
                    continue

                state_feat, cand_feat, mask = featurize_topk(hand, ctx, ranked[:TOPK], top_k=TOPK)
                states.append(state_feat)
                candidates.append(cand_feat)
                masks.append(mask)
                labels.append(label)
                raw_sample_id = item.get("sample_id") or f"{line_no}:{stats['contexts']}"
                sample_ids.append(f"s{source_idx}:{raw_sample_id}")
                replay_id = replay_group_from_sample_id(canonical_sample_id(raw_sample_id))
                replay_ids.append(replay_id)
                retrieval_ranks.append(int(rank_pos or 0))
                human_kinds.append(str(human.get("type") or "UNKNOWN"))
                current_kinds.append(str(ctx_payload.get("current_kind") or "Lead"))
                if len(labels) % max(1, args.progress_every) == 0:
                    elapsed = time.perf_counter() - start_time
                    print(
                        f"stored={len(labels)} scored={stats['scored']} top10_hit={stats['top10_hit']} "
                        f"source={source_idx} elapsed={elapsed:.1f}s",
                        flush=True,
                    )
                source_samples = len(labels) - source_start_samples
                if len(labels) >= args.samples or source_samples >= per_source_target:
                    break
            source_samples = len(labels) - source_start_samples
            if len(labels) >= args.samples or source_samples >= per_source_target:
                break

    if not states:
        raise RuntimeError(f"no training samples written; stats={dict(stats)}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "feature_version": FEATURE_VERSION,
        "top_k": TOPK,
        "state_dim": STATE_DIM,
        "candidate_dim": CANDIDATE_DIM,
        "source_path": source_paths,
        "args": vars(args),
        "stats": dict(stats),
        "dataset_version": "plm_proxy_top10_v1",
        "split_policy": "blake2b(plm_proxy_split_v1:replay_id), 80/10/10",
    }
    np.savez_compressed(
        output,
        state=np.stack(states).astype(np.float32),
        candidates=np.stack(candidates).astype(np.float32),
        mask=np.stack(masks).astype(np.float32),
        label=np.array(labels, dtype=np.int64),
        retrieval_rank=np.array(retrieval_ranks, dtype=np.int16),
        sample_id=np.array(sample_ids),
        replay_id=np.array(replay_ids),
        split=np.array([replay_split_v1(value) for value in replay_ids]),
        current_kind=np.array(current_kinds),
        human_kind=np.array(human_kinds),
        metadata=json.dumps(metadata, ensure_ascii=False),
    )
    elapsed = time.perf_counter() - start_time
    scored = max(1, stats["scored"])
    print(f"wrote={output} samples={len(labels)} elapsed={elapsed:.1f}s")
    print(f"top10_hit={stats['top10_hit']}/{stats['scored']} ({stats['top10_hit'] / scored:.1%})")
    for bucket in ("lead", "follow"):
        subtotal = stats[f"{bucket}_scored"]
        if subtotal:
            print(f"{bucket}_top10_hit={stats[f'{bucket}_top10_hit']}/{subtotal} ({stats[f'{bucket}_top10_hit'] / subtotal:.1%})")


if __name__ == "__main__":
    main()
