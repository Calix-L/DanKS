#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import sys


def main() -> None:
    health = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    expected_checkpoint = str(Path(os.environ["CHECKPOINT"]).resolve())
    checks = {
        "ok": health.get("ok") is True,
        "top_n": health.get("top_n") == 10,
        "max_partitions": health.get("max_partitions") == 8,
        "lead_max_partitions": health.get("lead_max_partitions") == 8,
        "follow_max_partitions": health.get("follow_max_partitions") == 8,
        "policy_mode": health.get("policy_mode") == "selector",
        "selector_device": health.get("selector_device") == "onnx-cpu",
        "selector_model_class": health.get("selector_model_class")
        == "OnnxPhase14Selector",
        "history_protocol": health.get("history_protocol")
        == "public_action_history_v1_15x64",
        "history_event_semantics": health.get("history_event_semantics")
        == "actual_public_actions_deduplicated_v1",
        "checkpoint": str(Path(str(health.get("selector_checkpoint"))).resolve()) == expected_checkpoint,
        "sample_policy": health.get("sample_policy") is False,
        "rank_top_k_only": health.get("rank_top_k_only") is True,
        "fast_approx_rank": health.get("fast_approx_rank") is True,
        "post_selector_constraints": health.get("post_selector_constraints") is False,
        "exact_best_cache_size": health.get("exact_best_cache_size") == 0,
        "partitioner_cache_size": health.get("partitioner_cache_size") == 0,
        "retrieval_profile": health.get("retrieval_profile") == "asset-tier-v2",
        "break_group_weight": float(health.get("break_group_weight", 0.0)) == 46.0,
        "approx_action_limit": health.get("approx_action_limit") == 96,
        "partition_hand_count_window": health.get("partition_hand_count_window") == 1,
        "partition_hand_count_window_min_hand": health.get("partition_hand_count_window_min_hand") == 21,
        "partition_hand_count_max_covers": health.get("partition_hand_count_max_covers") == 256,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise SystemExit(f"policy alignment check failed: {failures}; health={health}")
    print(json.dumps({"ok": True, "checks": sorted(checks)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
