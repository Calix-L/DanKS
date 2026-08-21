#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from DanKS.plm_eval.summarize import collect  # noqa: E402
from DanKS.plm_eval.pailemen_native import (  # noqa: E402
    EXPECTED_GDAI_SHA256,
    PAILEMEN_NATIVE_OPPONENT,
    TARGET_RULES,
    sha256_file,
    validate_native_eval_manifest,
)


LOG_PATHS_RE = re.compile(
    r"Logs: create=(?P<create>\S+) join=(?P<join>\S+) robot=(?P<robot>\S+)"
)
REPLAY_RE = re.compile(r'"replay_id":"(?P<replay>[0-9a-fA-F]+)"')
RULES_4136_RE = re.compile(r"id:4136 data:(?P<data>\{.*\})")
DESK_RE = re.compile(r"SUCCESS desk_id=(?P<desk>\d+)")
CHAIRS_RE = re.compile(
    r"desk_id=(?P<desk>\d+) create_pid=\d+ join_pid=\d+ "
    r"create_chair=(?P<create>\d+) join_chair=(?P<join>\d+)"
)
OPPOSITE_RE = re.compile(r"opposite_ok desk_id=(?P<desk>\d+)")
NATIVE_RE = re.compile(
    r"native_opponent_verified opponent_id=(?P<opponent>\S+) "
    r"gdai_sha256=(?P<sha>[0-9a-f]{64}) call_robot_url=(?P<url>\S+) "
    r"robot_count=(?P<count>\d+)"
)
FORBIDDEN_RUNTIME_MARKERS = (
    "Traceback (most recent call last)",
    "DiscardFail",
    "AI offer failed",
    "panic:",
    "lead PASS",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail-closed finalizer for PaiLeMen client-native Rule AI evaluations."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--gdai-bin", required=True, type=Path)
    parser.add_argument("--expected-matches", required=True, type=int)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _resolve_logged_path(raw: str, run_dir: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    for candidate in (ROOT / path, run_dir / path, path):
        if candidate.exists():
            return candidate.resolve()
    return path.resolve()


def _terminal_payload(lines: list[str]) -> dict[str, Any] | None:
    for line in reversed(lines):
        match = RULES_4136_RE.search(line)
        if not match:
            continue
        try:
            return json.loads(match.group("data")).get("DATA") or {}
        except json.JSONDecodeError:
            continue
    return None


def _table_from_status(status_path: Path, run_dir: Path) -> tuple[dict[str, Any] | None, list[str]]:
    failures: list[str] = []
    status_text = status_path.read_text(encoding="utf-8", errors="replace")
    paths_matches = list(LOG_PATHS_RE.finditer(status_text))
    desk_matches = list(DESK_RE.finditer(status_text))
    native_matches = list(NATIVE_RE.finditer(status_text))
    paths_match = paths_matches[-1] if paths_matches else None
    desk_match = desk_matches[-1] if desk_matches else None
    native_match = native_matches[-1] if native_matches else None
    if not paths_match:
        return None, [f"{status_path.name}:client_log_paths_missing"]
    if not desk_match:
        return None, [f"{status_path.name}:success_desk_missing"]
    final_desk = int(desk_match.group("desk"))
    chairs_match = next(
        (
            match
            for match in reversed(list(CHAIRS_RE.finditer(status_text)))
            if int(match.group("desk")) == final_desk
        ),
        None,
    )
    opposite_verified = any(
        int(match.group("desk")) == final_desk
        for match in OPPOSITE_RE.finditer(status_text)
    )
    if not chairs_match:
        failures.append(f"{status_path.name}:chair_assignment_missing")
    if not native_match:
        failures.append(f"{status_path.name}:native_identity_record_missing")
    paths = {
        key: _resolve_logged_path(value, run_dir)
        for key, value in paths_match.groupdict().items()
    }
    for key, path in paths.items():
        if not path.is_file():
            failures.append(f"{status_path.name}:{key}_log_missing:{path}")
    if failures:
        return None, failures
    create_lines = paths["create"].read_text(
        encoding="utf-8", errors="replace"
    ).splitlines()
    join_lines = paths["join"].read_text(encoding="utf-8", errors="replace").splitlines()
    robot_response = paths["robot"].read_text(encoding="utf-8", errors="replace").strip()
    combined_text = "\n".join(create_lines + join_lines)
    terminal = _terminal_payload(create_lines) or _terminal_payload(join_lines)
    terminal_data = (terminal or {}).get("data") or {}
    room_rules = terminal_data.get("rule") or {}
    replay_match = REPLAY_RE.search(combined_text)
    if not opposite_verified:
        failures.append(f"{status_path.name}:opposite_seats_not_verified")
    if "native_opponent_verified" not in status_text:
        failures.append(f"{status_path.name}:native_opponent_identity_not_verified")
    if chairs_match and abs(int(chairs_match.group("create")) - int(chairs_match.group("join"))) != 2:
        failures.append(f"{status_path.name}:chairs_not_opposite")
    if native_match:
        if native_match.group("opponent") != PAILEMEN_NATIVE_OPPONENT:
            failures.append(f"{status_path.name}:native_opponent_id_mismatch")
        if native_match.group("sha") != EXPECTED_GDAI_SHA256:
            failures.append(f"{status_path.name}:native_binary_hash_mismatch")
        if int(native_match.group("count")) != 2:
            failures.append(f"{status_path.name}:native_robot_count_mismatch")
    if robot_response != "ok":
        failures.append(f"{status_path.name}:call_robot_response:{robot_response!r}")
    if terminal is None:
        failures.append(f"{status_path.name}:terminal_4136_missing")
    if replay_match is None:
        failures.append(f"{status_path.name}:replay_id_missing")
    for marker in FORBIDDEN_RUNTIME_MARKERS:
        if marker in status_text or marker in combined_text:
            failures.append(f"{status_path.name}:runtime_marker:{marker}")
    result = collect([paths["create"], paths["join"]], set())
    summary = result["summary"]
    ai_uids = result["ai_uids"]
    if len(ai_uids) != 2:
        failures.append(f"{status_path.name}:ai_uid_count:{len(ai_uids)}")
    if summary.get("final_match_winner") not in (True, False):
        failures.append(f"{status_path.name}:final_match_winner_missing")
    table = {
        "status_log": str(status_path.resolve()),
        "status_log_sha256": sha256_file(status_path),
        "client_logs": {key: str(path) for key, path in paths.items()},
        "client_log_sha256": {key: sha256_file(path) for key, path in paths.items()},
        "desk_id": final_desk,
        "create_chair": int(chairs_match.group("create")) if chairs_match else None,
        "join_chair": int(chairs_match.group("join")) if chairs_match else None,
        "replay_id": replay_match.group("replay") if replay_match else "",
        "ai_uids": ai_uids,
        "opponent_id": PAILEMEN_NATIVE_OPPONENT,
        "opposite_seats_verified": opposite_verified,
        "call_robot_verified": robot_response == "ok",
        "call_robot_url": native_match.group("url") if native_match else "",
        "robot_count": int(native_match.group("count")) if native_match else 0,
        "terminal_4136_verified": terminal is not None,
        "room_rules": {key: room_rules.get(key) for key in TARGET_RULES},
        "rounds": int(summary.get("rounds", 0)),
        "round_wins": int(summary.get("wins", 0)),
        "round_losses": int(summary.get("losses", 0)),
        "round_draws": int(summary.get("draws", 0)),
        "gold_sum": int(summary.get("gold_sum", 0)),
        "avg_gold_per_round": float(summary.get("avg_gold_per_round", 0.0)),
        "final_match_winner": summary.get("final_match_winner"),
    }
    return table, failures


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    failures: list[str] = []
    checkpoint = args.checkpoint.expanduser().resolve()
    gdai_bin = args.gdai_bin.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    if not checkpoint.is_file():
        failures.append(f"checkpoint_missing:{checkpoint}")
    if not gdai_bin.is_file():
        failures.append(f"gdai_binary_missing:{gdai_bin}")
    checkpoint_sha = sha256_file(checkpoint) if checkpoint.is_file() else "missing"
    gdai_sha = sha256_file(gdai_bin) if gdai_bin.is_file() else "missing"
    if gdai_sha != EXPECTED_GDAI_SHA256:
        failures.append(f"gdai_binary_sha256:{gdai_sha}")
    tables: list[dict[str, Any]] = []
    status_paths = sorted(run_dir.glob("**/*_slot*.status.log"))
    for status_path in status_paths:
        table, table_failures = _table_from_status(status_path, run_dir)
        failures.extend(table_failures)
        if table is not None:
            tables.append(table)
    if len(tables) != args.expected_matches:
        failures.append(f"matches_completed:{len(tables)}!={args.expected_matches}")
    # Desk ids belong to a small server-side pool and are legitimately reused
    # after a table closes. The replay id is the durable match identity.
    replay_ids = [str(table.get("replay_id") or "") for table in tables]
    nonempty_replays = [value for value in replay_ids if value]
    if len(set(nonempty_replays)) != len(nonempty_replays):
        failures.append("duplicate_replay_ids")
    rounds = sum(int(table["rounds"]) for table in tables)
    round_wins = sum(int(table["round_wins"]) for table in tables)
    gold_sum = sum(int(table["gold_sum"]) for table in tables)
    metrics = {
        "final_match_wins": sum(table.get("final_match_winner") is True for table in tables),
        "final_match_losses": sum(table.get("final_match_winner") is False for table in tables),
        "final_match_win_rate": (
            sum(table.get("final_match_winner") is True for table in tables) / len(tables)
            if tables
            else 0.0
        ),
        "rounds": rounds,
        "round_wins": round_wins,
        "round_win_rate": round_wins / rounds if rounds else 0.0,
        "gold_sum": gold_sum,
        "avg_gold_per_round": gold_sum / rounds if rounds else 0.0,
        "legal_action_rate": 1.0 if not failures else 0.0,
    }
    manifest: dict[str, Any] = {
        "version": "pailemen_eval_manifest_v1",
        "opponent_id": PAILEMEN_NATIVE_OPPONENT,
        "evaluation_unit": "full_upgrade_match",
        "common_random_numbers": False,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "gdai_binary": str(gdai_bin),
        "gdai_binary_sha256": gdai_sha,
        "room_rules": dict(TARGET_RULES),
        "matches_expected": int(args.expected_matches),
        "matches_completed": len(tables),
        "tables": tables,
        "metrics": metrics,
        "quality_gate": {"passed": not failures, "failures": sorted(set(failures))},
    }
    validation_failures = validate_native_eval_manifest(
        {**manifest, "quality_gate": {"passed": True, "failures": []}},
        expected_checkpoint_sha256=checkpoint_sha,
        expected_matches=args.expected_matches,
    )
    if validation_failures:
        manifest["quality_gate"]["failures"] = sorted(
            set(manifest["quality_gate"]["failures"] + validation_failures)
        )
        manifest["quality_gate"]["passed"] = False
    return manifest


def main() -> None:
    args = parse_args()
    manifest = build_manifest(args)
    output = args.output or args.run_dir / "pailemen_eval_manifest.json"
    atomic_json(output, manifest)
    print(json.dumps({"output": str(output), **manifest["quality_gate"]}, ensure_ascii=False))
    if not manifest["quality_gate"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
