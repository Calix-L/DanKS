#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


OWN_UID_RE = re.compile(r"\bid:(?P<uid>\d+)\s+token:")
ROUND_DATA_RE = re.compile(r"id:4127 data:(?P<data>\{.*\})")
BALANCE_DATA_RE = re.compile(r"id:4136 data:(?P<data>\{.*\})")
CREATE_RE = re.compile(r"onCreateDesk msg:.*\{0 450 (?P<desk>\d+)\}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize remote gdai/callRobot evaluation logs against PLM robots."
    )
    parser.add_argument("logs", nargs="+", help="gdai_linux_local create/join logs")
    parser.add_argument("--ai-uid", type=int, action="append", default=[], help="UID controlled by our policy. Can be repeated.")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--fail-if-no-rounds", action="store_true")
    return parser.parse_args()


def _load_json_from_line(pattern: re.Pattern[str], line: str) -> dict[str, Any] | None:
    match = pattern.search(line)
    if not match:
        return None
    try:
        return json.loads(match.group("data"))
    except json.JSONDecodeError:
        return None


def _read(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def collect(paths: list[Path], explicit_ai_uids: set[int]) -> dict[str, Any]:
    ai_uids: set[int] = set(explicit_ai_uids)
    desks: set[int] = set()
    round_payloads: dict[int, dict[str, Any]] = {}
    balances: list[dict[str, Any]] = []

    for path in paths:
        for line in _read(path):
            if not explicit_ai_uids:
                own = OWN_UID_RE.search(line)
                if own:
                    ai_uids.add(int(own.group("uid")))
            desk = CREATE_RE.search(line)
            if desk:
                desks.add(int(desk.group("desk")))

            round_data = _load_json_from_line(ROUND_DATA_RE, line)
            if round_data:
                payload = round_data.get("DATA") or {}
                extra = payload.get("extra") or {}
                players = extra.get("players") or []
                round_count = int(payload.get("round_count") or 0)
                if round_count > 0 and players:
                    round_payloads.setdefault(round_count, payload)

            balance_data = _load_json_from_line(BALANCE_DATA_RE, line)
            if balance_data:
                payload = balance_data.get("DATA") or {}
                data = payload.get("data") or {}
                if data.get("teams"):
                    balances.append(payload)

    rounds = []
    wins = losses = draws = 0
    gold_sum = 0
    first_count = 0
    top3_count = 0
    valid_ai_round_entries = 0
    for round_count in sorted(round_payloads):
        payload = round_payloads[round_count]
        players = (payload.get("extra") or {}).get("players") or []
        ai_players = [p for p in players if int(p.get("uid", -1)) in ai_uids]
        ai_gold = sum(int(p.get("gold") or 0) for p in ai_players)
        gold_sum += ai_gold
        valid_ai_round_entries += len(ai_players)
        for p in ai_players:
            win_index = int(p.get("win_index") or 0)
            if win_index == 1:
                first_count += 1
            if 1 <= win_index <= 3:
                top3_count += 1
        if ai_gold > 0:
            outcome = "win"
            wins += 1
        elif ai_gold < 0:
            outcome = "loss"
            losses += 1
        else:
            outcome = "draw"
            draws += 1
        rounds.append(
            {
                "round_count": round_count,
                "ai_gold": ai_gold,
                "outcome": outcome,
                "ai_players": [
                    {
                        "uid": int(p.get("uid")),
                        "gold": int(p.get("gold") or 0),
                        "win_index": int(p.get("win_index") or 0),
                        "remaining": len(p.get("hand") or []),
                    }
                    for p in ai_players
                ],
                "players": [
                    {
                        "uid": int(p.get("uid")),
                        "gold": int(p.get("gold") or 0),
                        "win_index": int(p.get("win_index") or 0),
                        "remaining": len(p.get("hand") or []),
                    }
                    for p in players
                ],
            }
        )

    final_balance = balances[-1] if balances else None
    final_is_winner = None
    if final_balance:
        teams = ((final_balance.get("data") or {}).get("teams") or [])
        for team in teams:
            members = team.get("members") or []
            member_uids = {int(m.get("uid")) for m in members if m.get("uid") is not None}
            if ai_uids & member_uids:
                final_is_winner = bool(team.get("is_winner"))
                break

    total = wins + losses + draws
    return {
        "logs": [str(p) for p in paths],
        "desk_ids": sorted(desks),
        "ai_uids": sorted(ai_uids),
        "rounds": rounds,
        "summary": {
            "rounds": total,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": wins / total if total else 0.0,
            "gold_sum": gold_sum,
            "avg_gold_per_round": gold_sum / total if total else 0.0,
            "ai_round_entries": valid_ai_round_entries,
            "first_rate_per_entry": first_count / valid_ai_round_entries if valid_ai_round_entries else 0.0,
            "top3_rate_per_entry": top3_count / valid_ai_round_entries if valid_ai_round_entries else 0.0,
            "final_match_winner": final_is_winner,
        },
    }


def main() -> None:
    args = parse_args()
    paths = [Path(p) for p in args.logs]
    result = collect(paths, set(args.ai_uid))
    summary = result["summary"]
    if args.fail_if_no_rounds and int(summary["rounds"]) == 0:
        raise SystemExit("no completed rounds found")
    print(
        "remote_plm_eval "
        f"rounds={summary['rounds']} wins={summary['wins']} losses={summary['losses']} draws={summary['draws']} "
        f"win_rate={summary['win_rate']:.3f} gold_sum={summary['gold_sum']} "
        f"avg_gold={summary['avg_gold_per_round']:.2f} ai_uids={result['ai_uids']} "
        f"desk_ids={result['desk_ids']} final_match_winner={summary['final_match_winner']}"
    )
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote={output}")


if __name__ == "__main__":
    main()
