#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DATA_RE = re.compile(r"id:4127 data:(?P<data>\{.*\})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse gdai_linux_local round-over logs into PPO reward-json mapping.")
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--scale", type=float, default=3.0, help="Divide gold by this value. Use 1 for raw gold.")
    parser.add_argument("--clip", type=float, default=1.0, help="Clip absolute reward after scaling. Use <=0 to disable.")
    return parser.parse_args()


def _reward(gold: float, scale: float, clip: float) -> float:
    value = gold / scale if scale else gold
    if clip > 0:
        value = max(-clip, min(clip, value))
    return float(value)


def _round_payloads(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = DATA_RE.search(line)
        if not match:
            continue
        try:
            data = json.loads(match.group("data"))
        except json.JSONDecodeError:
            continue
        payload = data.get("DATA") or {}
        extra = payload.get("extra") or {}
        players = extra.get("players") or []
        if players:
            out.append(payload)
    return out


def main() -> None:
    args = parse_args()
    rewards: dict[str, float] = {}
    seen: set[tuple[int, int, int]] = set()
    for log in args.logs:
        for payload in _round_payloads(Path(log)):
            round_count = int(payload.get("round_count") or 0)
            episode_id = max(0, round_count - 1)
            for player in (payload.get("extra") or {}).get("players") or []:
                uid = int(player.get("uid"))
                gold = float(player.get("gold", 0.0))
                key_tuple = (round_count, uid, int(gold * 1000))
                if key_tuple in seen:
                    continue
                seen.add(key_tuple)
                value = _reward(gold, args.scale, args.clip)
                # interactive_server's gdai adapter currently uses game_id like
                # gdai_<uid> for the local client, so emit both episode and
                # round_count forms.
                rewards[f"gdai_{uid}:{uid}:{episode_id}"] = value
                rewards[f"gdai_{uid}:{uid}:{round_count}"] = value
                rewards[f"gdai_{uid}:{uid}"] = value

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rewards, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote={output} rewards={len(rewards)}")


if __name__ == "__main__":
    main()
