#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys


def main() -> None:
    path = Path(sys.argv[1])
    failures: list[str] = []
    records = 0
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            row = json.loads(line)
            records += 1
            action = row.get("chosen_action") or {}
            if row.get("mode") == "lead" and action.get("kind") == "PASS":
                failures.append(f"line {line_no}: lead PASS")
            if int(row.get("action_slot", -1)) not in range(10):
                failures.append(f"line {line_no}: action slot outside Top10")
            if row.get("policy_mode") != "selector":
                failures.append(f"line {line_no}: policy mode is not selector")
    if records == 0:
        failures.append("no policy decisions recorded")
    if failures:
        raise SystemExit("; ".join(failures[:20]))
    print(json.dumps({"ok": True, "records": records, "lead_pass": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
