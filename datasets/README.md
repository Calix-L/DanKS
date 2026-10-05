# GuanDan dataset

**English** | [简体中文](README.zh-CN.md)

Human GuanDan match records from [CardKS / KSCB](https://github.com/Calix-L/CardKS/tree/main/KSCB), linked here alongside the DanKS agent and game service.

| Complete promotion matches | Rounds | Decision points |
| ---: | ---: | ---: |
| 899 | 10,218 | 840,194 |

## Get the data

[Browse the original file](https://github.com/Calix-L/CardKS/blob/main/KSCB/data/guandan_matches.jsonl.gz) · [Download](https://raw.githubusercontent.com/Calix-L/CardKS/main/KSCB/data/guandan_matches.jsonl.gz) · [Format reference](https://github.com/Calix-L/CardKS/blob/main/KSCB/README.md)

From the DanKS repository root:

```bash
curl -fL https://raw.githubusercontent.com/Calix-L/CardKS/main/KSCB/data/guandan_matches.jsonl.gz \
  -o datasets/guandan_matches.jsonl.gz
```

The compressed file is approximately 17 MB. Local downloads are Git-ignored; the original data remains maintained in CardKS.

## Read a match

The file is gzip-compressed JSON Lines. Each line contains one complete promotion match with ordered `rounds`.

```python
import gzip
import json

with gzip.open("datasets/guandan_matches.jsonl.gz", "rt", encoding="utf-8") as stream:
    match = json.loads(next(stream))

print(match["sample_id"], match["round_count"])
print(match["rounds"][0].keys())
```

A match contains `game`, `sample_id`, `round_count`, and `rounds`. Each round includes `players`, `game_rule`, ordered `events`, and the terminal `result`. `game_rule` is a serialized JSON string; card codes are integers. In the event stream, `event_id` 3 denotes a play and 4 denotes a pass.

Use these records to analyze human choices or build your own replay and preprocessing pipeline. For DanKS training, first reconstruct the visible state and legal actions, then encode them for the chosen AI generation. The download is not a ready-made PPO rollout.

[Back to DanKS](../README.md)
