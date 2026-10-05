# 掼蛋数据集

[English](README.md) | **简体中文**

这里链接 [CardKS / KSCB](https://github.com/Calix-L/CardKS/tree/main/KSCB) 的人类掼蛋牌局记录，与 DanKS 智能体和网页对战服务配套阅读。

| 完整升级赛 | 小局 | 决策点 |
| ---: | ---: | ---: |
| 899 | 10,218 | 840,194 |

## 获取数据

[查看原始文件](https://github.com/Calix-L/CardKS/blob/main/KSCB/data/guandan_matches.jsonl.gz) · [直接下载](https://raw.githubusercontent.com/Calix-L/CardKS/main/KSCB/data/guandan_matches.jsonl.gz) · [完整格式说明](https://github.com/Calix-L/CardKS/blob/main/KSCB/README.zh-CN.md)

在 DanKS 仓库根目录运行：

```bash
curl -fL https://raw.githubusercontent.com/Calix-L/CardKS/main/KSCB/data/guandan_matches.jsonl.gz \
  -o datasets/guandan_matches.jsonl.gz
```

压缩文件约 17 MB。下载到本地的文件已加入 Git 忽略规则，原始数据继续由 CardKS 统一维护。

## 读取一场比赛

文件采用 gzip 压缩的 JSON Lines 格式，每行是一场完整升级赛，包含按顺序排列的 `rounds`。

```python
import gzip
import json

with gzip.open("datasets/guandan_matches.jsonl.gz", "rt", encoding="utf-8") as stream:
    match = json.loads(next(stream))

print(match["sample_id"], match["round_count"])
print(match["rounds"][0].keys())
```

比赛对象包含 `game`、`sample_id`、`round_count` 和 `rounds`。每个小局包含 `players`、`game_rule`、按序排列的 `events` 以及终局 `result`；`game_rule` 是序列化后的 JSON 字符串，牌使用整数编码。事件中 `event_id` 为 3 表示出牌，为 4 表示不出。

可以用它分析人类决策，或构建自己的回放与预处理流程。用于 DanKS 训练时，需要先重建可见状态与合法动作，再按所选 AI 版本编码；下载文件不是现成的 PPO rollout。

[返回 DanKS](../README.zh-CN.md)
