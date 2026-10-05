# V3Pro 接入指南

[← 返回 DanKS](../../README.zh-CN.md)

## V3Pro · 无需重训的决策增强

V3Pro 是 V3 的可选**纯源码增强包**，不是第四套网络，也不替换训练流程。它以 `DanKSPro` 与 V3 共存；原网络、特征和 PPO 训练实现仍由 `DanKS` 提供，不复制整套代码。

**出牌顺序：** Retrieval → 配牌 mask → 冻结 V3 → 等价理牌 → 残局验证。

| 组件 | 作用 | 明确边界 |
| --- | --- | --- |
| [配牌 Safe Gate](DanKSPro/safety.py) | 存在同牌型、同强度且保护指标不变差的替代出法时，屏蔽多花配牌或多拆天然炸弹的普通候选。 | 不是禁止配牌。可一手出完或对手剩 1–2 张时保留原 mask；不一刀切禁止送单、送双。 |
| [等价理牌规则](DanKSPro/rules.py) | 三带二保留原来的三张，仅在剩余拆分等价时换小对子；跟牌时保守选择足够压制的小炸弹。 | 不任意换三张、不改主动出炸弹，也不为换小牌破坏剩牌结构。 |
| [残局搜索](DanKSPro/endgame/runtime.py) | 公开暗牌枚举、冻结策略续局、精确 Minimax、完整证据下的平局处理，以及独立候选恢复。 | 全桌总剩牌 ≤16；11–16 张还要求暗牌分配数 ≤128。验证不完整就保留基础动作。 |

### 使用 V3Pro

使用 Python 3.11+，按照[开发指南](../../.github/guides/DEVELOPMENT.zh-CN.md)安装 V3 所需 PyTorch：

```bash
python -m pip install -e . -e versions/v3 -e versions/v3pro
python examples/v3pro_smoke.py
```

示例使用**随机初始化模型与合成残局**，验证接入和可恢复搜索，不代表模型棋力。训练好的主模型／专家权重、私有数据集及内部评测报告**不随仓库分发**。

```python
from DanKSPro import ProPolicy

policy = ProPolicy.from_checkpoints(
    "checkpoints/main.pt",
    specialist="checkpoints/endgame.pt",
    extended_specialist="checkpoints/endgame_extended.pt",
)
action_id, record = policy.act(hand, context, legal_actions, history=public_history)
```

上述路径由调用方自行提供，并非仓库附带文件。专家元数据中的父模型 SHA-256 必须与主权重一致。旧 checkpoint 如需 Python pickle 加载，只有**确认文件可信**时才能设置 `trusted=True`。没有专家权重时仍可运行前两个增强组件，学习型残局模块会明确放弃接管。

需要搜索时，将 record、公开信息重建的引擎状态、自家手牌、全部已出牌和四家绝对座位剩牌数传给 `policy.refine_endgame(...)`，完整调用见[可运行示例](../../examples/v3pro_smoke.py)。注意：Retrieval 的剩牌数按**自己、下家、队友、上家**排列；历史座位和搜索剩牌数使用**绝对座位**。请提供本小局完整公开历史，它同时用于记牌与序列特征，并独立复制到 record。适配器核对决策上下文、已出牌历史和合法动作后才转换候选编号与引擎位置。搜索可以检查召回之外的合法动作，但不能重新启用被 Safe Gate 明确屏蔽的动作。

残局移植沿用 r13 方法：准入范围内补齐合法动作、严格验证、11–16 张完整精确平局与策略续局验证、联合搜索不完整后的独立候选恢复，**不包含后续 r14 实验**。暗牌枚举下的 Minimax 仍存在不完全信息博弈的策略融合局限。预算是节点上限，**不是出牌墙钟期限**；本次为源码发布，不等于已验收的低延迟服务，也不把内部评测收益直接当作部署胜率。
