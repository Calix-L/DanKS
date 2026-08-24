<p align="right">
  <a href="README.md">English</a> | <strong>简体中文</strong>
</p>

<h1 align="center">DanKS</h1>

<p align="center">
  <strong>一个紧凑、代码优先的仓库，完整呈现三代掼蛋 AI 的技术演进</strong>
</p>

<p align="center">
  <a href="#快速开始">快速开始</a> ·
  <a href="#系统架构">系统架构</a> ·
  <a href="#三代技术路线">三代路线</a> ·
  <a href="#使用-ppo-训练-v3">训练</a> ·
  <a href="https://github.com/Calix-L/CardKS">CardKS 论文主页</a>
</p>

<p align="center">
  <a href="https://github.com/Calix-L/DanKS/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/Calix-L/DanKS/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://github.com/Calix-L/DanKS/releases/latest"><img alt="Release" src="https://img.shields.io/github/v/release/Calix-L/DanKS"></a>
  <a href="https://www.python.org/"><img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white"></a>
  <a href="LICENSE"><img alt="Apache-2.0 许可证" src="https://img.shields.io/badge/license-Apache--2.0-D22128"></a>
</p>

DanKS 保留了一个四人组队掼蛋智能体从结构化检索、学习型候选选择器，到基于 PPO 训练的记忆感知策略的演进过程。每一代都可独立安装，并共用 `DanKS` Python 命名空间；仓库同时提供一套共享的 108 张牌掼蛋规则引擎作为基础。

> 本仓库仅公开源代码。模型权重、数据集、评测记录、私有文档、凭据和部署自动化内容均不在公开范围内。

## 快速开始

最短的可运行路径是在 CPU 上使用 V3。以下命令以 Python 3.11 和 POSIX shell 为例：

```bash
git clone https://github.com/Calix-L/DanKS.git
cd DanKS
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e versions/v3
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python examples/retrieval_quickstart.py --version v3
python examples/v3_model_smoke.py
```

Windows PowerShell 请使用 `.venv\Scripts\Activate.ps1`。CUDA、昇腾 NPU、V1/V2、原生内核与开发环境请参阅[安装参考](#安装参考)。

## 系统架构

![DanKS V3 从掼蛋信息状态、结构化候选检索和 Actor-Critic 评分到 PPO 自对弈的完整流程](assets/danks-v3-architecture.png)

V3 将庞大且高度结构化的动作空间，压缩成一次边界清晰的策略决策：

1. **编码信息状态。** 策略网络接收可见手牌、公开出牌历史、合法动作和座位相关的对局上下文。
2. **检索结构化候选。** 有预算的拆牌搜索生成具有代表性的出牌方案，并概括牌数、对子、序列、花色、缺口以及剩余手牌结构。
3. **评分有界的 Top-K 候选集。** 共享编码器融合状态、候选动作和结构特征；Actor 对有效候选排序，Critic 估计当前状态价值。
4. **从自对弈中学习。** 轨迹数据为裁剪 PPO 更新提供 GAE 优势估计，在不增加推理阶段候选预算的前提下改进选择器。

上图展示完整的 V3 路径。V1 和 V2 是同一技术进程中更早、可独立使用的阶段：先引入结构化检索，再引入学习型选择，最终发展为记忆感知的 PPO 策略。

## 为什么选择 DanKS？

- **三代实现清晰可读** —— 可以对照完整代码，不会混用不同世代的特征或 checkpoint 协议。
- **每代都可隔离安装** —— 各代拥有独立的包元数据与依赖契约。
- **V3 端到端 PPO 训练** —— 包含 PPO learner、rollout 加载、优化器状态、持久化传输、召回与队伍信念建模。
- **共享掼蛋引擎** —— 合法动作、牌桌生命周期、进贡还贡与结算都收敛在一个可复用的包中。
- **公开边界精简** —— 不包含重复归档、自动生成的清单、模型二进制文件或平台专用评测代码。

## 三代技术路线

| 版本 | 核心思路 | 主要增量 | 入口 |
| --- | --- | --- | --- |
| **V1** | 结构化检索 | 候选评分和 NumPy 选择器 | [`ranker.py`](versions/v1/DanKS/retrieval/ranker.py) |
| **V2** | 学习型选择 | 更广的动作生成和 ONNX 选择器 | [`action_generator.py`](versions/v2/DanKS/retrieval/action_generator.py) |
| **V3** | 记忆感知策略学习 | 记牌、候选覆盖、召回、队伍信念和 PPO | [`model.py`](versions/v3/DanKS/training/model.py) |

三个版本被有意隔离。请一次只选择一代；它们的特征 schema 和 checkpoint 不可互换。

## 为什么要关注延迟结果？

<p align="center">
  <img
    src="assets/structure-aware-delayed-outcomes.png"
    alt="同一掼蛋状态下的三个候选动作导向不同的延迟结构结果"
    width="620"
  />
</p>

一个当下代价很低的动作，可能破坏手中唯一有用的组合；而主动消耗一张高价值牌，反而可能保留整体牌型结构，并带来更干净的后续出完路径。DanKS 将学习这种差异所需的职责进行了拆分：

- **Retrieval** 保留具有策略多样性的候选集，避免将完整组合动作空间粗暴展平。
- **结构特征** 显式表达每个候选会消耗什么、保留什么，以及出牌后留下什么。
- **Actor** 为当前状态下的合法候选动作评分。
- **Critic 和 GAE** 从后续轨迹结果中分配信用，使 PPO 能够偏好价值需要数次决策后才体现的动作。

上图是一个用于解释信用分配的概念示例。V3 在推理时会直接评分检索得到的候选动作，并不会显式模拟图中的三条未来分支。

## 仓库结构

```text
DanKS/
├── assets/             # 架构图与决策示意图
├── versions/
│   ├── v1/DanKS/       # 结构化检索 + NumPy 选择器
│   ├── v2/DanKS/       # 结构化检索 + ONNX 选择器
│   └── v3/DanKS/       # 结构化检索 + 神经策略 + PPO
├── guandan/engine/     # 共享 Python 规则引擎
├── examples/           # 可执行的引擎、检索和模型冒烟示例
├── tests/              # 仓库与引擎检查
├── README.zh-CN.md     # 完整的简体中文指南
├── pyproject.toml
├── LICENSE
└── NOTICE
```

## 安装参考

共享引擎和 V3 支持 Python 3.10 及更高版本；V1 和 V2 需要 Python 3.11 及更高版本。以下命令均从仓库根目录运行。由于三个世代使用同一个 `DanKS` 导入命名空间，每个虚拟环境中请**只安装一个世代**。

### 选择安装包

| 目标 | 安装命令 | 说明 |
| --- | --- | --- |
| 共享规则引擎与测试 | `python -m pip install -e '.[dev]'` | 不安装任何 AI 世代。 |
| V1 · 结构化检索 | `python -m pip install -e versions/v1` | NumPy 选择器；Python 3.11+。 |
| V2 · 学习型选择 | `python -m pip install -e versions/v2` | ONNX 选择器；Python 3.11+。 |
| V3 · PPO 策略 | `python -m pip install -e versions/v3` | 还需安装下方一种 PyTorch 构建。 |

### 为 V3 选择一种 PyTorch 构建

| 目标 | 命令 |
| --- | --- |
| Linux / Windows CPU | `python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu` |
| NVIDIA CUDA 12.8 | `python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128` |
| macOS CPU | `python -m pip install torch==2.8.0` |

如果平台需要不同 wheel，请参考 [PyTorch 官方安装矩阵](https://pytorch.org/get-started/previous-versions/)。使用 `python examples/v3_model_smoke.py` 检查安装，使用 `python -m DanKS.training.train_ppo --help` 查看 learner 的全部参数。

<details>
<summary><strong>昇腾 NPU 环境</strong></summary>

昇腾运行时与主机驱动及 CANN 安装紧密耦合。请先安装匹配的 CANN，再使用厂商提供的 PyTorch 与 `torch_npu` wheel。已验证组合记录在 [`requirements-training-npu.txt`](versions/v3/DanKS/environment/requirements-training-npu.txt)。

```bash
source /usr/local/Ascend/cann/set_env.sh
python3.10 -m venv --system-site-packages .venv-v3-npu
source .venv-v3-npu/bin/activate
python -m pip install -e versions/v3
python -m pip install --no-deps \
  /path/to/torch-2.7.1+cpu-cp310-cp310-manylinux_2_28_x86_64.whl \
  /path/to/torch_npu-2.7.1.post2-cp310-cp310-manylinux_2_28_x86_64.whl
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
python -m DanKS.training.train_ppo --help
```

不要在该环境中混装 CUDA 软件包。如果驱动、CANN、处理器架构或 Python 版本不同，请获取匹配的厂商 wheel，不要强行安装上述版本。

</details>

<details>
<summary><strong>已验证配置</strong></summary>

| 目标 | 系统 | Python | 框架 | 关键软件包 |
| --- | --- | --- | --- | --- |
| CI 与共享引擎 | Linux | 3.10, 3.12 | — | pytest 7+ |
| V1 | CPU | 3.11+ | NumPy 选择器 | NumPy 2.4.6 |
| V2 | CPU | 3.11+ | ONNX 选择器 | NumPy 2.4.6, ONNX Runtime 1.27.0 |
| V3 NVIDIA 服务器 | H100, driver 575.57.08 | 3.11.14 | PyTorch 2.8.0 + CUDA 12.8 | NumPy 2.4.6, pybind11 3.0.4 |
| V3 昇腾服务器 | Ubuntu 22.04.5, 910B2C, driver 24.1.0, CANN 8.5.0 | 3.10.12 | PyTorch 2.7.1 + torch_npu 2.7.1.post2 | NumPy 1.26.0, pybind11 3.0.4 |

这些配置用于复现，不是最低硬件要求。

</details>

### 可选的 V3 C++ 加速

优化后的检索内核目前支持 Linux 和 macOS，需要 C++17 编译器、Python 开发头文件和 `pybind11`；Windows 使用 Python 回退实现。请先安装一次平台工具链：

```bash
# Ubuntu/Debian
sudo apt-get update && sudo apt-get install -y build-essential python3-dev

# macOS（仅需执行一次）
xcode-select --install
```

然后在已激活的 V3 环境中，用一条命令完成两个内核的构建与验证：

```bash
danks-build-native
```

该命令会自动定位已安装的 V3 源码，成功后输出 `cover=True, actor=True`。Linux 构建使用主机特定的编译优化；macOS 将架构选择交给 Python 工具链，因此仍然支持 universal2 构建。更换 Python 版本或 CPU 架构后请重新运行。Windows 继续使用 Python 回退实现。

### 开发检查

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

## 运行示例

所有示例都有意保持简小，不需要预训练权重或私有数据：

```bash
# 共享规则引擎；可在基础环境中运行。
python examples/engine_quickstart.py

# 结构化检索；请在匹配的 V1、V2 或 V3 环境中运行。
python examples/retrieval_quickstart.py --version v3

# 完整 V3 网络前向传播；请在已安装 PyTorch 的 V3 环境中运行。
python examples/v3_model_smoke.py

# 通过 V3 PPO learner 执行一次合成优化器更新。
python examples/v3_ppo_smoke.py
```

每条命令都包含内部断言；如果契约被破坏，命令会以非零状态退出。

## 共享游戏引擎

规则引擎可以独立于三代 AI 使用：

```python
from guandan import Environment

game = Environment(first_player=0)
for seat in range(4):
    game.add_player(f"player-{seat}", seat)

messages = game.start()
assert all(len(player.hand_cards) == 27 for player in game.players)
```

公开 API 还导出了 `Move` 和 `Moves`，用于表示出牌动作与生成合法动作。

## 使用 PPO 训练 V3

激活并验证 V3 环境后，提供你自己的 rollout 和输出路径：

```bash
python -m DanKS.training.train_ppo \
  --rollout /path/to/rollout.npz \
  --output /path/to/checkpoint.pt \
  --device auto
```

Learner 期望 rollout 中包含状态、候选动作、mask、历史、动作、行为策略对数概率、优势和回报数组。使用 `--help` 查看优化、评估、加速器和初始化选项。

V3 训练实现位于 [`versions/v3/DanKS/training`](versions/v3/DanKS/training)，包含：

- 模型与特征定义；
- PPO 目标与战术重采样；
- checkpoint 与优化器状态处理；
- 持久化 learner 传输；
- 召回和队伍信念辅助路径；
- 感知 CPU、CUDA 和 NPU 的加速器辅助工具。

## 项目边界

DanKS 不分发预训练 checkpoint，也不分发复现私有训练所需的数据。请使用自己的 rollout 数据，并将生成的 checkpoint 保存在仓库之外。该源码树面向代码阅读、改造与独立实验。

## 参与贡献

我们欢迎聚焦的缺陷修复、测试、可移植性改进以及边界清晰的算法变更。提交 Pull Request 前，请阅读[贡献指南](.github/CONTRIBUTING.md)。

## 开源许可

DanKS 基于 [Apache License 2.0](LICENSE) 开源。第三方依赖仍适用各自的许可证；详见 [NOTICE](NOTICE)。
