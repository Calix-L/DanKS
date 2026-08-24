<p align="right">
  <a href="README.md">English</a> | <strong>简体中文</strong>
</p>

# DanKS

[![CI](https://github.com/Calix-L/DanKS/actions/workflows/ci.yml/badge.svg)](https://github.com/Calix-L/DanKS/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/Calix-L/DanKS)](https://github.com/Calix-L/DanKS/releases/latest)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-D22128)](LICENSE)

**一个紧凑、代码优先的仓库，完整呈现三代掼蛋 AI 的技术演进。**

DanKS 保留了一个四人组队掼蛋智能体从结构化检索、学习型候选选择器，到基于 PPO 训练的记忆感知策略的演进过程。每一代都可独立安装，并共用 `DanKS` Python 命名空间；仓库同时提供一套共享的 108 张牌掼蛋规则引擎作为基础。

> 本仓库仅公开源代码。模型权重、数据集、评测记录、私有文档、凭据和部署自动化内容均不在公开范围内。

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

## 环境搭建

共享引擎和 V3 支持 Python 3.10 及更高版本；V1 和 V2 需要 Python 3.11 及更高版本。请为每一代使用独立虚拟环境，使依赖与 `DanKS` 命名空间保持隔离。

### 已验证配置

| 目标 | 系统 | Python | 框架 | 关键软件包 |
| --- | --- | --- | --- | --- |
| CI 与共享引擎 | Linux | 3.10, 3.12 | — | pytest 7+ |
| V1 | CPU | 3.11+ | NumPy 选择器 | NumPy 2.4.6 |
| V2 | CPU | 3.11+ | ONNX 选择器 | NumPy 2.4.6, ONNX Runtime 1.27.0 |
| V3 NVIDIA 服务器 | H100, driver 575.57.08 | 3.11.14 | PyTorch 2.8.0 + CUDA 12.8 | NumPy 2.4.6, pybind11 3.0.4 |
| V3 昇腾服务器 | Ubuntu 22.04.5, 910B2C, driver 24.1.0, CANN 8.5.0 | 3.10.12 | PyTorch 2.7.1 + torch_npu 2.7.1.post2 | NumPy 1.26.0, pybind11 3.0.4 |

两行加速器配置复现了项目已验证的服务器环境；它们是参考配置，而不是最低硬件要求。

### 1. 克隆仓库并创建基础环境

```bash
git clone https://github.com/Calix-L/DanKS.git
cd DanKS
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pytest -q
```

Windows PowerShell 请使用 `.venv\Scripts\Activate.ps1` 激活环境。

### 2. 选择 V1 或 V2

创建 Python 3.11 环境，并仅安装一个世代：

```bash
# V1
python3.11 -m venv .venv-v1
source .venv-v1/bin/activate
python -m pip install --upgrade pip
python -m pip install -e versions/v1
python -c "import DanKS; print(DanKS.__file__)"
```

V2 请使用另一个虚拟环境：

```bash
python3.11 -m venv .venv-v2
source .venv-v2/bin/activate
python -m pip install --upgrade pip
python -m pip install -e versions/v2
python -c "import DanKS; print(DanKS.__file__)"
```

不要在同一环境中安装多个世代；它们的特征 schema、导入命名空间和 checkpoint 是相互独立的。

### 3. 构建 V3 CPU 或 NVIDIA 环境

创建全新的 Python 3.11 环境并安装 V3：

```bash
python3.11 -m venv .venv-v3
source .venv-v3/bin/activate
python -m pip install --upgrade pip
python -m pip install -e versions/v3
```

只安装一种 PyTorch 构建。下列命令对应已验证的服务器版本；请根据自己的机器，通过 [PyTorch 官方安装矩阵](https://pytorch.org/get-started/previous-versions/) 选择正确的 wheel 索引。

```bash
# Linux/Windows CPU
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu

# NVIDIA CUDA 12.8
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# macOS CPU
python -m pip install torch==2.8.0
```

验证环境：

```bash
python -c "import numpy, torch; print('numpy', numpy.__version__); print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"
python -m DanKS.training.train_ppo --help
```

### 4. 构建 V3 昇腾 NPU 环境

昇腾运行时与主机驱动及 CANN 安装紧密耦合。请先安装匹配的 CANN 版本，再使用厂商提供的 PyTorch 和 `torch_npu` wheel。公开的 [`requirements-training-npu.txt`](versions/v3/DanKS/environment/requirements-training-npu.txt) 记录了已验证的版本组合。

```bash
source /usr/local/Ascend/cann/set_env.sh
python3.10 -m venv --system-site-packages .venv-v3-npu
source .venv-v3-npu/bin/activate
python -m pip install -e versions/v3

# 请将路径替换为与当前平台匹配的厂商 wheel。
python -m pip install --no-deps \
  /path/to/torch-2.7.1+cpu-cp310-cp310-manylinux_2_28_x86_64.whl \
  /path/to/torch_npu-2.7.1.post2-cp310-cp310-manylinux_2_28_x86_64.whl

export TORCH_DEVICE_BACKEND_AUTOLOAD=0
```

确认 PyTorch 能够识别加速器：

```bash
python -c "import torch, torch_npu; print('torch', torch.__version__); print('torch_npu', torch_npu.__version__); print('npu', torch.npu.is_available())"
python -m DanKS.training.train_ppo --help
```

不要在 NPU 环境中安装 CUDA 软件包。如果你的驱动、CANN、处理器架构或 Python 版本不同，请从昇腾软件发行渠道获取匹配的 wheel 组合，不要强行安装上述版本。

### 5. 构建可选的 V3 C++ 内核

优化后的检索内核目前支持 Linux 和 macOS，需要 C++17 编译器、Python 开发头文件和 `pybind11`；Windows 使用 Python 回退实现。请在目标机器上已激活的 V3 环境中构建：

```bash
# Ubuntu/Debian
sudo apt-get update && sudo apt-get install -y build-essential python3-dev

# macOS（仅需执行一次）
xcode-select --install
```

```bash
cd versions/v3/DanKS/retrieval/native_cpp
python setup.py build_ext --inplace
cd ../../../../../

python -c "from DanKS.retrieval.native_cover import available; from DanKS.retrieval.native_actor_core import available as actor_available; print('cover', available()); print('actor', actor_available())"
```

两个输出都应为 `True`。Linux 构建使用主机特定的编译优化；macOS 将架构选择交给 Python 工具链，因此仍然支持 universal2 构建。更换 Python 版本或 CPU 架构后，请重新编译扩展。

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
