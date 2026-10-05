# DanKS 开发指南

[← 返回 DanKS](../../README.zh-CN.md)

## 安装参考

共享引擎和 V3 支持 Python 3.10 及更高版本；V1 和 V2 支持 Python 3.11 及更高版本。以下命令均从仓库根目录运行。推荐为每个版本创建独立虚拟环境，使 `DanKS` 命名空间与对应的特征、模型格式自然对齐。

### 选择安装包

| 目标 | 安装命令 | 说明 |
| --- | --- | --- |
| 共享规则引擎与测试 | `python -m pip install -e '.[dev]'` | 规则引擎与仓库测试套件。 |
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

昇腾运行时与主机驱动及 CANN 版本配套使用。安装匹配的 CANN 后，再安装厂商提供的 PyTorch 与 `torch_npu` wheel。已验证组合记录在 [`requirements-training-npu.txt`](../../versions/v3/DanKS/environment/requirements-training-npu.txt)。

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

建议将该虚拟环境专用于昇腾 NPU。其他驱动、CANN、处理器架构或 Python 版本可选用对应的厂商 wheel。

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

这些是经过验证的参考配置，其他兼容环境也可以运行 DanKS。

</details>

### 可选的 V3 C++ 加速

优化后的检索内核支持 Linux 和 macOS，需要 C++17 编译器、Python 开发头文件和 `pybind11`；Windows 自动使用 Python 实现。首先安装平台工具链：

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

该命令会自动定位已安装的 V3 源码，成功后输出 `cover=True, actor=True`。Linux 构建启用主机编译优化；macOS 由 Python 工具链选择架构并支持 universal2 构建。更换 Python 版本或 CPU 架构后重新运行即可。Windows 会自动选择 Python 实现。

### 开发检查

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

## 运行示例

示例覆盖规则引擎、结构化检索、完整网络前向传播和 PPO 更新，可直接从源码运行：

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

每条命令都包含自检断言，便于快速确认当前环境和代码路径运行正常。

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

激活并验证 V3 环境后，指定 rollout 与 checkpoint 输出路径：

```bash
python -m DanKS.training.train_ppo \
  --rollout /path/to/rollout.npz \
  --output /path/to/checkpoint.pt \
  --device auto
```

Learner 期望 rollout 中包含状态、候选动作、mask、历史、动作、行为策略对数概率、优势和回报数组。使用 `--help` 查看优化、评估、加速器和初始化选项。

V3 训练实现位于 [`versions/v3/DanKS/training`](../../versions/v3/DanKS/training)，包含：

- 模型与特征定义；
- PPO 目标与战术重采样；
- checkpoint 与优化器状态处理；
- 持久化 learner 传输；
- 召回和队伍信念辅助路径；
- 感知 CPU、CUDA 和 NPU 的加速器辅助工具。
