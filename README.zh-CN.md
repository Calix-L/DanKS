<p align="right">
  <a href="README.md">English</a> | <strong>简体中文</strong>
</p>

<p align="center">
  <a href="https://www.kingsoft.com/">
    <img src="assets/kingsoft-logo.png" alt="Kingsoft AI Product Center" width="420">
  </a>
</p>

<h1 align="center">DanKS：SOTA 级掼蛋智能体</h1>

<p align="center">
  <strong>完整开放三代技术路线</strong><br>
  PPO 策略学习 · V3Pro 决策增强 · 两代 KSPlay GuanDan Service
</p>

<p align="center">
  <a href="https://github.com/Calix-L/DanKS/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/Calix-L/DanKS/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://github.com/Calix-L/DanKS/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/Calix-L/DanKS?style=flat"></a>
  <a href="LICENSE"><img alt="Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-D22128"></a>
</p>

<p align="center">
  <a href="https://calixlin.com/CardKS/"><strong>在线体验 ↗</strong></a> ·
  <a href="#danks-如何思考">技术架构</a> ·
  <a href="#ksplay-guandan-service">网页服务</a> ·
  <a href="#人类掼蛋数据">数据集</a> ·
  <a href="#快速开始">快速开始</a> ·
  <a href="https://github.com/Calix-L/CardKS">研究主页</a>
</p>

<p align="center"><strong>不只选好这一手，更为后续每一步布局。</strong></p>

掼蛋不只是比牌大小。你需要与队友配合，在看不见其他人手牌的情况下争夺牌权，并为后续出牌保留空间。现在最省的一张牌，可能拆掉最有价值的组合；一次不出，也可能把主动权交给队友。

**DanKS 让决策关注整副手牌的未来。** 由 **Kingsoft AI Product Center（金山 AI 产品中心）** 发起，DanKS 将结构化召回与 PPO 策略学习结合，并把两代开源网页对战服务带进同一个仓库。你可以研究智能体、训练自己的策略、打造自己的牌桌，也可以直接坐下来，和它打一局。

<p align="center">
  <a href="https://calixlin.com/CardKS/">
    <img src="assets/danks-promotional-hero-v2.png" alt="DanKS：从掼蛋 AI 研究到可直接体验的网页牌桌" width="1000">
  </a>
</p>

## 在线体验

**坐下，出牌，挑战 DanKS。无需安装。**

[打开在线牌桌 →](https://calixlin.com/CardKS/)

你与一位 AI 队友合作，迎战两位 AI 对手。浏览器即可游玩，支持中文与英文界面。

<p align="center"><strong>观看简短对局预览</strong></p>

<p align="center">
  <a href="https://calixlin.com/CardKS/">
    <img src="assets/danks-online-demo.gif" alt="掼蛋在线体验版的动态对局预览" width="760">
  </a><br>
  <sub><a href="assets/danks-online-demo.png">高清牌桌预览</a> · <a href="assets/danks-social-preview.png">社交分享图</a></sub>
</p>

## DanKS 如何思考

**保留有价值的选择，再学习什么时候使用它。**

![信息状态、结构化候选召回、Actor-Critic 选择与 PPO 自博弈流程](assets/danks-overall-architecture.png)

1. **理解当前局面。** 编码自己的手牌、公开出牌历史、合法动作和队伍上下文。
2. **看到出牌之后。** 对候选动作分析剩余手牌，判断它保留了哪些组合与后续机会。
3. **从少而精的候选中决策。** Actor-Critic 联合评估状态、候选动作与剩牌结构。
4. **从后续结果中学习。** PPO 与 GAE 将当前选择和之后的得失联系起来。

召回负责找到有价值的选项，策略网络负责判断哪个选项适合当下。

<p align="center">
  <a href="assets/structure-aware-delayed-outcomes.png">
    <img src="assets/structure-aware-delayed-outcomes.png" alt="出小牌、出王与不出会保留不同的后续出牌空间" width="300">
  </a><br>
  <sub>同一副牌，三个选择，不同的未来。点击查看完整决策示例。</sub>
</p>

## 三代演进，一条主线

| AI 版本 | 核心能力 | 阅读入口 |
| --- | --- | --- |
| **V1** | 结构化召回与 NumPy 候选选择器 | [召回排序](versions/v1/DanKS/retrieval/ranker.py) |
| **V2** | 扩展候选生成与 ONNX 选择器 | [动作生成](versions/v2/DanKS/retrieval/action_generator.py) |
| **V3** | 记忆感知神经策略、队伍信念特征与 PPO 学习 | [策略网络](versions/v3/DanKS/training/model.py) · [PPO 训练](versions/v3/DanKS/training) |
| **V3Pro** | V3 推理增强：配牌保护、等价出牌规则与验证式残局搜索 | [策略入口](versions/v3pro/DanKSPro/policy.py) · [接入指南](versions/v3pro/USAGE.zh-CN.md) |

V3Pro 以独立的 `DanKSPro` 包扩展 V3，不替换原网络，也不要求重新训练。残局增强覆盖通过准入检查、全桌总剩牌不超过 16 张的局面；11–16 张时还要求暗牌分配数不超过 128。

## KSPlay GuanDan Service

**有了 AI，还要有一张好用的牌桌。现在，牌桌也开源了。**

两代 Service 均包含浏览器前端、房间后端、掼蛋裁判、完整源码理牌模块，以及标准外部 AI 接口。

- **Service V1 —— 经典牌桌。** 保留原版 CardKS 体验，提供简洁的开发起点。
- **Service V2 —— 全新牌桌。** 固定比例的桌面与手机横屏布局、模块化交互，以及改进的会话恢复。

[查看 Service V1 →](services/v1/README.zh-CN.md) · [查看 Service V2 →](services/v2/README.zh-CN.md)

Service 的 V1、V2 指**网页平台版本**，与 AI 网络代数独立。两代服务均可通过示例规则机器人本地运行，也可以按 [HTTP AI 接口](services/v2/docs/AI_INTERFACE.md)接入自己的模型。训练权重与私有 AI 部署环境不随仓库分发。

### 从这里开始二开

| 你想做什么 | 从哪里开始 |
| --- | --- |
| 改牌桌布局或选牌交互 | [V2 前端](services/v2/web) |
| 扩展房间、牌局流程或实时消息 | [V2 后端](services/v2/backend) |
| 调整理牌算法 | [Go 理牌源码](services/v2/arranger) |
| 接入新 AI | [AI 请求与返回协议](services/v2/docs/AI_INTERFACE.md) |
| 找到对应的修改模块 | [Service 二开指南](services/v2/docs/DEVELOPMENT.md) |

## 人类掼蛋数据

**从完整牌局中研究决策，而不只是孤立的一手牌。**

CardKS 维护的公开 [KSCB 掼蛋数据集](https://github.com/Calix-L/CardKS/blob/main/KSCB/data/guandan_matches.jsonl.gz)包含 **899 场完整升级赛、10,218 个小局、840,194 个决策点**。每个小局保留按顺序排列的事件，便于研究人类决策、队友配合与手牌结构的变化。

[查看数据入口 →](datasets/README.zh-CN.md) · [原始格式说明 →](https://github.com/Calix-L/CardKS/blob/main/KSCB/README.zh-CN.md)

`datasets/` 提供原始数据链接、下载方法与读取示例，数据继续由 CardKS 维护，不在 DanKS 重复存放。这些是牌局记录，而非预先编码好的 PPO 输入。

## 快速开始

### 跑起自己的网页牌桌

准备 **Python 3.12 与 Go 1.23+**。以下以 POSIX shell 为例：

```bash
git clone https://github.com/Calix-L/DanKS.git
cd DanKS/services/v2
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/build_arranger.py
python run.py
```

打开 **http://127.0.0.1:8000/solo**，创建牌桌并点击准备。其余座位由三个示例机器人填充；按 Ctrl+C 停止服务。

Windows PowerShell 使用 `py -3.12 -m venv .venv` 创建环境、`.venv\Scripts\Activate.ps1` 激活。要体验经典牌桌，将目录换成 `services/v1` 即可。

### 跑通 AI 代码

在独立的 **Python 3.11+** 环境中，从仓库根目录运行：

```bash
python3.11 -m venv .venv-ai
source .venv-ai/bin/activate
python -m pip install -e . -e versions/v3 -e versions/v3pro
python -m pip install torch==2.8.0
python examples/retrieval_quickstart.py --version v3
python examples/v3_model_smoke.py
python examples/v3pro_smoke.py
```

示例使用合成输入验证召回、网络推理与 V3Pro 接入，模型冒烟示例使用随机初始化权重。CPU/CUDA/NPU 环境、V1/V2 安装和 PPO learner 命令见[开发指南](.github/guides/DEVELOPMENT.zh-CN.md)。Linux/macOS 可在 V3 环境运行 `danks-build-native`，启用可选 C++ 召回加速。

## 仓库结构

```text
DanKS/
├── versions/           # AI：V1、V2、V3 与 V3Pro 扩展
├── services/
│   ├── v1/             # KSPlay GuanDan Service · 经典牌桌
│   └── v2/             # KSPlay GuanDan Service · 全新牌桌
├── guandan/engine/     # AI 侧共享掼蛋规则引擎
├── examples/           # 引擎、召回、网络与 PPO 可运行示例
├── datasets/           # 公开掼蛋数据入口与读取指南
├── assets/             # 品牌、对局预览与架构插图
└── .github/            # 贡献指南、开发指南与 CI
```

两代 Service 均可独立运行，各自保留规则与理牌模块。AI 各代仍为独立安装包，请为不同代数分别创建环境。

## 一起把它做得更好

训练一个新智能体，打造一张更好的牌桌，探索组队博弈的下一个想法。

欢迎算法、UI、跨平台适配与文档方面的贡献。参与方式见[贡献指南](.github/CONTRIBUTING.md)，也可以直接[提交 Issue](https://github.com/Calix-L/DanKS/issues)。

**项目仓库：** [GitHub](https://github.com/Calix-L/DanKS) · [AtomGit 国内镜像](https://atomgit.com/Calix_Lin/DanKS)<br>
**研究主页：** [CardKS](https://github.com/Calix-L/CardKS)<br>
**开源许可：** [Apache-2.0](LICENSE) · [许可与商标说明](NOTICE)

### Star 增长趋势

<p align="center">
  <a href="https://www.star-history.com/#Calix-L/DanKS&Date">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=Calix-L/DanKS&type=Date&theme=dark">
      <img alt="DanKS Star 增长趋势" src="https://api.star-history.com/svg?repos=Calix-L/DanKS&type=Date" width="560">
    </picture>
  </a>
</p>
