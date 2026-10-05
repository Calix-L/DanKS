# KSPlay GuanDan Service · V2

**面向 AI 对战与二次开发的掼蛋网页平台。**

[English](README.md) · [快速开始](#快速开始) · [二次开发](docs/DEVELOPMENT.md) · [AI 接口](docs/AI_INTERFACE.md)

从牌桌交互到规则裁判，KSPlay GuanDan Service 提供一套可独立运行的掼蛋对战体验。
开箱即可与三名规则机器人对局，也可以通过标准 HTTP 接口接入自己的 AI。

## 特性

- **完整牌桌体验**：固定比例牌桌、桌面与手机横屏布局、选牌、出牌和结算。
- **灵活理牌**：炸弹优先、同花顺优先、整牌优先、简单理牌，支持多种拆分方案切换。
- **实时对战**：WebSocket 状态同步、断线重连、原座位恢复与 AI 请求重试。
- **服务端裁判**：合法动作、回合推进和结算统一由后端处理。
- **模块化架构**：界面、房间、规则、理牌与 AI 接口分层，方便独立修改和扩展。

## 快速开始

环境：Python 3.12、Go 1.23+。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/build_arranger.py
python run.py
```

打开 **http://127.0.0.1:8000/solo**，输入昵称、创建牌桌并点击准备。
你坐在 1 号位，其余三个座位由规则机器人接管。按 Ctrl+C 停止服务。

Windows 使用 `py -3.12 -m venv .venv` 创建环境，使用 `.venv\Scripts\activate` 激活。

## 接入 AI

通过 `DANKS_AI_ENDPOINT` 指定选牌服务：

```bash
DANKS_AI_ENDPOINT=http://127.0.0.1:9000/select python run.py
```

平台发送当前行动者的手牌、公开牌局信息和合法动作，AI 返回所选动作索引。
模型作为独立服务运行，网页与游戏后端通过统一协议调用。
详见 [AI 接口文档](docs/AI_INTERFACE.md)。

## 项目结构

```text
web/           牌桌页面、交互与视觉素材
backend/       房间、HTTP API 与 WebSocket
rules/         掼蛋规则与裁判
arranger/      Go 手牌拆分与理牌算法
workers/       每桌裁判进程与规则机器人
integrations/  外部 AI 连接器
scripts/       本地源码构建工具
docs/          开发指南与协议说明
```

## 二次开发

浏览器、API 和完整对局的检查步骤见二次开发指南。
修改 Go 源码后，使用 `python scripts/build_arranger.py --refresh-manifest` 更新指纹并重新构建。

| 文档 | 内容 |
| --- | --- |
| [二次开发指南](docs/DEVELOPMENT.md) | 功能入口、模块职责与开发流程 |
| [游戏 API](docs/GAME_API.md) | 创建牌桌、出牌、理牌与实时通信 |
| [AI 接口](docs/AI_INTERFACE.md) | 观察数据、动作选择与请求协议 |
| [版本说明](docs/VERSION.md) | 第二代功能范围与运行方式 |

## 许可证

[Apache-2.0](LICENSE)。商标及第三方声明见 [NOTICE](NOTICE)。
