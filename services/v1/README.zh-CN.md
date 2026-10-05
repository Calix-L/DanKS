# KSPlay GuanDan Service · V1

**经典牌桌界面的掼蛋网页对战平台。**

[English](README.md) · [快速开始](#快速开始) · [二次开发](docs/DEVELOPMENT.md) · [AI 接口](docs/AI_INTERFACE.md)

KSPlay GuanDan Service 是网页服务第一代，提供一名玩家与三名机器人对战的掼蛋体验。
项目包含牌桌前端、房间后端、规则裁判和完整理牌算法，可独立运行，也可接入自己的 AI。

## 特性

- **经典牌桌**：保留第一代 CardKS 的界面、选牌与操作体验。
- **完整理牌**：炸弹优先、同花顺优先、整牌优先、简单理牌及多方案切换。
- **实时对局**：WebSocket 状态同步、原牌桌与座位恢复。
- **服务端裁判**：统一处理合法动作、回合推进和结算。
- **独立 AI 接口**：通过标准 HTTP 协议连接自己的选牌服务。

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
默认使用三名规则机器人，按 Ctrl+C 停止服务。

Windows 使用 `py -3.12 -m venv .venv` 创建环境，使用 `.venv\Scripts\activate` 激活。

## 接入 AI

```bash
DANKS_AI_ENDPOINT=http://127.0.0.1:9000/select python run.py
```

平台发送当前行动者的手牌、公开信息和合法动作，AI 返回动作索引。
模型独立于游戏服务运行，详见 [AI 接口文档](docs/AI_INTERFACE.md)。

## 项目结构

```text
web/           第一代牌桌页面与交互
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

[二次开发指南](docs/DEVELOPMENT.md) · [游戏 API](docs/GAME_API.md) · [版本说明](docs/VERSION.md)

## 许可证

[Apache-2.0](LICENSE)。商标及第三方声明见 [NOTICE](NOTICE)。
