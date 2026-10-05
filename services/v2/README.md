# KSPlay GuanDan Service · V2

**A GuanDan web platform for AI play and experimentation.**

[简体中文](README.zh-CN.md) · [Quick start](#quick-start) · [Development](docs/DEVELOPMENT.md) · [AI interface](docs/AI_INTERFACE.md)

KSPlay GuanDan Service brings the table UI, game referee and room service together in one
modular project. Play against three rule-based bots out of the box, or connect
your own AI through a standard HTTP interface.

## Features

- **Table experience** — fixed-aspect desktop and mobile-landscape layouts, card selection, play controls and results.
- **Flexible hand arrangement** — bombs-first, straight-flush-first, made-hand and simple modes, with alternative partitions.
- **Realtime play** — WebSocket updates, reconnects, seat recovery and external-AI retries.
- **Authoritative rules** — legal actions, turns and scoring handled by the game service.
- **Modular design** — separate UI, rooms, rules, hand arrangement and AI integration.

## Quick start

Requirements: Python 3.12 and Go 1.23+.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/build_arranger.py
python run.py
```

Open **http://127.0.0.1:8000/solo**, enter a nickname, create a table and click
Ready. You take seat 1; three rule-based bots fill the remaining seats.
Stop the service with Ctrl+C.

On Windows, create the environment with `py -3.12 -m venv .venv` and activate it
with `.venv\Scripts\activate`.

## Connect your AI

Set the action-selection endpoint:

```bash
DANKS_AI_ENDPOINT=http://127.0.0.1:9000/select python run.py
```

The service sends the acting player's hand, public game state and legal actions.
Your AI returns the selected action index. Models run as independent services,
keeping the game platform and model runtime decoupled.
See the [AI interface](docs/AI_INTERFACE.md) for the request and response contract.

## Project structure

```text
web/           Table UI, interactions and artwork
backend/       Rooms, HTTP API and WebSocket
rules/         GuanDan rules and referee
arranger/      Go hand splitter and layout algorithms
workers/       Per-table referee process and rule-based bots
integrations/  External AI connector
scripts/       Local source-build tools
docs/          Development guides and protocols
```

## Development

See the development guide for browser, API and full-game checks.
After editing Go source, update its fingerprint and rebuild with
`python scripts/build_arranger.py --refresh-manifest`.

| Guide | Contents |
| --- | --- |
| [Development](docs/DEVELOPMENT.md) | Entry points, module responsibilities and workflow |
| [Game API](docs/GAME_API.md) | Tables, actions, arrangement and realtime messages |
| [AI interface](docs/AI_INTERFACE.md) | Observations, action selection and protocol |
| [Version](docs/VERSION.md) | Generation-two scope and runtime behavior |

## License

[Apache-2.0](LICENSE). See [NOTICE](NOTICE) for trademark and third-party notices.
