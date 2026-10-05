# KSPlay GuanDan Service · V1

**A GuanDan web platform with the original CardKS table experience.**

[简体中文](README.zh-CN.md) · [Quick start](#quick-start) · [Development](docs/DEVELOPMENT.md) · [AI interface](docs/AI_INTERFACE.md)

KSPlay GuanDan Service is the first generation of the web game service. It combines the
table frontend, rooms, game referee and full hand arranger for one human and
three bots. Run it locally or connect your own AI through HTTP.

## Features

- **Classic table** — the original CardKS interface and card interactions.
- **Full hand arrangement** — bombs-first, straight-flush-first, made-hand and simple modes, with alternative layouts.
- **Realtime games** — WebSocket state updates and table/seat recovery.
- **Authoritative referee** — legal actions, turns and scoring handled server-side.
- **Independent AI integration** — a standard HTTP action-selection contract.

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
Ready. Three rule-based bots fill the other seats. Stop with Ctrl+C.

On Windows, create the environment with `py -3.12 -m venv .venv` and activate it
with `.venv\Scripts\activate`.

## Connect your AI

```bash
DANKS_AI_ENDPOINT=http://127.0.0.1:9000/select python run.py
```

The platform sends the acting player's hand, public information and legal actions;
your AI returns an action index. Models run independently of the game service.
See the [AI interface](docs/AI_INTERFACE.md).

## Project structure

```text
web/           Original table UI and interactions
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

[Development guide](docs/DEVELOPMENT.md) · [Game API](docs/GAME_API.md) · [Version](docs/VERSION.md)

## License

[Apache-2.0](LICENSE). See [NOTICE](NOTICE) for trademark and third-party notices.
