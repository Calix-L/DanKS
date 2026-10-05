# Secondary development / 二次开发

Start with the README local run commands, including the Go 1.23+ source
build via `python scripts/build_arranger.py`. Its local `runtime/` output is not a
release artifact. `BATTLE_ARRANGER_PATH` can point to a binary built outside the
source tree for testing. No original private repository,
server account or deployment directory is needed.

| Directory | Responsibility |
| --- | --- |
| `web/` | Preserved old table UI and required ES modules only |
| `backend/solo_app.py` | HTTP, asset allowlist, authenticated WebSocket routes |
| `backend/human_tables.py` | In-memory rooms, token/view isolation, versions, lifecycle and audit |
| `backend/human_table_models.py`, `models.py` | API and shared wire DTOs |
| `backend/presentation.py` and arranger source | Full source-based presentation splitter, no legal-move authority |
| `arranger/`, `scripts/build_arranger.py` | Complete portable Go module and source-build helper |
| `workers/table_worker.py` | One persistent NDJSON referee process per table |
| `rules/engine/` | Pure Python GuanDan rules, deck and turn state |
| `integrations/ai.py` | Independent validated HTTP contract |
| `workers/http_policy.py` | Referee-to-public-observation bridge |

## Common extensions

For another AI, implement the response in [AI_INTERFACE](AI_INTERFACE.md). Keep
model loading, weights, credentials and feature construction in your separate
service. Never forward other hands or raw deal messages. Test malformed response,
timeout, request-ID mismatch and illegal index cases before connecting it.

For another hand layout, implement `health`, `close` and `arrange` compatible with
`HandArranger`. Preserve the exact card multiset, including duplicate cards and
wild hearts, and return groups with `pattern`, `value`, `minor`, `cards`. Presentation
must never invent an action or change referee cards. Preserve the full source-based
splitter when modifying its portable invocation and layout contracts.

After reviewing edits to Go source, explicitly record their new fingerprint and
rebuild with `python scripts/build_arranger.py --refresh-manifest`. An ordinary
build validates the existing manifest rather than silently accepting changes.
`source_fingerprint` identifies the current public source (normalized LF);
`algorithm_source_sha` records historical algorithm lineage, not current bytes.
After rebuilding, repeat the four-mode arrangement and complete-game checks below.
Developer regression tests are maintained separately and are not bundled here.

For UI work, edit `web/play.js` and its modules, keeping the browser module-serving
allowlist in `solo_app.py` in sync. Such changes intentionally end byte identity:
update version/provenance notes and your separately maintained regression checks. This generation deliberately does not
backport the later UI's reconnect, pause/retry or selection changes.

For game rules, first record a reproducible failing case in your own development
harness. Use deterministic seeds, complete games, 108-card conservation,
level/joker/wildcard rules and invalid action checks. Do not alter natural
`A > ... > 3 > 2` to match historical display
utilities. Any new game is a separately reviewed extension; shared DTO values
do not mean a second game is runnable in this package.

## Local options and privacy

`python run.py --host 127.0.0.1 --port 8000` binds locally by default.
`DANKS_AI_ENDPOINT` enables external selection; `DANKS_AI_TIMEOUT_SECONDS` defaults
to 10. `BATTLE_SOLO_BOT_DELAY_MS` defaults to 2000. `BATTLE_BASE_PATH` supports a
URL prefix whose reverse proxy strips the prefix before forwarding.
`BATTLE_ENABLE_DOCS=1` enables local OpenAPI docs. `BATTLE_SOLO_LOG_ROOT` changes
the local audit directory. Do not expose internal details in public deployment.
Environment names are inherited compatibility interfaces, not deployment files.

For local validation, set a separate audit root with `BATTLE_SOLO_LOG_ROOT` and
use `PYTHONDONTWRITEBYTECODE=1` to avoid source-tree bytecode artifacts. Check that
`data/`, `runtime/`, `.venv/`, caches and logs are excluded when packaging.
The preserved stable-user-ID recovery is a sensitive legacy capability; use private
unpredictable IDs. Existing sockets are not promised to be hardened by later fixes.
There is no AI retry route here: fix the endpoint then create a new table.

## Manual validation checklist

This public package omits test source, not runtime modules. These checks are
actionable acceptance checks, not a replacement for your own automated regressions.

1. Build and start using the README, then request
   `curl -fsS http://127.0.0.1:8000/api/health`. Open `/solo` through the service,
   create a room, ready, select cards, play and pass. Inspect browser Console and
   Network for missing modules, failed requests and WebSocket state updates.
2. Exercise all four arrangement modes and alternative partitions. Compare
   physical card IDs and counts before/after: no duplicate card may disappear,
   no wildcard may change identity, and arrangement must not advance the referee.
   Include all 13 levels, 10/T/t aliases, duplicate ranks, wild hearts, jokers,
   empty hands and hands where bomb/straight-flush priorities conflict.
3. Using the request bodies in [GAME_API](GAME_API.md), inspect authenticated
   views and try missing/invalid tokens, stale versions, wrong decision IDs and
   out-of-range action indices. Confirm rejection without card or turn mutation.
   Inspect HTTP and WebSocket payloads: only the participant's own hand is private;
   other seats expose counts and public plays, never their hidden hands.
4. Finish several complete games with the example bots, including play/pass,
   trick clearing, finish order, settlement and the next hand. For rule edits,
   record deterministic seed/deal and inspect worker state in your local harness:
   hands plus played cards must conserve all 108 physical cards at every step.
   Check natural A-over-2, equal-rank rejection, current-level promotion, jokers,
   wildcard combinations and legal-action membership; visual play alone cannot
   establish these invariants.
5. With a separate local HTTP AI stub implementing [AI_INTERFACE](AI_INTERFACE.md),
   inspect actor-visible requests, then force malformed responses, request-ID
   mismatch, illegal indices and timeouts. Confirm a visible failure and no
   silent fallback or action commit. V1 has no retry route: repair the endpoint
   and create a new table. Check refresh/resume without assuming V2 hardening.

中文：公开包不附测试源码；按上述步骤检查网页、权限隔离、四种理牌与完整结算。
规则修改仍须在自己的验证工具中保留固定种子、108 张实体牌守恒与非法动作回归。

中文：界面、房间服务、裁判与 AI 连接器分层；二次开发先写失败用例，保护牌数守恒与私有视图。
模型与特征放在独立服务。替换展示算法仅改变布局，不能改变牌或决定合法动作。
修改旧版网页会结束字节一致性，须更新版本说明；勿宣称已移植新版重连与恢复改进。
