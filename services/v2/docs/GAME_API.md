# Browser ↔ game-service API

Routes below are relative to the game-service origin. This is not the AI interface;
the latter is specified in [AI_INTERFACE](AI_INTERFACE.md).

| Method / path | Purpose | Authentication |
| --- | --- | --- |
| `GET /solo` | Table webpage | None |
| `GET /api/health` | Service readiness | None |
| `POST /api/human-tables` | Create one-human GuanDan room | Returns session token |
| `POST /api/human-tables/{table}/join` | Resume existing human session | Original token or private stable user ID |
| `GET /api/human-tables/{table}/view` | Current actor-specific view | Bearer token |
| `POST /api/human-tables/{table}/ready` | Set readiness/start next hand | Bearer token |
| `POST /api/human-tables/{table}/actions` | Commit legal action | Bearer token |
| `POST /api/human-tables/{table}/arrange` | Change presentation mode | Bearer token |
| `POST /api/human-tables/{table}/retry` | Retry a paused external AI selection | Bearer token |
| `WS /api/human-tables/{table}/ws` | State updates | Initial auth frame |

Create request:

```json
{"game":"guandan","nickname":"Player","user_id":"a4296e08f1854b168f43b7014a0f5cfa"}
```

`user_id` is 16–128 characters using letters, digits, underscore or hyphen.
Generate an unpredictable value such as `crypto.randomUUID()` (browser) or
`uuid.uuid4().hex` (Python). **Treat it as a private recovery credential**, not a
public account name: the current join flow can rotate a token when the caller
supplies the original stable user ID without `resume_token`. Do not use nickname,
phone number or a predictable shared string. The response contains
`table_no`, `token`, `seat`, `role` and `view`. Keep the token private. HTTP player
requests use `Authorization: Bearer <token>`.

Ready request:

```json
{"ready":true,"expected_version":1}
```

Action request (`expected_version`, `decision_id` and `action_index` must come
from the **latest view**; generate `client_action_id` once for the submission):

```json
{"expected_version":1,"decision_id":"current-decision-id","action_index":2,"client_action_id":"unique-client-action-id"}
```

`client_action_id` is a unique idempotency identifier (1–256 characters); retain
it when retrying the same action. The browser must submit an entry in
`view.decision.actions`. The service rejects
stale versions, wrong actors and illegal indices. Refresh the view rather than
guessing a new action index. The browser uses physical-card selection only to
choose among this current legal list; it does not define legal moves.

Arrange request is `{"mode":1}` with mode 1..4: bomb priority, straight-flush
priority, made-hand priority and simple sorting. The full Go hand splitter is
included as source; repeated arrangement requests can cycle through alternative
decompositions. This is presentation logic, not AI inference.

Prefer resuming with the original token:

```json
{"nickname":"Player","user_id":"a4296e08f1854b168f43b7014a0f5cfa","resume_token":"original-session-token"}
```

WebSocket authentication must be the first frame, within five seconds:

```json
{"type":"auth","token":"session-token"}
```

The server sends `{"type":"state","state":{...}}` updates and ping frames.
See `web/frontend/room-api.mjs` for the client implementation. Do not put the
token in URL query strings. Player views contain only the player's own hand,
seat counts, public plays, legal actions when it is their turn and series state.
Sessions and tables are in memory; a service restart invalidates live rooms.

Recovering with the stable user ID rotates the token and revokes **all** existing
WebSocket subscriptions for that participant, including queued private views.
Old HTTP requests return 401; old sockets close with 4401. A missing table closes
with 4404. The browser stops reconnecting on either code and returns to entry:
4404 forgets the missing table, while 4401 keeps a deliberate Resume game action.
It never rotates credentials automatically in a reconnect loop. Ordinary network
disconnects back off from 750 ms to at most 10 seconds; a confirmed state resets
the delay, not a socket opening before authentication.

## Paused AI and retry

Healthy views have `failure: null`. An external selection error pauses the room
with `phase: "paused"` and `failure: {"code":"ai_unavailable","retryable":true}`.
This includes timeouts, HTTP errors and invalid AI responses. No action has been
committed for that failed selection; the current referee state/decision is kept.
The browser displays a persistent message and explicit Retry AI / Back to entry
controls. There is no automatic retry or bot substitution.

Retry body uses the latest **room** version:

```json
{"expected_version":4}
```

Only the authenticated participant can retry a paused selection fault. A stale
version or a non-retryable room returns 409. An accepted retry increments the
room version, clears the failure and schedules selection for the same referee
decision. If the AI is still unavailable, the room pauses again. If the request's
outcome is uncertain, refresh the view instead of blindly repeating it.

Unknown worker/referee failures use `phase: "error"` and
`failure: {"code":"engine_failure","retryable":false}`. They cannot be retried;
return to entry and create a new table. Browser messages contain no upstream
endpoint, traceback or hidden hands. A new table is a new game, not recovery.

Human/API failures generally use `{"detail":{"code":"...","message":"..."}}`;
schema validation failures use the standard FastAPI validation format. HTTP 401
means missing/invalid authentication, 404 a missing room, 409 a state conflict,
422 invalid input, and 502 a rules-worker failure. Never infer card ownership
from the HTTP response alone: render the returned/current authoritative view.
