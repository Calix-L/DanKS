# Generation-one browser ↔ game-service API

This is the old generation-one contract, not the external AI contract.
All paths are relative to the service origin.

| Method / path | Purpose |
| --- | --- |
| `GET /solo` | Old table webpage |
| `GET /api/health` | Readiness and public identity |
| `POST /api/human-tables` | Create one-human GuanDan table |
| `POST /api/human-tables/{table}/join` | Resume existing session |
| `GET /api/human-tables/{table}/view` | Authenticated private view |
| `POST /api/human-tables/{table}/ready` | Set readiness/start next hand |
| `POST /api/human-tables/{table}/actions` | Submit current legal action index |
| `POST /api/human-tables/{table}/arrange` | Choose layout mode 1..4 |
| `WS /api/human-tables/{table}/ws` | Authenticated state updates |

Create body:

```json
{"game":"guandan","nickname":"Player","user_id":"a4296e08f1854b168f43b7014a0f5cfa"}
```

`user_id` uses 16–128 letters/digits/underscore/hyphen. Generate an unpredictable
private value such as `crypto.randomUUID()` or `uuid.uuid4().hex`; never use a
nickname or public account name. The legacy join flow may rotate a session token
using this stable private ID without the previous token. It is a recovery
credential, not public profile information. The response contains `table_no`,
`token`, `seat`, `role`, `view`. Protect both ID and bearer token.

HTTP player requests use `Authorization: Bearer <token>`. Ready body is
`{"ready":true,"expected_version":1}`. Action fields must come from the latest
view, not this illustrative sample:

```json
{"expected_version":1,"decision_id":"current-decision-id","action_index":2,"client_action_id":"unique-client-action-id"}
```

Keep `client_action_id` when repeating the same uncertain submission. Stale
versions, wrong actors, illegal indices and conflicting reused IDs are rejected.
Generation one rejects the `selected_cards` substitution field. Browser card
selection only finds a matching action from the current legal list.

Arrange body is `{"mode":1}` (1..4). Prefer resuming with the original token:

```json
{"nickname":"Player","user_id":"a4296e08f1854b168f43b7014a0f5cfa","resume_token":"original-session-token"}
```

The first WebSocket frame, within five seconds, is
`{"type":"auth","token":"session-token"}`. Server updates are
`{"type":"state","state":{...}}`, with ping frames. Do not put tokens in URLs.
Views contain the viewer's hand and public information, not opponent hands.
Tables are in memory; a restart invalidates them.

## Legacy error and recovery limits

AI errors propagate without fallback and stop the room using the old `error`
flow. There is no dedicated paused-AI state or explicit selection retry route.
Restore your endpoint and create a new table. A new table is a new game, not
recovery. The old browser/reconnect behavior is intentionally unchanged; do not
assume later socket-revocation and recovery fixes have been backported.

Human/API failures normally use `{"detail":{"code":"...","message":"..."}}`.
401 means missing/invalid token, 404 missing room, 409 state conflict, 422 invalid
input and 502 worker failure. Always refresh/render the authoritative view.
