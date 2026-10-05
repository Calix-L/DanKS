# External AI interface: danks.ai.v1

The backend sends an HTTP POST to the endpoint configured by the operator.
This document specifies only the integration contract, not how to deploy an AI.

```json
{
  "schema_version": "danks.ai.v1",
  "request_id": "unique-request-id",
  "observation": {
    "seat": 1,
    "level": "7",
    "own_hand": ["S9", "HR"],
    "hand_counts": [8, 2, 11, 5],
    "is_lead": false,
    "public_history": [{"seat": 0, "action": ["Single", "8", ["S8"]]}]
  },
  "legal_actions": [
    {"action_index": 2, "type": "Single", "rank": "9", "cards": ["S9"], "label": "Single"}
  ]
}
```

Return status 200 and JSON:

```json
{"request_id": "unique-request-id", "action_index": 2}
```

Seats are zero-based (0..3), with teammate `(seat + 2) % 4`. Cards use suits
S/H/C/D and ranks 2..9/T/J/Q/K/A; SB is the small joker and HR the big joker.
Pass uses `type: "PASS"` and an empty cards list; there is no separate `pass`
boolean. History contains only committed actions of the
current hand, including passes, in chronological order.

`action_index` refers to the current legal list, not a reusable action ID.
Return an integer, not a string or boolean. The client rejects stale request IDs,
invalid indices, redirects, malformed JSON, HTTP errors and timeouts. It does not
silently substitute another bot. Setup/tribute phases use the example rule path;
the current solo adapter starts directly in the play phase.

Selection failures happen before a referee action is committed and propagate
into the generation-one legacy room error flow. This version has no explicit
selection retry route. Restore the endpoint and create a new table; the endpoint
must select an action, not mutate a game or initiate side effects. There is no
automatic retry or fallback. See [GAME_API](GAME_API.md).

No opponent hands, raw referee messages, model credentials, feature tensors or
deployment paths are included. The URL is configured only on the game backend;
it is not returned to the browser. Use HTTPS when crossing a trusted local
boundary. This basic candidate defines no AI authentication protocol; a private
connector may add authentication outside this published contract.
