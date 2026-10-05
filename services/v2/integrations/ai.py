"""Generic action-selection HTTP interface; contains no model implementation."""
from __future__ import annotations

import copy
import uuid
from urllib.parse import urlsplit

import httpx


class AIError(RuntimeError):
    pass


class AIClient:
    def __init__(self, endpoint, *, timeout=10.0, transport=None):
        parsed = urlsplit(endpoint)
        if (parsed.scheme not in {'http', 'https'} or not parsed.hostname
                or parsed.username or parsed.password or parsed.fragment):
            raise ValueError('AI endpoint must be HTTP(S), without credentials or fragment')
        if timeout <= 0:
            raise ValueError('AI timeout must be positive')
        self.endpoint = endpoint
        self.client = httpx.Client(timeout=timeout, transport=transport, trust_env=False,
                                   follow_redirects=False)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.client.close()

    def select(self, view):
        request_id = uuid.uuid4().hex
        payload = {'schema_version': 'danks.ai.v1', 'request_id': request_id,
                   'observation': {key: copy.deepcopy(view[key]) for key in
                     ('seat', 'level', 'own_hand', 'hand_counts', 'is_lead', 'public_history')},
                   'legal_actions': copy.deepcopy(view['legal_actions'])}
        try:
            response = self.client.post(self.endpoint, json=payload)
            response.raise_for_status()
            result = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise AIError('external AI request failed; no fallback was applied') from exc
        if not isinstance(result, dict) or result.get('request_id') != request_id:
            raise AIError('AI response request_id does not match')
        index = result.get('action_index')
        if type(index) is not int or index not in {a['action_index'] for a in view['legal_actions']}:
            raise AIError('AI response selected an illegal action index')
        return index
