"""Referee-to-public-observation bridge, with no model implementation."""
import copy
import os

from integrations.ai import AIClient


class HTTPPolicy:
    def __init__(self, endpoint):
        self.client = AIClient(endpoint, timeout=float(os.getenv('DANKS_AI_TIMEOUT_SECONDS', '10')))
        self.public_history = []

    def reset_table_history(self):
        self.public_history.clear()

    def record_action(self, seat, action):
        from .table_worker import _action_view
        normalized = _action_view(action, 0)
        self.public_history.append({'seat': seat, 'action': [normalized['type'],
            normalized['rank'], copy.deepcopy(normalized['cards'])]})

    def consume_messages(self, messages):
        # Raw referee messages can contain private dealt hands: never forward them.
        return 0

    def choose_action_from_environment(self, environment, *, seat):
        from .table_worker import _action_view, _valid_indices
        view = {'seat': seat, 'level': str(environment.rank),
                'own_hand': [str(card) for card in environment.players[seat].hand_cards],
                'hand_counts': [len(player.hand_cards) for player in environment.players],
                'is_lead': bool(environment.action_first),
                'public_history': self.public_history,
                'legal_actions': [_action_view(environment.legal_moves.action_list[i], i)
                                  for i in _valid_indices(environment)]}
        return self.client.select(view)

    def metadata(self):
        return {'id': 'external_http', 'ready': True}
