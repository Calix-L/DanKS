#!/usr/bin/env python3
"""Run all three V3Pro components on a synthetic state, without private weights.

The hand/history fixture tests interfaces; it is not an actual match replay or
playing-strength evaluation. A deployment reconstructs this state from its own
complete public round history instead of constructing a synthetic fixture.
"""
from collections import Counter
import random

import torch

from guandan.engine import Environment
from guandan.engine.types import Phase
from DanKS.training.model import EfficientTeamBeliefTop10Selector
from DanKS.training.schema import STATE_DIM, CANDIDATE_DIM
from DanKSPro import ProPolicy
from DanKSPro.adapter import public_context, wire_actions
from DanKSPro.endgame import exact, information
from DanKSPro.endgame.runtime import public_observation


def synthetic_position():
    random.seed(2026)
    env = Environment(allow_step_back=True, first_player=0)
    for seat in range(4):
        env.add_player(str(seat), seat)
    env.start()
    hands = (("S3", "S5"), ("S4",), ("S6",), ("S7",))
    for player, hand in zip(env.players, hands):
        player.hand_cards = []
        player.hearts_num = 0
        for card in hand:
            player.add_card(card, env.rank)
    env.state.reset()
    env.settlement.clear()
    env.first_action(0)
    env.action_first = True
    env.act([], 0, Phase.PLAY)
    env.state.current_pos = 0
    env.loop = env.play
    env.trace.clear()
    remaining = Counter(card for hand in hands for card in hand)
    played = list((Counter(information.canonical_double_deck()) - remaining).elements())
    history, offset = [], 0
    for seat, hand in enumerate(hands):
        for card in played[offset:offset + 27 - len(hand)]:
            history.append(dict(pos=seat, cards=[card], kind="Single",
                                rank=card if card in {"BJ", "RJ"} else card[1:]))
        offset += 27 - len(hand)
    return env, played, history


def main():
    torch.manual_seed(2026)
    torch.set_num_threads(1)
    model = EfficientTeamBeliefTop10Selector(STATE_DIM, CANDIDATE_DIM,
                                           hidden_dim=32, candidate_hidden_dim=24)
    # Same random network is used only to exercise expert proposal wiring.
    policy = ProPolicy(model, specialist=model, extended_specialist=model)
    env, played, history = synthetic_position()
    observation = public_observation(env)
    actions = wire_actions(env.legal_moves.action_list)
    context, events = public_context(observation, history, actions)
    baseline, record = policy.act(observation["hand"], context, actions, history=events)
    before = exact.restoration_key(env, 0)
    final = policy.refine_endgame(
        env, actor_seat=0, actor_hand=observation["hand"], played_cards=played,
        public_counts=observation["public_counts"], record=record)
    restored = exact.restoration_key(env, 0) == before
    trace = record["trace"]["endgame"]
    if not restored or final not in range(len(actions)):
        raise RuntimeError("search changed root state or returned an illegal action")
    if trace.get("reason") in {"adapter_error", "verification_error", "invalid_public_information"}:
        raise RuntimeError(f"search integration failed: {trace}")
    if not trace.get("proposed_indices") or "decision" not in trace:
        raise RuntimeError(f"example did not exercise complete search: {trace}")
    print(f"DanKS V3Pro ready: baseline={baseline}, final={final}, "
          f"worlds={trace['total_allocations']}, restored={restored}")
    print("Random model and synthetic position only; no strength or latency claim.")


if __name__ == "__main__":
    main()
