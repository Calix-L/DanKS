import { matchingActionsForSelection, isOwnPlayDecision, isSpectatorRole } from "./room-state.mjs?v=20261005-r1";
import { cardParts } from "./adapters.mjs?v=20260805-1";
import { guandanTypeLabel } from "./guandan-presentation.mjs?v=20261004-3";

// One logical group stays in one stack. Height is bounded by exposed-strip
// spacing, never by breaking a straight or bomb into unrelated columns.
export function boundedHandColumns(groups) {
  const logical = new Map();
  for (const group of groups) {
    const key = group.logicalKey ?? group.key ?? group.cards[0]?.id;
    if (!group.cards.length) continue;
    if (!logical.has(key)) logical.set(key, { ...group, key, logicalKey: key, cards: [] });
    logical.get(key).cards.push(...group.cards);
  }
  return [...logical.values()].map(group => ({ ...group, columnIndex: 0, columnCount: 1,
    selectionIds: group.cards.map(card => card.id) }));
}

export function fitHandStackReveal({ count, preferred, cardHeight, maxHeight }) {
  return count <= 1 ? preferred : Math.min(preferred, (maxHeight - cardHeight) / (count - 1));
}

// A visual index, not a hand decomposition or action recommendation. Preserve
// the physical card objects so equal card codes never become one selectable card.
export function rankDisplayGroups(ownHand, levelRank) {
  const level = String(levelRank ?? "").toUpperCase().replace(/^T$/, "10");
  const ranks = new Map();
  for (const card of ownHand) {
    const parts = cardParts(card.code);
    const rank = parts.joker ? parts.jokerVariant : parts.rank;
    if (!ranks.has(rank)) ranks.set(rank, []);
    ranks.get(rank).push(card);
  }
  const weight = rank => rank === "big" ? 30 : rank === "small" ? 29 : rank === level ? 28
    : ({ A: 14, K: 13, Q: 12, J: 11 }[rank] ?? (Number(rank) || 0));
  return [...ranks].sort(([a], [b]) => weight(b) - weight(a))
    .map(([rank, cards]) => ({ key: `rank:${rank}`, rank, pattern: 0, cards }));
}

// Preparation is local-only: it never enables submission or reads an opponent's
// action list. Use the same guard for cards, groups, clear and sweep controls.
export function canPrepareOwnHand(view, { pendingAction = null, pendingArrange = false, interactionFrozen = false } = {}) {
  return Boolean(view?.phase === "play" && Number.isInteger(view.viewer?.seat) && view.viewer.seat >= 0
    && !isSpectatorRole(view.viewer.role) && view.own_hand?.length
    && !pendingAction && !pendingArrange && !interactionFrozen);
}

// Expanded selection is a view of the current physical hand, never an action
// source. Reject stale/incomplete groups instead of binding duplicates by code.
export function handInspectionSignature(view, groups, options = {}) {
  if (!canPrepareOwnHand(view, options)) return null;
  const current = new Map(view.own_hand.map(card => [card.id, card.code]));
  if (current.size !== view.own_hand.length) return null;
  const used = new Set(), keys = new Set();
  for (const group of groups) {
    if (keys.has(group.key) || !group.cards.length) return null;
    keys.add(group.key);
    for (const card of group.cards) {
      if (!card.id || used.has(card.id) || !current.has(card.id) || current.get(card.id) !== card.code) return null;
      used.add(card.id);
    }
  }
  if (used.size !== current.size) return null;
  return JSON.stringify([view.table_no, view.round_no, view.game, view.viewer.seat, view.level,
    groups.map(group => [group.key, group.cards.map(card => [card.id, card.code])])]);
}

function sameHandContext(previous, next) {
  return previous?.table_no === next?.table_no && previous?.round_no === next?.round_no
    && previous?.viewer?.seat === next?.viewer?.seat && previous?.game === next?.game;
}

// Choices are capabilities of one snapshot, not reusable action indices.
export function actionChoiceContext(view) {
  if (view?.phase !== "play" || !isOwnPlayDecision(view) || !view.decision.id || !Number.isInteger(view.version)) return null;
  return JSON.stringify([view.table_no, view.round_no, view.game, view.level, view.phase,
    view.viewer.seat, view.viewer.role, view.version, view.decision]);
}

export function isCurrentActionChoice(view, action, context, options = {}) {
  if (!context || options.pendingAction || options.pendingArrange || options.interactionFrozen
    || context !== actionChoiceContext(view)) return false;
  return view.decision.actions.some(current => JSON.stringify(current) === JSON.stringify(action));
}

// Share exactly the selection occurrence policy with focus and display slots.
// A full bijection is required; a changed hand must never silently rebind UI.
export function handRefreshBindings(previous, next, groups, options = {}) {
  if (!sameHandContext(previous, next) || previous.level !== next.level
    || !handInspectionSignature(previous, groups, options) || !canPrepareOwnHand(next, options)) return null;
  const before = previous.own_hand.map(card => card.code).sort();
  const after = next.own_hand.map(card => card.code).sort();
  if (before.length !== after.length || before.some((code, i) => code !== after[i])
    || new Set(next.own_hand.map(card => card.id)).size !== next.own_hand.length) return null;
  const rebound = rebindDisplayGroups([{key: "refresh", cards: previous.own_hand}], next.own_hand)[0]?.cards ?? [];
  if (rebound.length !== previous.own_hand.length) return null;
  return new Map(previous.own_hand.map((card, i) => [card.id, rebound[i]]));
}

// IDs are reissued on engine advances. Only an unchanged full own-hand code
// multiset permits occurrence binding. Reserve every surviving exact ID first,
// including unselected duplicates, so no selected card steals another's ID.
export function reconcileHandSelection(previous, next, selectedIds, { pendingAction = null } = {}) {
  const cleared = () => new Set();
  if (!selectedIds.size || !canPrepareOwnHand(previous) || !canPrepareOwnHand(next)
    || !sameHandContext(previous, next)) return cleared();
  const decisionChanged = previous.decision?.id !== next.decision?.id;
  if (decisionChanged && (isOwnPlayDecision(previous) || pendingAction)) return cleared();
  const previousCodes = previous.own_hand.map(card => card.code).sort();
  const nextCodes = next.own_hand.map(card => card.code).sort();
  if (previousCodes.length !== nextCodes.length || previousCodes.some((code, i) => code !== nextCodes[i])) return cleared();
  const previousCards = new Map(previous.own_hand.map(card => [card.id, card]));
  if ([...selectedIds].some(id => !previousCards.has(id))) return cleared();
  if (pendingAction) {
    const nextCards = new Map(next.own_hand.map(card => [card.id, card]));
    return [...selectedIds].every(id => nextCards.get(id)?.code === previousCards.get(id).code)
      ? new Set(selectedIds) : cleared();
  }
  const rebound = rebindDisplayGroups([{ key: "selection", cards: previous.own_hand }], next.own_hand)[0]?.cards ?? [];
  if (rebound.length !== previous.own_hand.length) return cleared();
  const identities = new Map(previous.own_hand.map((card, i) => [card.id, rebound[i].id]));
  return new Set([...selectedIds].map(id => identities.get(id)));
}

export function beginHandSweep(selectedIds, startCardId) {
  return { intent: selectedIds.has(startCardId) ? "erase" : "select",
    visitedIds: new Set(), baselineSelectedIds: new Set(selectedIds) };
}

export function paintHandSweep(gesture, selectedIds, cardId) {
  if (!cardId || gesture.visitedIds.has(cardId)) return selectedIds;
  gesture.visitedIds.add(cardId);
  const next = new Set(selectedIds);
  if (gesture.intent === "erase") next.delete(cardId);
  else next.add(cardId);
  return next;
}

export function canContinueHandGesture(gesture, view, options) {
  const handCards = new Map((view?.own_hand ?? []).map(card => [card.id, card.code]));
  return canPrepareOwnHand(view, options)
    && gesture.decisionId === (view.decision?.id ?? "")
    && gesture.tableNo === view.table_no
    && gesture.roundNo === view.round_no
    && gesture.viewerSeat === view.viewer.seat
    && gesture.game === view.game
    && gesture.handIds.length === handCards.size
    && gesture.handIds.every((id, i) => handCards.has(id) && handCards.get(id) === gesture.handCodes[i]);
}

// A metadata/arrangement version bump is not proof that a play was accepted.
// Resolve only on an authoritative decision/seat change or a confirmed departure.
export function pendingActionResolved(pending, previousView, nextView) {
  if (!pending) return false;
  if (pending.tableNo !== nextView.table_no
    || previousView?.viewer?.seat !== nextView.viewer?.seat
    || pending.decisionId !== (nextView.decision?.id ?? "")) return true;
  const departureCount = pending.selectedIds?.length ?? 0;
  return departureCount > 0 && nextView.version > pending.version
    && (previousView?.own_hand?.length ?? 0) - nextView.own_hand.length === departureCount;
}

// Presentation only: the server's action list remains the legality authority.
export function selectionFeedback(view, selectedIds) {
  if (!selectedIds.size) return { kind: "empty", matches: [] };
  if (!isOwnPlayDecision(view)) return { kind: "prepared", matches: [] };
  const presentIds = new Set(view.own_hand.map(card => card.id));
  const matches = [...selectedIds].every(id => presentIds.has(id))
    ? matchingActionsForSelection(view.decision?.actions, view.own_hand, selectedIds, view.game)
    : [];
  return { kind: matches.length > 1 ? "ambiguous" : matches.length ? "valid" : "invalid", matches };
}

// Explain facts from the current server action list only. Card-count mismatch
// is observable; poker type/rank/wildcard legality is never recreated here.
export function selectionExplanation(view, selectedIds) {
  if (!selectedIds.size) return "点选手牌，也可整组选择";
  const ids = new Set(view.own_hand.map(card => card.id));
  if ([...selectedIds].some(id => !ids.has(id))) return "选牌已更新，请重新选择";
  if (!isOwnPlayDecision(view)) return `已预选 ${selectedIds.size} 张，轮到你时校验`;
  const { kind, matches } = selectionFeedback(view, selectedIds);
  if (kind === "valid") return `已选 ${selectedIds.size} 张 · ${guandanTypeLabel(matches[0])}`;
  if (kind === "ambiguous") return `已选 ${selectedIds.size} 张 · 提交时选择牌型`;
  const actions = view.decision?.actions ?? [];
  const plays = actions.filter(action => action.cards?.length);
  if (!plays.length && actions.some(action => String(action.type).toUpperCase() === "PASS")) return "当前只能不出";
  const counts = [...new Set(plays.map(action => action.cards.length))].sort((a, b) => a - b);
  if (counts.length && !counts.includes(selectedIds.size)) return `已选 ${selectedIds.size} 张 · 当前可出 ${counts.join("、")} 张的组合`;
  if (plays.length && view.current_trick?.mode === "follow") return `已选 ${selectedIds.size} 张 · 不能接当前这手牌`;
  return "当前组合不可出，请调整选牌";
}

export function toggleGroupSelection(selectedIds, cardIds) {
  const next = new Set(selectedIds);
  const allSelected = cardIds.every(id => next.has(id));
  for (const id of cardIds) {
    if (allSelected) next.delete(id);
    else next.add(id);
  }
  return next;
}

export function groupSelectionFeedback(selectedIds, cardIds) {
  const total = cardIds.length;
  const selected = cardIds.filter(id => selectedIds.has(id)).length;
  return { pressed: !selected ? "false" : selected === total ? "true" : "mixed", selected, total };
}

// The room protocol issues fresh IDs when engine state advances. Rebind only
// the presentation positions to current physical objects; never submit old IDs
// or reconstruct an action. Confirmed selected cards are removed first so an
// indistinguishable duplicate elsewhere in the hand keeps its familiar group.
export function rebindDisplayGroups(previous, ownHand, departedIds = new Set()) {
  const exact = new Map(ownHand.map(card => [card.id, card]));
  const reserved = new Set(previous.flatMap(group => group.cards)
    .filter(card => !departedIds.has(card.id) && exact.get(card.id)?.code === card.code).map(card => card.id));
  const used = new Set();
  const pools = new Map();
  for (const card of ownHand) {
    if (reserved.has(card.id)) continue;
    if (!pools.has(card.code)) pools.set(card.code, []);
    pools.get(card.code).push(card);
  }
  return previous.map(group => ({ ...group, cards: group.cards.flatMap(card => {
    if (departedIds.has(card.id)) return [];
    const match = exact.get(card.id)?.code === card.code && !used.has(card.id)
      ? exact.get(card.id) : pools.get(card.code)?.shift();
    if (!match || used.has(match.id)) return [];
    used.add(match.id);
    return [match];
  }) })).filter(group => group.cards.length);
}

// Keep familiar display positions between snapshots; an explicit arrangement
// or round change opts into the incoming order. Never change the actual hand.
export function stableHandGroups(previous, incoming, ownHand, { reset = false } = {}) {
  const current = new Map(ownHand.map(card => [card.id, card]));
  const used = new Set();
  const result = [];
  const add = group => {
    const cards = [];
    for (const card of group.cards ?? []) {
      if (current.has(card.id) && !used.has(card.id)) {
        cards.push(current.get(card.id));
        used.add(card.id);
      }
    }
    if (cards.length) result.push({ ...group, key: group.key ?? cards[0].id, cards });
  };
  if (!reset) previous.forEach(add);
  incoming.forEach(add);
  for (const card of ownHand) if (!used.has(card.id)) add({ pattern: 0, cards: [card] });
  return result;
}
