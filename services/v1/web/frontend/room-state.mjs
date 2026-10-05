import { cardParts, normalizeGuandanLevel } from "./adapters.mjs?v=20260801-30";

const GAMES = new Set(["guandan", "doudizhu"]);
const SPECTATOR_ROLES = new Set(["spectator", "observer"]);
const LOBBY_PHASES = new Set(["lobby", "waiting", "ready"]);
const TERMINAL_PHASES = new Set(["finished", "completed", "game_over", "gameover"]);
const ARRANGE_MODES = new Set([1, 2, 3, 4]);
const GUANDAN_RULE_PROFILES = new Set(["plm_release_v2", "arena_client_pdf_v1", "sports_bureau"]);

/**
 * Human-table snapshots deliberately use their own strict projection. This is
 * a privacy boundary: adding a field to a server object does not make it appear
 * in the browser automatically.
 */
export function normalizeRoomView(rawView) {
  const source = objectValue(rawView);
  const game = GAMES.has(source.game) ? source.game : "guandan";
  const phase = stringValue(source.phase, "lobby");
  const viewer = normalizeViewer(source.viewer);
  const seatCount = game === "doudizhu" ? 3 : 4;
  const seatsByPosition = new Map();

  for (const rawSeat of arrayValue(source.seats)) {
    const seat = normalizeSeat(rawSeat, seatCount);
    if (seat && !seatsByPosition.has(seat.position)) seatsByPosition.set(seat.position, seat);
  }

  const seats = Array.from({ length: seatCount }, (_, position) =>
    seatsByPosition.get(position) ?? emptySeat(position),
  );
  const decision = normalizeDecision(source.decision, viewer, seatCount);
  const maySeeOwnHand = !isSpectatorRole(viewer.role) && viewer.seat !== null;
  const ownHand = maySeeOwnHand ? normalizeOwnHand(source.own_hand) : [];
  const spectatorHands = (isSpectatorRole(viewer.role) || isTerminalPhase(phase))
    ? normalizeSpectatorHands(source.spectator_hands, seatCount)
    : [];
  const arrangeMode = maySeeOwnHand
    ? arrangeModeValue(source.arrange_mode, game === "doudizhu" ? 4 : 1)
    : null;
  const level = game === "guandan" ? normalizeGuandanLevel(source.level) || null : null;
  const guandanLevels = game === "guandan"
    ? normalizeGuandanLevels(source.guandan_levels, level)
    : null;

  return {
    table_no: stringValue(source.table_no),
    game,
    human_slots: clampInteger(source.human_slots, 1, 3, 1),
    phase,
    version: nonNegativeInteger(source.version, 0),
    round_no: positiveInteger(source.round_no, 1),
    level,
    guandan_levels: guandanLevels,
    series: game === "guandan" ? normalizeGuandanSeries(source.series, seatCount) : null,
    viewer,
    seats,
    spectator_count: nonNegativeInteger(source.spectator_count, 0),
    bot_policy: normalizeBotPolicy(source.bot_policy),
    rule_profile: game === "guandan" && GUANDAN_RULE_PROFILES.has(source.rule_profile)
      ? source.rule_profile
      : "",
    doudizhu_state: game === "doudizhu"
      ? normalizeDoudizhuState(source.doudizhu_state, phase, seatCount)
      : null,
    finish_order: normalizePositionList(source.finish_order, seatCount),
    latest_replay_match_id: normalizeReplayMatchId(source.latest_replay_match_id),
    replay_history: normalizeReplayHistory(source.replay_history),
    own_hand: ownHand,
    spectator_hands: spectatorHands,
    arrange_mode: arrangeMode,
    own_hand_layout: maySeeOwnHand
      ? normalizeOwnHandLayout(source.own_hand_layout, ownHand, arrangeMode)
      : null,
    decision,
    turn_started_at_ms: finiteNumberOrNull(source.turn_started_at_ms),
    turn_seconds: clampInteger(source.turn_seconds, 1, 3600, 20),
    result: normalizeResult(source.result, seatCount),
    server_time_ms: finiteNumberOrNull(source.server_time_ms),
  };
}

function normalizeReplayMatchId(value) {
  if (typeof value !== "string") return "";
  const matchId = value.trim();
  if (!matchId || matchId.length > 256 || /[\u0000-\u001f\u007f]/u.test(matchId)) return "";
  return matchId;
}

function normalizeReplayHistory(rawHistory) {
  const entries = [];
  for (const rawEntry of arrayValue(rawHistory)) {
    const source = objectValue(rawEntry);
    const roundNo = integerOrNull(source.round_no);
    const matchId = normalizeReplayMatchId(source.match_id);
    if (roundNo === null || roundNo <= 0 || !matchId) continue;
    entries.push({
      series_no: positiveInteger(source.series_no, 1),
      round_no: roundNo,
      match_id: matchId,
    });
  }
  return entries.slice(-64);
}

function normalizeSpectatorHands(rawHands, seatCount) {
  const source = arrayValue(rawHands);
  return Array.from({ length: seatCount }, (_, position) =>
    normalizeOwnHand(source[position]),
  );
}

function normalizeGuandanLevels(rawLevels, fallbackLevel) {
  const source = objectValue(rawLevels);
  const ourLevel = normalizeGuandanLevel(source.our_level) || "2";
  const opponentLevel = normalizeGuandanLevel(source.opponent_level) || "2";
  const currentLevel = normalizeGuandanLevel(source.current_level) || fallbackLevel || "2";
  const activeSide = ["our", "opponent"].includes(source.active_side)
    ? source.active_side
    : "";
  return {
    our_level: ourLevel,
    opponent_level: opponentLevel,
    current_level: currentLevel,
    active_side: activeSide,
    a_challenger_side: ["our", "opponent"].includes(source.a_challenger_side)
      ? source.a_challenger_side
      : "",
  };
}

function normalizeGuandanSeries(rawSeries, seatCount) {
  const source = objectValue(rawSeries);
  const rawHeadCounts = arrayValue(source.head_finish_count_by_seat);
  const levels = objectValue(source.team_levels);
  const victories = objectValue(source.team_victory_count);
  const attempts = objectValue(source.a_attempt_count);
  const teams = new Set(["even", "odd"]);
  const activeTeam = stringValue(source.active_team);
  const challenger = stringValue(source.a_challenger);
  const winner = stringValue(source.match_winner_team);
  return {
    schema_version: stringValue(source.schema_version),
    series_no: positiveInteger(source.series_no, 1),
    round_no: positiveInteger(source.round_no, 1),
    team_levels: {
      even: normalizeGuandanLevel(levels.even) || "2",
      odd: normalizeGuandanLevel(levels.odd) || "2",
    },
    current_level: normalizeGuandanLevel(source.current_level) || "2",
    active_team: teams.has(activeTeam) ? activeTeam : "",
    a_challenger: teams.has(challenger) ? challenger : "",
    a_attempt_count: {
      even: nonNegativeInteger(attempts.even, 0),
      odd: nonNegativeInteger(attempts.odd, 0),
    },
    head_finish_count_by_seat: Array.from({ length: seatCount }, (_, position) =>
      nonNegativeInteger(rawHeadCounts[position], 0)),
    team_victory_count: {
      even: nonNegativeInteger(victories.even, 0),
      odd: nonNegativeInteger(victories.odd, 0),
    },
    rounds_played: nonNegativeInteger(source.rounds_played, 0),
    match_winner_team: teams.has(winner) ? winner : "",
    match_finished: source.match_finished === true,
    viewer_team: teams.has(source.viewer_team) ? source.viewer_team : "",
    final_result: normalizeGuandanFinalResult(source.final_result, seatCount),
  };
}

function normalizeGuandanFinalResult(rawResult, seatCount) {
  const source = objectValue(rawResult);
  if (!Object.keys(source).length) return null;
  const teams = new Set(["even", "odd"]);
  const winner = stringValue(source.winner_team);
  const loser = stringValue(source.loser_team);
  if (!teams.has(winner) || !teams.has(loser) || winner === loser) return null;
  const levels = objectValue(source.team_levels);
  const victories = objectValue(source.team_victory_count);
  const attempts = objectValue(source.a_attempt_count);
  return {
    schema_version: stringValue(source.schema_version),
    series_no: positiveInteger(source.series_no, 1),
    winner_team: winner,
    loser_team: loser,
    team_levels: {
      even: normalizeGuandanLevel(levels.even) || "2",
      odd: normalizeGuandanLevel(levels.odd) || "2",
    },
    winning_team_level: normalizeGuandanLevel(source.winning_team_level) || "A",
    losing_team_level: normalizeGuandanLevel(source.losing_team_level) || "2",
    head_finish_count_by_seat: Array.from({ length: seatCount }, (_, position) =>
      nonNegativeInteger(arrayValue(source.head_finish_count_by_seat)[position], 0)),
    team_victory_count: {
      even: nonNegativeInteger(victories.even, 0),
      odd: nonNegativeInteger(victories.odd, 0),
    },
    a_attempt_count: {
      even: nonNegativeInteger(attempts.even, 0),
      odd: nonNegativeInteger(attempts.odd, 0),
    },
    winning_a_attempt_no: positiveInteger(source.winning_a_attempt_no, 1),
    last_finish_order: normalizePositionList(source.last_finish_order, seatCount),
    rounds_played: positiveInteger(source.rounds_played, 1),
    replay_ids: arrayValue(source.replay_ids).map(normalizeReplayMatchId).filter(Boolean),
  };
}

function normalizeDoudizhuState(rawState, phase, seatCount) {
  if (!rawState || typeof rawState !== "object" || Array.isArray(rawState)) return null;
  const source = objectValue(rawState);
  const landlordPosition = validPositionOrNull(
    source.landlord_position ?? source.landlord,
    seatCount,
  );
  const highestBidder = validPositionOrNull(
    source.highest_bidder ?? source.highestBidder,
    seatCount,
  );
  const revealBottom = String(phase ?? "").toLowerCase() === "play" || isTerminalPhase(phase);
  return {
    landlord_position: landlordPosition,
    highest_bid: clampInteger(source.highest_bid ?? source.highestBid, 0, 3, 0),
    highest_bidder: highestBidder,
    bids: normalizeSeatNumberMap(source.bids, seatCount, 0, 3),
    pending_double: normalizePositionList(source.pending_double ?? source.pendingDouble, seatCount),
    doubles: normalizeSeatBooleanMap(source.doubles, seatCount),
    redouble: source.redouble === true,
    base_score: nonNegativeInteger(source.base_score ?? source.baseScore, 0),
    bottom_cards: revealBottom
      ? arrayValue(source.bottom_cards ?? source.bottomCards).slice(0, 3).map(cardCode).filter(Boolean)
      : [],
    multiplier: positiveInteger(source.multiplier, 1),
    defender_multipliers: normalizeSeatPositiveIntegerMap(
      source.defender_multipliers ?? source.defenderMultipliers,
      seatCount,
    ),
    bombs: nonNegativeInteger(source.bombs, 0),
    rockets: nonNegativeInteger(source.rockets, 0),
  };
}

function normalizeBotPolicy(rawPolicy) {
  const source = objectValue(rawPolicy);
  const id = stringValue(source.id);
  if (!id) return null;
  const rawSha = stringValue(source.sha256).toLowerCase();
  const sha256 = /^[0-9a-f]{64}$/u.test(rawSha) ? rawSha : "";
  return {
    id,
    ready: source.ready === true,
    sha256,
    retrieval_profile: stringValue(source.retrieval_profile),
    break_group_weight: finiteNumberOrNull(source.break_group_weight),
    device: stringValue(source.device),
    history_protocol: stringValue(source.history_protocol),
    history_events: nonNegativeInteger(source.history_events, 0),
  };
}

export function isSpectatorRole(role) {
  return SPECTATOR_ROLES.has(String(role ?? "").toLowerCase());
}

export function isLobbyPhase(phase) {
  return LOBBY_PHASES.has(String(phase ?? "").toLowerCase());
}

export function isTerminalPhase(phase) {
  return TERMINAL_PHASES.has(String(phase ?? "").toLowerCase());
}

export function isReadyPhase(phase) {
  return isLobbyPhase(phase) || isTerminalPhase(phase);
}

export function isOwnDecision(view) {
  return Boolean(
    view?.decision &&
      !isSpectatorRole(view?.viewer?.role) &&
      Number.isInteger(view?.viewer?.seat) &&
      view.viewer.seat === view.decision.actor,
  );
}

export function isOwnPlayDecision(view) {
  return Boolean(
    isOwnDecision(view) &&
      String(view?.decision?.phase ?? "").toLowerCase() === "play",
  );
}

export function selectedCardCodes(ownHand, selectedIds) {
  const wanted = selectedIds instanceof Set ? selectedIds : new Set(arrayValue(selectedIds).map(String));
  return arrayValue(ownHand)
    .filter((card) => wanted.has(String(card?.id ?? "")))
    .map((card) => String(card?.code ?? ""))
    .filter(Boolean);
}

export function matchingActionsForSelection(actions, ownHand, selectedIds, game = "guandan") {
  const selectedCodes = selectedCardCodes(ownHand, selectedIds);
  if (!selectedCodes.length) return [];
  const signature = game === "doudizhu"
    ? doudizhuRankMultisetSignature
    : cardMultisetSignature;
  const selectedSignature = signature(selectedCodes);
  return arrayValue(actions).filter((action) => {
    const codes = actionCardCodes(action);
    return codes.length > 0 && signature(codes) === selectedSignature;
  });
}

export function toggleSelectionEntries(selectedIds, cardIds) {
  const next = selectedIds instanceof Set
    ? new Set([...selectedIds].map((value) => String(value)))
    : new Set(arrayValue(selectedIds).map((value) => String(value)));
  for (const rawCardId of arrayValue(cardIds)) {
    const cardId = stringValue(rawCardId);
    if (!cardId) continue;
    if (next.has(cardId)) next.delete(cardId);
    else next.add(cardId);
  }
  return next;
}

export function fitHandGroupStep({
  availableWidth = 0,
  groupCount = 0,
  naturalStep = 75,
  minimumStep = 26,
  lastGroupWidth = 74,
  paddingInline = 0,
  gap = 0,
} = {}) {
  const count = Math.max(0, Math.trunc(Number(groupCount) || 0));
  const natural = Math.max(0, Number(naturalStep) || 0);
  const minimum = Math.min(natural, Math.max(0, Number(minimumStep) || 0));
  const lastWidth = count ? Math.max(0, Number(lastGroupWidth) || 0) : 0;
  const padding = Math.max(0, Number(paddingInline) || 0);
  const spacing = Math.max(0, Number(gap) || 0);
  const precedingGroups = Math.max(0, count - 1);
  const fixedWidth = padding + lastWidth + spacing * precedingGroups;
  const naturalWidth = fixedWidth + natural * precedingGroups;
  const minimumWidth = fixedWidth + minimum * precedingGroups;
  const available = Math.max(0, Number(availableWidth) || 0);
  const fitted = precedingGroups && available
    ? (available - fixedWidth) / precedingGroups
    : natural;
  const step = Math.min(natural, Math.max(minimum, fitted));
  return {
    step,
    wrap: Boolean(count && available && available + 0.5 < minimumWidth),
    groupsPerRow: Math.max(
      1,
      Math.floor(Math.max(0, available - padding - lastWidth) /
        (minimum + spacing)) + 1,
    ),
    naturalWidth,
    minimumWidth,
  };
}

export function resultWinningPositions(result, game = "guandan", seatCount = 4) {
  if (!result || typeof result !== "object" || Array.isArray(result)) return [];
  const count = Math.max(1, Math.trunc(Number(seatCount) || 1));
  const explicit = normalizePositionList(result.winning_positions, count).sort((left, right) => left - right);
  if (explicit.length) return explicit;

  const side = stringValue(result.winning_side).toLowerCase();
  if (game === "guandan" && ["even", "odd"].includes(side)) {
    const parity = side === "even" ? 0 : 1;
    return Array.from({ length: count }, (_, position) => position)
      .filter((position) => position % 2 === parity);
  }

  const normalizedFinishOrder = normalizePositionList(result.finish_order, count);
  const order = normalizedFinishOrder.length
    ? normalizedFinishOrder
    : normalizePositionList(result.order, count);
  if (game === "guandan" && order.length) {
    const parity = order[0] % 2;
    return Array.from({ length: count }, (_, position) => position)
      .filter((position) => position % 2 === parity);
  }
  const winnerPosition = integerOrNull(result.winner_position ?? result.winnerPosition);
  return winnerPosition !== null && winnerPosition >= 0 && winnerPosition < count
    ? [winnerPosition]
    : [];
}

export function guandanRoundPresentation(result, viewerSeat) {
  if (!result || typeof result !== "object") return null;
  const viewerTeam = Number.isInteger(viewerSeat) && viewerSeat % 2 === 1 ? "odd" : "even";
  const opponentTeam = viewerTeam === "even" ? "odd" : "even";
  const winningTeam = ["even", "odd"].includes(result.winning_team)
    ? result.winning_team
    : "";
  if (!winningTeam) return null;
  const won = winningTeam === viewerTeam;
  const partnerFinish = clampInteger(result.partner_finish, 0, 4, 0);
  const upgradeLevels = clampInteger(result.upgrade_levels, 0, 3, 0)
    || ({ 2: 3, 3: 2, 4: 1 }[partnerFinish] ?? 0);
  const outcomeLabel = partnerFinish === 2
    ? "双上"
    : partnerFinish === 3
      ? "一三游"
      : partnerFinish === 4
        ? "一四游"
        : upgradeLevels === 3
          ? "双上"
          : upgradeLevels === 2
            ? "一三游"
            : upgradeLevels === 1
              ? "一四游"
              : "本局结算";
  const levelsBefore = objectValue(result.team_levels_before);
  const levelsAfter = objectValue(result.team_levels);
  const displayLevel = (value) => value === "T" ? "10" : value;
  const viewerBefore = displayLevel(normalizeGuandanLevel(levelsBefore[viewerTeam])) || "—";
  const viewerAfter = displayLevel(normalizeGuandanLevel(levelsAfter[viewerTeam])) || viewerBefore;
  const opponentBefore = displayLevel(normalizeGuandanLevel(levelsBefore[opponentTeam])) || "—";
  const opponentAfter = displayLevel(normalizeGuandanLevel(levelsAfter[opponentTeam])) || opponentBefore;
  const nextLevel = displayLevel(normalizeGuandanLevel(result.next_current_level)
    || normalizeGuandanLevel(result.next_level)
    || normalizeGuandanLevel(levelsAfter[winningTeam]))
    || "—";
  return {
    outcomeLabel,
    sideLabel: won ? "我方" : "对方",
    signedDelta: `${won ? "+" : "−"}${upgradeLevels}`,
    levelChange: `我方 ${viewerBefore} → ${viewerAfter}`,
    opponentLevel: `对方 ${opponentBefore} → ${opponentAfter}`,
    nextLevel: `下局级牌 ${nextLevel}`,
    won,
  };
}

export function actionCardCodes(action) {
  if (Array.isArray(action)) return arrayValue(action[2]).map(cardCode).filter(Boolean);
  const value = objectValue(action);
  const nested = objectValue(value.action);
  const wire = Array.isArray(value.wire) ? value.wire : Array.isArray(nested.wire) ? nested.wire : [];
  const rawCards =
    value.cards ?? value.card_codes ?? nested.cards ?? nested.card_codes ?? (Array.isArray(wire[2]) ? wire[2] : []);
  return arrayValue(rawCards).map(cardCode).filter(Boolean);
}

export function cardMultisetSignature(codes) {
  return countedSignature(arrayValue(codes).map(cardCode).filter(Boolean));
}

export function doudizhuRankMultisetSignature(codes) {
  const ranks = arrayValue(codes)
    .map(cardCode)
    .filter(Boolean)
    .map((code) => {
      const parts = cardParts(code);
      if (!parts.joker) return parts.rank;
      return parts.jokerVariant === "big" ? "RED_JOKER" : "SMALL_JOKER";
    })
    .filter(Boolean);
  return countedSignature(ranks);
}

function countedSignature(values) {
  const counts = new Map();
  for (const value of values) {
    counts.set(value, (counts.get(value) ?? 0) + 1);
  }
  return [...counts.entries()]
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([value, count]) => `${encodeURIComponent(value)}:${count}`)
    .join("|");
}

export function actionChoiceLabel(action, ordinal = 0) {
  const value = objectValue(action);
  const type = stringValue(value.label ?? value.type ?? value.kind ?? value.name, "合法出牌");
  const rank = stringValue(value.rank ?? value.main_rank);
  const cards = actionCardCodes(value);
  const semantic = rank ? `${type} · ${rank}` : type;
  return `${ordinal + 1}. ${semantic}${cards.length ? ` · ${cards.join(" ")}` : ""}`;
}

export function rotatedSeatPosition(position, viewerSeat, seatCount) {
  const count = Number.isInteger(seatCount) && seatCount > 0 ? seatCount : 4;
  const normalizedPosition = clampInteger(position, 0, count - 1, 0);
  if (!Number.isInteger(viewerSeat) || viewerSeat < 0 || viewerSeat >= count) return normalizedPosition;
  return (normalizedPosition - viewerSeat + count) % count;
}

export function seatVisualClass(game, position, viewerSeat = null) {
  const seatCount = game === "doudizhu" ? 3 : 4;
  const relative = rotatedSeatPosition(position, viewerSeat, seatCount);
  const classes = game === "doudizhu"
    ? ["seat-bottom", "seat-left", "seat-right"]
    : ["seat-bottom", "seat-right", "seat-top", "seat-left"];
  return classes[relative] ?? "seat-bottom";
}

function normalizeViewer(rawViewer) {
  const source = objectValue(rawViewer);
  return {
    role: stringValue(source.role, "spectator").toLowerCase(),
    seat: integerOrNull(source.seat),
    participant_id: stringValue(source.participant_id),
  };
}

function normalizeSeat(rawSeat, seatCount) {
  const source = objectValue(rawSeat);
  const position = integerOrNull(source.position);
  if (position === null || position < 0 || position >= seatCount) return null;
  return {
    position,
    kind: stringValue(source.kind, "open").toLowerCase(),
    display_name: stringValue(source.display_name),
    ready: source.ready === true,
    hand_count: nonNegativeInteger(source.hand_count, 0),
    play_area: normalizePlayArea(source.play_area),
    team: stringValue(source.team),
    role: stringValue(source.role),
  };
}

function emptySeat(position) {
  return {
    position,
    kind: "open",
    display_name: "",
    ready: false,
    hand_count: 0,
    play_area: { type: "", label: "", pass: false, cards: [] },
    team: "",
    role: "",
  };
}

function normalizePlayArea(rawPlayArea) {
  if (Array.isArray(rawPlayArea)) {
    const type = stringValue(rawPlayArea[0]);
    return {
      type,
      label: type,
      pass: type.toUpperCase() === "PASS",
      cards: arrayValue(rawPlayArea[2]).map(cardCode).filter(Boolean),
    };
  }
  const source = objectValue(rawPlayArea);
  return {
    type: stringValue(source.type ?? source.kind),
    label: stringValue(source.label ?? source.name),
    pass: source.pass === true || String(source.type ?? "").toUpperCase() === "PASS",
    cards: actionCardCodes(source),
  };
}

function normalizeOwnHand(rawHand) {
  return arrayValue(rawHand)
    .map((rawCard, index) => {
      const source = objectValue(rawCard);
      const code = cardCode(rawCard);
      if (!code) return null;
      const id = stringValue(source.id ?? source.instance_id, `card-${index}`);
      return { id, code };
    })
    .filter(Boolean);
}

export function normalizeOwnHandLayout(rawLayout, ownHand, defaultMode = 1) {
  const canonicalHand = normalizeOwnHand(ownHand);
  const source = objectValue(rawLayout);
  const mode = arrangeModeValue(source.mode, arrangeModeValue(defaultMode, 1));
  if (!canonicalHand.length) {
    return {
      version: stringValue(source.version),
      mode,
      source_sha: stringValue(source.source_sha),
      fallback: source.fallback === true,
      variant_index: nonNegativeInteger(source.variant_index, 0),
      variant_count: positiveInteger(source.variant_count, 1),
      groups: [],
    };
  }

  const byId = new Map(canonicalHand.map((card) => [card.id, card]));
  const unusedIds = new Set(byId.keys());
  const groups = [];
  const rawGroups = arrayValue(source.groups);
  if (!rawGroups.length) {
    return {
      version: stringValue(source.version),
      mode,
      source_sha: stringValue(source.source_sha),
      fallback: true,
      variant_index: nonNegativeInteger(source.variant_index, 0),
      variant_count: positiveInteger(source.variant_count, 1),
      groups: fallbackRankGroups(canonicalHand),
    };
  }
  for (const rawGroup of rawGroups) {
    const groupSource = objectValue(rawGroup);
    const cards = [];
    for (const rawCard of arrayValue(groupSource.cards)) {
      const rawSource = objectValue(rawCard);
      const requestedId = stringValue(rawSource.id ?? rawSource.instance_id);
      const requestedCode = cardCode(rawCard);
      let card = requestedId && unusedIds.has(requestedId) ? byId.get(requestedId) : null;
      if (card && requestedCode && card.code !== requestedCode) card = null;
      if (!card && requestedCode) {
        card = canonicalHand.find(
          (candidate) => unusedIds.has(candidate.id) && candidate.code === requestedCode,
        );
      }
      if (!card || !unusedIds.delete(card.id)) continue;
      cards.push(card);
    }
    if (!cards.length) continue;
    groups.push({
      pattern: scalarValue(groupSource.pattern),
      minor: scalarValue(groupSource.minor),
      value: scalarValue(groupSource.value),
      cards,
    });
  }

  const omitted = canonicalHand.filter((card) => unusedIds.has(card.id));
  if (omitted.length) {
    groups.push({ pattern: 0, minor: 0, value: null, cards: omitted });
  }
  if (!groups.length) groups.push(...fallbackRankGroups(canonicalHand));
  return {
    version: stringValue(source.version),
    mode,
    source_sha: stringValue(source.source_sha),
    fallback: source.fallback === true || omitted.length > 0,
    variant_index: nonNegativeInteger(source.variant_index, 0),
    variant_count: positiveInteger(source.variant_count, 1),
    groups,
  };
}

function fallbackRankGroups(hand) {
  const rankOrder = new Map(
    ["3", "4", "5", "6", "7", "8", "9", "T", "10", "J", "Q", "K", "A", "2", "JOKER"]
      .map((rank, index) => [rank, index]),
  );
  const groups = new Map();
  for (const card of hand) {
    const rank = cardParts(card.code).rank;
    if (!groups.has(rank)) groups.set(rank, []);
    groups.get(rank).push(card);
  }
  return [...groups.entries()]
    .sort(([left], [right]) => (rankOrder.get(left) ?? 99) - (rankOrder.get(right) ?? 99))
    .map(([rank, cards]) => ({ pattern: 0, minor: 0, value: rank, cards }));
}

function arrangeModeValue(value, fallback) {
  const mode = integerOrNull(value);
  return ARRANGE_MODES.has(mode) ? mode : fallback;
}

function normalizeDecision(rawDecision, viewer, seatCount) {
  if (!rawDecision || typeof rawDecision !== "object" || Array.isArray(rawDecision)) return null;
  const source = objectValue(rawDecision);
  const actor = integerOrNull(source.actor);
  if (actor === null || actor < 0 || actor >= seatCount) return null;
  const ownTurn = !isSpectatorRole(viewer.role) && viewer.seat === actor;
  const rawActions = ownTurn ? arrayValue(source.actions) : [];
  const actions = rawActions.map(normalizeLegalAction).filter(Boolean);
  return {
    id: stringValue(source.id),
    actor,
    phase: stringValue(source.phase, "play"),
    can_pass: ownTurn && source.can_pass === true,
    action_count: nonNegativeInteger(source.action_count, rawActions.length),
    actions,
  };
}

function normalizeLegalAction(rawAction) {
  const source = objectValue(rawAction);
  const actionIndex = integerOrNull(source.action_index ?? source.index);
  if (actionIndex === null || actionIndex < 0) return null;
  return {
    action_index: actionIndex,
    type: stringValue(source.type ?? source.kind),
    rank: stringValue(source.rank ?? source.main_rank),
    cards: actionCardCodes(source),
    label: stringValue(source.label ?? source.name),
  };
}

function normalizeResult(rawResult, seatCount) {
  if (rawResult === null || rawResult === undefined || rawResult === "") return null;
  if (["string", "number", "boolean"].includes(typeof rawResult)) return rawResult;
  const source = objectValue(rawResult);
  if (!Object.keys(source).length) return null;
  const normalizedFinishOrder = normalizePositionList(source.finish_order, seatCount);
  const normalizedOrder = normalizePositionList(source.order, seatCount);
  const finishOrder = normalizedFinishOrder.length === seatCount
    ? normalizedFinishOrder
    : normalizedOrder.length === seatCount
      ? normalizedOrder
      : [];
  const rawScores = arrayValue(source.scores ?? source.handScores ?? source.hand_scores);
  const rawTotalScores = arrayValue(source.total_scores ?? source.totalScores);
  const winnerPosition = integerOrNull(source.winner_position ?? source.winnerPosition);
  const levelsBefore = objectValue(source.team_levels_before);
  const teamLevels = objectValue(source.team_levels);
  const hasGuandanSettlement = [
    "winning_team",
    "upgrade_levels",
    "team_levels",
    "a_challenge_attempted",
    "final_result",
  ].some((key) => Object.prototype.hasOwnProperty.call(source, key));
  return {
    winner: scalarValue(source.winner),
    winner_position:
      winnerPosition !== null && winnerPosition >= 0 && winnerPosition < seatCount
        ? winnerPosition
        : null,
    winning_side: stringValue(source.winning_side ?? source.winningSide),
    winning_positions: normalizePositionList(
      source.winning_positions ?? source.winningPositions,
      seatCount,
    ).sort((left, right) => left - right),
    outcome: stringValue(source.outcome).toLowerCase(),
    summary: stringValue(source.summary ?? source.label),
    finish_order: finishOrder,
    ...(hasGuandanSettlement ? {
      winning_team: ["even", "odd"].includes(source.winning_team) ? source.winning_team : "",
      losing_team: ["even", "odd"].includes(source.losing_team) ? source.losing_team : "",
      partner_position: validPositionOrNull(source.partner_position, seatCount),
      partner_finish: clampInteger(source.partner_finish, 0, 4, 0),
      upgrade_levels: clampInteger(source.upgrade_levels, 0, 3, 0),
      previous_level: normalizeGuandanLevel(source.previous_level) || "",
      next_level: normalizeGuandanLevel(source.next_level) || "",
      next_current_level: normalizeGuandanLevel(source.next_current_level) || "",
      team_levels_before: {
        even: normalizeGuandanLevel(levelsBefore.even) || "",
        odd: normalizeGuandanLevel(levelsBefore.odd) || "",
      },
      team_levels: {
        even: normalizeGuandanLevel(teamLevels.even) || "",
        odd: normalizeGuandanLevel(teamLevels.odd) || "",
      },
      a_challenge_team: ["even", "odd"].includes(source.a_challenge_team)
        ? source.a_challenge_team
        : "",
      a_challenge_attempted: source.a_challenge_attempted === true,
      a_challenge_succeeded: source.a_challenge_succeeded === true,
      passed_a: source.passed_a === true,
      final_result: normalizeGuandanFinalResult(source.final_result, seatCount),
    } : {}),
    scores: rawScores.length
      ? Array.from({ length: Math.min(seatCount, rawScores.length) }, (_, position) =>
        finiteNumberOrNull(rawScores[position]))
      : [],
    landlord_position: validPositionOrNull(
      source.landlord_position ?? source.landlord,
      seatCount,
    ),
    base_score: nonNegativeInteger(source.base_score ?? source.baseScore, 0),
    bombs: nonNegativeInteger(source.bombs, 0),
    rockets: nonNegativeInteger(source.rockets, 0),
    spring: source.spring === true,
    reverse_spring: source.reverse_spring === true || source.reverseSpring === true,
    doubles: normalizeSeatBooleanMap(source.doubles, seatCount),
    redouble: source.redouble === true,
    total_scores: rawTotalScores.length
      ? Array.from({ length: Math.min(seatCount, rawTotalScores.length) }, (_, position) =>
        finiteNumberOrNull(rawTotalScores[position]))
      : [],
    rest_cards: normalizeRestCards(source.rest_cards ?? source.restCards),
  };
}

function normalizeSeatNumberMap(rawMap, seatCount, minimum, maximum) {
  const source = objectValue(rawMap);
  return Array.from({ length: seatCount }, (_, position) => {
    const value = integerOrNull(source[position] ?? source[String(position)]);
    return value !== null && value >= minimum && value <= maximum ? value : null;
  });
}

function normalizeSeatBooleanMap(rawMap, seatCount) {
  const source = objectValue(rawMap);
  return Array.from({ length: seatCount }, (_, position) => {
    const value = source[position] ?? source[String(position)];
    return typeof value === "boolean" ? value : null;
  });
}

function normalizeSeatPositiveIntegerMap(rawMap, seatCount) {
  const source = objectValue(rawMap);
  return Array.from({ length: seatCount }, (_, position) => {
    const value = integerOrNull(source[position] ?? source[String(position)]);
    return value !== null && value > 0 && value <= 2 ** 31 - 1 ? value : null;
  });
}

function validPositionOrNull(value, seatCount) {
  const position = integerOrNull(value);
  return position !== null && position >= 0 && position < seatCount ? position : null;
}

function normalizePositionList(rawPositions, seatCount) {
  const positions = [];
  const seen = new Set();
  for (const rawPosition of arrayValue(rawPositions)) {
    const position = integerOrNull(rawPosition);
    if (position === null || position < 0 || position >= seatCount || seen.has(position)) continue;
    seen.add(position);
    positions.push(position);
  }
  return positions;
}

function normalizeRestCards(rawRestCards) {
  const rows = arrayValue(rawRestCards);
  const sparseRows = rows.some(
    (row) => Array.isArray(row) && integerOrNull(row[0]) !== null && Array.isArray(row[1]),
  );
  if (sparseRows) {
    return rows
      .map((row) => {
        if (!Array.isArray(row) || !Array.isArray(row[1])) return null;
        const position = integerOrNull(row[0]);
        if (position === null || position < 0) return null;
        return { position, cards: row[1].map(cardCode).filter(Boolean) };
      })
      .filter(Boolean);
  }
  return rows.map((cards, position) => ({
    position,
    cards: arrayValue(cards).map(cardCode).filter(Boolean),
  }));
}

function cardCode(value) {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    return stringValue(value.code ?? value.card_code ?? value.card);
  }
  return stringValue(value);
}

function objectValue(value) {
  return value && typeof value === "object" && !Array.isArray(value) ? value : {};
}

function arrayValue(value) {
  return Array.isArray(value) ? value : [];
}

function stringValue(value, fallback = "") {
  if (value === null || value === undefined) return fallback;
  if (!["string", "number", "boolean"].includes(typeof value)) return fallback;
  return String(value);
}

function scalarValue(value) {
  return ["string", "number", "boolean"].includes(typeof value) ? value : null;
}

function integerOrNull(value) {
  if (Number.isInteger(value)) return value;
  if (typeof value === "string" && /^-?\d+$/.test(value.trim())) return Number(value);
  return null;
}

function nonNegativeInteger(value, fallback) {
  const integer = integerOrNull(value);
  return integer !== null && integer >= 0 ? integer : fallback;
}

function positiveInteger(value, fallback) {
  const integer = integerOrNull(value);
  return integer !== null && integer > 0 ? integer : fallback;
}

function clampInteger(value, minimum, maximum, fallback) {
  const integer = integerOrNull(value);
  if (integer === null) return fallback;
  return Math.min(maximum, Math.max(minimum, integer));
}

function finiteNumberOrNull(value) {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}
