const SUITS = {
  S: { symbol: "♠", name: "Spades", red: false },
  H: { symbol: "♥", name: "Hearts", red: true },
  C: { symbol: "♣", name: "Clubs", red: false },
  D: { symbol: "♦", name: "Diamonds", red: true },
};

const RANKS = {
  T: "10",
  J: "J",
  Q: "Q",
  K: "K",
  A: "A",
  B: "JOKER",
  R: "JOKER",
  X: "JOKER",
  Y: "JOKER",
};

const GUANDAN_LEVELS = new Set(["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]);

const PHASES = {
  play: "Play",
  tribute: "Tribute",
  back: "Return Tribute",
  return_tribute: "Return Tribute",
  bidding: "Bidding",
  bid: "Bidding",
  doubling: "Doubling",
  deal: "Deal",
  settle: "Settlement",
  finished: "Finished",
  "\u51fa\u724c": "Play",
  "\u8fdb\u8d21": "Tribute",
  "\u8fd8\u8d21": "Return Tribute",
  "\u53eb\u5730\u4e3b": "Bidding",
  "\u52a0\u500d": "Doubling",
  "\u53d1\u724c": "Deal",
  "\u7ed3\u7b97": "Settlement",
  "\u7ed3\u675f": "Finished",
};

const ACTIONS = {
  Action: "Action",
  action: "Action",
  PASS: "PASS",
  pass: "PASS",
  Single: "Single",
  single: "Single",
  Pair: "Pair",
  pair: "Pair",
  Trips: "Triple",
  triplet: "Triple",
  ThreeWithTwo: "Triple with Pair",
  airplane: "Airplane",
  TwoTrips: "Two Triples",
  ThreePair: "Three Consecutive Pairs",
  Straight: "Straight",
  straight: "Straight",
  Bomb: "Bomb",
  bomb: "Bomb",
  rocket: "Rocket",
  StraightFlush: "Straight Flush",
  tribute: "Tribute",
  back: "Return Tribute",
  "\u5355\u5f20": "Single",
  "\u5bf9\u5b50": "Pair",
  "\u4e09\u5f20": "Triple",
  "\u4e09\u5e26\u4e8c": "Triple with Pair",
  "\u98de\u673a": "Airplane",
  "\u4e09\u987a\u5b50": "Two Triples",
  "\u4e09\u8fde\u5bf9": "Three Consecutive Pairs",
  "\u987a\u5b50": "Straight",
  "\u70b8\u5f39": "Bomb",
  "\u706b\u7bad": "Rocket",
  "\u540c\u82b1\u987a": "Straight Flush",
  "\u8fdb\u8d21": "Tribute",
  "\u8fd8\u8d21": "Return Tribute",
};

const GAME_ADAPTERS = {
  guandan: {
    id: "guandan",
    name: "GuanDan",
    shortName: "GD",
    seatCount: 4,
    defaultPolicies: ["DanKS", "RuleBot"],
    policyAliases: ["DanKS", "RuleBot"],
    roleForPosition(position, player) {
      if (player?.role && !/^seat_\d+$/.test(player.role)) {
        return roleDisplayName(player.role);
      }
      const team = teamDisplayName(player?.team);
      if (team) return team;
      return position % 2 === 0 ? "Even-seat Team" : "Odd-seat Team";
    },
    metricItems(metrics = {}) {
      return [
        ["Level", metrics.level ?? metrics.current_level],
        ["Trick", metrics.trick_index],
        ["Finish Order", Array.isArray(metrics.finish_order) ? metrics.finish_order.join(" → ") : metrics.finish_order],
      ].filter(([, value]) => value !== undefined && value !== null && value !== "");
    },
    seatLabel(position) {
      return `Seat ${position}`;
    },
  },
  doudizhu: {
    id: "doudizhu",
    name: "DouDizhu",
    shortName: "DDZ",
    seatCount: 3,
    defaultPolicies: ["DouKS", "RuleBot"],
    policyAliases: ["DouKS", "RuleBot"],
    roleForPosition(position, player, metrics = {}) {
      const roleLabels = {
        landlord: "Landlord",
        landlord_down: "Downstream Farmer",
        landlord_up: "Upstream Farmer",
        farmer: "Farmer",
      };
      if (player?.role) return roleLabels[player.role] || roleDisplayName(player.role);
      const landlord = Number(metrics.landlord_position);
      return position === landlord ? "Landlord" : "Farmer";
    },
    metricItems(metrics = {}) {
      return [
        ["Multiplier", metrics.multiplier],
        ["Bombs", metrics.bombs],
        ["ADP", metrics.adp],
      ].filter(([, value]) => value !== undefined && value !== null && value !== "");
    },
    seatLabel(position) {
      return position === 0 ? "Primary View" : `Seat ${position}`;
    },
  },
};

export function getGameAdapter(game) {
  return GAME_ADAPTERS[game] || GAME_ADAPTERS.guandan;
}

export function normalizeGuandanLevel(value) {
  const raw = String(value ?? "").trim().toUpperCase();
  const canonical = raw === "10" ? "T" : raw;
  return GUANDAN_LEVELS.has(canonical) ? canonical : "";
}

export function policyDisplayName(value) {
  const fields =
    value && typeof value === "object"
      ? [value.policy_id, value.id, value.display_name, value.name, value.version]
      : [value];
  const search = fields.filter(Boolean).join(" ");
  if (/danks|danzero/iu.test(search)) return "DanKS";
  if (/douks|(?:^|[_\s-])dmc(?:$|[_\s-])|role[_\s-]?aware/iu.test(search)) return "DouKS";
  if (/rulebot|rule[_\s-]?agent|rlcard/iu.test(search)) return "RuleBot";
  const safeField = fields.find((field) => field && !hasHan(field));
  return String(safeField ?? (fields.some(Boolean) ? "Policy" : ""));
}

export function gameName(game) {
  return getGameAdapter(game).name;
}

export function phaseName(phase) {
  if (!phase) return "—";
  const value = PHASES[phase] || PHASES[String(phase).toLowerCase()] || String(phase);
  return hasHan(value) ? "Phase" : value;
}

export function cardParts(value) {
  const code = String(value ?? "").trim();
  if (!code) return { code: "", rank: "?", suit: "", suitName: "", red: false, joker: false };

  const upper = code.toUpperCase();
  if (["SB", "BJ", "BLACKJOKER", "B", "X", "\u5c0f\u738b"].includes(upper)) {
    return {
      code,
      rank: "JOKER",
      suit: "",
      suitName: "Small Joker",
      red: false,
      joker: true,
      jokerVariant: "small",
    };
  }
  if (["HR", "RJ", "REDJOKER", "R", "Y", "\u5927\u738b"].includes(upper)) {
    return {
      code,
      rank: "JOKER",
      suit: "",
      suitName: "Big Joker",
      red: true,
      joker: true,
      jokerVariant: "big",
    };
  }

  const suitInfo = SUITS[upper[0]];
  if (suitInfo && upper.length > 1) {
    const rawRank = upper.slice(1);
    return {
      code,
      rank: RANKS[rawRank] || rawRank,
      suit: suitInfo.symbol,
      suitName: suitInfo.name,
      red: suitInfo.red,
      joker: false,
    };
  }

  return {
    code,
    rank: RANKS[upper] || upper,
    suit: "",
    suitName: "",
    red: ["R", "Y"].includes(upper),
    joker: ["B", "R", "X", "Y"].includes(upper),
  };
}

export function extractAction(value) {
  if (!value) {
    return {
      type: "None",
      label: "",
      cards: [],
      pass: false,
      empty: true,
      index: null,
      raw: value,
    };
  }

  if (Array.isArray(value)) {
    const type = String(value[0] ?? "None");
    const cards = Array.isArray(value[2]) ? value[2] : [];
    const pass = type.toUpperCase() === "PASS";
    const empty =
      !pass && cards.length === 0 && ["", "NONE", "NULL"].includes(type.trim().toUpperCase());
    return {
      type,
      label: empty ? "" : actionLabel(type, value[1], cards),
      cards,
      pass,
      empty,
      index: null,
      raw: value,
    };
  }

  if (typeof value === "string") {
    const pass = ["PASS", "\u8fc7", "\u4e0d\u51fa"].includes(value.toUpperCase());
    const empty = !pass && ["", "NONE", "NULL"].includes(value.trim().toUpperCase());
    return {
      type: pass ? "PASS" : value,
      label: empty ? "" : pass ? "PASS" : hasHan(value) ? "Action" : value,
      cards: [],
      pass,
      empty,
      index: null,
      raw: value,
    };
  }

  const nested = value.action && value.action !== value ? extractAction(value.action) : null;
  const wire = Array.isArray(value.wire) ? extractAction(value.wire) : null;
  const explicitType = value.type ?? value.kind ?? nested?.type ?? wire?.type;
  const type = explicitType ?? (value.pass ? "PASS" : "action");
  const cards = arrayValue(value.cards ?? value.card_list ?? nested?.cards ?? wire?.cards);
  const pass = Boolean(value.pass) || String(type).toUpperCase() === "PASS";
  const providedLabel = value.label ?? value.name;
  const displayLabel = hasHan(providedLabel) ? null : providedLabel;
  const empty =
    !pass &&
    cards.length === 0 &&
    (explicitType === undefined ||
      explicitType === null ||
      ["", "NONE", "NULL"].includes(String(explicitType).trim().toUpperCase())) &&
    !String(providedLabel ?? "").trim();
  const labelLooksStructural =
    displayLabel !== undefined &&
    displayLabel !== null &&
    [String(type).toLowerCase(), "pass"].includes(String(displayLabel).toLowerCase());
  const label =
    empty
      ? ""
      : pass
        ? "PASS"
        : (labelLooksStructural ? null : displayLabel) ??
          (value.rank || !wire ? actionLabel(type, value.rank, cards) : wire.label) ??
          nested?.label ??
          wire?.label;
  return {
    type: String(type),
    label: String(label || (empty ? "" : "Action")),
    cards,
    pass,
    empty,
    index: integerOrNull(value.action_index ?? value.index ?? nested?.index ?? wire?.index),
    score: numberOrNull(value.score ?? value.value ?? nested?.score ?? wire?.score),
    raw: value,
  };
}

export function actionLabel(type, rank, cards = []) {
  const normalized = String(type || "action");
  if (normalized.toUpperCase() === "PASS") return "PASS";
  const candidate = ACTIONS[normalized] || ACTIONS[normalized.toLowerCase()] || normalized;
  const base = hasHan(candidate) ? "Action" : candidate;
  const rankValue = RANKS[String(rank ?? "").toUpperCase()] || rank;
  const rankLabel = rankValue && rankValue !== "PASS" && !hasHan(rankValue) ? ` ${rankValue}` : "";
  const count = cards.length ? ` · ${cards.length} ${cards.length === 1 ? "Card" : "Cards"}` : "";
  return `${base}${rankLabel}${count}`;
}

export function describeAction(action, actorName = "Player") {
  const normalized = extractAction(action);
  return normalized.pass ? `${actorName} PASS` : `${actorName} played ${normalized.label}`;
}

export function resultLabel(result) {
  if (result === undefined || result === null || result === "") return "Pending";
  if (typeof result === "string") {
    const side = teamDisplayName(result);
    if (side) return `${side} Wins`;
    const policy = policyDisplayName(result);
    return policy && policy !== "Policy" ? `${policy} Wins` : hasHan(result) ? "Settled" : result;
  }
  if (typeof result === "number") return `Seat ${result} Wins`;
  if (typeof result === "object" && !Object.keys(result).length) return "Pending";
  const winningSide = teamDisplayName(result.winning_side ?? result.winner_team);
  const winnerPosition = integerOrNull(result.winner_position ?? result.winner);
  const winnerName = policyDisplayName(
    result.winner_name ?? (typeof result.winner === "string" ? result.winner : ""),
  );
  const explicitLabel = firstEnglish(result.label, result.summary);
  return (
    explicitLabel ||
    (winnerName && winnerName !== "Policy" ? `${winnerName} Wins` : "") ||
    (winningSide ? `${winningSide} Wins` : "") ||
    (winnerPosition !== null ? `Seat ${winnerPosition} Wins` : "Settled")
  );
}

export function statusInfo(status) {
  const key = String(status || "unknown").toLowerCase();
  const table = {
    queued: ["QUEUED", "queued"],
    starting: ["STARTING", "running"],
    running: ["LIVE", "running"],
    completed: ["COMPLETED", "completed"],
    finished: ["COMPLETED", "completed"],
    failed: ["FAILED", "failed"],
    error: ["FAILED", "failed"],
    interrupted: ["INTERRUPTED", "interrupted"],
    cancelled: ["CANCELLED", "interrupted"],
    legacy: ["LEGACY REPLAY", "legacy"],
    unknown: ["UNKNOWN", "idle"],
  };
  const [label, tone] = table[key] || [hasHan(status) ? "UNKNOWN" : String(status), "idle"];
  return { key, label, tone, terminal: ["completed", "finished", "failed", "error", "interrupted", "cancelled"].includes(key) };
}

function teamDisplayName(value) {
  const raw = String(value ?? "").trim();
  const key = raw.toLowerCase().replace(/[\s-]+/g, "_");
  const labels = {
    even: "Even-seat Team",
    even_team: "Even-seat Team",
    even_seat_team: "Even-seat Team",
    odd: "Odd-seat Team",
    odd_team: "Odd-seat Team",
    odd_seat_team: "Odd-seat Team",
    a: "Team A",
    a_team: "Team A",
    team_a: "Team A",
    method_a: "Method A",
    b: "Team B",
    b_team: "Team B",
    team_b: "Team B",
    method_b: "Method B",
    landlord: "Landlord",
    landlord_side: "Landlord Side",
    farmers: "Farmers",
    farmer: "Farmers",
    farmer_side: "Farmer Side",
    "\u5076\u6570\u4f4d\u961f": "Even-seat Team",
    "\u5947\u6570\u4f4d\u961f": "Odd-seat Team",
    "a_\u961f": "Team A",
    "a_\u65b9\u6cd5": "Method A",
    "b_\u961f": "Team B",
    "b_\u65b9\u6cd5": "Method B",
    "\u5730\u4e3b": "Landlord",
    "\u5730\u4e3b\u65b9": "Landlord Side",
    "\u519c\u6c11": "Farmers",
    "\u519c\u6c11\u65b9": "Farmer Side",
  };
  return labels[key] || "";
}

function roleDisplayName(value) {
  const raw = String(value ?? "").trim();
  const key = raw.toLowerCase().replace(/[\s-]+/g, "_");
  const labels = {
    landlord: "Landlord",
    landlord_down: "Downstream Farmer",
    landlord_up: "Upstream Farmer",
    farmer: "Farmer",
    "\u5730\u4e3b": "Landlord",
    "\u4e0b\u5bb6\u519c\u6c11": "Downstream Farmer",
    "\u4e0a\u5bb6\u519c\u6c11": "Upstream Farmer",
    "\u519c\u6c11": "Farmer",
    "\u5076\u6570\u4f4d\u961f": "Even-seat Team",
    "\u5947\u6570\u4f4d\u961f": "Odd-seat Team",
  };
  if (labels[key]) return labels[key];
  if (!raw) return "";
  if (hasHan(raw)) return "Player Role";
  return raw
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function firstEnglish(...values) {
  return values.map((value) => String(value ?? "").trim()).find((value) => value && !hasHan(value)) ?? "";
}

function hasHan(value) {
  return /[\u3400-\u9fff]/u.test(String(value ?? ""));
}

function arrayValue(value) {
  if (Array.isArray(value)) return value;
  if (value === undefined || value === null || value === "") return [];
  return [value];
}

function integerOrNull(value) {
  if (Number.isInteger(value)) return value;
  if (typeof value === "string" && /^-?\d+$/.test(value)) return Number(value);
  return null;
}

function numberOrNull(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}
