/** Public-history boundary. Never infer events from hands or legal actions. */
export const MAX_TRICK_EVENTS = 128;
const CARD_CODES = new Set(Array.from("SHCD").flatMap((suit) => Array.from("23456789TJQKA", (rank) => `${suit}${rank}`)).concat(["SB", "HR"]));
const TYPES = new Set(["Single", "Pair", "Trips", "ThreeWithTwo", "Straight", "ThreePair", "TwoTrips", "StraightFlush", "Bomb", "FourKings"]);
const RANKS = new Set([..."23456789TJQKABR", "10"]);
const object = (value) => value && typeof value === "object" && !Array.isArray(value) ? value : {};

export function normalizeTrickHistory(raw, seatCount = 4) {
  const source = object(raw);
  if (source.available !== true || !Number.isInteger(seatCount) || seatCount < 1 || seatCount > 4) {
    return { available: false, current: null, previous: null };
  }
  return { available: true, current: normalizeTrick(source.current, seatCount), previous: normalizeTrick(source.previous, seatCount) };
}

function normalizeTrick(raw, seatCount) {
  const source = object(raw);
  if (!Number.isSafeInteger(source.number) || source.number < 1 || !Array.isArray(source.events)) return null;
  const events = [];
  let complete = source.complete === true && source.events.length <= MAX_TRICK_EVENTS;
  for (const rawEvent of source.events.slice(-MAX_TRICK_EVENTS)) {
    const event = normalizeEvent(rawEvent, seatCount);
    if (event) events.push(event);
    else complete = false;
  }
  return { number: source.number, complete, events };
}

function normalizeEvent(raw, seatCount) {
  const source = object(raw);
  if (!Number.isInteger(source.seat) || source.seat < 0 || source.seat >= seatCount || !Array.isArray(source.cards)) return null;
  if (source.pass === true && source.type === "PASS" && source.rank === "PASS" && source.cards.length === 0) {
    return { seat: source.seat, cards: [], pass: true, type: "PASS", rank: "PASS" };
  }
  if (source.pass !== false || !TYPES.has(source.type) || !RANKS.has(source.rank) || source.cards.length < 1 || source.cards.length > 27 || source.cards.some((card) => typeof card !== "string" || !CARD_CODES.has(card))) return null;
  return { seat: source.seat, cards: [...source.cards], pass: false, type: source.type, rank: source.rank };
}
