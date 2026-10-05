import { cardParts } from "./adapters.mjs?v=20260805-1";
import { createPlayingCard } from "./card-ui.mjs?v=20260804-2";

function asArray(value) {
  return Array.isArray(value) ? value : [];
}

function englishOnly(value, fallback = "") {
  const text = String(value ?? "");
  return /[\u3400-\u9fff]/u.test(text) ? fallback : text;
}

/** Keep replay and live spectator hands on the exact same grouping contract. */
export function normalizedLayout(player) {
  const source = player?.handLayout ?? player?.hand_layout;
  const groups = asArray(source?.groups ?? source).map((group, index) => {
    if (Array.isArray(group)) {
      return { pattern: "", minor: null, value: null, cards: group, index };
    }
    return {
      pattern: String(group?.pattern ?? group?.type ?? ""),
      minor: group?.minor ?? null,
      value: group?.value ?? null,
      cards: asArray(group?.cards ?? group?.card_list),
      index,
    };
  });
  if (!groups.length && asArray(player?.hand).length) {
    return {
      version: "flat-fallback",
      mode: "rank-groups",
      groups: fallbackHandGroups(player.hand),
    };
  }
  return {
    version: source?.version ?? "",
    mode: source?.mode ?? "",
    groups,
  };
}

export function fallbackHandGroups(cards) {
  const orderedGroups = [];
  const chunksByRank = new Map();
  asArray(cards).forEach((card) => {
    const code = typeof card === "object" ? card?.code : card;
    const rank = cardParts(code).rank || String(code);
    let chunks = chunksByRank.get(rank);
    if (!chunks) {
      chunks = [];
      chunksByRank.set(rank, chunks);
    }
    let group = chunks[chunks.length - 1];
    if (!group || group.cards.length >= 4) {
      group = {
        pattern: "rank-group",
        minor: null,
        value: rank,
        cards: [],
        index: orderedGroups.length,
      };
      chunks.push(group);
      orderedGroups.push(group);
    }
    group.cards.push(code);
  });
  return orderedGroups;
}

export function renderHandLayout(container, layout, cardContext = {}) {
  const documentRef = cardContext.documentRef ?? globalThis.document;
  if (!documentRef?.createElement) throw new TypeError("a DOM document is required to render a hand");
  container.replaceChildren();
  container.classList.remove("unknown-layout");
  container.dataset.mode = layout.mode || "";
  const groups = asArray(layout.groups);
  const cardCount = groups.reduce((total, group) => total + asArray(group.cards).length, 0);
  container.dataset.groups = String(groups.length);
  container.dataset.cards = String(cardCount);
  container.classList.toggle("dense", groups.length > 12 || cardCount > 24);
  container.classList.toggle("very-dense", groups.length > 18 || cardCount > 34);
  groups.forEach((group) => {
    const stack = documentRef.createElement("div");
    stack.className = "meld-stack";
    stack.dataset.pattern = group.pattern;
    stack.dataset.size = String(asArray(group.cards).length);
    stack.title = [
      englishOnly(group.pattern, "Card group"),
      englishOnly(group.value, ""),
    ].filter((value) => value !== "" && value !== null).join(" · ");
    asArray(group.cards).forEach((card) => {
      const code = typeof card === "object" ? card?.code : card;
      stack.appendChild(createPlayingCard(code, "hand", cardContext));
    });
    container.appendChild(stack);
  });
}
