import { cardParts, normalizeGuandanLevel } from "./adapters.mjs?v=20260801-30";

/**
 * Build the shared visual description used by both spectator and human-table
 * cards. Keeping the tactical GuanDan marks here prevents the two surfaces
 * from drifting apart again.
 */
export function playingCardDescriptor(value, { game = "", levelRank = "" } = {}) {
  const parts = cardParts(value);
  const canonicalLevel = normalizeGuandanLevel(levelRank);
  const canonicalCardRank = parts.rank === "10" ? "T" : parts.rank;
  const isLevelCard =
    game === "guandan" && Boolean(canonicalLevel) && !parts.joker && canonicalCardRank === canonicalLevel;
  const isWildLevelCard = isLevelCard && parts.suit === "♥";
  const classes = [];
  if (parts.red) classes.push("red");
  if (parts.joker) {
    classes.push("joker", `joker-${parts.jokerVariant === "big" ? "big" : "small"}`);
  }
  if (isLevelCard) classes.push("level-card");
  if (isWildLevelCard) classes.push("wild-level-card");

  const baseTitle = parts.joker
    ? parts.suitName
    : parts.suitName
      ? `${parts.suitName} ${parts.rank}`
      : parts.rank;

  return {
    parts,
    classes,
    isLevelCard,
    isWildLevelCard,
    title: isWildLevelCard
      ? `${baseTitle} · Wild heart-level card`
      : isLevelCard
        ? `${baseTitle} · Level card`
        : baseTitle,
  };
}

export function createPlayingCard(
  value,
  size = "hand",
  { game = "", levelRank = "", documentRef = globalThis.document } = {},
) {
  if (!documentRef?.createElement) throw new TypeError("a DOM document is required to render a card");
  const descriptor = playingCardDescriptor(value, { game, levelRank });
  const { parts } = descriptor;
  const card = documentRef.createElement("span");
  card.className = ["playing-card", size, ...descriptor.classes].join(" ");
  card.dataset.rank = parts.rank;
  card.dataset.game = game;
  card.title = descriptor.title;

  const corner = documentRef.createElement("span");
  corner.className = "card-corner";
  const rank = documentRef.createElement("b");
  rank.textContent = parts.rank;
  corner.appendChild(rank);
  card.appendChild(corner);

  if (!parts.joker) {
    const suit = documentRef.createElement("i");
    suit.textContent = parts.suit;
    corner.appendChild(suit);
    const pip = documentRef.createElement("em");
    pip.textContent = parts.suit;
    card.appendChild(pip);
  }

  if (descriptor.isLevelCard) {
    const mark = documentRef.createElement("span");
    mark.className = `card-level-mark${descriptor.isWildLevelCard ? " wild" : ""}`;
    mark.textContent = descriptor.isWildLevelCard ? "W" : "L";
    mark.setAttribute("aria-hidden", "true");
    card.appendChild(mark);
  }

  return card;
}
