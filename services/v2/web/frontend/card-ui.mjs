import { cardParts, normalizeGuandanLevel } from "./adapters.mjs?v=20260801-30";

// Exact card bounds in the supplied sheets: the white gutters are not cells.
const CARD_SHEETS = {
  "♣": { name: "clubs", atlasWidth: 3789, atlasHeight: 2915, width: 649, height: 907, xs: [0, 782, 1564, 2346, 3128], ys: [0, 1004, 2008] },
  "♠": { name: "spades", atlasWidth: 3784, atlasHeight: 2914, width: 648, height: 906, xs: [0, 782, 1564, 2346, 3128], ys: [0, 1004, 2008] },
  "♥": { name: "hearts", atlasWidth: 3775, atlasHeight: 2913, width: 648, height: 906, xs: [0, 781, 1563, 2345, 3127], ys: [0, 1003, 2007] },
  "♦": { name: "diamonds", atlasWidth: 3775, atlasHeight: 2913, width: 648, height: 906, xs: [0, 781, 1563, 2345, 3127], ys: [0, 1003, 2007] },
};
const FACE_RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "A", "J", "Q", "K"];

function sheetFrame(sheet, index) {
  const column = index % 5;
  const row = Math.floor(index / 5);
  return {
    src: new URL(`../assets/cards/${sheet.name}-user-v1.jpg`, import.meta.url).href,
    atlasWidth: sheet.atlasWidth, atlasHeight: sheet.atlasHeight,
    width: sheet.width, height: sheet.height,
    x: sheet.xs[column], y: sheet.ys[row], column, row,
  };
}

export function playingCardArtwork(value) {
  const parts = cardParts(value);
  if (parts.joker) return sheetFrame(CARD_SHEETS[parts.jokerVariant === "big" ? "♥" : "♠"], 13);
  const index = FACE_RANKS.indexOf(parts.rank);
  const sheet = CARD_SHEETS[parts.suit];
  return sheet && index >= 0 ? sheetFrame(sheet, index) : null;
}

export function cardBackArtwork() {
  return sheetFrame(CARD_SHEETS["♠"], 14);
}

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
  { game = "", levelRank = "", documentRef = globalThis.document,
    showJokerLabel = false, jokerLabels = { big: "BIG", small: "SMALL" }, jokerArtwork = "" } = {},
) {
  if (!documentRef?.createElement) throw new TypeError("a DOM document is required to render a card");
  const descriptor = playingCardDescriptor(value, { game, levelRank });
  const { parts } = descriptor;
  const card = documentRef.createElement("span");
  card.className = ["playing-card", size, ...descriptor.classes].join(" ");
  card.dataset.rank = parts.rank;
  card.dataset.game = game;
  card.title = descriptor.title;
  if (parts.joker && showJokerLabel) card.className += " joker-labelled";

  const corner = documentRef.createElement("span");
  corner.className = "card-corner";
  const rank = documentRef.createElement("b");
  rank.textContent = parts.joker && showJokerLabel
    ? jokerLabels[parts.jokerVariant === "big" ? "big" : "small"]
    : parts.rank;
  corner.appendChild(rank);
  card.appendChild(corner);

  if (parts.joker && jokerArtwork) {
    const image = documentRef.createElement("img");
    image.className = "joker-artwork";
    image.src = jokerArtwork;
    image.alt = "";
    image.draggable = false;
    image.setAttribute("aria-hidden", "true");
    card.appendChild(image);
  }

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
