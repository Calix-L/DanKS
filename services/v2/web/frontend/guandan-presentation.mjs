import { isLobbyPhase, isSpectatorRole } from "./room-state.mjs?v=20261005-r1";

// Display only. Type/rank interpretation and legality belong to the referee.
const TYPE_LABELS = {
  single: "单张", pair: "对子", trips: "三张", triple: "三张", triplet: "三张",
  threewithtwo: "三带二", fullhouse: "三带二", threepair: "三连对",
  twotrips: "钢板", straight: "顺子", straightflush: "同花顺",
  bomb: "炸弹", jokerbomb: "天王炸", kingbomb: "天王炸", fourkings: "天王炸", rocket: "天王炸", pass: "不出",
};

export function guandanTypeLabel(action) {
  const type = String(action?.type ?? "").toLowerCase().replace(/[\s_-]/gu, "");
  return TYPE_LABELS[type] ?? String(action?.label || action?.type || "");
}

export function guandanRankLabel(action) {
  if (guandanTypeLabel(action) === "天王炸") return "";
  const rank = String(action?.rank ?? "").toUpperCase();
  return ({ T: "10", B: "小王", R: "大王" })[rank] ?? rank;
}

export function wildcardCaption(action, level) {
  const rank = String(action?.rank ?? "").toUpperCase().replace(/^T$/u, "10");
  const levelRank = String(level ?? "").toUpperCase().replace(/^10$/u, "T");
  if (!rank || !/^[2-9TJQKA]$/u.test(levelRank) || ["单张", "不出", ""].includes(guandanTypeLabel(action))) return "";
  const hasHeartLevel = (action?.cards ?? []).some(code => String(code).toUpperCase().replace(/10$/u, "T") === `H${levelRank}`);
  // This does not say which missing card a wildcard replaced. Some actions
  // allow multiple assignments; only the chosen referee type/rank is public.
  return hasHeartLevel ? `含红心${levelRank.replace(/^T$/u, "10")} · 按${guandanTypeLabel(action)}${rank}判定` : "";
}

export function confirmedTableFeedback(previous, next) {
  const none = { playSeats: [], ownTurn: false };
  if (!previous || !next || previous.table_no !== next.table_no || previous.round_no !== next.round_no
    || previous.viewer?.seat !== next.viewer?.seat || previous.viewer?.role !== next.viewer?.role
    || next.version <= previous.version || next.phase !== "play"
    || !previous.decision?.id || !next.decision?.id || previous.decision.id === next.decision.id) return none;
  const signature = area => JSON.stringify([Boolean(area?.pass), area?.type ?? "", area?.rank ?? "", area?.cards ?? []]);
  const playSeats = next.seats.filter(seat => {
    if (!(seat.play_area?.cards?.length || seat.play_area?.pass)) return false;
    const before = previous.seats.find(item => item.position === seat.position);
    return before && signature(before.play_area) !== signature(seat.play_area);
  }).map(seat => seat.position);
  const result = { playSeats, ownTurn: next.viewer?.role === "player" && next.decision.phase === "play"
    && next.decision.actor === next.viewer.seat && previous.decision.actor !== next.viewer.seat };
  const oldTrick = previous.trick_history?.available && previous.trick_history.current;
  const completed = next.trick_history?.available && next.trick_history.previous;
  const newTrick = next.trick_history?.current;
  // A room version also changes for metadata. An exact public-history prefix
  // plus one committed action proves the boundary was not skipped on reconnect.
  if (previous.current_trick?.mode === "follow" && next.current_trick?.mode === "lead"
    && oldTrick?.complete && completed?.complete && newTrick?.complete
    && completed.number === oldTrick.number && newTrick.number === oldTrick.number + 1
    && newTrick.events.length === 0 && completed.events.length === oldTrick.events.length + 1
    && JSON.stringify(completed.events.slice(0, -1)) === JSON.stringify(oldTrick.events)) result.trickCleared = true;
  return result;
}

export function currentPublicPlayArea(view, seat) {
  if (!view.trick_history?.available || !view.trick_history.current) return seat.play_area;
  const event = [...view.trick_history.current.events].reverse().find(item => item.seat === seat.position);
  if (event) return { cards: event.cards, pass: event.pass, type: event.type, rank: event.rank, label: "" };
  // A partial initial observation can still have an explicit public target.
  const target = view.current_trick?.target;
  if (target?.seat === seat.position) return { ...target, pass: false, label: "" };
  return { cards: [], pass: false, type: "", rank: "", label: "" };
}
// Display-only projection. Readiness and dealing remain server-owned.
export function soloLobbyPresentation(view, { pending = false, frozen = false, error = "" } = {}) {
  const ownSeat = view?.seats?.find(seat => seat.position === view.viewer?.seat);
  const visible = isLobbyPhase(view?.phase) && !isSpectatorRole(view?.viewer?.role) && ownSeat?.kind === "human";
  const others = (view?.seats ?? []).filter(seat => seat.position !== view?.viewer?.seat);
  const aiReady = others.length === 3 && others.every(seat => seat.kind === "bot" && seat.ready === true);
  const ready = Boolean(ownSeat?.ready);
  return {
    visible, ready, busy: Boolean(pending), disabled: !visible || pending || frozen,
    button: pending ? ready ? "正在取消…" : "正在准备…"
      : ready ? "已准备 · 点击取消" : aiReady ? "开始对局" : "准备",
    status: frozen ? "连接恢复后即可准备。" : pending ? "正在确认准备状态…"
      : error || (ready ? "已准备，等待发牌。" : aiReady ? "3 位 AI 已准备，点击开始发牌。" : "等待其他玩家准备，你可以先准备。"),
  };
}
