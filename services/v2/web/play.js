import { createPlayingCard } from "./frontend/card-ui.mjs?v=20261004-7";
import {
  fallbackHandGroups,
  renderHandLayout,
} from "./frontend/hand-layout.mjs?v=20261004-7";
import { resolveArenaBasePath } from "./frontend/base-path.mjs?v=20260805-1";
import { reviewHref } from "./frontend/navigation.mjs?v=20260810-1";
import { installSoloLocale } from "./frontend/solo-english.mjs?v=20261005-r1";
import { selectionFeedback, selectionExplanation, pendingActionResolved, toggleGroupSelection, groupSelectionFeedback, stableHandGroups, canContinueHandGesture, canPrepareOwnHand, reconcileHandSelection, beginHandSweep, paintHandSweep, rankDisplayGroups, boundedHandColumns, rebindDisplayGroups, fitHandStackReveal, handInspectionSignature, handRefreshBindings, actionChoiceContext, isCurrentActionChoice } from "./frontend/hand-interactions.mjs?v=20261005-r1";
import { guandanTypeLabel, guandanRankLabel, wildcardCaption, confirmedTableFeedback, currentPublicPlayArea, soloLobbyPresentation } from "./frontend/guandan-presentation.mjs?v=20261005-r1";
import { createTableSound } from "./frontend/table-sound.mjs?v=20261004-3";
import { TABLE_VIEWPORT as HUMAN_TABLE_VIEWPORT, tableFrameMetrics, loadHandSpread, saveHandSpread, tableIsFullscreen, toggleTableFullscreen, loadControlPreferences, saveControlPreferences, effectiveControlSize, tableFlightVector } from "./frontend/table-frame.mjs?v=20261005-1";
import {
  findRecentSoloTable,
  forgetRecentSoloTable,
  loadRecentSoloTables,
  rememberRecentSoloTable,
  scopeSoloStorage,
  resilientStorage,
} from "./frontend/solo-session.mjs?v=20261005-4";
import {
  HumanTableApi,
  clearRoomToken,
  loadRoomToken,
  saveRoomToken,
} from "./frontend/room-api.mjs?v=20261005-r1";
import {
  actionChoiceLabel,
  fitHandGroupStep,
  guandanRoundPresentation,
  isLobbyPhase,
  isOwnDecision,
  isOwnPlayDecision,
  isReadyPhase,
  isSpectatorRole,
  isTerminalPhase,
  matchingActionsForSelection,
  normalizeRoomView,
  resultWinningPositions,
  selectedCardCodes,
  seatVisualClass,
  toggleSelectionEntries,
} from "./frontend/room-state.mjs?v=20261005-r1";
import {
  decisionClock,
  renderCountdownTick,
} from "./frontend/human-turn-visuals.mjs?v=20260804-1";

const api = new HumanTableApi(resolveArenaBasePath());
const soloLocalStorage = playerStorage("localStorage");
const soloSessionStorage = playerStorage("sessionStorage");
function playerStorage(kind) {
  let storage = null;
  try { storage = scopeSoloStorage(window[kind], resolveArenaBasePath()); } catch { /* Use page memory. */ }
  return resilientStorage(storage, { onUnavailable: () => {
    document.querySelectorAll("[data-storage-notice]").forEach(node => { node.hidden = false; });
  } });
}
const tableSound = createTableSound(window);
const $ = (id) => document.getElementById(id);
const SOLO_MODE = document.body.dataset.arenaMode === "solo";
const HUMAN_CARD_VISUALS = {
  showJokerLabel: true,
  jokerLabels: { big: "大王", small: "小王" },
  jokerArtwork: new URL("./assets/cards/joker-jester-v1.webp", import.meta.url).href,
};
const HUMAN_AI_AVATARS = {
  1: new URL("./assets/avatars/avatar-1.webp", import.meta.url).href,
  2: new URL("./assets/avatars/avatar-2.webp", import.meta.url).href,
  3: new URL("./assets/avatars/avatar-3.webp", import.meta.url).href,
};
const SOLO_USER_ID_KEY = "cardks.solo.user_id.v1";
const state = {
  tableNo: "",
  token: "",
  view: null,
  lobbyError: "",
  selectedIds: new Set(),
  clock: null,
  closeSocket: null,
  socketGeneration: 0,
  reconnectTimer: null,
  reconnectAttempt: 0,
  pendingAction: null,
  actionError: null,
  handSpread: false,
  controlPreferences: { size: "auto", side: "right" },
  historyRenderSignature: "",
  trickCueTimer: null,
  trickMotionTimer: null,
  pendingMutation: false,
  retryError: "",
  pendingArrange: false,
  connectionState: "idle",
  handLayoutObserver: null,
  handLayoutFrame: null,
  selectionGesture: null,
  suppressSelectionClick: false,
  suppressSelectionClickTimer: null,
  playAnimationSeats: new Set(),
  recentRememberSignature: "",
  displayGroups: [],
  displayContext: "",
  expandedHandSignature: null,
  actionChoices: null,
  controlsTrigger: "tableControlsButton",
  rulesTrigger: "rulesButton",
  historyTrigger: "trickHistoryButton",
  resetHandLayout: false,
  swipeSelection: false,
  cancelledSelectionPointerId: null,
  handDisplayMode: "rank",
  turnCueTimer: null,
};

const PHASE_LABELS = {
  lobby: "大厅等待",
  waiting: "大厅等待",
  ready: "等待准备",
  deal: "正在发牌",
  starting: "正在发牌",
  paused: "AI 暂时不可用",
  error: "牌局无法继续",
  play: "出牌阶段",
  tribute: "进贡阶段",
  back: "还贡阶段",
  return_tribute: "还贡阶段",
  bidding: "叫地主",
  bid: "叫地主",
  doubling: "加倍阶段",
  double: "农民加倍",
  redouble: "地主再加倍",
  settle: "结算中",
  finished: "对局结束",
  completed: "对局结束",
  game_over: "对局结束",
  gameover: "对局结束",
};

const SLOT_LABELS = {
  guandan: ["单人场", "双人场", "三人场"],
};
const ARRANGE_MODE_LABELS = {
  1: "炸弹优先",
  2: "同花顺优先",
  3: "整牌优先",
  4: "简单理牌",
};
// The table is authored in one stable 16:9 coordinate space.  The browser
// viewport (including browser zoom) only changes this outer scale; it never
// changes the relative seat/play/hand coordinates inside the table.

initialize().catch(error => {
  window.danksPageBoot?.fail();
  console.error("DanKS initialization failed", error);
});

async function initialize() {
  document.body.classList.add("human-fixed-viewport");
  if (SOLO_MODE) initializeSoloMode();
  initializeTablePresentation();
  updateHumanTableViewportScale();
  bindEntryControls();
  bindRoomControls();
  initializeOwnHandResizeObserver();
  if (!SOLO_MODE) renderHumanSlotOptions("guandan");
  window.setInterval(updateCountdownDisplay, 200);
  window.addEventListener("resize", updateHumanTableViewportScale);
  window.visualViewport?.addEventListener("resize", handleVisualViewportResize, { passive: true });
  window.addEventListener("beforeunload", destroyRoomRuntime);
  window.addEventListener("pagehide", event => { if (!event.persisted) tableSound.close(); });
  window.addEventListener("pointerdown", () => tableSound.arm(), { passive: true });
  window.addEventListener("keydown", () => tableSound.arm());
  await restoreFromSession();
  window.danksPageBoot?.ready();
}

function initializeTablePresentation() {
  // Accessing localStorage itself may throw in restricted browser contexts.
  let storage;
  try { storage = soloLocalStorage; } catch { storage = null; }
  state.handSpread = loadHandSpread(storage);
  state.controlPreferences = loadControlPreferences(storage);
  renderHandSpreadControl();
  renderControlPreferences();
  $("expandHandButton").addEventListener("click", openExpandedHand);
  $("expandedHandDialog").addEventListener("close", () => {
    state.expandedHandSignature = null;
    const trigger = $("expandHandButton");
    if (!trigger.disabled && !$("ownHandPanel").hidden && trigger.getClientRects().length) trigger.focus({ preventScroll: true });
  });
  $("expandedHandGroups").addEventListener("click", event => {
    const signature = currentHandInspectionSignature();
    if (!signature || signature !== state.expandedHandSignature) {
      closeExpandedHand();
      return;
    }
    const cardButton = event.target.closest(".expanded-hand-card");
    if (cardButton && !cardButton.disabled) {
      toggleCardSelection(cardButton.dataset.cardId);
      return;
    }
    const groupButton = event.target.closest(".expanded-group-select");
    const group = state.displayGroups.find(item => item.key === groupButton?.dataset.groupKey);
    if (group && !groupButton.disabled && canSelectOwnHand()) {
      state.selectedIds = toggleGroupSelection(state.selectedIds, group.selectionIds);
      state.actionError = null;
      syncOwnHandSelection();
    }
  });
  for (const id of ["tableControlsButton", "mobileTableToolsButton"]) {
    $(id).addEventListener("click", () => { state.controlsTrigger = id; openDialog($("tableControlsDialog")); });
  }
  $("tableControlsDialog").addEventListener("close", () => {
    // Opening another native dialog must not pull focus back into the table.
    if (!document.querySelector("dialog[open]")) restoreVisibleFocus(state.controlsTrigger);
  });
  $("tableToolsActions").addEventListener("click", event => {
    const target = event.target.closest("button[data-table-tool]")?.dataset.tableTool;
    if (!target) return;
    if (target === "copy") { copyTableNumber(); return; }
    $("tableControlsDialog").close();
    if (target === "rules") { state.rulesTrigger = state.controlsTrigger; openDialog($("rulesDialog")); }
    if (target === "history") { state.historyTrigger = state.controlsTrigger; renderTrickHistory(state.view); openDialog($("trickHistoryDialog")); }
    if (target === "hand") openExpandedHand();
  });
  document.querySelectorAll("button[data-control-size], button[data-control-side]").forEach(button => {
    button.addEventListener("click", () => {
      if (button.disabled || state.pendingAction || state.pendingArrange) return;
      cancelSelectionGesture();
      if (button.dataset.controlSize) state.controlPreferences.size = button.dataset.controlSize;
      if (button.dataset.controlSide) state.controlPreferences.side = button.dataset.controlSide;
      saveControlPreferences(storage, state.controlPreferences);
      renderControlPreferences();
      scheduleOwnHandSizing();
    });
  });
  $("trickHistoryButton").addEventListener("click", () => { state.historyTrigger = "trickHistoryButton"; renderTrickHistory(state.view); openDialog($("trickHistoryDialog")); });
  $("trickHistoryDialog").addEventListener("close", () => {
    restoreVisibleFocus(state.historyTrigger);
  });
  const fullscreenButton = $("fullscreenButton");
  document.querySelector(".solo-topbar-tools")?.prepend(fullscreenButton);
  fullscreenButton.addEventListener("click", async () => {
    if (fullscreenButton.disabled) return;
    fullscreenButton.disabled = true;
    try {
      const outcome = await toggleTableFullscreen(document, $("humanPlayApp"));
      if (outcome === "unsupported") showToast("当前浏览器不支持全屏，请横屏游玩。", "warning");
      if (outcome === "denied") showToast("未能进入全屏，可稍后重试。", "warning");
    } finally { fullscreenButton.disabled = false; syncFullscreenControl(); }
  });
  for (const event of ["fullscreenchange", "webkitfullscreenchange"]) document.addEventListener(event, () => {
    syncFullscreenControl(); updateHumanTableViewportScale(); scheduleOwnHandSizing();
  });
  syncFullscreenControl();
  $("handSpreadButton").addEventListener("click", () => {
    if (state.pendingAction || state.pendingArrange || interactionFrozen()) return;
    cancelSelectionGesture();
    state.handSpread = !state.handSpread;
    saveHandSpread(storage, state.handSpread);
    renderHandSpreadControl();
    scheduleOwnHandSizing();
  });
}

function syncFullscreenControl() {
  const active = tableIsFullscreen(document);
  const button = $("fullscreenButton");
  button.textContent = active ? "退出全屏" : "全屏";
  button.setAttribute("aria-pressed", String(active));
}

function renderHandSpreadControl() {
  $("ownHandPanel").classList.toggle("hand-spread", state.handSpread);
  const button = $("handSpreadButton");
  button.setAttribute("aria-pressed", String(state.handSpread));
  button.textContent = state.handSpread ? "收紧叠牌" : "展开叠牌";
  button.disabled = Boolean(state.pendingAction || state.pendingArrange || interactionFrozen());
}

function renderControlPreferences() {
  const viewport = window.visualViewport;
  const size = effectiveControlSize(state.controlPreferences,
    viewport?.width || window.innerWidth, viewport?.height || window.innerHeight);
  $("humanTableStage").dataset.controlSize = size;
  $("humanTableStage").dataset.controlSide = state.controlPreferences.side;
  document.querySelectorAll("[data-control-size], [data-control-side]").forEach(button => {
    // Only preference buttons have pressed state; the canvas has the same data attributes.
    if (button.tagName !== "BUTTON") return;
    const active = button.dataset.controlSize ? button.dataset.controlSize === state.controlPreferences.size
      : button.dataset.controlSide === state.controlPreferences.side;
    button.setAttribute("aria-pressed", String(active));
    button.disabled = Boolean(state.pendingAction || state.pendingArrange);
  });
}

function handleVisualViewportResize() {
  updateHumanTableViewportScale();
  scheduleOwnHandSizing();
}

function initializeSoloMode() {
  document.body.classList.add("solo-player-arena");
  document.title = "DanKS · GuanDan（掼蛋）";
  document.querySelectorAll([
    ".back-to-spectator",
    ".topbar-review-link",
    ".human-slot-switch",
    "#spectatorText",
    "#spectatorRuleBadge",
    "#latestReplayLink",
    "#resultReplayLink",
    "#seriesReplayButton",
  ].join(",")).forEach((node) => { node.hidden = true; });
  $("createTab").textContent = "创建单人桌";
  $("joinTab").textContent = "桌号继续";
  $("entryTitle").textContent = "单人挑战 DanKS";
  const introduction = $("entryPanel").querySelector(":scope > p");
  if (introduction) {
    introduction.textContent = "DanKS 是 CardKS 面向 GuanDan（掼蛋）的游戏 AI；你固定坐在 1 号位，其余三个座位全部由 Bot 接管。";
  }
  $("createTableButton").firstChild.textContent = "创建单人桌 ";
  $("joinTableButton").firstChild.textContent = "继续此桌 ";
  const joinNote = document.querySelector(".join-note");
  if (joinNote) joinNote.textContent = "此设备会记住你的座位，返回原桌即可继续。";
  $("lobbyHelpText").textContent = "点击准备后发牌；Bot 默认准备，20 秒倒计时不会替你出牌。";
  renderRecentTableCard();
  installSoloLocale(document, soloLocalStorage);
}

function updateHumanTableViewportScale() {
  const stage = $("humanTableStage");
  if (!stage) return;
  const viewport = window.visualViewport;
  const width = viewport?.width || window.innerWidth || HUMAN_TABLE_VIEWPORT.width;
  const height = viewport?.height || window.innerHeight || HUMAN_TABLE_VIEWPORT.height;
  const { scale, portraitHint, shortLandscape, toolsRail } = tableFrameMetrics(width, height, { mobileTools: SOLO_MODE });
  const scaleText = scale.toFixed(5);
  if (stage.dataset.shortLandscape !== String(shortLandscape)
    || stage.style.getPropertyValue("--human-table-scale") !== scaleText) cancelSelectionGesture();
  stage.dataset.shortLandscape = String(shortLandscape);
  stage.style.setProperty("--mobile-tools-rail", `${toolsRail}px`);
  $("humanPlayApp").dataset.mobileTools = String(SOLO_MODE && width <= 1024);
  $("mobileTableToolsButton").hidden = !SOLO_MODE || width > 1024;
  stage.style.setProperty("--human-table-scale", scaleText);
  const hint = $("landscapeHint");
  if (hint) hint.hidden = !portraitHint;
  renderControlPreferences();
}

function bindEntryControls() {
  const tabs = [$("createTab"), $("joinTab")];
  tabs[0].addEventListener("click", () => showEntryTab("create"));
  tabs[1].addEventListener("click", () => showEntryTab("join"));
  tabs.forEach((tab, index) => {
    tab.addEventListener("keydown", (event) => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
      event.preventDefault();
      const nextIndex = event.key === "Home"
        ? 0
        : event.key === "End"
          ? tabs.length - 1
          : (index + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
      showEntryTab(nextIndex === 0 ? "create" : "join");
      tabs[nextIndex].focus();
    });
  });
  $("createForm").addEventListener("submit", createTable);
  $("joinForm").addEventListener("submit", joinTable);
  $("resumeRecentTableButton").addEventListener("click", () => resumeRecentSoloTable());
}

function bindRoomControls() {
  renderTableSoundControl();
  $("tableSoundButton").addEventListener("click", async () => {
    await tableSound.toggle();
    renderTableSoundControl();
    if (tableSound.enabled) tableSound.play("turn");
  });
  $("readyButton").addEventListener("click", toggleReady);
  $("retryRoomButton").addEventListener("click", retryRoom);
  $("leaveFailedRoomButton").addEventListener("click", () => returnToEntry({forget:true}));
  $("resultReadyButton").addEventListener("click", toggleReady);
  $("seriesRematchButton").addEventListener("click", toggleReady);
  $("seriesReplayButton").addEventListener("click", toggleSeriesReplayList);
  $("seriesCopyButton").addEventListener("click", copySeriesResult);
  $("passButton").addEventListener("click", submitPass);
  $("playButton").addEventListener("click", prepareSelectedPlay);
  $("reconcileActionButton").addEventListener("click", () => {
    const pending = state.pendingAction;
    if (!pending?.uncertain || interactionFrozen()) return;
    pending.uncertain = false;
    state.actionError = null;
    renderOwnHand(state.view, false);
    sendPendingAction(pending); // Reuse the identical idempotency key, never a new play.
  });
  $("clearSelectionButton").addEventListener("click", clearHandSelection);
  $("swipeSelectButton").addEventListener("click", () => {
    if (!canSelectOwnHand()) return;
    cancelSelectionGesture();
    state.swipeSelection = !state.swipeSelection;
    syncOwnHandSelection();
  });
  $("copyTableBtn").addEventListener("click", copyTableNumber);
  $("rulesButton").addEventListener("click", () => { state.rulesTrigger = "rulesButton"; openDialog($("rulesDialog")); });
  const arrangeMenuButton = $("arrangeMenuButton");
  $("rankDisplayButton").addEventListener("click", () => {
    if (interactionFrozen() || state.pendingArrange || state.pendingAction || !state.view) return;
    cancelSelectionGesture();
    state.handDisplayMode = state.handDisplayMode === "rank" ? "arranged" : "rank";
    state.resetHandLayout = true;
    setArrangeMenuOpen(false);
    renderOwnHand(state.view, false);
  });
  arrangeMenuButton.addEventListener("click", () => {
    setArrangeMenuOpen(arrangeMenuButton.getAttribute("aria-expanded") !== "true");
  });
  document.querySelectorAll("[data-arrange-mode]").forEach((button) => {
    button.addEventListener("click", () => {
      setArrangeMenuOpen(false);
      arrangeOwnHand(Number(button.dataset.arrangeMode));
    });
  });
  document.addEventListener("click", (event) => {
    if (!$("arrangeToolbar").contains(event.target)) setArrangeMenuOpen(false);
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && arrangeMenuButton.getAttribute("aria-expanded") === "true") {
      setArrangeMenuOpen(false, { restoreFocus: true });
      return;
    }
    if (event.key === "Escape" && !event.isComposing && $("ownHandPanel").contains(event.target) && !document.querySelector("dialog[open]")) {
      clearHandSelection();
    }
  });
  $("actionDialog").addEventListener("close", () => {
    if ($("actionDialog").open) return;
    state.actionChoices = null;
    $("actionChoiceList").replaceChildren();
    restoreVisibleFocus("playButton");
  });
  $("rulesDialog").addEventListener("close", () => restoreVisibleFocus(state.rulesTrigger));
  const hand = $("ownHand");
  const viewport = $("ownHandViewport");
  viewport.addEventListener("scroll", dismissHandScrollHint, { passive: true });
  hand.addEventListener("pointerdown", beginSelectionGesture);
  hand.addEventListener("pointermove", extendSelectionGesture);
  hand.addEventListener("lostpointercapture", cancelSelectionGesture);
  hand.addEventListener("click", handleOwnHandClick);
  document.addEventListener("pointerup", finishSelectionGesture);
  document.addEventListener("pointercancel", cancelSelectionGesture);
  window.addEventListener("blur", cancelSelectionGesture);
  window.addEventListener("resize", scheduleOwnHandSizing);
}

function initializeOwnHandResizeObserver() {
  if (typeof ResizeObserver !== "function") return;
  state.handLayoutObserver = new ResizeObserver(scheduleOwnHandSizing);
  state.handLayoutObserver.observe($("ownHandPanel"));
  state.handLayoutObserver.observe($("ownHandViewport"));
}

function scheduleOwnHandSizing() {
  if (state.handLayoutFrame !== null) return;
  state.handLayoutFrame = window.requestAnimationFrame(() => {
    state.handLayoutFrame = null;
    updateOwnHandSizing();
  });
}

function dismissHandScrollHint() {
  const hint = $("handScrollHint");
  if ($("ownHandViewport").scrollLeft <= 2) return;
  hint.dataset.dismissed = "true";
  hint.hidden = true;
}

function updateOwnHandSizing() {
  const panel = $("ownHandPanel");
  const viewport = $("ownHandViewport");
  const hand = $("ownHand");
  const groupCount = hand.children.length;
  const groups = [...hand.children];
  groups.forEach((group) => group.classList.remove("row-end"));
  if (panel.hidden || groupCount === 0) {
    panel.style.removeProperty("--own-hand-panel-width");
    hand.style.removeProperty("--own-hand-group-step");
    hand.classList.remove("wrapped-layout");
    const hint = $("handScrollHint");
    delete hint.dataset.dismissed;
    hint.hidden = true;
    $("humanTableStage").style.setProperty("--own-hand-panel-height", "0px");
    return;
  }

  const styles = window.getComputedStyle(hand);
  if (document.documentElement.classList.contains("fixed-aspect-table")) {
    const preferred = cssPixelValue(styles.getPropertyValue("--own-hand-rank-reveal"), 28);
    const cardHeight = cssPixelValue(styles.getPropertyValue("--own-hand-card-height"), 122);
    const maxHeight = cssPixelValue(styles.getPropertyValue("--own-hand-stack-max-height"), 248);
    groups.forEach(group => group.style.setProperty("--own-hand-stack-reveal", `${fitHandStackReveal({
      count: group.querySelectorAll(".hand-card-button").length, preferred, cardHeight, maxHeight,
    })}px`));
  }
  const naturalStep = cssPixelValue(styles.getPropertyValue("--own-hand-natural-step"), 75);
  const lastGroupWidth = cssPixelValue(styles.getPropertyValue("--own-hand-card-width"), 74);
  const minimumStep = cssPixelValue(styles.getPropertyValue("--own-hand-min-step"), 26);
  const paddingInline = cssPixelValue(styles.paddingLeft) + cssPixelValue(styles.paddingRight);
  const gap = cssPixelValue(styles.columnGap || styles.gap);
  const naturalMetrics = fitHandGroupStep({
    groupCount,
    naturalStep,
    minimumStep,
    lastGroupWidth,
    paddingInline,
    gap,
  });
  const panelWidth = panel.clientWidth;
  const viewportWidth = viewport.clientWidth;
  const fixedPanelChrome = Math.max(0, panelWidth - viewportWidth);
  const desiredPanelWidth = Math.max(760, naturalMetrics.naturalWidth + fixedPanelChrome);
  panel.style.setProperty("--own-hand-panel-width", `${Math.ceil(desiredPanelWidth)}px`);

  const fittedMetrics = fitHandGroupStep({
    availableWidth: viewport.clientWidth,
    groupCount,
    naturalStep,
    minimumStep,
    lastGroupWidth,
    paddingInline,
    gap,
  });
  const mobileTouchLayout = !document.documentElement.classList.contains("fixed-aspect-table") && window.matchMedia([
    "(max-width: 760px)",
    "(min-width: 761px) and (max-width: 1024px) and (max-height: 600px) and (orientation: landscape)",
  ].join(", ")).matches;
  if (mobileTouchLayout) {
    hand.style.removeProperty("--own-hand-group-step");
  } else {
    hand.style.setProperty("--own-hand-group-step", `${fittedMetrics.step.toFixed(2)}px`);
  }
  hand.classList.toggle("wrapped-layout", fittedMetrics.wrap && !mobileTouchLayout);
  if (fittedMetrics.wrap && !mobileTouchLayout) {
    groups.forEach((group, index) => {
      group.classList.toggle(
        "row-end",
        (index + 1) % fittedMetrics.groupsPerRow === 0 || index === groups.length - 1,
      );
    });
  }
  $("humanTableStage").style.setProperty(
    "--own-hand-panel-height",
    `${panel.offsetHeight}px`,
  );
  const hint = $("handScrollHint");
  hint.hidden = viewport.scrollWidth <= viewport.clientWidth + 1;
  if (hint.dataset.dismissed === "true") hint.hidden = true;
}

function destroyRoomRuntime() {
  closeActionChoices();
  closeExpandedHand();
  closeLiveSocket();
  clearTrickMotion();
  cancelSelectionGesture();
  state.handLayoutObserver?.disconnect();
  if (state.handLayoutFrame !== null) window.cancelAnimationFrame(state.handLayoutFrame);
  if (state.suppressSelectionClickTimer !== null) {
    window.clearTimeout(state.suppressSelectionClickTimer);
  }
}

async function restoreFromSession() {
  const url = new URL(window.location.href);
  if (url.searchParams.has("token")) {
    url.searchParams.delete("token");
    window.history.replaceState({}, "", url);
  }
  const tableNo = String(url.searchParams.get("table") ?? "").trim();
  if (!tableNo) return;
  window.danksPageBoot?.restoring();
  if (!SOLO_MODE) $("joinTableNo").value = tableNo;
  const token = loadRoomToken(tableNo, soloSessionStorage);
  if (!token) {
    if (SOLO_MODE) {
      const recent = findRecentSoloTable(soloLocalStorage, tableNo);
      if (recent) {
        await resumeRecentSoloTable(recent);
        return;
      }
      $("joinTableNo").value = tableNo;
      showEntryTab("join");
      window.setTimeout(() => $("joinNickname").focus(), 0);
    } else {
      showEntryTab("join");
      window.setTimeout(() => $("joinTableNo").focus(), 0);
    }
    return;
  }

  setConnection("reconnecting", "正在恢复桌面");
  try {
    const rawView = await api.viewTable(tableNo, token);
    state.tableNo = tableNo;
    state.token = token;
    applyRoomView(extractView(rawView, tableNo));
    enterRoomShell();
    connectLiveSocket();
  } catch (error) {
    if (SOLO_MODE && (error?.status === 401 || error?.status === 403)) {
      clearRoomToken(tableNo, soloSessionStorage);
      const recent = findRecentSoloTable(soloLocalStorage, tableNo);
      if (recent) {
        await resumeRecentSoloTable(recent);
        return;
      }
    } else if (error?.status === 401 || error?.status === 403 || error?.status === 404) {
      clearRoomToken(tableNo, soloSessionStorage);
      if (SOLO_MODE && error?.status === 404) {
        forgetRecentSoloTable(soloLocalStorage, tableNo);
      }
    }
    setConnection("offline", "需要重新加入");
    if (SOLO_MODE) {
      $("entryPanel").hidden = false;
      $("joinTableNo").value = tableNo;
      showEntryTab("join");
    } else {
      showEntryTab("join");
    }
    showToast(errorMessage(error), "error");
    showEntryRestoreError(error);
    renderRecentTableCard();
  }
}

function showEntryRestoreError(error) {
  const message = $("entryRestoreMessage");
  message.textContent = "未能恢复对局。请检查网络，或重新输入桌号继续。";
  message.hidden = false;
}

function renderRecentTableCard() {
  if (!SOLO_MODE) return;
  const recent = loadRecentSoloTables(soloLocalStorage)[0] ?? null;
  const card = $("recentTableCard");
  card.hidden = !recent;
  if (!recent) return;
  $("recentTableTitle").textContent = `桌号 ${recent.tableNo}`;
  const nickname = document.createElement("span");
  nickname.dataset.l10nIgnore = "true";
  nickname.textContent = recent.nickname;
  $("recentTableMeta").replaceChildren(nickname, document.createTextNode(" · 返回原座位继续"));
}

async function resumeRecentSoloTable(recentTable = null) {
  if (!SOLO_MODE) return;
  const recent = recentTable ?? loadRecentSoloTables(soloLocalStorage)[0] ?? null;
  if (!recent) {
    renderRecentTableCard();
    return;
  }
  const button = $("resumeRecentTableButton");
  $("entryRestoreMessage").hidden = true;
  window.danksPageBoot?.restoring();
  setButtonPending(button, true, "正在恢复…");
  setConnection("reconnecting", "正在恢复桌面");
  try {
    const response = await api.joinTable(recent.tableNo, {
      nickname: recent.nickname,
      user_id: soloUserId(),
    });
    adoptJoinResponse(response, recent.nickname);
    showToast("已回到上次对局。", "success");
  } catch (error) {
    if (error?.status === 403 || error?.status === 404) {
      clearRoomToken(recent.tableNo, soloSessionStorage);
      forgetRecentSoloTable(soloLocalStorage, recent.tableNo);
    }
    setConnection("offline", "恢复失败");
    renderRecentTableCard();
    showToast(errorMessage(error), "error");
    showEntryRestoreError(error);
  } finally {
    setButtonPending(button, false, "继续对局");
    window.danksPageBoot?.ready();
  }
}

function showEntryTab(tab) {
  const create = tab === "create";
  $("createTab").setAttribute("aria-selected", String(create));
  $("joinTab").setAttribute("aria-selected", String(!create));
  $("createTab").tabIndex = create ? 0 : -1;
  $("joinTab").tabIndex = create ? -1 : 0;
  $("createForm").hidden = !create;
  $("joinForm").hidden = create;
}

function renderHumanSlotOptions(game) {
  const container = $("humanSlotOptions");
  container.replaceChildren();
  SLOT_LABELS[game].forEach((labelText, index) => {
    const slots = index + 1;
    const input = document.createElement("input");
    input.type = "radio";
    input.name = "humanSlots";
    input.id = `humanSlots${slots}`;
    input.value = String(slots);
    input.checked = index === 0;
    const label = document.createElement("label");
    label.htmlFor = input.id;
    label.textContent = `${labelText}（${slots} 人类）`;
    container.append(input, label);
  });
  container.style.gridTemplateColumns = `repeat(${SLOT_LABELS[game].length}, 1fr)`;
}

async function createTable(event) {
  event.preventDefault();
  const button = $("createTableButton");
  const game = "guandan";
  const nickname = $("createNickname").value.trim();
  if (!nickname) return;

  setButtonPending(button, true, "正在创建…");
  try {
    const response = await api.createTable({ game, nickname, user_id: soloUserId() });
    adoptJoinResponse(response, nickname);
    showToast("桌子已创建，请准备。", "success");
  } catch (error) {
    showToast(errorMessage(error), "error");
  } finally {
    setButtonPending(button, false, "创建并入座");
  }
}

async function joinTable(event) {
  event.preventDefault();
  const button = $("joinTableButton");
  const tableNo = $("joinTableNo").value.trim();
  const nickname = $("joinNickname").value.trim();
  if (!tableNo || !nickname) return;

  setButtonPending(button, true, "正在加入…");
  try {
    const response = await api.joinTable(tableNo, {
      nickname,
      user_id: SOLO_MODE ? soloUserId() : undefined,
      resume_token: undefined,
    });
    adoptJoinResponse(response, nickname);
    const role = String(response?.role ?? response?.view?.viewer?.role ?? "");
    showToast(
      SOLO_MODE
        ? "已恢复桌子。"
        : isSpectatorRole(role) ? "桌子已满，已进入观战席。" : "已加入桌子。",
      "success",
    );
  } catch (error) {
    showToast(errorMessage(error), "error");
  } finally {
    setButtonPending(button, false, SOLO_MODE ? "继续此桌" : "加入桌子");
  }
}

function soloUserId() {
  let value = String(soloLocalStorage?.getItem(SOLO_USER_ID_KEY) ?? "").trim();
  if (/^[A-Za-z0-9_-]{16,128}$/u.test(value)) return value;
  const random = window.crypto?.randomUUID?.().replaceAll("-", "")
    ?? String(Date.now().toString(36) + Math.random().toString(36).slice(2));
  value = "solo_" + random;
  soloLocalStorage?.setItem(SOLO_USER_ID_KEY, value);
  return value;
}

function adoptJoinResponse(response, nickname = "") {
  const tableNo = String(response?.table_no ?? "").trim();
  const token = String(response?.token ?? "");
  if (!tableNo || !token || !response?.view) throw new Error("服务器未返回完整的入桌信息。");
  closeLiveSocket();
  state.tableNo = tableNo;
  state.token = token;
  state.selectedIds.clear();
  state.pendingAction = null;
  state.actionError = null;
  state.retryError = "";
  state.reconnectAttempt = 0;
  saveRoomToken(tableNo, token, soloSessionStorage);
  if (SOLO_MODE) {
    rememberRecentSoloTable(soloLocalStorage, {
      tableNo,
      nickname: nickname || "玩家",
      updatedAt: Date.now(),
    });
    renderRecentTableCard();
  }
  updateTableUrl(tableNo);
  applyRoomView(extractView(response.view, tableNo));
  enterRoomShell();
  connectLiveSocket();
}

function enterRoomShell() {
  $("entryPanel").hidden = true;
  $("roomIdentity").hidden = false;
  $("roomStatusStrip").hidden = false;
}

function updateTableUrl(tableNo) {
  const url = new URL(window.location.href);
  url.searchParams.set("table", tableNo);
  url.searchParams.delete("token");
  window.history.replaceState({}, "", url);
}

function connectLiveSocket() {
  if (!state.tableNo || !state.token) return;
  clearReconnectTimer();
  const generation = ++state.socketGeneration;
  state.closeSocket?.();
  state.closeSocket = null;
  setConnection("reconnecting", "正在连接实时桌面");
  try {
    state.closeSocket = api.connect(state.tableNo, state.token, {
      lastVersion: state.view?.version ?? 0,
      onOpen: () => {
        if (generation !== state.socketGeneration) return;
        setConnection("reconnecting", "正在验证实时桌面");
      },
      onState: (rawView) => {
        if (generation !== state.socketGeneration) return;
        state.reconnectAttempt = 0;
        setConnection("online", "实时已连接", { render: false });
        applyRoomView(extractView(rawView, state.tableNo));
      },
      onClose: (event) => {
        if (generation !== state.socketGeneration) return;
        state.closeSocket = null;
        if (event?.code === 4404 || event?.code === 4401) {
          returnToEntry({forget: event.code === 4404, message: event.code === 4404
            ? "此桌已不存在，请创建新桌。"
            : "当前会话已失效，请点击继续对局重新进入。"});
          return;
        }
        scheduleReconnect();
      },
      onError: () => {
        if (generation === state.socketGeneration) {
          setConnection("reconnecting", "连接中断，正在重连");
        }
      },
    });
  } catch (error) {
    showToast(errorMessage(error), "error");
    scheduleReconnect();
  }
}

function scheduleReconnect() {
  if (!state.tableNo || !state.token || state.reconnectTimer) return;
  state.reconnectAttempt += 1;
  const delayMs = Math.min(10_000, 750 * (2 ** Math.min(state.reconnectAttempt - 1, 4)));
  setConnection("reconnecting", `${Math.ceil(delayMs / 1000)} 秒后重连`);
  state.reconnectTimer = window.setTimeout(() => {
    state.reconnectTimer = null;
    connectLiveSocket();
  }, delayMs);
}

function closeLiveSocket() {
  state.socketGeneration += 1;
  clearReconnectTimer();
  state.closeSocket?.();
  state.closeSocket = null;
}

function clearReconnectTimer() {
  if (state.reconnectTimer) window.clearTimeout(state.reconnectTimer);
  state.reconnectTimer = null;
}

function returnToEntry({forget = false, message = ""} = {}) {
  const tableNo = state.tableNo;
  closeLiveSocket();
  clearRoomToken(tableNo, soloSessionStorage);
  if (forget) forgetRecentSoloTable(soloLocalStorage, tableNo);
  cancelSelectionGesture(undefined, {restoreSelection:false});
  closeActionChoices();
  clearTrickMotion();
  for (const dialog of document.querySelectorAll("dialog[open]")) dialog.close();
  state.tableNo = "";
  state.token = "";
  state.view = null;
  state.clock = null;
  state.pendingAction = null;
  state.pendingMutation = false;
  state.pendingArrange = false;
  state.retryError = "";
  state.selectedIds.clear();
  state.displayGroups = [];
  state.displayContext = "";
  state.recentRememberSignature = "";
  for (const id of ["humanSeatLayer", "ownHand", "expandedHandGroups"]) $(id).replaceChildren();
  for (const id of ["roomIdentity", "roomStatusStrip", "ownHandPanel", "lobbyHelp",
    "resultBanner", "seriesResultOverlay", "guandanLevelHud", "currentTrickContext", "roomFailurePanel", "trickHistoryButton"]) $(id).hidden = true;
  $("humanTableStage").classList.remove("solo-lobby", "round-terminal", "spectator-view");
  $("entryPanel").hidden = false;
  $("entryRestoreMessage").hidden = !message;
  $("entryRestoreMessage").textContent = message;
  $("joinTableNo").value = forget ? "" : tableNo;
  showEntryTab("create");
  renderRecentTableCard();
  const url = new URL(window.location.href);
  url.searchParams.delete("table");
  url.searchParams.delete("token");
  window.history.replaceState({}, "", url);
  setConnection("idle", "未连接");
  $(forget ? "createNickname" : "resumeRecentTableButton").focus({preventScroll:true});
}

function applyRoomView(rawView) {
  const nextView = normalizeRoomView(rawView);
  if (state.view?.table_no === nextView.table_no && nextView.version < state.view.version) return;
  if (state.actionChoices && state.actionChoices.context !== actionChoiceContext(nextView)) closeActionChoices();
  const keepInspection = state.expandedHandSignature && state.expandedHandSignature === currentHandInspectionSignature();
  const bindings = handRefreshBindings(state.view, nextView, state.displayGroups, {
    pendingAction: state.pendingAction, pendingArrange: state.pendingArrange, interactionFrozen: interactionFrozen(),
  });
  if (bindings) {
    // Retag existing DOM nodes before rendering. Focus and exposed hitboxes stay
    // on the same occurrence, while every click reads its current physical ID.
    for (const container of [$("ownHand"), $("expandedHandGroups")]) {
      container.querySelectorAll("button[data-card-id]").forEach(button => {
        const card = bindings.get(button.dataset.cardId);
        if (card) button.dataset.cardId = card.id;
      });
    }
    state.displayGroups = state.displayGroups.map(group => ({ ...group, cards: group.cards.map(card => bindings.get(card.id)) }));
  }
  // Discard an old gesture before filtering or clearing authoritative selection.
  // Its cancellation baseline must never leak into a new server decision.
  if (state.selectionGesture && !canContinueHandGesture(state.selectionGesture, nextView)) {
    cancelSelectionGesture(undefined, { restoreSelection: false });
  }
  const feedback = confirmedTableFeedback(state.view, nextView);
  if (feedback.trickCleared) collectConfirmedTrick();
  if (state.view && (state.view.table_no !== nextView.table_no || state.view.round_no !== nextView.round_no)) {
    clearTrickMotion();
    if ($("trickHistoryDialog").open) $("trickHistoryDialog").close();
  }
  const enteredSettlement = state.view?.phase === "play" && isTerminalPhase(nextView.phase)
    && state.view.table_no === nextView.table_no && state.view.viewer.seat === nextView.viewer.seat;
  state.playAnimationSeats = new Set(feedback.playSeats);
  if (!bindings && state.view?.table_no === nextView.table_no && state.view.round_no === nextView.round_no
    && state.view.viewer.seat === nextView.viewer.seat && nextView.version > state.view.version) {
    const submittedIds = state.pendingAction?.selectedIds ?? [];
    const confirmedDeparture = state.view.own_hand.length - nextView.own_hand.length === submittedIds.length;
    state.displayGroups = rebindDisplayGroups(state.displayGroups, nextView.own_hand,
      new Set(confirmedDeparture ? submittedIds : []));
  }
  const previousDecisionId = state.view?.decision?.id ?? "";
  const nextDecisionId = nextView.decision?.id ?? "";
  state.selectedIds = reconcileHandSelection(state.view, nextView, state.selectedIds, { pendingAction: state.pendingAction });
  if (previousDecisionId !== nextDecisionId) state.actionError = null;
  if (pendingActionResolved(state.pendingAction, state.view, nextView)) {
    state.pendingAction = null;
    state.selectedIds.clear();
    state.actionError = null;
  }

  state.view = nextView;
  if (keepInspection && bindings) state.expandedHandSignature = handInspectionSignature(nextView, state.displayGroups);
  state.tableNo = nextView.table_no || state.tableNo;
  if (SOLO_MODE && nextView.phase === "game_over" && nextView.series?.match_finished) {
    forgetRecentSoloTable(soloLocalStorage, state.tableNo);
    state.recentRememberSignature = "";
    renderRecentTableCard();
  } else if (SOLO_MODE && !isSpectatorRole(nextView.viewer.role)) {
    const ownSeat = nextView.seats.find((seat) => seat.position === nextView.viewer.seat);
    const nickname = String(ownSeat?.display_name ?? "").trim();
    const signature = `${state.tableNo}\u0000${nickname}`;
    if (state.tableNo && nickname && signature !== state.recentRememberSignature) {
      rememberRecentSoloTable(soloLocalStorage, {
        tableNo: state.tableNo,
        nickname,
        updatedAt: Date.now(),
      });
      state.recentRememberSignature = signature;
      renderRecentTableCard();
    }
  }
  state.clock = decisionClock(nextView, Date.now());
  renderRoom();
  if (feedback.trickCleared) {
    const cue = $("trickFlowCue");
    cue.textContent = `本轮结束，${nextView.decision.actor + 1} 号位先出`;
    cue.hidden = false;
    if (state.trickCueTimer !== null) window.clearTimeout(state.trickCueTimer);
    state.trickCueTimer = window.setTimeout(() => { cue.hidden = true; state.trickCueTimer = null; }, 1200);
  }
  if (enteredSettlement && !isSpectatorRole(nextView.viewer.role)) {
    window.requestAnimationFrame(() => {
      const target = $("seriesResultOverlay").hidden ? $("resultReadyButton") : $("seriesRematchButton");
      if (!target.hidden && !target.disabled) target.focus({ preventScroll: true });
    });
  }
  if (feedback.ownTurn) {
    const dock = $("actionDock");
    dock.classList.remove("turn-arrived");
    if (state.turnCueTimer !== null) window.clearTimeout(state.turnCueTimer);
    dock.classList.add("turn-arrived");
    state.turnCueTimer = window.setTimeout(() => { dock.classList.remove("turn-arrived"); state.turnCueTimer = null; }, 240);
  }
  if (feedback.ownTurn || feedback.playSeats.length) tableSound.play(feedback.ownTurn ? "turn" : "play");
}

function renderTableSoundControl() {
  const button = $("tableSoundButton");
  button.setAttribute("aria-pressed", String(tableSound.enabled));
  button.textContent = tableSound.enabled ? "提示音：开" : "提示音：关";
  button.disabled = !(window.AudioContext || window.webkitAudioContext);
}

function renderCurrentTrick(view) {
  const context = $("currentTrickContext");
  const trick = view.current_trick;
  context.hidden = view.game !== "guandan" || view.phase !== "play" || trick?.mode === "unknown" || !trick;
  if (context.hidden) return;
  const target = trick.target;
  context.dataset.mode = trick.mode;
  $("currentTrickHeading").textContent = target ? "当前压牌" : "自由出牌";
  const actor = view.seats.find(seat => seat.position === view.decision?.actor);
  const rank = guandanRankLabel(target);
  $("currentTrickSummary").textContent = target
    ? `${target.seat + 1} 号位 · ${guandanTypeLabel(target)}${rank ? ` · ${rank}` : ""}`
    : actor ? `等待 ${seatDisplayName(actor)}出牌` : "请选择合法牌型";
  const caption = target ? wildcardCaption(target, view.level) : "";
  $("currentTrickWildcard").hidden = !caption;
  $("currentTrickWildcard").textContent = caption;
}

function renderTrickHistory(view) {
  if (!view) return;
  const history = view.trick_history;
  const previous = history?.previous;
  const signature = JSON.stringify([view.table_no, view.round_no, view.level, history]);
  if (signature === state.historyRenderSignature) return;
  state.historyRenderSignature = signature;
  const list = $("trickHistoryList");
  list.replaceChildren();
  $("trickHistoryMeta").textContent = previous ? `第 ${previous.number} 轮 · ${previous.events.length} 次出牌` : "";
  const empty = $("trickHistoryEmpty");
  empty.hidden = Boolean(previous?.events.length && previous.complete);
  empty.textContent = !history?.available ? "当前服务未提供上一轮记录。"
    : previous && !previous.complete ? "这一轮记录不完整，仅展示已确认的公开动作。"
    : "本局还没有完成的一轮出牌。";
  for (const event of previous?.events ?? []) {
    const item = document.createElement("li");
    const name = document.createElement("strong");
    name.className = "trick-history-seat";
    const seatLabel = document.createElement("span");
    seatLabel.textContent = `${event.seat + 1} 号位`;
    const nickname = document.createElement("span");
    nickname.textContent = seatDisplayName(view.seats[event.seat]);
    if (view.seats[event.seat]?.display_name) nickname.dataset.l10nIgnore = "true";
    name.append(seatLabel, " · ", nickname);
    const content = document.createElement("div");
    const label = document.createElement("span");
    label.className = "trick-history-type";
    label.textContent = event.pass ? "不出" : `${guandanTypeLabel(event)}${guandanRankLabel(event) ? ` · ${guandanRankLabel(event)}` : ""}`;
    content.appendChild(label);
    if (!event.pass) {
      const cards = document.createElement("div");
      cards.className = "trick-history-cards";
      event.cards.forEach(code => cards.appendChild(createPlayingCard(code, "table", {
        game: view.game, levelRank: view.level ?? "", ...HUMAN_CARD_VISUALS,
      })));
      content.appendChild(cards);
    }
    item.append(name, content);
    list.appendChild(item);
  }
}

function clearTrickMotion() {
  if (state.trickMotionTimer !== null) window.clearTimeout(state.trickMotionTimer);
  if (state.trickCueTimer !== null) window.clearTimeout(state.trickCueTimer);
  state.trickMotionTimer = null;
  state.trickCueTimer = null;
  $("trickMotionLayer").replaceChildren();
  $("trickFlowCue").hidden = true;
}

function collectConfirmedTrick() {
  clearTrickMotion();
  if (!SOLO_MODE || window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) return;
  const stageRect = $("humanTableStage").getBoundingClientRect();
  const scale = stageRect.width / HUMAN_TABLE_VIEWPORT.width;
  if (scale <= 0) return;
  document.querySelectorAll(".human-seat.live-identity .seat-play-area.has-played-cards").forEach(area => {
    const cards = area.querySelector(".play-cards");
    const rect = cards.getBoundingClientRect();
    const ghost = cards.cloneNode(true);
    ghost.className = "trick-collected-cards";
    const visualCard = cards.firstElementChild;
    const cardRect = visualCard?.getBoundingClientRect();
    if (!cardRect) return;
    ghost.style.setProperty("--collected-card-width", `${cardRect.width / scale}px`);
    ghost.style.setProperty("--collected-card-height", `${cardRect.height / scale}px`);
    const secondCard = cards.children[1];
    ghost.style.setProperty("--collected-card-overlap", secondCard ? window.getComputedStyle(secondCard).marginLeft : "0px");
    ghost.style.left = `${(rect.left - stageRect.left) / scale}px`;
    ghost.style.top = `${(rect.top - stageRect.top) / scale}px`;
    ghost.style.width = `${rect.width / scale}px`;
    ghost.style.height = `${rect.height / scale}px`;
    $("trickMotionLayer").appendChild(ghost);
  });
  state.trickMotionTimer = window.setTimeout(() => { $("trickMotionLayer").replaceChildren(); state.trickMotionTimer = null; }, 190);
}

function renderRoom() {
  const view = state.view;
  if (!view) return;
  const spectator = isSpectatorRole(view.viewer.role);
  const lobby = isLobbyPhase(view.phase);
  const terminal = isTerminalPhase(view.phase);
  $("humanTableStage").classList.toggle("spectator-view", spectator);
  $("humanTableStage").classList.toggle("round-terminal", terminal);
  $("humanTableStage").classList.toggle("solo-lobby", SOLO_MODE && lobby && !spectator);
  $("humanTableStage").dataset.game = view.game;
  $("humanTableStage").dataset.seats = String(view.seats.length);
  $("roomGameChip").textContent = "GD";
  $("roomTitle").textContent = `桌号 ${view.table_no || state.tableNo}`;
  $("roomMeta").textContent = `${gameLabel(view.game)} · ${view.human_slots} 个人类座位`;
  $("phaseText").textContent = phaseLabel(view.phase);
  $("viewerBadge").textContent = spectator ? "观众" : `玩家 · ${Number(view.viewer.seat) + 1} 号位`;
  $("roundLevelMeta").textContent = view.game === "guandan" && view.level
    ? `第 ${view.round_no} 局 · 级牌 ${displayGuandanLevel(view.level)}`
    : `第 ${view.round_no} 局`;
  $("spectatorText").textContent = `${view.spectator_count} 人观战`;
  renderReplayLinks(view);
  $("humanGameMark").textContent = gameLabel(view.game);
  $("humanGameCode").textContent = "GUANDAN";
  $("lobbyHelp").hidden = !lobby;
  $("lobbyKicker").hidden = !SOLO_MODE || spectator;
  $("lobbyIntroduction").hidden = !SOLO_MODE || spectator;
  $("lobbyTitle").textContent = SOLO_MODE && !spectator ? "单人挑战" : "等待准备";
  $("lobbyHelpText").textContent = SOLO_MODE
    ? "点击准备后开始单人挑战；其余三个座位均由 Bot 接管。"
    : "可点击空位或人机座位换座；每位人类玩家都须准备，全员准备后发牌并确定本局级牌。";
  $("ruleBadgeStrip").hidden = false;
  renderGuandanLevelHud(view);
  $("rulesButton").hidden = false;
  renderRulesDialog();

  renderSeats(view);
  renderCurrentTrick(view);
  $("trickHistoryButton").hidden = view.game !== "guandan" || lobby || view.phase === "deal" || spectator;
  if ($("trickHistoryDialog").open) renderTrickHistory(view);
  renderControlPreferences();
  renderReadyControl(view, { spectator, lobby });
  renderOwnHand(view, spectator);
  renderResult(view);
  renderRoomFailure(view);
  updateCountdownDisplay();
}

function renderRoomFailure(view) {
  const paused = view.phase === "paused";
  const failed = paused || view.phase === "error";
  const retryable = paused && view.failure?.retryable === true;
  $("roomFailurePanel").hidden = !failed;
  $("roomFailureTitle").textContent = paused ? "AI 暂时不可用" : "牌局无法继续";
  $("roomFailureText").textContent = state.retryError || (retryable
    ? "牌局已保留，重试后从当前回合继续，不会更换机器人。"
    : "本局无法继续，请返回入口创建新桌。");
  $("retryRoomButton").hidden = !retryable;
  $("retryRoomButton").disabled = state.pendingMutation || interactionFrozen();
  $("retryRoomButton").textContent = state.pendingMutation ? "正在重试…" : "重试 AI";
  $("retryRoomButton").setAttribute("aria-busy", String(state.pendingMutation));
  $("leaveFailedRoomButton").disabled = state.pendingMutation;
}

function displayGuandanLevel(level) {
  return String(level ?? "").toUpperCase() === "T" ? "10" : String(level ?? "");
}

function renderGuandanLevelHud(view) {
  const hud = $("guandanLevelHud");
  const levels = view.game === "guandan" ? view.guandan_levels : null;
  hud.hidden = !levels;
  if (!levels) return;
  const activeSide = levels.active_side;
  const activeLabel = activeSide === "opponent" ? "对方" : activeSide === "our" ? "我方" : "";
  $("ourLevelValue").textContent = displayGuandanLevel(levels.our_level);
  $("opponentLevelValue").textContent = displayGuandanLevel(levels.opponent_level);
  $("ourLevelCard").classList.toggle("active", activeSide === "our");
  $("opponentLevelCard").classList.toggle("active", activeSide === "opponent");
  $("currentLevelText").textContent = `当前级牌：${activeLabel ? `${activeLabel} ` : ""}${displayGuandanLevel(levels.current_level)}`;
}

function renderRulesDialog() {
  $("rulesDialogKicker").textContent = "本平台掼蛋桌固定配置";
  $("rulesDialogTitle").textContent = "掼蛋玩法与客户端说明";
  $("guandanRulesBody").hidden = false;
}

function turnCountdownElement(label) {
  const countdown = document.createElement("span");
  countdown.className = "turn-countdown human-turn-countdown";
  countdown.setAttribute("role", "timer");
  countdown.setAttribute("aria-label", label);
  const face = document.createElement("span");
  face.className = "turn-clock-face";
  const value = document.createElement("b");
  value.className = "turn-countdown-value";
  value.textContent = "20";
  face.appendChild(value);
  countdown.appendChild(face);
  return countdown;
}

function renderSeats(view) {
  const container = $("humanSeatLayer");
  container.replaceChildren();
  const spectator = isSpectatorRole(view.viewer.role);
  const terminal = isTerminalPhase(view.phase);
  const revealHands = spectator || terminal;
  const viewerSeat = spectator ? null : view.viewer.seat;
  const ownTurn = isOwnDecision(view);
  const compactIdentity = SOLO_MODE && !spectator && !terminal && !isLobbyPhase(view.phase) && view.phase !== "deal";
  const soloLobby = SOLO_MODE && !spectator && isLobbyPhase(view.phase);

  view.seats.forEach((seat) => {
    const node = document.createElement("section");
    const visualClass = seatVisualClass(view.game, seat.position, viewerSeat);
    const viewerOwnSeat = !spectator && seat.position === view.viewer.seat;
    const currentActor = seat.position === view.decision?.actor;
    // A seat's committed action stays visible while the other seats answer it.
    // Once play rotates back to that seat, its previous action belongs to the
    // completed cycle and must disappear before the new decision is rendered.
    // Keep this guard in the renderer as well as the room projection so an
    // older/stale view can never put the previous cards or PASS behind the
    // current turn clock.
    const visiblePlayArea = currentActor
      ? { cards: [], pass: false, label: "", type: "" }
      : compactIdentity ? currentPublicPlayArea(view, seat) : seat.play_area;
    node.className = `human-seat ${visualClass} kind-${safeClassName(seat.kind)}`;
    node.classList.toggle("viewer-seat", viewerOwnSeat);
    node.classList.toggle("terminal-reveal", revealHands && !spectator);
    node.classList.toggle("live-identity", compactIdentity);
    node.classList.toggle("current-actor", currentActor);
    node.classList.toggle("passed", visiblePlayArea.pass);
    node.classList.toggle("play-entering", state.playAnimationSeats.has(seat.position));
    node.dataset.position = String(seat.position);
    node.dataset.relation = Number.isInteger(view.viewer.seat) && seat.position % 2 === view.viewer.seat % 2 ? "partner" : "opponent";

    const tag = document.createElement("header");
    tag.className = "player-tag";
    const avatarUrl = (compactIdentity || soloLobby) && !viewerOwnSeat && seat.kind === "bot"
      ? HUMAN_AI_AVATARS[seat.position]
      : "";
    tag.classList.toggle("has-avatar", Boolean(avatarUrl));
    if (avatarUrl) {
      const avatar = document.createElement("img");
      avatar.className = "seat-avatar";
      avatar.src = avatarUrl;
      avatar.alt = "";
      avatar.width = 48;
      avatar.height = 48;
      avatar.draggable = false;
      avatar.setAttribute("aria-hidden", "true");
      tag.appendChild(avatar);
    }
    const number = document.createElement("span");
    number.className = "seat-number";
    number.textContent = String(seat.position + 1);
    const identity = document.createElement("div");
    identity.className = "player-identity";
    const name = document.createElement("strong");
    name.className = "player-name";
    if (seat.display_name && !(soloLobby && seat.kind === "bot")) name.dataset.l10nIgnore = "true";
    name.textContent = soloLobby && seat.kind === "bot"
      ? visualClass === "seat-top" ? "AI 队友" : visualClass === "seat-left" ? "左侧对手" : "右侧对手"
      : seatDisplayName(seat);
    const role = document.createElement("small");
    role.className = "player-role";
    role.textContent = seatRoleLabel(seat, viewerOwnSeat, view);
    identity.append(name, role);
    const status = document.createElement("b");
    const finishIndex = view.finish_order.indexOf(seat.position);
    const finishLabel = finishIndex >= 0
      ? finishPlaceLabel(finishIndex, view.seats.length)
      : "";
    status.className = `ready-mark${seat.ready ? " ready" : ""}`;
    status.classList.toggle("finish-place", Boolean(finishLabel));
    status.classList.toggle("low-card-count", compactIdentity && !finishLabel && (seat.hand_count === 1 || seat.hand_count === 2));
    const hideRemainingCount = compactIdentity && !viewerOwnSeat && !finishLabel && seat.hand_count > 10;
    status.hidden = hideRemainingCount;
    status.textContent = hideRemainingCount ? "" : finishLabel || (soloLobby ? seat.ready ? "已准备" : "未准备" : seatStatusLabel(seat, view.phase));
    if (finishLabel) status.setAttribute("aria-label", `${seatDisplayName(seat)}：${finishLabel}`);
    tag.append(number, identity, status);
    if (compactIdentity && currentActor && !viewerOwnSeat) {
      const turn = document.createElement("span");
      turn.className = "seat-turn-label";
      turn.textContent = "出牌中";
      tag.appendChild(turn);
    }

    if (canMoveToSeat(view, seat)) {
      const target = document.createElement("button");
      target.className = "seat-target";
      target.type = "button";
      target.setAttribute("aria-label", `换到 ${seat.position + 1} 号座位`);
      target.disabled = state.pendingMutation || interactionFrozen();
      target.addEventListener("click", () => moveToSeat(seat.position));
      tag.appendChild(target);
    }
    node.appendChild(tag);

    if (revealHands) {
      const hand = view.spectator_hands?.[seat.position] ?? [];
      if (hand.length) node.appendChild(spectatorHandElement(hand, view, { terminal }));
    } else if (!viewerOwnSeat && seat.hand_count > 0) {
      node.appendChild(opponentHandElement(seat.hand_count, { showCount: !compactIdentity }));
    }
    const playArea = playAreaElement(visiblePlayArea, view.game, view.level, compactIdentity ? HUMAN_CARD_VISUALS : {});
    const knownTrick = compactIdentity && view.current_trick?.mode !== "unknown" && Boolean(view.current_trick);
    const responseTarget = knownTrick && view.current_trick.target?.seat === seat.position;
    playArea.classList.toggle("response-target", responseTarget);
    playArea.classList.toggle("previous-play", knownTrick && !responseTarget && Boolean(visiblePlayArea.cards.length || visiblePlayArea.pass));
    if (responseTarget) {
      const badge = document.createElement("span");
      badge.className = "response-target-badge";
      badge.textContent = "当前压牌";
      playArea.appendChild(badge);
    }
    node.appendChild(playArea);
    if (currentActor && state.clock && (spectator || !viewerOwnSeat)) {
      const countdown = turnCountdownElement(`${seatDisplayName(seat)}的本回合剩余秒数`);
      countdown.dataset.actorSeat = String(seat.position);
      // The clock is part of this player's identity, not a free-floating table
      // layer. Keeping it inside the tag makes the actor-to-clock mapping
      // structural and prevents it from drifting over another seat's cards.
      tag.appendChild(countdown);
    }
    if (currentActor && ownTurn) node.setAttribute("aria-current", "true");
    container.appendChild(node);
    if (compactIdentity && state.playAnimationSeats.has(seat.position) && !visiblePlayArea.pass) {
      const cards = playArea.querySelector(".play-cards");
      const source = viewerOwnSeat ? $("ownHand") : tag;
      const scale = $("humanTableStage").getBoundingClientRect().width / HUMAN_TABLE_VIEWPORT.width;
      const vector = tableFlightVector(source.getBoundingClientRect(), cards.getBoundingClientRect(), scale);
      cards.style.setProperty("--flight-x", `${vector.x}px`);
      cards.style.setProperty("--flight-y", `${vector.y}px`);
      cards.classList.add("confirmed-flight");
    }
  });
  state.playAnimationSeats.clear();
}

function playAreaSignature(playArea) {
  return JSON.stringify({
    pass: Boolean(playArea?.pass),
    cards: (playArea?.cards ?? []).map((card) => card?.code ?? card),
    label: playArea?.label ?? "",
  });
}

function spectatorHandElement(hand, view, { terminal = false } = {}) {
  const container = document.createElement("div");
  container.className = "spectator-hand hand-zone";
  container.setAttribute("aria-label", `观战可见手牌，共 ${hand.length} 张`);
  if (terminal) {
    const count = document.createElement("span");
    count.className = "terminal-hand-count";
    count.textContent = `剩余 ${hand.length} 张`;
    container.appendChild(count);
  }
  const layout = document.createElement("div");
  layout.className = "hand-layout";
  const cards = hand.map((card) => typeof card === "object" ? card.code : card);
  renderHandLayout(layout, {
    version: "live-spectator-fallback",
    mode: "rank-groups",
    groups: fallbackHandGroups(cards),
  }, {
    game: view.game,
    levelRank: view.level ?? "",
  });
  container.appendChild(layout);
  return container;
}

function opponentHandElement(handCount, { showCount = true } = {}) {
  const container = document.createElement("div");
  container.className = "opponent-hand";
  if (!showCount) container.setAttribute("aria-hidden", "true");
  const visibleBacks = Math.min(7, Math.max(1, handCount));
  for (let index = 0; index < visibleBacks; index += 1) {
    const back = document.createElement("span");
    back.className = "card-back";
    back.setAttribute("aria-hidden", "true");
    container.appendChild(back);
  }
  if (showCount) {
    const count = document.createElement("b");
    count.className = "opponent-hand-count";
    count.textContent = `${handCount} 张`;
    container.appendChild(count);
  }
  return container;
}

function playAreaElement(playArea, game, levelRank, cardVisuals = {}) {
  const area = document.createElement("div");
  area.className = "seat-play-area";
  const label = document.createElement("strong");
  label.className = "play-label";
  const typeLabel = game === "guandan" ? guandanTypeLabel(playArea) : playArea.label;
  const rank = game === "guandan" ? guandanRankLabel(playArea) : String(playArea.rank ?? "").replace(/^T$/u, "10");
  label.textContent = playArea.pass ? "不出" : `${typeLabel}${rank ? ` · ${rank}` : ""}`;
  const caption = game === "guandan" ? wildcardCaption(playArea, levelRank) : "";
  if (caption) area.setAttribute("aria-label", caption);
  if (caption) label.title = caption;
  const cards = document.createElement("div");
  cards.className = "play-cards";
  const visibleCards = playArea.cards.slice(0, 12);
  area.classList.toggle("has-played-cards", visibleCards.length > 0);
  cards.dataset.cardCount = String(visibleCards.length);
  visibleCards.forEach((code) =>
    cards.appendChild(createPlayingCard(code, "table", { game, levelRank, ...cardVisuals })),
  );
  area.append(label, cards);
  return area;
}

function renderReadyControl(view, { spectator, lobby }) {
  const controls = $("humanRoomControls");
  const ownSeat = view.seats.find((seat) => seat.position === view.viewer.seat);
  controls.hidden = spectator || !lobby || !ownSeat;
  if (controls.hidden) return;
  const ready = ownSeat.ready;
  const preparation = soloLobbyPresentation(view, { pending: state.pendingMutation, frozen: interactionFrozen(), error: state.lobbyError });
  $("readyButton").classList.toggle("ready", ready);
  $("readyButton").disabled = SOLO_MODE ? preparation.disabled : state.pendingMutation || interactionFrozen();
  $("readyButton").setAttribute("aria-busy", String(state.pendingMutation));
  $("readyButtonText").textContent = SOLO_MODE ? preparation.button : ready ? "已准备 · 点击取消" : "点击准备";
  if (SOLO_MODE) {
    $("lobbyHelpText").textContent = preparation.status;
    $("lobbyHelp").classList.toggle("has-error", Boolean(state.lobbyError) && !state.pendingMutation);
  }
}

function renderOwnHand(view, spectator) {
  renderControlPreferences();
  const panel = $("ownHandPanel");
  const hasSeat = !spectator && Number.isInteger(view.viewer.seat);
  const terminal = isTerminalPhase(view.phase);
  const gameStarted = !isLobbyPhase(view.phase) && !["deal"].includes(String(view.phase).toLowerCase());
  // At settlement every seat is rendered by the same read-only reveal layer.
  // Keeping the interactive own-hand panel visible here duplicated the bottom
  // hand, consumed the only safe lower lane, and made its card scale differ
  // from the other three seats.
  panel.hidden = terminal || !hasSeat || (!gameStarted && view.own_hand.length === 0);
  $("humanTableStage").classList.toggle("own-hand-visible", !panel.hidden);
  if (panel.hidden) {
    closeExpandedHand();
    $("expandHandButton").disabled = true;
    cancelSelectionGesture();
    state.displayGroups = [];
    state.displayContext = "";
    $("ownHand").replaceChildren();
    renderArrangeToolbar(view);
    $("arrangeToolbar").hidden = true;
    $("ownTurnCountdown").hidden = true;
    $("ownTurnCountdown").removeAttribute("data-actor-seat");
    $("ownFinishBadge").hidden = true;
    $("ownFinishBadge").textContent = "";
    scheduleOwnHandSizing();
    return;
  }

  const ownSeat = view.seats.find((seat) => seat.position === view.viewer.seat);
  if (ownSeat) {
    $("ownSeatNumber").textContent = String(ownSeat.position + 1);
    $("ownSeatName").textContent = seatDisplayName(ownSeat);
    if (ownSeat.display_name) $("ownSeatName").dataset.l10nIgnore = "true";
    else delete $("ownSeatName").dataset.l10nIgnore;
    $("ownSeatRole").textContent = seatRoleLabel(ownSeat, true, view);
    $("ownHandRemaining").textContent = `${ownSeat.hand_count} 张手牌`;
    $("expandHandButton").setAttribute("aria-label", `展开选牌，${ownSeat.hand_count} 张手牌`);
    const finishIndex = view.finish_order.indexOf(ownSeat.position);
    const finishLabel = finishIndex >= 0
      ? finishPlaceLabel(finishIndex, view.seats.length)
      : "";
    $("ownFinishBadge").hidden = !finishLabel;
    $("ownFinishBadge").textContent = finishLabel;
    const ownClockVisible = ownSeat.position === view.decision?.actor && Boolean(state.clock);
    $("ownTurnCountdown").hidden = !ownClockVisible;
    if (ownClockVisible) {
      $("ownTurnCountdown").dataset.actorSeat = String(ownSeat.position);
    } else {
      $("ownTurnCountdown").removeAttribute("data-actor-seat");
    }
  } else {
    $("ownTurnCountdown").hidden = true;
    $("ownTurnCountdown").removeAttribute("data-actor-seat");
    $("ownFinishBadge").hidden = true;
    $("ownFinishBadge").textContent = "";
  }

  const selectable = canSelectOwnHand();
  const hand = $("ownHand");
  const layout = view.own_hand_layout;
  const rawGroups = layout?.groups?.length
    ? layout.groups
    : [{ pattern: 0, minor: 0, value: null, cards: view.own_hand }];
  const incomingGroups = state.handDisplayMode === "rank"
    ? rankDisplayGroups(view.own_hand, view.level)
    : layout?.fallback && rawGroups.length === 1 && rawGroups[0].cards.length > 8
    ? rawGroups[0].cards.map((card) => ({ pattern: 0, minor: 0, value: null, cards: [card] }))
    : rawGroups;
  const context = `${view.table_no}:${view.round_no}:${view.viewer.seat}:${view.game}`;
  const reset = state.resetHandLayout || state.displayContext !== context;
  const membershipChanged = state.displayGroups.flatMap(group => group.cards).map(card => card.id).sort().join("|") !== view.own_hand.map(card => card.id).sort().join("|");
  if (reset || membershipChanged || !selectable) cancelSelectionGesture();
  const groups = boundedHandColumns(stableHandGroups(state.displayGroups, incomingGroups, view.own_hand, { reset }));
  state.displayGroups = groups;
  state.displayContext = context;
  state.resetHandLayout = false;
  const focusId = document.activeElement?.closest?.(".hand-card-button")?.dataset.cardId;
  const scrollLeft = $("ownHandViewport").scrollLeft;
  const existingStacks = new Map([...hand.children].map(stack => [stack.dataset.groupKey, stack]));
  const existingCards = new Map([...hand.querySelectorAll(".hand-card-button")].map(button => [button.dataset.cardId, button]));
  hand.dataset.displayMode = state.handDisplayMode;
  hand.classList.toggle("fallback-layout", state.handDisplayMode !== "rank" && Boolean(layout?.fallback));
  groups.forEach((group, groupIndex) => {
    const stack = existingStacks.get(group.key) ?? document.createElement("div");
    stack.className = "own-meld-stack";
    stack.dataset.groupKey = group.key;
    stack.dataset.pattern = String(group.pattern ?? 0);
    stack.dataset.size = String(group.cards.length);
    stack.dataset.logicalKey = group.logicalKey;
    stack.dataset.columnIndex = String(group.columnIndex);
    stack.dataset.columnCount = String(group.columnCount);
    stack.classList.toggle("logical-start", group.columnCount > 1 && group.columnIndex === 0);
    stack.classList.toggle("logical-continuation", group.columnIndex > 0);
    stack.classList.toggle("logical-end", group.columnCount > 1 && group.columnIndex === group.columnCount - 1);
    stack.setAttribute("role", "group");
    stack.setAttribute("aria-label", `理牌分组 ${groupIndex + 1}，${group.cards.length} 张`);
    let groupButton = stack.querySelector(".select-group-button");
    if (group.selectionIds.length > 1 && group.columnIndex === 0) {
      if (!groupButton) {
        groupButton = document.createElement("button");
        groupButton.type = "button";
        groupButton.className = "select-group-button";
        stack.prepend(groupButton);
      }
      groupButton.dataset.groupKey = group.key;
      groupButton.dataset.groupIndex = String(groupIndex + 1);
      groupButton.disabled = !selectable;
      groupButton.setAttribute("aria-label", `选择整组 ${groupIndex + 1}，${group.selectionIds.length} 张`);
      groupButton.textContent = `${group.selectionIds.length} 张`;
    } else { groupButton?.remove(); }
    let continuation = stack.querySelector(".group-continuation-label");
    if (group.columnIndex > 0) {
      if (!continuation) {
        continuation = document.createElement("span");
        continuation.className = "group-continuation-label";
        continuation.setAttribute("aria-hidden", "true");
        stack.prepend(continuation);
      }
      continuation.textContent = "同组";
    } else { continuation?.remove(); }
    const wantedIds = new Set(group.cards.map(card => card.id));
    stack.querySelectorAll(".hand-card-button").forEach(button => { if (!wantedIds.has(button.dataset.cardId)) button.remove(); });
    group.cards.forEach((card, index) => {
      const button = existingCards.get(card.id) ?? ownHandCardButton(card, view, selectable);
      button.disabled = !selectable;
      if (button.dataset.visualKey !== `${view.game}:${card.code}:${view.level}`) {
        button.replaceChildren(createPlayingCard(card.code, "hand", { game: view.game, levelRank: view.level ?? "", ...HUMAN_CARD_VISUALS }));
        button.setAttribute("aria-label", `选择 ${card.code}`);
        button.dataset.visualKey = `${view.game}:${card.code}:${view.level}`;
      }
      const position = index + (group.selectionIds.length > 1 && group.columnIndex === 0 || group.columnIndex > 0 ? 1 : 0);
      if (stack.children[position] !== button) stack.insertBefore(button, stack.children[position] ?? null);
    });
    if (hand.children[groupIndex] !== stack) hand.insertBefore(stack, hand.children[groupIndex] ?? null);
  });
  for (const stack of [...hand.children]) if (!groups.some(group => group.key === stack.dataset.groupKey)) stack.remove();
  if (focusId) existingCards.get(focusId)?.isConnected && existingCards.get(focusId).focus({ preventScroll: true });
  $("ownHandViewport").scrollLeft = scrollLeft;
  renderArrangeToolbar(view);
  syncOwnHandSelection();
  scheduleOwnHandSizing();
}

function ownHandCardButton(card, view, selectable) {
  const button = document.createElement("button");
  button.className = "hand-card-button";
  button.type = "button";
  button.dataset.cardId = card.id;
  button.dataset.visualKey = `${view.game}:${card.code}:${view.level}`;
  button.classList.toggle("selected", state.selectedIds.has(card.id));
  button.setAttribute("aria-pressed", String(state.selectedIds.has(card.id)));
  button.setAttribute("aria-label", `选择 ${card.code}`);
  button.disabled = !selectable;
  button.appendChild(createPlayingCard(card.code, "hand", {
    game: view.game,
    levelRank: view.level ?? "",
    ...HUMAN_CARD_VISUALS,
  }));
  return button;
}

function renderArrangeToolbar(view) {
  const toolbar = $("arrangeToolbar");
  const available = view.own_hand.length > 0 && !isSpectatorRole(view.viewer.role) && !isTerminalPhase(view.phase);
  $("tableToolsArrange").hidden = !available;
  $("tableToolsActions").querySelector('[data-table-tool="hand"]').disabled = !canSelectOwnHand();
  $("tableToolsActions").querySelector('[data-table-tool="copy"]').disabled = !state.tableNo;
  toolbar.hidden = !available;
  if (!available) {
    setArrangeMenuOpen(false);
    return;
  }
  $("arrangeLabel").textContent = "理牌";
  $("arrangeCurrentMode").textContent = state.handDisplayMode === "rank" ? "牌型分组" : ARRANGE_MODE_LABELS[view.arrange_mode] ?? "选择方式";
  $("rankDisplayButton").setAttribute("aria-pressed", String(state.handDisplayMode === "rank"));
  $("rankDisplayButton").disabled = state.pendingArrange || Boolean(state.pendingAction) || interactionFrozen();
  $("arrangeMenuButton").disabled = state.pendingArrange || Boolean(state.pendingAction) || interactionFrozen();
  if (document.documentElement.classList.contains("fixed-aspect-table")) setArrangeMenuOpen(false);
  document.querySelectorAll("button[data-arrange-mode]").forEach((button) => {
    const mode = Number(button.dataset.arrangeMode);
    const active = state.handDisplayMode === "arranged" && mode === view.arrange_mode;
    button.setAttribute("aria-pressed", String(active));
    const variantCount = view.own_hand_layout?.variant_count ?? 1;
    const variantIndex = (view.own_hand_layout?.variant_index ?? 0) + 1;
    if (active && variantCount > 1) {
      button.dataset.variantCount = `方案 ${variantIndex}/${variantCount} ↻`;
      button.setAttribute("aria-label", `${ARRANGE_MODE_LABELS[mode]}，方案 ${variantIndex}/${variantCount}，重复点击切换方案`);
    } else {
      delete button.dataset.variantCount;
      button.setAttribute("aria-label", ARRANGE_MODE_LABELS[mode]);
    }
    button.disabled = state.pendingArrange || Boolean(state.pendingAction) || interactionFrozen();
  });
  $("arrangeHint").textContent = state.pendingArrange
    ? "正在重新分组；已选牌保持不变…"
    : view.own_hand_layout?.fallback
      ? "理牌服务暂不可用，当前按安全回退顺序显示；规则手牌未改变。"
      : "重复点击当前方式切换方案";
}

function setArrangeMenuOpen(open, { restoreFocus = false } = {}) {
  const toolbar = $("arrangeToolbar");
  const trigger = $("arrangeMenuButton");
  const controls = $("guandanArrangeControls");
  if (document.documentElement.classList.contains("fixed-aspect-table")) {
    trigger.hidden = true;
    controls.hidden = toolbar.hidden;
    toolbar.classList.remove("menu-open");
    trigger.setAttribute("aria-expanded", "false");
    return;
  }
  const nextOpen = Boolean(open && !toolbar.hidden && !trigger.disabled);
  toolbar.classList.toggle("menu-open", nextOpen);
  trigger.setAttribute("aria-expanded", String(nextOpen));
  controls.hidden = !nextOpen;
  if (nextOpen) {
    window.requestAnimationFrame(() => {
      (controls.querySelector('[aria-pressed="true"]') || controls.querySelector("button"))?.focus({ preventScroll: true });
    });
  } else if (restoreFocus) {
    trigger.focus({ preventScroll: true });
  }
}

function renderActionControls(view) {
  $("actionDock").hidden = view.phase === "paused" || view.phase === "error";
  const ownPlayTurn = isOwnPlayDecision(view);
  const pending = Boolean(state.pendingAction);
  $("selectedCount").textContent = `${state.selectedIds.size} 张已选`;
  $("actionDock").classList.toggle("pending", pending);
  $("actionDock").setAttribute("aria-busy", String(pending));
  $("playButton").setAttribute("aria-busy", String(pending));
  const uncertain = Boolean(state.pendingAction?.uncertain);
  $("reconcileActionButton").hidden = !uncertain;
  $("reconcileActionButton").disabled = interactionFrozen();
  renderHandSpreadControl();
  $("playActionButtons").hidden = false;
  $("passButton").disabled = !ownPlayTurn || !view.decision?.can_pass || pending || state.pendingArrange || interactionFrozen();
  const feedback = selectionFeedback(view, state.selectedIds);
  $("playButton").disabled = !ownPlayTurn || feedback.matches.length === 0 || pending || state.pendingArrange || interactionFrozen();
  $("clearSelectionButton").disabled = !canSelectOwnHand() || state.selectedIds.size === 0;
  $("swipeSelectButton").disabled = !canSelectOwnHand();
  $("expandHandButton").disabled = !canSelectOwnHand();
  if (!canSelectOwnHand()) state.swipeSelection = false;
  $("swipeSelectButton").setAttribute("aria-pressed", String(state.swipeSelection));
  $("ownHandWorkspace").classList.toggle("swipe-selection-enabled", state.swipeSelection);
  const status = $("selectionStatus");
  status.dataset.kind = state.actionError ? "error" : canSelectOwnHand() ? feedback.kind : "empty";
  const feedbackText = view.phase === "paused" ? "牌局已暂停，可重试 AI 或返回入口。"
    : view.phase === "error" ? "本局无法继续，请返回入口创建新桌。"
    : uncertain ? (interactionFrozen()
    ? "未收到出牌确认，恢复连接后继续。" : "未收到出牌确认，可点击「核对出牌」。")
    : pending ? "正在确认出牌…"
    : interactionFrozen() ? "连接恢复后可继续选牌"
    : state.actionError?.text ? state.actionError.text
    : state.pendingArrange ? "正在重新分组…"
    : state.selectedIds.size ? selectionExplanation(view, state.selectedIds)
    : !ownPlayTurn ? (canSelectOwnHand() ? "可预选手牌，轮到你时校验" : "轮到你后可选牌")
    : state.swipeSelection ? "滑动选牌；再次点击滑选可恢复浏览"
    : "点选手牌，也可整组选择";
  // Avoid re-announcing identical polite feedback on repeated state snapshots.
  if (status.dataset.feedbackText !== feedbackText) {
    status.dataset.feedbackText = feedbackText;
    status.textContent = feedbackText;
  }
  $("actionDock").classList.toggle("own-turn", ownPlayTurn);
  if (view.phase === "paused" || view.phase === "error") {
    $("turnHint").textContent = view.phase === "paused" ? "AI 暂时不可用" : "牌局无法继续";
  } else if (pending) {
    $("turnHint").textContent = "已提交，等待服务器确认…";
  } else if (ownPlayTurn) {
    $("turnHint").textContent = view.decision?.can_pass
      ? "轮到你：可出牌或不出"
      : "你先出：请选择牌";
  } else if (isTerminalPhase(view.phase)) {
    $("turnHint").textContent = "本局已结束；所有人类玩家准备后开始下一局";
  } else {
    const actor = view.seats.find((seat) => seat.position === view.decision?.actor);
    $("turnHint").textContent = actor
      ? `等待 ${seatDisplayName(actor)}出牌`
      : "等待下一个决策";
  }
}

function toggleCardSelection(cardId) {
  if (!canSelectOwnHand()) return;
  state.selectedIds = toggleSelectionEntries(state.selectedIds, [cardId]);
  state.actionError = null;
  syncOwnHandSelection();
}

function clearHandSelection() {
  if (!canSelectOwnHand()) return;
  cancelSelectionGesture();
  state.selectedIds.clear();
  state.actionError = null;
  syncOwnHandSelection();
}

function beginSelectionGesture(event) {
  // A fresh down is a new intentional interaction, not a cancelled pointer's release.
  state.cancelledSelectionPointerId = null;
  if ((event.pointerType !== "mouse" && !state.swipeSelection) || event.button !== 0 || event.isPrimary === false) return;
  const button = handCardButtonFromEvent(event);
  if (!button || button.disabled || !canSelectOwnHand()) return;
  event.preventDefault();
  cancelSelectionGesture();
  const startCardId = button.dataset.cardId;
  state.selectionGesture = {
    ...beginHandSweep(state.selectedIds, startCardId),
    pointerId: event.pointerId,
    decisionId: state.view.decision?.id ?? "",
    tableNo: state.view.table_no,
    roundNo: state.view.round_no,
    viewerSeat: state.view.viewer.seat,
    game: state.view.game,
    handIds: state.view.own_hand.map(card => card.id),
    handCodes: state.view.own_hand.map(card => card.code),
  };
  $("ownHand").classList.add("sweep-selecting");
  button.focus({ preventScroll: true });
  $("ownHand").setPointerCapture?.(event.pointerId);
  state.selectedIds = paintHandSweep(state.selectionGesture, state.selectedIds, startCardId);
  state.actionError = null;
  syncOwnHandSelection();
}

function extendSelectionGesture(event) {
  const gesture = state.selectionGesture;
  if (!gesture || event.pointerId !== gesture.pointerId) return;
  if (!(event.buttons & 1) || !canSelectOwnHand()) {
    finishSelectionGesture(event);
    return;
  }
  const button = handCardButtonFromPoint(event);
  const cardId = button?.dataset.cardId;
  if (!button || button.disabled || !cardId || gesture.visitedIds.has(cardId)) return;
  state.selectedIds = paintHandSweep(gesture, state.selectedIds, cardId);
  state.actionError = null;
  syncOwnHandSelection();
}

function finishSelectionGesture(event) {
  const gesture = state.selectionGesture;
  if (!gesture || (event?.pointerId !== undefined && event.pointerId !== gesture.pointerId)) return;
  state.suppressSelectionClick = true;
  if (state.suppressSelectionClickTimer !== null) {
    window.clearTimeout(state.suppressSelectionClickTimer);
  }
  state.suppressSelectionClickTimer = window.setTimeout(() => {
    state.suppressSelectionClick = false;
    state.suppressSelectionClickTimer = null;
  }, 0);
  state.selectionGesture = null;
  if ($("ownHand").hasPointerCapture?.(gesture.pointerId)) {
    $("ownHand").releasePointerCapture(gesture.pointerId);
  }
  $("ownHand").classList.remove("sweep-selecting");
}

function cancelSelectionGesture(event, { restoreSelection = true } = {}) {
  const gesture = state.selectionGesture;
  if (!gesture) {
    $("ownHand")?.classList.remove("sweep-selecting");
    return;
  }
  if (event?.pointerId !== undefined && event.pointerId !== gesture.pointerId) return;
  const pointerId = gesture?.pointerId;
  state.cancelledSelectionPointerId = pointerId;
  state.selectionGesture = null;
  if (pointerId !== undefined && $("ownHand")?.hasPointerCapture?.(pointerId)) {
    $("ownHand").releasePointerCapture(pointerId);
  }
  $("ownHand")?.classList.remove("sweep-selecting");
  if (restoreSelection && gesture?.baselineSelectedIds) {
    state.selectedIds = new Set(gesture.baselineSelectedIds);
    syncOwnHandSelection();
  }
}

function handleOwnHandClick(event) {
  if (state.cancelledSelectionPointerId !== null && event.detail > 0) {
    state.cancelledSelectionPointerId = null;
    event.preventDefault();
    event.stopPropagation();
    return;
  }
  if (state.suppressSelectionClick) {
    event.preventDefault();
    event.stopPropagation();
    state.suppressSelectionClick = false;
    if (state.suppressSelectionClickTimer !== null) {
      window.clearTimeout(state.suppressSelectionClickTimer);
      state.suppressSelectionClickTimer = null;
    }
    return;
  }
  const groupButton = event.target.closest?.(".select-group-button");
  if (groupButton && !groupButton.disabled && canSelectOwnHand()) {
    const group = state.displayGroups.find(item => item.key === groupButton.dataset.groupKey);
    if (group) {
      state.selectedIds = toggleGroupSelection(state.selectedIds, group.selectionIds);
      state.actionError = null;
    }
    syncOwnHandSelection();
    return;
  }
  const button = handCardButtonFromEvent(event);
  if (!button || button.disabled) return;
  toggleCardSelection(button.dataset.cardId);
}

function handCardButtonFromEvent(event) {
  const button = event?.target?.closest?.(".hand-card-button");
  return button && $("ownHand").contains(button) ? button : null;
}

function handCardButtonFromPoint(event) {
  if (!Number.isFinite(event?.clientX) || !Number.isFinite(event?.clientY)) {
    return handCardButtonFromEvent(event);
  }
  const hand = $("ownHand");
  const node = document.elementFromPoint(event.clientX, event.clientY);
  const button = node?.closest?.(".hand-card-button");
  return button && hand.contains(button) && !button.disabled ? button : null;
}

function canSelectOwnHand() {
  return canPrepareOwnHand(state.view, { pendingAction: state.pendingAction,
    pendingArrange: state.pendingArrange, interactionFrozen: interactionFrozen() });
}

function syncOwnHandSelection() {
  $("ownHand").querySelectorAll(".hand-card-button").forEach((button) => {
    const selected = state.selectedIds.has(button.dataset.cardId);
    button.classList.toggle("selected", selected);
    button.setAttribute("aria-pressed", String(selected));
  });
  $("ownHand").querySelectorAll(".select-group-button").forEach(button => {
    const group = state.displayGroups.find(item => item.key === button.dataset.groupKey);
    const feedback = groupSelectionFeedback(state.selectedIds, group?.selectionIds ?? []);
    button.setAttribute("aria-pressed", feedback.pressed);
    button.textContent = feedback.pressed === "mixed" ? `${feedback.selected}/${feedback.total} 张` : `${feedback.total} 张`;
    button.setAttribute("aria-label", `选择整组 ${button.dataset.groupIndex}，${feedback.total} 张${feedback.pressed === "mixed" ? `，已选 ${feedback.selected} 张` : ""}`);
  });
  $("ownHand").querySelectorAll(".own-meld-stack").forEach(stack => {
    const group = state.displayGroups.find(item => item.key === stack.dataset.groupKey);
    stack.classList.toggle("logical-selected", Boolean(group?.selectionIds.every(id => state.selectedIds.has(id))));
  });
  if (state.view) renderActionControls(state.view);
  syncExpandedHandSelection();
}

function currentHandInspectionSignature() {
  return handInspectionSignature(state.view, state.displayGroups, {
    pendingAction: state.pendingAction, pendingArrange: state.pendingArrange, interactionFrozen: interactionFrozen(),
  });
}

function closeExpandedHand() {
  state.expandedHandSignature = null;
  if ($("expandedHandDialog").open) $("expandedHandDialog").close();
}

function openExpandedHand() {
  const signature = currentHandInspectionSignature();
  if (!signature) return;
  cancelSelectionGesture();
  state.expandedHandSignature = signature;
  const sections = state.displayGroups.map((group, index) => {
    const section = document.createElement("section");
    section.className = "expanded-hand-group";
    section.style.setProperty("--expanded-group-width", `${Math.max(180, group.cards.length * 70 - 2)}px`);
    const heading = document.createElement("div");
    heading.className = "expanded-group-heading";
    const title = document.createElement("strong");
    title.id = `expanded-hand-group-${index}`;
    title.textContent = `分组 ${index + 1} · ${group.cards.length} 张`;
    section.setAttribute("aria-labelledby", title.id);
    const select = document.createElement("button");
    select.type = "button";
    select.className = "expanded-group-select";
    select.dataset.groupKey = group.key;
    select.dataset.groupIndex = String(index + 1);
    select.textContent = "整组选择";
    heading.append(title, select);
    const list = document.createElement("div");
    list.className = "expanded-group-cards";
    for (const card of group.cards) {
      const button = ownHandCardButton(card, state.view, true);
      button.className = "expanded-hand-card";
      list.append(button);
    }
    section.append(heading, list);
    return section;
  });
  $("expandedHandGroups").replaceChildren(...sections);
  syncExpandedHandSelection();
  openDialog($("expandedHandDialog"));
}

function syncExpandedHandSelection() {
  if (!state.expandedHandSignature) return;
  const signature = currentHandInspectionSignature();
  if (!signature || state.expandedHandSignature !== signature) {
    closeExpandedHand();
    return;
  }
  $("expandedHandGroups").querySelectorAll(".expanded-hand-card").forEach(button => {
    const selected = state.selectedIds.has(button.dataset.cardId);
    button.classList.toggle("selected", selected);
    button.setAttribute("aria-pressed", String(selected));
  });
  $("expandedHandGroups").querySelectorAll(".expanded-group-select").forEach(button => {
    const group = state.displayGroups.find(item => item.key === button.dataset.groupKey);
    const feedback = groupSelectionFeedback(state.selectedIds, group.selectionIds);
    button.setAttribute("aria-pressed", feedback.pressed);
    button.setAttribute("aria-label", `选择整组 ${button.dataset.groupIndex}，${feedback.total} 张，已选 ${feedback.selected} 张`);
  });
  $("expandedHandStatus").textContent = `已选 ${state.selectedIds.size} 张 · 完成后回到牌桌出牌`;
}

function prepareSelectedPlay() {
  const view = state.view;
  if (!isOwnPlayDecision(view) || state.pendingAction || state.pendingArrange || interactionFrozen()) return;
  const selectedCards = selectedCardCodes(view.own_hand, state.selectedIds);
  // A stale/unknown physical ID must not silently disappear from a submission.
  if (selectedCards.length !== state.selectedIds.size) return;
  const matches = matchingActionsForSelection(
    view.decision.actions,
    view.own_hand,
    state.selectedIds,
    view.game,
  );
  if (matches.length === 0) {
    showToast("这组选牌不在服务器给出的合法动作中。", "warning");
    return;
  }
  if (matches.length === 1) {
    submitLegalAction(matches[0], { selectedCards });
    return;
  }
  showActionChoices(matches, selectedCards);
}

function showActionChoices(actions, selectedCards = []) {
  const context = actionChoiceContext(state.view);
  if (!context) return;
  state.actionChoices = { context, selection: JSON.stringify([...state.selectedIds].sort()) };
  const list = $("actionChoiceList");
  list.replaceChildren();
  actions.forEach((action, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = actionChoiceLabel(state.view?.game === "guandan" ? { ...action, label: guandanTypeLabel(action) } : action, index);
    button.addEventListener("click", () => {
      if (!actionChoicesAreCurrent() || !isCurrentActionChoice(state.view, action, context)) { closeActionChoices(); return; }
      closeActionChoices();
      submitLegalAction(action, { selectedCards, context });
    });
    list.appendChild(button);
  });
  openDialog($("actionDialog"));
}

function actionChoicesAreCurrent() {
  return Boolean(state.actionChoices && !state.pendingAction && !state.pendingArrange && !interactionFrozen()
    && state.actionChoices.context === actionChoiceContext(state.view)
    && state.actionChoices.selection === JSON.stringify([...state.selectedIds].sort()));
}

function closeActionChoices() {
  state.actionChoices = null;
  // Clear handlers immediately; the native close event is queued asynchronously.
  $("actionChoiceList").replaceChildren();
  if ($("actionDialog").open) $("actionDialog").close();
}

function submitPass() {
  const decision = state.view?.decision;
  if (!isOwnPlayDecision(state.view) || !decision?.can_pass || state.pendingAction || state.pendingArrange || interactionFrozen()) return;
  const passAction = decision.actions.find((action) =>
    action.type.toUpperCase() === "PASS" || action.label.toUpperCase() === "PASS",
  );
  if (!passAction) {
    showToast("服务器没有提供“不出”的动作索引，未提交。", "error");
    return;
  }
  submitLegalAction(passAction);
}

async function submitLegalAction(action, { selectedCards = [], context = actionChoiceContext(state.view) } = {}) {
  const view = state.view;
  if (!state.token || !isCurrentActionChoice(view, action, context, {
    pendingAction: state.pendingAction, pendingArrange: state.pendingArrange, interactionFrozen: interactionFrozen(),
  })) return;
  if (action.cards?.length && selectedCardCodes(view.own_hand, state.selectedIds).length !== state.selectedIds.size) return;
  // Plays must still correspond to exactly the physical selection shown now.
  if (action.cards?.length && !matchingActionsForSelection(view.decision.actions, view.own_hand, state.selectedIds, view.game)
    .some(current => JSON.stringify(current) === JSON.stringify(action))) return;
  const request = {
    action_index: action.action_index,
    expected_version: view.version,
    decision_id: view.decision.id,
    client_action_id: newClientActionId(),
  };
  state.pendingAction = {
    decisionId: view.decision.id,
    version: view.version,
    clientActionId: request.client_action_id,
    selectedIds: action.cards?.length ? [...state.selectedIds] : [],
    request,
    tableNo: state.tableNo,
    token: state.token,
  };
  state.actionError = null;
  renderOwnHand(view, false);
  await sendPendingAction(state.pendingAction);
}

async function sendPendingAction(pending) {
  try {
    const response = await api.submitAction(pending.tableNo, pending.token, pending.request);
    if (state.pendingAction !== pending || state.tableNo !== pending.tableNo) return;
    applyResponseSnapshot(response);
    // HTTP success alone is not authoritative play confirmation. An unreadable
    // body or stale snapshot must retain the same key and offer reconciliation.
    if (state.pendingAction === pending) await reconcilePendingAction(pending);
  } catch (error) {
    if (state.pendingAction !== pending || state.tableNo !== pending.tableNo
      || state.view?.decision?.id !== pending.decisionId) return;
    if (!error?.status || error.status >= 500) {
      await reconcilePendingAction(pending);
      return;
    }
    state.pendingAction = null;
    state.actionError = { text: error.status === 409
      ? "桌面已变化，正在更新，请重新确认选牌。"
      : "出牌未成功，选牌已保留，可重新提交。" };
    renderOwnHand(state.view, false);
    if (error.status === 409) await refreshAuthoritativeView();
  }
}

async function reconcilePendingAction(pending) {
  if (state.pendingAction !== pending || state.tableNo !== pending.tableNo) return;
  pending.uncertain = true;
  state.actionError = { text: "未收到出牌确认，正在核对桌面…" };
  renderOwnHand(state.view, false);
  await refreshAuthoritativeView();
}

async function moveToSeat(position) {
  // The public service fixes the human at seat 0; no seat mutation route exists.
  return;
}

async function retryRoom() {
  const view = state.view;
  if (!state.token || view?.phase !== "paused" || !view.failure?.retryable
    || state.pendingMutation || state.pendingAction || interactionFrozen()) return;
  const tableNo = state.tableNo, token = state.token;
  state.pendingMutation = true;
  state.retryError = "";
  renderRoom();
  try {
    const response = await api.retryTable(tableNo, token, {expected_version:view.version});
    if (state.tableNo !== tableNo || state.token !== token) return;
    applyResponseSnapshot(response);
  } catch (error) {
    if (state.tableNo !== tableNo || state.token !== token) return;
    state.retryError = "尚未恢复，请检查 AI 服务后再次重试。";
    // Outcome may be unknown: refresh before permitting another versioned retry.
    await refreshAuthoritativeView();
  } finally {
    if (state.tableNo === tableNo && state.token === token) {
      state.pendingMutation = false;
      renderRoom();
    }
  }
}

async function toggleReady() {
  const view = state.view;
  const ownSeat = view?.seats.find((seat) => seat.position === view.viewer.seat);
  if (!ownSeat || !isReadyPhase(view.phase) || isSpectatorRole(view.viewer.role) || state.pendingMutation || interactionFrozen()) return;
  const tableNo = state.tableNo, token = state.token;
  state.lobbyError = "";
  state.pendingMutation = true;
  renderRoom();
  try {
    const response = await api.setReady(tableNo, token, {
      ready: !ownSeat.ready,
      expected_version: view.version,
    });
    if (state.tableNo !== tableNo || state.token !== token) return;
    applyResponseSnapshot(response);
  } catch (error) {
    if (state.tableNo !== tableNo || state.token !== token) return;
    state.lobbyError = "准备未成功，请重试。";
    showToast(errorMessage(error), "error");
    if (error?.status === 409) refreshAuthoritativeView();
  } finally {
    if (state.tableNo === tableNo && state.token === token) {
      state.pendingMutation = false;
      renderRoom();
    }
  }
}

async function arrangeOwnHand(mode) {
  const view = state.view;
  if (
    view?.game !== "guandan" ||
    !Number.isInteger(view.viewer?.seat) ||
    ![1, 2, 3, 4].includes(mode) ||
    state.pendingArrange ||
    state.pendingAction ||
    interactionFrozen()
  ) return;
  const tableNo = state.tableNo, token = state.token;
  state.pendingArrange = true;
  renderOwnHand(view, false);
  try {
    const response = await api.arrangeHand(tableNo, token, { mode });
    if (state.tableNo !== tableNo || state.token !== token) return;
    state.handDisplayMode = "arranged";
    state.resetHandLayout = true;
    applyResponseSnapshot(response);
  } catch (error) {
    if (state.tableNo !== tableNo || state.token !== token) return;
    showToast(errorMessage(error), "error");
  } finally {
    if (state.tableNo === tableNo && state.token === token) {
      state.pendingArrange = false;
      if (state.view) renderOwnHand(state.view, false);
    }
  }
}

async function refreshAuthoritativeView() {
  const tableNo = state.tableNo, token = state.token;
  try {
    const response = await api.viewTable(tableNo, token);
    if (state.tableNo !== tableNo || state.token !== token) return;
    applyRoomView(extractView(response, tableNo));
  } catch (error) {
    if (state.tableNo !== tableNo || state.token !== token) return;
    showToast(errorMessage(error), "error");
  }
}

function applyResponseSnapshot(response) {
  const candidate = response?.view ?? response;
  if (candidate?.seats && candidate?.version !== undefined) {
    applyRoomView(extractView(candidate, state.tableNo));
  }
}

function updateCountdownDisplay() {
  if (interactionFrozen()) {
    document.querySelectorAll(".human-turn-countdown").forEach((node) => {
      const value = node.querySelector(".turn-countdown-value");
      if (value) value.textContent = "—";
      node.classList.add("frozen");
      node.classList.remove("urgent");
    });
    return;
  }
  renderCountdownTick(state.clock, Date.now(), (value) => {
    document.querySelectorAll(".human-turn-countdown").forEach((node) => {
      const clockValue = node.querySelector(".turn-countdown-value");
      if (clockValue) clockValue.textContent = value === null ? "" : String(value);
      node.classList.toggle("elapsed", value === 0);
      node.classList.toggle("urgent", value !== null && value > 0 && value <= 5);
      node.classList.remove("frozen");
    });
  });
}

function renderResult(view) {
  const terminal = isTerminalPhase(view.phase);
  const seriesFinished = Boolean(
    view.game === "guandan" &&
      view.phase === "game_over" &&
      view.series?.match_finished &&
      view.series?.final_result,
  );
  $("resultBanner").hidden = seriesFinished || (!terminal && view.result === null);
  if (!$("resultBanner").hidden) {
    $("resultEyebrow").textContent = `第 ${view.round_no} 局结果`;
    $("resultText").textContent = resultSummary(view.result, view);
    renderResultDetails(view);
  } else {
    clearResultDetails();
  }
  renderSeriesResult(view, seriesFinished);
  const readyButton = $("resultReadyButton");
  const ownSeat = view.seats.find((seat) => seat.position === view.viewer.seat);
  readyButton.hidden = seriesFinished || !terminal || isSpectatorRole(view.viewer.role) || !ownSeat;
  if (!readyButton.hidden) {
    readyButton.textContent = state.pendingMutation ? "正在开始下一局…"
      : ownSeat.ready ? "已准备 · 等待其他玩家" : SOLO_MODE ? "继续下一局" : "准备下一局";
    readyButton.setAttribute("aria-busy", String(state.pendingMutation));
    readyButton.classList.toggle("ready", ownSeat.ready);
    readyButton.disabled = state.pendingMutation || interactionFrozen();
  }
}

function renderReplayLinks(view) {
  if (SOLO_MODE) {
    $("latestReplayLink").hidden = true;
    $("resultReplayLink").hidden = true;
    $("roomReviewLink").hidden = true;
    return;
  }
  const history = Array.isArray(view.replay_history) ? view.replay_history : [];
  const latestHistory = history.length ? history[history.length - 1] : null;
  const currentRoundReplay = [...history]
    .reverse()
    .find((item) => item.round_no === view.round_no);
  const latestMatchId = view.latest_replay_match_id || latestHistory?.match_id || "";
  setReplayLink(
    $("latestReplayLink"),
    latestMatchId,
    latestHistory ? `回放第 ${latestHistory.round_no} 局` : "最近回放",
  );
  setReplayLink($("resultReplayLink"), currentRoundReplay?.match_id, "查看本局回放");
  const reviewLink = $("roomReviewLink");
  reviewLink.href = reviewHref(latestMatchId);
  reviewLink.title = latestMatchId
    ? `打开牌局 ${latestMatchId} 的人工评测`
    : "打开人工评测牌局列表";
}

function setReplayLink(link, matchId, label) {
  const replayId = String(matchId ?? "");
  link.hidden = !replayId;
  link.textContent = label;
  if (!replayId) {
    link.removeAttribute("href");
    link.removeAttribute("title");
    return;
  }
  link.href = `./index.html?match=${encodeURIComponent(replayId)}`;
  link.title = `打开回放 ${replayId}`;
}

function renderResultDetails(view) {
  const result = view.result;
  if (!result || typeof result !== "object" || Array.isArray(result)) {
    clearResultDetails();
    return;
  }

  const winningSide = resultWinningSideLabel(result, view);
  $("resultWinningSide").hidden = !winningSide;
  $("resultWinningSide").textContent = winningSide;
  renderResultRoundMeta(result, view);

  const order = Array.isArray(result.finish_order) ? result.finish_order : [];
  const finishList = $("resultFinishOrder");
  finishList.replaceChildren();
  order.forEach((position, index) => {
    const seat = view.seats.find((candidate) => candidate.position === position);
    if (!seat) return;
    const item = document.createElement("li");
    const place = document.createElement("span");
    place.textContent = finishPlaceLabel(index, order.length);
    const seatNo = document.createElement("b");
    seatNo.textContent = `${position + 1} 号位`;
    const name = document.createElement("small");
    if (seat.display_name) name.dataset.l10nIgnore = "true";
    name.textContent = seatDisplayName(seat);
    item.append(place, seatNo, name);
    finishList.appendChild(item);
  });
  $("resultOrderSection").hidden = finishList.children.length === 0;

}

function guandanTeamLabel(team) {
  return team === "even" ? "1、3 号位队" : team === "odd" ? "2、4 号位队" : "未知队伍";
}

function renderSeriesResult(view, visible) {
  const overlay = $("seriesResultOverlay");
  overlay.hidden = !visible;
  if (!visible) {
    $("seriesReplayList").hidden = true;
    $("seriesReplayList").replaceChildren();
    $("seriesLastRoundMeta").hidden = true;
    $("seriesLastRoundMeta").classList.remove("won", "lost");
    return;
  }
  const result = view.series.final_result;
  const viewerTeam = view.series.viewer_team;
  const spectator = isSpectatorRole(view.viewer.role);
  const viewerWon = !spectator && viewerTeam === result.winner_team;
  $("seriesResultTitle").textContent = spectator
    ? `${guandanTeamLabel(result.winner_team)}获胜`
    : viewerWon ? "胜利" : "惜败";
  $("seriesResultEyebrow").textContent = `第 ${result.series_no} 场 · 掼蛋整场结算`;
  $("seriesResultSubtitle").textContent = `第 ${result.winning_a_attempt_no} 次过 A 成功`;
  renderSeriesLastRoundMeta(view);
  $("seriesWinnerLevel").textContent = "打 A · 过 A 成功";
  $("seriesLoserLevel").textContent = `最终打到 ${displayGuandanLevel(result.losing_team_level)}`;
  renderSeriesPlayers($("seriesWinnerPlayers"), view, result, result.winner_team);
  renderSeriesPlayers($("seriesLoserPlayers"), view, result, result.loser_team);
  const finalOrder = result.last_finish_order
    .map((position, index) => `${finishPlaceLabel(index, 4)} ${position + 1}号位`)
    .join(" · ");
  $("seriesResultMeta").textContent = `整场共 ${result.rounds_played} 副 · 最后一副：${finalOrder}`;
  const rematch = $("seriesRematchButton");
  const ownSeat = view.seats.find((seat) => seat.position === view.viewer.seat);
  rematch.hidden = spectator || !ownSeat;
  rematch.disabled = state.pendingMutation || interactionFrozen();
  rematch.textContent = ownSeat?.ready ? "已准备 · 等待其他玩家" : "再来一场";
  rematch.classList.toggle("ready", Boolean(ownSeat?.ready));
  renderSeriesReplayLinks(view, result);
}

function renderSeriesLastRoundMeta(view) {
  const meta = $("seriesLastRoundMeta");
  const presentation = guandanRoundPresentation(view.result, view.viewer.seat);
  meta.hidden = !presentation;
  if (!presentation) return;
  meta.classList.toggle("won", presentation.won);
  meta.classList.toggle("lost", !presentation.won);
  $("seriesLastRoundDelta").textContent = `${presentation.sideLabel}${presentation.outcomeLabel} · ${presentation.signedDelta}`;
  $("seriesLastRoundLevels").textContent = `${presentation.levelChange} · ${presentation.opponentLevel}`;
  $("seriesLastRoundNext").textContent = presentation.nextLevel;
}

function renderSeriesPlayers(container, view, result, team) {
  container.replaceChildren();
  const positions = team === "even" ? [0, 2] : [1, 3];
  positions.forEach((position) => {
    const seat = view.seats.find((candidate) => candidate.position === position);
    if (!seat) return;
    const card = document.createElement("article");
    card.className = "series-player-card";
    const identity = document.createElement("div");
    const seatNo = document.createElement("span");
    seatNo.textContent = `${position + 1} 号位`;
    const playerName = document.createElement("strong");
    playerName.textContent = seatDisplayName(seat);
    if (seat.display_name) playerName.dataset.l10nIgnore = "true";
    identity.append(seatNo, playerName);
    const stats = document.createElement("dl");
    const headLabel = document.createElement("dt");
    headLabel.textContent = "头游次数";
    const headValue = document.createElement("dd");
    headValue.textContent = String(result.head_finish_count_by_seat[position] ?? 0);
    const winLabel = document.createElement("dt");
    winLabel.textContent = "队伍获胜次数";
    const winValue = document.createElement("dd");
    winValue.textContent = String(result.team_victory_count[team] ?? 0);
    stats.append(headLabel, headValue, winLabel, winValue);
    card.append(identity, stats);
    container.appendChild(card);
  });
}

function renderSeriesReplayLinks(view, result) {
  const container = $("seriesReplayList");
  container.replaceChildren();
  const historyById = new Map(
    view.replay_history
      .filter((item) => item.series_no === result.series_no)
      .map((item) => [item.match_id, item]),
  );
  result.replay_ids.forEach((matchId, index) => {
    const item = historyById.get(matchId);
    const link = document.createElement("a");
    link.href = `./index.html?match=${encodeURIComponent(matchId)}`;
    link.target = "_blank";
    link.rel = "noopener";
    link.textContent = `第 ${item?.round_no ?? index + 1} 副回放`;
    container.appendChild(link);
  });
  $("seriesReplayButton").disabled = container.children.length === 0;
}

function toggleSeriesReplayList() {
  const list = $("seriesReplayList");
  if (!list.children.length) {
    showToast("本场暂时没有可用回放。", "warning");
    return;
  }
  list.hidden = !list.hidden;
  $("seriesReplayButton").textContent = list.hidden ? "整场回放" : "收起回放";
}

async function copySeriesResult() {
  const view = state.view;
  const result = view?.series?.final_result;
  if (!result) return;
  const teamPlayers = (team) => (team === "even" ? [0, 2] : [1, 3])
    .map((position) => `${position + 1}号位 ${seatDisplayName(view.seats[position])}（头游 ${result.head_finish_count_by_seat[position]} 次）`)
    .join("、");
  const summary = [
    `掼蛋第 ${result.series_no} 场：${guandanTeamLabel(result.winner_team)}过 A 获胜`,
    `获胜队：${teamPlayers(result.winner_team)}，队伍获胜 ${result.team_victory_count[result.winner_team]} 次`,
    `对方：${teamPlayers(result.loser_team)}，最终打到 ${displayGuandanLevel(result.losing_team_level)}`,
    `整场 ${result.rounds_played} 副，第 ${result.winning_a_attempt_no} 次过 A 成功`,
  ].join("\n");
  try {
    await navigator.clipboard.writeText(summary);
    showToast("整场战绩已复制。", "success");
  } catch {
    showToast(summary, "warning");
  }
}

function renderResultRoundMeta(result, view) {
  const meta = $("resultRoundMeta");
  const presentation = view.game === "guandan"
    ? guandanRoundPresentation(result, view.viewer.seat)
    : null;
  meta.hidden = !presentation;
  if (!presentation) {
    meta.classList.remove("won", "lost");
    return;
  }
  meta.classList.toggle("won", presentation.won);
  meta.classList.toggle("lost", !presentation.won);
  $("resultRoundDelta").textContent = `${presentation.sideLabel}${presentation.outcomeLabel} · ${presentation.signedDelta}`;
  $("resultLevelChange").textContent = presentation.levelChange;
  $("resultOpponentLevel").textContent = presentation.opponentLevel;
  $("resultNextLevel").textContent = presentation.nextLevel;
}

function clearResultDetails() {
  $("resultWinningSide").hidden = true;
  $("resultWinningSide").textContent = "";
  $("resultRoundMeta").hidden = true;
  $("resultRoundMeta").classList.remove("won", "lost");
  $("resultFinishOrder").replaceChildren();
  $("resultOrderSection").hidden = true;
}

function resultSummary(result, view) {
  if (result === null || result === undefined || result === "") return "对局已结束";
  if (["string", "number", "boolean"].includes(typeof result)) return String(result);
  if (result.outcome === "head_last") {
    if (result.summary) {
      return `头游方获胜 · ${String(result.summary).replace(/平局/gu, "头末同队")}`;
    }
    const winningSide = resultWinningSideLabel(result, view).replace(/^胜方：/u, "");
    return winningSide
      ? `头游方获胜 · 头末同队 · ${winningSide}`
      : "头游方获胜 · 头末同队";
  }
  if (result.summary) return result.summary;
  const winningSide = resultWinningSideLabel(result, view).replace(/^胜方：/u, "");
  if (winningSide) return `${winningSide}获胜`;
  if (result.winner !== null && result.winner !== "") return `${result.winner} 获胜`;
  if (Number.isInteger(result.winner_position)) return `${result.winner_position + 1} 号位获胜`;
  if (result.finish_order?.length) {
    return `完赛顺序：${result.finish_order.map((position) => `${position + 1} 号位`).join(" → ")}`;
  }
  return "对局已结束";
}

function resultWinningSideLabel(result, view) {
  const positions = resultWinningPositions(result, view.game, view.seats.length);
  if (view.game === "guandan" && positions.length) {
    return `胜方：${positions.map((position) => position + 1).join("、")} 号位队`;
  }
  const side = String(result.winning_side ?? "").toLowerCase();
  if (side === "landlord") return "胜方：地主";
  if (["defenders", "farmers", "farmer"].includes(side)) return "胜方：农民";
  if (result.winner !== null && result.winner !== undefined && result.winner !== "") {
    return `胜方：${result.winner}`;
  }
  if (Number.isInteger(result.winner_position)) return `胜方：${result.winner_position + 1} 号位`;
  return side ? `胜方：${side}` : "";
}

function finishPlaceLabel(index, total) {
  if (total === 4) return ["头游", "二游", "三游", "末游"][index] ?? `第 ${index + 1} 名`;
  return `第 ${index + 1} 名`;
}

function canMoveToSeat(view, seat) {
  if (SOLO_MODE) return false;
  const ownSeat = view?.seats?.find((candidate) => candidate.position === view.viewer?.seat);
  return Boolean(
    view &&
      !isSpectatorRole(view.viewer.role) &&
      isLobbyPhase(view.phase) &&
      ownSeat &&
      !ownSeat.ready &&
      seat.position !== view.viewer.seat &&
      ["open", "bot"].includes(seat.kind),
  );
}

function seatDisplayName(seat) {
  if (seat.kind === "open") return "空位";
  if (seat.display_name) return seat.display_name;
  return seat.kind === "bot" ? "人机玩家" : "人类玩家";
}

function seatRoleLabel(seat, ownSeat, view) {
  if (seat.kind === "open") return "点击可换到此座位";
  const kind = seat.kind === "bot" ? "人机" : "人类玩家";
  if (ownSeat) return "你 · 人类玩家";
  if (Number.isInteger(view.viewer?.seat)) {
    const relation = seat.position % 2 === view.viewer.seat % 2 ? "队友" : "对手";
    if (SOLO_MODE) return `${relation} · AI`;
    return [relation, kind, seat.kind === "bot" ? "默认准备" : ""].filter(Boolean).join(" · ");
  }
  return [seat.team === "even" ? "双数位队" : seat.team === "odd" ? "单数位队" : "", kind]
    .filter(Boolean)
    .join(" · ");
}

function seatStatusLabel(seat, phase) {
  if (!isReadyPhase(phase)) return `${seat.hand_count} 张`;
  if (seat.kind === "open") return "可入座";
  return seat.ready || seat.kind === "bot" ? "已准备" : "未准备";
}

async function copyTableNumber() {
  if (!state.tableNo) return;
  try {
    await navigator.clipboard.writeText(state.tableNo);
    showToast(`桌号 ${state.tableNo} 已复制。`, "success");
  } catch {
    showToast(`桌号：${state.tableNo}`, "warning");
  }
}

function setConnection(kind, text, { render = true } = {}) {
  state.connectionState = kind;
  if (kind !== "online") closeActionChoices();
  $("connectionChip").dataset.state = kind;
  $("connectionText").textContent = text;
  const frozen = Boolean(state.tableNo) && !["online", "idle"].includes(kind);
  $("connectionFreeze").hidden = !frozen;
  $("humanTableStage").classList.toggle("connection-frozen", frozen);
  if (state.view && render) renderRoom();
}

function interactionFrozen() {
  return Boolean(state.tableNo) && state.connectionState !== "online";
}

function openDialog(dialog) {
  if (typeof dialog.showModal === "function") dialog.showModal();
  else dialog.setAttribute("open", "");
}

function restoreVisibleFocus(id) {
  const target = $(id);
  if (target && !target.disabled && target.getClientRects().length && !document.querySelector("dialog[open]")) target.focus({ preventScroll: true });
}

function setButtonPending(button, pending, label) {
  button.disabled = pending;
  const svg = button.querySelector("svg");
  button.replaceChildren(document.createTextNode(label));
  if (svg) button.appendChild(svg);
}

function showToast(message, tone = "") {
  const toast = document.createElement("div");
  toast.className = `toast${tone ? ` ${tone}` : ""}`;
  toast.textContent = String(message);
  $("humanToastStack").appendChild(toast);
  window.requestAnimationFrame(() => toast.classList.add("show"));
  window.setTimeout(() => {
    toast.classList.remove("show");
    window.setTimeout(() => toast.remove(), 220);
  }, 3600);
}

function extractView(payload, tableNo) {
  const candidate = payload?.view ?? payload ?? {};
  return { ...candidate, table_no: candidate.table_no ?? tableNo };
}

function phaseLabel(value) {
  const key = String(value ?? "").toLowerCase();
  return PHASE_LABELS[key] ?? String(value || "大厅");
}

function gameLabel() {
  return "掼蛋";
}

function newClientActionId() {
  if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
  return `action-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function safeClassName(value) {
  return String(value || "unknown").toLowerCase().replace(/[^a-z0-9_-]/g, "-");
}

function cssPixelValue(value, fallback = 0) {
  const number = Number.parseFloat(String(value ?? ""));
  return Number.isFinite(number) ? number : fallback;
}

function errorMessage(error) {
  return String(error?.message || "操作失败，请稍后重试。");
}
