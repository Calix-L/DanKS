import { createPlayingCard } from "./frontend/card-ui.mjs?v=20260805-3";
import {
  fallbackHandGroups,
  renderHandLayout,
} from "./frontend/hand-layout.mjs?v=20260805-1";
import { resolveArenaBasePath } from "./frontend/base-path.mjs?v=20260805-1";
import { reviewHref } from "./frontend/navigation.mjs?v=20260810-1";
import { installSoloLocale } from "./frontend/solo-english.mjs?v=20260829-1";
import {
  findRecentSoloTable,
  forgetRecentSoloTable,
  loadRecentSoloTables,
  rememberRecentSoloTable,
} from "./frontend/solo-session.mjs?v=20260828-1";
import {
  HumanTableApi,
  clearRoomToken,
  loadRoomToken,
  saveRoomToken,
} from "./frontend/room-api.mjs?v=20260826-1";
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
} from "./frontend/room-state.mjs?v=20260828-1";
import {
  decisionClock,
  renderCountdownTick,
} from "./frontend/human-turn-visuals.mjs?v=20260804-1";

const api = new HumanTableApi(resolveArenaBasePath());
const $ = (id) => document.getElementById(id);
const SOLO_MODE = document.body.dataset.arenaMode === "solo";
const SOLO_USER_ID_KEY = "cardks.solo.user_id.v1";
const state = {
  tableNo: "",
  token: "",
  view: null,
  selectedIds: new Set(),
  clock: null,
  closeSocket: null,
  socketGeneration: 0,
  reconnectTimer: null,
  reconnectAttempt: 0,
  pendingAction: null,
  pendingMutation: false,
  pendingArrange: false,
  connectionState: "idle",
  handLayoutObserver: null,
  handLayoutFrame: null,
  selectionGesture: null,
  suppressSelectionClick: false,
  suppressSelectionClickTimer: null,
  playAnimationSeats: new Set(),
  recentRememberSignature: "",
};

const PHASE_LABELS = {
  lobby: "大厅等待",
  waiting: "大厅等待",
  ready: "等待准备",
  deal: "正在发牌",
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
const HUMAN_TABLE_VIEWPORT = { width: 1600, height: 900 };

initialize();

function initialize() {
  document.body.classList.add("human-fixed-viewport");
  if (SOLO_MODE) initializeSoloMode();
  updateHumanTableViewportScale();
  bindEntryControls();
  bindRoomControls();
  initializeOwnHandResizeObserver();
  if (!SOLO_MODE) renderHumanSlotOptions("guandan");
  window.setInterval(updateCountdownDisplay, 200);
  window.addEventListener("resize", updateHumanTableViewportScale);
  window.visualViewport?.addEventListener("resize", handleVisualViewportResize, { passive: true });
  window.addEventListener("beforeunload", destroyRoomRuntime);
  restoreFromSession();
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
  installSoloLocale(document);
}

function updateHumanTableViewportScale() {
  const stage = $("humanTableStage");
  if (!stage) return;
  const viewport = window.visualViewport;
  const width = Math.max(320, viewport?.width || window.innerWidth || HUMAN_TABLE_VIEWPORT.width);
  const height = Math.max(320, viewport?.height || window.innerHeight || HUMAN_TABLE_VIEWPORT.height);
  const scale = Math.min(
    width / HUMAN_TABLE_VIEWPORT.width,
    height / HUMAN_TABLE_VIEWPORT.height,
  );
  stage.style.setProperty("--human-table-scale", Math.max(0.36, scale).toFixed(5));
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
  $("readyButton").addEventListener("click", toggleReady);
  $("resultReadyButton").addEventListener("click", toggleReady);
  $("seriesRematchButton").addEventListener("click", toggleReady);
  $("seriesReplayButton").addEventListener("click", toggleSeriesReplayList);
  $("seriesCopyButton").addEventListener("click", copySeriesResult);
  $("passButton").addEventListener("click", submitPass);
  $("playButton").addEventListener("click", prepareSelectedPlay);
  $("copyTableBtn").addEventListener("click", copyTableNumber);
  $("rulesButton").addEventListener("click", () => openDialog($("rulesDialog")));
  const arrangeMenuButton = $("arrangeMenuButton");
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
    }
  });
  $("actionDialog").addEventListener("close", () => {
    $("actionChoiceList").replaceChildren();
    $("playButton").focus({ preventScroll: true });
  });
  $("rulesDialog").addEventListener("close", () => $("rulesButton").focus({ preventScroll: true }));
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
  const panelWidth = panel.getBoundingClientRect().width;
  const viewportWidth = viewport.getBoundingClientRect().width;
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
  const mobileTouchLayout = window.matchMedia([
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
    `${Math.ceil(panel.getBoundingClientRect().height)}px`,
  );
  const hint = $("handScrollHint");
  hint.hidden = viewport.scrollWidth <= viewport.clientWidth + 1;
  if (hint.dataset.dismissed === "true") hint.hidden = true;
}

function destroyRoomRuntime() {
  closeLiveSocket();
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
  if (!SOLO_MODE) $("joinTableNo").value = tableNo;
  const token = loadRoomToken(tableNo);
  if (!token) {
    if (SOLO_MODE) {
      const recent = findRecentSoloTable(window.localStorage, tableNo);
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
      clearRoomToken(tableNo);
      const recent = findRecentSoloTable(window.localStorage, tableNo);
      if (recent) {
        await resumeRecentSoloTable(recent);
        return;
      }
    } else if (error?.status === 401 || error?.status === 403 || error?.status === 404) {
      clearRoomToken(tableNo);
      if (SOLO_MODE && error?.status === 404) {
        forgetRecentSoloTable(window.localStorage, tableNo);
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
    renderRecentTableCard();
  }
}

function renderRecentTableCard() {
  if (!SOLO_MODE) return;
  const recent = loadRecentSoloTables(window.localStorage)[0] ?? null;
  const card = $("recentTableCard");
  card.hidden = !recent;
  if (!recent) return;
  $("recentTableTitle").textContent = `桌号 ${recent.tableNo}`;
  $("recentTableMeta").textContent = `${recent.nickname} · 返回原座位继续`;
}

async function resumeRecentSoloTable(recentTable = null) {
  if (!SOLO_MODE) return;
  const recent = recentTable ?? loadRecentSoloTables(window.localStorage)[0] ?? null;
  if (!recent) {
    renderRecentTableCard();
    return;
  }
  const button = $("resumeRecentTableButton");
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
      clearRoomToken(recent.tableNo);
      forgetRecentSoloTable(window.localStorage, recent.tableNo);
    }
    setConnection("offline", "恢复失败");
    renderRecentTableCard();
    showToast(errorMessage(error), "error");
  } finally {
    setButtonPending(button, false, "继续对局");
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
  const humanSlots = SOLO_MODE
    ? 1
    : Number(document.querySelector('input[name="humanSlots"]:checked')?.value ?? 1);
  const nickname = $("createNickname").value.trim();
  if (!nickname) return;

  setButtonPending(button, true, "正在创建…");
  try {
    const response = SOLO_MODE
      ? await api.request("/api/human-tables", {
          method: "POST",
          body: { game, nickname, user_id: soloUserId() },
        })
      : await api.createTable({ game, human_slots: humanSlots, nickname });
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
  let value = String(window.localStorage?.getItem(SOLO_USER_ID_KEY) ?? "").trim();
  if (/^[A-Za-z0-9_-]{16,128}$/u.test(value)) return value;
  const random = window.crypto?.randomUUID?.().replaceAll("-", "")
    ?? String(Date.now().toString(36) + Math.random().toString(36).slice(2));
  value = "solo_" + random;
  window.localStorage?.setItem(SOLO_USER_ID_KEY, value);
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
  saveRoomToken(tableNo, token);
  if (SOLO_MODE) {
    rememberRecentSoloTable(window.localStorage, {
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
        state.reconnectAttempt = 0;
        setConnection("reconnecting", "正在验证实时桌面");
      },
      onState: (rawView) => {
        if (generation !== state.socketGeneration) return;
        setConnection("online", "实时已连接");
        applyRoomView(extractView(rawView, state.tableNo));
      },
      onClose: () => {
        if (generation !== state.socketGeneration) return;
        state.closeSocket = null;
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

function applyRoomView(rawView) {
  const nextView = normalizeRoomView(rawView);
  if (state.view?.table_no === nextView.table_no && nextView.version < state.view.version) return;
  state.playAnimationSeats = new Set();
  if (state.view) {
    nextView.seats.forEach((seat) => {
      const previous = state.view.seats.find((item) => item.position === seat.position);
      if (previous && playAreaSignature(previous.play_area) !== playAreaSignature(seat.play_area)) {
        if (seat.play_area.cards.length || seat.play_area.pass) state.playAnimationSeats.add(seat.position);
      }
    });
  }
  const previousDecisionId = state.view?.decision?.id ?? "";
  const nextDecisionId = nextView.decision?.id ?? "";
  const currentIds = new Set(nextView.own_hand.map((card) => card.id));
  state.selectedIds = new Set([...state.selectedIds].filter((id) => currentIds.has(id)));

  if (previousDecisionId !== nextDecisionId) state.selectedIds.clear();
  if (
    state.pendingAction &&
    (state.pendingAction.decisionId !== nextDecisionId || nextView.version > state.pendingAction.version)
  ) {
    state.pendingAction = null;
    state.selectedIds.clear();
  }

  state.view = nextView;
  state.tableNo = nextView.table_no || state.tableNo;
  if (SOLO_MODE && nextView.phase === "game_over" && nextView.series?.match_finished) {
    forgetRecentSoloTable(window.localStorage, state.tableNo);
    state.recentRememberSignature = "";
    renderRecentTableCard();
  } else if (SOLO_MODE && !isSpectatorRole(nextView.viewer.role)) {
    const ownSeat = nextView.seats.find((seat) => seat.position === nextView.viewer.seat);
    const nickname = String(ownSeat?.display_name ?? "").trim();
    const signature = `${state.tableNo}\u0000${nickname}`;
    if (state.tableNo && nickname && signature !== state.recentRememberSignature) {
      rememberRecentSoloTable(window.localStorage, {
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
}

function renderRoom() {
  const view = state.view;
  if (!view) return;
  const spectator = isSpectatorRole(view.viewer.role);
  const lobby = isLobbyPhase(view.phase);
  const terminal = isTerminalPhase(view.phase);
  $("humanTableStage").classList.toggle("spectator-view", spectator);
  $("humanTableStage").classList.toggle("round-terminal", terminal);
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
  $("lobbyHelpText").textContent = SOLO_MODE
    ? "点击准备后开始单人挑战；其余三个座位均由 Bot 接管。"
    : "可点击空位或人机座位换座；每位人类玩家都须准备，全员准备后发牌并确定本局级牌。";
  $("ruleBadgeStrip").hidden = false;
  renderGuandanLevelHud(view);
  $("rulesButton").hidden = false;
  renderRulesDialog();

  renderSeats(view);
  renderReadyControl(view, { spectator, lobby });
  renderOwnHand(view, spectator);
  renderResult(view);
  updateCountdownDisplay();
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
      : seat.play_area;
    node.className = `human-seat ${visualClass} kind-${safeClassName(seat.kind)}`;
    node.classList.toggle("viewer-seat", viewerOwnSeat);
    node.classList.toggle("terminal-reveal", revealHands && !spectator);
    node.classList.toggle("current-actor", currentActor);
    node.classList.toggle("passed", visiblePlayArea.pass);
    node.classList.toggle("play-entering", state.playAnimationSeats.has(seat.position));
    node.dataset.position = String(seat.position);

    const tag = document.createElement("header");
    tag.className = "player-tag";
    const number = document.createElement("span");
    number.className = "seat-number";
    number.textContent = String(seat.position + 1);
    const identity = document.createElement("div");
    const name = document.createElement("strong");
    name.className = "player-name";
    name.textContent = seatDisplayName(seat);
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
    status.textContent = finishLabel || seatStatusLabel(seat, view.phase);
    if (finishLabel) status.setAttribute("aria-label", `${seatDisplayName(seat)}：${finishLabel}`);
    tag.append(number, identity, status);

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
      node.appendChild(opponentHandElement(seat.hand_count));
    }
    node.appendChild(playAreaElement(visiblePlayArea, view.game, view.level));
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

function opponentHandElement(handCount) {
  const container = document.createElement("div");
  container.className = "opponent-hand";
  const visibleBacks = Math.min(7, Math.max(1, handCount));
  for (let index = 0; index < visibleBacks; index += 1) {
    const back = document.createElement("span");
    back.className = "card-back";
    back.setAttribute("aria-hidden", "true");
    container.appendChild(back);
  }
  const count = document.createElement("b");
  count.className = "opponent-hand-count";
  count.textContent = `${handCount} 张`;
  container.appendChild(count);
  return container;
}

function playAreaElement(playArea, game, levelRank) {
  const area = document.createElement("div");
  area.className = "seat-play-area";
  const label = document.createElement("strong");
  label.className = "play-label";
  label.textContent = playArea.pass ? "不出" : playArea.label;
  const cards = document.createElement("div");
  cards.className = "play-cards";
  const visibleCards = playArea.cards.slice(0, 12);
  cards.dataset.cardCount = String(visibleCards.length);
  visibleCards.forEach((code) =>
    cards.appendChild(createPlayingCard(code, "table", { game, levelRank })),
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
  $("readyButton").classList.toggle("ready", ready);
  $("readyButton").disabled = state.pendingMutation || interactionFrozen();
  $("readyButtonText").textContent = ready ? "已准备 · 点击取消" : "点击准备";
}

function renderOwnHand(view, spectator) {
  cancelSelectionGesture();
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
    $("ownHand").replaceChildren();
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
    $("ownSeatRole").textContent = seatRoleLabel(ownSeat, true, view);
    $("ownHandRemaining").textContent = `${ownSeat.hand_count} 张手牌`;
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

  const ownPlayTurn = isOwnPlayDecision(view);
  const selectable = ownPlayTurn && !state.pendingAction && !state.pendingArrange && !interactionFrozen();
  const hand = $("ownHand");
  hand.replaceChildren();
  const layout = view.own_hand_layout;
  const rawGroups = layout?.groups?.length
    ? layout.groups
    : [{ pattern: 0, minor: 0, value: null, cards: view.own_hand }];
  const groups = layout?.fallback && rawGroups.length === 1 && rawGroups[0].cards.length > 8
    ? rawGroups[0].cards.map((card) => ({ pattern: 0, minor: 0, value: null, cards: [card] }))
    : rawGroups;
  hand.classList.toggle("fallback-layout", Boolean(layout?.fallback));
  groups.forEach((group, groupIndex) => {
    const stack = document.createElement("div");
    stack.className = "own-meld-stack";
    stack.dataset.pattern = String(group.pattern ?? 0);
    stack.dataset.size = String(group.cards.length);
    stack.setAttribute("role", "group");
    stack.setAttribute("aria-label", `理牌分组 ${groupIndex + 1}，${group.cards.length} 张`);
    group.cards.forEach((card) => stack.appendChild(ownHandCardButton(card, view, selectable)));
    hand.appendChild(stack);
  });
  renderArrangeToolbar(view);
  renderActionControls(view);
  scheduleOwnHandSizing();
}

function ownHandCardButton(card, view, selectable) {
  const button = document.createElement("button");
  button.className = "hand-card-button";
  button.type = "button";
  button.dataset.cardId = card.id;
  button.classList.toggle("selected", state.selectedIds.has(card.id));
  button.setAttribute("aria-pressed", String(state.selectedIds.has(card.id)));
  button.setAttribute("aria-label", `选择 ${card.code}`);
  button.disabled = !selectable;
  button.appendChild(createPlayingCard(card.code, "hand", {
    game: view.game,
    levelRank: view.level ?? "",
  }));
  return button;
}

function renderArrangeToolbar(view) {
  const toolbar = $("arrangeToolbar");
  const available = view.own_hand.length > 0 && !isSpectatorRole(view.viewer.role);
  toolbar.hidden = !available;
  if (!available) {
    setArrangeMenuOpen(false);
    return;
  }
  $("arrangeLabel").textContent = "理牌";
  $("arrangeCurrentMode").textContent = ARRANGE_MODE_LABELS[view.arrange_mode] ?? "选择方式";
  $("arrangeMenuButton").disabled = state.pendingArrange || interactionFrozen();
  toolbar.querySelectorAll("[data-arrange-mode]").forEach((button) => {
    const mode = Number(button.dataset.arrangeMode);
    const active = mode === view.arrange_mode;
    button.setAttribute("aria-pressed", String(active));
    const variantCount = view.own_hand_layout?.variant_count ?? 1;
    if (active && variantCount > 1) {
      button.dataset.variantCount = `${(view.own_hand_layout?.variant_index ?? 0) + 1}/${variantCount}`;
    } else {
      delete button.dataset.variantCount;
    }
    button.disabled = state.pendingArrange || interactionFrozen();
  });
  $("arrangeHint").textContent = state.pendingArrange
    ? "正在重新分组；已选牌保持不变…"
    : view.own_hand_layout?.fallback
      ? "理牌服务暂不可用，当前按安全回退顺序显示；规则手牌未改变。"
      : `方案 ${(view.own_hand_layout?.variant_index ?? 0) + 1}/${view.own_hand_layout?.variant_count ?? 1}；重复点击当前策略可切换。`;
}

function setArrangeMenuOpen(open, { restoreFocus = false } = {}) {
  const toolbar = $("arrangeToolbar");
  const trigger = $("arrangeMenuButton");
  const controls = $("guandanArrangeControls");
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
  const ownPlayTurn = isOwnPlayDecision(view);
  const pending = Boolean(state.pendingAction);
  $("selectedCount").textContent = `${state.selectedIds.size} 张已选`;
  $("actionDock").classList.toggle("pending", pending);
  $("playActionButtons").hidden = false;
  $("passButton").disabled = !ownPlayTurn || !view.decision?.can_pass || pending || interactionFrozen();
  $("playButton").disabled = !ownPlayTurn || state.selectedIds.size === 0 || pending || interactionFrozen();
  if (pending) {
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
  syncOwnHandSelection();
}

function beginSelectionGesture(event) {
  if (event.pointerType !== "mouse" || event.button !== 0 || event.isPrimary === false) return;
  const button = handCardButtonFromEvent(event);
  if (!button || button.disabled || !canSelectOwnHand()) return;
  event.preventDefault();
  cancelSelectionGesture();
  const startCardId = button.dataset.cardId;
  state.selectionGesture = {
    pointerId: event.pointerId,
    lastCardId: startCardId,
    baselineSelectedIds: new Set(state.selectedIds),
  };
  $("ownHand").classList.add("sweep-selecting");
  button.focus({ preventScroll: true });
  $("ownHand").setPointerCapture?.(event.pointerId);
  toggleCardSelection(startCardId);
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
  if (!button || button.disabled || !cardId) {
    gesture.lastCardId = null;
    return;
  }
  if (cardId === gesture.lastCardId) return;
  gesture.lastCardId = cardId;
  toggleCardSelection(cardId);
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

function cancelSelectionGesture(event) {
  const gesture = state.selectionGesture;
  if (!gesture) {
    $("ownHand")?.classList.remove("sweep-selecting");
    return;
  }
  if (event?.pointerId !== undefined && event.pointerId !== gesture.pointerId) return;
  const pointerId = gesture?.pointerId;
  state.selectionGesture = null;
  if (pointerId !== undefined && $("ownHand")?.hasPointerCapture?.(pointerId)) {
    $("ownHand").releasePointerCapture(pointerId);
  }
  $("ownHand")?.classList.remove("sweep-selecting");
  if (gesture?.baselineSelectedIds) {
    state.selectedIds = new Set(gesture.baselineSelectedIds);
    syncOwnHandSelection();
  }
}

function handleOwnHandClick(event) {
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
  return Boolean(
    isOwnPlayDecision(state.view) &&
      !state.pendingAction &&
      !state.pendingArrange &&
      !interactionFrozen(),
  );
}

function syncOwnHandSelection() {
  $("ownHand").querySelectorAll(".hand-card-button").forEach((button) => {
    const selected = state.selectedIds.has(button.dataset.cardId);
    button.classList.toggle("selected", selected);
    button.setAttribute("aria-pressed", String(selected));
  });
  if (state.view) renderActionControls(state.view);
}

function prepareSelectedPlay() {
  const view = state.view;
  if (!isOwnPlayDecision(view) || state.pendingAction || interactionFrozen()) return;
  const selectedCards = selectedCardCodes(view.own_hand, state.selectedIds);
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
  const list = $("actionChoiceList");
  list.replaceChildren();
  actions.forEach((action, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = actionChoiceLabel(action, index);
    button.addEventListener("click", () => {
      $("actionDialog").close();
      submitLegalAction(action, { selectedCards });
    });
    list.appendChild(button);
  });
  openDialog($("actionDialog"));
}

function submitPass() {
  const decision = state.view?.decision;
  if (!isOwnPlayDecision(state.view) || !decision?.can_pass || state.pendingAction || interactionFrozen()) return;
  const passAction = decision.actions.find((action) =>
    action.type.toUpperCase() === "PASS" || action.label.toUpperCase() === "PASS",
  );
  if (!passAction) {
    showToast("服务器没有提供“不出”的动作索引，未提交。", "error");
    return;
  }
  submitLegalAction(passAction);
}

async function submitLegalAction(action, { selectedCards = [] } = {}) {
  const view = state.view;
  if (!isOwnDecision(view) || state.pendingAction || !state.token || interactionFrozen()) return;
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
  };
  renderOwnHand(view, false);
  try {
    const response = await api.submitAction(state.tableNo, state.token, request);
    applyResponseSnapshot(response);
  } catch (error) {
    state.pendingAction = null;
    renderOwnHand(state.view, false);
    showToast(errorMessage(error), "error");
    if (error?.status === 409) refreshAuthoritativeView();
  }
}

async function moveToSeat(position) {
  const view = state.view;
  const seat = view?.seats.find((item) => item.position === position);
  if (!seat || !canMoveToSeat(view, seat) || state.pendingMutation || interactionFrozen()) return;
  state.pendingMutation = true;
  renderRoom();
  try {
    const response = await api.takeSeat(state.tableNo, state.token, {
      position,
      expected_version: view.version,
    });
    applyResponseSnapshot(response);
  } catch (error) {
    showToast(errorMessage(error), "error");
    if (error?.status === 409) refreshAuthoritativeView();
  } finally {
    state.pendingMutation = false;
    renderRoom();
  }
}

async function toggleReady() {
  const view = state.view;
  const ownSeat = view?.seats.find((seat) => seat.position === view.viewer.seat);
  if (!ownSeat || !isReadyPhase(view.phase) || isSpectatorRole(view.viewer.role) || state.pendingMutation || interactionFrozen()) return;
  state.pendingMutation = true;
  renderRoom();
  try {
    const response = await api.setReady(state.tableNo, state.token, {
      ready: !ownSeat.ready,
      expected_version: view.version,
    });
    applyResponseSnapshot(response);
  } catch (error) {
    showToast(errorMessage(error), "error");
    if (error?.status === 409) refreshAuthoritativeView();
  } finally {
    state.pendingMutation = false;
    renderRoom();
  }
}

async function arrangeOwnHand(mode) {
  const view = state.view;
  if (
    view?.game !== "guandan" ||
    !Number.isInteger(view.viewer?.seat) ||
    ![1, 2, 3, 4].includes(mode) ||
    state.pendingArrange ||
    interactionFrozen()
  ) return;
  state.pendingArrange = true;
  renderOwnHand(view, false);
  try {
    const response = await api.arrangeHand(state.tableNo, state.token, { mode });
    applyResponseSnapshot(response);
  } catch (error) {
    showToast(errorMessage(error), "error");
  } finally {
    state.pendingArrange = false;
    if (state.view) renderOwnHand(state.view, false);
  }
}

async function refreshAuthoritativeView() {
  try {
    const response = await api.viewTable(state.tableNo, state.token);
    applyRoomView(extractView(response, state.tableNo));
  } catch (error) {
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
    });
    return;
  }
  renderCountdownTick(state.clock, Date.now(), (value) => {
    document.querySelectorAll(".human-turn-countdown").forEach((node) => {
      const clockValue = node.querySelector(".turn-countdown-value");
      if (clockValue) clockValue.textContent = value === null ? "" : String(value);
      node.classList.toggle("elapsed", value === 0);
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
    readyButton.textContent = ownSeat.ready ? "已准备 · 等待其他玩家" : "准备下一局";
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
    const name = document.createElement("strong");
    name.textContent = seatDisplayName(seat);
    identity.append(seatNo, name);
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

function setConnection(kind, text) {
  state.connectionState = kind;
  $("connectionChip").dataset.state = kind;
  $("connectionText").textContent = text;
  const frozen = Boolean(state.tableNo) && !["online", "idle"].includes(kind);
  $("connectionFreeze").hidden = !frozen;
  $("humanTableStage").classList.toggle("connection-frozen", frozen);
  if (state.view) renderRoom();
}

function interactionFrozen() {
  return Boolean(state.tableNo) && state.connectionState !== "online";
}

function openDialog(dialog) {
  if (typeof dialog.showModal === "function") dialog.showModal();
  else dialog.setAttribute("open", "");
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
