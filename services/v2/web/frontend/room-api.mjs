export const HUMAN_TABLES_PATH = "/api/human-tables";
export const TOKEN_STORAGE_PREFIX = "arena.play.token:";

export class HumanTableApiError extends Error {
  constructor(message, { status = 0, payload = null, cause = null } = {}) {
    super(message, { cause });
    this.name = "HumanTableApiError";
    this.status = status;
    this.payload = payload;
  }
}

export class HumanTableApi {
  constructor(baseUrl = "", { fetchImpl = null, webSocketFactory = null, requestTimeoutMs = 15000 } = {}) {
    this.baseUrl = String(baseUrl).replace(/\/$/, "");
    this.fetchImpl = fetchImpl ?? (
      typeof globalThis.fetch === "function" ? globalThis.fetch.bind(globalThis) : null
    );
    this.webSocketFactory = webSocketFactory;
    this.requestTimeoutMs = Number.isFinite(requestTimeoutMs) && requestTimeoutMs > 0 ? requestTimeoutMs : 15000;
  }

  createTable({ game, nickname, user_id }) {
    return this.request(HUMAN_TABLES_PATH, {
      method: "POST",
      body: { game, nickname, user_id },
    });
  }

  joinTable(tableNo, { nickname, user_id, resume_token } = {}) {
    const body = { nickname, user_id };
    if (resume_token) body.resume_token = resume_token;
    return this.request(this.tablePath(tableNo, "/join"), {
      method: "POST",
      body,
    });
  }

  viewTable(tableNo, token) {
    return this.request(this.tablePath(tableNo, "/view"), { token });
  }

  retryTable(tableNo, token, { expected_version }) {
    return this.request(this.tablePath(tableNo, "/retry"), {
      method: "POST",
      token,
      body: { expected_version },
    });
  }

  setReady(tableNo, token, { ready, expected_version }) {
    return this.request(this.tablePath(tableNo, "/ready"), {
      method: "POST",
      token,
      body: { ready, expected_version },
    });
  }

  arrangeHand(tableNo, token, { mode }) {
    return this.request(this.tablePath(tableNo, "/arrange"), {
      method: "POST",
      token,
      body: { mode },
    });
  }

  submitAction(
    tableNo,
    token,
    { action_index, expected_version, decision_id, client_action_id, selected_cards },
  ) {
    const body = { action_index, expected_version, decision_id, client_action_id };
    if (selected_cards !== undefined) body.selected_cards = selected_cards;
    return this.request(this.tablePath(tableNo, "/actions"), {
      method: "POST",
      token,
      body,
    });
  }

  connect(tableNo, token, { lastVersion = 0, onState, onOpen, onClose, onError } = {}) {
    const socketUrl = websocketUrl(this.baseUrl, this.tablePath(tableNo, "/ws"));
    const socket = this.webSocketFactory
      ? this.webSocketFactory(socketUrl)
      : new WebSocket(socketUrl);

    socket.addEventListener("open", () => {
      socket.send(JSON.stringify({ type: "auth", token, last_version: Number(lastVersion) || 0 }));
      onOpen?.();
    });
    socket.addEventListener("message", (event) => {
      let message;
      try {
        message = JSON.parse(String(event.data));
      } catch (error) {
        onError?.(new HumanTableApiError("收到了无法识别的桌面消息。", { cause: error }));
        return;
      }
      if (message?.type === "state") onState?.(message.state);
      if (message?.type === "ping" && socket.readyState === 1) {
        socket.send(JSON.stringify({ type: "pong" }));
      }
    });
    socket.addEventListener("close", (event) => onClose?.(event));
    socket.addEventListener("error", (event) => onError?.(event));
    return () => socket.close();
  }

  tablePath(tableNo, suffix = "") {
    const value = String(tableNo ?? "").trim();
    if (!value) throw new HumanTableApiError("请输入桌号。");
    return `${HUMAN_TABLES_PATH}/${encodeURIComponent(value)}${suffix}`;
  }

  async request(path, { method = "GET", token = "", body } = {}) {
    if (typeof this.fetchImpl !== "function") {
      throw new HumanTableApiError("当前浏览器无法连接对局服务。");
    }
    const controller = new AbortController();
    let timer;
    const operation = async () => {
      let response;
      try {
        response = await this.fetchImpl(`${this.baseUrl}${path}`, {
          signal: controller.signal,
          method,
          cache: "no-store",
          credentials: "same-origin",
          headers: {
            Accept: "application/json",
            ...(body === undefined ? {} : { "Content-Type": "application/json" }),
            ...(token ? { Authorization: `Bearer ${token}` } : {}),
          },
          ...(body === undefined ? {} : { body: JSON.stringify(body) }),
        });
      } catch (error) {
        throw new HumanTableApiError("无法连接对局服务。", { cause: error });
      }

      const contentType = response.headers?.get?.("content-type") ?? "";
      let payload = null;
      try {
        payload = contentType.includes("application/json") ? await response.json() : await response.text();
      } catch {
        payload = null;
      }
      if (!response.ok) {
        const message = responseErrorMessage(payload);
        throw new HumanTableApiError(message || `请求失败（${response.status}）。`, {
          status: response.status,
          payload,
        });
      }
      return payload;
    };
    try {
      // Cover both fetch and body consumption. A timeout means unknown outcome,
      // not server rejection; callers retain their original idempotency key.
      return await Promise.race([operation(), new Promise((_, reject) => {
        timer = setTimeout(() => {
          reject(new HumanTableApiError("请求超时，尚未收到服务器确认。"));
          controller.abort();
        }, this.requestTimeoutMs);
      })]);
    } finally {
      clearTimeout(timer);
    }
  }
}

export function tokenStorageKey(tableNo) {
  return `${TOKEN_STORAGE_PREFIX}${String(tableNo ?? "").trim()}`;
}

export function saveRoomToken(tableNo, token, storage = globalThis.sessionStorage) {
  if (!storage || !tableNo || !token) return false;
  try { return storage.setItem(tokenStorageKey(tableNo), String(token)) !== false; }
  catch { return false; }
}

export function loadRoomToken(tableNo, storage = globalThis.sessionStorage) {
  if (!storage || !tableNo) return "";
  try { return storage.getItem(tokenStorageKey(tableNo)) ?? ""; }
  catch { return ""; }
}

export function clearRoomToken(tableNo, storage = globalThis.sessionStorage) {
  if (!storage || !tableNo) return false;
  try { return storage.removeItem(tokenStorageKey(tableNo)) !== false; }
  catch { return false; }
}

export function websocketUrl(baseUrl, path, locationValue = globalThis.location) {
  const fallbackOrigin = locationValue?.origin ?? "http://localhost";
  const url = new URL(`${String(baseUrl || "")}${path}`, fallbackOrigin);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  return url.toString();
}

function responseErrorMessage(payload, depth = 0) {
  if (depth > 3 || payload === null || payload === undefined) return "";
  if (["string", "number", "boolean"].includes(typeof payload)) return String(payload);
  if (typeof payload !== "object" || Array.isArray(payload)) return "";
  for (const key of ["message", "detail", "error"]) {
    const message = responseErrorMessage(payload[key], depth + 1);
    if (message) return message;
  }
  return typeof payload.code === "string" ? payload.code : "";
}
