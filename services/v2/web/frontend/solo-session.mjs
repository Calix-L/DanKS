const RECENT_TABLES_KEY = "cardks.solo.recent_tables.v1";
const MAX_RECENT_TABLES = 3;

// This fallback lives only in the current page. Never move session tokens to
// localStorage, and never claim that in-memory data survives reload or closure.
export function resilientStorage(storage, { onUnavailable } = {}) {
  const memory = new Map();
  let memoryOnly = !storage;
  let notified = false;
  const unavailable = () => {
    memoryOnly = true;
    if (!notified) { notified = true; onUnavailable?.(); }
  };
  return {
    getItem(key) {
      if (!memoryOnly) {
        try {
          const value = storage.getItem(key);
          if (value === null) memory.delete(key); else memory.set(key, String(value));
          return value;
        } catch { unavailable(); }
      } else unavailable();
      return memory.get(key) ?? null;
    },
    setItem(key, value) {
      memory.set(key, String(value));
      if (!memoryOnly) {
        try { storage.setItem(key, String(value)); return true; }
        catch { unavailable(); }
      } else unavailable();
      return false;
    },
    removeItem(key) {
      memory.delete(key);
      if (!memoryOnly) {
        try { storage.removeItem(key); return true; }
        catch { unavailable(); }
      } else unavailable();
      return false;
    },
  };
}

// Keep the deployed production contract intact; only /test has separate browser data.
export function scopeSoloStorage(storage, basePath = "") {
  if (!storage || String(basePath).replace(/\/+$/u, "") !== "/test") return storage;
  const prefix = "cardks.test:";
  return {
    getItem(key) { return storage.getItem(prefix + key); },
    setItem(key, value) { storage.setItem(prefix + key, value); },
    removeItem(key) { storage.removeItem(prefix + key); },
  };
}

export function loadRecentSoloTables(storage) {
  if (!storage?.getItem) return [];
  try {
    const rows = JSON.parse(storage.getItem(RECENT_TABLES_KEY) ?? "[]");
    if (!Array.isArray(rows)) return [];
    return rows
      .map(normalizeRecentTable)
      .filter(Boolean)
      .sort((left, right) => right.updatedAt - left.updatedAt)
      .slice(0, MAX_RECENT_TABLES);
  } catch {
    return [];
  }
}

export function rememberRecentSoloTable(storage, recentTable) {
  if (!storage?.setItem) return [];
  const normalized = normalizeRecentTable(recentTable);
  if (!normalized) return loadRecentSoloTables(storage);
  const rows = [
    normalized,
    ...loadRecentSoloTables(storage).filter((row) => row.tableNo !== normalized.tableNo),
  ].slice(0, MAX_RECENT_TABLES);
  try { storage.setItem(RECENT_TABLES_KEY, JSON.stringify(rows)); } catch { /* Optional history cannot block play. */ }
  return rows;
}

export function forgetRecentSoloTable(storage, tableNo) {
  if (!storage?.setItem) return [];
  const target = String(tableNo ?? "").trim();
  const rows = loadRecentSoloTables(storage).filter((row) => row.tableNo !== target);
  try { storage.setItem(RECENT_TABLES_KEY, JSON.stringify(rows)); } catch { /* Optional history cannot block play. */ }
  return rows;
}

export function findRecentSoloTable(storage, tableNo = "") {
  const target = String(tableNo ?? "").trim();
  const rows = loadRecentSoloTables(storage);
  return target ? rows.find((row) => row.tableNo === target) ?? null : rows[0] ?? null;
}

function normalizeRecentTable(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const tableNo = String(value.tableNo ?? "").trim();
  const nickname = String(value.nickname ?? "").trim();
  const updatedAt = Number(value.updatedAt);
  if (!/^[A-Za-z0-9_-]{1,24}$/u.test(tableNo) || !nickname || nickname.length > 32) return null;
  if (!Number.isFinite(updatedAt) || updatedAt < 0) return null;
  return { tableNo, nickname, updatedAt };
}
