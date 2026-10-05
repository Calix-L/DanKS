const RECENT_TABLES_KEY = "cardks.solo.recent_tables.v1";
const MAX_RECENT_TABLES = 3;

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
  storage.setItem(RECENT_TABLES_KEY, JSON.stringify(rows));
  return rows;
}

export function forgetRecentSoloTable(storage, tableNo) {
  if (!storage?.setItem) return [];
  const target = String(tableNo ?? "").trim();
  const rows = loadRecentSoloTables(storage).filter((row) => row.tableNo !== target);
  storage.setItem(RECENT_TABLES_KEY, JSON.stringify(rows));
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
