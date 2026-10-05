// Presentation geometry only. Never changes room state, card IDs or actions.
export const TABLE_VIEWPORT = Object.freeze({ width: 1600, height: 900 });

export function tableFrameMetrics(width, height, { mobileTools = false } = {}) {
  const gutter = width <= 1024 ? 8 : 24;
  const portraitHint = width <= 760 && height > width;
  const shortLandscape = width > height && height <= 600 && width <= 1024;
  const toolsRail = mobileTools && width <= 1024 && width > height ? 60 : 0;
  const scale = Math.min(1,
    Math.max(1, width - gutter * 2 - toolsRail) / TABLE_VIEWPORT.width,
    Math.max(1, height - gutter * 2 - (portraitHint ? 56 : 0)) / TABLE_VIEWPORT.height,
  );
  return { scale, gutter, portraitHint, shortLandscape, toolsRail };
}

const HAND_SPREAD_KEY = "cardks.solo.hand-spread.v1";
export function loadHandSpread(storage) {
  try { return storage?.getItem(HAND_SPREAD_KEY) === "true"; } catch { return false; }
}
export function saveHandSpread(storage, spread) {
  try { storage?.setItem(HAND_SPREAD_KEY, String(Boolean(spread))); } catch { /* Optional preference only. */ }
}
const CONTROL_PREFERENCES_KEY = "cardks.solo.controls.v1";
function normalizeControlPreferences(value) {
  return {
    size: ["auto", "standard", "large"].includes(value?.size) ? value.size : "auto",
    side: value?.side === "left" ? "left" : "right",
  };
}
export function loadControlPreferences(storage) {
  try { return normalizeControlPreferences(JSON.parse(storage?.getItem(CONTROL_PREFERENCES_KEY) || "null")); }
  catch { return normalizeControlPreferences(null); }
}
export function saveControlPreferences(storage, preferences) {
  try { storage?.setItem(CONTROL_PREFERENCES_KEY, JSON.stringify(normalizeControlPreferences(preferences))); }
  catch { /* Display preference only; storage failure never blocks play. */ }
}
export function effectiveControlSize(preferences, width, height) {
  const { size } = normalizeControlPreferences(preferences);
  if (size !== "auto") return size;
  return tableFrameMetrics(width, height).shortLandscape ? "large" : "standard";
}
export function tableFlightVector(source, target, scale) {
  const values = [source?.left, source?.top, source?.width, source?.height,
    target?.left, target?.top, target?.width, target?.height, scale];
  if (!values.every(Number.isFinite) || scale <= 0) return { x: 0, y: 0 };
  return {
    x: (source.left + source.width / 2 - target.left - target.width / 2) / scale,
    y: (source.top + source.height / 2 - target.top - target.height / 2) / scale,
  };
}
export function tableIsFullscreen(documentRef) {
  return Boolean(documentRef.fullscreenElement || documentRef.webkitFullscreenElement);
}
export async function toggleTableFullscreen(documentRef, app) {
  const active = tableIsFullscreen(documentRef);
  const method = active ? documentRef.exitFullscreen || documentRef.webkitExitFullscreen
    : app.requestFullscreen || app.webkitRequestFullscreen;
  if (!method || !active && documentRef.fullscreenEnabled === false) return "unsupported";
  try {
    await method.call(active ? documentRef : app);
    return "requested"; // The fullscreenchange event, not this request, owns UI state.
  } catch { return "denied"; }
}
