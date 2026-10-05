export const HUMAN_TURN_SECONDS = 20;

export function decisionClock(view, receivedAtMs = Date.now()) {
  const decisionId = String(view?.decision?.id ?? "");
  const turnStartedAtMs = finiteNumberOrNull(view?.turn_started_at_ms);
  if (!decisionId || turnStartedAtMs === null) return null;
  const configuredSeconds = Number(view?.turn_seconds);
  const turnSeconds = Number.isFinite(configuredSeconds) && configuredSeconds > 0
    ? Math.floor(configuredSeconds)
    : HUMAN_TURN_SECONDS;
  const serverTimeMs = finiteNumberOrNull(view?.server_time_ms) ?? Number(receivedAtMs);
  return {
    decisionId,
    turnStartedAtMs,
    turnSeconds,
    serverTimeMs,
    receivedAtMs: Number(receivedAtMs),
  };
}

export function decisionClockKey(clock) {
  return clock ? `${clock.decisionId}:${clock.turnStartedAtMs}` : "";
}

export function humanCountdownValue(clock, nowMs = Date.now()) {
  if (!clock) return null;
  const elapsedClientMs = Math.max(0, Number(nowMs) - Number(clock.receivedAtMs));
  const estimatedServerNowMs = Number(clock.serverTimeMs) + elapsedClientMs;
  const deadlineMs = Number(clock.turnStartedAtMs) + Number(clock.turnSeconds) * 1000;
  const remaining = Math.ceil((deadlineMs - estimatedServerNowMs) / 1000);
  return Math.max(0, Math.min(Number(clock.turnSeconds), remaining));
}

/**
 * One visual tick. Its only capability is rendering a number; reaching zero
 * cannot submit, pass, click, fetch, or otherwise act for a player.
 */
export function renderCountdownTick(clock, nowMs, renderValue) {
  const value = humanCountdownValue(clock, nowMs);
  if (typeof renderValue === "function") renderValue(value);
  return value;
}

function finiteNumberOrNull(value) {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}
