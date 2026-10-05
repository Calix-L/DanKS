const ENTRYPOINTS = new Set(["index.html", "solo", "play", "play.html", "review", "review.html"]);

function safeMatchId(value) {
  const result = String(value ?? "").trim();
  return result && result.length <= 256 && !/[\u0000-\u001f\u007f]/u.test(result) ? result : "";
}

function localPage(locationValue, page) {
  const parts = String(locationValue?.pathname ?? "/").split("/").filter(Boolean);
  if (ENTRYPOINTS.has(parts.at(-1))) parts.pop();
  return parts.length ? `/${parts.join("/")}/${page}` : `./${page}`;
}

export function reviewHref(matchId = "", locationValue = globalThis.location) {
  const page = localPage(locationValue, "review.html");
  const normalized = safeMatchId(matchId);
  return normalized ? `${page}?match=${encodeURIComponent(normalized)}` : page;
}

export function reviewMatchFromQuery(search = globalThis.location?.search) {
  try {
    return safeMatchId(new URLSearchParams(String(search ?? "")).get("match"));
  } catch {
    return "";
  }
}
