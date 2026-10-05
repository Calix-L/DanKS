const ENTRYPOINTS = new Set(["index.html", "play", "play.html", "solo"]);

export function normalizeArenaBasePath(value) {
  const raw = String(value ?? "").trim();
  if (!raw || raw === "/") return "";
  if (raw.includes("://") || /[?#\\]/u.test(raw)) {
    throw new TypeError("Arena base path must be a URL path.");
  }
  const segments = raw.split("/").filter(Boolean);
  if (!segments.length || segments.some((segment) => segment === "." || segment === "..")) {
    throw new TypeError("Arena base path contains an invalid segment.");
  }
  return `/${segments.join("/")}`;
}

export function inferArenaBasePath(locationValue = globalThis.location) {
  const pathname = String(locationValue?.pathname ?? "/");
  const segments = pathname.split("/").filter(Boolean);
  if (!segments.length) return "";

  const finalSegment = segments.at(-1);
  if (ENTRYPOINTS.has(finalSegment)) segments.pop();
  return normalizeArenaBasePath(segments.join("/"));
}

export function resolveArenaBasePath({
  explicit = globalThis.__PAICE_ARENA_BASE_PATH__,
  documentValue = globalThis.document,
  locationValue = globalThis.location,
} = {}) {
  if (explicit !== undefined && explicit !== null) {
    return normalizeArenaBasePath(explicit);
  }
  const declared = documentValue?.documentElement?.dataset?.arenaBasePath;
  if (declared !== undefined && declared !== "" && declared !== "auto") {
    return normalizeArenaBasePath(declared);
  }
  return inferArenaBasePath(locationValue);
}
