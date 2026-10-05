// Local, synthesized cues. No asset/network dependency and no game timing.
const PREFERENCE_KEY = "cardks.table.sound.v1";

export function createTableSound(host) {
  let enabled = false, context = null, generation = 0;
  try { enabled = host.localStorage?.getItem(PREFERENCE_KEY) === "on"; } catch { /* Storage is optional. */ }
  const persist = () => {
    try { host.localStorage?.setItem(PREFERENCE_KEY, enabled ? "on" : "off"); } catch { /* Playing works without storage. */ }
  };
  const unlock = async () => {
    const Audio = host.AudioContext ?? host.webkitAudioContext;
    if (!Audio) return false;
    context ??= new Audio();
    if (context.state !== "running") await context.resume();
    return true;
  };
  return {
    get enabled() { return enabled; },
    async arm() {
      if (!enabled || context?.state === "running") return;
      try { await unlock(); } catch { /* A later gesture may retry. */ }
    },
    async toggle() {
      const request = ++generation;
      if (enabled) { enabled = false; persist(); return false; }
      try {
        if (!await unlock()) return false;
        if (request !== generation) return enabled;
        enabled = true;
      } catch { enabled = false; }
      persist();
      return enabled;
    },
    play(kind) {
      if (!enabled || !context || context.state !== "running") return;
      try {
        const oscillator = context.createOscillator(), gain = context.createGain();
        const now = context.currentTime;
        oscillator.frequency.setValueAtTime(kind === "turn" ? 659 : 240, now);
        gain.gain.setValueAtTime(0.025, now);
        gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.12);
        oscillator.connect(gain); gain.connect(context.destination);
        oscillator.onended = () => { oscillator.disconnect(); gain.disconnect(); };
        oscillator.start(); oscillator.stop(now + 0.14);
      } catch { /* Audio failures must never interrupt a game. */ }
    },
    async close() {
      generation++; enabled = false;
      try { await context?.close(); } catch { /* Page teardown. */ }
      context = null;
    },
  };
}
