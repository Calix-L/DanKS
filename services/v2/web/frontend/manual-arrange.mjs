// Display-only manual partition editor. The referee remains the action authority.

function validate(groups, selectedIds) {
  const seen = new Set(), keys = new Set();
  for (const group of groups) {
    if (!group.key || keys.has(group.key) || !group.cards?.length) throw new TypeError("Invalid or duplicate group");
    keys.add(group.key);
    for (const card of group.cards) {
      if (!card.id || !card.code || seen.has(card.id)) throw new TypeError("Invalid or duplicate physical card");
      seen.add(card.id);
    }
  }
  if ([...selectedIds].some(id => !seen.has(id))) throw new TypeError("Stale selection");
}

export function manualSelectionInfo(groups, selectedIds) {
  validate(groups, selectedIds);
  const touched = groups.filter(g => g.cards.some(c => selectedIds.has(c.id)));
  const whole = touched.length === 1 && touched[0].cards.every(c => selectedIds.has(c.id));
  const index = whole ? groups.indexOf(touched[0]) : -1;
  return { selected: selectedIds.size, touched: touched.length, whole, index };
}

export function manualCapabilities(groups, selectedIds) {
  const { whole, index } = manualSelectionInfo(groups, selectedIds);
  const touched = groups.filter(g => g.cards.some(c => selectedIds.has(c.id)));
  return {
    group: selectedIds.size > 1 && !whole,
    split: touched.some(g => g.cards.length > 1),
    left: index > 0,
    right: index >= 0 && index < groups.length - 1,
  };
}

// The caller supplies the full bijection verified by handRefreshBindings.
// Equal card codes must never merge their distinct physical occurrences.
export function rebindManualHistory(history, bindings) {
  try {
    return history.map(entry => {
      validate(entry.groups, new Set(entry.selectedIds));
      const groups = entry.groups.map(group => ({ ...group, cards: group.cards.map(card => {
        const next = bindings.get(card.id);
        if (!next || next.code !== card.code) throw new TypeError("Changed hand");
        return next;
      }) }));
      const selectedIds = entry.selectedIds.map(id => bindings.get(id).id);
      validate(groups, new Set(selectedIds));
      return { ...entry, groups, selectedIds };
    });
  } catch {
    return [];
  }
}

export function editHandGroups(groups, selectedIds, operation) {
  const capabilities = manualCapabilities(groups, selectedIds);
  if (!Object.hasOwn(capabilities, operation)) throw new TypeError("Unknown manual arrangement operation");
  const result = { groups, selectedIds: new Set(selectedIds), changed: false };
  if (!capabilities[operation]) return result;
  const usedKeys = new Set(groups.map(g => g.key));
  const fresh = cards => {
    const base = `manual:${JSON.stringify(cards.map(c => c.id))}`;
    let key = base, suffix = 0;
    while (usedKeys.has(key)) key = `${base}:${++suffix}`;
    usedKeys.add(key);
    return { key, pattern: 0, minor: 0, value: null, manual: true, cards };
  };
  let next;
  if (operation === "group") {
    const selected = groups.flatMap(g => g.cards.filter(c => selectedIds.has(c.id)));
    const first = groups.findIndex(g => g.cards.some(c => selectedIds.has(c.id)));
    next = groups.flatMap((group, index) => {
      const rest = group.cards.filter(c => !selectedIds.has(c.id));
      const remaining = rest.length === group.cards.length ? [group] : rest.length ? [fresh(rest)] : [];
      return index === first ? [fresh(selected), ...remaining] : remaining;
    });
  } else if (operation === "split") {
    next = groups.flatMap(group => {
      if (!group.cards.some(c => selectedIds.has(c.id)) || group.cards.length === 1) return [group];
      const rest = group.cards.filter(c => !selectedIds.has(c.id));
      const parts = group.cards.filter(c => selectedIds.has(c.id)).map(c => fresh([c]));
      // Preserve the side on which the selected run began; unrelated cards stay together.
      return selectedIds.has(group.cards[0].id)
        ? [...parts, ...(rest.length ? [fresh(rest)] : [])]
        : [...(rest.length ? [fresh(rest)] : []), ...parts];
    });
  } else {
    next = [...groups];
    const index = next.findIndex(g => g.cards.some(c => selectedIds.has(c.id)));
    const destination = index + (operation === "left" ? -1 : 1);
    [next[index], next[destination]] = [next[destination], next[index]];
  }
  return { groups: next, selectedIds: new Set(selectedIds), changed: true };
}
