---
version: alpha
name: KSPlay GuanDan Service
description: A bilingual single-player Guandan table with familiar green felt and quiet recovery controls.
colors:
  primary: "#8df0bd"
  background: "#034b35"
  text: "#f7f1df"
  accent: "#e6c77d"
typography:
  sans:
    fontFamily: 'Inter, system-ui, "PingFang SC", "Microsoft YaHei", sans-serif'
  serif:
    fontFamily: 'Georgia, "Times New Roman", "Songti SC", serif'
  mono:
    fontFamily: '"SFMono-Regular", Consolas, "Liberation Mono", monospace'
rounded:
  DEFAULT: "18px"
  button: "24px"
spacing:
  recovery-padding: "22px"
components:
  button: {}
  card: {}
  dialog: {}
---

# KSPlay GuanDan Service design context

## Overview

The reference is a physical green-felt card table, not an administration dashboard.
Players should recognize their seat, current turn, cards and actions immediately.
The public application supports Chinese and English, desktop and landscape phones.
Target usage is casual Guandan play; no additional geographic-market claim is made.
Keep the existing cream type, mint controls and gold accents. Avoid added badges,
oversized dialogs or a competing visual treatment for routine states.

This file documents, not generates, runtime tokens. `web/styles.css` owns palette
and font variables; `web/play.css` owns table geometry and component states. Runtime
CSS remains canonical. This public package omits developer test source; manually
check shared geometry and localized state labels using the checklist below.

## Colors and typography

Cream is primary readable text, muted green secondary copy, mint primary actions,
and gold selected/current-turn emphasis. Error recovery uses the same deep-felt
surface, a gold boundary and localized text; meaning must not rely on color alone.
System Chinese fallbacks preserve mixed-script readability. Monospace is limited
to small technical identifiers; no new font dependency is required.

## Layout, depth and shapes

One uniformly scaled 16:9 canvas fits the viewport. Landscape is recommended on
phones. Cards, identity lanes, clocks and the action dock retain reserved areas;
do not independently stretch these elements. Hand grouping preserves physical
card identity. Ordinary ranks are A > K > Q > J > 10 > 9 ... > 3 > 2; current level
and jokers are promoted separately. Never use the landlord-game 2-above-A order.

Failure recovery is a persistent 400px panel at 150px from the canvas top,
with 22px padding and 18px corners. Its controls have 48px unscaled height and
24px corners. It appears only for paused/error rooms and never changes hand layout.
Shadows establish overlays; they must not obscure cards or action labels.

## Components and behaviour

Native buttons expose busy/disabled states and cream focus outlines. Native
dialogs own modality and focus restoration. Card selection has click and keyboard
alternatives to dragging. Existing short card motion obeys reduced-motion settings.
Recovery uses no animation or skeleton: the previous authoritative hand remains.

AI failure pauses the room before any move is committed. The player can explicitly
retry the same decision or return to entry. Never silently substitute the example
bot for a failed configured AI. Permanent engine faults offer only return to entry.
Auth revocation and missing rooms stop automatic reconnect, clear private card UI
and show an actionable entry message. Ordinary network reconnect uses capped delay.
Late room responses must not reintroduce a departed room or overwrite its successor.

| Capability | Canonical owner | Source of truth | Allowed variants | Verification |
| --- | --- | --- | --- | --- |
| Form | `web/play.js` entry handlers | Solo API schema | Native validation and localized request errors | Submit valid/invalid entry and inspect API responses in both locales |
| Toast | `web/play.js` showToast | Localized operation feedback | Brief nonblocking feedback only | Trigger feedback in Chinese/English and check duration and overlap |
| Table Selection | `web/frontend/hand-interactions.mjs` | Authoritative legal actions and physical card IDs | Click, sweep, keyboard and full-card picker | Exercise each input method, checking duplicate physical cards and legal-list membership |
| Scrollbar | `web/play.css` native dialogs | Viewport-bounded history/picker | Internal dialog body scroll | Open long history/picker on desktop and landscape phone; scroll without moving the table |

## Do and do not

- Do preserve existing table colors and ordinary playing-card faces.
- Do keep actionable errors visible until recovery or exit.
- Do localize dynamic error and busy labels through `solo-english.mjs`.
- Do not duplicate API schema or invent legal actions in the browser.
- Do not disclose private model endpoints, internal errors or opponent hands.

## Manual verification checklist

- Open `/solo` from the local service at desktop and landscape-phone sizes.
  Check uniform 16:9 scaling, reserved card/action lanes and mixed-script labels.
- Switch both locales and exercise entry, dialogs, toast, focus, selection and
  busy/disabled controls. Check keyboard alternatives and reduced motion; use a
  physical touch device for gesture acceptance, not desktop simulation alone.
- Exercise every arrangement mode and candidate partition; compare physical
  card IDs/counts and natural A-over-2 ordering, with level/jokers promoted.
- Force a local AI failure and inspect the persistent 400px recovery panel,
  150px top offset, 22px padding, 18px corners and 48px controls with 24px corners.
  The previous authoritative hand must stay unchanged until explicit retry/exit.
- Disconnect/reconnect, revoke authentication, remove a room and delay old room
  responses. Check bounded reconnect, versioned retry, entry recovery and cleanup
  of private views. Follow the API and complete-game checks in
  [DEVELOPMENT](docs/DEVELOPMENT.md); do not infer correctness from appearance.

Developer module regressions were run separately before this test-free repack;
their source is not shipped. These manual checks and prior regressions are not an
exhaustive accessibility audit or proof of every game state.
