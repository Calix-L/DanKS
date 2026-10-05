# KSPlay GuanDan Service · V1

This is generation one: the old deployed CardKS browser plus an adapted public
backend derived from that version's captured service/worker. It is not identical
private production inference. It does not contain generation-two UI, recovery
improvements or public-trick-history additions.

## Exact versus adapted

| Component | Boundary |
| --- | --- |
| `web/play.html`, `play.js`, `play.css`, `styles.css`, logo | byte-identical captured files |
| `web/frontend/` | byte-identical ten-module static import closure, including legacy shared utilities |
| DTOs, storage, models and backend initializer | byte-identical captured files; shared compatibility types are not another game's runtime |
| `backend/solo_app.py` | adapted captured file: portable settings, `web/` location, local audit path and generic audit label |
| `backend/human_tables.py` | adapted captured file: public worker target, generic profile, GuanDan-only runtime boundary and generic metadata |
| `workers/table_worker.py` | adapted captured worker: direct public rules imports, no server paths/private loader/other-game adapter; generic HTTP policy and public committed history |
| `rules/`, settings and AI connector | reviewed pure public/reference components, not inference implementation |
| `backend/presentation.py` and arranger source | full source-based splitter with portable local build/invocation, separate from inference |

The full advanced splitter supports the original layout modes and alternate
layouts. The old UI wording, rank-display utilities and reconnect behavior are
preserved, including historical limitations.
The public referee uses ordinary `A > ... > 3 > 2`, with level/joker promotion.
Its example bot's natural sort follows that same order. Do not infer rules from
old browser display order.

Generation one submits only the canonical `action_index`; `selected_cards`
substitution is rejected. It retains existing stale-version and action-ID checks,
but does not acquire the later generation's AI pause/retry or connection recovery
fixes. Endpoint failure enters the old error flow, without model fallback.

## Source identity

[SOURCE_IDENTITY.json](SOURCE_IDENTITY.json) records exact-file SHA-256 hashes,
the hashes of adapted captured inputs, and the public replacement list. The old
public entry page and its JS/CSS were also compared against served bytes:

| File | SHA-256 |
| --- | --- |
| `play.html` | `b8b09fc6ecd41e50b6723a048a262251581a09829a5d894eafc2fd3512e873d8` |
| `play.js` | `4c41797b92ebf73e6e62ea68d16a1233fec9842f77a69ed6c08508c61429c6b7` |
| `play.css` | `7a1bf31d5a97357d63d6c87a43ddacce6cc8b89bd7dbb80fd34dbd7f9e938692` |

Identity checks prove matching bytes. The owner has explicitly confirmed source
and artwork ownership for this release; these rights statements are recorded in
[NOTICE](../NOTICE), while trademarks remain separate. No private capture tree,
source-location paths, deployment
topology, credentials, models, runtime binaries, logs or data are included.

中文：第一代为旧版网页与该版本后端的公共适配，网页字节一致，完整高级拆牌源码保留，
私有推理不包含；不应宣称已具备第二代恢复改进。所有者已确认源码与素材权利。
