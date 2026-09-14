# IISH Daily — Brand Kit (v2, REBRAND — 2026-09-14)

> ⚠️ **SUPERSEDES the 2026-07 identity.** Ethan confirmed 2026-09-14 this rebrand is approved
> (Nicki/Emily signed off) and fully replaces the old wordmark/palette/patterns — not a parallel
> exploration. Source: brand deck screenshot, saved in full at
> `references/inputs/iish/rebrand-2026-09/brand-deck-source.png` (crops of each section alongside it).
> The old identity's files are archived, not deleted, at
> `references/inputs/iish/_archive-2026-jul-identity/` — the old version of this doc (8-color
> checker/burst system) is preserved in git history: `git show 45dfe12^:references/docs/iish-brand-kit.md`.
>
> **Update 2026-09-14 (same day, corrected mapping):** Ethan confirmed the pink-bg/dark-red-text
> pairing on the "Unflavored" card **is** the real packaging/SKU color assignment, and that
> **Blue Raspberry is now an active 4th flavor in the visual/design system** alongside Unflavored,
> Peach Rings, and Tropical — the squiggle-pattern language is intentionally built for 4-flavor
> variety, hand-drawn feel, colors combined/mixed across flavors. **One open item that's separate
> from design:** the locked Pharmachem SFP formula line is still only Unflavored/Peach Rings/Tropical
> (`iish-launch-resources/strategy/iish-brand-book.md` CLAUDE.md) — Blue Raspberry doesn't have a
> regulatory-approved formula yet as of this writing. That's a formulation/compliance question for
> Emily/Nicki/Pharmachem, not a design blocker — build the visual system for all 4, just don't say
> Blue Raspberry is "shipping" without checking formula status first.

## Logo
Cursive/script lowercase **"iish"** — connected single-stroke ligature, the *s* loops up into the
ascender of the *h*, two round dots over the *ii*. **No terminal period** in this version (the old
"iish." had one — confirmed as an intentional change, not an oversight, per 2026-09-14 approval).

File: `references/inputs/iish/rebrand-2026-09/wordmark-black-on-white.png` (clean isolated crop,
black on off-white). Full deck also shows the wordmark recolored solid in every palette color
(see `wordmark-colorways.png`) — always solid one color, never multi-color/gradient.

## Palette — sampled from the source deck (see caveat below)

Colors below were pixel-sampled directly from the screenshot with PIL, not eyeballed — but a
screenshot is still a compressed, possibly color-shifted copy of the real design file. **Treat
these as close approximations, not final production hex values.** Get the real exported swatches
(Figma/Adobe file) before these go anywhere near print, packaging proofs, or a style guide PDF.

| Role | Approx. Hex | Where it's used |
|---|---|---|
| Anchor (pink) | `#F2C2DA` | **Unflavored** packaging background |
| Dark red / berry | `#8A2A3D` | **Unflavored** wordmark, text, and squiggle icon (sampled range `#7F1829`–`#9A3F52` depending on element — get the real export) |
| Light blue / periwinkle | `#ADC7DB` | **Peach Rings** packaging background |
| Coral/orange | `#F0765A` | **Peach Rings** wordmark, text, and squiggle icon |
| Dark green | `#456B58` | **Tropical** packaging background |
| Sage/light green | `#8FA895` | **Tropical** wordmark, text, and squiggle icon |
| Mauve/mulberry | `#A97696` | **Blue Raspberry** packaging background |
| Lavender-blue | `#AAA7C1` | **Blue Raspberry** wordmark, text, and squiggle icon |
| Tan/beige | `#E5D6CE` | Secondary/neutral field — seen paired with mauve wordmark in one deck panel, not tied to a specific flavor card |
| Mustard/yellow | `#F3D476` | Seen paired with sage bg in one deck panel, not tied to a specific flavor card |

All values pixel-sampled directly from the screenshot with PIL (not eyeballed), but a screenshot is
still a compressed, color-shifted copy of the real file. **Treat as close approximations, not final
production hex** — get the real exported swatches (Figma/Adobe) before print, packaging proofs, or a
style guide PDF. The tan/mauve and sage/yellow pairings appear in the deck's wordmark-colorway test
panel but don't reconcile with the 4 named flavor cards below — kept here for reference only.

## Flavor → color lock (bg / accent, per the 4 explicitly-labeled packaging cards)

| Flavor | Background | Wordmark / text / icon | Icon shape |
|---|---|---|---|
| **Unflavored** | Pink `#F2C2DA` | Dark red/berry `#8A2A3D` | Abstract continuous-line squiggle (no object) |
| **Peach Rings** | Light blue `#ADC7DB` | Coral `#F0765A` | Peach-shaped continuous-line squiggle |
| **Tropical** | Dark green `#456B58` | Sage `#8FA895` | Pineapple-shaped continuous-line squiggle |
| **Blue Raspberry** | Mauve `#A97696` | Lavender-blue `#AAA7C1` | Berry-cluster continuous-line squiggle |

This is the primary, confirmed mapping (2026-09-14) — bg and accent always swap in pairs, one flavor
never borrows another's colors. Formula status note above applies to Blue Raspberry; the other 3 are
regulatory-locked SKUs.

## Pattern / illustration style — two distinct assets, don't conflate them
1. **Per-flavor icon squiggle** (single color = that flavor's accent) — a continuous-line hand-drawn
   doodle, shape varies per flavor (abstract / peach / pineapple / berry-cluster), used on that
   flavor's own packaging card. This is the one to reference per-SKU generation.
2. **Multi-color squiggle tangle** (`wordmark-colorways.png`'s companion crop, sage+coral+slateblue+
   pink intertwined) — several flavors' squiggles overlaid in one piece of line art. This is a
   **family/texture asset**, not a single-flavor icon: use it for anything that represents the whole
   flavor range at once (a "meet the flavors" spread, a collage background, packaging-line group
   shots) where combining colors reads as intentional variety, not a mistake.

Both replace the old geometric half-circle-checker / citrus-burst patterns entirely — solid color
field + line-art icon(s), no more busy geometric backgrounds.

## Typography
Deck doesn't specify a font name — the flavor-label type (`UNFLAVORED`, `CREATINE + COLLAGEN`) is a
bold, condensed, geometric sans in all caps with wide letter-spacing on the sub-label. **No file/name
confirmed yet** — get the actual typeface from whoever built the deck (Figma link) before using it
in production copy; don't guess a system-font substitute without asking.

## Photography & film direction
Not addressed in this deck — carry forward the old brand book's golden-hour/real-kitchen photography
law (`iish-launch-resources/strategy/iish-brand-book.md` §4) until told otherwise; this rebrand looks
scoped to logo/palette/pattern/packaging-card system, not photography style.

## Prompt pattern for this identity
> "Flat packaging card for iish [FLAVOR] creatine + collagen — solid [COLOR] background, the
> cursive lowercase 'iish' wordmark (no period, connected ii-s-h ligature, two dots over the ii) in
> [WORDMARK COLOR], bold condensed caps label '[FLAVOR NAME]' and 'CREATINE + COLLAGEN' beneath it,
> one thin continuous-line squiggle icon of [ICON SUBJECT] in the lower portion. Clean, minimal,
> modern — not retro/geometric. Flat graphic design, not photorealistic lifestyle photography."
Use `wordmark-black-on-white.png` and the relevant flavor card crop from `flavor-mockups.png` as
`image_input` references — this system is flat/graphic, so image-conditioning matters even more
than the old photoreal-lifestyle prompts did for holding the wordmark shape correct.

## Generation paths (unchanged — this is a visual-identity change, not an infrastructure change)
Same Doppler-keyed providers as before: Nano Banana Pro (Google/Kie), GPT Image 1.5 (WaveSpeed —
key currently dead, needs rotation), video via Kling/Veo/Sora.
