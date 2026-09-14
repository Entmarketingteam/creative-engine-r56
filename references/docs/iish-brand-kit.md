# IISH Daily — Brand Kit (v2, REBRAND — 2026-09-14)

> ⚠️ **SUPERSEDES the 2026-07 identity.** Ethan confirmed 2026-09-14 this rebrand is approved
> (Nicki/Emily signed off) and fully replaces the old wordmark/palette/patterns — not a parallel
> exploration. Source: brand deck screenshot, saved in full at
> `references/inputs/iish/rebrand-2026-09/brand-deck-source.png` (crops of each section alongside it).
> The old identity's files are archived, not deleted, at
> `references/inputs/iish/_archive-2026-jul-identity/` — the old version of this doc (8-color
> checker/burst system) is preserved in git history: `git show 45dfe12^:references/docs/iish-brand-kit.md`.
>
> **Open item:** flavor names in the source deck are internally inconsistent (see below) and don't
> fully match the regulatory-locked SFP formula line (Unflavored / Peach Rings / Tropical — the only
> three real formulated SKUs, per `iish-launch-resources/strategy/iish-brand-book.md` and the silo
> rule). Per Ethan 2026-09-14: apply the new visual system to those three real flavors only.
> **Do not use "Fruit Punch," "Lemon Lime," or "Blue Raspberry"** as shipped flavor names/colors —
> they appear in the deck but have no formulated product behind them. Flag to Ethan/Emily if the deck
> author meant something different; don't silently guess.

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
| Anchor (pink) | `#F2C2DA` | Background field behind primary wordmark; **Unflavored** flavor color |
| Secondary (tan/beige) | `#E4D3C6` | Secondary background field |
| Mauve/purple | `#9A5A83` | Wordmark-on-tan color; labeled "Fruit Punch" in deck — **not a real flavor, don't ship** |
| Coral/orange | `#F0765A` | Wordmark-on-blue color; labeled **"Peach Rings"** in the swatch grid — matches the real formulated flavor, use this |
| Yellow/mustard | `#F3D476` | Swatch grid "Unflavored" light variant |
| Dark green | `#4A6E58` | Swatch grid labels this "Unflavored" but the flavor-mockup section uses the same green for **"Tropical"** (pineapple icon) — use for **Tropical**, the icon makes the intent clear |
| Sage/light green | `#AABCB3` | Paired light variant of the dark green |
| Burgundy/maroon | `#852133` | Labeled "Fruit Punch" dark variant — **not a real flavor** |
| Light blue | `#ADC7DB` | Wordmark-on-sage color; labeled "Lemon Lime" in grid, "Peach" in flavor mockups — inconsistent naming, **do not use for real Peach Rings** (coral already owns that flavor) — hold this color until Ethan/Emily clarify what it's actually for |
| Slate blue | `#6E86A0` | Dark variant of the light blue |

## Flavor → color lock (applying the new system to the REAL 3 flavors only)

| Real flavor (locked formula) | Color | Line-art icon |
|---|---|---|
| **Unflavored** | Pink `#F2C2DA` bg, deep berry/maroon wordmark+text | Abstract continuous-line squiggle (no object), see `flavor-mockups.png` far left |
| **Peach Rings** | Coral/orange `#F0765A` | Apple/peach-shaped continuous-line squiggle |
| **Tropical** | Dark green `#4A6E58` bg, sage-green icon | Pineapple-shaped continuous-line squiggle |

Mauve/burgundy and the two blues are **held** — they belong to flavor names that don't exist yet.
Don't generate packaging using them until the flavor lineup question is resolved.

## Pattern / illustration style
**Continuous single-line squiggly doodle icons**, thin stroke, monochrome (icon color = the
flavor's accent color, e.g. coral icon on the coral card). Replaces the old geometric
half-circle-checker / citrus-burst patterns entirely. No more checker/burst backgrounds — the new
system uses solid color fields + one line-art icon, much more minimal.

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
