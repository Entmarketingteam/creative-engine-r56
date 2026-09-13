# IISH Daily — Brand Kit

> Sourced from the LOCKED visual identity in `~/Desktop/iish-launch-resources/strategy/iish-brand-book.md` §4
> (ground truth: `~/Downloads/iish - brand book.pdf`, 13pp + logo/colorway packs, 2026-07-13).
> This is the "brand kit" a tool like Lovart would hold internally — feed it as reference/context on
> every IISH generation call so output stays on-brand without re-deriving colors/type/logo each time.
> Do not invent colors, fonts, or logo treatments outside what's below — if something's not here, it's
> not locked yet (check §11 Open Locks in the brand book).

## Logo
Lowercase **"iish."** wordmark — playful reversed **s**, double dots over the *ii*, terminal period.
Never all-caps, never respelled. Wordmark recolors per surface from the palette below.

Files (in `references/inputs/iish/`):
- `iish_logo_black.svg` / `.png` — primary, for light grounds
- `iish_logo_white.svg` / `.png` — for dark grounds
- `board_logo_colorways.png` — logo recolored across the full palette
- `board_logo_on_color.png` — logo placed on colored backgrounds (contrast reference)
- `board_flavor_color_system.png` — the 3-flavor colorway pattern system

## Palette (LOCKED — 8 colors)

| Token | Hex | Pantone | Role |
|---|---|---|---|
| Cool Pink | `#F5D5DC` | 698 C | Blush ground — backgrounds, delicate highlights |
| Berry | `#6B2240` | 7421 C | Anchor — headings, dark UI, primary wordmark |
| Barbie Pink | `#E03C84` | 812 C | Bold CTA + feature highlights |
| Tangerine | `#D95F38` | 7416 C | Energetic accents (Peach Rings colorway lead) |
| Violet-Blue | `#8E9DC8` | 2718 C | Calm accents/backgrounds (Unflavored colorway lead) |
| Leaf | `#4E6B35` | 7490 C | Earthy nature accents/icons |
| Rainforest | `#1F5C52` | 7721 C | Deep green anchor (shaker, Tropical colorway) |
| Chartreuse | `#A8B840` | 381 C | Vivid pop against dark grounds (Tropical lead) |

## Flavor colorway system
- **Peach Rings** → tangerine/peach ring pattern
- **Unflavored** → violet-blue/pink half-circle checker
- **Tropical** → green/citrus burst pattern

Formats: stand-up gusset bag (scoop) + single-serve stick packs. **No jar** (dropped 2026-08-02).
Merch: rainforest-green shaker, cool-pink lid.

## Typography
- **Real Head Pro** — display/headings, caps
- **Cormorant Garamond SB** — serif accent + italic taglines ("it's in simple habits")
- **SF Pro Medium** — body/UI

## Aesthetic
**Retro-playful joy, not wellness minimalism.** Geometric retro patterns (half-circle checker,
ring/donut motifs), candy colorways per flavor, color-blocked logo. Fun, warm, confident —
deliberately breaks from Rhode/Alo muted-luxury.

## Photography & film direction
Golden-hour natural light, real kitchens/counters/gym bags, real women 32–48. No lab coats, no
strobe, no clinical anything. Phone-shot-friendly standards for build-in-public content.

## Proven prompt (image) — from brand book §5, use `board_flavor_color_system.png` as `image_input`
> "Editorial lifestyle photo of an iish. collagen + creatine stand-up pouch — retro geometric
> half-circle pattern in soft pink and periwinkle blue, playful lowercase 'iish.' berry wordmark —
> on a sun-drenched real kitchen counter with a morning coffee cup and a kid's cereal bowl at the
> edge of frame. Golden hour light through a window, soft shadows. Joyful, warm, lived-in — candy-
> colored retro optimism, not luxury, not clinical. Photorealistic, 8k."

## Prior working generation (proof this pipeline already works)
2026-08-30: `ent-marketing:creative` skill generated a real 6-shot photorealistic product carousel
("Midnight Glow" concept) via **Nano Banana Pro on Replicate**, chaining `image_input` off a first
hero shot so packaging/wordmark held consistent across all 6 shots. `REPLICATE_API_TOKEN` confirmed
live in Doppler `ent-agency-automation/dev`. That output was a color-direction *exploration*, not
these locked colors — regenerate against this file's real palette/assets instead of reusing it.

## Available generation paths (all pre-paid, in Doppler `ent-agency-automation/dev`)
| Need | Model | Key(s) | Notes |
|---|---|---|---|
| Image | Nano Banana Pro | `GOOGLE_API_KEY` (direct, default) or `KIE_API_KEY` / `REPLICATE_API_TOKEN` (fallback) | `tools/image_gen.py`, provider="google"\|"kie" |
| Image | GPT Image 2 | `OPENAI_API_KEY` direct, or via `KIE_API_KEY` | Not yet wired into `tools/providers/` — add if needed |
| Video | Seedance 2.x | `KIE_API_KEY` or `WAVESPEED_API_KEY` | `tools/video_gen.py`, provider="kie"\|"wavespeed" |
| Video | Veo 3.1 | `GOOGLE_API_KEY` | Native audio/dialogue, default per CLAUDE.md |
| Manual only | Midjourney | none (Discord, no API) | Use for one-off explorations, not the automated pipeline |
| Compile/layout | Canva | claude.ai MCP connector (already live this session) | Assemble a shareable kit sheet/deck from generated assets |
