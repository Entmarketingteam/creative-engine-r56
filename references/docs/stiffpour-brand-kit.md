# Stiff Pour — Brand Kit (dry run, 2026-09-16)

> Second brand tested through the creative engine, after IISH Daily — proving the pipeline
> generalizes rather than being IISH-specific. Nicki's friend's brand, stiffpour.co, single
> product: the Stiff Pour wine aerator. Sourced from real material, not invented:
> - `StiffPour_BrandingGuide.pdf` (this folder) — pulled fresh from Google Drive
>   "FINALIZED STIFF POUR BRANDING GUIDE AND DOWNLOADS" folder, modified 2026-09-16 (today) —
>   this is the current, locked identity per that folder name.
> - `wordmark-stiffpour.png` / `logomark-sp.png` — from `SP_LOGOS.zip` in that same Drive folder.
> - `product-in-hand-current-packaging.jpg` — a real photo (Nicki holding the actual retail box),
>   pulled from Drive folder `STIFF POUR/nicki-pics` (also mirrored locally at
>   `~/Desktop/stiffpour-site/reference/brand/nicki-pics/`, but that local copy was unreadable at
>   the time — EPERM/quarantine issue on that specific subtree, not a code problem. Worked around
>   it by pulling the same files straight from Drive instead — worth re-checking now that Desktop
>   access came back, in case it was actually part of the same access issue.)
> - Live site (stiffpour.co, fetched 2026-09-16): still running the OLDER identity — "Pinky
>   Promise Stiff Pour w weiner" logo, minimal/neutral color treatment, tagline **"When life gets
>   hard."** Confirms the current retail box (see product photo) already uses navy cursive
>   script on white — the finalized guide is an *extension* of what's live, not a wild pivot.
> - `~/Desktop/stiffpour-site/CLAUDE.md` (the Shopify theme rebuild repo) describes a DIFFERENT
>   provisional palette ("near-black base, wine-dark accent, warm highlight") — **that guidance is
>   now superseded** by the finalized guide below; flag to Ethan/Emily before the site rebuild
>   locks colors, since the two disagree.

## Logo
Cursive script **"stiff pour"** wordmark (lowercase, connected script) + a separate **"sp"**
monogram logo mark (interlocked cursive s+p). Matches what's already on the real retail
packaging — this is not a new logo, just the same one carried into an expanded system.

Files: `wordmark-stiffpour.png`, `logomark-sp.png` (both black on transparent/white).

## Palette (LOCKED, "option #2" — pulled from real PDF text, not sampled/eyeballed)

| Role | Hex | Source |
|---|---|---|
| Navy (anchor) | `#161441` | Matches the navy cursive text already on the real retail box |
| Magenta | `#a9227e` | New — expansion color |
| Yellow | `#f2e11d` | New — expansion color |
| Cream | `#f7f5e4` | New — expansion color |
| Black / White | — | Logo neutral pairing |

These came straight out of the branding-guide PDF's text layer (hex codes printed directly in
the doc), not pixel-sampled — highest-confidence palette we've sourced yet across either brand.

## Typography
- **Pinky Promise** — display/header/logo font (script). Actual font files exist:
  `~/Desktop/stiffpour-site/reference/brand/fonts/PinkyPromise.ttf` / `.otf` (was unreadable
  locally during the EPERM incident — recheck now that Desktop access is back).
- **Priori Sans OT** — subhead/body font.

## Voice (from `~/Desktop/stiffpour-site/CLAUDE.md`)
Playful, cheeky, innuendo-forward — "When life gets *hard*." Liquid Death confidence, premium
barware finish. Never corporate, never generic ecommerce filler, never explain the joke.

## Product
Single SKU: **Stiff Pour wine aerator**. No flavor/variant system (unlike IISH) — this brand's
"variety" surface is packaging/color treatment and content, not SKUs.

## Prompt pattern for this identity
> "Photorealistic product shot of the Stiff Pour wine aerator retail box — white box, navy
> cursive 'stiff pour' script wordmark (#161441), tagline 'when life gets hard.' in navy sans
> beneath it — [SCENE]. Confident, premium-but-playful barware mood, not corporate, not
> clinical. Photorealistic, 8k."
Use `product-in-hand-current-packaging.jpg` as the `image_input` reference for packaging
fidelity — it's a real photo of the actual retail box, the strongest ground truth available.

## Generation paths
Same Doppler-keyed providers as IISH: Nano Banana Pro via Google (`GOOGLE_API_KEY`) proven best
for holding wordmark/packaging fidelity; Kie/WaveSpeed for video if needed later.

## Open items (flag, don't guess)
1. The site-rebuild repo's palette guidance conflicts with this finalized guide — needs
   Ethan/Emily to confirm which wins before `stiffpour-site` theme work locks colors.
2. Local `~/Desktop/stiffpour-site/reference/brand/` was unreadable mid-session on 2026-09-16
   (EPERM on every file inside, `ls` worked but open didn't) while the rest of Desktop read fine
   at that moment — then later the *entire* Desktop folder lost access session-wide and had to be
   re-granted in System Settings. Possibly the same root cause the whole time. Recheck that
   specific subtree now that access is restored before trusting this note long-term.
3. Original local logo set includes files literally named "w weiner" (an explicit graphic pun
   variant) — not yet seen (blocked at the time), may differ from the clean `SP_LOGOS.zip` set
   used here. Confirm which is meant for which channel before shipping broadly.
