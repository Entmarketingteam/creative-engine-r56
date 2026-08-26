# Preset Library — Proven Prompt Formulas

Named, reusable prompt templates that have already produced strong results, kept here so the next
asset (different subject, different brand, different style, image vs. video) is a **variable swap**,
not a from-scratch rewrite. Add a new preset here whenever a prompt earns its keep in review.

Each preset lists: what it's for, the model it was verified against, the full template with
`{{variables}}` marked, and what specifically to change when reusing it.

---

## Preset: Editorial Portrait ("Vogue" style)

**Use for:** posed, magazine-quality beauty/product portraits where the subject is clearly recognizable
and the product/text must render exactly. Still image, with an optional slow-motion companion video.

**Verified against:** Nano Banana Pro (image), Veo 3.1 (video companion)
**Origin:** Jones Road Eyeshadow Stick ad → adapted for Nicki Entenmann x Bloom Clear Protein (2026-08-24)

### Image template

```
{{aspect_ratio}}. A high-fashion beauty editorial portrait, full bleed.
Primary subject — the woman shown in the reference images (match her exact facial identity, hair,
and features precisely — this is the priority reference, follow it closely), {{age_range}},
wearing/holding {{product}} — {{how_the_product_is_used_or_worn}}.
{{PRODUCT_NAME}} text clearly visible and spelled correctly on {{product_surface}}.
Wearing {{wardrobe}}, {{accessories}}.
{{background_color}} studio background seamlessly extending to all edges of the frame —
no text, no logos, no masthead, no cover lines, pure portrait.
Soft diffused beauty lighting, key light slightly above and to the right, gentle fill from the left,
subtle rim light separating subject from background.
Natural skin texture with visible pores, luminous but not airbrushed, real skin.
Shot on medium format Hasselblad, 80mm lens, f/2.8, razor-sharp focus on eyes and product.
Editorial high-fashion photography.
text_accuracy: preserve all visible product text exactly — {{PRODUCT_NAME}}, spelled correctly,
no substitutions.
No watermarks, no heavy contouring, no dramatic makeup, no extra packaging text not in the reference.
Using input images for subject facial identity (priority) and the final reference image for
product identity — preserve product text exactly as shown.
```

### Video companion template (same subject/product, no dialogue)

```
The woman shown in the reference images (matching her exact facial identity), wearing {{wardrobe}},
slowly turns her head toward camera with a quiet confident expression. She lifts {{product}}
near {{gesture_location}}, holds it still for a beat, then the faintest smile.
Slow deliberate movement, editorial energy, soft diffused beauty lighting, {{background_color}}
background. No dialogue, ambient studio silence, magazine cover coming to life.
```

**To reuse for a real magazine-cover version** (not just the portrait crop): add the actual masthead
block — `The word {{MAGAZINE}} in large bold serif capitals ... with the model's head overlapping in
front of the letters so {{MAGAZINE}} appears behind and partially occluded by the model, exactly like
a real {{MAGAZINE}} cover layout. Cover headline text along the left side reading {{HEADLINE}}...` —
this is what makes a *literal* magazine cover vs. the clean portrait above. Keep the rest of the
template identical; only the masthead/headline block changes per outlet.

**Swap checklist:**
- New subject → new reference image set uploaded via `tools/kie_upload.py`; the "match her exact
  facial identity" line stays word-for-word, it's the reference images that carry identity, not the text.
- New brand/product → swap `{{PRODUCT_NAME}}`, `{{product_surface}}`, `{{how_the_product_is_used_or_worn}}`.
  Keep the `text_accuracy:` line — it's what stops the model from garbling product text.
- New visual style (not editorial) → this whole preset is the "posed" register. For a candid/UGC
  register instead, use the Selfie-Cam preset below, not a tweak of this one.
- New asset type (photo → quick video) → generate the still first, mark it Approved in Airtable,
  then use the video companion template pointed at the same reference images. Don't regenerate a
  fresh subject likeness for the video — feed it the approved still.

---

## Preset: Selfie-Cam UGC ("SELCAM")

**Use for:** candid, authentic, UGC-style video where the subject appears to be filming themselves —
the opposite register from the editorial preset. This is the format CLAUDE.md's "Workflow 2" already
documents structurally; this entry is the *worked, proven example* to copy from.

**Verified against:** Kling 3.0 (Kie AI)
**Origin:** Bloom Clear Protein "Kitchen Pour" — Nicki Entenmann demo (2026-08-24)

### Structured template

```
dialogue: {{first_person_line_under_150_chars}}
action: {{what_the_hands_and_body_do_with_the_product}}. {{ends_with}} Maintains eye contact with camera.
camera: cinematic locked-off shot, slow subtle push-in, {{lighting_description}}, shallow depth of field
emotion: {{2_to_3_word_emotional_register}}
voice_type: {{tone}}, {{age_gender}}, {{energy_descriptor}}
```

**Real example (Bloom Clear Protein, kitchen pour):**
```
dialogue: okay this is my new morning ritual... bloom clear protein just hits different
action: character finishes pouring a scoop of powder into a tall glass of water, the pink cloud
swirls as it dissolves. She sets the scoop down, picks up the glass and gives it a gentle swirl,
then lifts it toward camera with a satisfied smile. Maintains eye contact with camera.
camera: cinematic locked-off shot, slow subtle push-in, warm morning golden hour light through
kitchen windows, shallow depth of field
emotion: calm confidence, morning zen, quiet satisfaction
voice_type: warm, relaxed, young adult female, soft-spoken morning energy
```

**Swap checklist:**
- New subject → same rule as above: reference images carry identity, the template text doesn't
  need a name or description of the person.
- New product/moment → rewrite `dialogue` and `action` around the new use-case; keep `camera` starting
  with "fixed camera, no music" (or "cinematic locked-off shot" for a slightly more produced feel)
  and always keep "maintains eye contact with camera" — it's what sells the authenticity.
  For Veo 3.1 instead of Kling/Sora, collapse this into prose with dialogue in quotes — see CLAUDE.md
  Workflow 2, "For Veo 3.1."
- New model (Sora 2 Pro, or whatever supersedes Kling 3.0) → the structured fields
  (`dialogue`/`action`/`camera`/`emotion`/`voice_type`) are a prompt convention, not a Kie AI API
  parameter — they'll keep working as long as the underlying model accepts a single text prompt.
  Re-verify max prompt length against `kie-ai-api.md` when switching models.

---

## Voice cloning (ElevenLabs) — not yet integrated, consent-gated

Explored 2026-08-24 for the Nicki x Bloom test. **Do not build this without explicit sign-off from
the voice owner** — ElevenLabs' Instant Voice Clone requires whoever uploads the sample to affirm
they hold the rights/consent to clone that specific voice; that's a ToS requirement with legal
weight, not an internal policy call. Confirmed at the time: ~365s of candidate source audio
identified (121s/93s/41s/92s clips, well above the 1-3 min minimum) but nothing was converted,
transcribed, or uploaded pending confirmation this is actually Nicki's voice and she's consented.

If/when cleared to proceed: add `tools/providers/elevenlabs.py` following the same provider
pattern as `google.py` / `kie.py` / `wavespeed.py`, and add `ELEVENLABS_API_KEY` to `.claude/.env`
(pull from Doppler `ent-agency-automation` once added — it isn't there yet).
