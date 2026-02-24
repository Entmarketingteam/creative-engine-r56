# Generate Content

You are running the Creative Content Engine workflow. Follow these steps precisely.

## Step 1: Gather Inputs

Ask the user for:
1. **Product name** - what product are we making ads for?
2. **Reference images** - confirm they've placed product photos in `references/inputs/`
3. **Number of variations** - how many ad variations to generate
4. **Style/mood preferences** - any specific aesthetic, aspect ratio, cinematic vs UGC, character details, scenes, etc.
5. **Image model** (optional) - Nano Banana Pro (default), Nano Banana, or GPT Image 1.5. Provider defaults to Google AI Studio (Nano Banana models) or WaveSpeed AI (GPT Image 1.5).

## Step 2: Analyze Reference Images

Read/view the reference images in `references/inputs/` to understand:
- Product appearance, color, shape, texture
- Brand name and visible text
- Key features to highlight

Consult `references/docs/prompt-best-practices.md` for model-specific prompt guidelines.

## Step 3: Write Prompts & Create Airtable Records

**IMPORTANT: Always create records in Airtable FIRST with the prompts, then tell the user to review them in Airtable before proceeding.** Do NOT just show prompts in chat — the user reviews and approves prompts in Airtable.

1. **Upload reference images** to Kie.ai (one-time, reuse URLs):
   ```python
   import sys; sys.path.insert(0, '.')
   from tools.kie_upload import upload_references

   ref_urls = upload_references(["references/inputs/product.jpg"])
   ```

2. **Get the next available Index** and **create Airtable records** with prompts AND reference images attached:
   ```python
   from tools.airtable import create_records_batch, get_next_index

   start_index = get_next_index()  # Ensures unique Index across batches

   records = create_records_batch([
       {
           "Index": start_index,
           "Ad Name": "ProductName - Description",
           "Product": "Product Name",
           "Reference Images": [{"url": ref_urls[0]}],
           "Image Prompt": "9:16. ...",
           "Image Model": "Nano Banana Pro",  # or "Nano Banana" / "GPT Image 1.5"
           "Image Status": "Pending",
       },
       {
           "Index": start_index + 1,
           # ... next variation
       },
       # ... more variations (increment from start_index)
   ])
   ```

3. **Tell the user to review prompts in Airtable.** Summarize what was created (names, brief prompt descriptions) and ask them to check the Image Prompt fields in Airtable. Wait for the user to confirm the prompts look good before proceeding to generation.

Each record MUST have:
- **Unique Index** — call `get_next_index()` to get the starting value, then increment for each record
- Descriptive **Ad Name** (product + scene description)
- **Reference Images** attached (the Kie.ai hosted URL for the reference used by that specific ad)
- **Image Prompt** following best practices
- **Image Model** — set at record creation ("Nano Banana Pro", "Nano Banana", or "GPT Image 1.5")
- **Image Status** set to "Pending"

Different ads can use different reference images — attach the appropriate one per record.

## Step 4: Cost Confirmation (MANDATORY - DO NOT SKIP)

**HARD RULE: NEVER call any generation endpoint without showing cost and getting explicit user confirmation first.**

After the user has reviewed and approved the prompts in Airtable, show:
1. Exactly what will be generated (number of records, 2 variations each)
2. Model and provider being used (e.g., "Nano Banana Pro via Google AI Studio")
3. Per-unit cost and total (use `config.get_cost(model, provider)` for accurate pricing)
4. **STOP and wait** for the user to explicitly confirm

Example: "This will generate 4 ads x 2 variations = 8 images using Nano Banana Pro (Google) at ~$0.13 each = **~$1.04 total**. Ready to generate?"

If doing images and videos separately, confirm each batch separately. Never combine.

## Step 5: Generate Images

**IMPORTANT: Always generate ALL images first. Do NOT proceed to videos until the user has reviewed and approved images in Airtable.**

```python
from tools.airtable import get_pending_images
from tools.image_gen import generate_batch

records = get_pending_images()
# Default: Google AI Studio with Nano Banana Pro
results = generate_batch(records, reference_paths=["references/inputs/product.jpg"])
# Override provider: generate_batch(records, reference_paths=[...], provider="kie")
# Override model: generate_batch(records, reference_paths=[...], model="nano-banana")
```

All generated images go directly to Airtable (Generated Image 1 + Generated Image 2) — NO local downloads.

Tell the user to review images in Airtable and mark as "Approved" or "Rejected".

## Step 6: Write Video Prompts into Airtable (ONLY after image approval)

Only proceed when the user explicitly says to continue with videos.

**Pay close attention to what the user asks for:**
- If they say "use Generated Image 1" or "just Image 1" → set `preferred_image=1` AND `num_variations=1` (1 video per record)
- If they say "use Generated Image 2" or "just Image 2" → set `preferred_image=2` AND `num_variations=1` (1 video per record)
- If they say "use both images" or "2 variations" → `num_variations=2` (default, 2 videos per record)
- If they don't specify, default to `num_variations=2` with `preferred_image=None`

When the user picks a single image, they want **1 video per record**, not 2 videos using the same image.

1. Check which records the user wants to generate videos for
2. Video model defaults: **Veo 3.1** for authentic/natural style (native audio/dialogue), **Kling 3.0** for cinematic look
3. Write video prompts and update Airtable for each record:

```python
from tools.airtable import update_record

# For Veo 3.1 (default) — natural description with dialogue in quotes:
update_record(record_id, {
    "Video Prompt": "A young woman holds up the serum bottle to camera and says \"so I just tried this serum and honestly my skin has never felt this good,\" she gently turns it to show the label while maintaining eye contact. Fixed camera, amateur iPhone selfie video, warm natural daylight, casual excited tone.",
    "Video Model": "Veo 3.1",
    "Video Status": "Pending",
})

# For Kling 3.0 / Sora 2 Pro — structured format:
update_record(record_id, {
    "Video Prompt": "dialogue: so I just tried this serum...\naction: character holds up the serum bottle\ncamera: fixed camera, no music, amateur iPhone selfie video\nemotion: excited, genuine surprise\nvoice_type: casual, friendly, young adult female",
    "Video Model": "Sora 2 Pro",  # or "Kling 3.0" for cinematic
    "Video Status": "Pending",
})
```

Each record MUST have:
- **Video Prompt** — format depends on model:
  - **Veo 3.1**: Natural description with dialogue in quotes (native audio generation)
  - **Kling/Sora**: Structured with `dialogue:`, `action:`, `camera:`, `emotion:`, `voice_type:` fields
- **Video Model** — "Veo 3.1" (UGC), "Sora 2 Pro", or "Kling 3.0" (cinematic)
- **Video Status** set to "Pending"

Tell the user to review the video prompts in Airtable. **STOP and wait** for the user to confirm the prompts look good before proceeding to generation.

## Step 7: Provider Selection & Cost Confirmation & Generate Videos

### Provider Selection (for Kling/Sora only)

For Kling 3.0 or Sora 2 Pro models, ask the user which provider to use:
- **Kie AI** (default) — `generate_batch(pending)` (no override needed)
- **WaveSpeed AI** (backup) — `generate_batch(pending, provider="wavespeed")`

Veo 3.1 always uses Google AI Studio (no WaveSpeed option).

### Cost Confirmation & Generation

**HARD RULE: Show cost and get explicit confirmation before generating.**

```python
from tools.airtable import get_pending_videos
from tools.video_gen import generate_batch

pending = get_pending_videos()
# Default (Kie AI for Kling/Sora, Google for Veo):
results = generate_batch(pending, num_variations=2)
# Use WaveSpeed for Kling/Sora:
results = generate_batch(pending, num_variations=2, provider="wavespeed")
# User said "use Image 1" → 1 video per record:
results = generate_batch(pending, preferred_image=1, num_variations=1)
```

Parameters:
- `preferred_image` (1 or 2): Which Generated Image to use as start frame. Default `None` (first available).
- `num_variations` (1 or 2): How many videos to generate per record. Default 2. When the user picks a single image, set this to 1.
- `provider` (optional): Override provider for Kling/Sora — `"wavespeed"` to use WaveSpeed AI instead of Kie AI.

All generated videos go directly to Airtable (Generated Video 1 + Generated Video 2) — NO local downloads.

## Key Rules

- **Records first, review in Airtable, then generate** — always create Airtable records with prompts + references, then have the user review prompts in Airtable before generating anything
- **Reference images in Airtable** — attach reference images to every record so the user can visually confirm what references are being used
- **Images first, videos second** — never skip ahead to videos
- **All outputs go to Airtable only** — no local downloads
- **Cost confirmation is mandatory** before every generation batch
- **2 variations per record by default** — every ad gets Generated Image 1 + 2, Generated Video 1 + 2. But if the user picks a single image (e.g., "use Image 1"), generate only 1 video per record
- **1 reference image per ad** by default unless user specifies more
- **Resolution defaults to 1K** for speed; only use 2K/4K if user requests it
- **Video model defaults**: Veo 3.1 for UGC (native audio), Kling 3.0 for cinematic
- **Image provider defaults**: Google AI Studio (can override to Kie AI with `provider="kie"`)
- **Video provider for Kling/Sora**: Kie AI default, WaveSpeed AI backup (`provider="wavespeed"`)
- **Index field** must be unique — always call `get_next_index()` before creating records
- **Video Prompt** serves as both the script description and the video prompt
