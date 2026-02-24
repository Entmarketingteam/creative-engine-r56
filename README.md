# Creative Content Engine — Community Template

> **For AI agents:** If you're reading this, you are setting up the Creative Content Engine for a new user. Follow the instructions below to walk them through setup step by step. Once setup is complete, refer to `CLAUDE.md` for all operational workflows.

---

## What This Is

The **Creative Content Engine** is an AI-powered pipeline that generates ad images and video content at scale. It uses multiple AI providers (Google AI Studio, Kie AI, WaveSpeed AI) for generation and Airtable as the review hub where users approve or reject outputs.

**The workflow:**
1. User provides a product and reference photos
2. Agent writes creative prompts and generates image variations
3. User reviews and approves/rejects in Airtable
4. Approved images become start frames for video ads
5. User reviews final videos in Airtable

---

## First-Time Setup (Agent Instructions)

When a user opens this project for the first time, walk them through these steps in order. Do not skip any step. Confirm each step is complete before moving to the next.

### Step 1: Install Python Dependencies

Run this from the project root:

```bash
pip install -r .claude/requirements.txt
```

This installs `requests`, `python-dotenv`, and `flask`.

### Step 2: Create the Environment File

The user needs to create `.claude/.env` from the provided template:

```bash
cp .claude/.env.example .claude/.env
```

Then have them fill in their API keys. Walk them through each one:

| Key | Where to Get It | Required? |
|-----|----------------|-----------|
| `GOOGLE_API_KEY` | https://aistudio.google.com/apikey | Yes (default provider for images + Veo 3.1 video) |
| `KIE_API_KEY` | https://kie.ai/api-key | Yes (file hosting + Kling/Sora video + fallback images) |
| `AIRTABLE_API_KEY` | https://airtable.com/create/tokens | Yes (review hub) |
| `AIRTABLE_BASE_ID` | From the Airtable base URL (`appXXXXXX`) | Yes |
| `WAVESPEED_API_KEY` | https://wavespeed.ai (API settings) | Optional (backup video provider) |

**Airtable PAT scopes required:** `data.records:read`, `data.records:write`, `schema.bases:read`, `schema.bases:write`

**Important:** The user needs to create an Airtable base first (any empty base), then grab the Base ID from the URL — it starts with `app`.

### Step 3: Create the Airtable Table

Run the setup script to create the `Content` table automatically:

```bash
python .claude/setup_airtable.py
```

This creates a pre-configured table with all the right fields, statuses, and select options. If the table already exists, it safely skips creation.

### Step 4: Verify Setup

Run a quick credentials check:

```python
import sys; sys.path.insert(0, '.')
from tools.config import check_credentials
missing = check_credentials()
if not missing:
    print("All credentials configured!")
```

If any keys are missing, the function will list them.

### Step 5: Ready to Generate

Setup is complete. The user can now:

1. Place product reference images in `references/inputs/`
2. Ask to generate content (e.g., "Generate 5 ad variations for [product name]")
3. Review outputs in their Airtable base

Refer to `CLAUDE.md` for the full operational workflows, prompt guidelines, cost awareness rules, and provider system documentation.

---

## What's Included

```
.claude/                  - Agent config & setup
  .env.example            - API key template (fill in your keys)
  requirements.txt        - Python dependencies
  setup_airtable.py       - One-time Airtable table creation
  commands/
    generate-content.md   - /generate-content slash command workflow

references/               - Reference materials
  docs/
    prompt-best-practices.md  - Comprehensive prompt writing guide
    setup-guide.md            - DIY build-your-own guide
    kie-ai-api.md             - Kie AI API reference
  inputs/                     - Sample reference images & videos

tools/                    - Python generation pipeline
  config.py               - API keys, endpoints, cost matrix
  airtable.py             - Airtable CRUD operations
  kie_upload.py           - File hosting via Kie.ai
  image_gen.py            - Multi-provider image generation
  video_gen.py            - Multi-provider video generation
  video_analyze.py        - Reference video analysis (Gemini)
  utils.py                - Polling, downloads, status printing
  providers/              - Provider abstraction layer
    google.py             - Google AI Studio (Nano Banana, Veo 3.1)
    kie.py                - Kie AI (Nano Banana Pro, Kling, Sora)
    wavespeed.py          - WaveSpeed AI (Kling, Sora backup)

outputs/                  - Generated assets (local cache)
CLAUDE.md                 - Full agent instructions & workflows
```

---

## Supported Models

| Model | Type | Default Provider | Cost |
|-------|------|-----------------|------|
| Nano Banana | Image | Google AI Studio | ~$0.04 |
| Nano Banana Pro | Image | Google AI Studio | ~$0.13 |
| Veo 3.1 | Video | Google AI Studio | ~$0.50 |
| Kling 3.0 | Video | Kie AI | ~$0.30 |
| Sora 2 Pro | Video | Kie AI | ~$0.30 |

---

## Quick Reference

| Task | Command |
|------|---------|
| Generate image ads | "Generate [N] ad variations for [product]" |
| Create videos | "Create videos for the approved images" |
| Analyze reference video | "Analyze the reference video in references/inputs/" |
| Check pipeline status | "What's pending in Airtable?" |

---

*Built with the Antigravity Creative Engine*
