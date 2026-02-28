# Replicate Provider

Added `tools/providers/replicate.py` as a third generation provider alongside Google and WaveSpeed.

## Models

### Image
| Model | Replicate Path | Cost |
|-------|---------------|------|
| `flux-schnell` | `black-forest-labs/flux-schnell` | $0.003/image |
| `flux-dev` | `black-forest-labs/flux-dev` | $0.01/image |

### Video
| Model | Replicate Path | Cost |
|-------|---------------|------|
| `ltx-video` | `lightricks/ltx-video` | $0.04/video |
| `wan-2.1` | `wavespeed-ai/wan2.1-i2v-480p` | $0.08/video |
| `cogvideox` | `fofr/cogvideox-5b` | $0.07/video |
| `minimax-video` | `minimax/video-01-live` | $0.10/video |

## Cost Impact (200-ad campaign)

| Asset | Previous | With Replicate | Savings |
|-------|----------|----------------|---------|
| 200 images (Nano Banana Pro/Google) | $26.00 | $0.60 (FLUX Schnell) | 98% |
| 200 videos (Kling/WaveSpeed) | $60.00 | $8.00 (LTX Video) | 87% |
| **Total** | **$86.00** | **$8.60** | **90%** |

## Usage

```python
from tools.providers import get_image_provider, get_video_provider

# FLUX Schnell (default for flux-schnell model)
provider, name = get_image_provider("flux-schnell")
task_id = provider.submit_image(prompt, aspect_ratio="9:16")
result = provider.poll_image(task_id)

# LTX Video (default for ltx-video model)
provider, name = get_video_provider("ltx-video")
task_id = provider.submit_video(prompt, image_url="https://...", duration="5")
result = provider.poll_video(task_id)

# Override to Replicate explicitly
from tools.providers import get_image_provider
provider, name = get_image_provider("flux-dev", provider_override="replicate")
```

## Setup

Add to `.claude/.env`:
```
REPLICATE_API_TOKEN=r8_xxxx
```

Token lives in Doppler: `doppler secrets get REPLICATE_API_TOKEN --project ent-agency-automation --config dev --plain`

## Notes

- FLUX models are text-to-image only — reference images (`reference_urls`) are not used
- All Replicate generation is async (submit → poll), consistent with WaveSpeed pattern
- FLUX Schnell typically completes in under 10 seconds
- LTX Video and Wan 2.1 are image-to-video — pass `image_url` for the start frame
- Model paths in `_IMAGE_MODELS` / `_VIDEO_MODELS` dicts can be updated if Replicate changes slugs
