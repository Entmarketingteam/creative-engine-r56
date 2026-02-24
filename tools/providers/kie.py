"""
Kie AI provider — image generation (Nano Banana Pro) and video generation
(Kling 3.0, Sora 2 Pro) via Kie AI's createTask/recordInfo API.

All generation is ASYNCHRONOUS (submit → poll).
"""

from .. import config
from ..utils import submit_kie_task, poll_kie_task, poll_kie_tasks_parallel

# Provider sync flags — Kie AI is always async
image_IS_SYNC = False
video_IS_SYNC = False

# --- Kie AI model IDs ---
_IMAGE_MODELS = {
    "nano-banana": "nano-banana-pro",       # Kie only has Pro variant
    "nano-banana-pro": "nano-banana-pro",
}

_VIDEO_MODELS = {
    "kling-3.0": "kling-3.0/video",
    "sora-2-pro": "sora-2-pro-image-to-video",
}


def submit_image(prompt, reference_urls=None, aspect_ratio="9:16",
                 resolution="1K", model="nano-banana-pro", **kwargs):
    """
    Submit an image generation task to Kie AI.

    Args:
        prompt: Image generation prompt
        reference_urls: List of hosted reference image URLs (uploaded to Kie.ai)
        aspect_ratio: Aspect ratio string
        resolution: "1K", "2K", or "4K"
        model: Internal model name

    Returns:
        str: task_id for polling
    """
    model_id = _IMAGE_MODELS.get(model, "nano-banana-pro")
    payload = {
        "model": model_id,
        "input": {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "output_format": "png",
            "image_input": reference_urls if reference_urls else [],
        },
    }
    return submit_kie_task(payload)


def poll_image(task_id, max_wait=300, poll_interval=5, quiet=False):
    """Poll a Kie AI image task. Returns GenerationResult dict."""
    return poll_kie_task(task_id, max_wait=max_wait,
                         poll_interval=poll_interval, quiet=quiet)


def submit_video(prompt, image_url=None, model="sora-2-pro",
                 duration="5", mode="pro", aspect_ratio="9:16", **kwargs):
    """
    Submit a video generation task to Kie AI.

    Args:
        prompt: Video prompt text
        image_url: Source image URL (start frame)
        model: "kling-3.0" or "sora-2-pro"
        duration: Video duration in seconds
        mode: "std" or "pro" (Kling 3.0 only)
        aspect_ratio: Aspect ratio string

    Returns:
        str: task_id for polling
    """
    model_id = _VIDEO_MODELS.get(model)
    if not model_id:
        raise ValueError(f"Kie AI doesn't support video model: '{model}'. "
                         f"Available: {list(_VIDEO_MODELS.keys())}")

    if model == "kling-3.0":
        payload = {
            "model": model_id,
            "input": {
                "mode": mode,
                "prompt": prompt,
                "duration": str(duration),
                "multi_shots": False,
                "sound": True,
            },
        }
        if image_url:
            payload["input"]["image_urls"] = [image_url]
        else:
            payload["input"]["aspect_ratio"] = aspect_ratio

    elif model == "sora-2-pro":
        sora_ratio = aspect_ratio
        if aspect_ratio in ("9:16", "portrait"):
            sora_ratio = "portrait"
        elif aspect_ratio in ("16:9", "1:1", "landscape"):
            sora_ratio = "landscape"

        n_frames = "10"
        if str(duration) in ("15", "12", "13", "14"):
            n_frames = "15"

        payload = {
            "model": model_id,
            "input": {
                "prompt": prompt,
                "aspect_ratio": sora_ratio,
                "n_frames": n_frames,
                "size": "high",
                "remove_watermark": True,
                "upload_method": "s3",
            },
        }
        if image_url:
            payload["input"]["image_urls"] = [image_url]
    else:
        raise ValueError(f"No payload builder for model: {model}")

    return submit_kie_task(payload)


def poll_video(task_id, max_wait=600, poll_interval=10, quiet=False):
    """Poll a Kie AI video task. Returns GenerationResult dict."""
    return poll_kie_task(task_id, max_wait=max_wait,
                         poll_interval=poll_interval, quiet=quiet)


def poll_tasks_parallel(task_ids, max_wait=300, poll_interval=5):
    """Poll multiple Kie AI tasks concurrently. Returns dict of task_id → result."""
    return poll_kie_tasks_parallel(task_ids, max_wait=max_wait,
                                   poll_interval=poll_interval)
