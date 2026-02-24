"""
WaveSpeed AI provider — image generation (GPT Image 1.5) and
video generation (Kling 3.0, Sora 2 Pro) via WaveSpeed's REST API.

All generation is ASYNCHRONOUS (submit → poll).
WaveSpeed returns a dynamic polling URL in the submit response.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

from .. import config
from ..utils import (
    print_status,
    submit_wavespeed_task,
    poll_wavespeed_task,
)

# Provider sync flags — WaveSpeed is always async
image_IS_SYNC = False
video_IS_SYNC = False

# --- WaveSpeed image model IDs ---
_IMAGE_MODELS = {
    "gpt-image-1.5": "openai/gpt-image-1.5/edit",
}

# --- WaveSpeed video model IDs ---
_VIDEO_MODELS = {
    "kling-3.0": "kwaivgi/kling-v3.0-pro/image-to-video",
    "kling-3.0-std": "kwaivgi/kling-v3.0-std/image-to-video",
    "sora-2": "openai/sora-2/image-to-video",
    "sora-2-pro": "openai/sora-2/image-to-video-pro",
}

# Module-level storage: task_id → poll_url
# Populated by submit_image/submit_video, consumed by poll_image/poll_video/poll_tasks_parallel
_task_poll_urls = {}


# --- Image helpers ---

def _map_image_size(aspect_ratio):
    """Map aspect ratio to GPT Image 1.5 size parameter."""
    return {
        "9:16": "1024*1536",
        "2:3":  "1024*1536",
        "16:9": "1536*1024",
        "3:2":  "1536*1024",
        "1:1":  "1024*1024",
    }.get(aspect_ratio, "auto")


def _map_image_quality(resolution):
    """Map resolution string to GPT Image 1.5 quality parameter."""
    if resolution in ("2K", "4K"):
        return "high"
    return "medium"


def submit_image(prompt, reference_urls=None, aspect_ratio="9:16",
                 resolution="1K", model="gpt-image-1.5", **kwargs):
    """
    Submit an image generation task to WaveSpeed AI (GPT Image 1.5).

    Args:
        prompt: Image prompt text
        reference_urls: List of reference image URLs (product references)
        aspect_ratio: Aspect ratio string (e.g., "9:16")
        resolution: "1K", "2K", or "4K"
        model: "gpt-image-1.5"

    Returns:
        str: task_id for polling
    """
    model_id = _IMAGE_MODELS.get(model)
    if not model_id:
        raise ValueError(f"WaveSpeed doesn't support image model: '{model}'. "
                         f"Available: {list(_IMAGE_MODELS.keys())}")

    payload = {
        "prompt": prompt,
        "size": _map_image_size(aspect_ratio),
        "quality": _map_image_quality(resolution),
        "input_fidelity": "high",
        "output_format": "jpeg",
    }
    if reference_urls:
        payload["images"] = list(reference_urls)

    task_info = submit_wavespeed_task(model_id, payload)
    _task_poll_urls[task_info["task_id"]] = task_info["poll_url"]
    return task_info["task_id"]


def poll_image(task_id, max_wait=300, poll_interval=5, quiet=False):
    """
    Poll a WaveSpeed image task. Returns GenerationResult dict.

    Args:
        task_id: The task ID returned by submit_image
        max_wait: Maximum seconds to wait
        poll_interval: Seconds between checks
        quiet: Suppress status messages

    Returns:
        dict: GenerationResult with status, result_url, task_id
    """
    poll_url = _task_poll_urls.get(task_id)
    if not poll_url:
        raise Exception(f"No poll URL stored for WaveSpeed task {task_id}. "
                        "Was submit_image called in this session?")
    return poll_wavespeed_task(task_id, poll_url, max_wait=max_wait,
                               poll_interval=poll_interval, quiet=quiet)


def submit_video(prompt, image_url=None, model="sora-2-pro",
                 duration="5", aspect_ratio="9:16", mode="pro", **kwargs):
    """
    Submit a video generation task to WaveSpeed AI.

    Args:
        prompt: Video prompt text
        image_url: Source image URL (start frame)
        model: "kling-3.0", "kling-3.0-std", "sora-2", or "sora-2-pro"
        duration: Video duration in seconds
        aspect_ratio: Aspect ratio string
        mode: "std" or "pro" — Kling quality mode (default: "pro")

    Returns:
        str: task_id for polling
    """
    # For Kling, select pro/std model variant based on mode
    if model == "kling-3.0" and mode == "std":
        model_id = _VIDEO_MODELS.get("kling-3.0-std")
    else:
        model_id = _VIDEO_MODELS.get(model)
    if not model_id:
        raise ValueError(f"WaveSpeed doesn't support video model: '{model}'. "
                         f"Available: {list(_VIDEO_MODELS.keys())}")

    if model.startswith("kling"):
        payload = {
            "prompt": prompt,
            "duration": int(duration),
            "cfg_scale": 0.5,
            "sound": True,
        }
        if image_url:
            payload["image"] = image_url

    elif model.startswith("sora"):
        # Map duration: WaveSpeed Sora accepts 4/8/12
        dur_int = int(duration)
        if dur_int <= 5:
            ws_duration = 4
        elif dur_int <= 10:
            ws_duration = 8
        else:
            ws_duration = 12

        payload = {
            "prompt": prompt,
            "duration": ws_duration,
        }
        if model == "sora-2-pro":
            payload["resolution"] = "1080p"
        if image_url:
            payload["image"] = image_url
    else:
        raise ValueError(f"No payload builder for model: {model}")

    task_info = submit_wavespeed_task(model_id, payload)

    # Store poll_url for later retrieval by poll functions
    _task_poll_urls[task_info["task_id"]] = task_info["poll_url"]

    return task_info["task_id"]


def poll_video(task_id, max_wait=600, poll_interval=10, quiet=False):
    """
    Poll a WaveSpeed video task. Returns GenerationResult dict.

    Args:
        task_id: The task ID returned by submit_video
        max_wait: Maximum seconds to wait
        poll_interval: Seconds between checks
        quiet: Suppress status messages

    Returns:
        dict: GenerationResult with status, result_url, task_id
    """
    poll_url = _task_poll_urls.get(task_id)
    if not poll_url:
        raise Exception(f"No poll URL stored for WaveSpeed task {task_id}. "
                        "Was submit_video called in this session?")

    return poll_wavespeed_task(task_id, poll_url, max_wait=max_wait,
                               poll_interval=poll_interval, quiet=quiet)


def poll_tasks_parallel(task_ids, max_wait=600, poll_interval=10):
    """
    Poll multiple WaveSpeed tasks concurrently.

    Args:
        task_ids: List of task ID strings (from submit_video)
        max_wait: Max seconds to wait per task
        poll_interval: Seconds between checks

    Returns:
        dict: task_id → GenerationResult
    """
    if not task_ids:
        return {}

    total = len(task_ids)
    completed = []
    results = {}

    def _poll_one(tid):
        result = poll_video(tid, max_wait=max_wait,
                            poll_interval=poll_interval, quiet=True)
        completed.append(tid)
        print_status(f"Task {tid[:12]}... done ({len(completed)}/{total})", "OK")
        return result

    max_workers = min(total, 20)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_poll_one, tid): tid
            for tid in task_ids
        }
        for future in as_completed(futures):
            tid = futures[future]
            try:
                results[tid] = future.result()
            except Exception as e:
                completed.append(tid)
                print_status(f"Task {tid[:12]}... failed: {e}", "XX")
                results[tid] = {
                    "status": "error",
                    "task_id": tid,
                    "error": str(e),
                }

    return results
