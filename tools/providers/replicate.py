"""
Replicate provider — image generation (FLUX.1 Schnell, FLUX.1 Dev) and
video generation (LTX Video, Wan 2.1, CogVideoX, Minimax Video) via Replicate's REST API.

All generation is ASYNCHRONOUS (submit → poll).
Replicate returns a prediction URL in the submit response for polling.

Cost per unit (approximate):
  FLUX.1 Schnell:  $0.003/image
  FLUX.1 Dev:      $0.01/image
  LTX Video:       $0.04/video
  Wan 2.1:         $0.08/video
  CogVideoX:       $0.05–0.10/video
  Minimax Video:   $0.10/video
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from .. import config
from ..utils import print_status

# Provider sync flags — Replicate is always async
image_IS_SYNC = False
video_IS_SYNC = False

# Replicate model paths: "<owner>/<model-name>"
# Uses the latest deployed version via the model-level predictions endpoint.
_IMAGE_MODELS = {
    "flux-schnell": "black-forest-labs/flux-schnell",
    "flux-dev":     "black-forest-labs/flux-dev",
}

_VIDEO_MODELS = {
    "ltx-video":      "lightricks/ltx-video",
    "wan-2.1":        "wavespeed-ai/wan2.1-i2v-480p",
    "cogvideox":      "fofr/cogvideox-5b",
    "minimax-video":  "minimax/video-01-live",
}

# Module-level storage: task_id → poll_url
_task_poll_urls = {}

_HEADERS = lambda: {
    "Authorization": f"Token {config.REPLICATE_API_TOKEN}",
    "Content-Type": "application/json",
    "Prefer": "wait",  # for fast models — ignored if unsupported
}


def _submit_prediction(model_path, input_payload):
    """
    Submit a prediction to Replicate via the model-level endpoint.

    Args:
        model_path: "<owner>/<name>" string
        input_payload: dict of model-specific input params

    Returns:
        dict: {"task_id": str, "poll_url": str}
    """
    url = f"{config.REPLICATE_API_URL}/models/{model_path}/predictions"
    # Remove "Prefer: wait" for video (long-running) — images only
    headers = {
        "Authorization": f"Token {config.REPLICATE_API_TOKEN}",
        "Content-Type": "application/json",
    }
    response = requests.post(url, headers=headers, json={"input": input_payload})

    if response.status_code not in (200, 201):
        raise Exception(
            f"Replicate API error {response.status_code}: {response.text}"
        )

    data = response.json()
    task_id = data.get("id")
    poll_url = data.get("urls", {}).get("get")

    if not task_id or not poll_url:
        raise Exception(f"Missing id or poll URL in Replicate response: {data}")

    return {"task_id": task_id, "poll_url": poll_url}


def _poll_prediction(task_id, poll_url, max_wait=300, poll_interval=5, quiet=False):
    """
    Poll a Replicate prediction until succeeded/failed/canceled.

    Returns:
        dict: {"status": "success", "task_id": str, "result_url": str}
    """
    headers = {"Authorization": f"Token {config.REPLICATE_API_TOKEN}"}
    start_time = time.time()
    retry_count = 0

    while time.time() - start_time < max_wait:
        response = requests.get(poll_url, headers=headers)

        if response.status_code != 200:
            retry_count += 1
            if retry_count > 10:
                raise Exception(f"Poll failed after retries: {response.text}")
            elapsed = int(time.time() - start_time)
            if not quiet:
                print_status(
                    f"Status check returned {response.status_code}, retrying... ({elapsed}s)", "!!"
                )
            time.sleep(poll_interval)
            continue

        data = response.json()
        status = data.get("status", "unknown")
        retry_count = 0

        if status == "succeeded":
            output = data.get("output")
            # Output is a list of URLs for images, or a single URL string for video
            if isinstance(output, list) and output:
                result_url = output[0]
            elif isinstance(output, str) and output:
                result_url = output
            else:
                raise Exception(f"No usable output in succeeded prediction: {data}")
            if not quiet:
                print_status("Prediction completed successfully!", "OK")
            return {"status": "success", "task_id": task_id, "result_url": result_url}

        elif status in ("failed", "canceled"):
            error = data.get("error") or status
            raise Exception(f"Replicate prediction {status}: {error}")

        else:
            elapsed = int(time.time() - start_time)
            mins, secs = divmod(elapsed, 60)
            if not quiet:
                print_status(f"Status: {status} ({mins}m {secs}s elapsed)", "..")
            time.sleep(poll_interval)

    raise Exception(f"Timeout waiting for Replicate prediction after {max_wait}s")


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------

def _map_aspect_ratio_flux(aspect_ratio):
    """Map aspect ratio string to FLUX's accepted format."""
    mapping = {
        "9:16": "9:16",
        "16:9": "16:9",
        "1:1":  "1:1",
        "2:3":  "2:3",
        "3:2":  "3:2",
        "4:5":  "4:5",
    }
    return mapping.get(aspect_ratio, "9:16")


def submit_image(prompt, reference_urls=None, aspect_ratio="9:16",
                 resolution="1K", model="flux-schnell", **kwargs):
    """
    Submit an image generation task to Replicate (FLUX models).

    Args:
        prompt: Image prompt text
        reference_urls: Unused — FLUX is text-to-image only
        aspect_ratio: Aspect ratio string (e.g., "9:16")
        resolution: "1K", "2K", or "4K" (maps to output_quality)
        model: "flux-schnell" or "flux-dev"

    Returns:
        str: task_id for polling
    """
    model_path = _IMAGE_MODELS.get(model)
    if not model_path:
        raise ValueError(
            f"Replicate doesn't support image model: '{model}'. "
            f"Available: {list(_IMAGE_MODELS.keys())}"
        )

    output_quality = 90 if resolution in ("2K", "4K") else 80

    input_payload = {
        "prompt": prompt,
        "aspect_ratio": _map_aspect_ratio_flux(aspect_ratio),
        "num_outputs": 1,
        "output_format": "jpg",
        "output_quality": output_quality,
    }

    if model == "flux-dev":
        input_payload["guidance"] = 3.5

    task_info = _submit_prediction(model_path, input_payload)
    _task_poll_urls[task_info["task_id"]] = task_info["poll_url"]
    return task_info["task_id"]


def poll_image(task_id, max_wait=120, poll_interval=3, quiet=False):
    """
    Poll a Replicate image task. Returns GenerationResult dict.

    Args:
        task_id: The task ID returned by submit_image
        max_wait: Maximum seconds to wait (FLUX Schnell is usually <10s)
        poll_interval: Seconds between checks
        quiet: Suppress status messages

    Returns:
        dict: GenerationResult with status, result_url, task_id
    """
    poll_url = _task_poll_urls.get(task_id)
    if not poll_url:
        raise Exception(
            f"No poll URL stored for Replicate task {task_id}. "
            "Was submit_image called in this session?"
        )
    return _poll_prediction(task_id, poll_url, max_wait=max_wait,
                            poll_interval=poll_interval, quiet=quiet)


# ---------------------------------------------------------------------------
# Video generation
# ---------------------------------------------------------------------------

def _build_video_payload(prompt, image_url, model, duration, aspect_ratio):
    """Build model-specific video input payload."""

    if model == "ltx-video":
        payload = {
            "prompt": prompt,
            "num_frames": min(int(duration) * 24, 257),  # LTX: up to 257 frames
        }
        if image_url:
            payload["image"] = image_url
        return payload

    elif model == "wan-2.1":
        payload = {
            "prompt": prompt,
            "num_frames": min(int(duration) * 16, 81),  # Wan: typically 81 frames
        }
        if image_url:
            payload["image"] = image_url
        return payload

    elif model == "cogvideox":
        payload = {
            "prompt": prompt,
            "num_frames": min(int(duration) * 8, 49),
        }
        if image_url:
            payload["image"] = image_url
        return payload

    elif model == "minimax-video":
        payload = {
            "prompt": prompt,
        }
        if image_url:
            payload["first_frame_image"] = image_url
        return payload

    else:
        raise ValueError(f"No payload builder for Replicate video model: {model}")


def submit_video(prompt, image_url=None, model="ltx-video",
                 duration="5", aspect_ratio="9:16", **kwargs):
    """
    Submit a video generation task to Replicate.

    Args:
        prompt: Video prompt text
        image_url: Source image URL (start frame)
        model: "ltx-video", "wan-2.1", "cogvideox", or "minimax-video"
        duration: Video duration in seconds
        aspect_ratio: Aspect ratio string (informational — not all models accept it)

    Returns:
        str: task_id for polling
    """
    model_path = _VIDEO_MODELS.get(model)
    if not model_path:
        raise ValueError(
            f"Replicate doesn't support video model: '{model}'. "
            f"Available: {list(_VIDEO_MODELS.keys())}"
        )

    input_payload = _build_video_payload(prompt, image_url, model, duration, aspect_ratio)
    task_info = _submit_prediction(model_path, input_payload)
    _task_poll_urls[task_info["task_id"]] = task_info["poll_url"]
    return task_info["task_id"]


def poll_video(task_id, max_wait=600, poll_interval=10, quiet=False):
    """
    Poll a Replicate video task. Returns GenerationResult dict.

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
        raise Exception(
            f"No poll URL stored for Replicate task {task_id}. "
            "Was submit_video called in this session?"
        )
    return _poll_prediction(task_id, poll_url, max_wait=max_wait,
                            poll_interval=poll_interval, quiet=quiet)


# ---------------------------------------------------------------------------
# Parallel polling
# ---------------------------------------------------------------------------

def poll_tasks_parallel(task_ids, max_wait=600, poll_interval=5):
    """
    Poll multiple Replicate tasks concurrently.

    Args:
        task_ids: List of task ID strings (from submit_image or submit_video)
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
        poll_url = _task_poll_urls.get(tid)
        if not poll_url:
            raise Exception(f"No poll URL for task {tid}")
        result = _poll_prediction(tid, poll_url, max_wait=max_wait,
                                  poll_interval=poll_interval, quiet=True)
        completed.append(tid)
        print_status(f"Task {tid[:12]}... done ({len(completed)}/{total})", "OK")
        return result

    max_workers = min(total, 20)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_poll_one, tid): tid for tid in task_ids}
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
