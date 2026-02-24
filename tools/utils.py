"""
Shared utilities for Creative Content Engine.
Task submission, parallel polling, file downloads, and status printing.
"""

import time
import json
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from . import config


def print_status(message, symbol="->"):
    """Print a formatted status message."""
    print(f"  {symbol} {message}")


def submit_kie_task(payload):
    """
    Submit a task to Kie AI and return the taskId without polling.

    Args:
        payload: Complete request payload (model + input)

    Returns:
        str: The taskId for polling later
    """
    headers = {
        "Authorization": f"Bearer {config.KIE_API_KEY}",
        "Content-Type": "application/json",
    }

    response = requests.post(config.KIE_CREATE_URL, headers=headers, json=payload)

    if response.status_code != 200:
        raise Exception(f"Kie.ai API error: {response.status_code} - {response.text}")

    result = response.json()
    if result.get("code") != 200:
        raise Exception(f"Kie.ai error: {result.get('msg')} (code: {result.get('code')})")

    task_id = result.get("data", {}).get("taskId")
    if not task_id:
        raise Exception(f"No taskId in response: {result}")

    return task_id


def poll_kie_task(task_id, max_wait=300, poll_interval=5, quiet=False):
    """
    Poll a Kie AI task until completion.

    Args:
        task_id: The Kie AI task ID
        max_wait: Maximum seconds to wait
        poll_interval: Seconds between status checks
        quiet: If True, suppress per-poll status messages (used in parallel polling)

    Returns:
        dict with 'status', 'task_id', and 'result_url'

    Raises:
        Exception on failure or timeout
    """
    headers = {"Authorization": f"Bearer {config.KIE_API_KEY}"}
    start_time = time.time()
    retry_count = 0

    while time.time() - start_time < max_wait:
        response = requests.get(
            f"{config.KIE_STATUS_URL}?taskId={task_id}",
            headers=headers,
        )

        if response.status_code != 200:
            retry_count += 1
            if retry_count > 10:
                raise Exception(f"Status check failed after retries: {response.text}")
            elapsed = int(time.time() - start_time)
            if not quiet:
                print_status(f"Status check returned {response.status_code}, retrying... ({elapsed}s)", "!!")
            time.sleep(poll_interval)
            continue

        result = response.json()

        if result.get("code") != 200:
            retry_count += 1
            elapsed = int(time.time() - start_time)
            if retry_count <= 20:
                if not quiet:
                    print_status(f"API returned code {result.get('code')}, retrying... ({elapsed}s)", "!!")
                time.sleep(poll_interval)
                continue
            else:
                raise Exception(f"Status API error: {result.get('msg')} (code: {result.get('code')})")

        retry_count = 0
        data = result.get("data", {})
        state = data.get("state", "unknown")

        if state == "success":
            result_json_str = data.get("resultJson", "{}")
            result_json = json.loads(result_json_str)
            result_urls = result_json.get("resultUrls", [])
            if result_urls:
                if not quiet:
                    print_status("Task completed successfully!", "OK")
                return {
                    "status": "success",
                    "task_id": task_id,
                    "result_url": result_urls[0],
                }
            else:
                raise Exception("No result URLs in completed task")

        elif state == "fail":
            fail_msg = data.get("failMsg", "Unknown error")
            raise Exception(f"Task failed: {fail_msg}")

        else:
            elapsed = int(time.time() - start_time)
            mins, secs = divmod(elapsed, 60)
            if not quiet:
                print_status(f"Status: {state} ({mins}m {secs}s elapsed)", "..")
            time.sleep(poll_interval)

    raise Exception(f"Timeout waiting for task after {max_wait}s")


def poll_kie_tasks_parallel(task_ids, max_wait=300, poll_interval=5):
    """
    Poll multiple Kie AI tasks concurrently using threads.

    Args:
        task_ids: List of task ID strings
        max_wait: Maximum seconds to wait per task
        poll_interval: Seconds between status checks per task

    Returns:
        dict mapping task_id -> result dict (with 'status', 'task_id', 'result_url')
        Failed tasks have 'status': 'error' and 'error' key.
    """
    if not task_ids:
        return {}

    total = len(task_ids)
    completed = []
    results = {}

    def _poll_one(task_id):
        result = poll_kie_task(task_id, max_wait=max_wait, poll_interval=poll_interval, quiet=True)
        completed.append(task_id)
        print_status(f"Task {task_id[:12]}... done ({len(completed)}/{total})", "OK")
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
                results[tid] = {"status": "error", "task_id": tid, "error": str(e)}

    return results


def submit_wavespeed_task(model_id, payload):
    """
    Submit a task to WaveSpeed AI and return the task info.

    Args:
        model_id: Full WaveSpeed model path (e.g., "kwaivgi/kling-v3.0-pro/image-to-video")
        payload: Request body (prompt, image, duration, etc.)

    Returns:
        dict: {"task_id": str, "poll_url": str}
    """
    headers = {
        "Authorization": f"Bearer {config.WAVESPEED_API_KEY}",
        "Content-Type": "application/json",
    }

    url = f"{config.WAVESPEED_API_URL}/{model_id}"
    response = requests.post(url, headers=headers, json=payload)

    if response.status_code != 200:
        raise Exception(f"WaveSpeed API error: {response.status_code} - {response.text}")

    result = response.json()
    # WaveSpeed may wrap response in a "data" key or return flat
    data = result.get("data", result)
    task_id = data.get("id")
    poll_url = data.get("urls", {}).get("get")

    if not task_id or not poll_url:
        raise Exception(f"Missing id or poll URL in WaveSpeed response: {result}")

    return {"task_id": task_id, "poll_url": poll_url}


def poll_wavespeed_task(task_id, poll_url, max_wait=600, poll_interval=10, quiet=False):
    """
    Poll a WaveSpeed AI task until completion.

    Args:
        task_id: The WaveSpeed task ID (for logging)
        poll_url: The dynamic polling URL from submit response
        max_wait: Maximum seconds to wait
        poll_interval: Seconds between status checks
        quiet: If True, suppress per-poll status messages

    Returns:
        dict with 'status', 'task_id', and 'result_url'

    Raises:
        Exception on failure or timeout
    """
    headers = {"Authorization": f"Bearer {config.WAVESPEED_API_KEY}"}
    start_time = time.time()
    retry_count = 0

    while time.time() - start_time < max_wait:
        response = requests.get(poll_url, headers=headers)

        if response.status_code != 200:
            retry_count += 1
            if retry_count > 10:
                raise Exception(f"Status check failed after retries: {response.text}")
            elapsed = int(time.time() - start_time)
            if not quiet:
                print_status(f"Status check returned {response.status_code}, retrying... ({elapsed}s)", "!!")
            time.sleep(poll_interval)
            continue

        result = response.json()
        # WaveSpeed may wrap response in a "data" key or return flat
        data = result.get("data", result)
        status = data.get("status", "unknown")
        retry_count = 0

        if status == "completed":
            outputs = data.get("outputs", [])
            if outputs:
                if not quiet:
                    print_status("Task completed successfully!", "OK")
                return {
                    "status": "success",
                    "task_id": task_id,
                    "result_url": outputs[0],
                }
            else:
                raise Exception("No outputs in completed WaveSpeed task")

        elif status == "failed":
            error_msg = data.get("error", "Unknown error")
            raise Exception(f"WaveSpeed task failed: {error_msg}")

        else:
            elapsed = int(time.time() - start_time)
            mins, secs = divmod(elapsed, 60)
            if not quiet:
                print_status(f"Status: {status} ({mins}m {secs}s elapsed)", "..")
            time.sleep(poll_interval)

    raise Exception(f"Timeout waiting for WaveSpeed task after {max_wait}s")


def poll_wavespeed_tasks_parallel(tasks, max_wait=600, poll_interval=10):
    """
    Poll multiple WaveSpeed AI tasks concurrently using threads.

    Args:
        tasks: List of dicts with 'task_id' and 'poll_url'
        max_wait: Maximum seconds to wait per task
        poll_interval: Seconds between status checks per task

    Returns:
        dict mapping task_id -> result dict (with 'status', 'task_id', 'result_url')
        Failed tasks have 'status': 'error' and 'error' key.
    """
    if not tasks:
        return {}

    total = len(tasks)
    completed = []
    results = {}

    def _poll_one(task_info):
        result = poll_wavespeed_task(
            task_info["task_id"], task_info["poll_url"],
            max_wait=max_wait, poll_interval=poll_interval, quiet=True,
        )
        completed.append(task_info["task_id"])
        print_status(f"Task {task_info['task_id'][:12]}... done ({len(completed)}/{total})", "OK")
        return result

    max_workers = min(total, 20)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_poll_one, t): t["task_id"]
            for t in tasks
        }
        for future in as_completed(futures):
            tid = futures[future]
            try:
                results[tid] = future.result()
            except Exception as e:
                completed.append(tid)
                print_status(f"Task {tid[:12]}... failed: {e}", "XX")
                results[tid] = {"status": "error", "task_id": tid, "error": str(e)}

    return results


def download_file(url, output_path):
    """
    Download a file from URL to local path.

    Args:
        url: Source URL
        output_path: Local file path to save to

    Returns:
        Path object of the saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print_status(f"Downloading to: {output_path.name}")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print_status(f"Download complete: {output_path}", "OK")
    return output_path
