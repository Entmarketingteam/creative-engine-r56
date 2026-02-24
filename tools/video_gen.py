"""
Video generation module — multi-provider.
Routes to Google AI Studio (Veo 3.1) or Kie AI (Kling 3.0, Sora 2 Pro)
based on the Video Model field in each Airtable record.

All video generation is asynchronous (submit → poll) regardless of provider.
A single batch may contain records using different models/providers —
the batch logic groups by provider for polling.
"""

import time
from . import config
from .utils import print_status
from .airtable import update_record
from .providers import get_video_provider


def _resolve_model(record_model_field, default_model=None):
    """Map Airtable Video Model field value to internal model name."""
    mapping = {
        "Kling 3.0": "kling-3.0",
        "Sora 2": "sora-2",
        "Sora 2 Pro": "sora-2-pro",
        "Veo 3.1": "veo-3.1",
    }
    default = default_model or config.DEFAULT_VIDEO_MODEL
    return mapping.get(record_model_field, default)


def _get_image_url(fields, preferred_image=None):
    """Get generated image URL from a record.

    Args:
        fields: Record fields dict
        preferred_image: 1 or 2 to select a specific image, None for first available
    """
    if preferred_image:
        img_field = f"Generated Image {preferred_image}"
        generated_image = fields.get(img_field, [])
        if generated_image:
            return generated_image[0].get("url")
        return None

    for img_field in ("Generated Image 1", "Generated Image 2"):
        generated_image = fields.get(img_field, [])
        if generated_image:
            return generated_image[0].get("url")
    return None


def generate_ugc_video(prompt, image_url=None, model=None, duration="5",
                       mode="pro", aspect_ratio="9:16", provider=None):
    """
    Generate a single video (submit + poll).

    Args:
        prompt: Text prompt describing the desired video motion
        image_url: Source image URL (start frame)
        model: Video model name (default: config.DEFAULT_VIDEO_MODEL)
        duration: Video duration in seconds
        mode: "std" or "pro" (Kling 3.0 only)
        aspect_ratio: Aspect ratio string
        provider: Provider override

    Returns:
        dict with 'status', 'task_id', and 'result_url'
    """
    model = model or config.DEFAULT_VIDEO_MODEL
    provider_module, provider_name = get_video_provider(model, provider)

    print_status(f"Generating video via {provider_name} ({model})...")
    print_status(f"Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
    if image_url:
        print_status(f"Source image: {image_url[:60]}...")

    task_id = provider_module.submit_video(
        prompt, image_url=image_url, model=model,
        duration=duration, mode=mode, aspect_ratio=aspect_ratio
    )
    print_status(f"Task created: {task_id}", "OK")

    # Choose appropriate timeout
    max_wait = 900 if model in ("sora-2", "sora-2-pro", "veo-3.1") else 600
    return provider_module.poll_video(task_id, max_wait=max_wait, poll_interval=10)


def generate_for_record(record, model=None, duration="5", preferred_image=None,
                        num_variations=2, provider=None, aspect_ratio="9:16",
                        mode="pro"):
    """
    Generate video variations for a single Airtable record.

    Args:
        record: Airtable record dict (with 'id' and 'fields')
        model: Default video model (overridden by record's Video Model field)
        duration: Default video duration in seconds
        preferred_image: 1 or 2 to select a specific generated image as start frame
        num_variations: Number of video variations to generate (1 or 2, default 2)
        provider: Provider override
        aspect_ratio: Aspect ratio string (default: "9:16")
        mode: "std" or "pro" — Kling 3.0 quality mode (default: "pro")

    Returns:
        list of result dicts, or None if skipped
    """
    record_id = record["id"]
    fields = record.get("fields", {})
    ad_name = fields.get("Ad Name", "untitled")

    image_url = _get_image_url(fields, preferred_image=preferred_image)
    if not image_url:
        print_status(f"Skipping '{ad_name}' - no Generated Image", "!!")
        return None

    video_prompt = fields.get("Video Prompt", "")
    if not video_prompt:
        print_status(f"Skipping '{ad_name}' - no Video Prompt set", "!!")
        return None

    record_model = _resolve_model(fields.get("Video Model", ""), model)
    provider_module, provider_name = get_video_provider(record_model, provider)

    num_variations = max(1, min(2, num_variations))
    var_range = range(1, num_variations + 1)

    print(f"\n--- Generating {num_variations} video variation(s) for: {ad_name} ({provider_name}/{record_model}) ---")
    print_status(f"Prompt: {video_prompt[:100]}{'...' if len(video_prompt) > 100 else ''}")

    # Submit variations with 1s gap
    task_ids = []
    for var_num in var_range:
        print_status(f"Submitting variation {var_num}/{num_variations}...")
        task_id = provider_module.submit_video(
            video_prompt, image_url=image_url, model=record_model,
            duration=duration, aspect_ratio=aspect_ratio, mode=mode
        )
        task_ids.append(task_id)
        print_status(f"Task {task_id}", "OK")
        if var_num < num_variations:
            time.sleep(1)

    # Poll concurrently
    print_status("Polling for results...")
    max_wait = 900 if record_model in ("sora-2", "sora-2-pro", "veo-3.1") else 600
    results_map = provider_module.poll_tasks_parallel(task_ids, max_wait=max_wait, poll_interval=10)

    # Build Airtable update
    update_fields = {"Video Status": "Generated"}
    results = []
    for var_num, task_id in enumerate(task_ids, 1):
        result = results_map[task_id]
        if result.get("status") == "error":
            raise Exception(f"Variation {var_num} failed: {result.get('error')}")
        results.append(result)
        update_fields[f"Generated Video {var_num}"] = [{"url": result["result_url"]}]

    update_record(record_id, update_fields)
    print_status(f"Airtable updated for '{ad_name}' ({num_variations} variation(s))", "OK")

    return results


def generate_batch(records, model=None, duration="5", preferred_image=None,
                   num_variations=2, provider=None, aspect_ratio="9:16",
                   mode="pro"):
    """
    Generate videos for multiple Airtable records in parallel.
    Supports mixed-provider batches (e.g., some records use Veo, others use Sora).

    Phase 1: Submit all tasks, tagging each with its provider.
    Phase 2: Group by provider, poll each group with its provider's poll_tasks_parallel.
    Phase 3: Update Airtable per record.

    Args:
        records: List of Airtable record dicts
        model: Default video model (overridden per-record by Video Model field)
        duration: Default video duration in seconds
        preferred_image: 1 or 2 to select a specific generated image
        num_variations: Videos per record (1 or 2, default 2)
        provider: Provider override (applies to all records)
        aspect_ratio: Aspect ratio string (default: "9:16")
        mode: "std" or "pro" — Kling 3.0 quality mode (default: "pro")

    Returns:
        list of results (None for skipped/failed records)
    """
    default_model = model or config.DEFAULT_VIDEO_MODEL

    actionable = [
        r for r in records
        if (r.get("fields", {}).get("Generated Image 1") or r.get("fields", {}).get("Generated Image 2"))
        and r.get("fields", {}).get("Video Prompt")
    ]
    count = len(actionable)

    if count == 0:
        print_status("No records ready for video generation", "!!")
        return []

    num_variations = max(1, min(2, num_variations))
    var_range = range(1, num_variations + 1)
    videos_total = count * num_variations

    # Build cost summary — may have mixed models
    model_counts = {}
    for record in actionable:
        rm = _resolve_model(record.get("fields", {}).get("Video Model", ""), default_model)
        model_counts[rm] = model_counts.get(rm, 0) + 1

    print(f"\n{'=' * 50}")
    print(f"  Video Generation Batch")
    print(f"{'=' * 50}")
    print(f"  Records: {count}")
    print(f"  Videos per record: {num_variations}")
    print(f"  Total videos: {videos_total}")

    total_cost = 0.0
    for m, cnt in model_counts.items():
        _, pname = get_video_provider(m, provider)
        unit = config.get_cost(m, pname)
        subtotal = cnt * num_variations * unit
        total_cost += subtotal
        print(f"  - {m} ({pname}): {cnt} records x {num_variations} = {cnt * num_variations} videos @ ${unit:.2f} = ${subtotal:.2f}")

    print(f"  Total estimated cost: ~${total_cost:.2f}")
    print(f"{'=' * 50}\n")

    # --- Phase 1: Submit all tasks ---
    print(f"\n--- Phase 1: Submitting {videos_total} tasks ---")
    # submissions: list of (record, var_num, task_id, provider_module)
    submissions = []

    for record in actionable:
        fields = record.get("fields", {})
        ad_name = fields.get("Ad Name", "untitled")
        video_prompt = fields.get("Video Prompt", "")
        image_url = _get_image_url(fields, preferred_image=preferred_image)
        record_model = _resolve_model(fields.get("Video Model", ""), default_model)
        rec_provider_module, rec_provider_name = get_video_provider(record_model, provider)

        for var_num in var_range:
            print_status(f"Submitting: {ad_name} (variation {var_num}) [{record_model}/{rec_provider_name}]")
            try:
                task_id = rec_provider_module.submit_video(
                    video_prompt, image_url=image_url, model=record_model,
                    duration=duration, aspect_ratio=aspect_ratio, mode=mode
                )
                submissions.append((record, var_num, task_id, rec_provider_module))
                print_status(f"Task {task_id}", "OK")
            except Exception as e:
                print_status(f"Submit failed: {e}", "XX")
                submissions.append((record, var_num, None, rec_provider_module))
            time.sleep(1)

    # --- Phase 2: Poll — grouped by provider ---
    print(f"\n--- Phase 2: Polling tasks ---")
    # Group task_ids by provider module
    provider_tasks = {}
    for _, _, task_id, pmod in submissions:
        if task_id is not None:
            if pmod not in provider_tasks:
                provider_tasks[pmod] = []
            provider_tasks[pmod].append(task_id)

    results_map = {}
    for pmod, task_ids in provider_tasks.items():
        prov_name = pmod.__name__.split(".")[-1]
        print_status(f"Polling {len(task_ids)} tasks via {prov_name}...")
        max_wait = 900  # generous timeout for all video providers
        group_results = pmod.poll_tasks_parallel(task_ids, max_wait=max_wait, poll_interval=10)
        results_map.update(group_results)

    # --- Phase 3: Update Airtable per record ---
    print(f"\n--- Phase 3: Updating Airtable ---")
    record_tasks = {}  # record_id -> [(var_num, task_id)]
    for record, var_num, task_id, _ in submissions:
        rid = record["id"]
        if rid not in record_tasks:
            record_tasks[rid] = []
        record_tasks[rid].append((var_num, task_id))

    record_map = {r["id"]: r for r in actionable}
    results = []
    succeeded = 0
    videos_generated = 0

    for rid, tasks in record_tasks.items():
        record = record_map[rid]
        ad_name = record.get("fields", {}).get("Ad Name", "untitled")
        update_fields = {}
        record_ok = True

        for var_num, task_id in tasks:
            if task_id is None:
                record_ok = False
                continue
            result = results_map.get(task_id, {})
            if result.get("status") == "error":
                print_status(f"'{ad_name}' variation {var_num} failed: {result.get('error')}", "XX")
                record_ok = False
            else:
                update_fields[f"Generated Video {var_num}"] = [{"url": result["result_url"]}]
                videos_generated += 1

        if update_fields:
            update_fields["Video Status"] = "Generated"
            update_record(rid, update_fields)
            print_status(f"Airtable updated for '{ad_name}'", "OK")

        if record_ok:
            succeeded += 1
        results.append(update_fields if update_fields else None)

    print(f"\n{'=' * 50}")
    print(f"  Batch complete: {succeeded}/{count} records ({videos_generated} videos)")
    print(f"  Actual cost: ~${total_cost:.2f}")
    print(f"{'=' * 50}\n")

    return results
