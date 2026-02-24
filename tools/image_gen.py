"""
Image generation module — multi-provider.
Routes to Google AI Studio (default) or Kie AI based on model/provider selection.

Supports:
- Nano Banana (Google: gemini-2.5-flash-image, Kie: nano-banana-pro)
- Nano Banana Pro (Google: gemini-3-pro-image-preview, Kie: nano-banana-pro)

Google is synchronous (no polling). Kie AI is async (submit → poll).
"""

import time
from . import config
from .utils import print_status
from .kie_upload import upload_references
from .airtable import update_record
from .providers import get_image_provider, is_sync


VALID_RATIOS = ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]

_MODEL_DISPLAY_NAMES = {
    "nano-banana": "Nano Banana",
    "nano-banana-pro": "Nano Banana Pro",
    "gpt-image-1.5": "GPT Image 1.5",
}

# Reverse mapping: Airtable display name -> internal model name
_AIRTABLE_TO_INTERNAL = {v: k for k, v in _MODEL_DISPLAY_NAMES.items()}


def _detect_aspect_ratio(prompt):
    """Detect aspect ratio from prompt prefix (e.g., '9:16. ...')."""
    for ratio in VALID_RATIOS:
        if prompt.startswith(f"{ratio}.") or prompt.startswith(f"{ratio} "):
            return ratio
    return "9:16"


def generate_ugc_image(prompt, reference_paths=None, reference_urls=None,
                       aspect_ratio="9:16", resolution="1K",
                       model=None, provider=None):
    """
    Generate a single UGC image (submit + poll if async, or direct if sync).

    Args:
        prompt: Image generation prompt
        reference_paths: Local file paths (used by Google provider)
        reference_urls: Hosted URLs (used by Kie AI provider)
        aspect_ratio: Aspect ratio string
        resolution: "1K", "2K", or "4K"
        model: Image model name (default: config.DEFAULT_IMAGE_MODEL)
        provider: Provider override (default: model's default)

    Returns:
        dict with 'status', 'task_id', and 'result_url'
    """
    model = model or config.DEFAULT_IMAGE_MODEL
    provider_module, provider_name = get_image_provider(model, provider)
    sync = is_sync(provider_module, "image")

    print_status(f"Generating image via {provider_name} ({model})...")
    if reference_paths:
        print_status(f"Using {len(reference_paths)} reference image(s)")
    elif reference_urls:
        print_status(f"Using {len(reference_urls)} reference image(s)")

    if sync:
        # Google: synchronous, pass local file paths
        return provider_module.submit_image(
            prompt, reference_paths=reference_paths,
            aspect_ratio=aspect_ratio, resolution=resolution, model=model
        )
    else:
        # Kie AI: async, pass hosted URLs
        task_id = provider_module.submit_image(
            prompt, reference_urls=reference_urls,
            aspect_ratio=aspect_ratio, resolution=resolution, model=model
        )
        print_status(f"Task created: {task_id}", "OK")
        return provider_module.poll_image(task_id, max_wait=300, poll_interval=5)


def generate_for_record(record, reference_paths=None, reference_urls=None,
                        model=None, provider=None, aspect_ratio=None,
                        resolution="1K", num_variations=2):
    """
    Generate image variations for a single Airtable record.

    Args:
        record: Airtable record dict (with 'id' and 'fields')
        reference_paths: Local file paths (for Google)
        reference_urls: Pre-uploaded Kie.ai hosted URLs (for Kie AI)
        model: Image model name
        provider: Provider override
        aspect_ratio: Override aspect ratio (default: auto-detect from prompt, fallback "9:16")
        resolution: Image resolution — "1K", "2K", or "4K" (default: "1K")
        num_variations: Number of image variations to generate, 1 or 2 (default: 2)

    Returns:
        list of result dicts, or None if skipped
    """
    model = model or config.DEFAULT_IMAGE_MODEL
    provider_module, provider_name = get_image_provider(model, provider)
    sync = is_sync(provider_module, "image")

    record_id = record["id"]
    fields = record.get("fields", {})
    ad_name = fields.get("Ad Name", "untitled")
    prompt = fields.get("Image Prompt", "")

    if not prompt:
        print_status(f"Skipping '{ad_name}' - no Image Prompt set", "!!")
        return None

    effective_ratio = aspect_ratio or _detect_aspect_ratio(prompt)
    num_variations = max(1, min(2, num_variations))
    var_range = range(1, num_variations + 1)

    print(f"\n--- Generating {num_variations} image variation(s) for: {ad_name} ({provider_name}) ---")
    print_status(f"Aspect ratio: {effective_ratio}")
    print_status(f"Resolution: {resolution}")
    print_status(f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")

    results = []
    if sync:
        # Google: each call returns result directly
        for var_num in var_range:
            print_status(f"Generating variation {var_num}/{num_variations}...")
            result = provider_module.submit_image(
                prompt, reference_paths=reference_paths,
                aspect_ratio=effective_ratio, resolution=resolution, model=model
            )
            results.append(result)
            print_status(f"Done -> {result['result_url'][:50]}...", "OK")
            if var_num < num_variations:
                time.sleep(1)
    else:
        # Kie AI: submit all, then poll concurrently
        task_ids = []
        for var_num in var_range:
            print_status(f"Submitting variation {var_num}/{num_variations}...")
            task_id = provider_module.submit_image(
                prompt, reference_urls=reference_urls,
                aspect_ratio=effective_ratio, resolution=resolution, model=model
            )
            task_ids.append(task_id)
            print_status(f"Task {task_id}", "OK")
            if var_num < num_variations:
                time.sleep(1)

        print_status("Polling for results...")
        results_map = provider_module.poll_tasks_parallel(task_ids, max_wait=300, poll_interval=5)
        for task_id in task_ids:
            result = results_map[task_id]
            if result.get("status") == "error":
                raise Exception(f"Generation failed: {result.get('error')}")
            results.append(result)

    # Update Airtable
    update_fields = {
        "Image Status": "Generated",
        "Image Model": _MODEL_DISPLAY_NAMES.get(model, model),
    }
    for var_num, result in enumerate(results, 1):
        update_fields[f"Generated Image {var_num}"] = [{"url": result["result_url"]}]

    update_record(record_id, update_fields)
    print_status(f"Airtable updated for '{ad_name}' ({num_variations} variation(s))", "OK")

    return results


def _resolve_record_model(record, fallback_model=None, fallback_provider=None):
    """
    Resolve the model and provider for a single record from its Airtable fields.

    Reads the record's 'Image Model' field (Airtable display name like "Nano Banana")
    and maps it to the internal model name. Falls back to the provided model or default.

    Returns:
        tuple: (internal_model, provider_module, provider_name)
    """
    fields = record.get("fields", {})
    airtable_model = fields.get("Image Model")

    if airtable_model:
        internal = _AIRTABLE_TO_INTERNAL.get(airtable_model)
        if internal:
            provider_module, provider_name = get_image_provider(internal, fallback_provider)
            return internal, provider_module, provider_name

    model = fallback_model or config.DEFAULT_IMAGE_MODEL
    provider_module, provider_name = get_image_provider(model, fallback_provider)
    return model, provider_module, provider_name


def generate_batch(records, reference_paths=None, model=None, provider=None,
                   aspect_ratio=None, resolution="1K", num_variations=2):
    """
    Generate images for multiple Airtable records.

    Respects each record's 'Image Model' field from Airtable. If a record has no
    Image Model set, falls back to the `model` argument or config.DEFAULT_IMAGE_MODEL.

    Routes to the appropriate provider based on per-record model selection.
    - SYNC providers (Google): submit returns result directly, no Phase 2 polling.
    - ASYNC providers (Kie AI): submit returns task_id, Phase 2 polls all concurrently.

    Args:
        records: List of Airtable record dicts
        reference_paths: Local file paths for reference images
        model: Fallback image model name (default: config.DEFAULT_IMAGE_MODEL)
        provider: Provider override applied to all records (default: model's default)
        aspect_ratio: Override aspect ratio for all records (default: auto-detect from prompt)
        resolution: Image resolution — "1K", "2K", or "4K" (default: "1K")
        num_variations: Images per record, 1 or 2 (default: 2)

    Returns:
        list of results (None for skipped/failed records)
    """
    actionable = [r for r in records if r.get("fields", {}).get("Image Prompt")]
    count = len(actionable)

    if count == 0:
        print_status("No records with Image Prompt found - nothing to generate", "!!")
        return []

    # --- Resolve per-record models and build cost summary ---
    record_models = {}  # record_id -> (internal_model, provider_module, provider_name)
    cost_groups = {}    # (model, provider_name) -> record_count
    for record in actionable:
        internal, pmod, pname = _resolve_record_model(record, model, provider)
        record_models[record["id"]] = (internal, pmod, pname)
        key = (internal, pname)
        cost_groups[key] = cost_groups.get(key, 0) + 1

    num_variations = max(1, min(2, num_variations))
    var_range = range(1, num_variations + 1)
    images_total = count * num_variations
    total_cost = 0.0

    print(f"\n{'=' * 50}")
    print(f"  Image Generation Batch")
    print(f"{'=' * 50}")
    print(f"  Records: {count}")
    print(f"  Images per record: {num_variations} (variations)")
    print(f"  Total images: {images_total}")
    if aspect_ratio:
        print(f"  Aspect ratio: {aspect_ratio} (override)")
    print(f"  Resolution: {resolution}")
    for (m, pname), rec_count in cost_groups.items():
        unit = config.get_cost(m, pname)
        group_cost = rec_count * num_variations * unit
        total_cost += group_cost
        display = _MODEL_DISPLAY_NAMES.get(m, m)
        print(f"  {display} via {pname}: {rec_count} record(s) x {num_variations} = {rec_count * num_variations} images @ ${unit:.2f} = ${group_cost:.2f}")
    print(f"  Total estimated cost: ${total_cost:.2f}")
    print(f"{'=' * 50}\n")

    # Upload reference images for any async (Kie AI) providers
    reference_urls = None
    needs_async = any(not is_sync(pm, "image") for _, pm, _ in record_models.values())
    if reference_paths and needs_async:
        print_status("Uploading reference images (one-time)...")
        reference_urls = upload_references(reference_paths)
        print_status(f"Uploaded {len(reference_urls)} reference image(s)", "OK")

    # --- Phase 1: Submit / Generate all images ---
    print(f"\n--- Phase 1: Generating {images_total} images ---")

    # submissions: list of (record, var_num, task_id_or_result, model, provider_module, provider_name, is_sync)
    submissions = []

    for record in actionable:
        fields = record.get("fields", {})
        ad_name = fields.get("Ad Name", "untitled")
        prompt = fields.get("Image Prompt", "")
        effective_ratio = aspect_ratio or _detect_aspect_ratio(prompt)

        rec_model, rec_pmod, rec_pname = record_models[record["id"]]
        rec_sync = is_sync(rec_pmod, "image")
        display_model = _MODEL_DISPLAY_NAMES.get(rec_model, rec_model)

        for var_num in var_range:
            print_status(f"Generating: {ad_name} (variation {var_num}) [{display_model} via {rec_pname}]")
            try:
                if rec_sync:
                    result = rec_pmod.submit_image(
                        prompt, reference_paths=reference_paths,
                        aspect_ratio=effective_ratio, resolution=resolution, model=rec_model
                    )
                    submissions.append((record, var_num, result, rec_model, rec_pmod, rec_pname, True))
                    print_status(f"Done -> {result['result_url'][:50]}...", "OK")
                else:
                    task_id = rec_pmod.submit_image(
                        prompt, reference_urls=reference_urls,
                        aspect_ratio=effective_ratio, resolution=resolution, model=rec_model
                    )
                    submissions.append((record, var_num, task_id, rec_model, rec_pmod, rec_pname, False))
                    print_status(f"Task {task_id}", "OK")
            except Exception as e:
                print_status(f"Failed: {e}", "XX")
                submissions.append((record, var_num, None, rec_model, rec_pmod, rec_pname, rec_sync))
            time.sleep(1)

    # --- Phase 2: Poll async tasks (grouped by provider) ---
    results_map = {}

    # Sync results: store directly
    for record, var_num, result, rec_model, rec_pmod, rec_pname, rec_sync in submissions:
        if rec_sync and result and isinstance(result, dict):
            key = f"{record['id']}_var{var_num}"
            results_map[key] = result

    # Async results: group task_ids by provider module and poll each
    async_by_provider = {}  # provider_module -> [task_id, ...]
    for record, var_num, task_id, rec_model, rec_pmod, rec_pname, rec_sync in submissions:
        if not rec_sync and task_id is not None:
            async_by_provider.setdefault(rec_pmod, []).append(task_id)

    for pmod, task_ids in async_by_provider.items():
        print(f"\n--- Phase 2: Polling {len(task_ids)} async tasks ---")
        polled = pmod.poll_tasks_parallel(task_ids, max_wait=300, poll_interval=5)
        results_map.update(polled)

    # --- Phase 3: Update Airtable per record ---
    print(f"\n--- Phase 3: Updating Airtable ---")
    record_tasks = {}  # record_id -> [(var_num, key)]
    for record, var_num, id_or_result, rec_model, rec_pmod, rec_pname, rec_sync in submissions:
        rid = record["id"]
        if rid not in record_tasks:
            record_tasks[rid] = []
        if rec_sync:
            key = f"{rid}_var{var_num}" if id_or_result else None
        else:
            key = id_or_result  # task_id or None
        record_tasks[rid].append((var_num, key))

    record_map = {r["id"]: r for r in actionable}
    results = []
    succeeded = 0
    images_generated = 0
    actual_cost = 0.0

    for rid, tasks in record_tasks.items():
        record = record_map[rid]
        ad_name = record.get("fields", {}).get("Ad Name", "untitled")
        rec_model, _, rec_pname = record_models[rid]
        update_fields = {}
        record_ok = True

        for var_num, key in tasks:
            if key is None:
                record_ok = False
                continue
            result = results_map.get(key, {})
            if result.get("status") == "error":
                print_status(f"'{ad_name}' variation {var_num} failed: {result.get('error')}", "XX")
                record_ok = False
            else:
                update_fields[f"Generated Image {var_num}"] = [{"url": result["result_url"]}]
                images_generated += 1
                actual_cost += config.get_cost(rec_model, rec_pname)

        if update_fields:
            update_fields["Image Status"] = "Generated"
            update_fields["Image Model"] = _MODEL_DISPLAY_NAMES.get(rec_model, rec_model)
            update_record(rid, update_fields)
            print_status(f"Airtable updated for '{ad_name}'", "OK")

        if record_ok:
            succeeded += 1
        results.append(update_fields if update_fields else None)

    print(f"\n{'=' * 50}")
    print(f"  Batch complete: {succeeded}/{count} records ({images_generated} images)")
    print(f"  Actual cost: ${actual_cost:.2f}")
    print(f"{'=' * 50}\n")

    return results
