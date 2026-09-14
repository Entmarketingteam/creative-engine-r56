import sys, os
sys.path.insert(0, '.')

from tools.image_gen import generate_ugc_image
from tools.video_gen import generate_ugc_video
from tools.kie_upload import upload_reference
from tools.utils import print_status, download_file

REF_LOCAL = "references/inputs/iish/board_flavor_color_system.png"
OUT_DIR = "creative-output/iish-brand-kit-proof"
os.makedirs(OUT_DIR, exist_ok=True)

BASE = (
    "Editorial lifestyle photo of an iish. collagen + creatine stand-up pouch "
    "— {pattern} — playful lowercase 'iish.' berry wordmark — on a sun-drenched real "
    "kitchen counter with a morning coffee cup and a kid's cereal bowl at the edge of frame. "
    "Golden hour light through a window, soft shadows. Joyful, warm, lived-in — candy-colored "
    "retro optimism, not luxury, not clinical. Photorealistic, 8k. Use the attached reference "
    "image for the exact colorway/pattern system — match it precisely, do not invent new colors."
)

FLAVORS = {
    "unflavored": "retro geometric half-circle checker pattern in soft violet-blue and cool pink",
    "peach-rings": "retro geometric half-circle checker pattern in tangerine and peach-ring pink",
    "tropical": "retro geometric citrus-burst pattern in rainforest green and chartreuse",
}

results = {"images": {}, "videos": {}}

# --- Step 0: upload reference for providers that need a hosted URL (WaveSpeed) ---
print_status("=== Uploading reference board to Kie.ai hosting ===")
ref_url = upload_reference(REF_LOCAL)
print_status(f"ref_url: {ref_url}", "OK")

# --- Step 1: Nano Banana Pro (Google) — 3 flavor hero shots ---
print_status("=== Nano Banana Pro (Google) — 3 flavors ===")
for flavor, pattern in FLAVORS.items():
    prompt = BASE.format(pattern=pattern)
    try:
        res = generate_ugc_image(
            prompt, reference_paths=[REF_LOCAL],
            aspect_ratio="4:5", resolution="1K",
            model="nano-banana-pro", provider="google",
        )
        local_path = os.path.join(OUT_DIR, f"nanobanana-{flavor}.png")
        download_file(res["result_url"], local_path)
        results["images"][f"nanobanana-{flavor}"] = {**res, "local_path": local_path}
        print_status(f"nanobanana-{flavor} -> {local_path}", "OK")
    except Exception as e:
        results["images"][f"nanobanana-{flavor}"] = {"status": "error", "error": str(e)}
        print_status(f"nanobanana-{flavor} FAILED: {e}", "XX")

# --- Step 2: GPT Image 1.5 (WaveSpeed) — 3 flavor hero shots ---
print_status("=== GPT Image 1.5 (WaveSpeed) — 3 flavors ===")
for flavor, pattern in FLAVORS.items():
    prompt = BASE.format(pattern=pattern)
    try:
        res = generate_ugc_image(
            prompt, reference_urls=[ref_url],
            aspect_ratio="4:5", resolution="1K",
            model="gpt-image-1.5", provider="wavespeed",
        )
        local_path = os.path.join(OUT_DIR, f"gptimage-{flavor}.png")
        download_file(res["result_url"], local_path)
        results["images"][f"gptimage-{flavor}"] = {**res, "local_path": local_path}
        print_status(f"gptimage-{flavor} -> {local_path}", "OK")
    except Exception as e:
        results["images"][f"gptimage-{flavor}"] = {"status": "error", "error": str(e)}
        print_status(f"gptimage-{flavor} FAILED: {e}", "XX")

# --- Step 3: animate the Unflavored Nano Banana hero shot across 3 video models ---
start = results["images"].get("nanobanana-unflavored", {})
start_url = start.get("result_url")

if start_url:
    video_prompt = (
        "Camera slowly pushes in on the iish. collagen + creatine pouch on the kitchen "
        "counter. Steam gently rises from the coffee cup beside it. Warm morning light "
        "shifts softly. Calm, joyful, lived-in mood, no text, no logos changing."
    )
    video_jobs = [
        ("veo-3.1", "google"),
        ("kling-3.0", "kie"),
        ("sora-2-pro", "kie"),
    ]
    print_status("=== Video: Veo 3.1 / Kling 3.0 / Sora 2 Pro (from unflavored hero) ===")
    for model, provider in video_jobs:
        try:
            res = generate_ugc_video(
                video_prompt, image_url=start_url,
                model=model, duration="5", aspect_ratio="4:5", provider=provider,
            )
            local_path = os.path.join(OUT_DIR, f"video-{model}.mp4")
            download_file(res["result_url"], local_path)
            results["videos"][model] = {**res, "local_path": local_path}
            print_status(f"video-{model} -> {local_path}", "OK")
        except Exception as e:
            results["videos"][model] = {"status": "error", "error": str(e)}
            print_status(f"video-{model} FAILED: {e}", "XX")
else:
    print_status("No unflavored hero image available — skipping video step", "XX")

print("\n=== SUMMARY ===")
for k, v in results["images"].items():
    print("IMG", k, "->", v.get("local_path") or v.get("error"))
for k, v in results["videos"].items():
    print("VID", k, "->", v.get("local_path") or v.get("error"))
