import sys
sys.path.insert(0, '.')

from tools.image_gen import generate_ugc_image
from tools.utils import print_status

REF = "references/inputs/iish/board_flavor_color_system.png"

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

results = {}
for flavor, pattern in FLAVORS.items():
    prompt = BASE.format(pattern=pattern)
    print_status(f"=== {flavor} ===")
    try:
        res = generate_ugc_image(
            prompt,
            reference_paths=[REF],
            aspect_ratio="4:5",
            resolution="1K",
            model="nano-banana-pro",
            provider="google",
        )
        results[flavor] = res
        print_status(f"{flavor}: {res.get('result_url')}", "OK")
    except Exception as e:
        results[flavor] = {"status": "error", "error": str(e)}
        print_status(f"{flavor} FAILED: {e}", "XX")

print("\n=== SUMMARY ===")
for flavor, res in results.items():
    print(flavor, "->", res)
