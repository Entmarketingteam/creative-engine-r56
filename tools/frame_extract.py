"""
Frame extraction with sharpness ranking.

Pulls every frame from a source video at native resolution, scores each for
sharpness (Laplacian variance — a standard blur-detection metric: a sharp
image has high-frequency edge detail, a blurry one doesn't), and saves the
top N sharpest, time-spaced frames as full-resolution stills.

Resolution is capped by whatever the source video was actually shot/exported
at — this cannot invent detail that isn't there. Point it at the highest
quality source file available (ideally original camera-roll footage, not a
social-platform re-download, which is already heavily recompressed).

Usage:
    python3 tools/frame_extract.py <video_path> <output_dir> [--top N] [--min-gap SECONDS]
"""

import subprocess
import sys
import argparse
import tempfile
import shutil
from pathlib import Path

from PIL import Image, ImageFilter
import numpy as np

_LAPLACIAN_KERNEL = ImageFilter.Kernel((3, 3), [0, 1, 0, 1, -4, 1, 0, 1, 0], scale=1)


def _sharpness_score(image_path):
    """Laplacian variance sharpness score. Higher = sharper."""
    img = Image.open(image_path).convert("L")
    edges = img.filter(_LAPLACIAN_KERNEL)
    arr = np.asarray(edges, dtype=np.float64)
    return float(arr.var())


def extract_frames(video_path, work_dir, fps=2):
    """Extract frames at native resolution, `fps` frames per second of source video."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(work_dir / "frame_%06d.jpg")
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-qscale:v", "2",  # highest quality JPEG encode (2 = near-lossless)
        pattern,
        "-loglevel", "error",
    ]
    subprocess.run(cmd, check=True)
    return sorted(work_dir.glob("frame_*.jpg"))


def rank_and_select(frames, top_n=8, min_gap_frames=4):
    """Score all frames, pick the top N sharpest with a minimum spacing between picks
    (so we don't return 8 near-identical frames from the same half-second)."""
    scored = [(f, _sharpness_score(f)) for f in frames]
    scored.sort(key=lambda x: x[1], reverse=True)

    selected = []
    used_indices = set()
    for f, score in scored:
        idx = int(f.stem.split("_")[1])
        if any(abs(idx - u) < min_gap_frames for u in used_indices):
            continue
        selected.append((f, score))
        used_indices.add(idx)
        if len(selected) >= top_n:
            break
    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video_path")
    ap.add_argument("output_dir")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--fps", type=float, default=2.0, help="frames sampled per second of source video")
    ap.add_argument("--min-gap", type=int, default=4, help="min frame-index spacing between picks")
    args = ap.parse_args()

    video_path = Path(args.video_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        print(f"Extracting frames from {video_path.name} at {args.fps} fps...")
        frames = extract_frames(video_path, tmp, fps=args.fps)
        print(f"  {len(frames)} candidate frames extracted at native resolution")

        print("Scoring sharpness...")
        selected = rank_and_select(frames, top_n=args.top, min_gap_frames=args.min_gap)

        for i, (f, score) in enumerate(selected, 1):
            out_path = output_dir / f"{video_path.stem}_sharp_{i:02d}.jpg"
            shutil.copy(f, out_path)
            print(f"  [{i}] sharpness={score:.1f}  ->  {out_path}")

    print(f"\nDone. {len(selected)} sharpest frames saved to {output_dir}")


if __name__ == "__main__":
    main()
