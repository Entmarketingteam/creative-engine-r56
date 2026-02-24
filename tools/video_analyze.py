"""
Video analysis module — uses Gemini Files API to analyze reference UGC videos.

Flow:
  1. Upload local video to Gemini Files API (resumable upload, any size)
  2. Poll until state == ACTIVE (Gemini finishes processing)
  3. Call generateContent with the file URI + analysis prompt
  4. Delete the file from Gemini (cleanup)
  5. Return structured analysis for use in prompt writing

The analysis is returned as a dict with keys matching UGC prompt dimensions
(hook, person, setting, camera, product_interaction, pacing, tone, dialogue,
audio, authenticity_score, prompt_notes) plus a formatted 'summary' string
ready to paste into a prompt-writing session.
"""

import os
import time
import mimetypes
from pathlib import Path

import requests

from . import config
from .utils import print_status


# --- Constants ---

_UPLOAD_URL = "https://generativelanguage.googleapis.com/upload/v1beta/files"
_FILES_URL = "https://generativelanguage.googleapis.com/v1beta/files/{name}"
_GENERATE_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

_ANALYSIS_MODEL = "gemini-2.0-flash"

_SUPPORTED_MIME_TYPES = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".avi": "video/avi",
    ".webm": "video/webm",
    ".wmv": "video/wmv",
    ".mpg": "video/mpeg",
    ".mpeg": "video/mpeg",
    ".flv": "video/x-flv",
    ".3gp": "video/3gpp",
}

_DEFAULT_ANALYSIS_PROMPT = """Analyze this UGC (User Generated Content) ad video in detail.
Extract everything that would help recreate or build on this style for new UGC ads.

Return your analysis in EXACTLY this format (keep the labels, fill in the values):

HOOK: [What happens in the first 2-3 seconds that grabs attention]
PERSON: [Gender, approximate age range, appearance, clothing style]
SETTING: [Background, location, indoor/outdoor, time of day, lighting quality and color]
CAMERA: [Angle — selfie/eye-level/below/above; distance — close-up/mid/wide; movement — static/slight drift/handheld shake]
PRODUCT INTERACTION: [How the product is held, shown, or referenced; specific angles; label visibility]
PACING: [Overall speed — fast/medium/slow; cut frequency; use of pauses or holds]
TONE & ENERGY: [Emotional register — e.g. genuinely excited, calm and informative, playful, surprised, etc.]
DIALOGUE: [Key phrases or direct quotes; speaking style — natural/casual vs scripted; pace of speech]
AUDIO: [Music — yes/no/type; ambient sound; voice characteristics — tone, accent, energy]
AUTHENTICITY SCORE: [1–10 score and one sentence reason]
PROMPT NOTES:
- [Key element 1 to emphasize when writing image/video prompts]
- [Key element 2]
- [Key element 3]
"""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _auth_headers():
    return {"x-goog-api-key": config.GOOGLE_API_KEY}


def _get_mime_type(file_path):
    """Return MIME type for the video file. Raises if unsupported."""
    ext = Path(file_path).suffix.lower()
    mime = _SUPPORTED_MIME_TYPES.get(ext)
    if not mime:
        # Fallback: let mimetypes library try
        mime, _ = mimetypes.guess_type(str(file_path))
    if not mime or not mime.startswith("video/"):
        raise ValueError(
            f"Unsupported video format: '{ext}'. "
            f"Supported: {', '.join(_SUPPORTED_MIME_TYPES.keys())}"
        )
    return mime


def _upload_video(file_path):
    """
    Upload a local video file to Gemini Files API via resumable upload.

    Returns:
        dict: File metadata with 'name' and 'uri' fields
    """
    file_path = Path(file_path)
    file_size = file_path.stat().st_size
    mime_type = _get_mime_type(file_path)
    display_name = file_path.name

    print_status(f"Uploading '{display_name}' ({file_size / 1024 / 1024:.1f} MB) to Gemini Files API...")

    # --- Step 1: Initiate resumable upload ---
    init_headers = {
        **_auth_headers(),
        "X-Goog-Upload-Protocol": "resumable",
        "X-Goog-Upload-Command": "start",
        "X-Goog-Upload-Header-Content-Length": str(file_size),
        "X-Goog-Upload-Header-Content-Type": mime_type,
        "Content-Type": "application/json",
    }
    init_response = requests.post(
        _UPLOAD_URL,
        headers=init_headers,
        json={"file": {"display_name": display_name}},
        timeout=30,
    )
    if init_response.status_code != 200:
        raise Exception(
            f"Upload init failed ({init_response.status_code}): {init_response.text[:400]}"
        )

    upload_url = init_response.headers.get("x-goog-upload-url")
    if not upload_url:
        raise Exception("No upload URL returned from Gemini Files API init")

    # --- Step 2: Upload file bytes ---
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    upload_headers = {
        "Content-Length": str(file_size),
        "X-Goog-Upload-Offset": "0",
        "X-Goog-Upload-Command": "upload, finalize",
    }
    upload_response = requests.post(
        upload_url,
        headers=upload_headers,
        data=file_bytes,
        timeout=300,
    )
    if upload_response.status_code not in (200, 201):
        raise Exception(
            f"Upload failed ({upload_response.status_code}): {upload_response.text[:400]}"
        )

    file_meta = upload_response.json().get("file", upload_response.json())
    file_name = file_meta.get("name")
    file_uri = file_meta.get("uri")

    if not file_name or not file_uri:
        raise Exception(f"Missing name/uri in upload response: {file_meta}")

    print_status(f"Uploaded: {file_name}", "OK")
    return {"name": file_name, "uri": file_uri, "mimeType": mime_type}


def _wait_for_active(file_name, max_wait=120, poll_interval=5):
    """
    Poll the Files API until the file state is ACTIVE (ready for analysis).

    Returns:
        dict: File metadata once ACTIVE
    """
    start_time = time.time()
    print_status("Waiting for Gemini to process video...")

    # file_name from upload response is "files/xyz"; strip prefix to avoid
    # double "files/" in the URL (...v1beta/files/files/xyz → 404)
    name_id = file_name.removeprefix("files/")

    while time.time() - start_time < max_wait:
        url = _FILES_URL.format(name=name_id)
        response = requests.get(url, headers=_auth_headers(), timeout=30)

        if response.status_code != 200:
            elapsed = int(time.time() - start_time)
            print_status(f"File poll returned {response.status_code}, retrying... ({elapsed}s)", "!!")
            time.sleep(poll_interval)
            continue

        file_meta = response.json()
        state = file_meta.get("state", "UNKNOWN")

        if state == "ACTIVE":
            print_status("Video ready for analysis", "OK")
            return file_meta
        elif state == "FAILED":
            raise Exception(f"Gemini file processing failed: {file_meta}")
        else:
            elapsed = int(time.time() - start_time)
            print_status(f"File state: {state} ({elapsed}s elapsed)", "..")
            time.sleep(poll_interval)

    raise Exception(f"File did not become ACTIVE within {max_wait}s")


def _delete_file(file_name):
    """Delete a file from Gemini Files API after analysis (cleanup)."""
    name_id = file_name.removeprefix("files/")
    url = _FILES_URL.format(name=name_id)
    response = requests.delete(url, headers=_auth_headers(), timeout=30)
    if response.status_code in (200, 204):
        print_status(f"Cleaned up Gemini file: {file_name}", "OK")
    else:
        # Non-fatal — just warn
        print_status(f"Could not delete Gemini file {file_name}: {response.status_code}", "!!")


def _run_analysis(file_uri, mime_type, prompt):
    """
    Call generateContent with the uploaded video file and analysis prompt.

    Returns:
        str: Raw text response from Gemini
    """
    url = _GENERATE_URL.format(model=_ANALYSIS_MODEL)
    payload = {
        "contents": [{
            "parts": [
                {"fileData": {"mimeType": mime_type, "fileUri": file_uri}},
                {"text": prompt},
            ]
        }]
    }
    headers = {**_auth_headers(), "Content-Type": "application/json"}

    print_status(f"Running analysis with {_ANALYSIS_MODEL}...")
    response = requests.post(url, headers=headers, json=payload, timeout=120)

    if response.status_code != 200:
        raise Exception(
            f"Gemini generateContent failed ({response.status_code}): {response.text[:500]}"
        )

    result = response.json()
    candidates = result.get("candidates", [])
    if not candidates:
        raise Exception(f"No candidates in Gemini response: {result}")

    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts).strip()
    if not text:
        raise Exception(f"Empty text in Gemini analysis response")

    return text


def _parse_analysis(raw_text):
    """
    Parse the structured analysis text into a dict.
    Gracefully handles missing fields.
    """
    result = {
        "hook": "",
        "person": "",
        "setting": "",
        "camera": "",
        "product_interaction": "",
        "pacing": "",
        "tone": "",
        "dialogue": "",
        "audio": "",
        "authenticity_score": "",
        "prompt_notes": [],
        "raw": raw_text,
    }

    field_map = {
        "HOOK:": "hook",
        "PERSON:": "person",
        "SETTING:": "setting",
        "CAMERA:": "camera",
        "PRODUCT INTERACTION:": "product_interaction",
        "PACING:": "pacing",
        "TONE & ENERGY:": "tone",
        "DIALOGUE:": "dialogue",
        "AUDIO:": "audio",
        "AUTHENTICITY SCORE:": "authenticity_score",
    }

    lines = raw_text.splitlines()
    current_key = None
    prompt_notes_mode = False

    for line in lines:
        stripped = line.strip()

        if stripped.upper().startswith("PROMPT NOTES"):
            prompt_notes_mode = True
            current_key = None
            continue

        if prompt_notes_mode:
            if stripped.startswith("-"):
                result["prompt_notes"].append(stripped.lstrip("- ").strip())
            elif stripped:
                result["prompt_notes"].append(stripped)
            continue

        matched = False
        for label, key in field_map.items():
            if stripped.upper().startswith(label):
                current_key = key
                value = stripped[len(label):].strip().strip("[]")
                if value:
                    result[key] = value
                matched = True
                break

        if not matched and current_key and stripped:
            # Continuation of previous field
            result[current_key] = (result[current_key] + " " + stripped).strip()

    return result


def _format_summary(analysis, video_name=""):
    """Build a clean summary string ready to use when writing prompts."""
    lines = [f"## Reference Video Analysis: {video_name}" if video_name else "## Reference Video Analysis"]
    lines.append("")

    field_labels = [
        ("hook", "Hook"),
        ("person", "Person"),
        ("setting", "Setting"),
        ("camera", "Camera"),
        ("product_interaction", "Product Interaction"),
        ("pacing", "Pacing"),
        ("tone", "Tone & Energy"),
        ("dialogue", "Dialogue"),
        ("audio", "Audio"),
        ("authenticity_score", "Authenticity"),
    ]

    for key, label in field_labels:
        val = analysis.get(key, "")
        if val:
            lines.append(f"**{label}:** {val}")

    notes = analysis.get("prompt_notes", [])
    if notes:
        lines.append("\n**Prompt Notes:**")
        for note in notes:
            lines.append(f"  - {note}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_video(file_path, prompt=None):
    """
    Analyze a single reference video with Gemini.

    Uploads the video to Gemini Files API, waits for processing,
    runs the analysis, cleans up the file, and returns structured results.

    Args:
        file_path: Local path to the video file
        prompt: Custom analysis prompt (uses default UGC analysis prompt if None)

    Returns:
        dict with keys:
            hook, person, setting, camera, product_interaction, pacing,
            tone, dialogue, audio, authenticity_score, prompt_notes (list),
            raw (full text), summary (formatted string)
    """
    if not config.GOOGLE_API_KEY:
        raise Exception("GOOGLE_API_KEY not set — add it to .claude/.env")

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Video file not found: {file_path}")

    analysis_prompt = prompt or _DEFAULT_ANALYSIS_PROMPT
    file_name = None

    try:
        # 1. Upload
        file_meta = _upload_video(file_path)
        file_name = file_meta["name"]

        # 2. Wait for processing
        _wait_for_active(file_name)

        # 3. Analyze
        raw_text = _run_analysis(file_meta["uri"], file_meta["mimeType"], analysis_prompt)

    finally:
        # 4. Cleanup regardless of success/failure
        if file_name:
            _delete_file(file_name)

    # 5. Parse and return
    analysis = _parse_analysis(raw_text)
    analysis["summary"] = _format_summary(analysis, video_name=file_path.name)

    print_status(f"Analysis complete for '{file_path.name}'", "OK")
    return analysis


def analyze_multiple(file_paths, prompt=None):
    """
    Analyze multiple reference videos and return a combined summary.

    Runs analysis sequentially (Gemini Files API has no meaningful
    parallelism benefit for video processing time).

    Args:
        file_paths: List of local video file paths
        prompt: Custom analysis prompt (uses default if None)

    Returns:
        dict with:
            analyses: list of individual analysis dicts (one per video)
            combined_summary: formatted string combining all analyses
    """
    analyses = []
    for i, path in enumerate(file_paths, 1):
        print(f"\n--- Analyzing video {i}/{len(file_paths)}: {Path(path).name} ---")
        analysis = analyze_video(path, prompt=prompt)
        analyses.append(analysis)

    combined_lines = [f"## Reference Video Analysis ({len(analyses)} video(s))\n"]
    for i, analysis in enumerate(analyses, 1):
        combined_lines.append(f"### Video {i}: {analysis.get('summary', '')}")
        combined_lines.append("")

    combined_summary = "\n".join(combined_lines)

    return {
        "analyses": analyses,
        "combined_summary": combined_summary,
    }
