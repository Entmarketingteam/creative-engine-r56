"""
Kie AI file upload module.
Uploads reference product images to Kie.ai's free hosting (3-day persistence).
"""

import requests
from pathlib import Path
from . import config
from .utils import print_status


def upload_reference(file_path, api_key=None):
    """
    Upload a file to Kie.ai's file hosting and return the public URL.
    Uses the same KIE_API_KEY - no extra credentials needed.
    Files are hosted for 3 days (free).

    Args:
        file_path: Path to the local file
        api_key: Optional API key override (defaults to config)

    Returns:
        str: The hosted download URL

    Raises:
        FileNotFoundError: If file doesn't exist
        Exception: If upload fails
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    api_key = api_key or config.KIE_API_KEY
    if not api_key:
        raise ValueError("KIE_API_KEY is required")

    print_status(f"Uploading to Kie.ai: {file_path.name}")

    headers = {"Authorization": f"Bearer {api_key}"}

    with open(file_path, "rb") as f:
        files = {"file": (file_path.name, f)}
        # CRITICAL: uploadPath must be included or the request fails with 400
        data = {"uploadPath": "creative-cloner"}

        response = requests.post(
            config.KIE_FILE_UPLOAD_URL,
            headers=headers,
            files=files,
            data=data,
        )

    if response.status_code != 200:
        raise Exception(f"Kie.ai upload failed: {response.status_code} - {response.text}")

    result = response.json()

    if result.get("success") or result.get("code") == 200:
        file_url = result.get("data", {}).get("downloadUrl")
        if file_url:
            print_status(f"Upload successful: {file_url}", "OK")
            return file_url
        else:
            raise Exception(f"No downloadUrl in response: {result}")
    else:
        raise Exception(f"Upload failed: {result.get('msg', result)}")


def upload_references(file_paths, api_key=None):
    """
    Upload multiple reference files and return their hosted URLs.

    Args:
        file_paths: List of local file paths
        api_key: Optional API key override

    Returns:
        list[str]: List of hosted URLs
    """
    urls = []
    for path in file_paths:
        url = upload_reference(path, api_key)
        urls.append(url)
    return urls
