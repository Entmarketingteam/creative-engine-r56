"""
Airtable CRUD operations for Creative Content Engine.
Handles schema creation, record management, and status tracking.
"""

import requests
from . import config
from .utils import print_status


def _headers():
    """Standard Airtable API headers."""
    return {
        "Authorization": f"Bearer {config.AIRTABLE_API_KEY}",
        "Content-Type": "application/json",
    }


def _table_url():
    """Base URL for the Content table."""
    return f"{config.AIRTABLE_API_URL}/{config.AIRTABLE_BASE_ID}/{config.AIRTABLE_TABLE_NAME}"


# --- Schema Creation ---


def create_ugc_table():
    """
    Create the Content table in Airtable using the Metadata API.
    This is idempotent - if the table already exists, it will raise an error
    that can be caught by the caller.

    Returns:
        dict: The created table metadata
    """
    url = f"https://api.airtable.com/v0/meta/bases/{config.AIRTABLE_BASE_ID}/tables"

    table_schema = {
        "name": config.AIRTABLE_TABLE_NAME,
        "fields": [
            {"name": "Index", "type": "number", "options": {"precision": 0}},
            {"name": "Ad Name", "type": "singleLineText"},
            {"name": "Product", "type": "singleLineText"},
            {"name": "Reference Images", "type": "multipleAttachments"},
            {"name": "Image Prompt", "type": "multilineText"},
            {
                "name": "Image Model",
                "type": "singleSelect",
                "options": {
                    "choices": [
                        {"name": "Nano Banana", "color": "yellowBright"},
                        {"name": "Nano Banana Pro", "color": "orangeBright"},
                        {"name": "GPT Image 1.5", "color": "tealBright"},
                    ]
                },
            },
            {
                "name": "Image Status",
                "type": "singleSelect",
                "options": {
                    "choices": [
                        {"name": "Pending", "color": "yellowBright"},
                        {"name": "Generated", "color": "cyanBright"},
                        {"name": "Approved", "color": "greenBright"},
                        {"name": "Rejected", "color": "redBright"},
                    ]
                },
            },
            {"name": "Generated Image 1", "type": "multipleAttachments"},
            {"name": "Generated Image 2", "type": "multipleAttachments"},
            {"name": "Video Prompt", "type": "multilineText"},
            {
                "name": "Video Model",
                "type": "singleSelect",
                "options": {
                    "choices": [
                        {"name": "Kling 3.0", "color": "purpleBright"},
                        {"name": "Sora 2", "color": "cyanDark1"},
                        {"name": "Sora 2 Pro", "color": "blueBright"},
                        {"name": "Veo 3.1", "color": "greenBright"},
                    ]
                },
            },
            {
                "name": "Video Status",
                "type": "singleSelect",
                "options": {
                    "choices": [
                        {"name": "Pending", "color": "yellowBright"},
                        {"name": "Generated", "color": "cyanBright"},
                        {"name": "Approved", "color": "greenBright"},
                        {"name": "Rejected", "color": "redBright"},
                    ]
                },
            },
            {"name": "Generated Video 1", "type": "multipleAttachments"},
            {"name": "Generated Video 2", "type": "multipleAttachments"},
        ],
    }

    print_status("Creating Content table in Airtable...")

    response = requests.post(url, headers=_headers(), json=table_schema)

    if response.status_code == 200:
        result = response.json()
        print_status(f"Table created: {result.get('name')} (ID: {result.get('id')})", "OK")
        return result
    elif response.status_code == 422:
        error = response.json()
        if "DUPLICATE_TABLE_NAME" in str(error):
            print_status("Table 'Content' already exists - skipping creation", "OK")
            return {"name": config.AIRTABLE_TABLE_NAME, "exists": True}
        raise Exception(f"Airtable schema error: {error}")
    else:
        raise Exception(f"Airtable API error ({response.status_code}): {response.text}")


def add_image_model_field():
    """
    Add the 'Image Model' singleSelect field to an existing Content table.
    Safe to call if the field already exists (returns existing field info).

    Returns:
        dict: The created or existing field metadata
    """
    # Look up the table ID by name
    meta_url = f"https://api.airtable.com/v0/meta/bases/{config.AIRTABLE_BASE_ID}/tables"
    resp = requests.get(meta_url, headers=_headers())
    if resp.status_code != 200:
        raise Exception(f"Failed to list tables: {resp.text}")

    tables = resp.json().get("tables", [])
    table_id = next(
        (t["id"] for t in tables if t["name"] == config.AIRTABLE_TABLE_NAME), None
    )
    if not table_id:
        raise Exception(f"Table '{config.AIRTABLE_TABLE_NAME}' not found in base")

    field_url = f"https://api.airtable.com/v0/meta/bases/{config.AIRTABLE_BASE_ID}/tables/{table_id}/fields"
    field_schema = {
        "name": "Image Model",
        "type": "singleSelect",
        "options": {
            "choices": [
                {"name": "Nano Banana", "color": "yellowBright"},
                {"name": "Nano Banana Pro", "color": "orangeBright"},
                {"name": "GPT Image 1.5", "color": "tealBright"},
            ]
        },
    }

    response = requests.post(field_url, headers=_headers(), json=field_schema)

    if response.status_code == 200:
        result = response.json()
        print_status(f"Field 'Image Model' created (ID: {result.get('id')})", "OK")
        return result
    elif response.status_code == 422:
        error = response.json()
        if "DUPLICATE_FIELD_NAME" in str(error):
            print_status("Field 'Image Model' already exists — skipping", "OK")
            return {"name": "Image Model", "exists": True}
        raise Exception(f"Airtable schema error: {error}")
    else:
        raise Exception(f"Airtable API error ({response.status_code}): {response.text}")


# --- Record CRUD ---


def create_record(fields):
    """
    Create a single record in the UGC Ads table.

    Args:
        fields: dict of field name -> value

    Returns:
        dict: The created record
    """
    response = requests.post(
        _table_url(),
        headers=_headers(),
        json={"fields": fields},
    )

    if response.status_code != 200:
        raise Exception(f"Airtable create failed: {response.text}")

    record = response.json()
    print_status(f"Created record: {record.get('id')}", "OK")
    return record


def create_records_batch(records_fields):
    """
    Create multiple records in batches of 10 (Airtable limit).

    Args:
        records_fields: list of field dicts

    Returns:
        list: All created records
    """
    all_created = []

    for i in range(0, len(records_fields), 10):
        batch = records_fields[i : i + 10]
        records = [{"fields": f} for f in batch]

        response = requests.post(
            _table_url(),
            headers=_headers(),
            json={"records": records},
        )

        if response.status_code != 200:
            raise Exception(f"Airtable batch create failed (batch {i}): {response.text}")

        created = response.json().get("records", [])
        all_created.extend(created)
        print_status(f"Created {len(all_created)}/{len(records_fields)} records", "OK")

    return all_created


def get_records(filter_formula=None):
    """
    Get records from the UGC Ads table with optional filtering.

    Args:
        filter_formula: Airtable formula string, e.g. '{Image Status} = "Approved"'

    Returns:
        list: All matching records (handles pagination)
    """
    params = {}
    if filter_formula:
        params["filterByFormula"] = filter_formula

    all_records = []
    offset = None

    while True:
        if offset:
            params["offset"] = offset

        response = requests.get(_table_url(), headers=_headers(), params=params)

        if response.status_code != 200:
            raise Exception(f"Airtable query failed: {response.text}")

        data = response.json()
        all_records.extend(data.get("records", []))

        offset = data.get("offset")
        if not offset:
            break

    return all_records


def update_record(record_id, fields):
    """
    Update a single record.

    Args:
        record_id: Airtable record ID
        fields: dict of field name -> new value

    Returns:
        dict: The updated record
    """
    url = f"{_table_url()}/{record_id}"

    response = requests.patch(url, headers=_headers(), json={"fields": fields})

    if response.status_code != 200:
        raise Exception(f"Airtable update failed: {response.text}")

    return response.json()


# --- Index Management ---


def get_next_index():
    """
    Query all records to find the current max Index, return max + 1.
    Ensures unique sequential Index values across batches.

    Returns:
        int: The next available Index value
    """
    records = get_records()
    if not records:
        return 1

    max_index = 0
    for record in records:
        idx = record.get("fields", {}).get("Index")
        if idx is not None and idx > max_index:
            max_index = int(idx)

    return max_index + 1


# --- Convenience Queries ---


def get_pending_images():
    """Get records where Image Status is Pending."""
    return get_records('{Image Status} = "Pending"')


def get_approved_images():
    """Get records where Image Status is Approved (ready for video generation)."""
    return get_records('{Image Status} = "Approved"')


def get_pending_videos():
    """Get records where Video Status is Pending."""
    return get_records('{Video Status} = "Pending"')


def get_approved_videos():
    """Get records where Video Status is Approved."""
    return get_records('{Video Status} = "Approved"')
