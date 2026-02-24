"""
Creative Content Engine - Airtable Setup
=========================================
One-time setup script that creates the Content table in your Airtable base.

Usage:
    python .claude/setup_airtable.py

Prerequisites:
    1. Copy .claude/.env.example to .claude/.env
    2. Fill in your API keys
    3. Make sure your Airtable PAT has 'schema.bases:write' scope
"""

import sys
from pathlib import Path

# Add project root to path so we can import the tools package
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools import config
from tools.airtable import create_ugc_table


def main():
    print("\n" + "=" * 55)
    print("  Creative Content Engine - Airtable Setup")
    print("=" * 55 + "\n")

    # Check .env exists
    if not config.ENV_PATH.exists():
        print("  .claude/.env file not found!")
        print()
        print("  Quick setup:")
        print(f"    1. Copy .claude/.env.example to .claude/.env")
        print(f"    2. Fill in your API keys")
        print(f"    3. Run this script again")
        sys.exit(1)

    # Check credentials
    missing = config.check_credentials()
    if missing:
        sys.exit(1)

    # Create the table
    try:
        result = create_ugc_table()

        print("\n" + "-" * 55)
        if result.get("exists"):
            print("  Table 'Content' already exists in your base.")
        else:
            print("  Table 'Content' created successfully!")
        print()
        print(f"  Open your base: https://airtable.com/{config.AIRTABLE_BASE_ID}")
        print("-" * 55)
        print()
        print("  Next steps:")
        print("    1. Open the link above to see your new table")
        print("    2. Place product reference images in the references/inputs/ folder")
        print("    3. Ask Claude Code to generate content!")
        print()

    except Exception as e:
        print(f"\n  Error: {e}")
        print()
        print("  Troubleshooting:")
        print("    - Make sure AIRTABLE_BASE_ID is correct (starts with 'app')")
        print("    - Make sure your PAT has 'schema.bases:write' scope")
        print("    - Check that you have access to the base")
        sys.exit(1)


if __name__ == "__main__":
    main()
