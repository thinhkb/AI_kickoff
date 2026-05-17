"""
Build the API registry from the API configuration file.
Exports: api_registry.jsonl + alias_dictionary.json
"""
import sys
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.paths import (
    API_CONFIG_FILE, API_REGISTRY_FILE, ALIAS_DICTIONARY_FILE, ensure_dirs,
)
from src.api.api_catalog_loader import load_api_catalog, load_alias_dictionary
from src.utils.io_utils import write_jsonl, write_json
from src.utils.logging_utils import logger


def main():
    ensure_dirs()

    # Load API catalog
    apis = load_api_catalog(API_CONFIG_FILE)
    logger.info(f"Loaded {len(apis)} APIs")

    # Export to JSONL
    api_dicts = []
    for api in apis:
        api_dicts.append({
            "func_code": api.func_code,
            "name": api.name,
            "description": api.description,
            "example_question": api.example_question,
            "method": api.method,
            "path": api.path,
            "body_params": api.body_params,
        })
    write_jsonl(api_dicts, API_REGISTRY_FILE)
    logger.info(f"Wrote API registry to {API_REGISTRY_FILE}")

    # Load and export alias dictionary
    alias_dict = load_alias_dictionary(API_CONFIG_FILE)
    write_json(alias_dict, ALIAS_DICTIONARY_FILE)
    logger.info(f"Wrote alias dictionary with {len(alias_dict)} categories to {ALIAS_DICTIONARY_FILE}")

    # Summary
    for name, mapping in alias_dict.items():
        logger.info(f"  {name}: {len(mapping)} entries")


if __name__ == "__main__":
    main()
