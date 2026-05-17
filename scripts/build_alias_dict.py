"""
Build alias dictionary from the API configuration file.
This is a subset of build_api_registry.py focused on alias normalization.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.paths import API_CONFIG_FILE, ALIAS_DICTIONARY_FILE, ensure_dirs
from src.api.api_catalog_loader import load_alias_dictionary
from src.utils.io_utils import write_json
from src.utils.logging_utils import logger


def main():
    ensure_dirs()
    alias_dict = load_alias_dictionary(API_CONFIG_FILE)
    write_json(alias_dict, ALIAS_DICTIONARY_FILE)
    logger.info(f"Alias dictionary saved with {len(alias_dict)} categories")
    for name, mapping in alias_dict.items():
        logger.info(f"  {name}: {len(mapping)} entries")


if __name__ == "__main__":
    main()
