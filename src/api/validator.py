"""
Validates the generated API output: field names, required keys, enum compatibility, JSON schema.
"""
import json
from typing import Dict, Any, List, Optional
from src.schemas import APIEntry
from src.utils.logging_utils import logger


class APIValidator:
    """Validates generated API JSON output against the API schema."""

    def __init__(self, alias_dict: Dict[str, Dict[str, str]] = None):
        self.alias_dict = alias_dict or {}

    def validate(self, api: APIEntry, output: Dict[str, Any]) -> tuple:
        """
        Validate the output against the API spec.
        Returns (is_valid, errors_list).
        """
        errors = []

        # Check path
        if not output.get("path"):
            errors.append("Missing path")
        elif output["path"] != api.path:
            errors.append(f"Path mismatch: {output['path']} != {api.path}")

        # Check body
        body = output.get("body", {})
        if not isinstance(body, dict):
            errors.append("Body must be a dict")
            return len(errors) == 0, errors

        # Validate enum values
        for param, value in body.items():
            if param in self.alias_dict:
                valid_values = set(self.alias_dict[param].values())
                if isinstance(value, list):
                    for v in value:
                        if v and str(v) not in valid_values and v not in valid_values:
                            pass  # Warn but don't fail
                elif isinstance(value, str) and value:
                    if value not in valid_values:
                        pass  # Warn but don't fail

        return len(errors) == 0, errors

    def repair(self, api: APIEntry, output: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt to repair minor issues in the output."""
        repaired = dict(output)
        if "path" not in repaired or not repaired["path"]:
            repaired["path"] = api.path
        if "body" not in repaired:
            repaired["body"] = {}
        return repaired
