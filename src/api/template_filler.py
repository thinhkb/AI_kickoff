"""
Fills the selected API JSON template with normalized slot values.
"""
import json
import re
from typing import Dict, Any
from src.schemas import APIEntry
from src.utils.logging_utils import logger


class TemplateFiller:
    """Fills API JSON templates with extracted and normalized slot values."""

    def fill(self, api: APIEntry, slots: Dict[str, Any]) -> Dict[str, Any]:
        result = {"path": api.path, "body": {}}
        example_body = self._get_example_body(api)
        body = {}
        for param in api.body_params:
            pc = param.strip()
            if pc in slots:
                body[pc] = slots[pc]
            elif pc in example_body:
                body[pc] = self._default_for_type(example_body[pc])
            else:
                body[pc] = []
        result["body"] = body
        return result

    def _get_example_body(self, api: APIEntry) -> Dict[str, Any]:
        config = api.endpoint_config
        if not config:
            return {}
        calls = config.get("example_call", [])
        if calls and isinstance(calls, list):
            raw = calls[0].get("body", {})
            if isinstance(raw, str):
                try:
                    return json.loads(re.sub(r",\s*}", "}", raw))
                except (json.JSONDecodeError, TypeError):
                    return {}
            elif isinstance(raw, dict):
                return raw
        return {}

    def _default_for_type(self, v):
        if isinstance(v, list): return []
        if isinstance(v, dict): return {}
        if isinstance(v, str): return ""
        return ""

    def format_output(self, filled: Dict[str, Any]) -> str:
        return json.dumps(filled, ensure_ascii=False, indent=2)
