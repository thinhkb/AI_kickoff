"""
Fills the selected API JSON template with normalized slot values.
"""
import copy
import json
import re
from collections import OrderedDict
from typing import Dict, Any, List, Optional

from configs.paths import EXAMPLE_DATA_FILE
from src.schemas import APIEntry
from src.utils.io_utils import read_excel_sheet
from src.utils.logging_utils import logger


class TemplateFiller:
    """Fills API JSON bodies from API schema plus calibrated sample templates."""

    def __init__(self):
        self._calibrated_templates: Optional[Dict[str, OrderedDict]] = None

    def fill(self, api: APIEntry, slots: Dict[str, Any]) -> Dict[str, Any]:
        template = self._body_template(api)
        body = OrderedDict()

        for param, default in template.items():
            if param == "isCompany" and default is True:
                body[param] = True
            elif param in slots and slots[param] is not None:
                body[param] = slots[param]
            else:
                body[param] = copy.deepcopy(default)

        self._postprocess_body(api, slots, body)
        return {"path": api.path, "body": dict(body)}

    def _body_template(self, api: APIEntry) -> OrderedDict:
        calibrated = self._load_calibrated_templates().get(api.path)
        if calibrated:
            return copy.deepcopy(calibrated)

        family = self._family_template(api.path)
        if family:
            return family

        schema = self._schema_template(api)
        if schema:
            return schema

        example_body = self._get_example_body(api)
        return OrderedDict(
            (k, self._neutral_default(k, v)) for k, v in example_body.items()
        )

    def _load_calibrated_templates(self) -> Dict[str, OrderedDict]:
        if self._calibrated_templates is not None:
            return self._calibrated_templates

        templates: Dict[str, OrderedDict] = {}
        try:
            rows = read_excel_sheet(EXAMPLE_DATA_FILE, "example_result")
        except Exception as exc:
            logger.debug(f"Could not load API calibration templates: {exc}")
            self._calibrated_templates = templates
            return templates

        for row in rows:
            if row.get("func_code") != "call_api":
                continue
            parsed = self._parse_answer(row.get("func_param"))
            if not parsed:
                continue
            path = parsed.get("path")
            body = parsed.get("body")
            if not path or not isinstance(body, dict):
                continue
            templates[path] = OrderedDict(
                (k, self._neutral_default(k, v)) for k, v in body.items()
            )

        self._calibrated_templates = templates
        return templates

    def _parse_answer(self, raw: Any) -> Optional[Dict[str, Any]]:
        if not raw:
            return None
        if isinstance(raw, dict):
            return raw
        raw_text = str(raw)
        try:
            parsed = json.loads(raw_text)
            if isinstance(parsed, dict) and isinstance(parsed.get("body"), str):
                body_text = parsed["body"].strip()
                body_text = re.sub(r",\s*}", "}", body_text)
                try:
                    parsed["body"] = json.loads(body_text)
                except json.JSONDecodeError:
                    pass
            return parsed
        except json.JSONDecodeError:
            pass

        path_match = re.search(r'"path"\s*:\s*"([^"]+)"', raw_text)
        if not path_match:
            return None
        body = OrderedDict()
        for key, value in re.findall(r'"(fromDate|toDate)"\s*:\s*"([^"]+)"', raw_text):
            body[key] = value
        for key in [
            "projectStatus", "projectList", "projectType",
            "organization", "customerList",
        ]:
            if re.search(rf'"{key}"\s*:\s*\[\s*\]', raw_text):
                body[key] = []
        return {"path": path_match.group(1), "body": body}

    def _neutral_default(self, key: str, value: Any) -> Any:
        if key in {"fromDate", "toDate"}:
            return ""
        if key in {"fromDateProject", "toDateProject", "standardComparison"}:
            return None
        if isinstance(value, list):
            return []
        if isinstance(value, dict):
            return {}
        if key in {"type", "sort"} and isinstance(value, int):
            return value
        if isinstance(value, bool):
            return value
        if value is None:
            return None
        return ""

    def _family_template(self, path: str) -> OrderedDict:
        if "/project-performance/" in path:
            return OrderedDict([
                ("fromDate", ""),
                ("toDate", ""),
                ("projectStatus", []),
                ("projectList", []),
                ("projectType", []),
                ("organization", []),
            ])

        if "/project-overview/company" in path:
            return OrderedDict([
                ("fromDate", ""),
                ("toDate", ""),
                ("projectStatus", []),
                ("projectType", []),
            ])

        if "/project-overview/organization" in path:
            return OrderedDict([
                ("fromDate", ""),
                ("toDate", ""),
                ("projectStatus", []),
                ("projectType", []),
                ("organization", []),
            ])

        return OrderedDict()

    def _schema_template(self, api: APIEntry) -> OrderedDict:
        params: OrderedDict[str, Any] = OrderedDict()
        type_by_name: Dict[str, str] = {}

        for name in self._parse_request_body_params(api):
            params[name] = self._default_for_param(name, "")

        for section in ("required_params", "optional_params"):
            for item in api.endpoint_config.get(section, []) or []:
                if not isinstance(item, dict):
                    continue
                name = self._canonical_param_name(str(item.get("name", "")).strip())
                if not name or name in {"page", "size"}:
                    continue
                type_by_name[name] = str(item.get("type", ""))
                if name not in params:
                    params[name] = self._default_for_param(name, type_by_name[name])

        return params

    def _parse_request_body_params(self, api: APIEntry) -> List[str]:
        request = api.endpoint_config.get("request", {}) or {}
        raw = request.get("body", [])
        if isinstance(raw, list):
            return [self._canonical_param_name(str(x).strip()) for x in raw if str(x).strip()]
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw.replace("'", '"'))
                if isinstance(parsed, list):
                    return [self._canonical_param_name(str(x).strip()) for x in parsed if str(x).strip()]
            except json.JSONDecodeError:
                return [
                    self._canonical_param_name(x.strip())
                    for x in re.findall(r"'([^']+)'", raw)
                    if x.strip()
                ]
        return []

    def _canonical_param_name(self, name: str) -> str:
        fixes = {
            "IsAllCustomer": "isAllCustomer",
            "isallcustomer": "isAllCustomer",
        }
        return fixes.get(name, name)

    def _default_for_param(self, name: str, type_name: str) -> Any:
        list_params = {
            "organization", "projectType", "projectStatus", "projectList",
            "customerList", "position", "level", "trainGroup", "assetGroup",
            "lcntType", "lcntOption", "lcntOptionDoing", "gtStatus",
            "hdStatus", "bidPlanType", "lcntDomainType",
        }
        if name in list_params or "List" in type_name:
            return []
        if name in {"fromDateProject", "toDateProject", "standardComparison", "sort"}:
            return None
        if "Boolean" in type_name or name.startswith("is"):
            return False
        if "Integer" in type_name or name == "type":
            return 0
        return ""

    def _postprocess_body(
        self,
        api: APIEntry,
        slots: Dict[str, Any],
        body: OrderedDict,
    ) -> None:
        if "isCompany" in body:
            if slots.get("organization"):
                if body["isCompany"] is not True:
                    body["isCompany"] = False
            elif "isCompany" in slots:
                body["isCompany"] = slots["isCompany"]

        if "isAllProject" in body and body["isAllProject"] in ("", None):
            body["isAllProject"] = True
        if "isAllCustomer" in body and body["isAllCustomer"] in ("", None):
            body["isAllCustomer"] = True

        qsort = slots.get("sort")
        if "sort" in body and qsort is not None:
            body["sort"] = qsort
        if "standardComparison" in body and "standardComparison" in slots:
            body["standardComparison"] = slots["standardComparison"]

        if "type" in body and "type" in slots:
            body["type"] = slots["type"]

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

    def format_output(self, filled: Dict[str, Any]) -> str:
        return json.dumps(filled, ensure_ascii=False, indent=2)
