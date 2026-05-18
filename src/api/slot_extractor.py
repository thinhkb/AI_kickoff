"""
Extracts required entities and parameter values from the question.
Uses deterministic rules first and LLM fallback only for unresolved ambiguity.
"""
import re
import unicodedata
from typing import Dict, Any, List, Optional

from src.schemas import APIEntry
from src.preprocess.time_normalizer import extract_date_range
from rapidfuzz import fuzz, process


class SlotExtractor:
    """Extracts reusable API slot values from a Vietnamese question."""

    def __init__(self, alias_dict: Dict[str, Dict[str, str]] = None):
        self.alias_dict = alias_dict or {}

    def extract(self, question: str, api: APIEntry) -> Dict[str, Any]:
        slots: Dict[str, Any] = {}

        date_range = extract_date_range(question)
        if date_range:
            slots["fromDate"], slots["toDate"] = date_range

        org = self._extract_organization(question)
        if org is not None:
            slots["organization"] = org
            slots["isCompany"] = False
        elif self._is_company_query(question):
            slots["isCompany"] = True

        for name, extractor in [
            ("projectType", self._extract_project_type),
            ("projectStatus", self._extract_project_status),
            ("projectList", self._extract_project_list),
            ("customerList", lambda q: self._extract_alias_list(q, "customerList")),
            ("position", self._extract_position),
            ("level", self._extract_level),
        ]:
            value = extractor(question)
            if value is not None:
                slots[name] = value

        for param in [
            "lcntOption", "lcntOptionDoing", "lcntType",
            "bidPlanType", "lcntDomainType",
        ]:
            value = self._extract_alias_list(question, param)
            if value is not None:
                slots[param] = value

        self._extract_procurement_short_aliases(question, slots)

        self._extract_common_scalars(question, api, slots)

        for param in api.body_params:
            param_clean = param.strip()
            if param_clean not in slots:
                value = self._extract_generic_param(question, param_clean)
                if value is not None:
                    slots[param_clean] = value

        return slots

    def _extract_organization(self, question: str) -> Optional[List[str]]:
        aliases = dict(self.alias_dict.get("organization", {}))
        aliases.update(self.alias_dict.get("orgAlias", {}))

        found = []
        for key, value in sorted(
            aliases.items(),
            key=lambda kv: max(len(str(kv[0])), len(str(kv[1]))),
            reverse=True,
        ):
            if self._contains_alias(question, value) or self._contains_alias(question, key):
                if value and value not in found:
                    found.append(value)

        # In the public examples, TTCN appears as broad company context for
        # SOC questions rather than an organization filter.
        found = [x for x in found if x != "TTCN"]
        fuzzy = self._extract_fuzzy_org_token(question, aliases)
        if fuzzy and fuzzy != "TTCN" and fuzzy not in found:
            found.append(fuzzy)
        return found or None

    def _extract_project_type(self, question: str) -> Optional[List[str]]:
        patterns = [
            (r"(?<!\w)t\s*&\s*m(?!\w)|(?<!\w)tm(?!\w)", "T&M"),
            (r"(?<!\w)pre\s*sales?(?!\w)|(?<!\w)presales?(?!\w)|tiền\s*bán", "presales"),
            (r"(?<!\w)package(?!\w)", "package"),
            (r"(?<!\w)(?:odc|osdc)(?!\w)", "odc/osdc"),
        ]
        found = [value for pattern, value in patterns if re.search(pattern, question, re.IGNORECASE)]
        return list(dict.fromkeys(found)) or None

    def _extract_project_status(self, question: str) -> Optional[List[str]]:
        patterns = [
            (r"(?<!\w)closed?(?!\w)|đã\s*đóng|đóng", "closed"),
            (r"in[-\s]?progress|đang\s+(?:thực hiện|tiến)", "in-progress"),
            (r"(?<!\w)open(?!\w)|đang\s*mở", "open"),
            (r"(?<!\w)hold(?!\w)|tạm\s*dừng|hoàn\s*thành", "hold"),
            (r"(?:trạng thái|status)\s+(?:là\s+)?presale|trạng thái\s+tiền\s*bán", "presale"),
        ]
        found = [value for pattern, value in patterns if re.search(pattern, question, re.IGNORECASE)]
        return list(dict.fromkeys(found)) or None

    def _extract_project_list(self, question: str) -> Optional[List[Any]]:
        project_info = self.alias_dict.get("project_info", {})
        found_with_pos: List[tuple[int, Any]] = []
        for name, pid in sorted(project_info.items(), key=lambda kv: len(kv[0]), reverse=True):
            pos = self._find_alias_position(question, name)
            if pos >= 0:
                try:
                    value: Any = int(pid)
                except (TypeError, ValueError):
                    value = str(pid)
                found_with_pos.append((pos, value))
        found = []
        for _, value in sorted(found_with_pos, key=lambda item: item[0]):
            if value not in found:
                found.append(value)
        return found or None

    def _extract_position(self, question: str) -> Optional[List[str]]:
        if not re.search(r"(?<!\w)(role|position)(?!\w)|vị\s*trí", question, re.IGNORECASE):
            return None
        return self._extract_alias_list(question, "position")

    def _extract_level(self, question: str) -> Optional[List[str]]:
        if not re.search(r"(?<!\w)level(?!\w)|cấp\s*bậc", question, re.IGNORECASE):
            return None
        return self._extract_alias_list(question, "level")

    def _extract_alias_list(self, question: str, param: str) -> Optional[List[Any]]:
        aliases = self.alias_dict.get(param, {})
        if not aliases:
            return None
        found = []
        for key, value in sorted(
            aliases.items(),
            key=lambda kv: max(len(str(kv[0])), len(str(kv[1]))),
            reverse=True,
        ):
            value_text = str(value).strip()
            value_match_allowed = not value_text.isdigit()
            if self._contains_alias(question, key) or (
                value_match_allowed and self._contains_alias(question, value)
            ):
                found.append(value)
        return list(dict.fromkeys(found)) or None

    def _extract_generic_param(self, question: str, param: str) -> Optional[Any]:
        value = self._extract_alias_list(question, param)
        if value is None:
            return None
        return value if len(value) > 1 else value[0]

    def _extract_common_scalars(
        self,
        question: str,
        api: APIEntry,
        slots: Dict[str, Any],
    ) -> None:
        q = question.lower()
        path = api.path.lower()

        if "standardComparison" not in slots:
            if re.search(r"\b(above|vượt|trên|lớn hơn|cao hơn)\b", q):
                slots["standardComparison"] = 1
            elif re.search(r"\b(below|dưới|nhỏ hơn|thấp hơn)\b", q):
                slots["standardComparison"] = 2

        if "sort" not in slots:
            if "xếp hạng" in q or "ranking" in path or "theo dự án" in q:
                slots["sort"] = 2
            elif "tăng dần" in q or "asc" in q:
                slots["sort"] = 1
            elif "giảm dần" in q or "desc" in q:
                slots["sort"] = 2

        if "type" not in slots:
            if "/ra-report/" in path:
                if any(part in path for part in [
                    "/free-ru", "/get-free-effort-bu", "/get-sum-free-effort",
                ]):
                    slots["type"] = 2
                else:
                    slots["type"] = 3
            elif "/recruitment/new-organization" in path:
                slots["type"] = 4
            elif "/recruitment/" in path:
                slots["type"] = 2
            elif "/employee-info/" in path:
                slots["type"] = 5 if ("level" in q or "trung tâm" in q or "nhân sự" in q) else 3

    def _extract_procurement_short_aliases(self, question: str, slots: Dict[str, Any]) -> None:
        q = self._ascii_upper(question)
        lcnt_type_map = {
            "DTRR": "1",
            "DT RR": "1",
            "DTHC": "2",
            "DT HC": "2",
            "CHCT": "4",
            "CDT": "5",
            "MSTT": "3",
        }
        found = slots.get("lcntType", [])
        for alias, value in lcnt_type_map.items():
            if re.search(rf"(?<![A-Z0-9]){re.escape(alias)}(?![A-Z0-9])", q):
                found.append(value)
        if found:
            slots["lcntType"] = list(dict.fromkeys(found))

    def _contains_alias(self, question: str, alias: Any) -> bool:
        if alias is None:
            return False
        alias_text = str(alias).strip()
        if not alias_text:
            return False
        if re.fullmatch(r"[\w+]+", alias_text, flags=re.UNICODE):
            pattern = rf"(?<!\w){re.escape(alias_text)}(?!\w)"
            return re.search(pattern, question, re.IGNORECASE | re.UNICODE) is not None
        return alias_text.lower() in question.lower()

    def _find_alias_position(self, question: str, alias: Any) -> int:
        if alias is None:
            return -1
        alias_text = str(alias).strip()
        if not alias_text:
            return -1
        if re.fullmatch(r"[\w+]+", alias_text, flags=re.UNICODE):
            pattern = rf"(?<!\w){re.escape(alias_text)}(?!\w)"
            match = re.search(pattern, question, re.IGNORECASE | re.UNICODE)
            return match.start() if match else -1
        return question.lower().find(alias_text.lower())

    def _is_company_query(self, question: str) -> bool:
        return re.search(
            r"(?<!\w)(cả|toàn)\s+công\s+ty(?!\w)|(?<!\w)công\s+ty(?!\w)",
            question,
            re.IGNORECASE,
        ) is not None

    def _extract_fuzzy_org_token(self, question: str, aliases: Dict[str, str]) -> Optional[str]:
        values = [str(v) for v in aliases.values() if v]
        if not values:
            return None
        value_by_norm = {self._ascii_upper(v): v for v in values}
        choices = list(value_by_norm.keys())

        for token in re.findall(r"\bT[\wĐđ]{4,}\b", question, flags=re.UNICODE):
            token_norm = self._ascii_upper(token)
            if token_norm in value_by_norm:
                return value_by_norm[token_norm]
            result = process.extractOne(token_norm, choices, scorer=fuzz.WRatio)
            if result and result[1] >= 82:
                return value_by_norm[result[0]]
        return None

    def _ascii_upper(self, text: Any) -> str:
        normalized = unicodedata.normalize("NFKD", str(text))
        stripped = "".join(ch for ch in normalized if not unicodedata.combining(ch))
        return stripped.replace("Đ", "D").replace("đ", "d").upper()

    def _default_value(self, param: str) -> Any:
        list_params = {
            "organization", "projectType", "projectStatus", "projectList",
            "position", "level", "trainGroup", "assetGroup",
            "lcntType", "lcntOption", "lcntOptionDoing", "gtStatus",
            "hdStatus", "bidPlanType", "lcntDomainType", "customerList",
        }
        if param in list_params:
            return []
        return ""
