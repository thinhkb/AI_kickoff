"""
Extracts required entities and parameter values from the question.
Uses deterministic rules first and LLM fallback only for unresolved ambiguity.
"""
import re
from typing import Dict, Any, List, Optional

from src.schemas import APIEntry
from src.preprocess.time_normalizer import extract_date_range
from src.utils.logging_utils import logger


class SlotExtractor:
    """
    Extracts slot values (parameters) from the question for a selected API.
    """

    def __init__(self, alias_dict: Dict[str, Dict[str, str]] = None):
        self.alias_dict = alias_dict or {}

    def extract(
        self,
        question: str,
        api: APIEntry,
    ) -> Dict[str, Any]:
        """
        Extract all required parameters for the API from the question.

        Returns a dict of {param_name: extracted_value}.
        """
        slots = {}

        # Extract date range (almost all APIs need fromDate/toDate)
        date_range = extract_date_range(question)
        if date_range:
            from_date, to_date = date_range
            if "fromDate" in api.body_params or any("Date" in p for p in api.body_params):
                slots["fromDate"] = from_date
                slots["toDate"] = to_date

        # Extract organization
        org = self._extract_organization(question)
        if org is not None and "organization" in api.body_params:
            slots["organization"] = org

        # Extract project type
        pt = self._extract_project_type(question)
        if pt is not None and "projectType" in api.body_params:
            slots["projectType"] = pt

        # Extract project status
        ps = self._extract_project_status(question)
        if ps is not None and "projectStatus" in api.body_params:
            slots["projectStatus"] = ps

        # Extract project list
        pl = self._extract_project_list(question)
        if pl is not None and "projectList" in api.body_params:
            slots["projectList"] = pl

        # Extract other params with alias matching
        for param in api.body_params:
            if param not in slots:
                value = self._extract_generic_param(question, param)
                if value is not None:
                    slots[param] = value

        # Set empty defaults for unextracted params
        for param in api.body_params:
            param_clean = param.strip()
            if param_clean not in slots:
                slots[param_clean] = self._default_value(param_clean)

        return slots

    def _extract_organization(self, question: str) -> Optional[List[str]]:
        """Extract organization names from the question."""
        org_aliases = self.alias_dict.get("organization", {})
        org_aliases_combined = {**org_aliases}

        # Also check orgAlias
        org_alias_extra = self.alias_dict.get("orgAlias", {})
        org_aliases_combined.update(org_alias_extra)

        found = []
        q_lower = question.lower()

        # Direct match by value (abbreviation like TTPMVT)
        for key, value in org_aliases_combined.items():
            if value and value.upper() in question.upper():
                if value not in found:
                    found.append(value)

        # Match by key (full name)
        for key, value in org_aliases_combined.items():
            if key.lower() in q_lower:
                if value not in found:
                    found.append(value)

        # Check for "cả công ty" / "toàn công ty" → empty list (all)
        if re.search(r"(cả\s+công\s+ty|toàn\s+công\s+ty|toàn\s+cty)", question, re.IGNORECASE):
            return []

        if found:
            return found
        return None

    def _extract_project_type(self, question: str) -> Optional[List[str]]:
        """Extract project type from question."""
        pt_aliases = self.alias_dict.get("projectType", {})
        found = []
        q_upper = question.upper()

        for key, value in pt_aliases.items():
            if key.upper() in q_upper or value.upper() in q_upper:
                if value not in found:
                    found.append(value)

        if found:
            return found
        return None

    def _extract_project_status(self, question: str) -> Optional[List[str]]:
        """Extract project status from question."""
        ps_aliases = self.alias_dict.get("projectStatus", {})
        found = []
        q_lower = question.lower()

        for key, value in ps_aliases.items():
            if key.lower() in q_lower:
                if value not in found:
                    found.append(value)

        if found:
            return found
        return None

    def _extract_project_list(self, question: str) -> Optional[List[str]]:
        """Extract specific project names/IDs from question."""
        project_info = self.alias_dict.get("project_info", {})

        found = []
        for name, pid in project_info.items():
            if name in question:
                found.append(str(pid))

        if found:
            return found
        return None

    def _extract_generic_param(self, question: str, param: str) -> Optional[Any]:
        """
        Try to extract a generic parameter value using alias dict.
        """
        param_clean = param.strip()
        aliases = self.alias_dict.get(param_clean, {})

        if not aliases:
            return None

        found = []
        q_lower = question.lower()

        for key, value in aliases.items():
            if key.lower() in q_lower or (value and str(value).lower() in q_lower):
                found.append(value)

        if found:
            return found if len(found) > 1 else found[0]
        return None

    def _default_value(self, param: str) -> Any:
        """Return default value for an unfilled parameter."""
        # List params get empty lists, others get empty string
        list_params = {
            "organization", "projectType", "projectStatus", "projectList",
            "position", "level", "trainGroup", "assetGroup",
            "lcntType", "lcntOption", "lcntOptionDoing", "gtStatus",
            "hdStatus", "bidPlanType", "lcntDomainType",
        }
        if param in list_params:
            return []
        return ""
