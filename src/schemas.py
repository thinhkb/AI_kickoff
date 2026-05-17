"""
Data schemas for input/output.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import json


@dataclass
class InputSample:
    """A single input question from the dataset."""
    id: int
    question: str
    note: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict) -> "InputSample":
        return cls(
            id=int(d.get("id", 0)),
            question=str(d.get("fun_question", "")),
            note=d.get("note"),
        )


@dataclass
class PredictionResult:
    """Output prediction for a single question."""
    id: int
    function_code: str
    function_answer: str
    time_response: float

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "function_code": self.function_code,
            "function_answer": self.function_answer,
            "time_response": round(self.time_response, 2),
        }


@dataclass
class APIEntry:
    """A single API from the registry."""
    func_code: str
    name: str
    description: str
    example_question: str
    endpoint_config: Dict[str, Any]
    # Parsed fields
    method: str = ""
    path: str = ""
    body_params: List[str] = field(default_factory=list)
    query_params: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "APIEntry":
        endpoint_raw = d.get("Endpoint config", "{}")
        try:
            endpoint = json.loads(endpoint_raw) if isinstance(endpoint_raw, str) else endpoint_raw
        except (json.JSONDecodeError, TypeError):
            endpoint = {}

        request_info = endpoint.get("request", {})
        method = request_info.get("method", "POST")
        path = request_info.get("path", "")

        # Parse body params
        body_raw = request_info.get("body", "[]")
        body_params = []
        if isinstance(body_raw, str):
            try:
                body_params = json.loads(body_raw.replace("'", '"'))
            except (json.JSONDecodeError, TypeError):
                # Try to extract param names from string
                import re
                body_params = re.findall(r"'(\w+)'", body_raw)
        elif isinstance(body_raw, list):
            body_params = body_raw

        return cls(
            func_code=d.get("func_code", ""),
            name=d.get("name", ""),
            description=d.get("description", ""),
            example_question=d.get("Example question", ""),
            endpoint_config=endpoint,
            method=method,
            path=path,
            body_params=body_params,
        )


@dataclass
class DocumentChunk:
    """A chunk from a parsed PDF document."""
    doc_id: str
    page: int
    chunk_id: int
    text: str
    heading: str = ""
    chunk_type: str = "text"  # text, table, formula
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "page": self.page,
            "chunk_id": self.chunk_id,
            "text": self.text,
            "heading": self.heading,
            "chunk_type": self.chunk_type,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DocumentChunk":
        return cls(
            doc_id=d["doc_id"],
            page=d["page"],
            chunk_id=d["chunk_id"],
            text=d["text"],
            heading=d.get("heading", ""),
            chunk_type=d.get("chunk_type", "text"),
            metadata=d.get("metadata", {}),
        )
