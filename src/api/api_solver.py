"""
End-to-end solver for the call_api branch.
"""
import json
from typing import Dict, Any, Optional, List
from src.schemas import APIEntry
from src.api.api_retriever import APIRetriever
from src.api.api_reranker import APIReranker
from src.api.slot_extractor import SlotExtractor
from src.api.slot_normalizer import SlotNormalizer
from src.api.template_filler import TemplateFiller
from src.api.validator import APIValidator
from src.utils.logging_utils import logger


class APISolver:
    """
    Solves call_api questions:
    1. Retrieve top API candidates
    2. Rerank to select best API
    3. Extract slots from question
    4. Normalize aliases/abbreviations
    5. Fill JSON template
    6. Validate output
    """

    def __init__(
        self,
        retriever: APIRetriever,
        reranker: Optional[APIReranker] = None,
        alias_dict: Dict[str, Dict[str, str]] = None,
    ):
        self.retriever = retriever
        self.reranker = reranker or APIReranker()
        self.alias_dict = alias_dict or {}
        self.slot_extractor = SlotExtractor(alias_dict=self.alias_dict)
        self.slot_normalizer = SlotNormalizer(alias_dict=self.alias_dict)
        self.template_filler = TemplateFiller()
        self.validator = APIValidator(alias_dict=self.alias_dict)

    def solve(self, question: str) -> str:
        """
        Solve an API question. Returns JSON string.
        """
        logger.info(f"API solver: {question[:80]}...")

        # Step 1: Retrieve candidates
        candidates = self.retriever.retrieve(question, top_k=30)
        logger.info(f"  Retrieved {len(candidates)} API candidates")

        if not candidates:
            logger.warning("  No API candidates found!")
            return json.dumps({"path": "", "body": {}}, ensure_ascii=False)

        # Step 2: Rerank
        reranked = self.reranker.rerank(question, candidates, top_k=5)
        best_api = reranked[0][0] if reranked else candidates[0][0]
        logger.info(f"  Selected API: {best_api.func_code} ({best_api.name})")

        # Step 3: Extract slots
        slots = self.slot_extractor.extract(question, best_api)
        logger.info(f"  Extracted slots: {list(slots.keys())}")

        # Step 4: Normalize
        normalized = self.slot_normalizer.normalize_slots(
            slots, best_api.body_params,
        )

        # Step 5: Fill template
        filled = self.template_filler.fill(best_api, normalized)

        # Step 6: Validate and repair
        is_valid, errors = self.validator.validate(best_api, filled)
        if not is_valid:
            logger.warning(f"  Validation errors: {errors}")
            filled = self.validator.repair(best_api, filled)

        return self.template_filler.format_output(filled)
