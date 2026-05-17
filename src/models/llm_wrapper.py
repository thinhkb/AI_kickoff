"""
Lightweight LLM wrapper for controlled fallback tasks.
"""
from typing import Optional, Dict, Any
from src.utils.logging_utils import logger
from configs.model_config import LLM_API_BASE, LLM_MAX_TOKENS, LLM_TEMPERATURE


class LLMWrapper:
    """Wrapper for LLM API (vLLM/SGLang compatible)."""

    def __init__(self, api_base: str = None, model: str = None):
        self.api_base = api_base or LLM_API_BASE
        self.model = model
        self.client = None

    def _ensure_client(self):
        if self.client is not None:
            return
        try:
            from openai import OpenAI
            self.client = OpenAI(base_url=self.api_base, api_key="EMPTY")
            logger.info(f"Connected to LLM at {self.api_base}")
        except ImportError:
            logger.warning("openai package not installed, LLM unavailable")
        except Exception as e:
            logger.warning(f"LLM connection failed: {e}")

    def generate(self, prompt: str, max_tokens: int = LLM_MAX_TOKENS,
                 temperature: float = LLM_TEMPERATURE) -> Optional[str]:
        """Generate text from a prompt."""
        self._ensure_client()
        if not self.client:
            return None
        try:
            resp = self.client.chat.completions.create(
                model=self.model or "default",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens, temperature=temperature,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"LLM generation failed: {e}")
            return None
