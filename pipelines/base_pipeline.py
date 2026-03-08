"""
Base Pipeline
=============
Abstract base class and shared utilities for all RAG pipelines.
"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Unified response object returned by all pipeline implementations."""
    question: str
    answer: str
    retrieved_contexts: list[str]
    retrieved_metadata: list[dict]
    pipeline_name: str
    latency_seconds: float = 0.0
    retrieval_scores: list[float] = field(default_factory=list)
    reasoning_trace: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "pipeline": self.pipeline_name,
            "question": self.question,
            "answer": self.answer,
            "contexts": self.retrieved_contexts,
            "retrieval_scores": self.retrieval_scores,
            "latency_s": self.latency_seconds,
            "reasoning_trace": self.reasoning_trace,
            "metadata": self.metadata,
        }

    def print_summary(self) -> None:
        print(f"\n{'='*70}")
        print(f"Pipeline : {self.pipeline_name}")
        print(f"Question : {self.question}")
        print(f"{'='*70}")
        print(f"Answer:\n{self.answer}")
        print(f"\nRetrieved {len(self.retrieved_contexts)} context(s) "
              f"[latency: {self.latency_seconds:.2f}s]")
        if self.reasoning_trace:
            print("\nReasoning trace:")
            for step in self.reasoning_trace:
                print(f"  > {step}")
        print("="*70)


class BasePipeline(ABC):
    """
    Abstract base for all RAG pipeline implementations.

    Subclasses must implement:
        - build_index(dataset_path)
        - query(question, **kwargs) -> RAGResponse
    """

    PIPELINE_NAME: str = "base"

    def __init__(self, config_path: str | Path = "config/config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._llm_client = None
        self._setup_logging()

    def _load_config(self) -> dict:
        if self.config_path.exists():
            with open(self.config_path, encoding="utf-8") as f:
                return yaml.safe_load(f)
        logger.warning("Config not found at %s, using defaults", self.config_path)
        return {}

    def _setup_logging(self) -> None:
        log_cfg = self.config.get("logging", {})
        logging.basicConfig(
            level=getattr(logging, log_cfg.get("level", "INFO")),
            format=log_cfg.get("format", "%(asctime)s | %(levelname)s | %(message)s"),
        )

    def get_llm_client(self):
        """Lazy-load the LLM client (OpenAI or Ollama)."""
        if self._llm_client is not None:
            return self._llm_client

        llm_cfg = self.config.get("llm", {})
        provider = llm_cfg.get("provider", "openai")

        if provider == "openai":
            try:
                from openai import OpenAI
                api_key = llm_cfg.get("api_key") or os.getenv("OPENAI_API_KEY")
                self._llm_client = OpenAI(api_key=api_key)
                logger.info("OpenAI client initialized (model: %s)", llm_cfg.get("model"))
            except ImportError:
                raise ImportError("Install openai: pip install openai")

        elif provider == "ollama":
            try:
                from openai import OpenAI
                base_url = llm_cfg.get("ollama_base_url", "http://localhost:11434/v1")
                self._llm_client = OpenAI(base_url=base_url, api_key="ollama")
                logger.info("Ollama client initialized (model: %s)", llm_cfg.get("ollama_model"))
            except ImportError:
                raise ImportError("Install openai: pip install openai")

        return self._llm_client

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Call the configured LLM with a prompt."""
        client = self.get_llm_client()
        llm_cfg = self.config.get("llm", {})
        provider = llm_cfg.get("provider", "openai")

        model = (
            llm_cfg.get("ollama_model", "llama3")
            if provider == "ollama"
            else llm_cfg.get("model", "gpt-4o-mini")
        )
        temperature = temperature if temperature is not None else llm_cfg.get("temperature", 0.1)
        max_tokens = max_tokens or llm_cfg.get("max_tokens", 1024)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error("LLM generation failed: %s", e)
            raise

    @abstractmethod
    def build_index(self, dataset_path: str | Path) -> None:
        """Load documents, chunk them, and build the retrieval index."""
        ...

    @abstractmethod
    def query(self, question: str, **kwargs) -> RAGResponse:
        """Answer a question using the RAG pipeline."""
        ...

    def timed_query(self, question: str, **kwargs) -> RAGResponse:
        """Wrapper that adds latency measurement."""
        start = time.perf_counter()
        response = self.query(question, **kwargs)
        response.latency_seconds = time.perf_counter() - start
        return response

    @property
    def retrieval_config(self) -> dict:
        return self.config.get("retrieval", {})
