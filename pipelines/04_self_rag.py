"""
Pipeline 04: Self-RAG (Self-Reflective Retrieval-Augmented Generation)
=======================================================================
An adaptive pipeline where the LLM judges its own retrieval and generation
at each step using special reflection tokens.

Architecture:
  Query
    ↓
  [RETRIEVE?] – LLM decides if retrieval is needed
    ↓ (if YES)
  Vector Retrieval
    ↓
  For each retrieved chunk:
    [ISREL?]  – Is this chunk relevant?
    [ISSUP?]  – Does the chunk support the answer?
    ↓
  Filter to relevant, supporting chunks
    ↓
  Generate answer segment
    ↓
  [ISUSE?]  – Is the generated answer useful?
    ↓
  Aggregate final answer

Reference:
  Asai et al. (2023) "Self-RAG: Learning to Retrieve, Generate, and Critique"
  https://arxiv.org/abs/2310.11511

Why Self-RAG for CLA?
  Medical Q&A often involves complex, multi-faceted questions. Self-RAG's
  adaptive retrieval avoids:
  - Retrieving when not needed (factoid questions with known answers)
  - Using irrelevant or misleading context
  - Generating unsupported hallucinations

Usage:
    python pipelines/04_self_rag.py
    python pipelines/04_self_rag.py --question "What is LAM?"
    python pipelines/04_self_rag.py --question "..." --max_iter 3
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipelines.base_pipeline import BasePipeline, RAGResponse
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reflection token definitions (adapted from Asai et al.)
# ---------------------------------------------------------------------------

class ReflectionTokens:
    RETRIEVE_YES = "[Retrieve]"
    RETRIEVE_NO = "[No Retrieve]"
    ISREL_RELEVANT = "[Relevant]"
    ISREL_IRRELEVANT = "[Irrelevant]"
    ISSUP_FULL = "[Fully supported]"
    ISSUP_PARTIAL = "[Partially supported]"
    ISSUP_NONE = "[No support]"
    ISUSE_5 = "[Utility:5]"
    ISUSE_4 = "[Utility:4]"
    ISUSE_3 = "[Utility:3]"
    ISUSE_2 = "[Utility:2]"
    ISUSE_1 = "[Utility:1]"


@dataclass
class ReflectionResult:
    """Stores all reflection judgments for a single retrieval + generation step."""
    chunk_text: str
    chunk_metadata: dict
    chunk_score: float
    is_relevant: bool = False
    relevance_reasoning: str = ""
    is_supported: str = "none"  # "full" | "partial" | "none"
    support_reasoning: str = ""
    generated_segment: str = ""
    utility_score: int = 0


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a medical AI assistant for Complex Lymphatic Anomalies (CLA) with
self-reflective capabilities. You carefully evaluate the relevance and supportiveness of
retrieved evidence before generating answers. Be precise and honest about uncertainty."""

RETRIEVE_DECISION_PROMPT = """Given the following medical question about Complex Lymphatic Anomalies,
decide whether you need to retrieve information from the medical literature.

Question: {question}

Options:
- [Retrieve]: Retrieval is needed because this question requires specific facts, statistics,
  drug doses, or clinical details that may not be in general knowledge.
- [No Retrieve]: This is a very general question that can be answered from common medical
  knowledge without specific document retrieval.

Respond with ONLY one of these tokens followed by a one-line reason:
[Retrieve] or [No Retrieve]"""

RELEVANCE_PROMPT = """Evaluate whether the following passage is RELEVANT to answering the question.

Question: {question}

Passage:
{passage}

Is this passage relevant to answering the question?
- [Relevant]: The passage contains information that directly helps answer the question.
- [Irrelevant]: The passage is about a different topic or does not help answer the question.

Respond with ONLY one token ([Relevant] or [Irrelevant]) followed by a brief reason (max 30 words)."""

GENERATION_WITH_CONTEXT_PROMPT = """Use ONLY the following relevant passage to generate a partial answer
to the question. Be specific and cite facts from the passage.

Passage:
{passage}

Question: {question}

Partial answer based on this passage:"""

SUPPORT_PROMPT = """Given the passage and the generated answer segment, assess whether the passage
FULLY SUPPORTS, PARTIALLY SUPPORTS, or provides NO SUPPORT for the answer.

Passage:
{passage}

Generated Answer Segment:
{answer_segment}

Evaluate:
- [Fully supported]: Every claim in the answer segment is directly stated in the passage.
- [Partially supported]: Some claims are supported, but others go beyond what the passage says.
- [No support]: The answer contradicts or is unrelated to the passage.

Respond with ONLY one token and a brief reason (max 30 words)."""

UTILITY_PROMPT = """Rate the utility of this answer for addressing the question on a scale of 1-5.

Question: {question}
Answer: {answer}

Rating scale:
[Utility:5] = Complete, accurate, and directly addresses the question
[Utility:4] = Mostly complete and accurate with minor gaps
[Utility:3] = Partially answers the question
[Utility:2] = Addresses the question tangentially
[Utility:1] = Does not address the question

Respond with ONLY one token ([Utility:1] through [Utility:5])."""

SYNTHESIS_PROMPT = """Synthesize the following partial answer segments into a comprehensive,
coherent final answer. Remove redundancy and ensure logical flow.

Partial Segments:
{segments}

Question: {question}

Final synthesized answer:"""

NO_RETRIEVAL_PROMPT = """Answer the following medical question about Complex Lymphatic Anomalies
from your medical knowledge. Be clear about any uncertainty.

Question: {question}

Answer:"""


# ---------------------------------------------------------------------------
# Self-RAG Pipeline
# ---------------------------------------------------------------------------

class SelfRAGPipeline(BasePipeline):
    """
    Self-Reflective RAG implementation for CLA medical Q&A.

    The pipeline uses 4 reflection checkpoints:
    1. RETRIEVE? – adaptive retrieval decision
    2. ISREL?   – relevance filtering per chunk
    3. ISSUP?   – support verification per generated segment
    4. ISUSE?   – utility scoring of final answer
    """

    PIPELINE_NAME = "Self_RAG"

    def __init__(self, config_path: str | Path = "config/config.yaml"):
        super().__init__(config_path)
        self.vector_store: VectorStoreManager | None = None

    def build_index(self, dataset_path: str | Path) -> None:
        logger.info("[Self-RAG] Building vector index...")
        processor = DocumentProcessor.from_config(self.config_path)
        chunks = processor.process_dataset(dataset_path)
        self.vector_store = VectorStoreManager.from_config(self.config_path)
        self.vector_store.index_chunks(chunks)
        logger.info("[Self-RAG] Index ready.")

    def _load_vector_store(self) -> None:
        if self.vector_store is None:
            self.vector_store = VectorStoreManager.from_config(self.config_path)

    # -----------------------------------------------------------------------
    # Reflection steps
    # -----------------------------------------------------------------------

    def decide_retrieval(self, question: str) -> tuple[bool, str]:
        """Step 1: Should we retrieve? Returns (retrieve_bool, reasoning)."""
        prompt = RETRIEVE_DECISION_PROMPT.format(question=question)
        response = self.generate(prompt, temperature=0.0, max_tokens=80)
        should_retrieve = ReflectionTokens.RETRIEVE_YES in response
        return should_retrieve, response.strip()

    def assess_relevance(self, question: str, passage: str) -> tuple[bool, str]:
        """Step 2: Is this passage relevant?"""
        prompt = RELEVANCE_PROMPT.format(question=question, passage=passage)
        response = self.generate(prompt, temperature=0.0, max_tokens=80)
        is_relevant = ReflectionTokens.ISREL_RELEVANT in response
        return is_relevant, response.strip()

    def generate_with_context(self, question: str, passage: str) -> str:
        """Generate an answer segment conditioned on a single passage."""
        prompt = GENERATION_WITH_CONTEXT_PROMPT.format(passage=passage, question=question)
        return self.generate(prompt, system_prompt=SYSTEM_PROMPT, max_tokens=256)

    def assess_support(self, passage: str, answer_segment: str) -> tuple[str, str]:
        """Step 3: Is the answer supported by the passage? Returns (level, reasoning)."""
        prompt = SUPPORT_PROMPT.format(passage=passage, answer_segment=answer_segment)
        response = self.generate(prompt, temperature=0.0, max_tokens=80)
        if ReflectionTokens.ISSUP_FULL in response:
            level = "full"
        elif ReflectionTokens.ISSUP_PARTIAL in response:
            level = "partial"
        else:
            level = "none"
        return level, response.strip()

    def assess_utility(self, question: str, answer: str) -> int:
        """Step 4: Utility score (1-5) for the generated answer."""
        prompt = UTILITY_PROMPT.format(question=question, answer=answer)
        response = self.generate(prompt, temperature=0.0, max_tokens=30)
        match = re.search(r'\[Utility:(\d)\]', response)
        return int(match.group(1)) if match else 3

    # -----------------------------------------------------------------------
    # Main query
    # -----------------------------------------------------------------------

    def query(
        self,
        question: str,
        k: int | None = None,
        max_iterations: int | None = None,
        **kwargs,
    ) -> RAGResponse:
        self._load_vector_store()
        self_rag_cfg = self.config.get("self_rag", {})
        k = k or self.retrieval_config.get("top_k", 5)
        max_iter = max_iterations or self_rag_cfg.get("max_iterations", 3)
        relevance_threshold = self_rag_cfg.get("relevance_threshold", 0.7)
        support_threshold = self_rag_cfg.get("support_threshold", 0.6)

        reasoning_trace: list[str] = []
        all_results: list[ReflectionResult] = []

        # Step 1: Decide retrieval
        should_retrieve, retrieve_reason = self.decide_retrieval(question)
        reasoning_trace.append(f"RETRIEVE? {'YES' if should_retrieve else 'NO'} — {retrieve_reason[:80]}")

        if not should_retrieve:
            answer = self.generate(
                NO_RETRIEVAL_PROMPT.format(question=question),
                system_prompt=SYSTEM_PROMPT,
            )
            utility = self.assess_utility(question, answer)
            reasoning_trace.append(f"No retrieval. Generated direct answer. Utility={utility}/5")
            return RAGResponse(
                question=question,
                answer=answer,
                retrieved_contexts=[],
                retrieved_metadata=[],
                retrieval_scores=[],
                pipeline_name=self.PIPELINE_NAME,
                reasoning_trace=reasoning_trace,
                metadata={"retrieval_used": False, "utility_score": utility},
            )

        # Step 2: Retrieve
        retrieved = self.vector_store.search(question, k=k)
        reasoning_trace.append(f"Retrieved {len(retrieved)} candidate chunks")

        # Step 3: For each chunk – assess relevance, generate, assess support
        for i, result_dict in enumerate(retrieved[:max_iter]):
            reasoning_trace.append(f"\n--- Chunk {i+1} (score={result_dict['score']:.3f}) ---")

            is_relevant, rel_reason = self.assess_relevance(question, result_dict["text"])
            reasoning_trace.append(f"  ISREL? {'RELEVANT' if is_relevant else 'IRRELEVANT'}: {rel_reason[:60]}")

            reflection = ReflectionResult(
                chunk_text=result_dict["text"],
                chunk_metadata=result_dict["metadata"],
                chunk_score=result_dict["score"],
                is_relevant=is_relevant,
                relevance_reasoning=rel_reason,
            )

            if is_relevant:
                segment = self.generate_with_context(question, result_dict["text"])
                support_level, sup_reason = self.assess_support(result_dict["text"], segment)
                reflection.generated_segment = segment
                reflection.is_supported = support_level
                reflection.support_reasoning = sup_reason
                reasoning_trace.append(f"  ISSUP? {support_level.upper()}: {sup_reason[:60]}")

            all_results.append(reflection)

        # Step 4: Filter to relevant + supported segments
        good_results = [
            r for r in all_results
            if r.is_relevant and r.is_supported in ("full", "partial")
        ]
        reasoning_trace.append(f"\nKept {len(good_results)}/{len(all_results)} chunks after reflection filtering")

        if not good_results:
            # Fall back to using all relevant chunks
            good_results = [r for r in all_results if r.is_relevant] or all_results[:1]
            reasoning_trace.append("Fallback: using all retrieved chunks (no fully-supported segments found)")

        # Step 5: Synthesize final answer
        if len(good_results) == 1:
            final_answer = good_results[0].generated_segment or self.generate(
                GENERATION_WITH_CONTEXT_PROMPT.format(
                    passage=good_results[0].chunk_text, question=question
                ),
                system_prompt=SYSTEM_PROMPT,
            )
        else:
            segments_text = "\n\n".join(
                f"[Segment {i+1}]:\n{r.generated_segment}"
                for i, r in enumerate(good_results)
                if r.generated_segment
            )
            if segments_text:
                synthesis_prompt = SYNTHESIS_PROMPT.format(
                    segments=segments_text, question=question
                )
                final_answer = self.generate(synthesis_prompt, system_prompt=SYSTEM_PROMPT)
            else:
                # Generate directly from filtered chunks
                context = "\n\n".join(r.chunk_text for r in good_results)
                final_answer = self.generate(
                    f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
                    system_prompt=SYSTEM_PROMPT,
                )

        # Step 6: Assess final utility
        utility = self.assess_utility(question, final_answer)
        reasoning_trace.append(f"\nISUSE? Utility={utility}/5")

        if utility < 3:
            reasoning_trace.append("Low utility. Augmenting with broader context...")
            broader_context = "\n\n".join(r.chunk_text for r in all_results[:3])
            final_answer = self.generate(
                f"Using all available context:\n{broader_context}\n\nQuestion: {question}\n\nAnswer:",
                system_prompt=SYSTEM_PROMPT,
            )
            utility = self.assess_utility(question, final_answer)
            reasoning_trace.append(f"Revised utility: {utility}/5")

        return RAGResponse(
            question=question,
            answer=final_answer,
            retrieved_contexts=[r.chunk_text for r in good_results],
            retrieved_metadata=[r.chunk_metadata for r in good_results],
            retrieval_scores=[r.chunk_score for r in good_results],
            pipeline_name=self.PIPELINE_NAME,
            reasoning_trace=reasoning_trace,
            metadata={
                "utility_score": utility,
                "total_chunks_retrieved": len(retrieved),
                "chunks_after_filtering": len(good_results),
                "relevance_details": [
                    {
                        "relevant": r.is_relevant,
                        "support": r.is_supported,
                        "score": r.chunk_score,
                    }
                    for r in all_results
                ],
            },
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Self-RAG Pipeline for CLA Q&A")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--dataset", default="data/pseudo_dataset/cla_documents.json")
    parser.add_argument("--question", default=None)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_iter", type=int, default=3)
    parser.add_argument("--verbose", action="store_true", help="Show reflection trace")
    args = parser.parse_args()

    pipeline = SelfRAGPipeline(config_path=args.config)
    pipeline.build_index(args.dataset)

    demo_questions = [
        "What is sirolimus and how does it work for CLA?",
        "How is chylothorax diagnosed in patients with GSD?",
        "What are the side effects of sirolimus in children?",
        "What is the difference between GLA and GSD?",
    ]

    questions = [args.question] if args.question else demo_questions

    print("\n=== Self-RAG Pipeline Demo ===\n")
    for question in questions[:3]:
        response = pipeline.timed_query(question, k=args.top_k, max_iterations=args.max_iter)
        response.print_summary()
        if args.verbose and response.reasoning_trace:
            print("\nDetailed Reflection Trace:")
            for step in response.reasoning_trace:
                print(f"  {step}")
        meta = response.metadata
        print(f"\n  Chunks retrieved: {meta.get('total_chunks_retrieved', 'N/A')}")
        print(f"  After filtering: {meta.get('chunks_after_filtering', 'N/A')}")
        print(f"  Utility score: {meta.get('utility_score', 'N/A')}/5")


if __name__ == "__main__":
    main()
