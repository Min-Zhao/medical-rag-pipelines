"""
Pipeline 05: Multi-hop RAG (Iterative Chain-of-Thought Retrieval)
=================================================================
Decomposes complex questions into sub-questions and iteratively retrieves
evidence across multiple hops, building a cumulative reasoning chain.

Architecture:
  Complex Question
        ↓
  [Decompose] → Sub-question 1
        ↓
  Retrieve(SQ1) → Context_1
        ↓
  [Partial Answer 1 + Remaining Question]
        ↓
  [Decompose] → Sub-question 2
        ↓
  Retrieve(SQ2) → Context_2
        ↓
  [Partial Answer 2]
        ...
        ↓
  [Synthesize All Partial Answers] → Final Answer

Why Multi-hop RAG?
  Many CLA clinical questions require multi-step reasoning:
  - "Can sirolimus cure LAM and what happens if you stop it?"
    → Hop 1: Does sirolimus cure LAM? (no, it stabilizes)
    → Hop 2: What happens when you stop sirolimus in LAM?
  - "Which mutation determines whether trametinib or alpelisib is better?"
    → Hop 1: What mutation does trametinib target?
    → Hop 2: What mutation does alpelisib target?
    → Hop 3: How do these compare clinically?

Reference:
  Press et al. (2023) "Measuring and Narrowing the Compositionality Gap in Language Models"
  Yang et al. (2018) "HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering"

Usage:
    python pipelines/05_multihop_rag.py
    python pipelines/05_multihop_rag.py --question "What mutation is treated by trametinib and what is the response rate?"
    python pipelines/05_multihop_rag.py --max_hops 3
"""

from __future__ import annotations

import argparse
import json
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
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ReasoningHop:
    """One step in the multi-hop reasoning chain."""
    hop_number: int
    sub_question: str
    retrieved_chunks: list[dict]
    partial_answer: str
    is_sufficient: bool = False
    evidence_summary: str = ""


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert medical researcher specializing in Complex Lymphatic Anomalies.
You solve complex questions through systematic multi-step reasoning, gathering evidence iteratively.
Always be precise, cite specific facts, and acknowledge when evidence is insufficient."""

DECOMPOSE_PROMPT = """You are answering a complex medical question about Complex Lymphatic Anomalies (CLA).
Break down the question into the MOST IMPORTANT atomic sub-question that should be answered FIRST.

Full question: {question}

Information already gathered (from previous hops):
{gathered_info}

Generate the single most important sub-question to answer next.
If the question is already answerable from gathered information, output: [SUFFICIENT]

Output ONLY:
Sub-question: <your sub-question>
OR
[SUFFICIENT]"""

PARTIAL_ANSWER_PROMPT = """Given the following retrieved medical context, answer the sub-question.
Be concise and focus only on what the context reveals.

Sub-question: {sub_question}

Retrieved Context:
{context}

Partial answer (based only on the above context):"""

SUFFICIENCY_CHECK_PROMPT = """Given the original question and the evidence gathered so far,
decide if enough information has been gathered to provide a complete answer.

Original Question: {question}

Evidence Gathered:
{gathered_info}

Is the gathered evidence SUFFICIENT to fully answer the original question?
Respond with exactly one of:
[SUFFICIENT] - Yes, I can now fully answer the question
[INSUFFICIENT] - More information is needed; continue retrieval

Decision:"""

SYNTHESIS_PROMPT = """Synthesize the following evidence gathered through multi-hop retrieval
into a comprehensive, well-structured final answer.

Original Question: {question}

Evidence Chain:
{evidence_chain}

Requirements:
- Directly answer the original question
- Integrate all relevant evidence from the hops
- Use precise medical terminology
- Note any important caveats or uncertainties
- Structure the answer logically

Final Answer:"""

DIRECT_ANSWER_PROMPT = """Answer the following medical question about Complex Lymphatic Anomalies
based on the provided context.

Context:
{context}

Question: {question}

Answer:"""


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class MultiHopRAGPipeline(BasePipeline):
    """
    Iterative multi-hop RAG for complex CLA questions.

    Algorithm:
    1. Decompose: Identify the next atomic sub-question to answer
    2. Retrieve: Dense search for the sub-question
    3. Partially Answer: Generate answer from retrieved context
    4. Accumulate: Add to evidence chain
    5. Sufficient?: Check if original question can be answered
    6. Repeat (up to max_hops)
    7. Synthesize: Final answer from accumulated evidence

    Compared to basic RAG:
    ┌─────────────────────────────────────────────────────────┐
    │ Basic RAG   : 1 query → N chunks → 1 answer             │
    │ Multi-hop   : K queries → K×N chunks → K answers → 1    │
    │               each query informed by previous answers    │
    └─────────────────────────────────────────────────────────┘
    """

    PIPELINE_NAME = "MultiHop_RAG"

    def __init__(self, config_path: str | Path = "config/config.yaml"):
        super().__init__(config_path)
        self.vector_store: VectorStoreManager | None = None

    def build_index(self, dataset_path: str | Path) -> None:
        logger.info("[MultiHop-RAG] Building vector index...")
        processor = DocumentProcessor.from_config(self.config_path)
        chunks = processor.process_dataset(dataset_path)
        self.vector_store = VectorStoreManager.from_config(self.config_path)
        self.vector_store.index_chunks(chunks)
        logger.info("[MultiHop-RAG] Index ready.")

    def _load_vector_store(self) -> None:
        if self.vector_store is None:
            self.vector_store = VectorStoreManager.from_config(self.config_path)

    # -----------------------------------------------------------------------
    # Core multi-hop steps
    # -----------------------------------------------------------------------

    def decompose_question(
        self, question: str, gathered_info: str
    ) -> tuple[str | None, bool]:
        """
        Returns (sub_question, is_sufficient).
        If is_sufficient=True, sub_question is None.
        """
        prompt = DECOMPOSE_PROMPT.format(
            question=question,
            gathered_info=gathered_info if gathered_info else "Nothing gathered yet."
        )
        response = self.generate(prompt, temperature=0.1, max_tokens=120)

        if "[SUFFICIENT]" in response:
            return None, True

        match = re.search(r"Sub-question:\s*(.+)", response, re.IGNORECASE)
        if match:
            sub_q = match.group(1).strip()
            return sub_q, False

        # Fallback: use the full response as sub-question
        cleaned = response.replace("[SUFFICIENT]", "").strip()
        return cleaned if cleaned else question, False

    def retrieve_for_subquestion(self, sub_question: str, k: int = 3) -> list[dict]:
        """Retrieve chunks relevant to a sub-question."""
        results = self.vector_store.search(sub_question, k=k)
        threshold = self.retrieval_config.get("similarity_threshold", 0.25)
        filtered = [r for r in results if r["score"] >= threshold]
        return filtered or results[:1]

    def generate_partial_answer(self, sub_question: str, retrieved: list[dict]) -> str:
        """Generate a partial answer from retrieved context for a sub-question."""
        context = "\n\n".join(
            f"[Source: {r['metadata'].get('doc_id', 'unknown')}]\n{r['text']}"
            for r in retrieved
        )
        prompt = PARTIAL_ANSWER_PROMPT.format(
            sub_question=sub_question,
            context=context,
        )
        return self.generate(prompt, system_prompt=SYSTEM_PROMPT, max_tokens=300)

    def check_sufficiency(self, question: str, gathered_info: str) -> bool:
        """Check if gathered evidence is sufficient to answer the original question."""
        prompt = SUFFICIENCY_CHECK_PROMPT.format(
            question=question,
            gathered_info=gathered_info,
        )
        response = self.generate(prompt, temperature=0.0, max_tokens=30)
        return "[SUFFICIENT]" in response

    def synthesize_final_answer(
        self, question: str, hops: list[ReasoningHop]
    ) -> str:
        """Synthesize all hop evidence into a coherent final answer."""
        evidence_chain = "\n\n".join(
            f"Hop {hop.hop_number}: {hop.sub_question}\n"
            f"Evidence: {hop.partial_answer}"
            for hop in hops
        )
        prompt = SYNTHESIS_PROMPT.format(
            question=question,
            evidence_chain=evidence_chain,
        )
        return self.generate(prompt, system_prompt=SYSTEM_PROMPT, max_tokens=600)

    # -----------------------------------------------------------------------
    # Main query
    # -----------------------------------------------------------------------

    def query(
        self,
        question: str,
        k: int | None = None,
        max_hops: int | None = None,
        **kwargs,
    ) -> RAGResponse:
        self._load_vector_store()
        multihop_cfg = self.config.get("multihop", {})
        k = k or multihop_cfg.get("hop_top_k", 3)
        max_hops = max_hops or multihop_cfg.get("max_hops", 3)

        reasoning_trace: list[str] = []
        hops: list[ReasoningHop] = []
        all_retrieved: list[dict] = []
        gathered_info = ""

        reasoning_trace.append(f"Original question: {question}")
        reasoning_trace.append(f"Max hops: {max_hops}")

        for hop_num in range(1, max_hops + 1):
            reasoning_trace.append(f"\n=== HOP {hop_num}/{max_hops} ===")

            # Decompose: get next sub-question
            sub_question, is_sufficient = self.decompose_question(question, gathered_info)

            if is_sufficient:
                reasoning_trace.append(f"  [SUFFICIENT] Evidence gathered is enough to answer.")
                break

            reasoning_trace.append(f"  Sub-question: {sub_question}")

            # Retrieve for sub-question
            retrieved = self.retrieve_for_subquestion(sub_question, k=k)
            all_retrieved.extend(retrieved)
            reasoning_trace.append(
                f"  Retrieved {len(retrieved)} chunks "
                f"(top score: {retrieved[0]['score']:.3f} — {retrieved[0]['metadata'].get('doc_id', '?')})"
            )

            # Generate partial answer
            partial_answer = self.generate_partial_answer(sub_question, retrieved)
            reasoning_trace.append(f"  Partial answer: {partial_answer[:120]}...")

            hop = ReasoningHop(
                hop_number=hop_num,
                sub_question=sub_question,
                retrieved_chunks=retrieved,
                partial_answer=partial_answer,
            )
            hops.append(hop)

            # Update gathered info
            gathered_info += f"\nHop {hop_num} – Q: {sub_question}\nA: {partial_answer}\n"

            # Check sufficiency
            if hop_num < max_hops:
                is_sufficient = self.check_sufficiency(question, gathered_info)
                hop.is_sufficient = is_sufficient
                if is_sufficient:
                    reasoning_trace.append(f"  [SUFFICIENT] after hop {hop_num}. Stopping early.")
                    break

        reasoning_trace.append(f"\nCompleted {len(hops)} hops. Synthesizing final answer...")

        # Synthesize
        if len(hops) == 1:
            # Single hop: use partial answer directly or refine
            final_answer = self.synthesize_final_answer(question, hops)
        else:
            final_answer = self.synthesize_final_answer(question, hops)

        reasoning_trace.append("Synthesis complete.")

        # Deduplicate retrieved contexts
        seen_texts: set[str] = set()
        unique_retrieved = []
        for r in all_retrieved:
            key = r["text"][:100]
            if key not in seen_texts:
                seen_texts.add(key)
                unique_retrieved.append(r)

        return RAGResponse(
            question=question,
            answer=final_answer,
            retrieved_contexts=[r["text"] for r in unique_retrieved],
            retrieved_metadata=[r["metadata"] for r in unique_retrieved],
            retrieval_scores=[r["score"] for r in unique_retrieved],
            pipeline_name=self.PIPELINE_NAME,
            reasoning_trace=reasoning_trace,
            metadata={
                "num_hops": len(hops),
                "hops": [
                    {
                        "hop": h.hop_number,
                        "sub_question": h.sub_question,
                        "partial_answer": h.partial_answer,
                        "sources": [r["metadata"].get("doc_id") for r in h.retrieved_chunks],
                    }
                    for h in hops
                ],
            },
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Multi-hop RAG Pipeline for CLA Q&A")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--dataset", default="data/pseudo_dataset/cla_documents.json")
    parser.add_argument("--question", default=None)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--max_hops", type=int, default=3)
    parser.add_argument("--show_hops", action="store_true", help="Show hop details")
    args = parser.parse_args()

    pipeline = MultiHopRAGPipeline(config_path=args.config)
    pipeline.build_index(args.dataset)

    demo_questions = [
        "What genetic mutation causes KLA and which drug targets it, and what is the response rate?",
        "Can sirolimus cure LAM or does it only stabilize it, and what happens when you stop it?",
        "How does Noonan syndrome cause central conducting lymphatic anomaly and how is it treated?",
    ]

    questions = [args.question] if args.question else demo_questions

    print("\n=== Multi-hop RAG Pipeline Demo ===\n")
    for question in questions[:2]:
        response = pipeline.timed_query(question, k=args.top_k, max_hops=args.max_hops)
        response.print_summary()

        if args.show_hops and response.metadata.get("hops"):
            print("\nReasoning Hops:")
            for hop in response.metadata["hops"]:
                print(f"\n  Hop {hop['hop']}:")
                print(f"    Sub-Q  : {hop['sub_question']}")
                print(f"    Sources: {hop['sources']}")
                print(f"    Answer : {hop['partial_answer'][:150]}...")

        print(f"\n  Total hops: {response.metadata.get('num_hops', 'N/A')}")
        print(f"  Total chunks retrieved: {len(response.retrieved_contexts)}")
        print(f"  Latency: {response.latency_seconds:.2f}s")


if __name__ == "__main__":
    main()
