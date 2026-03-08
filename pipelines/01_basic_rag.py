"""
Pipeline 01: Basic RAG
======================
Standard dense retrieval + LLM generation pipeline.

Architecture:
  Query → Embed → Vector Search → Top-K Chunks → Prompt → LLM → Answer

This is the baseline implementation against which all advanced pipelines
are benchmarked.

Usage:
    python pipelines/01_basic_rag.py
    python pipelines/01_basic_rag.py --question "What is sirolimus used for in CLA?"
    python pipelines/01_basic_rag.py --eval  # run full evaluation
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running from any directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipelines.base_pipeline import BasePipeline, RAGResponse
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStoreManager
from src.evaluation import RAGEvaluator, RAGSample

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System & RAG prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a specialized medical AI assistant focused on Complex Lymphatic Anomalies (CLA),
including Gorham-Stout Disease (GSD), Generalized Lymphatic Anomaly (GLA), Kaposiform
Lymphangiomatosis (KLA), Central Conducting Lymphatic Anomaly (CCLA), and
Lymphangioleiomyomatosis (LAM).

Guidelines:
- Answer only from the provided context. Do not hallucinate.
- If the context does not contain enough information, say so clearly.
- Use precise medical terminology appropriate for the question's audience.
- Cite the source document when the information is very specific (e.g., [Source: doc_002]).
- For clinical decisions, always recommend consulting a specialist."""

RAG_PROMPT_TEMPLATE = """Use the following retrieved medical literature to answer the question.

Retrieved Context:
{context}

Question: {question}

Provide a comprehensive, accurate answer based solely on the context above.
Answer:"""


class BasicRAGPipeline(BasePipeline):
    """
    Baseline RAG: embed query → cosine similarity search → LLM generation.

    Steps:
    1. Encode query with the same embedding model used during indexing
    2. Retrieve top-k most similar chunks from ChromaDB / FAISS
    3. Optionally filter by similarity threshold
    4. Inject retrieved text into a structured prompt
    5. Generate answer with the configured LLM
    """

    PIPELINE_NAME = "BasicRAG"

    def __init__(self, config_path: str | Path = "config/config.yaml"):
        super().__init__(config_path)
        self.vector_store: VectorStoreManager | None = None

    def build_index(self, dataset_path: str | Path) -> None:
        """
        Load the CLA dataset, chunk documents, embed, and store in vector DB.
        Idempotent: skips already-indexed chunks.
        """
        logger.info("[BasicRAG] Building index from: %s", dataset_path)

        processor = DocumentProcessor.from_config(self.config_path)
        chunks = processor.process_dataset(dataset_path)

        self.vector_store = VectorStoreManager.from_config(self.config_path)
        self.vector_store.index_chunks(chunks)

        logger.info("[BasicRAG] Index ready: %d total chunks", len(chunks))

    def _load_vector_store(self) -> None:
        """Attach to an existing vector store (without re-indexing)."""
        if self.vector_store is None:
            self.vector_store = VectorStoreManager.from_config(self.config_path)

    def retrieve(self, question: str, k: int | None = None) -> list[dict]:
        """Retrieve top-k chunks for a question."""
        self._load_vector_store()
        k = k or self.retrieval_config.get("top_k", 5)
        threshold = self.retrieval_config.get("similarity_threshold", 0.3)

        results = self.vector_store.search(question, k=k)

        # Filter by similarity threshold
        filtered = [r for r in results if r["score"] >= threshold]
        if not filtered and results:
            logger.warning(
                "No results above threshold %.2f; returning best result", threshold
            )
            filtered = results[:1]

        return filtered

    def format_context(self, retrieved: list[dict]) -> str:
        """Format retrieved chunks as numbered context blocks."""
        blocks = []
        for i, r in enumerate(retrieved, 1):
            title = r["metadata"].get("title", "Unknown")
            year = r["metadata"].get("year", "")
            source = r["metadata"].get("doc_id", "")
            blocks.append(
                f"[{i}] Source: {title} ({year}) [{source}]\n{r['text']}"
            )
        return "\n\n".join(blocks)

    def query(self, question: str, k: int | None = None, **kwargs) -> RAGResponse:
        """End-to-end: retrieve + generate."""
        retrieved = self.retrieve(question, k=k)
        context = self.format_context(retrieved)

        prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)
        answer = self.generate(prompt, system_prompt=SYSTEM_PROMPT)

        return RAGResponse(
            question=question,
            answer=answer,
            retrieved_contexts=[r["text"] for r in retrieved],
            retrieved_metadata=[r["metadata"] for r in retrieved],
            retrieval_scores=[r["score"] for r in retrieved],
            pipeline_name=self.PIPELINE_NAME,
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Basic RAG Pipeline for CLA Q&A")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--dataset", default="data/pseudo_dataset/cla_documents.json")
    parser.add_argument("--eval_questions", default="data/pseudo_dataset/eval_questions.json")
    parser.add_argument("--question", default=None, help="Single question to answer")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--eval", action="store_true", help="Run full evaluation")
    parser.add_argument("--output", default="results/basic_rag_eval.json")
    args = parser.parse_args()

    pipeline = BasicRAGPipeline(config_path=args.config)
    pipeline.build_index(args.dataset)

    if args.question:
        response = pipeline.timed_query(args.question, k=args.top_k)
        response.print_summary()
        return

    # Demo questions
    demo_questions = [
        "What is Gorham-Stout disease and how does it cause bone loss?",
        "What is the recommended sirolimus dose for pediatric CLA patients?",
        "How is chylothorax managed in CLA?",
        "What genetic mutations drive kaposiform lymphangiomatosis?",
        "What serum biomarker distinguishes LAM from other CLAs?",
    ]

    if not args.eval:
        print("\n=== Basic RAG Pipeline Demo ===\n")
        for question in demo_questions[:3]:
            response = pipeline.timed_query(question, k=args.top_k)
            response.print_summary()
        return

    # Full evaluation
    from src.evaluation import RAGEvaluator, RAGSample
    import json

    with open(args.eval_questions, encoding="utf-8") as f:
        eval_data = json.load(f)
    eval_questions = eval_data.get("questions", [])

    samples = []
    print(f"\nRunning evaluation on {len(eval_questions)} questions...")
    for qa in eval_questions:
        response = pipeline.timed_query(qa["question"], k=args.top_k)
        samples.append(RAGSample(
            question=qa["question"],
            answer=response.answer,
            contexts=response.retrieved_contexts,
            reference_answer=qa.get("reference_answer", ""),
        ))
        print(f"  Q: {qa['question'][:60]}... [latency: {response.latency_seconds:.2f}s]")

    evaluator = RAGEvaluator(use_llm_judge=False)
    results = evaluator.evaluate_batch(samples)
    summary = evaluator.summarize(results)

    print("\n=== Evaluation Summary ===")
    for metric, score in summary.items():
        print(f"  {metric:<25}: {score:.4f}")

    evaluator.save_results(results, args.output)
    print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
