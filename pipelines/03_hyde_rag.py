"""
Pipeline 03: HyDE RAG (Hypothetical Document Embeddings)
=========================================================
Uses an LLM to generate a hypothetical ideal answer first, then uses
the EMBEDDING of that hypothetical answer (rather than the question)
to retrieve from the vector store.

Architecture:
  Query → LLM → Hypothetical Answer(s) → Embed → Vector Search
                                              ↑
                                   (optional: ensemble of K hypothetical docs)
                                              ↓
                             Retrieved Chunks → Prompt → LLM → Final Answer

Reference:
  Gao et al. (2022) "Precise Zero-Shot Dense Retrieval without Relevance Labels"
  https://arxiv.org/abs/2212.10496

Why HyDE?
  Embedding a SHORT, SPECIFIC question often fails to retrieve LONG, DETAILED
  medical texts because the semantic gap between a query and a passage is large.
  A hypothetical answer written in the same style as the target passages
  dramatically improves retrieval precision in specialized domains.

Usage:
    python pipelines/03_hyde_rag.py
    python pipelines/03_hyde_rag.py --question "What is the MILES trial?"
    python pipelines/03_hyde_rag.py --question "..." --n_hyp 3 --ensemble
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipelines.base_pipeline import BasePipeline, RAGResponse
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a rare disease medical expert specializing in Complex Lymphatic Anomalies.
Provide accurate, evidence-based answers."""

HYPOTHETICAL_DOC_PROMPT = """You are a medical expert writing a passage from a clinical review article
about Complex Lymphatic Anomalies (CLA). Write a short, detailed, factual passage (150-200 words)
that directly answers the following question. Write as if this text appears in a peer-reviewed journal.
Do NOT include an introduction like "In answer to your question..." – start directly with the content.

Question: {question}

Write the hypothetical passage:"""

FINAL_ANSWER_PROMPT = """You retrieved the following passages from the CLA medical literature.
Use them to provide a comprehensive, accurate answer.

Retrieved Context:
{context}

Question: {question}

Answer:"""


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class HyDERAGPipeline(BasePipeline):
    """
    HyDE (Hypothetical Document Embeddings) RAG Pipeline.

    Retrieval improvement over basic RAG:
    ┌─────────────────────────────────────────────────────┐
    │  Basic RAG : embed(question)  → search              │
    │  HyDE RAG  : embed(LLM(question)) → search          │
    │              └─ or ensemble of N hypothetical docs   │
    └─────────────────────────────────────────────────────┘

    In the CLA domain, a question like "What does sirolimus inhibit?" has a
    much better embedding match to retrieved paragraphs when first expanded
    into a hypothetical passage like "Sirolimus (rapamycin) inhibits mTORC1
    by binding FKBP12, thereby suppressing lymphatic endothelial cell
    proliferation..."
    """

    PIPELINE_NAME = "HyDE_RAG"

    def __init__(self, config_path: str | Path = "config/config.yaml"):
        super().__init__(config_path)
        self.vector_store: VectorStoreManager | None = None

    def build_index(self, dataset_path: str | Path) -> None:
        logger.info("[HyDE-RAG] Building vector index...")
        processor = DocumentProcessor.from_config(self.config_path)
        chunks = processor.process_dataset(dataset_path)
        self.vector_store = VectorStoreManager.from_config(self.config_path)
        self.vector_store.index_chunks(chunks)
        logger.info("[HyDE-RAG] Index ready.")

    def _load_vector_store(self) -> None:
        if self.vector_store is None:
            self.vector_store = VectorStoreManager.from_config(self.config_path)

    def generate_hypothetical_document(self, question: str) -> str:
        """Ask the LLM to write a hypothetical ideal answer passage."""
        prompt = HYPOTHETICAL_DOC_PROMPT.format(question=question)
        hyp_doc = self.generate(
            prompt,
            system_prompt=SYSTEM_PROMPT,
            temperature=0.3,
            max_tokens=300,
        )
        logger.debug("[HyDE-RAG] Hypothetical doc: %s...", hyp_doc[:100])
        return hyp_doc

    def generate_multiple_hypothetical_docs(self, question: str, n: int = 3) -> list[str]:
        """Generate N diverse hypothetical documents and return all."""
        docs = []
        for i in range(n):
            doc = self.generate(
                HYPOTHETICAL_DOC_PROMPT.format(question=question),
                system_prompt=SYSTEM_PROMPT,
                temperature=0.5 + i * 0.1,  # slight temperature variation for diversity
                max_tokens=300,
            )
            docs.append(doc)
        return docs

    def embed_and_average(self, texts: list[str]) -> list[float]:
        """Embed multiple texts and return their centroid embedding."""
        self._load_vector_store()
        embeddings = self.vector_store.embed_texts(texts)
        arr = np.array(embeddings)
        centroid = arr.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        return centroid.tolist()

    def retrieve(
        self,
        question: str,
        k: int = 5,
        n_hypothetical: int = 1,
        use_ensemble: bool = False,
    ) -> tuple[list[dict], list[str]]:
        """
        Returns (retrieved_chunks, hypothetical_docs).
        - n_hypothetical: number of hypothetical docs to generate
        - use_ensemble: average embeddings of all hypothetical docs
        """
        self._load_vector_store()

        if n_hypothetical == 1:
            hyp_docs = [self.generate_hypothetical_document(question)]
        else:
            hyp_docs = self.generate_multiple_hypothetical_docs(question, n=n_hypothetical)

        if use_ensemble and len(hyp_docs) > 1:
            # Ensemble: average embeddings of all hypothetical documents
            query_embedding = self.embed_and_average(hyp_docs)
            retrieved = self.vector_store.search_by_embedding(query_embedding, k=k)
        else:
            # Use the first (or only) hypothetical document for retrieval
            query_embedding = self.vector_store.embed_query(hyp_docs[0])
            retrieved = self.vector_store.search_by_embedding(query_embedding, k=k)

        threshold = self.retrieval_config.get("similarity_threshold", 0.3)
        filtered = [r for r in retrieved if r["score"] >= threshold]
        return (filtered or retrieved[:1]), hyp_docs

    def format_context(self, retrieved: list[dict]) -> str:
        blocks = []
        for i, r in enumerate(retrieved, 1):
            title = r["metadata"].get("title", "Unknown")
            source = r["metadata"].get("doc_id", "")
            blocks.append(f"[{i}] {title} [{source}]\n{r['text']}")
        return "\n\n".join(blocks)

    def query(
        self,
        question: str,
        k: int | None = None,
        n_hypothetical: int | None = None,
        use_ensemble: bool | None = None,
        **kwargs,
    ) -> RAGResponse:
        hyde_cfg = self.config.get("hyde", {})
        k = k or self.retrieval_config.get("top_k", 5)
        n_hypothetical = n_hypothetical or hyde_cfg.get("num_hypothetical_docs", 1)
        use_ensemble = use_ensemble if use_ensemble is not None else hyde_cfg.get("use_ensemble", False)

        reasoning_trace = [f"Generating {n_hypothetical} hypothetical document(s)..."]

        retrieved, hyp_docs = self.retrieve(
            question,
            k=k,
            n_hypothetical=n_hypothetical,
            use_ensemble=use_ensemble and n_hypothetical > 1,
        )

        for i, doc in enumerate(hyp_docs, 1):
            reasoning_trace.append(f"Hypothetical doc {i}: {doc[:100]}...")
        reasoning_trace.append(
            f"Retrieved {len(retrieved)} chunks using "
            f"{'ensemble' if use_ensemble else 'single'} HyDE embedding"
        )

        context = self.format_context(retrieved)
        prompt = FINAL_ANSWER_PROMPT.format(context=context, question=question)
        answer = self.generate(prompt, system_prompt=SYSTEM_PROMPT)
        reasoning_trace.append("Final answer generated")

        return RAGResponse(
            question=question,
            answer=answer,
            retrieved_contexts=[r["text"] for r in retrieved],
            retrieved_metadata=[r["metadata"] for r in retrieved],
            retrieval_scores=[r["score"] for r in retrieved],
            pipeline_name=self.PIPELINE_NAME,
            reasoning_trace=reasoning_trace,
            metadata={
                "hypothetical_documents": hyp_docs,
                "n_hypothetical": n_hypothetical,
                "use_ensemble": use_ensemble,
            },
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="HyDE RAG Pipeline for CLA Q&A")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--dataset", default="data/pseudo_dataset/cla_documents.json")
    parser.add_argument("--question", default=None)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--n_hyp", type=int, default=1, help="Number of hypothetical docs")
    parser.add_argument("--ensemble", action="store_true", help="Use embedding ensemble")
    args = parser.parse_args()

    pipeline = HyDERAGPipeline(config_path=args.config)
    pipeline.build_index(args.dataset)

    demo_questions = [
        "What is the MILES trial and what did it show about sirolimus?",
        "How does chylothorax form in Gorham-Stout disease?",
        "What targeted therapies exist for PIK3CA-mutant GLA?",
    ]

    questions = [args.question] if args.question else demo_questions

    print("\n=== HyDE RAG Pipeline Demo ===\n")
    for question in questions[:3]:
        response = pipeline.timed_query(
            question,
            k=args.top_k,
            n_hypothetical=args.n_hyp,
            use_ensemble=args.ensemble,
        )
        response.print_summary()
        print("\nHypothetical Document(s) used for retrieval:")
        for i, doc in enumerate(response.metadata.get("hypothetical_documents", []), 1):
            print(f"\n  [Doc {i}]: {doc[:300]}...")


if __name__ == "__main__":
    main()
