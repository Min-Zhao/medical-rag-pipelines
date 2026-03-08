"""
Pipeline 02: Knowledge Graph RAG (KG-RAG)
==========================================
Augments dense vector retrieval with structured knowledge graph context.

Architecture:
  Query → [Entity Extraction] → [KG Neighborhood Lookup]
        ↘ [Vector Search]
                    ↓ Merge & Deduplicate
              [KG Triples + Vector Chunks]
                    ↓
              [LLM Generation]

Why KG-RAG?
  Standard RAG retrieves semantically similar chunks but struggles with:
  - Relationship queries ("What does sirolimus inhibit?")
  - Multi-entity questions ("How do PIK3CA and mTOR relate to GLA?")
  - Out-of-context implicit connections

  The knowledge graph provides explicit entity-relation triples that
  complement chunk-level retrieval.

Usage:
    python pipelines/02_knowledge_graph_rag.py
    python pipelines/02_knowledge_graph_rag.py --question "What gene mutations cause KLA?"
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipelines.base_pipeline import BasePipeline, RAGResponse
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStoreManager
from src.knowledge_graph import KnowledgeGraphBuilder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a specialized medical AI assistant for Complex Lymphatic Anomalies (CLA).
You have access to both retrieved literature passages AND structured knowledge graph triples.
Use BOTH sources to provide comprehensive, accurate answers.
Do not hallucinate. Cite sources when possible."""

KG_RAG_PROMPT_TEMPLATE = """Answer the medical question using:
1. Knowledge Graph Triples (structured entity-relationship data)
2. Retrieved Literature Passages (raw text from papers)

{kg_context}

Retrieved Literature:
{vector_context}

Question: {question}

Synthesize information from BOTH sources above. When the KG triples and literature agree,
state this clearly. If they conflict, note the discrepancy.
Answer:"""


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class KnowledgeGraphRAGPipeline(BasePipeline):
    """
    Hybrid KG + Vector RAG pipeline.

    Index construction:
      1. Chunk & embed documents → ChromaDB (vector retrieval)
      2. Extract entities & relations → NetworkX knowledge graph

    Query-time:
      1. Extract entities from question via rule-based NER
      2. Retrieve KG neighborhood (depth=1-2 hops) → structured triples
      3. Dense vector search → top-k chunks
      4. Concatenate KG triples + vector chunks as context
      5. LLM generation
    """

    PIPELINE_NAME = "KnowledgeGraphRAG"

    def __init__(self, config_path: str | Path = "config/config.yaml"):
        super().__init__(config_path)
        self.vector_store: VectorStoreManager | None = None
        self.kg_builder = KnowledgeGraphBuilder()
        self._kg_built = False

    def build_index(self, dataset_path: str | Path) -> None:
        logger.info("[KG-RAG] Building vector index and knowledge graph...")

        processor = DocumentProcessor.from_config(self.config_path)
        documents = processor.get_documents(dataset_path)
        chunks = processor.process_documents(documents)

        # 1. Build vector store
        self.vector_store = VectorStoreManager.from_config(self.config_path)
        self.vector_store.index_chunks(chunks)

        # 2. Build knowledge graph
        kg_cfg = self.config.get("knowledge_graph", {})
        logger.info("[KG-RAG] Extracting entities and relations from %d documents...", len(documents))
        self.kg_builder.build_from_documents(documents)
        self._kg_built = True

        stats = self.kg_builder.get_stats()
        logger.info(
            "[KG-RAG] KG built: %d nodes, %d edges. Top entities: %s",
            stats["num_nodes"],
            stats["num_edges"],
            [n for n, _ in stats["top_degree_nodes"][:5]],
        )

        # Optionally save the graph
        kg_path = Path("data/vector_store/knowledge_graph.json")
        self.kg_builder.save(kg_path)
        logger.info("[KG-RAG] Knowledge graph saved to %s", kg_path)

    def _load_stores(self) -> None:
        if self.vector_store is None:
            self.vector_store = VectorStoreManager.from_config(self.config_path)

        if not self._kg_built:
            kg_path = Path("data/vector_store/knowledge_graph.json")
            if kg_path.exists():
                self.kg_builder.load(kg_path)
                self._kg_built = True
            else:
                logger.warning(
                    "Knowledge graph not found at %s. Call build_index() first.", kg_path
                )

    def retrieve_vector(self, question: str, k: int = 5) -> list[dict]:
        threshold = self.retrieval_config.get("similarity_threshold", 0.3)
        results = self.vector_store.search(question, k=k)
        filtered = [r for r in results if r["score"] >= threshold]
        return filtered or results[:1]

    def retrieve_kg(self, question: str, depth: int = 2) -> str:
        """Extract entities from the question and return KG triple context."""
        if not self._kg_built:
            return ""
        kg_context = self.kg_builder.get_context_for_query(question, max_triples=15)
        return kg_context

    def format_vector_context(self, retrieved: list[dict]) -> str:
        blocks = []
        for i, r in enumerate(retrieved, 1):
            title = r["metadata"].get("title", "Unknown")
            source = r["metadata"].get("doc_id", "")
            blocks.append(f"[{i}] {title} [{source}]\n{r['text']}")
        return "\n\n".join(blocks)

    def query(self, question: str, k: int | None = None, **kwargs) -> RAGResponse:
        self._load_stores()
        k = k or self.retrieval_config.get("top_k", 5)

        # 1. KG retrieval
        kg_context = self.retrieve_kg(question)
        reasoning_trace = []

        if kg_context:
            entity_count = kg_context.count("-->")
            reasoning_trace.append(f"KG retrieval: {entity_count} triples found")
        else:
            reasoning_trace.append("KG retrieval: no matching entities in graph")

        # 2. Vector retrieval
        vector_results = self.retrieve_vector(question, k=k)
        reasoning_trace.append(f"Vector retrieval: {len(vector_results)} chunks (top score: {vector_results[0]['score']:.3f})")

        # 3. Build prompt
        vector_context = self.format_vector_context(vector_results)
        kg_section = kg_context if kg_context else "No relevant knowledge graph triples found."

        prompt = KG_RAG_PROMPT_TEMPLATE.format(
            kg_context=kg_section,
            vector_context=vector_context,
            question=question,
        )

        # 4. Generate
        answer = self.generate(prompt, system_prompt=SYSTEM_PROMPT)
        reasoning_trace.append("LLM generation completed")

        return RAGResponse(
            question=question,
            answer=answer,
            retrieved_contexts=[r["text"] for r in vector_results],
            retrieved_metadata=[r["metadata"] for r in vector_results],
            retrieval_scores=[r["score"] for r in vector_results],
            pipeline_name=self.PIPELINE_NAME,
            reasoning_trace=reasoning_trace,
            metadata={"kg_context": kg_context},
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Knowledge Graph RAG for CLA Q&A")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--dataset", default="data/pseudo_dataset/cla_documents.json")
    parser.add_argument("--question", default=None)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--show_kg", action="store_true", help="Print KG stats")
    args = parser.parse_args()

    pipeline = KnowledgeGraphRAGPipeline(config_path=args.config)
    pipeline.build_index(args.dataset)

    if args.show_kg:
        stats = pipeline.kg_builder.get_stats()
        print("\n=== Knowledge Graph Statistics ===")
        print(f"  Nodes: {stats['num_nodes']}")
        print(f"  Edges: {stats['num_edges']}")
        print(f"  Entity types: {stats['entity_type_counts']}")
        print(f"  Top nodes by degree:")
        for node, degree in stats["top_degree_nodes"][:8]:
            print(f"    {node} (degree={degree})")

    questions = args.question and [args.question] or [
        "What gene mutations are found in kaposiform lymphangiomatosis?",
        "How does sirolimus work to treat complex lymphatic anomalies?",
        "What is the relationship between PIK3CA mutations and GLA?",
        "What biomarkers are used to monitor CLA treatment?",
    ]

    print("\n=== Knowledge Graph RAG Pipeline Demo ===\n")
    for question in questions[:3]:
        response = pipeline.timed_query(question, k=args.top_k)
        response.print_summary()
        if response.metadata.get("kg_context"):
            print(f"\nKG Context used:\n{response.metadata['kg_context'][:500]}...")


if __name__ == "__main__":
    main()
