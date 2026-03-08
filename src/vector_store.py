"""
Vector Store Manager
====================
Manages document embedding and vector storage using ChromaDB (default)
or FAISS. Supports OpenAI and local sentence-transformer embeddings.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

import yaml

from .document_processor import Chunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedding providers
# ---------------------------------------------------------------------------

class EmbeddingProvider:
    """Base class for embedding backends."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI text-embedding models."""

    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Install openai: pip install openai")

        self.model = model
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        logger.info("Using OpenAI embeddings: %s", model)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        texts = [t.replace("\n", " ") for t in texts]
        response = self.client.embeddings.create(input=texts, model=self.model)
        return [item.embedding for item in response.data]


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Local sentence-transformers embeddings (no API cost)."""

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Install sentence-transformers: pip install sentence-transformers")

        logger.info("Loading SentenceTransformer: %s", model_name)
        self.model = SentenceTransformer(model_name)
        logger.info("SentenceTransformer loaded")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings.tolist()


def get_embedding_provider(config: dict) -> EmbeddingProvider:
    """Factory: instantiate the correct embedding provider from config."""
    emb_cfg = config.get("embeddings", {})
    provider = emb_cfg.get("provider", "sentence_transformers")

    if provider == "openai":
        api_key = config.get("llm", {}).get("api_key") or os.getenv("OPENAI_API_KEY")
        return OpenAIEmbeddingProvider(
            model=emb_cfg.get("model", "text-embedding-3-small"),
            api_key=api_key,
        )
    else:
        return SentenceTransformerEmbeddingProvider(
            model_name=emb_cfg.get("st_model", "BAAI/bge-large-en-v1.5")
        )


# ---------------------------------------------------------------------------
# ChromaDB vector store
# ---------------------------------------------------------------------------

class ChromaVectorStore:
    """Persistent ChromaDB vector store."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        persist_dir: str = "./data/vector_store",
        collection_name: str = "cla_documents",
    ):
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError("Install chromadb: pip install chromadb")

        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self.embedding_provider = embedding_provider
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB collection '%s' has %d documents",
            collection_name,
            self.collection.count(),
        )

    def add_chunks(self, chunks: list[Chunk], batch_size: int = 32) -> None:
        """Embed and upsert chunks into ChromaDB."""
        existing_ids = set(self.collection.get(include=[])["ids"])

        new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]
        if not new_chunks:
            logger.info("All %d chunks already indexed, skipping", len(chunks))
            return

        logger.info("Embedding %d new chunks (batch_size=%d)...", len(new_chunks), batch_size)
        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i : i + batch_size]
            texts = [c.full_text_for_embedding for c in batch]
            embeddings = self.embedding_provider.embed_texts(texts)
            self.collection.upsert(
                ids=[c.chunk_id for c in batch],
                embeddings=embeddings,
                documents=[c.text for c in batch],
                metadatas=[{**c.metadata, "chunk_index": c.chunk_index, "doc_id": c.doc_id} for c in batch],
            )
            logger.debug("Upserted batch %d/%d", i // batch_size + 1, (len(new_chunks) - 1) // batch_size + 1)

        logger.info("Indexed %d chunks into ChromaDB", len(new_chunks))

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: dict | None = None,
    ) -> list[dict]:
        """Return top-k most similar chunks with scores."""
        query_embedding = self.embedding_provider.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self.collection.count()),
            where=filter_metadata,
            include=["documents", "metadatas", "distances"],
        )
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        return [
            {
                "text": doc,
                "metadata": meta,
                "score": 1.0 - dist,  # cosine similarity from distance
            }
            for doc, meta, dist in zip(docs, metas, distances)
        ]

    def similarity_search_by_embedding(
        self, query_embedding: list[float], k: int = 5
    ) -> list[dict]:
        """Search using a pre-computed embedding vector."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]
        return [
            {"text": doc, "metadata": meta, "score": 1.0 - dist}
            for doc, meta, dist in zip(docs, metas, distances)
        ]

    def count(self) -> int:
        return self.collection.count()


# ---------------------------------------------------------------------------
# FAISS vector store (alternative)
# ---------------------------------------------------------------------------

class FAISSVectorStore:
    """In-memory FAISS index with optional disk persistence."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        index_path: str = "./data/vector_store/faiss.index",
    ):
        try:
            import faiss
            import numpy as np
        except ImportError:
            raise ImportError("Install faiss-cpu: pip install faiss-cpu")

        self.embedding_provider = embedding_provider
        self.index_path = Path(index_path)
        self._texts: list[str] = []
        self._metadatas: list[dict] = []
        self._index = None
        self._dim: int | None = None
        import numpy as np
        self._np = np
        self._faiss = faiss

        if self.index_path.exists():
            self._load()

    def add_chunks(self, chunks: list[Chunk], batch_size: int = 32) -> None:
        all_embeddings = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.full_text_for_embedding for c in batch]
            embs = self.embedding_provider.embed_texts(texts)
            all_embeddings.extend(embs)
            self._texts.extend(c.text for c in batch)
            self._metadatas.extend({**c.metadata, "doc_id": c.doc_id} for c in batch)

        embeddings_np = self._np.array(all_embeddings, dtype=self._np.float32)
        if self._index is None:
            self._dim = embeddings_np.shape[1]
            self._index = self._faiss.IndexFlatIP(self._dim)
            self._faiss.normalize_L2(embeddings_np)

        self._faiss.normalize_L2(embeddings_np)
        self._index.add(embeddings_np)
        logger.info("FAISS index now has %d vectors", self._index.ntotal)

    def similarity_search(self, query: str, k: int = 5) -> list[dict]:
        query_emb = self._np.array(
            [self.embedding_provider.embed_query(query)], dtype=self._np.float32
        )
        self._faiss.normalize_L2(query_emb)
        scores, indices = self._index.search(query_emb, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append({
                    "text": self._texts[idx],
                    "metadata": self._metadatas[idx],
                    "score": float(score),
                })
        return results

    def _load(self) -> None:
        import pickle
        self._index = self._faiss.read_index(str(self.index_path))
        meta_path = self.index_path.with_suffix(".pkl")
        if meta_path.exists():
            with open(meta_path, "rb") as f:
                data = pickle.load(f)
                self._texts = data["texts"]
                self._metadatas = data["metadatas"]
        logger.info("Loaded FAISS index with %d vectors", self._index.ntotal)

    def save(self) -> None:
        import pickle
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self._index, str(self.index_path))
        with open(self.index_path.with_suffix(".pkl"), "wb") as f:
            pickle.dump({"texts": self._texts, "metadatas": self._metadatas}, f)
        logger.info("FAISS index saved to %s", self.index_path)


# ---------------------------------------------------------------------------
# Manager (facade)
# ---------------------------------------------------------------------------

class VectorStoreManager:
    """
    High-level façade that wires together the embedding provider and
    vector store backend, following the project config.

    Example:
        manager = VectorStoreManager.from_config("config/config.yaml")
        manager.index_chunks(chunks)
        results = manager.search("What causes Gorham-Stout disease?", k=5)
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        store_type: str = "chroma",
        persist_dir: str = "./data/vector_store",
        collection_name: str = "cla_documents",
    ):
        self.embedding_provider = embedding_provider
        if store_type == "faiss":
            self.store: ChromaVectorStore | FAISSVectorStore = FAISSVectorStore(
                embedding_provider, index_path=f"{persist_dir}/faiss.index"
            )
        else:
            self.store = ChromaVectorStore(
                embedding_provider, persist_dir=persist_dir, collection_name=collection_name
            )

    @classmethod
    def from_config(cls, config_path: str | Path) -> "VectorStoreManager":
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        emb_provider = get_embedding_provider(cfg)
        vs_cfg = cfg.get("vector_store", {})
        return cls(
            embedding_provider=emb_provider,
            store_type=vs_cfg.get("type", "chroma"),
            persist_dir=vs_cfg.get("persist_directory", "./data/vector_store"),
            collection_name=vs_cfg.get("collection_name", "cla_documents"),
        )

    def index_chunks(self, chunks: list[Chunk]) -> None:
        self.store.add_chunks(chunks)

    def search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: dict | None = None,
    ) -> list[dict]:
        if isinstance(self.store, ChromaVectorStore):
            return self.store.similarity_search(query, k=k, filter_metadata=filter_metadata)
        return self.store.similarity_search(query, k=k)

    def search_by_embedding(self, embedding: list[float], k: int = 5) -> list[dict]:
        if isinstance(self.store, ChromaVectorStore):
            return self.store.similarity_search_by_embedding(embedding, k=k)
        raise NotImplementedError("FAISS does not support pre-computed embedding search via this interface")

    def embed_query(self, text: str) -> list[float]:
        return self.embedding_provider.embed_query(text)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self.embedding_provider.embed_texts(texts)
