"""
Document Processor
==================
Handles loading, chunking, and embedding of CLA documents.
Supports JSON dataset, PDF, and plain-text inputs.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Document:
    """Raw document loaded from the dataset."""
    doc_id: str
    title: str
    full_text: str
    abstract: str = ""
    source_type: str = ""
    disease_entity: str = ""
    year: int = 0
    journal: str = ""
    authors: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_metadata_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "source_type": self.source_type,
            "disease_entity": self.disease_entity,
            "year": self.year,
            "journal": self.journal,
            "authors": ", ".join(self.authors),
            "keywords": ", ".join(self.keywords),
        }


@dataclass
class Chunk:
    """A text chunk derived from a Document, ready for embedding."""
    chunk_id: str
    doc_id: str
    text: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def full_text_for_embedding(self) -> str:
        """Prepend title for better embedding quality."""
        title = self.metadata.get("title", "")
        return f"Title: {title}\n\n{self.text}" if title else self.text


# ---------------------------------------------------------------------------
# Document loader
# ---------------------------------------------------------------------------

class DocumentLoader:
    """Loads documents from various sources into Document objects."""

    @staticmethod
    def from_json(dataset_path: str | Path) -> list[Document]:
        """Load documents from the pseudo-dataset JSON file."""
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {path}. "
                "Run: python data/pseudo_dataset/generate_dataset.py"
            )

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        documents = []
        for raw in data.get("documents", []):
            doc = Document(
                doc_id=raw["id"],
                title=raw.get("title", ""),
                full_text=raw.get("full_text", raw.get("abstract", "")),
                abstract=raw.get("abstract", ""),
                source_type=raw.get("source_type", ""),
                disease_entity=raw.get("disease_entity", "CLA"),
                year=raw.get("year", 0),
                journal=raw.get("journal", ""),
                authors=raw.get("authors", []),
                keywords=raw.get("keywords", []),
            )
            documents.append(doc)

        logger.info("Loaded %d documents from %s", len(documents), path)
        return documents

    @staticmethod
    def from_pdf_directory(directory: str | Path) -> list[Document]:
        """Load PDFs from a directory using PyPDF."""
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("Install pypdf: pip install pypdf")

        docs = []
        pdf_dir = Path(directory)
        for i, pdf_path in enumerate(sorted(pdf_dir.glob("*.pdf"))):
            reader = PdfReader(str(pdf_path))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            docs.append(Document(
                doc_id=f"pdf_{i:04d}",
                title=pdf_path.stem,
                full_text=text,
            ))
        logger.info("Loaded %d PDFs from %s", len(docs), directory)
        return docs

    @staticmethod
    def from_text_files(directory: str | Path) -> list[Document]:
        """Load plain text files from a directory."""
        docs = []
        txt_dir = Path(directory)
        for i, txt_path in enumerate(sorted(txt_dir.glob("*.txt"))):
            text = txt_path.read_text(encoding="utf-8")
            docs.append(Document(
                doc_id=f"txt_{i:04d}",
                title=txt_path.stem,
                full_text=text,
            ))
        return docs


# ---------------------------------------------------------------------------
# Text splitter
# ---------------------------------------------------------------------------

class RecursiveTextSplitter:
    """
    Splits text into overlapping chunks, respecting paragraph and sentence
    boundaries for better semantic coherence.
    """

    SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> list[str]:
        """Return list of text chunks."""
        text = self._clean_text(text)
        return self._split(text, self.SEPARATORS)

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _split(self, text: str, separators: list[str]) -> list[str]:
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        for sep in separators:
            if sep == "":
                parts = list(text)
                break
            if sep in text:
                parts = text.split(sep)
                parts = [p + sep for p in parts[:-1]] + [parts[-1]]
                break
        else:
            parts = [text]

        chunks: list[str] = []
        current = ""

        for part in parts:
            if len(current) + len(part) <= self.chunk_size:
                current += part
            else:
                if current:
                    chunks.append(current.strip())
                if len(part) > self.chunk_size:
                    sub = self._split(part, separators[1:] if len(separators) > 1 else [""])
                    chunks.extend(sub)
                    current = ""
                else:
                    overlap_start = max(0, len(current) - self.chunk_overlap)
                    current = current[overlap_start:] + part

        if current.strip():
            chunks.append(current.strip())

        return [c for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# Main processor
# ---------------------------------------------------------------------------

class DocumentProcessor:
    """
    Orchestrates document loading, chunking, and preparation for embedding.

    Example:
        processor = DocumentProcessor.from_config("config/config.yaml")
        chunks = processor.process_dataset("data/pseudo_dataset/cla_documents.json")
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.splitter = RecursiveTextSplitter(chunk_size, chunk_overlap)
        self.loader = DocumentLoader()

    @classmethod
    def from_config(cls, config_path: str | Path) -> "DocumentProcessor":
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        retrieval = cfg.get("retrieval", {})
        return cls(
            chunk_size=retrieval.get("chunk_size", 512),
            chunk_overlap=retrieval.get("chunk_overlap", 64),
        )

    def process_dataset(self, dataset_path: str | Path) -> list[Chunk]:
        """Load JSON dataset and return all chunks."""
        documents = DocumentLoader.from_json(dataset_path)
        return self.process_documents(documents)

    def process_documents(self, documents: list[Document]) -> list[Chunk]:
        """Chunk a list of Document objects."""
        all_chunks: list[Chunk] = []
        for doc in documents:
            chunks = self._chunk_document(doc)
            all_chunks.extend(chunks)
        logger.info(
            "Processed %d documents into %d chunks",
            len(documents),
            len(all_chunks),
        )
        return all_chunks

    def _chunk_document(self, doc: Document) -> list[Chunk]:
        """Split one document into chunks, using full_text."""
        text = doc.full_text or doc.abstract
        if not text:
            logger.warning("Document %s has no text, skipping", doc.doc_id)
            return []

        raw_chunks = self.splitter.split_text(text)
        meta = doc.to_metadata_dict()

        return [
            Chunk(
                chunk_id=f"{doc.doc_id}_chunk_{i:04d}",
                doc_id=doc.doc_id,
                text=raw,
                chunk_index=i,
                metadata=meta,
            )
            for i, raw in enumerate(raw_chunks)
            if raw.strip()
        ]

    def get_documents(self, dataset_path: str | Path) -> list[Document]:
        """Load and return Document objects (no chunking)."""
        return DocumentLoader.from_json(dataset_path)
