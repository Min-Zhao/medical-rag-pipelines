# medical-rag-pipelines
End-to-end implementations of multiple RAG architectures for medical question answering and rare disease chatbot systems.

## Overview

Rare diseases like CLAs affect fewer than 200,000 people in the US, yet patients and clinicians often struggle to find accurate, up-to-date information. This project demonstrates how LLM-powered chatbots with different RAG strategies can improve access to specialized medical knowledge.

### RAG Pipelines Implemented

| Pipeline | Description | Best For |
|----------|-------------|----------|
| **01 · Basic RAG** | Dense vector retrieval + LLM generation | General Q&A, fast prototyping |
| **02 · Knowledge Graph RAG** | Entity-relation graph + vector hybrid | Mechanistic / relationship queries |
| **03 · HyDE RAG** | Hypothetical Document Embeddings | Precision-sensitive retrieval |
| **04 · Self-RAG** | Adaptive retrieval with self-reflection | Complex, multi-part questions |
| **05 · Multi-hop RAG** | Iterative chain-of-thought retrieval | Multi-step reasoning queries |

### Case Study: Complex Lymphatic Anomalies (CLAs)

CLA encompasses a spectrum of rare disorders including:
- **Gorham-Stout Disease (GSD)** – vanishing bone disease with osteolysis
- **Generalized Lymphatic Anomaly (GLA)** – multifocal lymphatic malformations
- **Central Conducting Lymphatic Anomaly (CCLA)** – lymphatic flow dysfunction
- **Kaposiform Lymphangiomatosis (KLA)** – aggressive multisystem involvement

## Project Structure

```
RAG_pipeline/
├── README.md
├── requirements.txt
├── config/
│   └── config.yaml               # API keys, model configs, paths
├── data/
│   └── pseudo_dataset/
│       ├── generate_dataset.py   # Generates pseudo CLA documents
│       └── cla_documents.json    # Curated pseudo dataset (auto-generated)
├── src/
│   ├── __init__.py
│   ├── document_processor.py     # Chunking, embedding, ingestion
│   ├── vector_store.py           # Vector DB management (ChromaDB / FAISS)
│   ├── knowledge_graph.py        # KG construction from documents
│   └── evaluation.py             # RAG evaluation metrics
├── pipelines/
│   ├── __init__.py
│   ├── 01_basic_rag.py           # Baseline dense retrieval RAG
│   ├── 02_knowledge_graph_rag.py # KG-enhanced hybrid RAG
│   ├── 03_hyde_rag.py            # Hypothetical Document Embeddings RAG
│   ├── 04_self_rag.py            # Self-reflective adaptive RAG
│   └── 05_multihop_rag.py        # Iterative multi-hop RAG
├── app/
│   └── chatbot_app.py            # Streamlit chatbot demo
└── notebooks/
    ├── 01_dataset_exploration.ipynb
    └── 02_pipeline_comparison.ipynb
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API keys

Copy and edit the config file:
```bash
cp config/config.yaml config/config_local.yaml
# Add your OpenAI API key (or use local Ollama)
```

### 3. Generate the pseudo dataset

```bash
python data/pseudo_dataset/generate_dataset.py
```

### 4. Run a pipeline

```bash
# Basic RAG
python pipelines/01_basic_rag.py

# Knowledge Graph RAG
python pipelines/02_knowledge_graph_rag.py

# HyDE RAG
python pipelines/03_hyde_rag.py

# Self-RAG
python pipelines/04_self_rag.py

# Multi-hop RAG
python pipelines/05_multihop_rag.py
```

### 5. Launch the chatbot demo

```bash
streamlit run app/chatbot_app.py
```

## Dataset

The pseudo dataset simulates a curated corpus of CLA-related literature including:
- Clinical case reports and case series
- Review articles on pathogenesis and treatment
- Diagnostic guideline summaries
- Drug/therapy mechanism descriptions
- Patient-facing educational content

Documents are generated to cover realistic CLA domain knowledge for benchmarking purposes. In a production setting, replace with actual PubMed abstracts, clinical guidelines (NORD, NIH), and consented patient education materials.

## Evaluation Metrics

All pipelines are evaluated using:
- **Faithfulness** – Are answers grounded in retrieved context?
- **Answer Relevancy** – Does the answer address the question?
- **Context Recall** – Were relevant documents retrieved?
- **Context Precision** – What fraction of retrieved docs are relevant?
- **BERTScore** – Semantic similarity to reference answers

## Requirements

- Python 3.10+
- OpenAI API key (or local Ollama with `llama3`, `mistral`, etc.)
- ~2 GB disk for ChromaDB vector store

## Citation

If you use this framework in your research, please cite this GitHub repository

## License

MIT License
