"""
CLA RAG Chatbot – Streamlit Demo App
=====================================
Interactive chatbot comparing 5 RAG pipeline architectures
for Complex Lymphatic Anomalies (CLA) rare disease Q&A.

Launch:
    cd RAG_pipeline
    streamlit run app/chatbot_app.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CLA RAG Chatbot",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
.pipeline-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: bold;
    margin-right: 4px;
}
.badge-basic    { background: #4CAF50; color: white; }
.badge-kg       { background: #2196F3; color: white; }
.badge-hyde     { background: #FF9800; color: white; }
.badge-self     { background: #9C27B0; color: white; }
.badge-multihop { background: #F44336; color: white; }

.metric-card {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 12px;
    margin: 4px;
    text-align: center;
    border: 1px solid #dee2e6;
}
.answer-box {
    background: #f1f8e9;
    border-left: 4px solid #4CAF50;
    padding: 12px;
    border-radius: 4px;
    margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Lazy pipeline loader (cached)
# ---------------------------------------------------------------------------

PIPELINE_INFO = {
    "Basic RAG": {
        "module": "pipelines.01_basic_rag",
        "class": "BasicRAGPipeline",
        "badge_class": "badge-basic",
        "description": "Standard dense retrieval + LLM. Fast and simple baseline.",
        "icon": "⚡",
    },
    "Knowledge Graph RAG": {
        "module": "pipelines.02_knowledge_graph_rag",
        "class": "KnowledgeGraphRAGPipeline",
        "badge_class": "badge-kg",
        "description": "Combines KG entity triples with vector retrieval. Best for relationship queries.",
        "icon": "🕸️",
    },
    "HyDE RAG": {
        "module": "pipelines.03_hyde_rag",
        "class": "HyDERAGPipeline",
        "badge_class": "badge-hyde",
        "description": "Generates hypothetical answers to improve retrieval precision.",
        "icon": "💡",
    },
    "Self-RAG": {
        "module": "pipelines.04_self_rag",
        "class": "SelfRAGPipeline",
        "badge_class": "badge-self",
        "description": "Self-reflective pipeline that judges its own retrieval and output.",
        "icon": "🪞",
    },
    "Multi-hop RAG": {
        "module": "pipelines.05_multihop_rag",
        "class": "MultiHopRAGPipeline",
        "badge_class": "badge-multihop",
        "description": "Iterative multi-step reasoning for complex questions.",
        "icon": "🔗",
    },
}

DATASET_PATH = "data/pseudo_dataset/cla_documents.json"
CONFIG_PATH = "config/config.yaml"

EXAMPLE_QUESTIONS = [
    "What is Gorham-Stout disease and what causes the bone loss?",
    "What is the recommended sirolimus dose for children with CLA?",
    "How is chylothorax diagnosed and treated in CLA patients?",
    "What genetic mutations drive kaposiform lymphangiomatosis?",
    "What serum biomarker distinguishes LAM from other CLA subtypes?",
    "What is the MILES trial and what did it prove about sirolimus?",
    "How does Noonan syndrome relate to central conducting lymphatic anomaly?",
    "What are the side effects of sirolimus in CLA treatment?",
    "Can sirolimus cure LAM or does it only stabilize the disease?",
    "What targeted therapies exist for PIK3CA-mutant GLA?",
]


@st.cache_resource(show_spinner="Loading pipeline...")
def load_pipeline(pipeline_name: str, config_path: str, dataset_path: str):
    """Load and index a pipeline (cached across Streamlit reruns)."""
    import importlib
    info = PIPELINE_INFO[pipeline_name]
    mod = importlib.import_module(info["module"])
    cls = getattr(mod, info["class"])
    pipeline = cls(config_path=config_path)
    if Path(dataset_path).exists():
        pipeline.build_index(dataset_path)
    return pipeline


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image("https://img.icons8.com/color/96/dna-helix.png", width=60)
    st.title("CLA RAG Chatbot")
    st.caption("Rare Disease Q&A | Complex Lymphatic Anomalies")
    st.divider()

    st.subheader("⚙️ Configuration")

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="Required for OpenAI models. Alternatively set OPENAI_API_KEY env var.",
    )
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    llm_model = st.selectbox(
        "LLM Model",
        ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        help="gpt-4o-mini recommended for speed/cost balance",
    )

    st.divider()
    st.subheader("🔬 Pipeline Selection")

    selected_pipelines = st.multiselect(
        "Active Pipelines",
        list(PIPELINE_INFO.keys()),
        default=["Basic RAG"],
        help="Select one or more pipelines to compare",
    )

    top_k = st.slider("Top-K Retrieval", min_value=1, max_value=10, value=5)

    st.divider()
    st.subheader("📋 Example Questions")
    example_clicked = None
    for q in EXAMPLE_QUESTIONS[:6]:
        if st.button(q[:55] + "...", key=f"ex_{q[:20]}", use_container_width=True):
            example_clicked = q

    st.divider()
    st.markdown("""
    **About CLA subtypes:**
    - 🦴 GSD – Gorham-Stout Disease
    - 🫁 GLA – Generalized Lymphatic Anomaly
    - 🩺 CCLA – Central Conducting Lymphatic Anomaly
    - 🔴 KLA – Kaposiform Lymphangiomatosis
    - 🌬️ LAM – Lymphangioleiomyomatosis
    """)


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("🧬 CLA Rare Disease RAG Chatbot")
st.caption("Comparing RAG pipeline architectures on Complex Lymphatic Anomaly Q&A")

tabs = st.tabs(["💬 Chat", "📊 Compare Pipelines", "📚 Knowledge Base", "📖 Pipeline Guide"])

# ---- TAB 1: CHAT ----
with tabs[0]:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                for pipeline_name, content in msg["content"].items():
                    info = PIPELINE_INFO.get(pipeline_name, {})
                    icon = info.get("icon", "")
                    badge = info.get("badge_class", "")
                    st.markdown(
                        f'<span class="pipeline-badge {badge}">{icon} {pipeline_name}</span>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(content["answer"])
                    with st.expander(f"📎 Sources ({len(content.get('contexts', []))} chunks) | ⏱ {content.get('latency', 0):.2f}s"):
                        for i, ctx in enumerate(content.get("contexts", [])[:3], 1):
                            meta = content.get("metadata_list", [{}] * 10)[i - 1] if content.get("metadata_list") else {}
                            title = meta.get("title", f"Chunk {i}")
                            st.markdown(f"**[{i}] {title}**")
                            st.markdown(f"> {ctx[:400]}...")
                        if content.get("reasoning_trace"):
                            st.markdown("**Reasoning trace:**")
                            st.code("\n".join(content["reasoning_trace"][:5]), language="text")
            else:
                st.markdown(msg["content"])

    # Input
    user_input = example_clicked or st.chat_input(
        "Ask a question about Complex Lymphatic Anomalies (CLA)..."
    )

    if user_input:
        if not api_key and not os.getenv("OPENAI_API_KEY"):
            st.error("Please enter your OpenAI API key in the sidebar to use the chatbot.")
            st.stop()

        if not selected_pipelines:
            st.warning("Please select at least one pipeline in the sidebar.")
            st.stop()

        if not Path(DATASET_PATH).exists():
            st.error(
                f"Dataset not found at `{DATASET_PATH}`. "
                "Run: `python data/pseudo_dataset/generate_dataset.py`"
            )
            st.stop()

        # Show user message
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Query all selected pipelines
        assistant_content = {}
        with st.chat_message("assistant"):
            for pipeline_name in selected_pipelines:
                info = PIPELINE_INFO[pipeline_name]
                icon = info.get("icon", "")
                badge = info.get("badge_class", "")
                st.markdown(
                    f'<span class="pipeline-badge {badge}">{icon} {pipeline_name}</span>',
                    unsafe_allow_html=True,
                )
                with st.spinner(f"Running {pipeline_name}..."):
                    try:
                        pipeline = load_pipeline(pipeline_name, CONFIG_PATH, DATASET_PATH)
                        response = pipeline.timed_query(user_input, k=top_k)
                        st.markdown(response.answer)
                        with st.expander(
                            f"📎 {len(response.retrieved_contexts)} sources | ⏱ {response.latency_seconds:.2f}s"
                        ):
                            for i, (ctx, meta) in enumerate(
                                zip(response.retrieved_contexts[:3], response.retrieved_metadata[:3]), 1
                            ):
                                title = meta.get("title", f"Chunk {i}")
                                year = meta.get("year", "")
                                st.markdown(f"**[{i}] {title} ({year})**")
                                st.markdown(f"> {ctx[:400]}...")
                            if response.reasoning_trace:
                                st.markdown("**Reasoning trace:**")
                                st.code("\n".join(response.reasoning_trace[:8]), language="text")

                        assistant_content[pipeline_name] = {
                            "answer": response.answer,
                            "contexts": response.retrieved_contexts,
                            "metadata_list": response.retrieved_metadata,
                            "latency": response.latency_seconds,
                            "reasoning_trace": response.reasoning_trace,
                        }
                    except Exception as e:
                        st.error(f"Error: {e}")
                        assistant_content[pipeline_name] = {
                            "answer": f"Error: {e}", "contexts": [], "latency": 0
                        }

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": assistant_content,
        })

    if st.button("🗑️ Clear Chat History", type="secondary"):
        st.session_state.chat_history = []
        st.rerun()


# ---- TAB 2: COMPARE PIPELINES ----
with tabs[1]:
    st.subheader("Pipeline Architecture Comparison")

    compare_question = st.text_area(
        "Comparison Question",
        value="What is the recommended treatment for a child with GLA and bone lesions?",
        height=80,
    )

    pipelines_to_compare = st.multiselect(
        "Pipelines to Compare",
        list(PIPELINE_INFO.keys()),
        default=["Basic RAG", "HyDE RAG", "Knowledge Graph RAG"],
    )

    if st.button("🚀 Run Comparison", type="primary"):
        if not api_key and not os.getenv("OPENAI_API_KEY"):
            st.error("OpenAI API key required.")
        elif not Path(DATASET_PATH).exists():
            st.error(f"Dataset not found at {DATASET_PATH}. Run the dataset generator first.")
        else:
            results_cols = st.columns(len(pipelines_to_compare))
            for col, pipeline_name in zip(results_cols, pipelines_to_compare):
                with col:
                    info = PIPELINE_INFO[pipeline_name]
                    st.markdown(f"### {info['icon']} {pipeline_name}")
                    st.caption(info["description"])
                    with st.spinner("Running..."):
                        try:
                            pipeline = load_pipeline(pipeline_name, CONFIG_PATH, DATASET_PATH)
                            response = pipeline.timed_query(compare_question, k=top_k)
                            st.markdown(f"**Answer:**\n{response.answer}")
                            st.metric("Latency", f"{response.latency_seconds:.2f}s")
                            st.metric("Chunks Retrieved", len(response.retrieved_contexts))
                            if response.reasoning_trace:
                                with st.expander("Reasoning"):
                                    st.code("\n".join(response.reasoning_trace), language="text")
                        except Exception as e:
                            st.error(str(e))

    st.divider()
    st.subheader("Architecture Overview")

    arch_data = {
        "Pipeline": ["Basic RAG", "KG-RAG", "HyDE RAG", "Self-RAG", "Multi-hop RAG"],
        "Retrieval Strategy": [
            "Dense vector search",
            "Vector + KG triples",
            "HyDE embedding search",
            "Adaptive + relevance filter",
            "Iterative per sub-question",
        ],
        "Key Strength": [
            "Speed, simplicity",
            "Relationship queries",
            "Precision in specialized domains",
            "Hallucination reduction",
            "Complex multi-part questions",
        ],
        "Relative Cost": ["Low", "Medium", "Medium", "High", "High"],
        "LLM Calls": ["1", "1", "2 (hyp + answer)", "3-5 (judge calls)", "2-6 (per hop)"],
    }

    import pandas as pd
    st.dataframe(pd.DataFrame(arch_data), use_container_width=True)


# ---- TAB 3: KNOWLEDGE BASE ----
with tabs[2]:
    st.subheader("📚 CLA Document Corpus")

    if Path(DATASET_PATH).exists():
        import json
        with open(DATASET_PATH, encoding="utf-8") as f:
            dataset = json.load(f)
        docs = dataset.get("documents", [])

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Documents", len(docs))
        col2.metric("Disease Subtypes", len({d.get("disease_entity") for d in docs}))
        col3.metric("Source Types", len({d.get("source_type") for d in docs}))
        col4.metric("Year Range", f"{min(d.get('year',0) for d in docs)}–{max(d.get('year',0) for d in docs)}")

        st.divider()

        filter_entity = st.selectbox(
            "Filter by Disease",
            ["All"] + sorted({d.get("disease_entity", "") for d in docs}),
        )
        filter_type = st.selectbox(
            "Filter by Source Type",
            ["All"] + sorted({d.get("source_type", "") for d in docs}),
        )

        filtered_docs = [
            d for d in docs
            if (filter_entity == "All" or d.get("disease_entity") == filter_entity)
            and (filter_type == "All" or d.get("source_type") == filter_type)
        ]

        for doc in filtered_docs:
            with st.expander(f"📄 {doc['title']} ({doc.get('year', '')} | {doc.get('source_type', '')})"):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(f"**Abstract:** {doc.get('abstract', 'N/A')[:400]}...")
                    keywords = ", ".join(doc.get("keywords", []))
                    if keywords:
                        st.markdown(f"**Keywords:** `{keywords}`")
                with col2:
                    st.markdown(f"**Journal:** {doc.get('journal', 'N/A')}")
                    st.markdown(f"**Authors:** {', '.join(doc.get('authors', []))}")
                    st.markdown(f"**Disease:** {doc.get('disease_entity', 'N/A')}")
                    st.markdown(f"**ID:** `{doc.get('id', 'N/A')}`")
    else:
        st.warning(
            f"Dataset not found at `{DATASET_PATH}`.\n\n"
            "Run: `python data/pseudo_dataset/generate_dataset.py`"
        )
        if st.button("Generate Dataset Now"):
            import subprocess
            with st.spinner("Generating dataset..."):
                result = subprocess.run(
                    [sys.executable, "data/pseudo_dataset/generate_dataset.py"],
                    capture_output=True, text=True
                )
            if result.returncode == 0:
                st.success("Dataset generated!")
                st.rerun()
            else:
                st.error(f"Error: {result.stderr}")


# ---- TAB 4: PIPELINE GUIDE ----
with tabs[3]:
    st.subheader("📖 RAG Pipeline Guide")

    for name, info in PIPELINE_INFO.items():
        badge_class = info["badge_class"]
        icon = info["icon"]
        with st.expander(f"{icon} **{name}**", expanded=False):
            st.markdown(f"""
**Description:** {info['description']}

**How it works:**
""")
            if name == "Basic RAG":
                st.markdown("""
1. **Embed** the user query using the configured embedding model
2. **Search** ChromaDB / FAISS for the top-K most similar chunks
3. **Filter** by similarity threshold
4. **Inject** retrieved chunks as context into the LLM prompt
5. **Generate** the final answer

**Best for:** General Q&A, factual lookups, fast prototyping
**Limitations:** May miss relevant documents with different surface forms
""")
            elif name == "Knowledge Graph RAG":
                st.markdown("""
1. **Extract entities** from the query using biomedical NER (CLA entity dictionary)
2. **Traverse the KG** (depth=2 hops) to find related entities and relationships
3. **Format KG triples** as structured context (e.g., `sirolimus --[INHIBITS]--> mTORC1`)
4. **Vector retrieval** in parallel for semantic context
5. **Combine** KG triples + vector chunks in the LLM prompt

**Best for:** Mechanism/relationship questions ("How does X affect Y?")
**Limitations:** KG quality depends on entity extraction accuracy
""")
            elif name == "HyDE RAG":
                st.markdown("""
1. **Generate hypothetical** passage: LLM writes what the ideal answer would look like
2. **Embed** the hypothetical passage (not the original query)
3. **Search** using the hypothetical passage's embedding
4. **Optional ensemble:** Average embeddings of N hypothetical docs for robustness
5. **Generate** the final answer from retrieved chunks

**Best for:** Precision-sensitive retrieval in specialized domains
**Limitations:** Extra LLM call for hypothesis generation; depends on LLM quality
""")
            elif name == "Self-RAG":
                st.markdown("""
Implements 4 self-reflection checkpoints:
1. **RETRIEVE?** – LLM decides if retrieval is needed
2. **ISREL?** – Per-chunk relevance judgment
3. **ISSUP?** – Per-segment supportedness verification
4. **ISUSE?** – Final utility score (1-5)

Filters chunks that are irrelevant or unsupported before synthesis.

**Best for:** Reducing hallucinations, complex questions with risk of wrong answers
**Limitations:** Higher latency due to multiple LLM judge calls
""")
            elif name == "Multi-hop RAG":
                st.markdown("""
1. **Decompose** the complex question into the most important sub-question
2. **Retrieve** for that sub-question (hop 1)
3. **Partially answer** → accumulate evidence
4. **Check sufficiency** → if not sufficient, go to hop 2
5. **Repeat** for max_hops iterations
6. **Synthesize** all partial answers into a final coherent answer

**Best for:** Multi-faceted questions requiring chained reasoning
**Limitations:** Highest latency; N×K LLM calls per query
""")

    st.divider()
    st.subheader("When to Use Each Pipeline")

    guidance = {
        "Question Type": [
            "Simple fact lookup",
            "Drug mechanism",
            "Relationship between entities",
            "Complex multi-part question",
            "High-stakes clinical question",
            "Speed-critical application",
        ],
        "Recommended Pipeline": [
            "Basic RAG",
            "Knowledge Graph RAG",
            "Knowledge Graph RAG",
            "Multi-hop RAG",
            "Self-RAG",
            "Basic RAG or HyDE RAG",
        ],
        "Reasoning": [
            "Fast, sufficient for direct lookups",
            "KG triples capture drug-target relationships",
            "KG explicitly represents entity connections",
            "Iterative retrieval builds up evidence",
            "Self-reflection reduces hallucination risk",
            "Fewer LLM calls = lower latency",
        ],
    }
    import pandas as pd
    st.dataframe(pd.DataFrame(guidance), use_container_width=True)
