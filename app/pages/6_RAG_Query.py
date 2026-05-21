# -*- coding: utf-8 -*-
"""
Created on Sat Jan 24 10:40:33 2026

@author: hamza khlefat
"""

"""
Retrieval-Augmented Generation (RAG) Diagnostic Interface
Neural search and Large Language Model (LLM) integration for natural language network analysis.
"""

import os
import sys

# Standard module path resolution
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
import pandas as pd

from src.config import Cols, DB
from src.db import read_sql
from src.rag_system import get_rag_system
from src.llm_insights import get_insights_generator

st.set_page_config(page_title="Neural Diagnostic Query", layout="wide")
st.title("Neural-Augmented Diagnostic Query System")

st.markdown("""
### Advanced Natural Language Interface
Submit complex operational queries in plain language. The diagnostic engine implements:
1. **Semantic Retrieval:** High-dimensional vector search to isolate relevant telemetry subsets.
2. **Contextual Synthesis:** Large Language Model (LLM) reasoning to generate technical insights from retrieved data.
""")

cols = Cols()
db = DB()

# ============================================================================
# SYSTEM CONFIGURATION (SIDEBAR)
# ============================================================================
st.sidebar.header("Engine Configuration")

use_vector_search = st.sidebar.checkbox(
    "Enable Neural Vector Search",
    value=True,
    help="Utilize semantic embeddings for context retrieval. Disabling reverts to keyword matching."
)

use_llm = st.sidebar.checkbox(
    "Enable LLM Response Generation",
    value=True,
    help="Utilize LLM to synthesize natural language conclusions from retrieved telemetry."
)

n_context_docs = st.sidebar.slider("Retrieval Depth (Context Samples)", 1, 10, 3)

if use_vector_search:
    st.sidebar.info("Methodology: Neural Semantic Retrieval")
    st.sidebar.caption("Embedding Model: all-MiniLM-L6-v2")
else:
    st.sidebar.warning("Methodology: Deterministic Keyword Search")

# ============================================================================
# INITIALIZATION LOGIC
# ============================================================================
@st.cache_resource
def load_rag_system(use_vector):
    """Initialize neural search engine components."""
    return get_rag_system(use_vector_db=use_vector)

@st.cache_resource
def load_llm_for_rag():
    """Initialize generative synthesis model."""
    return get_insights_generator(use_ollama=True, model_name="llama2")

with st.spinner("Initializing neural components..."):
    try:
        rag = load_rag_system(use_vector_search)
        st.success("Retrieval System: Operational")
    except Exception as e:
        st.error(f"System Error: Initialization failure - {e}")
        st.stop()

# LLM Component Provisioning
llm_gen = None
if use_llm:
    with st.spinner("Provisioning generative engine..."):
        try:
            llm_gen = load_llm_for_rag()
        except Exception as e:
            st.warning(f"Generative Service Unavailable: {e}. Reverting to raw retrieval mode.")

# ============================================================================
# DATA INGESTION & NEURAL INDEXING
# ============================================================================
@st.cache_data
def load_data_for_rag(n=50000):
    """Fetch randomized telemetry subset for system indexing."""
    return read_sql(db.path, f"SELECT * FROM {db.table} ORDER BY RANDOM() LIMIT ?", (n,))

sample_size = st.sidebar.slider("Indexing Sample Volume", 5000, 1500000, 30000, 5000)

# Dataset Retrieval
df = load_data_for_rag(sample_size)

if df is None or df.empty:
    st.error("Data Source Error: Unable to retrieve records for indexing.")
    st.stop()

# Vector Space Mapping
@st.cache_data
def index_data(_rag_system, _df, cell_col, kpi_list, max_docs):
    """Execute high-dimensional mapping of telemetry data."""
    return _rag_system.index_dataframe(
        df=_df,
        cell_col=cell_col,
        kpi_cols=kpi_list,
        max_docs=max_docs
    )

kpi_cols = [cols.throughput, cols.prb_dl, cols.ho_sr, cols.drop_rate, cols.rrc_users_max,
            cols.traffic_dl_gb, cols.harq_ack_64qam]
kpi_cols = [c for c in kpi_cols if c in df.columns]

with st.spinner("Executing Neural Mapping (Dimensional Indexing)..."):
    try:
        index_info = index_data(rag, df, "cell_id", kpi_cols, max_docs=min(5000, len(df)))
        st.success(f"Indexing Complete: {index_info['indexed_docs']} operational profiles mapped.")
    except Exception as e:
        st.error(f"Indexing Error: Neural mapping failed - {e}")
        st.stop()

# ============================================================================
# OPERATIONAL QUERY INTERFACE
# ============================================================================
st.divider()
st.header("Diagnostic Query Form")

st.markdown("**Standard Query Templates:**")

examples = [
    "Identify root causes for underperformance in cell_id 12345.",
    "Which sectors exhibit abnormal PRB utilization density?",
    "Cross-reference low throughput sectors with high call drop rate.",
    "Retrieve cells with mobility performance (HO_SR) below 95%.",
    "Identify geographic clusters with capacity saturation.",
    "Rank entities with the highest composite performance impact.",
]

example_cols = st.columns(3)
for i, ex in enumerate(examples):
    col = example_cols[i % 3]
    if col.button(f"Template: {ex}", key=f"ex_{i}"):
        st.session_state["rag_question"] = ex

# Query Input Field
default_question = st.session_state.get("rag_question", "")
question = st.text_input(
    "Operational Query Input:",
    value=default_question,
    placeholder="e.g., Conduct a performance audit for cell identifier X.",
    key="user_question"
)

# ============================================================================
# INFERENCE EXECUTION
# ============================================================================
if st.button("Execute Neural Inference", type="primary", disabled=not question):
    if question:
        with st.spinner("Processing neural search and generative synthesis..."):
            try:
                # Query inference
                result = rag.rag_query(
                    question=question,
                    llm_generator=llm_gen if use_llm else None,
                    n_context=n_context_docs
                )
                
                st.divider()
                
                # Inference Output
                st.subheader("Diagnostic Conclusion")
                st.markdown(result["answer"])
                
                st.divider()
                
                # Context Evidence Review
                st.subheader(f"Retrieved Evidence Context ({len(result['context'])} Artifacts)")
                
                for i, doc_info in enumerate(result["context"], 1):
                    with st.expander(f"Telemetry Artifact {i}" + 
                                   (f" (Semantic Distance: {doc_info.get('distance', 0):.4f})" if 'distance' in doc_info else "")):
                        st.code(doc_info["document"], language="text")
                        
                        if "metadata" in doc_info:
                            st.json(doc_info["metadata"])
                
                st.divider()
                
                # Source Identity
                st.subheader("Data Lineage & Primary Sources")
                
                if result["sources"]:
                    sources_df = pd.DataFrame(result["sources"])
                    st.dataframe(sources_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"Inference Failure: {e}")
                import traceback
                st.code(traceback.format_exc())

# ============================================================================
# SYSTEM STATUS TELEMETRY
# ============================================================================
st.divider()
st.caption(
    f"Inference Mode: {'Neural Vector Search' if use_vector_search else 'Keyword Match'} | "
    f"Synthesis Engine: {'Operational' if use_llm else 'Deactivated'} | "
    f"Active Mapping Density: {index_info.get('indexed_docs', 0)} documents"
)
