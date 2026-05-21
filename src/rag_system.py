# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Created on Sat Jan 24 14:55:22 2026

@author: hamza khlefat
"""

"""
Retrieval-Augmented Generation (RAG) Engine for Network KPI Analysis
Integrates high-dimensional vector similarity search with Large Language Models (LLMs) for technical diagnostics.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import json


class RAGSystem:
    """Neural retrieval engine for high-dimensional telemetry analysis."""
    
    def __init__(self, embedding_model="all-MiniLM-L6-v2"):
        """
        Initialize the neural retrieval engine.
        
        Args:
            embedding_model: Pre-trained sentence transformer model identifier.
        """
        self.embedding_model_name = embedding_model
        self.embedder = None
        self.vector_db = None
        self.documents = []
        self._initialized = False
    
    def _lazy_load(self):
        """Coordinate lazy loading of neural embedding models and vector infrastructure."""
        if self._initialized:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            import chromadb
            
            # Initialization of embedding transformer
            self.embedder = SentenceTransformer(self.embedding_model_name)
            
            # Provisioning of ephemeral vector database instance
            client = chromadb.Client()
            self.vector_db = client.create_collection(
                name="kpi_knowledge",
                metadata={"hnsw:space": "cosine"}
            )
            
            self._initialized = True
        except Exception as e:
            raise RuntimeError(f"RAG System Initialization Failure: {e}")
    
    def index_dataframe(self, df: pd.DataFrame, cell_col="cell_id", 
                       kpi_cols: List[str] = None, max_docs: int = 5000):
        """
        Index operational telemetry for high-dimensional retrieval.
        Constructs semantic profiles from cell-level longitudinal statistics.
        """
        self._lazy_load()
        
        if cell_col not in df.columns:
            raise ValueError(f"Spatial Identifier Error: Column '{cell_col}' not detected in dataset.")
        
        if kpi_cols is None:
            # Default to all identified numerical attributes
            kpi_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Aggregate longitudinal telemetry by spatial identifier
        agg_dict = {col: ['mean', 'std', 'min', 'max'] for col in kpi_cols if col in df.columns}
        cell_stats = df.groupby(cell_col).agg(agg_dict)
        
        # Normalization of hierarchical column structure
        cell_stats.columns = ['_'.join(col).strip() for col in cell_stats.columns.values]
        cell_stats = cell_stats.reset_index()
        
        # Dimensionality control: Sampling volume reduction
        if len(cell_stats) > max_docs:
            cell_stats = cell_stats.sample(n=max_docs, random_state=42)
        
        # Documentary construct preparation
        documents = []
        metadatas = []
        ids = []
        
        for idx, row in cell_stats.iterrows():
            cell_id = row[cell_col]
            
            # Semantic profile construction
            doc_text = f"Cell {cell_id} Performance Narrative:\n"
            
            for col in cell_stats.columns:
                if col != cell_col:
                    doc_text += f"{col}: {row[col]:.4g}\n"
            
            documents.append(doc_text)
            
            # Metadata association
            metadata = {cell_col: str(cell_id)}
            for col in cell_stats.columns:
                if col != cell_col:
                    metadata[col] = float(row[col]) if not pd.isna(row[col]) else 0.0
            
            metadatas.append(metadata)
            ids.append(f"cell_{cell_id}_{idx}")
        
        # Neural embedding generation and vector space ingestion
        embeddings = self.embedder.encode(documents, show_progress_bar=True)
        
        self.vector_db.add(
            documents=documents,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=ids
        )
        
        self.documents = documents
        
        return {"indexed_docs": len(documents), "kpi_cols": kpi_cols}
    
    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """
        Execute semantic proximity search against the high-dimensional index.
        
        Returns:
            Collection of retrieved artifacts with associated semantic distance metrics.
        """
        self._lazy_load()
        
        if not self.documents:
            raise ValueError("Indexing Requirement: Document repository is empty. Execute index_dataframe() before retrieval.")
        
        # Query transformation into vector space
        query_embedding = self.embedder.encode([query])[0]
        
        # Vector space heuristic lookup
        results = self.vector_db.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(n_results, len(self.documents))
        )
        
        # Object mapping and normalization
        formatted = []
        for i in range(len(results['documents'][0])):
            formatted.append({
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })
        
        return formatted
    
    def rag_query(self, question: str, llm_generator=None, n_context: int = 3) -> Dict:
        """
        End-to-end RAG pipeline: Neural retrieval followed by generative synthesis.
        """
        # Retrieval of semantically pertinent context
        context_docs = self.search(question, n_results=n_context)
        
        if not context_docs:
            return {
                "answer": "Diagnostic Limitation: No semantically relevant telemetry artifacts detected for the specified query.",
                "context": [],
                "sources": []
            }
        
        # Context block synthesis
        context_text = "\n\n".join([
            f"Evidence Artifact {i+1}:\n{doc['document']}"
            for i, doc in enumerate(context_docs)
        ])
        
        # Generative inference execution
        if llm_generator is not None:
            prompt = f"""Role: AI Diagnostic Engineer.
Objective: Synthesize a technical response using the provided operational context.

Target Query: {question}

Retrieved Telemetry Evidence:
{context_text}

Task: Formulate a technical diagnostic summary based strictly on the empirical data provided above."""
            
            try:
                answer = llm_generator.generate_text(prompt, max_tokens=400)
            except Exception as e:
                answer = f"Inference failure during synthetic generation: {e}\n\nReverting to raw evidence retrieval."
        else:
            # Fallback for retrieval-only mode
            answer = "Synthesis engine inactive. Reverting to raw operational telemetry artifacts."
        
        return {
            "answer": answer,
            "context": context_docs,
            "sources": [doc['metadata'] for doc in context_docs]
        }


class SimpleKeywordRAG:
    """Fallback: Deterministic keyword-based retrieval (utilized when neural infrastructure is unavailable)."""
    
    def __init__(self):
        self.documents = []
        self.metadatas = []
    
    def index_dataframe(self, df: pd.DataFrame, cell_col="cell_id", 
                       kpi_cols: List[str] = None, max_docs: int = 5000):
        """Execute deterministic keyword-based indexing."""
        
        if kpi_cols is None:
            kpi_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Aggregate telemetry by spatial identifier
        agg_dict = {col: 'mean' for col in kpi_cols if col in df.columns}
        cell_stats = df.groupby(cell_col).agg(agg_dict).reset_index()
        
        if len(cell_stats) > max_docs:
            cell_stats = cell_stats.sample(n=max_docs, random_state=42)
        
        # Document construct preparation
        for _, row in cell_stats.iterrows():
            doc = f"Cell {row[cell_col]} Profile: " + ", ".join([
                f"{col}={row[col]:.4g}" for col in kpi_cols if col in row.index
            ])
            
            self.documents.append(doc)
            self.metadatas.append(row.to_dict())
        
        return {"indexed_docs": len(self.documents)}
    
    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """Execute keyword-based matching heuristics."""
        
        query_lower = query.lower()
        keywords = query_lower.split()
        
        # Heuristic scoring based on term frequency
        scores = []
        for doc in self.documents:
            doc_lower = doc.lower()
            score = sum(1 for kw in keywords if kw in doc_lower)
            scores.append(score)
        
        # Strategic subset retrieval
        top_indices = np.argsort(scores)[::-1][:n_results]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Matching requirement
                results.append({
                    'document': self.documents[idx],
                    'metadata': self.metadatas[idx],
                    'score': scores[idx]
                })
        
        return results
    
    def rag_query(self, question: str, llm_generator=None, n_context: int = 3) -> Dict:
        """Standard query interface for deterministic keyword retrieval."""
        
        context_docs = self.search(question, n_results=n_context)
        
        if not context_docs:
            return {
                "answer": "Query Result: No operational profiles matched the specified keyword tokens.",
                "context": [],
                "sources": []
            }
        
        # Baseline report construction
        answer = "Keyword Analysis: The following telemetry profiles exhibit the highest term correlation with your query:\n\n"
        for i, doc in enumerate(context_docs, 1):
            answer += f"{i}. {doc['document']}\n"
        
        return {
            "answer": answer,
            "context": context_docs,
            "sources": [doc['metadata'] for doc in context_docs]
        }


def get_rag_system(use_vector_db: bool = True, embedding_model: str = "all-MiniLM-L6-v2"):
    """
    Factory function for retrieval system provisioning.
    
    Args:
        use_vector_db: Enable high-dimensional neural search.
        embedding_model: Identifier for targeted neural embedding model.
    """
    if use_vector_db:
        try:
            return RAGSystem(embedding_model=embedding_model)
        except Exception as e:
            # Revert to deterministic fallback upon neural component failure
            return SimpleKeywordRAG()
    else:
        return SimpleKeywordRAG()
