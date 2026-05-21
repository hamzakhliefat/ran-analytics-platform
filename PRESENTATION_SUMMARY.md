# RAN Analytics Platform - Presentation Summary

## 1. Executive Summary
The **RAN Analytics Platform** is a comprehensive, full-stack data science solution designed to optimize Radio Access Network (RAN) performance. It leverages advanced Machine Learning (ML) and Generative AI (LLMs) to transform raw network telemetry into actionable strategic intelligence.

**Goal:** Automate failure detection, diagnose root causes of degradation, and predict future network triggers to ensure superior subscriber experience.

---

## 2. Key Findings & Insights
Through the analysis of over **1.4 Million** network KPI records, the platform identified critical performance drivers.

### 📍 Regional Performance (Example: Ajlun)
*   **Mean Throughput:** ~17.58 Gbps
*   **PRB Utilization:** ~39.2%
*   **Handover Success Rate (HO SR):** **14.1%** (Critical Issue)
    *   *Insight:* This extremely low handover rate in Ajlun points to severe mobility or neighbor definition issues that need immediate optimization.

### 📉 Performance Correlations
*   **Congestion Drivers:**
    *   **High PRB Utilization (>85%)** shows a strong negative trend with User Throughput (as seen in the Inter-KPI Relationship Explorer).
    *   **Action:** Capacity expansion or load balancing is required when PRB saturation is detected.

*   **Mobility & Coverage:**
    *   **Low Handover Success Rate (<95%)** is strongly correlated with poor RF coverage or misconfigured neighbor relations (NRT).
    *   **Action:** Audit neighbor relations and optimize A3 offset parameters.

*   **Quality of Service (QoS):**
    *   **Call Drop Rate Anomalies (>1%)** typically cluster with high PRB utilization and low throughput, suggesting that congestion is a leading cause of dropped calls, not just coverage holes.
    *   **Seasonality:** Throughput exhibits distinct weekly and daily cyclic patterns, allowing for accurate time-series forecasting.

---

## 3. Technical Methodology & Techniques
The platform integrates a modern "Neural-Augmented" architecture:

### A. Machine Learning (Predictive Modeling)
*   **Regression Models:**
    *   **Best Model:** **Random Forest** (Identified as Optimal Architecture).
    *   **Performance Metrics:**
        *   **R² Score:** ~0.80 (Explains 80% of throughput variance).
        *   **MAE:** ~283 Mbps (Mean Absolute Error).
    *   **Target:** Predicting User Throughput based on load and signal quality.
*   **Classification:**
    *   **Goal:** Anomaly Detection (identifying cells with abnormal drop rates).
    *   **Handling Imbalance:** Uses SMOTE/Class Weighting to handle rare failure events.
*   **Deep Learning (Forecasting):**
    *   **Model:** LSTM (Long Short-Term Memory) networks using TensorFlow/Keras.
    *   **Application:** 7 to 30-day time-series forecasting of network traffic.
*   **Explainability (XAI):**
    *   **SHAP (SHapley Additive exPlanations):** Used to demystify "Black Box" models, quantifying exactly how much feature (e.g., "Active Users") contributes to a prediction (e.g., "Low Throughput").

### B. Generative AI (GenAI) & NLP
*   **Local LLM Inference:** Integrates **Ollama (Llama 2, Mistral)** to generate human-readable technical audits for underperforming cells.
*   **RAG (Retrieval-Augmented Generation):**
    *   **Vector DB:** ChromaDB stores semantic embeddings of KPI data.
    *   **Function:** Allows engineers to ask natural language questions (e.g., *"Why is Cell X failing?"*) and retrieve context-aware answers.

### C. Data Engineering & Stack
*   **ETL Pipeline:** Pandas & NumPy for cleaning 1.5M+ rows; SQLite for production storage.
*   **Visualization:** Plotly & Streamlit for interactive, real-time dashboards.

---

## 4. Project Workflow (Step-by-Step)
The application follows a structured data science lifecycle, mirrored by the navigation:

1.  **Data Ingestion & Cleaning:**
    *   Loading raw counters (CSV/Excel).
    *   Cleaning missing values and handling outliers.
    *   Storing structured data in a local SQLite database.

2.  **Exploratory Data Analysis (EDA):**
    *   Statistical profiling of the network (Mean, Std Dev).
    *   Visualizing distributions to catch data quality issues early.

3.  **KPI Analysis & Diagnostics:**
    *   **Correlation Heatmaps:** Identifying relationships between metrics (e.g., Interference vs. Quality).
    *   **Trend Analysis:** Monitoring performance over time.

4.  **Machine Learning Lab (Training):**
    *   Splitting data into Training (80%) and Testing (20%) sets.
    *   Training multiple models in parallel.
    *   Evaluating performance using RMSE, MAE, R², and F1-Score.

5.  **Root Cause Analysis (RCA):**
    *   **Case Study (Site 70432):**
        *   **Dominant Driver:** `NPM_DL_PRB_Utilization` was identified by SHAP as the most significant contributor to performance degradation.
        *   **Recommendation:** Technical intervention focusing on stabilization of PRB load is recommended.

6.  **Strategic Simulation:**
    *   **"What-If" Scenarios:** Simulating changes to operational parameters.
    *   **Example Simulation:** Adjusting network parameters resulted in a projected throughput of **21.30 Gbps** (a **9.79% decrease** from baseline), helping engineers avoid detrimental config changes before deployment.

7.  **Executive Reporting:**
    *   High-level dashboards summarizing network health for management.

---

## 5. Future Roadmap & Scalability
*   **Scalability:** Implementation of **Dask** for distributed computing to handle nationwide datasets (10M+ rows).
*   **Cloud Ready:** Architecture designed for easy deployment to AWS/Azure.
*   **Automation:** Full loop automation where the system auto-tunes network parameters based on ML predictions.
