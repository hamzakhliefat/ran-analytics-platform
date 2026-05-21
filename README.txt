# RAN Network KPI Analysis Platform  

## Network Data Scientist Engineering Case Study

### 🎯 Project Overview

A comprehensive full-stack data science platform for analyzing RAN (Radio Access Network) KPIs, identifying performance issues, and applying machine learning techniques for predictive insights and root cause analysis.

**Dataset:** >1.4M rows of network KPI data  
**Technologies:** Python, Streamlit, scikit-learn, TensorFlow, Ollama, ChromaDB

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- 8GB+ RAM (16GB recommended for LLM/RAG features)
- Windows/Linux/macOS

### Installation

1. **Clone/Navigate to project directory:**
   ```bash
   cd "c:\Users\hamza\RAN Analytics Platform"
   ```

2. **Create virtual environment (recommended):**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
### Data Setup

4. **Build database:**
   ```bash
   python scripts/build_db.py
   ```
   
   This will:
   - Load raw data
   - Clean and transform
   - Store in SQLite database (`db/ran_kpis.sqlite`)

### Run Application

5. **Launch Streamlit app:**
   ```bash
   streamlit run app/Home.py
   ```

6. **Open browser:**  
   Navigate to `http://localhost:8501`

---

## ✨ Features Implementation

### ✅ Core Requirements (All Completed)

#### 1. **Data Handling & Cleansing**
- ✅ Load large datasets (>1.5M rows) from Excel/CSV
- ✅ Clean data: handle missing values, outliers, duplicates
- ✅ SQLite database for efficient querying
- ✅ Pandas/NumPy data processing

#### 2. **Analysis & Visualization**
- ✅ Summary statistics by site_id and gov
- ✅ Correlation heatmaps for KPIs
- ✅ Identify underperforming cells (HO_SR < 95%)
- ✅ Root cause analysis (PRB utilization → congestion)
- ✅ Professional visualizations (Plotly, Seaborn, Matplotlib)

#### 3. **ML Modeling**
- ✅ **3+ Regression Models:** LinearRegression, ElasticNet, GradientBoosting, RandomForest
- ✅ **3+ Classification Models:** LogisticRegression, GradientBoosting, RandomForest
- ✅ Predict throughput, PRB utilization, RRC users, HO_SR
- ✅ Anomaly detection for drop rates
- ✅ 80/20 train/test split (explicitly shown in UI)
- ✅ Cross-validation (5-fold)
- ✅ Model comparison: MAE, RMSE, R², F1, Precision, Recall, ROC-AUC

#### 4. **Model Explainability**
- ✅ SHAP analysis for all models
- ✅ Feature importance visualizations
- ✅ Beeswarm plots, dependence plots

#### 5. **Full-Stack Platform**
- ✅ Streamlit multi-page application
- ✅ Interactive data exploration
- ✅ Real-time model training
- ✅ Visual explanations

### Bonus Features (All Completed)

#### 1. **Deep Learning (LSTM)**
- ✅ Time-series forecasting using TensorFlow LSTM
- ✅ 7/14/30-day throughput predictions
- ✅ Baseline vs LSTM comparison

#### 2. **LLM Integration**
- ✅ **Ollama** integration for local LLM inference
- ✅ Support for Llama 2, Mistral, and Phi-2 models
- ✅ Cell performance analysis and root cause diagnostics
- ✅ Parameter optimization recommendations

#### 3. **RAG System**
- ✅ ChromaDB vector database
- ✅ Sentence-transformers embeddings
- ✅ Semantic search over KPI data
- ✅ Natural language Q&A: "Why is cell_id X underperforming?"

#### 4. **Scalability**
- ✅ Dask for parallel processing
- ✅ Architecture documentation
- ✅ Cloud deployment guidelines

---


## 📊 Application Pages

### 1. **Home** 
- Database connection status
- Quick overview

### 2. **Data Explorer**
- Dataset statistics
- Data quality metrics
- Missing value analysis
- Outlier detection
- Distribution visualizations

### 3. **KPI Analysis**
- Summary statistics by site/gov
- Correlation heatmaps
- Underperforming cells identification
- Trend analysis (single/multi-cell)
- KPI relationship explorer
- Before/after degradation analysis
- Rule-based root cause evidence

### 4. **ML Lab**  
- **Regression:** Predict throughput, PRB, RRC users, HO_SR
- **Classification:** Anomaly detection
- Train 3+ models with comparison
- Train/test split visualization
- SHAP explanations
- **LSTM Forecasting** (Deep Learning bonus)

### 5. **Root Cause Analysis**
- Identify performance degradation
- Correlation-based drivers
- Model-based feature importance

### 6. **LLM Insights** 
- AI-powered network analysis
- Cell performance insights
- Anomaly explanations
- Optimization recommendations

### 7. **RAG Q&A**  
- Natural language queries
- Semantic vector search
- LLM-generated answers
- Context retrieval

---

## 🏗️ Project Structure

```
RAN Analytics Platform - Hamza Khlefat/
├── app/
│   ├── Home.py                  # Main entry point
│   └── pages/
│       ├── 1_Data_Explorer.py   # Data quality & exploration
│       ├── 2_KPI_Analysis.py    # KPI analysis & visualization
│       ├── 3_ML_Lab.py          # ML training & LSTM
│       ├── 4_Root_Cause.py      # Root cause analysis
│       ├── 5_LLM_Insights.py    # LLM-powered insights
│       └── 6_RAG_Query.py       # RAG Q&A system
├── src/
│   ├── clean.py                 # Data cleaning utilities
│   ├── config.py                # Configuration
│   ├── db.py                    # Database utilities
│   ├── models_regression.py     # Regression models
│   ├── models_anomaly.py        # Classification models
│   ├── forecasting.py           # LSTM time-series
│   ├── llm_insights.py          # LLM integration
│   ├── rag_system.py            # RAG system
│   ├── rca.py                   # Root cause analysis
│   └── viz_trends.py            # Visualization helpers
├── scripts/
│   └── build_db.py              # Database builder
├── data/
│   └── raw/                     # Place dataset here
├── db/
│   └── ran_kpis.sqlite          # Generated database
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

##   Key Technologies

| Category | Technologies |
|----------|-------------|
| **Data Processing** | Pandas, NumPy, SQLite, Dask |
| **ML/DL** | scikit-learn, TensorFlow, SHAP |
| **LLM/NLP** | **Ollama** (Llama 2, Mistral, Phi-2), sentence-transformers |
| **Vector DB** | ChromaDB |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Web App** | Streamlit |

---

##   Model Performance Summary

### Regression Models
- **Best Model:** GradientBoosting / RandomForest (typically)
- **Metrics:** MAE, RMSE, R²
- **Cross-Validation:** 5-fold

### Classification Models
- **Best Model:** RandomForest (balanced classes)
- **Metrics:** F1, Precision, Recall, ROC-AUC
- **Imbalance Handling:** Class weighting, stratified sampling

### LSTM Forecasting
- **Architecture:** LSTM(16) → Dense(1)
- **Lookback:** 30 days
- **Epochs:** 10 (configurable)

---

##   Key Insights

1. **High PRB Utilization** → strong negative correlation with throughput (congestion)
2. **Low HO Success Rate** → correlated with poor RF coverage or neighbor issues
3. **Drop Rate Anomalies** → often cluster with high PRB and low throughput
4. **Temporal Patterns:** Weekly and daily cycles evident in throughput data

---

##   Scalability

- **Current:** Handles 1.5M+ rows efficiently
- **Scaling:** Dask integration for distributed processing
- **Cloud Deployment:** Ready for AWS/Azure deployment
- See `docs/SCALABILITY.md` for detailed architecture

---

##   Author

**Hamza Khlefat | Contact: +962799693068 | Email: hamzakhliefat@yahoo.com**  

---


