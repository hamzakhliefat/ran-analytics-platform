# -*- coding: utf-8 -*-
"""
Created on Sun Jan 25 13:15:00 2026

@author: hamza khlefat
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Resolve root directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.config import Cols, DB
from src.db import read_sql

st.set_page_config(page_title="Strategic Simulation", layout="wide")
st.title("Strategic 'What-If' Simulation Canvas")

st.markdown("""
### Analytical Sensitivity Interface
This module enables engineers to simulate the impact of parameter adjustments on key performance objectives. 
By utilizing the optimal predictive models identified in the **ML Lab**, we can forecast the outcome of strategic 
network changes before deployment.
""")

def human_format(num, base_unit="bps"):
    """Format large numbers into human-readable strings with logical unit transitions."""
    magnitude = 0
    if base_unit in ["bps", "Mbps", "kbps"]:
        units = ["bps", "Kbps", "Mbps", "Gbps", "Tbps"]
    else:
        units = [base_unit, f"K{base_unit}", f"M{base_unit}", f"G{base_unit}"]

    while abs(num) >= 1000 and magnitude < len(units) - 1:
        magnitude += 1
        num /= 1000.0
    
    return f"{num:.2f} {units[magnitude]}".strip()

# ============================================================================
# STATE VERIFICATION
# ============================================================================
ml_pack = st.session_state.get("ml_pack")
ml_task = st.session_state.get("ml_task")

if not ml_pack:
    st.warning("Prediction Engine Unavailable: No optimized model detected in memory.")
    st.info("Resolution: Navigate to the **ML Lab** and execute a training cycle for a Regression target to enable simulation.")
    st.stop()

# ============================================================================
# SIMULATION CONFIGURATION
# ============================================================================
target = ml_task.split(" (")[0] if ml_task else "Target Metric"
model = ml_pack["best_model"]
features = ml_pack["feature_names"]
X_train = ml_pack["X_train"]

st.sidebar.header("Simulation Parameters")
st.sidebar.info(f"Active Model: {ml_pack['best_model_name']}")

# Initialize simulation inputs with baseline medians
sim_inputs = {}
baseline_values = X_train.median()

st.sidebar.subheader("Adjust Operational Variables")
for feat in features:
    val = float(baseline_values[feat])
    # Define bounds based on observed data distribution
    f_min = float(X_train[feat].min())
    f_max = float(X_train[feat].max())
    # Add 20% margin for "what-if" scenarios
    f_min_adj = f_min * 0.8 if f_min >= 0 else f_min * 1.2
    f_max_adj = f_max * 1.2
    
    sim_inputs[feat] = st.sidebar.slider(
        f"Simulated {feat}",
        min_value=f_min_adj,
        max_value=f_max_adj,
        value=val,
        format="%.4f"
    )

# ============================================================================
# INFERENCE ENGINE
# ============================================================================
input_df = pd.DataFrame([sim_inputs])
prediction = float(model.predict(input_df)[0])

# Baseline Prediction (using medians)
baseline_pred = float(model.predict(pd.DataFrame([baseline_values]))[0])
delta = prediction - baseline_pred
delta_pct = (delta / baseline_pred * 100) if baseline_pred != 0 else 0

# ============================================================================
# VISUALIZATION
# ============================================================================
st.divider()
st.header("Simulation Results & Impact Analysis")

col1, col2 = st.columns([1, 1])

with col1:
    # Scale values for visual consistency in Gauge
    gauge_val, gauge_unit = prediction, ""
    gauge_baseline = baseline_pred
    
    if "Throughput" in target:
        # Determine magnitude based on prediction
        magnitude = 0
        units = ["bps", "Kbps", "Mbps", "Gbps", "Tbps"]
        temp_val = prediction
        while abs(temp_val) >= 1000 and magnitude < len(units) - 1:
            magnitude += 1
            temp_val /= 1000.0
        
        gauge_unit = units[magnitude]
        div = 1000.0 ** magnitude
        gauge_val = prediction / div
        gauge_baseline = baseline_pred / div

    # Gauge Chart for Prediction
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = gauge_val,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Simulated {target} ({gauge_unit})" if gauge_unit else f"Simulated {target}"},
        delta = {'reference': gauge_baseline, 'increasing': {'color': "green" if "Throughput" in target or "Success" in target else "red"}},
        number = {'suffix': f" {gauge_unit}", 'valueformat': '.2f'},
        gauge = {
            'axis': {'range': [None, max(gauge_val, gauge_baseline) * 1.5], 'tickformat': '.2s'},
            'bar': {'color': "#3498db"},
            'steps': [
                {'range': [0, gauge_baseline], 'color': "#f1f2f6"},
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': gauge_baseline
            }
        }
    ))
    
    fig.update_layout(height=400, margin=dict(l=30, r=30, t=50, b=30))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Comparative Diagnostics")
    
    # Determine display delta
    display_delta = delta
    if "Throughput" in target:
        display_delta = delta / div # Re-use the divisor from the gauge logic

    st.metric(
        label=f"Projected {target}",
        value=human_format(prediction, "bps") if "Throughput" in target else f"{prediction:.4f}",
        delta=f"{display_delta:.2f} {gauge_unit} ({delta_pct:.2f}%)" if "Throughput" in target else f"{delta:.4f} ({delta_pct:.2f}%)",
        delta_color="normal"
    )
    
    st.markdown(f"""
    **Scenario Summary:**
    Adjusting the operational parameters from their network medians to the simulated values results in a 
    **{abs(delta_pct):.2f}% {"increase" if delta > 0 else "decrease"}** in projected {target}.
    
    **Baseline Context (Medians):**
    - Current predicted performance: `{human_format(baseline_pred, "bps") if "Throughput" in target else f"{baseline_pred:.4f}"}`
    - Analysis based on `{ml_pack['best_model_name']}` architecture.
    """)
    
    if st.button("Reset to Network Baselines", use_container_width=True):
        st.rerun()

# ============================================================================
# SENSITIVITY ANALYSIS (Linear Sweep)
# ============================================================================
st.divider()
st.header("Sensitivity Profile: Key Driver Impact")

selected_feat = st.selectbox("Select Target Variable for Sensitivity Sweep:", features)

# Create a range for the sweep
sweep_range = np.linspace(X_train[selected_feat].min(), X_train[selected_feat].max(), 50)
sweep_results = []

for val in sweep_range:
    temp_input = baseline_values.copy()
    temp_input[selected_feat] = val
    pred = model.predict(pd.DataFrame([temp_input]))[0]
    sweep_results.append(pred)

sweep_df = pd.DataFrame({
    selected_feat: sweep_range,
    f"Predicted {target}": sweep_results
})

# Determine scaling for plot
max_metric = max(max(sweep_results), prediction)
scale_factor = 1.0
unit_label = ""

if "Throughput" in target:
    if max_metric >= 1e9:
        scale_factor = 1e9
        unit_label = " (Gbps)"
    elif max_metric >= 1e6:
        scale_factor = 1e6
        unit_label = " (Mbps)"

fig_sweep = go.Figure()
fig_sweep.add_trace(go.Scatter(
    x=sweep_df[selected_feat],
    y=sweep_df[f"Predicted {target}"] / scale_factor,
    mode='lines',
    line=dict(color='#3498db', width=3),
    name=f"Sensitivity Curve{unit_label}"
))

# Highlight current simulated point
fig_sweep.add_trace(go.Scatter(
    x=[sim_inputs[selected_feat]],
    y=[prediction / scale_factor],
    mode='markers',
    marker=dict(size=12, color='#e74c3c', symbol='diamond'),
    name=f"Current Simulation Point{unit_label}"
))

fig_sweep.update_layout(
    title=f"Sensitivity Profile: Impact of {selected_feat} on {target}",
    xaxis_title=selected_feat,
    yaxis_title=f"Predicted {target}{unit_label}",
    height=450,
    margin=dict(l=30, r=30, t=50, b=30)
)

st.plotly_chart(fig_sweep, use_container_width=True)

st.divider()
st.caption(f"Engine Identification: {ml_pack['best_model_name']} | Model Source: ML Lab session state")
