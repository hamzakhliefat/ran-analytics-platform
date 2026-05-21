# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 10:30:55 2026

@author: hamza khlefat
"""

from __future__ import annotations

"""
Large Language Model (LLM) Integration for Network KPI Analysis
Implements local inference using Ollama for automated performance diagnostics.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class OllamaInsightsGenerator:
    """Neural inference engine for generating network performance insights."""
    
    def __init__(self, model_name="llama2"):
        """
        Initialize the neural inference engine.
        
        Args:
            model_name: Identifier for the Ollama model (e.g., llama2, mistral).
        """
        self.model_name = model_name
        self._initialized = False
    
    def _lazy_load(self):
        """Coordinate lazy loading of neural components and verify Ollama connectivity."""
        if self._initialized:
            return
        
        try:
            import ollama
            # Connectivity verification
            ollama.list()
            self._initialized = True
        except Exception as e:
            raise RuntimeError(
                f"Ollama Infrastructure Unavailable: {e}\n\n"
                "Installation Requirements:\n"
                "1. Download binaries from https://ollama.ai\n"
                "2. Execute system installation\n"
                "3. Provision model: 'ollama pull llama2'"
            )
    
    def generate_text(self, prompt: str, max_tokens: int = 500) -> str:
        """Execute text generation against the local LLM endpoint."""
        self._lazy_load()
        
        try:
            import ollama
            
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'temperature': 0.7,
                    'num_predict': max_tokens,
                }
            )
            
            return response['response'].strip()
        
        except Exception as e:
            return f"Inference Error: {e}"
    
    def analyze_cell_performance(self, cell_stats: Dict) -> str:
        """Execute a technical audit of cell-level performance metrics."""
        
        prompt = f"""Role: Senior RAN Optimization Engineer.
Context: Conduct a professional technical analysis of the following cell performance profile.

Cell Identifier: {cell_stats.get('cell_id', 'Unknown')}
Throughput Magnitude: {cell_stats.get('throughput', 0):.2f} Mbps
PRB Utilization Density: {cell_stats.get('prb_util', 0):.2%}
Handover Success Rate: {cell_stats.get('ho_sr', 0):.2%}
Observed Drop Rate: {cell_stats.get('drop_rate', 0):.4f}

Objective: Identify performance impairments and derive potential root causes.
Focus areas:
1. Resource Saturation and Capacity Constraints (PRB Utilization)
2. RF Interface Integrity and Mobility Performance (HO_SR)
3. Quality of Service (QoS) Degradation (Drop Rate)

Provide 3-4 professional technical recommendations."""
        
        return self.generate_text(prompt, max_tokens=400)
    
    def explain_anomaly(self, cell_id: str, kpi_name: str, kpi_value: float, 
                       threshold: float, context: Dict) -> str:
        """Analyze and explain a specific KPI anomaly event."""
        
        prompt = f"""Role: Telecom Domain Expert.
Objective: Analyze a performance anomaly in the RAN environment.

Entity: {cell_id}
Anomaly Event: {kpi_name} = {kpi_value:.4f} (Threshold: {threshold:.4f})

Supporting Context:
- PRB Utilization: {context.get('prb', 0):.2%}
- Active Subscribers: {context.get('users', 0)}
- Aggregate Payload: {context.get('traffic', 0):.2f} GB

Task: Evaluate the probable root causes for this {kpi_name} deviation. Provide technical rationale."""
        
        return self.generate_text(prompt, max_tokens=300)
    
    def generate_summary_insights(self, df: pd.DataFrame, kpi_cols: List[str]) -> str:
        """Generate high-level network health summaries from aggregate data."""
        
        # Aggregate statistical computation
        stats = []
        for col in kpi_cols:
            if col in df.columns:
                stats.append(f"- {col}: Mean={df[col].mean():.4f}, StdDev={df[col].std():.4f}")
        
        stats_text = "\n".join(stats)
        
        prompt = f"""Role: Principal Network Analyst.
Objective: Aggregate Performance Summary for the RAN Network.

Population Size: {len(df)} operational units
Statistical Profile:
{stats_text}

Task: Provide 4-5 strategic insights regarding network health and stability.
Primary dimensions:
1. Aggregate Network Integrity
2. Recurring Performance Patterns
3. High-Risk Operational Areas
4. Strategic Optimization Priorities"""
        
        return self.generate_text(prompt, max_tokens=500)
    
    def suggest_optimization(self, bottleneck_type: str, severity: str, 
                             affected_cells: int) -> str:
        """Formulate technical optimization strategies for identified bottlenecks."""
        
        prompt = f"""Role: Principal RAN Optimization Architect.
Objective: Formulation of Technical Mitigation Strategies.

Identified Impairment:
- Category: {bottleneck_type}
- Criticality: {severity}
- Observed Impact: {affected_cells} operational units

Objective: Provide specific technical countermeasures across temporal horizons.
Structure recommendations by:
1. Immediate Mitigation (Operational Quick Wins)
2. Intermediate Optimization (Tactical Parameter Tuning)
3. Strategic Resolution (Capacity Expansion/Infrastructure Alignment)

Focus on RAN parameters, threshold adjustment, and structural configuration."""
        
        return self.generate_text(prompt, max_tokens=600)


class SimpleRuleBasedInsights:
    """Fallback: Deterministic rule-based diagnostics (utilized when neural endpoints are unavailable)."""
    
    @staticmethod
    def analyze_cell_performance(cell_stats: Dict) -> str:
        """Execute heuristic analysis of cell-level performance metrics."""
        issues = []
        
        prb = cell_stats.get('prb_util', 0)
        ho_sr = cell_stats.get('ho_sr', 1.0)
        drop_rate = cell_stats.get('drop_rate', 0)
        throughput = cell_stats.get('throughput', 0)
        
        if prb > 0.85:
            issues.append(f"Critical: High PRB Utilization ({prb:.1%}) - Resource Saturation Detected. Recommendations:\n"
                         "  • Implement inter-frequency/inter-RAT load balancing\n"
                         "  • Evaluate multi-carrier aggregation strategies\n"
                         "  • Review capacity expansion requirements (additional carrier/sector deployment)")
        
        if ho_sr < 0.95:
            issues.append(f"Critical: Low Handover Success Rate ({ho_sr:.1%}) - Mobility Impairment Detected. Recommendations:\n"
                         "  • Audit Neighbor Relation Table (NRT) for inconsistencies\n"
                         "  • Analyze RF coverage overlap and interference profiles\n"
                         "  • Optimize handover parameters (A3 Offset, Time-To-Trigger)")
        
        if drop_rate > 0.01:
            issues.append(f"Critical: High Call Drop Rate ({drop_rate:.3%}) - QoS Integrity Failure. Recommendations:\n"
                         "  • Inspect for external/internal interference (PCI conflicts)\n"
                         "  • Audit RF Link Budget (RSRP/SINR thresholds)\n"
                         "  • Evaluate resource availability (potential congestion-driven drops)")
        
        if throughput < 10:
            issues.append(f"Warning: Reduced Throughput Magnitude ({throughput:.1f} Mbps) - User Experience Degradation. Recommendations:\n"
                         "  • Audit backhaul transport bandwidth\n"
                         "  • Review MAC scheduler configuration and MCS distribution\n"
                         "  • Correlate with active congestion indicators")
        
        if not issues:
            return "Operational Integrity: Cell performance metrics are within established thresholds.\n\nBaseline monitoring should proceed."
        
        return "\n\n".join(issues)
    
    @staticmethod
    def explain_anomaly(cell_id: str, kpi_name: str, kpi_value: float, 
                       threshold: float, context: Dict) -> str:
        """Deterministic explanation for KPI anomaly events."""
        
        explanation = f"Diagnostic Report: Cell {cell_id} - Abnormal {kpi_name}\n\n"
        explanation += f"Observation: {kpi_value:.4f} (Control Limit: {threshold:.4f})\n\n"
        
        if "drop" in kpi_name.lower():
            explanation += "Probable Root Cause Framework:\n"
            explanation += "1. High-frequency interference or coverage impairment\n"
            explanation += "2. Resource exhaustion (Capacity saturation)\n"
            explanation += "3. Hardware/Hardware component failure or misconfiguration\n"
            explanation += "4. Handover failure cascades (Neighbor misalignment)\n"
            
            if context.get('prb', 0) > 0.8:
                explanation += "\nDiagnostic Correlation: Elevated PRB utilization suggests congestion as a primary contributor to drop events."
        
        elif "throughput" in kpi_name.lower():
            explanation += "Probable Root Cause Framework:\n"
            explanation += "1. Network congestion (PRB utilization saturation)\n"
            explanation += "2. Backhaul transport capacity constraints\n"
            explanation += "3. Sub-optimal RF conditions (SINR/BLER variance)\n"
            explanation += "4. Scheduler misconfiguration\n"
            
            if context.get('prb', 0) > 0.8:
                explanation += "\nDiagnostic Correlation: PRB saturation confirms capacity bottlenecks as the primary impairment vector."
        
        return explanation

    @staticmethod
    def generate_text(prompt: str, max_tokens: int = 500) -> str:
        """Deterministic fallback for generic text generation requests."""
        return ("Executive Briefing Fallback: Detailed neural synthesis is currently unavailable. "
                "Aggregated network metrics indicate that performance is being monitored against established baseline thresholds. "
                "Please enable Ollama for AI-driven strategic interpretation.")


def get_insights_generator(use_ollama: bool = True, model_name: str = "llama2"):
    """
    Factory function for diagnostic insight generators.
    
    Args:
        use_ollama: Enable Large Language Model (local inference).
        model_name: Identifier for the targeted LLM model.
    """
    if use_ollama:
        try:
            return OllamaInsightsGenerator(model_name=model_name)
        except Exception as e:
            # Revert to deterministic fallback upon inference failure
            return SimpleRuleBasedInsights()
    else:
        return SimpleRuleBasedInsights()

