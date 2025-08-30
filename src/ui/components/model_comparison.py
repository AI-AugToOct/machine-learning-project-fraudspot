"""
Ensemble Voting Explanation Component

This module handles ensemble voting analysis interface including:
- Real-time voting breakdown display
- Individual model contribution analysis
- Performance metrics visualization
- Ensemble testing interface
"""

import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add the src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.core.constants import get_feature_importance_ranking, get_feature_weights

logger = logging.getLogger(__name__)


def render_model_comparison_tab() -> None:
    """
    Render the ensemble voting analysis tab.
    
    This tab shows:
    - Ensemble voting breakdown and decision analysis
    - Individual model contributions to final decision
    - Real-time voting analysis with test cases
    - Model status information
    """
    st.markdown("### 🎯 Ensemble Voting Analysis")
    
    render_ensemble_voting_analysis()


def render_ensemble_voting_analysis():
    """Render real-time ensemble voting analysis"""
    st.markdown("#### 🎯 Real-Time Ensemble Voting Breakdown")
    
    # Test case selector
    test_cases = {
        "HungerStation (Legitimate)": {
            'company': 'HungerStation',
            'job_title': 'Software Engineer',
            'job_description': 'Join our engineering team to build food delivery products.',
            'poster_verified': 0,
            'has_company_logo': 1,
            'completeness_score': 0.9,
            'total_suspicious_keywords': 0
        },
        "Obvious Scam": {
            'company': 'Easy Money LLC',
            'job_title': 'Work from Home - Make Money Fast',
            'job_description': 'Earn $5000 weekly! No experience! Send money first!',
            'poster_verified': 0,
            'has_company_logo': 0,
            'total_suspicious_keywords': 5,
            'total_urgency_keywords': 4
        },
        "Borderline Case": {
            'company': 'Unknown Tech Solutions',
            'job_title': 'Developer Position',
            'job_description': 'Looking for developer. Good salary.',
            'poster_verified': 0,
            'has_company_logo': 0,
            'completeness_score': 0.6
        }
    }
    
    selected_case = st.selectbox(
        "Choose test case to analyze:",
        options=list(test_cases.keys()),
        help="Select a test case to see how the ensemble voting works"
    )
    
    if st.button("🔍 Analyze Ensemble Voting", type="primary"):
        with st.spinner("Running ensemble analysis..."):
            try:
                from ...pipeline.pipeline_manager import PipelineManager
                pm = PipelineManager()
                
                # Get prediction
                result = pm.predict(test_cases[selected_case])
                
                if result.get('success', True):
                    # Display overall result
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        risk_color = {"VERY LOW": "green", "LOW": "green", "MODERATE": "orange", "HIGH": "red", "VERY HIGH": "red"}
                        st.metric("Risk Level", result.get('risk_level', 'Unknown'), delta_color="inverse")
                    with col2:
                        st.metric("Fraud Probability", f"{result.get('fraud_probability', 0):.1%}")
                    with col3:
                        is_fraud = result.get('is_fraud', False)
                        st.metric("Final Decision", "FRAUD" if is_fraud else "LEGITIMATE")
                    
                    # Voting breakdown
                    st.markdown("#### 🗳️ Individual Model Votes")
                    
                    if 'individual_predictions' in result:
                        voting_data = []
                        individual_preds = result['individual_predictions']
                        individual_probs = result.get('individual_probabilities', {})
                        
                        for model, prediction in individual_preds.items():
                            prob = individual_probs.get(model, 0.5)
                            vote = "FRAUD" if prediction else "LEGITIMATE"
                            
                            voting_data.append({
                                'Model': model.replace('_', ' ').title(),
                                'Vote': vote,
                                'Probability': f"{prob:.1%}",
                                'Confidence': "High" if abs(prob - 0.5) > 0.3 else "Medium"
                            })
                        
                        # Display voting table
                        df = pd.DataFrame(voting_data)
                        st.dataframe(df, width='stretch')
                        
                        # Voting visualization
                        fraud_votes = sum(individual_preds.values())
                        total_votes = len(individual_preds)
                        
                        fig = go.Figure(data=[
                            go.Bar(name='FRAUD', x=['Ensemble Vote'], y=[fraud_votes], marker_color='red'),
                            go.Bar(name='LEGITIMATE', x=['Ensemble Vote'], y=[total_votes - fraud_votes], marker_color='green')
                        ])
                        fig.update_layout(
                            title="Voting Breakdown",
                            yaxis_title="Number of Votes",
                            barmode='stack'
                        )
                        st.plotly_chart(fig, width='stretch')
                        
                        # Ensemble decision logic
                        st.markdown("#### 🧠 Decision Logic")
                        voting_result = result.get('voting_result', 'N/A')
                        st.info(f"**Voting Result:** {voting_result}")
                        
                        if fraud_votes >= total_votes / 2:
                            st.warning(f"✋ **FLAGGED AS FRAUD** - Majority vote ({fraud_votes}/{total_votes} models agree)")
                        else:
                            st.success(f"✅ **CONSIDERED LEGITIMATE** - Majority vote ({total_votes - fraud_votes}/{total_votes} models agree)")
                    
                    # Risk factors analysis
                    if 'risk_factors' in result and result['risk_factors']:
                        st.markdown("#### ⚠️ Key Risk Factors")
                        for factor in result['risk_factors'][:5]:
                            if "✅" in factor:
                                st.success(factor)
                            elif "🚨" in factor or "⚠️" in factor:
                                st.warning(factor)
                            else:
                                st.info(factor)
                    
                    # Ensemble metadata
                    if 'ensemble_info' in result:
                        info = result['ensemble_info']
                        st.markdown("#### 📊 Ensemble Statistics")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Models Used", f"{len(info.get('models_used', []))}/4")
                        with col2:
                            st.metric("Consensus Type", info.get('voting_consensus', 'N/A').title())
                        with col3:
                            st.metric("Agreement Level", f"{info.get('model_agreement', 0):.1%}")
                
                else:
                    st.error("❌ Ensemble prediction failed")
                    if 'error' in result:
                        st.error(f"Error: {result['error']}")
                        
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.info("💡 Make sure models are trained: python train_model_cli.py --model all_models")
    
    # Show model status at the bottom
    render_ensemble_model_status()


def render_ensemble_model_status():
    """Display current ensemble model status"""
    st.divider()
    st.markdown("#### 📊 Ensemble Model Status")
    
    try:
        from ...core.ensemble_predictor import EnsemblePredictor
        ensemble = EnsemblePredictor()
        
        # Load models first, then get status
        ensemble.load_models()
        status = ensemble.get_model_status()
        
        model_names = {
            'random_forest': '🌲 Random Forest',
            'logistic_regression': '📈 Logistic Regression',
            'naive_bayes': '🧠 Naive Bayes',
            'svm': '🎯 SVM'
        }
        
        status_data = []
        for model_id, display_name in model_names.items():
            model_status = status['model_status'].get(model_id, 'missing')
            status_emoji = "✅" if model_status == 'loaded' else "❌"
            status_text = "Ready" if model_status == 'loaded' else "Needs Training"
            
            status_data.append({
                'Model': display_name,
                'Status': f"{status_emoji} {status_text}",
                'Notes': 'Ready for Ensemble' if model_status == 'loaded' else 'Train via CLI'
            })
        
        df_status = pd.DataFrame(status_data)
        st.dataframe(df_status, width='stretch')
        
        # Overall ensemble readiness
        ready_count = len([s for s in status['model_status'].values() if s == 'loaded'])
        total_count = len(model_names)
        
        if ready_count == total_count:
            st.success(f"🎯 **Ensemble Ready**: {ready_count}/{total_count} models loaded")
            st.info("📊 **Weights**: RF(45.4%) > LR(24.5%) > NB(15.2%) > SVM(14.8%)")
        elif ready_count > 0:
            st.warning(f"⚠️ **Partial Ensemble**: {ready_count}/{total_count} models loaded - some models missing")
        else:
            st.error(f"❌ **No Models Available**: {ready_count}/{total_count} models loaded")
        
        # Training instructions
        if ready_count < total_count:
            st.info("""
            💡 **To train missing models:**
            ```bash
            python train_model_cli.py --model all_models --no-interactive
            ```
            """)
            
    except Exception as e:
        st.error(f"❌ Cannot check model status: {str(e)}")
        st.info("""
        💡 **Train models first:**
        ```bash
        python train_model_cli.py --model all_models --no-interactive
        ```
        """)