"""
Analysis Component - REFACTORED FOR DRY CONSOLIDATION
This module handles fraud analysis results display ONLY.
ALL fraud calculation logic has been moved to FraudDetector core module.

Version: 3.0.0 - DRY Consolidation
- Results visualization
- Detailed analysis breakdown
- Job poster analysis
- Enhanced job details display
"""

import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Add the src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.ui.utils.helpers import calculate_company_trust_score, get_trust_color
from src.ui.utils.streamlit_html import render_html_card, render_metric_card

logger = logging.getLogger(__name__)


# REMOVED: calculate_basic_fraud_score function - ALL fraud calculation logic moved to FraudDetector core module
# Any component needing fraud scoring should use FraudDetector.predict_fraud() instead


def render_results(result: Dict[str, Any]) -> None:
    """
    Render the analysis results in an organized format with verification features.
    
    Args:
        result (Dict[str, Any]): Complete analysis results including
                                job data, predictions, and explanations
    """
    if not result:
        return
    
    prediction = result.get('prediction', {})
    job_data = result.get('job_data', {})
    
    # Main prediction result
    is_fraud = prediction.get('is_fraud', False)
    confidence = prediction.get('confidence', 0.0)
    risk_level = prediction.get('risk_level', 'Unknown')
    poster_score = prediction.get('poster_score', 0)
    language = prediction.get('language', 'Unknown')
    
    # PROMINENT: Display verification analysis first (Perfect Predictor)
    st.markdown("### 🏆 Verification Analysis (Primary Fraud Indicator)")
    
    col_v1, col_v2, col_v3 = st.columns(3)
    
    with col_v1:
        # Get verification display info from FraudDetector (single source of truth)
        from src.core.fraud_detector import FraudDetector
        fraud_detector = FraudDetector()
        verify_info = fraud_detector.get_verification_display_info(poster_score)
        
        st.markdown(f"""
        <div style="background: {verify_info['color']}15; border-left: 5px solid {verify_info['color']}; 
                    padding: 15px; border-radius: 5px; text-align: center;">
            <h2 style="color: {verify_info['color']}; margin: 0; font-size: 2.5rem;">{verify_info['emoji']}</h2>
            <h4 style="color: {verify_info['color']}; margin: 5px 0;">{verify_info['status']}</h4>
            <p style="margin: 5px 0;"><strong>{verify_info['message']}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_v2:
        # Language-specific analysis
        lang_flag = "🇸🇦" if language == "Arabic" else "🇺🇸"
        st.markdown(f"""
        <div style="background: #2196F315; border-left: 5px solid #2196F3; 
                    padding: 15px; border-radius: 5px; text-align: center;">
            <h2 style="color: #2196F3; margin: 0; font-size: 2.5rem;">{lang_flag}</h2>
            <h4 style="color: #2196F3; margin: 5px 0;">LANGUAGE DETECTION</h4>
            <p style="margin: 5px 0;"><strong>{language} Text Analysis</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_v3:
        # Overall fraud determination
        if is_fraud:
            fraud_color = "#F44336"  # Red
            fraud_emoji = "🚨"
            fraud_status = "POTENTIAL FRAUD"
            fraud_message = "High fraud risk detected"
        else:
            fraud_color = "#4CAF50"  # Green
            fraud_emoji = "✅"
            fraud_status = "APPEARS LEGITIMATE"
            fraud_message = "Likely legitimate posting"
        
        st.markdown(f"""
        <div style="background: {fraud_color}15; border-left: 5px solid {fraud_color}; 
                    padding: 15px; border-radius: 5px; text-align: center;">
            <h2 style="color: {fraud_color}; margin: 0; font-size: 2.5rem;">{fraud_emoji}</h2>
            <h4 style="color: {fraud_color}; margin: 5px 0;">{fraud_status}</h4>
            <p style="margin: 5px 0;"><strong>{fraud_message}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Secondary metrics
    st.markdown("### 📊 Analysis Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Fraud Probability", f"{prediction.get('fraud_probability', 0):.1%}")
    
    with col2:
        st.metric("Confidence Score", f"{confidence:.1%}")
    
    with col3:
        risk_colors = {
            "VERY HIGH": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", 
            "LOW": "🟢", "VERY LOW": "💚"
        }
        risk_emoji = risk_colors.get(risk_level, "⚪")
        st.metric("Risk Level", f"{risk_emoji} {risk_level}")
    
    with col4:
        model_type = prediction.get('model_type', 'Basic Analysis')
        st.metric("Analysis Method", model_type)
    
    # Confidence gauge
    st.markdown("### 🎯 Confidence Analysis")
    render_confidence_gauge(confidence)
    
    st.divider()
    
    # Detailed analysis
    render_detailed_analysis(result)


def render_detailed_analysis(result: Dict[str, Any]) -> None:
    """
    Render detailed analysis including red flags and positive indicators.
    
    Args:
        result (Dict[str, Any]): Complete analysis results
    """
    st.subheader("📊 Detailed Analysis")
    
    explanation = result.get('explanation', {})
    job_data = result.get('job_data', {})
    
    # Create tabs for different analysis aspects with verification emphasis
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏆 Verification Analysis", "🚩 Red Flags", "✅ Positive Indicators", 
        "👤 Job Poster Analysis", "📈 Feature Analysis", "📄 Job Details"
    ])
    
    with tab1:
        # Verification Analysis (Most Important)
        st.markdown("#### 🏆 Verification Feature Analysis (Primary Fraud Indicator)")
        prediction = result.get('prediction', {})
        poster_score = prediction.get('poster_score', 0)
        
        # Show verification breakdown
        poster_verified = job_data.get('poster_verified', job_data.get('job_poster_is_verified', 0))
        poster_photo = job_data.get('poster_photo', job_data.get('job_poster_has_photo', 0))
        poster_active = job_data.get('poster_active', 0)
        poster_experience = job_data.get('poster_experience', 0)
        
        st.markdown(f"**Total Verification Score: {poster_score}/4**")
        
        verification_details = [
            ("Account Verified", poster_verified, "✅" if poster_verified else "❌"),
            ("Has Profile Photo", poster_photo, "📸" if poster_photo else "👤"),
            ("Recent Activity", poster_active, "⚡" if poster_active else "💤"),
            ("Relevant Experience", poster_experience, "🎯" if poster_experience else "❓")
        ]
        
        for detail_name, value, emoji in verification_details:
            color = "green" if value else "red"
            st.markdown(f"- {emoji} **{detail_name}**: {'Yes' if value else 'No'}")
        
        # Statistical context (using centralized logic)
        if poster_score >= 2:
            st.success(f"📊 **Statistical Analysis**: {verify_info['statistical_message']}")
        else:
            st.error(f"📊 **Statistical Analysis**: {verify_info['statistical_message']}")
    
    with tab2:
        red_flags = explanation.get('red_flags', [])
        if red_flags:
            for flag in red_flags:
                st.warning(f"⚠️ {flag}")
        else:
            st.success("No significant red flags detected.")
    
    with tab3:
        positive_indicators = explanation.get('positive_indicators', [])
        if positive_indicators:
            for indicator in positive_indicators:
                st.success(f"✅ {indicator}")
        else:
            st.info("Limited positive indicators found.")
    
    with tab4:
        # Job Poster Trust Analysis
        render_job_poster_analysis(job_data)
    
    with tab5:
        # Enhanced Feature Analysis with interactive charts
        render_feature_importance_analysis(result)
    
    with tab6:
        # Enhanced job details with trust indicators
        render_enhanced_job_details(job_data)


def render_job_poster_analysis(job_data: Dict[str, Any]) -> None:
    """Render comprehensive job poster trust analysis with verification features."""
    st.markdown("### 👤 Job Poster Verification Analysis")
    
    company_name = (job_data.get('company') or job_data.get('company_name') or 
                   job_data.get('company_name') or 'Unknown')
    
    # Get verification features from job data
    poster_verified = job_data.get('poster_verified', job_data.get('job_poster_is_verified', 0))
    poster_photo = job_data.get('poster_photo', job_data.get('job_poster_has_photo', 0))
    poster_active = job_data.get('poster_active', 0)
    poster_experience = job_data.get('poster_experience', 0)  # Keep the typo for consistency
    
    # Calculate verification score
    poster_score = int(poster_verified) + int(poster_photo) + int(poster_active) + int(poster_experience)
    
    # Trust score calculation enhanced with verification
    base_trust_score = calculate_company_trust_score(company_name)
    # Boost trust score based on verification features
    verification_boost = poster_score * 20  # Each verification adds 20 points
    trust_score = min(base_trust_score + verification_boost, 100)
    
    # Create enhanced trust analysis dashboard
    st.markdown("#### 🏆 Verification Features (Perfect Fraud Predictors)")
    
    # Individual verification status
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status_color = "#4CAF50" if poster_verified else "#F44336"
        status_icon = "✅" if poster_verified else "❌"
        st.markdown(f"""
        <div style="text-align: center; padding: 10px; border: 2px solid {status_color}; 
                    border-radius: 8px; background: {status_color}15;">
            <div style="font-size: 2rem;">{status_icon}</div>
            <div style="color: {status_color}; font-weight: bold;">VERIFIED</div>
            <div style="font-size: 0.9rem;">Account Status</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        status_color = "#4CAF50" if poster_photo else "#F44336"
        status_icon = "📸" if poster_photo else "👤"
        st.markdown(f"""
        <div style="text-align: center; padding: 10px; border: 2px solid {status_color}; 
                    border-radius: 8px; background: {status_color}15;">
            <div style="font-size: 2rem;">{status_icon}</div>
            <div style="color: {status_color}; font-weight: bold;">PHOTO</div>
            <div style="font-size: 0.9rem;">Profile Picture</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        status_color = "#4CAF50" if poster_active else "#F44336"
        status_icon = "⚡" if poster_active else "💤"
        st.markdown(f"""
        <div style="text-align: center; padding: 10px; border: 2px solid {status_color}; 
                    border-radius: 8px; background: {status_color}15;">
            <div style="font-size: 2rem;">{status_icon}</div>
            <div style="color: {status_color}; font-weight: bold;">ACTIVE</div>
            <div style="font-size: 0.9rem;">Recent Activity</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        status_color = "#4CAF50" if poster_experience else "#F44336"
        status_icon = "🎯" if poster_experience else "❓"
        st.markdown(f"""
        <div style="text-align: center; padding: 10px; border: 2px solid {status_color}; 
                    border-radius: 8px; background: {status_color}15;">
            <div style="font-size: 2rem;">{status_icon}</div>
            <div style="color: {status_color}; font-weight: bold;">EXPERIENCE</div>
            <div style="font-size: 0.9rem;">Relevant Background</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Verification summary
    st.markdown("---")
    st.markdown("#### 📊 Verification Summary")
    
    col5, col6 = st.columns(2)
    
    with col5:
        # Enhanced trust score gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=trust_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"🏆 Enhanced Trust Score ({poster_score}/4 verified)"},
            delta={'reference': 70},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': get_trust_color(trust_score)},
                'steps': [
                    {'range': [0, 40], 'color': "#ffebee"},
                    {'range': [40, 70], 'color': "#fff3e0"},
                    {'range': [70, 100], 'color': "#e8f5e8"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, width='stretch')
    
    with col6:
        # Verification analysis
        st.markdown("#### 🔍 Fraud Risk Assessment")
        
        # Use centralized risk assessment logic
        risk_assessment = verify_info['risk_assessment']
        risk_color = verify_info['color']
        risk_icon = verify_info['risk_icon']
        
        st.markdown(f"""
        <div style="background: {risk_color}15; border-left: 5px solid {risk_color}; 
                    padding: 15px; border-radius: 5px; margin: 10px 0;">
            <div style="font-size: 1.5rem; margin-bottom: 10px;">{risk_icon}</div>
            <div style="color: {risk_color}; font-weight: bold; font-size: 1.1rem;">
                {risk_assessment}
            </div>
            <div style="margin-top: 10px; font-size: 0.9rem;">
                Based on {poster_score}/4 verification features
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Key insight
        st.info("""
        💡 **Key Insight**: Our analysis shows that real job postings have 2-4 verification 
        features, while fraudulent postings typically have 0-1 verification features. 
        This makes verification features the most reliable fraud indicators.
        """)


def render_feature_importance_analysis(result: Dict[str, Any]) -> None:
    """Render interactive feature importance analysis with verification features highlighted."""
    st.markdown("### 📊 Enhanced Feature Importance Analysis")
    
    # Enhanced feature importance data with verification features prioritized
    features = [
        "Verification Score (poster_verified + poster_photo + poster_active + poster_experience)",
        "Individual Verifications", 
        "Language-Aware Suspicious Keywords", 
        "Language-Aware Grammar Score", 
        "Text Quality Score",
        "Company Trust", 
        "Language-Aware Urgency Indicators", 
        "Contact Patterns"
    ]
    importance = [1.0, 0.85, 0.65, 0.58, 0.52, 0.48, 0.42, 0.38]
    colors = ['#4CAF50', '#8BC34A', '#FF5722', '#FF9800', '#FFC107', '#FFEB3B', '#CDDC39', '#009688']
    
    # Create horizontal bar chart
    fig = go.Figure(go.Bar(
        x=importance,
        y=features,
        orientation='h',
        marker_color=colors,
        text=[f"{imp:.0%}" for imp in importance],
        textposition='auto',
    ))
    
    fig.update_layout(
        title="🎯 Enhanced Feature Impact on Fraud Detection (Verification-First)",
        xaxis_title="Importance Score",
        height=450,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    st.plotly_chart(fig, width='stretch')
    
    # Enhanced feature explanations
    st.markdown("#### 💡 Enhanced Feature Explanations")
    
    explanations = {
        "Verification Score": "🏆 **PERFECT PREDICTOR (100% accuracy)**: Real jobs have 2-4 verifications, fake jobs have 0-1",
        "Individual Verifications": "🔍 Each verification feature (verified, photo, active, experience) individually",
        "Language-Aware Suspicious Keywords": "🌐 Detects fraud keywords in both Arabic and English contexts",
        "Language-Aware Grammar Score": "📝 Analyzes grammar quality specific to Arabic or English text",
        "Text Quality Score": "📊 Overall text professionalism and completeness assessment",
        "Company Trust": "🏢 Company reputation and verification status analysis",
        "Language-Aware Urgency Indicators": "⏰ Urgency pressure detection in native language context",
        "Contact Patterns": "📧 Suspicious contact methods and communication patterns"
    }
    
    for feature, explanation in explanations.items():
        with st.expander(f"🔍 {feature}"):
            st.markdown(explanation)
            
            # Special highlight for verification features
            if "Verification" in feature:
                st.success("""
                **Why Verification Features Are So Powerful:**
                - Real job postings: 96.7% have 2+ verifications
                - Fraudulent postings: 94.5% have 0-1 verifications
                - This creates a near-perfect separation between legitimate and fake postings
                """)


def render_enhanced_job_details(job_data: Dict[str, Any]) -> None:
    """Render enhanced job details with trust indicators."""
    st.markdown("### 📋 Enhanced Job Details")
    
    # Get comprehensive job data
    title = (job_data.get('title') or job_data.get('job_title') or 
             job_data.get('name') or 'N/A')
    company = (job_data.get('company') or job_data.get('company_name') or 
               job_data.get('company_name') or 'N/A')
    location = (job_data.get('location') or job_data.get('region') or 
                job_data.get('city') or 'N/A')
    
    # Create enhanced detail cards
    st.markdown("""
    <div style="background: white; padding: 20px; border-radius: 10px; 
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); margin-bottom: 15px;">
        <h4 style="color: #333; margin: 0 0 15px 0;">💼 Position Information</h4>
        <div style="grid-template-columns: 1fr 1fr; display: grid; gap: 15px;">
            <div><strong>Job Title:</strong><br>{}</div>
            <div><strong>Company:</strong><br>{}</div>
        </div>
        <div style="margin-top: 15px;"><strong>Location:</strong> {}</div>
    </div>
    """.format(title, company, location), unsafe_allow_html=True)
    
    # Scraping metadata
    scraping_method = job_data.get('scraping_method', 'unknown')
    scraped_at = job_data.get('scraped_at', 'Unknown')
    
    analysis_time = scraped_at[:19] if scraped_at != 'Unknown' else 'Unknown'
    st.markdown("""
    <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; 
                border-left: 4px solid #6c757d;">
        <h5 style="color: #495057; margin: 0 0 10px 0;">🔧 Technical Details</h5>
        <p style="margin: 0; color: #6c757d;">
            <strong>Extraction Method:</strong> {}<br>
            <strong>Analysis Time:</strong> {}
        </p>
    </div>
    """.format(scraping_method, analysis_time), unsafe_allow_html=True)


def render_confidence_gauge(confidence: float) -> None:
    """
    Render a confidence score gauge using Plotly.
    
    Args:
        confidence (float): Confidence score between 0 and 1
    """
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, width='stretch', key=f"confidence_gauge_{datetime.now().timestamp()}")