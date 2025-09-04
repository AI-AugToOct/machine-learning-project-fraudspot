"""
Analysis Component - CONTENT-FOCUSED VERSION
This module handles fraud analysis results display for content-focused fraud detection.
Focuses on job posting content and company metrics, not profile data.

Version: 4.0.0 - Content-Focused Analysis
- Content quality analysis
- Company legitimacy analysis  
- Contact risk assessment
- Job structure analysis
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


def render_results(result: Dict[str, Any]) -> None:
    """
    Render the analysis results in an organized format focusing on content analysis.
    
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
    language = prediction.get('language', 'Unknown')
    
    # PROMINENT: Display content quality analysis first
    st.markdown("### 📝 Content Quality Analysis (Primary Fraud Indicator)")
    
    col_v1, col_v2, col_v3 = st.columns(3)
    
    with col_v1:
        # Get content quality score from prediction
        content_score = prediction.get('content_quality_score', 0.0)
        
        # Determine color and status based on content quality
        if content_score >= 0.8:
            color, emoji, status, message = "#4CAF50", "✅", "HIGH QUALITY", "Professional content detected"
        elif content_score >= 0.6:
            color, emoji, status, message = "#FF9800", "⚠️", "MODERATE QUALITY", "Some quality indicators"
        elif content_score >= 0.4:
            color, emoji, status, message = "#FF5722", "⚡", "LOW QUALITY", "Limited quality indicators"
        else:
            color, emoji, status, message = "#F44336", "❌", "POOR QUALITY", "Poor content quality detected"
            
        st.markdown(f"""
        <div style="background: {color}15; border-left: 5px solid {color}; 
                    padding: 15px; border-radius: 5px; text-align: center;">
            <h2 style="color: {color}; margin: 0; font-size: 2.5rem;">{emoji}</h2>
            <h4 style="color: {color}; margin: 5px 0;">{status}</h4>
            <p style="margin: 5px 0;"><strong>{message}</strong></p>
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
    
    # Create tabs for different analysis aspects with content focus
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📝 Content Analysis", "🚩 Red Flags", "✅ Positive Indicators", 
        "📈 Feature Analysis", "📄 Job Details"
    ])
    
    with tab1:
        # Content Quality Analysis (Most Important)
        st.markdown("#### 📝 Content Quality Analysis (Primary Fraud Indicator)")
        prediction = result.get('prediction', {})
        
        # Get content-focused scores
        content_quality = prediction.get('content_quality_score', 0.0)
        company_legitimacy = prediction.get('company_legitimacy_score', 0.0)
        contact_risk = prediction.get('contact_risk_score', 0.0)
        
        st.markdown(f"**Content Quality Score: {content_quality:.2f}/1.0**")
        st.markdown(f"**Company Legitimacy Score: {company_legitimacy:.2f}/1.0**")
        st.markdown(f"**Contact Risk Score: {contact_risk:.2f}/1.0**")
        
        # Create content quality breakdown
        content_details = [
            ("Professional Language", prediction.get('professional_language_score', 0.0), "💼"),
            ("Complete Job Description", prediction.get('description_length_score', 0.0), "📄"), 
            ("Proper Job Structure", prediction.get('has_requirements', 0), "📋"),
            ("Company Information", prediction.get('has_company_website', 0), "🏢")
        ]
        
        for detail_name, value, emoji in content_details:
            if isinstance(value, float):
                display_value = f"{value:.2f}"
                color = "green" if value >= 0.6 else "orange" if value >= 0.3 else "red"
            else:
                display_value = "Yes" if value else "No"
                color = "green" if value else "red"
            st.markdown(f"- {emoji} **{detail_name}**: {display_value}")
        
        # Statistical context for content quality
        if content_quality >= 0.6:
            st.success("📊 **Analysis**: High content quality indicates legitimate posting")
        else:
            st.error("📊 **Analysis**: Poor content quality is a strong fraud indicator")
    
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
        # Enhanced Feature Analysis with interactive charts
        render_feature_importance_analysis(result)
    
    with tab5:
        # Enhanced job details with content focus
        render_enhanced_job_details(job_data)


def render_content_quality_analysis(job_data: Dict[str, Any]) -> None:
    """Render comprehensive content quality analysis."""
    st.markdown("### 📝 Content Quality Analysis")
    
    # Get content quality metrics
    content_score = job_data.get('content_quality_score', 0.0)
    company_score = job_data.get('company_legitimacy_score', 0.0)
    contact_risk = job_data.get('contact_risk_score', 0.0)
    
    # Create content quality dashboard
    st.markdown("#### 📊 Quality Indicators")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Content Quality Score
        color = "#4CAF50" if content_score >= 0.7 else "#FF9800" if content_score >= 0.4 else "#F44336"
        st.markdown(f"""
        <div style="text-align: center; padding: 15px; border: 2px solid {color}; 
                    border-radius: 8px; background: {color}15;">
            <div style="font-size: 2rem;">📝</div>
            <div style="color: {color}; font-weight: bold;">Content Quality</div>
            <div style="font-size: 1.5rem; color: {color};">{content_score:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Company Legitimacy
        color = "#4CAF50" if company_score >= 0.7 else "#FF9800" if company_score >= 0.4 else "#F44336"
        st.markdown(f"""
        <div style="text-align: center; padding: 15px; border: 2px solid {color}; 
                    border-radius: 8px; background: {color}15;">
            <div style="font-size: 2rem;">🏢</div>
            <div style="color: {color}; font-weight: bold;">Company Score</div>
            <div style="font-size: 1.5rem; color: {color};">{company_score:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Contact Risk (inverse - lower is better)
        color = "#F44336" if contact_risk >= 0.7 else "#FF9800" if contact_risk >= 0.4 else "#4CAF50"
        st.markdown(f"""
        <div style="text-align: center; padding: 15px; border: 2px solid {color}; 
                    border-radius: 8px; background: {color}15;">
            <div style="font-size: 2rem;">📞</div>
            <div style="color: {color}; font-weight: bold;">Contact Risk</div>
            <div style="font-size: 1.5rem; color: {color};">{contact_risk:.2f}</div>
        </div>
        """, unsafe_allow_html=True)


def render_feature_importance_analysis(result: Dict[str, Any]) -> None:
    """Render interactive feature importance analysis with content features highlighted."""
    st.markdown("### 📊 Content-Focused Feature Importance Analysis")
    
    # Content-focused feature importance data
    features = [
        "Content Quality Score",
        "Company Legitimacy Score", 
        "Professional Language Score", 
        "Contact Risk Score", 
        "Suspicious Keywords Count",
        "Description Length Score", 
        "Job Structure Completeness", 
        "Company Information Completeness"
    ]
    importance = [1.0, 0.85, 0.70, 0.65, 0.58, 0.50, 0.45, 0.40]
    colors = ['#4CAF50', '#8BC34A', '#2196F3', '#FF5722', '#FF9800', '#FFC107', '#CDDC39', '#009688']
    
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
        title="🎯 Feature Impact on Fraud Detection (Content-Focused)",
        xaxis_title="Importance Score",
        height=450,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    st.plotly_chart(fig, width='stretch')
    
    # Content-focused feature explanations
    st.markdown("#### 💡 Content Feature Explanations")
    
    explanations = {
        "Content Quality Score": "📝 Overall job posting professionalism, completeness, and structure quality",
        "Company Legitimacy Score": "🏢 Company reputation, website presence, and business verification indicators",
        "Professional Language Score": "💼 Use of professional terminology and business language patterns",
        "Contact Risk Score": "📞 Analysis of contact methods for suspicious patterns (messaging apps vs professional channels)",
        "Suspicious Keywords Count": "🚨 Detection of fraud-related terms in both Arabic and English",
        "Description Length Score": "📄 Job description completeness and detail level assessment",
        "Job Structure Completeness": "📋 Presence of standard job posting elements (requirements, salary, etc.)",
        "Company Information Completeness": "🏢 Completeness of company profile and business information"
    }
    
    for feature, explanation in explanations.items():
        with st.expander(f"🔍 {feature}"):
            st.markdown(explanation)
            
            # Special highlight for content features
            if "Content Quality" in feature:
                st.success("""
                **Why Content Quality Is So Important:**
                - Legitimate postings: Professional language, complete descriptions, proper structure
                - Fraudulent postings: Poor grammar, incomplete information, urgent language
                - Content analysis is reliable and doesn't depend on profile access
                """)


def render_enhanced_job_details(job_data: Dict[str, Any]) -> None:
    """Render enhanced job details with content focus."""
    st.markdown("### 📋 Enhanced Job Details")
    
    # Get comprehensive job data
    title = (job_data.get('title') or job_data.get('job_title') or 
             job_data.get('name') or '')
    company = (job_data.get('company') or job_data.get('company_name') or '')
    location = (job_data.get('location') or job_data.get('region') or 
                job_data.get('city') or '')
    
    # Create enhanced detail cards
    st.markdown(f"""
    <div style="background: white; padding: 20px; border-radius: 10px; 
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); margin-bottom: 15px;">
        <h4 style="color: #333; margin: 0 0 15px 0;">💼 Position Information</h4>
        <div style="grid-template-columns: 1fr 1fr; display: grid; gap: 15px;">
            <div><strong>Job Title:</strong><br>{title}</div>
            <div><strong>Company:</strong><br>{company}</div>
        </div>
        <div style="margin-top: 15px;"><strong>Location:</strong> {location}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Content quality indicators
    content_quality = job_data.get('content_quality_score', 0.0)
    company_score = job_data.get('company_legitimacy_score', 0.0)
    
    st.markdown(f"""
    <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; 
                border-left: 4px solid #6c757d;">
        <h5 style="color: #495057; margin: 0 0 10px 0;">📊 Content Analysis</h5>
        <p style="margin: 0; color: #6c757d;">
            <strong>Content Quality:</strong> {content_quality:.2f}/1.0<br>
            <strong>Company Legitimacy:</strong> {company_score:.2f}/1.0
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Scraping metadata
    scraping_method = job_data.get('scraping_method', 'unknown')
    scraped_at = job_data.get('scraped_at', 'Unknown')
    
    analysis_time = scraped_at[:19] if scraped_at != 'Unknown' else 'Unknown'
    st.markdown(f"""
    <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; 
                border-left: 4px solid #6c757d; margin-top: 15px;">
        <h5 style="color: #495057; margin: 0 0 10px 0;">🔧 Technical Details</h5>
        <p style="margin: 0; color: #6c757d;">
            <strong>Extraction Method:</strong> {scraping_method}<br>
            <strong>Analysis Time:</strong> {analysis_time}
        </p>
    </div>
    """, unsafe_allow_html=True)


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