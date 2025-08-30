"""
Feature Display Component

This module handles the display of extracted features and fraud analysis results including:
- Feature vectors and analysis
- Risk assessment visualization
- Pattern detection results
- Feature extraction magic display
- Detailed feature analysis
"""

import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add the src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))


def show_feature_extraction_magic(job_data: Dict[str, Any]) -> None:
    """
    Show the magic of feature extraction with visual feedback.
    
    Args:
        job_data (Dict[str, Any]): Job data to process
    """
    st.info("🎯 Now extracting 34 intelligent features from the job posting...")
    
    # Create columns for feature categories
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.container():
            st.markdown("### 📝 Text Analysis")
            progress = st.progress(0)
            status = st.empty()
            
            steps = [
                "Checking suspicious keywords...",
                "Analyzing grammar quality...", 
                "Calculating sentiment scores..."
            ]
            
            for i, step in enumerate(steps, 1):
                status.text(step)
                progress.progress(i * 33)
                time.sleep(0.3)
            
            status.empty()
            st.success("12 text features extracted!")
    
    with col2:
        with st.container():
            st.markdown("### 🏗️ Structure Analysis")
            progress = st.progress(0)
            status = st.empty()
            
            steps = [
                "Checking required sections...",
                "Validating contact patterns...",
                "Analyzing formatting..."
            ]
            
            for i, step in enumerate(steps, 1):
                status.text(step)
                progress.progress(i * 33)
                time.sleep(0.3)
                
            status.empty()
            st.success("10 structural features extracted!")
    
    with col3:
        with st.container():
            st.markdown("### 🔍 Pattern Detection")  
            progress = st.progress(0)
            status = st.empty()
            
            steps = [
                "Matching fraud patterns...",
                "Checking legitimacy signals...",
                "Computing risk indicators..."
            ]
            
            for i, step in enumerate(steps, 1):
                status.text(step)
                progress.progress(i * 33)
                time.sleep(0.3)
                
            status.empty()
            st.success("12 pattern features extracted!")
    
    # Show feature vector summary
    st.success("✨ **Feature Vector Created:** 34 dimensions of intelligence extracted!")


def show_detailed_feature_extraction(job_data: Dict[str, Any], features) -> None:
    """
    Show detailed breakdown of extracted features with explanations.
    
    Args:
        job_data (Dict[str, Any]): Original job data
        features: Extracted feature vector (DataFrame or dict)
    """
    # Handle both DataFrame and dict input
    if isinstance(features, pd.DataFrame) and features.empty:
        return
    if isinstance(features, dict) and not features:
        return
    
    st.markdown("### 🔬 **Feature Engineering Deep Dive**")
    st.info("🎯 Here's exactly how we transformed raw job posting text into 34 intelligent features for machine learning:")
    
    # Convert features to dict for easier access
    if isinstance(features, pd.DataFrame):
        feat_dict = features.iloc[0].to_dict() if not features.empty else {}
    else:
        feat_dict = features if isinstance(features, dict) else {}
    
    # Text Analysis Features
    with st.expander("📝 **Text Analysis Features (12 features)**", expanded=True):
        st.markdown("#### Suspicious Language Detection:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sus_count = feat_dict.get('suspicious_keywords_count', 0)
            st.metric("🚨 Suspicious Keywords", sus_count)
            st.caption("Terms like 'guaranteed income', 'easy money', 'no experience needed'")
            
            if sus_count > 0:
                st.warning(f"⚠️ Found {sus_count} suspicious terms")
            else:
                st.success("✅ No suspicious keywords detected")
        
        with col2:
            gram_score = feat_dict.get('grammar_score', 0)
            st.metric("📝 Grammar Quality", f"{gram_score:.1f}%")
            st.caption("Professional writing quality assessment")
            
            if gram_score >= 80:
                st.success("✅ Professional grammar")
            elif gram_score >= 60:
                st.warning("⚠️ Average grammar quality")
            else:
                st.error("❌ Poor grammar quality")
        
        with col3:
            sent_score = feat_dict.get('sentiment_score', 0)
            st.metric("🎭 Sentiment Score", f"{sent_score:.2f}")
            st.caption("Emotional tone: -1 (negative) to +1 (positive)")
            
            if sent_score > 0.3:
                st.success("😊 Positive tone")
            elif sent_score < -0.3:
                st.error("😞 Negative tone")
            else:
                st.info("😐 Neutral tone")
        
        # Second row
        col4, col5, col6 = st.columns(3)
        
        with col4:
            text_len = feat_dict.get('text_length', 0)
            st.metric("📏 Text Length", text_len)
            st.caption("Character count of job description")
            
        with col5:
            caps_ratio = feat_dict.get('capital_letter_ratio', 0)
            st.metric("🔤 Caps Usage", f"{caps_ratio:.1%}")
            st.caption("Excessive caps indicate unprofessionalism")
            
        with col6:
            read_score = feat_dict.get('readability_score', 0)
            st.metric("📖 Readability", f"{read_score:.0f}")
            st.caption("Flesch reading ease (0-100)")
    
    # Structural Analysis Features  
    with st.expander("🏗️ **Structural Analysis Features (10 features)**", expanded=True):
        st.markdown("#### Job Posting Structure Quality:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            has_salary = feat_dict.get('has_salary_range', False)
            st.metric("💰 Salary Info", "✅ Present" if has_salary else "❌ Missing")
            st.caption("Legitimate jobs usually specify salary")
            
        with col2:
            contact_methods = feat_dict.get('contact_methods_count', 0)
            st.metric("📞 Contact Methods", contact_methods)
            st.caption("Email, phone, application links")
            
        with col3:
            req_sections = feat_dict.get('required_sections_present', 0)
            st.metric("📋 Required Sections", req_sections)
            st.caption("Requirements, responsibilities, benefits")
    
    # Pattern Detection Features
    with st.expander("🔍 **Fraud Pattern Detection Features (12 features)**", expanded=True):
        st.markdown("#### AI-Powered Fraud Pattern Matching:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            email_sus = feat_dict.get('email_domain_suspicious', False)
            st.metric("📧 Email Domain", "🚨 Suspicious" if email_sus else "✅ Professional")
            st.caption("gmail.com, yahoo.com are red flags for jobs")
            
        with col2:
            urgency = feat_dict.get('urgency_indicators', 0)
            st.metric("⏰ Urgency Signals", urgency)
            st.caption("'Apply now!', 'Limited time', 'Urgent'")
            
        with col3:
            unrealistic = feat_dict.get('unrealistic_promises', 0)
            st.metric("💸 Unrealistic Promises", unrealistic)
            st.caption("'Guaranteed success', 'Easy money'")
    
    # Feature Summary
    st.markdown("---")
    st.markdown("### 📊 **Feature Vector Summary**")
    
    total_features = len(feat_dict)
    suspicious_features = sum([
        feat_dict.get('suspicious_keywords_count', 0),
        1 if feat_dict.get('email_domain_suspicious', False) else 0,
        feat_dict.get('urgency_indicators', 0),
        feat_dict.get('unrealistic_promises', 0)
    ])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🔢 Total Features", total_features)
    with col2:
        st.metric("🚨 Risk Indicators", suspicious_features)
    with col3:
        confidence_level = "High" if suspicious_features == 0 else "Medium" if suspicious_features <= 2 else "Low"
        st.metric("✅ Confidence Level", confidence_level)