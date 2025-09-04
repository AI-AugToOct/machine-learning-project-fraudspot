"""
UI Orchestrator - CONTENT-FOCUSED VERSION
This module orchestrates the UI pipeline for content-focused fraud detection.
Focuses on job posting content and company metrics, not profile data.

Version: 4.0.0 - Content-Focused Orchestration
"""

import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

# Add the src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
# Also add parent directory for relative imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import ONLY from core modules and services (single source of truth)
from src.core.fraud_pipeline import FraudDetectionPipeline
from src.services import ScrapingService, SerializationService
from src.ui.components.analysis import render_results

logger = logging.getLogger(__name__)


def _clear_analysis_session():
    """Clear all analysis-related session state to prevent data pollution between jobs."""
    keys_to_clear = []
    
    # Collect all keys that might cause pollution
    for key in list(st.session_state.keys()):
        if any(prefix in key for prefix in [
            'analysis_done', 'current_job_data', 'async_results', 'ready_for_analysis',
            'analysis_done_', 'cached_fraud_detector', 'cached_ensemble', 
            'fraud_prediction_', 'analysis_result_', 'job_features_'
        ]):
            keys_to_clear.append(key)
    
    # Clear collected keys
    for key in keys_to_clear:
        del st.session_state[key]
    
    logger.info(f"Cleared {len(keys_to_clear)} session state keys for fresh analysis")


def _get_cached_scraping_service():
    """Get pre-cached ScrapingService from main.py initialization."""
    # Use pre-cached scraping service from main.py initialization
    if 'cached_scraping_service' in st.session_state:
        return st.session_state['cached_scraping_service']
    else:
        # Fallback if not initialized (shouldn't happen)
        logger.warning("⚠️ Scraping service not pre-cached, creating new instance")
        # Direct service instantiation
        st.session_state['cached_scraping_service'] = ScrapingService()
        return st.session_state['cached_scraping_service']


def render_analysis_section_from_url(url: str, fraud_loader=None) -> None:
    """
    Render the main analysis section using content-focused approach.
    
    Args:
        url (str): The LinkedIn job URL to analyze
        fraud_loader: Deprecated, kept for compatibility
    """
    if not url:
        return
    
    # Clear previous analysis session state to prevent data pollution
    _clear_analysis_session()
    
    st.subheader("Analysis Results")
        
    # Instructions
    with st.expander("How it works", expanded=False):
        st.markdown("1. **URL Method**: Paste LinkedIn job URL for automatic scraping")
        st.markdown("2. **HTML Method**: If URL fails, copy the job page HTML manually")
        st.markdown("3. **Manual Method**: Enter job details manually as backup")
    
    # Scrape job posting with minimal spinner
    try:
        import time
        start_time = time.time()
        
        # Use cached ScrapingService to prevent repeated initialization
        scraping_service = _get_cached_scraping_service()
        with st.spinner("Loading job details..."):
            job_data = scraping_service.scrape_job_posting(url)
        
        elapsed_time = time.time() - start_time
        
        # ALL display logic OUTSIDE spinner so it shows immediately
        if not job_data or not job_data.get('success', False):
            error_msg = job_data.get('error_message', 'Unknown error') if job_data else 'Failed to scrape'
            scraping_method = job_data.get('scraping_method', 'unknown') if job_data else 'unknown'
            
            st.error(f"Job scraping failed ({scraping_method}): {error_msg}")
            
            # Clean alternative options
            with st.expander("Alternative Options", expanded=True):
                tab1, tab2, tab3 = st.tabs(["HTML Copy", "Manual Entry", "Retry"])
                
                with tab1:
                    st.markdown("""
                    1. Open the LinkedIn job page
                    2. Right-click → View Page Source
                    3. Copy all (Ctrl+A, Ctrl+C)
                    4. Paste in **HTML Content** tab
                    """)
                
                with tab2:
                    st.markdown("""
                    1. Go to **Manual Input** tab
                    2. Enter job details manually
                    3. Click Analyze
                    """)
                
                with tab3:
                    st.markdown("""
                    1. Check your connection
                    2. Wait a moment
                    3. Try the URL again
                    """)
            
            return
        
        # Success - Display job data IMMEDIATELY (outside spinner)
        st.success(f"Job details loaded in {elapsed_time:.1f} seconds")
        
        # Get job data
        job_posting = job_data
        
        if job_posting:
            # Display job information immediately (no waiting)
            st.markdown("---")
            st.markdown("### Job Analysis")
            
            # Display job card immediately after successful scraping
            from src.ui.components.job_display import display_modern_job_card
            display_modern_job_card(job_data)
            
            # Run content-focused fraud analysis
            st.markdown("---")
            st.markdown("### Content-Focused Fraud Analysis")
            st.info("📊 Analyzing job posting content and company data...")
            
            # Run analysis without profile dependency
            try:
                _run_fraud_analysis_pipeline(job_data)
            except Exception as e:
                st.error(f"Content analysis failed: {str(e)}")
                st.info("Job data is still available above")
    
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}") 
        st.error(f"Analysis failed: {str(e)}")
        st.info("Try using the HTML Content or Manual Input tabs instead.")


def render_analysis_section_from_html(html_content: str, fraud_loader=None) -> None:
    """
    Render analysis section for HTML content using core modules.
    """
    st.subheader("Analysis Results")
    
    # Show input method
    st.info("Analyzing from HTML content")
    
    # Analysis pipeline
    with st.spinner("Analyzing job posting from HTML..."):
        try:
            # Step 1: Parse HTML content using ScrapingService
            st.write("Parsing HTML content...")
            # Note: HTML parsing needs to be implemented in ScrapingService
            from src.scraper.linkedin_scraper import scrape_from_html
            job_data = scrape_from_html(html_content)
            
            if not job_data or not job_data.get('success', False):
                error_msg = job_data.get('error_message', 'Unknown parsing error') if job_data else 'Failed to parse HTML'
                st.error(f"Failed to parse HTML content: {error_msg}")
                st.info("Please ensure you copied the complete LinkedIn job page HTML.")
                return
            
            # Continue with the rest of the analysis pipeline using core modules
            _run_analysis_pipeline(job_data)
            
        except Exception as e:
            logger.error(f"HTML analysis failed: {str(e)}")
            st.error(f"HTML analysis failed: {str(e)}")
            st.info("Please check the HTML content and try again.")


def render_analysis_section_from_manual(job_data: Dict[str, Any], fraud_loader=None) -> None:
    """
    Render analysis section for manual input using core modules.
    """
    st.subheader("Analysis Results")
    
    # Show input method
    st.info("Analyzing manually entered job details")
    
    # Analysis pipeline
    with st.spinner("Analyzing job posting from manual input..."):
        try:
            # Run the analysis pipeline using core modules
            _run_analysis_pipeline(job_data)
            
        except Exception as e:
            logger.error(f"Manual analysis failed: {str(e)}")
            st.error(f"Manual analysis failed: {str(e)}")
            st.info("Please check the job details and try again.")


def _run_analysis_pipeline(job_data: Dict[str, Any]) -> None:
    """
    Common analysis pipeline using ONLY core modules.
    """
    # Show clean job card first
    from src.ui.components.job_display import display_modern_job_card
    display_modern_job_card(job_data)
    
    # Extract features using SerializationService (behind the scenes)
    serialization_service = SerializationService()
    features = serialization_service.prepare_single_prediction(job_data)
    
    # Step 4: Use NEW FraudDetectionPipeline for ALL prediction logic (unified)
    st.write("Using unified FraudDetectionPipeline for content-focused analysis...")
    from src.core.fraud_pipeline import FraudDetectionPipeline
    pipeline = FraudDetectionPipeline()
    fraud_result = pipeline.process(job_data)
    prediction = fraud_result.to_ui_dict()
    fraud_indicators = prediction.get('fraud_indicators', {})
    
    # Handle prediction results from FraudDetector
    if not prediction.get('success', True):
        st.error("Analysis Error")
        st.write(prediction.get('error', 'Analysis failed'))
        
        col1, col2 = st.columns(2)
        with col1:
            st.error("ML models required for production system")
            st.info("Please ensure ML models are properly loaded")
        
        with col2:
            if st.button("Skip Analysis"):
                st.stop()
        
        return
    
    # Combine results from FraudDetector (single source of truth)
    result = {
        'job_data': job_data,
        'features': features,
        'prediction': prediction,
        'fraud_indicators': fraud_indicators,
        'timestamp': datetime.now().isoformat(),
        'scraping_method': job_data.get('scraping_method', 'unknown'),
        'explanation': prediction.get('explanation', {}),
        'metadata': {
            'feature_count': len(features) if isinstance(features, dict) else 0,
            'analysis_engine': 'FraudSpot v4.0 - Content-Focused',
            'core_modules_used': True,
            'prediction_method': prediction.get('prediction_method', 'FraudDetector')
        }
    }
    
    # Display results
    render_results(result)
    
    # Track analysis in session state for real statistics
    if 'analysis_history' not in st.session_state:
        st.session_state['analysis_history'] = []
    
    # Add this analysis to history
    analysis_summary = {
        'timestamp': result['timestamp'],
        'is_fraud': prediction.get('is_fraud', False),
        'confidence': prediction.get('confidence', 0.0),
        'company': job_data.get('company_name', ''),
        'method': result['scraping_method']
    }
    st.session_state['analysis_history'].append(analysis_summary)


def _run_fraud_analysis_pipeline(job_data: Dict[str, Any]) -> None:
    """
    Run content-focused fraud analysis pipeline using ONLY core modules.
    
    Args:
        job_data: Scraped job data
    """
    try:
        # Job card already displayed - start with fraud analysis
        from src.ui.components.fraud_dashboard import display_comprehensive_fraud_dashboard

        # Initialize FraudDetectionPipeline (unified system)
        try:
            # Use cached FraudDetectionPipeline to prevent models loading multiple times
            if 'cached_fraud_pipeline' not in st.session_state:
                st.session_state['cached_fraud_pipeline'] = FraudDetectionPipeline()
                logger.info("✅ Created and cached FraudDetectionPipeline instance")
            else:
                logger.info("♻️ Using cached FraudDetectionPipeline instance")
            
            fraud_pipeline = st.session_state['cached_fraud_pipeline']
            
            logger.info("Using cached unified pipeline to prevent duplicate loading")
            
            # Extract job posting data and add unique identifier
            job_posting = job_data
            
            # Add unique analysis identifier to prevent cross-contamination
            job_posting['analysis_id'] = f"{hash(str(job_data))}_{int(time.time() * 1000)}"
            
            logger.info(f"Starting content-focused analysis for job ID: {job_posting.get('analysis_id', 'unknown')}")
            
            if not job_posting:
                st.warning("No job data available for analysis")
                return
            
            # Use FraudDetectionPipeline for content-focused analysis
            enhanced_job_data = {**job_posting}
            
            # Make fresh ML prediction using cached unified pipeline
            logger.info(f"Making content-focused ML prediction for job {enhanced_job_data.get('analysis_id', 'unknown')}")
            fraud_result = fraud_pipeline.process(enhanced_job_data)
            prediction = fraud_result.to_ui_dict()
            logger.info(f"Content-focused ML prediction completed: fraud_probability={prediction.get('fraud_probability', 'N/A')}")
            
            # NO DEFAULT VALUES - USE ONLY REAL PREDICTION DATA
            if not prediction:
                st.error("❌ No prediction data available from ML models")
                return
            
        except Exception as model_error:
            st.error(f"❌ Failed to load ML models: {model_error}")
            st.info("💡 Train models first using: python train_model_cli.py --model all_models --no-interactive")
            
            # PRODUCTION MODE: No rule-based fallback available
            st.error("🤖 Production system requires ML models only")
            st.info("Please ensure ML models are properly loaded and available")
            return  # Exit without attempting analysis
        
        # Handle prediction results  
        if not prediction.get('success', True):
            # PRODUCTION MODE: No rule-based fallback available
            st.error(f"ML analysis failed: {prediction.get('error', 'Unknown error')}")
            st.error("🤖 Production system requires ML models only")
            st.info("Please ensure ML models are properly loaded and available")
            return  # Exit without showing analysis
        else:
            # Use the enhanced job data for analysis
            analysis_data = {**job_posting}  # Start with original job data
            
            # Add prediction results WITHOUT overwriting enriched fields
            for key, value in prediction.items():
                if key not in analysis_data:  # Only add if not already present
                    analysis_data[key] = value
            
            # Preserve content-focused enriched fields (both field name formats)
            enriched_fields = {
                'company_legitimacy_score': job_posting.get('company_legitimacy_score'),
                'content_quality_score': job_posting.get('content_quality_score'),
                'contact_risk_score': job_posting.get('contact_risk_score'),
                'company_followers': job_posting.get('company_followers') or job_posting.get('followers'),
                'company_employees': job_posting.get('company_employees') or job_posting.get('employees_in_linkedin'),
                'company_founded': job_posting.get('company_founded') or job_posting.get('founded'),
                'company_size': job_posting.get('company_size'),
                'has_company_logo': job_posting.get('has_company_logo'),
                'has_company_website': job_posting.get('has_company_website'),
                # Add alternative field mappings
                'followers': job_posting.get('followers') or job_posting.get('company_followers'),
                'employees_in_linkedin': job_posting.get('employees_in_linkedin') or job_posting.get('company_employees'),
                'founded': job_posting.get('founded') or job_posting.get('company_founded'),
                'job_industries': job_posting.get('job_industries'),
                'industry': job_posting.get('industry') or job_posting.get('job_industries')
            }
            
            # Only update if the enriched value exists and is not None (0 is a valid value)
            for field, value in enriched_fields.items():
                if value is not None:
                    analysis_data[field] = value
                    logger.info(f"✅ PRESERVED ENRICHED FIELD: {field}={value}")
            
            logger.info(f"🎯 CONTENT-FOCUSED ANALYSIS DATA - content_score: {analysis_data.get('content_quality_score', 'MISSING')}, company_score: {analysis_data.get('company_legitimacy_score', 'MISSING')}")
            
            # Render content-focused fraud analysis dashboard
            display_comprehensive_fraud_dashboard(analysis_data)
            
            # Store in session state for history
            _store_analysis_result(job_posting, prediction)
    
    except Exception as e:
        logger.error(f"Content-focused fraud analysis pipeline failed: {str(e)}")
        st.error(f"Content-focused fraud analysis failed: {str(e)}")


def _store_analysis_result(job_posting: Dict[str, Any], prediction: Dict[str, Any]) -> None:
    """Store analysis result in session state for history."""
    if 'analysis_history' not in st.session_state:
        st.session_state['analysis_history'] = []
    
    analysis_summary = {
        'timestamp': pd.Timestamp.now(),
        'job_title': job_posting.get('job_title', ''),
        'company': job_posting.get('company_name', ''),
        'fraud_score': prediction.get('fraud_score', 0),
        'risk_level': prediction.get('risk_level', '')
    }
    st.session_state['analysis_history'].append(analysis_summary)