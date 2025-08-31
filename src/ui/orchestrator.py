"""
UI Orchestrator - REFACTORED FOR DRY CONSOLIDATION
This module orchestrates the UI pipeline using ONLY core modules and services.
ALL business logic has been moved to core modules.

Version: 3.0.0 - DRY Consolidation
"""

import hashlib
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

from src.ui.components.analysis import render_results
from src.ui.components.job_poster import display_fraud_focused_profile, display_job_poster_details

# Import ONLY from core modules and services (single source of truth)
from src.core import FraudDetector
from src.services import ScrapingService, SerializationService

logger = logging.getLogger(__name__)


def _clear_analysis_session():
    """Clear all analysis-related session state to prevent data pollution between jobs."""
    keys_to_clear = []
    
    # Collect all keys that might cause pollution
    for key in list(st.session_state.keys()):
        if any(prefix in key for prefix in [
            'profile_', 'analysis_done', 'current_job_data', 
            'show_profile_fragment', 'async_results', 'ready_for_analysis',
            'no_profile_analysis_done', 'initial_analysis_done_',
            'cached_fraud_detector', 'cached_ensemble', 'fraud_prediction_',
            'analysis_result_', 'job_features_', 'verification_data_'
        ]):
            keys_to_clear.append(key)
    
    # Clear collected keys
    for key in keys_to_clear:
        del st.session_state[key]
    
    logger.info(f"Cleared {len(keys_to_clear)} session state keys for fresh analysis")


def _get_cached_scraping_service():
    """Get cached ScrapingService to prevent repeated initialization."""
    if 'cached_scraping_service' not in st.session_state:
        st.session_state['cached_scraping_service'] = ScrapingService()
        logger.info("Created cached ScrapingService instance")
    return st.session_state['cached_scraping_service']




def render_analysis_section_from_url(url: str, fraud_loader=None) -> None:
    """
    Render the main analysis section using ONLY core modules.
    
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
        # ONLY wrap the actual scraping call in spinner
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
            
            # Handle profile section FIRST (to start async fetch)
            st.markdown("---")
            st.markdown("### Job Poster Profile")
            
            # Show initial analysis without profile
            st.markdown("---")
            st.markdown("### Initial Fraud Analysis")
            st.info("📊 Analyzing job posting data...")
            
            # Run initial analysis without profile (force fresh analysis each time)
            try:
                _run_fraud_analysis_pipeline(job_data)
            except Exception as e:
                st.error(f"Initial fraud analysis failed: {str(e)}")
                st.info("Job data is still available above")
            
            # Add manual profile fetch button
            profile_url = job_posting.get('poster_profile_url')
            if profile_url:
                st.markdown("---")
                st.markdown("### Profile Verification")
                
                # Check if profile already fetched
                import hashlib
                url_hash = hashlib.md5(profile_url.encode()).hexdigest()[:12]
                profile_key = f"profile_{url_hash}"
                profile_data_key = f"{profile_key}_data"
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    if st.button("🔍 Fetch Profile", key=f"fetch_{profile_key}"):
                        # Clear any existing profile data
                        if profile_data_key in st.session_state:
                            del st.session_state[profile_data_key]
                        
                        with st.spinner("Fetching profile data..."):
                            try:
                                scraping_service = _get_cached_scraping_service()
                                profile_data = scraping_service.scrape_profile(profile_url)
                                st.session_state[profile_data_key] = profile_data
                                st.success("✅ Profile fetched successfully!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Profile fetch failed: {str(e)}")
                
                with col2:
                    st.caption(f"Profile URL: {profile_url[:50]}..." if len(profile_url) > 50 else f"Profile URL: {profile_url}")
                
                # Display profile if available
                if profile_data_key in st.session_state:
                    profile_data = st.session_state[profile_data_key]
                    if profile_data and isinstance(profile_data, dict) and profile_data.get('success'):
                        st.markdown("---")
                        st.markdown("#### Profile Information")
                        try:
                            display_fraud_focused_profile(profile_data, job_data)
                            
                            # Add re-analysis button
                            st.markdown("---")
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                if st.button("🔄 Re-analyze with Profile", key=f"reanalyze_{profile_key}"):
                                    st.markdown("### Updated Fraud Analysis (Including Profile)")
                                    st.info("✅ Including profile verification data in analysis")
                                    try:
                                        enhanced_job_data = {**job_data, 'profile_data': profile_data}
                                        _run_fraud_analysis_pipeline(enhanced_job_data)
                                    except Exception as e:
                                        st.error(f"Re-analysis failed: {str(e)}")
                            with col2:
                                st.caption("Compare the analysis results before and after profile verification")
                        except Exception as e:
                            st.error(f"❌ Error displaying profile: {str(e)}")
                    else:
                        st.warning("⚠️ Profile is private or could not be accessed")
                        st.info("💡 This is common for private LinkedIn profiles")
            else:
                st.info("No profile URL available for this job posting")
    
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
    
    # Step 4: Use FraudDetector for ALL prediction logic (single source of truth)
    st.write("Using FraudDetector for comprehensive analysis...")
    fraud_detector = FraudDetector()
    prediction = fraud_detector.predict_fraud(job_data, use_ml=True)
    fraud_indicators = prediction.get('fraud_indicators', {})
    
    # Handle prediction results from FraudDetector
    if not prediction.get('success', True):
        st.error("Analysis Error")
        st.write(prediction.get('error', 'Analysis failed'))
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Try Rule-Based Analysis", type="primary"):
                prediction = fraud_detector.predict_fraud(job_data, use_ml=False)
                st.rerun()
        
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
            'analysis_engine': 'FraudSpot v3.0 - DRY Architecture',
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
        'company': job_data.get('company_name', 'Unknown'),
        'method': result['scraping_method']
    }
    st.session_state['analysis_history'].append(analysis_summary)


# REMOVED: make_real_prediction function - ALL prediction logic moved to FraudDetector core module


# REMOVED: make_fallback_prediction function - ALL prediction logic moved to FraudDetector core module


# REMOVED: extract_red_flags function - ALL analysis logic moved to FraudDetector core module


# REMOVED: extract_positive_indicators function - ALL analysis logic moved to FraudDetector core module


def _run_fraud_analysis_pipeline(job_data: Dict[str, Any]) -> None:
    """
    Run fraud analysis pipeline using ONLY core modules.
    
    Args:
        job_data: Scraped job data
    """
    # Check if profile data is available for enhanced analysis
    profile_url = job_data.get('poster_profile_url')
    profile_data = None
    
    if profile_url:
        # Get profile data from session state if available
        import hashlib
        profile_hash = hashlib.md5(profile_url.encode()).hexdigest()[:12]
        profile_data_key = f"profile_{profile_hash}_data"
        
        if profile_data_key in st.session_state:
            profile_data = st.session_state[profile_data_key]
            if profile_data and profile_data.get('success'):
                st.info("✅ Including profile verification data in analysis")
            else:
                profile_data = None
                st.info("ℹ️ Profile data not available - analyzing job posting only")
    try:
        # Job card already displayed - start with fraud analysis
        from src.ui.components.fraud_dashboard import display_comprehensive_fraud_dashboard

        # Initialize FraudDetector with ensemble models (force fresh instance)
        try:
            from src.core.ensemble_predictor import EnsemblePredictor
            # Create completely fresh instances to prevent cached predictions
            ensemble = EnsemblePredictor()
            ensemble.load_models()
            fraud_detector = FraudDetector(model_pipeline=ensemble)
            
            logger.info("Created fresh FraudDetector and EnsemblePredictor instances")
            
            # Extract job posting data and add unique identifier
            job_posting = job_data
            
            # Add unique analysis identifier to prevent cross-contamination
            job_posting['analysis_id'] = f"{hash(str(job_data))}_{int(time.time() * 1000)}"
            
            logger.info(f"Starting analysis for job ID: {job_posting.get('analysis_id', 'unknown')}")
            
            if not job_posting:
                st.warning("No job data available for analysis")
                return
            
            # Use FraudDetector for comprehensive analysis
            # Include profile data if available for enhanced analysis
            enhanced_job_data = {**job_posting}
            if profile_data:
                enhanced_job_data['profile_data'] = profile_data
            
            # Wait for profile data if URL exists (synchronous for demo)
            if profile_url and not profile_data:
                st.info("⏳ Waiting for profile data to complete fraud analysis...")
                # Check for async profile completion
                import hashlib
                profile_hash = hashlib.md5(profile_url.encode()).hexdigest()[:12]
                profile_data_key = f"profile_{profile_hash}_data"
                profile_status_key = f"profile_{profile_hash}_status"
                
                # Wait up to 60 seconds for profile data
                max_wait = 60
                wait_time = 0
                while wait_time < max_wait and st.session_state.get(profile_status_key) == 'pending':
                    time.sleep(1)
                    wait_time += 1
                    
                    # Check for completed async results
                    if 'async_results' in st.session_state:
                        result_key = f"result_profile_{profile_hash}"
                        if result_key in st.session_state['async_results']:
                            result = st.session_state['async_results'][result_key]
                            if result['status'] == 'complete':
                                profile_data = result['data']
                                if profile_data and profile_data.get('success'):
                                    enhanced_job_data['profile_data'] = profile_data
                                    st.success("✅ Profile data loaded - including in analysis")
                                break
                            elif result['status'] == 'error':
                                st.warning("⚠️ Profile data failed to load - proceeding with job data only")
                                break
                    
                    # Update UI every 5 seconds
                    if wait_time % 5 == 0:
                        st.info(f"⏳ Still waiting for profile data... ({wait_time}s elapsed)")
                
                if wait_time >= max_wait:
                    st.warning("⚠️ Profile data timeout - proceeding with job data only")
            
            # Make fresh ML prediction with unique job identifier
            logger.info(f"Making ML prediction for job {enhanced_job_data.get('analysis_id', 'unknown')}")
            prediction = fraud_detector.predict_fraud(enhanced_job_data, use_ml=True)
            logger.info(f"ML prediction completed: fraud_score={prediction.get('fraud_score', 'N/A')}")
            
            # Validate prediction data to prevent format errors
            if not prediction:
                prediction = {'fraud_score': 0, 'risk_level': 'Unknown', 'error': 'No prediction data'}
            
            # Ensure required fields exist with safe defaults
            prediction.setdefault('fraud_score', 0)
            prediction.setdefault('risk_level', 'Unknown')
            prediction.setdefault('explanation', {})
            
        except Exception as model_error:
            st.error(f"❌ Failed to load ML models: {model_error}")
            st.info("💡 Train models first using: python train_model_cli.py --model all_models --no-interactive")
            
            # Show button but don't automatically fallback
            if st.button("📊 Use Rule-Based Analysis Instead"):
                fraud_detector = FraudDetector()  # No model - explicit fallback only
                # Use enhanced data with profile if available
                enhanced_job_data = {**job_posting}
                if profile_data:
                    enhanced_job_data['profile_data'] = profile_data
                
                # Also wait for profile data in rule-based mode if URL exists
                if profile_url and not profile_data:
                    st.info("⏳ Waiting for profile data for rule-based analysis...")
                    import hashlib
                    profile_hash = hashlib.md5(profile_url.encode()).hexdigest()[:12]
                    
                    # Wait up to 30 seconds for profile data (shorter for rule-based)
                    max_wait = 30
                    wait_time = 0
                    while wait_time < max_wait and st.session_state.get(f"profile_{profile_hash}_status") == 'pending':
                        time.sleep(1)
                        wait_time += 1
                        
                        # Check for completed async results
                        if 'async_results' in st.session_state:
                            result_key = f"result_profile_{profile_hash}"
                            if result_key in st.session_state['async_results']:
                                result = st.session_state['async_results'][result_key]
                                if result['status'] == 'complete':
                                    profile_data = result['data']
                                    if profile_data and profile_data.get('success'):
                                        enhanced_job_data['profile_data'] = profile_data
                                        st.success("✅ Profile data loaded - including in rule-based analysis")
                                    break
                                elif result['status'] == 'error':
                                    st.warning("⚠️ Profile data failed - using rule-based with job data only")
                                    break
                        
                        if wait_time % 10 == 0:
                            st.info(f"⏳ Still waiting... ({wait_time}s elapsed)")
                    
                    if wait_time >= max_wait:
                        st.warning("⚠️ Profile timeout - using rule-based with job data only")
                
                prediction = fraud_detector.predict_fraud(enhanced_job_data, use_ml=False)
                
                # Validate prediction data to prevent format errors
                if not prediction:
                    prediction = {'fraud_score': 0, 'risk_level': 'Unknown', 'error': 'No prediction data'}
                
                # Ensure required fields exist with safe defaults
                prediction.setdefault('fraud_score', 0)
                prediction.setdefault('risk_level', 'Unknown')
                prediction.setdefault('explanation', {})
            else:
                # Don't show analysis at all - user must explicitly request rule-based
                st.info("Analysis unavailable - ML models not loaded")
                return
        
        # Analysis completed - no need for status message
        
        # Handle prediction results
        if not prediction.get('success', True):
            # Only show rule-based option on explicit user request
            st.error(f"ML analysis failed: {prediction.get('error', 'Unknown error')}")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📊 Use Rule-Based Analysis"):
                    # Use enhanced data with profile if available
                    enhanced_job_data = {**job_posting}
                    if profile_data:
                        enhanced_job_data['profile_data'] = profile_data
                    
                    # Wait for profile data in rule-based fallback mode too
                    if profile_url and not profile_data:
                        st.info("⏳ Loading profile data for rule-based analysis...")
                        import hashlib
                        profile_hash = hashlib.md5(profile_url.encode()).hexdigest()[:12]
                        
                        # Check async results immediately for quick response
                        if 'async_results' in st.session_state:
                            result_key = f"result_profile_{profile_hash}"
                            if result_key in st.session_state['async_results']:
                                result = st.session_state['async_results'][result_key]
                                if result['status'] == 'complete':
                                    profile_data = result['data']
                                    if profile_data and profile_data.get('success'):
                                        enhanced_job_data['profile_data'] = profile_data
                                        st.success("✅ Profile data loaded for rule-based analysis")
                    
                    prediction = fraud_detector.predict_fraud(enhanced_job_data, use_ml=False)
                
                # Validate prediction data to prevent format errors
                if not prediction:
                    prediction = {'fraud_score': 0, 'risk_level': 'Unknown', 'error': 'No prediction data'}
                
                # Ensure required fields exist with safe defaults
                prediction.setdefault('fraud_score', 0)
                prediction.setdefault('risk_level', 'Unknown')
                prediction.setdefault('explanation', {})
                st.rerun()
            with col2:
                if st.button("📄 Show Job Data Only"):
                    # Just display the job data without any analysis
                    st.info("Job data displayed - analysis skipped")
                    return
        else:
            # Prepare data for fraud dashboard with prediction results
            analysis_data = {
                **job_posting,
                **prediction  # Include all prediction results
            }
            
            # Render fraud analysis dashboard
            display_comprehensive_fraud_dashboard(analysis_data)
            
            # Store in session state for history
            _store_analysis_result(job_posting, prediction)
    
    except Exception as e:
        logger.error(f"Fraud analysis pipeline failed: {str(e)}")
        st.error(f"Fraud analysis failed: {str(e)}")


# Profile fetch functionality removed - now handled by scraper directly
# to prevent UI blocking. Profile data comes with initial scrape when available.


def _store_analysis_result(job_posting: Dict[str, Any], prediction: Dict[str, Any]) -> None:
    """Store analysis result in session state for history."""
    if 'analysis_history' not in st.session_state:
        st.session_state['analysis_history'] = []
    
    analysis_summary = {
        'timestamp': pd.Timestamp.now(),
        'job_title': job_posting.get('job_title', 'Unknown'),
        'company': job_posting.get('company_name', 'Unknown'),
        'fraud_score': prediction.get('fraud_score', 0),
        'risk_level': prediction.get('risk_level', 'Unknown')
    }
    st.session_state['analysis_history'].append(analysis_summary)


# Fragment removed - now at module level


