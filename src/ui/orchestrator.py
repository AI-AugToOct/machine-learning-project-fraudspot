"""
UI Orchestrator - REFACTORED FOR DRY CONSOLIDATION
This module orchestrates the UI pipeline using ONLY core modules and services.
ALL business logic has been moved to core modules.

Version: 3.0.0 - DRY Consolidation
"""

import logging
import os
import sys
import threading
import time
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx

# Add the src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.ui.components.analysis import render_results
from src.ui.components.job_poster import display_job_poster_details

# Import ONLY from core modules and services (single source of truth)
from ..core import FraudDetector
from ..services import ScrapingService, SerializationService

logger = logging.getLogger(__name__)


def render_analysis_section_from_url(url: str, fraud_loader=None) -> None:
    """
    Render the main analysis section using ONLY core modules.
    
    Args:
        url (str): The LinkedIn job URL to analyze
        fraud_loader: Deprecated, kept for compatibility
    """
    if not url:
        return
    
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
        
        # Use ScrapingService (single source of truth)
        scraping_service = ScrapingService()
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
            
            # Check if profile URL is available for separate fetching
            profile_url = job_posting.get('poster_profile_url')
            if profile_url:
                # Start profile fetching
                _handle_async_profile_fetching(profile_url, scraping_service)
                
                # Check profile status to decide if we should run analysis
                import hashlib
                url_hash = hashlib.md5(profile_url.encode()).hexdigest()[:12]
                profile_key = f"profile_{url_hash}"
                status_key = f"{profile_key}_status"
                
                if status_key in st.session_state and st.session_state[status_key] == 'pending':
                    # Profile is still loading - DO NOT run analysis
                    st.markdown("---")
                    st.markdown("### Fraud Analysis")
                    st.info("⏳ Waiting for profile data to complete analysis...")
                    st.caption("Analysis will begin automatically once profile data is loaded.")
                    return  # Exit without running analysis
                else:
                    # Profile loaded or failed - can run analysis
                    st.markdown("---")
                    st.markdown("### Fraud Analysis")
            else:
                st.info("No profile information available for this job posting")
                st.caption("Profile data may not be available for all job postings")
                
                # No profile URL available, run analysis immediately
                st.markdown("---")
                st.markdown("### Fraud Analysis")
            
            # Run fraud analysis only if we reach here (no pending profile)
            try:
                _run_fraud_analysis_pipeline(job_data)
            except Exception as e:
                st.error(f"Fraud analysis failed: {str(e)}")
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
    try:
        # Job card already displayed - start with fraud analysis
        from src.ui.components.fraud_dashboard import display_comprehensive_fraud_dashboard

        # Initialize FraudDetector with ensemble models
        try:
            from src.core.ensemble_predictor import EnsemblePredictor
            ensemble = EnsemblePredictor()
            ensemble.load_models()
            fraud_detector = FraudDetector(model_pipeline=ensemble)
            
            # Extract job posting data
            job_posting = job_data
            
            if not job_posting:
                st.warning("No job data available for analysis")
                return
            
            # Use FraudDetector for comprehensive analysis
            prediction = fraud_detector.predict_fraud(job_posting, use_ml=True)
            
        except Exception as model_error:
            st.error(f"❌ Failed to load ML models: {model_error}")
            st.info("💡 Train models first using: python train_model_cli.py --model all_models --no-interactive")
            
            # Show button but don't automatically fallback
            if st.button("📊 Use Rule-Based Analysis Instead"):
                fraud_detector = FraudDetector()  # No model - explicit fallback only
                prediction = fraud_detector.predict_fraud(job_posting, use_ml=False)
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
                    prediction = fraud_detector.predict_fraud(job_posting, use_ml=False)
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


@st.fragment(run_every=2)  # Auto-refresh every 2 seconds without full page reload
def _profile_fragment(profile_url: str, scraping_service, profile_key: str, url_hash: str) -> None:
    """Fragment for profile fetching that updates independently."""
    # Key definitions
    status_key = f"{profile_key}_status"
    data_key = f"{profile_key}_data"
    error_key = f"{profile_key}_error"
    timestamp_key = f"{profile_key}_timestamp"
    
    # Check for async results from background thread
    if 'async_results' in st.session_state:
        result_key = f"result_{profile_key}"
        if result_key in st.session_state['async_results']:
            result = st.session_state['async_results'][result_key]
            
            # Update session state from thread result
            if result['status'] == 'complete':
                st.session_state[data_key] = result['data']
                st.session_state[f"{profile_key}_elapsed"] = result['elapsed']
                st.session_state[status_key] = 'complete'
            elif result['status'] == 'error':
                st.session_state[error_key] = result['error']
                st.session_state[status_key] = 'error'
            
            # Clean up the result
            del st.session_state['async_results'][result_key]

    # Get current status or default to pending
    status = st.session_state.get(status_key, 'pending')
    
    if status == 'pending':
        # Calculate elapsed time
        fetch_start_time = st.session_state.get(f"{profile_key}_fetch_start")
        current_time = time.time()
        if fetch_start_time:
            elapsed = int(current_time - fetch_start_time)
        else:
            elapsed = int(current_time - st.session_state.get(timestamp_key, current_time))
        
        # Display loading info
        st.info(f"🔄 Fetching profile data... ({elapsed}s elapsed)")
        
        # Progress indicator
        max_time_seconds = 1800  # 30 minutes maximum
        raw_progress = min(elapsed / max_time_seconds, 1.0)
        progress = max(0.01, raw_progress) if elapsed > 0 else 0.01
        st.progress(progress)
        
        # Status caption
        if elapsed < 30:
            st.caption("⏳ Initial request processing...")
        elif elapsed < 120:
            st.caption("🔄 Profile data being scraped...")  
        else:
            st.caption("⌛ Complex profile - this may take up to 5 minutes")
        
        # Control buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Check Status", key=f"refresh_{profile_key}"):
                pass  # Fragment auto-refreshes
        with col2:
            if st.button("⏹️ Cancel", key=f"cancel_{profile_key}"):
                # Clear the async request
                for key in [status_key, data_key, error_key, timestamp_key, f"{profile_key}_fetch_start"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.info("Profile fetch cancelled")
                return
                
    elif status == 'complete':
        profile_data = st.session_state[data_key]
        elapsed_time = st.session_state.get(f"{profile_key}_elapsed", 0)
        
        if profile_data and profile_data.get('success'):
            st.success(f"✅ Profile loaded successfully in {elapsed_time:.1f} seconds")
            # Display profile information
            from src.ui.components.job_poster import display_job_poster_details
            display_job_poster_details(profile_data)
        else:
            st.warning("⚠️ Profile is private or could not be accessed")
            st.info("💡 This is common for private LinkedIn profiles and does not affect fraud analysis")
        
        # Cleanup button
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🗑️ Clear", key=f"clear_{profile_key}"):
                for key in [status_key, data_key, error_key, f"{profile_key}_elapsed", timestamp_key, f"{profile_key}_fetch_start"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
        with col2:
            st.caption("Profile data cached for this session")
    
    elif status == 'error':
        error_msg = st.session_state[error_key] or "Unknown error"
        st.error(f"❌ Profile fetching failed: {error_msg}")
        
        # Retry/dismiss buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Retry", key=f"retry_{profile_key}"):
                # Reset state to trigger new fetch
                for key in [status_key, data_key, error_key, timestamp_key, f"{profile_key}_fetch_start"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
        with col2:
            if st.button("❌ Dismiss", key=f"dismiss_{profile_key}"):
                for key in [status_key, data_key, error_key, timestamp_key, f"{profile_key}_fetch_start"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.info("Profile fetch dismissed - fraud analysis above remains valid")
                return


def _handle_async_profile_fetching(profile_url: str, scraping_service) -> None:
    """Handle async profile fetching with improved error handling and state management."""
    
    # Create collision-resistant unique key using URL-based hash
    import hashlib
    url_hash = hashlib.md5(profile_url.encode()).hexdigest()[:12]
    profile_key = f"profile_{url_hash}"
    status_key = f"{profile_key}_status"
    data_key = f"{profile_key}_data"
    error_key = f"{profile_key}_error"
    timestamp_key = f"{profile_key}_timestamp"
    
    # Check for stale data (older than 5 minutes)
    current_time = time.time()
    if timestamp_key in st.session_state:
        if current_time - st.session_state[timestamp_key] > 300:  # 5 minutes
            logger.info("Clearing stale profile data")
            for key in [status_key, data_key, error_key, timestamp_key, f"{profile_key}_fetch_start"]:
                if key in st.session_state:
                    del st.session_state[key]
    
    # Initialize session state for this profile if not exists
    if status_key not in st.session_state:
        st.session_state[status_key] = 'pending'
        st.session_state[data_key] = None
        st.session_state[error_key] = None
        # Initialize with actual start time for immediate timer display
        st.session_state[timestamp_key] = current_time
        st.session_state[f"{profile_key}_fetch_start"] = current_time
        
        # Use a shared results dictionary to communicate with thread (Streamlit-compatible)
        if 'async_results' not in st.session_state:
            st.session_state['async_results'] = {}
            
        result_key = f"result_{profile_key}"
        
        # Start async profile fetch in background thread
        def fetch_profile_async():
            try:
                # Record the actual start time locally
                profile_start = time.time()
                logger.info(f"Starting async profile fetch: {profile_url}")
                profile_data = scraping_service.scrape_profile(profile_url)
                profile_elapsed = time.time() - profile_start
                
                # Store result in shared dictionary (thread-safe)
                st.session_state['async_results'][result_key] = {
                    'status': 'complete',
                    'data': profile_data,
                    'elapsed': profile_elapsed,
                    'timestamp': time.time()
                }
                
                logger.info(f"Async profile fetch completed in {profile_elapsed:.1f}s")
                
            except Exception as e:
                logger.error(f"Async profile fetch failed: {str(e)}", exc_info=True)
                # Store error result in shared dictionary
                st.session_state['async_results'][result_key] = {
                    'status': 'error',
                    'error': str(e),
                    'timestamp': time.time()
                }
        
        # Start the thread
        try:
            thread = threading.Thread(target=fetch_profile_async, daemon=True)
            add_script_run_ctx(thread)  # Add Streamlit context for session state access
            thread.start()
            logger.info(f"Started async profile fetch thread for {profile_url}")
        except Exception as e:
            logger.error(f"Failed to start profile fetch thread: {e}")
            st.session_state[status_key] = 'error'
            st.session_state[error_key] = f"Failed to start background fetch: {e}"
    
    # Use fragment for profile display (auto-refreshes every 2 seconds without full page reload)
    _profile_fragment(profile_url, scraping_service, profile_key, url_hash)