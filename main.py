"""
FraudSpot - LinkedIn Job Fraud Detection System

Professional fraud detection application with clean UI and non-blocking operation.
Version: 3.0.0
"""

import logging
import os
import sys
from datetime import datetime

import streamlit as st

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.app_initializer import get_initialization_status, initialize_fraudspot

# Import configuration and fraud patterns
from src.core.constants import FraudKeywords

# Import modular UI components
from src.ui.components.header import render_page_header
from src.ui.components.input_forms import render_html_input, render_manual_input, render_url_input
from src.ui.components.model_comparison import render_model_comparison_tab
from src.ui.components.sidebar import render_info_panel, render_sidebar
from src.ui.orchestrator import (
    render_analysis_section_from_html,
    render_analysis_section_from_manual,
    render_analysis_section_from_url,
)
from src.ui.utils.streamlit_html import inject_global_css

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def initialize_application() -> None:
    """Initialize all application components using centralized system."""
    if not st.session_state.get('app_initialized', False):
        logger.info("🚀 CENTRALIZED INITIALIZATION: Starting FraudSpot...")
        
        with st.spinner("Loading ML models and initializing components (this happens only once)..."):
            # Use centralized initialization system
            success = initialize_fraudspot()
            
            if success:
                logger.info("✅ CENTRALIZED INITIALIZATION: Success")
                status = get_initialization_status()
                logger.info(f"📋 Initialization status: {status}")
            else:
                logger.error("❌ CENTRALIZED INITIALIZATION: Failed")
                st.error("Application initialization failed. Please refresh the page.")
    else:
        logger.debug("📌 Using centralized cached components")


def setup_page_config() -> None:
    """Configure Streamlit page with professional settings."""
    st.set_page_config(
        page_title="FraudSpot - Job Fraud Detection",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': None,
            'Report a bug': None,
            'About': 'FraudSpot v3.0 - AI-Powered Fraud Detection'
        }
    )
    
    # Inject centralized CSS once
    inject_global_css()


def initialize_fraud_pattern_loader() -> None:
    """Initialize fraud pattern loader if enabled."""
    enable_fraud_patterns = os.getenv('ENABLE_FRAUD_PATTERNS', 'false').lower() in ['true', '1', 'yes']
    
    if enable_fraud_patterns:
        st.info("Fraud pattern detection enabled (demo mode)")
    
    # FraudPatternLoader has been deprecated - using consolidated core modules
    return None


def render_url_analysis_tab(fraud_loader = None) -> None:
    """Render the LinkedIn URL analysis tab."""
    st.markdown("### Enter LinkedIn Job URL")
    url = render_url_input()
    if url:
        render_analysis_section_from_url(url, fraud_loader)


def render_html_analysis_tab(fraud_loader = None) -> None:
    """Render the HTML content analysis tab."""
    st.markdown("### Paste LinkedIn Job HTML")
    st.info("Backup method: If URL scraping fails, copy the LinkedIn job page HTML and paste it here.")
    html_content = render_html_input()
    if html_content:
        render_analysis_section_from_html(html_content, fraud_loader)


def render_manual_analysis_tab(fraud_loader = None) -> None:
    """Render the manual job details entry tab."""
    st.markdown("### Manual Job Details Entry")
    st.info("Alternative method: Enter job details manually for analysis.")
    job_data = render_manual_input()
    if job_data:
        render_analysis_section_from_manual(job_data, fraud_loader)


def render_voting_explanation_analysis_tab() -> None:
    """Render the ensemble voting explanation and training tab."""
    st.markdown("### Ensemble Voting Explanation")
    st.info("ML Pipeline: View how the ensemble voting system makes fraud detection decisions.")
    render_model_comparison_tab()


def render_main_interface(fraud_loader = None) -> None:
    """Render the main application interface with tabs."""
    # Main content area with tabs
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create tabs for different input methods
        tab1, tab2, tab3, tab4 = st.tabs([
            "LinkedIn URL", 
            "HTML Content", 
            "Manual Input", 
            "Voting Explanation"
        ])
        
        with tab1:
            render_url_analysis_tab(fraud_loader)
        
        with tab2:
            render_html_analysis_tab(fraud_loader)
        
        with tab3:
            render_manual_analysis_tab(fraud_loader)

        with tab4:
            render_voting_explanation_analysis_tab()
    
    with col2:
        # Information panel
        render_info_panel()


def main() -> None:
    """
    Main application entry point.
    
    Clean, professional, non-blocking fraud detection system.
    """
    try:
        # Page configuration MUST come first
        setup_page_config()
        
        # Initialize application using centralized system (fixes 4x loading issue)
        initialize_application()
        
        # Initialize fraud pattern loader
        fraud_loader = initialize_fraud_pattern_loader()
        
        # Render header
        render_page_header()
        
        # Render sidebar
        render_sidebar()
        
        # Render main interface
        render_main_interface(fraud_loader)
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error(f"An error occurred: {str(e)}")
        st.info("Please refresh the page and try again.")


if __name__ == "__main__":
    main()