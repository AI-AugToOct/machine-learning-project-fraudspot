"""
Sidebar Component

This module handles the sidebar interface including:
- Analysis settings and options
- Application information
- Real-time statistics
- Info panel with fraud detection tips
"""

import streamlit as st


def render_sidebar() -> None:
    """
    Render the sidebar with additional options and information.
    """
    with st.sidebar:
        st.header("🛠️ Options")
        
        # Analysis options
        st.subheader("Analysis Settings")
        sensitivity = st.slider(
            "Detection Sensitivity",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            help="Higher values increase fraud detection sensitivity"
        )
        
        # Store sensitivity threshold in session state for use by fraud detector
        st.session_state['fraud_detection_threshold'] = 1.0 - sensitivity  # Invert: high sensitivity = low threshold
        
        # Ensemble Model System (No Selection - Always Use Ensemble)
        st.subheader("🤖 Ensemble Model System")
        
        # Show ensemble status and models
        try:
            from ...core.ensemble_predictor import EnsemblePredictor
            ensemble = EnsemblePredictor()
            
            # Load models first, then get status
            ensemble.load_models()
            status = ensemble.get_model_status()
            
            # Always use ensemble - no selection dropdown
            st.success("🎯 **Active System: Ensemble Voting**")
            st.markdown("**Combined Models:**")
            
            # Show status of each model in ensemble
            model_display_names = {
                'random_forest': '🌲 Random Forest',
                'logistic_regression': '📈 Logistic Regression',
                'naive_bayes': '🧠 Naive Bayes',
                'svm': '🎯 Support Vector Machine'
            }
            
            models_loaded = 0
            for model_name, display_name in model_display_names.items():
                model_status = status['model_status'].get(model_name, 'missing')
                if model_status == 'loaded':
                    st.markdown(f"  • {display_name} ✅")
                    models_loaded += 1
                else:
                    st.markdown(f"  • {display_name} ❌ (missing)")
            
            # Store ensemble settings in session state (no user selection)
            st.session_state['selected_model'] = 'ensemble'
            st.session_state['use_ensemble'] = True
            
            # Show voting strategy
            st.info("🗳️ **Strategy:** Majority vote from all available models")
            
            # Show warning if not all models available
            if models_loaded < 4:
                st.warning(f"⚠️ Only {models_loaded}/4 models loaded. Train missing models for best results.")
            else:
                st.success(f"✅ All {models_loaded}/4 models ready")
                
        except Exception as e:
            st.error(f"❌ Error loading ensemble: {str(e)}")
            st.info("💡 Train models first using: python train_model_cli.py --model all_models")
            # Fallback to single model if ensemble completely fails
            st.session_state['selected_model'] = 'random_forest'
            st.session_state['use_ensemble'] = False
        
        # Batch processing option
        st.subheader("Batch Processing")
        if st.button("Upload Multiple URLs"):
            st.info("Batch processing feature coming soon!")
        
        # Application info
        st.divider()
        st.subheader("ℹ️ About")
        st.write("""
        This application analyzes job postings using:
        - Natural Language Processing
        - Machine Learning Classification
        - Pattern Recognition
        - Suspicious Keyword Detection
        """)
        
        # Real-time Statistics (when available)
        st.divider()
        st.subheader("📊 Statistics")
        
        # Only show statistics if we have real data
        if 'analysis_history' in st.session_state and st.session_state['analysis_history']:
            history = st.session_state['analysis_history']
            total_analyses = len(history)
            fraud_detected = sum(1 for analysis in history if analysis.get('is_fraud', False))
            fraud_rate = (fraud_detected / total_analyses * 100) if total_analyses > 0 else 0
            
            st.metric("Total Analyses", total_analyses)
            st.metric("Fraud Detected", fraud_detected)
            st.metric("Fraud Rate", f"{fraud_rate:.1f}%")
        else:
            st.info("📈 Statistics will appear after analyzing job postings")


def render_info_panel() -> None:
    """
    Render an information panel with tips and guidelines.
    """
    st.subheader("💡 Fraud Detection Tips")
    
    with st.expander("Common Red Flags"):
        st.write("""
        - Requests for upfront payments
        - Vague job descriptions
        - Personal email addresses instead of company domains
        - Promises of easy money or high pay for minimal work
        - Urgency tactics ("limited time offer")
        - Poor grammar and spelling
        - Requests for personal financial information
        """)
    
    with st.expander("Positive Indicators"):
        st.write("""
        - Detailed job description with specific requirements
        - Company website and professional email domain
        - Realistic salary expectations
        - Clear application process
        - Professional communication
        - Verifiable company information
        """)
    
    with st.expander("Safety Tips"):
        st.write("""
        - Research the company independently
        - Never pay upfront fees
        - Verify job postings on company websites
        - Be cautious of remote positions with immediate starts
        - Trust your instincts
        """)