"""
App initialization module for one-time model loading.
Ensures models, feature engines, and configs are loaded only once per app lifetime.
"""

import json
import logging
import os
import streamlit as st
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


def load_model_metrics() -> Dict[str, Any]:
    """Load model metrics and ensemble configuration."""
    metrics_path = Path("models/model_metrics.json")
    if metrics_path.exists():
        try:
            with open(metrics_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load model metrics: {e}")
    return {}


def initialize_fraudspot() -> bool:
    """
    Initialize fraud detection system with models and configurations.
    This runs only once per app lifetime.
    """
    try:
        # Import here to avoid circular imports
        from src.core.fraud_pipeline import FraudDetectionPipeline
        from src.core.feature_engine import FeatureEngine
        
        # Load model metrics and ensemble config
        if 'model_metrics' not in st.session_state:
            st.session_state['model_metrics'] = load_model_metrics()
            ensemble_config = st.session_state['model_metrics'].get('ensemble_config', {})
            logger.info(f"📊 Loaded ensemble config: threshold={ensemble_config.get('fraud_threshold', 0.65)}")
        
        # Initialize and cache the fraud pipeline
        if 'cached_fraud_pipeline' not in st.session_state:
            logger.info("🚀 Loading ML models for the first time...")
            pipeline = FraudDetectionPipeline()
            
            # Pass ensemble config to pipeline if it has the attribute
            if hasattr(pipeline, 'ensemble_config'):
                pipeline.ensemble_config = st.session_state['model_metrics'].get('ensemble_config', {})
                logger.info("✅ Applied ensemble config to pipeline")
            
            st.session_state['cached_fraud_pipeline'] = pipeline
            logger.info(f"✅ Cached {len(pipeline.models)} models")
        
        # Initialize and cache feature engine
        if 'cached_feature_engine' not in st.session_state:
            st.session_state['cached_feature_engine'] = FeatureEngine()
            logger.info("✅ Cached FeatureEngine")
        
        # Mark app as initialized
        st.session_state['app_initialized'] = True
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        return False


def get_initialization_status() -> Dict[str, Any]:
    """Get the current initialization status."""
    status = {
        'initialized': st.session_state.get('app_initialized', False),
        'models_loaded': 0,
        'feature_engine': 'cached_feature_engine' in st.session_state,
        'ensemble_config': {}
    }
    
    if 'cached_fraud_pipeline' in st.session_state:
        pipeline = st.session_state['cached_fraud_pipeline']
        status['models_loaded'] = len(pipeline.models) if hasattr(pipeline, 'models') else 0
    
    if 'model_metrics' in st.session_state:
        status['ensemble_config'] = st.session_state['model_metrics'].get('ensemble_config', {})
    
    return status