"""
Core Module - SINGLE SOURCE OF TRUTH
This module contains all core business logic with zero duplication.

Classes:
- DataProcessor: ALL data preprocessing operations
- FeatureEngine: ALL feature engineering operations  
- FraudDetector: ALL fraud detection and prediction logic

Constants:
- FraudKeywords: All fraud detection keywords (multilingual)
- ModelConstants: All ML model parameters and thresholds
- DataConstants: All data processing constants
- ScrapingConstants: All scraping-related constants
- UIConstants: All UI-related constants

Version: 1.0.0 - DRY Consolidation
"""

from .constants import DataConstants, FraudKeywords, ModelConstants, ScrapingConstants, UIConstants
from .data_model import DataRange, FraudResult, JobPostingData
from .data_processor import DataProcessor, prepare_scraped_data_for_ml
from .feature_engine import FeatureEngine
from .fraud_pipeline import FraudDetectionPipeline, detect_fraud

# Export all core functionality
__all__ = [
    # Core classes
    'DataProcessor',
    'FeatureEngine', 
    'FraudDetectionPipeline',
    'JobPostingData',
    'FraudResult',
    'DataRange',
    
    # Convenience functions
    'prepare_scraped_data_for_ml',
    'detect_fraud',
    
    # Constants
    'FraudKeywords',
    'ModelConstants',
    'DataConstants', 
    'ScrapingConstants',
    'UIConstants'
]