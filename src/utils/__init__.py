"""
Utilities Module for Job Post Fraud Detector

This module provides utility functions and helpers used throughout
the fraud detection system.

Components:
- model_utils: Model operations and utilities
- data_utils: Data validation and processing utilities
- evaluation_utils: Model evaluation and reporting utilities
- cache_manager: Caching functionality for performance optimization

 Version: 3.0.0 - Refactored for DRY compliance
"""

# Utils classes moved to services for DRY compliance
# Use ModelService and EvaluationService instead
# from .data_utils import DataUtils  # Moved to DataProcessor
# from .evaluation_utils import EvaluationUtils  # Moved to EvaluationService
# from .model_utils import ModelUtils  # Moved to ModelService

# Import cache manager functions if available
try:
    from .cache_manager import (
        cache_scraping_result,
        clear_old_cache,
        get_cache_statistics,
        get_cached_result,
        initialize_cache,
    )
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False

__all__ = [
]

# Add cache functions to exports if available
if CACHE_AVAILABLE:
    __all__.extend([
        'initialize_cache',
        'cache_scraping_result',
        'get_cached_result',
        'clear_old_cache',
        'get_cache_statistics'
    ])