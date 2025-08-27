"""
Utilities Module for Job Post Fraud Detector

This module provides utility functions and helpers used throughout
the fraud detection system.

Components:
- text_preprocessing: Text cleaning and preprocessing functions
- validators: Input validation and sanitization
- cache_manager: Caching functionality for performance optimization

 Version: 1.0.0
"""

from .text_preprocessing import (
    clean_text,
    remove_html_tags,
    normalize_whitespace,
    remove_special_characters,
    expand_contractions,
    remove_stopwords,
    lemmatize_text,
    extract_urls,
    extract_phone_numbers
)

from .validators import (
    validate_url,
    validate_email,
    validate_job_data,
    sanitize_input,
    check_data_quality
)

from .cache_manager import (
    initialize_cache,
    cache_scraping_result,
    get_cached_result,
    clear_old_cache,
    get_cache_statistics
)

__all__ = [
    # text_preprocessing functions
    'clean_text',
    'remove_html_tags',
    'normalize_whitespace',
    'remove_special_characters',
    'expand_contractions',
    'remove_stopwords',
    'lemmatize_text',
    'extract_urls',
    'extract_phone_numbers',
    
    # validators functions
    'validate_url',
    'validate_email',
    'validate_job_data',
    'sanitize_input',
    'check_data_quality',
    
    # cache_manager functions
    'initialize_cache',
    'cache_scraping_result',
    'get_cached_result',
    'clear_old_cache',
    'get_cache_statistics'
]