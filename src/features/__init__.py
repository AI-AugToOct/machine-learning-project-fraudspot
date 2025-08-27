"""
Features Module for Job Post Fraud Detector

This module provides comprehensive feature extraction and engineering capabilities
for analyzing job postings and detecting fraudulent patterns.

Components:
- text_features: Text-based feature extraction (NLP, suspicious keywords, sentiment)
- structural_features: Structural feature extraction (formatting, completeness, metadata)
- feature_engineering: Feature combination, selection, and preprocessing

 Version: 1.0.0
"""

from .text_features import (
    extract_suspicious_keywords,
    calculate_grammar_score,
    analyze_sentiment,
    calculate_readability_scores,
    extract_contact_patterns,
    detect_urgency_indicators,
    analyze_salary_mentions,
    extract_email_domains,
    calculate_text_statistics,
    extract_ngrams
)

from .structural_features import (
    analyze_job_structure,
    check_required_sections,
    analyze_formatting,
    calculate_description_length_score,
    analyze_experience_requirements,
    check_company_info_completeness,
    analyze_location_specificity,
    detect_application_method
)

from .feature_engineering import (
    create_feature_vector,
    extract_tfidf_features,
    combine_text_features,
    combine_structural_features,
    apply_feature_selection,
    normalize_features,
    engineer_interaction_features,
    create_feature_names
)

__all__ = [
    # text_features functions
    'extract_suspicious_keywords',
    'calculate_grammar_score',
    'analyze_sentiment',
    'calculate_readability_scores',
    'extract_contact_patterns',
    'detect_urgency_indicators',
    'analyze_salary_mentions',
    'extract_email_domains',
    'calculate_text_statistics',
    'extract_ngrams',
    
    # structural_features functions
    'analyze_job_structure',
    'check_required_sections',
    'analyze_formatting',
    'calculate_description_length_score',
    'analyze_experience_requirements',
    'check_company_info_completeness',
    'analyze_location_specificity',
    'detect_application_method',
    
    # feature_engineering functions
    'create_feature_vector',
    'extract_tfidf_features',
    'combine_text_features',
    'combine_structural_features',
    'apply_feature_selection',
    'normalize_features',
    'engineer_interaction_features',
    'create_feature_names'
]