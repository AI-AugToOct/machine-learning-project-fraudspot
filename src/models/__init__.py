"""
Models Module for Job Post Fraud Detector

This module provides machine learning model training, prediction, and utilities
for the fraud detection system.

Components:
- train_model: Model training pipeline and evaluation
- predict: Prediction pipeline for new job postings
- model_utils: Utility functions for model management

 Version: 1.0.0
"""

from .train_model import (
    load_training_data,
    preprocess_training_data,
    train_random_forest,
    train_gradient_boosting,
    train_svm,
    train_ensemble,
    evaluate_model,
    save_model,
    create_training_report
)

from .predict import (
    load_model,
    predict_fraud,
    calculate_confidence_score,
    determine_risk_level,
    extract_top_features,
    generate_explanation,
    create_prediction_report
)

from .model_utils import (
    validate_model_input,
    get_model_metadata,
    compare_models,
    update_model_performance,
    cleanup_old_models,
    export_model_artifacts
)

__all__ = [
    # train_model functions
    'load_training_data',
    'preprocess_training_data',
    'train_random_forest',
    'train_gradient_boosting',
    'train_svm',
    'train_ensemble',
    'evaluate_model',
    'save_model',
    'create_training_report',
    
    # predict functions
    'load_model',
    'predict_fraud',
    'calculate_confidence_score',
    'determine_risk_level',
    'extract_top_features',
    'generate_explanation',
    'create_prediction_report',
    
    # model_utils functions
    'validate_model_input',
    'get_model_metadata',
    'compare_models',
    'update_model_performance',
    'cleanup_old_models',
    'export_model_artifacts'
]