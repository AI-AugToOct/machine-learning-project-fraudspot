"""
Model Utilities for Job Fraud Detection

This module provides common utilities for model operations including
loading, saving, evaluation, and configuration management.

Version: 1.0.0
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


class ModelUtils:
    """Utility class for common model operations."""
    
    @staticmethod
    def save_model_with_metadata(model: Any, model_path: str, metadata: Dict[str, Any] = None) -> bool:
        """
        Save model with comprehensive metadata.
        
        Args:
            model: Trained model to save
            model_path (str): Path to save model
            metadata (Dict[str, Any], optional): Additional metadata
            
        Returns:
            bool: Success status
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save model
            joblib.dump(model, model_path)
            
            # Prepare metadata
            model_metadata = {
                'saved_at': datetime.now().isoformat(),
                'model_type': type(model).__name__,
                'model_path': model_path,
                'version': '1.0.0'
            }
            
            if metadata:
                model_metadata.update(metadata)
            
            # Save metadata
            metadata_path = model_path.replace('.joblib', '_metadata.joblib')
            joblib.dump(model_metadata, metadata_path)
            
            logger.info(f"Model and metadata saved to {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    @staticmethod
    def load_model_with_metadata(model_path: str) -> Tuple[Any, Dict[str, Any]]:
        """
        Load model with its metadata.
        
        Args:
            model_path (str): Path to model file
            
        Returns:
            Tuple[Any, Dict[str, Any]]: Model and metadata
        """
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Load model
            model = joblib.load(model_path)
            
            # Load metadata
            metadata_path = model_path.replace('.joblib', '_metadata.joblib')
            metadata = {}
            
            if os.path.exists(metadata_path):
                metadata = joblib.load(metadata_path)
            
            logger.info(f"Model and metadata loaded from {model_path}")
            return model, metadata
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None, {}
    
    @staticmethod
    def calculate_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                      y_pred_proba: np.ndarray = None) -> Dict[str, Any]:
        """
        Calculate comprehensive model evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities (optional)
            
        Returns:
            Dict[str, Any]: Comprehensive metrics
        """
        try:
            metrics = {
                # Basic metrics
                'accuracy': float(accuracy_score(y_true, y_pred)),
                'precision': float(precision_score(y_true, y_pred, average='binary', zero_division=0)),
                'recall': float(recall_score(y_true, y_pred, average='binary', zero_division=0)),
                'f1_score': float(f1_score(y_true, y_pred, average='binary', zero_division=0)),
                
                # Confusion matrix
                'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
                
                # Classification report
                'classification_report': classification_report(y_true, y_pred, output_dict=True, zero_division=0),
                
                # Sample counts
                'n_samples': len(y_true),
                'n_positive': int(np.sum(y_true)),
                'n_negative': int(len(y_true) - np.sum(y_true))
            }
            
            # Add confusion matrix details
            cm = confusion_matrix(y_true, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                metrics.update({
                    'true_negatives': int(tn),
                    'false_positives': int(fp),
                    'false_negatives': int(fn),
                    'true_positives': int(tp),
                    'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
                    'sensitivity': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
                })
            
            # Add probability-based metrics
            if y_pred_proba is not None:
                if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] > 1:
                    # Binary classification probabilities
                    pos_proba = y_pred_proba[:, 1]
                elif y_pred_proba.ndim == 1:
                    # Single probability array
                    pos_proba = y_pred_proba
                else:
                    pos_proba = None
                
                if pos_proba is not None:
                    try:
                        metrics['roc_auc'] = float(roc_auc_score(y_true, pos_proba))
                        
                        # Precision-Recall AUC
                        from sklearn.metrics import auc, precision_recall_curve
                        precision, recall, _ = precision_recall_curve(y_true, pos_proba)
                        metrics['pr_auc'] = float(auc(recall, precision))
                        
                    except Exception as e:
                        logger.warning(f"Could not calculate AUC metrics: {str(e)}")
                        metrics['roc_auc'] = 0.0
                        metrics['pr_auc'] = 0.0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {'error': str(e)}
    
    @staticmethod
    def extract_feature_importance(model: Any, feature_names: List[str] = None) -> List[Tuple[str, float]]:
        """
        Extract feature importance from model.
        
        Args:
            model: Trained model
            feature_names (List[str], optional): Names of features
            
        Returns:
            List[Tuple[str, float]]: List of (feature_name, importance) tuples
        """
        try:
            importances = None
            
            # Extract based on model type
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # Linear models
                if len(model.coef_.shape) > 1:
                    importances = np.abs(model.coef_[0])
                else:
                    importances = np.abs(model.coef_)
            else:
                logger.warning(f"Model {type(model).__name__} does not have feature importance")
                return []
            
            # Create feature names if not provided
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(len(importances))]
            
            # Combine and sort
            feature_importance = list(zip(feature_names, importances))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            logger.info(f"Extracted {len(feature_importance)} feature importances")
            return feature_importance
            
        except Exception as e:
            logger.error(f"Error extracting feature importance: {str(e)}")
            return []
    
    @staticmethod
    def validate_model_performance(metrics: Dict[str, Any], thresholds: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Validate model performance against thresholds.
        
        Args:
            metrics (Dict[str, Any]): Model performance metrics
            thresholds (Dict[str, float], optional): Performance thresholds
            
        Returns:
            Dict[str, Any]: Validation results
        """
        # Default thresholds for fraud detection
        default_thresholds = {
            'min_accuracy': 0.7,
            'min_precision': 0.6,
            'min_recall': 0.6,
            'min_f1_score': 0.6,
            'min_roc_auc': 0.7
        }
        
        if thresholds:
            default_thresholds.update(thresholds)
        
        try:
            validation_results = {
                'is_valid': True,
                'warnings': [],
                'recommendations': [],
                'scores': {}
            }
            
            # Check each metric against threshold
            for metric_name, threshold in default_thresholds.items():
                metric_key = metric_name.replace('min_', '')
                metric_value = metrics.get(metric_key, 0.0)
                validation_results['scores'][metric_key] = metric_value
                
                if metric_value < threshold:
                    validation_results['is_valid'] = False
                    validation_results['warnings'].append(
                        f"Low {metric_key}: {metric_value:.3f} < {threshold:.3f}"
                    )
            
            # Generate recommendations
            if metrics.get('recall', 0) < 0.6:
                validation_results['recommendations'].append(
                    "Low recall: Consider class balancing or threshold tuning"
                )
            
            if metrics.get('precision', 0) < 0.6:
                validation_results['recommendations'].append(
                    "Low precision: Review feature engineering or model selection"
                )
            
            if metrics.get('f1_score', 0) < 0.6:
                validation_results['recommendations'].append(
                    "Low F1 score: Consider hyperparameter tuning"
                )
            
            # Overall assessment
            f1_score = metrics.get('f1_score', 0)
            if f1_score >= 0.8:
                validation_results['assessment'] = "Excellent performance"
            elif f1_score >= 0.7:
                validation_results['assessment'] = "Good performance"
            elif f1_score >= 0.6:
                validation_results['assessment'] = "Acceptable performance"
            else:
                validation_results['assessment'] = "Poor performance - needs improvement"
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating model performance: {str(e)}")
            return {'is_valid': False, 'error': str(e)}
    
    @staticmethod
    def compare_models(model_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Compare multiple model results.
        
        Args:
            model_results (Dict[str, Dict[str, Any]]): Results for each model
            
        Returns:
            pd.DataFrame: Comparison table
        """
        try:
            comparison_data = []
            
            for model_name, results in model_results.items():
                metrics = results.get('metrics', {})
                
                row = {
                    'Model': model_name,
                    'Accuracy': metrics.get('accuracy', 0.0),
                    'Precision': metrics.get('precision', 0.0),
                    'Recall': metrics.get('recall', 0.0),
                    'F1_Score': metrics.get('f1_score', 0.0),
                    'ROC_AUC': metrics.get('roc_auc', 0.0),
                    'Training_Time': results.get('training_time', 0.0),
                    'Status': 'Success' if results.get('success', False) else 'Failed'
                }
                
                comparison_data.append(row)
            
            # Create DataFrame and sort by F1 score
            comparison_df = pd.DataFrame(comparison_data)
            if not comparison_df.empty:
                comparison_df = comparison_df.sort_values('F1_Score', ascending=False)
                comparison_df = comparison_df.round(4)
            
            logger.info(f"Model comparison completed for {len(model_results)} models")
            return comparison_df
            
        except Exception as e:
            logger.error(f"Error comparing models: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    def get_available_models(models_dir: str = "models") -> List[Dict[str, Any]]:
        """
        Get list of available saved models.
        
        Args:
            models_dir (str): Directory containing saved models
            
        Returns:
            List[Dict[str, Any]]: List of available models with metadata
        """
        try:
            if not os.path.exists(models_dir):
                return []
            
            models = []
            
            for file in os.listdir(models_dir):
                if file.endswith('.joblib') and not file.endswith('_metadata.joblib'):
                    model_path = os.path.join(models_dir, file)
                    metadata_path = model_path.replace('.joblib', '_metadata.joblib')
                    
                    model_info = {
                        'name': file.replace('.joblib', ''),
                        'path': model_path,
                        'size_mb': os.path.getsize(model_path) / 1024 / 1024,
                        'modified': datetime.fromtimestamp(os.path.getmtime(model_path)).isoformat()
                    }
                    
                    # Add metadata if available
                    if os.path.exists(metadata_path):
                        try:
                            metadata = joblib.load(metadata_path)
                            model_info.update(metadata)
                        except Exception as e:
                            logger.warning(f"Could not read metadata for {file}: {str(e)}")
                    
                    models.append(model_info)
            
            # Sort by modification date (newest first)
            models.sort(key=lambda x: x['modified'], reverse=True)
            
            logger.info(f"Found {len(models)} available models")
            return models
            
        except Exception as e:
            logger.error(f"Error getting available models: {str(e)}")
            return []
    
    @staticmethod
    def get_model_instance(model_type: str, random_state: int = 42, **kwargs):
        """
        Get an instance of the specified model type.
        
        Args:
            model_type (str): Type of model to instantiate
            random_state (int): Random state for reproducibility
            **kwargs: Additional model parameters
            
        Returns:
            sklearn model instance
        """
        try:
            if model_type == 'random_forest':
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(
                    n_estimators=kwargs.get('n_estimators', 150),  # More trees for stability
                    max_depth=kwargs.get('max_depth', 15),  # Limit depth to prevent overfitting
                    min_samples_split=kwargs.get('min_samples_split', 20),  # Require more samples to split (prevent overfitting)
                    min_samples_leaf=kwargs.get('min_samples_leaf', 10),  # Require more samples in leaf nodes (prevent overfitting)
                    max_features=kwargs.get('max_features', 'sqrt'),  # Use sqrt of features to reduce dominance
                    class_weight=kwargs.get('class_weight', 'balanced_subsample'),  # Better handling of imbalanced data
                    random_state=random_state,
                    n_jobs=kwargs.get('n_jobs', -1)
                )
            elif model_type == 'logistic_regression':
                from sklearn.linear_model import LogisticRegression
                return LogisticRegression(
                    C=kwargs.get('C', 1.0),
                    max_iter=kwargs.get('max_iter', 1000),
                    random_state=random_state,
                    n_jobs=kwargs.get('n_jobs', -1)
                )
            elif model_type == 'svm':
                # Use SGD Classifier for much faster training on large datasets
                from sklearn.linear_model import SGDClassifier
                return SGDClassifier(
                    loss=kwargs.get('loss', 'hinge'),  # SVM-like loss function
                    alpha=kwargs.get('alpha', 0.0001),  # Regularization strength (similar to 1/C)
                    max_iter=kwargs.get('max_iter', 1000),
                    class_weight=kwargs.get('class_weight', 'balanced'),  # Handle imbalanced data
                    random_state=random_state,
                    n_jobs=kwargs.get('n_jobs', -1)
                )
            elif model_type == 'naive_bayes':
                variant = kwargs.get('variant', 'gaussian')  # Back to GaussianNB for continuous features
                if variant == 'bernoulli':
                    from sklearn.naive_bayes import BernoulliNB
                    return BernoulliNB(
                        alpha=kwargs.get('alpha', 1.0),
                        binarize=kwargs.get('binarize', 0.0)
                    )
                elif variant == 'complement':
                    from sklearn.naive_bayes import ComplementNB
                    return ComplementNB(
                        alpha=kwargs.get('alpha', 1.0)
                    )
                else:  # gaussian
                    from sklearn.naive_bayes import GaussianNB
                    return GaussianNB(
                        var_smoothing=kwargs.get('var_smoothing', 1e-9)
                    )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
                
        except Exception as e:
            logger.error(f"Error creating model instance for {model_type}: {str(e)}")
            raise