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
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
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
        Calculate comprehensive model evaluation metrics + business impact analysis.
        Enhanced with MODEL_ENHANCEMENT_PLAN Phase 1 business metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities (optional)
            
        Returns:
            Dict[str, Any]: Comprehensive metrics with business impact
        """
        try:
            # Calculate class distribution
            n_positive = int(np.sum(y_true))
            n_negative = int(len(y_true) - n_positive)
            fraud_rate = float(n_positive / len(y_true))
            
            metrics = {
                # Basic metrics
                'accuracy': float(accuracy_score(y_true, y_pred)),
                'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
                'precision': float(precision_score(y_true, y_pred, average='binary', zero_division=0)),
                'recall': float(recall_score(y_true, y_pred, average='binary', zero_division=0)),
                'f1_score': float(f1_score(y_true, y_pred, average='binary', zero_division=0)),
                
                # Imbalanced data metrics
                'matthews_corrcoef': float(matthews_corrcoef(y_true, y_pred)),
                
                # Confusion matrix
                'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
                
                # Classification report
                'classification_report': classification_report(y_true, y_pred, output_dict=True, zero_division=0),
                
                # Sample counts and class distribution
                'n_samples': len(y_true),
                'n_positive': n_positive,
                'n_negative': n_negative,
                'fraud_rate': fraud_rate
            }
            
            # Add warning if accuracy is suspiciously high with imbalanced data
            if metrics['accuracy'] > 0.90 and fraud_rate < 0.10:
                metrics['warning'] = f"⚠️  High accuracy ({metrics['accuracy']:.1%}) with only {fraud_rate:.1%} fraud - check for overfitting!"
            
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
                
                # Enhanced Business Impact Metrics (MODEL_ENHANCEMENT_PLAN Phase 1)
                business_metrics = ModelUtils.calculate_business_impact(cm)
                metrics.update(business_metrics)
            
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
                        
                        # Precision-Recall AUC (more informative for imbalanced data)
                        from sklearn.metrics import auc, average_precision_score, precision_recall_curve
                        precision, recall, _ = precision_recall_curve(y_true, pos_proba)
                        metrics['pr_auc'] = float(auc(recall, precision))
                        metrics['average_precision'] = float(average_precision_score(y_true, pos_proba))
                        
                        # Add baseline comparison for PR-AUC (random classifier performance)
                        fraud_rate = float(np.sum(y_true) / len(y_true))
                        metrics['pr_baseline'] = fraud_rate  # Random classifier PR-AUC = fraud rate
                        
                        # Flag if PR-AUC is not much better than random
                        if metrics['pr_auc'] < fraud_rate * 2:
                            metrics['pr_warning'] = f"⚠️  PR-AUC ({metrics['pr_auc']:.3f}) barely better than random ({fraud_rate:.3f})"
                        
                    except Exception as e:
                        logger.warning(f"Could not calculate AUC metrics: {str(e)}")
                        metrics['roc_auc'] = 0.0
                        metrics['pr_auc'] = 0.0
                        metrics['average_precision'] = 0.0
            
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
    def calculate_business_impact(confusion_matrix: np.ndarray, 
                                costs: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Calculate business impact metrics from MODEL_ENHANCEMENT_PLAN Phase 1.
        
        Args:
            confusion_matrix: 2x2 confusion matrix
            costs: Cost dictionary with business impact values
            
        Returns:
            Dict[str, Any]: Business impact metrics
        """
        # Default business costs for fraud detection
        if costs is None:
            costs = {
                'false_positive': 10,    # Cost of reviewing legitimate job ($10)
                'false_negative': 1000,  # Cost of missing fraud ($1000)
                'true_positive': 5,      # Cost of reviewing detected fraud ($5)
                'true_negative': 0       # No cost for correctly identifying legitimate
            }
        
        try:
            tn, fp, fn, tp = confusion_matrix.ravel()
            
            # Calculate total cost
            total_cost = (
                fp * costs['false_positive'] +
                fn * costs['false_negative'] +
                tp * costs['true_positive'] +
                tn * costs['true_negative']
            )
            
            # Calculate savings (fraud prevented)
            fraud_prevented_value = tp * costs['false_negative']  # Value of fraud we caught
            
            # Calculate ROI
            review_costs = fp * costs['false_positive'] + tp * costs['true_positive']
            roi = (fraud_prevented_value - review_costs) / max(review_costs, 1)
            
            # Calculate business metrics
            business_metrics = {
                # Core business metrics
                'total_cost': float(total_cost),
                'fraud_prevented_value': float(fraud_prevented_value),
                'review_costs': float(review_costs),
                'roi': float(roi),
                
                # Key performance indicators
                'fraud_catch_rate': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,  # Recall
                'false_positive_rate': float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
                'cost_per_detection': float(total_cost / tp) if tp > 0 else float('inf'),
                'cost_per_review': float(review_costs / (fp + tp)) if (fp + tp) > 0 else 0.0,
                
                # Business insights
                'fraud_losses_prevented': float(tp * costs['false_negative']),
                'unnecessary_reviews': int(fp),
                'missed_fraud_cost': float(fn * costs['false_negative']),
                
                # Efficiency metrics
                'precision_cost_ratio': float(tp / (fp + tp)) if (fp + tp) > 0 else 0.0,
                'net_benefit': float(fraud_prevented_value - total_cost)
            }
            
            # Add business assessment
            if roi > 5.0:
                business_metrics['business_assessment'] = "Excellent ROI - High business value"
            elif roi > 2.0:
                business_metrics['business_assessment'] = "Good ROI - Positive business impact"
            elif roi > 0.5:
                business_metrics['business_assessment'] = "Moderate ROI - Break-even performance"
            else:
                business_metrics['business_assessment'] = "Poor ROI - High cost, low value"
            
            # Add recommendations
            recommendations = []
            if business_metrics['false_positive_rate'] > 0.10:
                recommendations.append("High false positive rate - consider threshold tuning")
            if business_metrics['fraud_catch_rate'] < 0.70:
                recommendations.append("Low fraud detection rate - improve recall")
            if business_metrics['cost_per_detection'] > 100:
                recommendations.append("High cost per detection - optimize model efficiency")
            
            business_metrics['business_recommendations'] = recommendations
            
            return business_metrics
            
        except Exception as e:
            logger.error(f"Error calculating business impact: {str(e)}")
            return {'business_error': str(e)}
    
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
        # Enhanced thresholds for fraud detection (MODEL_ENHANCEMENT_PLAN Phase 1)
        default_thresholds = {
            'min_accuracy': 0.7,
            'min_precision': 0.4,      # Lowered - acceptable for fraud detection
            'min_recall': 0.6,         # Prioritize catching fraud
            'min_f1_score': 0.5,       # Realistic for imbalanced data
            'min_roc_auc': 0.7,
            'min_pr_auc': 0.3,         # PR-AUC more important for imbalanced
            'min_mcc': 0.3,            # Matthews Correlation Coefficient
            'max_false_positive_rate': 0.15,  # Business constraint
            'min_roi': 1.0             # Business ROI threshold
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
            
            # Enhanced recommendations with business focus
            if metrics.get('recall', 0) < 0.6:
                validation_results['recommendations'].append(
                    "Low recall: Consider class balancing or threshold tuning - missing too much fraud"
                )
            
            if metrics.get('precision', 0) < 0.3:  # More lenient threshold
                validation_results['recommendations'].append(
                    "Very low precision: Review feature engineering - too many false alarms"
                )
            
            if metrics.get('f1_score', 0) < 0.5:  # More realistic threshold
                validation_results['recommendations'].append(
                    "Low F1 score: Consider hyperparameter tuning or data augmentation"
                )
            
            # Business-specific recommendations
            if metrics.get('roi', 0) < 1.0:
                validation_results['recommendations'].append(
                    "Low ROI: Model costs exceed benefits - optimize for business value"
                )
            
            if metrics.get('false_positive_rate', 0) > 0.15:
                validation_results['recommendations'].append(
                    "High false positive rate: Too many manual reviews - adjust threshold"
                )
            
            if metrics.get('pr_auc', 0) < metrics.get('pr_baseline', 0.05) * 2:
                validation_results['recommendations'].append(
                    "Poor PR-AUC: Model barely better than random - needs major improvements"
                )
            
            # Overall assessment with business context
            f1_score = metrics.get('f1_score', 0)
            roi = metrics.get('roi', 0)
            pr_auc = metrics.get('pr_auc', 0)
            
            # Multi-criteria assessment
            if f1_score >= 0.6 and roi >= 3.0 and pr_auc >= 0.4:
                validation_results['assessment'] = "Excellent performance - Strong business value"
            elif f1_score >= 0.5 and roi >= 2.0 and pr_auc >= 0.3:
                validation_results['assessment'] = "Good performance - Positive business impact"
            elif f1_score >= 0.4 and roi >= 1.0 and pr_auc >= 0.2:
                validation_results['assessment'] = "Acceptable performance - Break-even value"
            elif f1_score >= 0.3 or roi >= 0.5:
                validation_results['assessment'] = "Poor performance - Limited business value"
            else:
                validation_results['assessment'] = "Unacceptable performance - No business value"
            
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
                    'PR_AUC': metrics.get('pr_auc', 0.0),           # More important for imbalanced
                    'MCC': metrics.get('matthews_corrcoef', 0.0),   # Better for imbalanced data
                    'ROI': metrics.get('roi', 0.0),                 # Business metric
                    'Cost_Per_Detection': metrics.get('cost_per_detection', float('inf')),
                    'Training_Time': results.get('training_time', 0.0),
                    'Status': 'Success' if results.get('success', False) else 'Failed'
                }
                
                comparison_data.append(row)
            
            # Create DataFrame and sort by business value (ROI then F1)
            comparison_df = pd.DataFrame(comparison_data)
            if not comparison_df.empty:
                # Sort by ROI first (business value), then F1 score
                comparison_df = comparison_df.sort_values(['ROI', 'F1_Score'], ascending=[False, False])
                comparison_df = comparison_df.round(4)
                
                # Add business rank
                comparison_df['Business_Rank'] = range(1, len(comparison_df) + 1)
            
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
    def get_model_instance(model_type: str, random_state: int = 42, use_smote: bool = False, **kwargs):
        """
        Get an instance of the specified model type.
        
        Args:
            model_type (str): Type of model to instantiate
            random_state (int): Random state for reproducibility
            use_smote (bool): Whether SMOTE will be used (affects class_weight)
            **kwargs: Additional model parameters
            
        Returns:
            sklearn model instance
        """
        try:
            if model_type == 'random_forest':
                from sklearn.ensemble import RandomForestClassifier

                # Avoid double balancing: if SMOTE is used, don't use class_weight
                if use_smote:
                    class_weight = None
                    logger.info("🔄 Random Forest: Using SMOTE balancing, disabled class_weight")
                else:
                    class_weight = kwargs.get('class_weight', 'balanced')
                    logger.info("⚖️  Random Forest: Using class_weight balancing, no SMOTE")
                
                return RandomForestClassifier(
                    n_estimators=kwargs.get('n_estimators', 100),  # Reduced from 150 to prevent overfitting
                    max_depth=kwargs.get('max_depth', 12),  # Reduced from 15 to prevent overfitting
                    min_samples_split=kwargs.get('min_samples_split', 25),  # Increased from 20 to prevent overfitting
                    min_samples_leaf=kwargs.get('min_samples_leaf', 15),  # Increased from 10 to prevent overfitting
                    max_features=kwargs.get('max_features', 'sqrt'),  # Use sqrt of features to reduce dominance
                    class_weight=class_weight,  # Conditional balancing
                    random_state=random_state,
                    n_jobs=kwargs.get('n_jobs', -1)
                )
            elif model_type == 'logistic_regression':
                from sklearn.linear_model import LogisticRegression
                return LogisticRegression(
                    C=kwargs.get('C', 1.0),
                    max_iter=kwargs.get('max_iter', 1000),
                    class_weight=kwargs.get('class_weight', 'balanced'),  # Handle class imbalance
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
            elif model_type == 'ensemble':
                # Ensemble model removed - use random forest as default
                return RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=random_state,
                    n_jobs=-1
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