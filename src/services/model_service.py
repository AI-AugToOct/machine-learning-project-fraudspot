"""
Model Service - SINGLE SOURCE for Model Management
This service handles all ML model operations without duplicating business logic.

Version: 3.0.0 - DRY Consolidation
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

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

from ..core import ModelConstants

logger = logging.getLogger(__name__)


class ModelService:
    """
    SINGLE SOURCE for all model management operations with caching.
    
    This service handles:
    - Model loading and saving with automatic caching
    - Model registry management
    - Model metadata tracking  
    - Model validation and versioning
    - Singleton pattern to prevent multiple instances
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls, models_dir: str = "models"):
        """Implement singleton pattern to prevent multiple instances."""
        if cls._instance is None:
            cls._instance = super(ModelService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize the model service (only once due to singleton).
        
        Args:
            models_dir: Directory containing saved models
        """
        if not ModelService._initialized:
            self.models_dir = models_dir
            self.loaded_models = {}
            self._ensure_models_directory()
            ModelService._initialized = True
            logger.info(f"ModelService initialized as singleton with models_dir: {models_dir}")
        else:
            logger.debug("ModelService singleton already initialized")
    
    def load_model(self, model_name: str, model_path: str = None) -> Optional[Any]:
        """
        Load ML model by name or path with intelligent caching.
        
        Args:
            model_name: Unique model identifier
            model_path: Optional explicit path to model file
            
        Returns:
            Loaded model pipeline or None if loading fails
        """
        try:
            # Check cache first to prevent repeated loading
            if model_name in self.loaded_models:
                cached_model = self.loaded_models[model_name]
                logger.debug(f"Model loaded from cache: {model_name}")
                return cached_model['pipeline']
            
            # Use provided path or construct from model name
            if model_path is None:
                model_path = os.path.join(self.models_dir, f"{model_name}_pipeline.joblib")
            
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return None
            
            # Load the model from disk
            model_pipeline = joblib.load(model_path)
            
            # Cache the loaded model with metadata
            self.loaded_models[model_name] = {
                'pipeline': model_pipeline,
                'path': model_path,
                'loaded_at': self._get_current_timestamp(),
                'file_size': os.path.getsize(model_path)
            }
            
            logger.info(f"Model loaded from disk and cached: {model_name} ({os.path.getsize(model_path)//1024}KB)")
            return model_pipeline
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            return None
    
    def save_model(self, model_pipeline: Any, model_name: str, metadata: Dict[str, Any] = None) -> bool:
        """
        Save ML model with metadata.
        
        Args:
            model_pipeline: Trained model pipeline to save
            model_name: Unique model identifier
            metadata: Optional model metadata
            
        Returns:
            bool: Success status
        """
        try:
            # Construct save paths
            model_path = os.path.join(self.models_dir, f"{model_name}_pipeline.joblib")
            metadata_path = os.path.join(self.models_dir, f"{model_name}_metadata.joblib")
            
            # Save the model pipeline
            joblib.dump(model_pipeline, model_path)
            
            # Prepare and save metadata
            full_metadata = {
                'model_name': model_name,
                'model_path': model_path,
                'saved_at': self._get_current_timestamp(),
                'model_type': getattr(model_pipeline, '__class__', {}).get('__name__', 'Unknown'),
                'pipeline_version': '1.0.0'
            }
            
            if metadata:
                full_metadata.update(metadata)
            
            joblib.dump(full_metadata, metadata_path)
            
            # Update cache
            self.loaded_models[model_name] = {
                'pipeline': model_pipeline,
                'path': model_path,
                'loaded_at': full_metadata['saved_at'],
                'metadata': full_metadata
            }
            
            logger.info(f"Model saved successfully: {model_name} -> {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model {model_name}: {str(e)}")
            return False
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get list of all available models.
        
        Returns:
            List of model information dictionaries
        """
        available_models = []
        
        try:
            if not os.path.exists(self.models_dir):
                return available_models
            
            # Scan for model files
            for filename in os.listdir(self.models_dir):
                if filename.endswith('_pipeline.joblib'):
                    model_name = filename.replace('_pipeline.joblib', '')
                    model_path = os.path.join(self.models_dir, filename)
                    metadata_path = os.path.join(self.models_dir, f"{model_name}_metadata.joblib")
                    
                    # Load metadata if available
                    metadata = {}
                    if os.path.exists(metadata_path):
                        try:
                            metadata = joblib.load(metadata_path)
                        except Exception as e:
                            logger.warning(f"Failed to load metadata for {model_name}: {e}")
                    
                    # Get file stats
                    stat = os.stat(model_path)
                    
                    model_info = {
                        'name': model_name,
                        'path': model_path,
                        'size_mb': round(stat.st_size / (1024 * 1024), 2),
                        'modified_at': stat.st_mtime,
                        'is_loaded': model_name in self.loaded_models,
                        'metadata': metadata
                    }
                    
                    available_models.append(model_info)
            
            # Sort by modification time (newest first)
            available_models.sort(key=lambda x: x['modified_at'], reverse=True)
            
            logger.info(f"Found {len(available_models)} available models")
            
        except Exception as e:
            logger.error(f"Error scanning for models: {str(e)}")
        
        return available_models
    
    def get_best_model(self, criteria: str = 'newest') -> Optional[Dict[str, Any]]:
        """
        Get the best available model based on criteria.
        
        Args:
            criteria: Selection criteria ('newest', 'smallest', 'by_performance')
            
        Returns:
            Best model info or None
        """
        available_models = self.get_available_models()
        
        if not available_models:
            return None
        
        if criteria == 'newest':
            return available_models[0]  # Already sorted by newest
        elif criteria == 'smallest':
            return min(available_models, key=lambda x: x['size_mb'])
        elif criteria == 'by_performance':
            # Look for performance metrics in metadata
            models_with_metrics = [
                model for model in available_models 
                if model['metadata'].get('test_f1_score') is not None
            ]
            if models_with_metrics:
                return max(models_with_metrics, key=lambda x: x['metadata'].get('test_f1_score', 0))
            else:
                return available_models[0]  # Fallback to newest
        
        return available_models[0]
    
    def validate_model(self, model_name: str) -> Dict[str, Any]:
        """
        Validate model integrity and compatibility.
        
        Args:
            model_name: Model to validate
            
        Returns:
            Validation results
        """
        validation_result = {
            'is_valid': False,
            'errors': [],
            'warnings': [],
            'model_info': {}
        }
        
        try:
            # Check if model exists
            model_path = os.path.join(self.models_dir, f"{model_name}_pipeline.joblib")
            if not os.path.exists(model_path):
                validation_result['errors'].append(f"Model file not found: {model_path}")
                return validation_result
            
            # Try to load the model
            try:
                model_pipeline = joblib.load(model_path)
                validation_result['model_info']['loadable'] = True
            except Exception as e:
                validation_result['errors'].append(f"Model failed to load: {str(e)}")
                return validation_result
            
            # Check if model has required methods
            required_methods = ['predict', 'predict_proba']
            for method in required_methods:
                if not hasattr(model_pipeline, method):
                    validation_result['errors'].append(f"Model missing required method: {method}")
            
            # Load and validate metadata
            metadata_path = os.path.join(self.models_dir, f"{model_name}_metadata.joblib")
            if os.path.exists(metadata_path):
                try:
                    metadata = joblib.load(metadata_path)
                    validation_result['model_info']['metadata'] = metadata
                except Exception as e:
                    validation_result['warnings'].append(f"Failed to load metadata: {str(e)}")
            else:
                validation_result['warnings'].append("No metadata file found")
            
            # If no errors, mark as valid
            if not validation_result['errors']:
                validation_result['is_valid'] = True
                logger.info(f"Model validation passed: {model_name}")
            else:
                logger.error(f"Model validation failed: {model_name}")
            
        except Exception as e:
            validation_result['errors'].append(f"Validation error: {str(e)}")
            logger.error(f"Model validation error for {model_name}: {str(e)}")
        
        return validation_result
    
    def get_model_predictions_stats(self, model_name: str) -> Dict[str, Any]:
        """Get prediction statistics for a model (if tracked)."""
        # This would typically integrate with a model monitoring system
        return {
            'model_name': model_name,
            'prediction_count': 0,  # Would track actual usage
            'average_confidence': 0.0,
            'last_prediction': None
        }
    
    def cleanup_old_models(self, keep_count: int = 5) -> int:
        """
        Clean up old model files, keeping only the most recent.
        
        Args:
            keep_count: Number of recent models to keep
            
        Returns:
            Number of models deleted
        """
        try:
            available_models = self.get_available_models()
            
            if len(available_models) <= keep_count:
                return 0
            
            models_to_delete = available_models[keep_count:]
            deleted_count = 0
            
            for model_info in models_to_delete:
                try:
                    # Delete model file
                    os.remove(model_info['path'])
                    
                    # Delete metadata file if exists
                    metadata_path = model_info['path'].replace('_pipeline.joblib', '_metadata.joblib')
                    if os.path.exists(metadata_path):
                        os.remove(metadata_path)
                    
                    # Remove from cache
                    if model_info['name'] in self.loaded_models:
                        del self.loaded_models[model_info['name']]
                    
                    deleted_count += 1
                    logger.info(f"Deleted old model: {model_info['name']}")
                    
                except Exception as e:
                    logger.error(f"Failed to delete model {model_info['name']}: {str(e)}")
            
            logger.info(f"Cleaned up {deleted_count} old models")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error during model cleanup: {str(e)}")
            return 0
    
    def _ensure_models_directory(self):
        """Ensure models directory exists."""
        try:
            os.makedirs(self.models_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create models directory: {str(e)}")
    
    def compare_models(self, model_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Compare multiple model results and create a comparison DataFrame.
        
        Args:
            model_results: Dictionary mapping model names to their metrics
            
        Returns:
            DataFrame with model comparison
        """
        try:
            comparison_data = []
            
            for model_name, results in model_results.items():
                metrics = results.get('metrics', {})
                comparison_data.append({
                    'Model': model_name,
                    'Accuracy': metrics.get('test_accuracy', 0),
                    'F1 Score': metrics.get('test_f1_score', 0),
                    'Precision': metrics.get('test_precision', 0),
                    'Recall': metrics.get('test_recall', 0),
                    'ROC AUC': metrics.get('test_roc_auc', 0),
                    'Training Time': results.get('training_time', 0)
                })
            
            df = pd.DataFrame(comparison_data)
            return df.sort_values('F1 Score', ascending=False).reset_index(drop=True)
            
        except Exception as e:
            logger.error(f"Error comparing models: {str(e)}")
            return pd.DataFrame()
    
    def calculate_model_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              y_pred_proba: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate comprehensive metrics for model evaluation.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='binary'),
                'recall': recall_score(y_true, y_pred, average='binary'),
                'f1_score': f1_score(y_true, y_pred, average='binary')
            }
            
            # Add ROC AUC if probabilities provided
            if y_pred_proba is not None:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            
            logger.info(f"Model metrics calculated: F1={metrics['f1_score']:.3f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating model metrics: {str(e)}")
            return {}
    
    def extract_feature_importance(self, model: Any, feature_names: List[str] = None) -> List[Tuple[str, float]]:
        """
        Extract feature importance from trained model.
        
        Args:
            model: Trained model with feature importance
            feature_names: List of feature names (optional)
            
        Returns:
            List of (feature_name, importance) tuples sorted by importance
        """
        try:
            # Try to get feature importance from the model
            importance = None
            
            # For sklearn models
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # For linear models, use absolute values of coefficients
                importance = np.abs(model.coef_).flatten()
            
            if importance is None:
                logger.warning("Model does not have feature importance information")
                return []
            
            # Create feature names if not provided
            if feature_names is None:
                feature_names = [f'feature_{i}' for i in range(len(importance))]
            
            # Combine names and importance, sort by importance
            feature_importance = list(zip(feature_names, importance))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            logger.info(f"Extracted importance for {len(feature_importance)} features")
            return feature_importance
            
        except Exception as e:
            logger.error(f"Error extracting feature importance: {str(e)}")
            return []
    
    def validate_model_performance(self, metrics: Dict[str, Any], 
                                 thresholds: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Validate model performance against thresholds.
        
        Args:
            metrics: Model evaluation metrics
            thresholds: Performance thresholds to validate against
            
        Returns:
            Validation results with pass/fail for each metric
        """
        try:
            # Default thresholds
            if thresholds is None:
                thresholds = {
                    'accuracy': 0.8,
                    'precision': 0.75,
                    'recall': 0.75,
                    'f1_score': 0.75
                }
            
            validation_results = {
                'overall_pass': True,
                'metric_results': {},
                'warnings': [],
                'recommendations': []
            }
            
            for metric_name, threshold in thresholds.items():
                metric_value = metrics.get(metric_name, 0)
                passed = metric_value >= threshold
                
                validation_results['metric_results'][metric_name] = {
                    'value': metric_value,
                    'threshold': threshold,
                    'passed': passed
                }
                
                if not passed:
                    validation_results['overall_pass'] = False
                    validation_results['warnings'].append(
                        f"{metric_name.title()}: {metric_value:.3f} < {threshold:.3f}"
                    )
            
            # Generate recommendations
            if not validation_results['overall_pass']:
                validation_results['recommendations'].append(
                    "Consider feature engineering or hyperparameter tuning"
                )
            
            logger.info(f"Model validation: {'PASSED' if validation_results['overall_pass'] else 'FAILED'}")
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating model performance: {str(e)}")
            return {'overall_pass': False, 'error': str(e)}

    def _get_current_timestamp(self) -> str:
        """Get current timestamp string."""
        from datetime import datetime
        return datetime.now().isoformat()


# Export main class
__all__ = ['ModelService']