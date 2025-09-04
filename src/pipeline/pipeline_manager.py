"""
Pipeline Manager - REFACTORED FOR DRY CONSOLIDATION
This module orchestrates the ML pipeline using ONLY core modules.
ALL business logic has been moved to core modules.

Version: 3.0.0 - DRY Consolidation
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Import ONLY from core modules (single source of truth)
from ..core import DataProcessor, FeatureEngine, FraudDetectionPipeline, ModelConstants
# Ensemble model removed for content-focused approach
from ..services import ModelService, SerializationService
from ..utils.model_utils import ModelUtils

logger = logging.getLogger(__name__)


class PipelineManager:
    """
    REFACTORED Pipeline Manager using DRY principles.
    
    This class now serves as a simple orchestrator that delegates ALL business logic
    to core modules. No duplication, no embedded logic.
    
    Uses:
    - DataProcessor: ALL data preprocessing
    - FeatureEngine: ALL feature engineering
    - FraudDetector: ALL fraud detection logic
    - ModelService: Model management
    - SerializationService: Data conversion
    """
    
    def __init__(self, model_type: str = 'ensemble', 
                 balance_method: str = 'borderline_smote', 
                 scaling_method: str = 'standard', 
                 config: Dict[str, Any] = None):
        """
        Initialize pipeline manager with enhanced ensemble models (Phase 2).
        
        Args:
            model_type: ML model type
            balance_method: Class balancing method  
            scaling_method: Feature scaling method
            config: Additional configuration
        """
        # Use constants from core module
        self.config = config or ModelConstants.DEFAULT_MODEL_CONFIG.copy()
        self.config.update({
            'model_type': model_type,
            'balance_method': balance_method,
            'scaling_method': scaling_method
        })
        
        # Initialize core modules directly (unified architecture)
        self.data_processor = DataProcessor(balance_method, scaling_method)
        self.feature_engine = FeatureEngine()
        self.fraud_detector = FraudDetectionPipeline()
        
        # Initialize services directly
        self.model_service = ModelService()
        self.serialization_service = SerializationService()
        
        # Pipeline state
        self.data = None
        self.trained_model = None
        self.pipeline = None
        self.training_results = {}
        
        logger.info("PipelineManager initialized with DRY architecture")
    
    def load_data(self, data_path: str, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Load and validate training data.
        
        Args:
            data_path: Path to CSV training data
            sample_size: Optional sample size for testing
            
        Returns:
            pd.DataFrame: Loaded data
        """
        logger.info(f"Loading data from {data_path}")
        
        try:
            # Load raw data
            self.data = pd.read_csv(data_path)
            
            if self.data.empty:
                raise ValueError("No data loaded")
            
            # Sample if requested
            if sample_size and len(self.data) > sample_size:
                self.data = self.data.sample(
                    n=sample_size, 
                    random_state=self.config['random_state']
                )
                logger.info(f"Sampled {sample_size} records")
            
            logger.info(f"Data loaded successfully: {self.data.shape}")
            return self.data
            
        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}")
            raise
    
    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare data for training using CORRECTED pipeline order.
        
        BEST PRACTICE: FeatureEngine → DataProcessor → Model
        This ensures DataProcessor only sees ML features, not raw categorical columns.
        
        Returns:
            Tuple: X_train, X_test, y_train, y_test (all processed)
        """
        if self.data is None or self.data.empty:
            raise ValueError("No data loaded. Call load_data() first.")
        
        logger.info("Preparing data for training with corrected pipeline architecture")
        
        try:
            # Step 1: Split data
            X = self.data.drop('fraudulent', axis=1)
            y = self.data['fraudulent']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config['test_size'],
                random_state=self.config['random_state'],
                stratify=y
            )
            
            # Step 2: CORRECTED ORDER - Generate features FIRST with FeatureEngine
            logger.info("Step 2: Feature engineering (before data processing)")
            X_train_features = self.feature_engine.fit_transform(X_train)
            X_test_features = self.feature_engine.transform(X_test)
            
            # Step 3: CORRECTED ORDER - Process features with DataProcessor
            logger.info("Step 3: Data processing (scaling ML features only)")
            X_train_processed = self.data_processor.fit_transform(X_train_features)
            X_test_processed = self.data_processor.transform(X_test_features)
            
            # Step 4: Balance classes using DataProcessor
            logger.info("Step 4: Class balancing")
            X_train_balanced, y_train_balanced = self.data_processor.balance_classes(
                X_train_processed, y_train
            )
            
            logger.info(f"Corrected data preparation completed - Train: {X_train_balanced.shape}, Test: {X_test_processed.shape}")
            logger.info("Pipeline now follows best practice: Raw → FeatureEngine → DataProcessor → Model")
            
            return X_train_balanced, X_test_processed, y_train_balanced, y_test
            
        except Exception as e:
            logger.error(f"Data preparation failed: {str(e)}")
            raise
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Train ML model using utilities.
        
        Args:
            X_train: Training features (can be balanced with SMOTE)
            y_train: Training targets (can be balanced with SMOTE)
            
        Returns:
            Dict: Basic training info (NOT performance metrics - use evaluate_model for that)
        """
        logger.info(f"Training {self.config['model_type']} model")
        
        try:
            # Get model instance from utilities (with SMOTE awareness to prevent double balancing)
            balance_method = self.config.get('balance_method', 'smote')
            use_smote = balance_method == 'smote'
            
            self.trained_model = ModelUtils.get_model_instance(
                self.config['model_type'], 
                random_state=self.config['random_state'],
                use_smote=use_smote
            )
            
            # Train the model
            self.trained_model.fit(X_train, y_train)
            
            # Store basic training info (NOT performance metrics)
            self.training_results = {
                'model_type': self.config['model_type'],
                'training_samples': len(X_train),
                'feature_count': X_train.shape[1],
                'trained': True
            }
            
            logger.info(f"Model training completed - {len(X_train)} samples, {X_train.shape[1]} features")
            logger.warning("⚠️  Training metrics removed - use evaluate_model() for realistic performance")
            
            return self.training_results
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate trained model.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dict: Evaluation results
        """
        if self.trained_model is None:
            raise ValueError("No trained model. Call train_model() first.")
        
        logger.info("Evaluating model on test data")
        
        try:
            # Make predictions
            y_pred = self.trained_model.predict(X_test)
            y_pred_proba = None
            if hasattr(self.trained_model, 'predict_proba'):
                y_pred_proba = self.trained_model.predict_proba(X_test)
            
            # Calculate evaluation metrics using utilities
            evaluation_results = ModelUtils.calculate_comprehensive_metrics(
                y_test, y_pred, y_pred_proba
            )
            
            logger.info(f"Model evaluation completed - Test F1: {evaluation_results['f1_score']:.3f}")
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            raise
    
    def build_pipeline(self) -> Pipeline:
        """
        Build sklearn pipeline with CORRECTED component order.
        
        BEST PRACTICE ORDER: FeatureEngine → DataProcessor → Model
        This ensures consistency between training and prediction.
        
        Returns:
            Pipeline: Complete deployment pipeline
        """
        if any(comp is None for comp in [self.data_processor, self.feature_engine, self.trained_model]):
            raise ValueError("Pipeline components not ready. Complete training first.")
        
        logger.info("Building deployment pipeline with corrected architecture")
        
        try:
            # Create pipeline with CORRECTED order - same as training
            pipeline_steps = [
                ('feature_engine', self.feature_engine),    # First: Raw → ML features
                ('data_processor', self.data_processor),    # Second: Scale ML features
                ('model', self.trained_model)               # Third: Make prediction
            ]
            
            self.pipeline = Pipeline(pipeline_steps)
            
            # Set fraud detector to use this pipeline
            self.fraud_detector.set_model_pipeline(self.pipeline)
            
            logger.info("Deployment pipeline built successfully with corrected order: FeatureEngine → DataProcessor → Model")
            return self.pipeline
            
        except Exception as e:
            logger.error(f"Pipeline building failed: {str(e)}")
            raise
    
    def save_pipeline(self, model_name: str = None) -> bool:
        """
        Save pipeline using ModelService.
        
        Args:
            model_name: Name for saved model
            
        Returns:
            bool: Success status
        """
        if self.pipeline is None:
            self.build_pipeline()
        
        model_name = model_name or f"{self.config['model_type']}_pipeline"
        
        logger.info(f"Saving pipeline: {model_name}")
        
        try:
            # Prepare metadata
            metadata = {
                'config': self.config,
                'training_results': self.training_results,
                'pipeline_version': '2.0.0-DRY',
                'core_modules_used': True
            }
            
            # Use ModelService for saving
            success = self.model_service.save_model(self.pipeline, model_name, metadata)
            
            if success:
                logger.info(f"Pipeline saved successfully: {model_name}")
            else:
                logger.error(f"Pipeline saving failed: {model_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Pipeline saving error: {str(e)}")
            return False
    
    def load_pipeline(self, model_name: str) -> bool:
        """
        Load pipeline using ModelService.
        
        Args:
            model_name: Name of saved model
            
        Returns:
            bool: Success status
        """
        logger.info(f"Loading pipeline: {model_name}")
        
        try:
            # Use ModelService for loading
            loaded_pipeline = self.model_service.load_model(model_name)
            
            if loaded_pipeline is None:
                logger.error(f"Failed to load pipeline: {model_name}")
                return False
            
            self.pipeline = loaded_pipeline
            
            # Set fraud detector to use loaded pipeline
            self.fraud_detector.set_model_pipeline(self.pipeline)
            
            logger.info(f"Pipeline loaded successfully: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Pipeline loading error: {str(e)}")
            return False
    
    def predict(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make fraud prediction using ENSEMBLE VOTING (always).
        No more single model predictions - ensemble is the default and only method.
        
        Args:
            job_data: Raw job posting data
            
        Returns:
            Dict: Complete prediction results from ensemble
        """
        # Check if we're being called from ensemble (avoid recursion)
        if hasattr(self, '_use_ensemble') and not self._use_ensemble:
            # Internal call from ensemble - use original single model
            return self._predict_single_model(job_data)
        
        # Normal call - use ensemble as default
        from ..core.ensemble_predictor import EnsemblePredictor
        
        try:
            ensemble = EnsemblePredictor()
            result = ensemble.predict(job_data)
            logger.info(f"Ensemble prediction completed: {result.get('risk_level', 'Unknown')} ({result.get('voting_result', 'N/A')})")
            return result
            
        except Exception as e:
            logger.error(f"Ensemble prediction failed: {str(e)}")
            # Fallback to Random Forest only if ensemble completely fails
            try:
                logger.warning("Using Random Forest fallback")
                self._use_ensemble = False
                result = self._predict_single_model(job_data)
                result['prediction_method'] = 'Fallback (Random Forest Only)'
                result['ensemble_failed'] = True
                return result
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                return {
                    'success': False,
                    'error': str(e),
                    'fallback_error': str(fallback_error),
                    'is_fraud': True,  # Conservative default
                    'confidence': 0.1
                }
            finally:
                self._use_ensemble = True
    
    def predict_direct_with_features(self, job_data: Dict[str, Any], features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        CRITICAL FIX: Direct prediction using pre-computed features to prevent circular loops.
        This method ensures enriched company/profile data reaches the actual sklearn models.
        """
        if self.pipeline is None:
            raise ValueError("No pipeline available for prediction")
        
        try:
            logger.info(f"🎯 Direct prediction with pre-computed features for {self.config.get('model_type', 'unknown')} model")
            
            # Use the sklearn model directly with features_df (no circular calls)
            prediction = self.pipeline.predict(features_df.iloc[0:1])[0]
            
            # Get probability if available
            if hasattr(self.pipeline, 'predict_proba'):
                prediction_proba = self.pipeline.predict_proba(features_df.iloc[0:1])[0]
                fraud_probability = float(prediction_proba[1])  # Probability of class 1 (fraud)
            elif hasattr(self.pipeline, 'decision_function'):
                # For SVM/SGD - use decision function and convert to probability-like score
                decision_score = self.pipeline.decision_function(features_df.iloc[0:1])[0]
                fraud_probability = 1 / (1 + np.exp(-decision_score))  # Sigmoid conversion
            else:
                # Use binary prediction as rough probability
                fraud_probability = float(prediction)
            
            # Log enriched data usage
            company_size = job_data.get('company_size_category', 'Unknown')
            content_quality = job_data.get('content_quality_score', 'N/A')
            legitimacy = job_data.get('legitimacy_score', 'N/A') 
            logger.info(f"🔧 Using enriched data: size={company_size}, content={content_quality}, legitimacy={legitimacy}")
            
            return {
                'success': True,
                'model_used': True,
                'is_fraud': bool(prediction),
                'fraud_probability': fraud_probability,
                'prediction_method': f'{self.config.get("model_type", "sklearn")} (direct)',
                'risk_factors': []  # Individual models don't generate detailed risk factors
            }
            
        except Exception as e:
            logger.error(f"Direct prediction with features failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'is_fraud': True,  # Conservative default
                'confidence': 0.1
            }
    
    def _predict_single_model(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Original single model prediction method (used internally by ensemble).
        This method is now only called from within the ensemble for individual model predictions.
        """
        if self.fraud_detector is None:
            raise ValueError("No fraud detector available")
        
        try:
            # Use FraudDetectionPipeline for single model prediction
            fraud_result = self.fraud_detector.process(job_data)
            
            # Convert FraudResult to legacy format for compatibility
            prediction_result = fraud_result.to_ui_dict()
            prediction_result.update({
                'success': True,
                'is_fraud': fraud_result.is_fraud,
                'confidence': fraud_result.confidence
            })
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"Single model prediction failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'is_fraud': True,  # Conservative default
                'confidence': 0.1
            }
    
    def predict_batch(self, job_data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Predict fraud for multiple job postings.
        
        Args:
            job_data_list: List of job posting data
            
        Returns:
            List: Prediction results for each job
        """
        logger.info(f"Starting batch prediction for {len(job_data_list)} jobs")
        
        results = []
        for i, job_data in enumerate(job_data_list):
            try:
                result = self.predict(job_data)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch prediction failed for job {i}: {str(e)}")
                results.append({
                    'success': False,
                    'error': str(e),
                    'job_index': i
                })
        
        logger.info(f"Batch prediction completed: {len(results)} results")
        return results
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get available models using ModelService."""
        return self.model_service.get_available_models()
    
    def validate_pipeline(self, model_name: str) -> Dict[str, Any]:
        """Validate pipeline using ModelService."""
        return self.model_service.validate_model(model_name)
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about current pipeline configuration."""
        return {
            'config': self.config,
            'has_data': self.data is not None,
            'has_trained_model': self.trained_model is not None,
            'has_pipeline': self.pipeline is not None,
            'training_results': self.training_results,
            'core_modules': {
                'data_processor': type(self.data_processor).__name__,
                'feature_engine': type(self.feature_engine).__name__,
                'fraud_detector': type(self.fraud_detector).__name__
            },
            'version': '3.0.0-DRY'
        }


# Export main class
__all__ = ['PipelineManager']