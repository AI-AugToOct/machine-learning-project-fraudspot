"""
Unified Fraud Detection Pipeline - Single Source of Truth

This module consolidates ALL fraud detection logic into a single pipeline.
Replaces: fraud_detector.py, solid_fraud_detector.py, realistic_fraud_predictor.py,
         ensemble_predictor.py, prediction_engine.py, prediction_enhancer.py
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

# Suppress numpy warnings about mean of empty slices
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')

from .constants import ModelConstants
from .data_model import FraudResult, JobPostingData
from .feature_engine import FeatureEngine

logger = logging.getLogger(__name__)


class FraudDetectionPipeline:
    """
    Single unified pipeline for ALL fraud detection operations.
    
    This class consolidates all fraud prediction functionality:
    - Data transformation and validation
    - Feature engineering
    - ML model predictions (ensemble of all models)
    - Risk classification
    - Result explanation and confidence scoring
    """
    
    def __init__(self, model_dir: str = "models"):
        """Initialize the unified fraud detection pipeline."""
        self.model_dir = Path(model_dir)
        self.feature_engine = FeatureEngine()
        
        # Load all available models
        self.models = self._load_models()
        
        # Load ensemble configuration from model_metrics.json
        self.ensemble_config = self._load_ensemble_config()
        
        logger.info(f"FraudDetectionPipeline initialized with {len(self.models)} models")
    
    def set_model_pipeline(self, pipeline):
        """Set sklearn pipeline for training integration."""
        # Store reference to training pipeline for compatibility with PipelineManager
        self.training_pipeline = pipeline
        logger.info("Training pipeline set for FraudDetectionPipeline")
    
    def _load_models(self) -> Dict[str, Any]:
        """Load all available ML models."""
        models = {}
        model_files = {
            'logistic_regression': 'logistic_regression_pipeline.joblib',
            'random_forest': 'random_forest_pipeline.joblib',
            'svm': 'svm_pipeline.joblib',
            'naive_bayes': 'naive_bayes_pipeline.joblib'
        }
        
        for model_name, filename in model_files.items():
            model_path = self.model_dir / filename
            if model_path.exists():
                try:
                    model = joblib.load(model_path)
                    models[model_name] = model
                    logger.info(f"✅ Loaded {model_name} model from {filename}")
                except Exception as e:
                    logger.warning(f"❌ Failed to load {model_name}: {e}")
            else:
                logger.warning(f"❌ Model file not found: {filename}")
        
        return models
    
    def _load_ensemble_config(self) -> Dict[str, Any]:
        """Load ensemble configuration from model metrics."""
        metrics_path = self.model_dir / "model_metrics.json"
        if metrics_path.exists():
            try:
                with open(metrics_path, 'r') as f:
                    data = json.load(f)
                    config = data.get('ensemble_config', {})
                    logger.info(f"✅ Loaded ensemble config: threshold={config.get('fraud_threshold', 0.65)}")
                    return config
            except Exception as e:
                logger.error(f"Failed to load ensemble config: {e}")
        
        # Default configuration
        default_config = {
            'fraud_threshold': 0.65,
            'confidence_threshold': 0.75,
            'weight_calculation_method': 'f1_based'
        }
        logger.info("Using default ensemble configuration")
        return default_config
    
    def process(self, raw_input: Dict[str, Any]) -> FraudResult:
        """
        Main pipeline entry point - process raw input and return complete result.
        
        Args:
            raw_input: Raw job posting data from scraper/UI
            
        Returns:
            FraudResult: Complete fraud analysis result
        """
        logger.info("🚀 FRAUD PIPELINE START ================================")
        
        # Step 1: Transform to unified data model
        job_data = JobPostingData.from_scraper_data(raw_input)
        logger.info(f"✅ Created unified JobPostingData: content_quality={job_data.content_quality_score}")
        
        # Step 2: Generate ML features
        try:
            feature_dict = self.feature_engine.generate_features(job_data.to_ml_features())
            logger.info(f"✅ Generated {len(feature_dict)} features")
        except Exception as e:
            logger.error(f"❌ Feature generation failed: {e}")
            feature_dict = job_data.to_ml_features()
        
        # Step 3: ML predictions (ensemble)
        ml_predictions = self._get_ml_predictions(feature_dict)
        
        # Step 4: Calculate final fraud probability
        fraud_probability = self._calculate_final_probability(ml_predictions, job_data)
        
        # Step 5: Risk assessment
        risk_level, risk_score = self._assess_risk(fraud_probability, job_data)
        
        # Step 6: Generate explanations
        explanation, risk_factors, protective_factors = self._generate_explanation(
            job_data, feature_dict, ml_predictions, fraud_probability
        )
        
        # Step 7: Calculate confidence
        confidence = self._calculate_confidence(ml_predictions, job_data)
        
        # Step 8: Create result
        result = FraudResult(
            is_fraudulent=fraud_probability > 0.5,
            confidence_score=confidence,
            risk_level=risk_level,
            risk_factors=risk_factors,
            protective_factors=protective_factors,
            model_predictions=ml_predictions,
            feature_importance={},  # TODO: Calculate actual feature importance
            fraud_probability=fraud_probability,
            explanation=explanation,
            metrics=self._generate_metrics(job_data, feature_dict, ml_predictions)
        )
        
        logger.info(f"🏁 FRAUD PIPELINE COMPLETE: {fraud_probability:.1%} fraud probability, {risk_level} risk")
        return result
    
    def _get_ml_predictions(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Get predictions from all available models."""
        predictions = {}
        
        if not self.models:
            logger.error("No ML models available - system requires trained models")
            raise ValueError("ML models not available. System cannot function without trained models.")
        
        # Convert to DataFrame for sklearn
        df = pd.DataFrame([features])
        
        for model_name, model in self.models.items():
            try:
                # Try predict_proba first (most models support this)
                if hasattr(model, 'predict_proba'):
                    prediction = model.predict_proba(df)[0][1]  # Get fraud probability
                elif hasattr(model, 'decision_function'):
                    # For models like SGDClassifier, use decision_function and convert to probability
                    decision_score = model.decision_function(df)[0]
                    # Convert decision score to probability using sigmoid
                    import numpy as np
                    prediction = 1 / (1 + np.exp(-decision_score))
                else:
                    # Fallback to basic prediction
                    prediction = float(model.predict(df)[0])
                
                predictions[model_name] = float(prediction)
                logger.info(f"📊 {model_name}: {prediction:.3f}")
            except Exception as e:
                logger.warning(f"❌ {model_name} prediction failed: {e}")
        
        return predictions
    
    def _get_dynamic_weights(self) -> Dict[str, float]:
        """Get dynamic model weights from model_metrics.json."""
        metrics_path = self.model_dir / "model_metrics.json"
        default_weights = {
            'logistic_regression': 0.25,
            'random_forest': 0.30,
            'svm': 0.25,
            'naive_bayes': 0.20
        }
        
        if not metrics_path.exists():
            logger.warning("model_metrics.json not found, using default weights")
            return default_weights
        
        try:
            with open(metrics_path, 'r') as f:
                data = json.load(f)
                models = data.get('models', {})
                
                # Extract weights from metrics
                weights = {}
                for model_name, model_data in models.items():
                    if model_data.get('status') == 'active':
                        weights[model_name] = model_data.get('weight', 0.0)
                
                if weights and sum(weights.values()) > 0:
                    logger.info(f"✅ Using dynamic weights: {weights}")
                    return weights
                else:
                    logger.warning("No valid weights in model_metrics.json, using defaults")
                    return default_weights
                    
        except Exception as e:
            logger.error(f"Failed to load dynamic weights: {e}")
            return default_weights
    
    def _calculate_final_probability(self, predictions: Dict[str, float], job_data: JobPostingData) -> float:
        """Calculate final fraud probability from ensemble of predictions."""
        if not predictions:
            return 0.5  # Unknown
        
        # Load dynamic weights from model metrics
        weights = self._get_dynamic_weights()
        
        weighted_sum = 0
        total_weight = 0
        
        for model, prediction in predictions.items():
            weight = weights.get(model, 0.1)
            weighted_sum += prediction * weight
            total_weight += weight
        
        final_prob = weighted_sum / total_weight if total_weight > 0 else 0.5
        
        # Apply company size adjustment
        if job_data.company_followers and job_data.company_followers > 50000:
            final_prob *= 0.8  # Large companies less likely fraud
        elif job_data.company_employees and job_data.company_employees > 1000:
            final_prob *= 0.85
        
        return np.clip(final_prob, 0.0, 1.0)
    
    def _assess_risk(self, fraud_probability: float, job_data: JobPostingData) -> Tuple[str, float]:
        """Assess risk level based on dynamic threshold from model_metrics.json."""
        # Use dynamic threshold from ensemble config
        threshold = self.ensemble_config.get('fraud_threshold', 0.65)
        
        # Dynamic risk levels based on threshold
        if fraud_probability < threshold * 0.3:  # < 0.195 with default 0.65
            return 'very_low', fraud_probability
        elif fraud_probability < threshold * 0.6:  # < 0.39 with default 0.65
            return 'low', fraud_probability
        elif fraud_probability < threshold:  # < 0.65 with default
            return 'medium', fraud_probability
        elif fraud_probability < threshold * 1.2:  # < 0.78 with default 0.65
            return 'high', fraud_probability
        else:
            return 'very_high', fraud_probability
    
    def _generate_explanation(self, job_data: JobPostingData, features: Dict[str, Any], 
                            predictions: Dict[str, float], fraud_prob: float) -> Tuple[str, List[str], List[str]]:
        """Generate human-readable explanation of the prediction."""
        risk_factors = []
        protective_factors = []
        
        # Analyze content quality factors
        content_quality = job_data.content_quality_score
        if content_quality < 0.3:
            risk_factors.append("Low content quality score")
        elif content_quality < 0.6:
            risk_factors.append("Moderate content quality issues")
        else:
            protective_factors.append(f"High content quality ({content_quality:.1%})")
        
        # Analyze company factors
        if job_data.company_followers:
            if job_data.company_followers > 10000:
                protective_factors.append(f"Large company following ({job_data.get_company_followers('display')})")
            elif job_data.company_followers < 100:
                risk_factors.append("Very small company following")
        
        if job_data.company_employees:
            if job_data.company_employees > 1000:
                protective_factors.append(f"Large company size ({job_data.get_company_employees('display')})")
            elif job_data.company_employees < 10:
                risk_factors.append("Very small company size")
        
        # Generate main explanation
        if fraud_prob < 0.3:
            explanation = f"Low fraud risk ({fraud_prob:.1%}). Job posting shows strong legitimacy indicators."
        elif fraud_prob < 0.7:
            explanation = f"Moderate fraud risk ({fraud_prob:.1%}). Some concerns identified, proceed with caution."
        else:
            explanation = f"High fraud risk ({fraud_prob:.1%}). Multiple red flags detected."
        
        return explanation, risk_factors, protective_factors
    
    def _calculate_confidence(self, predictions: Dict[str, float], job_data: JobPostingData) -> float:
        """Calculate confidence in the prediction."""
        if not predictions:
            return 0.5
        
        # Base confidence on prediction variance
        pred_values = list(predictions.values())
        if len(pred_values) > 1:
            variance = np.var(pred_values)
            confidence = 1.0 - min(variance, 0.25) * 4  # Scale variance to confidence
        else:
            confidence = 0.7
        
        # Boost confidence if we have enriched data
        if job_data.has_enriched_data():
            confidence = min(confidence + 0.15, 1.0)
        
        return confidence
    
    def _generate_metrics(self, job_data: JobPostingData, features: Dict[str, Any], 
                         predictions: Dict[str, float]) -> Dict[str, Any]:
        """Generate metrics for UI display."""
        return {
            'content_quality_score': job_data.content_quality_score,
            'company_legitimacy_score': job_data.company_legitimacy_score,
            'company_followers': job_data.company_followers,
            'company_employees': job_data.company_employees,
            'company_founded': job_data.company_founded,
            'contact_risk_score': job_data.contact_risk_score,
            
            # Use computed scores directly from job_data (calculated from real scraped data)
            'legitimacy_score': job_data.company_legitimacy_score,
            
            # Include additional UI-required scores from features
            'contact_professionalism_score': features.get('professional_language_score', 0),
            'salary_realism_score': features.get('salary_mentioned', 0),
            'company_followers_score': features.get('company_followers_score', 0),
            'company_employees_score': features.get('company_employees_score', 0),
            'company_founded_score': features.get('has_company_founded', 0),
            'experience_score': features.get('experience_level_encoded'),
            'content_quality_score': features.get('content_quality_score'),
            'has_enriched_data': job_data.has_enriched_data(),
            'models_used': list(predictions.keys()),
            'feature_count': len(features)
        }


# Convenience function for backward compatibility
def detect_fraud(job_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function for fraud detection.
    Maintains backward compatibility with existing code.
    """
    pipeline = FraudDetectionPipeline()
    result = pipeline.process(job_data)
    return result.to_ui_dict()