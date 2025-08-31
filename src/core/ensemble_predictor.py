"""
Ensemble Predictor - The Default Fraud Detection Method

This module provides ensemble voting functionality that combines predictions
from Random Forest, Logistic Regression, Naive Bayes, and SVM models to make 
more accurate and reliable fraud predictions using performance-based weighting.

Version: 3.0.0 - 4-Model Performance-Weighted Ensemble
"""

import logging
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """
    Ensemble voting system - combines Random Forest, Logistic Regression, Naive Bayes, and SVM.
    This is now the DEFAULT and ONLY prediction method for fraud detection.
    Uses performance-based weighting derived from F1 scores.
    """
    
    def __init__(self):
        self.models_loaded = False
        self.models = {}
        # Performance-based weights from F1² scores (emphasize performance differences)
        self.model_weights = {
            'random_forest': 0.454,       # F1²=0.552 -> 45.4% - Best performer gets much more weight
            'logistic_regression': 0.245, # F1²=0.298 -> 24.5% - Good secondary model  
            'naive_bayes': 0.152,         # F1²=0.185 -> 15.2% - Supporting vote
            'svm': 0.148                  # F1²=0.180 -> 14.8% - Minimal influence
        }
    
    def load_models(self):
        """Load all trained models for ensemble prediction"""
        from ..pipeline.pipeline_manager import PipelineManager
        
        successful_loads = 0
        for model_type in ['random_forest', 'logistic_regression', 'naive_bayes', 'svm']:
            try:
                pm = PipelineManager(model_type=model_type)
                # Prevent recursion by marking this as internal call
                pm._use_ensemble = False  
                if pm.load_pipeline(model_type):
                    self.models[model_type] = pm
                    successful_loads += 1
                    logger.info(f"✅ Loaded {model_type} for ensemble")
            except Exception as e:
                logger.warning(f"⚠️ Could not load {model_type}: {e}")
                continue
        
        self.models_loaded = True
        logger.info(f"🎯 Ensemble loaded with {successful_loads}/4 models")
        
        if successful_loads == 0:
            raise ValueError("No models available for ensemble prediction")
    
    def predict(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main prediction method using majority voting.
        This is THE fraud detection method for the entire platform.
        """
        if not self.models_loaded:
            self.load_models()
        
        if not self.models:
            raise ValueError("No models available for ensemble prediction")
        
        predictions = {}
        probabilities = {}
        risk_factors_all = []
        
        # Collect predictions from each model
        for model_name, model in self.models.items():
            try:
                result = model.predict(job_data)
                predictions[model_name] = result.get('is_fraud', False)
                probabilities[model_name] = result.get('fraud_probability', 0.5)
                
                # Collect unique risk factors
                factors = result.get('risk_factors', [])
                risk_factors_all.extend(factors)
                
                logger.debug(f"{model_name}: fraud={result.get('is_fraud')}, prob={result.get('fraud_probability', 0):.2%}")
                
            except Exception as e:
                logger.error(f"❌ Model {model_name} prediction failed: {e}")
                continue
        
        if not predictions:
            raise ValueError("All models failed to make predictions")
        
        # Calculate weighted probability average
        weighted_prob = 0.0
        total_weight = 0.0
        for model_name, prob in probabilities.items():
            weight = self.model_weights.get(model_name, 0.33)
            weighted_prob += prob * weight
            total_weight += weight
        
        if total_weight > 0:
            weighted_prob /= total_weight
        
        # Majority voting (at least 50% of models must agree it's fraud)
        fraud_votes = sum(predictions.values())
        total_models = len(predictions)
        is_fraud = fraud_votes >= (total_models / 2)
        
        # Calculate confidence based on model agreement
        if fraud_votes == total_models or fraud_votes == 0:
            confidence = 0.95  # Full agreement (all models agree)
        elif fraud_votes == total_models - 1 or fraud_votes == 1:
            confidence = 0.75  # Strong majority (one dissenter)
        else:
            confidence = 0.60  # Split decision (needs attention)
        
        # Determine risk level based on weighted probability
        if weighted_prob < 0.30:
            risk_level = "VERY LOW"
        elif weighted_prob < 0.50:
            risk_level = "LOW"
        elif weighted_prob < 0.65:
            risk_level = "MODERATE"
        elif weighted_prob < 0.80:
            risk_level = "HIGH"
        else:
            risk_level = "VERY HIGH"
        
        # Special handling for legitimate companies with unverified posters
        if not is_fraud and 0.4 <= weighted_prob <= 0.6:
            # Borderline case - check for legitimacy signals
            has_logo = job_data.get('has_company_logo', 0)
            legitimacy = job_data.get('legitimacy_score', 0)
            
            if has_logo and legitimacy > 0.7:
                risk_level = "LOW"
                weighted_prob *= 0.8  # Reduce probability by 20%
                risk_factors_all.append("✅ Ensemble: Legitimate company indicators detected - risk reduced")
        
        # Remove duplicate risk factors and limit to top 5
        unique_factors = []
        seen = set()
        for factor in risk_factors_all:
            if factor not in seen:
                unique_factors.append(factor)
                seen.add(factor)
            if len(unique_factors) >= 5:
                break
        
        # Debug logging for ensemble prediction
        logger.info(f"🎯 Ensemble final: fraud_score={weighted_prob:.3f}, fraud_votes={fraud_votes}/{total_models}, confidence={confidence}")
        
        return {
            'success': True,
            'model_failed': False,
            'model_used': True,
            'is_fraud': is_fraud,
            'fraud_probability': weighted_prob,
            'fraud_score': weighted_prob,
            'confidence': confidence,
            'risk_level': risk_level,
            'prediction_method': 'Ensemble Voting (RF+LR+SVM)',
            'voting_result': f"{fraud_votes}/{total_models} models flagged as fraud",
            'individual_predictions': predictions,
            'individual_probabilities': probabilities,
            'risk_factors': unique_factors,
            'ensemble_info': {
                'models_used': list(self.models.keys()),
                'weighted_average': weighted_prob,
                'voting_consensus': 'unanimous' if fraud_votes in [0, total_models] else 'majority',
                'model_agreement': confidence
            }
        }
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all models in the ensemble"""
        status = {
            'ensemble_ready': self.models_loaded,
            'models_available': len(self.models),
            'expected_models': 4,
            'model_status': {}
        }
        
        for model_type in ['random_forest', 'logistic_regression', 'naive_bayes', 'svm']:
            if model_type in self.models:
                status['model_status'][model_type] = 'loaded'
            else:
                status['model_status'][model_type] = 'missing'
        
        return status


# Export the main class
__all__ = ['EnsemblePredictor']