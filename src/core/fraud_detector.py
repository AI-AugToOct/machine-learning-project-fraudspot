"""
Fraud Detector - SINGLE SOURCE OF TRUTH
This module handles ALL fraud detection and prediction logic.

Version: 3.0.0 - DRY Consolidation
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .constants import FraudKeywords, ModelConstants, UIConstants
from .feature_engine import FeatureEngine

logger = logging.getLogger(__name__)


class FraudDetector:
    """
    SINGLE SOURCE OF TRUTH for ALL fraud detection operations.
    
    This class consolidates all fraud prediction functionality:
    - ML model-based predictions
    - Rule-based scoring systems  
    - Risk level determination
    - Risk factor extraction
    - Confidence scoring
    
    Provides unified interface for all fraud detection needs.
    """
    
    def __init__(self, model_pipeline=None):
        """
        Initialize the unified fraud detector.
        
        Args:
            model_pipeline: Trained ML pipeline (optional)
        """
        self.model_pipeline = model_pipeline
        self.feature_engine = FeatureEngine()
        
        logger.info("FraudDetector initialized - single source for all fraud detection")
    
    def predict_fraud(self, job_data: Dict[str, Any], use_ml: bool = True) -> Dict[str, Any]:
        """
        SINGLE FUNCTION for all fraud prediction needs.
        
        Args:
            job_data: Raw job posting data
            use_ml: Whether to use ML model if available
            
        Returns:
            Dict: Unified prediction result with all analysis details
        """
        logger.info("Starting unified fraud prediction")
        
        try:
            # Step 1: Generate complete feature set
            features_df = self.feature_engine.generate_complete_feature_set(job_data)
            
            # Step 2: Try ML prediction first if requested and available
            if use_ml and self.model_pipeline is not None:
                ml_result = self._predict_with_ml_model(features_df)
                if not ml_result.get('model_failed', False):
                    # Enhance ML result with additional analysis
                    return self._enhance_prediction_result(job_data, features_df, ml_result)
            
            # Step 2.5: If ML was requested but not available, return error (no silent fallback)
            if use_ml and self.model_pipeline is None:
                logger.error("ML prediction requested but no model pipeline available")
                return {
                    'success': False,
                    'error': 'ML models not loaded',
                    'model_failed': True,
                    'model_used': False,
                    'prediction_method': 'Error - No Models'
                }
            
            # Step 3: Fall back to rule-based prediction (only when explicitly requested)
            rule_result = self._predict_with_rule_based_system(job_data, features_df)
            return self._enhance_prediction_result(job_data, features_df, rule_result)
            
        except Exception as e:
            logger.error(f"Error in fraud prediction: {str(e)}")
            return self._create_error_result(str(e))
    
    def _predict_with_ml_model(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Make prediction using trained ML model."""
        if self.model_pipeline is None:
            return {'model_failed': True, 'error': 'No model pipeline available'}
        
        try:
            logger.info("Making ML-based fraud prediction")
            
            # Make prediction
            prediction = self.model_pipeline.predict(features_df.iloc[0:1])[0]
            
            # Get probability if available, otherwise use decision function or fallback
            if hasattr(self.model_pipeline, 'predict_proba'):
                prediction_proba = self.model_pipeline.predict_proba(features_df.iloc[0:1])[0]
                fraud_probability = float(prediction_proba[1])  # Probability of class 1 (fraud)
            elif hasattr(self.model_pipeline, 'decision_function'):
                # For SVM/SGD - use decision function and convert to probability-like score
                decision_score = self.model_pipeline.decision_function(features_df.iloc[0:1])[0]
                # Convert to 0-1 range using sigmoid approximation
                from math import exp
                fraud_probability = 1 / (1 + exp(-decision_score))
            else:
                # Fallback to binary prediction
                fraud_probability = float(prediction)
            
            # Determine risk level based on ML probability
            risk_level = self._get_risk_level_from_probability(fraud_probability)
            
            # Calculate confidence based on available probability information
            if hasattr(self.model_pipeline, 'predict_proba'):
                confidence = float(abs(prediction_proba[1] - prediction_proba[0]))
                proba_list = prediction_proba.tolist()
            else:
                # For models without predict_proba, use distance from 0.5 threshold as confidence
                confidence = abs(fraud_probability - 0.5) * 2  # Scale to 0-1 range
                proba_list = [1 - fraud_probability, fraud_probability]
            
            return {
                'model_failed': False,
                'model_used': True,
                'prediction_method': 'ML Model',
                'is_fraud': self._apply_threshold_decision(fraud_probability, bool(prediction)),
                'fraud_probability': fraud_probability,
                'fraud_score': fraud_probability,
                'confidence': confidence,
                'risk_level': risk_level,
                'prediction_proba': proba_list
            }
            
        except Exception as e:
            logger.error(f"ML model prediction failed: {str(e)}")
            return {'model_failed': True, 'error': str(e)}
    
    def _predict_with_rule_based_system(self, job_data: Dict[str, Any], features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Make prediction using rule-based scoring system.
        This is the SINGLE implementation of rule-based fraud detection.
        """
        logger.info("Making rule-based fraud prediction")
        
        try:
            # Convert features to dict for easier access
            features_dict = features_df.iloc[0].to_dict() if len(features_df) > 0 else {}
            
            # Calculate fraud score using balanced approach
            fraud_score, risk_factors = self._calculate_fraud_score_with_verification(job_data, features_dict)
            
            # Determine risk level and fraud classification based on final score
            risk_level, is_fraud, confidence = self._classify_risk_from_probability(fraud_score)
            
            return {
                'model_failed': False,
                'model_used': False,
                'prediction_method': 'Rule-based System',
                'is_fraud': is_fraud,
                'fraud_probability': fraud_score,
                'fraud_score': fraud_score,
                'confidence': confidence,
                'risk_level': risk_level,
                'risk_factors': risk_factors,
                'poster_score': features_dict.get('poster_score', 0)
            }
            
        except Exception as e:
            logger.error(f"Rule-based prediction failed: {str(e)}")
            return {'model_failed': True, 'error': str(e)}
    
    def _calculate_fraud_score_with_verification(self, job_data: Dict[str, Any], features_dict: Dict[str, Any]) -> Tuple[float, List[str]]:
        """
        Calculate fraud score using verification features as primary predictors.
        This is the SINGLE implementation of fraud scoring logic.
        """
        fraud_score = 0.0
        risk_factors = []
        
        # BALANCED: Verification Features Analysis (important but not overwhelming)
        poster_score = features_dict.get('poster_score', 0)
        
        if poster_score >= 3:
            # Highly verified = Very likely legitimate
            fraud_score = 0.05  # Very low fraud risk
            risk_factors.append(f"✅ EXCELLENT: High verification score ({poster_score}/4) - Very likely legitimate")
        elif poster_score == 2:
            # Moderately verified = Likely legitimate  
            fraud_score = 0.15  # Low fraud risk
            risk_factors.append(f"✅ GOOD: Moderate verification score ({poster_score}/4) - Likely legitimate")
        elif poster_score == 1:
            # Low verification = Moderate risk (reduced from 75%)
            fraud_score = 0.25  # Moderate fraud risk
            risk_factors.append(f"⚠️ CAUTION: Low verification score ({poster_score}/4) - Consider other factors")
        else:  # poster_score == 0
            # No verification = Suspicious but check content (further reduced)
            fraud_score = 0.15  # Further reduced base suspicion for private profiles (many legitimate recruiters use private profiles)
            risk_factors.append(f"ℹ️ Private poster profile - Analyzing job content quality")
        
        # Add language-aware adjustments
        language = features_dict.get('language', 0)
        lang_name = 'Arabic' if language == 1 else 'English'
        
        # Suspicious keywords (language-specific) - INCREASED WEIGHT
        if 'total_suspicious_keywords' in features_dict:
            suspicious_count = features_dict['total_suspicious_keywords']
            if suspicious_count > 0:
                fraud_score += min(suspicious_count * 0.12, 0.4)  # Increased from 0.05/0.2 to 0.12/0.4
                risk_factors.append(f"🚨 Suspicious {lang_name} keywords detected: {suspicious_count}")
        
        # Urgency indicators (language-specific) - INCREASED WEIGHT
        if 'total_urgency_keywords' in features_dict:
            urgency_count = features_dict['total_urgency_keywords']
            if urgency_count > 2:
                fraud_score += min(urgency_count * 0.08, 0.3)  # Increased from 0.03/0.15 to 0.08/0.3
                risk_factors.append(f"⚡ Excessive urgency pressure in {lang_name}: {urgency_count} indicators")
        
        # Professional language score
        professional_score = features_dict.get('professional_language_score', 0.7)
        if professional_score < 0.5:
            fraud_score += 0.1
            risk_factors.append(f"Poor {lang_name} professional language quality: {professional_score:.1%}")
        
        # Contact risk
        contact_risk = features_dict.get('contact_risk_score', 0)
        if contact_risk > 0.5:
            fraud_score += contact_risk * 0.15
            risk_factors.append("Suspicious contact methods detected")
        
        # Completeness check
        completeness = features_dict.get('completeness_score', 1)
        if completeness < 0.5:
            fraud_score += 0.05
            risk_factors.append("Job posting lacks essential information")
        
        # COMPENSATING FACTORS: Quality content can reduce fraud score for unverified posters
        if poster_score <= 1:  # Only apply for low/unverified posters
            compensating_factors = []
            reduction_factor = 1.0
            
            # High quality professional language
            if professional_score > 0.8:
                reduction_factor *= 0.8  # 20% reduction
                compensating_factors.append("high-quality professional language")
            
            # Complete job posting
            if completeness > 0.8:
                reduction_factor *= 0.85  # 15% reduction  
                compensating_factors.append("comprehensive job details")
            
            # Company indicators
            has_logo = features_dict.get('has_company_logo', 0)
            if has_logo:
                reduction_factor *= 0.9  # 10% reduction
                compensating_factors.append("company logo present")
            
            # Good legitimacy score from company analysis (STRONGER PROTECTION)
            legitimacy_score = features_dict.get('legitimacy_score', 0.5)
            if legitimacy_score > 0.8:
                reduction_factor *= 0.3  # 70% reduction for very legitimate companies
                compensating_factors.append("very high company legitimacy score")
            elif legitimacy_score > 0.7:
                reduction_factor *= 0.4  # 60% reduction for legitimate companies (stronger protection)
                compensating_factors.append("high company legitimacy score")
            elif legitimacy_score < 0.3:
                reduction_factor *= 1.2  # 20% increase for suspicious company
                risk_factors.append("⚠️ Low company legitimacy detected")
            
            # Apply compensating factors
            if compensating_factors:
                original_score = fraud_score
                fraud_score *= reduction_factor
                risk_factors.append(f"✅ COMPENSATING: Reduced risk due to {', '.join(compensating_factors)} (was {original_score:.1%}, now {fraud_score:.1%})")
        
        # Normalize to [0, 1] range
        fraud_score = min(max(fraud_score, 0.0), 1.0)
        
        return fraud_score, risk_factors
    
    def _classify_risk_from_verification(self, poster_score: int) -> Tuple[str, bool, float]:
        """
        DEPRECATED: Legacy method for verification-only classification.
        Use _classify_risk_from_probability for balanced approach.
        """
        if poster_score >= 3:
            return "VERY LOW", False, 0.95
        elif poster_score == 2:
            return "LOW", False, 0.85
        elif poster_score == 1:
            return "HIGH", True, 0.80
        else:  # poster_score == 0
            return "VERY HIGH", True, 0.95
    
    def _classify_risk_from_probability(self, fraud_probability: float) -> Tuple[str, bool, float]:
        """
        Classify risk level based on balanced fraud probability.
        This provides more nuanced risk assessment than poster-only scoring.
        """
        if fraud_probability < 0.25:
            return "VERY LOW", False, 0.90
        elif fraud_probability < 0.40:
            return "LOW", False, 0.80
        elif fraud_probability < 0.65:
            return "MODERATE", False, 0.70  # Still not flagged as fraud
        elif fraud_probability < 0.75:
            return "HIGH", True, 0.75
        else:  # fraud_probability >= 0.70
            return "VERY HIGH", True, 0.85
    
    def _get_risk_level_from_probability(self, probability: float) -> str:
        """Convert fraud probability to risk level."""
        thresholds = ModelConstants.RISK_THRESHOLDS
        
        if probability <= thresholds['very_low']:
            return "VERY LOW"
        elif probability <= thresholds['low']:
            return "LOW"
        elif probability <= thresholds['high']:
            return "MODERATE"
        elif probability <= thresholds['very_high']:
            return "HIGH"
        else:
            return "VERY HIGH"
    
    def get_verification_display_info(self, poster_score: int) -> Dict[str, Any]:
        """
        SINGLE SOURCE for verification display information.
        Returns standardized display info for UI components.
        
        Args:
            poster_score (int): Verification score (0-4)
            
        Returns:
            Dict containing display info: color, emoji, status, message
        """
        if poster_score >= 3:
            return {
                'color': '#4CAF50',  # Green
                'emoji': '✅',
                'status': 'HIGHLY VERIFIED',
                'message': f'Verification Score: {poster_score}/4',
                'risk_assessment': 'VERY LOW RISK - Highly verified poster',
                'risk_icon': '✅',
                'statistical_message': '96.7% of real job postings have 2+ verification features'
            }
        elif poster_score == 2:
            return {
                'color': '#8BC34A',  # Light Green
                'emoji': '☑️',
                'status': 'MODERATELY VERIFIED',
                'message': f'Verification Score: {poster_score}/4',
                'risk_assessment': 'LOW RISK - Moderately verified poster',
                'risk_icon': '☑️',
                'statistical_message': '96.7% of real job postings have 2+ verification features'
            }
        elif poster_score == 1:
            return {
                'color': '#FF9800',  # Orange
                'emoji': '⚠️',
                'status': 'LOW VERIFICATION',
                'message': f'Verification Score: {poster_score}/4',
                'risk_assessment': 'HIGH RISK - Poorly verified poster',
                'risk_icon': '⚠️',
                'statistical_message': '94.5% of fraudulent postings have 0-1 verification features'
            }
        else:  # poster_score == 0
            return {
                'color': '#F44336',  # Red
                'emoji': '❌',
                'status': 'UNVERIFIED',
                'message': f'Verification Score: {poster_score}/4',
                'risk_assessment': 'VERY HIGH RISK - Unverified poster',
                'risk_icon': '❌',
                'statistical_message': '94.5% of fraudulent postings have 0-1 verification features'
            }

    def _enhance_prediction_result(self, job_data: Dict[str, Any], features_df: pd.DataFrame, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance prediction result with additional analysis."""
        if prediction.get('model_failed', False):
            return prediction
        
        features_dict = features_df.iloc[0].to_dict() if len(features_df) > 0 else {}
        
        # Add detailed risk analysis
        prediction['risk_analysis'] = self._extract_detailed_risk_analysis(job_data, features_dict)
        
        # Add feature importance information
        prediction['feature_analysis'] = self._analyze_key_features(features_dict)
        
        # Add verification breakdown
        prediction['verification_breakdown'] = self._get_verification_breakdown(features_dict)
        
        # Add language-aware analysis
        prediction['language_analysis'] = self._get_language_analysis(features_dict)
        
        # Ensure UI compatibility
        prediction['color'] = UIConstants.RISK_COLORS.get(prediction['risk_level'], 'gray')
        
        return prediction
    
    def _extract_detailed_risk_analysis(self, job_data: Dict[str, Any], features_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Extract detailed risk factors and positive indicators."""
        risk_analysis = {
            'red_flags': [],
            'positive_indicators': [],
            'neutral_factors': []
        }
        
        # Red flags (risk factors)
        poster_score = features_dict.get('poster_score', 0)
        if poster_score <= 1:
            risk_analysis['red_flags'].append(f"CRITICAL: Low verification score ({poster_score}/4)")
        
        # Individual verification checks
        if not features_dict.get('poster_verified', 0):
            risk_analysis['red_flags'].append("Job poster is not verified")
        if not features_dict.get('poster_photo', 0):
            risk_analysis['red_flags'].append("Job poster has no profile photo")
        if not features_dict.get('poster_active', 0):
            risk_analysis['red_flags'].append("Job poster shows no recent activity")
        if not features_dict.get('poster_experience', 0):
            risk_analysis['red_flags'].append("Job poster lacks relevant experience")
        
        # Language-aware keyword analysis
        language = features_dict.get('language', 0)
        lang_name = 'Arabic' if language == 1 else 'English'
        
        if features_dict.get('total_suspicious_keywords', 0) > 0:
            count = features_dict['total_suspicious_keywords']
            risk_analysis['red_flags'].append(f"Contains {count} suspicious {lang_name} keywords")
        
        if features_dict.get('total_urgency_keywords', 0) > 2:
            count = features_dict['total_urgency_keywords']
            risk_analysis['red_flags'].append(f"Multiple urgency indicators in {lang_name} ({count} found)")
        
        # Positive indicators
        if poster_score >= 3:
            risk_analysis['positive_indicators'].append(f"EXCELLENT: High verification score ({poster_score}/4)")
        elif poster_score == 2:
            risk_analysis['positive_indicators'].append(f"GOOD: Moderate verification score ({poster_score}/4)")
        
        # Individual positive checks
        if features_dict.get('poster_verified', 0):
            risk_analysis['positive_indicators'].append("Job poster is verified")
        if features_dict.get('poster_photo', 0):
            risk_analysis['positive_indicators'].append("Job poster has profile photo")
        if features_dict.get('poster_active', 0):
            risk_analysis['positive_indicators'].append("Job poster shows recent activity")
        if features_dict.get('poster_experience', 0):
            risk_analysis['positive_indicators'].append("Job poster has relevant experience")
        
        # Professional language quality
        professional_score = features_dict.get('professional_language_score', 0.7)
        if professional_score > 0.8:
            risk_analysis['positive_indicators'].append(f"Professional {lang_name} language quality ({professional_score:.1%})")
        
        # Low suspicious content
        if features_dict.get('total_suspicious_keywords', 0) == 0:
            risk_analysis['positive_indicators'].append(f"No suspicious {lang_name} keywords detected")
        
        return risk_analysis
    
    def _analyze_key_features(self, features_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the most important features for this prediction."""
        key_features = {}
        
        # Verification features (most important)
        key_features['verification'] = {
            'poster_score': features_dict.get('poster_score', 0),
            'poster_verified': features_dict.get('poster_verified', 0),
            'poster_photo': features_dict.get('poster_photo', 0),
            'poster_experience': features_dict.get('poster_experience', 0),
            'poster_active': features_dict.get('poster_active', 0),
            'importance': 'CRITICAL - 100% accuracy predictor'
        }
        
        # Text quality features
        key_features['text_quality'] = {
            'professional_language_score': features_dict.get('professional_language_score', 0),
            'urgency_language_score': features_dict.get('urgency_language_score', 0),
            'contact_professionalism_score': features_dict.get('contact_professionalism_score', 0),
            'importance': 'HIGH - Language quality indicators'
        }
        
        # Content analysis
        key_features['content_analysis'] = {
            'suspicious_keywords': features_dict.get('total_suspicious_keywords', 0),
            'urgency_keywords': features_dict.get('total_urgency_keywords', 0),
            'quality_keywords': features_dict.get('total_quality_keywords', 0),
            'importance': 'MEDIUM - Content pattern analysis'
        }
        
        return key_features
    
    def _get_verification_breakdown(self, features_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed breakdown of verification features."""
        return {
            'total_score': f"{features_dict.get('poster_score', 0)}/4",
            'breakdown': {
                'verified': bool(features_dict.get('poster_verified', 0)),
                'has_photo': bool(features_dict.get('poster_photo', 0)),
                'has_experience': bool(features_dict.get('poster_experience', 0)),
                'is_active': bool(features_dict.get('poster_active', 0))
            },
            'interpretation': self._interpret_verification_score(features_dict.get('poster_score', 0))
        }
    
    def _interpret_verification_score(self, score: int) -> str:
        """Interpret verification score."""
        interpretations = {
            4: "Fully verified - Excellent legitimacy indicators",
            3: "Highly verified - Strong legitimacy indicators", 
            2: "Moderately verified - Some legitimacy indicators",
            1: "Low verification - Limited legitimacy indicators",
            0: "No verification - High fraud risk"
        }
        return interpretations.get(score, "Unknown verification level")
    
    def _get_language_analysis(self, features_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Get language-specific analysis."""
        language = features_dict.get('language', 0)
        lang_name = 'Arabic' if language == 1 else 'English'
        
        return {
            'detected_language': lang_name,
            'language_code': language,
            'keywords_analyzed': f"{lang_name} keyword patterns used",
            'professional_score': features_dict.get('professional_language_score', 0),
            'urgency_indicators': features_dict.get('total_urgency_keywords', 0),
            'suspicious_patterns': features_dict.get('total_suspicious_keywords', 0)
        }
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create error result with safe defaults."""
        return {
            'model_failed': True,
            'model_used': False,
            'error': error_message,
            'prediction_method': 'Error Fallback',
            'is_fraud': True,  # Conservative default
            'fraud_probability': 0.5,
            'fraud_score': 0.5,
            'confidence': 0.1,
            'risk_level': 'UNKNOWN',
            'color': 'gray',
            'risk_factors': [f"Analysis error: {error_message}"]
        }
    
    def _apply_threshold_decision(self, fraud_probability: float, default_prediction: bool) -> bool:
        """Apply user-defined threshold to fraud probability."""
        try:
            import streamlit as st

            # Get threshold from session state (set by sidebar slider)
            threshold = st.session_state.get('fraud_detection_threshold', 0.5)
            return fraud_probability > threshold
        except:
            # Fallback to model's default prediction if session state unavailable
            return default_prediction
    
    def set_model_pipeline(self, pipeline):
        """Set or update the ML model pipeline."""
        self.model_pipeline = pipeline
        logger.info("ML model pipeline updated in FraudDetector")
    
    def get_prediction_summary(self, result: Dict[str, Any]) -> str:
        """Get a human-readable summary of the prediction."""
        if result.get('model_failed', False):
            return f"Analysis failed: {result.get('error', 'Unknown error')}"
        
        risk_level = result['risk_level']
        confidence = result['confidence']
        method = result['prediction_method']
        
        return f"{risk_level} risk (confidence: {confidence:.1%}) using {method}"


# Convenience function for simple fraud detection
def detect_fraud(job_data: Dict[str, Any], model_pipeline=None) -> Dict[str, Any]:
    """
    SINGLE FUNCTION for simple fraud detection.
    
    Args:
        job_data: Raw job posting data
        model_pipeline: Optional trained ML pipeline
        
    Returns:
        Dict: Complete fraud analysis result
    """
    detector = FraudDetector(model_pipeline)
    return detector.predict_fraud(job_data)


# Export main classes and functions
__all__ = ['FraudDetector', 'detect_fraud']