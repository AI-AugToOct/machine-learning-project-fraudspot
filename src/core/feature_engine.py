"""
Feature Engine - CONTENT-FOCUSED IMPLEMENTATION
This module handles feature engineering for content and company-based fraud detection.

CONTENT-FIRST APPROACH:
- Focuses on job content quality, company legitimacy, and text analysis
- Removed all poster/profile-related features for more reliable detection
- Models learn from job posting content and company verification data
- No dependency on unreliable profile scraping

Version: 4.0.0 - Content-Focused (Profile features removed)
"""

import logging
import re
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .constants import DataConstants, FraudKeywords, ModelConstants

logger = logging.getLogger(__name__)


class FeatureEngine(BaseEstimator, TransformerMixin):
    """
    Content-focused feature engine for fraud detection.
    
    This class consolidates content and company-based feature engineering:
    - Text-based features (suspicious keywords, grammar, sentiment)
    - Company features (legitimacy, size, website verification)
    - Job content features (completeness, quality, structure)
    - Contact risk features (messaging apps, professional channels)
    
    Generates a focused feature set for reliable ML model training.
    """
    
    def __init__(self):
        """Initialize the content-focused feature engine."""
        # Define the new content-focused feature columns (28 features with network quality)
        self.feature_names_ = [
            # Company features (9) - EXPANDED with network quality
            'company_legitimacy_score', 'company_followers_score', 'company_employees_score',
            'has_company_website', 'has_company_size', 'has_company_founded',
            'network_quality_score', 'follower_employee_ratio', 'suspicious_network_flag',
            
            # Content quality features (5)
            'content_quality_score', 'description_length_score', 'title_word_count',
            'professional_language_score', 'urgency_language_score',
            
            # Contact risk features (4)
            'contact_risk_score', 'has_whatsapp', 'has_telegram', 'has_professional_email',
            
            # Text analysis features (3)
            'suspicious_keywords_count', 'arabic_suspicious_score', 'english_suspicious_score',
            
            # Job structure features (4)
            'has_salary_info', 'has_requirements', 'has_location', 'has_experience_level',
            
            # Additional content features (2)
            'company_website_quality', 'email_domain_legitimacy',
            
            # Computed indicators (1)
            'fraud_risk_score'
        ]
        
        self.is_fitted = False
        self._feature_cache = {}
        self._generation_count = 0
        
        logger.info("FeatureEngine initialized - content-focused version with enhanced company analysis")
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Fit the feature engine (mainly for compatibility)."""
        self.is_fitted = True
        logger.info("Content-focused FeatureEngine fitted")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data to content-focused feature set."""
        logger.info("Transforming data with content-focused feature engine")
        
        if isinstance(X, dict):
            # Single job posting
            df = pd.DataFrame([X])
            single_row = True
        else:
            df = X.copy()
            single_row = False
        
        # Generate content-focused feature set
        df_features = self.generate_complete_feature_set(df)
        
        # Return single row if input was single job
        return df_features.iloc[0:1] if single_row else df_features
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def generate_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate content-focused features for fraud detection.
        
        Args:
            data: Dictionary from job scraping (content and company data only)
            
        Returns:
            Dict: Enhanced feature dictionary ready for ML models
        """
        logger.info("🔧 Generating content-focused features")
        
        # Start with input features
        features = data.copy()
        
        # Add company analysis
        features.update(self._analyze_company_features(features))
        
        # Add content quality analysis  
        features.update(self._analyze_content_features(features))
        
        # Add contact risk analysis
        features.update(self._analyze_contact_features(features))
        
        # Add text analysis
        features.update(self._analyze_language_features(features))
        
        # Add missing ML model features
        features.update(self._generate_missing_features(features))
        
        # Return only the 25 defined features, not all input data
        final_features = {}
        for feature_name in self.feature_names_:
            final_features[feature_name] = features.get(feature_name, 0.0)
        
        logger.info(f"✅ Generated {len(final_features)} content-focused features")
        return final_features
    
    def _analyze_company_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze company legitimacy and verification features."""
        analysis = {}
        
        # Company legitimacy score (enhanced)
        company_score = features.get('company_legitimacy_score', 0.0)
        analysis['company_legitimacy_score'] = company_score
        
        # Individual company indicators
        analysis['has_company_website'] = int(bool(features.get('company_website')))
        analysis['has_company_size'] = int(bool(features.get('company_size')))
        analysis['has_company_founded'] = int(bool(features.get('company_founded')))
        
        # Company followers score (normalized)
        followers = features.get('company_followers', 0)
        analysis['company_followers_score'] = min(followers / 10000, 1.0) if followers else 0.0
        
        # Company employees score (normalized)
        employees = features.get('company_employees', 0)
        analysis['company_employees_score'] = min(employees / 1000, 1.0) if employees else 0.0
        
        # Network quality score - NEW FRAUD INDICATOR
        def calculate_network_quality_score(followers, employees):
            """Calculate network quality score (0-1) based on follower/employee ratio."""
            if not followers or not employees or employees == 0:
                return 0.5  # Unknown, neutral
                
            try:
                followers = float(followers) if followers else 0
                employees = float(employees) if employees else 0
                
                if employees == 0:
                    return 0.5
                    
                ratio = followers / employees
            except (ValueError, TypeError):
                return 0.5
            
            # Sweet spot: 10-200 followers per employee (legitimate companies)
            if 10 <= ratio <= 200:
                # Peak score around ratio=50
                optimal_distance = abs(ratio - 50) / 150
                return 0.85 + (0.15 * (1 - optimal_distance))
            # Highly suspicious: extreme ratios  
            elif ratio > 1000:
                # Very suspicious - likely bot followers (lower score = more suspicious)
                return max(0.01, 0.3 - (ratio - 1000) / 20000)
            elif ratio < 1:
                # Very suspicious - no social presence
                return ratio * 0.3
            # Borderline cases
            elif 200 < ratio <= 1000:
                # Linear decline from good to suspicious
                return 0.7 - ((ratio - 200) / 800 * 0.4)
            else:  # 1 <= ratio < 10
                # Linear increase from suspicious to acceptable
                return 0.3 + ((ratio - 1) / 9 * 0.4)
        
        network_quality = calculate_network_quality_score(followers, employees)
        analysis['network_quality_score'] = network_quality
        analysis['follower_employee_ratio'] = followers / employees if employees else 0
        analysis['suspicious_network_flag'] = int(network_quality < 0.3 or (followers and employees and followers/employees > 1000))
        
        logger.debug(f"Company analysis: legitimacy={company_score:.2f}, followers={followers}, employees={employees}, network_quality={network_quality:.3f}")
        return analysis
    
    def _analyze_content_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze job content quality and completeness."""
        analysis = {}
        
        # Content quality score from scraper
        content_score = features.get('content_quality_score', 0.0)
        analysis['content_quality_score'] = content_score
        
        # Description quality
        description = features.get('job_description', '') or features.get('description', '')
        desc_length = len(str(description)) if description else 0
        analysis['description_length_score'] = min(desc_length / 1000, 1.0)
        
        # Title quality
        title = features.get('job_title', '')
        title_words = len(str(title).split()) if title else 0
        analysis['title_word_count'] = title_words
        
        # Professional language assessment
        analysis['professional_language_score'] = self._calculate_professional_score(features)
        
        # Urgency language assessment (lower = more urgent = worse)
        analysis['urgency_language_score'] = self._calculate_urgency_score(features)
        
        # Job structure completeness
        analysis['has_salary_info'] = int(bool(features.get('salary_info')))
        analysis['has_requirements'] = int(bool(features.get('requirements')))
        analysis['has_location'] = int(bool(features.get('location')))
        analysis['has_experience_level'] = int(bool(features.get('experience_level')))
        
        logger.debug(f"Content analysis: quality={content_score:.2f}, desc_len={desc_length}, title_words={title_words}")
        return analysis
    
    def _analyze_contact_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze contact methods for fraud risk indicators."""
        analysis = {}
        
        # Contact risk score from scraper
        contact_risk = features.get('contact_risk_score', 0.0)
        analysis['contact_risk_score'] = contact_risk
        
        # Individual contact method flags
        analysis['has_whatsapp'] = int(bool(features.get('has_whatsapp')))
        analysis['has_telegram'] = int(bool(features.get('has_telegram')))
        analysis['has_professional_email'] = int(bool(features.get('has_email')))
        
        logger.debug(f"Contact analysis: risk={contact_risk:.2f}, whatsapp={analysis['has_whatsapp']}, telegram={analysis['has_telegram']}")
        return analysis
    
    def _analyze_language_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze text for suspicious language patterns."""
        analysis = {}
        
        # Get text content for analysis
        description = str(features.get('job_description', '')) + ' ' + str(features.get('description', ''))
        title = str(features.get('job_title', ''))
        full_text = f"{title} {description}".lower()
        
        # Count suspicious keywords
        suspicious_count = 0
        english_suspicious = 0
        arabic_suspicious = 0
        
        # English suspicious keywords
        for keyword in FraudKeywords.ENGLISH_SUSPICIOUS:
            if keyword.lower() in full_text:
                suspicious_count += 1
                english_suspicious += 1
        
        # Arabic suspicious keywords  
        for keyword in FraudKeywords.ARABIC_SUSPICIOUS:
            if keyword in full_text:
                suspicious_count += 1
                arabic_suspicious += 1
        
        analysis['suspicious_keywords_count'] = suspicious_count
        analysis['english_suspicious_score'] = min(english_suspicious / 10, 1.0)
        analysis['arabic_suspicious_score'] = min(arabic_suspicious / 10, 1.0)
        
        logger.debug(f"Language analysis: suspicious_count={suspicious_count}, english={english_suspicious}, arabic={arabic_suspicious}")
        return analysis
    
    def _calculate_professional_score(self, features: Dict[str, Any]) -> float:
        """Calculate professional language quality score."""
        description = str(features.get('job_description', '')) + ' ' + str(features.get('description', ''))
        title = str(features.get('job_title', ''))
        full_text = f"{title} {description}".lower()
        
        # Professional indicators
        professional_terms = [
            'experience', 'qualification', 'responsibility', 'requirement',
            'skill', 'education', 'degree', 'certificate', 'professional',
            'company', 'team', 'role', 'position', 'opportunity'
        ]
        
        professional_count = sum(1 for term in professional_terms if term in full_text)
        return min(professional_count / len(professional_terms), 1.0)
    
    def _calculate_urgency_score(self, features: Dict[str, Any]) -> float:
        """Calculate urgency language score (lower = more urgent = worse)."""
        description = str(features.get('job_description', '')) + ' ' + str(features.get('description', ''))
        title = str(features.get('job_title', ''))
        full_text = f"{title} {description}".lower()
        
        # Urgency indicators (suspicious)
        urgency_terms = [
            'urgent', 'immediate', 'asap', 'quick', 'fast', 'instant',
            'now hiring', 'start immediately', 'easy money', 'guaranteed'
        ]
        
        urgency_count = sum(1 for term in urgency_terms if term in full_text)
        return max(1.0 - (urgency_count / 5), 0.0)  # Higher urgency = lower score
    
    def _generate_missing_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Generate any missing features needed for ML model compatibility."""
        analysis = {}
        
        # Overall fraud risk score (composite)
        company_risk = 1.0 - features.get('company_legitimacy_score', 0.5)
        content_risk = 1.0 - features.get('content_quality_score', 0.5)
        contact_risk = features.get('contact_risk_score', 0.0)
        language_risk = features.get('english_suspicious_score', 0.0) + features.get('arabic_suspicious_score', 0.0)
        
        # Weighted composite risk
        fraud_risk = (
            company_risk * 0.4 +      # Company legitimacy most important
            content_risk * 0.3 +      # Content quality second
            contact_risk * 0.2 +      # Contact methods third
            language_risk * 0.1       # Language patterns least weight
        )
        
        analysis['fraud_risk_score'] = min(fraud_risk, 1.0)
        
        # Additional content features
        analysis['company_website_quality'] = self._analyze_website_quality(features)
        analysis['email_domain_legitimacy'] = self._analyze_email_legitimacy(features)
        
        return analysis
    
    def _analyze_website_quality(self, features: Dict[str, Any]) -> float:
        """Analyze company website quality indicators."""
        website = features.get('company_website', '')
        if not website:
            return 0.0
        
        quality_score = 0.5  # Base score for having a website
        
        # Check for professional domains
        professional_domains = ['.com', '.org', '.gov', '.edu']
        if any(domain in website.lower() for domain in professional_domains):
            quality_score += 0.3
        
        # Check for HTTPS
        if website.startswith('https://'):
            quality_score += 0.2
        
        return min(quality_score, 1.0)
    
    def _analyze_email_legitimacy(self, features: Dict[str, Any]) -> float:
        """Analyze email domain legitimacy."""
        email_domains = features.get('email_domains', [])
        if not email_domains:
            return 0.0
        
        # Check for professional email domains
        professional_indicators = ['company', 'corp', 'inc', 'ltd', '.com', '.org']
        suspicious_indicators = ['gmail', 'yahoo', 'hotmail', 'outlook']
        
        legitimacy_score = 0.5  # Base score
        
        for domain in email_domains:
            domain_lower = domain.lower()
            if any(indicator in domain_lower for indicator in professional_indicators):
                legitimacy_score += 0.3
            elif any(indicator in domain_lower for indicator in suspicious_indicators):
                legitimacy_score -= 0.2
        
        return max(0.0, min(legitimacy_score, 1.0))
    
    def generate_complete_feature_set(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate complete feature set for ML model training."""
        logger.info(f"Generating complete content-focused feature set for {len(df)} records")
        
        df_features = df.copy()
        
        # Generate features for each row
        for idx in df_features.index:
            row_data = df_features.loc[idx].to_dict()
            enhanced_features = self.generate_features(row_data)
            
            # Update the row with new features
            for feature_name in self.feature_names_:
                df_features.loc[idx, feature_name] = enhanced_features.get(feature_name, 0.0)
        
        # Select only the required feature columns
        final_features = df_features[self.feature_names_].copy()
        
        # Fill any remaining NaN values
        final_features = final_features.fillna(0.0)
        
        logger.info(f"✅ Generated {len(self.feature_names_)} content-focused features")
        return final_features
    
    def get_feature_names(self) -> List[str]:
        """Get the list of feature names."""
        return self.feature_names_.copy()
    
    def validate_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate the generated features."""
        validation = {
            'total_features': len(df.columns),
            'expected_features': len(self.feature_names_),
            'missing_features': [],
            'extra_features': [],
            'nan_features': [],
            'validation_passed': True
        }
        
        # Check for missing expected features
        for feature in self.feature_names_:
            if feature not in df.columns:
                validation['missing_features'].append(feature)
                validation['validation_passed'] = False
        
        # Check for NaN values
        for col in df.columns:
            if df[col].isna().any():
                validation['nan_features'].append(col)
        
        # Log validation results
        if validation['validation_passed']:
            logger.info("✅ Feature validation passed")
        else:
            logger.warning(f"❌ Feature validation failed: missing {validation['missing_features']}")
        
        return validation


# Export main class
__all__ = ['FeatureEngine']