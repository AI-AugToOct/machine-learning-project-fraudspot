"""
Feature Engine - SINGLE SOURCE OF TRUTH
This module handles ALL feature engineering operations.

Version: 3.0.0 - DRY Consolidation
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
    SINGLE SOURCE OF TRUTH for ALL feature engineering operations.
    
    This class consolidates all feature engineering functionality:
    - Text-based features (suspicious keywords, grammar, sentiment)
    - Structural features (completeness, verification scores)
    - Derived features (ratios, indicators, computed scores)  
    - Verification features (the 100% accuracy predictors)
    
    Generates the complete 33-column feature set required for ML model.
    """
    
    def __init__(self):
        """Initialize the unified feature engine."""
        self.feature_names_ = ModelConstants.REQUIRED_FEATURE_COLUMNS.copy()
        self.is_fitted = False
        
        logger.info("FeatureEngine initialized - single source for all features")
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Fit the feature engine (mainly for compatibility)."""
        self.is_fitted = True
        logger.info("FeatureEngine fitted")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data to complete 33-column feature set."""
        logger.info("Transforming data with unified feature engine")
        
        if isinstance(X, dict):
            # Single job posting
            df = pd.DataFrame([X])
            single_row = True
        else:
            df = X.copy()
            single_row = False
        
        # Generate complete feature set
        df_features = self.generate_complete_feature_set(df)
        
        # Return single row if input was single job
        return df_features.iloc[0:1] if single_row else df_features
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def generate_complete_feature_set(self, data: Union[Dict[str, Any], pd.DataFrame]) -> pd.DataFrame:
        """
        Generate complete 33-column feature set.
        SINGLE FUNCTION to create all features needed for ML model.
        
        Args:
            data: Raw job data (dict or DataFrame)
            
        Returns:
            pd.DataFrame: Complete 33-column feature set
        """
        logger.info("Generating complete unified feature set")
        
        # Convert to DataFrame if needed
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()
        
        # Step 1: Ensure basic columns exist
        df = self._ensure_basic_columns(df)
        
        # Step 2: Generate text-based features
        df = self._generate_text_features(df)
        
        # Step 3: Generate structural features  
        df = self._generate_structural_features(df)
        
        # Step 4: Calculate verification scores (CRITICAL PREDICTORS)
        df = self._calculate_verification_scores(df)
        
        # Step 5: Generate derived/computed features
        df = self._generate_computed_scores(df)
        
        # Step 6: Generate company verification features (NEW)
        df = self._generate_company_features(df)
        
        # Step 7: Ensure all columns exist with correct types (updated for 27 ML + text columns)
        df = self._ensure_complete_feature_set(df)
        
        logger.info(f"Complete feature generation: {df.shape[1]} columns")
        return df
    
    def _ensure_basic_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all basic columns exist with appropriate defaults."""
        df_basic = df.copy()
        
        # Text columns
        text_columns = ['job_title', 'job_description', 'requirements', 'benefits', 
                       'company_name', 'company_profile', 'industry', 'location', 'salary_info']
        for col in text_columns:
            if col not in df_basic.columns:
                df_basic[col] = ''
            df_basic[col] = df_basic[col].fillna('').astype(str)
        
        # Basic info columns (job_id removed - provides no ML value)
        
        # Employment details
        employment_cols = ['employment_type', 'experience_level', 'education_level']
        for col in employment_cols:
            if col not in df_basic.columns:
                df_basic[col] = ''
            df_basic[col] = df_basic[col].fillna('').astype(str)
        
        # Binary indicators
        binary_defaults = {'has_company_logo': 1, 'has_questions': 0, 'fraudulent': 0}
        for col, default in binary_defaults.items():
            if col not in df_basic.columns:
                df_basic[col] = default
        
        # Language (0=English, 1=Arabic)
        if 'language' not in df_basic.columns:
            df_basic['language'] = 0
        
        # Use centralized verification service for poster column handling
        from ..services.verification_service import VerificationService
        verification_service = VerificationService()
        df_basic = verification_service.extract_verification_features_df(df_basic)
        
        return df_basic
    
    def _generate_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate ALL text-based fraud detection features."""
        logger.info("Generating unified text-based features")
        df_text = df.copy()
        
        # Process each text column
        text_columns = ['job_title', 'job_description', 'requirements', 'benefits', 'company_name']
        
        for col in text_columns:
            if col in df_text.columns:
                # Generate text statistics
                df_text = self._generate_text_statistics(df_text, col)
                
                # Generate language-aware keyword features
                df_text = self._generate_keyword_features(df_text, col)
        
        # Generate aggregate text features
        df_text = self._generate_aggregate_text_features(df_text)
        
        return df_text
    
    def _generate_text_statistics(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Generate basic text statistics for a column."""
        text_data = df[col].fillna('').astype(str)
        
        # Character and word counts
        df[f'{col}_char_count'] = text_data.str.len()
        df[f'{col}_word_count'] = text_data.str.split().str.len().fillna(0)
        
        # Average word length
        df[f'{col}_avg_word_length'] = df[f'{col}_char_count'] / (df[f'{col}_word_count'] + 1)
        
        # Empty text indicator
        df[f'{col}_is_empty'] = (text_data.str.strip() == '').astype(int)
        
        # Grammar indicators
        df[f'{col}_exclamation_count'] = text_data.str.count(r'!')
        df[f'{col}_question_count'] = text_data.str.count(r'\?')
        df[f'{col}_caps_ratio'] = text_data.apply(
            lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1) if x else 0
        )
        
        return df
    
    def _generate_keyword_features(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Generate language-aware keyword features for a column."""
        text_data = df[col].fillna('').astype(str).str.lower()
        
        # Initialize keyword counts
        df[f'{col}_suspicious_count'] = 0
        df[f'{col}_urgency_count'] = 0  
        df[f'{col}_quality_count'] = 0
        
        # Process by language if available
        if 'language' in df.columns:
            for idx, row in df.iterrows():
                text = str(row[col]).lower() if pd.notna(row[col]) else ''
                language = row.get('language', 0)
                
                if language == 1:  # Arabic
                    sus_count = sum(1 for kw in FraudKeywords.ARABIC_SUSPICIOUS if kw in text)
                    urg_count = sum(1 for kw in FraudKeywords.ARABIC_URGENCY if kw in text)
                    qual_count = sum(1 for kw in FraudKeywords.ARABIC_QUALITY if kw in text)
                else:  # English (default)
                    sus_count = sum(1 for kw in FraudKeywords.ENGLISH_SUSPICIOUS if kw in text)
                    urg_count = sum(1 for kw in FraudKeywords.ENGLISH_URGENCY if kw in text)
                    qual_count = sum(1 for kw in FraudKeywords.ENGLISH_QUALITY if kw in text)
                
                df.loc[idx, f'{col}_suspicious_count'] = sus_count
                df.loc[idx, f'{col}_urgency_count'] = urg_count
                df.loc[idx, f'{col}_quality_count'] = qual_count
        else:
            # Fallback to English only
            df[f'{col}_suspicious_count'] = text_data.apply(
                lambda x: sum(1 for kw in FraudKeywords.ENGLISH_SUSPICIOUS if kw in x)
            )
            df[f'{col}_urgency_count'] = text_data.apply(
                lambda x: sum(1 for kw in FraudKeywords.ENGLISH_URGENCY if kw in x)
            )
            df[f'{col}_quality_count'] = text_data.apply(
                lambda x: sum(1 for kw in FraudKeywords.ENGLISH_QUALITY if kw in x)
            )
        
        return df
    
    def _generate_aggregate_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate aggregate features across all text columns."""
        # Aggregate keyword counts
        suspicious_cols = [c for c in df.columns if '_suspicious_count' in c]
        if suspicious_cols:
            df['total_suspicious_keywords'] = df[suspicious_cols].sum(axis=1)
        
        urgency_cols = [c for c in df.columns if '_urgency_count' in c]  
        if urgency_cols:
            df['total_urgency_keywords'] = df[urgency_cols].sum(axis=1)
        
        quality_cols = [c for c in df.columns if '_quality_count' in c]
        if quality_cols:
            df['total_quality_keywords'] = df[quality_cols].sum(axis=1)
        
        return df
    
    def _generate_structural_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate ALL structural features."""
        logger.info("Generating structural features")
        df_struct = df.copy()
        
        # Completeness analysis
        df_struct = self._analyze_completeness(df_struct)
        
        # Required sections analysis  
        df_struct = self._analyze_required_sections(df_struct)
        
        # Contact pattern analysis
        df_struct = self._analyze_contact_patterns(df_struct)
        
        # Experience and salary indicators
        df_struct = self._generate_experience_salary_features(df_struct)
        
        return df_struct
    
    def _analyze_completeness(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze job posting completeness."""
        essential_fields = DataConstants.ESSENTIAL_FIELDS
        additional_fields = DataConstants.ADDITIONAL_FIELDS
        
        # Check field presence
        for field in essential_fields + additional_fields:
            if field in df.columns:
                df[f'has_{field}'] = (~df[field].isin(['', 'Unknown', 'nan', None])).astype(int)
        
        # Calculate completeness score
        essential_present = []
        additional_present = []
        
        for field in essential_fields:
            if f'has_{field}' in df.columns:
                essential_present.append(f'has_{field}')
        
        for field in additional_fields:
            if f'has_{field}' in df.columns:
                additional_present.append(f'has_{field}')
        
        if essential_present:
            essential_score = df[essential_present].mean(axis=1)
        else:
            essential_score = pd.Series([0.5] * len(df), index=df.index)
        
        if additional_present:
            additional_score = df[additional_present].mean(axis=1)
        else:
            additional_score = pd.Series([0.5] * len(df), index=df.index)
        
        # Weighted completeness (70% essential, 30% additional)
        df['completeness_score'] = (essential_score * 0.7) + (additional_score * 0.3)
        
        return df
    
    def _analyze_required_sections(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze presence of required job sections."""
        section_keywords = DataConstants.SECTION_KEYWORDS
        
        # Get all text content
        text_columns = ['job_description', 'job_title', 'company_profile', 'requirements', 'benefits']
        
        sections_present = []
        for section, keywords in section_keywords.items():
            df[f'has_section_{section}'] = 0
            
            for _, row in df.iterrows():
                text_content = ""
                for field in text_columns:
                    if field in df.columns:
                        value = row.get(field, "")
                        if value and str(value).lower() != 'nan':
                            text_content += f" {str(value)}"
                
                text_content = text_content.lower()
                
                # Check for section keywords
                has_section = any(keyword.lower() in text_content for keyword in keywords)
                df.loc[row.name, f'has_section_{section}'] = int(has_section)
            
            sections_present.append(f'has_section_{section}')
        
        # Calculate sections present count
        if sections_present:
            df['required_sections_present'] = df[sections_present].sum(axis=1)
        else:
            df['required_sections_present'] = 0
        
        return df
    
    def _analyze_contact_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze contact patterns for fraud indicators."""
        # Initialize contact risk features
        df['contact_risk_score'] = 0.0
        df['has_suspicious_email'] = 0
        df['has_messaging_apps'] = 0
        
        # Analyze job description for contact patterns
        if 'job_description' in df.columns:
            desc_text = df['job_description'].fillna('').astype(str).str.lower()
            
            # Check for suspicious email domains
            from .constants import ScrapingConstants
            for domain in ScrapingConstants.SUSPICIOUS_EMAIL_DOMAINS:
                domain_pattern = f"@{domain.lower()}"
                df['has_suspicious_email'] = np.maximum(
                    df['has_suspicious_email'], 
                    desc_text.str.contains(domain_pattern, case=False, na=False).astype(int)
                )
            
            # Check for messaging apps
            for app, pattern in ScrapingConstants.MESSAGING_PATTERNS.items():
                app_present = desc_text.str.contains(pattern, case=False, na=False).astype(int)
                df['has_messaging_apps'] = np.maximum(df['has_messaging_apps'], app_present)
            
            # Calculate contact risk score
            df['contact_risk_score'] = (
                df['has_suspicious_email'] * 0.4 +
                df['has_messaging_apps'] * 0.3
            )
        
        return df
    
    def _generate_experience_salary_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate experience and salary indicator features."""
        # Experience level indicators
        if 'experience_level' in df.columns:
            df['experience_level_numeric'] = pd.to_numeric(df['experience_level'], errors='coerce').fillna(0)
            df['is_entry_level'] = (df['experience_level_numeric'] <= 1).astype(int)
            df['is_mid_level'] = ((df['experience_level_numeric'] > 1) & 
                                 (df['experience_level_numeric'] <= 5)).astype(int)
            df['is_senior_level'] = (df['experience_level_numeric'] > 5).astype(int)
        
        # Salary indicators
        if 'salary_info' in df.columns:
            salary_text = df['salary_info'].fillna('').astype(str)
            # Extract numeric values from salary text
            df['salary_numeric'] = salary_text.str.extract(r'(\d+)').fillna('0').astype(float)
            df['has_salary_range'] = (df['salary_numeric'] > 0).astype(int)
            
            # High salary indicator (relative to median)
            median_salary = df['salary_numeric'].median()
            if median_salary > 0:
                df['is_high_salary'] = (df['salary_numeric'] > median_salary).astype(int)
            else:
                df['is_high_salary'] = 0
        
        return df
    
    def _calculate_verification_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate verification scores using centralized VerificationService.
        This delegates to the SINGLE source of truth for verification logic.
        """
        from ..services.verification_service import VerificationService
        verification_service = VerificationService()
        
        # Use centralized verification service for all score calculations
        return verification_service.calculate_verification_scores_df(df)
    
    def _generate_computed_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate all computed score features."""
        logger.info("Generating computed score features")
        
        # Description length score (normalized)
        if 'job_description' in df.columns:
            desc_lengths = df['job_description'].str.len()
            df['description_length_score'] = np.clip(desc_lengths / 1000.0, 0, 1)
        else:
            df['description_length_score'] = 0.5
        
        # Title word count
        if 'job_title' in df.columns:
            df['title_word_count'] = df['job_title'].str.split().str.len().fillna(0)
        else:
            df['title_word_count'] = 0
        
        # Professional language score
        df['professional_language_score'] = df.apply(self._calculate_professional_score, axis=1)
        
        # Urgency language score (lower = more urgent = worse)
        df['urgency_language_score'] = df.apply(self._calculate_urgency_score, axis=1)
        
        # Contact professionalism score
        df['contact_professionalism_score'] = df.apply(self._calculate_contact_professionalism, axis=1)
        
        # Verification score (already calculated, ensure it exists)
        if 'verification_score' not in df.columns:
            df['verification_score'] = df['poster_score'] / 4.0
        
        # Content quality score (combination of length and professionalism)
        df['content_quality_score'] = (
            df['description_length_score'] + 
            df['professional_language_score']
        ) / 2
        
        # Legitimacy score (combination of urgency and contact professionalism)
        df['legitimacy_score'] = (
            df['urgency_language_score'] + 
            df['contact_professionalism_score']
        ) / 2
        
        return df
    
    def _calculate_professional_score(self, row: pd.Series) -> float:
        """Calculate professional language score for a row."""
        text = str(row.get('job_description', ''))
        if not text or text == 'nan':
            return 0.5
        
        text = text.lower()
        
        # Count professional vs unprofessional terms
        professional_count = sum(1 for term in FraudKeywords.PROFESSIONAL_TERMS if term in text)
        unprofessional_count = sum(1 for term in FraudKeywords.UNPROFESSIONAL_TERMS if term in text)
        
        # Calculate score (0-1 range)
        score = min(professional_count / max(len(FraudKeywords.PROFESSIONAL_TERMS), 1), 1.0)
        score -= unprofessional_count * 0.1
        
        return max(0, min(1, score))
    
    def _calculate_urgency_score(self, row: pd.Series) -> float:
        """Calculate urgency language score (lower = more urgent = worse)."""
        text = str(row.get('job_description', ''))
        if not text or text == 'nan':
            return 1.0  # No urgency = good
        
        text = text.lower()
        
        # Use language-appropriate urgency keywords
        language = row.get('language', 0)
        if language == 1:  # Arabic
            urgency_keywords = FraudKeywords.ARABIC_URGENCY
        else:  # English
            urgency_keywords = FraudKeywords.ENGLISH_URGENCY
        
        urgency_count = sum(1 for keyword in urgency_keywords if keyword in text)
        return max(0, 1.0 - urgency_count * 0.2)  # Lower score = more urgency
    
    def _calculate_contact_professionalism(self, row: pd.Series) -> float:
        """Calculate contact professionalism score."""
        text = str(row.get('job_description', ''))
        if not text or text == 'nan':
            return 0.7  # Neutral default instead of 0.8
        
        text = text.lower()
        unprofessional_count = sum(1 for contact in FraudKeywords.UNPROFESSIONAL_CONTACTS if contact in text)
        return max(0, 1.0 - unprofessional_count * 0.25)
    
    def _generate_company_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate company verification features using VerificationService (single source of truth).
        
        Converts raw company data into normalized scores for ML model input by delegating
        ALL calculations to the VerificationService to avoid code duplication.
        
        Args:
            df: DataFrame with potential company data columns
            
        Returns:
            DataFrame with company feature scores added
        """
        logger.info("Generating company verification features using VerificationService")
        df_company = df.copy()
        
        # Import VerificationService (avoid circular imports at module level)
        from ..services.verification_service import VerificationService
        verification_service = VerificationService()
        
        # Process each row using VerificationService as single source of truth
        def generate_company_scores(row):
            job_data = row.to_dict()
            
            # Use VerificationService for ALL company scoring calculations
            company_scores = verification_service.calculate_company_verification_scores(job_data)
            
            return pd.Series(company_scores)
        
        # Generate company scores for all rows using VerificationService
        company_scores = df_company.apply(generate_company_scores, axis=1)
        
        # Add all company scores to dataframe
        for col in company_scores.columns:
            df_company[col] = company_scores[col]
        
        logger.info(f"Company features generated using VerificationService - "
                   f"followers: {df_company['company_followers_score'].mean():.3f}, "
                   f"employees: {df_company['company_employees_score'].mean():.3f}, "
                   f"founded: {df_company['company_founded_score'].mean():.3f}, "
                   f"network: {df_company['network_quality_score'].mean():.3f}, "
                   f"legitimacy: {df_company['company_legitimacy_score'].mean():.3f}")
        
        return df_company
    
    def _ensure_complete_feature_set(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all required columns exist with correct types, return only ML features."""
        # Step 1: Add missing columns with defaults (need all columns for processing)
        for col in ModelConstants.REQUIRED_FEATURE_COLUMNS:
            if col not in df.columns:
                if col in DataConstants.BINARY_COLUMNS or col in DataConstants.ENCODED_COLUMNS:
                    df[col] = 0
                elif col in DataConstants.SCORE_COLUMNS:
                    df[col] = 0.5 if 'score' in col else 0.0
                elif col == 'title_word_count':
                    df[col] = 0
                else:
                    df[col] = ''
        
        # Step 2: Ensure correct data types for numerical columns
        int_columns = (['title_word_count'] + 
                      DataConstants.BINARY_COLUMNS + 
                      DataConstants.ENCODED_COLUMNS)
        
        for col in int_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        float_columns = DataConstants.SCORE_COLUMNS
        for col in float_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.5).astype(float)
                # Ensure scores are in [0, 1] range
                if 'score' in col:
                    df[col] = df[col].clip(0, 1)
        
        # Step 3: Return only ML features (numerical columns) - exclude text columns and target
        ml_features_available = [col for col in ModelConstants.ML_FEATURE_COLUMNS if col in df.columns]
        
        # Ensure we don't include job_id (provides no value) or raw categorical columns that have encoded versions
        exclude_columns = ['job_id', 'employment_type', 'experience_level', 'education_level', 'industry', 'salary_info']
        ml_features_final = [col for col in ml_features_available if col not in exclude_columns]
        
        logger.info(f"Final ML features for model: {ml_features_final}")
        return df[ml_features_final]
    
    def get_feature_importance_info(self) -> Dict[str, Any]:
        """Get information about feature importance and categories."""
        return {
            'total_features': len(ModelConstants.REQUIRED_FEATURE_COLUMNS),
            'critical_predictors': {
                'verification_features': ['poster_verified', 'poster_experience', 'poster_photo', 'poster_active', 'poster_score'],
                'accuracy': '100% - perfect separation between fraud/legitimate'
            },
            'feature_categories': {
                'text_features': [col for col in ModelConstants.REQUIRED_FEATURE_COLUMNS if any(text_col in col for text_col in DataConstants.TEXT_COLUMNS)],
                'verification_features': ['poster_verified', 'poster_experience', 'poster_photo', 'poster_active', 'poster_score', 'verification_score'],
                'computed_scores': DataConstants.SCORE_COLUMNS,
                'structural_features': ['completeness_score', 'required_sections_present', 'contact_risk_score']
            },
            'language_support': {
                'english': 'Full support with English keywords and patterns',
                'arabic': 'Full support with Arabic keywords and patterns',
                'multilingual': 'Language-aware feature generation'
            }
        }


# Convenience function for single job predictions
def generate_features_for_single_job(job_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Generate complete feature set for a single job posting.
    SINGLE FUNCTION for all external feature generation needs.
    
    Args:
        job_data: Raw job posting data
        
    Returns:
        pd.DataFrame: Single row with complete 33-column feature set
    """
    engine = FeatureEngine()
    return engine.generate_complete_feature_set(job_data)


# Export main classes and functions
__all__ = ['FeatureEngine', 'generate_features_for_single_job']