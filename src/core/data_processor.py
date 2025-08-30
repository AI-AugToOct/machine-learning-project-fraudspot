"""
Data Processor - SINGLE SOURCE OF TRUTH
This module handles ALL data preprocessing operations.
Consolidates logic from transformers.py, data_harmonizer.py, and serializers.py

Version: 3.0.0 - DRY Consolidation
"""

import logging
import re
from html import unescape
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .constants import DataConstants, FraudKeywords, ModelConstants

logger = logging.getLogger(__name__)


class DataProcessor(BaseEstimator, TransformerMixin):
    """
    SINGLE SOURCE OF TRUTH for ALL data preprocessing operations.
    
    This class consolidates all data preprocessing functionality that was
    previously scattered across multiple modules:
    - Data cleaning and validation
    - Feature encoding and normalization  
    - Class balancing and scaling
    - Format conversion and harmonization
    """
    
    def __init__(self, 
                 balance_method: str = 'smote',
                 scaling_method: str = 'standard',
                 steps: List[str] = None):
        """
        Initialize the unified data processor.
        
        Args:
            balance_method: Method for class balancing ('smote', 'random', 'none')
            scaling_method: Method for feature scaling ('standard', 'minmax', 'none')  
            steps: Processing steps to apply
        """
        self.balance_method = balance_method
        self.scaling_method = scaling_method
        self.steps = steps or ['clean', 'encode', 'normalize']
        
        # Initialize transformers
        self.label_encoders = {}
        self.scaler = None
        self.is_fitted = False
        
        logger.info(f"DataProcessor initialized with balance={balance_method}, scaling={scaling_method}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Fit the data processor to the data.
        
        Args:
            X: Feature matrix
            y: Target vector (optional)
        """
        try:
            logger.info("Fitting DataProcessor...")
            
            # Apply processing steps
            X_processed = X.copy()
            
            if 'clean' in self.steps:
                X_processed = self._clean_data(X_processed)
            
            if 'encode' in self.steps:
                X_processed = self._encode_features(X_processed)
            
            if 'normalize' in self.steps and self.scaling_method != 'none':
                self._fit_scaler(X_processed)
            
            self.is_fitted = True
            logger.info("DataProcessor fitting completed")
            return self
            
        except Exception as e:
            logger.error(f"Error fitting DataProcessor: {str(e)}")
            raise
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using fitted processor.
        
        Args:
            X: Feature matrix to transform
            
        Returns:
            Transformed feature matrix
        """
        try:
            if not self.is_fitted:
                raise ValueError("DataProcessor must be fitted before transform")
                
            X_processed = X.copy()
            
            if 'clean' in self.steps:
                X_processed = self._clean_data(X_processed)
            
            if 'encode' in self.steps:
                X_processed = self._encode_features(X_processed)
            
            if 'normalize' in self.steps and self.scaling_method != 'none':
                X_processed = self._scale_features(X_processed)
            
            logger.info(f"Data transformation completed: {X.shape} -> {X_processed.shape}")
            return X_processed
            
        except Exception as e:
            logger.error(f"Error transforming data: {str(e)}")
            raise
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess raw data."""
        try:
            df_cleaned = df.copy()
            
            # Clean text fields
            text_columns = ['job_title', 'job_description', 'requirements', 'benefits', 'company_profile']
            for col in text_columns:
                if col in df_cleaned.columns:
                    df_cleaned[col] = df_cleaned[col].apply(self._clean_text_field)
            
            # Handle missing values
            df_cleaned = self._handle_missing_values(df_cleaned)
            
            # Validate data types
            df_cleaned = self._validate_data_types(df_cleaned)
            
            return df_cleaned
            
        except Exception as e:
            logger.error(f"Error in data cleaning: {str(e)}")
            return df
    
    def _clean_text_field(self, text: Any) -> str:
        """Clean individual text field."""
        if pd.isna(text) or text is None:
            return ""
        
        try:
            # Convert to string
            text_str = str(text)
            
            # Extract from HTML if present
            if '<' in text_str and '>' in text_str:
                text_str = self._extract_text_from_html(text_str)
            
            # Basic text cleaning
            text_str = text_str.strip()
            text_str = re.sub(r'\s+', ' ', text_str)  # Normalize whitespace
            text_str = re.sub(r'[^\w\s\.,!?;:()\-]', '', text_str)  # Remove special chars
            
            return text_str
            
        except Exception as e:
            logger.warning(f"Error cleaning text field: {str(e)}")
            return str(text) if text else ""
    
    def _extract_text_from_html(self, html_content: str) -> str:
        """Extract clean text from HTML content."""
        try:
            if not html_content or '<' not in html_content:
                return html_content
            
            # Remove script and style elements
            html_content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
            html_content = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
            
            # Remove HTML tags
            html_content = re.sub(r'<[^>]+>', ' ', html_content)
            
            # Decode HTML entities
            html_content = unescape(html_content)
            
            # Clean up whitespace
            html_content = re.sub(r'\s+', ' ', html_content).strip()
            
            return html_content
            
        except Exception as e:
            logger.warning(f"Error extracting text from HTML: {str(e)}")
            return html_content
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in dataset."""
        try:
            df_filled = df.copy()
            
            # Fill missing values based on column type
            for col in df_filled.columns:
                if df_filled[col].dtype == 'object':
                    # Text columns
                    df_filled[col] = df_filled[col].fillna('')
                elif df_filled[col].dtype in ['int64', 'float64']:
                    # Numeric columns
                    if col.endswith('_count') or col.endswith('_score'):
                        df_filled[col] = df_filled[col].fillna(0)
                    else:
                        df_filled[col] = df_filled[col].fillna(df_filled[col].median())
                else:
                    # Boolean or other types
                    df_filled[col] = df_filled[col].fillna(0)
            
            return df_filled
            
        except Exception as e:
            logger.error(f"Error handling missing values: {str(e)}")
            return df
    
    def _validate_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure proper data types for all columns."""
        try:
            df_typed = df.copy()
            
            # Convert numeric columns
            numeric_cols = ['telecommuting', 'has_company_logo', 'has_questions', 'employment_type']
            for col in numeric_cols:
                if col in df_typed.columns:
                    df_typed[col] = pd.to_numeric(df_typed[col], errors='coerce').fillna(0).astype(int)
            
            # Convert boolean-like columns
            bool_cols = ['fraudulent']
            for col in bool_cols:
                if col in df_typed.columns:
                    df_typed[col] = df_typed[col].astype(int)
            
            return df_typed
            
        except Exception as e:
            logger.error(f"Error validating data types: {str(e)}")
            return df
    
    def _encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features."""
        try:
            df_encoded = df.copy()
            
            # Encode categorical columns
            categorical_cols = ['required_experience', 'required_education', 'industry', 'function']
            for col in categorical_cols:
                if col in df_encoded.columns:
                    df_encoded[col] = self._encode_column(df_encoded[col], col)
            
            return df_encoded
            
        except Exception as e:
            logger.error(f"Error encoding features: {str(e)}")
            return df
    
    def _encode_column(self, series: pd.Series, col_name: str) -> pd.Series:
        """Encode a single categorical column."""
        try:
            if col_name not in self.label_encoders:
                self.label_encoders[col_name] = LabelEncoder()
            
            # Handle missing values
            series_filled = series.fillna('Unknown')
            
            def safe_encode(value):
                try:
                    return self.label_encoders[col_name].transform([value])[0]
                except ValueError:
                    # Handle unseen categories
                    return -1
            
            if not hasattr(self.label_encoders[col_name], 'classes_'):
                # Fit the encoder
                return pd.Series(self.label_encoders[col_name].fit_transform(series_filled))
            else:
                # Transform using fitted encoder
                return series_filled.apply(safe_encode)
                
        except Exception as e:
            logger.error(f"Error encoding column {col_name}: {str(e)}")
            return series.fillna(0)
    
    def _normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize numeric features."""
        try:
            if self.scaling_method == 'none':
                return df
            
            df_normalized = df.copy()
            
            # Get numeric columns
            numeric_cols = df_normalized.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols and self.scaler:
                df_normalized[numeric_cols] = self.scaler.transform(df_normalized[numeric_cols])
            
            return df_normalized
            
        except Exception as e:
            logger.error(f"Error normalizing data: {str(e)}")
            return df
    
    def _fit_scaler(self, df: pd.DataFrame):
        """Fit scaler on numeric data."""
        try:
            # Get numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_cols:
                logger.warning("No numeric columns found for scaling")
                return
            
            if self.scaling_method == 'standard':
                self.scaler = StandardScaler()
            else:
                logger.warning(f"Unsupported scaling method: {self.scaling_method}")
                return
            
            self.scaler.fit(df[numeric_cols])
            logger.info(f"Fitted {self.scaling_method} scaler on {len(numeric_cols)} columns")
            
        except Exception as e:
            logger.error(f"Error fitting scaler: {str(e)}")
    
    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale features using fitted scaler."""
        try:
            if not self.scaler:
                return df
            
            df_scaled = df.copy()
            numeric_cols = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                df_scaled[numeric_cols] = self.scaler.transform(df_scaled[numeric_cols])
            
            return df_scaled
            
        except Exception as e:
            logger.error(f"Error scaling features: {str(e)}")
            return df
    
    def balance_classes(self, X: pd.DataFrame, y: pd.Series) -> tuple:
        """Balance classes in the dataset."""
        try:
            if self.balance_method == 'none':
                return X, y
            elif self.balance_method == 'smote':
                smote = SMOTE(random_state=42)
                X_balanced, y_balanced = smote.fit_resample(X, y)
                return pd.DataFrame(X_balanced, columns=X.columns), pd.Series(y_balanced)
            elif self.balance_method == 'random':
                return self._random_oversample(X, y)
            else:
                logger.warning(f"Unknown balance method: {self.balance_method}")
                return X, y
                
        except Exception as e:
            logger.error(f"Error balancing classes: {str(e)}")
            return X, y
    
    def _random_oversample(self, X: pd.DataFrame, y: pd.Series) -> tuple:
        """Perform random oversampling."""
        try:
            # Get class counts
            class_counts = y.value_counts()
            max_count = class_counts.max()
            
            X_balanced = X.copy()
            y_balanced = y.copy()
            
            # Oversample minority classes
            for class_label, count in class_counts.items():
                if count < max_count:
                    minority_indices = y[y == class_label].index
                    n_samples = max_count - count
                    
                    # Sample with replacement
                    oversample_indices = np.random.choice(minority_indices, size=n_samples, replace=True)
                    
                    X_additional = X.loc[oversample_indices]
                    y_additional = y.loc[oversample_indices]
                    
                    X_balanced = pd.concat([X_balanced, X_additional], ignore_index=True)
                    y_balanced = pd.concat([y_balanced, y_additional], ignore_index=True)
            
            # Combine
            X_balanced = pd.concat([X, X_additional], ignore_index=True)
            y_balanced = pd.concat([y, y_additional], ignore_index=True)
            
            logger.info(f"Random oversampling: {len(X)} → {len(X_balanced)} samples")
            return X_balanced, y_balanced
            
        except Exception as e:
            logger.error(f"Error in random oversampling: {str(e)}")
            return X, y
    
    def validate_dataframe(self, df: pd.DataFrame, required_columns: List[str] = None,
                          min_rows: int = 10) -> Dict[str, Any]:
        """
        Comprehensive DataFrame validation (consolidated from data_utils.py).
        
        Args:
            df: DataFrame to validate
            required_columns: Required column names
            min_rows: Minimum number of rows required
            
        Returns:
            Dict[str, Any]: Validation results
        """
        try:
            validation_result = {
                'is_valid': True,
                'warnings': [],
                'errors': [],
                'statistics': {},
                'missing_required': [],
                'missing_optional': []
            }
            
            # Check if DataFrame is empty
            if df is None or df.empty:
                validation_result['is_valid'] = False
                validation_result['errors'].append("DataFrame is None or empty")
                return validation_result
            
            # Check minimum rows
            if len(df) < min_rows:
                validation_result['warnings'].append(f"Low row count: {len(df)} < {min_rows}")
            
            # Check required columns
            if required_columns:
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    validation_result['is_valid'] = False
                    validation_result['missing_required'] = missing_columns
                    validation_result['errors'].append(f"Missing required columns: {missing_columns}")
            
            # Calculate comprehensive statistics
            validation_result['statistics'] = {
                'n_rows': len(df),
                'n_columns': len(df.columns),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
                'missing_values': df.isnull().sum().sum(),
                'duplicate_rows': df.duplicated().sum()
            }
            
            # Check for high missing values
            missing_percentage = (validation_result['statistics']['missing_values'] / 
                                (len(df) * len(df.columns))) * 100
            if missing_percentage > 20:
                validation_result['warnings'].append(f"High missing values: {missing_percentage:.1f}%")
            
            # Check for high duplication
            dup_percentage = (validation_result['statistics']['duplicate_rows'] / len(df)) * 100
            if dup_percentage > 5:
                validation_result['warnings'].append(f"High duplicate rate: {dup_percentage:.1f}%")
            
            # Check data types
            type_counts = df.dtypes.value_counts().to_dict()
            validation_result['statistics']['data_types'] = {str(k): v for k, v in type_counts.items()}
            
            # Check for constant columns
            constant_columns = []
            for col in df.columns:
                if df[col].nunique() <= 1:
                    constant_columns.append(col)
            
            if constant_columns:
                validation_result['warnings'].append(f"Constant columns found: {constant_columns}")
                validation_result['statistics']['constant_columns'] = constant_columns
            
            logger.info(f"DataFrame validation completed. Valid: {validation_result['is_valid']}")
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating DataFrame: {str(e)}")
            return {'is_valid': False, 'errors': [str(e)]}


def prepare_scraped_data_for_ml(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    SINGLE FUNCTION to prepare scraped data for ML pipeline.
    Consolidates logic from multiple serializers.
    
    Args:
        raw_data: Raw scraped data (usually from Bright Data API)
        
    Returns:
        Dict: ML-ready data with proper field mapping
    """
    try:
        if not raw_data:
            logger.warning("Empty raw data provided")
            return {}
        
        # Initialize result
        ml_data = {}
        
        # Extract basic job information
        ml_data['job_title'] = raw_data.get('job_title', '').strip()
        ml_data['job_description'] = raw_data.get('job_description', '').strip()
        ml_data['company_profile'] = raw_data.get('company_profile', '').strip()
        ml_data['requirements'] = raw_data.get('requirements', '').strip()
        ml_data['benefits'] = raw_data.get('benefits', '').strip()
        
        # Extract job metadata
        ml_data['telecommuting'] = int(bool(raw_data.get('telecommuting', False)))
        ml_data['has_company_logo'] = int(bool(raw_data.get('has_company_logo', False)))
        ml_data['has_questions'] = int(bool(raw_data.get('has_questions', False)))
        ml_data['employment_type'] = raw_data.get('employment_type', 'Unknown')
        ml_data['required_experience'] = raw_data.get('required_experience', 'Unknown')
        ml_data['required_education'] = raw_data.get('required_education', 'Unknown')
        ml_data['industry'] = raw_data.get('industry', 'Unknown')
        ml_data['function'] = raw_data.get('function', 'Unknown')
        
        # Extract location information
        location_raw = raw_data.get('location', {})
        if isinstance(location_raw, dict):
            ml_data['location'] = location_raw.get('text', 'Unknown')
        else:
            ml_data['location'] = str(location_raw) if location_raw else 'Unknown'
        
        # Extract company information
        company_raw = raw_data.get('company', {})
        if isinstance(company_raw, dict):
            ml_data['company'] = company_raw.get('name', 'Unknown')
        else:
            ml_data['company'] = str(company_raw) if company_raw else 'Unknown'
        
        # Extract poster information and verification features
        poster_raw = raw_data.get('poster', {})
        company_name = ml_data['company']
        
        if isinstance(poster_raw, dict):
            verification_features = _extract_verification_features(poster_raw, company_name)
            ml_data.update(verification_features)
        else:
            # Default verification features
            ml_data.update({
                'poster_verified': 0,
                'poster_photo': 0,
                'poster_experience': 0,
                'poster_active': 0
            })
        
        # Set default fraudulent label
        ml_data['fraudulent'] = raw_data.get('fraudulent', 0)
        
        logger.info(f"Prepared ML data with {len(ml_data)} features")
        return ml_data
        
    except Exception as e:
        logger.error(f"Error preparing scraped data for ML: {str(e)}")
        return {}


def _extract_verification_features(poster: Dict[str, Any], company_name: str) -> Dict[str, int]:
    """Extract verification features from job poster data."""
    verification_features = {}
    
    # poster_verified: Check for verification indicators
    verification_features['poster_verified'] = 1 if (
        poster.get('is_verified') or 
        poster.get('is_premium') or
        poster.get('verified_badge') or
        poster.get('name')  # Having a name indicates some verification
    ) else 0
    
    # poster_photo: Check for profile photo
    verification_features['poster_photo'] = 1 if (
        poster.get('profile_photo_url') or 
        poster.get('profile_image') or
        poster.get('photo_url')
    ) else 0
    
    # poster_experience: Check for experience at posting company
    poster_experience = poster.get('experience', [])
    company_lower = company_name.lower()
    poster_exp = 0
    
    if poster_experience and company_lower:
        for exp in poster_experience:
            if isinstance(exp, dict):
                exp_company = exp.get('company_name', '').lower()
                if company_lower in exp_company or exp_company in company_lower:
                    poster_exp = 1
                    break
            elif isinstance(exp, str) and company_lower in exp.lower():
                poster_exp = 1
                break
    
    verification_features['poster_experience'] = poster_exp
    
    # poster_active: Check for activity indicators
    verification_features['poster_active'] = 1 if (
        poster.get('num_posts', 0) > 0 or
        poster.get('recent_activity') or
        poster.get('last_active') or
        poster.get('title')  # Having a title indicates activity
    ) else 0
    
    return verification_features


# Export main classes and functions
__all__ = ['DataProcessor', 'prepare_scraped_data_for_ml']