"""
Data Processor - CONTENT-FOCUSED VERSION
This module handles data preprocessing for content-focused fraud detection.
Focuses on job posting content and company metrics, not profile data.

Version: 4.0.0 - Content-Focused Data Processing
"""

import logging
import re
from html import unescape
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer, SimpleImputer
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
        self.imputer = None
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
            
            # Fit imputer for any remaining NaN values
            self._fit_imputer(X_processed)
            
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
            
            # Apply imputation for any remaining NaN values
            X_processed = self._impute_features(X_processed)
            
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
                    # Text columns - fill with empty string
                    df_filled[col] = df_filled[col].fillna('')
                elif df_filled[col].dtype in ['int64', 'float64']:
                    # Numeric columns
                    if col.endswith('_count') or col.endswith('_score'):
                        # Count and score columns - fill with 0
                        df_filled[col] = df_filled[col].fillna(0)
                    else:
                        # Other numeric columns - fill with median or 0 if median unavailable
                        median_val = df_filled[col].median()
                        fill_val = median_val if not pd.isna(median_val) else 0
                        df_filled[col] = df_filled[col].fillna(fill_val)
                else:
                    # Boolean or other types - fill with False/0
                    df_filled[col] = df_filled[col].fillna(False)
            
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
                    # Convert to numeric and fill NaN with 0 before int conversion
                    df_typed[col] = pd.to_numeric(df_typed[col], errors='coerce')
                    df_typed[col] = df_typed[col].fillna(0).astype(int)
            
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
            # NO DEFAULTS: Keep original values including NaN
            series_filled = series  # No fillna
            
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
            # NO DEFAULTS: Keep NaN values for model to handle
            return series  # No fillna
    
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
    
    def _fit_imputer(self, df: pd.DataFrame):
        """Fit imputer for handling NaN values."""
        try:
            # Get numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_cols:
                logger.info("No numeric columns found for imputation")
                return
            
            # Use median imputation for numeric data (per our plan)
            self.imputer = SimpleImputer(strategy='median')
            # Suppress numpy warnings about empty slices when computing median
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning, 
                                      message='Mean of empty slice')
                self.imputer.fit(df[numeric_cols])
            logger.info(f"Fitted imputer on {len(numeric_cols)} numeric columns")
            
        except Exception as e:
            logger.error(f"Error fitting imputer: {str(e)}")
    
    def _impute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values using fitted imputer."""
        try:
            if not self.imputer:
                return df
            
            df_imputed = df.copy()
            numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                # Suppress numpy warnings about empty slices when computing median
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=RuntimeWarning, 
                                          message='Mean of empty slice')
                    df_imputed[numeric_cols] = self.imputer.transform(df_imputed[numeric_cols])
                logger.info(f"Imputed NaN values in {len(numeric_cols)} numeric columns")
            
            return df_imputed
            
        except Exception as e:
            logger.error(f"Error imputing features: {str(e)}")
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
    
    def clean_and_enhance_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ENHANCED data cleaning with intelligent preprocessing.
        Implements Phase 1 of the Model Enhancement Plan.
        
        Args:
            df: Raw DataFrame with job posting data
            
        Returns:
            pd.DataFrame: Cleaned and enhanced DataFrame
        """
        logger.info("🧹 Starting enhanced data cleaning pipeline")
        
        try:
            df_clean = df.copy()
            
            # Step 1: Drop unnecessary columns (based on analysis)
            columns_to_drop = ['salary_range', 'job_id']  # salary_range: 84% missing, not used in Arabic
            existing_drops = [col for col in columns_to_drop if col in df_clean.columns]
            if existing_drops:
                df_clean = df_clean.drop(columns=existing_drops)
                logger.info(f"📉 Dropped columns: {existing_drops}")
            
            # Step 2: Create missing indicator features BEFORE filling
            missing_indicators = self._create_missing_indicators(df_clean)
            df_clean = pd.concat([df_clean, missing_indicators], axis=1)
            
            # Step 3: Apply intelligent imputation
            df_clean = self._apply_intelligent_imputation(df_clean)
            
            # Step 4: Create data quality features
            df_clean = self._create_quality_features(df_clean)
            
            logger.info(f"✅ Data cleaning complete. Shape: {df_clean.shape}")
            return df_clean
            
        except Exception as e:
            logger.error(f"Error in enhanced data cleaning: {str(e)}")
            raise
    
    def _create_missing_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create binary indicators for missing values in key columns."""
        indicators = pd.DataFrame(index=df.index)
        
        # Key columns that indicate fraud when missing
        key_columns = ['company_profile', 'requirements', 'benefits', 
                      'required_experience', 'required_education', 'function']
        
        for col in key_columns:
            if col in df.columns:
                indicators[f'missing_{col}'] = df[col].isna().astype(int)
        
        # Count total missing fields
        indicators['total_missing_count'] = indicators.sum(axis=1)
        
        logger.info(f"📊 Created {len(indicators.columns)} missing indicator features")
        return indicators
    
    def _apply_intelligent_imputation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply domain-knowledge based imputation strategies."""
        df_imputed = df.copy()
        
        # Department: Infer from job title
        if 'department' in df.columns:
            df_imputed['department'] = self._impute_department(df_imputed)
        
        # Required education: Based on job level
        if 'required_education' in df.columns:
            df_imputed['required_education'] = self._impute_education(df_imputed)
        
        # Benefits: Generate standard packages
        if 'benefits' in df.columns:
            df_imputed['benefits'] = self._impute_benefits(df_imputed)
        
        # Required experience: From title seniority
        if 'required_experience' in df.columns:
            df_imputed['required_experience'] = self._impute_experience(df_imputed)
        
        # Function: Map from title keywords
        if 'function' in df.columns:
            df_imputed['function'] = self._impute_function(df_imputed)
        
        logger.info("🔧 Applied intelligent imputation to all columns")
        return df_imputed
    
    def _impute_department(self, df: pd.DataFrame) -> pd.Series:
        """Impute department based on job title keywords."""
        department_map = {
            'engineer|developer|programmer|software': 'Engineering',
            'sales|account|business development': 'Sales',
            'marketing|digital|social media': 'Marketing',
            'hr|human resources|recruiter': 'Human Resources',
            'finance|accounting|financial': 'Finance',
            'design|ui|ux|graphic': 'Design',
            'manager|director|executive': 'Management',
            'analyst|data|research': 'Analytics'
        }
        
        department = df['department'].copy()
        title = df['title'].fillna('').str.lower()
        
        for pattern, dept in department_map.items():
            mask = department.isna() & title.str.contains(pattern, na=False)
            department.loc[mask] = dept
        
        # Fill remaining with 'General'
        department.fillna('General', inplace=True)
        return department
    
    def _impute_education(self, df: pd.DataFrame) -> pd.Series:
        """Impute education requirements based on job seniority."""
        education = df['required_education'].copy()
        title = df['title'].fillna('').str.lower()
        
        # Senior/Lead positions
        mask = education.isna() & title.str.contains('senior|lead|principal|architect', na=False)
        education.loc[mask] = "Bachelor's or Master's degree"
        
        # Management positions
        mask = education.isna() & title.str.contains('manager|director|vp|head', na=False)
        education.loc[mask] = "Master's degree preferred"
        
        # Entry level
        mask = education.isna() & title.str.contains('junior|entry|intern|trainee', na=False)
        education.loc[mask] = "Bachelor's degree"
        
        # Fill remaining
        education.fillna("Bachelor's degree or equivalent experience", inplace=True)
        return education
    
    def _impute_benefits(self, df: pd.DataFrame) -> pd.Series:
        """Generate realistic benefits packages."""
        benefits = df['benefits'].copy()
        
        benefit_templates = [
            "Health insurance, paid time off, professional development opportunities",
            "Competitive salary, medical benefits, flexible working hours",
            "Health coverage, annual leave, training and development programs",
            "Medical insurance, vacation days, career advancement opportunities",
            "Healthcare benefits, PTO, skill development programs"
        ]
        
        # Randomly assign templates to missing benefits
        mask = benefits.isna()
        import random
        for idx in benefits[mask].index:
            benefits.loc[idx] = random.choice(benefit_templates)
        
        return benefits
    
    def _impute_experience(self, df: pd.DataFrame) -> pd.Series:
        """Impute experience requirements from title."""
        experience = df['required_experience'].copy()
        title = df['title'].fillna('').str.lower()
        
        # Experience mapping
        exp_map = {
            'entry|junior|trainee|intern': "0-2 years",
            'mid|intermediate': "2-5 years", 
            'senior': "5+ years",
            'lead|principal': "7+ years",
            'manager|director': "8+ years",
            'vp|head|chief': "10+ years"
        }
        
        for pattern, exp in exp_map.items():
            mask = experience.isna() & title.str.contains(pattern, na=False)
            experience.loc[mask] = exp
        
        experience.fillna("2-5 years", inplace=True)
        return experience
    
    def _impute_function(self, df: pd.DataFrame) -> pd.Series:
        """Impute function from job title keywords."""
        function = df['function'].copy()
        title = df['title'].fillna('').str.lower()
        
        function_map = {
            'engineer|developer|programmer|tech|software|it': 'Information Technology',
            'sales|account|business': 'Sales',
            'marketing|digital|social': 'Marketing', 
            'finance|accounting|financial': 'Finance',
            'hr|human resources': 'Human Resources',
            'design|ui|ux|creative': 'Design',
            'operations|logistics|supply': 'Operations',
            'customer|support|service': 'Customer Service'
        }
        
        for pattern, func in function_map.items():
            mask = function.isna() & title.str.contains(pattern, na=False)
            function.loc[mask] = func
        
        function.fillna('General', inplace=True)
        return function
    
    def _create_quality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create data completeness and quality features."""
        df_enhanced = df.copy()
        
        # Information completeness score
        total_cols = len(df.columns)
        df_enhanced['info_completeness'] = df.notna().sum(axis=1) / total_cols
        
        # Company legitimacy indicators  
        df_enhanced['has_company_logo'] = df.get('has_company_logo', 0)
        df_enhanced['has_company_profile'] = (~df['company_profile'].isna()).astype(int)
        df_enhanced['has_requirements'] = (~df['requirements'].isna()).astype(int)
        df_enhanced['has_benefits'] = (~df['benefits'].isna()).astype(int)
        df_enhanced['has_questions'] = df.get('has_questions', 0)
        
        logger.info("📈 Created data quality features")
        return df_enhanced
    
    def balance_classes_enhanced(self, X: pd.DataFrame, y: pd.Series) -> tuple:
        """
        ENHANCED class balancing with better SMOTE implementation.
        Uses BorderlineSMOTE for more realistic synthetic samples.
        """
        try:
            if self.balance_method == 'none':
                return X, y
            elif self.balance_method == 'smote':
                # Use BorderlineSMOTE for better quality synthetic samples
                smote = BorderlineSMOTE(
                    sampling_strategy=0.2,  # 1:5 ratio, not 1:1
                    random_state=42,
                    k_neighbors=3,
                    kind='borderline-1'
                )
                logger.info("🔄 Using BorderlineSMOTE with 1:5 ratio for better balance")
                X_balanced, y_balanced = smote.fit_resample(X, y)
                
                # Add small amount of noise to prevent overfitting
                noise = np.random.normal(0, 0.01, X_balanced.shape)
                X_balanced_noisy = X_balanced + noise
                
                logger.info(f"⚖️  Class balancing: {len(X)} → {len(X_balanced)} samples")
                return pd.DataFrame(X_balanced_noisy, columns=X.columns), pd.Series(y_balanced)
                
            elif self.balance_method == 'adasyn':
                # Alternative: ADASYN for adaptive synthetic samples
                adasyn = ADASYN(sampling_strategy=0.2, random_state=42)
                X_balanced, y_balanced = adasyn.fit_resample(X, y)
                return pd.DataFrame(X_balanced, columns=X.columns), pd.Series(y_balanced)
                
            else:
                return self.balance_classes(X, y)  # Fall back to original
                
        except Exception as e:
            logger.error(f"Error in enhanced class balancing: {str(e)}")
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
        
        # Extract company information - handle multiple field names
        company_raw = raw_data.get('company', raw_data.get('company_name', {}))
        if isinstance(company_raw, dict):
            ml_data['company'] = company_raw.get('name', 'Unknown')
        else:
            ml_data['company'] = str(company_raw) if company_raw else 'Unknown'
        
        # Company metrics from enriched scraping - preserve None/NaN for ML-first approach
        ml_data['company_followers'] = raw_data.get('company_followers')
        ml_data['company_employees'] = raw_data.get('company_employees')
        ml_data['company_founded'] = raw_data.get('company_founded')
        ml_data['company_size'] = raw_data.get('company_size')
        
        # Company score fields - preserve None/NaN
        ml_data['company_followers_score'] = raw_data.get('company_followers_score')
        ml_data['company_employees_score'] = raw_data.get('company_employees_score')
        ml_data['company_founded_score'] = raw_data.get('company_founded_score')
        
        # Content quality metrics from enrichment
        ml_data['content_quality_score'] = raw_data.get('content_quality_score', 0.0)
        ml_data['professional_language_score'] = raw_data.get('professional_language_score', 0.0)
        ml_data['urgency_language_score'] = raw_data.get('urgency_language_score', 0.0)
        
        # Contact risk metrics
        ml_data['contact_risk_score'] = raw_data.get('contact_risk_score', 0.0)
        ml_data['has_whatsapp'] = raw_data.get('has_whatsapp', 0)
        ml_data['has_telegram'] = raw_data.get('has_telegram', 0)
        ml_data['has_professional_email'] = raw_data.get('has_professional_email', 0)
        
        # Company enrichment flags
        ml_data['company_enrichment_success'] = raw_data.get('company_enrichment_success', False)
        ml_data['profile_enrichment_success'] = raw_data.get('profile_enrichment_success', False)
        
        # Text analysis features
        ml_data['suspicious_keywords_count'] = raw_data.get('suspicious_keywords_count', 0)
        ml_data['arabic_suspicious_score'] = raw_data.get('arabic_suspicious_score', 0.0)
        ml_data['english_suspicious_score'] = raw_data.get('english_suspicious_score', 0.0)
        
        # Job structure features
        ml_data['has_salary_info'] = raw_data.get('has_salary_info', 0)
        ml_data['has_requirements'] = raw_data.get('has_requirements', 0)
        ml_data['has_location'] = raw_data.get('has_location', 0)
        ml_data['has_experience_level'] = raw_data.get('has_experience_level', 0)
        
        # Fraud risk score (composite)
        ml_data['fraud_risk_score'] = raw_data.get('fraud_risk_score', 0.0)
        
        logger.info("✅ Content-focused data processing completed")
        
        # Set default fraudulent label
        ml_data['fraudulent'] = raw_data.get('fraudulent', 0)
        
        logger.info(f"✅ Prepared content-focused ML data with {len(ml_data)} features")
        return ml_data
        
    except Exception as e:
        logger.error(f"Error preparing scraped data for ML: {str(e)}")
        return {}




# Export main classes and functions
__all__ = ['DataProcessor', 'prepare_scraped_data_for_ml']