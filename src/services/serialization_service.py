"""
Serialization Service - SINGLE SOURCE for Data Conversion
This service handles all data format conversions without duplicating business logic.

Version: 3.0.0 - DRY Consolidation
"""

import logging
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd

from ..core import DataConstants, ModelConstants
from ..core.data_processor import prepare_scraped_data_for_ml

logger = logging.getLogger(__name__)


class SerializationService:
    """
    SINGLE SOURCE for all data serialization and conversion operations.
    
    This service handles:
    - API response to DataFrame conversion
    - DataFrame to ML-ready format conversion  
    - Type validation and standardization
    - Format compatibility checking
    """
    
    def __init__(self):
        """Initialize the serialization service."""
        # No external services needed
        
        logger.info("SerializationService initialized")
    
    def api_to_dataframe(self, api_responses: Union[Dict[str, Any], List[Dict[str, Any]]]) -> pd.DataFrame:
        """
        Convert API response(s) to standardized DataFrame.
        
        Args:
            api_responses: Single response dict or list of response dicts
            
        Returns:
            pd.DataFrame: Standardized DataFrame
        """
        logger.info("Converting API response(s) to DataFrame")
        
        try:
            # Normalize to list format
            if isinstance(api_responses, dict):
                responses = [api_responses]
            elif isinstance(api_responses, list):
                responses = api_responses
            else:
                raise ValueError(f"Unsupported API response type: {type(api_responses)}")
            
            if not responses:
                return pd.DataFrame()
            
            # Process each response for ML compatibility
            processed_responses = []
            for response in responses:
                try:
                    # Use core data processor for ML preparation
                    ml_ready_data = prepare_scraped_data_for_ml(response)
                    processed_responses.append(ml_ready_data)
                except Exception as e:
                    logger.error(f"Failed to process response: {str(e)}")
                    # Add a minimal record to prevent complete failure
                    processed_responses.append(self._create_minimal_record(response, str(e)))
            
            # Create DataFrame
            df = pd.DataFrame(processed_responses)
            
            # Ensure column consistency
            df = self._ensure_column_consistency(df)
            
            logger.info(f"API to DataFrame conversion completed: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"API to DataFrame conversion failed: {str(e)}")
            return pd.DataFrame()
    
    def dataframe_to_ml_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert DataFrame to ML-ready format with all required columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            pd.DataFrame: ML-ready DataFrame with 33 required columns
        """
        logger.info("Converting DataFrame to ML format")
        
        try:
            if df.empty:
                return self._create_empty_ml_dataframe()
            
            # Use core feature engine directly
            from ..core.feature_engine import FeatureEngine
            feature_engine = FeatureEngine()
            
            # Process each row to ensure complete feature set
            processed_rows = []
            for _, row in df.iterrows():
                try:
                    row_dict = row.to_dict()
                    features_df = feature_engine.generate_complete_feature_set(row_dict)
                    processed_rows.append(features_df.iloc[0].to_dict())
                except Exception as e:
                    logger.error(f"Failed to process row: {str(e)}")
                    # Skip rows that can't be processed - no fallback values
                    continue
            
            # Create final DataFrame
            ml_df = pd.DataFrame(processed_rows)
            
            # Ensure exact column order and types
            ml_df = self._standardize_ml_format(ml_df)
            
            logger.info(f"DataFrame to ML format conversion completed: {ml_df.shape}")
            return ml_df
            
        except Exception as e:
            logger.error(f"DataFrame to ML format conversion failed: {str(e)}")
            return self._create_empty_ml_dataframe()
    
    def prepare_single_prediction(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare single job posting data for prediction.
        
        Args:
            raw_data: Raw job posting data
            
        Returns:
            Dict: ML-ready data for single prediction
        """
        logger.info("Preparing single prediction data")
        
        try:
            # Use core data processor
            ml_ready_data = prepare_scraped_data_for_ml(raw_data)
            
            # Generate complete feature set directly
            from ..core.feature_engine import FeatureEngine
            feature_engine = FeatureEngine()
            features_df = feature_engine.generate_complete_feature_set(ml_ready_data)
            
            # Log content quality status
            content_score = ml_ready_data.get('content_quality_score', 0)
            logger.info(f"Content quality ready - score: {content_score:.2f}")
            
            # Return as dictionary
            return features_df.iloc[0].to_dict()
            
        except Exception as e:
            logger.error(f"Single prediction preparation failed: {str(e)}")
            # Fail properly instead of returning default values
            raise ValueError(f"Cannot prepare prediction data: {str(e)}")
    
    def validate_ml_format(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate DataFrame for ML compatibility.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dict: Validation results
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'column_info': {}
        }
        
        try:
            # Check for required columns
            required_columns = set(ModelConstants.REQUIRED_FEATURE_COLUMNS)
            present_columns = set(df.columns)
            
            missing_columns = required_columns - present_columns
            extra_columns = present_columns - required_columns
            
            if missing_columns:
                validation_result['errors'].append(f"Missing required columns: {list(missing_columns)}")
                validation_result['is_valid'] = False
            
            if extra_columns:
                validation_result['warnings'].append(f"Extra columns found: {list(extra_columns)}")
            
            # Check data types
            type_errors = self._check_column_types(df)
            if type_errors:
                validation_result['errors'].extend(type_errors)
                validation_result['is_valid'] = False
            
            # Check for null values in critical columns
            critical_columns = DataConstants.BINARY_COLUMNS + DataConstants.SCORE_COLUMNS
            for col in critical_columns:
                if col in df.columns and df[col].isnull().any():
                    validation_result['errors'].append(f"Null values found in critical column: {col}")
                    validation_result['is_valid'] = False
            
            # Generate column info
            validation_result['column_info'] = {
                'total_columns': len(df.columns),
                'required_columns': len(required_columns),
                'missing_columns': len(missing_columns),
                'extra_columns': len(extra_columns),
                'rows': len(df)
            }
            
            logger.info(f"ML format validation: {'PASSED' if validation_result['is_valid'] else 'FAILED'}")
            
        except Exception as e:
            validation_result['errors'].append(f"Validation error: {str(e)}")
            validation_result['is_valid'] = False
            logger.error(f"ML format validation error: {str(e)}")
        
        return validation_result
    
    def _ensure_column_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure consistent columns across DataFrame."""
        # Add missing basic columns with defaults
        basic_columns = {
            'job_title': '',
            'job_description': '',
            'company_name': '',
            'location': '',
            'fraudulent': 0
        }
        
        for col, default_value in basic_columns.items():
            if col not in df.columns:
                df[col] = default_value
        
        return df
    
    def _standardize_ml_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize DataFrame to exact ML format."""
        # Ensure all required columns exist
        for col in ModelConstants.REQUIRED_FEATURE_COLUMNS:
            if col not in df.columns:
                if col in DataConstants.BINARY_COLUMNS or col in DataConstants.ENCODED_COLUMNS:
                    df[col] = 0
                elif col in DataConstants.SCORE_COLUMNS:
                    df[col] = np.nan  # ML-FIRST: Use NaN, not defaults
                elif col == 'title_word_count':
                    df[col] = 0
                else:
                    df[col] = ''
        
        # Reorder columns
        df = df[ModelConstants.REQUIRED_FEATURE_COLUMNS]
        
        # Ensure correct data types
        int_columns = (['title_word_count'] + 
                      DataConstants.BINARY_COLUMNS + 
                      DataConstants.ENCODED_COLUMNS)
        
        for col in int_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        float_columns = DataConstants.SCORE_COLUMNS
        for col in float_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)  # Keep NaN values
                # Only clip valid values, preserve NaN
                if 'score' in col:
                    df[col] = df[col].clip(0, 1)  # Clips only non-NaN values
        
        return df
    
    def _check_column_types(self, df: pd.DataFrame) -> List[str]:
        """Check if column data types are correct."""
        type_errors = []
        
        try:
            # Check integer columns
            int_columns = (['title_word_count'] + 
                          DataConstants.BINARY_COLUMNS + 
                          DataConstants.ENCODED_COLUMNS)
            
            for col in int_columns:
                if col in df.columns and not pd.api.types.is_integer_dtype(df[col]):
                    type_errors.append(f"Column {col} should be integer type")
            
            # Check float columns
            float_columns = DataConstants.SCORE_COLUMNS
            for col in float_columns:
                if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                    type_errors.append(f"Column {col} should be numeric type")
            
        except Exception as e:
            type_errors.append(f"Type checking error: {str(e)}")
        
        return type_errors
    
    def _create_minimal_record(self, original_data: Dict[str, Any], error_message: str) -> Dict[str, Any]:
        """Create minimal record from failed processing."""
        # Use default verification features
        verification_features = {
            'poster_verified': 0,
            'poster_photo': 0,
            'poster_experience': 0,
            'content_quality_score': 0.0,
            'company_legitimacy_score': 0.0
        }
        
        minimal_record = {
            'job_title': str(original_data.get('job_title', 'Processing Error')),
            'job_description': f"Processing failed: {error_message}",
            'company_name': str(original_data.get('company_name', 'Unknown')),
            'location': str(original_data.get('location', 'Unknown')),
            'fraudulent': 0,
            'language': 0
        }
        
        # Add verification features
        minimal_record.update(verification_features)
        
        return minimal_record
    
    
    def _create_empty_ml_dataframe(self) -> pd.DataFrame:
        """Create empty DataFrame with all required ML columns."""
        empty_data = {}
        
        for col in ModelConstants.REQUIRED_FEATURE_COLUMNS:
            if col in DataConstants.BINARY_COLUMNS or col in DataConstants.ENCODED_COLUMNS:
                empty_data[col] = pd.Series([], dtype=int)
            elif col in DataConstants.SCORE_COLUMNS:
                empty_data[col] = pd.Series([], dtype=float)
            elif col == 'title_word_count':
                empty_data[col] = pd.Series([], dtype=int)
            else:
                empty_data[col] = pd.Series([], dtype=str)
        
        return pd.DataFrame(empty_data)
    
    def get_conversion_stats(self) -> Dict[str, Any]:
        """Get serialization service statistics."""
        return {
            'service_status': 'active',
            'supported_formats': ['api_dict', 'dataframe', 'ml_ready'],
            'required_columns': len(ModelConstants.REQUIRED_FEATURE_COLUMNS),
            'column_categories': {
                'text_columns': len(DataConstants.TEXT_COLUMNS),
                'binary_columns': len(DataConstants.BINARY_COLUMNS),
                'score_columns': len(DataConstants.SCORE_COLUMNS),
                'encoded_columns': len(DataConstants.ENCODED_COLUMNS)
            }
        }


# Export main class
__all__ = ['SerializationService']