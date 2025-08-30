"""
Centralized Encoding Utilities

This module provides all ordinal encoding mappings used throughout the project.
Single source of truth for converting text values to numeric codes.

Version: 1.0.0
Author: System Integration Fix
"""

import logging
from typing import Dict, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)

# ================================
# ORDINAL ENCODING MAPPINGS
# ================================

EXPERIENCE_MAPPING = {
    '': 0, 'nan': 0, 'unknown': 0, 'unspecified': 0,  # Unspecified
    'entry': 1, 'entry level': 1, 'internship': 1, 'intern': 1,
    'associate': 2, 'associate level': 2, '1-2 years': 2, '0-1 years': 2,
    'mid': 3, 'mid-level': 3, 'mid-senior level': 3, '3-5 years': 3, '2-4 years': 3,
    'senior': 4, 'senior level': 4, '5+ years': 4, '4+ years': 4, '5-10 years': 4,
    'executive': 5, 'executive level': 5, 'director': 5, 'manager': 5, '10+ years': 5,
    # Numeric values (from Arabic data)
    '0': 1, '1': 2, '2': 3, '3': 4, '4': 5,
    # Bright Data specific values
    'not applicable': 0,
    # Additional variations
    'junior': 1, 'experienced': 4, 'lead': 4, 'principal': 5
}

EDUCATION_MAPPING = {
    '': 0, 'nan': 0, 'unknown': 0, 'unspecified': 0,  # Unspecified
    'none': 1, 'no formal education': 1, 'no education': 1,
    'high school': 2, 'high school or equivalent': 2, 'secondary': 2, 'diploma': 2,
    'associate': 3, 'associate degree': 3, 'some college coursework completed': 3, 
    'vocational': 3, 'trade school': 3, 'certificate': 3,
    'bachelor': 4, "bachelor's": 4, "bachelor's degree": 4, 'undergraduate': 4, 'ba': 4, 'bs': 4,
    'master': 5, "master's": 5, "master's degree": 5, 'graduate': 5, 'ma': 5, 'ms': 5, 'mba': 5,
    'phd': 6, 'doctorate': 6, 'doctoral': 6, 'doctoral degree': 6, 'ph.d.': 6,
    # Additional variations
    'certification': 4, 'professional degree': 5
}

EMPLOYMENT_MAPPING = {
    '': 0, 'nan': 0, 'unknown': 0, 'unspecified': 0,  # Unspecified
    'contract': 1, 'contractor': 1, 'freelance': 1,
    'part-time': 2, 'part time': 2, 'دوام جزئي': 2,  # Arabic part-time
    'internship': 3, 'intern': 3, 'apprentice': 3,
    'temporary': 4, 'temp': 4, 'seasonal': 4, 'عقد مؤقت': 4,  # Arabic temporary
    'full-time': 5, 'full time': 5, 'permanent': 5, 'دوام كامل': 5,  # Arabic full-time
    'other': 6, 'remote': 6, 'hybrid': 6, 'عمل عن بعد': 6,  # Arabic remote work
    # Additional variations
    'volunteer': 3, 'casual': 2, 'consulting': 1
}

# ================================
# REVERSE MAPPINGS (for debugging)
# ================================

EXPERIENCE_REVERSE = {v: k for k, v in EXPERIENCE_MAPPING.items() if k != '' and k != 'nan'}
EDUCATION_REVERSE = {v: k for k, v in EDUCATION_MAPPING.items() if k != '' and k != 'nan'}
EMPLOYMENT_REVERSE = {v: k for k, v in EMPLOYMENT_MAPPING.items() if k != '' and k != 'nan'}

# ================================
# ENCODING FUNCTIONS
# ================================

def encode_experience_level(value: Union[str, int, float], default: int = 0) -> int:
    """
    Encode experience level to ordinal value.
    
    Args:
        value: Experience level as string, int, or float
        default: Default value if encoding fails
        
    Returns:
        int: Encoded experience level (0-5)
    """
    if pd.isna(value):
        return default
    
    # Handle numeric inputs
    if isinstance(value, (int, float)):
        # Convert to int and clamp to valid range
        int_val = int(value)
        return max(0, min(5, int_val))
    
    # Handle string inputs
    value_str = str(value).lower().strip()
    encoded = EXPERIENCE_MAPPING.get(value_str, default)
    
    if encoded == default and value_str not in ['', 'nan', 'unknown', 'unspecified']:
        logger.debug(f"Unknown experience level: '{value}' -> using default {default}")
    
    return encoded

def encode_education_level(value: Union[str, int, float], default: int = 0) -> int:
    """
    Encode education level to ordinal value.
    
    Args:
        value: Education level as string, int, or float
        default: Default value if encoding fails
        
    Returns:
        int: Encoded education level (0-6)
    """
    if pd.isna(value):
        return default
    
    # Handle numeric inputs
    if isinstance(value, (int, float)):
        int_val = int(value)
        return max(0, min(6, int_val))
    
    # Handle string inputs
    value_str = str(value).lower().strip()
    encoded = EDUCATION_MAPPING.get(value_str, default)
    
    if encoded == default and value_str not in ['', 'nan', 'unknown', 'unspecified']:
        logger.debug(f"Unknown education level: '{value}' -> using default {default}")
    
    return encoded

def encode_employment_type(value: Union[str, int, float], default: int = 0) -> int:
    """
    Encode employment type to ordinal value.
    
    Args:
        value: Employment type as string, int, or float
        default: Default value if encoding fails
        
    Returns:
        int: Encoded employment type (0-6)
    """
    if pd.isna(value):
        return default
    
    # Handle numeric inputs
    if isinstance(value, (int, float)):
        int_val = int(value)
        return max(0, min(6, int_val))
    
    # Handle string inputs
    value_str = str(value).lower().strip()
    encoded = EMPLOYMENT_MAPPING.get(value_str, default)
    
    if encoded == default and value_str not in ['', 'nan', 'unknown', 'unspecified']:
        logger.debug(f"Unknown employment type: '{value}' -> using default {default}")
    
    return encoded

def encode_language(value: Union[str, int, float], default: int = 0) -> int:
    """
    Encode language to numeric value.
    
    Args:
        value: Language as string, int, or float
        default: Default value (0 = English)
        
    Returns:
        int: 0 = English, 1 = Arabic
    """
    if pd.isna(value):
        return default
    
    if isinstance(value, (int, float)):
        return 1 if int(value) == 1 else 0
    
    value_str = str(value).lower().strip()
    
    # Arabic indicators
    if value_str in ['arabic', 'ar', '1', 'عربي']:
        return 1
    
    # English indicators (default)
    return 0

# ================================
# BATCH ENCODING FUNCTIONS
# ================================

def encode_dataframe_columns(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """
    Apply ordinal encoding to standard columns in DataFrame.
    
    Args:
        df: DataFrame with text columns to encode
        inplace: Whether to modify DataFrame in place
        
    Returns:
        pd.DataFrame: DataFrame with encoded columns added
    """
    if not inplace:
        df = df.copy()
    
    # Encode experience level
    if 'experience_level' in df.columns:
        df['experience_level_encoded'] = df['experience_level'].apply(encode_experience_level)
        logger.debug(f"Encoded experience_level: {df['experience_level_encoded'].value_counts().to_dict()}")
    
    # Encode education level
    if 'education_level' in df.columns:
        df['education_level_encoded'] = df['education_level'].apply(encode_education_level)
        logger.debug(f"Encoded education_level: {df['education_level_encoded'].value_counts().to_dict()}")
    
    # Encode employment type
    if 'employment_type' in df.columns:
        df['employment_type_encoded'] = df['employment_type'].apply(encode_employment_type)
        logger.debug(f"Encoded employment_type: {df['employment_type_encoded'].value_counts().to_dict()}")
    
    # Encode language
    if 'language' in df.columns and df['language'].dtype == 'object':
        df['language'] = df['language'].apply(encode_language)
        logger.debug(f"Encoded language: {df['language'].value_counts().to_dict()}")
    
    return df

def get_encoding_info() -> Dict[str, Dict]:
    """
    Get information about all available encodings.
    
    Returns:
        Dict: Encoding information for debugging
    """
    return {
        'experience_levels': {
            'mapping': EXPERIENCE_MAPPING,
            'reverse': EXPERIENCE_REVERSE,
            'range': '0-5'
        },
        'education_levels': {
            'mapping': EDUCATION_MAPPING,
            'reverse': EDUCATION_REVERSE,
            'range': '0-6'
        },
        'employment_types': {
            'mapping': EMPLOYMENT_MAPPING,
            'reverse': EMPLOYMENT_REVERSE,
            'range': '0-6'
        },
        'languages': {
            'mapping': {'english': 0, 'arabic': 1},
            'reverse': {0: 'english', 1: 'arabic'},
            'range': '0-1'
        }
    }

# ================================
# VALIDATION FUNCTIONS
# ================================

def validate_encoded_values(df: pd.DataFrame) -> Dict[str, bool]:
    """
    Validate that encoded columns have correct value ranges.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Dict[str, bool]: Validation results
    """
    results = {}
    
    if 'experience_level_encoded' in df.columns:
        valid_range = df['experience_level_encoded'].between(0, 5, inclusive='both').all()
        results['experience_level_encoded'] = valid_range
    
    if 'education_level_encoded' in df.columns:
        valid_range = df['education_level_encoded'].between(0, 6, inclusive='both').all()
        results['education_level_encoded'] = valid_range
    
    if 'employment_type_encoded' in df.columns:
        valid_range = df['employment_type_encoded'].between(0, 6, inclusive='both').all()
        results['employment_type_encoded'] = valid_range
    
    if 'language' in df.columns:
        valid_range = df['language'].isin([0, 1]).all()
        results['language'] = valid_range
    
    return results