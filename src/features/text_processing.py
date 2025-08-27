"""
Text Processing Module

Essential text preprocessing functions for fraud detection.
Implemented by Orchestration Engineer for immediate project needs.

Author: Orchestration Engineer - Infrastructure & Deployment
Version: 1.0.0
"""

import re
import string
import logging
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def clean_text(text: str, remove_arabic: bool = False) -> str:
    """
    Clean and normalize text for processing.
    
    Args:
        text (str): Input text to clean
        remove_arabic (bool): Whether to remove Arabic characters
        
    Returns:
        str: Cleaned text
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower().strip()
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove Arabic characters if requested
    if remove_arabic:
        text = re.sub(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]', '', text)
    
    # Clean common issues
    text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)  # Keep alphanumeric and Arabic
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def extract_basic_text_features(text: str) -> Dict[str, Union[int, float]]:
    """
    Extract basic statistical features from text.
    
    Args:
        text (str): Input text
        
    Returns:
        Dict[str, Union[int, float]]: Dictionary of text features
    """
    if pd.isna(text) or not isinstance(text, str):
        return {
            'char_count': 0,
            'word_count': 0,
            'sentence_count': 0,
            'avg_word_length': 0.0,
            'uppercase_ratio': 0.0,
            'digit_ratio': 0.0,
            'special_char_ratio': 0.0
        }
    
    cleaned_text = clean_text(text)
    
    # Basic counts
    char_count = len(text)
    words = cleaned_text.split() if cleaned_text else []
    word_count = len(words)
    sentence_count = len([s for s in re.split(r'[.!?]+', text) if s.strip()])
    
    # Calculate ratios
    avg_word_length = np.mean([len(word) for word in words]) if words else 0.0
    uppercase_count = sum(1 for c in text if c.isupper())
    digit_count = sum(1 for c in text if c.isdigit())
    special_count = sum(1 for c in text if c in string.punctuation)
    
    uppercase_ratio = uppercase_count / char_count if char_count > 0 else 0.0
    digit_ratio = digit_count / char_count if char_count > 0 else 0.0
    special_char_ratio = special_count / char_count if char_count > 0 else 0.0
    
    return {
        'char_count': char_count,
        'word_count': word_count,
        'sentence_count': max(sentence_count, 1),  # Avoid division by zero
        'avg_word_length': float(avg_word_length),
        'uppercase_ratio': float(uppercase_ratio),
        'digit_ratio': float(digit_ratio),
        'special_char_ratio': float(special_char_ratio)
    }


def detect_suspicious_patterns(text: str) -> Dict[str, bool]:
    """
    Detect suspicious patterns that may indicate fraud.
    
    Args:
        text (str): Input text to analyze
        
    Returns:
        Dict[str, bool]: Dictionary of detected patterns
    """
    if pd.isna(text) or not isinstance(text, str):
        return {
            'has_email': False,
            'has_phone': False,
            'has_url': False,
            'excessive_caps': False,
            'urgent_language': False,
            'money_mention': False,
            'contact_request': False
        }
    
    text_lower = text.lower()
    
    # Pattern detection
    patterns = {
        'has_email': bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)),
        'has_phone': bool(re.search(r'(\+?\d[\d\s\-\(\)]{7,}\d)', text)),
        'has_url': bool(re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)),
        'excessive_caps': len([c for c in text if c.isupper()]) > len(text) * 0.3,
        'urgent_language': any(word in text_lower for word in ['urgent', 'immediate', 'asap', 'quickly', 'fast', 'now']),
        'money_mention': any(word in text_lower for word in ['$', 'dollar', 'money', 'cash', 'salary', 'pay', 'earn']),
        'contact_request': any(phrase in text_lower for phrase in ['contact me', 'call me', 'email me', 'whatsapp', 'telegram'])
    }
    
    return patterns


def calculate_text_quality_score(text: str) -> float:
    """
    Calculate a simple text quality score (0-1).
    
    Args:
        text (str): Input text
        
    Returns:
        float: Quality score between 0 and 1
    """
    if pd.isna(text) or not isinstance(text, str) or len(text.strip()) == 0:
        return 0.0
    
    features = extract_basic_text_features(text)
    suspicious = detect_suspicious_patterns(text)
    
    # Quality indicators (higher is better)
    quality_score = 0.5  # Base score
    
    # Length indicators
    if features['word_count'] >= 10:  # Reasonable length
        quality_score += 0.1
    if features['word_count'] >= 50:  # Good length
        quality_score += 0.1
        
    # Language quality indicators
    if 2 <= features['avg_word_length'] <= 8:  # Reasonable word length
        quality_score += 0.1
    if features['sentence_count'] > 1:  # Multiple sentences
        quality_score += 0.1
        
    # Penalize suspicious patterns
    if features['uppercase_ratio'] > 0.3:  # Too many caps
        quality_score -= 0.1
    if suspicious['excessive_caps']:
        quality_score -= 0.1
    if suspicious['urgent_language']:
        quality_score -= 0.05
        
    # Bonus for professional indicators
    if not suspicious['urgent_language'] and features['word_count'] > 20:
        quality_score += 0.1
    
    return max(0.0, min(1.0, quality_score))


def process_text_column(df: pd.DataFrame, column_name: str, prefix: str = None) -> pd.DataFrame:
    """
    Process a text column and add derived features.
    
    Args:
        df (pd.DataFrame): Input dataframe
        column_name (str): Name of text column to process
        prefix (str): Prefix for new feature columns
        
    Returns:
        pd.DataFrame: Dataframe with new text features
    """
    if column_name not in df.columns:
        logger.warning(f"Column '{column_name}' not found in dataframe")
        return df
    
    df_result = df.copy()
    prefix = prefix or f"{column_name}_"
    
    # Extract features for each text entry
    logger.info(f"Processing text features for column: {column_name}")
    
    text_features = df[column_name].apply(extract_basic_text_features)
    suspicious_patterns = df[column_name].apply(detect_suspicious_patterns)
    quality_scores = df[column_name].apply(calculate_text_quality_score)
    
    # Add basic features
    for feature_name in ['char_count', 'word_count', 'sentence_count', 'avg_word_length',
                        'uppercase_ratio', 'digit_ratio', 'special_char_ratio']:
        df_result[f"{prefix}{feature_name}"] = [features[feature_name] for features in text_features]
    
    # Add suspicious pattern features
    for pattern_name in ['has_email', 'has_phone', 'has_url', 'excessive_caps',
                        'urgent_language', 'money_mention', 'contact_request']:
        df_result[f"{prefix}{pattern_name}"] = [patterns[pattern_name] for patterns in suspicious_patterns]
    
    # Add quality score
    df_result[f"{prefix}quality_score"] = quality_scores
    
    logger.info(f"Added {len(text_features[0]) + len(suspicious_patterns[0]) + 1} text features for {column_name}")
    
    return df_result


def normalize_arabic_text(text: str) -> str:
    """
    Basic Arabic text normalization.
    
    Args:
        text (str): Input Arabic text
        
    Returns:
        str: Normalized Arabic text
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Basic Arabic normalization
    text = text.strip()
    
    # Normalize Arabic characters
    arabic_normalizations = {
        'أ': 'ا', 'إ': 'ا', 'آ': 'ا',  # Alif variations
        'ة': 'ه',  # Taa marbuta to haa
        'ي': 'ى', 'ئ': 'ى', 'ؤ': 'و'  # Ya and Waw variations
    }
    
    for old_char, new_char in arabic_normalizations.items():
        text = text.replace(old_char, new_char)
    
    # Remove diacritics
    text = re.sub(r'[\u064B-\u0652\u0670\u0640]', '', text)
    
    # Clean whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def extract_contact_info(text: str) -> Dict[str, List[str]]:
    """
    Extract contact information from text.
    
    Args:
        text (str): Input text
        
    Returns:
        Dict[str, List[str]]: Extracted contact information
    """
    if pd.isna(text) or not isinstance(text, str):
        return {'emails': [], 'phones': [], 'urls': []}
    
    # Extract emails
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    
    # Extract phone numbers
    phones = re.findall(r'(\+?\d[\d\s\-\(\)]{7,}\d)', text)
    
    # Extract URLs
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    
    return {
        'emails': emails,
        'phones': phones,
        'urls': urls
    }
