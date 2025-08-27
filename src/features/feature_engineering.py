"""
Feature Engineering and Combination

This module combines text and structural features, applies feature selection,
normalization, and creates the final feature vectors for machine learning models.

 Version: 1.0.0
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from .text_features import (
    extract_suspicious_keywords, calculate_grammar_score,
    analyze_sentiment, calculate_readability_scores,
    extract_contact_patterns, detect_urgency_indicators,
    calculate_text_statistics
)
from .structural_features import (
    analyze_job_structure, check_required_sections,
    analyze_formatting, calculate_description_length_score,
    analyze_experience_requirements, check_company_info_completeness,
    analyze_location_specificity, calculate_posting_quality_score
)
from ..config import FEATURE_CONFIG, NGRAM_CONFIG

logger = logging.getLogger(__name__)


def create_feature_vector(job_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Create a complete feature vector from job posting data.
    
    This is the main function that combines all feature extraction methods
    to create a comprehensive feature representation for ML models.
    
    Args:
        job_data (Dict[str, Any]): Complete job posting data
        
    Returns:
        pd.DataFrame: Feature vector as a single-row DataFrame
        
    TODO: Implement complete feature engineering pipeline:
        1. Extract text-based features
        2. Extract structural features
        3. Combine all features into single vector
        4. Handle missing values and data types
        5. Apply feature scaling if needed
        6. Return properly formatted DataFrame
    """
    if not job_data:
        logger.warning("Empty job data provided")
        return pd.DataFrame()
    
    try:
        # Extract all feature categories
        text_features = _extract_all_text_features(job_data)
        structural_features = _extract_all_structural_features(job_data)
        
        # Combine features into single dictionary
        all_features = {**text_features, **structural_features}
        
        # Convert to DataFrame
        feature_df = pd.DataFrame([all_features])
        
        # Handle missing values
        feature_df = feature_df.fillna(0)
        
        logger.info(f"Created feature vector with {len(all_features)} features")
        return feature_df
        
    except Exception as e:
        logger.error(f"Error creating feature vector: {str(e)}")
        return pd.DataFrame()


def extract_tfidf_features(text: str, vectorizer: TfidfVectorizer = None) -> np.ndarray:
    """
    Extract TF-IDF features from text.
    
    Args:
        text (str): Text to vectorize
        vectorizer (TfidfVectorizer, optional): Pre-fitted vectorizer
        
    Returns:
        np.ndarray: TF-IDF feature vector
        
    TODO: Implement TF-IDF feature extraction with:
        - Handle both training (fit_transform) and prediction (transform) modes
        - Configure n-gram ranges and vocabulary limits
        - Apply text preprocessing
        - Return dense feature array
    """
    if not text:
        return np.array([])
    
    try:
        if vectorizer is None:
            # Create new vectorizer with default config
            vectorizer = TfidfVectorizer(
                max_features=NGRAM_CONFIG['max_features'],
                min_df=NGRAM_CONFIG['min_df'],
                max_df=NGRAM_CONFIG['max_df'],
                ngram_range=(1, 2),  # Unigrams and bigrams
                stop_words='english'
            )
            # For new vectorizer, we need to fit first
            tfidf_matrix = vectorizer.fit_transform([text])
        else:
            # Use pre-fitted vectorizer
            tfidf_matrix = vectorizer.transform([text])
        
        return tfidf_matrix.toarray()[0]
        
    except Exception as e:
        logger.error(f"Error extracting TF-IDF features: {str(e)}")
        return np.array([])


def combine_text_features(text_features: Dict[str, Any]) -> np.ndarray:
    """
    Combine various text-based features into a single array.
    
    Args:
        text_features (Dict[str, Any]): Dictionary of text features
        
    Returns:
        np.ndarray: Combined text feature array
        
    TODO: Implement text feature combination with:
        - Handle different data types (counts, scores, lists)
        - Normalize feature scales appropriately
        - Create meaningful feature names
        - Handle missing or invalid features
    """
    try:
        features = []
        
        # Suspicious keyword features
        if 'suspicious_keywords' in text_features:
            keyword_counts = text_features['suspicious_keywords']
            features.extend([
                len(keyword_counts),  # Total unique suspicious keywords
                sum(keyword_counts.values()),  # Total suspicious keyword occurrences
                max(keyword_counts.values()) if keyword_counts else 0  # Max count for any keyword
            ])
        else:
            features.extend([0, 0, 0])
        
        # Sentiment features
        if 'sentiment' in text_features:
            sentiment = text_features['sentiment']
            features.extend([
                sentiment.get('positive', 0),
                sentiment.get('negative', 0),
                sentiment.get('neutral', 0),
                sentiment.get('compound', 0)
            ])
        else:
            features.extend([0, 0, 1, 0])  # Default to neutral
        
        # Readability features
        if 'readability' in text_features:
            readability = text_features['readability']
            features.extend([
                readability.get('flesch_reading_ease', 0),
                readability.get('flesch_kincaid_grade', 0)
            ])
        else:
            features.extend([0, 0])
        
        # Text statistics
        if 'text_stats' in text_features:
            stats = text_features['text_stats']
            features.extend([
                stats.get('word_count', 0),
                stats.get('sentence_count', 0),
                stats.get('avg_words_per_sentence', 0),
                stats.get('avg_word_length', 0)
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        # Contact pattern features
        if 'contact_patterns' in text_features:
            contact = text_features['contact_patterns']
            features.extend([
                contact.get('total_emails', 0),
                contact.get('suspicious_domain_count', 0)
            ])
        else:
            features.extend([0, 0])
        
        # Other features
        features.extend([
            text_features.get('grammar_score', 0),
            text_features.get('urgency_indicators', 0),
            1 if text_features.get('salary_mentions', {}).get('has_salary', False) else 0
        ])
        
        return np.array(features, dtype=float)
        
    except Exception as e:
        logger.error(f"Error combining text features: {str(e)}")
        return np.array([])


def combine_structural_features(structural_features: Dict[str, Any]) -> np.ndarray:
    """
    Combine structural features into a single array.
    
    Args:
        structural_features (Dict[str, Any]): Dictionary of structural features
        
    Returns:
        np.ndarray: Combined structural feature array
        
    TODO: Implement structural feature combination
    """
    try:
        features = []
        
        # Job structure features
        if 'job_structure' in structural_features:
            structure = structural_features['job_structure']
            features.extend([
                1 if structure.get('has_title', False) else 0,
                1 if structure.get('has_company', False) else 0,
                1 if structure.get('has_description', False) else 0,
                1 if structure.get('has_location', False) else 0,
                1 if structure.get('has_salary', False) else 0,
                structure.get('completeness_score', 0)
            ])
        else:
            features.extend([0, 0, 0, 0, 0, 0])
        
        # Required sections
        if 'required_sections' in structural_features:
            sections = structural_features['required_sections']
            section_count = sum(1 for present in sections.values() if present)
            features.append(section_count / len(sections) if sections else 0)
        else:
            features.append(0)
        
        # Formatting features
        if 'formatting' in structural_features:
            formatting = structural_features['formatting']
            features.extend([
                formatting.get('bullet_points', 0),
                formatting.get('paragraphs', 0),
                formatting.get('headings', 0),
                formatting.get('formatting_score', 0)
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        # Other structural features
        features.extend([
            structural_features.get('length_score', 0),
            structural_features.get('company_completeness', 0),
            structural_features.get('location_specificity', 0),
            structural_features.get('quality_score', 0)
        ])
        
        return np.array(features, dtype=float)
        
    except Exception as e:
        logger.error(f"Error combining structural features: {str(e)}")
        return np.array([])


def apply_feature_selection(features: pd.DataFrame, selector: SelectKBest = None, 
                          target: pd.Series = None) -> pd.DataFrame:
    """
    Apply feature selection to reduce dimensionality.
    
    Args:
        features (pd.DataFrame): Feature matrix
        selector (SelectKBest, optional): Pre-fitted feature selector
        target (pd.Series, optional): Target variable for fitting new selector
        
    Returns:
        pd.DataFrame: Selected features
        
    TODO: Implement feature selection with:
        - Handle both training and prediction modes
        - Support different selection methods
        - Maintain feature names after selection
        - Log selection statistics
    """
    if features.empty:
        return features
    
    try:
        if selector is None and target is not None:
            # Create and fit new selector
            from ..config import FEATURE_SELECTION_CONFIG
            k = min(FEATURE_SELECTION_CONFIG.get('k_features', 50), features.shape[1])
            selector = SelectKBest(score_func=f_classif, k=k)
            selected_features = selector.fit_transform(features, target)
        elif selector is not None:
            # Use pre-fitted selector
            selected_features = selector.transform(features)
        else:
            # No selection applied
            return features
        
        # Create DataFrame with selected features
        if hasattr(selector, 'get_feature_names_out'):
            feature_names = selector.get_feature_names_out(features.columns)
        else:
            # Fallback for older scikit-learn versions
            selected_indices = selector.get_support(indices=True)
            feature_names = features.columns[selected_indices]
        
        selected_df = pd.DataFrame(selected_features, 
                                  columns=feature_names, 
                                  index=features.index)
        
        logger.info(f"Feature selection: {features.shape[1]} -> {selected_df.shape[1]} features")
        return selected_df
        
    except Exception as e:
        logger.error(f"Error in feature selection: {str(e)}")
        return features


def normalize_features(features: pd.DataFrame, scaler: StandardScaler = None) -> pd.DataFrame:
    """
    Normalize features using StandardScaler.
    
    Args:
        features (pd.DataFrame): Feature matrix
        scaler (StandardScaler, optional): Pre-fitted scaler
        
    Returns:
        pd.DataFrame: Normalized features
        
    TODO: Implement feature normalization with:
        - Handle both training and prediction modes
        - Preserve feature names and index
        - Handle missing values appropriately
        - Log normalization statistics
    """
    if features.empty:
        return features
    
    try:
        if scaler is None:
            # Create and fit new scaler
            scaler = StandardScaler()
            normalized_data = scaler.fit_transform(features)
        else:
            # Use pre-fitted scaler
            normalized_data = scaler.transform(features)
        
        # Create DataFrame with normalized data
        normalized_df = pd.DataFrame(normalized_data, 
                                   columns=features.columns, 
                                   index=features.index)
        
        logger.info(f"Features normalized using StandardScaler")
        return normalized_df
        
    except Exception as e:
        logger.error(f"Error normalizing features: {str(e)}")
        return features


def engineer_interaction_features(features: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features between existing features.
    
    Args:
        features (pd.DataFrame): Base feature matrix
        
    Returns:
        pd.DataFrame: Features with added interaction terms
        
    TODO: Implement interaction feature engineering with:
        - Create meaningful feature interactions
        - Avoid creating too many features (curse of dimensionality)
        - Use domain knowledge for relevant interactions
        - Name interaction features clearly
    """
    if features.empty:
        return features
    
    try:
        # Make a copy to avoid modifying original
        enhanced_features = features.copy()
        
        # Define meaningful interactions based on domain knowledge
        # Example interactions for fraud detection
        
        # Interaction: Poor grammar + Suspicious keywords
        if 'grammar_score' in features.columns and 'suspicious_keyword_count' in features.columns:
            enhanced_features['grammar_suspicious_interaction'] = (
                (1 - features['grammar_score']) * features['suspicious_keyword_count']
            )
        
        # Interaction: Missing company info + Urgency indicators
        if 'company_completeness' in features.columns and 'urgency_indicators' in features.columns:
            enhanced_features['missing_company_urgency'] = (
                (1 - features['company_completeness']) * features['urgency_indicators']
            )
        
        # Interaction: Short description + High positive sentiment (potentially fake)
        if 'word_count' in features.columns and 'sentiment_positive' in features.columns:
            enhanced_features['short_description_positive'] = (
                (features['word_count'] < 200).astype(int) * features['sentiment_positive']
            )
        
        logger.info(f"Added interaction features: {features.shape[1]} -> {enhanced_features.shape[1]}")
        return enhanced_features
        
    except Exception as e:
        logger.error(f"Error creating interaction features: {str(e)}")
        return features


def create_feature_names(features: pd.DataFrame) -> List[str]:
    """
    Create descriptive names for all features.
    
    Args:
        features (pd.DataFrame): Feature matrix
        
    Returns:
        List[str]: List of feature names with descriptions
        
    TODO: Implement comprehensive feature naming with:
        - Descriptive names for interpretability
        - Consistent naming convention
        - Group similar features logically
        - Include feature type information
    """
    try:
        feature_descriptions = []
        
        for col in features.columns:
            # Add description based on feature name patterns
            if 'suspicious' in col:
                feature_descriptions.append(f"{col} (Fraud Indicator)")
            elif 'sentiment' in col:
                feature_descriptions.append(f"{col} (Text Sentiment)")
            elif 'grammar' in col:
                feature_descriptions.append(f"{col} (Text Quality)")
            elif 'completeness' in col or 'quality' in col:
                feature_descriptions.append(f"{col} (Structure Quality)")
            elif 'count' in col:
                feature_descriptions.append(f"{col} (Count Feature)")
            elif 'score' in col:
                feature_descriptions.append(f"{col} (Scoring Feature)")
            elif 'has_' in col:
                feature_descriptions.append(f"{col} (Presence Indicator)")
            else:
                feature_descriptions.append(col)
        
        return feature_descriptions
        
    except Exception as e:
        logger.error(f"Error creating feature names: {str(e)}")
        return list(features.columns)


# Helper functions for internal use

def _extract_all_text_features(job_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract all text-based features from job data."""
    text_features = {}
    
    job_description = job_data.get('job_description', '')
    
    if FEATURE_CONFIG['text_features']['suspicious_keywords']:
        text_features['suspicious_keywords'] = extract_suspicious_keywords(job_description)
        text_features['suspicious_keyword_count'] = len(text_features['suspicious_keywords'])
        text_features['suspicious_keyword_total'] = sum(text_features['suspicious_keywords'].values())
    
    if FEATURE_CONFIG['text_features']['grammar_score']:
        text_features['grammar_score'] = calculate_grammar_score(job_description)
    
    if FEATURE_CONFIG['text_features']['sentiment_analysis']:
        text_features['sentiment'] = analyze_sentiment(job_description)
        sentiment = text_features['sentiment']
        text_features['sentiment_positive'] = sentiment['positive']
        text_features['sentiment_negative'] = sentiment['negative']
        text_features['sentiment_compound'] = sentiment['compound']
    
    if FEATURE_CONFIG['text_features']['readability_scores']:
        text_features['readability'] = calculate_readability_scores(job_description)
        readability = text_features['readability']
        text_features['flesch_reading_ease'] = readability['flesch_reading_ease']
        text_features['flesch_kincaid_grade'] = readability['flesch_kincaid_grade']
    
    if FEATURE_CONFIG['text_features']['contact_patterns']:
        text_features['contact_patterns'] = extract_contact_patterns(job_description)
        contact = text_features['contact_patterns']
        text_features['total_emails'] = contact.get('total_emails', 0)
        text_features['suspicious_domain_count'] = contact.get('suspicious_domain_count', 0)
    
    if FEATURE_CONFIG['text_features']['text_statistics']:
        text_features['text_stats'] = calculate_text_statistics(job_description)
        stats = text_features['text_stats']
        text_features['word_count'] = stats.get('word_count', 0)
        text_features['sentence_count'] = stats.get('sentence_count', 0)
        text_features['avg_words_per_sentence'] = stats.get('avg_words_per_sentence', 0)
    
    if FEATURE_CONFIG['text_features'].get('urgency_indicators', True):
        text_features['urgency_indicators'] = detect_urgency_indicators(job_description)
    
    return text_features


def _extract_all_structural_features(job_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract all structural features from job data."""
    structural_features = {}
    
    if FEATURE_CONFIG['structural_features']['job_structure']:
        structural_features['job_structure'] = analyze_job_structure(job_data)
        structure = structural_features['job_structure']
        structural_features['has_title'] = structure.get('has_title', False)
        structural_features['has_company'] = structure.get('has_company', False)
        structural_features['has_description'] = structure.get('has_description', False)
        structural_features['completeness_score'] = structure.get('completeness_score', 0)
    
    if FEATURE_CONFIG['structural_features']['required_sections']:
        structural_features['required_sections'] = check_required_sections(job_data)
        sections = structural_features['required_sections']
        structural_features['required_sections_count'] = sum(sections.values())
    
    if FEATURE_CONFIG['structural_features']['formatting_analysis']:
        html_content = job_data.get('html_content', '')
        structural_features['formatting'] = analyze_formatting(html_content)
        formatting = structural_features['formatting']
        structural_features['bullet_points'] = formatting.get('bullet_points', 0)
        structural_features['paragraphs'] = formatting.get('paragraphs', 0)
        structural_features['formatting_score'] = formatting.get('formatting_score', 0)
    
    if FEATURE_CONFIG['structural_features']['length_analysis']:
        job_description = job_data.get('job_description', '')
        structural_features['length_score'] = calculate_description_length_score(job_description)
    
    if FEATURE_CONFIG['structural_features']['company_info']:
        structural_features['company_completeness'] = check_company_info_completeness(job_data)
    
    if FEATURE_CONFIG['structural_features']['location_analysis']:
        location = job_data.get('location', '')
        structural_features['location_specificity'] = analyze_location_specificity(location)
    
    # Overall quality score
    structural_features['quality_score'] = calculate_posting_quality_score(job_data)
    
    return structural_features