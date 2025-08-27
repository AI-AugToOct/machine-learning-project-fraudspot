"""
Configuration Constants for Job Post Fraud Detector

This module contains all configuration constants, parameters, and settings
used throughout the fraud detection system. Centralized configuration
makes it easy to tune the system and maintain consistency.

 Version: 1.0.0
"""

from typing import Dict, List, Any
import os

# ================================
# SUSPICIOUS KEYWORDS AND PATTERNS
# ================================

SUSPICIOUS_KEYWORDS = [
    # Communication channels (non-professional)
    "whatsapp", "telegram", "wire transfer", "western union", 
    "moneygram", "personal email", "gmail", "yahoo", "hotmail",
    
    # Payment red flags
    "upfront fee", "registration fee", "processing fee", "administration fee",
    "deposit required", "cash only", "payment required", "advance payment",
    "wire money", "send money", "transfer funds",
    
    # Work-from-home scams
    "work from home", "no experience needed", "easy money", "quick money",
    "unlimited income", "be your own boss", "financial freedom", 
    "earn while you sleep", "passive income", "make money fast",
    
    # Urgency indicators
    "urgent", "immediate start", "limited time", "act now", "don't wait",
    "expires soon", "hurry", "limited spots", "first come first served",
    
    # Vague/unrealistic promises
    "no skills required", "anyone can do", "guaranteed income", 
    "100% success rate", "risk-free", "guaranteed job", "instant approval",
    
    # MLM/Pyramid scheme indicators
    "network marketing", "multi-level marketing", "mlm", "pyramid",
    "recruit others", "build your team", "downline", "upline",
    
    # Fake credentials
    "work visa guaranteed", "no background check", "no references needed",
    "fake id accepted", "undocumented workers welcome"
]

# Suspicious domains commonly used in fraud
SUSPICIOUS_EMAIL_DOMAINS = [
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com",
    "mail.com", "yandex.com", "protonmail.com", "tutanota.com"
]

# Professional domains that indicate legitimacy
TRUSTED_EMAIL_DOMAINS = [
    "linkedin.com", "indeed.com", "monster.com", "glassdoor.com",
    "microsoft.com", "google.com", "amazon.com", "apple.com"
]

# ========================
# MODEL CONFIGURATIONS
# ========================

MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42
    },
    'xgboost': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 6,
        'random_state': 42
    },
    'svm': {
        'kernel': 'rbf',
        'C': 1.0,
        'probability': True,
        'random_state': 42
    },
    'logistic_regression': {
        'C': 1.0,
        'max_iter': 1000,
        'random_state': 42
    },
    'naive_bayes': {
        'alpha': 1.0
    }
}

# Feature selection and preprocessing
FEATURE_SELECTION_CONFIG = {
    'method': 'selectkbest',  # 'selectkbest', 'rfe', 'lasso'
    'k_features': 50,
    'score_func': 'f_classif'
}

FEATURE_IMPORTANCE_THRESHOLD = 0.01
MIN_FEATURE_CORRELATION = 0.05

# ========================
# CONFIDENCE THRESHOLDS
# ========================

CONFIDENCE_THRESHOLDS = {
    'high_risk': 0.8,      # 80%+ confidence of fraud
    'medium_risk': 0.5,    # 50-80% confidence of fraud
    'low_risk': 0.3        # 30-50% confidence of fraud
}

PREDICTION_THRESHOLDS = {
    'fraud_threshold': 0.5,    # Threshold for binary fraud classification
    'confidence_threshold': 0.3  # Minimum confidence for predictions
}

# ========================
# SCRAPING CONFIGURATION
# ========================

SCRAPING_CONFIG = {
    'timeout': 30,              # Request timeout in seconds
    'rate_limit_delay': 2,      # Delay between requests
    'max_retries': 3,           # Maximum retry attempts
    'retry_delay': 5,           # Delay between retries
    'user_agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# LinkedIn-specific patterns
LINKEDIN_URL_PATTERNS = [
    r'https://www\.linkedin\.com/jobs/view/\d+',
    r'https://linkedin\.com/jobs/view/\d+',
    r'https://www\.linkedin\.com/jobs/collections/recommended/\?currentJobId=\d+'
]

# Headers for web scraping
DEFAULT_HEADERS = {
    'User-Agent': SCRAPING_CONFIG['user_agent'],
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
}

# ========================
# TEXT PROCESSING CONFIG
# ========================

TEXT_PROCESSING_CONFIG = {
    'min_text_length': 50,      # Minimum text length for analysis
    'max_text_length': 10000,   # Maximum text length to process
    'language': 'english',      # Language for stopwords and stemming
    'remove_stopwords': True,   # Whether to remove stopwords
    'lemmatize': True,          # Whether to lemmatize text
    'remove_punctuation': True, # Whether to remove punctuation
    'lowercase': True           # Whether to convert to lowercase
}

# N-gram configuration
NGRAM_CONFIG = {
    'unigram_range': (1, 1),
    'bigram_range': (2, 2),
    'trigram_range': (3, 3),
    'max_features': 1000,
    'min_df': 2,
    'max_df': 0.95
}

# ========================
# FEATURE ENGINEERING
# ========================

FEATURE_CONFIG = {
    'text_features': {
        'suspicious_keywords': True,
        'grammar_score': True,
        'sentiment_analysis': True,
        'readability_scores': True,
        'text_statistics': True,
        'contact_patterns': True,
        'tfidf_features': True,
        'ngram_features': True
    },
    'structural_features': {
        'job_structure': True,
        'required_sections': True,
        'formatting_analysis': True,
        'length_analysis': True,
        'company_info': True,
        'location_analysis': True,
        'application_method': True
    }
}

# Required job posting sections
REQUIRED_JOB_SECTIONS = [
    'responsibilities', 'requirements', 'qualifications', 
    'benefits', 'description', 'duties'
]

# ========================
# CACHING CONFIGURATION
# ========================

CACHE_CONFIG = {
    'cache_dir': 'data/cache',
    'cache_expiry_days': 7,     # Cache results for 7 days
    'max_cache_size_mb': 100,   # Maximum cache size in MB
    'enable_cache': True        # Whether to enable caching
}

# ========================
# MODEL PATHS
# ========================

MODEL_PATHS = {
    'base_dir': 'data/models',
    'trained_model': 'fraud_detector_model.pkl',
    'vectorizer': 'tfidf_vectorizer.pkl',
    'feature_selector': 'feature_selector.pkl',
    'label_encoder': 'label_encoder.pkl',
    'scaler': 'feature_scaler.pkl'
}

# ========================
# EVALUATION METRICS
# ========================

EVALUATION_METRICS = [
    'accuracy', 'precision', 'recall', 'f1_score', 
    'roc_auc', 'confusion_matrix', 'classification_report'
]

# Cross-validation configuration
CV_CONFIG = {
    'cv_folds': 5,
    'shuffle': True,
    'stratify': True,
    'random_state': 42
}

# ========================
# LOGGING CONFIGURATION
# ========================

LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': 'logs/fraud_detector.log',
    'max_bytes': 10485760,  # 10MB
    'backup_count': 5
}

# ========================
# DATA PATHS
# ========================

DATA_PATHS = {
    'raw_data': 'data/raw',
    'processed_data': 'data/processed',
    'training_data': 'data/processed/training_data.csv',
    'test_data': 'data/processed/test_data.csv',
    'validation_data': 'data/processed/validation_data.csv'
}

# ========================
# STREAMLIT CONFIGURATION
# ========================

STREAMLIT_CONFIG = {
    'page_title': 'Job Post Fraud Detector',
    'page_icon': '🕵️',
    'layout': 'wide',
    'sidebar_state': 'expanded',
    'theme': {
        'primary_color': '#1f77b4',
        'background_color': '#ffffff',
        'secondary_background_color': '#f0f2f6',
        'text_color': '#262730'
    }
}

# ========================
# VISUALIZATION CONFIG
# ========================

PLOT_CONFIG = {
    'figure_size': (10, 6),
    'dpi': 100,
    'color_palette': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
    'style': 'whitegrid',
    'font_scale': 1.1
}

# ========================
# HELPER FUNCTIONS
# ========================

def get_model_path(model_name: str) -> str:
    """Get the full path for a model file."""
    return os.path.join(MODEL_PATHS['base_dir'], MODEL_PATHS.get(model_name, model_name))

def get_data_path(data_type: str) -> str:
    """Get the full path for a data file."""
    return DATA_PATHS.get(data_type, os.path.join('data', data_type))

def is_suspicious_domain(domain: str) -> bool:
    """Check if an email domain is suspicious."""
    return domain.lower() in SUSPICIOUS_EMAIL_DOMAINS

def is_trusted_domain(domain: str) -> bool:
    """Check if an email domain is trusted."""
    return domain.lower() in TRUSTED_EMAIL_DOMAINS

def get_risk_level(confidence: float) -> str:
    """Determine risk level based on confidence score."""
    if confidence >= CONFIDENCE_THRESHOLDS['high_risk']:
        return 'High'
    elif confidence >= CONFIDENCE_THRESHOLDS['medium_risk']:
        return 'Medium'
    elif confidence >= CONFIDENCE_THRESHOLDS['low_risk']:
        return 'Low'
    else:
        return 'Very Low'

# ========================
# EXPORT ALL CONSTANTS
# ========================

__all__ = [
    'SUSPICIOUS_KEYWORDS',
    'SUSPICIOUS_EMAIL_DOMAINS',
    'TRUSTED_EMAIL_DOMAINS',
    'MODEL_PARAMS',
    'FEATURE_SELECTION_CONFIG',
    'CONFIDENCE_THRESHOLDS',
    'PREDICTION_THRESHOLDS',
    'SCRAPING_CONFIG',
    'LINKEDIN_URL_PATTERNS',
    'DEFAULT_HEADERS',
    'TEXT_PROCESSING_CONFIG',
    'NGRAM_CONFIG',
    'FEATURE_CONFIG',
    'REQUIRED_JOB_SECTIONS',
    'CACHE_CONFIG',
    'MODEL_PATHS',
    'EVALUATION_METRICS',
    'CV_CONFIG',
    'LOGGING_CONFIG',
    'DATA_PATHS',
    'STREAMLIT_CONFIG',
    'PLOT_CONFIG',
    'get_model_path',
    'get_data_path',
    'is_suspicious_domain',
    'is_trusted_domain',
    'get_risk_level'
]