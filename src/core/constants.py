"""
Core Constants - SINGLE SOURCE OF TRUTH
This module contains ALL constants, keywords, patterns, thresholds, and configurations.
NO OTHER MODULE should define these constants.

Consolidated from:
- Original constants.py
- config.py (merged and deleted)
- All scattered constants throughout codebase

Version: 3.0.0 - Complete DRY Consolidation
"""

import os
from typing import Any, Dict, List

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class EnvironmentConstants:
    """Single source for environment configuration"""
    
    ENVIRONMENT = os.getenv('ENVIRONMENT', 'local')  # 'local' or 'cloud'
    USE_BRIGHT_DATA = os.getenv('USE_BRIGHT_DATA', 'true').lower() == 'true'
    BD_API_KEY = os.getenv('BD_API_KEY', '')
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    @staticmethod
    def get_scraping_mode():
        """Determine scraping mode based on environment"""
        return 'bright_data'  # Always use Bright Data
    
    @staticmethod
    def is_cloud_environment():
        """Check if running in cloud environment"""
        return EnvironmentConstants.ENVIRONMENT == 'cloud' or EnvironmentConstants.USE_BRIGHT_DATA


class BrightDataConstants:
    """Single source for Bright Data configuration"""
    
    CONFIG = {
        'api_key': EnvironmentConstants.BD_API_KEY.strip('"') if EnvironmentConstants.BD_API_KEY else '',
        'base_url': 'https://api.brightdata.com',
        'timeout': 60,  # 60 seconds for initial request
        'max_retries': 3,
        'rate_limit_delay': 1.0,  # seconds between requests
        
        # Correct endpoint for data scraping
        'trigger_endpoint': 'https://api.brightdata.com/datasets/v3/scrape',
        
        # Correct dataset IDs from official Bright Data GitHub examples
        'dataset_ids': {
            'profiles': 'gd_l1viktl72bvl7bjuj0',    # LinkedIn Profiles
            'companies': 'gd_l1vikfnt1wgvvqz95w',   # LinkedIn Companies  
            'jobs': 'gd_lpfll7v5hcqtkxl6l',         # LinkedIn Jobs
            'posts': 'gd_lyy3tktm25m4avu764'       # LinkedIn Posts
        },
        
        # API parameter keys
        'api_params': {
            'DATASET_ID': 'dataset_id',
            'FORMAT': 'format',
            'URL': 'url'
        },
        
        # Request configuration
        'request_config': {
            'format': 'json',
            'include_additional_data': True,
            'max_pages': 1,
            'country': 'US',
            'language': 'en'
        },
        
        # Comprehensive field extraction for fraud detection
        'extraction_fields': {
            'job_fields': [
                'job_title', 'job_description', 'job_functions', 'job_industries',
                'location', 'employment_type', 'experience_level', 'salary_range',
                'application_deadline', 'posted_date', 'application_count',
                'job_id', 'job_url', 'easy_apply', 'remote_allowed'
            ],
            'company_fields': [
                'company_name', 'company_id', 'company_url', 'company_logo',
                'company_size', 'company_type', 'company_industry', 'company_headquarters',
                'company_founded', 'company_employees', 'company_followers', 'company_funding',
                'company_website', 'company_description', 'company_specialties'
            ],
            'poster_fields': [
                'poster_name', 'poster_title', 'poster_company', 'poster_location',
                'poster_profile_url', 'poster_connections', 'poster_followers',
                'poster_experience', 'poster_education', 'poster_skills',
                'poster_certifications', 'poster_languages', 'poster_recommendations',
                'poster_endorsements', 'poster_activity', 'poster_verified',
                'poster_premium', 'poster_profile_photo'
            ],
            'verification_fields': [
                'hiring_team', 'company_employees_on_linkedin', 'similar_companies',
                'job_posting_company_match', 'poster_company_relationship',
                'company_recent_hires', 'company_growth_rate', 'job_repost_count'
            ]
        }
    }
    
    @staticmethod
    def get_config():
        """Get Bright Data configuration"""
        return BrightDataConstants.CONFIG


class FraudKeywords:
    """Single source for all fraud detection keywords (multilingual)"""
    
    # CONSOLIDATED English suspicious keywords (from config.py + constants.py)
    ENGLISH_SUSPICIOUS = [
        # Communication channels (non-professional)
        'whatsapp', 'telegram', 'wire transfer', 'western union', 
        'moneygram', 'personal email', 'gmail', 'yahoo', 'hotmail',
        
        # Payment red flags
        'upfront fee', 'registration fee', 'processing fee', 'administration fee',
        'deposit required', 'cash only', 'payment required', 'advance payment',
        'wire money', 'send money', 'transfer funds',
        
        # Work-from-home scams
        'work from home', 'no experience needed', 'easy money', 'quick money',
        'unlimited income', 'be your own boss', 'financial freedom', 
        'earn while you sleep', 'passive income', 'make money fast',
        
        # MLM/Pyramid scheme indicators
        'network marketing', 'multi-level marketing', 'mlm', 'pyramid',
        'recruit others', 'build your team', 'downline', 'upline',
        
        # Fake credentials
        'work visa guaranteed', 'no background check', 'no references needed',
        'fake id accepted', 'undocumented workers welcome',
        
        # Vague/unrealistic promises
        'no skills required', 'anyone can do', 'guaranteed income', 
        '100% success rate', 'risk-free', 'guaranteed job', 'instant approval'
    ]
    
    # Arabic suspicious keywords
    ARABIC_SUSPICIOUS = [
        'فوري', 'عاجل', 'بدون خبرة', 'دخل مضمون', 'مال سهل', 'ربح سريع',
        'العمل من المنزل', 'فرصة ذهبية', 'استثمار مطلوب', 'رسوم التدريب',
        'رسوم التسجيل', 'رسوم المعالجة', 'واتساب', 'تليجرام'
    ]
    
    # English urgency keywords
    ENGLISH_URGENCY = [
        'urgent', 'immediate', 'asap', 'hurry', 'quick', 'fast', 'now',
        'limited time', 'act now', 'expires soon', 'deadline', 'rush',
        'critical', 'emergency', 'don\'t wait', 'limited spots', 
        'first come first served'
    ]
    
    # Arabic urgency keywords  
    ARABIC_URGENCY = [
        'فوري', 'عاجل', 'سريع', 'الآن', 'مستعجل', 'محدود الوقت',
        'انتهت الصلاحية', 'موعد نهائي', 'طارئ', 'حرج'
    ]
    
    # English quality/professional keywords
    ENGLISH_QUALITY = [
        'benefits', 'insurance', 'pension', '401k', 'vacation', 'career',
        'growth', 'development', 'team', 'professional', 'experience',
        'skills', 'qualifications', 'responsibilities', 'requirements'
    ]
    
    # Arabic quality/professional keywords
    ARABIC_QUALITY = [
        'مزايا', 'تأمين', 'معاش', 'إجازة', 'مهنة', 'نمو', 'تطوير',
        'فريق', 'مهني', 'خبرة', 'مهارات', 'مؤهلات', 'مسؤوليات',
        'متطلبات', 'وظيفة', 'عمل', 'موظف'
    ]
    
    # Professional terms (English + Arabic)
    PROFESSIONAL_TERMS = [
        # English
        'experience', 'skills', 'qualifications', 'responsibilities',
        'requirements', 'benefits', 'team', 'company', 'position',
        # Arabic  
        'خبرة', 'مهارات', 'مؤهلات', 'مسؤوليات', 'متطلبات',
        'مزايا', 'فريق', 'شركة', 'منصب', 'وظيفة', 'عمل', 'موظف'
    ]
    
    # Unprofessional terms (English + Arabic)
    UNPROFESSIONAL_TERMS = [
        # English
        'easy money', 'quick cash', 'work from home', 'no experience',
        'urgent', 'asap', 'immediate', 'guaranteed income',
        # Arabic
        'مال سهل', 'ربح سريع', 'عمل من المنزل', 'بلا خبرة',
        'عاجل', 'فوري', 'دخل مضمون', 'اتصل الآن'
    ]
    
    # Unprofessional contact patterns
    UNPROFESSIONAL_CONTACTS = [
        # English
        'whatsapp', 'telegram', 'personal email', 'gmail.com',
        'yahoo.com', 'hotmail.com', 'call now', 'text me',
        # Arabic  
        'واتساب', 'واتس اب', 'تليجرام', 'ايميل شخصي',
        'اتصل الآن', 'راسلني', 'جيميل', 'ياهو'
    ]


class DataConstants:
    """Single source for data processing constants"""
    
    # Essential job posting fields
    ESSENTIAL_FIELDS = ['job_title', 'company_name', 'job_description', 'location']
    
    # Additional important fields
    ADDITIONAL_FIELDS = ['salary_info', 'requirements', 'contact_info', 'experience_level']
    
    # Text columns for processing
    TEXT_COLUMNS = [
        'job_title', 'job_description', 'requirements', 'benefits', 
        'company_name', 'company_profile', 'industry', 'location', 'salary_info'
    ]
    
    # Binary indicator columns
    BINARY_COLUMNS = [
        'has_company_logo', 'has_questions', 'fraudulent',
        'poster_verified', 'poster_experience', 'poster_photo', 'poster_active'
    ]
    
    # Encoded categorical columns
    ENCODED_COLUMNS = [
        'language', 'experience_level_encoded', 'education_level_encoded',
        'employment_type_encoded'
    ]
    
    # Score/computed columns (0.0 to 1.0 range)
    SCORE_COLUMNS = [
        'description_length_score', 'professional_language_score',
        'urgency_language_score', 'contact_professionalism_score',
        'verification_score', 'content_quality_score', 'legitimacy_score'
    ]
    
    # Required job posting sections
    REQUIRED_SECTIONS = [
        'responsibilities', 'requirements', 'qualifications', 
        'benefits', 'description', 'duties'
    ]
    
    # Section keyword mappings
    SECTION_KEYWORDS = {
        'responsibilities': ['responsibilities', 'duties', 'role', 'tasks', 'job duties'],
        'requirements': ['requirements', 'qualifications', 'skills', 'experience', 'must have'],
        'qualifications': ['qualifications', 'requirements', 'skills', 'preferred', 'desired'],
        'benefits': ['benefits', 'perks', 'compensation', 'insurance', 'vacation'],
        'description': ['description', 'overview', 'about', 'summary', 'position'],
        'duties': ['duties', 'responsibilities', 'tasks', 'job functions', 'daily tasks']
    }
    
    # Data paths
    DATA_PATHS = {
        'raw_data': 'data/raw',
        'processed_data': 'data/processed',
        'training_data': 'data/processed/training_data.csv',
        'test_data': 'data/processed/test_data.csv',
        'validation_data': 'data/processed/validation_data.csv'
    }


class ModelConstants:
    """Single source for all ML model constants"""
    
    # ML features only (numerical columns for training - 27 total, NO job_id)
    ML_FEATURE_COLUMNS = [
        # Binary indicators
        'has_company_logo', 'has_questions',
        # Poster verification columns (CRITICAL PREDICTORS)
        'poster_verified', 'poster_experience', 'poster_photo', 'poster_active', 'poster_score',
        'is_highly_verified', 'is_unverified', 'verification_ratio',
        # Language and encoding
        'language', 'experience_level_encoded', 'education_level_encoded', 
        'employment_type_encoded',
        # Computed score columns (calculated from text)
        'description_length_score', 'title_word_count', 'professional_language_score',
        'urgency_language_score', 'contact_professionalism_score',
        'verification_score', 'content_quality_score', 'legitimacy_score',
        # Company verification features (NEW - 5 additional features)
        'company_followers_score', 'company_employees_score', 'company_founded_score', 
        'network_quality_score', 'company_legitimacy_score'
    ]
    
    # Required feature columns (33 total - includes ML features + text for processing + target)
    # Composed from ML_FEATURE_COLUMNS + DataConstants.TEXT_COLUMNS + raw categorical + target
    REQUIRED_FEATURE_COLUMNS = (ML_FEATURE_COLUMNS + 
                                DataConstants.TEXT_COLUMNS + 
                                ['employment_type', 'experience_level', 'education_level'] +  # Raw categorical for encoding
                                ['fraudulent'])  # Target
    
    # Verification score weights (based on analysis - these are 100% accuracy predictors)
    VERIFICATION_WEIGHTS = {
        'poster_verified': 0.4,      # Highest weight - LinkedIn verification
        'poster_experience': 0.3,    # Experience at posting company
        'poster_photo': 0.2,         # Profile photo presence
        'poster_active': 0.1         # Recent activity indicators
    }
    
    # Risk level thresholds (based on verification analysis)
    RISK_THRESHOLDS = {
        'very_low': 0.15,    # poster_score >= 3 (highly verified)
        'low': 0.30,         # poster_score == 2 (moderately verified) 
        'high': 0.75,        # poster_score == 1 (low verification)
        'very_high': 0.95    # poster_score == 0 (no verification)
    }
    
    # Model parameters (consolidated from config.py)
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
    
    # Default model configuration
    DEFAULT_MODEL_CONFIG = {
        'model_type': 'random_forest',
        'test_size': 0.2,
        'validation_size': 0.2,
        'random_state': 42,
        'balance_method': 'smote',
        'scaling_method': 'standard'
    }
    
    # Feature selection and preprocessing
    FEATURE_SELECTION_CONFIG = {
        'method': 'selectkbest',  # 'selectkbest', 'rfe', 'lasso'
        'k_features': 50,
        'score_func': 'f_classif'
    }
    
    FEATURE_IMPORTANCE_THRESHOLD = 0.01
    MIN_FEATURE_CORRELATION = 0.05



class ScrapingConstants:
    """Single source for scraping-related constants"""
    
    # LinkedIn URL patterns
    LINKEDIN_URL_PATTERNS = [
        r'https://www\.linkedin\.com/jobs/view/\d+',
        r'https://linkedin\.com/jobs/view/\d+',
        r'https://www\.linkedin\.com/jobs/search.*currentJobId=\d+',
        r'.*linkedin\.com.*jobs.*view.*\d+'
    ]
    
    # Suspicious email domains
    SUSPICIOUS_EMAIL_DOMAINS = [
        'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com',
        'aol.com', 'mail.com', 'protonmail.com', 'tempmail.com'
    ]
    
    # Trusted email domains
    TRUSTED_EMAIL_DOMAINS = [
        'linkedin.com', 'indeed.com', 'monster.com', 'glassdoor.com',
        'microsoft.com', 'google.com', 'amazon.com', 'apple.com'
    ]
    
    # Messaging app patterns
    MESSAGING_PATTERNS = {
        'whatsapp': r'\bwhatsapp\b',
        'telegram': r'\btelegram\b', 
        'viber': r'\bviber\b',
        'wechat': r'\bwechat\b'
    }
    
    # Phone number patterns
    PHONE_PATTERNS = [
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # US format
        r'\+\d{1,3}[-.s]?\d{3,4}[-.s]?\d{3,4}[-.s]?\d{3,4}',  # International
        r'\(\d{3}\)s?\d{3}[-.]?\d{4}'  # (XXX) XXX-XXXX
    ]
    
    # Scraping configuration
    SCRAPING_CONFIG = {
        'timeout': 30,              # Request timeout in seconds
        'rate_limit_delay': 2,      # Delay between requests
        'max_retries': 3,           # Maximum retry attempts
        'retry_delay': 5,           # Delay between retries
        'user_agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    # Headers for web scraping
    DEFAULT_HEADERS = {
        'User-Agent': SCRAPING_CONFIG['user_agent'],
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }


class TextProcessingConstants:
    """Single source for text processing configuration"""
    
    CONFIG = {
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


class UIConstants:
    """Single source for UI-related constants"""
    
    # Risk level colors
    RISK_COLORS = {
        'VERY LOW': 'green',
        'LOW': 'lightgreen', 
        'MODERATE': 'yellow',
        'HIGH': 'orange',
        'VERY HIGH': 'red',
        'UNKNOWN': 'gray'
    }
    
    # Confidence thresholds for UI display
    CONFIDENCE_THRESHOLDS = {
        'high': 0.8,
        'medium': 0.6,
        'low': 0.4
    }
    
    # Display formatting
    SCORE_DISPLAY_FORMAT = {
        'decimal_places': 3,
        'percentage_format': True,
        'show_confidence': True
    }
    
    # Streamlit configuration
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
    
    # Plot configuration
    PLOT_CONFIG = {
        'figure_size': (10, 6),
        'dpi': 100,
        'color_palette': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
        'style': 'whitegrid',
        'font_scale': 1.1
    }


class UtilityConstants:
    """Single source for utility configurations"""
    
    # Cache configuration
    CACHE_CONFIG = {
        'cache_dir': 'data/cache',
        'cache_expiry_days': 7,     # Cache results for 7 days
        'max_cache_size_mb': 100,   # Maximum cache size in MB
        'enable_cache': True        # Whether to enable caching
    }
    
    # Model paths
    MODEL_PATHS = {
        'base_dir': 'data/models',
        'trained_model': 'fraud_detector_model.pkl',
        'vectorizer': 'tfidf_vectorizer.pkl',
        'feature_selector': 'feature_selector.pkl',
        'label_encoder': 'label_encoder.pkl',
        'scaler': 'feature_scaler.pkl'
    }
    
    # Evaluation metrics
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
    
    # Logging configuration (simplified - no duplication with logging_config.py)
    LOGGING_CONFIG = {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'log_file': 'logs/fraud_detector.log',
        'max_bytes': 10485760,  # 10MB
        'backup_count': 5
    }


class ThresholdConstants:
    """Single source for all thresholds and limits"""
    
    # Confidence thresholds for predictions
    CONFIDENCE_THRESHOLDS = {
        'high_risk': 0.8,      # 80%+ confidence of fraud
        'medium_risk': 0.5,    # 50-80% confidence of fraud
        'low_risk': 0.3        # 30-50% confidence of fraud
    }
    
    # Prediction thresholds
    PREDICTION_THRESHOLDS = {
        'fraud_threshold': 0.5,    # Threshold for binary fraud classification
        'confidence_threshold': 0.3  # Minimum confidence for predictions
    }


class FeatureWeightConstants:
    """Single source for feature importance weights (from config.py)"""
    
    # Default feature weights - higher values indicate more importance
    DEFAULT_FEATURE_WEIGHTS = {
        # Primary Dataset Features - Verification Features (Most Important)
        'poster_verified': 1.0,          # Job poster verification status (0/1)
        'poster_experience': 0.9,      # Job poster experience level
        'poster_photo': 0.7,           # Profile photo presence (0/1)
        'poster_active': 0.8,          # Recent activity indicator (0/1)
        
        # Text Features - Raw text columns (High Importance for ML processing)
        'job_title': 0.8,          # Job title text
        'job_description': 0.9,    # Job description text
        'job_requirements': 0.7,   # Job requirements text
        'company_name': 0.6,       # Company name
        
        # Company Information (Medium-High Importance)
        'company_type': 0.6,       # Company type (private, public, etc.)
        'company_size': 0.5,       # Company size category
        'industry': 0.5,           # Industry sector
        
        # Job Details (Medium Importance)
        'education_level': 0.6,    # Education requirements
        'employment_type': 0.4,    # Employment type (full-time, part-time, etc.)
        'experience_level': 0.7,   # Experience requirements
        'salary': 0.5,             # Salary information
        
        # Location Features (Medium Importance)
        'region': 0.4,             # Region/province
        'city': 0.3,               # City
        
        # Additional computed features that may be generated during feature engineering
        'suspicious_keyword_count': 0.9,    # Count of suspicious keywords
        'urgency_indicators': 0.8,          # Urgency-related language
        'grammar_score': 0.7,               # Grammar quality score
        'readability_score': 0.6,           # Text readability
        'red_flags_count': 0.9,             # Total red flags detected
        'quality_score': 0.8,               # Overall posting quality
        'completeness_score': 0.7,          # Information completeness
        'text_length': 0.4,                 # Length of job description
        'contact_patterns': 0.8,            # Contact information patterns
        'sentiment_score': 0.5,             # Sentiment analysis of text
        
        # Profile availability features
        'profile_private': 0.15,            # Very low weight
        'company_legitimacy_score': 0.9,    # High weight - company verification
        'job_quality_indicators': 0.85,     # High weight - job content quality
        'posting_professionalism': 0.9,     # High weight - professional posts
        'description_completeness': 0.8,    # Medium-high weight
    }
    
    # Feature categories with their base weights
    FEATURE_CATEGORIES = {
        'verification': {
            'weight_multiplier': 1.2,
            'features': ['poster_verified', 'poster_experience', 'poster_photo', 'poster_active']
        },
        'text_content': {
            'weight_multiplier': 1.1,
            'features': ['job_title', 'job_description', 'job_requirements', 'company_name']
        },
        'computed_text': {
            'weight_multiplier': 1.0,
            'features': ['suspicious_keyword_count', 'urgency_indicators', 'grammar_score', 'readability_score']
        },
        'company_info': {
            'weight_multiplier': 0.95,
            'features': ['company_type', 'company_size', 'industry']
        },
        'job_details': {
            'weight_multiplier': 0.9,
            'features': ['education_level', 'employment_type', 'experience_level', 'salary']
        },
        'location': {
            'weight_multiplier': 0.8,
            'features': ['region', 'city']
        },
        'computed_structural': {
            'weight_multiplier': 1.05,
            'features': ['red_flags_count', 'quality_score', 'completeness_score', 'contact_patterns']
        }
    }
    
    # Risk-based weight adjustments
    RISK_ADJUSTMENTS = {
        'high_risk_features': {
            'multiplier': 1.5,
            'features': ['poster_verified', 'suspicious_keyword_count', 'red_flags_count', 'poster_experience']
        },
        'medium_risk_features': {
            'multiplier': 1.2,
            'features': ['urgency_indicators', 'grammar_score', 'quality_score', 'poster_photo', 'job_description']
        },
        'low_risk_features': {
            'multiplier': 0.8,
            'features': ['region', 'city']
        }
    }


# HELPER FUNCTIONS (moved from config.py)

def get_model_path(model_name: str) -> str:
    """Get the full path for a model file."""
    import os
    return os.path.join(UtilityConstants.MODEL_PATHS['base_dir'], 
                       UtilityConstants.MODEL_PATHS.get(model_name, model_name))


def get_data_path(data_type: str) -> str:
    """Get the full path for a data file."""
    import os
    return DataConstants.DATA_PATHS.get(data_type, os.path.join('data', data_type))


def is_suspicious_domain(domain: str) -> bool:
    """Check if an email domain is suspicious."""
    return domain.lower() in ScrapingConstants.SUSPICIOUS_EMAIL_DOMAINS


def is_trusted_domain(domain: str) -> bool:
    """Check if an email domain is trusted."""
    return domain.lower() in ScrapingConstants.TRUSTED_EMAIL_DOMAINS


def get_risk_level(confidence: float) -> str:
    """Determine risk level based on confidence score."""
    if confidence >= ThresholdConstants.CONFIDENCE_THRESHOLDS['high_risk']:
        return 'High'
    elif confidence >= ThresholdConstants.CONFIDENCE_THRESHOLDS['medium_risk']:
        return 'Medium'
    elif confidence >= ThresholdConstants.CONFIDENCE_THRESHOLDS['low_risk']:
        return 'Low'
    else:
        return 'Very Low'


def get_feature_weights(weight_strategy: str = 'default') -> Dict[str, float]:
    """
    Get feature weights based on the specified strategy.
    
    Args:
        weight_strategy (str): Weight strategy ('default', 'conservative', 'aggressive', 'balanced')
        
    Returns:
        Dict[str, float]: Dictionary of feature weights
    """
    try:
        base_weights = FeatureWeightConstants.DEFAULT_FEATURE_WEIGHTS.copy()
        
        if weight_strategy == 'conservative':
            # Conservative approach - reduce all weights slightly, emphasize verification
            adjusted_weights = {}
            for feature, weight in base_weights.items():
                if feature in FeatureWeightConstants.FEATURE_CATEGORIES['verification']['features']:
                    adjusted_weights[feature] = weight * 1.3  # Boost verification features
                else:
                    adjusted_weights[feature] = weight * 0.8  # Reduce other features
            return adjusted_weights
            
        elif weight_strategy == 'aggressive':
            # Aggressive approach - boost high-risk indicators significantly
            adjusted_weights = {}
            for feature, weight in base_weights.items():
                if feature in FeatureWeightConstants.RISK_ADJUSTMENTS['high_risk_features']['features']:
                    adjusted_weights[feature] = weight * 1.6  # Strong boost for high-risk
                elif feature in FeatureWeightConstants.RISK_ADJUSTMENTS['medium_risk_features']['features']:
                    adjusted_weights[feature] = weight * 1.3  # Medium boost
                else:
                    adjusted_weights[feature] = weight * 0.9  # Slight reduction for others
            return adjusted_weights
            
        elif weight_strategy == 'balanced':
            # Balanced approach - apply category-based adjustments
            adjusted_weights = {}
            for feature, weight in base_weights.items():
                category_multiplier = 1.0
                for category, config in FeatureWeightConstants.FEATURE_CATEGORIES.items():
                    if feature in config['features']:
                        category_multiplier = config['weight_multiplier']
                        break
                adjusted_weights[feature] = weight * category_multiplier
            return adjusted_weights
            
        else:  # default strategy
            return base_weights
            
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error getting feature weights: {str(e)}")
        return FeatureWeightConstants.DEFAULT_FEATURE_WEIGHTS


def get_feature_importance_ranking(weights: Dict[str, float] = None) -> List[tuple]:
    """
    Get features ranked by their importance (weight).
    
    Args:
        weights (Dict[str, float], optional): Feature weights dictionary
        
    Returns:
        List[tuple]: List of (feature_name, weight) tuples sorted by weight (descending)
    """
    try:
        import logging
        logger = logging.getLogger(__name__)
        
        if weights is None:
            weights = get_feature_weights()
        
        # Sort features by weight in descending order
        ranked_features = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        
        logger.info(f"Generated feature importance ranking for {len(ranked_features)} features")
        return ranked_features
        
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error ranking feature importance: {str(e)}")
        return []


def get_bright_data_config():
    """Get Bright Data configuration"""
    return BrightDataConstants.CONFIG


# Export all constants for easy importing
__all__ = [
    # Environment and Configuration
    'EnvironmentConstants',
    'BrightDataConstants', 
    
    # Core Business Constants
    'FraudKeywords',
    'ModelConstants', 
    'DataConstants',
    'ScrapingConstants',
    
    # Processing and UI
    'TextProcessingConstants',
    'UIConstants',
    'UtilityConstants',
    'ThresholdConstants',
    'FeatureWeightConstants',
    
    # Helper Functions
    'get_model_path',
    'get_data_path',
    'is_suspicious_domain',
    'is_trusted_domain',
    'get_risk_level',
    'get_feature_weights',
    'get_feature_importance_ranking',
    'get_bright_data_config'
]