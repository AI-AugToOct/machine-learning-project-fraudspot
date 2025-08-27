# ML-OPS Engineer: Data Pipeline Implementation Guide

## Your Responsibility

You are the **Data Pipeline Specialist** responsible for implementing data loading, preprocessing, and exploratory data analysis components of the ML pipeline. Note: Support utility functions (text processing, validation, logging) are handled by the Orchestration Engineer.

## Files to Implement

### 1. `src/data/data_loader.py`

**Primary Focus: Data loading and management**

#### Key Functions to Implement:

- `load_csv_data()` - Load CSV datasets efficiently
- `load_fraud_dataset()` - Load fraud training data
- `load_legitimate_dataset()` - Load legitimate job data
- `combine_datasets()` - Merge and balance datasets
- `split_data()` - Train/validation/test splitting
- `save_processed_data()` - Save preprocessed datasets
- `get_data_info()` - Dataset information and statistics

### 2. `src/data/preprocessing.py`

**Primary Focus: Data cleaning and preprocessing**

#### Key Functions to Implement:

- `clean_job_data()` - Clean and standardize job data
- `handle_missing_values()` - Missing value imputation
- `normalize_text_fields()` - Text field standardization
- `encode_categorical()` - Categorical variable encoding
- `remove_duplicates()` - Duplicate detection and removal
- `validate_data_types()` - Data type validation and conversion
- `create_preprocessing_pipeline()` - Scikit-learn preprocessing pipeline

### 3. `src/data/eda.py`

**Primary Focus: Exploratory data analysis**

#### Key Functions to Implement:

- `generate_data_summary()` - Comprehensive data summary
- `analyze_class_distribution()` - Target variable analysis
- `create_correlation_matrix()` - Feature correlation analysis
- `plot_feature_distributions()` - Distribution visualizations
- `analyze_text_features()` - Text-specific analysis
- `identify_outliers()` - Outlier detection and analysis
- `generate_eda_report()` - Comprehensive EDA report

#### Required Libraries:

```python
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import plotly.express as px, plotly.graph_objects as go
```

## Implementation Approach

### Step 1: Data Loading (Foundation)

Start with `data_loader.py` to establish data pipeline:

1. **CSV Data Loading**

   ```python
   def load_csv_data(file_path: str) -> pd.DataFrame:
       # Load CSV with proper encoding
       # Handle different formats and separators
       # Basic validation and error handling
   ```

2. **Dataset Combination**

   ```python
   def combine_datasets(fraud_df: pd.DataFrame, legitimate_df: pd.DataFrame) -> pd.DataFrame:
       # Add labels and combine datasets
       # Balance classes if needed
       # Shuffle data for training
   ```

### Step 2: Data Preprocessing (Critical for ML Pipeline)

Implement `preprocessing.py` for data preparation:

1. **Data Cleaning**

   ```python
   def clean_job_data(df: pd.DataFrame) -> pd.DataFrame:
       # Standardize text fields
       # Handle encoding issues
       # Remove malformed entries
   ```

2. **Missing Value Handling**

   ```python
   def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
       # Identify patterns in missing data
       # Apply appropriate imputation strategies
       # Document data quality issues
   ```

### Step 3: Exploratory Data Analysis

Implement `eda.py` for insights:

1. **Data Summary**
   ```python
   def generate_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
       # Basic statistics and info
       # Missing value analysis
       # Class distribution analysis
   ```

## Key Configuration References

Use these from `src/config.py`:

- `LOGGING_CONFIG` - Logging parameters
- `SUSPICIOUS_EMAIL_DOMAINS` - Email domain blacklist
- `TEXT_PROCESSING_CONFIG` - Text processing settings

## Text Preprocessing Patterns

### HTML Cleaning Template:

```python
def remove_html_tags(text: str) -> str:
    import re
    from html import unescape
  
    # Remove HTML tags
    clean = re.sub('<[^<]+?>', '', text)
    # Convert HTML entities
    clean = unescape(clean)
    return clean
```

### Stopword Removal Template:

```python
def remove_stopwords(text: str, language: str = 'english') -> str:
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
  
    stop_words = set(stopwords.words(language))
    tokens = word_tokenize(text.lower())
    filtered = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered)
```

## Data Validation Patterns

### URL Validation Template:

```python
def validate_url(url: str) -> bool:
    from urllib.parse import urlparse
  
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False
```

### Email Validation Template:

```python
def validate_email(email: str) -> Dict[str, Any]:
    import re
  
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    is_valid = bool(re.match(pattern, email))
  
    if is_valid:
        domain = email.split('@')[1].lower()
        is_suspicious = domain in SUSPICIOUS_EMAIL_DOMAINS
  
    return {
        'is_valid': is_valid,
        'domain': domain if is_valid else '',
        'is_suspicious': is_suspicious if is_valid else True
    }
```

## Logging Configuration Pattern

### Basic Logging Setup:

```python
def setup_logging(config: Optional[Dict[str, Any]] = None) -> None:
    import logging.handlers
  
    config = config or LOGGING_CONFIG
  
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, config['level']))
  
    # File handler with rotation
    if config.get('log_file'):
        os.makedirs(os.path.dirname(config['log_file']), exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            config['log_file'],
            maxBytes=config.get('max_bytes', 10485760),
            backupCount=config.get('backup_count', 5)
        )
        file_handler.setFormatter(logging.Formatter(config['format']))
        logger.addHandler(file_handler)
```

## Security Considerations

### Input Sanitization:

```python
def sanitize_input(data: Any) -> Any:
    if isinstance(data, str):
        # Remove dangerous characters
        data = re.sub(r'[<>"\';\\]', '', data)
        # Limit length
        data = data[:1000]
    elif isinstance(data, dict):
        return {k: sanitize_input(v) for k, v in data.items()}
  
    return data
```

## Testing Strategy

1. **Unit Testing**: Test each function with various inputs
2. **Security Testing**: Test with malicious inputs
3. **Performance Testing**: Ensure functions are fast enough
4. **Integration Testing**: Verify integration with other modules

## Error Handling Template

```python
def your_function(params) -> ReturnType:
    if not params:
        logger.warning("Empty parameters provided")
        return default_value
  
    try:
        result = process_params(params)
        logger.debug(f"Successfully processed {type(params)}")
        return result
    except Exception as e:
        logger.error(f"Error in your_function: {str(e)}")
        return safe_default_value
```

## Expected Deliverables

1. **Complete text_preprocessing.py** with all NLP utilities
2. **Complete data_validation.py** with security validation
3. **Complete logging_config.py** with production-ready logging
4. **Integration testing** with other team members' code
5. **Security validation** of all input handling

## Integration Points

Your support functions are used by:

- **Feature Engineer**: Text preprocessing for feature extraction
- **ML Engineer**: Data validation for model training
- **All Engineers**: Logging and error handling
- **Streamlit App**: Input validation and sanitization

## Success Criteria

- All stub functions replaced with robust implementations
- No security vulnerabilities in input handling
- Proper logging throughout the application
- Text preprocessing supports multiple languages
- Data validation catches malformed and malicious input

## Performance Requirements

- **Text Preprocessing**: <50ms for typical job posting
- **Data Validation**: <10ms per validation check
- **Logging**: Minimal performance impact (<1ms per log entry)
- **Memory Usage**: <100MB for preprocessing operations

## Advanced Features to Consider

1. **Language Detection**: Support for multiple languages
2. **Advanced Sanitization**: XSS and SQL injection prevention
3. **Structured Logging**: JSON-formatted logs for analysis
4. **Performance Monitoring**: Track function execution times
5. **Cache Integration**: Cache expensive operations

## Quality Standards

1. **Robustness**: Handle all edge cases gracefully
2. **Security**: Prevent all common attack vectors
3. **Performance**: Meet speed requirements
4. **Maintainability**: Clear, well-documented code
5. **Testing**: Comprehensive test coverage

## Tips for Success

1. **Security First**: Always validate and sanitize inputs
2. **Handle Edge Cases**: Empty strings, None values, encoding issues
3. **Use Standard Libraries**: NLTK, re, logging, urllib
4. **Performance Matters**: Your functions are called frequently
5. **Test Thoroughly**: Security and edge case testing is critical

Remember: Your support functions are the foundation that everything else builds on (No Pressure). Robust, secure, and fast implementations are essential for system reliability!
