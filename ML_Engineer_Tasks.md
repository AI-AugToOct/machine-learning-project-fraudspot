# ML Engineer: Machine Learning Implementation Guide

## Your Responsibility

You are the **Model Training and Inference Specialist** responsible for implementing the complete machine learning pipeline for fraud detection.

## Files to Implement

### 1. `src/models/train_model.py`

**Primary Focus: Model training pipeline**

#### Key Functions to Implement:

- `load_training_data()` - Load and prepare training datasets
- `preprocess_training_data()` - Data preprocessing and feature engineering
- `train_model()` - Train fraud detection models
- `evaluate_model()` - Comprehensive model evaluation
- `perform_cross_validation()` - Cross-validation for robustness
- `hyperparameter_tuning()` - Optimize model parameters
- `save_model()` - Persist trained models
- `compare_models()` - Compare different algorithms
- `main_training_pipeline()` - Orchestrate complete training

### 2. `src/models/predict.py`

**Primary Focus: Real-time inference system**

#### Key Functions to Implement:

- `load_model()` - Load trained models for inference
- `predict_fraud()` - Generate fraud predictions
- `calculate_confidence_score()` - Assess prediction confidence
- `determine_risk_level()` - Map confidence to risk levels
- `generate_explanation()` - Human-readable explanations
- `extract_top_features()` - Feature importance for interpretability

#### Required Libraries:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
import joblib
```

## Implementation Approach

### Step 1: Prediction System (Start Here)

Begin with `predict.py` since it's needed for the Streamlit app:

1. **Model Loading**

   ```python
   def load_model(model_path: str = None) -> Any:
       # Use joblib.load() to load pickled models
       # Handle missing files gracefully
       # Load associated preprocessors (scaler, vectorizer)
   ```
2. **Fraud Prediction**

   ```python
   def predict_fraud(model: Any, features: pd.DataFrame) -> Dict[str, Any]:
       # Use model.predict_proba() for probability scores
       # Apply threshold from PREDICTION_THRESHOLDS config
       # Return structured prediction results
   ```

### Step 2: Training Pipeline

Implement `train_model.py` for model development:

1. **Data Loading**

   ```python
   def load_training_data(data_path: str) -> Tuple[pd.DataFrame, pd.Series]:
       # Load from CSV/JSON files
       # Separate features from target
       # Handle missing values
   ```
2. **Model Training**

   ```python
   def train_model(X: pd.DataFrame, y: pd.Series, model_type: str = 'random_forest') -> Dict[str, Any]:
       # Use MODEL_PARAMS from config for hyperparameters
       # Support multiple model types
       # Return trained model and preprocessors
   ```

### Step 3: Integration and Testing

1. Test prediction functions with the Streamlit app
2. Implement training pipeline for model development
3. Validate end-to-end integration

## Key Configuration References

Use these from `src/config.py`:

- `MODEL_PARAMS` - Hyperparameters for different models
- `PREDICTION_THRESHOLDS` - Fraud classification thresholds
- `CONFIDENCE_THRESHOLDS` - Risk level mapping
- `MODEL_PATHS` - File paths for model persistence
- `EVALUATION_METRICS` - Metrics to calculate

## Model Types to Support

Implement these algorithms with parameters from config:

1. **Random Forest** (`random_forest`)
2. **XGBoost** (`xgboost`)
3. **Logistic Regression** (`logistic_regression`)
4. **SVM** (`svm`)
5. **Naive Bayes** (`naive_bayes`)

## Training Data Structure

Expected input format:

```python
# Features: All extracted features from Feature Engineer
X = pd.DataFrame({
    'suspicious_keyword_count': [2, 0, 1, ...],
    'sentiment_compound': [0.5, -0.2, 0.8, ...],
    'grammar_score': [0.9, 0.3, 0.8, ...],
    # ... more features
})

# Target: Binary fraud labels
y = pd.Series([1, 0, 1, ...])  # 1 = fraud, 0 = legitimate
```

## Prediction Output Format

Your `predict_fraud()` function should return:

```python
{
    'is_fraud': bool,              # Binary prediction
    'fraud_probability': float,    # Probability of fraud (0-1)
    'confidence': float,          # Prediction confidence (0-1)
    'risk_level': str,            # 'High', 'Medium', 'Low', 'Very Low'
    'probabilities': List[float]  # [legitimate_prob, fraud_prob]
}
```

## Model Persistence

Save models with associated objects:

```python
model_data = {
    'model': trained_model,
    'scaler': feature_scaler,
    'vectorizer': tfidf_vectorizer,
    'feature_selector': selector,
    'metadata': {
        'version': '1.0',
        'created': datetime.now(),
        'performance': evaluation_metrics
    }
}
```

## Evaluation Framework

Implement comprehensive evaluation:

```python
def evaluate_model(model, X_test, y_test):
    # Generate predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
  
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    return metrics
```

## Testing Strategy

1. **Unit Testing**: Test each function with mock data
2. **Integration Testing**: Use with real features from Feature Engineer
3. **Performance Testing**: Validate model accuracy and speed
4. **Streamlit Testing**: Ensure predictions display correctly

## Error Handling Template

```python
def your_function(params) -> ReturnType:
    try:
        # Your implementation
        result = process_data(params)
        logger.info(f"Successfully processed {len(data)} samples")
        return result
    except Exception as e:
        logger.error(f"Error in your_function: {str(e)}")
        return default_safe_value
```

## Expected Deliverables

1. **Complete predict.py** with all inference functions
2. **Complete train_model.py** with full training pipeline
3. **Trained model files** saved to data/models/
4. **Model evaluation reports** with performance metrics
5. **Integration testing** with Streamlit application

## Integration Points

Your ML functions will be used by:

- The Streamlit application for real-time fraud detection
- The training pipeline for model development and evaluation
- Feature Engineer's feature engineering for model validation

## Success Criteria

- All stub functions replaced with working implementations
- Models can be trained, saved, and loaded successfully
- Real-time predictions work in the Streamlit app
- Prediction confidence and risk levels are accurate
- Model explanations are meaningful and helpful

## Performance Targets

- **Prediction Speed**: <100ms per job posting
- **Model Accuracy**: >90% on validation set
- **Memory Usage**: <500MB for loaded model
- **File Size**: <50MB for saved model

## Advanced Features to Implement

1. **Model Ensemble**: Combine multiple models for better accuracy
2. **Online Learning**: Update models with new data
3. **Feature Importance**: Explain which features drive predictions
4. **Threshold Optimization**: Find optimal classification thresholds
5. **Model Monitoring**: Track prediction performance over time

## Tips for Success

1. **Start with Simple Models**: Get RandomForest working first
2. **Use Cross-Validation**: Ensure robust model evaluation
3. **Handle Imbalanced Data**: Consider SMOTE or class weighting
4. **Feature Scaling**: Normalize features for SVM and neural networks
5. **Model Interpretability**: Focus on explainable predictions

Remember: Your models are the core intelligence of the fraud detection system (No Pressure). Accurate, fast, and interpretable models are crucial for user trust and system success!
