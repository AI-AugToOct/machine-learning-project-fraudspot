"""
Validation Utilities Module

Essential validation functions for fraud detection system.
Implemented by Orchestration Engineer for immediate project needs.

Version: 1.0.0
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import json

logger = logging.getLogger(__name__)


def validate_dataframe_schema(df: pd.DataFrame, required_columns: List[str], 
                            optional_columns: List[str] = None) -> Dict[str, Any]:
    """
    Validate DataFrame schema against required and optional columns.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (List[str]): Required column names
        optional_columns (List[str], optional): Optional column names
        
    Returns:
        Dict[str, Any]: Validation results
    """
    validation_result = {
        'is_valid': True,
        'missing_required': [],
        'missing_optional': [],
        'extra_columns': [],
        'issues': [],
        'summary': {}
    }
    
    if df is None or df.empty:
        validation_result['is_valid'] = False
        validation_result['issues'].append('DataFrame is None or empty')
        return validation_result
    
    df_columns = set(df.columns)
    required_set = set(required_columns)
    optional_set = set(optional_columns) if optional_columns else set()
    expected_columns = required_set | optional_set
    
    # Check missing required columns
    missing_required = required_set - df_columns
    if missing_required:
        validation_result['is_valid'] = False
        validation_result['missing_required'] = list(missing_required)
        validation_result['issues'].append(f"Missing required columns: {missing_required}")
    
    # Check missing optional columns
    missing_optional = optional_set - df_columns
    if missing_optional:
        validation_result['missing_optional'] = list(missing_optional)
    
    # Check extra columns
    extra_columns = df_columns - expected_columns
    if extra_columns:
        validation_result['extra_columns'] = list(extra_columns)
        validation_result['issues'].append(f"Unexpected columns found: {extra_columns}")
    
    # Summary statistics
    validation_result['summary'] = {
        'total_columns': len(df.columns),
        'total_rows': len(df),
        'required_present': len(required_set & df_columns),
        'optional_present': len(optional_set & df_columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
    }
    
    logger.info(f"Schema validation completed. Valid: {validation_result['is_valid']}")
    return validation_result


def validate_data_quality(df: pd.DataFrame, target_column: str = None) -> Dict[str, Any]:
    """
    Comprehensive data quality validation.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        target_column (str, optional): Target column name
        
    Returns:
        Dict[str, Any]: Data quality metrics
    """
    if df is None or df.empty:
        return {
            'is_valid': False,
            'issues': ['DataFrame is None or empty'],
            'quality_score': 0.0
        }
    
    quality_metrics = {
        'is_valid': True,
        'issues': [],
        'quality_score': 0.0,
        'completeness': {},
        'consistency': {},
        'duplicates': {},
        'target_analysis': {}
    }
    
    # Completeness analysis
    total_cells = df.size
    missing_cells = df.isnull().sum().sum()
    completeness_ratio = 1 - (missing_cells / total_cells) if total_cells > 0 else 0
    
    quality_metrics['completeness'] = {
        'overall_ratio': completeness_ratio,
        'missing_by_column': df.isnull().sum().to_dict(),
        'complete_rows': len(df.dropna()),
        'total_rows': len(df)
    }
    
    # Duplicate analysis
    duplicate_rows = df.duplicated().sum()
    quality_metrics['duplicates'] = {
        'count': int(duplicate_rows),
        'ratio': duplicate_rows / len(df) if len(df) > 0 else 0
    }
    
    # Consistency checks
    consistency_issues = []
    
    # Check for consistent data types
    for col in df.select_dtypes(include=[object]).columns:
        unique_types = set(type(val).__name__ for val in df[col].dropna())
        if len(unique_types) > 2:  # Allow some variation
            consistency_issues.append(f"Column {col} has inconsistent data types: {unique_types}")
    
    quality_metrics['consistency'] = {
        'issues': consistency_issues,
        'has_issues': len(consistency_issues) > 0
    }
    
    # Target column analysis
    if target_column and target_column in df.columns:
        target_series = df[target_column]
        quality_metrics['target_analysis'] = {
            'missing_count': target_series.isnull().sum(),
            'unique_values': target_series.nunique(),
            'value_counts': target_series.value_counts().to_dict(),
            'data_type': str(target_series.dtype)
        }
        
        # Check for class imbalance (if binary or small categorical)
        if target_series.nunique() <= 10:
            value_counts = target_series.value_counts()
            if len(value_counts) == 2:
                imbalance_ratio = value_counts.min() / value_counts.max()
                if imbalance_ratio < 0.1:
                    quality_metrics['issues'].append(f"Severe class imbalance in {target_column}: {imbalance_ratio:.3f}")
    
    # Calculate overall quality score
    quality_score = completeness_ratio * 0.4  # 40% weight on completeness
    quality_score += (1 - min(duplicate_rows / len(df), 0.5)) * 0.3  # 30% weight on uniqueness
    quality_score += (1 if not consistency_issues else 0.7) * 0.3  # 30% weight on consistency
    
    quality_metrics['quality_score'] = quality_score
    quality_metrics['is_valid'] = quality_score >= 0.7  # Threshold for acceptable quality
    
    if quality_score < 0.7:
        quality_metrics['issues'].append(f"Overall data quality score too low: {quality_score:.3f}")
    
    logger.info(f"Data quality validation completed. Score: {quality_score:.3f}")
    return quality_metrics


def validate_text_columns(df: pd.DataFrame, text_columns: List[str]) -> Dict[str, Any]:
    """
    Validate text columns for common issues.
    
    Args:
        df (pd.DataFrame): DataFrame containing text columns
        text_columns (List[str]): List of text column names
        
    Returns:
        Dict[str, Any]: Text validation results
    """
    validation_results = {
        'is_valid': True,
        'issues': [],
        'column_analysis': {}
    }
    
    for col in text_columns:
        if col not in df.columns:
            validation_results['issues'].append(f"Text column '{col}' not found")
            validation_results['is_valid'] = False
            continue
        
        col_data = df[col].dropna()
        if col_data.empty:
            validation_results['issues'].append(f"Text column '{col}' is empty after removing nulls")
            continue
        
        # Analyze text characteristics
        text_lengths = col_data.astype(str).str.len()
        
        col_analysis = {
            'total_entries': len(col_data),
            'empty_strings': (col_data.astype(str).str.strip() == '').sum(),
            'avg_length': text_lengths.mean(),
            'min_length': text_lengths.min(),
            'max_length': text_lengths.max(),
            'very_short_count': (text_lengths < 10).sum(),  # Less than 10 characters
            'very_long_count': (text_lengths > 1000).sum(),  # More than 1000 characters
        }
        
        # Check for potential encoding issues
        encoding_issues = 0
        for text in col_data.astype(str).head(100):  # Sample check
            if '?' in text or '�' in text:  # Common encoding error markers
                encoding_issues += 1
        
        col_analysis['potential_encoding_issues'] = encoding_issues
        
        # Flag issues
        if col_analysis['empty_strings'] > len(col_data) * 0.1:
            validation_results['issues'].append(f"Column '{col}' has too many empty strings: {col_analysis['empty_strings']}")
        
        if col_analysis['avg_length'] < 5:
            validation_results['issues'].append(f"Column '{col}' has very short average length: {col_analysis['avg_length']:.1f}")
        
        if encoding_issues > 5:
            validation_results['issues'].append(f"Column '{col}' may have encoding issues: {encoding_issues} potential cases")
        
        validation_results['column_analysis'][col] = col_analysis
    
    if validation_results['issues']:
        validation_results['is_valid'] = False
    
    logger.info(f"Text validation completed for {len(text_columns)} columns")
    return validation_results


def validate_numerical_columns(df: pd.DataFrame, numerical_columns: List[str]) -> Dict[str, Any]:
    """
    Validate numerical columns for outliers and data quality.
    
    Args:
        df (pd.DataFrame): DataFrame containing numerical columns
        numerical_columns (List[str]): List of numerical column names
        
    Returns:
        Dict[str, Any]: Numerical validation results
    """
    validation_results = {
        'is_valid': True,
        'issues': [],
        'column_analysis': {}
    }
    
    for col in numerical_columns:
        if col not in df.columns:
            validation_results['issues'].append(f"Numerical column '{col}' not found")
            validation_results['is_valid'] = False
            continue
        
        col_data = pd.to_numeric(df[col], errors='coerce')
        valid_data = col_data.dropna()
        
        if valid_data.empty:
            validation_results['issues'].append(f"Numerical column '{col}' has no valid values")
            continue
        
        # Calculate statistics
        q1 = valid_data.quantile(0.25)
        q3 = valid_data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = valid_data[(valid_data < lower_bound) | (valid_data > upper_bound)]
        
        col_analysis = {
            'total_values': len(col_data),
            'valid_values': len(valid_data),
            'invalid_values': len(col_data) - len(valid_data),
            'mean': valid_data.mean(),
            'median': valid_data.median(),
            'std': valid_data.std(),
            'min': valid_data.min(),
            'max': valid_data.max(),
            'outlier_count': len(outliers),
            'outlier_ratio': len(outliers) / len(valid_data) if len(valid_data) > 0 else 0,
            'zero_count': (valid_data == 0).sum(),
            'negative_count': (valid_data < 0).sum()
        }
        
        # Flag issues
        if col_analysis['invalid_values'] > len(col_data) * 0.05:
            validation_results['issues'].append(f"Column '{col}' has too many invalid values: {col_analysis['invalid_values']}")
        
        if col_analysis['outlier_ratio'] > 0.1:
            validation_results['issues'].append(f"Column '{col}' has high outlier ratio: {col_analysis['outlier_ratio']:.3f}")
        
        if col_analysis['std'] == 0:
            validation_results['issues'].append(f"Column '{col}' has zero standard deviation (constant values)")
        
        validation_results['column_analysis'][col] = col_analysis
    
    if validation_results['issues']:
        validation_results['is_valid'] = False
    
    logger.info(f"Numerical validation completed for {len(numerical_columns)} columns")
    return validation_results


def validate_model_input(X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray] = None) -> Dict[str, Any]:
    """
    Validate model input data.
    
    Args:
        X: Feature matrix
        y: Target vector (optional)
        
    Returns:
        Dict[str, Any]: Validation results
    """
    validation_result = {
        'is_valid': True,
        'issues': [],
        'X_shape': None,
        'y_shape': None,
        'feature_analysis': {}
    }
    
    # Validate X
    if X is None:
        validation_result['issues'].append('Feature matrix X is None')
        validation_result['is_valid'] = False
        return validation_result
    
    if hasattr(X, 'shape'):
        validation_result['X_shape'] = X.shape
        
        if X.shape[0] == 0:
            validation_result['issues'].append('Feature matrix X has no rows')
            validation_result['is_valid'] = False
        
        if X.shape[1] == 0:
            validation_result['issues'].append('Feature matrix X has no columns')
            validation_result['is_valid'] = False
    
    # Check for NaN/inf values
    if isinstance(X, pd.DataFrame):
        nan_count = X.isnull().sum().sum()
        inf_count = np.isinf(X.select_dtypes(include=[np.number])).sum().sum()
    elif isinstance(X, np.ndarray):
        nan_count = np.isnan(X).sum() if X.dtype != object else 0
        inf_count = np.isinf(X).sum() if X.dtype != object else 0
    else:
        nan_count = inf_count = 0
    
    if nan_count > 0:
        validation_result['issues'].append(f'Feature matrix contains {nan_count} NaN values')
        validation_result['is_valid'] = False
    
    if inf_count > 0:
        validation_result['issues'].append(f'Feature matrix contains {inf_count} infinite values')
        validation_result['is_valid'] = False
    
    # Validate y if provided
    if y is not None:
        if hasattr(y, 'shape'):
            validation_result['y_shape'] = y.shape
            
            if len(y) != X.shape[0]:
                validation_result['issues'].append(f'Shape mismatch: X has {X.shape[0]} rows, y has {len(y)} values')
                validation_result['is_valid'] = False
        
        # Check for NaN values in y
        if isinstance(y, pd.Series):
            y_nan_count = y.isnull().sum()
        elif isinstance(y, np.ndarray):
            y_nan_count = np.isnan(y).sum() if y.dtype != object else 0
        else:
            y_nan_count = 0
        
        if y_nan_count > 0:
            validation_result['issues'].append(f'Target vector contains {y_nan_count} NaN values')
            validation_result['is_valid'] = False
    
    logger.info(f"Model input validation completed. Valid: {validation_result['is_valid']}")
    return validation_result


def validate_file_path(file_path: Union[str, Path], check_exists: bool = True, 
                      allowed_extensions: List[str] = None) -> Dict[str, Any]:
    """
    Validate file path and accessibility.
    
    Args:
        file_path: Path to validate
        check_exists: Whether to check if file exists
        allowed_extensions: List of allowed file extensions
        
    Returns:
        Dict[str, Any]: Validation results
    """
    validation_result = {
        'is_valid': True,
        'issues': [],
        'path_info': {}
    }
    
    if not file_path:
        validation_result['issues'].append('File path is empty or None')
        validation_result['is_valid'] = False
        return validation_result
    
    path_obj = Path(file_path)
    
    validation_result['path_info'] = {
        'absolute_path': str(path_obj.absolute()),
        'exists': path_obj.exists(),
        'is_file': path_obj.is_file() if path_obj.exists() else False,
        'extension': path_obj.suffix.lower(),
        'size_bytes': path_obj.stat().st_size if path_obj.exists() else 0
    }
    
    # Check existence
    if check_exists and not path_obj.exists():
        validation_result['issues'].append(f'File does not exist: {file_path}')
        validation_result['is_valid'] = False
    
    # Check if it's actually a file
    if check_exists and path_obj.exists() and not path_obj.is_file():
        validation_result['issues'].append(f'Path exists but is not a file: {file_path}')
        validation_result['is_valid'] = False
    
    # Check extension
    if allowed_extensions and path_obj.suffix.lower() not in [ext.lower() for ext in allowed_extensions]:
        validation_result['issues'].append(f'File extension {path_obj.suffix} not in allowed extensions: {allowed_extensions}')
        validation_result['is_valid'] = False
    
    # Check file size (basic check)
    if path_obj.exists() and validation_result['path_info']['size_bytes'] == 0:
        validation_result['issues'].append(f'File is empty: {file_path}')
    
    logger.info(f"File path validation completed for: {file_path}")
    return validation_result


def generate_validation_report(validation_results: Dict[str, Dict]) -> str:
    """
    Generate a comprehensive validation report.
    
    Args:
        validation_results: Dictionary of validation results from different validators
        
    Returns:
        str: Formatted validation report
    """
    report_lines = [
        "=" * 60,
        "DATA VALIDATION REPORT",
        "=" * 60,
        ""
    ]
    
    overall_valid = True
    total_issues = 0
    
    for validator_name, results in validation_results.items():
        report_lines.append(f"\n{validator_name.upper()} VALIDATION:")
        report_lines.append("-" * 40)
        
        if isinstance(results, dict):
            is_valid = results.get('is_valid', False)
            issues = results.get('issues', [])
            
            report_lines.append(f"Status: {'PASS' if is_valid else 'FAIL'}")
            
            if issues:
                total_issues += len(issues)
                overall_valid = False
                report_lines.append(f"Issues ({len(issues)}):")
                for issue in issues:
                    report_lines.append(f"  - {issue}")
            else:
                report_lines.append("No issues found")
                
            # Add summary info if available
            if 'summary' in results:
                report_lines.append("Summary:")
                for key, value in results['summary'].items():
                    report_lines.append(f"  {key}: {value}")
    
    # Overall summary
    report_lines.extend([
        "",
        "=" * 60,
        f"OVERALL VALIDATION: {'PASS' if overall_valid else 'FAIL'}",
        f"Total Issues Found: {total_issues}",
        "=" * 60
    ])
    
    return "\n".join(report_lines)
