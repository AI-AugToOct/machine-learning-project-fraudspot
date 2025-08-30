"""
Data Utilities for Job Fraud Detection

This module provides common utilities for data operations including
validation, statistics, transformations, and quality assessment.

Version: 1.0.0
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DataUtils:
    """Utility class for common data operations."""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: List[str] = None,
                          min_rows: int = 10) -> Dict[str, Any]:
        """
        Validate DataFrame structure and quality.
        
        Args:
            df (pd.DataFrame): DataFrame to validate
            required_columns (List[str], optional): Required column names
            min_rows (int): Minimum number of rows required
            
        Returns:
            Dict[str, Any]: Validation results
        """
        try:
            validation_result = {
                'is_valid': True,
                'warnings': [],
                'errors': [],
                'statistics': {}
            }
            
            # Check if DataFrame is empty
            if df.empty:
                validation_result['is_valid'] = False
                validation_result['errors'].append("DataFrame is empty")
                return validation_result
            
            # Check minimum rows
            if len(df) < min_rows:
                validation_result['warnings'].append(f"Low row count: {len(df)} < {min_rows}")
            
            # Check required columns
            if required_columns:
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    validation_result['is_valid'] = False
                    validation_result['errors'].append(f"Missing required columns: {missing_columns}")
            
            # Calculate statistics
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
    
    @staticmethod
    def analyze_class_balance(y: pd.Series) -> Dict[str, Any]:
        """
        Analyze class balance in target variable.
        
        Args:
            y (pd.Series): Target variable
            
        Returns:
            Dict[str, Any]: Class balance analysis
        """
        try:
            if y.empty:
                return {'error': 'Empty target variable'}
            
            class_counts = y.value_counts()
            class_percentages = y.value_counts(normalize=True) * 100
            
            analysis = {
                'class_counts': class_counts.to_dict(),
                'class_percentages': class_percentages.round(2).to_dict(),
                'n_classes': len(class_counts),
                'majority_class': class_counts.idxmax(),
                'minority_class': class_counts.idxmin(),
                'imbalance_ratio': class_counts.max() / class_counts.min(),
                'minority_percentage': class_percentages.min()
            }
            
            # Determine balance level
            if analysis['minority_percentage'] >= 40:
                analysis['balance_level'] = 'Balanced'
            elif analysis['minority_percentage'] >= 20:
                analysis['balance_level'] = 'Moderately Imbalanced'
            elif analysis['minority_percentage'] >= 10:
                analysis['balance_level'] = 'Highly Imbalanced'
            else:
                analysis['balance_level'] = 'Extremely Imbalanced'
            
            # Recommendations
            analysis['recommendations'] = []
            if analysis['minority_percentage'] < 30:
                analysis['recommendations'].append("Consider class balancing techniques (SMOTE, oversampling)")
            if analysis['minority_percentage'] < 10:
                analysis['recommendations'].append("Use stratified sampling and appropriate metrics (F1, AUC)")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing class balance: {str(e)}")
            return {'error': str(e)}
    
    @staticmethod
    def detect_data_quality_issues(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect various data quality issues.
        
        Args:
            df (pd.DataFrame): DataFrame to analyze
            
        Returns:
            Dict[str, Any]: Data quality issues found
        """
        try:
            issues = {
                'missing_values': {},
                'duplicate_rows': 0,
                'constant_columns': [],
                'high_cardinality': [],
                'outliers': {},
                'data_type_issues': [],
                'text_issues': {}
            }
            
            # Missing values analysis
            missing_counts = df.isnull().sum()
            missing_columns = missing_counts[missing_counts > 0]
            if not missing_columns.empty:
                issues['missing_values'] = {
                    col: {'count': int(count), 'percentage': (count / len(df)) * 100}
                    for col, count in missing_columns.items()
                }
            
            # Duplicate rows
            issues['duplicate_rows'] = int(df.duplicated().sum())
            
            # Constant columns
            for col in df.columns:
                if df[col].nunique() <= 1:
                    issues['constant_columns'].append(col)
            
            # High cardinality columns
            for col in df.select_dtypes(include=['object']).columns:
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio > 0.5 and df[col].nunique() > 10:
                    issues['high_cardinality'].append({
                        'column': col,
                        'unique_count': int(df[col].nunique()),
                        'unique_ratio': unique_ratio
                    })
            
            # Outliers in numeric columns
            for col in df.select_dtypes(include=[np.number]).columns:
                if df[col].nunique() > 2:  # Skip binary columns
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                    if outlier_count > 0:
                        issues['outliers'][col] = {
                            'count': int(outlier_count),
                            'percentage': (outlier_count / len(df)) * 100,
                            'bounds': {'lower': float(lower_bound), 'upper': float(upper_bound)}
                        }
            
            # Data type issues
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Check if numeric data stored as text
                    try:
                        pd.to_numeric(df[col], errors='raise')
                        issues['data_type_issues'].append(f"{col}: Numeric data stored as text")
                    except (ValueError, TypeError):
                        pass
            
            # Text column issues
            text_columns = df.select_dtypes(include=['object']).columns
            for col in text_columns:
                text_data = df[col].dropna().astype(str)
                if not text_data.empty:
                    avg_length = text_data.str.len().mean()
                    empty_strings = (text_data == '').sum()
                    very_short = (text_data.str.len() < 3).sum()
                    
                    issues['text_issues'][col] = {
                        'avg_length': float(avg_length),
                        'empty_strings': int(empty_strings),
                        'very_short_entries': int(very_short)
                    }
            
            logger.info("Data quality analysis completed")
            return issues
            
        except Exception as e:
            logger.error(f"Error detecting data quality issues: {str(e)}")
            return {'error': str(e)}
    
    @staticmethod
    def stratified_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2,
                        val_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, ...]:
        """
        Perform stratified split into train, validation, and test sets.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            test_size (float): Proportion for test set
            val_size (float): Proportion for validation set (from remaining data)
            random_state (int): Random seed
            
        Returns:
            Tuple: X_train, X_val, X_test, y_train, y_val, y_test
        """
        try:
            if X.empty or y.empty:
                raise ValueError("Empty data provided")
            
            # First split: separate test set
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=random_state
            )
            
            # Second split: separate train and validation from remaining data
            if val_size > 0:
                # Calculate validation size from remaining data
                val_size_adjusted = val_size / (1 - test_size)
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp, y_temp, test_size=val_size_adjusted, 
                    stratify=y_temp, random_state=random_state
                )
            else:
                X_train, y_train = X_temp, y_temp
                X_val, y_val = pd.DataFrame(), pd.Series()
            
            # Log split information
            total_samples = len(X)
            logger.info(f"Stratified split completed:")
            logger.info(f"  Total samples: {total_samples}")
            logger.info(f"  Train: {len(X_train)} ({len(X_train)/total_samples:.2%})")
            if not X_val.empty:
                logger.info(f"  Validation: {len(X_val)} ({len(X_val)/total_samples:.2%})")
            logger.info(f"  Test: {len(X_test)} ({len(X_test)/total_samples:.2%})")
            
            return X_train, X_val, X_test, y_train, y_val, y_test
            
        except Exception as e:
            logger.error(f"Error in stratified split: {str(e)}")
            raise
    
    @staticmethod
    def generate_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive data summary.
        
        Args:
            df (pd.DataFrame): DataFrame to summarize
            
        Returns:
            Dict[str, Any]: Data summary
        """
        try:
            if df.empty:
                return {'error': 'Empty DataFrame'}
            
            summary = {
                'basic_info': {
                    'n_rows': len(df),
                    'n_columns': len(df.columns),
                    'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
                },
                'data_types': df.dtypes.value_counts().to_dict(),
                'missing_values': {
                    'total': int(df.isnull().sum().sum()),
                    'percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
                    'by_column': df.isnull().sum().to_dict()
                },
                'duplicates': {
                    'count': int(df.duplicated().sum()),
                    'percentage': (df.duplicated().sum() / len(df)) * 100
                }
            }
            
            # Numeric columns summary
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                summary['numeric_summary'] = df[numeric_cols].describe().to_dict()
            
            # Categorical columns summary
            categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                categorical_summary = {}
                for col in categorical_cols:
                    categorical_summary[col] = {
                        'unique_count': int(df[col].nunique()),
                        'top_values': df[col].value_counts().head(5).to_dict()
                    }
                summary['categorical_summary'] = categorical_summary
            
            # Target variable analysis (if exists)
            if 'fraudulent' in df.columns:
                target_summary = DataUtils.analyze_class_balance(df['fraudulent'])
                summary['target_analysis'] = target_summary
            
            logger.info("Data summary generated successfully")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating data summary: {str(e)}")
            return {'error': str(e)}
    
    @staticmethod
    def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize column names.
        
        Args:
            df (pd.DataFrame): DataFrame with potentially problematic column names
            
        Returns:
            pd.DataFrame: DataFrame with cleaned column names
        """
        try:
            df_cleaned = df.copy()
            
            # Remove BOM and other invisible characters
            df_cleaned.columns = df_cleaned.columns.str.replace('\ufeff', '').str.strip()
            
            # Standardize column names
            df_cleaned.columns = (
                df_cleaned.columns
                .str.lower()
                .str.replace(' ', '_')
                .str.replace('-', '_')
                .str.replace('.', '_')
                .str.replace('(', '')
                .str.replace(')', '')
                .str.replace('[', '')
                .str.replace(']', '')
            )
            
            # Fix known typos (only keep job_desc mapping as it might still be needed)
            column_fixes = {
                'job_desc': 'job_description'  # Standardize
            }
            
            df_cleaned = df_cleaned.rename(columns=column_fixes)
            
            logger.info(f"Column names cleaned for {len(df_cleaned.columns)} columns")
            return df_cleaned
            
        except Exception as e:
            logger.error(f"Error cleaning column names: {str(e)}")
            return df
    
    @staticmethod
    def save_data_report(data_summary: Dict[str, Any], output_path: str) -> bool:
        """
        Save data analysis report to file.
        
        Args:
            data_summary (Dict[str, Any]): Data summary from generate_data_summary
            output_path (str): Path to save report
            
        Returns:
            bool: Success status
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            if output_path.endswith('.json'):
                import json
                with open(output_path, 'w') as f:
                    json.dump(data_summary, f, indent=2, default=str)
            else:
                # Generate markdown report
                report_lines = [
                    "# Data Analysis Report",
                    f"Generated at: {pd.Timestamp.now()}",
                    "",
                    "## Basic Information",
                ]
                
                if 'basic_info' in data_summary:
                    info = data_summary['basic_info']
                    report_lines.extend([
                        f"- Rows: {info.get('n_rows', 0):,}",
                        f"- Columns: {info.get('n_columns', 0)}",
                        f"- Memory Usage: {info.get('memory_usage_mb', 0):.2f} MB",
                        ""
                    ])
                
                if 'missing_values' in data_summary:
                    mv = data_summary['missing_values']
                    report_lines.extend([
                        "## Missing Values",
                        f"- Total Missing: {mv.get('total', 0):,} ({mv.get('percentage', 0):.1f}%)",
                        ""
                    ])
                
                if 'target_analysis' in data_summary:
                    target = data_summary['target_analysis']
                    report_lines.extend([
                        "## Target Variable Analysis",
                        f"- Balance Level: {target.get('balance_level', 'Unknown')}",
                        f"- Minority Percentage: {target.get('minority_percentage', 0):.1f}%",
                        ""
                    ])
                
                with open(output_path, 'w') as f:
                    f.write('\n'.join(report_lines))
            
            logger.info(f"Data report saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving data report: {str(e)}")
            return False