"""
Data Loading Module

This module handles loading various data sources for the fraud detection system
including CSV files, databases, and external data sources.

 Version: 3.0.0
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..core.constants import DataConstants

logger = logging.getLogger(__name__)


def load_training_data(data_path: str = None) -> pd.DataFrame:
    """
    Load training data from CSV files.
    
    Args:
        data_path (str, optional): Path to training data file
        
    Returns:
        pd.DataFrame: Loaded training data
    """
    try:
        # Default path to our main training dataset
        if data_path is None:
            data_path = 'src/features/mergedFakeWithRealData.csv'
        
        # Check if file exists
        if not os.path.exists(data_path):
            logger.error(f"Training data file not found: {data_path}")
            return pd.DataFrame()
        
        # Try different encodings
        encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(data_path, encoding=encoding)
                logger.info(f"Successfully loaded data with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            logger.error(f"Could not load {data_path} with any encoding")
            return pd.DataFrame()
        
        # Clean BOM characters from column names
        df.columns = df.columns.str.replace('﻿', '').str.strip()
        
        # Column names should already be standardized in CSV files
        # No need for column renaming since CSV files have been updated
        
        # Validate required columns
        required_columns = ['fraudulent']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return pd.DataFrame()
        
        # Log dataset statistics
        logger.info(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
        logger.info(f"Class distribution: {df['fraudulent'].value_counts().to_dict()}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading training data: {str(e)}")
        return pd.DataFrame()


def validate_data_schema(df: pd.DataFrame, required_columns: List[str]) -> Dict[str, Any]:
    """
    Validate dataset schema and structure.
    
    Args:
        df (pd.DataFrame): Dataset to validate
        required_columns (List[str]): Required column names
        
    Returns:
        Dict[str, Any]: Validation results
    """
    try:
        validation_result = {
            'is_valid': True,
            'missing_columns': [],
            'issues': [],
            'warnings': [],
            'statistics': {},
            'data_quality': {}
        }
        
        if df.empty:
            validation_result['is_valid'] = False
            validation_result['issues'].append("Dataset is empty")
            return validation_result
        
        # Check required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_result['is_valid'] = False
            validation_result['missing_columns'] = missing_columns
            validation_result['issues'].append(f"Missing required columns: {missing_columns}")
        
        # Data statistics
        validation_result['statistics'] = {
            'num_rows': len(df),
            'num_columns': len(df.columns),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024 / 1024,  # MB
            'duplicate_rows': df.duplicated().sum()
        }
        
        # Check for missing values
        missing_values = df.isnull().sum()
        columns_with_missing = missing_values[missing_values > 0]
        if not columns_with_missing.empty:
            validation_result['data_quality']['missing_values'] = columns_with_missing.to_dict()
            high_missing = columns_with_missing[columns_with_missing > len(df) * 0.5]
            if not high_missing.empty:
                validation_result['warnings'].append(f"Columns with >50% missing values: {list(high_missing.index)}")
        
        # Validate target column if exists
        if 'fraudulent' in df.columns:
            target_stats = df['fraudulent'].value_counts()
            validation_result['data_quality']['class_distribution'] = target_stats.to_dict()
            
            # Check class balance
            minority_class_ratio = min(target_stats) / len(df)
            if minority_class_ratio < 0.1:
                validation_result['warnings'].append(f"Highly imbalanced classes: {minority_class_ratio:.1%} minority class")
            elif minority_class_ratio < 0.2:
                validation_result['warnings'].append(f"Moderately imbalanced classes: {minority_class_ratio:.1%} minority class")
        
        # Check data types
        data_types = df.dtypes.value_counts().to_dict()
        validation_result['data_quality']['data_types'] = {str(k): v for k, v in data_types.items()}
        
        # Check for constant columns (no variation)
        constant_columns = []
        for col in df.columns:
            if df[col].nunique() <= 1:
                constant_columns.append(col)
        
        if constant_columns:
            validation_result['warnings'].append(f"Constant columns (may not be useful): {constant_columns}")
            validation_result['data_quality']['constant_columns'] = constant_columns
        
        # Check duplicate rows
        if validation_result['statistics']['duplicate_rows'] > 0:
            dup_ratio = validation_result['statistics']['duplicate_rows'] / len(df)
            validation_result['warnings'].append(f"Found {validation_result['statistics']['duplicate_rows']} duplicate rows ({dup_ratio:.1%})")
        
        # Text column analysis
        text_columns = ['job_title', 'job_desc', 'job_tasks', 'comp_name']
        text_analysis = {}
        for col in text_columns:
            if col in df.columns:
                empty_text = df[col].isnull() | (df[col] == '') | (df[col] == 'N/A')
                text_analysis[col] = {
                    'empty_count': empty_text.sum(),
                    'empty_ratio': empty_text.sum() / len(df),
                    'avg_length': df[col].dropna().astype(str).str.len().mean()
                }
        
        if text_analysis:
            validation_result['data_quality']['text_analysis'] = text_analysis
        
        # Overall validation status
        if validation_result['issues']:
            validation_result['is_valid'] = False
        
        logger.info(f"Data validation completed. Valid: {validation_result['is_valid']}")
        if validation_result['warnings']:
            logger.warning(f"Validation warnings: {len(validation_result['warnings'])} issues found")
        
        return validation_result
        
    except Exception as e:
        logger.error(f"Error validating data schema: {str(e)}")
        return {
            'is_valid': False,
            'missing_columns': [],
            'issues': [f"Validation error: {str(e)}"],
            'warnings': []
        }


def save_processed_data(df: pd.DataFrame, filename: str, data_type: str = 'processed') -> str:
    """
    Save processed data to appropriate directory.
    
    Args:
        df (pd.DataFrame): Data to save
        filename (str): Output filename
        data_type (str): Type of data (processed, raw, etc.)
        
    Returns:
        str: Path to saved file
    """
    try:
        if df.empty:
            logger.error("Cannot save empty DataFrame")
            return ""
        
        # Create directory structure
        data_dir = f"data/{data_type}"
        os.makedirs(data_dir, exist_ok=True)
        
        # Ensure filename has .csv extension
        if not filename.endswith('.csv'):
            filename += '.csv'
        
        file_path = os.path.join(data_dir, filename)
        
        # Save with UTF-8 encoding
        df.to_csv(file_path, index=False, encoding='utf-8')
        
        # Get absolute path
        abs_path = os.path.abspath(file_path)
        
        # Log operation
        file_size = os.path.getsize(abs_path) / 1024 / 1024  # MB
        logger.info(f"Saved {data_type} data: {abs_path}")
        logger.info(f"  Shape: {df.shape}, Size: {file_size:.2f} MB")
        
        return abs_path
        
    except Exception as e:
        logger.error(f"Error saving processed data: {str(e)}")
        return ""


# REMOVED: load_sample_data() - use load_training_data() with sample parameter instead


def get_data_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate comprehensive data statistics.
    
    Args:
        df (pd.DataFrame): Dataset to analyze
        
    Returns:
        Dict[str, Any]: Statistical summary
    """
    try:
        if df.empty:
            return {'shape': (0, 0), 'memory_usage': 0, 'missing_values': 0, 'error': 'Empty dataset'}
        
        stats = {}
        
        # Basic information
        stats['basic'] = {
            'shape': df.shape,
            'n_rows': len(df),
            'n_columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'columns': list(df.columns)
        }
        
        # Missing values analysis
        missing_values = df.isnull().sum()
        stats['missing_values'] = {
            'total_missing': missing_values.sum(),
            'missing_percentage': (missing_values.sum() / (len(df) * len(df.columns))) * 100,
            'columns_with_missing': missing_values[missing_values > 0].to_dict(),
            'complete_rows': len(df) - df.isnull().any(axis=1).sum()
        }
        
        # Data types
        dtype_counts = df.dtypes.value_counts()
        stats['data_types'] = {
            'type_distribution': {str(dtype): count for dtype, count in dtype_counts.items()},
            'column_types': {col: str(dtype) for col, dtype in df.dtypes.items()}
        }
        
        # Duplicates
        stats['duplicates'] = {
            'duplicate_rows': df.duplicated().sum(),
            'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100,
            'unique_rows': len(df.drop_duplicates())
        }
        
        # Cardinality (unique values per column)
        cardinality = {}
        for col in df.columns:
            unique_count = df[col].nunique()
            cardinality[col] = {
                'unique_values': unique_count,
                'unique_percentage': (unique_count / len(df)) * 100,
                'is_constant': unique_count <= 1,
                'is_binary': unique_count == 2,
                'is_categorical': unique_count < len(df) * 0.5
            }
        
        stats['cardinality'] = cardinality
        
        # Numerical columns statistics
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            numeric_stats = {}
            for col in numeric_columns:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    numeric_stats[col] = {
                        'mean': col_data.mean(),
                        'median': col_data.median(),
                        'std': col_data.std(),
                        'min': col_data.min(),
                        'max': col_data.max(),
                        'q25': col_data.quantile(0.25),
                        'q75': col_data.quantile(0.75),
                        'zeros': (col_data == 0).sum(),
                        'negative': (col_data < 0).sum()
                    }
            stats['numerical'] = numeric_stats
        
        # Text columns analysis
        text_columns = df.select_dtypes(include=['object']).columns
        if len(text_columns) > 0:
            text_stats = {}
            for col in text_columns:
                col_data = df[col].dropna().astype(str)
                if len(col_data) > 0:
                    text_lengths = col_data.str.len()
                    text_stats[col] = {
                        'avg_length': text_lengths.mean(),
                        'min_length': text_lengths.min(),
                        'max_length': text_lengths.max(),
                        'empty_strings': (col_data == '').sum(),
                        'most_common': col_data.value_counts().head(5).to_dict()
                    }
            stats['text'] = text_stats
        
        # Target variable analysis (if exists)
        if 'fraudulent' in df.columns:
            target_stats = df['fraudulent'].value_counts()
            stats['target'] = {
                'class_distribution': target_stats.to_dict(),
                'class_percentages': (target_stats / len(df) * 100).to_dict(),
                'is_balanced': abs(target_stats.iloc[0] - target_stats.iloc[1]) / len(df) < 0.1,
                'minority_class_ratio': min(target_stats) / len(df)
            }
        
        # Data quality flags
        quality_flags = []
        if stats['missing_values']['missing_percentage'] > 20:
            quality_flags.append("High missing values (>20%)")
        if stats['duplicates']['duplicate_percentage'] > 5:
            quality_flags.append("High duplicate rate (>5%)")
        if 'target' in stats and stats['target']['minority_class_ratio'] < 0.1:
            quality_flags.append("Highly imbalanced classes (<10% minority)")
        
        constant_cols = [col for col, info in cardinality.items() if info['is_constant']]
        if constant_cols:
            quality_flags.append(f"Constant columns found: {constant_cols}")
        
        stats['quality_flags'] = quality_flags
        
        logger.info(f"Generated statistics for dataset with {df.shape[0]} rows, {df.shape[1]} columns")
        return stats
        
    except Exception as e:
        logger.error(f"Error generating data statistics: {str(e)}")
        return {'shape': (0, 0), 'memory_usage': 0, 'missing_values': 0, 'error': str(e)}


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        df (pd.DataFrame): Dataset to split
        test_size (float): Proportion for test set
        random_state (int): Random seed for reproducibility
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, validation, test sets
    """
    try:
        if df.empty:
            logger.error("Cannot split empty dataset")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        from sklearn.model_selection import train_test_split

        # Check if we have target column for stratification
        target_col = None
        if 'fraudulent' in df.columns:
            target_col = 'fraudulent'
            y = df[target_col]
            X = df.drop(target_col, axis=1)
        else:
            logger.warning("No target column found, using random split")
            y = None
            X = df
        
        # For small datasets, adjust split ratios
        n_samples = len(df)
        if n_samples < 100:
            logger.warning(f"Small dataset ({n_samples} samples), using simple train/test split")
            if y is not None:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=y
                )
                train_df = pd.concat([X_train, y_train], axis=1)
                test_df = pd.concat([X_test, y_test], axis=1)
                return train_df, pd.DataFrame(), test_df  # Empty validation set for small data
            else:
                X_train, X_test = train_test_split(
                    X, test_size=test_size, random_state=random_state
                )
                return X_train, pd.DataFrame(), X_test
        
        # Standard split: 60% train, 20% validation, 20% test
        if y is not None:
            # First split: 80% train+val, 20% test
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # Second split: 60% train, 20% validation (from the 80%)
            val_size = 0.25  # 0.25 of 0.8 = 0.2 of total
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
            )
            
            # Combine features and targets back
            train_df = pd.concat([X_train, y_train], axis=1)
            val_df = pd.concat([X_val, y_val], axis=1)
            test_df = pd.concat([X_test, y_test], axis=1)
            
        else:
            # Without target column, random split
            X_temp, X_test = train_test_split(
                X, test_size=test_size, random_state=random_state
            )
            val_size = 0.25  # 0.25 of 0.8 = 0.2 of total
            X_train, X_val = train_test_split(
                X_temp, test_size=val_size, random_state=random_state
            )
            
            train_df, val_df, test_df = X_train, X_val, X_test
        
        # Log split statistics
        logger.info(f"Data split completed:")
        logger.info(f"  Train: {len(train_df)} samples ({len(train_df)/n_samples:.1%})")
        logger.info(f"  Validation: {len(val_df)} samples ({len(val_df)/n_samples:.1%})")
        logger.info(f"  Test: {len(test_df)} samples ({len(test_df)/n_samples:.1%})")
        
        # Log class distribution if target exists
        if target_col and target_col in train_df.columns:
            train_dist = train_df[target_col].value_counts()
            val_dist = val_df[target_col].value_counts() if not val_df.empty else "Empty"
            test_dist = test_df[target_col].value_counts()
            
            logger.info(f"Class distribution:")
            logger.info(f"  Train: {train_dist.to_dict()}")
            logger.info(f"  Validation: {val_dist.to_dict() if not val_df.empty else 'Empty'}")
            logger.info(f"  Test: {test_dist.to_dict()}")
        
        return train_df, val_df, test_df
        
    except Exception as e:
        logger.error(f"Error splitting data: {str(e)}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()