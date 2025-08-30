"""
Evaluation Service - SINGLE SOURCE for Model Evaluation Operations
This service handles all ML model evaluation operations without duplicating business logic.

Version: 3.0.0 - DRY Consolidation
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


class EvaluationService:
    """
    SINGLE SOURCE for all model evaluation operations.
    
    This service handles:
    - Model evaluation and metrics calculation
    - Report generation and formatting
    - Business impact analysis
    - Comparison reports and visualizations
    """
    
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize the evaluation service.
        
        Args:
            output_dir: Directory for saving evaluation reports
        """
        self.output_dir = output_dir
        self._ensure_output_directory()
        
        logger.info(f"EvaluationService initialized with output_dir: {output_dir}")
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      y_pred_proba: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='binary'),
                'recall': recall_score(y_true, y_pred, average='binary'),
                'f1_score': f1_score(y_true, y_pred, average='binary')
            }
            
            # Add ROC AUC if probabilities provided
            if y_pred_proba is not None:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            
            # Calculate business impact metrics
            business_metrics = self.calculate_business_metrics(y_true, y_pred)
            metrics.update(business_metrics)
            
            logger.info(f"Model evaluation completed: F1={metrics['f1_score']:.3f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error in model evaluation: {str(e)}")
            return {}
    
    def calculate_business_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 cost_false_positive: float = 1.0,
                                 cost_false_negative: float = 10.0) -> Dict[str, float]:
        """
        Calculate business impact metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            cost_false_positive: Cost of false positive (investigating legitimate job)
            cost_false_negative: Cost of false negative (missing fraudulent job)
            
        Returns:
            Dictionary of business metrics
        """
        try:
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            total_cost = (fp * cost_false_positive) + (fn * cost_false_negative)
            fraud_detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
            false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            return {
                'total_business_cost': total_cost,
                'fraud_detection_rate': fraud_detection_rate,
                'false_alarm_rate': false_alarm_rate,
                'true_positives': int(tp),
                'false_positives': int(fp),
                'true_negatives': int(tn),
                'false_negatives': int(fn)
            }
            
        except Exception as e:
            logger.error(f"Error calculating business metrics: {str(e)}")
            return {}
    
    def generate_training_report(self, metrics: Dict[str, Any], model_type: str) -> str:
        """
        Generate a formatted training report.
        
        Args:
            metrics: Training metrics dictionary
            model_type: Type of model trained
            
        Returns:
            Formatted training report string
        """
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            report = f"""
{'=' * 80}
                JOB FRAUD DETECTION - TRAINING REPORT
{'=' * 80}
Generated: {timestamp}

📊 MODEL PERFORMANCE
{'-' * 40}
Model Type: {model_type}
Accuracy:   {metrics.get('test_accuracy', 0):.4f}
Precision:  {metrics.get('test_precision', 0):.4f}
Recall:     {metrics.get('test_recall', 0):.4f}
F1 Score:   {metrics.get('test_f1_score', 0):.4f}
ROC AUC:    {metrics.get('test_roc_auc', 0):.4f}

💼 BUSINESS IMPACT
{'-' * 40}
Fraud Detection Rate: {metrics.get('fraud_detection_rate', 0):.2%}
False Alarm Rate:     {metrics.get('false_alarm_rate', 0):.2%}
Total Business Cost:  ${metrics.get('total_business_cost', 0):.2f}

🔢 CONFUSION MATRIX
{'-' * 40}
True Positives:  {metrics.get('true_positives', 0)}
False Positives: {metrics.get('false_positives', 0)}
True Negatives:  {metrics.get('true_negatives', 0)}
False Negatives: {metrics.get('false_negatives', 0)}

⏱️ TRAINING DETAILS
{'-' * 40}
Training Time: {metrics.get('training_time', 'N/A')} seconds
Data Size:     {metrics.get('data_size', 'N/A')} samples
Features Used: {metrics.get('n_features', 'N/A')}

📋 RECOMMENDATIONS
{'-' * 40}
{self._generate_recommendations(metrics)}

{'=' * 80}
"""
            return report
            
        except Exception as e:
            logger.error(f"Error generating training report: {str(e)}")
            return f"Error generating report: {str(e)}"
    
    def generate_model_comparison_report(self, model_results: Dict[str, Dict[str, Any]]) -> str:
        """
        Generate a formatted model comparison report.
        
        Args:
            model_results: Results for multiple models
            
        Returns:
            Formatted comparison report string
        """
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            report = f"""
{'=' * 80}
              JOB FRAUD DETECTION - MODEL COMPARISON REPORT
{'=' * 80}
Generated: {timestamp}

📊 MODEL COMPARISON TABLE
{'-' * 80}
{'Model':<15} {'Accuracy':<10} {'F1 Score':<10} {'Precision':<10} {'Recall':<10} {'ROC AUC':<10}
{'-' * 80}
"""
            
            # Sort models by F1 score (best first)
            sorted_models = sorted(
                model_results.items(),
                key=lambda x: x[1].get('metrics', {}).get('test_f1_score', 0),
                reverse=True
            )
            
            for model_name, results in sorted_models:
                metrics = results.get('metrics', {})
                report += f"{model_name:<15} "
                report += f"{metrics.get('test_accuracy', 0):<10.4f} "
                report += f"{metrics.get('test_f1_score', 0):<10.4f} "
                report += f"{metrics.get('test_precision', 0):<10.4f} "
                report += f"{metrics.get('test_recall', 0):<10.4f} "
                report += f"{metrics.get('test_roc_auc', 0):<10.4f}\n"
            
            report += f"{'-' * 80}\n\n"
            
            # Best model summary
            if sorted_models:
                best_model_name, best_results = sorted_models[0]
                best_metrics = best_results.get('metrics', {})
                
                report += f"""🏆 BEST PERFORMING MODEL
{'-' * 40}
Model: {best_model_name}
F1 Score: {best_metrics.get('test_f1_score', 0):.4f}
Accuracy: {best_metrics.get('test_accuracy', 0):.4f}
Training Time: {best_results.get('training_time', 'N/A')} seconds

💡 RECOMMENDATION: Use {best_model_name} for production deployment.

"""
            
            report += f"{'=' * 80}\n"
            return report
            
        except Exception as e:
            logger.error(f"Error generating comparison report: {str(e)}")
            return f"Error generating comparison report: {str(e)}"
    
    def save_evaluation_results(self, results: Dict[str, Any], 
                              filename: str = None) -> str:
        """
        Save evaluation results to file.
        
        Args:
            results: Evaluation results to save
            filename: Optional filename (auto-generated if None)
            
        Returns:
            Path to saved file
        """
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"evaluation_results_{timestamp}.json"
            
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Evaluation results saved to: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving evaluation results: {str(e)}")
            return ""
    
    def compare_models(self, model_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Compare multiple model results and create a comparison DataFrame.
        
        Args:
            model_results: Dictionary mapping model names to their results
            
        Returns:
            DataFrame with model comparison
        """
        try:
            comparison_data = []
            
            for model_name, results in model_results.items():
                metrics = results.get('metrics', {})
                comparison_data.append({
                    'Model': model_name,
                    'Accuracy': metrics.get('test_accuracy', 0),
                    'F1 Score': metrics.get('test_f1_score', 0),
                    'Precision': metrics.get('test_precision', 0),
                    'Recall': metrics.get('test_recall', 0),
                    'ROC AUC': metrics.get('test_roc_auc', 0),
                    'Training Time': results.get('training_time', 0),
                    'Business Cost': metrics.get('total_business_cost', 0)
                })
            
            df = pd.DataFrame(comparison_data)
            df = df.sort_values('F1 Score', ascending=False).reset_index(drop=True)
            
            logger.info(f"Model comparison completed for {len(comparison_data)} models")
            return df
            
        except Exception as e:
            logger.error(f"Error comparing models: {str(e)}")
            return pd.DataFrame()
    
    def _generate_recommendations(self, metrics: Dict[str, Any]) -> str:
        """Generate recommendations based on metrics."""
        recommendations = []
        
        f1_score = metrics.get('test_f1_score', 0)
        precision = metrics.get('test_precision', 0)
        recall = metrics.get('test_recall', 0)
        
        if f1_score < 0.7:
            recommendations.append("• Consider feature engineering or different algorithm")
        
        if precision < 0.8:
            recommendations.append("• High false positive rate - consider threshold tuning")
        
        if recall < 0.8:
            recommendations.append("• High false negative rate - consider class balancing")
        
        if not recommendations:
            recommendations.append("• Model performance is acceptable for production")
        
        return "\n".join(recommendations)
    
    def _ensure_output_directory(self):
        """Ensure output directory exists."""
        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create output directory: {str(e)}")


# Export main class
__all__ = ['EvaluationService']