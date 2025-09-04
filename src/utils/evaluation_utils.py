"""
Evaluation Utilities for Job Fraud Detection

This module provides common utilities for model evaluation including
report generation, visualization, and performance analysis.

Version: 1.0.0
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class EvaluationUtils:
    """Utility class for model evaluation operations."""
    
    @staticmethod
    def generate_training_report(training_results: Dict[str, Any], 
                               model_comparison: pd.DataFrame = None) -> str:
        """
        Generate comprehensive training report.
        
        Args:
            training_results (Dict[str, Any]): Training results
            model_comparison (pd.DataFrame, optional): Model comparison table
            
        Returns:
            str: Formatted training report
        """
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            report_lines = [
                "=" * 80,
                "                    JOB FRAUD DETECTION - TRAINING REPORT",
                "=" * 80,
                f"Generated: {timestamp}",
                "",
                "📊 TRAINING SUMMARY",
                "-" * 40
            ]
            
            # Basic information
            config = training_results.get('config', {})
            data_info = training_results.get('data_info', {})
            
            if config:
                report_lines.extend([
                    f"Model Type: {config.get('model_type', 'Unknown')}",
                    f"Weight Strategy: {config.get('weight_strategy', 'default')}",
                    f"Test Size: {config.get('test_size', 0.2):.1%}",
                    f"Random State: {config.get('random_state', 42)}",
                    f"SMOTE Enabled: {config.get('enable_smote', False)}",
                    ""
                ])
            
            # Dataset information
            if data_info:
                report_lines.extend([
                    "📁 DATASET INFORMATION",
                    "-" * 40,
                    f"Total Samples: {data_info.get('n_rows', 'Unknown'):,}",
                    f"Features: {data_info.get('n_columns', 'Unknown')}",
                    f"Memory Usage: {data_info.get('memory_usage_mb', 0):.2f} MB",
                ])
                
                if 'target_analysis' in data_info:
                    target = data_info['target_analysis']
                    report_lines.extend([
                        f"Fraud Rate: {target.get('minority_percentage', 0):.1f}%",
                        f"Balance Level: {target.get('balance_level', 'Unknown')}",
                    ])
                report_lines.append("")
            
            # Training metrics
            metrics = training_results.get('training_metrics', {})
            if metrics:
                report_lines.extend([
                    "🎯 TRAINING PERFORMANCE",
                    "-" * 40,
                    f"Accuracy:  {metrics.get('accuracy', 0):.3f}",
                    f"Precision: {metrics.get('precision', 0):.3f}",
                    f"Recall:    {metrics.get('recall', 0):.3f}",
                    f"F1 Score:  {metrics.get('f1_score', 0):.3f}",
                    f"ROC AUC:   {metrics.get('roc_auc', 0):.3f}",
                    ""
                ])
            
            # Evaluation metrics
            eval_metrics = training_results.get('evaluation_metrics', {})
            if eval_metrics:
                report_lines.extend([
                    "📈 TEST PERFORMANCE",
                    "-" * 40,
                    f"Accuracy:  {eval_metrics.get('accuracy', 0):.3f}",
                    f"Precision: {eval_metrics.get('precision', 0):.3f}",
                    f"Recall:    {eval_metrics.get('recall', 0):.3f}",
                    f"F1 Score:  {eval_metrics.get('f1_score', 0):.3f}",
                    f"ROC AUC:   {eval_metrics.get('roc_auc', 0):.3f}",
                    ""
                ])
                
                # Confusion matrix
                if 'confusion_matrix' in eval_metrics:
                    cm = eval_metrics['confusion_matrix']
                    if len(cm) == 2 and len(cm[0]) == 2:
                        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
                        report_lines.extend([
                            "📊 CONFUSION MATRIX",
                            "-" * 40,
                            f"True Negatives:  {tn:,}",
                            f"False Positives: {fp:,}",
                            f"False Negatives: {fn:,}",
                            f"True Positives:  {tp:,}",
                            f"Specificity:     {tn/(tn+fp):.3f}" if (tn+fp) > 0 else "Specificity: N/A",
                            f"Sensitivity:     {tp/(tp+fn):.3f}" if (tp+fn) > 0 else "Sensitivity: N/A",
                            ""
                        ])
            
            # Model comparison
            if model_comparison is not None and not model_comparison.empty:
                report_lines.extend([
                    "🏆 MODEL COMPARISON",
                    "-" * 40
                ])
                
                # Add top 3 models
                top_models = model_comparison.head(3)
                for i, (_, row) in enumerate(top_models.iterrows(), 1):
                    status_icon = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
                    report_lines.append(f"{status_icon} {row['Model']}: F1={row['F1_Score']:.3f}, Accuracy={row['Accuracy']:.3f}")
                
                report_lines.append("")
            
            # Feature importance
            feature_importance = training_results.get('feature_importance', [])
            if feature_importance:
                report_lines.extend([
                    "🔍 TOP FEATURE IMPORTANCE",
                    "-" * 40
                ])
                
                for i, (feature, importance) in enumerate(feature_importance[:10], 1):
                    report_lines.append(f"{i:2d}. {feature:<25} {importance:.4f}")
                
                report_lines.append("")
            
            # Performance assessment
            f1_score = eval_metrics.get('f1_score', metrics.get('f1_score', 0))
            if f1_score >= 0.8:
                assessment = "🌟 EXCELLENT - Model is production-ready"
            elif f1_score >= 0.7:
                assessment = "⚡ GOOD - Model performs well, consider minor tuning"
            elif f1_score >= 0.6:
                assessment = "⚠️ FAIR - Model needs improvement"
            else:
                assessment = "❌ POOR - Model requires significant work"
            
            report_lines.extend([
                "🎯 PERFORMANCE ASSESSMENT",
                "-" * 40,
                assessment,
                ""
            ])
            
            # Recommendations
            recommendations = EvaluationUtils.generate_recommendations(training_results)
            if recommendations:
                report_lines.extend([
                    "💡 RECOMMENDATIONS",
                    "-" * 40
                ])
                for i, rec in enumerate(recommendations, 1):
                    report_lines.append(f"{i}. {rec}")
                report_lines.append("")
            
            # Timing information
            timing = training_results.get('timing', {})
            if timing:
                report_lines.extend([
                    "⏱️ TIMING INFORMATION",
                    "-" * 40,
                    f"Training Time: {timing.get('training_time', 0):.2f}s",
                    f"Evaluation Time: {timing.get('evaluation_time', 0):.2f}s",
                    f"Total Pipeline Time: {timing.get('total_time', 0):.2f}s",
                    ""
                ])
            
            # Footer
            report_lines.extend([
                "=" * 80,
                "Generated by Job Fraud Detection System v1.0",
                f"Report ID: {abs(hash(str(training_results))) % 1000000:06d}",
                "=" * 80
            ])
            
            return "\n".join(report_lines)
            
        except Exception as e:
            logger.error(f"Error generating training report: {str(e)}")
            return f"Error generating report: {str(e)}"
    
    @staticmethod
    def generate_recommendations(training_results: Dict[str, Any]) -> List[str]:
        """
        Generate actionable recommendations based on training results.
        
        Args:
            training_results (Dict[str, Any]): Training results
            
        Returns:
            List[str]: List of recommendations
        """
        try:
            recommendations = []
            
            # Get metrics (prefer evaluation metrics over training metrics)
            metrics = training_results.get('evaluation_metrics', 
                                         training_results.get('training_metrics', {}))
            
            if not metrics:
                return ["No metrics available for recommendations"]
            
            # Performance-based recommendations
            f1_score = metrics.get('f1_score', 0)
            precision = metrics.get('precision', 0)
            recall = metrics.get('recall', 0)
            accuracy = metrics.get('accuracy', 0)
            
            if f1_score < 0.6:
                recommendations.append("Low F1 score: Consider hyperparameter tuning or different algorithms")
            
            if precision < 0.6:
                recommendations.append("Low precision: Review feature engineering to reduce false positives")
            
            if recall < 0.6:
                recommendations.append("Low recall: Consider class balancing techniques (SMOTE, cost-sensitive learning)")
            
            if accuracy - f1_score > 0.1:
                recommendations.append("High accuracy but low F1: Dataset may be imbalanced, focus on F1/AUC metrics")
            
            # Data-based recommendations
            data_info = training_results.get('data_info', {})
            target_analysis = data_info.get('target_analysis', {})
            
            if target_analysis:
                minority_pct = target_analysis.get('minority_percentage', 0)
                if minority_pct < 10:
                    recommendations.append("Extremely imbalanced dataset: Use SMOTE, cost-sensitive learning, or ensemble methods")
                elif minority_pct < 20:
                    recommendations.append("Imbalanced dataset: Consider oversampling or adjusting class weights")
            
            # Feature importance recommendations
            feature_importance = training_results.get('feature_importance', [])
            if feature_importance:
                # Check if content features are top contributors
                top_features = [feat[0] for feat in feature_importance[:5]]
                content_features = ['content_quality_score', 'company_legitimacy_score', 'contact_risk_score', 'suspicious_keywords_count']
                
                content_in_top = any(cf in top_features for cf in content_features)
                if not content_in_top:
                    recommendations.append("Content quality features not in top 5: Check feature engineering and weights")
            
            # Model-specific recommendations
            config = training_results.get('config', {})
            model_type = config.get('model_type', '')
            
            if model_type == 'random_forest' and f1_score < 0.7:
                recommendations.append("Random Forest underperforming: Try XGBoost or increase n_estimators")
            elif model_type == 'logistic_regression' and f1_score < 0.65:
                recommendations.append("Logistic Regression underperforming: Try feature scaling or polynomial features")
            
            # Training time recommendations
            timing = training_results.get('timing', {})
            training_time = timing.get('training_time', 0)
            
            if training_time > 300:  # 5 minutes
                recommendations.append("Long training time: Consider feature selection or simpler models for production")
            
            # If no issues found
            if not recommendations and f1_score >= 0.7:
                recommendations.append("Model performance is good - consider deploying to production")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return [f"Error generating recommendations: {str(e)}"]
    
    @staticmethod
    def create_model_comparison_report(comparison_df: pd.DataFrame) -> str:
        """
        Create a detailed model comparison report.
        
        Args:
            comparison_df (pd.DataFrame): Model comparison DataFrame
            
        Returns:
            str: Formatted comparison report
        """
        try:
            if comparison_df.empty:
                return "No model comparison data available"
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            report_lines = [
                "=" * 80,
                "                     MODEL COMPARISON REPORT",
                "=" * 80,
                f"Generated: {timestamp}",
                "",
                f"📊 COMPARED {len(comparison_df)} MODELS",
                "=" * 80
            ]
            
            # Summary statistics
            report_lines.extend([
                "📈 SUMMARY STATISTICS",
                "-" * 40,
                f"Best F1 Score:     {comparison_df['F1_Score'].max():.3f}",
                f"Average F1 Score:  {comparison_df['F1_Score'].mean():.3f}",
                f"F1 Score Range:    {comparison_df['F1_Score'].min():.3f} - {comparison_df['F1_Score'].max():.3f}",
                f"Best Accuracy:     {comparison_df['Accuracy'].max():.3f}",
                f"Average Accuracy:  {comparison_df['Accuracy'].mean():.3f}",
                ""
            ])
            
            # Detailed comparison
            report_lines.extend([
                "🏆 DETAILED COMPARISON",
                "-" * 40,
                f"{'Rank':<4} {'Model':<20} {'Accuracy':<9} {'Precision':<10} {'Recall':<8} {'F1':<8} {'ROC-AUC':<8} {'Status':<8}",
                "-" * 80
            ])
            
            for i, (_, row) in enumerate(comparison_df.iterrows(), 1):
                status_icon = "✅" if row.get('Status', 'Failed') == 'Success' else "❌"
                report_lines.append(
                    f"{i:<4} {row['Model']:<20} "
                    f"{row['Accuracy']:<9.3f} {row['Precision']:<10.3f} "
                    f"{row['Recall']:<8.3f} {row['F1_Score']:<8.3f} "
                    f"{row.get('ROC_AUC', 0):<8.3f} {status_icon:<8}"
                )
            
            report_lines.append("")
            
            # Performance analysis
            best_model = comparison_df.iloc[0]
            worst_model = comparison_df.iloc[-1]
            
            report_lines.extend([
                "🎯 PERFORMANCE ANALYSIS",
                "-" * 40,
                f"🥇 Best Model: {best_model['Model']}",
                f"   F1 Score: {best_model['F1_Score']:.3f}",
                f"   Accuracy: {best_model['Accuracy']:.3f}",
                f"   Balance: Precision={best_model['Precision']:.3f}, Recall={best_model['Recall']:.3f}",
                "",
                f"📊 Model Performance Distribution:",
                f"   Excellent (F1 ≥ 0.8): {(comparison_df['F1_Score'] >= 0.8).sum()} models",
                f"   Good (F1 ≥ 0.7):      {((comparison_df['F1_Score'] >= 0.7) & (comparison_df['F1_Score'] < 0.8)).sum()} models",
                f"   Fair (F1 ≥ 0.6):      {((comparison_df['F1_Score'] >= 0.6) & (comparison_df['F1_Score'] < 0.7)).sum()} models",
                f"   Poor (F1 < 0.6):      {(comparison_df['F1_Score'] < 0.6).sum()} models",
                ""
            ])
            
            # Recommendations
            recommendations = []
            
            if best_model['F1_Score'] >= 0.8:
                recommendations.append(f"✅ {best_model['Model']} is production-ready")
            elif best_model['F1_Score'] >= 0.7:
                recommendations.append(f"⚡ {best_model['Model']} is good - consider minor tuning")
            else:
                recommendations.append("⚠️ All models need improvement - review data quality and features")
            
            if 'Training_Time' in comparison_df.columns:
                fastest_model = comparison_df.loc[comparison_df['Training_Time'].idxmin()]
                recommendations.append(f"🚀 Fastest training: {fastest_model['Model']} ({fastest_model['Training_Time']:.1f}s)")
            
            # Check for balanced models
            comparison_df['Balance_Score'] = np.abs(comparison_df['Precision'] - comparison_df['Recall'])
            most_balanced = comparison_df.loc[comparison_df['Balance_Score'].idxmin()]
            recommendations.append(f"⚖️ Most balanced: {most_balanced['Model']} (Precision-Recall diff: {most_balanced['Balance_Score']:.3f})")
            
            report_lines.extend([
                "💡 RECOMMENDATIONS",
                "-" * 40
            ])
            
            for i, rec in enumerate(recommendations, 1):
                report_lines.append(f"{i}. {rec}")
            
            report_lines.extend([
                "",
                "=" * 80,
                "Generated by Job Fraud Detection System v1.0",
                "=" * 80
            ])
            
            return "\n".join(report_lines)
            
        except Exception as e:
            logger.error(f"Error creating model comparison report: {str(e)}")
            return f"Error creating comparison report: {str(e)}"
    
    @staticmethod
    def save_evaluation_results(results: Dict[str, Any], output_dir: str = "reports") -> List[str]:
        """
        Save evaluation results in multiple formats.
        
        Args:
            results (Dict[str, Any]): Evaluation results
            output_dir (str): Directory to save results
            
        Returns:
            List[str]: List of saved file paths
        """
        try:
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            saved_files = []
            
            # Save JSON results
            json_path = os.path.join(output_dir, f"evaluation_results_{timestamp}.json")
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            saved_files.append(json_path)
            
            # Save training report
            if 'training_results' in results:
                report_text = EvaluationUtils.generate_training_report(results['training_results'])
                report_path = os.path.join(output_dir, f"training_report_{timestamp}.txt")
                with open(report_path, 'w') as f:
                    f.write(report_text)
                saved_files.append(report_path)
            
            # Save model comparison
            if 'model_comparison' in results and isinstance(results['model_comparison'], pd.DataFrame):
                csv_path = os.path.join(output_dir, f"model_comparison_{timestamp}.csv")
                results['model_comparison'].to_csv(csv_path, index=False)
                saved_files.append(csv_path)
                
                # Create comparison report
                comparison_report = EvaluationUtils.create_model_comparison_report(results['model_comparison'])
                comparison_path = os.path.join(output_dir, f"comparison_report_{timestamp}.txt")
                with open(comparison_path, 'w') as f:
                    f.write(comparison_report)
                saved_files.append(comparison_path)
            
            logger.info(f"Evaluation results saved to {len(saved_files)} files in {output_dir}")
            return saved_files
            
        except Exception as e:
            logger.error(f"Error saving evaluation results: {str(e)}")
            return []
    
    @staticmethod
    def calculate_business_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                 cost_fp: float = 1.0, cost_fn: float = 10.0) -> Dict[str, float]:
        """
        Calculate business-oriented metrics for fraud detection.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            cost_fp: Cost of false positive (reviewing legitimate job)
            cost_fn: Cost of false negative (missing fraud)
            
        Returns:
            Dict[str, float]: Business metrics
        """
        try:
            from sklearn.metrics import confusion_matrix

            # Calculate confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            if cm.shape != (2, 2):
                return {'error': 'Binary classification required'}
            
            tn, fp, fn, tp = cm.ravel()
            
            # Business metrics
            total_cost = (fp * cost_fp) + (fn * cost_fn)
            cost_per_prediction = total_cost / len(y_true)
            
            # Fraud detection effectiveness
            fraud_detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
            false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            # Cost-benefit analysis
            total_fraud_value = tp + fn  # Assuming each fraud has unit value
            fraud_prevented = tp
            prevention_rate = fraud_prevented / total_fraud_value if total_fraud_value > 0 else 0
            
            metrics = {
                'total_cost': total_cost,
                'cost_per_prediction': cost_per_prediction,
                'fraud_detection_rate': fraud_detection_rate,
                'false_alarm_rate': false_alarm_rate,
                'prevention_rate': prevention_rate,
                'fraud_prevented': int(fraud_prevented),
                'false_alarms': int(fp),
                'missed_fraud': int(fn),
                'correctly_identified_legitimate': int(tn)
            }
            
            # ROI calculation (simplified)
            if cost_fn > 0:
                roi = (fraud_prevented * cost_fn - total_cost) / (fraud_prevented * cost_fn) if fraud_prevented > 0 else -1
                metrics['roi'] = roi
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating business metrics: {str(e)}")
            return {'error': str(e)}
    
    @staticmethod
    def generate_training_report(metrics: Dict[str, Any], model_type: str) -> str:
        """
        Generate a formatted training report.
        
        Args:
            metrics (Dict[str, Any]): Training metrics
            model_type (str): Type of model trained
            
        Returns:
            str: Formatted training report
        """
        try:
            report_lines = [
                "=" * 60,
                f"         {model_type.upper()} TRAINING REPORT",
                "=" * 60,
                "",
                f"🎯 MODEL PERFORMANCE METRICS",
                f"Accuracy: {metrics.get('accuracy', 0):.3f}",
                f"Precision: {metrics.get('precision', 0):.3f}",
                f"Recall: {metrics.get('recall', 0):.3f}",
                f"F1-Score: {metrics.get('f1_score', 0):.3f}",
                f"ROC-AUC: {metrics.get('roc_auc', 0):.3f}",
                "",
                f"📊 CONFUSION MATRIX",
                f"True Negatives: {metrics.get('tn', 'N/A')}",
                f"False Positives: {metrics.get('fp', 'N/A')}",
                f"False Negatives: {metrics.get('fn', 'N/A')}",
                f"True Positives: {metrics.get('tp', 'N/A')}",
                "",
                f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "=" * 60
            ]
            
            return "\n".join(report_lines)
            
        except Exception as e:
            logger.error(f"Error generating training report: {str(e)}")
            return f"Error generating report: {str(e)}"
    
    @staticmethod
    def generate_model_comparison_report(model_results: Dict[str, Dict[str, Any]]) -> str:
        """
        Generate a formatted model comparison report.
        
        Args:
            model_results (Dict[str, Dict[str, Any]]): Results for multiple models
            
        Returns:
            str: Formatted comparison report
        """
        try:
            report_lines = [
                "=" * 60,
                "         MODEL COMPARISON REPORT",
                "=" * 60,
                ""
            ]
            
            # Add individual model performance
            for model_type, results in model_results.items():
                if results.get('success', False):
                    metrics = results.get('metrics', {})
                    report_lines.extend([
                        f"🤖 {model_type.upper()}",
                        f"Accuracy: {metrics.get('accuracy', 0):.3f}",
                        f"Precision: {metrics.get('precision', 0):.3f}",
                        f"Recall: {metrics.get('recall', 0):.3f}",
                        f"F1-Score: {metrics.get('f1_score', 0):.3f}",
                        f"ROC-AUC: {metrics.get('roc_auc', 0):.3f}",
                        ""
                    ])
                else:
                    report_lines.extend([
                        f"❌ {model_type.upper()}: FAILED",
                        f"Error: {results.get('error', 'Unknown error')}",
                        ""
                    ])
            
            # Find best model
            best_model = None
            best_f1 = -1
            for model_type, results in model_results.items():
                if results.get('success', False):
                    f1_score = results.get('metrics', {}).get('f1_score', 0)
                    if f1_score > best_f1:
                        best_f1 = f1_score
                        best_model = model_type
            
            if best_model:
                report_lines.extend([
                    f"🏆 BEST MODEL: {best_model.upper()}",
                    f"Best F1-Score: {best_f1:.3f}",
                    ""
                ])
            
            report_lines.extend([
                f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "=" * 60
            ])
            
            return "\n".join(report_lines)
            
        except Exception as e:
            logger.error(f"Error generating model comparison report: {str(e)}")
            return f"Error generating report: {str(e)}"