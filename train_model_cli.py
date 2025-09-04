#!/usr/bin/env python3
"""
🤖 JOB FRAUD DETECTION - TRAINING CLI 🤖

Streamlined interactive CLI for training job fraud detection models using
the unified PipelineManager and DRY-compliant components.

Version: 3.0.0 - Refactored for DRY compliance
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to Python path
sys.path.append(str(Path(__file__).parent / 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Interactive CLI components
try:
    import questionary
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
    from rich.prompt import Confirm, Prompt
    from rich.table import Table
    from rich.text import Text
    
    RICH_AVAILABLE = True
    QUESTIONARY_AVAILABLE = True
        
except ImportError as e:
    print(f"📦 Install rich and questionary for enhanced UI: {e}")
    print("   pip install rich questionary")
    try:
        import questionary
        QUESTIONARY_AVAILABLE = True
        RICH_AVAILABLE = False
    except ImportError:
        QUESTIONARY_AVAILABLE = False
        RICH_AVAILABLE = False

# Import project modules
try:
    import pandas as pd

    from src.core.data_processor import DataProcessor
    from src.data.data_loader import load_training_data, save_processed_data, validate_data_schema
    from src.pipeline.pipeline_manager import PipelineManager
    from src.services.model_service import ModelService
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("🔧 Make sure you're running from the project root directory")
    print("📂 Current directory should contain 'src/' folder")
    sys.exit(1)


def generate_training_report(metrics: Dict[str, Any], model_type: str) -> str:
    """Generate a simple training report."""
    report = f"# Training Report - {model_type}\n\n"
    report += f"## Performance Metrics\n"
    report += f"- Accuracy: {metrics.get('accuracy', 0):.4f}\n"
    report += f"- F1 Score: {metrics.get('f1_score', 0):.4f}\n"
    report += f"- Precision: {metrics.get('precision', 0):.4f}\n"
    report += f"- Recall: {metrics.get('recall', 0):.4f}\n"
    report += f"\n## Training Details\n"
    report += f"- Model Type: {model_type}\n"
    report += f"- Training Time: {metrics.get('training_time', 'N/A')} seconds\n"
    return report


def generate_model_comparison_report(model_results: Dict[str, Dict[str, Any]]) -> str:
    """Generate a realistic model comparison report for imbalanced data."""
    from datetime import datetime
    
    report = "# Model Comparison Report (Realistic Metrics)\n\n"
    report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # Get fraud rate from first model (should be same for all)
    first_result = next(iter(model_results.values()))
    first_metrics = first_result.get('metrics', {})
    fraud_rate = first_metrics.get('fraud_rate', 0)
    
    report += f"**Dataset Context**: {fraud_rate:.1%} fraud rate (severely imbalanced)\n"
    report += f"**Baseline**: Random classifier would achieve {(1-fraud_rate):.1%} accuracy\n\n"
    
    # Create comprehensive table
    report += "| Model | F1 | Balanced Acc | PR-AUC | MCC | Precision | Recall | Accuracy* |\n"
    report += "|-------|----|--------------|---------|----|-----------|--------|----------|\n"
    
    for model_name, results in model_results.items():
        metrics = results.get('metrics', {})
        report += f"| {model_name} | "
        report += f"{metrics.get('f1_score', 0):.3f} | "
        report += f"{metrics.get('balanced_accuracy', 0):.3f} | "
        report += f"{metrics.get('pr_auc', 0):.3f} | "
        report += f"{metrics.get('matthews_corrcoef', 0):.3f} | "
        report += f"{metrics.get('precision', 0):.3f} | "
        report += f"{metrics.get('recall', 0):.3f} | "
        report += f"{metrics.get('accuracy', 0):.3f} |\n"
    
    report += "\n*Accuracy is misleading with imbalanced data - focus on F1, Balanced Accuracy, and PR-AUC\n"
    
    # Add warnings section
    warnings = []
    for model_name, results in model_results.items():
        metrics = results.get('metrics', {})
        if 'warning' in metrics:
            warnings.append(f"- **{model_name}**: {metrics['warning']}")
        if 'pr_warning' in metrics:
            warnings.append(f"- **{model_name}**: {metrics['pr_warning']}")
    
    if warnings:
        report += "\n## ⚠️ Warnings\n\n"
        report += "\n".join(warnings) + "\n"
    
    return report


def compare_model_results(model_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Compare multiple model results and create a comparison DataFrame.
    
    Args:
        model_results: Dictionary mapping model names to their metrics
        
    Returns:
        DataFrame with model comparison
    """
    comparison_data = []
    
    for model_name, results in model_results.items():
        metrics = results.get('metrics', {})
        comparison_data.append({
            'Model': model_name,
            'Accuracy': metrics.get('accuracy', 0),
            'F1 Score': metrics.get('f1_score', 0),
            'Precision': metrics.get('precision', 0),
            'Recall': metrics.get('recall', 0),
            'Training Time': results.get('training_time', 0)
        })
    
    return pd.DataFrame(comparison_data).sort_values('F1 Score', ascending=False)




class JobFraudTrainingCLI:
    """Streamlined interactive CLI using PipelineManager."""
    
    def __init__(self):
        if RICH_AVAILABLE:
            self.console = Console()
        else:
            self.console = None
        
        self.models_available = {
            'random_forest': 'Random Forest (Recommended)',
            'logistic_regression': 'Logistic Regression',
            'svm': 'Support Vector Machine',
            'naive_bayes': 'Naive Bayes',
            'all_models': '🔄 Train All Models for Comparison'
        }
        
        self.balance_methods = {
            'smote': 'SMOTE (Recommended)',
            'oversample': 'Random Oversampling',
            'undersample': 'Random Undersampling',
            'none': 'No Balancing'
        }
        
        # Initialize with dynamic dataset discovery
        self.dataset_options = self._discover_datasets()
        
    def _discover_datasets(self):
        """Dynamically discover all available CSV datasets."""
        import glob
        
        datasets = {}
        
        # Predefined special options first
        datasets['auto'] = '🤖 Auto-detect best available dataset'
        
        # Scan for CSV files in data directories
        data_dirs = ['data/raw', 'data/processed']
        
        for data_dir in data_dirs:
            if os.path.exists(data_dir):
                csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
                
                for csv_file in sorted(csv_files):
                    # Get relative path and create key
                    rel_path = os.path.relpath(csv_file)
                    file_name = os.path.basename(csv_file).replace('.csv', '')
                    
                    # Get file size
                    try:
                        size_mb = os.path.getsize(csv_file) / (1024 * 1024)
                        
                        # Try to get row count quickly
                        try:
                            with open(csv_file, 'r', encoding='utf-8') as f:
                                row_count = sum(1 for line in f) - 1  # Subtract header
                            rows_info = f"{row_count:,} rows"
                        except:
                            rows_info = f"{size_mb:.1f}MB"
                        
                        # Create descriptive labels based on filename
                        if 'network' in file_name:
                            emoji = '📊'
                            desc = 'Network Features Dataset'
                        elif 'merged_raw_data.csv' in csv_file:
                            emoji = '🔥'
                            desc = 'Merged Raw Dataset'
                        elif 'fake_job_postings' in file_name:
                            emoji = '🇺🇸'
                            desc = 'English Dataset'
                        elif 'mergedFakeWithRealData' in file_name:
                            emoji = '🇸🇦'
                            desc = 'Arabic Dataset'
                        elif 'multilingual' in file_name:
                            emoji = '🌍'
                            desc = 'Multilingual Dataset'
                        elif 'cleaned' in file_name:
                            emoji = '🧹'
                            desc = 'Cleaned Dataset'
                        elif 'jadarat' in file_name.lower():
                            emoji = '📚'
                            desc = 'Jadarat Dataset'
                        else:
                            emoji = '📄'
                            desc = file_name.replace('_', ' ').title()
                        
                        # Use filename as key for easy selection
                        key = file_name
                        datasets[key] = f'{emoji} {desc} ({rel_path} - {rows_info})'
                        
                    except Exception as e:
                        # Fallback if we can't read file info
                        datasets[file_name] = f'📄 {file_name} ({rel_path})'
        
        return datasets
    
    def _get_dataset_path(self, dataset_choice):
        """Get the file path for a selected dataset."""
        import glob
        
        if dataset_choice == 'auto':
            # Auto-detect best available dataset - prefer network version if available
            candidates = [
                "data/processed/merged_raw_data_with_network.csv",  # Prefer network features
                "data/processed/merged_raw_data.csv",
                "data/processed/multilingual_job_fraud_data.csv", 
                "data/raw/fake_job_postings.csv",
                "data/raw/mergedFakeWithRealData.csv"
            ]
            
            for candidate in candidates:
                if os.path.exists(candidate):
                    if self.console:
                        self.console.print(f"🤖 Auto-selected: {candidate}", style="green")
                    else:
                        print(f"🤖 Auto-selected: {candidate}")
                    return candidate
            
            # Fallback - find any CSV file
            for data_dir in ['data/processed', 'data/raw']:
                if os.path.exists(data_dir):
                    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
                    if csv_files:
                        return csv_files[0]
            
            raise FileNotFoundError("No CSV datasets found in data/raw or data/processed")
        
        # For specific dataset choices, find the matching file
        for data_dir in ['data/processed', 'data/raw']:
            if os.path.exists(data_dir):
                csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
                for csv_file in csv_files:
                    file_name = os.path.basename(csv_file).replace('.csv', '')
                    if file_name == dataset_choice:
                        if self.console:
                            self.console.print(f"✅ Using dataset: {csv_file}", style="green")
                        else:
                            print(f"✅ Using dataset: {csv_file}")
                        return csv_file
        
        # If not found, raise error with helpful message
        available_files = []
        for data_dir in ['data/processed', 'data/raw']:
            if os.path.exists(data_dir):
                csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
                available_files.extend([os.path.basename(f).replace('.csv', '') for f in csv_files])
        
        raise FileNotFoundError(f"Dataset '{dataset_choice}' not found. Available datasets: {', '.join(available_files)}")
    
    def print_header(self):
        """Print CLI header."""
        if self.console:
            header = Text("🤖 JOB FRAUD DETECTION - TRAINING CLI", style="bold blue")
            panel = Panel(
                header,
                subtitle="Powered by Unified Pipeline Manager v3.0.0",
                box=box.DOUBLE
            )
            self.console.print(panel)
        else:
            print("=" * 60)
            print("🤖 JOB FRAUD DETECTION - TRAINING CLI")
            print("Powered by Unified Pipeline Manager v3.0.0")
            print("=" * 60)
    
    def get_user_input(self):
        """Get training configuration from user."""
        config = {}
        
        # Dataset selection
        if QUESTIONARY_AVAILABLE:
            dataset_choices = [f"{k}: {v}" for k, v in self.dataset_options.items()]
            selected = questionary.select(
                "Select dataset for training:",
                choices=dataset_choices
            ).ask()
            dataset_choice = selected.split(':')[0]
        else:
            print("\nAvailable datasets:")
            for i, (key, desc) in enumerate(self.dataset_options.items(), 1):
                print(f"  {i}. {key}: {desc}")
            choice = input(f"Select dataset (1-{len(self.dataset_options)}) [5]: ").strip() or "5"
            dataset_keys = list(self.dataset_options.keys())
            dataset_choice = dataset_keys[int(choice) - 1]
        
        # Set data path based on choice
        config['data_path'] = self._get_dataset_path(dataset_choice)
        
        config['dataset_type'] = dataset_choice
        
        # Model type selection
        if QUESTIONARY_AVAILABLE:
            model_choices = [f"{k}: {v}" for k, v in self.models_available.items()]
            selected = questionary.select(
                "Select model type:",
                choices=model_choices
            ).ask()
            selected_key = selected.split(':')[0]
            
            if selected_key == 'all_models':
                config['model_type'] = 'random_forest'  # Default for single model path
                config['compare_models'] = True  # Enable comparison mode
            else:
                config['model_type'] = selected_key
                config['compare_models'] = False  # Single model mode
        else:
            print("\nAvailable models:")
            for i, (key, desc) in enumerate(self.models_available.items(), 1):
                print(f"  {i}. {key}: {desc}")
            choice = input(f"Select model (1-{len(self.models_available)}) [1]: ").strip() or "1"
            model_keys = list(self.models_available.keys())
            selected_key = model_keys[int(choice) - 1]
            
            if selected_key == 'all_models':
                config['model_type'] = 'random_forest'  # Default for single model path
                config['compare_models'] = True  # Enable comparison mode
            else:
                config['model_type'] = selected_key
                config['compare_models'] = False  # Single model mode
        
        # Class balancing method
        if QUESTIONARY_AVAILABLE:
            balance_choices = [f"{k}: {v}" for k, v in self.balance_methods.items()]
            selected = questionary.select(
                "Select class balancing method:",
                choices=balance_choices
            ).ask()
            config['balance_method'] = selected.split(':')[0]
        else:
            print("\nBalancing methods:")
            for i, (key, desc) in enumerate(self.balance_methods.items(), 1):
                print(f"  {i}. {key}: {desc}")
            choice = input("Select balancing method (1-4) [1]: ").strip() or "1"
            balance_keys = list(self.balance_methods.keys())
            config['balance_method'] = balance_keys[int(choice) - 1]
        
        # Output directory
        if QUESTIONARY_AVAILABLE:
            config['output_dir'] = questionary.text(
                "Enter output directory for trained models:",
                default="models"
            ).ask()
        else:
            config['output_dir'] = input("Enter output directory for trained models [models]: ").strip() or "models"
        
        # Training options (skip comparison question if already set by model selection)
        if 'compare_models' not in config:
            if QUESTIONARY_AVAILABLE:
                config['compare_models'] = questionary.confirm(
                    "Train and compare multiple models?", 
                    default=False
                ).ask()
            elif RICH_AVAILABLE:
                config['compare_models'] = Confirm.ask("Train and compare multiple models?", default=False)
            else:
                config['compare_models'] = input("Train and compare multiple models? (y/N): ").lower().startswith('y')
        
        # Always ask about report generation
        if QUESTIONARY_AVAILABLE:
            config['generate_report'] = questionary.confirm(
                "Generate detailed training report?", 
                default=True
            ).ask()
        elif RICH_AVAILABLE:
            config['generate_report'] = Confirm.ask("Generate detailed training report?", default=True)
        else:
            config['generate_report'] = input("Generate detailed training report? (Y/n): ").lower() != 'n'
        
        return config
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate user configuration."""
        # Check data file exists
        if not os.path.exists(config['data_path']):
            if self.console:
                self.console.print(f"❌ Data file not found: {config['data_path']}", style="red")
            else:
                print(f"❌ Data file not found: {config['data_path']}")
            return False
        
        # Create output directory if it doesn't exist
        os.makedirs(config['output_dir'], exist_ok=True)
        
        return True
    
    def run_single_model_training(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Train a single model using PipelineManager with proper API calls."""
        if self.console:
            self.console.print(f"🚀 Training {config['model_type']} model...", style="green")
        else:
            print(f"🚀 Training {config['model_type']} model...")
        
        try:
            import time
            start_time = time.time()
            
            # Initialize PipelineManager (fixed - removed preprocessing_steps parameter)
            pipeline_manager = PipelineManager(
                model_type=config['model_type'],
                balance_method=config.get('balance_method', 'smote'),
                scaling_method='standard'
            )
            
            # Show detailed progress if rich is available
            if RICH_AVAILABLE:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    transient=True
                ) as progress:
                    task = progress.add_task("Loading data...", total=5)
                    
                    # Step 1: Load data
                    pipeline_manager.load_data(config['data_path'])
                    progress.update(task, advance=1, description="Preparing data...")
                    
                    # Step 2: Prepare data (split, process, engineer features, balance)
                    X_train, X_test, y_train, y_test = pipeline_manager.prepare_data()
                    progress.update(task, advance=1, description="Training model...")
                    
                    # Step 3: Train model
                    training_info = pipeline_manager.train_model(X_train, y_train)
                    progress.update(task, advance=1, description="Evaluating model...")
                    
                    # Step 4: Evaluate model
                    metrics = pipeline_manager.evaluate_model(X_test, y_test)
                    progress.update(task, advance=1, description="Saving pipeline...")
                    
                    # Step 5: Save pipeline
                    pipeline_manager.save_pipeline(config['model_type'])
                    progress.update(task, completed=5, description="Training complete!")
            else:
                print("📊 Loading data...")
                pipeline_manager.load_data(config['data_path'])
                
                print("🔄 Preparing and processing data...")
                X_train, X_test, y_train, y_test = pipeline_manager.prepare_data()
                
                print("🎯 Training model...")
                training_info = pipeline_manager.train_model(X_train, y_train)
                
                print("📈 Evaluating performance...")
                metrics = pipeline_manager.evaluate_model(X_test, y_test)
                
                print("💾 Saving model...")
                pipeline_manager.save_pipeline(config['model_type'])
            
            # Calculate training time BEFORE using it
            training_time = time.time() - start_time
            
            # Save metrics to JSON file for dynamic weight calculation
            print("📊 Saving metrics for dynamic weight calculation...")
            self.save_metrics_to_json(config['model_type'], metrics, training_time)
            
            # Prepare results in expected format
            results = {
                'success': True,
                'metrics': metrics,  # Contains accuracy, f1_score, etc.
                'model_type': config['model_type'],
                'training_time': training_time
            }
            
            # Success message
            if self.console:
                self.console.print(f"✅ {config['model_type']} pipeline trained and saved successfully!", style="green")
                # Show realistic metrics with context
                test_f1 = metrics.get('f1_score', 0)
                test_acc = metrics.get('accuracy', 0) 
                balanced_acc = metrics.get('balanced_accuracy', 0)
                pr_auc = metrics.get('pr_auc', 0)
                fraud_rate = metrics.get('fraud_rate', 0)
                
                # Show key metrics with fraud context
                self.console.print(f"📊 Fraud Rate: {fraud_rate:.1%} | F1: {test_f1:.3f} | Balanced Acc: {balanced_acc:.3f} | PR-AUC: {pr_auc:.3f}", style="cyan")
                
                # Show warnings if present
                if 'warning' in metrics:
                    self.console.print(f"{metrics['warning']}", style="yellow")
                if 'pr_warning' in metrics:
                    self.console.print(f"{metrics['pr_warning']}", style="yellow")
            else:
                print(f"✅ {config['model_type']} pipeline trained and saved successfully!")
                test_f1 = metrics.get('f1_score', 0)
                test_acc = metrics.get('accuracy', 0)
                balanced_acc = metrics.get('balanced_accuracy', 0)
                pr_auc = metrics.get('pr_auc', 0)
                fraud_rate = metrics.get('fraud_rate', 0)
                
                print(f"📊 Fraud Rate: {fraud_rate:.1%} | F1: {test_f1:.3f} | Balanced Acc: {balanced_acc:.3f} | PR-AUC: {pr_auc:.3f}")
                
                # Show warnings if present
                if 'warning' in metrics:
                    print(f"{metrics['warning']}")
                if 'pr_warning' in metrics:
                    print(f"{metrics['pr_warning']}")
            
            return results
            
        except Exception as e:
            error_msg = f"Error training {config['model_type']} model: {str(e)}"
            logger.error(error_msg)
            if self.console:
                self.console.print(f"❌ {error_msg}", style="red")
            else:
                print(f"❌ {error_msg}")
            return {'success': False, 'error': error_msg}
    
    def run_model_comparison(self, config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Train and compare multiple models."""
        if self.console:
            self.console.print("🔄 Training multiple models for comparison...", style="blue")
        else:
            print("🔄 Training multiple models for comparison...")
        
        results = {}
        # Exclude 'all_models' option from actual training
        model_types = [k for k in self.models_available.keys() if k != 'all_models']
        
        for model_type in model_types:
            if self.console:
                self.console.print(f"Training {model_type}...", style="yellow")
            else:
                print(f"Training {model_type}...")
            
            # Update config for current model
            model_config = config.copy()
            model_config['model_type'] = model_type
            
            # Train model
            result = self.run_single_model_training(model_config)
            results[model_type] = result
            
            if result['success']:
                metrics = result.get('metrics', {})
                training_time = result.get('training_time', 0)
                f1_score = metrics.get('f1_score', 0)
                
                # Save metrics for each model in comparison mode
                self.save_metrics_to_json(model_type, metrics, training_time)
                
                if self.console:
                    self.console.print(f"✅ {model_type}: F1-Score = {f1_score:.3f}", style="green")
                else:
                    print(f"✅ {model_type}: F1-Score = {f1_score:.3f}")
            else:
                if self.console:
                    self.console.print(f"❌ {model_type}: Failed", style="red")
                else:
                    print(f"❌ {model_type}: Failed")
        
        # Generate comparison
        comparison_df = compare_model_results(results)
        
        if self.console:
            self.console.print("\n📊 Model Comparison Results:", style="bold")
            # Convert DataFrame to Rich table
            table = Table()
            for col in comparison_df.columns:
                table.add_column(col)
            
            for _, row in comparison_df.iterrows():
                table.add_row(*[str(val) for val in row])
            
            self.console.print(table)
        else:
            print("\n📊 Model Comparison Results:")
            print(comparison_df.to_string(index=False))
        
        return results
    
    def save_metrics_to_json(self, model_type: str, metrics: Dict[str, Any], training_time: float = 0):
        """Save model metrics to JSON file for dynamic weight calculation."""
        import json
        from datetime import datetime
        
        metrics_file = "models/model_metrics.json"
        
        try:
            # Auto-create directory if needed
            os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
            
            # Load existing metrics or create new structure
            if os.path.exists(metrics_file):
                try:
                    with open(metrics_file, 'r') as f:
                        data = json.load(f)
                    print(f"📊 Loaded existing metrics from {metrics_file}")
                except (json.JSONDecodeError, IOError) as e:
                    print(f"⚠️ Corrupted metrics file, recreating: {e}")
                    data = self._create_default_metrics_structure()
            else:
                print(f"📊 Creating new metrics file: {metrics_file}")
                data = self._create_default_metrics_structure()
            
            # Update model metrics
            data["models"][model_type] = {
                "f1_score": metrics.get("f1_score", 0.0),
                "precision": metrics.get("precision", 0.0),
                "recall": metrics.get("recall", 0.0),
                "accuracy": metrics.get("accuracy", 0.0),
                "balanced_accuracy": metrics.get("balanced_accuracy", 0.0),
                "pr_auc": metrics.get("pr_auc", 0.0),
                "training_time": training_time,
                "training_date": datetime.now().isoformat(),
                "model_path": f"models/{model_type}_pipeline.joblib",
                "status": "active"
            }
            
            # Recalculate weights based on F1 scores
            model_f1_scores = {name: model["f1_score"] for name, model in data["models"].items()}
            total_f1 = sum(model_f1_scores.values())
            
            if total_f1 > 0:
                for name, model in data["models"].items():
                    model["weight"] = model["f1_score"] / total_f1
            
            # Update timestamp
            data["last_updated"] = datetime.now().isoformat()
            
            # Save updated metrics
            os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
            with open(metrics_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Log the update
            f1_score = metrics.get("f1_score", 0.0)
            weight = data["models"][model_type]["weight"]
            print(f"✅ Metrics saved: {model_type} F1={f1_score:.3f} Weight={weight:.3f}")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not save metrics to JSON: {str(e)}")
    
    def _create_default_metrics_structure(self):
        """Create default metrics structure with precision-focused settings."""
        from datetime import datetime
        return {
            "last_updated": datetime.now().isoformat(),
            "data_source": "merged_raw_data.csv",
            "models": {},
            "ensemble_config": {
                "weight_calculation_method": "f1_based",
                "total_weight_sum": 1.0,
                "minimum_f1_threshold": 0.05,
                "fraud_threshold": 0.65,  # INCREASED from 0.5 to reduce false positives
                "confidence_threshold": 0.75,
                "precision_focused": True,
                "false_positive_penalty": 0.1
            },
            "dataset_info": {
                "total_samples": 19350,
                "fraud_rate": 0.0448,
                "class_balance": "imbalanced"
            },
            "version": "3.0.0"
        }
    
    def generate_training_report(self, results: Dict[str, Any], config: Dict[str, Any]):
        """Generate comprehensive training report."""
        if self.console:
            self.console.print("📝 Generating training report...", style="blue")
        else:
            print("📝 Generating training report...")
        
        try:
            if isinstance(results, dict) and 'metrics' in results:
                # Single model results
                report = generate_training_report(
                    results['metrics'],
                    results.get('model_type', config['model_type'])
                )
            else:
                # Multiple model results
                report = generate_model_comparison_report(results)
            
            # Save report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = os.path.join(config['output_dir'], f"training_report_{timestamp}.txt")
            
            with open(report_path, 'w') as f:
                f.write(report)
            
            if self.console:
                self.console.print(f"📄 Report saved to: {report_path}", style="green")
                self.console.print("\n" + "=" * 60)
                self.console.print("TRAINING REPORT PREVIEW", style="bold")
                self.console.print("=" * 60)
                self.console.print(report[:1000] + "..." if len(report) > 1000 else report)
            else:
                print(f"📄 Report saved to: {report_path}")
                print("\n" + "=" * 60)
                print("TRAINING REPORT PREVIEW")
                print("=" * 60)
                print(report[:1000] + "..." if len(report) > 1000 else report)
            
        except Exception as e:
            error_msg = f"Error generating report: {str(e)}"
            logger.error(error_msg)
            if self.console:
                self.console.print(f"❌ {error_msg}", style="red")
            else:
                print(f"❌ {error_msg}")
    
    def run(self):
        """Main CLI execution."""
        try:
            # Print header
            self.print_header()
            
            # Get user configuration
            config = self.get_user_input()
            
            # Validate configuration
            if not self.validate_config(config):
                return
            
            # Show configuration summary
            if self.console:
                self.console.print("\n🔧 Configuration Summary:", style="bold")
                config_table = Table()
                config_table.add_column("Setting", style="cyan")
                config_table.add_column("Value", style="green")
                
                for key, value in config.items():
                    config_table.add_row(key.replace('_', ' ').title(), str(value))
                
                self.console.print(config_table)
            else:
                print("\n🔧 Configuration Summary:")
                for key, value in config.items():
                    print(f"  {key.replace('_', ' ').title()}: {value}")
            
            # Confirm before proceeding
            if QUESTIONARY_AVAILABLE:
                if not questionary.confirm("Proceed with training?", default=True).ask():
                    return
            elif RICH_AVAILABLE:
                if not Confirm.ask("\nProceed with training?"):
                    return
            else:
                if input("\nProceed with training? (Y/n): ").lower() == 'n':
                    return
            
            # Start training
            start_time = time.time()
            
            if config['compare_models']:
                # Train multiple models
                results = self.run_model_comparison(config)
            else:
                # Train single model
                results = self.run_single_model_training(config)
            
            # Calculate training time
            training_time = time.time() - start_time
            
            # Generate report if requested
            if config['generate_report']:
                self.generate_training_report(results, config)
            
            # Print completion summary
            if self.console:
                self.console.print(f"\n🎉 Training completed in {training_time:.1f} seconds!", style="bold green")
            else:
                print(f"\n🎉 Training completed in {training_time:.1f} seconds!")
            
        except KeyboardInterrupt:
            if self.console:
                self.console.print("\n❌ Training interrupted by user", style="red")
            else:
                print("\n❌ Training interrupted by user")
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg)
            if self.console:
                self.console.print(f"\n❌ {error_msg}", style="red")
            else:
                print(f"\n❌ {error_msg}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Job Fraud Detection Model Training CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--data', '-d',
        type=str,
        help='Path to training data CSV file (or use --dataset for predefined options)'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='auto',
        help='Dataset name (filename without .csv) or "auto" to auto-detect best dataset. Available datasets will be discovered automatically from data/raw and data/processed directories.'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        choices=['random_forest', 'logistic_regression', 'svm', 'naive_bayes', 'all_models'],
        help='Model type to train (use all_models to train all types)'
    )
    
    parser.add_argument(
        '--balance', '-b',
        type=str,
        choices=['smote', 'oversample', 'undersample', 'none'],
        default='smote',
        help='Class balancing method'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='models',
        help='Output directory for trained models'
    )
    
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Train and compare all model types'
    )
    
    parser.add_argument(
        '--no-interactive',
        action='store_true',
        help='Run in non-interactive mode using provided arguments'
    )
    
    args = parser.parse_args()
    
    if args.no_interactive:
        # Non-interactive mode
        # Determine data path from --data or --dataset option
        if args.data:
            data_path = args.data
        else:
            # Use dynamic dataset discovery
            cli = JobFraudTrainingCLI()
            try:
                data_path = cli._get_dataset_path(args.dataset)
            except FileNotFoundError as e:
                print(f"❌ {e}")
                sys.exit(1)
        
        if not os.path.exists(data_path):
            print(f"❌ Data file not found: {data_path}")
            sys.exit(1)
        
        # Handle 'all_models' selection in non-interactive mode
        if args.model == 'all_models':
            compare_models = True
            model_type = 'random_forest'  # Default for single model path
        else:
            compare_models = args.compare
            model_type = args.model or 'random_forest'
        
        config = {
            'data_path': data_path,
            'model_type': model_type,
            'balance_method': args.balance,
            'output_dir': args.output,
            'compare_models': compare_models,
            'generate_report': True,
            'dataset_type': args.dataset if not args.data else 'custom'
        }
        
        # Initialize CLI and run with config
        cli = JobFraudTrainingCLI()
        
        # Validate config
        if not cli.validate_config(config):
            sys.exit(1)
        
        # Run training
        if config['compare_models']:
            results = cli.run_model_comparison(config)
        else:
            results = cli.run_single_model_training(config)
        
        # Generate report
        cli.generate_training_report(results, config)
        
    else:
        # Interactive mode
        cli = JobFraudTrainingCLI()
        cli.run()


if __name__ == "__main__":
    main()