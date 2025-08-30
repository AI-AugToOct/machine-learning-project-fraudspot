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
    """Generate a model comparison report."""
    report = "# Model Comparison Report\n\n"
    report += "| Model | Accuracy | F1 Score | Precision | Recall |\n"
    report += "|-------|----------|----------|-----------|--------|\n"
    
    for model_name, results in model_results.items():
        metrics = results.get('metrics', {})
        report += f"| {model_name} | {metrics.get('accuracy', 0):.4f} | "
        report += f"{metrics.get('f1_score', 0):.4f} | "
        report += f"{metrics.get('precision', 0):.4f} | "
        report += f"{metrics.get('recall', 0):.4f} |\n"
    
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
        
        self.dataset_options = {
            'english': '🇺🇸 English Dataset (17K samples)',
            'arabic': '🇸🇦 Arabic Dataset (2K samples)',
            'combined': '🌍 Combined Multilingual Dataset (19K+ samples)',
            'auto': '🤖 Auto-detect and Combine Available Datasets'
        }
        
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
            choice = input(f"Select dataset (1-{len(self.dataset_options)}) [4]: ").strip() or "4"
            dataset_keys = list(self.dataset_options.keys())
            dataset_choice = dataset_keys[int(choice) - 1]
        
        # Set data path based on choice
        if dataset_choice == 'english':
            config['data_path'] = "data/raw/fake_job_postings.csv"
        elif dataset_choice == 'arabic':
            config['data_path'] = "data/raw/mergedFakeWithRealData.csv"
        elif dataset_choice == 'combined':
            # Create combined dataset
            combined_path = "data/processed/multilingual_job_fraud_data.csv"
            if not os.path.exists(combined_path):
                try:
                    if self.console:
                        self.console.print("🔄 Creating combined multilingual dataset...", style="blue")
                    else:
                        print("🔍 Using existing multilingual dataset...")
                        # Use existing multilingual dataset
                        multilingual_path = "data/processed/multilingual_job_fraud_data.csv"
                        if os.path.exists(multilingual_path):
                            combined_path = multilingual_path
                            if self.console:
                                self.console.print("✅ Using existing multilingual dataset", style="green")
                            else:
                                print("✅ Using existing multilingual dataset")
                except Exception as e:
                    if self.console:
                        self.console.print(f"❌ Error creating combined dataset: {e}", style="red")
                    else:
                        print(f"❌ Error creating combined dataset: {e}")
                    # Fallback to Arabic dataset
                    combined_path = "data/raw/mergedFakeWithRealData.csv"
            config['data_path'] = combined_path
        else:  # auto
            # Auto-detect available datasets and combine
            english_path = "data/raw/fake_job_postings.csv"
            arabic_path = "data/raw/mergedFakeWithRealData.csv"
            combined_path = "data/processed/multilingual_job_fraud_data.csv"

            # Check which datasets are available
            available_datasets = []
            if os.path.exists(english_path):
                available_datasets.append("English")
            if os.path.exists(arabic_path):
                available_datasets.append("Arabic")
            
            if len(available_datasets) > 1:
                # Combine datasets
                if self.console:
                    self.console.print(f"🔍 Found {', '.join(available_datasets)} datasets. Creating combined dataset...", style="blue")
                else:
                    print(f"🔍 Found {', '.join(available_datasets)} datasets. Creating combined dataset...")
                
                try:
                    # Use existing multilingual dataset
                    multilingual_path = "data/processed/multilingual_job_fraud_data.csv"
                    if os.path.exists(multilingual_path):
                        config['data_path'] = multilingual_path
                except Exception as e:
                    # Fallback to largest available dataset
                    config['data_path'] = english_path if os.path.exists(english_path) else arabic_path
            else:
                # Use single available dataset
                config['data_path'] = english_path if os.path.exists(english_path) else arabic_path
        
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
            
            # Calculate training time
            training_time = time.time() - start_time
            
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
                # Show key metrics
                test_f1 = metrics.get('f1_score', 0)
                test_acc = metrics.get('accuracy', 0) 
                self.console.print(f"📊 F1-Score: {test_f1:.3f}, Accuracy: {test_acc:.3f}", style="cyan")
            else:
                print(f"✅ {config['model_type']} pipeline trained and saved successfully!")
                test_f1 = metrics.get('f1_score', 0)
                test_acc = metrics.get('accuracy', 0)
                print(f"📊 F1-Score: {test_f1:.3f}, Accuracy: {test_acc:.3f}")
            
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
                f1_score = metrics.get('f1_score', 0)
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
        choices=['english', 'arabic', 'combined', 'auto'],
        default='auto',
        help='Predefined dataset option: english (17K), arabic (2K), combined (19K+), or auto-detect'
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
            # Use dataset option to determine path
            dataset_choice = args.dataset
            if dataset_choice == 'english':
                data_path = "data/raw/fake_job_postings.csv"
            elif dataset_choice == 'arabic':
                data_path = "data/raw/mergedFakeWithRealData.csv"
            elif dataset_choice == 'combined':
                data_path = "data/processed/multilingual_job_fraud_data.csv"
                # Create combined dataset if it doesn't exist
                if not os.path.exists(data_path):
                    try:
                        print("🔍 Using existing multilingual dataset...")
                        multilingual_path = "data/processed/multilingual_job_fraud_data.csv"
                        if os.path.exists(multilingual_path):
                            data_path = multilingual_path
                            print("✅ Using existing multilingual dataset")
                    except Exception as e:
                        print(f"❌ Error creating combined dataset: {e}")
                        data_path = "data/raw/mergedFakeWithRealData.csv"
            else:  # auto
                # Auto-detect and combine if possible
                english_path = "data/raw/fake_job_postings.csv"
                arabic_path = "data/raw/mergedFakeWithRealData.csv"
                combined_path = "data/processed/multilingual_job_fraud_data.csv"

                if os.path.exists(english_path) and os.path.exists(arabic_path):
                    try:
                        print("🔍 Found both datasets. Using existing multilingual dataset...")
                        multilingual_path = "data/processed/multilingual_job_fraud_data.csv"
                        if os.path.exists(multilingual_path):
                            data_path = multilingual_path
                    except Exception as e:
                        print(f"❌ Error creating combined dataset: {e}")
                        data_path = english_path  # Fallback to larger dataset
                else:
                    data_path = english_path if os.path.exists(english_path) else arabic_path
        
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