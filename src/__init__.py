"""
Job Post Fraud Detector - Source Code Package

This package contains all the core modules for the job fraud detection system:
- scraper: LinkedIn job posting scraping functionality
- features: Feature extraction and engineering
- models: Machine learning model training and prediction
- utils: Utility functions and helpers

Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Tuwaiq ML Bootcamp"
__description__ = "Job Post Fraud Detector - ML-powered fraud detection for job postings"

# Package-level imports for easy access
from .config import *

__all__ = [
    '__version__',
    '__author__',
    '__description__',
]