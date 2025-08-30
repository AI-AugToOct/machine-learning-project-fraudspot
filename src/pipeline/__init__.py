"""
Pipeline module for job fraud detection system.

This module provides the main PipelineManager class that orchestrates
the complete machine learning pipeline from data loading to model deployment.
"""

from .pipeline_manager import PipelineManager

__all__ = ['PipelineManager']