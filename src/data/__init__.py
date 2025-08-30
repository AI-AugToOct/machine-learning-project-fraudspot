"""

Version: 3.0.0 - DRY Consolidation
IMPORTANT: This module now serves as a bridge to core modules.
"""

# Import from core modules (single source of truth)
from ..core import DataProcessor, FeatureEngine
from .data_loader import load_training_data


__all__ = [
    'load_training_data',
    'DataProcessor',
    'FeatureEngine',
]