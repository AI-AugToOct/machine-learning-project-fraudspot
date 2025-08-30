"""
Services Module - Application Service Layer
This module provides high-level services that orchestrate core business logic.

Services:
- ScrapingService: Unified interface for all scraping operations
- ModelService: Model management and lifecycle operations
- SerializationService: Data format conversion and serialization

Version: 3.0.0 - DRY Consolidation
"""

from .model_service import ModelService
from .scraping_service import ScrapingService
from .serialization_service import SerializationService

__all__ = [
    'ScrapingService',
    'ModelService', 
    'SerializationService'
]