"""
Scraping Service - SINGLE INTERFACE
This service provides a unified interface for all scraping operations.
Coordinates between different scrapers without duplicating business logic.

Version: 1.0.0 - DRY Consolidation
"""

import logging
from typing import Any, Dict, Optional

from ..core import DataConstants, ScrapingConstants

logger = logging.getLogger(__name__)


class ScrapingService:
    """
    SINGLE INTERFACE for all scraping operations.
    
    This service orchestrates scraping without containing business logic.
    All data processing is delegated to core modules.
    
    Implements singleton pattern to prevent duplicate API calls.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """Implement singleton pattern to prevent multiple instances."""
        if cls._instance is None:
            cls._instance = super(ScrapingService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the scraping service (only once)."""
        if not ScrapingService._initialized:
            self.scrapers = {}
            logger.info("ScrapingService initialized (singleton)")
            ScrapingService._initialized = True
    
    def scrape_job_posting(self, url: str, method: str = 'auto') -> Dict[str, Any]:
        """
        Scrape job posting using specified method.
        
        Args:
            url: LinkedIn job URL to scrape
            method: Scraping method ('auto', 'bright_data', 'html')
            
        Returns:
            Dict: Standardized job posting data ready for ML pipeline
        """
        logger.info(f"Scraping job posting: {url}")
        
        try:
            # Validate URL
            if not self._validate_url(url):
                return self._create_error_result("Invalid LinkedIn URL format", url)
            
            # Determine scraping method
            if method == 'auto':
                method = self._select_best_method()
            
            # Perform scraping
            raw_result = self._scrape_with_method(url, method)
            
            # Validate result
            if not raw_result.get('success', False):
                return raw_result  # Return error as-is
            
            # Return raw data directly to UI (ML preparation happens later in analysis phase)
            raw_result.update({
                'scraping_method': method,
                'scraping_success': True,
                'data_source': 'scraping_service',
                'url': url
            })
            
            logger.info("Job scraping completed successfully")
            return raw_result
            
        except Exception as e:
            logger.error(f"Scraping failed: {str(e)}")
            return self._create_error_result(str(e), url)
    
    def scrape_profile(self, profile_url: str) -> Dict[str, Any]:
        """
        Scrape LinkedIn profile separately.
        
        Args:
            profile_url: LinkedIn profile URL to scrape
            
        Returns:
            Dict: Profile data for display
        """
        logger.info(f"Scraping profile: {profile_url}")
        
        try:
            # Import profile scraper
            from ..scraper.linkedin_scraper import scrape_profile
            return scrape_profile(profile_url)
            
        except Exception as e:
            logger.error(f"Profile scraping failed: {str(e)}")
            return self._create_error_result(str(e), profile_url)
    
    def _validate_url(self, url: str) -> bool:
        """Validate LinkedIn job URL format."""
        if not url or not isinstance(url, str):
            return False
        
        # Use constants from core module
        import re
        for pattern in ScrapingConstants.LINKEDIN_URL_PATTERNS:
            if re.match(pattern, url):
                return True
        
        # Additional basic validation
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return 'linkedin.com' in parsed.netloc and 'jobs' in url
    
    def _select_best_method(self) -> str:
        """Select the best available scraping method."""
        # Priority: Bright Data > HTML fallback
        try:
            # Check if Bright Data is configured
            from ..core.constants import get_bright_data_config
            config = get_bright_data_config()
            if config.get('api_key'):
                return 'bright_data'
        except ImportError:
            pass
        
        # Fallback to HTML method
        return 'html'
    
    def _scrape_with_method(self, url: str, method: str) -> Dict[str, Any]:
        """Perform scraping with specified method."""
        if method == 'bright_data':
            return self._scrape_with_bright_data(url)
        elif method == 'html':
            return self._scrape_with_html_fallback(url)
        else:
            raise ValueError(f"Unknown scraping method: {method}")
    
    def _scrape_with_bright_data(self, url: str) -> Dict[str, Any]:
        """Scrape using Bright Data API."""
        try:
            # Import new scraper functions (avoid circular imports)
            from ..scraper.linkedin_scraper import scrape_job
            return scrape_job(url)
        except Exception as e:
            logger.error(f"Bright Data scraping failed: {str(e)}")
            return {'success': False, 'error': str(e), 'scraping_method': 'bright_data_failed'}
    
    def _scrape_with_html_fallback(self, url: str) -> Dict[str, Any]:
        """Fallback to HTML-based scraping."""
        logger.warning("Using HTML fallback method - limited data available")
        return {
            'success': False,
            'error': 'HTML fallback not implemented in service layer',
            'scraping_method': 'html_not_available'
        }
    
    def _create_error_result(self, error_message: str, url: str = None) -> Dict[str, Any]:
        """Create standardized error result."""
        return {
            'success': False,
            'error': error_message,
            'scraping_method': 'failed',
            'url': url,
            'data_source': 'scraping_service_error'
        }
    
    def validate_scraped_data(self, data: Dict[str, Any]) -> Dict[str, bool]:
        """
        Validate scraped data quality and completeness.
        
        Args:
            data: Scraped job data
            
        Returns:
            Dict: Validation results
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'completeness_score': 0.0
        }
        
        # Check essential fields
        essential_fields = DataConstants.ESSENTIAL_FIELDS
        missing_essential = []
        present_essential = 0
        
        for field in essential_fields:
            if field not in data or not data[field] or str(data[field]).strip() == '':
                missing_essential.append(field)
            else:
                present_essential += 1
        
        # Calculate completeness
        validation_result['completeness_score'] = present_essential / len(essential_fields)
        
        # Add errors for missing critical fields
        if missing_essential:
            validation_result['errors'].append(f"Missing essential fields: {missing_essential}")
            if len(missing_essential) >= len(essential_fields) // 2:
                validation_result['is_valid'] = False
        
        # Check data quality
        if 'job_description' in data:
            desc_length = len(str(data['job_description']))
            if desc_length < 50:
                validation_result['warnings'].append("Very short job description")
            elif desc_length > 10000:
                validation_result['warnings'].append("Extremely long job description")
        
        # Check for data integrity
        if 'company_name' in data and not data['company_name']:
            validation_result['warnings'].append("No company name provided")
        
        logger.info(f"Data validation: {'PASSED' if validation_result['is_valid'] else 'FAILED'} "
                   f"(completeness: {validation_result['completeness_score']:.1%})")
        
        return validation_result
    
    def get_scraping_stats(self) -> Dict[str, Any]:
        """Get scraping performance statistics."""
        # This would typically track success rates, response times, etc.
        return {
            'service_status': 'active',
            'available_methods': ['bright_data', 'html'],
            'preferred_method': self._select_best_method()
        }


# Export main class
__all__ = ['ScrapingService']