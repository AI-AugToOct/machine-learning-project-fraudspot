"""
Scraping Service - CONTENT-FOCUSED INTERFACE
This service provides a unified interface for job and company scraping only.
Profile scraping removed - focuses on content and company-based fraud detection.

Version: 2.0.0 - Profile-Free Implementation
"""

import logging
from typing import Any, Dict, Optional

from ..core import DataConstants, ScrapingConstants

logger = logging.getLogger(__name__)


class ScrapingService:
    """
    Content-focused scraping service for jobs and company data only.
    
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
            logger.info("ScrapingService initialized (singleton) - Content-focused version")
            ScrapingService._initialized = True
    
    def scrape_job_posting(self, url: str, method: str = 'auto') -> Dict[str, Any]:
        """
        Scrape job posting with company enrichment (no profile data).
        
        Args:
            url: LinkedIn job URL to scrape
            method: Scraping method ('auto', 'bright_data', 'html')
            
        Returns:
            Dict: Standardized job posting data ready for ML pipeline
        """
        logger.info(f"Scraping job posting (content-focused): {url}")
        
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
            
            # Map scraped field names to model-expected field names
            field_mappings = {
                'job_title': 'job_title',
                'company_name': 'company_name',
                'job_summary': 'description',
                'job_description': 'description',
                'job_location': 'location',
                'job_employment_type': 'employment_type',
                'job_seniority_level': 'experience_level',
                'job_industries': 'industry',
                'job_posted_time': 'posted_time',
                'job_num_applicants': 'applicants_count'
            }
            
            # Apply field name mappings
            for scraped_field, model_field in field_mappings.items():
                if scraped_field in raw_result:
                    raw_result[model_field] = raw_result[scraped_field]
                    logger.debug(f"🔄 MAPPED FIELD: {scraped_field} → {model_field}")
            
            # Extract additional job details for complete mapping (no profile fields)
            raw_result.update({
                'applicants_count': raw_result.get('job_num_applicants'),
                'posted_time': raw_result.get('job_posted_time'),
                'company_logo': raw_result.get('company_logo'),
                'job_posting_id': raw_result.get('job_posting_id'),
                'company_id': raw_result.get('company_id'),
                'base_salary': raw_result.get('base_salary', {}),
                'apply_link': raw_result.get('apply_link'),
                
                # Additional fields that might be present in scraped data
                'has_questions': 1 if raw_result.get('has_questions') else 0,
                'has_logo': 1 if raw_result.get('company_logo') else 0,
                'has_apply_link': 1 if raw_result.get('apply_link') else 0
            })
            logger.debug("🔄 MAPPED JOB FIELDS for content-based analysis")
            
            raw_result.update({
                'scraping_method': method,
                'scraping_success': True,
                'data_source': 'scraping_service',
                'url': url,
                'content_focused': True  # Mark as content-focused version
            })
            
            logger.info("Job scraping completed successfully (content-focused)")
            return raw_result
            
        except Exception as e:
            logger.error(f"Scraping failed: {str(e)}")
            return self._create_error_result(str(e), url)
    
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
        else:
            raise ValueError(f"Unknown scraping method: {method}. Only 'bright_data' is supported.")
    
    def _scrape_with_bright_data(self, url: str) -> Dict[str, Any]:
        """Scrape using Bright Data API with company enrichment only."""
        try:
            # Import scraper functions (avoid circular imports)
            from ..scraper.linkedin_scraper import scrape_company, scrape_job
            
            # Step 1: Scrape job data
            job_result = scrape_job(url)
            
            if not job_result.get('success'):
                return job_result
            
            # Step 2: Extract company URL and fetch company data
            company_url = job_result.get('company_url')
            if company_url:
                logger.info(f"Fetching company data from: {company_url}")
                
                try:
                    company_result = scrape_company(company_url)
                    
                    if company_result.get('success'):
                        # Map company data structure to job result
                        company_followers = company_result.get('company_followers', None)
                        company_employees = company_result.get('company_employees', None)
                        company_founded = company_result.get('company_founded', None)
                        company_size = company_result.get('company_size', None)
                        company_website = company_result.get('company_website', None)
                        company_legitimacy_score = company_result.get('company_legitimacy_score', None)
                        
                        # Extract additional company details for UI display
                        company_name = company_result.get('company_name', None)
                        industries = company_result.get('industries', None)
                        
                        job_result.update({
                            'company_followers': company_followers,
                            'company_employees': company_employees,
                            'company_founded': company_founded,
                            'company_size': company_size,
                            'company_website': company_website,
                            'company_legitimacy_score': company_legitimacy_score,
                            'industries': industries,
                            'company_enrichment_success': True,
                            
                            # Add direct mapping for data model
                            'employees_in_linkedin': company_employees,
                            'founded': company_founded,
                            'followers': company_followers,
                            
                            # Calculate company indicators from real data
                            'has_company_website': 1 if company_website else 0,
                            'has_company_size': 1 if company_size else 0,
                            'has_company_founded': 1 if company_founded else 0
                        })
                        logger.info(f"✅ COMPANY DATA: followers={company_followers}, employees={company_employees}, founded={company_founded}")
                    else:
                        logger.warning(f"Company scraping failed: {company_result.get('error')}")
                        job_result.update({
                            'company_followers': None,
                            'company_employees': None,
                            'company_founded': None,
                            'company_legitimacy_score': None,
                            'company_enrichment_success': False,
                            'company_enrichment_error': company_result.get('error')
                        })
                        
                except Exception as company_error:
                    logger.warning(f"Company scraping error: {str(company_error)}")
                    job_result.update({
                        'company_followers': None,
                        'company_employees': None,
                        'company_founded': None,
                        'company_legitimacy_score': None,
                        'company_enrichment_success': False,
                        'company_enrichment_error': str(company_error)
                    })
            else:
                logger.warning("No company URL found in job data")
                job_result.update({
                    'company_followers': None,
                    'company_employees': None,
                    'company_founded': None,
                    'company_legitimacy_score': None,
                    'company_enrichment_success': False,
                    'company_enrichment_error': 'No company URL found'
                })
            
            return job_result
            
        except Exception as e:
            logger.error(f"Bright Data scraping failed: {str(e)}")
            return {'success': False, 'error': str(e), 'scraping_method': 'bright_data_failed'}
    
    def _create_error_result(self, error_message: str, url: str = None) -> Dict[str, Any]:
        """Create standardized error result."""
        return {
            'success': False,
            'error': error_message,
            'scraping_method': 'failed',
            'url': url,
            'data_source': 'scraping_service_error',
            'content_focused': True
        }
    
    def validate_scraped_data(self, data: Dict[str, Any]) -> Dict[str, bool]:
        """
        Validate scraped data quality and completeness for content-focused analysis.
        
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
        
        # Check essential fields for content-based analysis
        content_essential_fields = [
            'job_title', 'company_name', 'job_description', 'location'
        ]
        
        missing_essential = []
        present_essential = 0
        
        for field in content_essential_fields:
            if field not in data or not data[field] or str(data[field]).strip() == '':
                missing_essential.append(field)
            else:
                present_essential += 1
        
        # Calculate completeness
        validation_result['completeness_score'] = present_essential / len(content_essential_fields)
        
        # Add errors for missing critical fields
        if missing_essential:
            validation_result['errors'].append(f"Missing essential content fields: {missing_essential}")
            if len(missing_essential) >= len(content_essential_fields) // 2:
                validation_result['is_valid'] = False
        
        # Check content quality
        if 'job_description' in data:
            desc_length = len(str(data['job_description']))
            if desc_length < 50:
                validation_result['warnings'].append("Very short job description")
            elif desc_length > 10000:
                validation_result['warnings'].append("Extremely long job description")
        
        # Check for company data integrity
        if 'company_name' in data and not data['company_name']:
            validation_result['warnings'].append("No company name provided")
        
        logger.info(f"Content validation: {'PASSED' if validation_result['is_valid'] else 'FAILED'} "
                   f"(completeness: {validation_result['completeness_score']:.1%})")
        
        return validation_result
    
    def get_scraping_stats(self) -> Dict[str, Any]:
        """Get scraping performance statistics."""
        return {
            'service_status': 'active',
            'available_methods': ['bright_data', 'html'],
            'preferred_method': self._select_best_method(),
            'content_focused': True,
            'profile_scraping_enabled': False
        }


# Export main class
__all__ = ['ScrapingService']