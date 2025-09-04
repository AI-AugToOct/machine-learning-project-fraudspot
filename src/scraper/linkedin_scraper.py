"""
LinkedIn Job & Company Scraper - Focused Implementation

This module provides functionality for scraping LinkedIn job postings and company data
using Bright Data's professional APIs for fraud detection based on content and company metrics.

Key Features:
- Complete job posting extraction with structured data points
- Company verification and legitimacy scoring
- Content-based fraud detection algorithms
- Professional API integration with structured JSON data
- Real-time data with no HTML parsing overhead

Note: Profile scraping removed - fraud detection now focuses on job content and company data only.

Version: 5.0.0 - Profile-Free Implementation
"""

import json
import logging
import os
import re
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from urllib.parse import quote, urlparse

import requests

# Import configuration
from ..core.constants import BrightDataConstants, ScrapingConstants, get_bright_data_config

logger = logging.getLogger(__name__)


class BrightDataLinkedInScraper:
    """
    LinkedIn scraper for job and company data only (no profile scraping).
    
    This class provides structured access to LinkedIn job and company data through 
    Bright Data's official APIs, focusing on content-based fraud detection.
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize the Bright Data LinkedIn scraper.
        
        Args:
            api_key (str, optional): Bright Data API key. Uses config if not provided.
        """
        self.api_key = api_key or BrightDataConstants.CONFIG['api_key']
        self.config = BrightDataConstants.CONFIG
        
        # Validate API key FIRST before setting up any session headers
        if not self.api_key or self.api_key.strip() == '':
            raise ValueError("Bright Data API key is required. Please set BD_API_KEY environment variable or pass api_key parameter.")
        
        self.session = requests.Session()
        
        # Set up authentication headers (only after API key validation passes)
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'User-Agent': 'FraudSpot-Detector/1.0'
        })
        
        # Use correct trigger endpoint and dataset IDs from config
        self.trigger_endpoint = self.config['trigger_endpoint']
        self.dataset_ids = self.config['dataset_ids']
        
        # Rate limiting configuration
        self.rate_limit_delay = self.config['rate_limit_delay']
        self.last_request_time = 0
        
        # Validate configuration
        if not self.trigger_endpoint:
            raise ValueError("Bright Data trigger endpoint not configured properly")
        
        if not self.dataset_ids.get('jobs'):
            raise ValueError("Bright Data jobs dataset ID not configured properly")
        
        logger.info(f"🚀 Bright Data LinkedIn Scraper initialized successfully")
        logger.info(f"📡 Endpoint: {self.trigger_endpoint}")
        logger.info(f"📊 Jobs Dataset ID: {self.dataset_ids['jobs']}")

    def _wait_for_rate_limit(self):
        """Implement rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()

    def _make_request(self, dataset_id: str, url: str, timeout: int = 30) -> Dict[str, Any]:
        """
        Make a request to Bright Data API with proper error handling and timeout.
        
        Args:
            dataset_id (str): Bright Data dataset ID for the LinkedIn data type
            url (str): LinkedIn URL to scrape
            timeout (int): Request timeout in seconds
            
        Returns:
            Dict[str, Any]: API response data or error information
        """
        self._wait_for_rate_limit()
        
        # Correct Bright Data API format: array payload with dataset_id as URL parameter
        payload = [{"url": url}]
        api_url = f"{self.trigger_endpoint}?dataset_id={dataset_id}"
        
        try:
            logger.info(f"🔄 Making Bright Data API request to: {api_url}")
            logger.info(f"📊 Dataset: {dataset_id}, URL: {url}")
            
            response = self.session.post(
                api_url,
                json=payload,
                timeout=timeout
            )
            
            logger.info(f"📡 Response Status: {response.status_code}")
            
            if response.status_code == 200:
                response_data = response.json()
                logger.info(f"✅ Bright Data API request successful")
                
                # Handle array response format - extract first item for jobs
                if isinstance(response_data, list) and response_data:
                    data = response_data[0]
                elif isinstance(response_data, dict):
                    data = response_data
                else:
                    logger.warning("⚠️ Unexpected response format from Bright Data API")
                    data = response_data
                
                return {
                    'success': True,
                    'data': data,
                    'status_code': response.status_code
                }
            elif response.status_code == 202:
                # 202 Accepted - Async processing with snapshot_id
                response_data = response.json()
                logger.info(f"🔄 Bright Data API request accepted for async processing")
                
                return {
                    'success': True,
                    'data': response_data,  # Contains snapshot_id
                    'status_code': response.status_code
                }
            else:
                error_message = f"HTTP {response.status_code}: {response.text}"
                logger.error(f"❌ Bright Data API error: {error_message}")
                return {
                    'success': False,
                    'error': error_message,
                    'status_code': response.status_code
                }
                
        except requests.exceptions.Timeout:
            error_message = f"Request timeout after {timeout} seconds"
            logger.error(f"⏰ {error_message}")
            return {'success': False, 'error': error_message, 'status_code': 408}
            
        except requests.exceptions.RequestException as e:
            error_message = f"Request failed: {str(e)}"
            logger.error(f"🔌 {error_message}")
            return {'success': False, 'error': error_message, 'status_code': 0}
        
        except json.JSONDecodeError as e:
            error_message = f"Invalid JSON response: {str(e)}"
            logger.error(f"📝 {error_message}")
            return {'success': False, 'error': error_message, 'status_code': response.status_code}
        
        except Exception as e:
            error_message = f"Unexpected error: {str(e)}"
            logger.error(f"💥 {error_message}")
            return {'success': False, 'error': error_message, 'status_code': 500}

    def _extract_job_data(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and structure job posting data only (no profile data)."""
        structured = {}
        
        # Job information - Map actual API field names
        structured.update({
            'job_title': job_data.get('job_title', ''),
            'company_name': job_data.get('company_name', ''),
            'job_description': job_data.get('job_description', '') or job_data.get('job_summary', ''),
            'location': job_data.get('job_location', ''),  # API uses job_location
            'salary_info': job_data.get('base_salary'),  # API uses base_salary
            'requirements': job_data.get('requirements', []),
            'posted_date': job_data.get('job_posted_date'),  # API uses job_posted_date
            'applicant_count': job_data.get('job_num_applicants'),  # API uses job_num_applicants
            'application_method': job_data.get('application_method', ''),
            'job_type': job_data.get('job_employment_type'),  # API uses job_employment_type
            'experience_level': job_data.get('job_seniority_level'),  # API uses job_seniority_level
            'industry': job_data.get('job_industries'),  # API uses job_industries
            'company_url': job_data.get('company_url'),  # CRITICAL: Keep original for enrichment
            'company_website': job_data.get('company_url'),  # Also store as website for compatibility
            'company_size': job_data.get('company_size'),
            'job_function': job_data.get('job_function'),
            'seniority_level': job_data.get('job_seniority_level'),
        })
        
        # Preserve original API field names for UI compatibility
        structured.update({
            'company_logo': job_data.get('company_logo'),
            'job_num_applicants': job_data.get('job_num_applicants'),
            'job_posted_date': job_data.get('job_posted_date'),
            'job_posted_time': job_data.get('job_posted_time'),
            'application_count': job_data.get('job_num_applicants'),  # Alias for UI
            'posted_date': job_data.get('job_posted_date'),  # Alias for UI
        })
        
        # Contact information
        contact_info = job_data.get('contact_info', {})
        structured.update({
            'contact_info': contact_info,
            'has_email': len(contact_info.get('emails', [])) > 0,
            'has_phone': len(contact_info.get('phones', [])) > 0,
            'has_whatsapp': len(contact_info.get('whatsapp', [])) > 0,
            'has_telegram': len(contact_info.get('telegram', [])) > 0,
        })
        
        return structured

    def _calculate_content_based_scores(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate fraud detection scores based only on job content and company data."""
        scores = {}
        
        # 1. Company Legitimacy Score (0-1) - Enhanced focus on company data
        company_score = 0.0
        
        if data.get('company_website'):
            company_score += 0.3
        if data.get('company_size'):
            company_score += 0.25
        if data.get('industry'):
            company_score += 0.25
        if data.get('company_name') and len(data.get('company_name', '')) > 2:
            company_score += 0.2
            
        scores['company_legitimacy_score'] = min(company_score, 1.0)
        
        # 2. Content Quality Score (0-1) - Based on job posting completeness
        content_factors = [
            bool(data.get('job_title') and len(data.get('job_title', '')) > 3),
            bool(data.get('job_description') and len(data.get('job_description', '')) > 50),
            bool(data.get('location')),
            bool(data.get('experience_level')),
            bool(data.get('requirements')),
            bool(data.get('salary_info')),
            bool(data.get('posted_date')),
            bool(data.get('company_name'))
        ]
        
        scores['content_quality_score'] = sum(content_factors) / len(content_factors)
        
        # 3. Contact Risk Score (0-1) - Higher is more suspicious
        contact_risk = 0.0
        
        if data.get('has_whatsapp'):
            contact_risk += 0.3
        if data.get('has_telegram'):
            contact_risk += 0.3
        if not data.get('has_email') and (data.get('has_whatsapp') or data.get('has_telegram')):
            contact_risk += 0.4  # Suspicious if only messaging apps
            
        scores['contact_risk_score'] = min(contact_risk, 1.0)
        
        # 4. Overall Fraud Risk Score (0-1, lower is better)
        risk_factors = []
        
        # High risk factors based on content only
        if not data.get('company_name'):
            risk_factors.append(0.4)
        if not data.get('job_description') or len(data.get('job_description', '')) < 50:
            risk_factors.append(0.3)
        if data.get('has_whatsapp') or data.get('has_telegram'):
            risk_factors.append(0.2)
        if not data.get('company_website'):
            risk_factors.append(0.1)
            
        fraud_risk = sum(risk_factors)
        scores['fraud_risk_score'] = min(fraud_risk, 1.0)
        
        return scores

    def _create_error_response(self, error: str, url: str = None) -> Dict[str, Any]:
        """Create standardized error response."""
        return {
            'success': False,
            'error': error,
            'url': url,
            'scraped_at': datetime.now().isoformat(),
            'scraping_method': 'bright_data_failed',
            'data_source': 'bright_data_professional',
            'api_version': '5.0.0'
        }


# MAIN PUBLIC FUNCTIONS

def scrape_job(url: str) -> Dict[str, Any]:
    """
    Scrape job data with company information for content-based fraud detection.
    
    Args:
        url (str): The LinkedIn job posting URL to scrape
        
    Returns:
        Dict[str, Any]: Dictionary containing job information:
            - Complete job details (title, description, requirements, salary)
            - Company information (size, industry, verification)
            - Content-based fraud detection scores
    """
    logger.info(f"Starting job scraping: {url}")
    
    try:
        # Validate LinkedIn URL
        if not validate_linkedin_url(url):
            raise ValueError(f"Invalid LinkedIn URL format: {url}")
        
        # Get Bright Data API key
        config = get_bright_data_config()
        api_key = config.get('api_key', '')
        
        if not api_key:
            raise ValueError("Bright Data API key not configured. Please set BD_API_KEY environment variable.")
        
        # Create scraper and get job data only
        scraper = BrightDataLinkedInScraper(api_key)
        
        job_result = scraper._make_request(
            dataset_id=scraper.dataset_ids['jobs'], 
            url=url,
            timeout=120
        )
        
        if not job_result.get('success'):
            logger.error(f"❌ Job scraping failed: {job_result.get('error')}")
            return scraper._create_error_response(
                error=job_result.get('error', 'Job scraping failed'),
                url=url
            )
        
        # Extract and structure job data
        job_data = job_result.get('data', {})
        
        # Structure job data (no profile processing)
        structured_data = scraper._extract_job_data(job_data)
        
        # Calculate content-based fraud scores
        fraud_scores = scraper._calculate_content_based_scores(structured_data)
        structured_data.update(fraud_scores)
        
        # Add metadata
        structured_data.update({
            'success': True,
            'scraped_at': datetime.now().isoformat(),
            'scraping_method': 'bright_data_content_only',
            'data_source': 'bright_data_professional',
            'api_version': '5.0.0'
        })
        
        logger.info(f"✅ Job scraping successful!")
        logger.info(f"📊 Job: '{structured_data.get('job_title')}' at '{structured_data.get('company_name')}'")
        
        return structured_data
        
    except Exception as e:
        error_msg = f"Job scraping error: {str(e)}"
        logger.error(error_msg)
        
        return {
            'success': False,
            'error_message': error_msg,
            'scraping_method': 'bright_data_job_error',
            'url': url,
            'scraped_at': datetime.now().isoformat(),
            'api_version': '5.0.0',
            'data_source': 'bright_data_professional'
        }


def validate_linkedin_url(url: str) -> bool:
    """
    Validate that the URL is a proper LinkedIn job posting URL.
    
    Args:
        url (str): URL to validate
        
    Returns:
        bool: True if valid LinkedIn job URL, False otherwise
    """
    if not url or not isinstance(url, str):
        return False
    
    # Check against known LinkedIn patterns
    for pattern in ScrapingConstants.LINKEDIN_URL_PATTERNS:
        if re.match(pattern, url):
            return True
    
    # Additional validation for LinkedIn domain
    parsed_url = urlparse(url)
    if 'linkedin.com' in parsed_url.netloc and 'jobs' in url:
        return True
    
    return False


def scrape_from_html(html_content: str) -> Dict[str, Any]:
    """
    Fallback function for HTML content processing.
    
    Note: With Bright Data, this function is mainly for compatibility.
    Bright Data provides structured JSON data, eliminating the need for HTML parsing.
    
    Args:
        html_content (str): Raw HTML content
        
    Returns:
        Dict[str, Any]: Basic extracted information (limited compared to Bright Data)
    """
    logger.warning("scrape_from_html called - consider using Bright Data for comprehensive results")
    
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Basic extraction (very limited compared to Bright Data)
        result = {
            'success': True,
            'scraping_method': 'html_fallback',
            'scraped_at': datetime.now().isoformat(),
            'job_title': soup.find('title').get_text() if soup.find('title') else '',
            'job_description': soup.get_text()[:1000],  # First 1000 chars
            'company_name': '',
            'location': '',
            'salary_info': None,
            'requirements': [],
            'contact_info': {'emails': [], 'phones': [], 'whatsapp': [], 'telegram': []},
            'posted_date': None,
            'applicant_count': None,
            'application_method': '',
            'company_website': None,
            'job_type': None,
            'experience_level': None,
            'industry': None,
            'html_content': html_content,
            # Content-based scores (not available from static HTML)
            'fraud_risk_score': None,
            'content_quality_score': None,
            'company_legitimacy_score': None,
            'contact_risk_score': None
        }
        
        logger.info("HTML fallback extraction completed (limited data available)")
        return result
        
    except Exception as e:
        logger.error(f"HTML fallback extraction failed: {str(e)}")
        return {
            'success': False,
            'error_message': f"HTML extraction failed: {str(e)}",
            'scraping_method': 'html_fallback_failed'
        }


def get_job_id_from_url(url: str) -> Optional[str]:
    """
    Extract job ID from LinkedIn URL.
    
    Args:
        url (str): LinkedIn job URL
        
    Returns:
        Optional[str]: Job ID or None if not found
    """
    try:
        # Extract job ID from various LinkedIn URL formats
        patterns = [
            r'/jobs/view/(\d+)',
            r'currentJobId=(\d+)',
            r'jobId=(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return None
        
    except Exception:
        return None


def scrape_company(url: str) -> Dict[str, Any]:
    """
    Scrape LinkedIn company data for enrichment.
    
    Args:
        url (str): The LinkedIn company URL to scrape
        
    Returns:
        Dict[str, Any]: Dictionary containing company information:
            - followers: Number of company followers
            - company_employees: Parsed employee count from company_size
            - company_founded: Company founding year
            - company_legitimacy_score: Calculated score (0-1)
    """
    logger.info(f"Starting company scraping: {url}")
    
    try:
        # Validate LinkedIn company URL
        if not url or 'linkedin.com/company/' not in url:
            raise ValueError(f"Invalid LinkedIn company URL format: {url}")
        
        # Get Bright Data API key
        config = get_bright_data_config()
        api_key = config.get('api_key', '')
        
        if not api_key:
            raise ValueError("Bright Data API key not configured. Please set BD_API_KEY environment variable.")
        
        # Create scraper and get company data
        scraper = BrightDataLinkedInScraper(api_key)
        
        company_result = scraper._make_request(
            dataset_id=scraper.dataset_ids['companies'], 
            url=url,
            timeout=120
        )
        
        if not company_result.get('success'):
            logger.error(f"❌ Company scraping failed: {company_result.get('error')}")
            return {
                'success': False,
                'error': company_result.get('error', 'Company scraping failed'),
                'url': url
            }
        
        # Extract company data
        company_data = company_result.get('data', [])
        if not company_data:
            logger.error("❌ No company data returned")
            return {
                'success': False,
                'error': 'No company data returned',
                'url': url
            }
        
        # Get first company result
        company_info = company_data[0] if isinstance(company_data, list) else company_data
        
        # Parse company size to get employee count
        company_size_str = company_info.get('company_size', '')
        company_employees = _parse_company_size_to_employees(company_size_str)
        
        # Extract basic company data
        enriched_data = {
            'success': True,
            'url': url,
            'company_name': company_info.get('name', ''),
            'company_followers': company_info.get('followers', 0),
            'company_employees': company_employees,
            'company_founded': company_info.get('founded'),
            'company_size': company_size_str,
            'company_website': company_info.get('website'),
            'employees_in_linkedin': company_info.get('employees_in_linkedin', 0),
            'industries': company_info.get('industries', ''),
        }
        
        # Calculate company-only legitimacy scores
        followers_score = min(enriched_data.get('company_followers', 0) / 10000, 1.0)
        employees_score = min(enriched_data.get('company_employees', 0) / 1000, 1.0)
        founded_score = 0.5 if not enriched_data.get('company_founded') else min((2024 - enriched_data.get('company_founded')) / 50, 1.0)
        
        company_legitimacy = (followers_score * 0.4 + employees_score * 0.4 + founded_score * 0.2)
        
        enrichment_scores = {
            'company_legitimacy_score': company_legitimacy
        }
        enriched_data.update(enrichment_scores)
        
        logger.info(f"✅ Company scraping completed: {enriched_data['company_name']}")
        return enriched_data
        
    except Exception as e:
        logger.error(f"❌ Company scraping failed: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'url': url
        }


def _parse_company_size_to_employees(company_size: str) -> int:
    """
    Parse LinkedIn company size string to estimated employee count.
    
    Args:
        company_size: String like "11-50 employees", "501-1000 employees"
        
    Returns:
        int: Estimated employee count (midpoint of range)
    """
    if not company_size:
        return 0
    
    try:
        # Common LinkedIn size formats
        size_mappings = {
            '1 employee': 1,
            '2-10 employees': 6,
            '11-50 employees': 30,
            '51-200 employees': 125,
            '201-500 employees': 350,
            '501-1000 employees': 750,
            '1001-5000 employees': 3000,
            '5001-10000 employees': 7500,
            '10000+ employees': 15000
        }
        
        # Direct mapping
        if company_size in size_mappings:
            return size_mappings[company_size]
        
        # Extract numbers from string like "11-50"
        import re
        numbers = re.findall(r'\d+', company_size)
        if len(numbers) >= 2:
            return (int(numbers[0]) + int(numbers[1])) // 2
        elif len(numbers) == 1:
            num = int(numbers[0])
            return num if num <= 10000 else 15000  # Cap at 15k for 10000+
        
        return 0
        
    except (ValueError, TypeError):
        return 0


# Export functions
__all__ = [
    'scrape_job',
    'scrape_company',
    'scrape_from_html',  # Keep for HTML fallback
    'validate_linkedin_url',
    'get_job_id_from_url',
    'BrightDataLinkedInScraper'
]