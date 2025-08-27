"""
LinkedIn Job Scraper - Complete Implementation

This module provides comprehensive functionality for scraping LinkedIn job postings
and extracting all relevant information for fraud detection analysis.

Key Features:
- Complete job posting extraction from LinkedIn URLs
- Dynamic content handling with Selenium
- Anti-bot detection and mitigation
- Comprehensive data extraction and validation
- Rate limiting and retry mechanisms

Version: 2.0.0 - Production Ready
"""

import re
import time
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from urllib.parse import urlparse, parse_qs
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup, Tag
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import TimeoutException, WebDriverException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager

from ..config import (
    LINKEDIN_URL_PATTERNS, DEFAULT_HEADERS, SCRAPING_CONFIG,
    SUSPICIOUS_KEYWORDS, SUSPICIOUS_EMAIL_DOMAINS
)
from .scraper_utils import (
    setup_session, handle_rate_limiting, retry_request,
    clean_scraped_text, detect_anti_bot_measures, setup_selenium_driver
)

logger = logging.getLogger(__name__)


def scrape_job_posting(url: str) -> Dict[str, Any]:
    """
    Scrape a LinkedIn job posting and extract all relevant information.
    
    This is the main function that orchestrates the entire scraping process.
    It handles both static and dynamic content, extracts all job information,
    and returns a structured dictionary of the scraped data.
    
    Args:
        url (str): The LinkedIn job posting URL to scrape
        
    Returns:
        Dict[str, Any]: Dictionary containing all extracted job information
            {
                'job_title': str,
                'company_name': str,
                'job_description': str,
                'location': str,
                'salary_info': Optional[str],
                'requirements': List[str],
                'contact_info': Dict[str, List[str]],
                'posted_date': Optional[str],
                'applicant_count': Optional[int],
                'application_method': str,
                'company_website': Optional[str],
                'job_type': Optional[str],
                'experience_level': Optional[str],
                'industry': Optional[str],
                'scraped_at': str,
                'url': str,
                'html_content': str,
                'success': bool,
                'error_message': Optional[str]
            }
            
    Raises:
        ValueError: If the URL is invalid
        requests.RequestException: If network request fails
        TimeoutException: If page loading times out
    """
    logger.info(f"Starting to scrape job posting: {url}")
    
    # Initialize result structure
    result = {
        'url': url,
        'scraped_at': datetime.now().isoformat(),
        'success': False,
        'error_message': None,
        'job_title': '',
        'company_name': '',
        'job_description': '',
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
        'html_content': ''
    }
    
    try:
        # Validate LinkedIn URL
        if not validate_linkedin_url(url):
            raise ValueError(f"Invalid LinkedIn URL format: {url}")
        
        # Try static scraping first (faster)
        static_result = _scrape_static_content(url)
        if static_result and static_result.get('success'):
            logger.info("Successfully scraped using static method")
            result.update(static_result)
            result['success'] = True
            return result
        
        # Fallback to dynamic scraping with Selenium
        logger.info("Static scraping failed, attempting dynamic scraping")
        dynamic_result = _scrape_dynamic_content(url)
        if dynamic_result and dynamic_result.get('success'):
            logger.info("Successfully scraped using dynamic method")
            result.update(dynamic_result)
            result['success'] = True
            return result
        
        # Both methods failed
        result['error_message'] = "Both static and dynamic scraping methods failed"
        logger.error(f"Failed to scrape job posting: {url}")
        
    except Exception as e:
        result['error_message'] = str(e)
        logger.error(f"Error scraping job posting {url}: {str(e)}")
    
    return result


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
    for pattern in LINKEDIN_URL_PATTERNS:
        if re.match(pattern, url):
            return True
    
    # Additional validation for LinkedIn domain
    parsed_url = urlparse(url)
    if 'linkedin.com' in parsed_url.netloc and 'jobs' in url:
        return True
    
    return False


def extract_job_title(soup: BeautifulSoup) -> str:
    """
    Extract job title from LinkedIn page.
    
    Args:
        soup (BeautifulSoup): Parsed HTML content
        
    Returns:
        str: Job title or empty string if not found
    """
    title_selectors = [
        'h1[data-test-id="job-title"]',
        'h1.t-24.t-bold.jobs-unified-top-card__job-title',
        'h1.job-title',
        'h1[class*="job-title"]',
        '.jobs-unified-top-card__job-title h1',
        'h1'  # Fallback
    ]
    
    for selector in title_selectors:
        element = soup.select_one(selector)
        if element:
            title = clean_scraped_text(element.get_text())
            if title and len(title) > 3:  # Basic validation
                return title
    
    return ''


def extract_company_name(soup: BeautifulSoup) -> str:
    """
    Extract company name from LinkedIn page.
    
    Args:
        soup (BeautifulSoup): Parsed HTML content
        
    Returns:
        str: Company name or empty string if not found
    """
    company_selectors = [
        'a[data-test-id="job-poster-name"]',
        '.jobs-unified-top-card__company-name a',
        '.company-name a',
        'a[class*="company"]',
        '.jobs-poster__name a'
    ]
    
    for selector in company_selectors:
        element = soup.select_one(selector)
        if element:
            company = clean_scraped_text(element.get_text())
            if company and len(company) > 1:
                return company
    
    return ''


def extract_job_description(soup: BeautifulSoup) -> str:
    """
    Extract job description from LinkedIn page.
    
    Args:
        soup (BeautifulSoup): Parsed HTML content
        
    Returns:
        str: Job description or empty string if not found
    """
    description_selectors = [
        '.jobs-description-content__text',
        '.jobs-box__html-content',
        '.description__text',
        '.jobs-description__content',
        'div[data-test-id="job-description"]',
        '.jobs-unified-description-container'
    ]
    
    for selector in description_selectors:
        element = soup.select_one(selector)
        if element:
            description = clean_scraped_text(element.get_text())
            if description and len(description) > 50:  # Ensure substantial content
                return description
    
    return ''


def extract_location(soup: BeautifulSoup) -> str:
    """
    Extract job location from LinkedIn page.
    
    Args:
        soup (BeautifulSoup): Parsed HTML content
        
    Returns:
        str: Location or empty string if not found
    """
    location_selectors = [
        '.jobs-unified-top-card__bullet',
        '.job-location',
        '[data-test-id="job-location"]',
        '.jobs-unified-top-card__subtitle-secondary-grouping span'
    ]
    
    for selector in location_selectors:
        elements = soup.select(selector)
        for element in elements:
            location = clean_scraped_text(element.get_text())
            # Check if this looks like a location (contains common location keywords)
            location_keywords = ['city', 'state', 'country', 'remote', 'hybrid', ',', 'CA', 'NY', 'TX']
            if location and any(keyword.lower() in location.lower() for keyword in location_keywords):
                return location
    
    return ''


def extract_salary_info(soup: BeautifulSoup) -> Optional[str]:
    """
    Extract salary information from LinkedIn page.
    
    Args:
        soup (BeautifulSoup): Parsed HTML content
        
    Returns:
        Optional[str]: Salary information or None if not found
    """
    salary_selectors = [
        '.salary',
        '[data-test-id="salary"]',
        '.jobs-unified-top-card__job-insight span',
        '.job-criteria__text'
    ]
    
    for selector in salary_selectors:
        elements = soup.select(selector)
        for element in elements:
            text = clean_scraped_text(element.get_text())
            # Check if text contains salary indicators
            salary_indicators = ['$', '€', '£', 'salary', 'per hour', '/hr', 'annually', 'k', 'thousand']
            if text and any(indicator.lower() in text.lower() for indicator in salary_indicators):
                return text
    
    return None


def extract_contact_info(content: str) -> Dict[str, List[str]]:
    """
    Extract contact information from job content.
    
    Args:
        content (str): Job description content
        
    Returns:
        Dict[str, List[str]]: Dictionary containing extracted contact info
    """
    contact_info = {
        'emails': [],
        'phones': [],
        'whatsapp': [],
        'telegram': []
    }
    
    if not content:
        return contact_info
    
    # Extract email addresses
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, content)
    contact_info['emails'] = list(set(emails))
    
    # Extract phone numbers (various formats)
    phone_patterns = [
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # US format
        r'\+\d{1,3}[-.\s]?\d{3,4}[-.\s]?\d{3,4}[-.\s]?\d{3,4}',  # International
        r'\(\d{3}\)\s?\d{3}[-.]?\d{4}'  # (XXX) XXX-XXXX
    ]
    
    phones = []
    for pattern in phone_patterns:
        phones.extend(re.findall(pattern, content))
    contact_info['phones'] = list(set(phones))
    
    # Extract WhatsApp references
    whatsapp_pattern = r'whatsapp[:\s]*([+]?\d+[\d\s-]+)'
    whatsapp_matches = re.findall(whatsapp_pattern, content, re.IGNORECASE)
    contact_info['whatsapp'] = whatsapp_matches
    
    # Extract Telegram references
    telegram_pattern = r'telegram[:\s]*(@?\w+)'
    telegram_matches = re.findall(telegram_pattern, content, re.IGNORECASE)
    contact_info['telegram'] = telegram_matches
    
    return contact_info


def extract_requirements(soup: BeautifulSoup) -> List[str]:
    """
    Extract job requirements from LinkedIn page.
    
    Args:
        soup (BeautifulSoup): Parsed HTML content
        
    Returns:
        List[str]: List of requirements
    """
    requirements = []
    
    # Look for requirements sections
    requirements_selectors = [
        '.jobs-unified-description-container ul li',
        '.description__text ul li',
        '.jobs-box__html-content ul li'
    ]
    
    for selector in requirements_selectors:
        elements = soup.select(selector)
        for element in elements:
            req_text = clean_scraped_text(element.get_text())
            if req_text and len(req_text) > 10:  # Filter out short/empty requirements
                requirements.append(req_text)
    
    # Also look for criteria sections
    criteria_elements = soup.select('.job-criteria__text')
    for element in criteria_elements:
        criteria_text = clean_scraped_text(element.get_text())
        if criteria_text:
            requirements.append(criteria_text)
    
    return requirements


def _scrape_static_content(url: str) -> Dict[str, Any]:
    """
    Attempt to scrape using static requests method.
    
    Args:
        url (str): LinkedIn job URL
        
    Returns:
        Dict[str, Any]: Scraped data or None if failed
    """
    try:
        session = setup_session()
        response = retry_request(session, url, max_retries=3)
        
        if not response or response.status_code != 200:
            return {'success': False}
        
        # Check for anti-bot measures
        if detect_anti_bot_measures(response.text):
            logger.warning("Anti-bot measures detected in static scraping")
            return {'success': False}
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract all information
        result = {
            'success': True,
            'job_title': extract_job_title(soup),
            'company_name': extract_company_name(soup),
            'job_description': extract_job_description(soup),
            'location': extract_location(soup),
            'salary_info': extract_salary_info(soup),
            'requirements': extract_requirements(soup),
            'html_content': response.text
        }
        
        # Extract contact info from description
        result['contact_info'] = extract_contact_info(result['job_description'])
        
        # Validate that we got essential information
        if not result['job_title'] or not result['company_name']:
            logger.warning("Static scraping didn't capture essential information")
            return {'success': False}
        
        return result
        
    except Exception as e:
        logger.error(f"Static scraping failed: {str(e)}")
        return {'success': False}


def _scrape_dynamic_content(url: str) -> Dict[str, Any]:
    """
    Scrape using Selenium for dynamic content.
    
    Args:
        url (str): LinkedIn job URL
        
    Returns:
        Dict[str, Any]: Scraped data or None if failed
    """
    driver = None
    try:
        # Setup Selenium driver
        driver = setup_selenium_driver()
        
        # Navigate to the URL
        driver.get(url)
        
        # Wait for essential elements to load
        wait = WebDriverWait(driver, 15)
        
        # Wait for job title to appear
        try:
            wait.until(EC.presence_of_element_located((By.TAG_NAME, "h1")))
        except TimeoutException:
            logger.warning("Timeout waiting for job title to load")
        
        # Handle "Show more" buttons to expand content
        _expand_job_content(driver, wait)
        
        # Get the page source after JavaScript execution
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')
        
        # Extract all information
        result = {
            'success': True,
            'job_title': extract_job_title(soup),
            'company_name': extract_company_name(soup),
            'job_description': extract_job_description(soup),
            'location': extract_location(soup),
            'salary_info': extract_salary_info(soup),
            'requirements': extract_requirements(soup),
            'html_content': page_source
        }
        
        # Extract additional dynamic content
        result.update(_extract_dynamic_elements(driver, wait))
        
        # Extract contact info from description
        result['contact_info'] = extract_contact_info(result['job_description'])
        
        # Validate essential information
        if not result['job_title'] or not result['company_name']:
            logger.warning("Dynamic scraping didn't capture essential information")
            return {'success': False}
        
        return result
        
    except Exception as e:
        logger.error(f"Dynamic scraping failed: {str(e)}")
        return {'success': False}
        
    finally:
        if driver:
            try:
                driver.quit()
            except Exception as e:
                logger.warning(f"Error closing driver: {str(e)}")


def _expand_job_content(driver: webdriver.Chrome, wait: WebDriverWait) -> None:
    """
    Click "Show more" buttons to expand job content.
    
    Args:
        driver: Selenium WebDriver instance
        wait: WebDriverWait instance
    """
    try:
        # Common selectors for "Show more" buttons
        show_more_selectors = [
            'button[aria-label="Show more, visually expands previously read content"]',
            '.jobs-description__footer-button',
            'button[data-test-id="show-more-button"]',
            '.show-more-less-html__button'
        ]
        
        for selector in show_more_selectors:
            try:
                show_more_button = wait.until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                )
                driver.execute_script("arguments[0].click();", show_more_button)
                time.sleep(1)  # Allow content to expand
                logger.info(f"Clicked show more button: {selector}")
                break
            except (TimeoutException, NoSuchElementException):
                continue
                
    except Exception as e:
        logger.debug(f"No expandable content found or error expanding: {str(e)}")


def _extract_dynamic_elements(driver: webdriver.Chrome, wait: WebDriverWait) -> Dict[str, Any]:
    """
    Extract elements that are only available after JavaScript execution.
    
    Args:
        driver: Selenium WebDriver instance
        wait: WebDriverWait instance
        
    Returns:
        Dict[str, Any]: Additional scraped information
    """
    result = {}
    
    try:
        # Extract posted date
        try:
            date_element = driver.find_element(By.CSS_SELECTOR, '.jobs-unified-top-card__subtitle-secondary-grouping time')
            result['posted_date'] = date_element.get_attribute('datetime') or date_element.text
        except NoSuchElementException:
            result['posted_date'] = None
        
        # Extract applicant count
        try:
            applicant_element = driver.find_element(By.CSS_SELECTOR, '.jobs-unified-top-card__subtitle-secondary-grouping span')
            applicant_text = applicant_element.text
            # Extract number from text like "50 applicants"
            applicant_match = re.search(r'(\d+)', applicant_text)
            result['applicant_count'] = int(applicant_match.group(1)) if applicant_match else None
        except (NoSuchElementException, ValueError):
            result['applicant_count'] = None
        
        # Extract job type and experience level
        try:
            criteria_elements = driver.find_elements(By.CSS_SELECTOR, '.jobs-unified-top-card__job-insight span')
            for element in criteria_elements:
                text = element.text.lower()
                if any(job_type in text for job_type in ['full-time', 'part-time', 'contract', 'temporary']):
                    result['job_type'] = element.text
                elif any(level in text for level in ['entry', 'associate', 'mid', 'senior', 'director', 'executive']):
                    result['experience_level'] = element.text
        except NoSuchElementException:
            pass
        
        # Extract industry
        try:
            industry_element = driver.find_element(By.CSS_SELECTOR, '.jobs-company__industry')
            result['industry'] = industry_element.text
        except NoSuchElementException:
            result['industry'] = None
        
        # Extract company website
        try:
            website_element = driver.find_element(By.CSS_SELECTOR, 'a[data-test-id="company-website-url"]')
            result['company_website'] = website_element.get_attribute('href')
        except NoSuchElementException:
            result['company_website'] = None
        
        # Determine application method
        try:
            apply_button = driver.find_element(By.CSS_SELECTOR, '.jobs-apply-button')
            if apply_button:
                result['application_method'] = 'LinkedIn Easy Apply'
            else:
                result['application_method'] = 'External Application'
        except NoSuchElementException:
            result['application_method'] = 'Unknown'
        
    except Exception as e:
        logger.warning(f"Error extracting dynamic elements: {str(e)}")
    
    return result


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