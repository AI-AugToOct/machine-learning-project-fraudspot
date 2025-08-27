"""
Scraper Utilities

This module provides utility functions and helpers for web scraping operations.
It includes session management, rate limiting, request retrying, and content
processing utilities used by the main scraping modules.

 Version: 1.0.0
"""

import time
import random
import logging
import hashlib
import os
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import re

from ..config import (
    SCRAPING_CONFIG, DEFAULT_HEADERS, CACHE_CONFIG
)

logger = logging.getLogger(__name__)

# Global session for connection pooling
_session = None
_last_request_time = 0


def setup_session() -> requests.Session:
    """
    Set up a requests session with proper configuration for web scraping.
    
    This function creates and configures a requests session with:
    - Custom headers to mimic a real browser
    - Connection pooling for better performance
    - Retry strategy for handling temporary failures
    - Timeout settings
    
    Returns:
        requests.Session: Configured session object
        
    Implementation by Orchestration Engineer - Infrastructure & Deployment:
        - Browser-like headers configuration
        - Connection pooling settings
        - Retry strategy with exponential backoff
        - Proxy support (if needed)
        - SSL certificate verification settings
        - Cookie jar for session persistence
    """
    global _session
    
    if _session is None:
        _session = requests.Session()
        
        # Set headers
        _session.headers.update(DEFAULT_HEADERS)
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=SCRAPING_CONFIG['max_retries'],
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=20
        )
        
        _session.mount("http://", adapter)
        _session.mount("https://", adapter)
        
        logger.info("Scraping session initialized")
    
    return _session


def handle_rate_limiting() -> None:
    """
    Handle rate limiting to avoid being blocked by LinkedIn.
    
    This function implements intelligent rate limiting by:
    - Tracking time between requests
    - Adding delays when necessary
    - Implementing random delays to appear more human-like
    
    Implementation by Orchestration Engineer - Infrastructure & Deployment:
        - Track time since last request
        - Add configurable delay between requests
        - Random jitter to avoid detection patterns
        - Exponential backoff for repeated requests
        - Respect robots.txt guidelines
    """
    global _last_request_time
    
    current_time = time.time()
    time_since_last = current_time - _last_request_time
    
    min_delay = SCRAPING_CONFIG['rate_limit_delay']
    
    if time_since_last < min_delay:
        delay = min_delay - time_since_last
        # Add random jitter (±50% of delay)
        jitter = random.uniform(-0.5 * delay, 0.5 * delay)
        total_delay = max(0, delay + jitter)
        
        if total_delay > 0:
            logger.debug(f"Rate limiting: waiting {total_delay:.2f} seconds")
            time.sleep(total_delay)
    
    _last_request_time = time.time()


def retry_request(session: requests.Session, url: str, max_retries: int = None) -> Optional[requests.Response]:
    """
    Make a request with retry logic and error handling.
    
    Args:
        session (requests.Session): The requests session to use
        url (str): The URL to request
        max_retries (int, optional): Maximum number of retries
        
    Returns:
        Optional[requests.Response]: The response object, None if all retries failed
        
    Implementation by Orchestration Engineer - Infrastructure & Deployment:
        - Exponential backoff between retries
        - Handle different types of errors differently
        - Log retry attempts and failures
        - Return response or None based on success/failure
        - Handle rate limiting responses (429)
    """
    if max_retries is None:
        max_retries = SCRAPING_CONFIG['max_retries']
    
    for attempt in range(max_retries + 1):
        try:
            # Handle rate limiting before each attempt
            handle_rate_limiting()
            
            response = session.get(
                url,
                timeout=SCRAPING_CONFIG['timeout']
            )
            
            # Handle rate limiting response
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', SCRAPING_CONFIG['retry_delay']))
                logger.warning(f"Rate limited, waiting {retry_after} seconds")
                time.sleep(retry_after)
                continue
            
            # Success
            if response.status_code == 200:
                return response
            
            # Client error (4xx) - don't retry
            if 400 <= response.status_code < 500:
                logger.error(f"Client error {response.status_code} for URL: {url}")
                return None
            
            # Server error (5xx) - retry
            logger.warning(f"Server error {response.status_code}, attempt {attempt + 1}/{max_retries + 1}")
            
        except requests.exceptions.Timeout:
            logger.warning(f"Request timeout, attempt {attempt + 1}/{max_retries + 1}")
        except requests.exceptions.ConnectionError:
            logger.warning(f"Connection error, attempt {attempt + 1}/{max_retries + 1}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            return None
        
        # Wait before retry (exponential backoff)
        if attempt < max_retries:
            delay = SCRAPING_CONFIG['retry_delay'] * (2 ** attempt)
            time.sleep(delay)
    
    logger.error(f"All retry attempts failed for URL: {url}")
    return None


def parse_job_metadata(soup: BeautifulSoup) -> Dict[str, Any]:
    """
    Parse metadata from job posting HTML.
    
    Args:
        soup (BeautifulSoup): Parsed HTML content
        
    Returns:
        Dict[str, Any]: Dictionary containing parsed metadata
            {
                'page_title': str,
                'meta_description': str,
                'canonical_url': str,
                'structured_data': Dict,
                'social_media_tags': Dict
            }
            
    Implementation by Orchestration Engineer - Infrastructure & Deployment:
        - Extract page title and meta description
        - Parse structured data (JSON-LD, microdata)
        - Extract Open Graph and Twitter Card tags
        - Find canonical URL
        - Extract any job-specific metadata
    """
    metadata = {
        'page_title': '',
        'meta_description': '',
        'canonical_url': '',
        'structured_data': {},
        'social_media_tags': {}
    }
    
    try:
        # Page title
        title_tag = soup.find('title')
        if title_tag:
            metadata['page_title'] = clean_scraped_text(title_tag.get_text())
        
        # Meta description
        description_tag = soup.find('meta', attrs={'name': 'description'})
        if description_tag:
            metadata['meta_description'] = description_tag.get('content', '')
        
        # Canonical URL
        canonical_tag = soup.find('link', attrs={'rel': 'canonical'})
        if canonical_tag:
            metadata['canonical_url'] = canonical_tag.get('href', '')
        
        # Structured data (JSON-LD)
        json_ld_scripts = soup.find_all('script', attrs={'type': 'application/ld+json'})
        for script in json_ld_scripts:
            try:
                import json
                structured_data = json.loads(script.string)
                if '@type' in structured_data:
                    metadata['structured_data'] = structured_data
                    break
            except:
                continue
        
        # Open Graph tags
        og_tags = soup.find_all('meta', attrs={'property': re.compile(r'^og:')})
        for tag in og_tags:
            property_name = tag.get('property', '').replace('og:', '')
            content = tag.get('content', '')
            metadata['social_media_tags'][property_name] = content
        
        return metadata
        
    except Exception as e:
        logger.error(f"Error parsing job metadata: {str(e)}")
        return metadata


def clean_scraped_text(text: str) -> str:
    """
    Clean and normalize scraped text content.
    
    Args:
        text (str): Raw text content from scraping
        
    Returns:
        str: Cleaned and normalized text
        
    Implementation by Orchestration Engineer - Infrastructure & Deployment:
        - Remove extra whitespace and normalize line breaks
        - Handle Unicode characters and encoding issues
        - Remove or replace HTML entities
        - Normalize quotation marks and dashes
        - Remove non-printable characters
        - Preserve meaningful formatting
    """
    if not text or not isinstance(text, str):
        return ""
    
    try:
        # Replace HTML entities
        import html
        text = html.unescape(text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Remove extra punctuation
        text = re.sub(r'[^\w\s\.,!?;:()\[\]\'\"@#$%&*+-=/<>]', '', text)
        
        # Normalize quotes
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r'['']', "'", text)
        
        # Remove excessive punctuation
        text = re.sub(r'\.{3,}', '...', text)
        text = re.sub(r'[!?]{2,}', lambda m: m.group(0)[0], text)
        
        return text.strip()
        
    except Exception as e:
        logger.error(f"Error cleaning text: {str(e)}")
        return text


def detect_anti_bot_measures(html_content: str) -> bool:
    """
    Detect if the page contains anti-bot measures or blocks.
    
    Args:
        html_content (str): The HTML content to analyze
        
    Returns:
        bool: True if anti-bot measures detected, False otherwise
        
    Implementation by Orchestration Engineer - Infrastructure & Deployment:
        - Check for CAPTCHA elements
        - Detect blocked/error pages
        - Look for bot detection scripts
        - Check for rate limiting messages
        - Identify incomplete page loads
    """
    if not html_content:
        return True
    
    try:
        # Common anti-bot indicators
        bot_indicators = [
            'captcha',
            'blocked',
            'bot detected',
            'rate limit',
            'too many requests',
            'access denied',
            'cloudflare',
            'challenge',
            'verification'
        ]
        
        html_lower = html_content.lower()
        
        for indicator in bot_indicators:
            if indicator in html_lower:
                logger.warning(f"Anti-bot measure detected: {indicator}")
                return True
        
        # Check for minimal content (possible block page)
        soup = BeautifulSoup(html_content, 'html.parser')
        text_content = soup.get_text()
        
        if len(text_content.strip()) < 500:  # Very little content
            logger.warning("Minimal content detected, possible bot block")
            return True
        
        # Check for missing job-specific content
        job_indicators = ['job', 'position', 'role', 'company', 'apply']
        if not any(indicator in html_lower for indicator in job_indicators):
            logger.warning("No job-related content found")
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"Error detecting anti-bot measures: {str(e)}")
        return True


def save_scraped_html(url: str, html_content: str, job_id: str = None) -> Optional[str]:
    """
    Save scraped HTML content to disk for debugging and analysis.
    
    Args:
        url (str): The URL that was scraped
        html_content (str): The HTML content to save
        job_id (str, optional): Job ID for filename
        
    Returns:
        Optional[str]: Path to saved file, None if saving failed
        
    Implementation by Orchestration Engineer - Infrastructure & Deployment:
        - Create appropriate directory structure
        - Generate unique filename based on URL/job ID
        - Save HTML with proper encoding
        - Add metadata header to saved file
        - Handle disk space and cleanup old files
    """
    if not CACHE_CONFIG.get('enable_cache', True):
        return None
    
    try:
        # Create cache directory if it doesn't exist
        cache_dir = CACHE_CONFIG['cache_dir']
        html_dir = os.path.join(cache_dir, 'html')
        os.makedirs(html_dir, exist_ok=True)
        
        # Generate filename
        if job_id:
            filename = f"job_{job_id}.html"
        else:
            # Use URL hash as filename
            url_hash = hashlib.md5(url.encode()).hexdigest()
            filename = f"page_{url_hash}.html"
        
        filepath = os.path.join(html_dir, filename)
        
        # Save HTML with metadata header
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"<!-- Scraped from: {url} -->\n")
            f.write(f"<!-- Scraped at: {time.strftime('%Y-%m-%d %H:%M:%S')} -->\n")
            f.write(html_content)
        
        logger.debug(f"HTML saved to: {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Error saving HTML: {str(e)}")
        return None


def extract_job_id_from_url(url: str) -> Optional[str]:
    """
    Extract job ID from LinkedIn job URL.
    
    Args:
        url (str): LinkedIn job URL
        
    Returns:
        Optional[str]: Job ID if found, None otherwise
        
    Implementation by Orchestration Engineer - Infrastructure & Deployment:
        - Parse URL components
        - Extract job ID from path or query parameters
        - Handle different LinkedIn URL formats
        - Validate extracted job ID format
    """
    try:
        parsed_url = urlparse(url)
        
        # Extract from path: /jobs/view/1234567890
        path_match = re.search(r'/jobs/view/(\d+)', parsed_url.path)
        if path_match:
            return path_match.group(1)
        
        # Extract from query parameters
        from urllib.parse import parse_qs
        query_params = parse_qs(parsed_url.query)
        
        if 'currentJobId' in query_params:
            return query_params['currentJobId'][0]
        
        if 'jobId' in query_params:
            return query_params['jobId'][0]
        
        return None
        
    except Exception as e:
        logger.error(f"Error extracting job ID from URL: {str(e)}")
        return None


def estimate_content_quality(soup: BeautifulSoup) -> Dict[str, float]:
    """
    Estimate the quality and completeness of scraped content.
    
    Args:
        soup (BeautifulSoup): Parsed HTML content
        
    Returns:
        Dict[str, float]: Quality metrics (0.0 to 1.0)
            {
                'completeness': float,
                'content_richness': float,
                'structure_quality': float,
                'overall_score': float
            }
            
    Implementation by Orchestration Engineer - Infrastructure & Deployment:
        - Check for presence of key job posting elements
        - Analyze text content length and richness
        - Evaluate HTML structure quality
        - Detect missing or incomplete sections
        - Calculate overall quality score
    """
    try:
        quality_metrics = {
            'completeness': 0.0,
            'content_richness': 0.0,
            'structure_quality': 0.0,
            'overall_score': 0.0
        }
        
        if not soup:
            return quality_metrics
        
        # Check completeness (presence of key elements)
        key_elements = [
            ('.job-title', 'h1'),  # Job title
            ('.company-name', '.company'),  # Company name
            ('.job-description', '.description'),  # Job description
            ('.location', '.job-location'),  # Location
        ]
        
        found_elements = 0
        for selectors in key_elements:
            for selector in selectors:
                if soup.select_one(selector):
                    found_elements += 1
                    break
        
        quality_metrics['completeness'] = found_elements / len(key_elements)
        
        # Analyze content richness
        text_content = soup.get_text()
        word_count = len(text_content.split())
        
        if word_count > 500:
            quality_metrics['content_richness'] = min(1.0, word_count / 1000)
        else:
            quality_metrics['content_richness'] = word_count / 500
        
        # Evaluate structure quality
        structure_indicators = [
            soup.find('title'),  # Has title
            soup.find_all('h1', 'h2', 'h3'),  # Has headings
            soup.find_all('p'),  # Has paragraphs
            soup.find_all('ul', 'ol'),  # Has lists
        ]
        
        structure_score = sum(1 for indicator in structure_indicators if indicator) / len(structure_indicators)
        quality_metrics['structure_quality'] = structure_score
        
        # Calculate overall score
        weights = {'completeness': 0.4, 'content_richness': 0.4, 'structure_quality': 0.2}
        quality_metrics['overall_score'] = sum(
            quality_metrics[metric] * weight 
            for metric, weight in weights.items()
        )
        
        return quality_metrics
        
    except Exception as e:
        logger.error(f"Error estimating content quality: {str(e)}")
        return {
            'completeness': 0.0,
            'content_richness': 0.0,
            'structure_quality': 0.0,
            'overall_score': 0.0
        }


def get_scraping_statistics() -> Dict[str, Any]:
    """
    Get statistics about scraping operations.
    
    Returns:
        Dict[str, Any]: Statistics about scraping performance
        
    Implementation by Orchestration Engineer - Infrastructure & Deployment:
        - Track successful vs failed scraping attempts
        - Monitor response times and performance
        - Count rate limiting incidents
        - Track content quality scores
        - Monitor cache hit rates
    """
    try:
        # For now, return basic statistics structure
        # In production, this would read from persistent storage
        stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'rate_limited_requests': 0,
            'average_response_time': 0.0,
            'cache_hit_rate': 0.0,
            'average_content_quality': 0.0,
            'session_initialized': _session is not None,
            'last_request_time': _last_request_time,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Add session information if available
        if _session:
            stats['session_headers'] = dict(_session.headers)
            stats['session_adapters'] = list(_session.adapters.keys())
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting scraping statistics: {str(e)}")
        return {
            'error': str(e),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }


def cleanup_old_cache() -> None:
    """
    Clean up old cached files based on configured retention period.
    
    Implementation by Orchestration Engineer - Infrastructure & Deployment:
        - Check file modification dates
        - Remove files older than configured retention period
        - Clean up empty directories
        - Log cleanup statistics
        - Handle cleanup errors gracefully
    """
    try:
        if not CACHE_CONFIG.get('enable_cache', True):
            return
        
        cache_dir = CACHE_CONFIG['cache_dir']
        if not os.path.exists(cache_dir):
            return
        
        retention_days = CACHE_CONFIG.get('cache_expiry_days', 7)
        cutoff_time = time.time() - (retention_days * 24 * 60 * 60)
        
        files_deleted = 0
        for root, dirs, files in os.walk(cache_dir):
            for file in files:
                filepath = os.path.join(root, file)
                if os.path.getmtime(filepath) < cutoff_time:
                    try:
                        os.remove(filepath)
                        files_deleted += 1
                    except OSError:
                        pass
        
        if files_deleted > 0:
            logger.info(f"Cleaned up {files_deleted} old cache files")
            
    except Exception as e:
        logger.error(f"Error cleaning up cache: {str(e)}")