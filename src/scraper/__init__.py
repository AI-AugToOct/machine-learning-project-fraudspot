"""
Scraper Module for Job Post Fraud Detector

This module provides functionality for scraping LinkedIn job postings
and extracting relevant information for fraud detection analysis.

Components:
- linkedin_scraper: Main scraping functionality for LinkedIn job posts
- scraper_utils: Utility functions and helpers for web scraping

 Version: 1.0.0
"""

from .linkedin_scraper import (
    scrape_job_posting,
    extract_job_title,
    extract_company_name,
    extract_job_description,
    extract_location,
    extract_salary_info,
    extract_requirements,
    extract_contact_info,
    validate_linkedin_url,
    handle_dynamic_content
)

from .scraper_utils import (
    setup_session,
    handle_rate_limiting,
    retry_request,
    parse_job_metadata,
    clean_scraped_text,
    detect_anti_bot_measures,
    save_scraped_html
)

__all__ = [
    # linkedin_scraper functions
    'scrape_job_posting',
    'extract_job_title',
    'extract_company_name',
    'extract_job_description',
    'extract_location',
    'extract_salary_info',
    'extract_requirements',
    'extract_contact_info',
    'validate_linkedin_url',
    'handle_dynamic_content',
    
    # scraper_utils functions
    'setup_session',
    'handle_rate_limiting',
    'retry_request',
    'parse_job_metadata',
    'clean_scraped_text',
    'detect_anti_bot_measures',
    'save_scraped_html'
]