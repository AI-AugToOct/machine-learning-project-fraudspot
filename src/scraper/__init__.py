"""
Scraper Module for Job Post Fraud Detector - Single Source Implementation

This module provides comprehensive functionality for scraping LinkedIn job postings
using Bright Data's professional APIs for fraud detection analysis.

Components:
- linkedin_scraper: Complete scraping functionality with Bright Data integration

Version: 3.0.0 - DRY Consolidation - Single Source of Truth
"""

from .linkedin_scraper import (
    BrightDataLinkedInScraper,
    get_job_id_from_url,
    scrape_from_html,
    scrape_job,
    scrape_profile,
    validate_linkedin_url,
)

__all__ = [
    'scrape_job',
    'scrape_profile',
    'scrape_from_html',
    'validate_linkedin_url',
    'get_job_id_from_url',
    'BrightDataLinkedInScraper'
]