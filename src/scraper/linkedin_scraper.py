"""
LinkedIn Job Scraper - Single Source Implementation

This module provides comprehensive functionality for scraping LinkedIn job postings
using Bright Data's professional APIs for maximum fraud detection capabilities.

Key Features:
- Complete job posting extraction with 50+ structured data points
- Comprehensive job poster profile analysis
- Company verification and legitimacy scoring
- Advanced fraud detection algorithms
- Professional API integration with structured JSON data
- Real-time data with no HTML parsing overhead

Version: 4.0.0 - DRY Consolidation - Single Source of Truth
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
    Professional Bright Data LinkedIn scraper for comprehensive fraud detection.
    
    This class provides structured access to LinkedIn data through Bright Data's
    official APIs, extracting all available data points for maximum fraud detection
    capabilities.
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

    def _poll_for_snapshot(self, snapshot_id: str, dataset_id: str, max_attempts: int = 180, polling_interval: float = 10.0) -> Dict[str, Any]:
        """
        Poll for snapshot completion with fixed 10-second intervals as per Bright Data requirements.
        Total timeout: 30 minutes (180 attempts * 10 seconds).
        
        Args:
            snapshot_id (str): Snapshot ID from initial API response
            dataset_id (str): Dataset ID for polling endpoint
            max_attempts (int): Maximum polling attempts (default: 180 for 30 minutes)
            polling_interval (float): Fixed polling interval in seconds (default: 10.0)
            
        Returns:
            Dict[str, Any]: Final snapshot data or error information
        """
        # Correct Bright Data polling endpoint - v3 without dataset_id parameter
        poll_url = f"https://api.brightdata.com/datasets/v3/snapshot/{snapshot_id}"
        
        for attempt in range(max_attempts):
            try:
                # Log polling attempt (first attempt is immediate)
                logger.info(f"⏳ Polling attempt {attempt + 1}/{max_attempts}...")
                
                response = self.session.get(poll_url, timeout=30)
                
                if response.status_code == 200:
                    snapshot_data = response.json()
                    
                    # Check if response contains actual profile data (ready snapshot)
                    if isinstance(snapshot_data, dict) and ('name' in snapshot_data or 'linkedin_id' in snapshot_data):
                        # Direct profile data response - snapshot is ready
                        logger.info("📊 Snapshot ready - received profile data directly")
                        return {
                            'success': True,
                            'data': snapshot_data,
                            'status_code': 200
                        }
                    
                    # Check for status-based response (old format)
                    status = snapshot_data.get('status', 'unknown')
                    logger.info(f"📊 Snapshot status: {status}")
                    
                    # Check for Bright Data completion status: 'ready' (official docs)
                    if status == 'ready':
                        # When ready, download the actual data using download endpoint
                        download_url = f"https://api.brightdata.com/datasets/snapshots/{snapshot_id}/download"
                        logger.info(f"📥 Snapshot ready - downloading from: {download_url}")
                        
                        download_response = self.session.get(download_url, timeout=60)
                        if download_response.status_code == 200:
                            try:
                                download_data = download_response.json()
                                # Handle download response format
                                if isinstance(download_data, list) and download_data:
                                    data = download_data
                                elif isinstance(download_data, dict):
                                    data = [download_data]
                                else:
                                    data = download_data
                            except Exception as e:
                                logger.error(f"❌ Failed to parse download response: {e}")
                                return {'success': False, 'error': f'Download parse error: {e}', 'status_code': 500}
                        else:
                            logger.error(f"❌ Download failed: HTTP {download_response.status_code}")
                            return {'success': False, 'error': f'Download failed: HTTP {download_response.status_code}', 'status_code': download_response.status_code}
                        
                        if data:
                            # Handle different response formats
                            profile_data = data[0] if isinstance(data, list) else data
                            
                            # Log the actual response format for debugging
                            logger.info(f"📊 Profile response keys: {list(profile_data.keys()) if isinstance(profile_data, dict) else f'Type: {type(profile_data)}'}")
                            
                            # More detailed logging of the response structure
                            if isinstance(profile_data, dict):
                                logger.info(f"📊 Response has {len(profile_data)} keys: {list(profile_data.keys())[:10]}")  # First 10 keys
                                # Accept any dictionary as valid profile data
                                logger.info(f"✅ Snapshot {status} - data downloaded successfully")
                                return {
                                    'success': True,
                                    'data': profile_data,
                                    'status_code': 200
                                }
                            elif isinstance(profile_data, list):
                                logger.info(f"📊 Response is list with {len(profile_data)} items")
                                if profile_data:
                                    first_item = profile_data[0]
                                    if isinstance(first_item, dict):
                                        logger.info("✅ Using first item from list as profile data")
                                        return {
                                            'success': True,
                                            'data': first_item,
                                            'status_code': 200
                                        }
                                logger.warning(f"⚠️ List format not usable: {profile_data[:2] if len(profile_data) > 2 else profile_data}")
                                return {'success': False, 'error': f'Unusable list format with {len(profile_data)} items', 'status_code': 422}
                            else:
                                logger.warning(f"⚠️ Unexpected profile data format: {type(profile_data)}, content: {str(profile_data)[:200]}")
                                return {'success': False, 'error': f'Unexpected profile data format: {type(profile_data)}', 'status_code': 422}
                        else:
                            logger.warning("⚠️ Snapshot completed but no data returned")
                            return {'success': False, 'error': 'No data in completed snapshot', 'status_code': 204}
                    
                    elif status in ['failed', 'error']:
                        error_msg = snapshot_data.get('error', 'Snapshot failed')
                        logger.error(f"❌ Snapshot failed: {error_msg}")
                        return {'success': False, 'error': error_msg, 'status_code': 500}
                    
                    elif status in ['collecting', 'building', 'running', 'pending', 'queued', 'processing']:
                        # Continue polling - these are intermediate statuses from Bright Data docs
                        if status == 'collecting':
                            logger.info("🔄 Data collection in progress...")
                        elif status == 'building':
                            logger.info("🔨 Snapshot building, almost ready...")
                        else:
                            logger.info(f"⏳ Status: {status} - continuing to poll...")
                        continue
                    
                    else:
                        logger.warning(f"⚠️ Unknown snapshot status: {status}")
                        continue
                        
                elif response.status_code == 202:
                    # HTTP 202 Accepted - Still processing, continue polling
                    logger.info("🔄 Snapshot still processing (HTTP 202) - continuing to poll...")
                    # Continue with sleep at the end of loop
                else:
                    logger.warning(f"⚠️ Polling request failed: HTTP {response.status_code}")
                    continue
                    
            except Exception as e:
                logger.warning(f"⚠️ Polling attempt {attempt + 1} failed: {str(e)}")
                
            # Sleep between attempts (except the last one)
            if attempt < max_attempts - 1:
                time.sleep(polling_interval)
        
        # Max attempts reached
        total_time = max_attempts * polling_interval
        logger.error(f"❌ Polling timeout after {max_attempts} attempts ({total_time} seconds)")
        return {'success': False, 'error': f'Polling timeout after {max_attempts} attempts ({total_time} seconds)', 'status_code': 408}

    def _make_profile_request(self, dataset_id: str, url: str, timeout: int = 120) -> Dict[str, Any]:
        """
        Make a profile request that requires polling for snapshots.
        
        Args:
            dataset_id (str): Bright Data profile dataset ID
            url (str): LinkedIn profile URL to scrape
            timeout (int): Request timeout in seconds
            
        Returns:
            Dict[str, Any]: Profile data or error information
        """
        # Initial request to trigger profile snapshot
        initial_result = self._make_request(dataset_id, url, timeout)
        
        if not initial_result.get('success'):
            return initial_result
        
        response_data = initial_result.get('data', {})
        
        # Check if we got immediate data (some profiles) or snapshot ID (most profiles)
        if isinstance(response_data, list) and response_data:
            # Immediate response with data
            logger.info("✅ Profile data received immediately")
            return {
                'success': True,
                'data': response_data[0],
                'status_code': 200
            }
        elif isinstance(response_data, dict) and response_data.get('snapshot_id'):
            # Async response - need to poll
            snapshot_id = response_data['snapshot_id']
            logger.info(f"🔄 Profile snapshot triggered, ID: {snapshot_id}")
            return self._poll_for_snapshot(snapshot_id, dataset_id)
        else:
            # Enhanced diagnostics for unexpected response format
            logger.warning("⚠️ Unexpected profile response format")
            
            if isinstance(response_data, dict):
                logger.info(f"📊 Response is dict with {len(response_data)} keys: {list(response_data.keys())[:10]}")
                # Check for other possible structures
                if 'data' in response_data:
                    logger.info("📊 Response has 'data' key - checking nested structure")
                    nested_data = response_data['data']
                    if isinstance(nested_data, list) and nested_data:
                        logger.info("📊 Found data in nested 'data' field, using first item")
                        return {
                            'success': True,
                            'data': nested_data[0],
                            'status_code': 200
                        }
                    elif isinstance(nested_data, dict):
                        logger.info("📊 Found dict in nested 'data' field, using directly")
                        return {
                            'success': True,
                            'data': nested_data,
                            'status_code': 200
                        }
                
                # Check if this might be the profile data itself (direct format)
                # LinkedIn profile data indicators based on Bright Data documentation
                linkedin_profile_fields = [
                    'name', 'headline', 'profile_url', 'current_company', 'linkedin_id', 
                    'position', 'about', 'experience', 'education', 'city', 'country_code',
                    'profile_info', 'followers', 'connections', 'avatar', 'url'
                ]
                
                if any(key in response_data for key in linkedin_profile_fields):
                    logger.info(f"📊 Response appears to be profile data directly (found: {[k for k in linkedin_profile_fields if k in response_data][:5]})")
                    return {
                        'success': True,
                        'data': response_data,
                        'status_code': 200
                    }
                
                # Check for error responses
                if 'error' in response_data or 'status' in response_data:
                    error_msg = response_data.get('error', 'API error')
                    status = response_data.get('status', 'unknown')
                    logger.warning(f"📊 API error response: {status} - {error_msg}")
                    return {
                        'success': False,
                        'error': f"{status}: {error_msg}",
                        'status_code': 422
                    }
                    
            elif isinstance(response_data, list):
                logger.info(f"📊 Response is list with {len(response_data)} items")
                if response_data:
                    logger.info("📊 Using first item from list")
                    return {
                        'success': True,
                        'data': response_data[0],
                        'status_code': 200
                    }
            else:
                logger.info(f"📊 Response type: {type(response_data)}")
                logger.info(f"📊 Response content (first 200 chars): {str(response_data)[:200]}")
            
            return {'success': False, 'error': 'Unexpected profile response format', 'status_code': 500}

    def _extract_poster_profile_url(self, job_data: Dict[str, Any]) -> Optional[str]:
        """Extract job poster profile URL from job data."""
        # Check job_poster nested object first (Bright Data API structure)
        job_poster = job_data.get('job_poster', {})
        if isinstance(job_poster, dict):
            poster_url = job_poster.get('url')
            if poster_url and 'linkedin.com/in/' in str(poster_url):
                return str(poster_url)
        
        # Try various fields where profile URL might be stored
        profile_url_fields = [
            'job_poster_profile_url',
            'poster_profile_url', 
            'hiring_manager_profile',
            'recruiter_profile_url',
            'contact_person_profile'
        ]
        
        for field in profile_url_fields:
            if field in job_data and job_data[field]:
                url = job_data[field]
                if 'linkedin.com/in/' in str(url):
                    return str(url)
        
        # Try to extract from nested contact information
        contact_info = job_data.get('contact_info', {})
        if isinstance(contact_info, dict):
            for key, value in contact_info.items():
                if 'linkedin' in str(key).lower() and 'linkedin.com/in/' in str(value):
                    return str(value)
        
        return None

    def _combine_job_and_poster_data(self, job_data: Dict[str, Any], poster_data: Dict[str, Any]) -> Dict[str, Any]:
        """Combine job and poster data into unified structure."""
        combined = {}
        
        # Job information
        combined.update({
            'job_title': job_data.get('job_title', ''),
            'company_name': job_data.get('company_name', ''),
            'job_description': job_data.get('job_description', ''),
            'location': job_data.get('location', ''),
            'salary_info': job_data.get('salary_info'),
            'requirements': job_data.get('requirements', []),
            'posted_date': job_data.get('posted_date'),
            'applicant_count': job_data.get('applicant_count'),
            'application_method': job_data.get('application_method', ''),
            'job_type': job_data.get('job_type'),
            'experience_level': job_data.get('experience_level'),
            'industry': job_data.get('industry'),
            'company_website': job_data.get('company_website'),
            'company_size': job_data.get('company_size'),
            'job_function': job_data.get('job_function'),
            'seniority_level': job_data.get('seniority_level'),
        })
        
        # Contact information
        contact_info = job_data.get('contact_info', {})
        combined.update({
            'contact_info': contact_info,
            'has_email': len(contact_info.get('emails', [])) > 0,
            'has_phone': len(contact_info.get('phones', [])) > 0,
            'has_whatsapp': len(contact_info.get('whatsapp', [])) > 0,
            'has_telegram': len(contact_info.get('telegram', [])) > 0,
        })
        
        # Poster information from job data
        combined.update({
            'job_poster_name': job_data.get('job_poster_name', ''),
            'job_poster_current_title': job_data.get('job_poster_current_title', ''),
            'job_poster_current_company': job_data.get('job_poster_current_company', ''),
        })
        
        # Enhanced poster information from profile data
        if poster_data:
            # Pass the entire profile data in 'poster' dict for data processor
            combined['poster'] = poster_data
            
            # Also extract specific fields for backward compatibility and display
            combined.update({
                'job_poster_headline': poster_data.get('headline', ''),
                'job_poster_location': poster_data.get('location', ''),
                'job_poster_summary': poster_data.get('summary', poster_data.get('about', '')),
                'job_poster_experiences': poster_data.get('experience', []),  # Note: 'experience' not 'experiences'
                'job_poster_education': poster_data.get('education', []),
                'job_poster_skills': poster_data.get('skills', []),
                'job_poster_connections': poster_data.get('connections', 0),
                'job_poster_followers': poster_data.get('followers', 0),
                'job_poster_has_photo': 1 if (poster_data.get('avatar') and not poster_data.get('default_avatar', False)) else 0,
                'job_poster_is_verified': 1 if poster_data.get('connections', 0) >= 500 else 0,  # Infer verification
                
                # Pass key Bright Data fields directly for data processor
                'avatar': poster_data.get('avatar'),
                'default_avatar': poster_data.get('default_avatar', False),
                'connections': poster_data.get('connections', 0),
                'followers': poster_data.get('followers', 0),
                'current_company': poster_data.get('current_company', {}),
                'experience': poster_data.get('experience', []),
                'education': poster_data.get('education', []),
                'activity': poster_data.get('activity', []),
                'recommendations_count': poster_data.get('recommendations_count', 0)
            })
        else:
            # Default values when no poster profile available
            combined.update({
                'job_poster_headline': '',
                'job_poster_location': '',
                'job_poster_summary': '',
                'job_poster_experiences': [],
                'job_poster_education': [],
                'job_poster_skills': [],
                'job_poster_connections': 0,
                'job_poster_followers': 0,
                'job_poster_has_photo': 0,
                'job_poster_is_verified': 0,
            })
        
        # Model compatibility fields - use VerificationService for consistent defaults
        # Only set if poster_data is not available (for fallback)
        if not poster_data:
            # Use verification service to get consistent default verification features
            from ..services.verification_service import VerificationService
            verification_service = VerificationService()
            verification_defaults = verification_service.extract_verification_features({})
            combined.update(verification_defaults)
        
        return combined

    def _calculate_fraud_detection_scores(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive fraud detection scores."""
        scores = {}
        
        # 1. Network Quality Score (0-1)
        connections = data.get('job_poster_connections', 0)
        experiences = len(data.get('job_poster_experiences', []))
        education = len(data.get('job_poster_education', []))
        
        network_score = 0.0
        if connections > 500:
            network_score += 0.4
        elif connections > 100:
            network_score += 0.2
        
        if experiences > 3:
            network_score += 0.3
        elif experiences > 1:
            network_score += 0.15
        
        if education > 0:
            network_score += 0.3
            
        scores['network_quality_score'] = min(network_score, 1.0)
        
        # 2. Profile Completeness Score (0-1)
        completeness_factors = [
            bool(data.get('job_poster_name')),
            bool(data.get('job_poster_current_title')),
            bool(data.get('job_poster_headline')),
            bool(data.get('job_poster_summary')),
            bool(data.get('job_poster_has_photo')),
            len(data.get('job_poster_experiences', [])) > 0,
            len(data.get('job_poster_education', [])) > 0,
            len(data.get('job_poster_skills', [])) > 0,
        ]
        
        scores['profile_completeness_score'] = sum(completeness_factors) / len(completeness_factors)
        
        # 3. Company Legitimacy Score (0-1)
        company_score = 0.5  # Neutral baseline
        
        if data.get('company_website'):
            company_score += 0.2
        if data.get('company_size'):
            company_score += 0.15
        if data.get('industry'):
            company_score += 0.15
            
        scores['company_legitimacy_score'] = min(company_score, 1.0)
        
        # 4. Poster Credibility Score (0-1)
        credibility_score = 0.0
        
        if data.get('job_poster_is_verified'):
            credibility_score += 0.3
        if data.get('job_poster_has_photo'):
            credibility_score += 0.2
        if connections > 50:
            credibility_score += 0.25
        if experiences > 2:
            credibility_score += 0.25
            
        scores['poster_credibility_score'] = min(credibility_score, 1.0)
        
        # 5. Overall Fraud Risk Score (0-1, lower is better)
        risk_factors = []
        
        # High risk factors
        if not data.get('job_poster_name'):
            risk_factors.append(0.3)
        if not data.get('company_name'):
            risk_factors.append(0.25)
        if connections < 10:
            risk_factors.append(0.2)
        if not data.get('job_poster_has_photo'):
            risk_factors.append(0.15)
        if experiences == 0:
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
            'api_version': '4.0.0'
        }


# MAIN PUBLIC FUNCTIONS

def scrape_job(url: str) -> Dict[str, Any]:
    """
    Scrape ONLY job data - returns in 2-3 seconds without waiting for profile.
    
    Args:
        url (str): The LinkedIn job posting URL to scrape
        
    Returns:
        Dict[str, Any]: Dictionary containing job information with optional profile URL:
            - Complete job details (title, description, requirements, salary)
            - Company information (size, industry, verification)
            - poster_profile_url: LinkedIn profile URL if found (for separate fetching)
            - Job-only fraud detection scores
    """
    logger.info(f"Starting job-only scraping: {url}")
    
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
        
        # Extract poster profile URL for separate fetching
        poster_profile_url = scraper._extract_poster_profile_url(job_data)
        if poster_profile_url:
            job_data['poster_profile_url'] = poster_profile_url
        
        # Calculate job-only fraud scores (without profile data)
        job_only_combined = scraper._combine_job_and_poster_data(job_data, {})
        fraud_scores = scraper._calculate_fraud_detection_scores(job_only_combined)
        job_data.update(fraud_scores)
        
        # Add metadata
        job_data.update({
            'success': True,
            'scraped_at': datetime.now().isoformat(),
            'scraping_method': 'bright_data_job_only',
            'data_source': 'bright_data_professional',
            'api_version': '4.0.0',
            'profile_available': bool(poster_profile_url)
        })
        
        logger.info(f"✅ Job-only scraping successful!")
        logger.info(f"📊 Job: '{job_data.get('job_title')}' at '{job_data.get('company_name')}'")
        logger.info(f"👤 Profile URL available: {bool(poster_profile_url)}")
        
        return job_data
        
    except Exception as e:
        error_msg = f"Job scraping error: {str(e)}"
        logger.error(error_msg)
        
        return {
            'success': False,
            'error_message': error_msg,
            'scraping_method': 'bright_data_job_error',
            'url': url,
            'scraped_at': datetime.now().isoformat(),
            'api_version': '4.0.0',
            'data_source': 'bright_data_professional'
        }


def scrape_profile(profile_url: str) -> Dict[str, Any]:
    """
    Scrape ONLY profile data - handles polling for LinkedIn profiles.
    
    Args:
        profile_url (str): The LinkedIn profile URL to scrape
        
    Returns:
        Dict[str, Any]: Dictionary containing profile information:
            - Profile details (name, title, headline, summary)
            - Experience and education history
            - Skills, connections, followers
            - Verification and credibility data
    """
    logger.info(f"Starting profile-only scraping: {profile_url}")
    
    try:
        # Get Bright Data API key
        config = get_bright_data_config()
        api_key = config.get('api_key', '')
        
        if not api_key:
            raise ValueError("Bright Data API key not configured. Please set BD_API_KEY environment variable.")
        
        # Create scraper and get profile data with polling
        scraper = BrightDataLinkedInScraper(api_key)
        
        if not scraper.dataset_ids.get('profiles'):
            raise ValueError("Profiles dataset not configured properly")
        
        profile_result = scraper._make_profile_request(
            dataset_id=scraper.dataset_ids['profiles'],
            url=profile_url,
            timeout=120
        )
        
        if profile_result.get('success'):
            profile_data = profile_result.get('data', {})
            
            # Add metadata
            profile_data.update({
                'success': True,
                'scraped_at': datetime.now().isoformat(),
                'scraping_method': 'bright_data_profile_only',
                'data_source': 'bright_data_professional',
                'api_version': '4.0.0',
                'profile_url': profile_url
            })
            
            logger.info("✅ Profile-only scraping successful!")
            logger.info(f"👤 Profile: '{profile_data.get('name')}' ({profile_data.get('headline')})")
            
            return profile_data
        else:
            logger.warning(f"⚠️ Profile scraping failed: {profile_result.get('error')}")
            return {
                'success': False,
                'error_message': profile_result.get('error', 'Profile scraping failed'),
                'scraping_method': 'bright_data_profile_failed',
                'profile_url': profile_url,
                'scraped_at': datetime.now().isoformat(),
                'api_version': '4.0.0',
                'data_source': 'bright_data_professional'
            }
        
    except Exception as e:
        error_msg = f"Profile scraping error: {str(e)}"
        logger.error(error_msg)
        
        return {
            'success': False,
            'error_message': error_msg,
            'scraping_method': 'bright_data_profile_error',
            'profile_url': profile_url,
            'scraped_at': datetime.now().isoformat(),
            'api_version': '4.0.0',
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
            # Poster information (not available from static HTML)
            'job_poster_name': None,
            'job_poster_current_title': None,
            'job_poster_current_company': None,
            'job_poster_has_photo': 0,
            'job_poster_is_verified': 0,
            'job_poster_experiences': [],
            # Model compatibility fields - use VerificationService for defaults
            # VerificationService will handle these when data is processed
            # Use VerificationService for consistent verification defaults
            # Fraud detection scores (not available from static HTML)
            'fraud_risk_score': 0.5,  # Default neutral score
            'network_quality_score': 0.0,
            'profile_completeness_score': 0.0,
            'company_legitimacy_score': 0.5,
            'poster_credibility_score': 0.0
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
            - network_quality_score: Calculated score (0-1)
            - legitimacy_score: Calculated score (0-1)
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
        
        # Calculate enrichment scores using VerificationService (single source of truth)
        from ..services.verification_service import VerificationService
        verification_service = VerificationService()
        enrichment_scores = verification_service.calculate_company_verification_scores(enriched_data)
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
    'scrape_profile',
    'scrape_company',
    'scrape_from_html',  # Keep for HTML fallback
    'validate_linkedin_url',
    'get_job_id_from_url',
    'BrightDataLinkedInScraper'
]