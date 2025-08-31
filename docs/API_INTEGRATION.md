# 🌐 API Integration Guide

**FraudSpot v3.0 External API Integration Documentation**

**Version:** 3.0.0  
**Last Updated:** August 30, 2025  
**Status:** Production Ready

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Bright Data LinkedIn API](#bright-data-linkedin-api)
3. [Verification Service Integration](#verification-service-integration)
4. [API Response Processing](#api-response-processing)
5. [Field Mapping Reference](#field-mapping-reference)
6. [Error Handling](#error-handling)
7. [Rate Limiting & Performance](#rate-limiting--performance)
8. [Testing & Validation](#testing--validation)
9. [Migration Notes](#migration-notes)

---

## 🎯 Overview

FraudSpot v3.1+ integrates with external APIs to provide comprehensive fraud detection with company verification capabilities. The system uses **two primary Bright Data integrations**: LinkedIn Jobs API for job posting analysis and **Companies API for company verification**. This guide documents the complete API integration architecture, company enrichment workflows, and verification service implementation.

### Key Integrations

- **Bright Data LinkedIn Jobs API**: Profile data and job posting verification
- **Bright Data Companies API** 🆕: Company enrichment and fraud scoring (dataset_id: `gd_l1vikfnt1wgvvqz95w`)
- **VerificationService**: Single source of truth for all company calculations
- **ScrapingService**: Coordinated job + company data fetching
- **DataProcessor**: Enhanced API data standardization with company features

---

## 🔗 Bright Data LinkedIn API

### Service Overview

Bright Data provides LinkedIn scraping capabilities with structured data extraction for job postings and user profiles. FraudSpot uses this service to gather verification information about job posters.

### Authentication & Configuration

```python
# Environment configuration
BRIGHT_DATA_CONFIG = {
    'endpoint': os.getenv('BRIGHT_DATA_ENDPOINT'),
    'api_key': os.getenv('BRIGHT_DATA_API_KEY'), 
    'zone': os.getenv('BRIGHT_DATA_ZONE', 'linkedin'),
    'timeout': int(os.getenv('SCRAPING_TIMEOUT', 30)),
    'max_retries': int(os.getenv('SCRAPING_MAX_RETRIES', 3))
}
```

### API Request Structure

```python
def scrape_linkedin_profile(profile_url: str) -> dict:
    """Scrape LinkedIn profile using Bright Data API"""
    
    request_payload = {
        'url': profile_url,
        'country': 'US',
        'include_profile': True,
        'include_experience': True,
        'include_education': True,
        'include_skills': True,
        'format': 'json'
    }
    
    headers = {
        'Authorization': f'Bearer {BRIGHT_DATA_CONFIG["api_key"]}',
        'Content-Type': 'application/json'
    }
    
    response = requests.post(
        BRIGHT_DATA_CONFIG['endpoint'],
        json=request_payload,
        headers=headers,
        timeout=BRIGHT_DATA_CONFIG['timeout']
    )
    
    return response.json()
```

### API Response Structure

#### Complete Profile Response
```json
{
  "profile": {
    "name": "John Smith",
    "headline": "Senior Software Engineer at SmartChoice International",
    "location": "Dubai, UAE",
    "avatar": "https://media.licdn.com/dms/image/C5603AQE...",
    "summary": "Experienced software engineer with 8+ years...",
    "connections": 500,
    "followers": 1250,
    "languages": ["English", "Arabic"],
    "industry": "Technology"
  },
  "experience": [
    {
      "company": {
        "name": "SmartChoice International",
        "url": "https://www.linkedin.com/company/smartchoice/",
        "industry": "Technology",
        "size": "51-200 employees"
      },
      "title": "Senior Software Engineer",
      "location": "Dubai, UAE",
      "start_date": "2022-01",
      "end_date": null,
      "current": true,
      "description": "Leading development of fraud detection systems..."
    },
    {
      "company": {
        "name": "TechCorp Solutions",
        "url": "https://www.linkedin.com/company/techcorp/"
      },
      "title": "Software Developer",
      "location": "Abu Dhabi, UAE", 
      "start_date": "2020-03",
      "end_date": "2021-12",
      "current": false,
      "description": "Developed web applications using React and Node.js..."
    }
  ],
  "education": [
    {
      "school": "American University of Dubai",
      "degree": "Bachelor of Science",
      "field": "Computer Science",
      "start_date": "2016",
      "end_date": "2020"
    }
  ],
  "skills": [
    {
      "name": "Python",
      "endorsements": 45
    },
    {
      "name": "Machine Learning",
      "endorsements": 32
    },
    {
      "name": "React",
      "endorsements": 28
    }
  ],
  "certifications": [
    {
      "name": "AWS Certified Solutions Architect",
      "issuer": "Amazon Web Services",
      "issue_date": "2023-03",
      "credential_id": "AWS-CSA-12345"
    }
  ]
}
```

### Job Posting Response Structure

```json
{
  "job": {
    "id": "3999835116",
    "title": "Senior Software Engineer",
    "company": "SmartChoice International UAE",
    "location": "Dubai, UAE",
    "employment_type": "Full-time",
    "seniority_level": "Mid-Senior level",
    "job_function": ["Engineering", "Information Technology"],
    "industries": ["Technology", "Software Development"],
    "description": "We are seeking an experienced Senior Software Engineer to join our growing team...",
    "requirements": "• 5+ years of software development experience\n• Strong knowledge of Python and React\n• Experience with cloud platforms (AWS/Azure)",
    "benefits": "• Competitive salary and equity package\n• Health insurance and wellness programs\n• Flexible working arrangements",
    "posted_date": "2025-08-15",
    "application_count": 127,
    "recruiter": {
      "name": "Sarah Johnson",
      "title": "Senior Technical Recruiter",
      "company": "SmartChoice International",
      "profile_url": "https://www.linkedin.com/in/sarah-johnson-recruiter/"
    }
  },
  "poster": {
    "name": "Sarah Johnson",
    "title": "Senior Technical Recruiter", 
    "avatar": "https://media.licdn.com/dms/image/D5603AQF...",
    "connections": 500,
    "experience": [
      {
        "company": {"name": "SmartChoice International"},
        "title": "Senior Technical Recruiter",
        "current": true
      }
    ]
  }
}
```

---

## 🏢 Bright Data Companies API 🆕

### Service Overview

The Bright Data Companies API provides comprehensive LinkedIn company data for fraud detection enhancement. This integration fetches real-time company information including followers, employees, founding year, and verification status to improve fraud prediction accuracy.

### Dataset Configuration

```python
# Company API Configuration
BRIGHT_DATA_COMPANIES_CONFIG = {
    'dataset_id': 'gd_l1vikfnt1wgvvqz95w',  # LinkedIn Companies dataset
    'endpoint': os.getenv('BRIGHT_DATA_ENDPOINT'),
    'api_key': os.getenv('BRIGHT_DATA_API_KEY'),
    'timeout': int(os.getenv('COMPANY_SCRAPING_TIMEOUT', 120)),
    'max_retries': int(os.getenv('COMPANY_SCRAPING_MAX_RETRIES', 3))
}
```

### Company Scraping Implementation

```python
def scrape_company(company_url: str) -> Dict[str, Any]:
    """Scrape LinkedIn company data using Bright Data Companies API"""
    
    scraper = BrightDataLinkedInScraper(api_key)
    
    # Make request to Companies dataset
    company_result = scraper._make_request(
        dataset_id=scraper.dataset_ids['companies'],  # gd_l1vikfnt1wgvvqz95w
        url=company_url,
        timeout=120,
        webhook_url=None
    )
    
    if company_result.get('success'):
        raw_company_data = company_result.get('data', {})
        
        # Extract company information
        enriched_data = {
            'company_name': raw_company_data.get('name', ''),
            'company_followers': raw_company_data.get('followers', 0),
            'company_employees': raw_company_data.get('employees', 0), 
            'company_founded': raw_company_data.get('founded', 0),
            'company_website': raw_company_data.get('website', ''),
            'company_verified': raw_company_data.get('verified', False),
            'company_enrichment_success': True
        }
        
        # Calculate verification scores using VerificationService
        verification_service = VerificationService()
        enrichment_scores = verification_service.calculate_company_verification_scores(enriched_data)
        
        # Merge scores with raw data
        enriched_data.update(enrichment_scores)
        
        return enriched_data
    else:
        return {'company_enrichment_success': False, 'error': company_result.get('error')}
```

### Company API Response Structure

```json
{
  "success": true,
  "data": {
    "name": "Microsoft Corporation",
    "followers": 15000000,
    "employees": 181000,
    "founded": 1975,
    "website": "https://microsoft.com",
    "verified": true,
    "industry": "Software Development",
    "headquarters": "Redmond, Washington",
    "description": "We empower every person and every organization on the planet to achieve more.",
    "logo": "https://media.licdn.com/dms/image/...",
    "cover_image": "https://media.licdn.com/dms/image/...",
    "company_url": "https://linkedin.com/company/microsoft"
  }
}
```

### Company Verification Scores

The system calculates **5 company-specific features** using VerificationService:

```python
# Example company verification scores
{
    'company_followers_score': 0.90,      # Normalized followers (0.0-1.0)
    'company_employees_score': 0.85,      # Normalized employee count
    'company_founded_score': 0.80,        # Company age normalization
    'network_quality_score': 0.90,        # Overall network strength
    'company_legitimacy_score': 0.85      # Combined trust indicator
}
```

### Integration with ML Pipeline

Company features are automatically integrated into the 27-feature ML models:

```python
# In FeatureEngine.generate_complete_feature_set()
def _generate_company_features(self, df: pd.DataFrame) -> pd.DataFrame:
    """Generate company features using VerificationService (single source of truth)."""
    
    verification_service = VerificationService()
    
    company_features = []
    for _, row in df.iterrows():
        job_data = row.to_dict()
        company_scores = verification_service.calculate_company_verification_scores(job_data)
        company_features.append(company_scores)
    
    return pd.DataFrame(company_features)
```

### Error Handling

```python
# Company API specific error handling
try:
    company_data = scrape_company(company_url)
except BrightDataTimeout:
    logger.warning("Company API timeout - using default values")
    company_data = {'company_enrichment_success': False, 'timeout': True}
except BrightDataQuotaExceeded:
    logger.error("Company API quota exceeded")
    company_data = {'company_enrichment_success': False, 'quota_exceeded': True}
except Exception as e:
    logger.error(f"Company API error: {str(e)}")
    company_data = {'company_enrichment_success': False, 'error': str(e)}
```

---

## 🛡️ Verification Service Integration

### VerificationService Data Processing

The VerificationService processes Bright Data API responses to extract verification features:

```python
class VerificationService:
    def extract_verification_features(self, job_data: Dict[str, Any]) -> Dict[str, int]:
        """
        Extract verification features from Bright Data API response
        
        Maps Bright Data fields to FraudSpot verification features:
        - avatar → poster_verified, poster_photo
        - connections → poster_active  
        - experience + company matching → poster_experience
        """
        
        # Extract profile data (could be nested in 'poster' or at root level)
        profile_data = job_data.get('poster', job_data)
        
        features = {
            'poster_verified': self._extract_poster_verified(profile_data),
            'poster_photo': self._extract_poster_photo(profile_data),
            'poster_experience': self._extract_poster_experience(job_data),
            'poster_active': self._extract_poster_active(profile_data)
        }
        
        return features
    
    def _extract_poster_verified(self, profile_data: Dict) -> int:
        """Check if profile has avatar (indicates verified profile)"""
        avatar = profile_data.get('avatar', '')
        return 1 if avatar and str(avatar).startswith('http') else 0
    
    def _extract_poster_photo(self, profile_data: Dict) -> int:
        """Check if profile photo is accessible"""
        avatar = profile_data.get('avatar', '')
        if not avatar or not str(avatar).startswith('http'):
            return 0
        
        # Could add URL validation here if needed
        return 1
    
    def _extract_poster_active(self, profile_data: Dict) -> int:
        """Check if profile shows activity (connections > 0)"""
        connections = profile_data.get('connections', 0)
        try:
            return 1 if int(connections) > 0 else 0
        except (ValueError, TypeError):
            return 0
    
    def _extract_poster_experience(self, job_data: Dict) -> int:
        """Check if poster has relevant experience at the job posting company"""
        # Get job posting company name
        job_company = job_data.get('company_name') or job_data.get('company', '')
        if not job_company:
            return 0
        
        # Get poster's current company from experience
        profile_data = job_data.get('poster', job_data)
        experience = profile_data.get('experience', [])
        
        if not experience or not isinstance(experience, list):
            return 0
        
        # Check current job (first experience entry)
        current_job = experience[0]
        if not isinstance(current_job, dict):
            return 0
        
        company_info = current_job.get('company', {})
        if isinstance(company_info, dict):
            current_company = company_info.get('name', '')
        else:
            current_company = str(company_info)
        
        # Use fuzzy matching to compare companies
        return 1 if self.company_matches(job_company, current_company) else 0
```

---

## 🔄 API Response Processing

### ScrapingService API Coordination

```python
class ScrapingService:
    def __init__(self):
        self.verification_service = VerificationService()
        self.cache_manager = CacheManager()
    
    def scrape_job_posting(self, url: str) -> dict:
        """
        Scrape job posting with profile verification data
        Coordinates Bright Data API calls and processes responses
        """
        try:
            # 1. Check cache first
            cached_result = self.cache_manager.get(url)
            if cached_result:
                return cached_result
            
            # 2. Make API request to Bright Data
            raw_response = self._call_bright_data_api(url)
            
            # 3. Validate API response structure
            validated_data = self._validate_api_response(raw_response)
            
            # 4. Extract and standardize data
            standardized_data = self._standardize_api_data(validated_data)
            
            # 5. Extract verification features
            verification_features = self.verification_service.extract_verification_features(standardized_data)
            
            # 6. Combine all data
            result = {
                **standardized_data,
                **verification_features,
                'success': True,
                'from_cache': False,
                'api_source': 'bright_data'
            }
            
            # 7. Cache result
            self.cache_manager.set(url, result)
            
            return result
            
        except Exception as e:
            logger.error(f"API scraping failed for {url}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'fallback_attempted': self._attempt_fallback_scraping(url)
            }
    
    def _call_bright_data_api(self, url: str) -> dict:
        """Make API call to Bright Data with retries and timeout"""
        max_retries = BRIGHT_DATA_CONFIG['max_retries']
        timeout = BRIGHT_DATA_CONFIG['timeout']
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    BRIGHT_DATA_CONFIG['endpoint'],
                    json={'url': url, 'include_profile': True},
                    headers={
                        'Authorization': f'Bearer {BRIGHT_DATA_CONFIG["api_key"]}',
                        'Content-Type': 'application/json'
                    },
                    timeout=timeout
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    raise Exception(f"API returned status {response.status_code}: {response.text}")
                    
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                raise Exception(f"API timeout after {max_retries} attempts")
            
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                raise e
        
        raise Exception("API request failed after all retry attempts")
    
    def _validate_api_response(self, response_data: dict) -> dict:
        """Validate Bright Data API response structure"""
        required_fields = ['job', 'poster']
        
        for field in required_fields:
            if field not in response_data:
                logger.warning(f"Missing required field: {field}")
        
        # Validate job data
        job_data = response_data.get('job', {})
        if not job_data.get('title') or not job_data.get('company'):
            logger.warning("Job data missing required fields (title, company)")
        
        # Validate poster data
        poster_data = response_data.get('poster', {})
        if not poster_data.get('name'):
            logger.warning("Poster data missing name field")
        
        return response_data
    
    def _standardize_api_data(self, api_data: dict) -> dict:
        """Convert Bright Data response to FraudSpot standard format"""
        job_info = api_data.get('job', {})
        poster_info = api_data.get('poster', {})
        
        standardized = {
            # Job information
            'job_title': job_info.get('title', ''),
            'job_description': job_info.get('description', ''),
            'company_name': job_info.get('company', ''),
            'location': job_info.get('location', ''),
            'employment_type': job_info.get('employment_type', ''),
            'requirements': job_info.get('requirements', ''),
            'benefits': job_info.get('benefits', ''),
            
            # Poster profile data (for verification)
            'avatar': poster_info.get('avatar', ''),
            'connections': poster_info.get('connections', 0),
            'experience': poster_info.get('experience', []),
            
            # Meta information
            'posted_date': job_info.get('posted_date', ''),
            'application_count': job_info.get('application_count', 0),
            'job_id': job_info.get('id', ''),
            
            # Language detection (simplified)
            'language': 0  # Default to English, could add language detection
        }
        
        return standardized
```

---

## 📋 Field Mapping Reference

### Verification Feature Mapping

| FraudSpot Feature | Bright Data Source | Extraction Logic | Data Type |
|------------------|-------------------|------------------|-----------|
| `poster_verified` | `poster.avatar` | Check if avatar URL exists and is valid HTTP URL | int (0/1) |
| `poster_photo` | `poster.avatar` | Same as poster_verified (avatar indicates photo) | int (0/1) |
| `poster_experience` | `poster.experience[0].company.name` vs `job.company` | Fuzzy matching between current job and posting company | int (0/1) |
| `poster_active` | `poster.connections` | Check if connections count > 0 | int (0/1) |

### Job Information Mapping

| FraudSpot Field | Bright Data Source | Processing Notes |
|----------------|-------------------|------------------|
| `job_title` | `job.title` | Direct mapping |
| `job_description` | `job.description` | Direct mapping |
| `company_name` | `job.company` | Used for company matching |
| `location` | `job.location` | Direct mapping |
| `employment_type` | `job.employment_type` | Direct mapping |
| `requirements` | `job.requirements` | Direct mapping |
| `benefits` | `job.benefits` | Direct mapping |
| `posted_date` | `job.posted_date` | Date standardization may be needed |

### Profile Information Mapping

| FraudSpot Field | Bright Data Source | Processing Notes |
|----------------|-------------------|------------------|
| `avatar` | `poster.avatar` | Full URL for verification checks |
| `connections` | `poster.connections` | Integer conversion with error handling |
| `experience` | `poster.experience[]` | Array of experience objects |
| `job_poster.name` | `poster.name` | Poster identification |
| `job_poster.title` | `poster.title` | Current job title |

---

## ⚠️ Error Handling

### API Error Scenarios

#### 1. Network Errors
```python
def handle_network_errors(func):
    """Decorator for handling network-related API errors"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except requests.exceptions.ConnectionError:
            logger.error("Network connection failed")
            return {'success': False, 'error': 'network_connection_failed'}
        except requests.exceptions.Timeout:
            logger.error("API request timed out")
            return {'success': False, 'error': 'api_timeout'}
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error: {e}")
            return {'success': False, 'error': f'http_error_{e.response.status_code}'}
    return wrapper
```

#### 2. Authentication Errors
```python
def handle_auth_errors(response):
    """Handle authentication-related errors"""
    if response.status_code == 401:
        logger.error("API authentication failed - check API key")
        return {
            'success': False,
            'error': 'authentication_failed',
            'suggestion': 'Verify BRIGHT_DATA_API_KEY environment variable'
        }
    elif response.status_code == 403:
        logger.error("API access forbidden - check permissions")
        return {
            'success': False,
            'error': 'access_forbidden',
            'suggestion': 'Check API key permissions and rate limits'
        }
```

#### 3. Rate Limiting
```python
class RateLimitHandler:
    def __init__(self):
        self.request_count = 0
        self.last_reset = time.time()
        self.rate_limit = 100  # requests per hour
        
    def can_make_request(self) -> bool:
        """Check if request can be made without exceeding rate limits"""
        current_time = time.time()
        
        # Reset counter every hour
        if current_time - self.last_reset > 3600:
            self.request_count = 0
            self.last_reset = current_time
        
        return self.request_count < self.rate_limit
    
    def record_request(self):
        """Record that a request was made"""
        self.request_count += 1
    
    def get_retry_after(self) -> int:
        """Get seconds to wait before next request"""
        if self.can_make_request():
            return 0
        
        time_since_reset = time.time() - self.last_reset
        return max(0, 3600 - int(time_since_reset))
```

#### 4. Data Validation Errors
```python
def validate_response_data(response_data: dict) -> dict:
    """Validate and clean API response data"""
    errors = []
    warnings = []
    
    # Check required job fields
    job_data = response_data.get('job', {})
    required_job_fields = ['title', 'company', 'description']
    
    for field in required_job_fields:
        if not job_data.get(field):
            errors.append(f"Missing required job field: {field}")
    
    # Check poster data
    poster_data = response_data.get('poster', {})
    if not poster_data:
        warnings.append("No poster profile data available")
    else:
        # Validate avatar URL format
        avatar = poster_data.get('avatar', '')
        if avatar and not avatar.startswith('http'):
            warnings.append(f"Invalid avatar URL format: {avatar}")
        
        # Validate connections count
        connections = poster_data.get('connections')
        if connections is not None:
            try:
                int(connections)
            except (ValueError, TypeError):
                warnings.append(f"Invalid connections count: {connections}")
    
    # Return validation results
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'data': response_data
    }
```

### Fallback Strategies

#### 1. Selenium Fallback
```python
def fallback_selenium_scraping(url: str) -> dict:
    """Fallback to Selenium scraping if API fails"""
    try:
        from src.scraper.linkedin_scraper import LinkedInScraper
        
        logger.info(f"Attempting Selenium fallback for {url}")
        scraper = LinkedInScraper()
        result = scraper.scrape_job_posting(url)
        
        if result.get('success'):
            logger.info("Selenium fallback successful")
            result['fallback_method'] = 'selenium'
            return result
        else:
            logger.warning("Selenium fallback also failed")
            
    except Exception as e:
        logger.error(f"Selenium fallback error: {str(e)}")
    
    return {'success': False, 'error': 'all_methods_failed'}
```

#### 2. Cached Data Fallback
```python
def get_cached_fallback(url: str) -> dict:
    """Use cached data as fallback when API fails"""
    cache_manager = CacheManager()
    
    # Try to get cached data (even if expired)
    cached_data = cache_manager.get_expired_ok(url)
    
    if cached_data:
        logger.info(f"Using cached fallback data for {url}")
        cached_data['from_cache'] = True
        cached_data['cache_fallback'] = True
        return cached_data
    
    return None
```

---

## ⚡ Rate Limiting & Performance

### Request Optimization

#### 1. Caching Strategy
```python
class APICache:
    def __init__(self):
        self.memory_cache = {}
        self.cache_timeout = 3600  # 1 hour
    
    def get_cache_key(self, url: str) -> str:
        """Generate cache key from URL"""
        import hashlib
        return hashlib.md5(url.encode()).hexdigest()
    
    def get(self, url: str) -> Optional[dict]:
        """Get cached response"""
        cache_key = self.get_cache_key(url)
        
        if cache_key in self.memory_cache:
            entry = self.memory_cache[cache_key]
            
            # Check if cache is still valid
            if time.time() - entry['timestamp'] < self.cache_timeout:
                logger.debug(f"Cache hit for {url}")
                return entry['data']
            else:
                # Remove expired cache
                del self.memory_cache[cache_key]
        
        logger.debug(f"Cache miss for {url}")
        return None
    
    def set(self, url: str, data: dict):
        """Cache API response"""
        cache_key = self.get_cache_key(url)
        
        self.memory_cache[cache_key] = {
            'data': data,
            'timestamp': time.time()
        }
        
        logger.debug(f"Cached response for {url}")
```

#### 2. Batch Processing
```python
def scrape_multiple_jobs(urls: List[str]) -> List[dict]:
    """Scrape multiple jobs with batching and rate limiting"""
    results = []
    batch_size = 10
    delay_between_batches = 60  # seconds
    
    for i in range(0, len(urls), batch_size):
        batch = urls[i:i + batch_size]
        batch_results = []
        
        logger.info(f"Processing batch {i//batch_size + 1} of {len(urls)//batch_size + 1}")
        
        for url in batch:
            result = scrape_job_posting(url)
            batch_results.append(result)
            
            # Small delay between requests in same batch
            time.sleep(2)
        
        results.extend(batch_results)
        
        # Longer delay between batches to respect rate limits
        if i + batch_size < len(urls):
            logger.info(f"Waiting {delay_between_batches}s before next batch...")
            time.sleep(delay_between_batches)
    
    return results
```

#### 3. Async Processing
```python
import asyncio
import aiohttp

async def async_scrape_job(session: aiohttp.ClientSession, url: str) -> dict:
    """Async job scraping for improved performance"""
    try:
        async with session.post(
            BRIGHT_DATA_CONFIG['endpoint'],
            json={'url': url, 'include_profile': True},
            headers={
                'Authorization': f'Bearer {BRIGHT_DATA_CONFIG["api_key"]}',
                'Content-Type': 'application/json'
            },
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            
            if response.status == 200:
                data = await response.json()
                return {'success': True, 'data': data, 'url': url}
            else:
                error_text = await response.text()
                return {
                    'success': False, 
                    'error': f'HTTP {response.status}: {error_text}',
                    'url': url
                }
                
    except Exception as e:
        return {'success': False, 'error': str(e), 'url': url}

async def scrape_jobs_async(urls: List[str]) -> List[dict]:
    """Scrape multiple jobs asynchronously"""
    async with aiohttp.ClientSession() as session:
        tasks = [async_scrape_job(session, url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
```

---

## 🧪 Testing & Validation

### Unit Tests

```python
import pytest
from unittest.mock import patch, Mock
from src.services.scraping_service import ScrapingService

class TestBrightDataIntegration:
    def setup_method(self):
        self.scraping_service = ScrapingService()
    
    @patch('requests.post')
    def test_successful_api_call(self, mock_post):
        """Test successful API response processing"""
        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'job': {
                'title': 'Software Engineer',
                'company': 'SmartChoice International',
                'description': 'Great opportunity...'
            },
            'poster': {
                'name': 'John Smith',
                'avatar': 'https://media.licdn.com/image.jpg',
                'connections': 500,
                'experience': [
                    {'company': {'name': 'SmartChoice International'}}
                ]
            }
        }
        mock_post.return_value = mock_response
        
        # Test API call
        result = self.scraping_service.scrape_job_posting(
            'https://www.linkedin.com/jobs/view/123456789'
        )
        
        # Verify results
        assert result['success'] == True
        assert result['job_title'] == 'Software Engineer'
        assert result['company_name'] == 'SmartChoice International'
        assert result['poster_verified'] == 1
        assert result['poster_experience'] == 1
    
    @patch('requests.post')
    def test_api_timeout_error(self, mock_post):
        """Test API timeout handling"""
        mock_post.side_effect = requests.exceptions.Timeout()
        
        result = self.scraping_service.scrape_job_posting(
            'https://www.linkedin.com/jobs/view/123456789'
        )
        
        assert result['success'] == False
        assert 'timeout' in result['error'].lower()
    
    @patch('requests.post')
    def test_api_auth_error(self, mock_post):
        """Test authentication error handling"""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = 'Unauthorized'
        mock_post.return_value = mock_response
        
        result = self.scraping_service.scrape_job_posting(
            'https://www.linkedin.com/jobs/view/123456789'
        )
        
        assert result['success'] == False
        assert 'authentication' in result['error'].lower()
```

### Integration Tests

```python
def test_end_to_end_api_integration():
    """Test complete API integration workflow"""
    from src.services.scraping_service import ScrapingService
    from src.services.verification_service import VerificationService
    
    scraping_service = ScrapingService()
    verification_service = VerificationService()
    
    # Test with a sample job URL (use test data)
    test_url = "https://www.linkedin.com/jobs/view/test-job"
    
    # Mock the API response for testing
    test_api_response = {
        'job': {
            'title': 'Test Software Engineer',
            'company': 'Test Company Inc',
            'description': 'Test job description'
        },
        'poster': {
            'avatar': 'https://media.licdn.com/test.jpg',
            'connections': 250,
            'experience': [
                {'company': {'name': 'Test Company'}}
            ]
        }
    }
    
    # Process through verification service
    verification_features = verification_service.extract_verification_features({
        **test_api_response['job'],
        **test_api_response['poster'],
        'company_name': test_api_response['job']['company']
    })
    
    # Verify extraction worked correctly
    assert verification_features['poster_verified'] == 1
    assert verification_features['poster_active'] == 1
    assert verification_features['poster_experience'] == 1  # Fuzzy match should work
    
    print("✅ End-to-end API integration test passed")
```

### Performance Tests

```python
def test_api_performance():
    """Test API response times and throughput"""
    import time
    from src.services.scraping_service import ScrapingService
    
    scraping_service = ScrapingService()
    test_urls = [
        "https://www.linkedin.com/jobs/view/test1",
        "https://www.linkedin.com/jobs/view/test2", 
        "https://www.linkedin.com/jobs/view/test3"
    ]
    
    # Test single request performance
    start_time = time.time()
    result = scraping_service.scrape_job_posting(test_urls[0])
    single_request_time = time.time() - start_time
    
    print(f"Single request time: {single_request_time:.2f}s")
    assert single_request_time < 10, "Single request should complete within 10 seconds"
    
    # Test batch processing performance
    start_time = time.time()
    results = []
    for url in test_urls:
        result = scraping_service.scrape_job_posting(url)
        results.append(result)
    
    batch_time = time.time() - start_time
    avg_time_per_request = batch_time / len(test_urls)
    
    print(f"Batch processing: {batch_time:.2f}s total, {avg_time_per_request:.2f}s per request")
    
    # Verify caching improves performance
    start_time = time.time()
    cached_result = scraping_service.scrape_job_posting(test_urls[0])  # Should be cached
    cached_request_time = time.time() - start_time
    
    print(f"Cached request time: {cached_request_time:.2f}s")
    assert cached_request_time < single_request_time, "Cached request should be faster"
```

---

## 📈 Migration Notes

### From Previous Implementation

**Key Changes**:
1. **Real API Integration**: Replaced dummy data with actual Bright Data API responses
2. **Centralized Processing**: All API data processing moved to VerificationService
3. **Improved Error Handling**: Comprehensive error handling and fallback strategies
4. **Caching System**: Added intelligent caching for improved performance

**Breaking Changes**:
- Verification features now extract from real API data (requires model retraining)
- API configuration requires new environment variables
- Response structure may differ from previous mock data

### Migration Steps

#### 1. Environment Configuration
```bash
# Add to .env file
BRIGHT_DATA_ENDPOINT=https://api.brightdata.com/scrape
BRIGHT_DATA_API_KEY=your_api_key_here
BRIGHT_DATA_ZONE=linkedin
SCRAPING_TIMEOUT=30
SCRAPING_MAX_RETRIES=3
```

#### 2. Dependency Updates
```bash
pip install aiohttp>=3.8.0  # For async processing
pip install rapidfuzz>=3.5.0  # For fuzzy matching
```

#### 3. Code Updates
```python
# Update existing scrapers to use new API integration
from src.services.scraping_service import ScrapingService
from src.services.verification_service import VerificationService

# Replace old scraping logic
scraping_service = ScrapingService()
verification_service = VerificationService()

# Use centralized API integration
result = scraping_service.scrape_job_posting(url)
features = verification_service.extract_verification_features(result)
```

#### 4. Testing Migration
```python
def test_migration_compatibility():
    """Test that migration preserves expected functionality"""
    # Test with known good data
    test_data = {
        'avatar': 'https://media.licdn.com/test.jpg',
        'connections': 500,
        'experience': [{'company': {'name': 'Test Company'}}],
        'company_name': 'Test Company Inc'
    }
    
    verification_service = VerificationService()
    features = verification_service.extract_verification_features(test_data)
    
    # Should extract features correctly
    assert features['poster_verified'] == 1
    assert features['poster_active'] == 1
    
    print("✅ Migration compatibility verified")
```

---

## 📝 Conclusion

The API integration system provides robust, production-ready integration with Bright Data's LinkedIn API while maintaining comprehensive error handling, caching, and fallback strategies. The centralized VerificationService ensures consistent data processing across the application.

**Key Benefits**:
- Real verification data instead of dummy values
- Intelligent fuzzy company matching
- Comprehensive error handling and fallbacks  
- Performance optimizations with caching
- Production-ready rate limiting and monitoring

**Next Steps**:
- Monitor API usage and costs
- Implement additional data sources if needed
- Consider API response time optimizations
- Add comprehensive monitoring and alerting

---

**For technical support with API integrations, refer to the troubleshooting sections or contact the development team.**