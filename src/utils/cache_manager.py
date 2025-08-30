"""
Cache Manager for Job Post Fraud Detector

This module provides caching functionality to store scraping results
and improve performance by avoiding repeated requests.

 Version: 1.0.0
"""

import hashlib
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from ..core.constants import UtilityConstants

logger = logging.getLogger(__name__)

# Cache storage
_cache_storage = {}
_cache_initialized = False


def initialize_cache() -> None:
    """
    Initialize the cache system and create necessary directories.
    
    Implementation by Orchestration Engineer - Infrastructure & Deployment:
        - Create cache directory if it doesn't exist
        - Load existing cache from disk
        - Set up cache cleanup scheduling
        - Initialize cache statistics
    """
    global _cache_initialized
    
    if _cache_initialized:
        return
    
    try:
        if UtilityConstants.CACHE_CONFIG.get('enable_cache', True):
            cache_dir = UtilityConstants.CACHE_CONFIG['cache_dir']
            os.makedirs(cache_dir, exist_ok=True)
            
            # Load existing cache from disk
            _load_cache_from_disk()
            
            logger.info("Cache system initialized")
        
        _cache_initialized = True
        
    except Exception as e:
        logger.error(f"Error initializing cache: {str(e)}")
        _cache_initialized = True  # Don't retry initialization


def cache_scraping_result(url: str, data: Dict[str, Any]) -> None:
    """
    Cache scraping result for a given URL.
    
    Args:
        url (str): The URL that was scraped
        data (Dict[str, Any]): The scraped data to cache
        
    Implementation by Orchestration Engineer - Infrastructure & Deployment:
        - Generate unique cache key from URL
        - Store data with timestamp
        - Handle cache size limits
        - Write to persistent storage
    """
    if not UtilityConstants.CACHE_CONFIG.get('enable_cache', True):
        return
    
    try:
        cache_key = _generate_cache_key(url)
        
        cache_entry = {
            'url': url,
            'data': data,
            'timestamp': time.time(),
            'created_at': datetime.now().isoformat()
        }
        
        _cache_storage[cache_key] = cache_entry
        
        # Persist to disk
        _save_cache_to_disk()
        
        logger.debug(f"Cached result for URL: {url}")
        
    except Exception as e:
        logger.error(f"Error caching result: {str(e)}")


def get_cached_result(url: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve cached result for a URL if it exists and is still valid.
    
    Args:
        url (str): The URL to look up in cache
        
    Returns:
        Optional[Dict[str, Any]]: Cached data if available, None otherwise
        
    Implementation by Orchestration Engineer - Infrastructure & Deployment:
        - Generate cache key and lookup
        - Check cache expiry
        - Return data or None based on validity
        - Update cache statistics
    """
    if not UtilityConstants.CACHE_CONFIG.get('enable_cache', True):
        return None
    
    try:
        cache_key = _generate_cache_key(url)
        
        if cache_key in _cache_storage:
            cache_entry = _cache_storage[cache_key]
            
            # Check if cache entry is still valid
            cache_age = time.time() - cache_entry['timestamp']
            max_age = UtilityConstants.CACHE_CONFIG.get('cache_expiry_days', 7) * 24 * 3600
            
            if cache_age < max_age:
                logger.debug(f"Cache hit for URL: {url}")
                return cache_entry['data']
            else:
                # Remove expired entry
                del _cache_storage[cache_key]
                logger.debug(f"Cache expired for URL: {url}")
        
        return None
        
    except Exception as e:
        logger.error(f"Error retrieving cached result: {str(e)}")
        return None


def clear_old_cache(days: int = 7) -> None:
    """
    Clear cache entries older than specified number of days.
    
    Args:
        days (int): Number of days to keep in cache (default: 7)
        
    Implementation by Orchestration Engineer - Infrastructure & Deployment:
        - Iterate through all cache entries
        - Remove entries older than specified days
        - Update persistent storage
        - Log cleanup statistics
    """
    try:
        cutoff_time = time.time() - (days * 24 * 3600)
        keys_to_remove = []
        
        for key, entry in _cache_storage.items():
            if entry['timestamp'] < cutoff_time:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del _cache_storage[key]
        
        if keys_to_remove:
            _save_cache_to_disk()
            logger.info(f"Cleared {len(keys_to_remove)} old cache entries")
        
    except Exception as e:
        logger.error(f"Error clearing old cache: {str(e)}")


def get_cache_statistics() -> Dict[str, Any]:
    """
    Get statistics about cache usage and performance.
    
    Returns:
        Dict[str, Any]: Cache statistics
        
    Implementation by Orchestration Engineer - Infrastructure & Deployment:
        - Total cache entries
        - Cache hit/miss rates
        - Cache size on disk
        - Average cache age
        - Most/least accessed entries
    """
    try:
        if not _cache_storage:
            return {
                'total_entries': 0,
                'cache_size_mb': 0,
                'oldest_entry': None,
                'newest_entry': None
            }
        
        timestamps = [entry['timestamp'] for entry in _cache_storage.values()]
        
        stats = {
            'total_entries': len(_cache_storage),
            'cache_size_mb': _estimate_cache_size_mb(),
            'oldest_entry': datetime.fromtimestamp(min(timestamps)).isoformat() if timestamps else None,
            'newest_entry': datetime.fromtimestamp(max(timestamps)).isoformat() if timestamps else None,
            'average_age_hours': (time.time() - sum(timestamps) / len(timestamps)) / 3600 if timestamps else 0
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting cache statistics: {str(e)}")
        return {'error': str(e)}


# Helper functions

def _generate_cache_key(url: str) -> str:
    """Generate a unique cache key for a URL."""
    return hashlib.md5(url.encode('utf-8')).hexdigest()


def _load_cache_from_disk() -> None:
    """Load cache from persistent storage."""
    try:
        cache_file = os.path.join(UtilityConstants.CACHE_CONFIG['cache_dir'], 'scraping_cache.json')
        
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                global _cache_storage
                _cache_storage = json.load(f)
            
            logger.debug(f"Loaded {len(_cache_storage)} cache entries from disk")
        
    except Exception as e:
        logger.warning(f"Could not load cache from disk: {str(e)}")


def _save_cache_to_disk() -> None:
    """Save cache to persistent storage."""
    try:
        cache_file = os.path.join(UtilityConstants.CACHE_CONFIG['cache_dir'], 'scraping_cache.json')
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(_cache_storage, f, indent=2, default=str)
        
        logger.debug("Cache saved to disk")
        
    except Exception as e:
        logger.warning(f"Could not save cache to disk: {str(e)}")


def _estimate_cache_size_mb() -> float:
    """Estimate cache size in memory."""
    try:
        import sys
        total_size = sys.getsizeof(_cache_storage)
        
        for entry in _cache_storage.values():
            total_size += sys.getsizeof(entry)
            
        return total_size / (1024 * 1024)  # Convert to MB
        
    except Exception:
        return 0.0