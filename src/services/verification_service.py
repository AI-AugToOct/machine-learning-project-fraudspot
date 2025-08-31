"""
Verification Service - Centralized Verification Logic

This service provides a single source of truth for all verification-related operations
across the fraud detection system. It handles verification feature extraction,
score calculation, company matching, and UI display data generation.

Version: 1.0.0 - Centralized Verification
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from rapidfuzz import fuzz
except ImportError:
    # Fallback to difflib if rapidfuzz not available
    import difflib
    fuzz = None

logger = logging.getLogger(__name__)


class VerificationService:
    """
    Centralized service for all verification-related operations.
    
    This service handles:
    - Verification feature extraction from job data
    - Poster verification score calculation (0-4)
    - Company name fuzzy matching
    - UI display data generation
    - Verification badge information
    """
    
    def __init__(self):
        """Initialize the verification service."""
        if fuzz is None:
            logger.warning("rapidfuzz not available, using difflib fallback for company matching")
        logger.info("VerificationService initialized")
    
    def extract_verification_features(self, job_data: Dict[str, Any]) -> Dict[str, int]:
        """
        Extract the 4 core verification fields from job data.
        
        Handles multiple field name formats:
        - poster_verified vs job_poster_is_verified
        - poster_photo vs job_poster_has_photo
        - poster_experience (ML field)
        - poster_active (ML field)
        
        Args:
            job_data: Job posting data dict
            
        Returns:
            Dict with verified 4 verification fields (0 or 1 each)
        """
        verification = {}
        
        # 1. POSTER_VERIFIED - Account/profile verified
        verification['poster_verified'] = int(bool(
            job_data.get('poster_verified', 0) or 
            job_data.get('job_poster_is_verified', 0)
        ))
        
        # 2. POSTER_PHOTO - Has profile photo
        verification['poster_photo'] = int(bool(
            job_data.get('poster_photo', 0) or
            job_data.get('job_poster_has_photo', 0)
        ))
        
        # 3. POSTER_EXPERIENCE - Relevant experience/company match
        verification['poster_experience'] = int(bool(
            job_data.get('poster_experience', 0)
        ))
        
        # 4. POSTER_ACTIVE - Recent activity
        verification['poster_active'] = int(bool(
            job_data.get('poster_active', 0)
        ))
        
        return verification
    
    def calculate_verification_score(self, job_data: Dict[str, Any]) -> int:
        """
        Calculate total verification score (0-4).
        
        Args:
            job_data: Job posting data dict
            
        Returns:
            int: Sum of 4 verification fields (0-4)
        """
        verification = self.extract_verification_features(job_data)
        return sum(verification.values())
    
    def get_verification_status(self, score: int) -> Tuple[str, str, str]:
        """
        Get verification status label, color, and icon based on score.
        
        Args:
            score: Verification score (0-4)
            
        Returns:
            Tuple of (label, color, icon)
        """
        if score >= 3:
            return ("Highly Verified", "#4CAF50", "🏆")
        elif score >= 2:
            return ("Partially Verified", "#FF9800", "⭐")
        elif score >= 1:
            return ("Minimally Verified", "#2196F3", "📝")
        else:
            return ("Unverified", "#F44336", "❌")
    
    def get_verification_badges(self, job_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get verification badge information for UI display.
        
        Args:
            job_data: Job posting data dict
            
        Returns:
            List of badge info dicts with label, key, icon, description, verified
        """
        verification = self.extract_verification_features(job_data)
        
        badges = [
            {
                'label': 'VERIFIED',
                'key': 'poster_verified',
                'icon': '✓',
                'description': 'Account Verified',
                'verified': bool(verification['poster_verified'])
            },
            {
                'label': 'PHOTO',
                'key': 'poster_photo',
                'icon': '📸',
                'description': 'Profile Photo',
                'verified': bool(verification['poster_photo'])
            },
            {
                'label': 'EXPERIENCE',
                'key': 'poster_experience',
                'icon': '🎯',
                'description': 'Relevant Experience',
                'verified': bool(verification['poster_experience'])
            },
            {
                'label': 'ACTIVE',
                'key': 'poster_active',
                'icon': '🔥',
                'description': 'Recent Activity',
                'verified': bool(verification['poster_active'])
            }
        ]
        
        return badges
    
    def get_verification_breakdown(self, job_data: Dict[str, Any]) -> List[Tuple[str, int, str]]:
        """
        Get verification breakdown for detailed analysis display.
        
        Args:
            job_data: Job posting data dict
            
        Returns:
            List of (description, value, icon) tuples
        """
        verification = self.extract_verification_features(job_data)
        
        return [
            ("Account Verified", verification['poster_verified'], 
             "✅" if verification['poster_verified'] else "❌"),
            ("Has Profile Photo", verification['poster_photo'], 
             "📸" if verification['poster_photo'] else "👤"),
            ("Recent Activity", verification['poster_active'], 
             "⚡" if verification['poster_active'] else "💤"),
            ("Relevant Experience", verification['poster_experience'], 
             "🎯" if verification['poster_experience'] else "❓")
        ]
    
    def company_matches(self, company1: str, company2: str) -> bool:
        """
        Smart company name matching using fuzzy logic.
        
        Handles variations like:
        - "SmartChoice International UAE" vs "SmartChoice International Limited"
        - "Google LLC" vs "Google Inc."
        - "Microsoft Corporation" vs "Microsoft"
        
        Args:
            company1, company2: Company names to compare
            
        Returns:
            bool: True if companies likely match (85%+ similarity)
        """
        if not company1 or not company2:
            return False
        
        # Clean basic suffixes to normalize company names
        basic_suffixes = [
            ' inc', ' inc.', ' ltd', ' ltd.', ' limited', 
            ' llc', ' corp', ' corp.', ' corporation'
        ]
        
        clean1 = company1.lower().strip()
        clean2 = company2.lower().strip()
        
        # Remove basic suffixes
        for suffix in basic_suffixes:
            clean1 = clean1.replace(suffix, '')
            clean2 = clean2.replace(suffix, '')
        
        clean1 = clean1.strip()
        clean2 = clean2.strip()
        
        # Exact match after cleaning
        if clean1 == clean2:
            return True
        
        # Use fuzzy matching
        if fuzz is not None:
            # Use rapidfuzz for better fuzzy matching
            # token_sort_ratio ignores word order and handles partial matches
            score = fuzz.token_sort_ratio(clean1, clean2)
            
            # Log for debugging
            logger.debug(f"Company match: '{clean1}' vs '{clean2}' - Score: {score}")
            
            # 85+ score indicates likely same company
            return score >= 85
        else:
            # Fallback to difflib if rapidfuzz unavailable
            similarity = difflib.SequenceMatcher(None, clean1, clean2).ratio()
            return similarity > 0.80 and len(clean1) > 3
    
    def get_verification_summary(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get comprehensive verification summary for the job posting.
        
        Args:
            job_data: Job posting data dict
            
        Returns:
            Dict with verification summary information
        """
        verification = self.extract_verification_features(job_data)
        score = sum(verification.values())
        label, color, icon = self.get_verification_status(score)
        
        return {
            'score': score,
            'max_score': 4,
            'percentage': int((score / 4) * 100),
            'label': label,
            'color': color,
            'icon': icon,
            'features': verification,
            'is_highly_verified': score >= 3,
            'is_verified': score >= 2,
            'is_unverified': score == 0
        }
    
    def is_profile_private(self, job_data: Dict[str, Any]) -> bool:
        """
        Check if job poster profile is private.
        
        Args:
            job_data: Job posting data dict
            
        Returns:
            bool: True if profile is private
        """
        return bool(job_data.get('profile_private', 1))
    
    def get_poster_name(self, job_data: Dict[str, Any]) -> str:
        """
        Get job poster name from various possible fields.
        
        Args:
            job_data: Job posting data dict
            
        Returns:
            str: Poster name or 'Unknown'
        """
        return (job_data.get('job_poster_name') or 
                job_data.get('poster_name') or
                job_data.get('job_poster', {}).get('name') if 
                isinstance(job_data.get('job_poster'), dict) else 'Unknown')
    
    # === DATAFRAME SUPPORT METHODS ===
    
    def extract_verification_features_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract and validate verification features from DataFrame.
        
        This method ensures all verification columns exist and are properly formatted
        for batch processing (training, prediction, etc.).
        
        Args:
            df: Input DataFrame with potential verification columns
            
        Returns:
            pd.DataFrame: DataFrame with validated verification columns
        """
        logger.info("Extracting verification features from DataFrame")
        df_result = df.copy()
        
        # Define verification columns with defaults
        verification_columns = {
            'poster_verified': 0,
            'poster_photo': 0, 
            'poster_experience': 0,
            'poster_active': 0
        }
        
        # Ensure all verification columns exist
        for col, default_value in verification_columns.items():
            if col not in df_result.columns:
                df_result[col] = default_value
                logger.info(f"Added missing verification column '{col}' with default value {default_value}")
            else:
                # Ensure binary values (0 or 1)
                df_result[col] = df_result[col].apply(lambda x: 1 if x else 0)
        
        logger.info(f"Verification features extracted for {len(df_result)} records")
        return df_result
    
    def calculate_verification_scores_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate verification scores and categories for entire DataFrame.
        
        This is the SINGLE SOURCE OF TRUTH for verification score calculation
        used in training, batch predictions, and feature engineering.
        
        Args:
            df: DataFrame with verification columns
            
        Returns:
            pd.DataFrame: DataFrame with calculated verification scores and categories
        """
        logger.info("Calculating verification scores for DataFrame")
        df_result = df.copy()
        
        # First ensure verification features are properly extracted
        df_result = self.extract_verification_features_df(df_result)
        
        # Calculate poster_score (0-4 scale)
        df_result['poster_score'] = (
            df_result['poster_verified'].astype(int) +
            df_result['poster_photo'].astype(int) +
            df_result['poster_experience'].astype(int) +
            df_result['poster_active'].astype(int)
        )
        
        # Create verification categories (powerful predictors)
        df_result['is_highly_verified'] = (df_result['poster_score'] >= 3).astype(int)
        df_result['is_unverified'] = (df_result['poster_score'] == 0).astype(int)
        df_result['verification_ratio'] = df_result['poster_score'] / 4.0
        
        # Language-verification interactions (if language column exists)
        if 'language' in df_result.columns:
            df_result['is_arabic'] = (df_result['language'] == 1).astype(int)
            df_result['is_english'] = (df_result['language'] == 0).astype(int)
            df_result['arabic_unverified'] = (df_result['is_arabic'] & (df_result['poster_score'] <= 1)).astype(int)
            df_result['english_unverified'] = (df_result['is_english'] & (df_result['poster_score'] <= 1)).astype(int)
        
        # Log summary statistics
        if not df_result.empty:
            score_stats = df_result['poster_score'].describe()
            highly_verified_pct = (df_result['is_highly_verified'].sum() / len(df_result)) * 100
            unverified_pct = (df_result['is_unverified'].sum() / len(df_result)) * 100
            
            logger.info(f"Verification scores calculated for {len(df_result)} records:")
            logger.info(f"  Mean poster_score: {score_stats['mean']:.2f}")
            logger.info(f"  Highly verified (>=3): {highly_verified_pct:.1f}%")
            logger.info(f"  Unverified (=0): {unverified_pct:.1f}%")
        
        return df_result
    
    def get_risk_thresholds(self) -> Dict[str, float]:
        """
        Get fraud risk thresholds based on verification scores.
        
        These thresholds define fraud probability based on poster_score:
        - poster_score >= 3: Very low fraud risk (highly verified)
        - poster_score == 2: Low fraud risk (moderately verified)
        - poster_score == 1: High fraud risk (low verification)
        - poster_score == 0: Very high fraud risk (no verification)
        
        Returns:
            Dict: Risk thresholds for different verification levels
        """
        return {
            'very_low': 0.15,    # poster_score >= 3 (highly verified)
            'low': 0.30,         # poster_score == 2 (moderately verified) 
            'high': 0.75,        # poster_score == 1 (low verification)
            'very_high': 0.95    # poster_score == 0 (no verification)
        }
    
    def classify_risk_from_verification(self, poster_score: int) -> Tuple[str, bool, float]:
        """
        Classify fraud risk based on verification score.
        
        Args:
            poster_score: Verification score (0-4)
            
        Returns:
            Tuple of (risk_level, is_high_risk, fraud_probability)
        """
        thresholds = self.get_risk_thresholds()
        
        if poster_score >= 3:
            return ('very_low', False, thresholds['very_low'])
        elif poster_score == 2:
            return ('low', False, thresholds['low'])
        elif poster_score == 1:
            return ('high', True, thresholds['high'])
        else:  # poster_score == 0
            return ('very_high', True, thresholds['very_high'])
    
    def classify_risk_with_company_context(self, poster_score: int, job_data: Dict[str, Any]) -> Tuple[str, bool, float]:
        """
        Enhanced risk classification using BOTH poster and company verification.
        Uses enriched company data when available.
        
        Args:
            poster_score: Poster verification score (0-4)
            job_data: Job posting data dict with potential company metrics
            
        Returns:
            Tuple of (risk_level, is_high_risk, fraud_probability)
        """
        # Get company verification metrics from enriched data
        network_quality = job_data.get('network_quality_score', 0.0)
        profile_completeness = job_data.get('profile_completeness_score', 0.0)
        company_legitimacy = job_data.get('legitimacy_score', 0.5)  # Default to neutral if missing
        
        # Check if profile is private
        is_private = self.is_profile_private(job_data)
        
        # Calculate combined trust score
        if is_private and poster_score == 0:
            # For private profiles, weight company data more heavily
            company_weight = 0.8
            poster_weight = 0.2
        elif poster_score >= 2:
            # Good poster verification: balance both
            company_weight = 0.4
            poster_weight = 0.6
        else:
            # Low poster verification: increase company weight
            company_weight = 0.6
            poster_weight = 0.4
        
        # Company trust score (0-1, higher is better)
        company_trust = (network_quality * 0.3 + 
                        profile_completeness * 0.3 + 
                        company_legitimacy * 0.4)
        
        # Poster trust score (0-1, higher is better)
        poster_trust = poster_score / 4.0
        
        # Combined trust score
        combined_trust = (company_trust * company_weight + 
                         poster_trust * poster_weight)
        
        # Convert trust to risk (inverse)
        fraud_probability = 1.0 - combined_trust
        
        # Apply known company patterns (without hardcoding specific companies)
        company_name = str(job_data.get('company_name', '')).lower()
        
        # Pattern-based trusted company detection
        trusted_patterns = [
            len(company_name) > 40,  # Very long official names
            'international' in company_name or 'الدولية' in company_name,
            'group' in company_name or 'مجموعة' in company_name,
            'authority' in company_name or 'هيئة' in company_name,
        ]
        
        # If company shows trusted patterns AND has good legitimacy score
        if sum(trusted_patterns) >= 2 and company_legitimacy > 0.7:
            fraud_probability = min(fraud_probability * 0.5, 0.3)  # Significant reduction
        
        # Determine risk level based on adjusted probability
        if fraud_probability < 0.25:
            return ('very_low', False, fraud_probability)
        elif fraud_probability < 0.45:
            return ('low', False, fraud_probability)
        elif fraud_probability < 0.65:
            return ('moderate', False, fraud_probability)
        elif fraud_probability < 0.80:
            return ('high', True, fraud_probability)
        else:
            return ('very_high', True, fraud_probability)
    
    def validate_verification_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate verification data quality in DataFrame.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dict: Validation results with statistics and warnings
        """
        validation = {
            'is_valid': True,
            'warnings': [],
            'statistics': {},
            'recommendations': []
        }
        
        verification_columns = ['poster_verified', 'poster_photo', 'poster_experience', 'poster_active']
        
        # Check for missing columns
        missing_columns = [col for col in verification_columns if col not in df.columns]
        if missing_columns:
            validation['warnings'].append(f"Missing verification columns: {missing_columns}")
        
        # Check data quality for existing columns
        for col in verification_columns:
            if col in df.columns:
                # Check for non-binary values
                unique_values = df[col].dropna().unique()
                non_binary = [v for v in unique_values if v not in [0, 1, True, False]]
                if non_binary:
                    validation['warnings'].append(f"Non-binary values in {col}: {non_binary}")
                
                # Check for null values
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    validation['warnings'].append(f"Null values in {col}: {null_count}")
        
        # Calculate statistics
        if 'poster_score' in df.columns or all(col in df.columns for col in verification_columns):
            if 'poster_score' not in df.columns:
                df = self.calculate_verification_scores_df(df)
            
            validation['statistics'] = {
                'total_records': len(df),
                'highly_verified_count': (df['poster_score'] >= 3).sum(),
                'unverified_count': (df['poster_score'] == 0).sum(),
                'mean_score': df['poster_score'].mean(),
                'score_distribution': df['poster_score'].value_counts().to_dict()
            }
        
        # Generate recommendations
        if validation['statistics'].get('unverified_count', 0) > len(df) * 0.8:
            validation['recommendations'].append("High percentage of unverified records - check data extraction")
        
        if len(validation['warnings']) > 0:
            validation['is_valid'] = False
        
        return validation
    
    def calculate_company_trust(self, job_data: Dict[str, Any]) -> float:
        """
        Calculate company trust score based on company characteristics and patterns.
        
        This method analyzes company name patterns and characteristics to determine
        trustworthiness, used as a component in legitimacy score calculation.
        
        Args:
            job_data: Job posting data dict with company information
            
        Returns:
            float: Company trust score (0-1, higher is more trustworthy)
        """
        company_name = str(job_data.get('company_name', '')).lower().strip()
        
        if not company_name:
            return 0.0
        
        trust_score = 0.0
        trust_factors = 0
        
        # Base trust from name length (legitimate companies often have proper names)
        if len(company_name) > 2:
            if len(company_name) > 30:
                trust_score += 0.3  # Very long official names often legitimate
            elif len(company_name) > 10:
                trust_score += 0.2  # Reasonable length
            else:
                trust_score += 0.1  # Short but not too short
            trust_factors += 1
        
        # Trust patterns in company names
        trusted_patterns = [
            # International/global presence indicators
            ('international' in company_name or 'الدولية' in company_name, 0.25),
            ('global' in company_name or 'عالمية' in company_name, 0.25),
            
            # Corporate structure indicators
            ('group' in company_name or 'مجموعة' in company_name, 0.2),
            ('corporation' in company_name or 'corp' in company_name, 0.15),
            ('limited' in company_name or 'ltd' in company_name, 0.15),
            ('company' in company_name or 'شركة' in company_name, 0.1),
            
            # Authority/official indicators
            ('authority' in company_name or 'هيئة' in company_name, 0.3),
            ('ministry' in company_name or 'وزارة' in company_name, 0.3),
            ('government' in company_name or 'حكومة' in company_name, 0.25),
            
            # Established business indicators
            ('solutions' in company_name or 'حلول' in company_name, 0.15),
            ('services' in company_name or 'خدمات' in company_name, 0.1),
            ('consulting' in company_name or 'استشارات' in company_name, 0.15),
            
            # Technology/professional indicators  
            ('technology' in company_name or 'تقنية' in company_name, 0.1),
            ('systems' in company_name or 'أنظمة' in company_name, 0.1),
        ]
        
        pattern_score = 0.0
        pattern_matches = 0
        
        for pattern, score in trusted_patterns:
            if pattern:
                pattern_score += score
                pattern_matches += 1
        
        # Add pattern score if any matches found
        if pattern_matches > 0:
            # Cap pattern contribution and normalize
            pattern_contribution = min(pattern_score, 0.6)  # Max 60% from patterns
            trust_score += pattern_contribution
            trust_factors += 1
        
        # Suspicious pattern detection (reduces trust)
        suspicious_patterns = [
            'urgent' in company_name or 'عاجل' in company_name,
            'hiring' in company_name or 'توظيف' in company_name,
            'job' in company_name or 'وظيفة' in company_name,
            len([c for c in company_name if c.isdigit()]) > 3,  # Too many numbers
            company_name.count(' ') > 8,  # Overly long/complex name
        ]
        
        suspicious_count = sum(suspicious_patterns)
        if suspicious_count > 0:
            trust_score -= suspicious_count * 0.1  # Reduce trust for suspicious patterns
        
        # Calculate final trust score
        if trust_factors > 0:
            final_score = max(trust_score / trust_factors, 0.0)
        else:
            final_score = 0.1  # Minimal trust if no factors
        
        # Ensure score is in valid range [0, 1]
        return max(0.0, min(final_score, 1.0))
    
    def calculate_company_verification_scores(self, job_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate all company verification scores from company data.
        
        SINGLE SOURCE OF TRUTH for company scoring calculations used throughout
        the system (scraper, feature engine, fraud detector).
        
        Args:
            job_data: Job posting data dict with company information
            
        Returns:
            Dict with all company verification scores (0-1 range)
        """
        scores = {}
        
        # Company Followers Score (Network Quality)
        followers = job_data.get('company_followers', 0)
        if followers and followers > 0:
            if followers >= 10000:
                scores['company_followers_score'] = min(0.8 + (followers - 10000) / 50000 * 0.2, 1.0)
            elif followers >= 1000:
                scores['company_followers_score'] = 0.6 + (followers - 1000) / 9000 * 0.2
            elif followers >= 100:
                scores['company_followers_score'] = 0.4 + (followers - 100) / 900 * 0.2
            else:
                scores['company_followers_score'] = max(followers / 100 * 0.4, 0.1)
        else:
            scores['company_followers_score'] = 0.1  # Default minimal score
        
        # Company Employees Score (Size-based legitimacy)
        employees = job_data.get('company_employees', 0)
        if employees and employees > 0:
            if employees >= 1000:
                scores['company_employees_score'] = 0.9  # Large company
            elif employees >= 100:
                scores['company_employees_score'] = 0.7  # Medium-large company
            elif employees >= 20:
                scores['company_employees_score'] = 0.5  # Medium company
            elif employees >= 5:
                scores['company_employees_score'] = 0.3  # Small company
            else:
                scores['company_employees_score'] = 0.1  # Very small company
        else:
            scores['company_employees_score'] = 0.2  # Small company default
        
        # Company Founded Score (Age-based legitimacy)
        founded = job_data.get('company_founded')
        if founded and founded > 1900:
            try:
                import datetime
                current_year = datetime.datetime.now().year
                company_age = current_year - int(founded)
                
                if company_age >= 20:
                    scores['company_founded_score'] = 0.8  # Very established
                elif company_age >= 10:
                    scores['company_founded_score'] = 0.6  # Established
                elif company_age >= 5:
                    scores['company_founded_score'] = 0.4  # Medium age
                elif company_age >= 1:
                    scores['company_founded_score'] = 0.3  # New but legitimate
                else:
                    scores['company_founded_score'] = 0.2  # Very new
            except (ValueError, TypeError):
                scores['company_founded_score'] = 0.3  # Error handling
        else:
            scores['company_founded_score'] = 0.3  # Unknown age default
        
        # Network Quality Score - use enriched data if available, calculate if not
        if 'network_quality_score' in job_data and job_data['network_quality_score'] is not None:
            scores['network_quality_score'] = float(job_data['network_quality_score'])
        else:
            scores['network_quality_score'] = scores['company_followers_score']
        
        # Company Trust Score (using existing method)
        scores['company_trust_score'] = self.calculate_company_trust(job_data)
        
        # Company Legitimacy Score - use enriched data if available, calculate if not
        if 'legitimacy_score' in job_data and job_data['legitimacy_score'] is not None:
            scores['company_legitimacy_score'] = float(job_data['legitimacy_score'])
        else:
            # Calculate as weighted average including trust score
            scores['company_legitimacy_score'] = (
                scores['company_employees_score'] * 0.3 +      # Size weight
                scores['company_founded_score'] * 0.3 +        # Age weight  
                scores['company_followers_score'] * 0.2 +      # Network weight
                scores['company_trust_score'] * 0.2            # Trust patterns
            )
        
        # Ensure all scores are in [0, 1] range
        for key, value in scores.items():
            scores[key] = max(0.0, min(float(value), 1.0))
        
        return scores


# Export main class
__all__ = ['VerificationService']