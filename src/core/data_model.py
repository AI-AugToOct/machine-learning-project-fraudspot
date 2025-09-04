"""
Content-Focused Data Model - Single Source of Truth

This module defines the content-focused data model for fraud detection.
All poster/profile features have been removed for reliability.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DataRange(Enum):
    """Defines the format range for data values."""
    RAW = "raw"           # Original values from source
    NORMALIZED = "norm"   # ML-ready normalized values
    DISPLAY = "display"   # UI-ready formatted values (e.g., "75%")


@dataclass
class JobPostingData:
    """
    Content-focused data model for job posting fraud detection.
    
    This class maintains data in multiple formats to serve different layers:
    - RAW: Original values from scraper
    - NORMALIZED: ML-ready normalized values
    - Methods provide appropriate format for each consumer
    
    All poster/profile features removed for reliability.
    """
    
    # Core fields (required)
    job_title: str
    company_name: str
    description: str
    location: str = ""
    
    # Job details
    employment_type: str = "Full-time"
    seniority_level: str = "Entry level"
    function: str = ""
    industries: str = ""
    
    # Content-focused features (no poster/profile data)
    content_quality_score: float = 0.0
    company_legitimacy_score: float = 0.0
    contact_risk_score: float = 0.0
    
    # Company metrics - nullable for ML-first approach
    company_followers: Optional[int] = None
    company_employees: Optional[int] = None
    company_founded: Optional[int] = None
    
    # Contact information
    company_website: Optional[str] = None
    company_linkedin: Optional[str] = None
    email_domains: List[str] = field(default_factory=list)
    
    # Timing data
    scraped_at: datetime = field(default_factory=datetime.now)
    posted_date: Optional[datetime] = None
    
    # ML features
    fraudulent: Optional[int] = None  # Target variable for training
    
    def __post_init__(self):
        """Post-initialization to ensure data consistency."""
        # Validate company metrics
        if self.company_followers and self.company_followers < 0:
            logger.warning(f"company_followers {self.company_followers} < 0, setting to 0")
            self.company_followers = 0
            
        if self.company_employees and self.company_employees < 0:
            logger.warning(f"company_employees {self.company_employees} < 0, setting to 0")
            self.company_employees = 0
    
    def get_company_followers(self, range: DataRange = DataRange.RAW) -> Any:
        """Get company followers in requested format."""
        if self.company_followers is None:
            return 0 if range != DataRange.DISPLAY else "Unknown"
        
        if range == DataRange.RAW:
            return self.company_followers
        elif range == DataRange.NORMALIZED:
            return min(self.company_followers / 10000, 1.0)
        else:  # DISPLAY
            if self.company_followers >= 1000000:
                return f"{self.company_followers/1000000:.1f}M"
            elif self.company_followers >= 1000:
                return f"{self.company_followers/1000:.1f}K"
            return str(self.company_followers)
    
    def get_company_employees(self, range: DataRange = DataRange.RAW) -> Any:
        """Get company employees in requested format."""
        if self.company_employees is None:
            return 0 if range != DataRange.DISPLAY else "Unknown"
        
        if range == DataRange.RAW:
            return self.company_employees
        elif range == DataRange.NORMALIZED:
            return min(self.company_employees / 1000, 1.0)
        else:  # DISPLAY
            if self.company_employees >= 1000:
                return f"{self.company_employees/1000:.1f}K"
            return str(self.company_employees)
    
    def get_company_founded(self, range: DataRange = DataRange.RAW) -> Any:
        """Get company founded year in requested format."""
        if self.company_founded is None:
            return 0 if range != DataRange.DISPLAY else "Unknown"
        
        if range == DataRange.RAW:
            return self.company_founded
        elif range == DataRange.NORMALIZED:
            current_year = datetime.now().year
            company_age = max(current_year - self.company_founded, 0)
            return min(company_age / 50, 1.0)  # Normalize to 50 years max
        else:  # DISPLAY
            return str(self.company_founded)
    
    def to_ml_features(self) -> Dict[str, Any]:
        """
        Convert to ML-ready feature dictionary.
        Used by feature engineering and model training.
        """
        return {
            # Basic job information
            'job_title': self.job_title,
            'company_name': self.company_name,
            'description': self.description,
            'location': self.location,
            'employment_type': self.employment_type,
            'seniority_level': self.seniority_level,
            'function': self.function,
            'industries': self.industries,
            
            # Content-focused scores
            'content_quality_score': self.content_quality_score,
            'company_legitimacy_score': self.company_legitimacy_score,
            'contact_risk_score': self.contact_risk_score,
            
            # Company metrics (preserve None for ML)
            'company_followers': self.company_followers,
            'company_employees': self.company_employees,
            'company_founded': self.company_founded,
            
            # Contact information
            'company_website': self.company_website,
            'company_linkedin': self.company_linkedin,
            'email_domains': self.email_domains,
            
            # Target variable
            'fraudulent': self.fraudulent
        }
    
    def to_display_dict(self) -> Dict[str, Any]:
        """
        Convert to UI-friendly display dictionary.
        Used by Streamlit interface.
        """
        return {
            'job_title': self.job_title,
            'company_name': self.company_name,
            'location': self.location,
            'employment_type': self.employment_type,
            
            'content_quality': f"{self.content_quality_score:.1%}",
            'company_legitimacy': f"{self.company_legitimacy_score:.1%}",
            'contact_risk': f"{self.contact_risk_score:.1%}",
            
            'company_followers': self.get_company_followers(DataRange.DISPLAY),
            'company_employees': self.get_company_employees(DataRange.DISPLAY),
            'company_founded': self.get_company_founded(DataRange.DISPLAY),
            
            'scraped_at': self.scraped_at.strftime('%Y-%m-%d %H:%M'),
            'posted_date': self.posted_date.strftime('%Y-%m-%d') if self.posted_date else 'Unknown'
        }
    
    def calculate_company_legitimacy_score(self) -> float:
        """Calculate company legitimacy score based on available company data."""
        score = 0.0
        factors = 0
        
        # Factor 1: Company age (20% weight)
        if self.company_founded:
            from datetime import datetime
            age_years = datetime.now().year - self.company_founded
            if age_years >= 10:
                score += 0.2  # Established company
            elif age_years >= 5:
                score += 0.15  # Mature company
            elif age_years >= 2:
                score += 0.1   # Young but established
            else:
                score += 0.05  # Very new
            factors += 1
        
        # Factor 2: Website presence (15% weight)  
        if self.company_website:
            score += 0.15
            factors += 1
        
        # Factor 3: Employee count legitimacy (25% weight)
        if self.company_employees:
            if self.company_employees >= 1000:
                score += 0.25  # Large company
            elif self.company_employees >= 100:
                score += 0.20  # Medium company
            elif self.company_employees >= 10:
                score += 0.15  # Small company
            else:
                score += 0.05  # Very small (potentially suspicious)
            factors += 1
        
        # Factor 4: Network legitimacy (40% weight)
        if self.company_followers and self.company_employees:
            ratio = self.company_followers / self.company_employees
            if ratio <= 100:  # Reasonable ratio
                score += 0.4
            elif ratio <= 500:  # Moderate ratio
                score += 0.3
            elif ratio <= 1000:  # High but acceptable
                score += 0.2
            elif ratio <= 5000:  # Very high (suspicious)
                score += 0.1
            else:  # Extreme ratio (very suspicious)
                score += 0.0
            factors += 1
        
        # Normalize by number of factors we could evaluate
        if factors > 0:
            return score / factors * 4  # Scale to 0-1 range
        else:
            return 0.0  # No data to evaluate
    
    def calculate_content_quality_score(self) -> float:
        """Calculate content quality score based on job posting content."""
        score = 0.0
        
        # Factor 1: Description length and completeness (30%)
        if self.description:
            desc_len = len(self.description)
            if desc_len >= 500:
                score += 0.3
            elif desc_len >= 200:
                score += 0.25
            elif desc_len >= 100:
                score += 0.15
            else:
                score += 0.05
        
        # Factor 2: Job title professionalism (20%)
        if self.job_title:
            title_lower = self.job_title.lower()
            # Professional titles get higher scores
            professional_terms = ['engineer', 'manager', 'analyst', 'specialist', 'director', 'coordinator', 'developer', 'consultant']
            spam_terms = ['urgent', '$', 'quick', 'easy', 'fast', '!!!', 'work from home', 'earn money']
            
            if any(term in title_lower for term in professional_terms):
                score += 0.2
            elif any(term in title_lower for term in spam_terms):
                score += 0.0  # Spam-like title
            else:
                score += 0.1  # Neutral title
        
        # Factor 3: Location specificity (15%)
        if self.location and self.location.lower() not in ['remote', 'work from home', 'anywhere']:
            score += 0.15
        elif self.location:
            score += 0.05  # Remote is okay but less specific
        
        # Factor 4: Employment type clarity (10%)
        if self.employment_type and self.employment_type in ['Full-time', 'Part-time', 'Contract', 'Internship']:
            score += 0.1
        
        # Factor 5: Function/industry specificity (10%)
        if self.function or self.industries:
            score += 0.1
        
        # Factor 6: Seniority level specified (15%)
        if self.seniority_level and self.seniority_level != 'Entry level':
            score += 0.15
        elif self.seniority_level:
            score += 0.1
        
        return min(score, 1.0)  # Cap at 1.0
    
    def calculate_contact_risk_score(self) -> float:
        """Calculate contact risk score based on contact methods mentioned."""
        risk_score = 0.0
        
        # Check description and title for risky contact methods
        full_text = f"{self.job_title} {self.description}".lower()
        
        # High risk indicators
        if 'whatsapp' in full_text:
            risk_score += 0.4
        if 'telegram' in full_text:
            risk_score += 0.3
        if any(term in full_text for term in ['contact us via', 'message us at', 'text us']):
            risk_score += 0.2
        
        # Suspicious email patterns
        if any(term in full_text for term in ['@gmail', '@yahoo', '@hotmail']):
            risk_score += 0.1
        
        return min(risk_score, 1.0)
    
    @classmethod
    def from_scraper_data(cls, raw_data: Dict[str, Any]) -> 'JobPostingData':
        """
        Create JobPostingData from scraper output.
        
        Args:
            raw_data: Dictionary from scraper (content-focused)
            
        Returns:
            JobPostingData instance with normalized values
        """
        logger.info("Creating content-focused JobPostingData from scraper data")
        
        # Extract company data with proper field mapping
        company_followers = raw_data.get('company_followers') or raw_data.get('followers')
        company_employees = raw_data.get('company_employees') or raw_data.get('employees')
        company_founded = raw_data.get('company_founded') or raw_data.get('founded')
        
        # Handle string values that should be integers
        if company_followers and isinstance(company_followers, str):
            try:
                company_followers = int(company_followers.replace(',', '').replace('+', ''))
            except ValueError:
                company_followers = None
        
        if company_employees and isinstance(company_employees, str):
            try:
                company_employees = int(company_employees.replace(',', '').replace('+', ''))
            except ValueError:
                company_employees = None
                
        if company_founded and isinstance(company_founded, str):
            try:
                company_founded = int(company_founded)
            except ValueError:
                company_founded = None
        
        # Extract email domains
        email_domains = raw_data.get('email_domains', [])
        if isinstance(email_domains, str):
            email_domains = [email_domains]
        
        # Create the object first with basic data
        instance = cls(
            # Core job information
            job_title=raw_data.get('job_title', ''),
            company_name=raw_data.get('company_name', ''),
            description=raw_data.get('description', '') or raw_data.get('job_description', ''),
            location=raw_data.get('location', ''),
            employment_type=raw_data.get('employment_type', 'Full-time'),
            seniority_level=raw_data.get('seniority_level', 'Entry level'),
            function=raw_data.get('function', ''),
            industries=raw_data.get('industries', ''),
            
            # Content-focused scores (will be calculated after)
            content_quality_score=0.0,
            company_legitimacy_score=0.0,
            contact_risk_score=0.0,
            
            # Company metrics (properly extracted from scraped data)
            company_followers=company_followers,
            company_employees=company_employees,
            company_founded=company_founded,
            
            # Contact information
            company_website=raw_data.get('company_website'),
            company_linkedin=raw_data.get('company_linkedin'),
            email_domains=email_domains,
            
            # Target variable for training
            fraudulent=raw_data.get('fraudulent')
        )
        
        # Calculate scores after object creation
        instance.content_quality_score = instance.calculate_content_quality_score()
        instance.company_legitimacy_score = instance.calculate_company_legitimacy_score()
        instance.contact_risk_score = instance.calculate_contact_risk_score()
        
        return instance
    
    def has_enriched_data(self) -> bool:
        """Check if we have enriched company data."""
        return any([
            self.company_followers is not None,
            self.company_employees is not None,
            self.company_founded is not None,
            self.company_website is not None
        ])


@dataclass
class FraudResult:
    """Result of fraud detection analysis."""
    
    is_fraudulent: bool
    confidence_score: float
    risk_level: str  # 'low', 'medium', 'high'
    risk_factors: List[str]
    protective_factors: List[str]
    model_predictions: Dict[str, float]
    feature_importance: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)
    fraud_probability: float = 0.0
    explanation: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'is_fraudulent': self.is_fraudulent,
            'confidence_score': self.confidence_score,
            'risk_level': self.risk_level,
            'risk_factors': self.risk_factors,
            'protective_factors': self.protective_factors,
            'model_predictions': self.model_predictions,
            'feature_importance': self.feature_importance,
            'timestamp': self.timestamp.isoformat(),
            'fraud_probability': self.fraud_probability,
            'explanation': self.explanation,
            'metrics': self.metrics
        }
    
    def to_ui_dict(self) -> Dict[str, Any]:
        """Convert to UI-compatible dictionary with expected field names."""
        return {
            'fraud_score': self.fraud_probability,  # UI expects 'fraud_score'
            'fraud_probability': self.fraud_probability,
            'is_fraudulent': self.is_fraudulent,
            'confidence': self.confidence_score,
            'confidence_score': self.confidence_score,
            'risk_level': self.risk_level,
            'risk_factors': self.risk_factors,
            'protective_factors': self.protective_factors,
            'model_predictions': self.model_predictions,
            'feature_importance': self.feature_importance,
            'explanation': self.explanation,
            'metrics': self.metrics,
            'timestamp': self.timestamp.isoformat()
        }


# Export main classes
__all__ = ['JobPostingData', 'FraudResult', 'DataRange']