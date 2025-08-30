"""
UI Helper Functions

This module contains utility functions for the Streamlit UI including:
- Trust score calculation
- Color scheme functions  
- Fraud pattern highlighting
- Text processing utilities
"""

import logging
import re
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


def calculate_company_trust_score(company_name: str) -> int:
    """Calculate trust score for company based on various factors."""
    if not company_name or company_name == 'N/A':
        return 0
    
    score = 50  # Base score
    
    # Well-known companies get higher trust
    trusted_companies = ['hungerstation', 'careem', 'uber', 'amazon', 'microsoft', 'google', 'apple']
    if any(trusted.lower() in company_name.lower() for trusted in trusted_companies):
        score += 40
    
    # Suspicious company indicators
    suspicious_names = ['جهة حكومية سرية', 'شركة المستقبل', 'شركة الثروة السريعة', 'مؤسسة النجاح الفوري']
    if any(sus.lower() in company_name.lower() for sus in suspicious_names):
        score -= 50
    
    # Company name quality
    if len(company_name) > 5 and company_name != 'N/A':
        score += 10
    
    return max(0, min(100, score))


def get_trust_color(score: int) -> str:
    """Get color based on trust score."""
    if score >= 80:
        return "#4CAF50"  # Green
    elif score >= 60:
        return "#FF9800"  # Orange  
    elif score >= 40:
        return "#FF5722"  # Red-orange
    else:
        return "#F44336"  # Red


def create_metric_card(title: str, value: any, color: str):
    """Create a professional metric card."""
    import streamlit as st
    st.markdown("""
    <div style="background: white; padding: 20px; border-radius: 10px; 
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); text-align: center;
                border-top: 4px solid {};">
        <h4 style="color: #666; margin: 0 0 10px 0; font-size: 14px;">{}</h4>
        <p style="color: #333; margin: 0; font-size: 24px; font-weight: bold;">{}</p>
    </div>
    """.format(color, title, value), unsafe_allow_html=True)


def highlight_fraud_indicators(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Highlight suspicious keywords and return both highlighted text and detected indicators.
    
    Returns:
        Tuple[str, List[Dict]]: (highlighted_text, detected_indicators)
    """
    if not text:
        return "", []
    
    # Define suspicious patterns
    suspicious_patterns = {
        'urgent_language': {
            'patterns': [
                r'\bURGENT\b', r'\bIMMEDIATE\b', r'\bASAP\b', r'\bFAST MONEY\b',
                r'\bQUICK CASH\b', r'\bEASY MONEY\b', r'\bNO EXPERIENCE\b'
            ],
            'color': '#ff4444',
            'weight': 0.3
        },
        'payment_red_flags': {
            'patterns': [
                r'\bUPFRONT\b', r'\bPAY FEE\b', r'\bREGISTRATION FEE\b',
                r'\bTRAINING FEE\b', r'\bDEPOSIT REQUIRED\b'
            ],
            'color': '#ff0000',
            'weight': 0.5
        },
        'vague_descriptions': {
            'patterns': [
                r'\bMAKE MONEY FROM HOME\b', r'\bWORK FROM ANYWHERE\b',
                r'\bFLEXIBLE SCHEDULE\b', r'\bUNLIMITED EARNING\b'
            ],
            'color': '#ff8800',
            'weight': 0.2
        },
        'grammar_errors': {
            'patterns': [
                r'[a-z]\.[A-Z]',  # Missing space after period
                r'\s{2,}',        # Multiple spaces
                r'[.]{2,}'        # Multiple periods
            ],
            'color': '#ffaa00',
            'weight': 0.1
        }
    }
    
    highlighted_text = text
    detected_indicators = []
    
    for category, info in suspicious_patterns.items():
        for pattern in info['patterns']:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                matched_text = match.group()
                
                # Add to detected indicators
                detected_indicators.append({
                    'type': category,
                    'text': matched_text,
                    'position': match.span(),
                    'severity': info['weight'],
                    'description': f"Suspicious {category.replace('_', ' ')}: '{matched_text}'"
                })
                
                # Highlight in text
                highlighted_text = highlighted_text.replace(
                    matched_text,
                    f'<span style="background-color: {info["color"]}; padding: 2px 4px; border-radius: 3px; color: white;">{matched_text}</span>',
                    1
                )
    
    return highlighted_text, detected_indicators


def extract_suspicious_patterns(text: str) -> Dict[str, Any]:
    """
    Extract and analyze suspicious patterns from job posting text.
    
    Args:
        text (str): Job posting text to analyze
        
    Returns:
        Dict[str, Any]: Analysis results with risk scores and indicators
    """
    if not text:
        return {'risk_score': 0.0, 'indicators': [], 'total_flags': 0}
    
    # Get highlighted text and indicators
    _, detected_indicators = highlight_fraud_indicators(text)
    
    # Calculate risk score based on indicators
    total_weight = sum(indicator['severity'] for indicator in detected_indicators)
    risk_score = min(1.0, total_weight)  # Cap at 1.0
    
    # Group indicators by type
    indicator_groups = {}
    for indicator in detected_indicators:
        category = indicator['type']
        if category not in indicator_groups:
            indicator_groups[category] = []
        indicator_groups[category].append(indicator)
    
    return {
        'risk_score': risk_score,
        'indicators': detected_indicators,
        'indicator_groups': indicator_groups,
        'total_flags': len(detected_indicators),
        'categories_affected': len(indicator_groups)
    }


def format_risk_level(risk_score: float) -> Tuple[str, str]:
    """
    Convert risk score to human-readable level and color.
    
    Args:
        risk_score (float): Risk score between 0.0 and 1.0
        
    Returns:
        Tuple[str, str]: (risk_level, color)
    """
    if risk_score >= 0.7:
        return "HIGH RISK", "#ff4444"
    elif risk_score >= 0.4:
        return "MEDIUM RISK", "#ff8800"
    elif risk_score >= 0.1:
        return "LOW RISK", "#ffaa00"
    else:
        return "MINIMAL RISK", "#44ff44"


def clean_text_for_analysis(text: str) -> str:
    """
    Clean and normalize text for fraud analysis.
    
    Args:
        text (str): Raw text to clean
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove special characters that might interfere with analysis
    text = re.sub(r'[^\w\s\.,!?;:()\[\]\'\"@#$%&*+\-=/<>]', '', text)
    
    # Normalize common abbreviations
    text = re.sub(r'\bu\s*r\s*g\s*e\s*n\s*t\b', 'URGENT', text, flags=re.IGNORECASE)
    text = re.sub(r'\ba\s*s\s*a\s*p\b', 'ASAP', text, flags=re.IGNORECASE)
    
    return text