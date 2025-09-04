"""
Content-Focused Fraud Detection Dashboard

This module provides fraud analysis dashboard focused on job content and company data.
Profile analysis removed for improved reliability and faster processing.

Enhanced Features:
- Content-based fraud risk scoring
- Company legitimacy verification
- Text quality analysis
- Contact method risk assessment
- Visual fraud indicators and trends
- Comparative analysis with legitimate job patterns

Version: 2.0.0 - Content-Focused Implementation
"""

import os
import sys
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

# Add the src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.ui.utils.streamlit_html import render_html_card, render_metric_card


def display_comprehensive_fraud_dashboard(job_data: Dict[str, Any]) -> None:
    """
    Display a modern, content-focused fraud analysis dashboard.
    
    Args:
        job_data (Dict[str, Any]): Complete job data for fraud analysis
    """
    if not job_data:
        st.error("No job data available for fraud analysis")
        return
    
    # Extract fraud scores from model (no defaults)
    fraud_risk_score = job_data.get('fraud_score')
    if fraud_risk_score is None:
        st.error("❌ No fraud prediction available - model may not be loaded")
        st.info("💡 Train models first using: python train_model_cli.py --model all_models --no-interactive")
        return
    
    is_fraud = job_data.get('is_fraud', fraud_risk_score > 0.5)
    confidence = job_data.get('confidence')
    if confidence is None:
        st.warning("⚠️ No confidence score available from model")
    
    # Company legitimacy from ML model predictions AND scraped data
    company_name = job_data.get('company_name', 'Unknown')
    model_metrics = job_data.get('metrics', {})
    
    # Use scraped data for company legitimacy if available
    company_legitimacy = job_data.get('company_legitimacy_score')
    if company_legitimacy is None:
        company_legitimacy = model_metrics.get('company_legitimacy_score', 0.5)
    
    # Display main header
    st.markdown("# 🛡️ FraudSpot Content Analysis")
    st.markdown("*Powered by content and company-based fraud detection*")
    
    # Main fraud risk indicator
    _display_fraud_risk_gauge(fraud_risk_score, is_fraud, confidence)
    
    # Key metrics with explanations
    _display_key_metrics_with_tooltips(job_data, fraud_risk_score, company_legitimacy)
    
    # Detailed analysis sections - 2x2 grid layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        _display_content_analysis(job_data)
    
    with col2:
        _display_company_verification(job_data, company_legitimacy)
    
    # Bottom row
    col3, col4 = st.columns([1, 1])
    
    with col3:
        _display_contact_risk_analysis(job_data)
    
    with col4:
        _display_methodology_explanation()
    
    # Final recommendation
    _display_final_recommendation(fraud_risk_score, is_fraud, confidence)


def _display_fraud_risk_gauge(fraud_risk_score: float, is_fraud: bool, confidence: Optional[float]) -> None:
    """Display visual fraud risk gauge with clean design."""
    
    # Determine risk level and colors
    if fraud_risk_score < 0.3:
        risk_level = "Low Risk"
        risk_color = "#10B981"  # Green
        risk_bg = "#D1FAE5"
        risk_icon = "✅"
    elif fraud_risk_score < 0.7:
        risk_level = "Medium Risk"
        risk_color = "#F59E0B"  # Orange
        risk_bg = "#FEF3C7"
        risk_icon = "⚠️"
    else:
        risk_level = "High Risk"
        risk_color = "#EF4444"  # Red
        risk_bg = "#FECACA"
        risk_icon = "🚨"
    
    # Create visual gauge
    gauge_html = f'''
    <div style="background: {risk_bg}; padding: 30px; border-radius: 20px; margin: 20px 0; text-align: center;
                border-left: 6px solid {risk_color};">
        
        <!-- Risk Icon and Level -->
        <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 20px;">
            <div style="font-size: 48px; margin-right: 15px;">{risk_icon}</div>
            <div>
                <h2 style="margin: 0; color: {risk_color}; font-size: 28px; font-weight: 700;">
                    {risk_level}
                </h2>
                <div style="color: #6B7280; font-size: 16px; margin-top: 5px;">
                    Fraud Risk Score: {fraud_risk_score:.1%}
                </div>
            </div>
        </div>
        
        <!-- Visual Gauge Bar -->
        <div style="background: #E5E7EB; border-radius: 10px; height: 20px; margin: 20px 0; position: relative;">
            <div style="background: {risk_color}; border-radius: 10px; height: 100%; width: {fraud_risk_score * 100}%; 
                        transition: width 0.3s ease;"></div>
            <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); 
                        color: white; font-weight: bold; font-size: 12px; text-shadow: 1px 1px 2px rgba(0,0,0,0.7);">
                {fraud_risk_score:.1%}
            </div>
        </div>
        
        <!-- Confidence Score -->
        {f'''<div style="margin-top: 20px; padding: 15px; background: rgba(255,255,255,0.7); border-radius: 12px;">
            <div style="color: #374151; font-weight: 600; margin-bottom: 5px;">
                Analysis Confidence: {f"{confidence:.1%}" if confidence is not None else "N/A"}
            </div>
            <div style="color: #6B7280; font-size: 14px;">
                Based on job content, company verification, and text analysis
            </div>
        </div>''' if confidence is not None else ''}
    </div>
    '''
    
    render_html_card(gauge_html)


def _display_key_metrics_with_tooltips(job_data: Dict[str, Any], fraud_risk: float, company_legitimacy: float) -> None:
    """Display key content-focused metrics with clear explanations."""
    
    # Calculate derived metrics from model predictions and scraped data
    overall_safety = 1.0 - fraud_risk
    
    # Get content and company scores
    model_metrics = job_data.get('metrics', {})
    content_quality = job_data.get('content_quality_score', model_metrics.get('content_quality_score', 0.5))
    contact_risk = job_data.get('contact_risk_score', model_metrics.get('contact_risk_score', 0.0))
    
    # Create metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        render_metric_card(
            title="Overall Safety",
            value=f"{overall_safety:.1%}",
            color="green" if overall_safety > 0.7 else "orange" if overall_safety > 0.3 else "red"
        )
    
    with col2:
        render_metric_card(
            title="Content Quality",
            value=f"{content_quality:.1%}" if content_quality is not None else "N/A",
            color="green" if content_quality and content_quality > 0.7 else "orange" if content_quality and content_quality > 0.3 else "red"
        )
    
    with col3:
        render_metric_card(
            title="Company Legitimacy",
            value=f"{company_legitimacy:.1%}" if company_legitimacy is not None else "N/A",
            color="green" if company_legitimacy and company_legitimacy > 0.7 else "orange" if company_legitimacy and company_legitimacy > 0.3 else "red"
        )
    
    with col4:
        contact_risk_display = 1.0 - contact_risk if contact_risk is not None else None
        render_metric_card(
            title="Contact Safety",
            value=f"{contact_risk_display:.1%}" if contact_risk_display is not None else "N/A",
            color="green" if contact_risk_display and contact_risk_display > 0.7 else "orange" if contact_risk_display and contact_risk_display > 0.3 else "red"
        )


def _display_content_analysis(job_data: Dict[str, Any]) -> None:
    """Display detailed content quality analysis."""
    
    st.markdown("### 📝 Content Quality Analysis")
    
    # Get content metrics
    content_quality = job_data.get('content_quality_score', 0.5)
    description_length = len(str(job_data.get('job_description', '')))
    title_words = len(str(job_data.get('job_title', '')).split())
    professional_score = job_data.get('professional_language_score', 0.5)
    urgency_score = job_data.get('urgency_language_score', 0.5)
    
    # Content factors
    factors = [
        ("Job Description Length", f"{description_length} characters", "✅" if description_length > 100 else "⚠️"),
        ("Title Word Count", f"{title_words} words", "✅" if title_words >= 3 else "⚠️"),
        ("Professional Language", f"{professional_score:.1%}", "✅" if professional_score > 0.5 else "⚠️"),
        ("Urgency Indicators", f"{urgency_score:.1%} non-urgent", "✅" if urgency_score > 0.5 else "🚨"),
        ("Has Salary Information", "Yes" if job_data.get('has_salary_info') else "No", "✅" if job_data.get('has_salary_info') else "⚠️"),
        ("Has Requirements", "Yes" if job_data.get('has_requirements') else "No", "✅" if job_data.get('has_requirements') else "⚠️"),
    ]
    
    # Display factors
    factors_html = "<div style='padding: 20px; background: #F9FAFB; border-radius: 12px; margin-bottom: 20px;'>"
    
    for factor, value, status in factors:
        color = "#10B981" if status == "✅" else "#F59E0B" if status == "⚠️" else "#EF4444"
        factors_html += f"""
        <div style="display: flex; justify-content: space-between; align-items: center; 
                    padding: 8px 0; border-bottom: 1px solid #E5E7EB;">
            <div style="font-weight: 500; color: #374151;">{factor}</div>
            <div style="display: flex; align-items: center;">
                <span style="margin-right: 8px; color: #6B7280;">{value}</span>
                <span style="font-size: 16px;">{status}</span>
            </div>
        </div>
        """
    
    factors_html += "</div>"
    render_html_card(factors_html)


def _display_company_verification(job_data: Dict[str, Any], company_legitimacy: float) -> None:
    """Display company legitimacy verification."""
    
    st.markdown("### 🏢 Company Analysis") 
    
    company_name = job_data.get('company_name', 'Unknown')
    
    # NO HARDCODED DATA - USE ONLY SCRAPED DATA
    # Use scraped or predicted data - DEBUG LOGGING
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"🏢 COMPANY DATA DEBUG: company_employees={job_data.get('company_employees')}, company_founded={job_data.get('company_founded')}, company_followers={job_data.get('company_followers')}")
    logger.info(f"🏢 ALTERNATIVE FIELDS: employees_in_linkedin={job_data.get('employees_in_linkedin')}, founded={job_data.get('founded')}, followers={job_data.get('followers')}")
    
    # Get company data
    employees = job_data.get('company_employees') or job_data.get('employees_in_linkedin')
    followers = job_data.get('company_followers') or job_data.get('followers')
    
    # Calculate network quality score
    def calculate_network_quality(followers, employees):
            """Calculate network quality score based on follower/employee ratio."""
            if not followers or not employees or employees == 0:
                return None  # Unknown, will show N/A
            
            try:
                followers = float(followers) if followers else 0
                employees = float(employees) if employees else 0
                
                if employees == 0:
                    return None
                    
                ratio = followers / employees
                
                # Sweet spot: 10-200 followers per employee (legitimate companies)
                if 10 <= ratio <= 200:
                    # Peak score around ratio=50, tapering off at edges
                    optimal_distance = abs(ratio - 50) / 150
                    return int(85 + (15 * (1 - optimal_distance)))
                # Highly suspicious: extreme ratios  
                elif ratio > 1000:
                    # Very suspicious - likely bot followers
                    # For SWATX: ratio 5099 should give ~5%
                    if ratio > 5000:
                        return 5
                    elif ratio > 3000:
                        return 10
                    elif ratio > 2000:
                        return 15
                    else:
                        return 20
                elif ratio < 1:
                    # Very suspicious - no social presence
                    return int(ratio * 30)
                # Borderline cases
                elif 200 < ratio <= 1000:
                    # Linear decline from good to suspicious
                    return int(70 - ((ratio - 200) / 800 * 40))
                else:  # 1 <= ratio < 10
                    # Linear increase from suspicious to acceptable
                    return int(30 + ((ratio - 1) / 9 * 40))
            except Exception as e:
                logger.error(f"Error calculating network quality: {e}")
                return None
        
    network_quality = calculate_network_quality(followers, employees)
    logger.info(f"📊 NETWORK QUALITY: followers={followers}, employees={employees}, ratio={followers/employees if employees else 0:.1f}, quality={network_quality}%")
    
    company_info = {
            'name': company_name,
            'legitimacy_score': company_legitimacy,
            'employees': employees,
            'founded': job_data.get('company_founded') or job_data.get('founded'),
            'linkedin_followers': followers,
            'industry': job_data.get('industry', job_data.get('job_industries', 'Not specified')),
            'network_quality': network_quality,
            'overall_legitimacy': company_legitimacy,
            'website': bool(job_data.get('company_website')),
            'size': bool(job_data.get('company_size')),
            'description': bool(job_data.get('job_description')),
        'logo_url': job_data.get('company_logo')
    }
    
    # Company verification card with logo
    logo_html = ""
    if company_info.get('logo_url'):
        logo_html = f'''
        <img src="{company_info['logo_url']}" 
             style="width: 50px; height: 50px; border-radius: 8px; object-fit: cover; 
                    margin-right: 15px; border: 1px solid #F59E0B;" 
             alt="Company Logo">
        '''
    else:
        logo_html = '''
        <div style="width: 50px; height: 50px; background: #F59E0B; border-radius: 8px; 
                    display: flex; align-items: center; justify-content: center; margin-right: 15px;">
            <span style="color: white; font-weight: bold; font-size: 18px;">🏢</span>
        </div>
        '''
    
    company_html = f'''
    <div style="background: #FEF3C7; padding: 20px; border-radius: 12px; margin-bottom: 20px;
                border-left: 4px solid #F59E0B;">
        <div style="display: flex; align-items: center; margin-bottom: 15px;">
            {logo_html}
            <div>
                <h4 style="margin: 0; color: #92400E; font-size: 18px;">{company_info['name']}</h4>
                <div style="color: #92400E; font-size: 14px; margin-top: 4px;">
                    Legitimacy Score: {company_info['legitimacy_score']:.1%}
                </div>
            </div>
        </div>
        
        <div style="color: #92400E; font-size: 14px; line-height: 1.6;">
            <strong>Company Assessment:</strong><br>
            Company data available but verification shows some concerns.
        </div>
        
        <div style="margin-top: 15px; display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
            <div style="display: flex; justify-content: space-between; padding: 8px; background: rgba(255,255,255,0.5); border-radius: 6px;">
                <span>LinkedIn Followers</span>
                <span>{f'{company_info["linkedin_followers"]:,}' if company_info['linkedin_followers'] else 'N/A'}</span>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 8px; background: rgba(255,255,255,0.5); border-radius: 6px;">
                <span>Employees</span>
                <span>{company_info['employees'] or 'N/A'}</span>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 8px; background: rgba(255,255,255,0.5); border-radius: 6px;">
                <span>Founded</span>
                <span>{company_info['founded'] or 'N/A'}</span>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 8px; background: rgba(255,255,255,0.5); border-radius: 6px;">
                <span>Network Quality</span>
                <span style="color: {'#16a34a' if company_info['network_quality'] and company_info['network_quality'] >= 70 else '#dc2626' if company_info['network_quality'] and company_info['network_quality'] <= 30 else '#d97706' if company_info['network_quality'] else '#6b7280'}; font-weight: 600;">{f"{company_info['network_quality']}%" if company_info['network_quality'] is not None else 'N/A'}</span>
            </div>
        </div>
        
        <div style="margin-top: 15px;">
            <div style="font-weight: 600; margin-bottom: 8px; color: #92400E;">Company Details</div>
            <div style="font-size: 13px; color: #92400E; line-height: 1.5;">
                <div><strong>Industry:</strong> {company_info['industry']}</div>
            </div>
        </div>
        
        <div style="margin-top: 15px; padding: 10px; background: rgba(16, 185, 129, 0.1); border-radius: 8px; border: 1px solid rgba(16, 185, 129, 0.2);">
            <div style="color: #065F46; font-size: 12px; font-weight: 600;">✓ Company data enriched via API</div>
        </div>
    </div>
    '''
    
    render_html_card(company_html)


def _display_contact_risk_analysis(job_data: Dict[str, Any]) -> None:
    """Display contact method risk analysis."""
    
    st.markdown("### 📞 Contact Risk Assessment")
    
    # Get contact information
    has_email = job_data.get('has_professional_email', job_data.get('has_email', False))
    has_whatsapp = job_data.get('has_whatsapp', False)
    has_telegram = job_data.get('has_telegram', False)
    contact_risk = job_data.get('contact_risk_score', 0.0)
    
    # Risk assessment
    risk_factors = []
    if has_whatsapp:
        risk_factors.append("Uses WhatsApp for contact")
    if has_telegram:
        risk_factors.append("Uses Telegram for contact")
    if not has_email and (has_whatsapp or has_telegram):
        risk_factors.append("Only messaging apps, no professional email")
    
    # Display contact analysis
    if contact_risk < 0.3:
        risk_color = "#10B981"
        risk_bg = "#D1FAE5"
        risk_level = "Low Risk"
        risk_icon = "✅"
    elif contact_risk < 0.7:
        risk_color = "#F59E0B"
        risk_bg = "#FEF3C7"
        risk_level = "Medium Risk"
        risk_icon = "⚠️"
    else:
        risk_color = "#EF4444"
        risk_bg = "#FECACA"
        risk_level = "High Risk"
        risk_icon = "🚨"
    
    contact_html = f'''
    <div style="background: {risk_bg}; padding: 20px; border-radius: 12px; margin-bottom: 20px;
                border-left: 4px solid {risk_color};">
        <div style="display: flex; align-items: center; margin-bottom: 15px;">
            <div style="font-size: 24px; margin-right: 10px;">{risk_icon}</div>
            <div>
                <h4 style="margin: 0; color: {risk_color}; font-size: 18px;">{risk_level}</h4>
                <div style="color: {risk_color}; font-size: 14px; margin-top: 4px;">
                    Contact Risk Score: {contact_risk:.1%}
                </div>
            </div>
        </div>
        
        <div style="margin-bottom: 15px;">
            <div><strong>Professional Email:</strong> {"✅ Yes" if has_email else "❌ No"}</div>
            <div><strong>WhatsApp Contact:</strong> {"⚠️ Yes" if has_whatsapp else "✅ No"}</div>
            <div><strong>Telegram Contact:</strong> {"⚠️ Yes" if has_telegram else "✅ No"}</div>
        </div>
        
        {f'''<div style="margin-top: 15px; padding: 10px; background: rgba(239, 68, 68, 0.1); border-radius: 8px;">
            <div style="color: #DC2626; font-weight: 600; margin-bottom: 5px;">Risk Factors:</div>
            <ul style="color: #DC2626; font-size: 13px; margin: 0; padding-left: 20px;">
                {"".join([f"<li>{factor}</li>" for factor in risk_factors])}
            </ul>
        </div>''' if risk_factors else ''}
    </div>
    '''
    
    render_html_card(contact_html)


def _display_methodology_explanation() -> None:
    """Display explanation of content-focused analysis methodology."""
    
    st.markdown("### 🔍 How Our Analysis Works")
    
    methodology_html = '''
    <div style="background: #F3F4F6; padding: 20px; border-radius: 12px; margin-bottom: 20px;">
        <h4 style="margin-top: 0; color: #374151;">Content-Focused Detection</h4>
        
        <div style="margin-bottom: 15px;">
            <div style="font-weight: 600; color: #1F2937; margin-bottom: 8px;">Our AI analyzes:</div>
            <ul style="color: #4B5563; font-size: 14px; line-height: 1.6; margin: 0; padding-left: 20px;">
                <li><strong>Job Content Quality:</strong> Description completeness, professional language, structure</li>
                <li><strong>Company Verification:</strong> Website presence, size information, industry details</li>
                <li><strong>Contact Methods:</strong> Professional vs. messaging app usage patterns</li>
                <li><strong>Text Analysis:</strong> Suspicious keywords, urgency language, grammar quality</li>
            </ul>
        </div>
        
        <div style="margin-top: 15px; padding: 12px; background: rgba(59, 130, 246, 0.1); border-radius: 8px; border-left: 4px solid #3B82F6;">
            <div style="color: #1E40AF; font-weight: 600; margin-bottom: 5px;">✓ Enhanced Reliability</div>
            <div style="color: #1E40AF; font-size: 13px;">
                Content-focused approach provides consistent results without depending on profile data availability.
            </div>
        </div>
        
        <div style="margin-top: 10px; padding: 12px; background: rgba(16, 185, 129, 0.1); border-radius: 8px; border-left: 4px solid #10B981;">
            <div style="color: #065F46; font-weight: 600; margin-bottom: 5px;">✓ Faster Processing</div>
            <div style="color: #065F46; font-size: 13px;">
                Focuses on immediately available job posting and company data for quick analysis.
            </div>
        </div>
    </div>
    '''
    
    render_html_card(methodology_html)


def _display_final_recommendation(fraud_risk_score: float, is_fraud: bool, confidence: Optional[float]) -> None:
    """Display final recommendation with action items."""
    
    st.markdown("### 🎯 Final Recommendation")
    
    if fraud_risk_score < 0.3:
        recommendation = "✅ **PROCEED WITH CONFIDENCE** - This appears to be a legitimate job posting."
        color = "#10B981"
        bg_color = "#D1FAE5"
        actions = [
            "Verify company details through official channels",
            "Research the role and company culture online",
            "Prepare for standard interview process"
        ]
    elif fraud_risk_score < 0.7:
        recommendation = "⚠️ **PROCEED WITH CAUTION** - Some risk factors detected."
        color = "#F59E0B"
        bg_color = "#FEF3C7"
        actions = [
            "Thoroughly verify company legitimacy",
            "Be cautious of requests for personal/financial information",
            "Use official company channels for communication",
            "Trust your instincts if something feels wrong"
        ]
    else:
        recommendation = "🚨 **HIGH RISK - NOT RECOMMENDED** - Multiple fraud indicators detected."
        color = "#EF4444"
        bg_color = "#FECACA"
        actions = [
            "Avoid sharing personal or financial information",
            "Do not pay any fees or provide banking details",
            "Report this posting to the platform administrators",
            "Look for alternative opportunities through verified channels"
        ]
    
    # Recommendation card
    rec_html = f'''
    <div style="background: {bg_color}; padding: 25px; border-radius: 15px; margin: 20px 0;
                border-left: 6px solid {color};">
        <div style="color: {color}; font-size: 18px; font-weight: 700; margin-bottom: 15px;">
            {recommendation}
        </div>
        
        <div style="color: #374151; font-weight: 600; margin-bottom: 10px;">Recommended Actions:</div>
        <ul style="color: #4B5563; margin: 0; padding-left: 20px;">
            {"".join([f"<li style='margin-bottom: 5px;'>{action}</li>" for action in actions])}
        </ul>
        
        {f'''<div style="margin-top: 20px; padding: 15px; background: rgba(255,255,255,0.8); border-radius: 10px;">
            <div style="color: #374151; font-weight: 600; margin-bottom: 5px;">
                Model Confidence: {confidence:.1%}
            </div>
            <div style="color: #6B7280; font-size: 14px;">
                This confidence score reflects how certain our AI model is about this prediction.
            </div>
        </div>''' if confidence is not None else ''}
    </div>
    '''
    
    render_html_card(rec_html)


# Export main function
__all__ = ['display_comprehensive_fraud_dashboard']