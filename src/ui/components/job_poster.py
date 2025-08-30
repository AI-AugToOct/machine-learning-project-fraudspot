"""
Job Poster Display Component - Bright Data Enhanced

This module handles the comprehensive display of job poster information using
Bright Data's rich LinkedIn profile data for advanced fraud detection.

Enhanced Features:
- Complete poster profile analysis (50+ data points)
- Network analysis and verification scores
- Professional history and credibility assessment
- Advanced fraud detection indicators
- Multi-factor verification system
- Social proof and activity tracking
"""

import logging
import os
import sys
from typing import Any, Dict, List

import streamlit as st

logger = logging.getLogger(__name__)

# Add the src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.ui.utils.streamlit_html import render_html_card, render_info_card, render_verification_badge


def display_job_poster_details(profile_data: Dict[str, Any]) -> None:
    """
    Display ALL profile data dynamically based on what's available.
    Uses appropriate UI elements for different data types.
    
    Args:
        profile_data (Dict[str, Any]): Raw profile data from Bright Data
    """
    if not profile_data:
        st.warning("No profile data available")
        return
    
    # Check if profile has success field and is successful
    if not profile_data.get('success', True):
        _display_private_profile_message(profile_data)
        return
    
    st.markdown("### 👤 Profile Information")
    
    # Skip metadata fields
    skip_fields = {'success', 'scraped_at', 'scraping_method', 'data_source', 
                   'api_version', 'profile_url', 'url', 'input_url', 'timestamp'}
    
    # Group certain fields for better display
    primary_fields = ['name', 'position', 'location', 'about', 'current_company']
    network_fields = ['connections', 'followers', 'recommendations_count']
    
    # Display primary information first
    st.markdown("#### Basic Information")
    for field in primary_fields:
        if field in profile_data and profile_data[field]:
            display_field(field, profile_data[field])
    
    # Display network stats in a grid
    display_network_stats(profile_data, network_fields)
    
    # Display all other fields dynamically
    st.markdown("#### Additional Profile Data")
    for field, value in profile_data.items():
        if field not in skip_fields and field not in primary_fields and field not in network_fields:
            if value:  # Only display if has data
                display_field(field, value)


def display_field(field_name: str, value: Any) -> None:
    """Display a field with appropriate UI element based on data type."""
    
    # Format field name for display
    display_name = field_name.replace('_', ' ').replace('-', ' ').title()
    
    if isinstance(value, list):
        if len(value) > 0:
            st.markdown(f"**{display_name}** ({len(value)} items)")
            
            # Check first item to determine display format
            if isinstance(value[0], dict):
                # Display as expandable cards for complex objects
                for item in value[:10]:  # Limit to first 10
                    with st.expander(get_item_title(item)):
                        display_dict(item)
            else:
                # Display as tags for simple lists
                display_tags(value)
    
    elif isinstance(value, dict):
        st.markdown(f"**{display_name}**")
        with st.container():
            display_dict(value)
    
    elif isinstance(value, str):
        if len(value) > 200:
            # Long text - use expander
            with st.expander(f"{display_name}"):
                st.text(value)
        elif 'url' in field_name.lower() or value.startswith('http'):
            # URL - make clickable
            st.markdown(f"**{display_name}**: [Link]({value})")
        else:
            # Short text - inline
            st.markdown(f"**{display_name}**: {value}")
    
    elif isinstance(value, (int, float)):
        # Numbers - show with appropriate formatting
        if 'score' in field_name.lower():
            st.metric(display_name, f"{value:.2%}")
        else:
            st.metric(display_name, f"{value:,}")
    
    elif isinstance(value, bool):
        # Boolean - show as badge
        if value:
            st.success(f"✅ {display_name}")
        else:
            st.info(f"❌ {display_name}")


def display_dict(data: dict) -> None:
    """Display dictionary data in a clean format."""
    for key, val in data.items():
        if val:
            formatted_key = key.replace('_', ' ').title()
            if isinstance(val, (list, dict)):
                st.json(val)  # For complex nested data
            else:
                st.write(f"• **{formatted_key}**: {val}")


def display_tags(items: list) -> None:
    """Display list items as tags."""
    tags_html = '<div style="display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0;">'
    for item in items[:20]:  # Limit to 20 tags
        tags_html += f'''
        <span style="background: #e3f2fd; color: #1976d2; 
                     padding: 4px 12px; border-radius: 16px; 
                     font-size: 14px;">{str(item)}</span>
        '''
    tags_html += '</div>'
    st.markdown(tags_html, unsafe_allow_html=True)


def get_item_title(item: dict) -> str:
    """Get a title for a dictionary item."""
    # Try common title fields
    for field in ['title', 'name', 'company', 'degree', 'position']:
        if field in item and item[field]:
            return str(item[field])
    # Fallback to first non-empty string value
    for val in item.values():
        if isinstance(val, str) and val:
            return val[:50]
    return "Item"


def display_network_stats(profile_data: dict, network_fields: list) -> None:
    """Display network statistics in a grid format."""
    network_data = {}
    for field in network_fields:
        if field in profile_data and profile_data[field]:
            network_data[field] = profile_data[field]
    
    if network_data:
        st.markdown("#### Network Statistics")
        cols = st.columns(len(network_data))
        for i, (field, value) in enumerate(network_data.items()):
            with cols[i]:
                display_name = field.replace('_', ' ').title()
                st.metric(display_name, f"{value:,}")


def display_verification_badges(job_data: Dict[str, Any]) -> None:
    """
    Display the 4 verification badges in a row with support for new fields.
    Handles private profiles by showing appropriate messaging.
    
    Args:
        job_data (Dict[str, Any]): Job posting data containing verification features
    """
    # Check if profile is private
    profile_private = job_data.get('profile_private', 1)
    
    if profile_private:
        st.markdown("### 🔒 Verification Status")
        st.info("**Job poster profile is private** - verification badges not available. Our fraud detection uses alternative indicators for assessment.")
        return
    
    st.markdown("### ✅ Verification Badges")
    
    # Define the 4 verification types using new fields
    verification_types = [
        {
            'label': 'VERIFIED',
            'key': 'job_poster_is_verified',
            'icon': '✓',
            'description': 'Account Verified'
        },
        {
            'label': 'EXPERIENCE',
            'key': 'job_poster_experiences',
            'icon': '🎯',
            'description': 'Relevant Experience'
        },
        {
            'label': 'PHOTO',
            'key': 'job_poster_has_photo',
            'icon': '📸',
            'description': 'Profile Photo'
        },
        {
            'label': 'ACTIVE',
            'key': 'poster_active',  # Keep from model compatibility
            'icon': '🔥',
            'description': 'Recent Activity'
        }
    ]
    
    # Create 4 columns for badges
    cols = st.columns(4)
    
    for col, badge_info in zip(cols, verification_types):
        with col:
            # Get verification status using new fields directly
            if badge_info['key'] == 'job_poster_experiences':
                # For experiences, check if there are any
                experiences = job_data.get(badge_info['key'], [])
                is_verified = len(experiences) > 0
            else:
                is_verified = bool(job_data.get(badge_info['key'], 0))
            
            # Render the badge
            render_verification_badge(
                label=badge_info['label'],
                verified=is_verified,
                icon=badge_info['icon']
            )
            
            # Add description below badge
            st.markdown(
                f"<p style='text-align: center; font-size: 11px; margin: 5px 0; color: #666;'>"
                f"{badge_info['description']}</p>",
                unsafe_allow_html=True
            )


def render_poster_summary(job_data: Dict[str, Any]) -> None:
    """
    Render a compact summary of poster information for the main display.
    Handles private profiles gracefully.
    
    Args:
        job_data (Dict[str, Any]): Job posting data
    """
    if not job_data:
        return
    
    # Check if profile is private
    profile_private = job_data.get('profile_private', 1)
    company = job_data.get('company_name', 'Unknown Company')
    
    if profile_private:
        # Show private profile summary
        private_summary_html = f'''
        <div style="background: white; 
                    padding: 20px; border-radius: 12px; margin: 15px 0;
                    border: 1px solid #e5e7eb; box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
                    border-left: 4px solid #6c757d;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h4 style="margin: 0; color: #495057;">🔒 Private Profile</h4>
                    <p style="margin: 5px 0 0 0; color: #666; font-size: 14px;">Posted by {company}</p>
                </div>
                <div style="text-align: right;">
                    <div style="color: #6c757d; font-weight: bold; font-size: 16px;">
                        Private
                    </div>
                    <div style="color: #666; font-size: 12px;">Profile</div>
                </div>
            </div>
        </div>
        '''
        render_html_card(private_summary_html)
        return
    
    # Get basic info using new fields
    poster_name = job_data.get('job_poster_name', 'Unknown')
    poster_company = job_data.get('job_poster_current_company', company)
    
    # Calculate quick verification score using new fields
    verified_features = sum([
        job_data.get('job_poster_is_verified', 0),
        1 if len(job_data.get('job_poster_experiences', [])) > 0 else 0,
        job_data.get('job_poster_has_photo', 0),
        job_data.get('poster_active', 0)  # Keep from model compatibility
    ])
    verification_percentage = int((verified_features / 4) * 100)
    
    # Choose color based on verification score
    if verification_percentage >= 75:
        color = "#4CAF50"  # Green
    elif verification_percentage >= 50:
        color = "#FF9800"  # Orange
    else:
        color = "#F44336"  # Red
    
    summary_html = f'''
    <div style="background: white; 
                padding: 20px; border-radius: 12px; margin: 15px 0;
                border: 1px solid #e5e7eb; box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
                border-left: 4px solid {color};">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h4 style="margin: 0; color: #333;">👤 {poster_name}</h4>
                <p style="margin: 5px 0 0 0; color: #666; font-size: 14px;">at {company}</p>
            </div>
            <div style="text-align: right;">
                <div style="color: {color}; font-weight: bold; font-size: 16px;">
                    {verification_percentage}%
                </div>
                <div style="color: #666; font-size: 12px;">Verified</div>
            </div>
        </div>
    </div>
    '''
    
    render_html_card(summary_html)


def _display_experience_timeline(experiences: List[Dict[str, str]]) -> None:
    """Display detailed experience timeline with Bright Data enhancements."""
    if not experiences:
        return
    
    st.markdown("### 💼 Professional Experience Timeline")
    
    timeline_html = '''
    <div style="background: white; padding: 25px; border-radius: 15px; 
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin-bottom: 20px;">
        <h3 style="color: #333; margin: 0 0 20px 0;">📋 Career History</h3>
        <div style="position: relative; padding-left: 30px;">
    '''
    
    for i, exp in enumerate(experiences[:8]):  # Show up to 8 experiences
        title = exp.get('title', 'N/A')
        company = exp.get('company', 'N/A')
        duration = exp.get('duration', 'Duration not specified')
        description = exp.get('description', '')
        
        # Color coding based on position
        color = '#667eea' if i == 0 else '#9C27B0' if i == 1 else '#FF9800' if i < 4 else '#4CAF50'
        
        timeline_html += f'''
        <div style="position: relative; margin-bottom: 25px; padding: 20px; 
                    background: #f8f9fa; border-radius: 10px; border-left: 4px solid {color};">
            <div style="position: absolute; left: -37px; top: 20px; width: 12px; height: 12px; 
                        background: {color}; border-radius: 50%; border: 3px solid white; 
                        box-shadow: 0 2px 4px rgba(0,0,0,0.2);"></div>
            <div style="position: absolute; left: -32px; top: 32px; width: 2px; height: calc(100% + 10px); 
                        background: #e0e0e0;"></div>
            
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 10px;">
                <div>
                    <h4 style="margin: 0; color: #333; font-size: 18px; font-weight: 600;">{title}</h4>
                    <p style="margin: 5px 0; color: #667eea; font-weight: 500; font-size: 16px;">{company}</p>
                </div>
                <span style="background: {color}; color: white; padding: 4px 12px; border-radius: 20px; 
                            font-size: 11px; font-weight: bold; white-space: nowrap;">{duration}</span>
            </div>
            {'<p style="margin: 10px 0 0 0; color: #666; font-size: 14px; line-height: 1.5;">' + description + '</p>' if description else ''}
        </div>
        '''
    
    timeline_html += '''
        </div>
    </div>
    '''
    
    render_html_card(timeline_html)


def _display_skills_and_certifications(skills: List[str], certifications: List[Dict]) -> None:
    """Display skills and certifications with enhanced Bright Data presentation."""
    st.markdown("### 🎯 Skills & Certifications")
    
    skills_html = '''
    <div style="background: white; padding: 25px; border-radius: 15px; 
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin-bottom: 20px;">
    '''
    
    # Skills section
    if skills:
        skills_html += '''
        <div style="margin-bottom: 25px;">
            <h4 style="color: #333; margin: 0 0 15px 0;">🛠️ Professional Skills</h4>
            <div style="display: flex; flex-wrap: wrap; gap: 8px;">
        '''
        
        for skill in skills[:20]:  # Show top 20 skills
            skills_html += f'''
            <span style="background: #667eea; 
                         color: white; padding: 8px 16px; border-radius: 25px; 
                         font-size: 13px; font-weight: 500; white-space: nowrap;">
                {skill}
            </span>
            '''
        
        skills_html += '</div></div>'
    
    # Certifications section
    if certifications:
        skills_html += '''
        <div>
            <h4 style="color: #333; margin: 0 0 15px 0;">🏆 Certifications</h4>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px;">
        '''
        
        for cert in certifications[:6]:  # Show top 6 certifications
            cert_name = cert.get('name', 'Certification') if isinstance(cert, dict) else str(cert)
            cert_org = cert.get('organization', 'Professional Organization') if isinstance(cert, dict) else 'Professional Organization'
            
            skills_html += f'''
            <div style="background: #f8f9fa; padding: 15px; border-radius: 10px; 
                        border-left: 4px solid #4CAF50;">
                <div style="font-weight: 600; color: #333; margin-bottom: 5px;">{cert_name}</div>
                <div style="color: #666; font-size: 13px;">{cert_org}</div>
            </div>
            '''
        
        skills_html += '</div></div>'
    
    skills_html += '</div>'
    render_html_card(skills_html)


def _display_fraud_risk_analysis(fraud_risk: float, network_quality: float, 
                                profile_completeness: float, credibility_score: float, 
                                job_data: Dict[str, Any]) -> None:
    """Display comprehensive fraud risk analysis dashboard."""
    st.markdown("### 🔍 Fraud Risk Analysis Dashboard")
    
    # Overall risk assessment
    risk_level = "LOW" if fraud_risk < 0.3 else "MEDIUM" if fraud_risk < 0.7 else "HIGH"
    risk_color = "#4CAF50" if fraud_risk < 0.3 else "#FF9800" if fraud_risk < 0.7 else "#F44336"
    
    # Red flags from Bright Data
    red_flags = job_data.get('red_flags_detected', [])
    
    fraud_html = f'''
    <div style="background: white; padding: 25px; border-radius: 15px; 
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin-bottom: 20px;">
        
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 25px;">
            <h3 style="color: #333; margin: 0;">🛡️ Fraud Risk Assessment</h3>
            <div style="background: {risk_color}; color: white; padding: 12px 24px; 
                        border-radius: 25px; font-size: 16px; font-weight: bold;">
                RISK: {risk_level} ({fraud_risk:.1%})
            </div>
        </div>
        
        <!-- Risk Factors Grid -->
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 25px;">
            <div style="text-align: center; padding: 20px; background: #f8f9fa; border-radius: 12px;">
                <div style="font-size: 32px; margin-bottom: 10px;">🔍</div>
                <div style="font-size: 24px; font-weight: bold; color: {risk_color}; margin-bottom: 5px;">
                    {fraud_risk:.1%}
                </div>
                <div style="font-size: 12px; color: #666;">Fraud Risk</div>
            </div>
            <div style="text-align: center; padding: 20px; background: #f8f9fa; border-radius: 12px;">
                <div style="font-size: 32px; margin-bottom: 10px;">🌐</div>
                <div style="font-size: 24px; font-weight: bold; color: {'#4CAF50' if network_quality > 0.7 else '#FF9800' if network_quality > 0.4 else '#F44336'}; margin-bottom: 5px;">
                    {network_quality:.1%}
                </div>
                <div style="font-size: 12px; color: #666;">Network Quality</div>
            </div>
            <div style="text-align: center; padding: 20px; background: #f8f9fa; border-radius: 12px;">
                <div style="font-size: 32px; margin-bottom: 10px;">📋</div>
                <div style="font-size: 24px; font-weight: bold; color: {'#4CAF50' if profile_completeness > 0.7 else '#FF9800' if profile_completeness > 0.4 else '#F44336'}; margin-bottom: 5px;">
                    {profile_completeness:.1%}
                </div>
                <div style="font-size: 12px; color: #666;">Profile Complete</div>
            </div>
            <div style="text-align: center; padding: 20px; background: #f8f9fa; border-radius: 12px;">
                <div style="font-size: 32px; margin-bottom: 10px;">⭐</div>
                <div style="font-size: 24px; font-weight: bold; color: {'#4CAF50' if credibility_score > 0.7 else '#FF9800' if credibility_score > 0.4 else '#F44336'}; margin-bottom: 5px;">
                    {credibility_score:.1%}
                </div>
                <div style="font-size: 12px; color: #666;">Credibility</div>
            </div>
        </div>
        
        <!-- Advanced Scores -->
        <div style="margin-bottom: 25px;">
            <h4 style="color: #333; margin: 0 0 15px 0;">📊 Advanced Fraud Indicators</h4>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
                <div style="background: #e3f2fd; padding: 15px; border-radius: 10px; text-align: center;">
                    <div style="font-weight: bold; color: #1565C0; font-size: 18px;">
                        {job_data.get('salary_realism_score', 0.8):.1%}
                    </div>
                    <div style="font-size: 12px; color: #666; margin-top: 4px;">Salary Realism</div>
                </div>
                <div style="background: #e8f5e8; padding: 15px; border-radius: 10px; text-align: center;">
                    <div style="font-weight: bold; color: #2E7D32; font-size: 18px;">
                        {job_data.get('contact_professionalism_score', 0.9):.1%}
                    </div>
                    <div style="font-size: 12px; color: #666; margin-top: 4px;">Contact Quality</div>
                </div>
                <div style="background: #fff3e0; padding: 15px; border-radius: 10px; text-align: center;">
                    <div style="font-weight: bold; color: #E65100; font-size: 18px;">
                        {job_data.get('social_proof_score', 0.7):.1%}
                    </div>
                    <div style="font-size: 12px; color: #666; margin-top: 4px;">Social Proof</div>
                </div>
            </div>
        </div>
    '''
    
    # Red flags section
    if red_flags:
        fraud_html += '''
        <div style="background: #ffebee; padding: 20px; border-radius: 10px; border-left: 4px solid #f44336;">
            <h4 style="color: #c62828; margin: 0 0 15px 0;">🚨 Detected Red Flags</h4>
            <ul style="margin: 0; padding-left: 20px; color: #666;">
        '''
        
        for flag in red_flags[:5]:  # Show top 5 red flags
            fraud_html += f'<li style="margin-bottom: 8px;">{flag}</li>'
        
        fraud_html += '</ul></div>'
    else:
        fraud_html += '''
        <div style="background: #e8f5e8; padding: 20px; border-radius: 10px; border-left: 4px solid #4caf50;">
            <h4 style="color: #2e7d32; margin: 0 0 10px 0;">✅ No Major Red Flags Detected</h4>
            <p style="margin: 0; color: #666;">The job poster profile shows positive indicators for legitimacy.</p>
        </div>
        '''
    
    fraud_html += '</div>'
    render_html_card(fraud_html)


def _display_private_profile_message(job_data: Dict[str, Any]) -> None:
    """
    Display a user-friendly message when job poster profile is private.
    
    Args:
        job_data (Dict[str, Any]): Job posting data
    """
    st.markdown("### 🔒 Job Poster Profile Information")
    
    # Get basic company info for context
    company_name = job_data.get('company_name', 'Unknown Company')
    job_title = job_data.get('job_title', 'Position')
    
    private_profile_html = f'''
    <div style="background: white; 
                padding: 30px; border-radius: 12px; margin: 20px 0;
                border: 1px solid #e5e7eb; box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
                border-left: 4px solid #6c757d; text-align: center;">
        
        <div style="font-size: 64px; margin-bottom: 20px; opacity: 0.7;">🔒</div>
        
        <h3 style="color: #495057; margin: 0 0 15px 0;">
            Job Poster Profile is Private
        </h3>
        
        <p style="color: #6c757d; margin: 0 0 20px 0; font-size: 16px; line-height: 1.5;">
            The LinkedIn profile of the person who posted this job is private and couldn't be accessed. 
            This is common for recruiters and hiring managers who maintain private profiles.
        </p>
        
        <div style="background: white; padding: 20px; border-radius: 10px; 
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 20px 0;">
            <h4 style="color: #333; margin: 0 0 15px 0;">Available Information:</h4>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; text-align: left;">
                <div>
                    <strong style="color: #6c757d;">Company:</strong><br>
                    <span style="color: #333; font-size: 16px;">{company_name}</span>
                </div>
                <div>
                    <strong style="color: #6c757d;">Position:</strong><br>
                    <span style="color: #333; font-size: 16px;">{job_title}</span>
                </div>
            </div>
        </div>
        
        <div style="background: #e3f2fd; padding: 15px; border-radius: 10px; 
                    border-left: 3px solid #2196f3; text-align: left;">
            <h5 style="color: #1565c0; margin: 0 0 10px 0;">💡 Fraud Detection Note:</h5>
            <p style="color: #1976d2; margin: 0; font-size: 14px;">
                Our fraud detection system has been adjusted to work effectively even without job poster 
                profile information. We analyze job content, company legitimacy, and posting patterns 
                to provide accurate fraud assessment.
            </p>
        </div>
        
    </div>
    '''
    
    render_html_card(private_profile_html)


def _display_profile_loading_state(job_data: Dict[str, Any]) -> None:
    """
    Display loading state while fetching job poster profile.
    
    Args:
        job_data (Dict[str, Any]): Job posting data
    """
    st.markdown("### 🔄 Job Poster Profile Information")
    
    poster_name = job_data.get('job_poster_name', 'Job Poster')
    
    loading_html = f'''
    <div style="background: white; 
                padding: 30px; border-radius: 12px; margin: 20px 0;
                border: 1px solid #e5e7eb; box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
                border-left: 4px solid #ff9800; text-align: center;">
        
        <div style="font-size: 48px; margin-bottom: 15px;">
            <div class="loading-spinner">🔄</div>
        </div>
        
        <h3 style="color: #e65100; margin: 0 0 10px 0;">
            Loading {poster_name}'s Profile...
        </h3>
        
        <p style="color: #f57c00; margin: 0; font-size: 14px;">
            Fetching detailed profile information from LinkedIn via Bright Data
        </p>
        
        <div style="margin-top: 20px;">
            <div style="background: rgba(255,255,255,0.8); padding: 15px; border-radius: 10px;">
                <div style="display: flex; justify-content: center; gap: 10px; align-items: center;">
                    <div style="width: 8px; height: 8px; background: #ff9800; border-radius: 50%; 
                                animation: pulse 1.5s ease-in-out infinite;"></div>
                    <div style="width: 8px; height: 8px; background: #ff9800; border-radius: 50%; 
                                animation: pulse 1.5s ease-in-out 0.2s infinite;"></div>
                    <div style="width: 8px; height: 8px; background: #ff9800; border-radius: 50%; 
                                animation: pulse 1.5s ease-in-out 0.4s infinite;"></div>
                </div>
            </div>
        </div>
    </div>
    
    <style>
    @keyframes pulse {{
        0%, 80%, 100% {{
            opacity: 0.3;
            transform: scale(1);
        }}
        40% {{
            opacity: 1;
            transform: scale(1.2);
        }}
    }}
    .loading-spinner {{
        animation: spin 2s linear infinite;
    }}
    @keyframes spin {{
        0% {{ transform: rotate(0deg); }}
        100% {{ transform: rotate(360deg); }}
    }}
    </style>
    '''
    
    render_html_card(loading_html)