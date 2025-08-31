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


def display_fraud_focused_profile(profile_data: Dict[str, Any], job_data: Dict[str, Any] = None) -> None:
    """
    Display profile as a unified access card/badge design like a professional ID card.
    
    Args:
        profile_data (Dict[str, Any]): Raw profile data from Bright Data
        job_data (Dict[str, Any]): Job posting data for comparison
    """
    if not profile_data:
        st.warning("No profile data available")
        return
    
    # Check if profile has success field and is successful
    if not profile_data.get('success', True):
        st.info("🔒 Profile is private - fraud analysis uses alternative indicators")
        return
    
    # Extract fraud-relevant data (adapted for Bright Data response structure)
    name = profile_data.get('name', 'Unknown')
    position = profile_data.get('position', 'Not specified')  
    current_company = profile_data.get('current_company', 'Not specified')
    
    # Handle location from city and country_code
    city = profile_data.get('city', '')
    country_code = profile_data.get('country_code', '')
    if city and country_code:
        location = f"{city}, {country_code}"
    elif city:
        location = city
    elif country_code:
        location = country_code
    else:
        location = profile_data.get('location', 'Not specified')
    
    about = profile_data.get('about', '')
    
    # Profile photo - use the correct 'avatar' field from Bright Data
    profile_photo_url = profile_data.get('avatar')
    
    # Activity indicators - use the 'activity' array from Bright Data
    activity_data = profile_data.get('activity', [])
    activity_count = len(activity_data) if isinstance(activity_data, list) else 0
    
    # Trust score calculation - since no is_verified field, use trust indicators
    recommendations_count = profile_data.get('recommendations_count', 0)
    connections = profile_data.get('connections', 0)  
    followers = profile_data.get('followers', 0)
    honors_and_awards = profile_data.get('honors_and_awards', [])
    has_awards = len(honors_and_awards) > 0 if isinstance(honors_and_awards, list) else False
    
    # Calculate trust score (0-5 points)
    trust_score = 0
    if recommendations_count > 0: trust_score += 1  # Has recommendations
    if connections >= 500: trust_score += 1         # Well connected (500+)
    if followers > 100: trust_score += 1            # Has followers
    if has_awards: trust_score += 1                 # Has awards/honors
    if profile_photo_url: trust_score += 1          # Has profile photo
    
    # Determine trust level and colors
    if trust_score >= 4:
        trust_level = "TRUSTED"
        trust_color = "#4CAF50"
        trust_icon = "🏆"
        badge_border_color = "#4CAF50"
    elif trust_score >= 2:
        trust_level = "ESTABLISHED"
        trust_color = "#FF9800"
        trust_icon = "⭐"
        badge_border_color = "#FF9800"
    else:
        trust_level = "NEW PROFILE"
        trust_color = "#9E9E9E"
        trust_icon = "👤"
        badge_border_color = "#9E9E9E"
    
    # Company match logic
    job_company = job_data.get('company_name', '') if job_data else ''
    poster_current_company = profile_data.get('current_company', {}).get('name', '') if isinstance(profile_data.get('current_company'), dict) else str(current_company) if current_company != 'Not specified' else ''
    
    # Smart company matching
    company_match = False
    if poster_current_company and job_company:
        poster_company_clean = poster_current_company.lower().replace(' limited', '').replace(' ltd', '').replace(' llc', '').replace(' inc', '')
        job_company_clean = job_company.lower().replace(' limited', '').replace(' ltd', '').replace(' llc', '').replace(' inc', '')
        
        if any(word in poster_company_clean and word in job_company_clean 
               for word in ['smartchoice', 'google', 'microsoft', 'apple', 'amazon'] 
               if len(word) > 3) or poster_company_clean == job_company_clean:
            company_match = True
    
    # Activity level
    if activity_count > 10:
        activity_status = "VERY ACTIVE"
        activity_color = "#4CAF50"
        activity_icon = "🔥"
    elif activity_count > 3:
        activity_status = "ACTIVE"
        activity_color = "#FF9800"
        activity_icon = "📊"
    elif activity_count > 0:
        activity_status = "SOME ACTIVITY"
        activity_color = "#2196F3"
        activity_icon = "📈"
    else:
        activity_status = "NO ACTIVITY"
        activity_color = "#f44336"
        activity_icon = "😴"
    
    # Create unified access card
    st.markdown("### 👤 Job Poster Profile")
    
    # Main unified access card
    # Prepare template variables
    profile_photo_html = f'<img src="{profile_photo_url}" style="width: 100%; height: 100%; object-fit: cover;" alt="Profile Photo">' if profile_photo_url else '<span style="font-size: 48px; color: #999;">👤</span>'
    photo_color = '#4CAF50' if profile_photo_url else '#f44336'
    photo_status = '✅ PHOTO' if profile_photo_url else '❌ NO PHOTO'
    company_bg = 'rgba(76, 175, 80, 0.1)' if company_match else 'rgba(255, 152, 0, 0.1)'
    company_border = '#4CAF50' if company_match else '#FF9800'
    company_icon = '🏢' if company_match else '🏭'
    company_status = 'SAME COMPANY' if company_match else 'EXTERNAL POSTER'
    recommendations_text = '✅ Recommendations' if recommendations_count > 0 else '❌ No Recommendations'
    connections_text = '✅ Well Connected' if connections >= 500 else '❌ Limited Connections'
    followers_text = '✅ Has Followers' if followers > 100 else '❌ Few Followers'
    awards_text = '✅ Awards/Honors' if has_awards else '❌ No Awards'
    photo_text = '✅ Profile Photo' if profile_photo_url else '❌ No Photo'
    about_section = f'<div style="margin-top: 20px; padding-top: 20px; border-top: 1px solid #e0e0e0;"><div style="color: #333; font-weight: bold; margin-bottom: 10px; font-size: 14px;">💭 ABOUT</div><div style="color: #666; font-size: 13px; line-height: 1.5;">{about[:200] + "..." if len(about) > 200 else about}</div></div>' if about and len(about.strip()) > 0 else ''
    profile_id = f'{name.upper().replace(" ", "-")}-{str(hash(name))[-6:]}'
    
    access_card_html = '''
    <div style="
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border: 3px solid {badge_border_color};
        border-radius: 16px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
        position: relative;
        overflow: hidden;
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
    ">
        <!-- Header stripe -->
        <div style="
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 8px;
            background: {badge_border_color};
        "></div>
        
        <!-- Trust badge corner -->
        <div style="
            position: absolute;
            top: 15px;
            right: 15px;
            background: {trust_color};
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
            text-transform: uppercase;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        ">
            {trust_icon} {trust_level}
        </div>
        
        <!-- Main content area -->
        <div style="display: flex; align-items: flex-start; gap: 25px; margin-top: 15px;">
            <!-- Profile photo section -->
            <div style="flex-shrink: 0;">
                <div style="
                    width: 120px;
                    height: 120px;
                    border-radius: 12px;
                    overflow: hidden;
                    border: 4px solid {badge_border_color};
                    box-shadow: 0 4px 16px rgba(0,0,0,0.15);
                    background: #f5f5f5;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                ">
                    {profile_photo_html}
                </div>
                <div style="
                    text-align: center;
                    margin-top: 10px;
                    color: {photo_color};
                    font-size: 11px;
                    font-weight: bold;
                ">
                    {photo_status}
                </div>
            </div>
            
            <!-- Profile information section -->
            <div style="flex-grow: 1;">
                <!-- Name and title -->
                <div style="margin-bottom: 20px;">
                    <h2 style="
                        margin: 0 0 8px 0;
                        color: #333;
                        font-size: 28px;
                        font-weight: 700;
                        line-height: 1.2;
                    ">{name}</h2>
                    <div style="
                        color: #666;
                        font-size: 16px;
                        font-weight: 500;
                        margin-bottom: 5px;
                    ">{position}</div>
                    <div style="
                        color: #888;
                        font-size: 14px;
                    ">{current_company} • {location}</div>
                </div>
                
                <!-- Status indicators row -->
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-bottom: 20px;">
                    <!-- Company match -->
                    <div style="
                        background: {company_bg};
                        border: 1px solid {company_border};
                        border-radius: 10px;
                        padding: 12px;
                        text-align: center;
                    ">
                        <div style="font-size: 20px; margin-bottom: 5px;">{company_icon}</div>
                        <div style="
                            color: {company_border};
                            font-size: 11px;
                            font-weight: bold;
                            text-transform: uppercase;
                        ">
                            {company_status}
                        </div>
                    </div>
                    
                    <!-- Activity level -->
                    <div style="
                        background: {activity_color}20;
                        border: 1px solid {activity_color};
                        border-radius: 10px;
                        padding: 12px;
                        text-align: center;
                    ">
                        <div style="font-size: 20px; margin-bottom: 5px;">{activity_icon}</div>
                        <div style="
                            color: {activity_color};
                            font-size: 11px;
                            font-weight: bold;
                            text-transform: uppercase;
                        ">
                            {activity_status}
                        </div>
                        <div style="color: #666; font-size: 10px; margin-top: 2px;">
                            {activity_count} items
                        </div>
                    </div>
                    
                    <!-- Network stats -->
                    <div style="
                        background: rgba(33, 150, 243, 0.1);
                        border: 1px solid #2196F3;
                        border-radius: 10px;
                        padding: 12px;
                        text-align: center;
                    ">
                        <div style="font-size: 20px; margin-bottom: 5px;">🌐</div>
                        <div style="
                            color: #2196F3;
                            font-size: 11px;
                            font-weight: bold;
                            text-transform: uppercase;
                        ">
                            NETWORK
                        </div>
                        <div style="color: #666; font-size: 10px; margin-top: 2px;">
                            {connections:,} connections
                        </div>
                    </div>
                </div>
                
                <!-- Trust score details -->
                <div style="
                    background: {trust_color}10;
                    border: 1px solid {trust_color};
                    border-radius: 10px;
                    padding: 15px;
                ">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <div style="
                                color: {trust_color};
                                font-size: 14px;
                                font-weight: bold;
                                margin-bottom: 5px;
                            ">
                                Trust Indicators ({trust_score}/5)
                            </div>
                            <div style="color: #666; font-size: 12px;">
                                {recommendations_text} • 
                                {connections_text} • 
                                {followers_text} • 
                                {awards_text} • 
                                {photo_text}
                            </div>
                        </div>
                        <div style="
                            color: {trust_color};
                            font-size: 24px;
                            font-weight: bold;
                        ">
                            {trust_score}/5
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- About section if available -->
        {about_section}
        
        <!-- Footer ID stripe -->
        <div style="
            margin-top: 20px;
            padding-top: 15px;
            border-top: 1px solid #e0e0e0;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 11px;
            color: #888;
        ">
            <div>FRAUDSPOT PROFILE ID</div>
            <div style="font-family: monospace; font-weight: bold;">
                {profile_id}
            </div>
        </div>
    </div>
    '''.format(
        badge_border_color=badge_border_color,
        trust_color=trust_color,
        trust_icon=trust_icon,
        trust_level=trust_level,
        profile_photo_html=profile_photo_html,
        photo_color=photo_color,
        photo_status=photo_status,
        name=name,
        position=position,
        current_company=current_company,
        location=location,
        company_bg=company_bg,
        company_border=company_border,
        company_icon=company_icon,
        company_status=company_status,
        activity_color=activity_color,
        activity_icon=activity_icon,
        activity_status=activity_status,
        activity_count=activity_count,
        connections=connections,
        trust_score=trust_score,
        recommendations_text=recommendations_text,
        connections_text=connections_text,
        followers_text=followers_text,
        awards_text=awards_text,
        photo_text=photo_text,
        about_section=about_section,
        profile_id=profile_id
    )
    
    # Render HTML card using the same method as other working cards
    render_html_card(access_card_html)


def _render_photo_card(photo_url: str, has_photo: bool) -> None:
    """Render profile photo card with actual image or placeholder."""
    if has_photo and photo_url:
        # Display actual photo
        photo_html = f'''
        <div style="text-align: center; padding: 15px; background: white; 
                    border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
            <img src="{photo_url}" 
                 style="width: 100px; height: 100px; border-radius: 50%; 
                        object-fit: cover; border: 3px solid #4CAF50;">
            <div style="margin-top: 10px; color: #4CAF50; font-weight: bold; font-size: 12px;">
                ✅ Photo Available
            </div>
        </div>
        '''
    else:
        # Placeholder
        photo_html = '''
        <div style="text-align: center; padding: 15px; background: white; 
                    border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
            <div style="width: 100px; height: 100px; border-radius: 50%; 
                        background: #f5f5f5; display: flex; align-items: center; 
                        justify-content: center; margin: 0 auto; border: 3px solid #ccc;">
                <span style="font-size: 40px; color: #999;">👤</span>
            </div>
            <div style="margin-top: 10px; color: #f44336; font-weight: bold; font-size: 12px;">
                ❌ No Photo
            </div>
        </div>
        '''
    
    st.markdown(photo_html, unsafe_allow_html=True)


def _render_basic_info_card(name: str, position: str, company: str, location: str) -> None:
    """Render basic profile information card."""
    info_html = f'''
    <div style="background: white; padding: 20px; border-radius: 12px; 
                box-shadow: 0 2px 8px rgba(0,0,0,0.1); height: 100%;">
        <h3 style="margin: 0 0 15px 0; color: #333; font-size: 20px;">{name}</h3>
        <div style="margin-bottom: 10px;">
            <strong style="color: #666;">Position:</strong><br>
            <span style="color: #333;">{position}</span>
        </div>
        <div style="margin-bottom: 10px;">
            <strong style="color: #666;">Company:</strong><br>
            <span style="color: #333;">{company}</span>
        </div>
        <div>
            <strong style="color: #666;">Location:</strong><br>
            <span style="color: #333;">{location}</span>
        </div>
    </div>
    '''
    st.markdown(info_html, unsafe_allow_html=True)


def _render_trust_card(trust_level: str, trust_label: str, trust_score: int) -> None:
    """Render trust score card."""
    if trust_level == "trusted":
        color = "#4CAF50"
        icon = "🏆"
    elif trust_level == "established": 
        color = "#FF9800"
        icon = "⭐"
    else:
        color = "#9E9E9E"
        icon = "👤"
    
    card_html = f'''
    <div style="background: white; padding: 15px; border-radius: 12px; 
                box-shadow: 0 2px 8px rgba(0,0,0,0.1); text-align: center; height: 120px;
                border-left: 4px solid {color};">
        <div style="font-size: 32px; margin-bottom: 10px;">{icon}</div>
        <div style="color: {color}; font-weight: bold; font-size: 14px;">{trust_label}</div>
        <div style="color: #666; font-size: 12px; margin-top: 5px;">{trust_score}/5 Indicators</div>
    </div>
    '''
    st.markdown(card_html, unsafe_allow_html=True)


def _render_company_match_card(company_match: bool, match_type: str, poster_company: str, job_company: str) -> None:
    """Render company match card."""
    if company_match:
        color = "#4CAF50"
        icon = "🏢"
        status = "Same Company"
        detail = "Poster works here"
    else:
        color = "#FF9800"
        icon = "🏭"
        status = "Different Company"
        detail = "External poster"
    
    card_html = f'''
    <div style="background: white; padding: 15px; border-radius: 12px; 
                box-shadow: 0 2px 8px rgba(0,0,0,0.1); text-align: center; height: 120px;
                border-left: 4px solid {color};">
        <div style="font-size: 32px; margin-bottom: 10px;">{icon}</div>
        <div style="color: {color}; font-weight: bold; font-size: 14px;">{status}</div>
        <div style="color: #666; font-size: 12px; margin-top: 5px;">{detail}</div>
    </div>
    '''
    st.markdown(card_html, unsafe_allow_html=True)


def _render_activity_card(activity_count: int, has_activity: bool) -> None:
    """Render activity level card."""
    if activity_count > 10:
        color = "#4CAF50"
        icon = "🔥"
        status = "Very Active"
    elif activity_count > 3:
        color = "#FF9800"  
        icon = "📊"
        status = "Active"
    elif has_activity:
        color = "#2196F3"
        icon = "📈"
        status = "Some Activity"
    else:
        color = "#f44336"
        icon = "😴"
        status = "No Activity"
    
    card_html = f'''
    <div style="background: white; padding: 15px; border-radius: 12px; 
                box-shadow: 0 2px 8px rgba(0,0,0,0.1); text-align: center; height: 120px;
                border-left: 4px solid {color};">
        <div style="font-size: 32px; margin-bottom: 10px;">{icon}</div>
        <div style="color: {color}; font-weight: bold; font-size: 14px;">{status}</div>
        <div style="color: #666; font-size: 12px; margin-top: 5px;">{activity_count} Items</div>
    </div>
    '''
    st.markdown(card_html, unsafe_allow_html=True)


def _render_photo_status_card(has_photo: bool) -> None:
    """Render photo status card."""
    color = "#4CAF50" if has_photo else "#f44336"
    icon = "📸" if has_photo else "🚫"
    status = "Has Photo" if has_photo else "No Photo"
    
    card_html = f'''
    <div style="background: white; padding: 15px; border-radius: 12px; 
                box-shadow: 0 2px 8px rgba(0,0,0,0.1); text-align: center; height: 120px;
                border-left: 4px solid {color};">
        <div style="font-size: 32px; margin-bottom: 10px;">{icon}</div>
        <div style="color: {color}; font-weight: bold; font-size: 14px;">{status}</div>
        <div style="color: #666; font-size: 12px; margin-top: 5px;">Profile Photo</div>
    </div>
    '''
    st.markdown(card_html, unsafe_allow_html=True)


def _render_about_card(about: str) -> None:
    """Render about section card."""
    # Truncate if too long
    display_about = about[:300] + "..." if len(about) > 300 else about
    
    about_html = f'''
    <div style="background: white; padding: 20px; border-radius: 12px; 
                box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-top: 15px;
                border-left: 4px solid #2196F3;">
        <h4 style="margin: 0 0 15px 0; color: #333;">💭 About</h4>
        <p style="margin: 0; color: #666; line-height: 1.6; font-size: 14px;">{display_about}</p>
    </div>
    '''
    st.markdown(about_html, unsafe_allow_html=True)


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
    
    # Use centralized verification service
    from ...services.verification_service import VerificationService
    verification_service = VerificationService()
    
    # Get verification badges from service
    badges = verification_service.get_verification_badges(job_data)
    
    # Create 4 columns for badges
    cols = st.columns(4)
    
    for col, badge_info in zip(cols, badges):
        with col:
            # Render the badge
            render_verification_badge(
                label=badge_info['label'],
                verified=badge_info['verified'],
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
    
    # Use centralized verification service
    from ...services.verification_service import VerificationService
    verification_service = VerificationService()
    
    # Get verification summary from service
    verification_summary = verification_service.get_verification_summary(job_data)
    verification_percentage = verification_summary['percentage']
    
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
            {f'<p style="margin: 10px 0 0 0; color: #666; font-size: 14px; line-height: 1.5;">{description}</p>' if description else ''}
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