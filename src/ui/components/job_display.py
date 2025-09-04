"""
Job Display Component

This module handles the display of job posting information including:
- Basic job details (title, company, location, salary)
- Trust scores and company verification
- Contact information
- Job description with fraud highlighting
"""

import html
import os
import sys
from typing import Any, Dict, List, Tuple

import streamlit as st

# Add the src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Import shared functions instead of duplicating (DRY principle)
from src.ui.utils.helpers import calculate_company_trust_score, get_trust_color
from src.ui.utils.streamlit_html import render_html_card, render_info_card, render_metric_card


def _format_applicant_count(count: int) -> str:
    """
    Format applicant count to match LinkedIn display format.
    
    Args:
        count (int): Number of applicants
        
    Returns:
        str: Formatted applicant count string
    """
    if count is None:
        return ""
    
    if count >= 200:
        return "Over 100 applicants"
    elif count >= 100:
        return f"Over {count//10*10} applicants"
    elif count >= 50:
        return f"Over {count//10*10} applicants"
    else:
        return f"{count} applicants"


def display_job_info_card(job_data: Dict[str, Any]) -> None:
    """
    Display basic job information in a professional card format.
    
    Args:
        job_data (Dict[str, Any]): Job posting data
    """
    if not job_data:
        st.warning("No job data available to display")
        return
    
    # Extract and escape data with comprehensive fallbacks
    title = html.escape(
        job_data.get('title') or 
        job_data.get('job_title') or 
        job_data.get('name') or 
        ''
    )
    
    company = html.escape(
        job_data.get('company') or 
        job_data.get('company_name') or 
        ''
    )
    
    location = html.escape(
        job_data.get('location') or 
        job_data.get('job_location') or 
        job_data.get('region') or 
        job_data.get('city') or 
        ''
    )
    
    salary = html.escape(
        job_data.get('salary_info') or 
        job_data.get('benefits') or 
        job_data.get('salary') or
        'Not specified'
    )
    
    # Calculate trust metrics
    trust_score = calculate_company_trust_score(company)
    trust_color = get_trust_color(trust_score)
    
    # Create the job info card with consistent styling
    job_info_html = f'''
    <div style="background: white; padding: 30px; border-radius: 12px; 
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08); 
                border: 1px solid #e5e7eb; margin: 20px 0;">
        
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 25px;">
            <h3 style="color: #1f2937; margin: 0; font-size: 20px; font-weight: 700;">📋 Job Information</h3>
            <div style="background: {trust_color}; color: white; padding: 8px 16px; 
                       border-radius: 6px; font-size: 13px; font-weight: 600;">
                Trust Score: {trust_score}/100
            </div>
        </div>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 25px; margin-bottom: 20px;">
            <div>
                <h4 style="color: #6b7280; margin: 0 0 8px 0; font-size: 14px; font-weight: 600;">🏢 Company</h4>
                <p style="color: #1f2937; margin: 0; font-size: 16px; font-weight: 500;">{company}</p>
            </div>
            <div>
                <h4 style="color: #6b7280; margin: 0 0 8px 0; font-size: 14px; font-weight: 600;">📍 Location</h4>
                <p style="color: #1f2937; margin: 0; font-size: 16px;">{location}</p>
            </div>
        </div>
        
        <div style="display: grid; grid-template-columns: 2fr 1fr; gap: 25px;">
            <div>
                <h4 style="color: #6b7280; margin: 0 0 8px 0; font-size: 14px; font-weight: 600;">💼 Job Title</h4>
                <p style="color: #1f2937; margin: 0; font-size: 18px; font-weight: 700;">{title}</p>
            </div>
            <div>
                <h4 style="color: #6b7280; margin: 0 0 8px 0; font-size: 14px; font-weight: 600;">💰 Compensation</h4>
                <p style="color: #1f2937; margin: 0; font-size: 16px;">{salary}</p>
            </div>
        </div>
    </div>
    '''
    
    render_html_card(job_info_html)


def display_contact_information(job_data: Dict[str, Any]) -> None:
    """
    Display contact information if available.
    
    Args:
        job_data (Dict[str, Any]): Job posting data
    """
    contact = job_data.get('contact_info', {})
    if not isinstance(contact, dict):
        return
    
    # Safely convert all contact info to strings
    def safe_string_list(items):
        """Convert any list items to strings safely"""
        if not items:
            return []
        safe_items = []
        for item in items:
            if isinstance(item, (str, int, float)):
                safe_items.append(str(item))
            elif isinstance(item, dict):
                # Handle dict items - extract meaningful string representation
                if 'value' in item:
                    safe_items.append(str(item['value']))
                elif 'number' in item:
                    safe_items.append(str(item['number']))
                else:
                    # Convert dict to string format
                    safe_items.append(str(item))
            else:
                safe_items.append(str(item))
        return safe_items
    
    emails = safe_string_list(contact.get('emails', []))
    phones = safe_string_list(contact.get('phones', []))
    whatsapp = safe_string_list(contact.get('whatsapp', []))
    telegram = safe_string_list(contact.get('telegram', []))
    
    # Check if we have any contact information
    if not any([emails, phones, whatsapp, telegram]):
        return
    
    st.markdown("### 📞 Contact Information")
    
    contact_html = '''
    <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; 
                border-left: 4px solid #007bff; margin-bottom: 15px;">
    '''
    
    if emails:
        email_list = ', '.join(emails[:3])  # Limit to first 3
        if len(emails) > 3:
            email_list += f" (+{len(emails) - 3} more)"
        contact_html += f'''
        <p><strong>📧 Emails:</strong> {html.escape(email_list)}</p>
        '''
    
    if phones:
        phone_list = ', '.join(phones[:3])  # Limit to first 3
        if len(phones) > 3:
            phone_list += f" (+{len(phones) - 3} more)"
        contact_html += f'''
        <p><strong>📞 Phones:</strong> {html.escape(phone_list)}</p>
        '''
    
    if whatsapp:
        whatsapp_list = ', '.join(whatsapp[:2])  # Limit to first 2
        if len(whatsapp) > 2:
            whatsapp_list += f" (+{len(whatsapp) - 2} more)"
        contact_html += f'''
        <p><strong>💬 WhatsApp:</strong> {html.escape(whatsapp_list)}</p>
        '''
    
    if telegram:
        telegram_list = ', '.join(telegram[:2])  # Limit to first 2
        if len(telegram) > 2:
            telegram_list += f" (+{len(telegram) - 2} more)"
        contact_html += f'''
        <p><strong>✈️ Telegram:</strong> {html.escape(telegram_list)}</p>
        '''
    
    contact_html += '</div>'
    
    render_html_card(contact_html)


def display_job_metrics(job_data: Dict[str, Any]) -> None:
    """
    Display key job metrics in a card format.
    
    Args:
        job_data (Dict[str, Any]): Job posting data
    """
    # Create metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        trust_score = calculate_company_trust_score(
            job_data.get('company_name', job_data.get('company'))
        )
        color = get_trust_color(trust_score)
        render_metric_card("Company Trust", f"{trust_score}%", color)
    
    with col2:
        # Job posting time from Bright Data - NO FALLBACKS
        posting_time = job_data.get('job_posted_time')
        if posting_time:
            display_text = str(posting_time)
            # Calculate freshness for color
            try:
                if 'ago' in str(posting_time).lower() or 'day' in str(posting_time).lower():
                    color = "#4CAF50"  # Recent
                else:
                    color = "#FF9800"  # Unknown age
            except:
                color = "#FF9800"
            render_metric_card("Posted", display_text, color)
    
    with col3:
        # Application count from Bright Data - NO FALLBACKS
        applications = job_data.get('job_num_applicants')
        if applications is not None:
            color = "#007bff"
            # Use helper function for consistent formatting
            display_text = _format_applicant_count(applications).replace(' applicants', '')
            render_metric_card("Applications", display_text, color)
    
    with col4:
        # Job type from Bright Data - NO FALLBACKS  
        job_type = job_data.get('job_employment_type')
        if job_type:
            color = "#28a745" if job_type == 'Full-time' else "#ffc107"
            render_metric_card("Type", str(job_type), color)


def display_job_description(job_data: Dict[str, Any]) -> None:
    """
    Display job description with fraud keyword highlighting.
    
    Args:
        job_data (Dict[str, Any]): Job posting data
    """
    # Try to get formatted HTML version first for better display
    job_formatted_html = job_data.get('job_description_formatted', '')
    job_desc = (job_data.get('job_description') or 
                job_data.get('job_summary') or 
                job_data.get('description') or '')
    
    if not job_desc and not job_formatted_html:
        return
    
    st.markdown("### 📝 Job Description")
    
    # If we have HTML content, process it for display
    if job_formatted_html and len(job_formatted_html) > len(str(job_desc)):
        def clean_html_for_display(html_content: str) -> str:
            """Clean HTML for display while preserving formatting"""
            import re
            from html import unescape

            # Unescape HTML entities
            text = unescape(str(html_content))
            
            # Remove button elements completely
            text = re.sub(r'<button[^>]*>.*?</button>', '', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<icon[^>]*>.*?</icon>', '', text, flags=re.DOTALL | re.IGNORECASE)
            
            # Convert some HTML tags to markdown/text equivalents
            text = re.sub(r'<strong[^>]*>(.*?)</strong>', r'**\1**', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<b[^>]*>(.*?)</b>', r'**\1**', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<em[^>]*>(.*?)</em>', r'*\1*', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<i[^>]*>(.*?)</i>', r'*\1*', text, flags=re.DOTALL | re.IGNORECASE)
            
            # Handle list items
            text = re.sub(r'<li[^>]*>', '• ', text, flags=re.IGNORECASE)
            text = re.sub(r'</li>', '\n', text, flags=re.IGNORECASE)
            text = re.sub(r'</?ul[^>]*>', '\n', text, flags=re.IGNORECASE)
            text = re.sub(r'</?ol[^>]*>', '\n', text, flags=re.IGNORECASE)
            
            # Handle paragraphs and line breaks
            text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
            text = re.sub(r'</?p[^>]*>', '\n\n', text, flags=re.IGNORECASE)
            
            # Remove remaining HTML tags
            text = re.sub(r'<[^>]+>', '', text)
            
            # Clean up whitespace
            text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
            text = re.sub(r'[ \t]+', ' ', text)
            text = text.strip()
            
            # Remove "Show more Show less" text that appears from LinkedIn
            text = re.sub(r'\s*Show more\s*Show less\s*', '', text, flags=re.IGNORECASE)
            text = re.sub(r'\s*Show more\s*', '', text, flags=re.IGNORECASE)
            text = re.sub(r'\s*Show less\s*', '', text, flags=re.IGNORECASE)
            
            return text
        
        job_desc = clean_html_for_display(job_formatted_html)
    
    # Common fraud keywords to highlight
    fraud_keywords = [
        'guaranteed income', 'easy money', 'no experience needed', 'work from home',
        'make money fast', 'unlimited earning potential', 'be your own boss',
        'financial freedom', 'urgent', 'apply now', 'limited time',
        'investment opportunity', 'pyramid', 'multi-level marketing', 'MLM'
    ]
    
    # First escape the HTML to prevent injection, then highlight keywords
    highlighted_desc = html.escape(str(job_desc))
    for keyword in fraud_keywords:
        if keyword.lower() in highlighted_desc.lower():
            # Case-insensitive replacement with highlight on escaped text
            import re
            escaped_keyword = html.escape(keyword)
            pattern = re.compile(re.escape(escaped_keyword), re.IGNORECASE)
            highlighted_desc = pattern.sub(
                f'<mark style="background-color: #ffcccc; padding: 2px 4px; border-radius: 3px;">{escaped_keyword}</mark>',
                highlighted_desc
            )
    
    # Display in a styled container with consistent card design
    desc_html = f'''
    <div style="background: white; padding: 25px; border-radius: 12px; 
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08); 
                border: 1px solid #e5e7eb; margin: 20px 0;
                max-height: 500px; overflow-y: auto;">
        <div style="line-height: 1.6; color: #374151;">{highlighted_desc}</div>
    </div>
    '''
    
    render_html_card(desc_html)
    
    # Show fraud indicator count if any keywords found
    fraud_count = sum(1 for keyword in fraud_keywords if keyword.lower() in job_desc.lower())
    if fraud_count > 0:
        st.warning(f"⚠️ Found {fraud_count} potentially suspicious keyword(s) highlighted above")


def display_enhanced_job_details(job_data: Dict[str, Any]) -> None:
    """
    Display enhanced job details with additional context.
    
    Args:
        job_data (Dict[str, Any]): Job posting data
    """
    if not job_data:
        return
    
    st.markdown("### 📄 Enhanced Job Details")
    
    # Get additional details using Bright Data field names
    industry = job_data.get('industry', job_data.get('job_industries', 'Not specified'))
    experience_level = job_data.get('experience_level', 
                                   job_data.get('job_seniority_level', 
                                              job_data.get('experience', 'Not specified')))
    company_size = job_data.get('company_size', 'Not specified')
    benefits = job_data.get('benefits', job_data.get('perks', 'Not specified'))
    
    # Create details grid
    col1, col2 = st.columns(2)
    
    with col1:
        render_info_card("Industry", industry, "🏭")
        render_info_card("Experience Level", experience_level, "📈")
    
    with col2:
        render_info_card("Company Size", company_size, "👥")
        render_info_card("Benefits", benefits, "🎁")
    
    # Display job description with fraud keyword highlighting
    display_job_description(job_data)
    
    # Display job URL if available
    job_url = job_data.get('url', job_data.get('job_url'))
    if job_url:
        st.markdown(f"**🔗 Original Posting:** [View on LinkedIn]({job_url})")


def display_scraped_data(job_data: Dict[str, Any]) -> None:
    """
    Main function to display clean, user-friendly job data card.
    Removes technical clutter and focuses on essential information.
    
    Args:
        job_data (Dict[str, Any]): Complete job posting data
    """
    if not job_data:
        st.error("No job data available to display")
        return
    
    # Display modern job details card
    display_modern_job_card(job_data)


def display_modern_job_card(job_data: Dict[str, Any]) -> None:
    """
    Display a modern, clean job details card with only essential information.
    Implements card-based design with beautiful colors and proper image rendering.
    
    Args:
        job_data (Dict[str, Any]): Job posting data
    """
    if not job_data:
        return
    
    # DEBUG: Log the actual data being passed
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"🎯 JOB DISPLAY DATA: posted_time={job_data.get('job_posted_time')}, applicants={job_data.get('job_num_applicants')}")
        
    # Extract essential job information - NO FALLBACKS, only real LinkedIn data
    job_title = job_data.get('job_title')
    company_name = job_data.get('company_name')
    location = job_data.get('job_location')  # Use correct API field name
    posted_time = job_data.get('job_posted_time')
    applicants = job_data.get('job_num_applicants')
    
    # Early return if essential data is missing
    if not job_title or not company_name:
        st.error("Essential job data missing - cannot display job card")
        return
    
    # Get company logo URL and render as image
    company_logo_url = job_data.get('company_logo', '')
    logo_html = ""
    if company_logo_url and 'http' in str(company_logo_url):
        logo_html = f'''
        <img src="{html.escape(str(company_logo_url))}" 
             style="width: 60px; height: 60px; border-radius: 12px; object-fit: cover; 
                    border: 2px solid #e0e7ff; margin-right: 20px;"
             alt="Company Logo"
             onerror="this.style.display='none'">
        '''
    else:
        # Fallback icon if no logo
        logo_html = '''
        <div style="width: 60px; height: 60px; border-radius: 12px; 
                    background: #667eea;
                    display: flex; align-items: center; justify-content: center; 
                    margin-right: 20px; color: white; font-size: 24px; font-weight: bold;">
            🏢
        </div>
        '''
    
    # Create clean, professional job card with white background
    job_card_html = f'''
    <div style="background: white; padding: 30px; border-radius: 12px; 
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08); 
                border: 1px solid #e5e7eb; margin: 20px 0;">
        
        <!-- Header with logo and basic info -->
        <div style="display: flex; align-items: flex-start; margin-bottom: 25px;">
            {logo_html}
            <div style="flex: 1;">
                <h2 style="margin: 0 0 8px 0; color: #1f2937; font-size: 28px; font-weight: 700; 
                           line-height: 1.2;">
                    {html.escape(job_title)}
                </h2>
                <div style="display: flex; align-items: center; gap: 20px; margin-bottom: 12px;">
                    <div style="display: flex; align-items: center; color: #4b5563; font-weight: 500;">
                        <span style="margin-right: 8px;">🏢</span>
                        {html.escape(company_name)}
                    </div>
                    {f'<div style="display: flex; align-items: center; color: #4b5563;"><span style="margin-right: 8px;">📍</span>{html.escape(location)}</div>' if location else ''}
                </div>
                <div style="display: flex; align-items: center; gap: 20px; color: #6b7280; font-size: 14px;">
                    {f'<div style="display: flex; align-items: center;"><span style="margin-right: 6px;">🕒</span>{html.escape(str(posted_time))}</div>' if posted_time else ''}
                    {f'<div style="display: flex; align-items: center;"><span style="margin-right: 6px;">👥</span>{_format_applicant_count(applicants)}</div>' if applicants is not None else ''}
                </div>
            </div>
        </div>
        
        <!-- Job Details Section -->
        <div style="background: #f9fafb; padding: 20px; border-radius: 8px; 
                    border: 1px solid #f3f4f6;">
            <div style="display: flex; align-items: center; margin-bottom: 15px;">
                <span style="font-size: 18px; margin-right: 10px;">📄</span>
                <h3 style="margin: 0; color: #374151; font-size: 16px; font-weight: 600;">
                    Job Details
                </h3>
            </div>
            <div style="color: #4b5563; line-height: 1.6; font-size: 14px;">
                <div style="margin-bottom: 12px;">
                    <strong style="color: #374151;">Employment:</strong> 
                    <span style="background: #e5e7eb; padding: 2px 8px; border-radius: 4px; font-size: 13px;">
                        {job_data.get('job_employment_type', 'Full-time')}
                    </span>
                    {f'<span style="margin-left: 15px;"><strong style="color: #374151;">Level:</strong> <span style="background: #dbeafe; padding: 2px 8px; border-radius: 4px; color: #1e40af; font-size: 13px;">{job_data.get("job_seniority_level", "Not specified")}</span></span>' if job_data.get('job_seniority_level') else ''}
                </div>
                {f'<div><strong style="color: #374151;">Industry:</strong> <span style="background: #fef3c7; padding: 2px 8px; border-radius: 4px; color: #92400e; font-size: 13px;">{job_data.get("job_industries", "Not specified")}</span></div>' if job_data.get('job_industries') else ''}
            </div>
        </div>
    </div>
    '''
    
    render_html_card(job_card_html)
    
    # Add job summary as Streamlit expander (native and functional)
    job_summary = _get_job_summary(job_data)
    if job_summary and job_summary != "No job description available.":
        with st.expander("📋 Full Job Description", expanded=False):
            st.markdown(job_summary)
    
    # Display additional details in collapsible sections if needed
    _display_additional_details(job_data)


def _get_job_summary(job_data: Dict[str, Any]) -> str:
    """Get a clean, concise job summary from available data."""
    # Try different description fields
    description = (job_data.get('job_description') or 
                  job_data.get('description') or 
                  job_data.get('job_summary') or '')
    
    if not description:
        return "No job description available."
    
    # Clean and truncate the description
    import re
    from html import unescape

    # Clean HTML if present
    text = unescape(str(description))
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    
    # Remove "Show more Show less" text that appears from LinkedIn
    text = re.sub(r'\s*Show more\s*Show less\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*Show more\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*Show less\s*', '', text, flags=re.IGNORECASE)
    
    # Return full text without truncation
    return html.escape(text)


def _display_additional_details(job_data: Dict[str, Any]) -> None:
    """Display additional job details in clean, collapsible format if available."""
    
    # Check if we have additional details worth showing
    additional_info = {}
    
    # Collect relevant additional information
    if job_data.get('job_employment_type'):
        additional_info['Employment Type'] = job_data.get('job_employment_type')
    
    if job_data.get('job_industries'):
        additional_info['Industry'] = job_data.get('job_industries')
    
    if job_data.get('job_seniority_level'):
        additional_info['Experience Level'] = job_data.get('job_seniority_level')
    
    # Company URL (if available)
    company_url = job_data.get('company_url')
    if company_url:
        additional_info['Company Website'] = f'<a href="{html.escape(str(company_url))}" target="_blank" style="color: #4299e1; text-decoration: none;">Visit Website →</a>'
    
    # Job URL
    job_url = job_data.get('url') or job_data.get('job_url')
    if job_url:
        additional_info['Original Posting'] = f'<a href="{html.escape(str(job_url))}" target="_blank" style="color: #4299e1; text-decoration: none;">View on LinkedIn →</a>'
    
    # Only show additional details if we have some
    if additional_info:
        with st.expander("🔍 Additional Details", expanded=False):
            cols = st.columns(2)
            items = list(additional_info.items())
            
            for i, (label, value) in enumerate(items):
                with cols[i % 2]:
                    if '<a href=' in str(value):
                        st.markdown(f"**{label}:** {value}", unsafe_allow_html=True)
                    else:
                        st.markdown(f"**{label}:** {value}")


def display_dynamic_job_data(job_data: Dict[str, Any]) -> None:
    """Display ALL job data dynamically based on what's available."""
    
    # Skip metadata fields
    skip_fields = {'success', 'scraped_at', 'scraping_method', 'data_source', 
                   'api_version', 'url', 'scraping_success', 'timestamp'}
    
    # Primary job information fields (display first)
    primary_fields = ['job_title', 'company_name', 'location', 'job_description', 'salary_info']
    contact_fields = ['contact_info']
    
    st.markdown("### 📋 Job Details")
    
    # Display primary job information first
    st.markdown("#### Basic Job Information")
    for field in primary_fields:
        if field in job_data and job_data[field]:
            display_job_field(field, job_data[field])
    
    # Display contact information separately
    if 'contact_info' in job_data and job_data['contact_info']:
        st.markdown("#### Contact Information")
        display_job_field('contact_info', job_data['contact_info'])
    
    # Display all other fields dynamically
    st.markdown("#### Additional Job Details")
    for field, value in job_data.items():
        if field not in skip_fields and field not in primary_fields and field not in contact_fields:
            if value:  # Only display if has data
                display_job_field(field, value)


def display_job_field(field_name: str, value: Any) -> None:
    """Display a job field with appropriate UI element based on data type."""
    
    # Format field name for display
    display_name = field_name.replace('_', ' ').replace('-', ' ').title()
    
    # Special handling for certain job fields
    if field_name == 'job_description':
        st.markdown(f"**{display_name}**")
        if len(str(value)) > 500:
            with st.expander("View Full Description"):
                st.text_area("", value, height=200, disabled=True)
        else:
            st.text_area("", value, height=100, disabled=True)
    
    elif field_name == 'contact_info' and isinstance(value, dict):
        st.markdown(f"**{display_name}**")
        display_contact_data(value)
    
    elif isinstance(value, list):
        if len(value) > 0:
            st.markdown(f"**{display_name}** ({len(value)} items)")
            
            # Check first item to determine display format
            if isinstance(value[0], dict):
                # Display as expandable cards for complex objects
                for item in value[:5]:  # Limit to first 5
                    with st.expander(get_job_item_title(item)):
                        display_job_dict(item)
            else:
                # Display as tags for simple lists
                display_job_tags(value)
    
    elif isinstance(value, dict):
        st.markdown(f"**{display_name}**")
        with st.container():
            display_job_dict(value)
    
    elif isinstance(value, str):
        if len(value) > 200:
            # Long text - use expander
            with st.expander(f"{display_name}"):
                st.text(value)
        elif 'url' in field_name.lower() or (isinstance(value, str) and value.startswith('http')):
            # URL - make clickable
            st.markdown(f"**{display_name}**: [Link]({value})")
        else:
            # Short text - inline
            st.markdown(f"**{display_name}**: {value}")
    
    elif isinstance(value, (int, float)):
        # Numbers - show with appropriate formatting
        if 'score' in field_name.lower():
            st.metric(display_name, f"{value:.2%}")
        elif field_name in ['salary', 'salary_info'] and value > 1000:
            st.metric(display_name, f"${value:,}")
        else:
            st.metric(display_name, f"{value:,}")
    
    elif isinstance(value, bool):
        # Boolean - show as badge
        if value:
            st.success(f"✅ {display_name}")
        else:
            st.info(f"❌ {display_name}")


def display_contact_data(contact_info: dict) -> None:
    """Display contact information in a structured way."""
    for contact_type, contact_list in contact_info.items():
        if contact_list and len(contact_list) > 0:
            formatted_type = contact_type.replace('_', ' ').title()
            if len(contact_list) == 1:
                st.write(f"• **{formatted_type}**: {contact_list[0]}")
            else:
                st.write(f"• **{formatted_type}** ({len(contact_list)}):")
                for contact in contact_list[:3]:  # Show first 3
                    st.write(f"  - {contact}")
                if len(contact_list) > 3:
                    st.write(f"  ... and {len(contact_list) - 3} more")


def display_job_dict(data: dict) -> None:
    """Display dictionary data in a clean format."""
    for key, val in data.items():
        if val:
            formatted_key = key.replace('_', ' ').title()
            if isinstance(val, (list, dict)):
                st.json(val)  # For complex nested data
            else:
                st.write(f"• **{formatted_key}**: {val}")


def display_job_tags(items: list) -> None:
    """Display list items as tags."""
    tags_html = '<div style="display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0;">'
    for item in items[:15]:  # Limit to 15 tags
        tags_html += f'''
        <span style="background: #e8f5e8; color: #2e7d32; 
                     padding: 4px 12px; border-radius: 16px; 
                     font-size: 14px;">{str(item)}</span>
        '''
    tags_html += '</div>'
    st.markdown(tags_html, unsafe_allow_html=True)


def get_job_item_title(item: dict) -> str:
    """Get a title for a dictionary item."""
    # Try common title fields
    for field in ['title', 'name', 'type', 'category', 'level']:
        if field in item and item[field]:
            return str(item[field])
    # Fallback to first non-empty string value
    for val in item.values():
        if isinstance(val, str) and val:
            return val[:50]
    return "Item"