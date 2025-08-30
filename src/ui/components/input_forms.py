"""
Input Forms Component

This module handles all input form interfaces including:
- URL input with validation
- HTML content input
- Manual job entry form
- Demo mode with sample data
"""

import os
import sys
from typing import Any, Dict

import streamlit as st

# Add the src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.scraper.linkedin_scraper import validate_linkedin_url


def render_url_input() -> str:
    """
    Render the URL input field with validation.
    
    Creates an input field for users to enter LinkedIn job URLs,
    validates the URL format, and provides feedback.
    
    Returns:
        str: The validated LinkedIn job URL, empty string if invalid
    """
    st.subheader("📎 Enter Job Post URL")
    
    url = st.text_input(
        "LinkedIn Job URL",
        placeholder="https://www.linkedin.com/jobs/view/...",
        help="Paste the complete LinkedIn job posting URL here"
    )
    
    if url:
        if validate_linkedin_url(url):
            st.success("✅ Valid LinkedIn URL detected")
            return url
        else:
            st.error("❌ Invalid LinkedIn URL. Please check the URL format.")
            st.info("Expected format: https://www.linkedin.com/jobs/view/[job-id]")
    
    return ""


def render_html_input() -> str:
    """
    Render HTML input section for manual HTML pasting.
    
    Returns:
        str: HTML content or empty string
    """
    st.markdown("#### Instructions:")
    st.markdown("""
    1. 🌐 Open the LinkedIn job posting in your browser
    2. 🖱️ Right-click on the page and select **"View Page Source"**
    3. 📋 Copy all the HTML content (Ctrl+A, Ctrl+C)
    4. 📝 Paste it in the text area below
    """)
    
    html_content = st.text_area(
        "LinkedIn Job HTML Content",
        height=200,
        placeholder="Paste the complete HTML source code of the LinkedIn job posting here...",
        help="Copy the entire HTML source from the LinkedIn job page"
    )
    
    if html_content and len(html_content.strip()) > 100:
        st.success("✅ HTML content received")
        return html_content.strip()
    elif html_content:
        st.warning("⚠️ HTML content seems too short. Please paste the complete page source.")
    
    return ""


def render_manual_input() -> Dict[str, Any]:
    """
    Render manual input form for job details.
    
    Returns:
        Dict[str, Any]: Job data or empty dict
    """
    with st.form("manual_job_input"):
        st.markdown("#### Enter Job Details Manually:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            job_title = st.text_input("Job Title *", placeholder="e.g., Marketing Manager")
            company_name = st.text_input("Company Name *", placeholder="e.g., TechCorp Inc.")
            location = st.text_input("Location", placeholder="e.g., New York, NY")
            salary_info = st.text_input("Salary Info", placeholder="e.g., $50,000 - $70,000")
        
        with col2:
            job_type = st.selectbox("Job Type", ["", "Full-time", "Part-time", "Contract", "Temporary"])
            experience_level = st.selectbox("Experience Level", ["", "Entry", "Associate", "Mid", "Senior", "Executive"])
            industry = st.text_input("Industry", placeholder="e.g., Technology")
        
        job_description = st.text_area(
            "Job Description *",
            height=200,
            placeholder="Enter the complete job description here...",
            help="Include requirements, responsibilities, benefits, and contact information"
        )
        
        submitted = st.form_submit_button("🔍 Analyze Job Posting")
        
        if submitted:
            if job_title and company_name and job_description:
                job_data = {
                    'job_title': job_title,
                    'company_name': company_name,
                    'job_description': job_description,
                    'location': location,
                    'salary_info': salary_info if salary_info else None,
                    'job_type': job_type if job_type else None,
                    'experience_level': experience_level if experience_level else None,
                    'industry': industry if industry else None,
                    'success': True,
                    'scraping_method': 'manual_input',
                    'requirements': []
                }
                return job_data
            else:
                st.error("❌ Please fill in at least Job Title, Company Name, and Job Description.")
    
    return {}

