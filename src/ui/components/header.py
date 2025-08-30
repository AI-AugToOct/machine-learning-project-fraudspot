"""
Header Component

This module handles the application header display including:
- Main application branding and title
- Feature highlights
- Live statistics
"""

import os
import sys

import streamlit as st

# Add the src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.ui.utils.streamlit_html import render_gradient_header, render_html_card


def render_main_header() -> None:
    """
    Render the main application header with branding and live stats.
    """
    # Main gradient header
    header_html = '''
    <div style="background: #2563eb; 
                padding: 30px; border-radius: 15px; margin-bottom: 30px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                border: 1px solid #e2e8f0;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div style="flex: 1;">
                <h1 style="color: white; font-size: 3rem; margin: 0; font-weight: 700;">
                    🕵️ FraudSpot AI
                </h1>
                <h3 style="color: rgba(255, 255, 255, 0.9); margin: 10px 0 0 0; font-weight: 400;">
                    Advanced Job Post Fraud Detection System
                </h3>
                <p style="color: rgba(255, 255, 255, 0.8); margin: 5px 0 0 0; font-size: 14px;">
                    Powered by Machine Learning | Tuwaiq ML Bootcamp Project
                </p>
            </div>
            
            <div style="display: flex; gap: 20px; text-align: center;">
                <div style="background: rgba(255, 255, 255, 0.2); padding: 15px; border-radius: 10px; min-width: 100px;">
                    <div style="color: white; font-size: 24px; font-weight: bold;">49</div>
                    <div style="color: rgba(255, 255, 255, 0.8); font-size: 12px;">AI Features</div>
                </div>
                <div style="background: rgba(255, 255, 255, 0.2); padding: 15px; border-radius: 10px; min-width: 100px;">
                    <div style="color: white; font-size: 24px; font-weight: bold;">ML</div>
                    <div style="color: rgba(255, 255, 255, 0.8); font-size: 12px;">Powered</div>
                </div>
                <div style="background: rgba(255, 255, 255, 0.2); padding: 15px; border-radius: 10px; min-width: 100px;">
                    <div style="color: white; font-size: 24px; font-weight: bold;">Real-time</div>
                    <div style="color: rgba(255, 255, 255, 0.8); font-size: 12px;">Analysis</div>
                </div>
            </div>
        </div>
    </div>
    '''
    
    render_html_card(header_html)


def render_feature_highlights() -> None:
    """

    """
    # Create 4 columns for feature highlights
    col1, col2, col3, col4 = st.columns(4)
    
    features = [
        {
            'icon': '🔍',
            'title': 'Smart Analysis',
            'description': 'AI-powered fraud detection'
        },
        {
            'icon': '⚡',
            'title': 'Real-time',
            'description': 'Instant analysis of LinkedIn job postings'
        },
        {
            'icon': '🛡️',
            'title': 'Secure',
            'description': 'Safe scraping with advanced protection'
        },
        {
            'icon': '📊',
            'title': 'Detailed Reports',
            'description': 'Comprehensive fraud risk assessment'
        }
    ]
    
    columns = [col1, col2, col3, col4]
    
    for col, feature in zip(columns, features):
        with col:
            feature_html = f'''
            <div style="text-align: center; padding: 20px;">
                <div style="font-size: 2rem; margin-bottom: 10px;">{feature['icon']}</div>
                <h4 style="color: #333; margin: 0 0 8px 0; font-size: 16px; font-weight: 600;">
                    {feature['title']}
                </h4>
                <p style="color: #666; margin: 0; font-size: 14px; line-height: 1.4;">
                    {feature['description']}
                </p>
            </div>
            '''
            render_html_card(feature_html)


def render_page_header() -> None:
    """
    Render the complete page header including main header and feature highlights.
    """
    render_main_header()
    render_feature_highlights()


def render_success_banner(message: str = "Job Data Successfully Extracted") -> None:
    """
    Render a success banner for successful operations.
    
    Args:
        message (str): Success message to display
    """
    banner_html = f'''
    <div style="background: #10b981; 
                padding: 20px; border-radius: 10px; margin-bottom: 20px;
                border: 1px solid #e2e8f0;">
        <h2 style="color: white; text-align: center; margin: 0;">
            ✅ {message}
        </h2>
    </div>
    '''
    
    render_html_card(banner_html)


def render_error_banner(message: str = "Error processing request") -> None:
    """
    Render an error banner for failed operations.
    
    Args:
        message (str): Error message to display
    """
    banner_html = f'''
    <div style="background: linear-gradient(90deg, #dc3545 0%, #c82333 100%); 
                padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h2 style="color: white; text-align: center; margin: 0;">
            ❌ {message}
        </h2>
    </div>
    '''
    
    render_html_card(banner_html)


def render_warning_banner(message: str = "Warning: Please review the information") -> None:
    """
    Render a warning banner for operations that need attention.
    
    Args:
        message (str): Warning message to display
    """
    banner_html = f'''
    <div style="background: linear-gradient(90deg, #ffc107 0%, #e0a800 100%); 
                padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h2 style="color: white; text-align: center; margin: 0;">
            ⚠️ {message}
        </h2>
    </div>
    '''
    
    render_html_card(banner_html)