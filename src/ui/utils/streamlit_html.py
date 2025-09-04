"""
Streamlit HTML Rendering Utilities

Clean, minimal HTML utilities with centralized CSS loading.
"""

import os
from typing import Optional

import streamlit as st
import streamlit.components.v1 as components


def inject_global_css():
    """Load centralized CSS once with error reporting."""
    # Absolute path to ensure correct loading
    css_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'static', 'css', 'style.css')
    
    try:
        if os.path.exists(css_path):
            with open(css_path, 'r', encoding='utf-8') as f:
                css_content = f.read()
                st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
        else:
            # Minimal professional fallback if file not found
            st.markdown("""
            <style>
            :root {
                --primary: #2563eb;
                --gray-100: #f1f5f9;
                --gray-200: #e2e8f0;
                --white: #ffffff;
            }
            .main .block-container {
                max-width: 1200px;
                padding: 2rem;
                margin: 0 auto;
            }
            .card {
                background: var(--white);
                border: 1px solid var(--gray-200);
                border-radius: 0.5rem;
                padding: 1.5rem;
                margin-bottom: 1.5rem;
                box-shadow: 0 1px 2px 0 rgb(0 0 0 / 0.05);
                min-height: 120px;
                display: flex;
                flex-direction: column;
                justify-content: center;
            }
            .app-header {
                background: var(--white);
                border: 1px solid var(--gray-200);
                border-radius: 0.75rem;
                padding: 2rem;
                margin-bottom: 2rem;
                box-shadow: 0 1px 2px 0 rgb(0 0 0 / 0.05);
            }
            </style>
            """, unsafe_allow_html=True)
    except Exception as e:
        # Log error but don't break the app
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"CSS loading failed: {str(e)}")


def render_html_card(html_content: str, height: Optional[int] = None) -> None:
    """
    Render HTML content without any CSS injection.
    
    Args:
        html_content: HTML to render
        height: Optional height for components
    """
    try:
        if hasattr(st, 'html'):
            st.html(html_content)
        else:
            if height:
                components.html(html_content, height=height)
            else:
                st.markdown(html_content, unsafe_allow_html=True)
    except Exception:
        st.markdown(html_content, unsafe_allow_html=True)


def render_metric_card(title: str, value: str, color: str = "#2563eb") -> None:
    """
    Create a clean metric card with consistent height.
    
    Args:
        title: Card title
        value: Card value
        color: Border color (default blue)
    """
    html = f'''
    <div class="card" style="border-left: 4px solid {color}; min-height: 100px; display: flex; flex-direction: column; justify-content: center; text-align: center;">
        <h4 style="color: #64748b; font-size: 0.875rem; margin: 0 0 0.5rem 0;">{title}</h4>
        <p style="color: #1e293b; font-size: 1.5rem; font-weight: 700; margin: 0;">{value}</p>
    </div>
    '''
    render_html_card(html)


def render_info_card(title: str, content: str, icon: str = "", background_color: str = "#ffffff") -> None:
    """
    Render a clean information card.
    
    Args:
        title: Card title
        content: Card content
        icon: Optional icon
        background_color: Background color
    """
    html = f'''
    <div class="card" style="background: {background_color};">
        <h4 style="color: #1e293b; margin: 0 0 0.5rem 0;">
            {icon} {title}
        </h4>
        <p style="color: #64748b; margin: 0; line-height: 1.5;">
            {content}
        </p>
    </div>
    '''
    render_html_card(html)


def render_gradient_header(title: str, subtitle: str, gradient: str = "") -> None:
    """
    Render a clean header without gradients.
    
    Args:
        title: Main title
        subtitle: Subtitle text
        gradient: Ignored - no gradients
    """
    html = f'''
    <div class="app-header">
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    '''
    render_html_card(html)


def render_verification_badge(label: str, verified: bool, icon: str = "✓", verified_color: str = "#10b981", failed_color: str = "#ef4444") -> None:
    """
    Render a verification badge.
    
    Args:
        label: Badge label
        verified: Whether verification passed
        icon: Badge icon
        verified_color: Color for verified state
        failed_color: Color for failed state
    """
    color = verified_color if verified else failed_color
    status_icon = icon if verified else "✗"
    
    html = f'''
    <div class="card" style="text-align: center; border-top: 3px solid {color};">
        <div style="font-size: 1.5rem; margin-bottom: 0.5rem; color: {color};">{status_icon}</div>
        <div style="font-weight: 600; color: #1e293b;">{label}</div>
    </div>
    '''
    render_html_card(html)