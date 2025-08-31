#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Disclaimer : This Script Was Created Using AI , The sole purpose
of this script is to generate synthetic fraudulent Arabic job postings for research and educational purposes only.
it may need further enhancements and modifications to suit specific requirements.

Enhanced Arabic Fraudulent Job Postings Generator

This script generates sophisticated synthetic fraudulent Arabic job postings with
advanced fraud patterns including experience mismatches, suspicious emails,
confidential government entities, and realistic red flags.

Key Features:
- Experience field vs description contradictions
- Personal email patterns (@gmail, @hotmail, etc.)
- Confidential government fraud patterns
- Sophisticated multi-layered fraud indicators

Author: Tuwaiq ML Bootcamp
Version: 3.0.0 - Enhanced
"""

import ast
import csv
import json
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# ============================
# ARABIC FRAUD KEYWORDS
# ============================

ARABIC_FRAUD_KEYWORDS = {
    # Unrealistic Rewards
    'unrealistic_rewards': [
        'راتب عالي جداً',
        'بدون خبرة',
        'العمل من المنزل',
        'فرصة ذهبية',
        'اكسب ٨٠٠ ريال يومياً',
        'اكسب ١٠٠٠ ريال يومياً',
        'دوام جزئي',
        'مكافآت شهرية مغرية',
        'عمولات يومية',
        'راتب ٥٠٠٠٠ ريال',
        'دخل مضمون',
        'أرباح خيالية',
        'ثروة سريعة',
        'راتب بدون عمل',
        'مال سهل'
    ],
    
    # Urgency and Pressure
    'urgency_pressure': [
        'مطلوب فورا',
        'وظيفة عاجلة',
        'بداية فورية',
        'مقابلة فورية',
        'للانضمام الفوري',
        'فرصة محدودة',
        'اتصل الآن',
        'لا تفوت الفرصة',
        'آخر موعد اليوم',
        'عجل قبل فوات الأوان',
        'مقاعد محدودة',
        'التحق فوراً'
    ],
    
    # Unprofessional Communication
    'unprofessional_comm': [
        'التواصل واتساب فقط',
        'مقابلة عبر تيليجرام',
        'انضم إلى قناتنا',
        'راسلنا على الواتساب',
        'للتواصل جيميل',
        'ايميل هوتميل',
        'ياهو للمراسلة',
        'فقط واتساب',
        'تيليجرام للتقديم',
        'لا مقابلات شخصية',
    ],
    
    # Money and Data Requests
    'money_data_requests': [
        'رسوم مسبقة',
        'رسوم رمزية',
        'رسوم معالجة الفيزا',
        'دفع مبلغ مادي',
        'صورة الهوية',
        'رقم الآيبان',
        'معلوماتك البنكية',
        'رسوم التدريب',
        'مبلغ التأمين',
        'رسوم الاشتراك',
        'دفع ضمان',
        'رسوم الدورة'
    ]
}

# ============================
# JOB TITLES AND DESCRIPTIONS
# ============================

ARABIC_JOB_TITLES = [
    'محاسب', 'بائع', 'سكرتير', 'مندوب مبيعات', 'خدمة عملاء', 'مصمم جرافيك',
    'مطور ويب', 'مترجم', 'كاتب محتوى', 'مسوق إلكتروني', 'محرر', 'مصور',
    'سائق', 'حارس أمن', 'عامل نظافة', 'مراقب كاميرات', 'مدخل بيانات',
    'استقبال', 'مساعد إداري', 'موظف مبيعات', 'مندوب توصيل', 'مشرف مبيعات',
    'مدير مكتب', 'أخصائي موارد بشرية', 'محلل مالي', 'مطور تطبيقات'
]

SAUDI_REGIONS = [
    'الرياض', 'المنطقة الشرقية', 'مكة المكرمة', 'المدينة المنورة',
    'القصيم', 'عسير', 'تبوك', 'حائل', 'الحدود الشمالية', 'جازان',
    'نجران', 'الباحة', 'الجوف'
]

SAUDI_CITIES = [
    'AR RIYADH', 'JEDDAH', 'AD DAMMAM', 'MAKKAH', 'AL MADINAH',
    'AT TAIF', 'BURAYDAH', 'TABUK', 'HAIL', 'ABHA',
    'KHOBAR', 'YANBU', 'NAJRAN', 'AL JUBAIL', 'ARAR'
]

COMPANY_TYPES = ['خاص', 'حكومي', 'غير ربحي', 'حكومي سري']
COMPANY_SIZES = ['صغيرة فئة أ', 'صغيرة فئة ب', 'متوسطة فئة أ', 'متوسطة فئة ب', 'كبيرة']
CONTRACT_TYPES = ['دوام كامل', 'دوام جزئي', 'عمل عن بعد', 'عقد مؤقت']

# Experience mismatch patterns
EXPERIENCE_MISMATCHES = [
    {"exper": "5 Years", "desc_text": "بدون خبرة مطلوبة", "title_modifier": "للخريجين الجدد"},
    {"exper": "0 Years", "desc_text": "خبرة لا تقل عن ٧ سنوات", "title_modifier": "خبرة عالية مطلوبة"},
    {"exper": "10 Years", "desc_text": "لا تشترط خبرة سابقة", "title_modifier": "بدون خبرة"},
    {"exper": "3 Years", "desc_text": "للمبتدئين فقط", "title_modifier": "مرحب بالجدد"},
    {"exper": "7 Years", "desc_text": "وظيفة للخريجين الجدد", "title_modifier": "فرصة للمبتدئين"},
    {"exper": "2 Years", "desc_text": "خبرة ١٥ سنة كحد أدنى", "title_modifier": "كبار المختصين"},
    {"exper": "12 Years", "desc_text": "مناسب للطلاب والخريجين", "title_modifier": "تدريب وتأهيل"},
    {"exper": "4 Years", "desc_text": "بدون أي متطلبات خبرة", "title_modifier": "عمل بسيط"}
]

# Suspicious email patterns
SUSPICIOUS_EMAIL_DOMAINS = ['@gmail.com', '@hotmail.com', '@yahoo.com', '@outlook.com']
EMAIL_ARABIC_REFS = [
    'للتواصل جيميل فقط',
    'راسلنا على الهوتميل', 
    'ياهو للمراسلة',
    'أرسل سيرتك على الجيميل',
    'للتقديم ايميل شخصي',
    'تواصل معنا عبر الايميل الشخصي'
]

# ============================
# FRAUD GENERATION FUNCTIONS
# ============================

class EnhancedArabicFraudGenerator:
    """Enhanced generator for creating sophisticated Arabic fraudulent job postings"""
    
    def __init__(self, legitimate_df: pd.DataFrame):
        self.legitimate_df = legitimate_df
        self.fraud_posts = []
        self.confidential_gov_count = 0
        self.experience_mismatch_count = 0
        
    def generate_suspicious_company_name(self, is_confidential_gov: bool = False) -> Dict[str, str]:
        """Generate suspicious company names with special handling for confidential government"""
        
        if is_confidential_gov:
            # Confidential government patterns (major fraud indicator)
            confidential_names = [
                'جهة حكومية سرية',
                'Confidential Government Entity', 
                'مؤسسة حكومية غير معلنة',
                'قطاع حكومي سري',
                'وزارة غير محددة',
                'هيئة حكومية مجهولة',
                'Government Agency - Confidential',
                'مؤسسة عامة سرية',
                'Confidential Government',
                'الإدارة الحكومية'
            ]
            
            return {
                "comp_name": random.choice(confidential_names),
                "comp_type": "حكومي سري",
                "comp_no": "CONFIDENTIAL"
            }
        
        # Regular suspicious patterns
        suspicious_prefixes = [
            'شركة', 'مؤسسة', 'مكتب', 'دار', 'بيت', 'مركز', 'معهد'
        ]
        
        suspicious_names = [
            'الثروة السريعة', 'المال السهل', 'النجاح المضمون', 'الأرباح الذهبية',
            'الفرص الذهبية', 'الدخل المضمون', 'النجاح الفوري', 'المكاسب السريعة',
            'العمل المنزلي', 'الربح الحلال', 'الاستثمار الآمن', 'التوظيف السريع',
            'الوظائف الذهبية', 'العمل عن بعد', 'الأعمال المنزلية', 'الربح اليومي'
        ]
        
        generic_names = [
            'الشركة العامة', 'المؤسسة الكبرى', 'الشركة الدولية', 'المؤسسة العالمية',
            'الشركة المحدودة', 'التجارة العامة', 'الأعمال المتنوعة', 'الخدمات الشاملة'
        ]
        
        if random.random() < 0.4:
            comp_name = f"{random.choice(suspicious_prefixes)} {random.choice(suspicious_names)}"
        else:
            comp_name = f"{random.choice(suspicious_prefixes)} {random.choice(generic_names)}"
        
        return {
            "comp_name": comp_name,
            "comp_type": random.choice(['خاص', 'غير ربحي']),
            "comp_no": self.generate_suspicious_contact_info()
        }
    
    
    
    
    def generate_experience_mismatch(self) -> Dict[str, str]:
        """Generate experience field vs description contradictions"""
        mismatch = random.choice(EXPERIENCE_MISMATCHES)
        self.experience_mismatch_count += 1
        return mismatch
    
    def generate_suspicious_email(self) -> str:
        """Generate personal email addresses for fraud posts"""
        username = f"job{random.randint(100, 9999)}"
        domain = random.choice(SUSPICIOUS_EMAIL_DOMAINS)
        return f"{username}{domain}"
    
    def inject_email_in_description(self, description: str, email: str) -> str:
        """Inject personal email into job description"""
        email_templates = [
            f"للتقديم أرسل سيرتك على {email}",
            f"راسلنا على {email} للتفاصيل", 
            f"تواصل معنا عبر {email}",
            f"للاستفسار {email}"
        ]
        
        # Add email reference to description
        email_ref = random.choice(email_templates)
        return f"{description} {email_ref}"
    
    def generate_suspicious_contact_info(self) -> str:
        """Generate suspicious contact information"""
        suspicious_numbers = [
            '1-999999', '2-888888', '3-777777', '1-123456', '9-999999'
        ]
        return random.choice(suspicious_numbers)
    
    def generate_fraudulent_description(self, fraud_type: str, has_exp_mismatch: bool = False, 
                                         exp_mismatch_text: str = "", has_email: bool = False, 
                                         email: str = "") -> str:
        """Generate fraudulent job description with specific fraud patterns"""
        
        base_descriptions = {
            'high_salary': [
                'نحن نقدم راتب عالي جداً يصل إلى ٥٠٠٠٠ ريال شهرياً بدون خبرة مطلوبة.',
                'فرصة ذهبية للحصول على راتب ٤٠٠٠٠ ريال مع العمل من المنزل فقط.',
                'اكسب ١٠٠٠ ريال يومياً مع مكافآت شهرية مغرية بدون أي خبرة سابقة.',
                'وظيفة براتب عالي جداً ٣٥٠٠٠ ريال شهرياً مع عمولات يومية إضافية.',
                'دخل مضمون يصل إلى ٦٠٠٠٠ ريال شهرياً من خلال دوام جزئي فقط.'
            ],
            
            'urgent': [
                'مطلوب فورا! وظيفة عاجلة للانضمام الفوري بداية فورية اليوم.',
                'فرصة محدودة! مقابلة فورية اليوم - لا تفوت هذه الفرصة الذهبية.',
                'عجل قبل فوات الأوان! آخر موعد للتقديم اليوم - اتصل الآن.',
                'مقاعد محدودة! للانضمام الفوري - بداية العمل غداً.',
                'وظيفة عاجلة! التحق فوراً - مقابلة فورية خلال ساعات.'
            ],
            
            'communication': [
                'للتقديم التواصل واتساب فقط على الرقم المرفق بدون مكالمات.',
                'مقابلة عبر تيليجرام وانضم إلى قناتنا للحصول على تفاصيل أكثر.',
                'راسلنا على الواتساب أو جيميل فقط - لا نقبل التقديم بطرق أخرى.',
                'للتواصل ايميل هوتميل أو ياهو للمراسلة - فقط واتساب للمتابعة.',
                'تيليجرام للتقديم فقط - لا مقابلات شخصية مطلوبة.'
            ],
            
            'money_request': [
                'مطلوب دفع رسوم رمزية ١٠٠ ريال كرسوم معالجة الفيزا والتدريب.',
                'يجب إحضار صورة الهوية ورقم الآيبان ومعلوماتك البنكية للتقديم.',
                'رسوم مسبقة ٥٠٠ ريال كرسوم التدريب وضمان الوظيفة مع استرداد كامل.',
                'مطلوب دفع مبلغ مادي ٢٠٠ ريال كرسوم الاشتراك في البرنامج.',
                'دفع ضمان ٣٠٠ ريال كمبلغ التأمين - يسترد بعد شهر من العمل.'
            ],
            
            'confidential_gov': [
                'وظيفة حكومية سرية براتب مرتفع جداً ومزايا مميزة.',
                'فرصة ذهبية للعمل في جهة حكومية مرموقة براتب يبدأ من ٢٠٠٠٠ ريال.',
                'وظيفة في قطاع حكومي سري - تفاصيل أكثر بعد التواصل.',
                'وظيفة حكومية ممتازة - اسم الجهة سيتم الاعلان عنه لاحقاً.',
                'وظيفة حساسة في قطاع حكومي مهم - للجاديين فقط.'
            ]
        }
        
        # Get base description
        description = random.choice(base_descriptions.get(fraud_type, base_descriptions['high_salary']))
        
        # Add experience mismatch text if present
        if has_exp_mismatch and exp_mismatch_text:
            description = f"{description} {exp_mismatch_text}"
        
        # Add email if present 
        if has_email and email:
            description = self.inject_email_in_description(description, email)
        
        return description
    
    def generate_fraudulent_tasks(self, fraud_type: str, has_exp_mismatch: bool = False, exp_text: str = "") -> str:
        """Generate fraudulent job tasks with enhanced patterns"""
        
        task_templates = {
            'high_salary': [
                "العمل من المنزل بمرونة كاملة مع راتب عالي جداً مضمون.",
                "مكافآت شهرية مغرية وعمولات يومية بدون خبرة مطلوبة.",
                "دخل مضمون مع أرباح خيالية من خلال العمل عن بعد.",
                "ثروة سريعة مع راتب بدون عمل شاق - فرصة ذهبية."
            ],
            
            'urgent': [
                "البدء فوراً - مطلوب للانضمام الفوري اليوم.",
                "مقابلة فورية مع بداية العمل خلال ٢٤ ساعة.",
                "وظيفة عاجلة تتطلب التحاقاً فورياً بدون تأخير.",
                "فرصة محدودة - اتصل الآن قبل انتهاء المقاعد."
            ],
            
            'communication': [
                "التواصل عبر واتساب فقط - لا مقابلات شخصية.",
                "مقابلة عبر تيليجرام وانضم للقناة للتفاصيل.",
                "راسل على جيميل أو هوتميل - واتساب للمتابعة الفورية.",
                "تيليجرام للتقديم - لا حاجة لزيارة المكتب."
            ],
            
            'money_request': [
                "دفع رسوم رمزية للتدريب وضمان الوظيفة مع الاسترداد.",
                "إحضار الهوية والآيبان والمعلومات البنكية للتقديم.",
                "رسوم معالجة الفيزا ورسوم الاشتراك مطلوبة مسبقاً.",
                "مبلغ التأمين مطلوب كضمان - يسترد بعد التوظيف."
            ],
            
            'confidential_gov': [
                "عمل في بيئة حكومية متميزة مع مزايا استثنائية.",
                "مهام حساسة ومهمة في قطاع حيوي.",
                "راتب مرتفع وتأمين شامل ومعاش تقاعدي.",
                "عمل في بيئة سرية - تفاصيل أكثر بعد التوظيف."
            ]
        }
        
        # Get base tasks
        tasks = task_templates.get(fraud_type, task_templates['high_salary'])
        selected_tasks = random.sample(tasks, min(3, len(tasks)))
        
        # Add experience mismatch text if present
        if has_exp_mismatch and exp_text:
            selected_tasks.append(f"ملاحظة: {exp_text}")
        
        # Format as list similar to legitimate data
        formatted_tasks = [f"   {task}" for task in selected_tasks]
        formatted_tasks.extend(['  ', '  '])  # Add empty entries like in original data
        
        return str(formatted_tasks)
    
    def generate_suspicious_benefits(self, fraud_type: str) -> str:
        """Generate suspicious benefits"""
        if fraud_type == 'high_salary':
            salary = random.choice(['25000.0', '35000.0', '50000.0', '75000.0', '100000.0'])
        elif fraud_type == 'confidential_gov':
            salary = random.choice(['20000.0', '30000.0', '40000.0', '50000.0'])
        else:
            # Still high but less extreme
            salary = random.choice(['15000.0', '20000.0', '25000.0', '30000.0'])
        
        return f"['Salary', '{salary}']"
    
    def generate_fraudulent_post(self, fraud_type: str, job_index: int) -> Dict[str, Any]:
        """Generate a complete fraudulent job posting with enhanced patterns"""
        
        # Determine if this is a confidential government post (40% chance)
        is_confidential_gov = (fraud_type == 'confidential_gov' or random.random() < 0.4)
        
        # Determine if this post has experience mismatch (60% chance)
        has_exp_mismatch = random.random() < 0.6
        
        # Determine if this post has suspicious email (50% chance) 
        has_email = random.random() < 0.5
        
        # Generate experience mismatch if applicable
        exp_mismatch = None
        if has_exp_mismatch:
            exp_mismatch = self.generate_experience_mismatch()
        
        # Generate suspicious email if applicable
        email = None
        if has_email:
            email = self.generate_suspicious_email()
        
        # Select base job title and add suspicious elements
        base_title = random.choice(ARABIC_JOB_TITLES)
        
        if fraud_type in ['urgent', 'high_salary'] or (exp_mismatch and exp_mismatch['title_modifier']):
            if random.random() < 0.3:
                if exp_mismatch and exp_mismatch['title_modifier']:
                    job_title = f"{exp_mismatch['title_modifier']} - {base_title}"
                else:
                    title_modifiers = ['مطلوب فورا', 'وظيفة عاجلة', 'فرصة ذهبية']
                    job_title = f"{random.choice(title_modifiers)} - {base_title}"
            else:
                job_title = base_title
        else:
            job_title = base_title
        
        # Generate company information
        company_info = self.generate_suspicious_company_name(is_confidential_gov)
        if is_confidential_gov:
            self.confidential_gov_count += 1
        
        # Generate fraudulent post
        fraud_post = {
            'job_title': job_title,
            'job_date': random.choice(['27/05/1444', '28/05/1444', '29/05/1444', '30/05/1444']),
            'job_desc': f"['{self.generate_fraudulent_description(fraud_type, has_exp_mismatch, exp_mismatch['desc_text'] if exp_mismatch else '', has_email, email)}']",
            'job_tasks': self.generate_fraudulent_tasks(fraud_type, has_exp_mismatch, exp_mismatch['desc_text'] if exp_mismatch else ''),
            'comp_name': company_info['comp_name'],
            'comp_no': company_info['comp_no'],
            'comp_type': company_info['comp_type'],
            'comp_size': random.choice(COMPANY_SIZES),
            'eco_activity': 'nan',  # Often missing in fraudulent posts
            'qualif': 'nan',  # Often vague or missing
            'region': random.choice(SAUDI_REGIONS),
            'city': random.choice(SAUDI_CITIES) + '...',
            'benefits': self.generate_suspicious_benefits(fraud_type),
            'contract': random.choice(CONTRACT_TYPES),
            'positions': f"0 / {random.randint(1, 50)}",  # Often inflated numbers
            'job_post_id': f"FRAUD{20220000000000 + job_index}",
            'exper': exp_mismatch['exper'] if exp_mismatch else random.choice(['0 Years', '1 Years', '2 Years']),
            'gender': random.choice(['both', 'M', 'F']),
            'fraudulent': 1  # Mark as fraudulent
        }
        
        return fraud_post
    
    def generate_fraud_dataset(self, num_fraud_posts: int = 530) -> List[Dict[str, Any]]:
        """Generate the complete enhanced fraudulent dataset"""
        
        # Enhanced fraud type distribution with confidential government as primary pattern
        fraud_types = ['confidential_gov', 'high_salary', 'urgent', 'communication', 'money_request']
        type_distribution = [0.4, 0.2, 0.15, 0.15, 0.1]  # 40%, 20%, 15%, 15%, 10%
        
        fraud_posts = []
        
        for i in range(num_fraud_posts):
            # Select fraud type based on distribution
            fraud_type = np.random.choice(fraud_types, p=type_distribution)
            
            # Generate fraudulent post
            fraud_post = self.generate_fraudulent_post(fraud_type, i)
            fraud_posts.append(fraud_post)
        
        return fraud_posts


def integrate_fraud_with_legitimate(legitimate_df: pd.DataFrame, fraud_posts: List[Dict[str, Any]]) -> pd.DataFrame:
    """Integrate fraudulent posts with legitimate data randomly"""
    
    print("Integrating fraudulent posts with legitimate data...")
    
    # Add fraudulent column to legitimate data (all 0s)
    legitimate_df['fraudulent'] = 0
    
    # Convert fraud posts to DataFrame
    fraud_df = pd.DataFrame(fraud_posts)
    
    # Combine datasets
    combined_df = pd.concat([legitimate_df, fraud_df], ignore_index=True)
    
    # Shuffle the entire dataset to randomly distribute fraud posts
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return combined_df


def validate_dataset_quality(df: pd.DataFrame, fraud_generator: EnhancedArabicFraudGenerator) -> Dict[str, Any]:
    """Validate the quality and structure of the enhanced generated dataset"""
    
    fraud_posts = df[df['fraudulent'] == 1]
    
    # Count experience mismatches
    exp_mismatch_patterns = ['بدون خبرة', 'لا تشترط خبرة', 'للخريجين الجدد', 'للمبتدئين']
    exp_mismatches = 0
    for _, row in fraud_posts.iterrows():
        desc_text = str(row['job_desc']).lower()
        if any(pattern in desc_text for pattern in exp_mismatch_patterns):
            if row['exper'] not in ['0 Years', '1 Years']:
                exp_mismatches += 1
    
    # Count suspicious emails
    email_patterns = ['@gmail', '@hotmail', '@yahoo', '@outlook', 'جيميل', 'هوتميل']
    email_frauds = 0
    for _, row in fraud_posts.iterrows():
        desc_text = str(row['job_desc']).lower()
        if any(pattern in desc_text for pattern in email_patterns):
            email_frauds += 1
    
    # Count confidential government posts
    conf_gov_count = len(fraud_posts[fraud_posts['comp_type'] == 'حكومي سري'])
    
    validation_results = {
        'total_rows': len(df),
        'legitimate_count': len(df[df['fraudulent'] == 0]),
        'fraudulent_count': len(df[df['fraudulent'] == 1]),
        'fraud_percentage': len(df[df['fraudulent'] == 1]) / len(df) * 100,
        'missing_values': df.isnull().sum().to_dict(),
        'columns': list(df.columns),
        'sample_fraud_titles': fraud_posts['job_title'].head(10).tolist(),
        'fraud_patterns': {
            'confidential_government_count': conf_gov_count,
            'confidential_government_percentage': (conf_gov_count / len(fraud_posts)) * 100,
            'experience_mismatches': exp_mismatches,
            'experience_mismatch_percentage': (exp_mismatches / len(fraud_posts)) * 100,
            'suspicious_emails': email_frauds,
            'suspicious_email_percentage': (email_frauds / len(fraud_posts)) * 100,
            'total_experience_mismatches_generated': fraud_generator.experience_mismatch_count,
            'total_confidential_gov_generated': fraud_generator.confidential_gov_count
        }
    }
    
    return validation_results


def main():
    """Main execution function"""
    
    print("🚀 Starting Arabic Fraudulent Job Postings Generator")
    print("=" * 60)
    
    # Load legitimate data
    print("📂 Loading legitimate job postings...")
    try:
        legitimate_df = pd.read_csv('data/raw/Jadarat_data.csv', encoding='utf-8-sig')
        print(f"✅ Loaded {len(legitimate_df)} legitimate job postings")
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return
    
    # Initialize enhanced fraud generator
    print("\n🔧 Initializing enhanced fraud generator...")
    fraud_generator = EnhancedArabicFraudGenerator(legitimate_df)
    
    # Generate fraudulent posts
    print("🎭 Generating 530 fraudulent job postings...")
    fraud_posts = fraud_generator.generate_fraud_dataset(530)
    print(f"✅ Generated {len(fraud_posts)} fraudulent posts")
    
    # Display fraud distribution
    fraud_types_count = {}
    for post in fraud_posts:
        # Determine fraud type based on content
        desc = post['job_desc'].lower()
        if 'راتب عالي' in desc or '50000' in post['benefits']:
            fraud_type = 'high_salary'
        elif 'مطلوب فورا' in desc or 'عاجل' in desc:
            fraud_type = 'urgent'
        elif 'واتساب' in desc or 'تيليجرام' in desc:
            fraud_type = 'communication'
        elif 'رسوم' in desc or 'دفع' in desc:
            fraud_type = 'money_request'
        else:
            fraud_type = 'mixed'
        
        fraud_types_count[fraud_type] = fraud_types_count.get(fraud_type, 0) + 1
    
    print("\n📊 Fraud Types Distribution:")
    for fraud_type, count in fraud_types_count.items():
        percentage = (count / len(fraud_posts)) * 100
        print(f"   {fraud_type}: {count} posts ({percentage:.1f}%)")
    
    # Integrate with legitimate data
    print("\n🔄 Integrating fraud posts with legitimate data...")
    combined_df = integrate_fraud_with_legitimate(legitimate_df, fraud_posts)
    print(f"✅ Created combined dataset with {len(combined_df)} total posts")
    
    # Validate dataset
    print("\n🔍 Validating dataset quality...")
    validation_results = validate_dataset_quality(combined_df, fraud_generator)
    
    print("\n📈 Dataset Statistics:")
    print(f"   Total rows: {validation_results['total_rows']:,}")
    print(f"   Legitimate posts: {validation_results['legitimate_count']:,}")
    print(f"   Fraudulent posts: {validation_results['fraudulent_count']:,}")
    print(f"   Fraud percentage: {validation_results['fraud_percentage']:.1f}%")
    
    print("\n🚨 Enhanced Fraud Patterns:")
    patterns = validation_results['fraud_patterns']
    print(f"   Confidential Government: {patterns['confidential_government_count']} ({patterns['confidential_government_percentage']:.1f}%)")
    print(f"   Experience Mismatches: {patterns['experience_mismatches']} ({patterns['experience_mismatch_percentage']:.1f}%)")
    print(f"   Suspicious Emails: {patterns['suspicious_emails']} ({patterns['suspicious_email_percentage']:.1f}%)")
    
    # Save combined dataset
    output_file = 'data/processed/arabic_job_postings_with_fraud.csv'
    print(f"\n💾 Saving combined dataset to {output_file}...")
    
    try:
        combined_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print("✅ Dataset saved successfully!")
        
        # Save validation report
        validation_file = 'data/processed/fraud_generation_report.json'
        with open(validation_file, 'w', encoding='utf-8') as f:
            json.dump(validation_results, f, ensure_ascii=False, indent=2)
        print(f"✅ Validation report saved to {validation_file}")
        
    except Exception as e:
        print(f"❌ Error saving dataset: {str(e)}")
        return
    
    # Display sample fraudulent posts
    print("\n🎭 Sample Fraudulent Job Titles:")
    fraud_sample = combined_df[combined_df['fraudulent'] == 1]['job_title'].head(10)
    for i, title in enumerate(fraud_sample, 1):
        print(f"   {i}. {title}")
    
    print("\n🎉 Arabic Fraudulent Job Postings Generation Complete!")
    print("=" * 60)
    
    return combined_df


if __name__ == "__main__":
    # Create necessary directories
    import os
    os.makedirs('data/processed', exist_ok=True)
    
    # Run the main function
    result_df = main()
    
    if result_df is not None:
        print(f"\n📋 Final Dataset Shape: {result_df.shape}")
        print("🔗 You can now use this dataset for training fraud detection models!")