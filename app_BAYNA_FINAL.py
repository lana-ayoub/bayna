
import os
import re
import numpy as np
import pandas as pd
import streamlit as st

import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from transformers import pipeline

# ============================================
# PAGE SETTINGS
# ============================================

st.set_page_config(
    page_title="بينه",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
    <style>
    :root {
        --bunya-navy: #17324D;
        --bunya-green: #2D7A5E;
        --bunya-soft: #F6F8FA;
        --bunya-border: #E2E8F0;
        --bunya-text: #263746;
        --bunya-muted: #6B7C8C;
    }

    html, body, [class*="css"] {
        direction: rtl;
        text-align: right;
        font-family: "Segoe UI", Tahoma, Arial, sans-serif;
    }

    .stApp {
        background: var(--bunya-soft);
    }

    .block-container {
        max-width: 1180px;
        padding-top: 1.4rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: var(--bunya-navy);
        letter-spacing: -0.2px;
    }

    p, label, .stCaption {
        color: var(--bunya-text);
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid var(--bunya-border);
        border-radius: 14px;
        padding: 14px 16px;
        box-shadow: 0 4px 14px rgba(23, 50, 77, 0.04);
    }

    div[data-testid="stMetricLabel"] {
        color: var(--bunya-muted);
    }

    div[data-testid="stMetricValue"] {
        color: var(--bunya-navy);
        font-weight: 700;
    }

    .stButton > button {
        border-radius: 10px;
        min-height: 44px;
        font-weight: 650;
        border: 1px solid #D7E0E7;
        background: white;
        color: var(--bunya-navy);
        transition: 0.15s ease-in-out;
    }

    .stButton > button:hover {
        border-color: var(--bunya-green);
        color: var(--bunya-green);
        box-shadow: 0 4px 14px rgba(45, 122, 94, 0.10);
    }

    button[kind="primary"] {
        background: var(--bunya-green) !important;
        color: white !important;
        border: none !important;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    textarea {
        border-radius: 10px !important;
    }

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid var(--bunya-border);
    }

    div[role="radiogroup"] {
        gap: 0.45rem;
        background: white;
        border: 1px solid var(--bunya-border);
        padding: 0.4rem;
        border-radius: 12px;
        margin-bottom: 1rem;
    }

    div[role="radiogroup"] label {
        background: #F8FAFC;
        border-radius: 9px;
        padding: 0.35rem 0.6rem;
    }

    .bunya-hero {
        background: white;
        border: 1px solid var(--bunya-border);
        border-radius: 20px;
        padding: 34px 28px 26px;
        text-align: center;
        margin-bottom: 22px;
        box-shadow: 0 8px 26px rgba(23, 50, 77, 0.05);
    }

    .bunya-hero h1 {
        margin: 0;
        font-size: 2.4rem;
    }

    .bunya-hero p {
        margin: 8px 0 0;
        color: var(--bunya-muted);
        font-size: 1.02rem;
    }

    .bunya-section-note {
        background: white;
        border: 1px solid var(--bunya-border);
        border-right: 4px solid var(--bunya-green);
        border-radius: 12px;
        padding: 13px 15px;
        margin: 8px 0 16px;
    }

    #MainMenu, footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================
# BUNYA VISUAL SYSTEM V2
# ============================================

st.markdown(
    """
    <style>
    :root {
        --bunya-green: #0F7A4A;
        --bunya-green-dark: #075F3A;
        --bunya-lime: #79C84B;
        --bunya-green-soft: #EEF8F2;
        --bunya-red: #C63D45;
        --bunya-red-soft: #FFF1F2;
        --bunya-amber: #C78517;
        --bunya-amber-soft: #FFF8E8;
        --bunya-blue: #2D6F98;
        --bunya-blue-soft: #EFF7FC;
        --bunya-ink: #17352A;
        --bunya-muted: #64756E;
        --bunya-border: #DDE8E1;
        --bunya-surface: #FFFFFF;
        --bunya-bg: #F5F8F6;
    }

    .stApp {
        background:
            radial-gradient(circle at 85% 0%, rgba(121, 200, 75, 0.11), transparent 28rem),
            radial-gradient(circle at 0% 25%, rgba(15, 122, 74, 0.07), transparent 24rem),
            var(--bunya-bg);
    }

    .block-container {
        max-width: 1200px;
        padding-top: 1.1rem;
    }

    h1, h2, h3, h4, h5 {
        color: var(--bunya-ink);
    }

    .bunya-landing-shell {
        max-width: 800px;
        margin: 3vh auto 0;
    }

    .bunya-landing {
        text-align: center;
        margin-top: -10px;
        margin-bottom: 26px;
    }

    .bunya-brand {
        color: var(--bunya-green-dark);
        font-size: clamp(3rem, 7vw, 5.2rem);
        font-weight: 850;
        line-height: 1;
        letter-spacing: -1.5px;
        margin-bottom: 16px;
    }

    .bunya-tagline {
        color: var(--bunya-ink);
        font-size: 1.35rem;
        font-weight: 750;
        line-height: 1.8;
    }

    .bunya-subtagline {
        color: var(--bunya-muted);
        font-size: 1rem;
        margin-top: 8px;
    }

    .bunya-choice-header {
        text-align: center;
        max-width: 760px;
        margin: 18px auto 30px;
    }

    .bunya-choice-header h1 {
        margin: 6px 0 8px;
        font-size: 2.15rem;
    }

    .bunya-choice-header p {
        color: var(--bunya-muted);
        margin: 0;
    }

    .bunya-eyebrow,
    .bunya-card-kicker {
        color: var(--bunya-green);
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.5px;
    }

    .bunya-choice-card {
        min-height: 235px;
        background: rgba(255,255,255,0.96);
        border: 1px solid var(--bunya-border);
        border-radius: 20px;
        padding: 28px;
        box-shadow: 0 12px 34px rgba(24, 72, 50, 0.08);
        position: relative;
        overflow: hidden;
        margin-bottom: 12px;
    }

    .bunya-choice-card::before {
        content: "";
        position: absolute;
        top: 0;
        right: 0;
        width: 100%;
        height: 5px;
        background: linear-gradient(90deg, var(--bunya-green), var(--bunya-lime));
    }

    .bunya-choice-card h3 {
        font-size: 1.55rem;
        margin: 10px 0 12px;
    }

    .bunya-choice-card p {
        color: var(--bunya-muted);
        line-height: 1.85;
        min-height: 74px;
    }

    .bunya-card-line {
        height: 1px;
        background: var(--bunya-border);
        margin: 18px 0 14px;
    }

    .bunya-choice-card span {
        color: var(--bunya-green-dark);
        font-weight: 700;
        font-size: 0.92rem;
    }

    .stButton > button {
        border-radius: 12px;
        min-height: 48px;
        font-weight: 750;
        border: 1px solid #CFE0D6;
        background: #FFFFFF;
        color: var(--bunya-green-dark);
        box-shadow: 0 4px 12px rgba(17, 92, 57, 0.04);
    }

    .stButton > button:hover {
        border-color: var(--bunya-green);
        color: var(--bunya-green-dark);
        background: var(--bunya-green-soft);
        box-shadow: 0 8px 20px rgba(17, 92, 57, 0.10);
    }

    button[kind="primary"] {
        background: linear-gradient(135deg, var(--bunya-green-dark), var(--bunya-green)) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 9px 22px rgba(15, 122, 74, 0.20) !important;
    }

    button[kind="primary"]:hover {
        transform: translateY(-1px);
        filter: brightness(1.03);
    }

    div[role="radiogroup"] {
        border: 1px solid var(--bunya-border);
        background: rgba(255,255,255,0.95);
        border-radius: 14px;
        padding: 0.45rem;
        box-shadow: 0 5px 18px rgba(24, 72, 50, 0.04);
    }

    div[role="radiogroup"] label {
        border-radius: 10px;
        transition: .15s ease;
    }

    div[role="radiogroup"] label:has(input:checked) {
        background: var(--bunya-green-soft) !important;
        color: var(--bunya-green-dark) !important;
        font-weight: 800;
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.97);
        border: 1px solid var(--bunya-border);
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(24, 72, 50, 0.06);
    }

    /* Positive = green */
    div[data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]),
    div[data-testid="stAlert"].st-emotion-cache-1xarl3l {
        border-right: 5px solid var(--bunya-green) !important;
    }

    /* Streamlit status colors - stronger visual signal */
    div[data-testid="stAlert"] {
        border-radius: 14px;
        box-shadow: 0 5px 16px rgba(20, 50, 35, 0.04);
        font-weight: 600;
    }

    div[data-testid="stNotificationContentSuccess"],
    div[data-testid="stAlertContentSuccess"] {
        color: var(--bunya-green-dark) !important;
    }

    div[data-testid="stNotificationContentError"],
    div[data-testid="stAlertContentError"] {
        color: var(--bunya-red) !important;
    }

    /* Metric deltas: positive green, negative red */
    div[data-testid="stMetricDelta"] svg {
        stroke-width: 2.5;
    }

    div[data-testid="stMetricDelta"]:has(svg[fill="currentColor"]) {
        font-weight: 800;
    }

    .bunya-section-note {
        background: #FFFFFF;
        border: 1px solid var(--bunya-border);
        border-right: 5px solid var(--bunya-green);
        border-radius: 14px;
        box-shadow: 0 5px 16px rgba(24, 72, 50, 0.04);
    }

    div[data-testid="stDataFrame"] {
        box-shadow: 0 6px 20px rgba(24, 72, 50, 0.05);
    }

    @media (max-width: 700px) {
        .bunya-landing-shell {
            margin-top: 0;
        }

        .bunya-brand {
            font-size: 3rem;
        }

        .bunya-tagline {
            font-size: 1.12rem;
        }

        .bunya-choice-card {
            min-height: auto;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================
# DATA
# ============================================

DATA_FILE = "SME_AI_Master_Data_v4_Location_Scoring.xlsx"

if not os.path.exists(DATA_FILE):
    st.error(
        "ملف البيانات الرئيسي غير موجود في جلسة Colab."
    )
    st.stop()

governorates = pd.read_excel(
    DATA_FILE,
    sheet_name="Governorate Master"
)

table4 = pd.read_excel(
    DATA_FILE,
    sheet_name="Table4 Structured"
)

demand = pd.read_excel(
    DATA_FILE,
    sheet_name="Demand Proxy 2017",
    usecols="A:P",
    nrows=28
)


# ============================================
# HELPERS
# ============================================

def normalize_isic4(value):

    if pd.isna(value):
        return None

    code = str(value).strip()

    if code.endswith(".0"):
        code = code[:-2]

    if code.isdigit():
        code = code.zfill(4)

    return code


def normalize(series, reverse=False):

    series = pd.to_numeric(
        series,
        errors="coerce"
    )

    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:
        return pd.Series(
            [50] * len(series),
            index=series.index
        )

    score = (
        (series - minimum)
        / (maximum - minimum)
        * 100
    )

    if reverse:
        score = 100 - score

    return score


table4["ISIC Code"] = (
    table4["ISIC Code"]
    .apply(normalize_isic4)
)

activities = table4[
    table4["Level"] == "Detailed activity (4-digit)"
].copy()


governorate_map = {
    "عمان": "Amman",
    "البلقاء": "Balqa",
    "الزرقاء": "Zarqa",
    "مادبا": "Madaba",
    "إربد": "Irbid",
    "المفرق": "Mafraq",
    "جرش": "Jarash",
    "عجلون": "Ajlun",
    "الكرك": "Karak",
    "الطفيلة": "Tafiela",
    "معان": "Ma'an",
    "العقبة": "Aqaba"
}


# ============================================
# LOCATION ANALYSIS
# ============================================

def find_clothing_demand_category():

    categories = demand["Category (EN)"].dropna()

    matches = categories[
        categories.astype(str)
        .str.lower()
        .str.contains("clothing|footwear")
    ]

    if not matches.empty:
        return matches.iloc[0]

    return None


def calculate_location_scores(
    isic_code,
    demand_category=None
):

    business = activities[
        activities["ISIC Code"] == isic_code
    ]

    if business.empty:
        raise ValueError(
            "لم يتم العثور على النشاط في بيانات المنشآت."
        )

    business = business.iloc[0]

    demand_row = None

    if demand_category is not None:

        match = demand[
            demand["Category (EN)"]
            == demand_category
        ]

        if not match.empty:
            demand_row = match.iloc[0]

    rows = []

    for arabic_name, english_name in governorate_map.items():

        gov_match = governorates[
            governorates["Governorate"]
            == english_name
        ]

        if gov_match.empty:
            continue

        gov_row = gov_match.iloc[0]

        population = float(
            gov_row["Population 2025"]
        )

        registration_growth = float(
            gov_row["H1 Growth 2023→2024"]
        )

        similar_businesses = pd.to_numeric(
            business.get(
                english_name,
                0
            ),
            errors="coerce"
        )

        if pd.isna(similar_businesses):
            similar_businesses = 0

        similar_businesses = float(
            similar_businesses
        )

        businesses_per_10k = (
            similar_businesses
            / population
            * 10000
        )

        if demand_row is not None:

            historical_demand = pd.to_numeric(
                demand_row.get(
                    arabic_name,
                    np.nan
                ),
                errors="coerce"
            )

        else:

            historical_demand = np.nan

        rows.append({
            "Governorate": english_name,
            "Population": population,
            "Historical Demand": historical_demand,
            "Similar Businesses": similar_businesses,
            "Businesses per 10k": businesses_per_10k,
            "Registration Growth": registration_growth
        })


    result = pd.DataFrame(rows)

    # حجم السوق
    result["Log Population"] = np.log1p(
        result["Population"]
    )

    result["Market Score"] = normalize(
        result["Log Population"]
    )

    # الطلب التاريخي
    if result["Historical Demand"].notna().any():

        result["Demand Score"] = normalize(
            result["Historical Demand"]
        )

    else:

        result["Demand Score"] = np.nan

    # البيع بالتجزئة:
    # كثافة أقل للأنشطة المشابهة = فرصة تنافسية أعلى نسبيًا
    result["Sector Score"] = normalize(
        result["Businesses per 10k"],
        reverse=True
    )

    # اتجاه التسجيل
    result["Trend Score"] = normalize(
        result["Registration Growth"]
    )

    components = [
        "Market Score",
        "Trend Score",
        "Sector Score"
    ]

    if result["Demand Score"].notna().any():
        components.append(
            "Demand Score"
        )

    result["Location Suitability Score"] = (
        result[components]
        .mean(axis=1)
    )

    result = result.sort_values(
        "Location Suitability Score",
        ascending=False
    ).reset_index(drop=True)

    result["Rank"] = result.index + 1

    return result, components


# ============================================
# CAPITAL PLANNER
# ============================================

def render_capital_planner(project):

    st.header(" تخطيط رأس المال")

    st.caption(
        "هذا تقدير تخطيطي مبني على التكاليف التي تدخلها، "
        "وليس تقديرًا سوقيًا مؤكدًا لرأس المال المطلوب."
    )

    st.subheader("تكاليف التأسيس")

    col1, col2 = st.columns(2)

    with col1:

        equipment = st.number_input(
            "معدات وأجهزة (JD)",
            min_value=0.0,
            value=0.0,
            step=50.0,
            key="cap_equipment"
        )

        inventory = st.number_input(
            "مخزون أولي (JD)",
            min_value=0.0,
            value=0.0,
            step=50.0,
            key="cap_inventory"
        )

        premises = st.number_input(
            "موقع / إيجار / تأمين أولي (JD)",
            min_value=0.0,
            value=0.0,
            step=50.0,
            key="cap_premises"
        )

        setup = st.number_input(
            "تجهيز وديكور (JD)",
            min_value=0.0,
            value=0.0,
            step=50.0,
            key="cap_setup"
        )

        licensing = st.number_input(
            "ترخيص وإجراءات قانونية (JD)",
            min_value=0.0,
            value=0.0,
            step=20.0,
            key="cap_licensing"
        )

    with col2:

        packaging = st.number_input(
            "تغليف (JD)",
            min_value=0.0,
            value=0.0,
            step=20.0,
            key="cap_packaging"
        )

        delivery_setup = st.number_input(
            "تجهيز التوصيل (JD)",
            min_value=0.0,
            value=0.0,
            step=20.0,
            key="cap_delivery"
        )

        launch_marketing = st.number_input(
            "تسويق الإطلاق (JD)",
            min_value=0.0,
            value=0.0,
            step=20.0,
            key="cap_launch_marketing"
        )

        technology = st.number_input(
            "تقنية / أنظمة / برامج (JD)",
            min_value=0.0,
            value=0.0,
            step=20.0,
            key="cap_technology"
        )

        other_startup = st.number_input(
            "تكاليف تأسيس أخرى (JD)",
            min_value=0.0,
            value=0.0,
            step=20.0,
            key="cap_other_startup"
        )


    st.subheader("التكاليف الشهرية")

    col1, col2 = st.columns(2)

    with col1:

        salaries = st.number_input(
            "رواتب شهرية (JD)",
            min_value=0.0,
            value=0.0,
            step=50.0,
            key="monthly_salaries"
        )

        rent = st.number_input(
            "إيجار شهري (JD)",
            min_value=0.0,
            value=0.0,
            step=20.0,
            key="monthly_rent"
        )

        utilities = st.number_input(
            "مياه / كهرباء / خدمات (JD)",
            min_value=0.0,
            value=0.0,
            step=10.0,
            key="monthly_utilities"
        )

    with col2:

        marketing = st.number_input(
            "تسويق شهري (JD)",
            min_value=0.0,
            value=0.0,
            step=20.0,
            key="monthly_marketing"
        )

        logistics = st.number_input(
            "نقل وتوصيل شهري (JD)",
            min_value=0.0,
            value=0.0,
            step=20.0,
            key="monthly_logistics"
        )

        other_monthly = st.number_input(
            "مصاريف شهرية أخرى (JD)",
            min_value=0.0,
            value=0.0,
            step=20.0,
            key="monthly_other"
        )


    st.subheader("افتراضات التخطيط")

    c1, c2 = st.columns(2)

    with c1:

        months = st.number_input(
            "عدد أشهر رأس المال التشغيلي",
            min_value=0,
            value=3,
            step=1,
            key="working_months"
        )

    with c2:

        reserve_percent = st.number_input(
            "نسبة الاحتياطي %",
            min_value=0.0,
            value=10.0,
            step=1.0,
            key="reserve_percent"
        )

    st.caption(
        "القيم الافتراضية 3 أشهر و10% احتياطي هي افتراضات "
        "قابلة للتعديل وليست قواعد مالية ثابتة."
    )


    if st.button(
        "احسب رأس المال ",
        use_container_width=True
    ):

        startup_cost = sum([
            equipment,
            inventory,
            premises,
            setup,
            licensing,
            packaging,
            delivery_setup,
            launch_marketing,
            technology,
            other_startup
        ])

        monthly_cost = sum([
            salaries,
            rent,
            utilities,
            marketing,
            logistics,
            other_monthly
        ])

        operating_capital = (
            monthly_cost * months
        )

        cumulative = (
            startup_cost
            + operating_capital
        )

        reserve_amount = (
            cumulative
            * reserve_percent
            / 100
        )

        total_required = (
            cumulative
            + reserve_amount
        )

        available = project.get(
            "Available Capital"
        )

        st.session_state.capital_result = {
            "startup": startup_cost,
            "monthly": monthly_cost,
            "operating": operating_capital,
            "reserve": reserve_amount,
            "total": total_required,
            "available": available
        }


    result = st.session_state.get(
        "capital_result"
    )

    if result:

        if result["total"] <= 0:

            st.warning(
                "أدخل بعض التكاليف أولًا حتى يتم حساب "
                "رأس المال التخطيطي."
            )

        else:

            st.subheader("نتيجة التخطيط")

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "تكاليف الإطلاق",
                f"{result['startup']:,.0f} JD"
            )

            c2.metric(
                "التكلفة الشهرية",
                f"{result['monthly']:,.0f} JD"
            )

            c3.metric(
                "رأس المال التشغيلي",
                f"{result['operating']:,.0f} JD"
            )

            c1, c2 = st.columns(2)

            c1.metric(
                "الاحتياطي",
                f"{result['reserve']:,.0f} JD"
            )

            c2.metric(
                "إجمالي رأس المال التخطيطي",
                f"{result['total']:,.0f} JD"
            )


            if (
                result["available"] is not None
                and result["available"] > 0
            ):

                difference = (
                    result["available"]
                    - result["total"]
                )

                st.subheader(
                    "مقارنة برأس المال المتوفر"
                )

                if difference >= 0:

                    st.success(
                        f"رأس المال المتوفر أعلى من التقدير "
                        f"التخطيطي بحوالي "
                        f"{difference:,.0f} JD."
                    )

                else:

                    st.warning(
                        f"يوجد فرق تخطيطي قدره "
                        f"{abs(difference):,.0f} JD "
                        f"بين رأس المال المتوفر والتقدير الحالي."
                    )



# ============================================
# PRE-LAUNCH RISK ANALYSIS
# ============================================

def render_prelaunch_risks(project):

    st.header(" تحليل مخاطر ما قبل الإطلاق")

    st.caption(
        "يعتمد هذا التحليل على قواعد Screening واضحة "
        "ومدخلات المشروع الحالية، ولا يمثل احتمالًا إحصائيًا للفشل أو النجاح."
    )

    # التفاصيل متاحة في Premium
    if project.get("Plan") != "Premium":

        st.info(
            " تحليل المخاطر التفصيلي متاح ضمن Premium."
        )

        return


    high_priority = []
    watch_points = []
    data_gaps = []
    positive_factors = []


    # ========================================
    # CAPITAL
    # ========================================

    capital = st.session_state.get(
        "capital_result"
    )

    available = project.get(
        "Available Capital"
    )

    required = None

    if capital and capital.get("total", 0) > 0:

        required = float(
            capital["total"]
        )


    if (
        available is not None
        and required is not None
    ):

        gap = (
            float(available)
            - required
        )

        if gap < 0:

            high_priority.append(
                f"رأس المال المتوفر أقل من التقدير "
                f"التخطيطي الحالي بحوالي "
                f"{abs(gap):,.0f} JD."
            )

        else:

            positive_factors.append(
                "رأس المال المتوفر يغطي التقدير "
                "التخطيطي الحالي لرأس المال."
            )

    elif available is None:

        data_gaps.append(
            "لم يتم تحديد رأس المال المتوفر، "
            "لذلك لا يمكن تقييم فجوة التمويل."
        )

    elif required is None:

        data_gaps.append(
            "لم يتم حساب خطة رأس المال بعد، "
            "لذلك لا يمكن تقييم فجوة التمويل."
        )


    # ========================================
    # LOCATION
    # ========================================

    location = st.session_state.get(
        "location_summary"
    )

    if location:

        rank = int(
            location["Rank"]
        )

        total_governorates = int(
            location.get(
                "Total Governorates",
                12
            )
        )

        if rank <= 3:

            positive_factors.append(
                f"المحافظة المختارة ضمن أعلى "
                f"{rank} نتائج في Screening الموقع."
            )

        elif rank >= 8:

            watch_points.append(
                f"ترتيب المحافظة المختارة "
                f"{rank} من {total_governorates} "
                f"في Screening الموقع."
            )

            if not project.get(
                "Can Relocate",
                False
            ):

                high_priority.append(
                    "الموقع المختار منخفض نسبيًا "
                    "في Screening الموقع، "
                    "ولا توجد مرونة حالية لتغييره."
                )


        # ====================================
        # DEMAND EVIDENCE
        # ====================================

        if not location.get(
            "Demand Category"
        ):

            data_gaps.append(
                "لا يوجد مؤشر طلب مباشر موثوق "
                "مرتبط بهذا النشاط ضمن البيانات الحالية."
            )

    else:

        data_gaps.append(
            "تحليل الموقع غير متوفر حاليًا."
        )


    # ========================================
    # OPERATING MODEL
    # ========================================

    operating_model = project.get(
        "Operating Model"
    )

    if operating_model in [
        "home",
        "home_based"
    ]:

        watch_points.append(
            "نموذج التشغيل من المنزل يحتاج إلى "
            "التحقق من ملاءمته لطبيعة النشاط "
            "ومتطلبات الترخيص قبل الإطلاق."
        )


    # ========================================
    # SECTOR INTERPRETATION
    # ========================================

    family = project.get(
        "Business Family"
    )

    strategy_map = {
        "manufacturing": "cluster",
        "construction": "cluster",
        "transport_logistics": "cluster",
        "real_estate": "ambiguous_market",
        "retail_trade": "competition",
        "hospitality_food": "competition",
        "personal_services": "competition",
        "professional_services": "competition",
        "business_support": "competition",
        "education": "competition",
        "entertainment": "competition",
        "information_technology": "competition",
        "financial_services": "competition",
        "health_social": "competition",
        "agriculture": "resource_based",
        "mining": "resource_based",
        "utilities_environment": "resource_based",
        "other": "competition"
    }

    strategy = strategy_map.get(
        family,
        "competition"
    )

    if strategy == "cluster":

        watch_points.append(
            "ارتفاع نشاط القطاع قد يدل على سوق نشط "
            "أو تجمع أعمال، لكنه قد يعني أيضًا "
            "منافسة أعلى؛ لذلك لا يتم تفسيره وحده "
            "كميزة مؤكدة."
        )

    elif strategy == "resource_based":

        data_gaps.append(
            "هذا القطاع يحتاج بيانات موارد ومتغيرات "
            "متخصصة للحصول على تحليل موقع أدق."
        )


    # ========================================
    # SAVE RISK RESULTS
    # ========================================

    st.session_state.prelaunch_risk_summary = {
        "high_priority": high_priority,
        "watch_points": watch_points,
        "data_gaps": data_gaps,
        "positive_factors": positive_factors
    }


    # ========================================
    # DISPLAY
    # ========================================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(" مخاطر عالية الأولوية")

        if high_priority:

            for item in high_priority:
                st.error(item)

        else:

            st.success(
                "لا توجد مخاطر عالية الأولوية "
                "ضمن البيانات الحالية."
            )


        st.subheader(" نقاط تحتاج متابعة")

        if watch_points:

            for item in watch_points:
                st.warning(item)

        else:

            st.write(
                "لا توجد نقاط متابعة إضافية حاليًا."
            )


    with col2:

        st.subheader(" فجوات البيانات")

        if data_gaps:

            for item in data_gaps:
                st.info(item)

        else:

            st.write(
                "لا توجد فجوات أساسية ظاهرة "
                "ضمن البيانات المستخدمة حاليًا."
            )


        st.subheader(" عوامل إيجابية")

        if positive_factors:

            for item in positive_factors:
                st.success(item)

        else:

            st.write(
                "لا توجد عوامل إيجابية إضافية "
                "مسجلة حاليًا."
            )



# ============================================
# TARGET CUSTOMER
# ============================================

def build_target_customer_profile(project):

    family = project.get(
        "Business Family",
        "other"
    )

    operating_model = project.get(
        "Operating Model",
        "unknown"
    )

    isic_code = str(
        project.get(
            "ISIC Code",
            ""
        )
    )

    # ----------------------------------------
    # Customer Segments + Needs
    # ----------------------------------------

    if family == "retail_trade":

        segments = [
            "مستهلكون أفراد يبحثون عن المنتج للاستخدام الشخصي.",
            "عملاء محليون ضمن منطقة المشروع.",
            "عملاء يقارنون بين السعر والجودة والتوفر."
        ]

        needs = [
            "سعر مناسب",
            "توفر المنتج",
            "سهولة الشراء",
            "جودة موثوقة"
        ]

    elif family == "hospitality_food":

        segments = [
            "عملاء أفراد يبحثون عن الطعام أو خدمات الضيافة.",
            "عملاء محليون ومتكررون.",
            "عملاء التوصيل والطلبات الرقمية."
        ]

        needs = [
            "الجودة",
            "السعر",
            "سرعة الخدمة",
            "النظافة",
            "التوصيل"
        ]

    elif family == "manufacturing":

        segments = [
            "تجار ومحلات قد تشترى بالجملة.",
            "شركات تحتاج المنتجات ضمن سلسلة التوريد.",
            "عملاء أفراد إذا كان المشروع يبيع مباشرة."
        ]

        needs = [
            "ثبات الجودة",
            "السعر",
            "القدرة على التوريد",
            "الالتزام بالمواعيد"
        ]

    elif family == "real_estate":

        if isic_code == "6810":

            segments = [
                "أفراد يبحثون عن شراء أو استئجار عقار.",
                "أسر تبحث عن سكن مناسب.",
                "مستثمرون يبحثون عن عقارات للاستثمار."
            ]

            needs = [
                "السعر",
                "الموقع",
                "حالة العقار",
                "وضوح المعلومات",
                "الثقة والشفافية"
            ]

        else:

            segments = [
                "أفراد يحتاجون خدمات عقارية.",
                "أصحاب أو مستثمرون في العقارات."
            ]

            needs = [
                "الثقة",
                "الموقع",
                "السعر",
                "وضوح المعلومات"
            ]

    elif family == "information_technology":

        segments = [
            "مشاريع صغيرة ومتوسطة تحتاج حلولًا تقنية.",
            "شركات تبحث عن تطوير أو دعم أنظمة رقمية.",
            "عملاء أفراد حسب طبيعة الخدمة."
        ]

        needs = [
            "حل واضح للمشكلة",
            "سهولة الاستخدام",
            "الدعم الفني",
            "السعر"
        ]

    elif family == "professional_services":

        segments = [
            "أفراد يحتاجون خدمات متخصصة.",
            "شركات ومؤسسات تحتاج خدمات مهنية.",
            "مشاريع صغيرة تحتاج خبرات خارجية."
        ]

        needs = [
            "الثقة",
            "الخبرة",
            "وضوح الخدمة",
            "سرعة الاستجابة"
        ]

    elif family == "education":

        segments = [
            "طلاب أو متعلمون يحتاجون الخدمة التعليمية.",
            "أولياء أمور حسب طبيعة الخدمة.",
            "مؤسسات تحتاج تدريبًا أو تطوير مهارات."
        ]

        needs = [
            "جودة التعليم",
            "النتائج",
            "المرونة",
            "السعر"
        ]

    elif family == "health_social":

        segments = [
            "أفراد يحتاجون الخدمة الصحية أو الاجتماعية.",
            "أسر تبحث عن خدمات مناسبة لأفرادها.",
            "جهات تحتاج خدمات متخصصة."
        ]

        needs = [
            "الثقة",
            "سهولة الوصول",
            "جودة الخدمة",
            "الاستجابة"
        ]

    else:

        segments = [
            "عملاء أفراد محتملون حسب طبيعة النشاط.",
            "شركات أو مؤسسات قد تستفيد من الخدمة.",
            "عملاء ضمن السوق المحلي للمشروع."
        ]

        needs = [
            "القيمة مقابل السعر",
            "الجودة",
            "الثقة",
            "سهولة الوصول"
        ]


    # ----------------------------------------
    # Marketing Channels
    # ----------------------------------------

    if operating_model in [
        "online",
        "home_online"
    ]:

        channels = [
            "وسائل التواصل الاجتماعي",
            "الإعلانات الرقمية",
            "التواصل المباشر عبر الإنترنت"
        ]

    elif operating_model in [
        "home",
        "home_based"
    ]:

        channels = [
            "وسائل التواصل الاجتماعي",
            "التوصيات والإحالات",
            "التسويق المحلي"
        ]

    elif operating_model in [
        "small_physical",
        "full_physical"
    ]:

        channels = [
            "الموقع الفعلي",
            "Google Maps والمنصات المحلية",
            "وسائل التواصل الاجتماعي",
            "التسويق المحلي"
        ]

    elif operating_model == "production":

        channels = [
            "التواصل مع التجار والشركات",
            "المعارض والفعاليات التجارية",
            "التسويق الرقمي B2B",
            "المبيعات المباشرة"
        ]

    else:

        channels = [
            "التسويق الرقمي",
            "التسويق المحلي"
        ]


    return {
        "segments": segments,
        "needs": needs,
        "channels": channels
    }


# ============================================
# CUSTOMER + RECOMMENDATIONS DISPLAY
# ============================================

def render_customer_and_recommendations(project):

    if project.get("Plan") != "Premium":

        st.info(
            " تحليل العميل المستهدف والتوصيات "
            "التفصيلية متاح ضمن Premium."
        )

        return


    profile = build_target_customer_profile(
        project
    )


    # ========================================
    # TARGET CUSTOMER
    # ========================================

    st.header(" العميل المستهدف")

    st.caption(
        "هذا توصيف أولي مبني على نوع النشاط "
        "وطريقة التشغيل، ويحتاج إلى التحقق "
        "من خلال بيانات فعلية من السوق."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.subheader(" الشرائح المحتملة")

        for item in profile["segments"]:
            st.write("•", item)

    with col2:

        st.subheader(" احتياجات العملاء")

        for item in profile["needs"]:
            st.write("•", item)

    with col3:

        st.subheader(" قنوات الوصول")

        for item in profile["channels"]:
            st.write("•", item)


    # ========================================
    # RECOMMENDATIONS
    # ========================================

    st.divider()

    st.header(" التوصيات")

    priorities = []
    recommendations = []
    validation_steps = []


    # ----------------------------------------
    # Capital
    # ----------------------------------------

    capital = st.session_state.get(
        "capital_result"
    )

    available = project.get(
        "Available Capital"
    )

    if (
        capital
        and capital.get("total", 0) > 0
        and available is not None
    ):

        difference = (
            float(available)
            - float(capital["total"])
        )

        if difference < 0:

            priorities.append(
                f"يوجد فرق تخطيطي في رأس المال "
                f"بحوالي {abs(difference):,.0f} JD. "
                f"يمكن مراجعة تكاليف البداية أو "
                f"تقسيم الإنفاق إلى مراحل."
            )

        else:

            recommendations.append(
                "رأس المال المتوفر يغطي التقدير "
                "التخطيطي الحالي."
            )


    # ----------------------------------------
    # Location
    # ----------------------------------------

    location = st.session_state.get(
        "location_summary"
    )

    if location:

        rank = int(
            location["Rank"]
        )

        if rank == 1:

            recommendations.append(
                "المحافظة المختارة حصلت على أعلى "
                "نتيجة في Screening الموقع الحالي."
            )

        elif rank <= 3:

            recommendations.append(
                f"المحافظة المختارة ضمن أعلى "
                f"{rank} نتائج في Screening الموقع."
            )

        else:

            validation_steps.append(
                "مقارنة المحافظة المختارة مع "
                "المحافظات الأعلى ترتيبًا قبل "
                "اتخاذ القرار النهائي."
            )


    # ----------------------------------------
    # Risk summary
    # ----------------------------------------

    risks = st.session_state.get(
        "prelaunch_risk_summary",
        {}
    )

    for item in risks.get(
        "high_priority",
        []
    ):

        if (
            "رأس المال" in item
            and any(
                "رأس المال" in x
                for x in priorities
            )
        ):
            continue

        if item not in priorities:
            priorities.append(item)


    for item in risks.get(
        "watch_points",
        []
    ):

        if item not in validation_steps:
            validation_steps.append(item)


    for item in risks.get(
        "data_gaps",
        []
    ):

        if item not in validation_steps:
            validation_steps.append(item)


    # ----------------------------------------
    # Customer validation
    # ----------------------------------------

    recommendations.append(
        "اختبار الجمهور المستهدف مبدئيًا قبل "
        "الإطلاق للتأكد من وجود اهتمام فعلي."
    )

    recommendations.append(
        "بدء تجربة تسويقية صغيرة باستخدام "
        "قنوات الوصول المقترحة قبل زيادة "
        "الإنفاق التسويقي."
    )

    validation_steps.append(
        "جمع بيانات فعلية من السوق قبل الالتزام "
        "الكامل بالمشروع، مثل استبيان أو مقابلات "
        "مع عملاء محتملين أو تجربة أولية صغيرة."
    )


    # ========================================
    # DISPLAY
    # ========================================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(" إجراءات ذات أولوية")

        if priorities:

            for item in priorities:
                st.warning(item)

        else:

            st.success(
                "لا توجد إجراءات عاجلة واضحة "
                "ضمن البيانات الحالية."
            )


        st.subheader(" توصيات")

        for item in recommendations:
            st.success(item)


    with col2:

        st.subheader(" خطوات التحقق")

        for item in validation_steps:
            st.info(item)



# ============================================
# EXISTING BUSINESS HELPERS
# ============================================

def safe_percent_change(current, previous):

    current = float(current)
    previous = float(previous)

    if previous == 0 and current == 0:
        return 0.0

    if previous == 0:
        return None

    return (
        (current - previous)
        / abs(previous)
        * 100
    )


def calculate_existing_kpis(snapshot):

    revenue = float(snapshot["Revenue"])
    expenses = float(snapshot["Expenses"])
    marketing = float(snapshot["Marketing Spend"])

    customers = int(snapshot["Customers"])
    transactions = int(snapshot["Transactions"])
    complaints = int(snapshot["Complaints"])

    profit = revenue - expenses

    profit_margin = (
        profit / revenue * 100
        if revenue > 0
        else 0
    )

    average_order_value = (
        revenue / transactions
        if transactions > 0
        else 0
    )

    complaint_rate = (
        complaints / customers * 100
        if customers > 0
        else 0
    )

    marketing_ratio = (
        marketing / revenue * 100
        if revenue > 0
        else 0
    )

    return {
        "Profit": profit,
        "Profit Margin %": profit_margin,
        "Average Order Value": average_order_value,
        "Complaint Rate %": complaint_rate,
        "Marketing / Revenue %": marketing_ratio
    }


def save_existing_month(month_name, snapshot):

    if "performance_history" not in st.session_state:
        st.session_state.performance_history = []

    record = {
        "Month": month_name,
        **snapshot,
        **calculate_existing_kpis(snapshot)
    }

    # إذا الشهر موجود، نحدثه بدل تكراره
    updated = False

    for i, old_record in enumerate(
        st.session_state.performance_history
    ):

        if old_record["Month"] == month_name:

            st.session_state.performance_history[i] = record
            updated = True
            break

    if not updated:
        st.session_state.performance_history.append(record)



# ============================================
# EXISTING BUSINESS CHANGE ANALYSIS
# ============================================

def analyze_existing_changes(previous, latest):

    warnings = []
    observations = []
    positives = []


    # ----------------------------------------
    # Percentage changes
    # ----------------------------------------

    revenue_change = safe_percent_change(
        latest["Revenue"],
        previous["Revenue"]
    )

    profit_change = safe_percent_change(
        latest["Profit"],
        previous["Profit"]
    )

    customer_change = safe_percent_change(
        latest["Customers"],
        previous["Customers"]
    )

    transaction_change = safe_percent_change(
        latest["Transactions"],
        previous["Transactions"]
    )

    complaint_rate_change = safe_percent_change(
        latest["Complaint Rate %"],
        previous["Complaint Rate %"]
    )

    expenses_change = safe_percent_change(
        latest["Expenses"],
        previous["Expenses"]
    )

    average_order_change = safe_percent_change(
        latest["Average Order Value"],
        previous["Average Order Value"]
    )


    # ----------------------------------------
    # Absolute differences
    # ----------------------------------------

    rating_diff = (
        float(latest["Customer Rating"])
        - float(previous["Customer Rating"])
    )

    margin_diff = (
        float(latest["Profit Margin %"])
        - float(previous["Profit Margin %"])
    )


    # ========================================
    # Revenue
    # ========================================

    if revenue_change is not None:

        if revenue_change <= -20:

            warnings.append(
                f"الإيرادات انخفضت بنسبة "
                f"{abs(revenue_change):.1f}%، "
                f"وهو تغير يحتاج مراجعة ذات أولوية."
            )

        elif revenue_change <= -10:

            observations.append(
                f"الإيرادات انخفضت بنسبة "
                f"{abs(revenue_change):.1f}%."
            )

        elif revenue_change >= 10:

            positives.append(
                f"الإيرادات ارتفعت بنسبة "
                f"{revenue_change:.1f}%."
            )


    # ========================================
    # Profit
    # ========================================

    if profit_change is not None:

        if profit_change <= -20:

            warnings.append(
                f"الربح انخفض بنسبة "
                f"{abs(profit_change):.1f}%، "
                f"ويحتاج تحليل أسباب الانخفاض."
            )

        elif profit_change <= -10:

            observations.append(
                f"الربح انخفض بنسبة "
                f"{abs(profit_change):.1f}%."
            )

        elif profit_change >= 10:

            positives.append(
                f"الربح ارتفع بنسبة "
                f"{profit_change:.1f}%."
            )


    # ========================================
    # Profit Margin
    # ========================================

    if margin_diff <= -5:

        warnings.append(
            f"هامش الربح انخفض بحوالي "
            f"{abs(margin_diff):.1f} نقطة مئوية."
        )

    elif margin_diff >= 3:

        positives.append(
            f"هامش الربح تحسن بحوالي "
            f"{margin_diff:.1f} نقطة مئوية."
        )


    # ========================================
    # Customers
    # ========================================

    if customer_change is not None:

        if customer_change <= -15:

            warnings.append(
                f"عدد العملاء انخفض بنسبة "
                f"{abs(customer_change):.1f}%."
            )

        elif customer_change <= -5:

            observations.append(
                f"عدد العملاء انخفض بنسبة "
                f"{abs(customer_change):.1f}%."
            )

        elif customer_change >= 10:

            positives.append(
                f"عدد العملاء ارتفع بنسبة "
                f"{customer_change:.1f}%."
            )


    # ========================================
    # Transactions
    # ========================================

    if transaction_change is not None:

        if transaction_change <= -20:

            warnings.append(
                f"عدد عمليات البيع انخفض بنسبة "
                f"{abs(transaction_change):.1f}%."
            )

        elif transaction_change <= -10:

            observations.append(
                f"عدد عمليات البيع انخفض بنسبة "
                f"{abs(transaction_change):.1f}%."
            )

        elif transaction_change >= 10:

            positives.append(
                f"عدد عمليات البيع ارتفع بنسبة "
                f"{transaction_change:.1f}%."
            )


    # ========================================
    # Complaint Rate
    # ========================================

    if complaint_rate_change is not None:

        if complaint_rate_change >= 25:

            warnings.append(
                f"معدل الشكاوى ارتفع بنسبة "
                f"{complaint_rate_change:.1f}%."
            )

        elif complaint_rate_change >= 10:

            observations.append(
                f"معدل الشكاوى ارتفع بنسبة "
                f"{complaint_rate_change:.1f}%."
            )

        elif complaint_rate_change <= -20:

            positives.append(
                f"معدل الشكاوى انخفض بنسبة "
                f"{abs(complaint_rate_change):.1f}%."
            )


    # ========================================
    # Customer Rating
    # ========================================

    if rating_diff <= -0.5:

        warnings.append(
            f"تقييم العملاء انخفض بمقدار "
            f"{abs(rating_diff):.1f} نقطة."
        )

    elif rating_diff >= 0.3:

        positives.append(
            f"تقييم العملاء تحسن بمقدار "
            f"{rating_diff:.1f} نقطة."
        )


    # ========================================
    # Combined Context
    # ========================================

    if (
        transaction_change is not None
        and average_order_change is not None
        and transaction_change < 0
        and average_order_change > 0
    ):

        observations.append(
            "انخفض عدد عمليات البيع بينما ارتفع "
            "متوسط قيمة الطلب؛ قد يكون التغير ناتجًا "
            "عن عدد أقل من العمليات بقيم أعلى."
        )


    if (
        expenses_change is not None
        and revenue_change is not None
        and expenses_change > 10
        and revenue_change <= 0
    ):

        observations.append(
            "المصاريف ارتفعت بينما الإيرادات لم ترتفع، "
            "لذلك يُنصح بمراجعة بنود التكلفة."
        )


    return {
        "warnings": list(dict.fromkeys(warnings)),
        "observations": list(dict.fromkeys(observations)),
        "positives": list(dict.fromkeys(positives))
    }



# ============================================
# CUSTOMER REVIEW NLP
# ============================================

aspect_keywords = {

    "الجودة": [
        "جودة", "جوده", "الخامة", "خامة",
        "القماش", "قماش",
        "تشطيب", "خياطة",
        "quality", "fabric"
    ],

    "المقاس": [
        "مقاس", "قياس",
        "ضيق", "واسع",
        "size"
    ],

    "السعر": [
        "سعر", "السعر",
        "غالي", "رخيص",
        "مرتفع", "price"
    ],

    "التأخير": [
        "تأخر", "تأخير",
        "تاخير", "متأخر",
        "التوصيل", "توصيل",
        "أخذت وقت", "اخذت وقت",
        "بطء", "بطيء",
        "delivery", "late"
    ],

    "الخدمة": [
        "خدمة", "الخدمة",
        "تعامل", "موظف",
        "موظفين", "استجابة",
        "service"
    ],

    "التغليف": [
        "تغليف", "التغليف",
        "باكج", "package",
        "packaging"
    ],

    "التجربة الرقمية": [
        "التطبيق",
        "تطبيق",
        "استخدام التطبيق",
        "واجهة التطبيق",
        "سهولة الاستخدام",
        "app",
        "application"
    ],

    "تنوع المنتجات": [
        "تشكيلة",
        "تنوع",
        "خيارات",
        "موديلات",
        "قطع",
        "منتجات",
        "اصناف",
        "أصناف",
        "variety",
        "collection"
    ],

    "العلامات التجارية": [
        "ماركات",
        "الماركات",
        "براند",
        "براندات",
        "علامات تجارية",
        "brand",
        "brands"
    ]
}


@st.cache_resource
def load_sentiment_model():

    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-xlm-roberta-base-sentiment"
    )


def analyze_review_sentiment(review):

    model = load_sentiment_model()

    result = model(
        str(review)
    )[0]

    label_map = {
        "negative": "سلبي",
        "neutral": "محايد",
        "positive": "إيجابي"
    }

    return {
        "Sentiment":
            label_map.get(
                result["label"].lower(),
                result["label"]
            ),

        "Confidence":
            round(
                float(result["score"]),
                3
            )
    }


def detect_review_aspects(review):

    text = str(review).lower()

    found = []

    for aspect, keywords in aspect_keywords.items():

        for keyword in keywords:

            if keyword.lower() in text:

                found.append(aspect)
                break

    return found


def split_review_into_clauses(review):

    clauses = re.split(
        r"\s*(?:بس|لكن|ولكن|،|,|؛|;)\s*|\s+و(?=ال)",
        str(review).strip()
    )

    return [
        x.strip()
        for x in clauses
        if x.strip()
    ]


def analyze_aspect_sentiment(review):

    results = []

    for clause in split_review_into_clauses(review):

        aspects = detect_review_aspects(
            clause
        )

        if not aspects:
            continue

        sentiment = analyze_review_sentiment(
            clause
        )

        for aspect in aspects:

            results.append({
                "Review": review,
                "Aspect": aspect,
                "Context": clause,
                "Aspect Sentiment":
                    sentiment["Sentiment"],
                "Confidence":
                    sentiment["Confidence"]
            })

    return results


def build_streamlit_aspect_summary(reviews):

    rows = []

    for review in reviews:

        rows.extend(
            analyze_aspect_sentiment(
                review
            )
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    summary = (
        df
        .groupby(
            ["Aspect", "Aspect Sentiment"]
        )
        .size()
        .unstack(fill_value=0)
    )

    for sentiment in [
        "إيجابي",
        "محايد",
        "سلبي"
    ]:

        if sentiment not in summary.columns:
            summary[sentiment] = 0

    summary = summary[
        ["إيجابي", "محايد", "سلبي"]
    ]

    summary["Total Mentions"] = (
        summary[
            ["إيجابي", "محايد", "سلبي"]
        ]
        .sum(axis=1)
    )

    summary["Positive %"] = (
        summary["إيجابي"]
        / summary["Total Mentions"]
        * 100
    )

    summary["Negative %"] = (
        summary["سلبي"]
        / summary["Total Mentions"]
        * 100
    )

    summary["Neutral %"] = (
        summary["محايد"]
        / summary["Total Mentions"]
        * 100
    )

    confidence = (
        df
        .groupby("Aspect")["Confidence"]
        .mean()
    )

    summary["Average Confidence"] = confidence

    return (
        summary
        .sort_values(
            ["Negative %", "Total Mentions"],
            ascending=[False, False]
        )
        .reset_index()
    )


def render_customer_review_nlp():

    st.header(" تحليل آراء العملاء")

    st.caption(
        "يستخدم النظام نموذج NLP متعدد اللغات لتحليل "
        "المشاعر وربطها بجوانب محددة من تجربة العميل."
    )

    if "customer_reviews" not in st.session_state:
        st.session_state.customer_reviews = []

    review = st.text_area(
        "رأي العميل",
        placeholder="مثال: الجودة ممتازة لكن التوصيل تأخر."
    )

    if st.button(
        "تحليل وإضافة الرأي ",
        use_container_width=True
    ):

        if not review.strip():

            st.warning(
                "اكتب رأي العميل أولًا."
            )

        else:

            with st.spinner(
                "جاري تحليل رأي العميل..."
            ):

                overall = analyze_review_sentiment(
                    review.strip()
                )

                st.session_state.customer_reviews.append(
                    review.strip()
                )

                st.success(
                    f"التصنيف العام: "
                    f"{overall['Sentiment']} "
                    f"— ثقة "
                    f"{overall['Confidence']:.2f}"
                )


    reviews = st.session_state.customer_reviews

    if reviews:

        st.subheader("آراء العملاء المدخلة")

        for i, item in enumerate(
            reviews,
            start=1
        ):
            st.write(
                f"{i}. {item}"
            )

        with st.spinner(
            "جاري بناء ملخص الجوانب..."
        ):

            summary = (
                build_streamlit_aspect_summary(
                    reviews
                )
            )

        if not summary.empty:

            st.subheader(
                " ملخص تحليل الجوانب"
            )

            display_summary = summary[
                [
                    "Aspect",
                    "Total Mentions",
                    "Positive %",
                    "Neutral %",
                    "Negative %",
                    "Rule-based Mentions",
                    "Average Model Confidence"
                ]
            ].copy()

            display_summary[
                "Positive %"
            ] = display_summary[
                "Positive %"
            ].round(1)

            display_summary[
                "Neutral %"
            ] = display_summary[
                "Neutral %"
            ].round(1)

            display_summary[
                "Negative %"
            ] = display_summary[
                "Negative %"
            ].round(1)

            if "Average Model Confidence" in display_summary.columns:

                display_summary[
                    "Average Model Confidence"
                ] = display_summary[
                    "Average Model Confidence"
                ].apply(
                    lambda x:
                    "—"
                    if pd.isna(x)
                    else round(float(x), 2)
                )

            st.dataframe(
                display_summary,
                use_container_width=True,
                hide_index=True
            )

            st.session_state.aspect_summary = summary

            render_grouped_review_alerts()

            # ====================================
            # INTEGRATED PERFORMANCE + NLP
            # ====================================




# ============================================
# HYBRID ASPECT SENTIMENT
# ============================================

def normalize_review_text(text):

    text = str(text).lower().strip()

    replacements = {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ى": "ي"
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def keyword_is_negated(text, keyword):

    text = normalize_review_text(text)
    keyword = normalize_review_text(keyword)

    negation_patterns = [
        f"مش {keyword}",
        f"مو {keyword}",
        f"ليس {keyword}",
        f"غير {keyword}"
    ]

    return any(
        pattern in text
        for pattern in negation_patterns
    )


def rule_based_aspect_sentiment(
    aspect,
    clause
):

    text = normalize_review_text(
        clause
    )


    # ========================================
    # DELAY / DELIVERY
    # ========================================

    if aspect == "التأخير":

        negative_terms = [
            "تاخر",
            "تاخير",
            "متاخر",
            "اخذت وقت",
            "اخذ وقت",
            "اخد وقت",
            "طولت",
            "طول",
            "بطيء",
            "بطيئه",
            "بطء"
        ]

        positive_terms = [
            "سريع",
            "سريعه",
            "وصل بسرعه",
            "التوصيل ممتاز"
        ]

        if any(
            term in text
            for term in negative_terms
        ):
            return "سلبي"

        if any(
            term in text
            for term in positive_terms
        ):
            return "إيجابي"


    # ========================================
    # PRICE
    # ========================================

    elif aspect == "السعر":

        negative_terms = [
            "غالي",
            "غاليه",
            "مرتفع",
            "مرتفعه",
            "مبالغ",
            "مكلف"
        ]

        positive_terms = [
            "رخيص",
            "رخيصه",
            "سعر مناسب",
            "السعر مناسب",
            "سعر معقول",
            "السعر معقول"
        ]

        for term in negative_terms:

            if (
                term in text
                and not keyword_is_negated(
                    text,
                    term
                )
            ):
                return "سلبي"

        if any(
            term in text
            for term in positive_terms
        ):
            return "إيجابي"


    # ========================================
    # QUALITY
    # ========================================

    elif aspect == "الجودة":

        positive_terms = [
            "ممتاز",
            "ممتازه",
            "جيد",
            "جيده",
            "رائعه",
            "رائع",
            "مرتبه",
            "الخامة حلوه",
            "الخامه حلوه"
        ]

        negative_terms = [
            "سيئ",
            "سيئه",
            "رديء",
            "رديئه",
            "ضعيف",
            "ضعيفه",
            "الخامة سيئه",
            "الخامه سيئه"
        ]

        if any(
            term in text
            for term in negative_terms
        ):
            return "سلبي"

        if any(
            term in text
            for term in positive_terms
        ):
            return "إيجابي"


    # ========================================
    # SIZE
    # ========================================

    elif aspect == "المقاس":

        negative_terms = [
            "غير مناسب",
            "مش مناسب",
            "ضيق",
            "واسع",
            "المقاس غلط",
            "القياس غلط"
        ]

        positive_terms = [
            "مناسب",
            "مظبوط",
            "مضبوط",
            "المقاس ممتاز"
        ]

        if any(
            term in text
            for term in negative_terms
        ):
            return "سلبي"

        if any(
            term in text
            for term in positive_terms
        ):
            return "إيجابي"


    # ========================================
    # SERVICE
    # ========================================

    elif aspect == "الخدمة":

        negative_terms = [
            "سيئ",
            "سيئه",
            "بطيء",
            "بطيئه",
            "غير متعاون",
            "مش متعاون",
            "تعامل سيئ"
        ]

        positive_terms = [
            "ممتاز",
            "ممتازه",
            "متعاون",
            "متعاونين",
            "سريع",
            "سريعه",
            "تعامل رائع"
        ]

        if any(
            term in text
            for term in negative_terms
        ):
            return "سلبي"

        if any(
            term in text
            for term in positive_terms
        ):
            return "إيجابي"


    # ========================================
    # PACKAGING
    # ========================================

    elif aspect == "التغليف":

        negative_terms = [
            "سيئ",
            "سيئه",
            "ممزق",
            "تالف",
            "مخرب"
        ]

        positive_terms = [
            "جميل",
            "جميله",
            "مرتب",
            "مرتبه",
            "ممتاز",
            "ممتازه"
        ]

        if any(
            term in text
            for term in negative_terms
        ):
            return "سلبي"

        if any(
            term in text
            for term in positive_terms
        ):
            return "إيجابي"


    # ========================================
    # DIGITAL EXPERIENCE
    # ========================================

    elif aspect == "التجربة الرقمية":

        negative_terms = [
            "صعب",
            "صعبه",
            "معقد",
            "معقده",
            "بطيء",
            "بطيئ",
            "يعلق",
            "تعليق",
            "ما بشتغل",
            "مش شغال",
            "لا يعمل"
        ]

        positive_terms = [
            "سهل",
            "سهله",
            "سلس",
            "سلسه",
            "واضح",
            "واضحه",
            "سريع",
            "سريعه",
            "سهولة الاستخدام"
        ]

        if any(
            term in text
            for term in negative_terms
        ):
            return "سلبي"

        if any(
            term in text
            for term in positive_terms
        ):
            return "إيجابي"


    # ========================================
    # PRODUCT VARIETY
    # ========================================

    elif aspect == "تنوع المنتجات":

        negative_terms = [
            "محدود",
            "محدوده",
            "قليل",
            "قليله",
            "ما في خيارات",
            "خيارات قليلة",
            "نفس القطع",
            "نفس المنتجات"
        ]

        positive_terms = [
            "متنوع",
            "متنوعه",
            "تنوع",
            "فريد",
            "فريده",
            "قطع فريدة",
            "خيارات كثيرة",
            "تشكيلة كبيرة",
            "تشكيله كبيره"
        ]

        if any(
            term in text
            for term in negative_terms
        ):
            return "سلبي"

        if any(
            term in text
            for term in positive_terms
        ):
            return "إيجابي"


    # ========================================
    # BRANDS
    # ========================================

    elif aspect == "العلامات التجارية":

        negative_terms = [
            "تقليد",
            "مقلد",
            "مقلده",
            "غير اصلي",
            "غير اصليه",
            "مش اصلي",
            "مش اصليه"
        ]

        positive_terms = [
            "عالميه",
            "عالمية",
            "معروفه",
            "معروفة",
            "اصليه",
            "أصلية",
            "اصلية",
            "ماركات عالمية"
        ]

        if any(
            term in text
            for term in negative_terms
        ):
            return "سلبي"

        if any(
            term in text
            for term in positive_terms
        ):
            return "إيجابي"


    return None


# ============================================
# OVERRIDE ASPECT ANALYSIS
# ============================================

def analyze_aspect_sentiment(review):

    results = []

    for clause in split_review_into_clauses(
        review
    ):

        aspects = detect_review_aspects(
            clause
        )

        if not aspects:
            continue


        for aspect in aspects:

            rule_sentiment = (
                rule_based_aspect_sentiment(
                    aspect,
                    clause
                )
            )


            # --------------------------------
            # Clear rule
            # --------------------------------

            if rule_sentiment is not None:

                sentiment_label = (
                    rule_sentiment
                )

                confidence = np.nan

                method = "Rule-based"


            # --------------------------------
            # NLP fallback
            # --------------------------------

            else:

                model_result = (
                    analyze_review_sentiment(
                        clause
                    )
                )

                sentiment_label = (
                    model_result[
                        "Sentiment"
                    ]
                )

                confidence = (
                    model_result[
                        "Confidence"
                    ]
                )

                method = "NLP Model"


            results.append({
                "Review":
                    review,

                "Aspect":
                    aspect,

                "Context":
                    clause,

                "Aspect Sentiment":
                    sentiment_label,

                "Confidence":
                    confidence,

                "Method":
                    method
            })


    return results


# ============================================
# OVERRIDE SUMMARY
# ============================================

def build_streamlit_aspect_summary(
    reviews
):

    rows = []

    for review in reviews:

        rows.extend(
            analyze_aspect_sentiment(
                review
            )
        )


    if not rows:
        return pd.DataFrame()


    df = pd.DataFrame(
        rows
    )


    summary = (
        df
        .groupby(
            [
                "Aspect",
                "Aspect Sentiment"
            ]
        )
        .size()
        .unstack(fill_value=0)
    )


    for sentiment in [
        "إيجابي",
        "محايد",
        "سلبي"
    ]:

        if sentiment not in summary.columns:
            summary[sentiment] = 0


    summary = summary[
        [
            "إيجابي",
            "محايد",
            "سلبي"
        ]
    ]


    summary["Total Mentions"] = (
        summary[
            [
                "إيجابي",
                "محايد",
                "سلبي"
            ]
        ]
        .sum(axis=1)
    )


    summary["Positive %"] = (
        summary["إيجابي"]
        / summary["Total Mentions"]
        * 100
    )


    summary["Neutral %"] = (
        summary["محايد"]
        / summary["Total Mentions"]
        * 100
    )


    summary["Negative %"] = (
        summary["سلبي"]
        / summary["Total Mentions"]
        * 100
    )


    # ثقة النموذج فقط
    model_rows = df[
        df["Method"]
        == "NLP Model"
    ]

    if not model_rows.empty:

        model_confidence = (
            model_rows
            .groupby("Aspect")[
                "Confidence"
            ]
            .mean()
        )

        summary[
            "Average Model Confidence"
        ] = model_confidence

    else:

        summary[
            "Average Model Confidence"
        ] = np.nan


    # عدد الحالات التي حُسمت بالقواعد
    rule_counts = (
        df[
            df["Method"]
            == "Rule-based"
        ]
        .groupby("Aspect")
        .size()
    )

    summary[
        "Rule-based Mentions"
    ] = (
        rule_counts
        .reindex(
            summary.index,
            fill_value=0
        )
    )


    return (
        summary
        .sort_values(
            [
                "Negative %",
                "Total Mentions"
            ],
            ascending=[
                False,
                False
            ]
        )
        .reset_index()
    )



# ============================================
# INTEGRATED BUSINESS INSIGHTS
# ============================================

def build_integrated_existing_insights(
    base_insights,
    aspect_summary
):

    priority_alerts = list(
        base_insights.get(
            "warnings",
            []
        )
    )

    business_context = list(
        base_insights.get(
            "observations",
            []
        )
    )

    positive_signals = list(
        base_insights.get(
            "positives",
            []
        )
    )


    if (
        isinstance(
            aspect_summary,
            pd.DataFrame
        )
        and not aspect_summary.empty
    ):

        for _, row in aspect_summary.iterrows():

            aspect = row["Aspect"]

            mentions = int(
                row["Total Mentions"]
            )

            negative = float(
                row["Negative %"]
            )

            positive = float(
                row["Positive %"]
            )


            # -------------------------------
            # Repeated negative feedback
            # -------------------------------

            if (
                mentions >= 2
                and negative >= 60
            ):

                priority_alerts.append(
                    f"آراء العملاء حول «{aspect}» "
                    f"تظهر اتجاهًا سلبيًا "
                    f"({negative:.0f}% من "
                    f"{mentions} إشارات)."
                )


            # -------------------------------
            # Repeated positive feedback
            # -------------------------------

            if (
                mentions >= 2
                and positive >= 60
            ):

                positive_signals.append(
                    f"آراء العملاء حول «{aspect}» "
                    f"تميل إلى الإيجابية "
                    f"({positive:.0f}% من الإشارات)."
                )


    return {
        "Priority Alerts":
            list(
                dict.fromkeys(
                    priority_alerts
                )
            ),

        "Business Context":
            list(
                dict.fromkeys(
                    business_context
                )
            ),

        "Positive Signals":
            list(
                dict.fromkeys(
                    positive_signals
                )
            )
    }


def render_integrated_existing_insights():

    base = st.session_state.get(
        "existing_base_insights"
    )

    summary = st.session_state.get(
        "aspect_summary"
    )

    if not base:
        return

    if (
        not isinstance(
            summary,
            pd.DataFrame
        )
        or summary.empty
    ):
        return


    integrated = (
        build_integrated_existing_insights(
            base,
            summary
        )
    )


    st.divider()

    st.header(
        " قراءة متكاملة للأداء وآراء العملاء"
    )

    st.caption(
        "يتم دمج تغيرات الأداء الشهرية مع "
        "أنماط آراء العملاء المتكررة. "
        "حدود التنبيه في هذه النسخة التجريبية "
        "قابلة للتعديل وليست قواعد عالمية ثابتة."
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.subheader(
            " تنبيهات ذات أولوية"
        )

        if integrated[
            "Priority Alerts"
        ]:

            for item in integrated[
                "Priority Alerts"
            ]:

                st.error(item)

        else:

            st.success(
                "لا توجد تنبيهات مرتفعة الأولوية."
            )


    with col2:

        st.subheader(
            " سياق الأداء"
        )

        if integrated[
            "Business Context"
        ]:

            for item in integrated[
                "Business Context"
            ]:

                st.info(item)

        else:

            st.write(
                "لا توجد ملاحظات سياقية إضافية."
            )


    with col3:

        st.subheader(
            " إشارات إيجابية"
        )

        if integrated[
            "Positive Signals"
        ]:

            for item in integrated[
                "Positive Signals"
            ]:

                st.success(item)

        else:

            st.write(
                "لا توجد إشارات إيجابية إضافية."
            )



# ============================================
# FINAL INTEGRATED EXISTING BUSINESS VIEW
# ============================================

def render_final_integrated_insights():

    history = st.session_state.get(
        "performance_history",
        []
    )

    summary = st.session_state.get(
        "aspect_summary"
    )

    # لازم يكون في شهرين
    if len(history) < 2:
        return

    # لازم يكون في تحليل آراء
    if (
        not isinstance(summary, pd.DataFrame)
        or summary.empty
    ):
        return


    performance = pd.DataFrame(
        history
    )

    previous = performance.iloc[-2]
    latest = performance.iloc[-1]


    # تحليل الأداء من آخر شهرين
    base = analyze_existing_changes(
        previous,
        latest
    )


    # دمج الأداء مع آراء العملاء
    integrated = (
        build_integrated_existing_insights(
            base,
            summary
        )
    )


    st.divider()

    st.header(
        " قراءة متكاملة للأداء وآراء العملاء"
    )

    st.caption(
        "يجمع هذا القسم بين تغير مؤشرات الأداء "
        "والأنماط المتكررة في آراء العملاء."
    )


    col1, col2, col3 = st.columns(3)


    # ========================================
    # ALERTS
    # ========================================

    with col1:

        st.subheader(
            " تنبيهات ذات أولوية"
        )

        alerts = integrated[
            "Priority Alerts"
        ]

        if alerts:

            for item in alerts:
                st.error(item)

        else:

            st.success(
                "لا توجد تنبيهات مرتفعة الأولوية."
            )


    # ========================================
    # CONTEXT
    # ========================================

    with col2:

        st.subheader(
            " سياق الأداء"
        )

        context = integrated[
            "Business Context"
        ]

        if context:

            for item in context:
                st.info(item)

        else:

            st.write(
                "لا توجد ملاحظات سياقية إضافية."
            )


    # ========================================
    # POSITIVE SIGNALS
    # ========================================

    with col3:

        st.subheader(
            " إشارات إيجابية"
        )

        positives = integrated[
            "Positive Signals"
        ]

        if positives:

            for item in positives:
                st.success(item)

        else:

            st.write(
                "لا توجد إشارات إيجابية إضافية."
            )



# ============================================
# GROUPBY REVIEW INSIGHTS
# ============================================

def build_grouped_review_insights(reviews):

    rows = []

    for review in reviews:

        results = analyze_aspect_sentiment(
            review
        )

        rows.extend(results)


    if not rows:
        return pd.DataFrame()


    df = pd.DataFrame(rows)


    # ========================================
    # GROUP BY ASPECT
    # ========================================

    grouped = (
        df
        .groupby("Aspect")
        .agg(
            Total_Mentions=(
                "Aspect",
                "size"
            ),

            Negative_Mentions=(
                "Aspect Sentiment",
                lambda x:
                    (x == "سلبي").sum()
            ),

            Positive_Mentions=(
                "Aspect Sentiment",
                lambda x:
                    (x == "إيجابي").sum()
            ),

            Neutral_Mentions=(
                "Aspect Sentiment",
                lambda x:
                    (x == "محايد").sum()
            )
        )
        .reset_index()
    )


    grouped["Negative %"] = (
        grouped["Negative_Mentions"]
        / grouped["Total_Mentions"]
        * 100
    )


    grouped["Positive %"] = (
        grouped["Positive_Mentions"]
        / grouped["Total_Mentions"]
        * 100
    )


    grouped["Neutral %"] = (
        grouped["Neutral_Mentions"]
        / grouped["Total_Mentions"]
        * 100
    )


    return grouped


def render_grouped_review_alerts():

    reviews = st.session_state.get(
        "customer_reviews",
        []
    )

    if not reviews:
        return


    grouped = build_grouped_review_insights(
        reviews
    )

    if grouped.empty:
        return


    st.divider()

    st.header(
        " قراءة متكاملة لآراء العملاء"
    )


    alerts = []
    positives = []


    for _, row in grouped.iterrows():

        aspect = row["Aspect"]

        mentions = int(
            row["Total_Mentions"]
        )

        negative = float(
            row["Negative %"]
        )

        positive = float(
            row["Positive %"]
        )


        # ====================================
        # NEGATIVE REPEATED SIGNAL
        # ====================================

        if (
            mentions >= 2
            and negative >= 60
        ):

            alerts.append(
                f"تكررت ملاحظات سلبية حول «{aspect}» "
                f"في {mentions} إشارات، "
                f"بنسبة سلبية {negative:.0f}%."
            )


        # ====================================
        # POSITIVE REPEATED SIGNAL
        # ====================================

        if (
            mentions >= 2
            and positive >= 60
        ):

            positives.append(
                f"تكررت ملاحظات إيجابية حول «{aspect}» "
                f"في {mentions} إشارات، "
                f"بنسبة إيجابية {positive:.0f}%."
            )


    col1, col2 = st.columns(2)


    with col1:

        st.subheader(
            " تنبيهات متكررة"
        )

        if alerts:

            for item in alerts:
                st.error(item)

        else:

            st.success(
                "لا توجد أنماط سلبية متكررة "
                "تتجاوز حد التنبيه الحالي."
            )


    with col2:

        st.subheader(
            " أنماط إيجابية"
        )

        if positives:

            for item in positives:
                st.success(item)

        else:

            st.write(
                "لا توجد أنماط إيجابية متكررة حاليًا."
            )



# ============================================
# EXISTING BUSINESS RECOMMENDATIONS
# ============================================

def build_existing_recommendations():

    recommendations = []
    investigation = []

    history = st.session_state.get(
        "performance_history",
        []
    )

    # ========================================
    # PERFORMANCE-BASED RECOMMENDATIONS
    # ========================================

    if len(history) >= 2:

        df = pd.DataFrame(history)

        previous = df.iloc[-2]
        latest = df.iloc[-1]

        revenue_change = safe_percent_change(
            latest["Revenue"],
            previous["Revenue"]
        )

        profit_change = safe_percent_change(
            latest["Profit"],
            previous["Profit"]
        )

        transaction_change = safe_percent_change(
            latest["Transactions"],
            previous["Transactions"]
        )

        customer_change = safe_percent_change(
            latest["Customers"],
            previous["Customers"]
        )

        complaint_change = safe_percent_change(
            latest["Complaint Rate %"],
            previous["Complaint Rate %"]
        )

        margin_diff = (
            float(latest["Profit Margin %"])
            - float(previous["Profit Margin %"])
        )


        # ------------------------------------
        # Revenue
        # ------------------------------------

        if (
            revenue_change is not None
            and revenue_change <= -10
        ):

            recommendations.append(
                "مراجعة أسباب انخفاض الإيرادات، "
                "مثل حجم الطلب، قنوات البيع، "
                "الأسعار والعروض التسويقية."
            )


        # ------------------------------------
        # Profit
        # ------------------------------------

        if (
            profit_change is not None
            and profit_change <= -10
        ):

            recommendations.append(
                "مراجعة هيكل التكاليف ومصادر الربح "
                "لتحديد البنود الأكثر تأثيرًا "
                "على انخفاض الربحية."
            )


        # ------------------------------------
        # Profit Margin
        # ------------------------------------

        if margin_diff <= -3:

            recommendations.append(
                "تحليل هامش الربح حسب المنتج أو الخدمة، "
                "ومراجعة التسعير والتكاليف المباشرة."
            )


        # ------------------------------------
        # Transactions
        # ------------------------------------

        if (
            transaction_change is not None
            and transaction_change <= -10
        ):

            recommendations.append(
                "فحص أسباب انخفاض عدد عمليات البيع، "
                "مع مراجعة سهولة الشراء، "
                "معدل التحويل وإعادة الشراء."
            )

            investigation.append(
                "تحقق مما إذا كان انخفاض العمليات "
                "متركزًا في منتج أو قناة بيع "
                "أو فترة زمنية محددة."
            )


        # ------------------------------------
        # Customers
        # ------------------------------------

        if (
            customer_change is not None
            and customer_change <= -5
        ):

            recommendations.append(
                "مراجعة اكتساب العملاء والاحتفاظ بهم، "
                "ومقارنة العملاء الجدد بالعملاء المتكررين."
            )


        # ------------------------------------
        # Complaints
        # ------------------------------------

        if (
            complaint_change is not None
            and complaint_change >= 10
        ):

            recommendations.append(
                "تصنيف الشكاوى حسب الموضوع "
                "وتحديد المشكلات الأكثر تكرارًا "
                "قبل اتخاذ إجراء تصحيحي."
            )

            investigation.append(
                "قارن ارتفاع الشكاوى مع نتائج "
                "تحليل آراء العملاء، دون افتراض "
                "أن أحدهما سبب مباشر للآخر."
            )


    # ========================================
    # REVIEW-BASED RECOMMENDATIONS
    # ========================================

    reviews = st.session_state.get(
        "customer_reviews",
        []
    )

    if reviews:

        grouped = build_grouped_review_insights(
            reviews
        )

        if not grouped.empty:

            for _, row in grouped.iterrows():

                aspect = row["Aspect"]

                mentions = int(
                    row["Total_Mentions"]
                )

                negative = float(
                    row["Negative %"]
                )


                # لا نعتبر التعليق الواحد نمطًا متكررًا
                if not (
                    mentions >= 2
                    and negative >= 60
                ):
                    continue


                # --------------------------------
                # Delay
                # --------------------------------

                if aspect == "التأخير":

                    recommendations.append(
                        "مراجعة مسار تنفيذ الطلب والتوصيل، "
                        "من وقت استلام الطلب حتى التسليم، "
                        "وتحديد المرحلة التي يحدث فيها التأخير."
                    )

                    investigation.append(
                        "تحقق مما إذا كانت ملاحظات التأخير "
                        "مرتبطة بأوقات أو مناطق أو "
                        "قنوات توصيل محددة."
                    )


                # --------------------------------
                # Price
                # --------------------------------

                elif aspect == "السعر":

                    recommendations.append(
                        "مراجعة تصور العملاء للقيمة مقابل السعر، "
                        "مع اختبار وضوح القيمة المقدمة "
                        "قبل اتخاذ قرار بتغيير الأسعار."
                    )

                    investigation.append(
                        "تحقق مما إذا كانت ملاحظات السعر "
                        "مرتبطة بمنتجات محددة أو بفئة معينة "
                        "من العملاء."
                    )


                # --------------------------------
                # Quality
                # --------------------------------

                elif aspect == "الجودة":

                    recommendations.append(
                        "مراجعة خطوات ضبط الجودة "
                        "وتحديد أكثر المنتجات أو الخدمات "
                        "ارتباطًا بالملاحظات السلبية."
                    )


                # --------------------------------
                # Size
                # --------------------------------

                elif aspect == "المقاس":

                    recommendations.append(
                        "مراجعة دليل المقاسات ووضوح معلومات القياس، "
                        "ومتابعة حالات الاستبدال المرتبطة بالمقاس."
                    )


                # --------------------------------
                # Service
                # --------------------------------

                elif aspect == "الخدمة":

                    recommendations.append(
                        "مراجعة نقاط التواصل مع العميل "
                        "وزمن الاستجابة وجودة التعامل "
                        "في مراحل الخدمة المختلفة."
                    )


                # --------------------------------
                # Packaging
                # --------------------------------

                elif aspect == "التغليف":

                    recommendations.append(
                        "مراجعة طريقة التغليف "
                        "والتحقق من المشكلات المتكررة "
                        "قبل تعديل مواد أو إجراءات التغليف."
                    )


                # --------------------------------
                # Digital Experience
                # --------------------------------

                elif aspect == "التجربة الرقمية":

                    recommendations.append(
                        "مراجعة خطوات استخدام التطبيق "
                        "وتحديد النقاط التي تسبب صعوبة "
                        "أو بطئًا للمستخدم."
                    )

                    investigation.append(
                        "تحقق مما إذا كانت المشكلة مرتبطة "
                        "بصفحة أو جهاز أو خطوة محددة "
                        "داخل التطبيق."
                    )


                # --------------------------------
                # Product Variety
                # --------------------------------

                elif aspect == "تنوع المنتجات":

                    recommendations.append(
                        "مراجعة تنوع التشكيلة والمنتجات "
                        "ومقارنة الخيارات المتاحة "
                        "مع أكثر ما يطلبه العملاء."
                    )

                    investigation.append(
                        "حدد المنتجات أو الفئات التي يطلب "
                        "العملاء تنوعًا أكبر فيها قبل "
                        "زيادة المخزون."
                    )


                # --------------------------------
                # Brands
                # --------------------------------

                elif aspect == "العلامات التجارية":

                    recommendations.append(
                        "مراجعة العلامات التجارية التي يهتم "
                        "بها العملاء وتحديد الأكثر ارتباطًا "
                        "بالطلب الفعلي."
                    )

                    investigation.append(
                        "تحقق من تفضيلات العملاء للعلامات "
                        "التجارية قبل توسيع التشكيلة."
                    )


    return {
        "Recommendations":
            list(
                dict.fromkeys(
                    recommendations
                )
            ),

        "Needs Investigation":
            list(
                dict.fromkeys(
                    investigation
                )
            )
    }


def render_existing_recommendations():

    result = build_existing_recommendations()

    recommendations = result[
        "Recommendations"
    ]

    investigation = result[
        "Needs Investigation"
    ]

    st.divider()

    st.header(
        " توصيات للتحسين"
    )

    st.caption(
        "هذه التوصيات ناتجة عن مؤشرات الأداء "
        "والأنماط المتكررة في آراء العملاء. "
        "هي نقاط دعم للقرار ولا تثبت سبب المشكلة بشكل مباشر."
    )

    col1, col2 = st.columns(2)


    with col1:

        st.subheader(
            " إجراءات مقترحة"
        )

        if recommendations:

            for item in recommendations:
                st.success(item)

        else:

            st.info(
                "لا توجد توصيات تلقائية إضافية "
                "ضمن المؤشرات الحالية."
            )


    with col2:

        st.subheader(
            " أمور تحتاج تحقق"
        )

        if investigation:

            for item in investigation:
                st.warning(item)

        else:

            st.write(
                "لا توجد نقاط تحقق إضافية حاليًا."
            )



# ============================================================
# UNIVERSAL NEW-BUSINESS ENGINE
# ============================================================

def get_business_family(isic_code):

    code = normalize_isic4(isic_code)

    if not code or not code[:2].isdigit():
        return "other"

    division = int(code[:2])

    if 1 <= division <= 3:
        return "agriculture"

    elif 5 <= division <= 9:
        return "mining"

    elif 10 <= division <= 33:
        return "manufacturing"

    elif 35 <= division <= 39:
        return "utilities_environment"

    elif 41 <= division <= 43:
        return "construction"

    elif 45 <= division <= 47:
        return "retail_trade"

    elif 49 <= division <= 53:
        return "transport_logistics"

    elif 55 <= division <= 56:
        return "hospitality_food"

    elif 58 <= division <= 63:
        return "information_technology"

    elif 64 <= division <= 66:
        return "financial_services"

    elif division == 68:
        return "real_estate"

    elif 69 <= division <= 75:
        return "professional_services"

    elif 77 <= division <= 82:
        return "business_support"

    elif division == 85:
        return "education"

    elif 86 <= division <= 88:
        return "health_social"

    elif 90 <= division <= 93:
        return "entertainment"

    elif 94 <= division <= 96:
        return "personal_services"

    return "other"


business_family_labels = {

    "التجارة والبيع":
        "retail_trade",

    "المطاعم والطعام والضيافة":
        "hospitality_food",

    "التصنيع":
        "manufacturing",

    "العقارات":
        "real_estate",

    "تكنولوجيا المعلومات":
        "information_technology",

    "الخدمات المهنية":
        "professional_services",

    "خدمات الأعمال":
        "business_support",

    "التعليم":
        "education",

    "الصحة والخدمات الاجتماعية":
        "health_social",

    "الخدمات الشخصية":
        "personal_services",

    "الترفيه":
        "entertainment",

    "النقل والخدمات اللوجستية":
        "transport_logistics",

    "الإنشاءات":
        "construction",

    "الخدمات المالية":
        "financial_services",

    "الزراعة":
        "agriculture",

    "التعدين":
        "mining",

    "المياه والطاقة والبيئة":
        "utilities_environment"
}


# ============================================================
# ACTIVITY CATALOG
# ============================================================

activity_catalog = (
    activities[
        [
            "ISIC Code",
            "Economic Activity"
        ]
    ]
    .dropna()
    .drop_duplicates(
        subset=["ISIC Code"]
    )
    .copy()
)

activity_catalog[
    "Business Family"
] = activity_catalog[
    "ISIC Code"
].apply(
    get_business_family
)

activity_catalog[
    "Search Text"
] = activity_catalog[
    "Economic Activity"
].astype(str)


# ============================================================
# AI SEARCH MODELS
# ============================================================

@st.cache_resource(show_spinner=False)
def load_activity_search_models():

    semantic_model = SentenceTransformer(
        "sentence-transformers/"
        "paraphrase-multilingual-MiniLM-L12-v2"
    )

    translation_name = (
        "Helsinki-NLP/opus-mt-ar-en"
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            translation_name
        )
    )

    translation_model = (
        AutoModelForSeq2SeqLM
        .from_pretrained(
            translation_name
        )
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    translation_model = (
        translation_model.to(device)
    )

    return (
        semantic_model,
        tokenizer,
        translation_model,
        device
    )


def translate_activity_query(text):

    (
        semantic_model,
        tokenizer,
        translation_model,
        device
    ) = load_activity_search_models()

    text = str(text).strip()

    if not text:
        return ""

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True
    )

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    with torch.no_grad():

        generated = (
            translation_model.generate(
                **inputs,
                max_length=128
            )
        )

    return tokenizer.decode(
        generated[0],
        skip_special_tokens=True
    )


def search_activities_streamlit(
    business_family,
    search_text,
    top_k=5
):

    search_text = str(
        search_text
    ).strip()

    if not search_text:
        return pd.DataFrame()


    (
        semantic_model,
        _,
        _,
        _
    ) = load_activity_search_models()


    translated_text = (
        translate_activity_query(
            search_text
        )
    )


    candidates = activity_catalog[
        activity_catalog[
            "Business Family"
        ] == business_family
    ].copy()


    if candidates.empty:
        return pd.DataFrame()


    documents = (
        candidates[
            "Search Text"
        ]
        .fillna("")
        .astype(str)
        .tolist()
    )


    document_vectors = (
        semantic_model.encode(
            documents,
            convert_to_tensor=True,
            normalize_embeddings=True
        )
    )


    arabic_vector = (
        semantic_model.encode(
            search_text,
            convert_to_tensor=True,
            normalize_embeddings=True
        )
    )


    english_vector = (
        semantic_model.encode(
            translated_text,
            convert_to_tensor=True,
            normalize_embeddings=True
        )
    )


    arabic_semantic = (
        util.cos_sim(
            arabic_vector,
            document_vectors
        )[0]
        .cpu()
        .numpy()
    )


    english_semantic = (
        util.cos_sim(
            english_vector,
            document_vectors
        )[0]
        .cpu()
        .numpy()
    )


    char_vectorizer = (
        TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            lowercase=True
        )
    )


    char_matrix = (
        char_vectorizer.fit_transform(
            [translated_text]
            + documents
        )
    )


    char_scores = (
        cosine_similarity(
            char_matrix[0:1],
            char_matrix[1:]
        )[0]
    )


    candidates[
        "Arabic Semantic"
    ] = arabic_semantic

    candidates[
        "English Semantic"
    ] = english_semantic

    candidates[
        "Text Match"
    ] = char_scores


    # Prototype hybrid weights
    candidates[
        "Final Search Score"
    ] = (
        0.40
        * candidates[
            "Arabic Semantic"
        ]

        + 0.30
        * candidates[
            "English Semantic"
        ]

        + 0.30
        * candidates[
            "Text Match"
        ]
    )


    return (
        candidates
        .sort_values(
            "Final Search Score",
            ascending=False
        )
        .head(top_k)[
            [
                "ISIC Code",
                "Economic Activity",
                "Business Family",
                "Final Search Score"
            ]
        ]
        .reset_index(drop=True)
    )


# ============================================================
# DEMAND MATCHING
# ============================================================

consumer_families = [
    "retail_trade",
    "hospitality_food",
    "personal_services",
    "education",
    "health_social",
    "entertainment"
]


def match_demand_for_activity(
    activity_name,
    business_family
):

    if business_family not in consumer_families:
        return None


    (
        semantic_model,
        _,
        _,
        _
    ) = load_activity_search_models()


    categories = (
        demand["Category (EN)"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


    if not categories:
        return None


    query_vector = (
        semantic_model.encode(
            str(activity_name),
            convert_to_tensor=True,
            normalize_embeddings=True
        )
    )


    demand_vectors = (
        semantic_model.encode(
            categories,
            convert_to_tensor=True,
            normalize_embeddings=True
        )
    )


    scores = util.cos_sim(
        query_vector,
        demand_vectors
    )[0]


    best_index = int(
        scores.argmax()
    )

    similarity = float(
        scores[best_index]
    )


    # Prototype confidence threshold
    if similarity < 0.45:
        return None


    return categories[
        best_index
    ]


# ============================================================
# LOCATION STRATEGY
# ============================================================

location_strategy = {

    "manufacturing":
        "cluster",

    "construction":
        "cluster",

    "transport_logistics":
        "cluster",

    "real_estate":
        "ambiguous_market",

    "retail_trade":
        "competition",

    "hospitality_food":
        "competition",

    "personal_services":
        "competition",

    "professional_services":
        "competition",

    "business_support":
        "competition",

    "education":
        "competition",

    "entertainment":
        "competition",

    "information_technology":
        "competition",

    "financial_services":
        "competition",

    "health_social":
        "competition",

    "agriculture":
        "resource_based",

    "mining":
        "resource_based",

    "utilities_environment":
        "resource_based",

    "other":
        "competition"
}


# ============================================================
# OVERRIDE LOCATION MODEL
# ============================================================

def calculate_location_scores(
    isic_code,
    demand_category=None,
    business_family=None
):

    if business_family is None:

        business_family = (
            get_business_family(
                isic_code
            )
        )


    business = activities[
        activities[
            "ISIC Code"
        ] == str(isic_code)
    ]


    if business.empty:

        raise ValueError(
            "لم يتم العثور على النشاط "
            "في بيانات المنشآت."
        )


    business = business.iloc[0]


    strategy = (
        location_strategy.get(
            business_family,
            "competition"
        )
    )


    demand_row = None

    if demand_category:

        match = demand[
            demand[
                "Category (EN)"
            ] == demand_category
        ]

        if not match.empty:
            demand_row = (
                match.iloc[0]
            )


    rows = []


    for (
        arabic_name,
        english_name
    ) in governorate_map.items():


        gov_match = governorates[
            governorates[
                "Governorate"
            ] == english_name
        ]


        if gov_match.empty:
            continue


        gov_row = (
            gov_match.iloc[0]
        )


        population = float(
            gov_row[
                "Population 2025"
            ]
        )


        registration_growth = float(
            gov_row[
                "H1 Growth 2023→2024"
            ]
        )


        similar_businesses = (
            pd.to_numeric(
                business.get(
                    english_name,
                    0
                ),
                errors="coerce"
            )
        )


        if pd.isna(
            similar_businesses
        ):
            similar_businesses = 0


        similar_businesses = float(
            similar_businesses
        )


        businesses_per_10k = (
            similar_businesses
            / population
            * 10000
        )


        if demand_row is not None:

            historical_demand = (
                pd.to_numeric(
                    demand_row.get(
                        arabic_name,
                        np.nan
                    ),
                    errors="coerce"
                )
            )

        else:

            historical_demand = (
                np.nan
            )


        rows.append({

            "Governorate":
                english_name,

            "Population":
                population,

            "Historical Demand":
                historical_demand,

            "Similar Businesses":
                similar_businesses,

            "Businesses per 10k":
                businesses_per_10k,

            "Registration Growth":
                registration_growth
        })


    result = pd.DataFrame(
        rows
    )


    # ----------------------------------------
    # Market
    # ----------------------------------------

    result[
        "Log Population"
    ] = np.log1p(
        result[
            "Population"
        ]
    )


    result[
        "Market Score"
    ] = normalize(
        result[
            "Log Population"
        ]
    )


    # ----------------------------------------
    # Historical demand
    # ----------------------------------------

    if result[
        "Historical Demand"
    ].notna().any():

        result[
            "Demand Score"
        ] = normalize(
            result[
                "Historical Demand"
            ]
        )

    else:

        result[
            "Demand Score"
        ] = np.nan


    # ----------------------------------------
    # Similar-business indicator
    # ----------------------------------------

    result[
        "Sector Activity Score"
    ] = normalize(
        result[
            "Businesses per 10k"
        ]
    )


    if strategy == "cluster":

        result[
            "Sector Score"
        ] = result[
            "Sector Activity Score"
        ]


    elif strategy == "competition":

        result[
            "Sector Score"
        ] = normalize(
            result[
                "Businesses per 10k"
            ],
            reverse=True
        )


    else:

        # Real estate / resource-based:
        # keep for context only.
        result[
            "Sector Score"
        ] = np.nan


    # ----------------------------------------
    # Trend
    # ----------------------------------------

    result[
        "Trend Score"
    ] = normalize(
        result[
            "Registration Growth"
        ]
    )


    # ----------------------------------------
    # Final supported components only
    # ----------------------------------------

    components = [
        "Market Score",
        "Trend Score"
    ]


    if result[
        "Demand Score"
    ].notna().any():

        components.append(
            "Demand Score"
        )


    if (
        strategy
        in [
            "cluster",
            "competition"
        ]
        and result[
            "Sector Score"
        ].notna().any()
    ):

        components.append(
            "Sector Score"
        )


    result[
        "Location Suitability Score"
    ] = (
        result[
            components
        ].mean(
            axis=1
        )
    )


    result = (
        result
        .sort_values(
            "Location Suitability Score",
            ascending=False
        )
        .reset_index(drop=True)
    )


    result["Rank"] = (
        result.index + 1
    )

    result["Strategy"] = (
        strategy
    )

    result[
        "Components Used"
    ] = ", ".join(
        components
    )


    return (
        result,
        components
    )



# ============================================
# SESSION STATE
# ============================================

if "page" not in st.session_state:
    st.session_state.page = "landing"

if "new_project" not in st.session_state:
    st.session_state.new_project = {}


# ============================================
# HOME
# ============================================

def _show_logo_if_available():

    logo_candidates = [
        "bunya_logo.png",
        "/content/bunya_logo.png"
    ]

    logo_path = next(
        (
            path
            for path in logo_candidates
            if os.path.exists(path)
        ),
        None
    )

    if logo_path:

        left, center, right = st.columns(
            [1.35, 1, 1.35]
        )

        with center:
            st.image(
                logo_path,
                use_container_width=True
            )


def show_landing():

    st.markdown(
        '<div class="bunya-landing-shell">',
        unsafe_allow_html=True
    )

    _show_logo_if_available()

    st.markdown(
        """
        <div class="bunya-landing">
            <div class="bunya-brand">بينه</div>
            <div class="bunya-tagline">
                منصة ذكية لدعم قرارات المشاريع الصغيرة والمتوسطة
            </div>
            <div class="bunya-subtagline">
                من الفكرة إلى القرار، ومن التشغيل إلى متابعة الأداء
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    left, center, right = st.columns(
        [1.4, 1, 1.4]
    )

    with center:
        if st.button(
            "ابدأ",
            use_container_width=True,
            type="primary",
            key="landing_start"
        ):
            st.session_state.page = "home"
            st.rerun()

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


def show_home():

    st.markdown(
        """
        <div class="bunya-choice-header">
            <div class="bunya-eyebrow">اختر مسار العمل</div>
            <h1>كيف تريد استخدام بينه اليوم؟</h1>
            <p>
                اختر المسار المناسب، ويمكنك العودة لهذه الصفحة في أي وقت.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(
        2,
        gap="large"
    )

    with col1:

        st.markdown(
            """
            <div class="bunya-choice-card bunya-choice-new">
                <div class="bunya-card-kicker">قبل الإطلاق</div>
                <h3>بناء مشروع جديد</h3>
                <p>
                    ابدأ من النشاط، ثم قيّم الموقع ورأس المال والمخاطر
                    والعميل المستهدف قبل اتخاذ قرار الإطلاق.
                </p>
                <div class="bunya-card-line"></div>
                <span>تحليل منظم خطوة بخطوة</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button(
            "ابدأ بناء مشروع جديد",
            use_container_width=True,
            type="primary",
            key="open_new_business"
        ):
            st.session_state.page = "new"
            st.rerun()

    with col2:

        st.markdown(
            """
            <div class="bunya-choice-card bunya-choice-existing">
                <div class="bunya-card-kicker">بعد التشغيل</div>
                <h3>متابعة مشروع قائم</h3>
                <p>
                    أدخل بيانات الأداء الشهرية، راقب التغيرات،
                    حلّل آراء العملاء، ثم راجع التنبيهات والتوصيات.
                </p>
                <div class="bunya-card-line"></div>
                <span>متابعة أداء وقراءة إشارات التغير</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button(
            "افتح مشروعًا قائمًا",
            use_container_width=True,
            key="open_existing_business"
        ):
            st.session_state.page = "existing"
            st.rerun()

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    left, center, right = st.columns([1.6, 1, 1.6])

    with center:
        if st.button(
            "العودة إلى الواجهة الرئيسية",
            use_container_width=True,
            key="back_to_landing"
        ):
            st.session_state.page = "landing"
            st.rerun()


# ============================================
# NEW BUSINESS HELPERS
# ============================================

def _render_new_project_summary(project):

    st.markdown(
        '<div class="bunya-section-note">'
        '<b>ملخص المشروع الحالي</b>'
        '</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "ISIC",
        project["ISIC Code"]
    )

    c2.metric(
        "المحافظة",
        project["Preferred Governorate"]
    )

    c3.metric(
        "الخطة",
        project["Plan"]
    )

    st.caption(
        f"النشاط: {project['Economic Activity']}"
    )


def _render_new_project_input():

    st.subheader(
        "بيانات المشروع"
    )

    st.caption(
        "ابدأ بتحديد القطاع ووصف النشاط، ثم اختر أقرب نشاط اقتصادي."
    )

    row1_col1, row1_col2 = st.columns(
        [1.25, 1],
        gap="large"
    )

    with row1_col1:

        sector_label = st.selectbox(
            "القطاع الرئيسي",
            list(
                business_family_labels.keys()
            )
        )

    with row1_col2:

        plan = st.radio(
            "نوع التحليل",
            ["Free", "Premium"],
            horizontal=True
        )

    selected_family = (
        business_family_labels[
            sector_label
        ]
    )

    activity_query = st.text_input(
        "اكتب وصف المشروع أو النشاط",
        placeholder=(
            "مثال: محل ملابس، شركة برمجيات، "
            "مكتب عقاري، مركز تدريب..."
        )
    )

    if st.button(
        "بحث عن النشاط",
        use_container_width=True,
        key="activity_search_button"
    ):

        if not activity_query.strip():

            st.warning(
                "اكتب وصف النشاط أولًا."
            )

        else:

            with st.spinner(
                "جاري البحث عن الأنشطة الأقرب..."
            ):

                results = (
                    search_activities_streamlit(
                        selected_family,
                        activity_query,
                        top_k=5
                    )
                )

                st.session_state[
                    "activity_search_results"
                ] = results


    activity_results = (
        st.session_state.get(
            "activity_search_results",
            pd.DataFrame()
        )
    )

    selected_activity = None

    if (
        isinstance(
            activity_results,
            pd.DataFrame
        )
        and not activity_results.empty
    ):

        with st.expander(
            "الأنشطة المقترحة",
            expanded=True
        ):

            show_results = (
                activity_results[
                    [
                        "ISIC Code",
                        "Economic Activity",
                        "Final Search Score"
                    ]
                ]
                .copy()
            )

            show_results[
                "Final Search Score"
            ] = show_results[
                "Final Search Score"
            ].round(3)

            st.dataframe(
                show_results,
                use_container_width=True,
                hide_index=True
            )

            activity_options = {}

            for _, row in (
                activity_results
                .iterrows()
            ):

                label = (
                    f"{row['ISIC Code']} - "
                    f"{row['Economic Activity']}"
                )

                activity_options[
                    label
                ] = row[
                    "ISIC Code"
                ]

            selected_label = (
                st.selectbox(
                    "اختر النشاط الأنسب",
                    list(
                        activity_options.keys()
                    )
                )
            )

            selected_code = (
                activity_options[
                    selected_label
                ]
            )

            selected_row = (
                activity_results[
                    activity_results[
                        "ISIC Code"
                    ] == selected_code
                ]
                .iloc[0]
            )

            selected_activity = {

                "ISIC Code":
                    str(
                        selected_row[
                            "ISIC Code"
                        ]
                    ),

                "Economic Activity":
                    selected_row[
                        "Economic Activity"
                    ],

                "Business Family":
                    selected_row[
                        "Business Family"
                    ]
            }

            st.success(
                "تم اختيار النشاط: "
                + str(
                    selected_activity[
                        "Economic Activity"
                    ]
                )
            )


    st.markdown("#### إعدادات المشروع")

    col1, col2 = st.columns(
        2,
        gap="large"
    )

    with col1:

        governorate = st.selectbox(
            "المحافظة المفضلة",
            list(
                governorate_map.values()
            ),
            index=4
        )

        operating_model = st.selectbox(
            "نموذج التشغيل",
            [
                "small_physical",
                "online",
                "hybrid",
                "home_based"
            ]
        )

    with col2:

        can_relocate = st.radio(
            "هل يمكن تغيير الموقع؟",
            ["نعم", "لا"],
            horizontal=True
        )

        available_capital = (
            st.number_input(
                "رأس المال المتوفر - اختياري (JD)",
                min_value=0.0,
                value=0.0,
                step=100.0
            )
        )


    if st.button(
        "حفظ وتحليل المشروع",
        use_container_width=True,
        type="primary"
    ):

        if selected_activity is None:

            st.warning(
                "ابحث عن النشاط واختر النشاط "
                "الأنسب أولًا."
            )

        else:

            st.session_state.new_project = {

                "ISIC Code":
                    selected_activity[
                        "ISIC Code"
                    ],

                "Economic Activity":
                    selected_activity[
                        "Economic Activity"
                    ],

                "Business Family":
                    selected_activity[
                        "Business Family"
                    ],

                "Preferred Governorate":
                    governorate,

                "Can Relocate":
                    can_relocate == "نعم",

                "Operating Model":
                    operating_model,

                "Available Capital":
                    available_capital
                    if available_capital > 0
                    else None,

                "Plan":
                    plan
            }

            # Clear dependent results from an older project.
            for key in [
                "location_summary",
                "capital_result",
                "prelaunch_risk_summary"
            ]:
                st.session_state.pop(
                    key,
                    None
                )

            st.success(
                "تم حفظ بيانات المشروع. "
                "يمكنك الآن فتح بقية الأقسام من شريط التنقل."
            )


def _get_new_location_context(project):

    demand_category = (
        match_demand_for_activity(
            project[
                "Economic Activity"
            ],
            project[
                "Business Family"
            ]
        )
    )

    location_df, components = (
        calculate_location_scores(
            project[
                "ISIC Code"
            ],
            demand_category,
            project[
                "Business Family"
            ]
        )
    )

    selected = location_df[
        location_df["Governorate"]
        == project["Preferred Governorate"]
    ].iloc[0]

    top = location_df.iloc[0]

    st.session_state.location_summary = {
        "Governorate":
            selected["Governorate"],

        "Rank":
            int(selected["Rank"]),

        "Score":
            float(
                selected[
                    "Location Suitability Score"
                ]
            ),

        "Top Governorate":
            top["Governorate"],

        "Total Governorates":
            len(location_df),

        "Demand Category":
            demand_category
    }

    return (
        demand_category,
        location_df,
        components,
        selected,
        top
    )


def _render_new_location(project):

    st.subheader(
        "تحليل ملاءمة الموقع"
    )

    st.caption(
        "مؤشر ملاءمة الموقع هو Screening Score "
        "لدعم المقارنة بين المحافظات، وليس احتمالًا لنجاح المشروع."
    )

    try:

        (
            demand_category,
            location_df,
            components,
            selected,
            top
        ) = _get_new_location_context(
            project
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "ملاءمة الموقع",
            f"{selected['Location Suitability Score']:.1f}",
            help="النتيجة من 100"
        )

        c2.metric(
            "ترتيب المحافظة",
            f"{int(selected['Rank'])}",
            help=(
                f"من أصل "
                f"{len(location_df)} محافظة"
            )
        )

        c3.metric(
            "أعلى محافظة",
            top["Governorate"]
        )

        if project["Plan"] == "Premium":

            col1, col2 = st.columns(
                2,
                gap="large"
            )

            with col1:

                st.markdown(
                    "#### أفضل المحافظات حسب المؤشر"
                )

                display_df = location_df[
                    [
                        "Rank",
                        "Governorate",
                        "Location Suitability Score"
                    ]
                ].head(5).copy()

                display_df[
                    "Location Suitability Score"
                ] = display_df[
                    "Location Suitability Score"
                ].round(1)

                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )

            with col2:

                st.markdown(
                    "#### مكونات تحليل المحافظة المختارة"
                )

                component_cols = [
                    col
                    for col in components
                    if (
                        col in selected.index
                        and pd.notna(
                            selected[col]
                        )
                    )
                ]

                component_data = pd.DataFrame({
                    "المؤشر":
                        component_cols,

                    "النتيجة":
                        [
                            round(
                                float(
                                    selected[col]
                                ),
                                1
                            )
                            for col
                            in component_cols
                        ]
                })

                st.dataframe(
                    component_data,
                    use_container_width=True,
                    hide_index=True
                )

            if demand_category is not None:

                st.info(
                    f"مؤشر الطلب المستخدم يعتمد على "
                    f"فئة الإنفاق الأسري التاريخية "
                    f"«{demand_category}» من HEIS 2017، "
                    f"ولا يمثل الطلب الحالي بشكل مباشر."
                )

            if (
                selected["Governorate"]
                != top["Governorate"]
            ):

                gap = (
                    top[
                        "Location Suitability Score"
                    ]
                    -
                    selected[
                        "Location Suitability Score"
                    ]
                )

                st.write(
                    f"أعلى نتيجة كانت في "
                    f"**{top['Governorate']}** "
                    f"بفارق **{gap:.1f} نقطة** "
                    f"عن المحافظة المختارة."
                )

    except Exception as e:

        st.error(
            f"تعذر إجراء تحليل الموقع: {e}"
        )


def show_new_business():

    top1, top2 = st.columns(
        [1, 4]
    )

    with top1:

        if st.button(
            "العودة للرئيسية",
            use_container_width=True
        ):
            st.session_state.page = "home"
            st.rerun()

    with top2:
        st.title(
            "مشروع جديد"
        )

    nav = st.radio(
        "أقسام المشروع الجديد",
        [
            "بيانات المشروع",
            "تحليل الموقع",
            "رأس المال",
            "المخاطر",
            "العميل والتوصيات"
        ],
        horizontal=True,
        label_visibility="collapsed",
        key="new_business_nav"
    )

    project = (
        st.session_state
        .get(
            "new_project",
            {}
        )
    )

    if project:
        _render_new_project_summary(
            project
        )

    if nav == "بيانات المشروع":

        _render_new_project_input()

    elif not project:

        st.info(
            "احفظ بيانات المشروع أولًا من قسم "
            "«بيانات المشروع» ثم افتح هذا القسم."
        )

    elif nav == "تحليل الموقع":

        _render_new_location(
            project
        )

    elif nav == "رأس المال":

        render_capital_planner(
            project
        )

    elif nav == "المخاطر":

        if not st.session_state.get(
            "location_summary"
        ):
            try:
                _get_new_location_context(
                    project
                )
            except Exception:
                pass

        render_prelaunch_risks(
            project
        )

    elif nav == "العميل والتوصيات":

        render_customer_and_recommendations(
            project
        )


# ============================================
# EXISTING BUSINESS HELPERS
# ============================================

def _render_existing_month_entry():

    st.subheader(
        "إضافة أو تحديث شهر"
    )

    st.caption(
        "أدخل بيانات الفترة الشهرية. "
        "إذا كررت اسم الشهر سيتم تحديثه بدل تكراره."
    )

    with st.form(
        "existing_month_form",
        clear_on_submit=False
    ):

        month_name = st.text_input(
            "اسم الشهر",
            placeholder="مثال: September 2026"
        )

        c1, c2 = st.columns(
            2,
            gap="large"
        )

        with c1:

            revenue = st.number_input(
                "الإيرادات (JD)",
                min_value=0.0,
                value=0.0,
                step=50.0
            )

            expenses = st.number_input(
                "المصروفات (JD)",
                min_value=0.0,
                value=0.0,
                step=50.0
            )

            marketing = st.number_input(
                "الإنفاق التسويقي (JD)",
                min_value=0.0,
                value=0.0,
                step=20.0
            )

            customers = st.number_input(
                "عدد العملاء",
                min_value=0,
                value=0,
                step=1
            )

        with c2:

            transactions = st.number_input(
                "عدد العمليات / المبيعات",
                min_value=0,
                value=0,
                step=1
            )

            employees = st.number_input(
                "عدد الموظفين",
                min_value=0,
                value=0,
                step=1
            )

            complaints = st.number_input(
                "عدد الشكاوى",
                min_value=0,
                value=0,
                step=1
            )

            rating = st.number_input(
                "تقييم العملاء",
                min_value=0.0,
                max_value=5.0,
                value=0.0,
                step=0.1
            )

        submitted = (
            st.form_submit_button(
                "حفظ الشهر",
                use_container_width=True,
                type="primary"
            )
        )

    if submitted:

        if not month_name.strip():

            st.warning(
                "اكتب اسم الشهر أولًا."
            )

        elif (
            revenue == 0
            and expenses == 0
            and customers == 0
            and transactions == 0
        ):

            st.warning(
                "أدخل بيانات فعلية للشهر قبل الحفظ."
            )

        else:

            snapshot = {
                "Revenue":
                    revenue,

                "Expenses":
                    expenses,

                "Marketing Spend":
                    marketing,

                "Customers":
                    customers,

                "Transactions":
                    transactions,

                "Employees":
                    employees,

                "Complaints":
                    complaints,

                "Customer Rating":
                    rating
            }

            save_existing_month(
                month_name.strip(),
                snapshot
            )

            st.success(
                f"تم حفظ {month_name.strip()}."
            )


    history = st.session_state.get(
        "performance_history",
        []
    )

    if history:

        with st.expander(
            "عرض الأشهر المحفوظة",
            expanded=False
        ):

            df = pd.DataFrame(
                history
            )

            display_columns = [
                "Month",
                "Revenue",
                "Expenses",
                "Profit",
                "Customers",
                "Transactions",
                "Complaints",
                "Customer Rating"
            ]

            st.dataframe(
                df[
                    display_columns
                ].round(2),
                use_container_width=True,
                hide_index=True
            )


def _render_existing_performance():

    st.subheader(
        "لوحة الأداء"
    )

    history = (
        st.session_state.get(
            "performance_history",
            []
        )
    )

    if len(history) < 2:

        st.info(
            "أدخل بيانات شهرين على الأقل "
            "حتى تظهر المقارنة وقراءة الأداء."
        )

        return

    df = pd.DataFrame(
        history
    )

    previous = df.iloc[-2]
    latest = df.iloc[-1]

    st.caption(
        f"{latest['Month']} مقارنة مع "
        f"{previous['Month']}"
    )

    revenue_change = safe_percent_change(
        latest["Revenue"],
        previous["Revenue"]
    )

    profit_change = safe_percent_change(
        latest["Profit"],
        previous["Profit"]
    )

    customer_change = safe_percent_change(
        latest["Customers"],
        previous["Customers"]
    )

    transaction_change = safe_percent_change(
        latest["Transactions"],
        previous["Transactions"]
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "الإيرادات",
        f"{latest['Revenue']:,.0f} JD",
        None
        if revenue_change is None
        else f"{revenue_change:.1f}%"
    )

    c2.metric(
        "الربح",
        f"{latest['Profit']:,.0f} JD",
        None
        if profit_change is None
        else f"{profit_change:.1f}%"
    )

    c3.metric(
        "العملاء",
        f"{int(latest['Customers'])}",
        None
        if customer_change is None
        else f"{customer_change:.1f}%"
    )

    c4.metric(
        "العمليات",
        f"{int(latest['Transactions'])}",
        None
        if transaction_change is None
        else f"{transaction_change:.1f}%"
    )

    insights = analyze_existing_changes(
        previous,
        latest
    )

    st.session_state[
        "existing_base_insights"
    ] = insights

    st.markdown(
        "#### قراءة الأداء"
    )

    st.caption(
        "التنبيهات التالية تعتمد على حدود تشغيلية "
        "شفافة في النسخة التجريبية، وهي قابلة للتعديل "
        "وليست حدودًا علمية ثابتة لجميع المشاريع."
    )

    col1, col2, col3 = st.columns(
        3,
        gap="medium"
    )

    with col1:

        st.markdown(
            "##### تنبيهات ذات أولوية"
        )

        if insights["warnings"]:

            for item in insights[
                "warnings"
            ]:
                st.error(item)

        else:

            st.success(
                "لا توجد إشارات ذات أولوية مرتفعة "
                "ضمن التغيرات الحالية."
            )

    with col2:

        st.markdown(
            "##### سياق الأداء"
        )

        if insights[
            "observations"
        ]:

            for item in insights[
                "observations"
            ]:
                st.info(item)

        else:

            st.write(
                "لا توجد ملاحظات سياقية إضافية حاليًا."
            )

    with col3:

        st.markdown(
            "##### إشارات إيجابية"
        )

        if insights[
            "positives"
        ]:

            for item in insights[
                "positives"
            ]:
                st.success(item)

        else:

            st.write(
                "لا توجد إشارات إيجابية إضافية حاليًا."
            )


def show_existing_business():

    top1, top2 = st.columns(
        [1, 4]
    )

    with top1:

        if st.button(
            "العودة للرئيسية",
            use_container_width=True
        ):
            st.session_state.page = "home"
            st.rerun()

    with top2:

        st.title(
            "مشروع قائم"
        )

    st.caption(
        "أدخل بيانات الأداء الشهرية، ثم انتقل بين "
        "الأقسام لمراجعة الأداء وآراء العملاء والتوصيات."
    )

    nav = st.radio(
        "أقسام المشروع القائم",
        [
            "البيانات الشهرية",
            "لوحة الأداء",
            "آراء العملاء",
            "التوصيات"
        ],
        horizontal=True,
        label_visibility="collapsed",
        key="existing_business_nav"
    )

    if nav == "البيانات الشهرية":

        _render_existing_month_entry()

    elif nav == "لوحة الأداء":

        _render_existing_performance()

    elif nav == "آراء العملاء":

        render_customer_review_nlp()

    elif nav == "التوصيات":

        render_existing_recommendations()

        render_final_integrated_insights()


# ============================================
# ROUTER
# ============================================

if st.session_state.page == "landing":
    show_landing()

elif st.session_state.page == "home":
    show_home()

elif st.session_state.page == "new":
    show_new_business()

elif st.session_state.page == "existing":
    show_existing_business()

else:
    st.session_state.page = "landing"
    st.rerun()
