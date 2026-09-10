# ============================================================
# FOREST FIRE RISK PREDICTION
# Streamlit Application
# ============================================================

import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# ============================================================
# LOAD FINAL MODEL
# ============================================================

model_config = joblib.load("forest_fire_final_model.pkl")

model = model_config["model"]
features = model_config["features"]
threshold = model_config["threshold"]


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Forest Fire Risk Prediction",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# RESET INPUTS
# ============================================================

def reset_inputs():
    st.session_state["precmax"] = 0.0
    st.session_state["precsum"] = 0.0
    st.session_state["precmin"] = 0.0
    st.session_state["ndvimedian"] = 0.0
    st.session_state["ndwimin"] = 0.0
    st.session_state["days"] = 0.0
    st.session_state["latitude"] = 0.0
    st.session_state["slope"] = 0.0
    st.session_state["date_input"] = ""


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

.stApp {
    min-height: 100vh;

    background:
        linear-gradient(
            rgba(5, 35, 45, 0.62),
            rgba(5, 65, 55, 0.70)
        ),
        url("https://images.unsplash.com/photo-1448375240586-882707db888b?auto=format&fit=crop&w=2400&q=85");

    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}


.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}


/* ============================================================
   HERO
   ============================================================ */

.hero-box {
    background:
        linear-gradient(
            135deg,
            rgba(4, 47, 70, 0.96),
            rgba(5, 105, 95, 0.92)
        );

    border-radius: 28px;

    padding: 34px 42px;

    margin-bottom: 30px;

    border: 1px solid rgba(255,255,255,0.25);

    box-shadow:
        0 20px 50px rgba(0,0,0,0.30);

    backdrop-filter: blur(10px);
}


.hero-title {
    color: #ffffff !important;

    font-size: 44px;

    font-weight: 850;

    line-height: 1.15;
}


.hero-subtitle {
    color: #dff6ff !important;

    font-size: 18px;

    font-weight: 600;

    line-height: 1.6;

    margin-top: 10px;
}


/* ============================================================
   SECTION TITLES
   ============================================================ */

.section-title {
    color: #ffffff !important;

    font-size: 28px;

    font-weight: 850;

    margin-top: 20px;

    margin-bottom: 20px;

    text-shadow:
        0 2px 8px rgba(0,0,0,0.70);
}


/* ============================================================
   INPUT CONTAINERS
   ============================================================ */

div[data-testid="stNumberInput"],
div[data-testid="stTextInput"] {
    background:
        rgba(255,255,255,0.14) !important;

    padding:
        9px 12px 8px 12px !important;

    border-radius:
        16px !important;

    border:
        1px solid rgba(186,230,253,0.45) !important;

    backdrop-filter:
        blur(8px);

    box-shadow:
        0 8px 20px rgba(0,0,0,0.14);
}


/* ============================================================
   INPUT LABELS
   ============================================================ */

div[data-testid="stNumberInput"] label,
div[data-testid="stTextInput"] label,
div[data-testid="stDateInput"] label {
    color: #ffffff !important;

    font-size: 16px !important;

    font-weight: 850 !important;

    opacity: 1 !important;

    text-shadow:
        0 2px 5px rgba(0,0,0,0.85) !important;
}


/* ============================================================
   INPUT FIELDS
   ============================================================ */

div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input {
    background:
        linear-gradient(
            135deg,
            rgba(248,250,252,0.98),
            rgba(224,242,254,0.97)
        ) !important;

    color: #082f49 !important;

    border:
        2px solid rgba(125,211,252,0.60) !important;

    border-radius:
        12px !important;

    font-size:
        17px !important;

    font-weight:
        750 !important;

    min-height:
        42px !important;

    box-shadow:
        0 5px 15px rgba(0,0,0,0.16) !important;
}


/* ============================================================
   INPUT FOCUS
   ============================================================ */

div[data-testid="stNumberInput"] input:focus,
div[data-testid="stTextInput"] input:focus {
    border:
        2px solid #38bdf8 !important;

    box-shadow:
        0 0 0 3px rgba(56,189,248,0.25),
        0 8px 20px rgba(0,0,0,0.18) !important;
}


/* ============================================================
   PLUS / MINUS
   ============================================================ */

div[data-testid="stNumberInput"] button {
    background:
        #082f49 !important;

    color:
        #ffffff !important;

    border:
        none !important;
}


div[data-testid="stNumberInput"] button:hover {
    background:
        #0c4a6e !important;
}


/* ============================================================
   RANGE TEXT
   ============================================================ */

.range-text {
    color:
        #bae6fd !important;

    font-size:
        12px !important;

    font-weight:
        750 !important;

    margin-top:
        -5px;

    margin-bottom:
        14px;

    text-shadow:
        0 2px 5px rgba(0,0,0,0.75);
}


/* ============================================================
   EXPANDER
   ============================================================ */

div[data-testid="stExpander"] {
    background:
        rgba(255,255,255,0.94);

    border:
        1px solid rgba(255,255,255,0.65);

    border-radius:
        16px;
}


div[data-testid="stExpander"] p,
div[data-testid="stExpander"] li,
div[data-testid="stExpander"] h1,
div[data-testid="stExpander"] h2,
div[data-testid="stExpander"] h3,
div[data-testid="stExpander"] strong {
    color:
        #082f49 !important;
}


/* ============================================================
   BUTTONS
   ============================================================ */

div.stButton > button {
    width:
        100%;

    min-height:
        54px;

    padding:
        13px 22px;

    background:
        linear-gradient(
            90deg,
            #dc2626,
            #f97316
        );

    color:
        #ffffff;

    border:
        none;

    border-radius:
        14px;

    font-size:
        18px;

    font-weight:
        850;

    box-shadow:
        0 10px 28px rgba(220,38,38,0.32);

    transition:
        all 0.2s ease;
}


div.stButton > button:hover {
    transform:
        translateY(-2px);

    box-shadow:
        0 15px 35px rgba(249,115,22,0.42);
}


/* ============================================================
   RISK CARD
   ============================================================ */

.result-card {
    border-radius:
        26px;

    padding:
        34px;

    margin-top:
        20px;

    margin-bottom:
        25px;

    text-align:
        center;

    box-shadow:
        0 20px 45px rgba(0,0,0,0.28);

    backdrop-filter:
        blur(8px);
}


.low-risk {
    background:
        linear-gradient(
            135deg,
            rgba(5,150,105,0.94),
            rgba(16,185,129,0.65)
        );

    border:
        1px solid rgba(167,243,208,0.45);
}


.medium-risk {
    background:
        linear-gradient(
            135deg,
            rgba(180,83,9,0.94),
            rgba(245,158,11,0.68)
        );

    border:
        1px solid rgba(254,215,170,0.45);
}


.high-risk {
    background:
        linear-gradient(
            135deg,
            rgba(185,28,28,0.95),
            rgba(239,68,68,0.70)
        );

    border:
        1px solid rgba(254,202,202,0.45);
}


.risk-icon {
    font-size:
        52px;

    margin-bottom:
        8px;
}


.risk-title {
    color:
        #ffffff !important;

    font-size:
        30px;

    font-weight:
        850;
}


.risk-description {
    color:
        #ffffff !important;

    font-size:
        17px;

    margin-top:
        12px;
}


/* ============================================================
   METRIC CARDS
   ============================================================ */

.metric-card {
    background:
        rgba(255,255,255,0.92);

    border:
        1px solid rgba(255,255,255,0.70);

    border-radius:
        20px;

    padding:
        22px;

    text-align:
        center;

    box-shadow:
        0 10px 30px rgba(15,23,42,0.16);

    backdrop-filter:
        blur(8px);
}


.metric-label {
    color:
        #075985 !important;

    font-size:
        13px;

    font-weight:
        750;
}


.metric-value {
    color:
        #052f4a !important;

    font-size:
        27px;

    font-weight:
        850;

    margin-top:
        7px;
}


/* ============================================================
   FOOTER
   ============================================================ */

.footer {
    text-align:
        center;

    color:
        #ffffff !important;

    padding:
        30px 0 10px 0;

    font-size:
        13px;

    font-weight:
        650;

    text-shadow:
        0 2px 5px rgba(0,0,0,0.50);
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# HERO HEADER
# ============================================================

hero_html = (
    '<div class="hero-box">'
    '<div class="hero-title">🔥 Forest Fire Risk Prediction</div>'
    '<div class="hero-subtitle">'
    'نظام ذكي للتنبؤ بخطر حرائق الغابات باستخدام Machine Learning'
    '</div>'
    '</div>'
)

st.markdown(
    hero_html,
    unsafe_allow_html=True
)


# ============================================================
# MAIN SECTION
# ============================================================

st.markdown(
    '<div class="section-title">🌍 الظروف البيئية والجغرافية</div>',
    unsafe_allow_html=True
)


# ============================================================
# INPUT COLUMNS
# ============================================================

col1, col2 = st.columns(2)


# ============================================================
# LEFT COLUMN
# ============================================================

with col1:

    precmax = st.number_input(
        "🌧️ Maximum Precipitation | أقصى كمية هطول",
        value=0.0,
        step=0.1,
        format="%.2f",
        key="precmax"
    )

    st.markdown(
        '<div class="range-text">🌐 Global Range: 0+ mm</div>',
        unsafe_allow_html=True
    )


    precsum = st.number_input(
        "🌧️ Total Precipitation | إجمالي الهطول",
        value=0.0,
        step=0.1,
        format="%.2f",
        key="precsum"
    )

    st.markdown(
        '<div class="range-text">🌐 Global Range: 0+ mm</div>',
        unsafe_allow_html=True
    )


    precmin = st.number_input(
        "💧 Minimum Precipitation | أقل كمية هطول",
        value=0.0,
        step=0.1,
        format="%.2f",
        key="precmin"
    )

    st.markdown(
        '<div class="range-text">🌐 Global Range: 0+ mm</div>',
        unsafe_allow_html=True
    )


    ndvimedian = st.number_input(
        "🌿 NDVI Median | الوسيط لمؤشر الغطاء النباتي",
        value=0.0,
        step=0.01,
        format="%.4f",
        key="ndvimedian"
    )

    st.markdown(
        '<div class="range-text">🌐 Global Range: -1.00 – +1.00 Index</div>',
        unsafe_allow_html=True
    )


# ============================================================
# RIGHT COLUMN
# ============================================================

with col2:

    ndwimin = st.number_input(
        "💦 NDWI Minimum | أقل قيمة لمؤشر المياه",
        value=0.0,
        step=0.01,
        format="%.4f",
        key="ndwimin"
    )

    st.markdown(
        '<div class="range-text">🌐 Global Range: -1.00 – +1.00 Index</div>',
        unsafe_allow_html=True
    )


    days = st.number_input(
        "📅 Days | عدد الأيام",
        value=0.0,
        step=1.0,
        format="%.0f",
        key="days"
    )

    st.markdown(
        '<div class="range-text">🌐 Global Range: 0+ Day</div>',
        unsafe_allow_html=True
    )


    latitude = st.number_input(
        "📍 Latitude | خط العرض",
        value=0.0,
        step=0.01,
        format="%.6f",
        key="latitude"
    )

    st.markdown(
        '<div class="range-text">🌐 Global Range: -90° – +90°</div>',
        unsafe_allow_html=True
    )


    slope = st.number_input(
        "⛰️ Slope | درجة انحدار الأرض",
        value=0.0,
        step=0.1,
        format="%.3f",
        key="slope"
    )

    st.markdown(
        '<div class="range-text">🌐 Global Range: 0° – 90°</div>',
        unsafe_allow_html=True
    )


# ============================================================
# DATE SECTION
# ============================================================

st.markdown(
    '<div class="section-title">📅 التاريخ</div>',
    unsafe_allow_html=True
)


date_input = st.text_input(
    "📅 Date | التاريخ",
    value="",
    placeholder="YYYY-MM-DD",
    key="date_input"
)


st.markdown(
    '<div class="range-text">🌐 Global Range: Any valid date | أي تاريخ صالح</div>',
    unsafe_allow_html=True
)


# ============================================================
# INPUT GUIDE
# ============================================================

with st.expander(
    "ℹ️ Input Guide | شرح المدخلات والوحدات"
):

    st.markdown(
        """
### 🌧️ Maximum Precipitation
أقصى كمية هطول خلال الفترة.

**Unit:** mm

---

### 🌧️ Total Precipitation
إجمالي كمية الهطول خلال الفترة.

**Unit:** mm

---

### 💧 Minimum Precipitation
أقل كمية هطول خلال الفترة.

**Unit:** mm

---

### 🌿 NDVI Median
مؤشر مرتبط بحالة وكثافة الغطاء النباتي.

**Unit:** Index

**General Range:** -1 إلى +1

---

### 💦 NDWI Minimum
مؤشر مرتبط بالمياه والرطوبة.

**Unit:** Index

**General Range:** -1 إلى +1

---

### 📅 Days
عدد الأيام المرتبط بالملاحظة.

**Unit:** Day

---

### 📍 Latitude
خط العرض للموقع.

**Unit:** Degree (°)

---

### ⛰️ Slope
درجة انحدار سطح الأرض.

**Unit:** Degree (°)

---

### 📅 Date
اكتب التاريخ بالشكل:

`YYYY-MM-DD`
"""
    )


st.divider()


# ============================================================
# PREDICT BUTTON
# ============================================================

predict_clicked = st.button(
    "🔥 Predict Fire Risk | توقّع خطر الحريق"
)


# ============================================================
# PREDICTION
# ============================================================

if predict_clicked:

    # --------------------------------------------------------
    # DATE VALIDATION
    # --------------------------------------------------------

    try:

        selected_date = pd.to_datetime(
            date_input,
            format="%Y-%m-%d"
        )

        year = selected_date.year

        day_of_year = selected_date.dayofyear

    except (ValueError, TypeError):

        st.error(
            "❌ برجاء إدخال تاريخ صحيح بالشكل: YYYY-MM-DD"
        )

        st.stop()


    # --------------------------------------------------------
    # CREATE MODEL INPUT
    # --------------------------------------------------------

    input_data = {
        "precmax": precmax,
        "day_of_year": day_of_year,
        "year": year,
        "precsum": precsum,
        "precmin": precmin,
        "ndvimedian": ndvimedian,
        "ndwimin": ndwimin,
        "days": days,
        "latitude": latitude,
        "slope": slope
    }


    input_df = pd.DataFrame(
        [input_data]
    )


    # Make sure feature order matches the saved final model
    input_df = input_df[features]


    # ========================================================
    # MODEL PREDICTION
    # ========================================================

    fire_probability = model.predict_proba(
        input_df
    )[0][1]


    # Apply the final validated threshold
    prediction = int(
        fire_probability >= threshold
    )


    # ========================================================
    # RISK LEVEL
    # ========================================================

    if fire_probability < 0.30:

        risk_level = "LOW RISK"
        risk_ar = "خطر منخفض"

        risk_message = (
            "الظروف الحالية تشير إلى احتمال منخفض لحدوث حريق."
        )

        risk_class = "low-risk"
        emoji = "🟢"
        gauge_color = "#22c55e"


    elif fire_probability < 0.75:

        risk_level = "MEDIUM RISK"
        risk_ar = "خطر متوسط"

        risk_message = (
            "الظروف الحالية تشير إلى احتمال متوسط لحدوث حريق."
        )

        risk_class = "medium-risk"
        emoji = "🟡"
        gauge_color = "#f59e0b"


    else:

        risk_level = "HIGH RISK"
        risk_ar = "خطر مرتفع"

        risk_message = (
            "الظروف الحالية تشير إلى احتمالية مرتفعة للحريق."
        )

        risk_class = "high-risk"
        emoji = "🔴"
        gauge_color = "#ef4444"


    # ========================================================
    # RESULT SECTION TITLE
    # ========================================================

    st.markdown(
        '<div class="section-title">🎯 نتيجة التنبؤ</div>',
        unsafe_allow_html=True
    )


    # ========================================================
    # PROBABILITY
    # ========================================================

    probability_percent = fire_probability * 100


    # ========================================================
    # PLOTLY GAUGE
    # ========================================================

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",

            value=probability_percent,

            title={
                "text":
                    "Fire Probability<br>"
                    "<span style='font-size:14px'>"
                    "احتمالية الحريق"
                    "</span>",

                "font": {
                    "size": 20,
                    "color": "#052f4a"
                }
            },

            number={
                "suffix": "%",

                "font": {
                    "size": 42,
                    "color": "#052f4a"
                }
            },

            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "#475569"
                },

                "bar": {
                    "color": gauge_color,
                    "thickness": 0.30
                },

                "bgcolor":
                    "rgba(255,255,255,0.88)",

                "borderwidth":
                    1,

                "bordercolor":
                    "#94a3b8",

                "steps": [
                    {
                        "range": [0, 30],
                        "color":
                            "rgba(34,197,94,0.20)"
                    },

                    {
                        "range": [30, 75],
                        "color":
                            "rgba(245,158,11,0.20)"
                    },

                    {
                        "range": [75, 100],
                        "color":
                            "rgba(239,68,68,0.20)"
                    }
                ],

                "threshold": {
                    "line": {
                        "color": "#0f172a",
                        "width": 3
                    },

                    "thickness":
                        0.80,

                    "value":
                        threshold * 100
                }
            }
        )
    )


    fig.update_layout(
        height=350,

        margin={
            "l": 30,
            "r": 30,
            "t": 35,
            "b": 15
        },

        paper_bgcolor="rgba(255,255,255,0.15)",

        plot_bgcolor="rgba(255,255,255,0)"
    )


    # Display gauge
    st.plotly_chart(
        fig,

        use_container_width=True,

        config={
            "displayModeBar": False
        }
    )


    # ========================================================
    # RISK RESULT
    # ========================================================

    result_html = (
        f'<div class="result-card {risk_class}">'
        f'<div class="risk-icon">{emoji}</div>'
        f'<div class="risk-title">{risk_level} — {risk_ar}</div>'
        f'<div class="risk-description">{risk_message}</div>'
        f'</div>'
    )

    st.markdown(
        result_html,
        unsafe_allow_html=True
    )


    # ========================================================
    # METRICS
    # ========================================================

    metric1, metric2, metric3 = st.columns(3)


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    with metric1:

        prediction_text = (
            "🔥 FIRE"
            if prediction == 1
            else "✅ NO FIRE"
        )

        metric_html = (
            '<div class="metric-card">'
            '<div class="metric-label">'
            'Prediction | التنبؤ'
            '</div>'
            f'<div class="metric-value">{prediction_text}</div>'
            '</div>'
        )

        st.markdown(
            metric_html,
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    with metric2:

        metric_html = (
            '<div class="metric-card">'
            '<div class="metric-label">'
            'Probability | الاحتمالية'
            '</div>'
            f'<div class="metric-value">{probability_percent:.2f}%</div>'
            '</div>'
        )

        st.markdown(
            metric_html,
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # Risk Level
    # --------------------------------------------------------

    with metric3:

        metric_html = (
            '<div class="metric-card">'
            '<div class="metric-label">'
            'Risk Level | مستوى الخطر'
            '</div>'
            f'<div class="metric-value">{risk_ar}</div>'
            '</div>'
        )

        st.markdown(
            metric_html,
            unsafe_allow_html=True
        )


    # ========================================================
    # DETAILS
    # ========================================================

    with st.expander(
        "📋 Prediction Details | تفاصيل التنبؤ"
    ):

        st.write(
            f"**Date | التاريخ:** {date_input}"
        )

        st.write(
            f"**Fire Probability | احتمال الحريق:** "
            f"{probability_percent:.2f}%"
        )

        st.write(
            f"**Prediction | التنبؤ:** "
            f"{'Fire' if prediction == 1 else 'No Fire'}"
        )

        st.write(
            f"**Risk Level | مستوى الخطر:** "
            f"{risk_ar}"
        )


# ============================================================
# NEW PREDICTION
# ============================================================

st.divider()


st.markdown(
    '<div class="section-title">🔄 توقع حالة جديدة</div>',
    unsafe_allow_html=True
)


st.button(
    "🧹 Clear All Inputs & Start New Prediction | "
    "مسح البيانات وبدء توقع جديد",
    on_click=reset_inputs
)


# ============================================================
# FOOTER
# ============================================================

footer_html = (
    '<div class="footer">'
    '🔥 Forest Fire Risk Prediction'
    '<br>'
    'XGBoost Machine Learning Project'
    '</div>'
)

st.markdown(
    footer_html,
    unsafe_allow_html=True
)