import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Set the page configuration to wide mode
st.set_page_config(page_title="ICM Underwriting Dashboard", layout="wide")

# CUSTOM CSS – adjust these values for finer tuning if needed
st.markdown("""
<style>
/* ---------- SIDEBAR ---------- */
[data-testid="stSidebar"] {
    background-color: #1E3A8A;
    padding: 20px;
}
.sidebar-title {
    text-align: center;
    color: #fff;
    font-size: 28px;
    font-weight: 700;
    margin-bottom: 20px;
}
.sidebar-item {
    text-align: center;
    color: #fff;
    font-size: 20px;
    padding: 10px;
    margin: 8px 0;
    border-bottom: 1px solid rgba(255,255,255,0.3);
}

/* ---------- HEADER ---------- */
.header {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    margin: 20px 0;
    color: #333;
}

/* ---------- SECTION TITLES ---------- */
.section-title {
    font-size: 34px;
    font-weight: 600;
    margin: 40px 0 20px;
    color: #333;
}

/* ---------- METRIC CARDS ---------- */
.metric-card {
    background-color: #fff;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 20px;
    text-align: center;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
}
.metric-card .title {
    font-size: 16px;
    color: #777;
    margin-bottom: 8px;
}
.metric-card .value {
    font-size: 36px;
    font-weight: 700;
    color: #333;
}

/* ---------- PORTFOLIO CARDS ---------- */
.portfolio-card {
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 20px;
    text-align: center;
    color: #fff;
}
.portfolio-card.one { background-color: #0D1321; }
.portfolio-card.two { background-color: #165A72; }
.portfolio-card.three { background-color: #4A69BD; }
.portfolio-card .value {
    font-size: 36px;
    font-weight: 700;
}
.portfolio-card .label {
    font-size: 16px;
}

/* ---------- FOOTER BUTTONS ---------- */
.footer {
    text-align: center;
    margin-top: 40px;
    margin-bottom: 20px;
}
.footer button {
    background-color: #1E88E5;
    border: none;
    color: #fff;
    padding: 15px 30px;
    font-size: 18px;
    border-radius: 8px;
    cursor: pointer;
    margin: 0 10px;
    transition: background-color 0.3s;
}
.footer button:hover {
    background-color: #1669A2;
}
</style>
""", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown('<div class="sidebar-title">ICM Underwriting</div>', unsafe_allow_html=True)
    for item in ["ICM", "UW", "TM", "CRMS", "Product", "SELF SERVICE"]:
        st.markdown(f'<div class="sidebar-item">{item}</div>', unsafe_allow_html=True)

# ---------- MAIN HEADER ----------
st.markdown('<div class="header">ICM Underwriting Dashboard</div>', unsafe_allow_html=True)

# ---------- MONITORING STATS SECTION ----------
st.markdown('<div class="section-title">Monitoring Stats</div>', unsafe_allow_html=True)
m_col1, m_col2, m_col3 = st.columns(3)

with m_col1:
    st.markdown("""
    <div class="metric-card">
        <div class="title">Total Workflows (Current Month)</div>
        <div class="value">300</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <div class="title">Annual Reviews</div>
        <div class="value">275</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <div class="title">Quarterly</div>
        <div class="value">25</div>
    </div>
    """, unsafe_allow_html=True)

with m_col2:
    st.markdown("""
    <div class="metric-card">
        <div class="title">Total Workflows (YTD)</div>
        <div class="value">1200 <span style="font-size:20px;">🔼100</span></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <div class="title">Annual Reviews (YTD)</div>
        <div class="value">900 <span style="font-size:20px;">🔼100</span></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <div class="title">Quarterly (YTD)</div>
        <div class="value">75 <span style="font-size:20px;">🔼15</span></div>
    </div>
    """, unsafe_allow_html=True)

with m_col3:
    st.markdown("""
    <div class="metric-card">
        <div class="title">Total Workflows (Last Year)</div>
        <div class="value">1100</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <div class="title">Annual Reviews (Last Year)</div>
        <div class="value">800</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <div class="title">Quarterly (Last Year)</div>
        <div class="value">60</div>
    </div>
    """, unsafe_allow_html=True)

# ---------- PORTFOLIO SECTION ----------
st.markdown('<div class="section-title">Portfolio</div>', unsafe_allow_html=True)
p1, p2, p3 = st.columns(3)

with p1:
    st.markdown("""
    <div class="portfolio-card one">
        <div class="value">6,280</div>
        <div class="label"># of Relationships <span style="font-size:18px;">🔼4%</span></div>
    </div>
    """, unsafe_allow_html=True)
with p2:
    st.markdown("""
    <div class="portfolio-card two">
        <div class="value">$1.6T</div>
        <div class="label">OSUC <span style="font-size:18px;">🔼2%</span></div>
    </div>
    """, unsafe_allow_html=True)
with p3:
    st.markdown("""
    <div class="portfolio-card three">
        <div class="value">RR 4-</div>
        <div class="label">Risk Profile <span style="font-size:18px;">🔼1 Notch</span></div>
    </div>
    """, unsafe_allow_html=True)

# ---------- ACTIVE WORKFLOW MANAGEMENT CHART ----------
st.markdown('<div class="section-title">Active Workflow Management: Annual Reviews</div>', unsafe_allow_html=True)
months = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
values = [500, 500, 500, 400, 600, 700, 800, 700, 600, 700, 600, 700]
fig = go.Figure(go.Bar(x=months, y=values, marker_color="#1E88E5"))
fig.update_layout(
    height=300,
    margin=dict(l=20, r=20, t=20, b=20),
    xaxis_title="Months",
    yaxis_title="Count",
    plot_bgcolor="rgba(0,0,0,0)"
)
st.plotly_chart(fig, use_container_width=True)

# ---------- STATUS OVERVIEW TABLE ----------
st.markdown('<div class="section-title">Status Overview</div>', unsafe_allow_html=True)
status_data = {
    "Category": ["Quarterly", "CCM", "Risk Ratings", "Covenants"],
    "Pending": [15, 20, 30, 18],
    "Past Due": [1, 2, 10, 1]
}
df_status = pd.DataFrame(status_data)
st.table(df_status)

# ---------- FOOTER BUTTONS ----------
st.markdown("""
<div class="footer">
    <button>Monitoring</button>
    <button>Workflow Management</button>
    <button>Portfolio</button>
</div>
""", unsafe_allow_html=True)
