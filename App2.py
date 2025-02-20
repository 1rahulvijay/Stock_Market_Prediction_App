import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Set page configuration to wide layout
st.set_page_config(layout="wide")

# CUSTOM CSS – tweak these values until the layout exactly matches your design
st.markdown("""
<style>
/* ----- SIDEBAR STYLES ----- */
[data-testid="stSidebar"] {
    background-color: #1E3A8A;
    padding: 20px;
}
.sidebar-title {
    text-align: center;
    color: #fff;
    font-size: 26px;
    font-weight: bold;
    margin-bottom: 30px;
}
.sidebar-item {
    text-align: center;
    color: #fff;
    font-size: 20px;
    margin: 12px 0;
    padding: 8px 0;
    border-bottom: 1px solid rgba(255,255,255,0.3);
}

/* ----- MAIN HEADER ----- */
.main-header {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    margin: 20px 0 40px;
    color: #333;
}

/* ----- SECTION HEADINGS ----- */
.section-heading {
    font-size: 32px;
    font-weight: bold;
    color: #333;
    margin: 40px 0 20px;
}

/* ----- METRIC BOXES ----- */
.metric-box {
    background-color: #fff;
    border-radius: 10px;
    padding: 25px;
    margin: 10px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    text-align: center;
}
.metric-title {
    font-size: 16px;
    color: #777;
}
.metric-value {
    font-size: 36px;
    font-weight: bold;
    color: #333;
}

/* ----- PORTFOLIO CARDS ----- */
.portfolio-card {
    border-radius: 10px;
    padding: 25px;
    margin: 10px;
    text-align: center;
    color: #fff;
}
.portfolio-1 { background-color: #0D1321; }
.portfolio-2 { background-color: #165A72; }
.portfolio-3 { background-color: #4A69BD; }
.portfolio-value {
    font-size: 36px;
    font-weight: bold;
}
.portfolio-label {
    font-size: 16px;
}

/* ----- FOOTER BUTTONS ----- */
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
    transition: background-color 0.3s ease;
}
.footer button:hover {
    background-color: #1669A2;
}
</style>
""", unsafe_allow_html=True)

# ----- SIDEBAR -----
with st.sidebar:
    st.markdown('<div class="sidebar-title">ICM Underwriting</div>', unsafe_allow_html=True)
    sidebar_items = ["ICM", "UW", "TM", "CRMS", "Product", "SELF SERVICE"]
    for item in sidebar_items:
        st.markdown(f'<div class="sidebar-item">{item}</div>', unsafe_allow_html=True)

# ----- MAIN HEADER -----
st.markdown('<div class="main-header">ICM Underwriting</div>', unsafe_allow_html=True)

# ----- MONITORING STATS SECTION -----
st.markdown('<div class="section-heading">Monitoring Stats</div>', unsafe_allow_html=True)
m_col1, m_col2, m_col3 = st.columns(3)

with m_col1:
    st.markdown(
        '<div class="metric-box">'
        '<div class="metric-title">Total Workflows (Current Month)</div>'
        '<div class="metric-value">300</div>'
        '</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="metric-box">'
        '<div class="metric-title">Annual Reviews</div>'
        '<div class="metric-value">275</div>'
        '</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="metric-box">'
        '<div class="metric-title">Quarterly</div>'
        '<div class="metric-value">25</div>'
        '</div>', unsafe_allow_html=True)

with m_col2:
    st.markdown(
        '<div class="metric-box">'
        '<div class="metric-title">Total Workflows (YTD)</div>'
        '<div class="metric-value">1200 <span style="font-size:20px;">🔼100</span></div>'
        '</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="metric-box">'
        '<div class="metric-title">Annual Reviews (YTD)</div>'
        '<div class="metric-value">900 <span style="font-size:20px;">🔼100</span></div>'
        '</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="metric-box">'
        '<div class="metric-title">Quarterly (YTD)</div>'
        '<div class="metric-value">75 <span style="font-size:20px;">🔼15</span></div>'
        '</div>', unsafe_allow_html=True)

with m_col3:
    st.markdown(
        '<div class="metric-box">'
        '<div class="metric-title">Total Workflows (Last Year)</div>'
        '<div class="metric-value">1100</div>'
        '</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="metric-box">'
        '<div class="metric-title">Annual Reviews (Last Year)</div>'
        '<div class="metric-value">800</div>'
        '</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="metric-box">'
        '<div class="metric-title">Quarterly (Last Year)</div>'
        '<div class="metric-value">60</div>'
        '</div>', unsafe_allow_html=True)

# ----- PORTFOLIO SECTION -----
st.markdown('<div class="section-heading">Portfolio</div>', unsafe_allow_html=True)
p1, p2, p3 = st.columns(3)
with p1:
    st.markdown(
        '<div class="portfolio-card portfolio-1">'
        '<div class="portfolio-value">6,280</div>'
        '<div class="portfolio-label"># of Relationships <span style="font-size:18px;">🔼4%</span></div>'
        '</div>', unsafe_allow_html=True)
with p2:
    st.markdown(
        '<div class="portfolio-card portfolio-2">'
        '<div class="portfolio-value">$1.6T</div>'
        '<div class="portfolio-label">OSUC <span style="font-size:18px;">🔼2%</span></div>'
        '</div>', unsafe_allow_html=True)
with p3:
    st.markdown(
        '<div class="portfolio-card portfolio-3">'
        '<div class="portfolio-value">RR 4-</div>'
        '<div class="portfolio-label">Risk Profile <span style="font-size:18px;">🔼1 Notch</span></div>'
        '</div>', unsafe_allow_html=True)

# ----- ACTIVE WORKFLOW MANAGEMENT CHART -----
st.markdown('<div class="section-heading">Active Workflow Management: Annual Reviews</div>', unsafe_allow_html=True)
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

# ----- STATUS OVERVIEW TABLE -----
st.markdown('<div class="section-heading">Status Overview</div>', unsafe_allow_html=True)
status_data = {
    "Category": ["Quarterly", "CCM", "Risk Ratings", "Covenants"],
    "Pending": [15, 20, 30, 18],
    "Past Due": [1, 2, 10, 1]
}
df_status = pd.DataFrame(status_data)
st.table(df_status)

# ----- FOOTER BUTTONS -----
st.markdown("""
<div class="footer">
    <button>Monitoring</button>
    <button>Workflow Management</button>
    <button>Portfolio</button>
</div>
""", unsafe_allow_html=True)
