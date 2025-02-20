import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Set the page configuration to wide layout.
st.set_page_config(layout="wide")

# CUSTOM CSS – adjust values if needed to further fine-tune spacing, fonts, etc.
st.markdown("""
<style>
/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #1E3A8A;
    padding: 20px;
}
.sidebar-title {
    color: white;
    font-size: 24px;
    font-weight: 600;
    margin-bottom: 30px;
    text-align: center;
}
.sidebar-item {
    color: white;
    font-size: 20px;
    margin: 12px 0;
    text-align: center;
    padding: 8px 0;
    border-bottom: 1px solid rgba(255,255,255,0.3);
}

/* Main header */
.main-header {
    text-align: center;
    font-size: 36px;
    font-weight: 700;
    margin: 30px 0 20px 0;
    color: #333;
}

/* Section headings */
.section-heading {
    font-size: 28px;
    font-weight: 600;
    margin: 40px 0 20px 0;
    color: #333;
    text-align: left;
}

/* Metric boxes */
.metric-box {
    background-color: #ffffff;
    border-radius: 12px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    padding: 20px;
    text-align: center;
    margin-bottom: 20px;
}
.metric-title {
    font-size: 16px;
    color: #777;
    margin-bottom: 8px;
}
.metric-value {
    font-size: 32px;
    font-weight: 700;
    color: #222;
}

/* Portfolio boxes */
.portfolio-box {
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin-bottom: 20px;
}
.portfolio-box.one { background-color: #0D1321; color: #fff; }
.portfolio-box.two { background-color: #165A72; color: #fff; }
.portfolio-box.three { background-color: #4A69BD; color: #fff; }
.portfolio-value {
    font-size: 32px;
    font-weight: 700;
}
.portfolio-label {
    font-size: 16px;
}

/* Footer buttons */
.footer-container {
    display: flex;
    justify-content: center;
    gap: 30px;
    margin-top: 40px;
    margin-bottom: 20px;
}
.footer-button {
    background-color: #1E88E5;
    border: none;
    color: #fff;
    padding: 12px 30px;
    font-size: 18px;
    border-radius: 8px;
    cursor: pointer;
    transition: background-color 0.3s ease;
}
.footer-button:hover {
    background-color: #1669A2;
}
</style>
""", unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.markdown('<div class="sidebar-title">ICM Underwriting</div>', unsafe_allow_html=True)
    sidebar_items = ["ICM", "UW", "TM", "CRMS", "Product", "SELF SERVICE"]
    for item in sidebar_items:
        st.markdown(f'<div class="sidebar-item">{item}</div>', unsafe_allow_html=True)

# MAIN HEADER
st.markdown('<div class="main-header">ICM Underwriting</div>', unsafe_allow_html=True)

# MONITORING STATS SECTION
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
        '<div class="metric-value">1200 <span style="font-size:16px;">🔼100</span></div>'
        '</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="metric-box">'
        '<div class="metric-title">Annual Reviews (YTD)</div>'
        '<div class="metric-value">900 <span style="font-size:16px;">🔼100</span></div>'
        '</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="metric-box">'
        '<div class="metric-title">Quarterly (YTD)</div>'
        '<div class="metric-value">75 <span style="font-size:16px;">🔼15</span></div>'
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

# PORTFOLIO SECTION
st.markdown('<div class="section-heading">Portfolio</div>', unsafe_allow_html=True)
p_col1, p_col2, p_col3 = st.columns(3)

with p_col1:
    st.markdown(
        '<div class="portfolio-box one">'
        '<div class="portfolio-value">6,280</div>'
        '<div class="portfolio-label"># of Relationships <span style="font-size:14px;">🔼4%</span></div>'
        '</div>', unsafe_allow_html=True)
with p_col2:
    st.markdown(
        '<div class="portfolio-box two">'
        '<div class="portfolio-value">$1.6T</div>'
        '<div class="portfolio-label">OSUC <span style="font-size:14px;">🔼2%</span></div>'
        '</div>', unsafe_allow_html=True)
with p_col3:
    st.markdown(
        '<div class="portfolio-box three">'
        '<div class="portfolio-value">RR 4-</div>'
        '<div class="portfolio-label">Risk Profile <span style="font-size:14px;">🔼1 Notch</span></div>'
        '</div>', unsafe_allow_html=True)

# ACTIVE WORKFLOW MANAGEMENT CHART
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

# STATUS OVERVIEW TABLE
st.markdown('<div class="section-heading">Status Overview</div>', unsafe_allow_html=True)
status_data = {
    "Category": ["Quarterly", "CCM", "Risk Ratings", "Covenants"],
    "Pending": [15, 20, 30, 18],
    "Past Due": [1, 2, 10, 1]
}
df_status = pd.DataFrame(status_data)
st.table(df_status)

# FOOTER BUTTONS
st.markdown("""
<div class="footer-container">
    <button class="footer-button">Monitoring</button>
    <button class="footer-button">Workflow Management</button>
    <button class="footer-button">Portfolio</button>
</div>
""", unsafe_allow_html=True)
