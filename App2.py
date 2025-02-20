import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Set wide layout
st.set_page_config(layout="wide")

# CUSTOM CSS – adjust these styles as needed for pixel-perfect matching
custom_css = """
<style>
/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #1E3A8A;
    padding: 20px;
}
.sidebar-title {
    color: white;
    font-size: 22px;
    font-weight: bold;
    margin-bottom: 20px;
}
.sidebar-item {
    color: white;
    font-size: 18px;
    padding: 10px 0;
    border-bottom: 1px solid white;
}

/* Header styling */
.header {
    text-align: center;
    font-size: 32px;
    font-weight: bold;
    margin: 10px 0 20px 0;
}

/* Metric Boxes */
.metric-box {
    background-color: white;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    text-align: center;
    margin-bottom: 20px;
}
.metric-title {
    font-size: 14px;
    color: #666;
    margin-bottom: 5px;
}
.metric-value {
    font-size: 28px;
    font-weight: bold;
    color: #333;
}

/* Portfolio Boxes */
.portfolio-box-1, .portfolio-box-2, .portfolio-box-3 {
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    color: white;
    margin-bottom: 20px;
}
.portfolio-box-1 { background-color: #0D1321; }
.portfolio-box-2 { background-color: #165A72; }
.portfolio-box-3 { background-color: #4A69BD; }
.portfolio-metric-value {
    font-size: 28px;
    font-weight: bold;
}
.portfolio-metric-title {
    font-size: 14px;
}

/* Footer Buttons */
.footer-btn {
    background-color: #1E88E5;
    border: none;
    color: white;
    padding: 10px 20px;
    border-radius: 5px;
    font-size: 16px;
    margin: 0 10px;
    cursor: pointer;
}
.footer-btn:hover {
    opacity: 0.9;
}
.footer-container {
    text-align: center;
    margin-top: 20px;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.markdown('<div class="sidebar-title">ICM Underwriting</div>', unsafe_allow_html=True)
    sidebar_items = ["ICM", "UW", "TM", "CRMS", "Product", "SELF SERVICE"]
    for item in sidebar_items:
        st.markdown(f'<div class="sidebar-item">{item}</div>', unsafe_allow_html=True)

# MAIN HEADER
st.markdown('<div class="header">ICM Underwriting</div>', unsafe_allow_html=True)

# MONITORING STATS
st.markdown("### Monitoring Stats")
col1, col2, col3 = st.columns(3)

with col1:
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

with col2:
    st.markdown(
        '<div class="metric-box">'
        '<div class="metric-title">Total Workflows (YTD)</div>'
        '<div class="metric-value">1200 🔼100</div>'
        '</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="metric-box">'
        '<div class="metric-title">Annual Reviews (YTD)</div>'
        '<div class="metric-value">900 🔼100</div>'
        '</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="metric-box">'
        '<div class="metric-title">Quarterly (YTD)</div>'
        '<div class="metric-value">75 🔼15</div>'
        '</div>', unsafe_allow_html=True)

with col3:
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
st.markdown("### Portfolio")
port_col1, port_col2, port_col3 = st.columns(3)

with port_col1:
    st.markdown(
        '<div class="portfolio-box-1">'
        '<div class="portfolio-metric-value">6,280</div>'
        '<div class="portfolio-metric-title"># of Relationships 🔼4%</div>'
        '</div>', unsafe_allow_html=True)

with port_col2:
    st.markdown(
        '<div class="portfolio-box-2">'
        '<div class="portfolio-metric-value">$1.6T</div>'
        '<div class="portfolio-metric-title">OSUC 🔼2%</div>'
        '</div>', unsafe_allow_html=True)

with port_col3:
    st.markdown(
        '<div class="portfolio-box-3">'
        '<div class="portfolio-metric-value">RR 4-</div>'
        '<div class="portfolio-metric-title">Risk Profile 🔼1 Notch</div>'
        '</div>', unsafe_allow_html=True)

# ACTIVE WORKFLOW MANAGEMENT CHART
st.markdown("### Active Workflow Management: Annual Reviews")
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
st.markdown("### Status Overview")
status_data = {
    "Category": ["Quarterly", "CCM", "Risk Ratings", "Covenants"],
    "Pending": [15, 20, 30, 18],
    "Past Due": [1, 2, 10, 1]
}
df_status = pd.DataFrame(status_data)
st.dataframe(df_status)

# FOOTER BUTTONS
st.markdown(
    '<div class="footer-container">'
    '<button class="footer-btn">Monitoring</button>'
    '<button class="footer-btn">Workflow Management</button>'
    '<button class="footer-btn">Portfolio</button>'
    '</div>',
    unsafe_allow_html=True
)
