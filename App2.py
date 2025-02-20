import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Set wide layout and page title
st.set_page_config(page_title="ICM Underwriting Dashboard", layout="wide")

# CUSTOM CSS – using CSS Grid for main content and fine-tuning fonts, colors, and spacing
st.markdown("""
<style>
/* ---------- Overall Body ---------- */
body {
    font-family: Arial, sans-serif;
    background: #f4f7fa;
    color: #333;
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background-color: #1E3A8A;
    padding: 20px;
}
[data-testid="stSidebar"] h2 {
    color: #fff;
    text-align: center;
    font-size: 26px;
    margin-bottom: 30px;
}
.sidebar-item {
    text-align: center;
    color: #fff;
    font-size: 20px;
    padding: 10px 0;
    margin: 8px 0;
    border-bottom: 1px solid rgba(255,255,255,0.3);
}

/* ---------- Main Content ---------- */
.app-header {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    margin: 30px 0 40px;
}
.section-title {
    font-size: 32px;
    font-weight: 600;
    margin: 40px 0 20px;
    border-bottom: 2px solid #ccc;
    padding-bottom: 5px;
}

/* Grid container for metrics & portfolio */
.metrics-grid, .portfolio-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 20px;
}

/* ---------- Metric Cards ---------- */
.metric-card {
    background: #fff;
    border-radius: 8px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.metric-title {
    font-size: 16px;
    color: #777;
    margin-bottom: 8px;
}
.metric-value {
    font-size: 28px;
    font-weight: bold;
}

/* ---------- Portfolio Cards ---------- */
.portfolio-card {
    border-radius: 8px;
    padding: 20px;
    text-align: center;
    color: #fff;
}
.portfolio-1 { background: #0D1321; }
.portfolio-2 { background: #165A72; }
.portfolio-3 { background: #4A69BD; }
.portfolio-value {
    font-size: 28px;
    font-weight: bold;
}
.portfolio-label {
    font-size: 16px;
}

/* ---------- Footer Buttons ---------- */
.footer {
    text-align: center;
    margin: 40px 0 20px;
}
.footer button {
    background: #1E88E5;
    border: none;
    color: #fff;
    padding: 12px 24px;
    font-size: 16px;
    border-radius: 6px;
    margin: 0 10px;
    cursor: pointer;
    transition: background 0.3s;
}
.footer button:hover {
    background: #1669A2;
}
</style>
""", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("<h2>ICM Underwriting</h2>", unsafe_allow_html=True)
    for item in ["ICM", "UW", "TM", "CRMS", "Product", "SELF SERVICE"]:
        st.markdown(f"<div class='sidebar-item'>{item}</div>", unsafe_allow_html=True)

# ---------- MAIN HEADER ----------
st.markdown("<div class='app-header'>ICM Underwriting Dashboard</div>", unsafe_allow_html=True)

# ---------- MONITORING STATS SECTION ----------
st.markdown("<div class='section-title'>Monitoring Stats</div>", unsafe_allow_html=True)

# Create metric cards in a grid – nine cards (three rows of three)
metrics = [
    ("Total Workflows (Current Month)", "300"),
    ("Annual Reviews", "275"),
    ("Quarterly", "25"),
    ("Total Workflows (YTD)", "1200 <span style='font-size:16px;'>🔼100</span>"),
    ("Annual Reviews (YTD)", "900 <span style='font-size:16px;'>🔼100</span>"),
    ("Quarterly (YTD)", "75 <span style='font-size:16px;'>🔼15</span>"),
    ("Total Workflows (Last Year)", "1100"),
    ("Annual Reviews (Last Year)", "800"),
    ("Quarterly (Last Year)", "60"),
]

metrics_html = "<div class='metrics-grid'>"
for title, value in metrics:
    metrics_html += f"""
    <div class='metric-card'>
      <div class='metric-title'>{title}</div>
      <div class='metric-value'>{value}</div>
    </div>
    """
metrics_html += "</div>"
st.markdown(metrics_html, unsafe_allow_html=True)

# ---------- PORTFOLIO SECTION ----------
st.markdown("<div class='section-title'>Portfolio</div>", unsafe_allow_html=True)
portfolio = [
    ("6,280", "# of Relationships <span style='font-size:16px;'>🔼4%</span>", "portfolio-1"),
    ("$1.6T", "OSUC <span style='font-size:16px;'>🔼2%</span>", "portfolio-2"),
    ("RR 4-", "Risk Profile <span style='font-size:16px;'>🔼1 Notch</span>", "portfolio-3"),
]
portfolio_html = "<div class='portfolio-grid'>"
for value, label, cls in portfolio:
    portfolio_html += f"""
    <div class='portfolio-card {cls}'>
      <div class='portfolio-value'>{value}</div>
      <div class='portfolio-label'>{label}</div>
    </div>
    """
portfolio_html += "</div>"
st.markdown(portfolio_html, unsafe_allow_html=True)

# ---------- ACTIVE WORKFLOW MANAGEMENT CHART ----------
st.markdown("<div class='section-title'>Active Workflow Management: Annual Reviews</div>", unsafe_allow_html=True)
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
st.markdown("<div class='section-title'>Status Overview</div>", unsafe_allow_html=True)
status_data = {
    "Category": ["Quarterly", "CCM", "Risk Ratings", "Covenants"],
    "Pending": [15, 20, 30, 18],
    "Past Due": [1, 2, 10, 1]
}
df_status = pd.DataFrame(status_data)
st.table(df_status)

# ---------- FOOTER BUTTONS ----------
st.markdown("""
<div class='footer'>
  <button>Monitoring</button>
  <button>Workflow Management</button>
  <button>Portfolio</button>
</div>
""", unsafe_allow_html=True)
