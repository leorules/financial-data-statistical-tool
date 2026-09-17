import streamlit as st

from market import ui

st.set_page_config(page_title="FDA", page_icon="📈", layout="wide")
st.logo("assets/fda.svg", size="large")
st.html("""<style>
.block-container {padding-top: 2.5rem; padding-bottom: 3rem; max-width: 1480px}
h1 {letter-spacing: -0.02em}
[data-testid="stMetricLabel"] p {font-size: 0.78rem; letter-spacing: 0.02em; opacity: 0.75}
[data-testid="stMetricValue"] {font-size: 1.55rem; font-weight: 600}
[data-testid="stSidebarUserContent"] {padding-top: 0.5rem}
.st-key-tiles [data-testid="stMetricValue"] {font-size: 1.2rem}
</style>""")

page = st.navigation({
    "Explore": [
        st.Page("pages/overview.py", title="Overview", icon=":material/candlestick_chart:", default=True),
        st.Page("pages/asset_classes.py", title="Asset Classes", icon=":material/category:"),
        st.Page("pages/screener.py", title="Screener", icon=":material/filter_alt:"),
        st.Page("pages/compare.py", title="Compare", icon=":material/stacked_line_chart:"),
        st.Page("pages/portfolio.py", title="Portfolio", icon=":material/account_balance_wallet:"),
    ],
    "Analyse": [
        st.Page("pages/correlation.py", title="Correlation", icon=":material/grid_on:"),
        st.Page("pages/statistics.py", title="Statistics", icon=":material/functions:"),
        st.Page("pages/scenarios.py", title="Stress Periods", icon=":material/history:"),
        st.Page("pages/code_lab.py", title="Code Lab", icon=":material/code:"),
    ],
    "Data": [st.Page("pages/data_manager.py", title="Data Manager", icon=":material/database:")],
})
ui.sidebar()
page.run()
