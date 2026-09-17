import plotly.express as px
import streamlit as st

from market import interpret, resample, stats, ui

s = ui.settings()
inst = ui.instruments()
ui.require(inst)

ui.header("Compare", "Relative performance, risk and drawdowns side by side, for tickers or whole asset classes.")
with st.container(border=True):
    tickers, groups = ui.subject_picker()
    log_scale = st.toggle("Log scale")
ui.require(tickers, "Pick at least one ticker or asset class.")

prices = resample.last(ui.price_matrix(tickers, s, groups=groups), s.freq).dropna(how="all")
rebased = prices / prices.bfill().iloc[0] * 100
r = prices.pct_change(fill_method=None).iloc[1:]

with st.container(border=True):
    ui.chart(px.line(rebased, log_y=log_scale, title="Growth of 100", labels={"value": "", "date": ""}), height=460)
    if s.currency == "native" and inst[inst.ticker.isin(tickers)].currency.nunique() > 1:
        st.caption("Mixed currencies in native terms. Choose AUD or USD in the sidebar to compare in one currency.")

table = stats.risk.summary(r)
table.insert(0, "total_return", prices.ffill().iloc[-1] / prices.bfill().iloc[0] - 1)
cols = ["total_return", "ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown", "calmar", "dd_length"]
with st.container(border=True):
    st.markdown("**Risk & return**")
    st.dataframe(table[cols], column_config=ui.percent(table, cols[:3] + ["max_drawdown"])
                 | {c: ui.NUM for c in ("sharpe", "sortino", "calmar")}
                 | {"dd_length": st.column_config.NumberColumn("dd_length", help="Periods from peak to recovery")})
    ui.download(table, "compare")

with st.container(border=True):
    fig = px.line(stats.risk.drawdown(r), title="Drawdown from peak", labels={"value": "", "date": ""})
    ui.chart(fig.update_yaxes(tickformat=".0%"), height=340)

ui.explain(interpret.compare(table))
