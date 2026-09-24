import pandas as pd
import plotly.express as px
import streamlit as st

from market import interpret, resample, stats, ui
from market import returns as rets

s = ui.settings()
inst = ui.instruments()
ui.require(inst)

ui.header("Compare", "Relative performance, risk and drawdowns side by side, for tickers or whole asset classes.")
with st.container(border=True):
    tickers, groups = ui.subject_picker()
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    stress_periods = ui.stress_picker("compare_stress", c1)
    log_scale = c2.toggle("Log scale")
ui.require(tickers, "Pick at least one ticker or asset class.")

loaded = resample.last(ui.price_matrix(tickers, s, groups=groups), s.freq).dropna(how="all")
# Index every series from the date they all exist: rebasing each to its own start puts a 16-year
# compounding beside a 10-year one and calls the totals comparable.
starts = [loaded[c].first_valid_index() for c in loaded if loaded[c].first_valid_index() is not None]
prices = loaded.loc[max(starts):] if starts else loaded
if len(starts) and max(starts) > loaded.index[0]:
    latest = max(loaded, key=lambda c: loaded[c].first_valid_index() or loaded.index[0])
    st.caption(f"Compared from {max(starts):%d %b %Y}, where every series has data ({latest} has the shortest "
               "history). Each would otherwise be indexed to a different starting date.")
rebased = prices / prices.bfill().iloc[0] * 100
aligned = rets.align_closes(prices, ui.regions(groups)) if s.on("align_closes") and s.freq == "D" else prices
r = aligned.pct_change(fill_method=None).iloc[1:]

with st.container(border=True):
    fig = px.line(rebased, log_y=log_scale, title="Growth of 100", labels={"value": "", "date": ""})
    ui.shade_stress(fig, stress_periods, s)
    ui.chart(fig, height=460)
    if s.currency == "native" and inst[inst.ticker.isin(tickers)].currency.nunique() > 1:
        st.caption("Mixed currencies in native terms. Choose AUD or USD in the sidebar to compare in one currency.")

bench = st.selectbox("Benchmark", ["None", *inst.ticker], format_func=lambda t: t if t == "None" else ui.label(t),
                     help="Adds beta, tracking error, information ratio and capture against this series")
bench_r = None
if bench != "None":
    bench_prices = resample.last(ui.price_matrix([bench], s), s.freq).get(bench)
    if bench_prices is not None:
        bench_r = bench_prices.pct_change(fill_method=None).reindex(r.index)

table = stats.risk.summary(r, bench_r, rf=ui.risk_free(s, r.index))
ui.adjustments_caption(s, "live_cash", "align_closes")
ui.close_time_caption(r.columns, s, groups)
table.insert(0, "total_return", prices.ffill().iloc[-1] / prices.bfill().iloc[0] - 1)
cols = ["total_return", "ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown", "calmar", "dd_length"]
if bench_r is not None:
    cols += ["beta", "tracking_error", "information_ratio", "up_capture", "down_capture"]
with st.container(border=True):
    st.markdown("**Risk & return**" + (f" vs {bench}" if bench_r is not None else ""))
    st.dataframe(table[cols], column_config=ui.percent(table, cols[:3] + ["max_drawdown"])
                 | {c: ui.NUM for c in ("sharpe", "sortino", "calmar", "beta", "information_ratio")}
                 | ui.percent(table, ["tracking_error", "up_capture", "down_capture"])
                 | {"dd_length": st.column_config.NumberColumn("dd_length", help="Periods from peak to recovery"),
                    "sharpe": st.column_config.NumberColumn("sharpe", help="Arithmetic mean excess return over "
                                                            "cash, divided by volatility; ann_return beside it "
                                                            "compounds geometrically, so the two use different means")})
    c1, c2 = st.columns([1, 4])
    with c1:
        ui.download(table, "compare")
    with c2:
        ui.report(f"Compare · {', '.join(prices.columns[:6])}{'…' if len(prices.columns) > 6 else ''}", s, [
            ("Risk and return", table[cols]),
            ("Correlation of returns", r.corr().round(3)),
            ("Drawdown", pd.DataFrame({"max drawdown": table.max_drawdown, "periods peak to recovery": table.dd_length})),
        ], "compare_tearsheet", notes=[f"Measured over {len(r)} {ui.FREQS[s.freq].lower()} returns from "
                                       f"{r.index[0]:%d %b %Y} to {r.index[-1]:%d %b %Y}."])

with st.container(border=True):
    fig = px.line(stats.risk.drawdown(r), title="Drawdown from peak", labels={"value": "", "date": ""})
    ui.shade_stress(fig, stress_periods, s)
    ui.chart(fig.update_yaxes(tickformat=".0%"), height=340)

ui.explain(interpret.compare(table))
