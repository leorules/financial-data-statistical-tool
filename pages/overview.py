from dataclasses import replace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from market import indicators, interpret, resample, stats, ui, universe
from market.config import benchmark_for

s = ui.settings()
inst = ui.instruments()
ui.require(inst)
PERIOD = {"D": "day", "W": "week", "M": "month"}[s.freq]
EVERY = {"D": "daily", "W": "weekly", "M": "monthly"}[s.freq]
RISK_COLUMNS = {
    "std": ("Std dev", f"Standard deviation of {EVERY} returns"),
    "vol_ann": ("Volatility", "Annualised standard deviation"),
    "downside_dev": ("Downside dev", "Annualised volatility of losses only"),
    "var_95": ("VaR 95%", f"Historical: 1 in 20 {PERIOD}s was this bad or worse"),
    "cvar_95": ("CVaR 95%", "Average return in those worst 5% of periods"),
    "var_99": ("VaR 99%", f"Historical: 1 in 100 {PERIOD}s was this bad or worse"),
    "worst": (f"Worst {PERIOD}", "Single worst period in the range"),
    "max_drawdown": ("Max drawdown", "Largest fall from a peak"),
    "sharpe": ("Sharpe", "Annual excess return over cash per unit of volatility"),
    "sortino": ("Sortino", "Like Sharpe, but only penalises downside volatility"),
}

ui.header("Overview", "Global markets at a glance, and a closer look at any instrument.")
ui.market_tiles()

classes = [c for c in universe.ASSET_CLASSES if c in set(inst.asset_class)]
with st.container(border=True):
    asset_class = st.segmented_control("Asset class", classes, default=classes[0], required=True)
    pool = inst[inst.asset_class == asset_class]
    c1, c2, c3, c4 = st.columns([2, 2, 4, 2], vertical_alignment="bottom")
    region = c1.selectbox("Region", ["All", *sorted(pool.exchange.dropna().unique())])
    pool = pool if region == "All" else pool[pool.exchange == region]
    available = universe.kinds(pool.type)
    kind = c2.selectbox("Kind", [*available, "All"], help="Indices, ETFs and individual stocks are listed separately")
    pool = pool if kind == "All" else pool[pool.type.isin(universe.KINDS[kind])]
    tickers = pool.ticker.tolist()
    ticker = c3.selectbox("Instrument", tickers, index=tickers.index("^AXJO") if "^AXJO" in tickers else 0,
                          format_func=ui.label)
    smas = c4.multiselect("Moving averages", [20, 50, 100, 200], default=[50, 200])
    daily = ui.prices((ticker,)).set_index("date").drop(columns="ticker")
    bars = resample.ohlcv(daily, s.freq)
    per_month = stats.periods_per_year(bars.index) / 12
    longest = max(2, int(len(bars) / per_month))
    c5, c6 = st.columns(2)
    months = c5.slider("Rolling volatility window", 1, longest, min(12, longest), format="%d months",
                       help="Annualised standard deviation of returns over this trailing window, from one month "
                            "up to the instrument's full history")
    stress_periods = ui.stress_picker("overview_stress", c6)

window = max(2, round(months * per_month))
for n in smas:
    bars[f"SMA {n}"] = indicators.sma(bars["close"], n)
bar_returns = bars["adj_close"].pct_change(fill_method=None)
# computed on full history so the start of the range is not blank
bars["Vol"] = bar_returns.rolling(window).std() * np.sqrt(stats.periods_per_year(bars.index))
view = bars.loc[str(s.start or bars.index[0]):str(s.end)]
ui.require(view, "No data in the selected range.")

r = view["adj_close"].pct_change(fill_method=None).dropna()
periods = stats.periods_per_year(view.index)
rf = ui.risk_free(s, r.index)
risk = stats.risk.cross_section(r.to_frame(ticker), periods, rf).iloc[0]

bench_ticker = benchmark_for(ticker, s.on("total_return"))
relative = corr = None
if bench_ticker != ticker:
    bench_wide = ui.price_matrix([bench_ticker], s)
    if bench_ticker in bench_wide:
        bench_r = resample.last(bench_wide, s.freq)[bench_ticker].pct_change(fill_method=None)
        both = pd.concat([r, bench_r], axis=1, keys=["a", "b"]).dropna()
        if len(both) > 20:
            relative = stats.risk.metrics(both.a, both.b, periods, rf)
            corr = both.a.corr(both.b)
k = st.columns(4)
k[0].metric("Last close", f"{view.close.iloc[-1]:,.2f}", f"{view.close.pct_change().iloc[-1]:+.2%}", border=True,
            help=f"Change over the last {PERIOD}")
k[1].metric("Period return", f"{view.adj_close.iloc[-1] / view.adj_close.iloc[0] - 1:+.2%}", border=True)
k[2].metric("Annual return", f"{(view.adj_close.iloc[-1] / view.adj_close.iloc[0]) ** (periods / max(len(r), 1)) - 1:+.2%}",
            border=True, help="Compound annual growth rate over the selected range")
k[3].metric(f"RSI (14 {PERIOD}s)", f"{indicators.rsi(bars.close).iloc[-1]:.0f}", border=True,
            help=f"Measured over 14 {PERIOD}s, following the sidebar interval. Above 70 is often read as "
                 "overbought, below 30 as oversold.")

with st.container(border=True):
    st.markdown(f"**:material/shield: Risk** · {len(r)} {EVERY} returns, {view.index[0]:%d %b %Y} – {view.index[-1]:%d %b %Y}")
    ui.adjustments_caption(s, "live_cash")
    if s.on("live_cash"):
        st.caption(ui.cash_label(s, r.index))
    tiles = [(key, *RISK_COLUMNS[key]) for key in ("std", "vol_ann", "var_95", "cvar_95", "var_99", "max_drawdown",
                                                   "sharpe", "sortino")]
    for line in (tiles[:4], tiles[4:]):
        for col, (key, name, help_text) in zip(st.columns(4), line):
            value = f"{risk[key]:.2f}" if key in ("sharpe", "sortino") else f"{risk[key]:.2%}"
            col.metric(name, value, help=help_text)
    if relative is not None:
        st.markdown(f"**Versus {ui.label(bench_ticker)}**")
        for col, (name, value, help_text) in zip(st.columns(4), [
            ("Beta", f"{relative.beta:.2f}", "Sensitivity to a 1% benchmark move"),
            ("Correlation", f"{corr:.2f}", "How closely the two move together"),
            ("Tracking error", f"{relative.tracking_error:.2%}", "Annualised volatility of the return difference"),
            ("Information ratio", f"{relative.information_ratio:.2f}", "Excess return per unit of tracking error"),
        ]):
            col.metric(name, value, help=help_text)

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.04,
                    subplot_titles=["", "Annualised volatility"])
fig.add_trace(go.Candlestick(x=view.index, open=view.open, high=view.high, low=view.low, close=view.close,
                             name=ticker, increasing_line_color=ui.UP, decreasing_line_color=ui.DOWN,
                             increasing_fillcolor=ui.UP, decreasing_fillcolor=ui.DOWN), row=1, col=1)
for n, slot in zip(smas, (1, 7, 4, 5)):  # slots away from the green/red candles
    fig.add_trace(go.Scatter(x=view.index, y=view[f"SMA {n}"], name=f"SMA {n}", mode="lines",
                             line_color=ui.series_color(slot)), row=1, col=1)
fig.add_trace(go.Scatter(x=view.index, y=view["Vol"], name=f"Vol {months}m · {window} {PERIOD}s", mode="lines",
                         line_color=ui.series_color(2)), row=2, col=1)
fig.add_hline(y=risk.vol_ann, line_dash="dot", line_color=ui.MUTED, row=2, col=1,
              annotation_text=f"range average {risk.vol_ann:.1%}", annotation_position="top right")
fig.update_yaxes(tickformat=".0%", rangemode="tozero", row=2, col=1)
fig.update_layout(xaxis_rangeslider_visible=False, title=ui.label(ticker))
ui.shade_stress(fig, stress_periods, s, subplots=True)
with st.container(border=True):
    if st.checkbox("Log price scale", key="overview_log",
                   help="Equal percentage moves take equal vertical space, so early years stay readable"):
        fig.update_yaxes(type="log", row=1, col=1)
    ui.chart(fig, height=640)

row = inst.set_index("ticker", drop=False).loc[ticker]
about = ui.description(ticker)
with st.container(border=True):
    st.markdown(f"**:material/info: About {row['name']}**")
    details = [row.type, row.asset_class, row.exchange, row.currency, about.get("sector"), about.get("industry")]
    st.caption(" · ".join(str(d) for d in details if d and str(d) != "nan"))
    st.markdown(about["text"])
    for note in about["notes"]:
        st.markdown(f":material/label: {note}")
    link = f" · [Website]({about['website']})" if about.get("website") else ""
    st.caption(f"Source: {about['source']}{link}")

log = ui.freshness(ticker)
if len(log):
    st.caption(f"Data to {log.last_date[0]:%d %b %Y} · refreshed {log.last_run[0]:%d %b %Y %H:%M} · "
               f"native currency · {ui.FREQS[s.freq].lower()} bars")

with st.container(border=True):
    st.markdown(f"**:material/table_rows: Risk metrics for all {len(pool)} instruments in this selection**",
                help="Uses the asset class, region and type chosen above, the sidebar date range and interval, and "
                     "each instrument's own trading days.")
    if st.toggle("Show table", value=len(pool) <= 100, key=f"risk_table_{asset_class}_{region}_{kind}"):
        wide = ui.price_matrix(pool.ticker.tolist(), replace(s, currency="native"))
        bars_wide = resample.last(wide, s.freq)
        # Each instrument uses its own trading days: carry the last price over gaps, then blank the gap rows.
        member_returns = bars_wide.ffill().pct_change(fill_method=None).where(bars_wide.notna())
        table = stats.risk.cross_section(member_returns, periods, ui.risk_free(s, member_returns.index))
        table = pool.set_index("ticker")[["name", "exchange", "type"]].join(table, how="inner")
        usable = (table.observations >= 20) & (table["std"] > 0)
        if (~usable).any():
            st.caption("Excluded (fewer than 20 returns or no price movement, e.g. suspended): "
                       + ", ".join(table.index[~usable]))
        table = table[usable].sort_values("vol_ann", ascending=False)
        st.dataframe(table, height=min(640, 36 * (len(table) + 1)), column_config={
            "name": st.column_config.TextColumn("Name", width="medium"), "exchange": "Region", "type": "Type",
            "observations": st.column_config.NumberColumn("Obs", help=f"Number of {EVERY} returns"),
            **{key: st.column_config.NumberColumn(name, help=help_text,
                                                  format="%.2f" if key in ("sharpe", "sortino") else "percent")
               for key, (name, help_text) in RISK_COLUMNS.items()},
        })
        if ticker in table.index:
            rank = int(table.index.get_loc(ticker)) + 1  # the table is sorted most volatile first
            st.caption(f"**{ticker}** ranks {rank} of {len(table)} by volatility "
                       f"({(rank - 1) / len(table):.0%} of this selection is more volatile).")
        ui.download(table, f"risk_{asset_class.lower().replace(' ', '_')}")
        ui.explain(interpret.risk_table(table, asset_class, PERIOD))
