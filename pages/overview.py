from dataclasses import replace

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from market import indicators, interpret, resample, stats, store, ui, universe

s = ui.settings()
inst = ui.instruments()
ui.require(inst)
PERIOD = {"D": "day", "W": "week", "M": "month"}[s.freq]
VOL_WINDOWS = {"D": [20, 63, 126, 252], "W": [4, 13, 26, 52], "M": [3, 6, 12]}[s.freq]
RISK_COLUMNS = {
    "std": ("Std dev", f"Standard deviation of {PERIOD}ly returns"),
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
    kind = c2.selectbox("Type", ["All", *pool.type.dropna().unique()])
    pool = pool if kind == "All" else pool[pool.type == kind]
    tickers = pool.ticker.tolist()
    ticker = c3.selectbox("Instrument", tickers, index=tickers.index("^AXJO") if "^AXJO" in tickers else 0,
                          format_func=ui.label)
    smas = c4.multiselect("Moving averages", [20, 50, 100, 200], default=[50, 200])
    c5, c6 = st.columns(2)
    vol_windows = c5.multiselect("Rolling volatility on chart", VOL_WINDOWS, default=VOL_WINDOWS[:2],
                                 format_func=lambda n: f"{n} {PERIOD}s",
                                 help="Annualised standard deviation of returns over each trailing window")
    stress_periods = ui.stress_picker("overview_stress", c6)

daily = ui.prices((ticker,)).set_index("date").drop(columns="ticker")
bars = resample.ohlcv(daily, s.freq)
for n in smas:
    bars[f"SMA {n}"] = indicators.sma(bars["close"], n)
bar_returns = bars["adj_close"].pct_change(fill_method=None)
for n in vol_windows:  # computed on full history so the start of the range is not blank
    bars[f"Vol {n}"] = bar_returns.rolling(n).std() * np.sqrt(stats.periods_per_year(bars.index))
view = bars.loc[str(s.start or bars.index[0]):str(s.end)]
ui.require(view, "No data in the selected range.")

r = view["adj_close"].pct_change(fill_method=None).dropna()
periods = stats.periods_per_year(view.index)
risk = stats.risk.cross_section(r.to_frame(ticker), periods).iloc[0]
k = st.columns(4)
k[0].metric("Last close", f"{view.close.iloc[-1]:,.2f}", f"{view.close.pct_change().iloc[-1]:+.2%}", border=True)
k[1].metric("Period return", f"{view.adj_close.iloc[-1] / view.adj_close.iloc[0] - 1:+.2%}", border=True)
k[2].metric("Annual return", f"{(view.adj_close.iloc[-1] / view.adj_close.iloc[0]) ** (periods / max(len(r), 1)) - 1:+.2%}",
            border=True, help="Compound annual growth rate over the selected range")
k[3].metric("RSI (14)", f"{indicators.rsi(bars.close).iloc[-1]:.0f}", border=True,
            help="Above 70 is often read as overbought, below 30 as oversold.")

with st.container(border=True):
    st.markdown(f"**:material/shield: Risk** · {len(r)} {PERIOD}ly returns, {view.index[0]:%d %b %Y} – {view.index[-1]:%d %b %Y}")
    tiles = [(key, *RISK_COLUMNS[key]) for key in ("std", "vol_ann", "var_95", "cvar_95", "var_99", "max_drawdown",
                                                   "sharpe", "sortino")]
    for line in (tiles[:4], tiles[4:]):
        for col, (key, name, help_text) in zip(st.columns(4), line):
            value = f"{risk[key]:.2f}" if key in ("sharpe", "sortino") else f"{risk[key]:.2%}"
            col.metric(name, value, help=help_text)

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.04,
                    subplot_titles=["", "Annualised volatility"])
fig.add_trace(go.Candlestick(x=view.index, open=view.open, high=view.high, low=view.low, close=view.close,
                             name=ticker, increasing_line_color=ui.UP, decreasing_line_color=ui.DOWN,
                             increasing_fillcolor=ui.UP, decreasing_fillcolor=ui.DOWN), row=1, col=1)
for n, slot in zip(smas, (1, 7, 4, 5)):  # slots away from the green/red candles
    fig.add_trace(go.Scatter(x=view.index, y=view[f"SMA {n}"], name=f"SMA {n}", mode="lines",
                             line_color=ui.series_color(slot)), row=1, col=1)
for n, slot in zip(vol_windows, (2, 3, 5, 8)):
    fig.add_trace(go.Scatter(x=view.index, y=view[f"Vol {n}"], name=f"Vol {n}{PERIOD[0]}", mode="lines",
                             line_color=ui.series_color(slot)), row=2, col=1)
fig.add_hline(y=risk.vol_ann, line_dash="dot", line_color=ui.MUTED, row=2, col=1,
              annotation_text=f"range average {risk.vol_ann:.1%}", annotation_position="top right")
fig.update_yaxes(tickformat=".0%", rangemode="tozero", row=2, col=1)
fig.update_layout(xaxis_rangeslider_visible=False, title=ui.label(ticker))
ui.shade_stress(fig, stress_periods, s, subplots=True)
with st.container(border=True):
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

log = store.query("SELECT last_date, last_run FROM ingest_log WHERE ticker = ?", [ticker])
if len(log):
    st.caption(f"Data to {log.last_date[0]:%d %b %Y} · refreshed {log.last_run[0]:%d %b %Y %H:%M} · "
               f"native currency · {ui.FREQS[s.freq].lower()} bars")

with st.container(border=True):
    st.markdown(f"**:material/table_rows: Risk metrics for all {len(pool)} instruments in this selection**",
                help="Uses the asset class, region and type chosen above, the sidebar date range and interval, and "
                     "each instrument's own trading days.")
    if st.toggle("Show table", value=len(pool) <= 100, key=f"risk_table_{asset_class}_{region}_{kind}"):
        wide = ui.price_matrix(pool.ticker.tolist(), replace(s, currency="native"))
        member_returns = resample.last(wide, s.freq).apply(lambda col: col.dropna().pct_change())
        table = stats.risk.cross_section(member_returns, periods)
        table = pool.set_index("ticker")[["name", "exchange", "type"]].join(table, how="inner")
        usable = (table.observations >= 20) & (table["std"] > 0)
        if (~usable).any():
            st.caption("Excluded (fewer than 20 returns or no price movement, e.g. suspended): "
                       + ", ".join(table.index[~usable]))
        table = table[usable].sort_values("vol_ann", ascending=False)
        st.dataframe(table, height=min(640, 36 * (len(table) + 1)), column_config={
            "name": st.column_config.TextColumn("Name", width="medium"), "exchange": "Region", "type": "Type",
            "observations": st.column_config.NumberColumn("Obs", help=f"Number of {PERIOD}ly returns"),
            **{key: st.column_config.NumberColumn(name, help=help_text,
                                                  format="%.2f" if key in ("sharpe", "sortino") else "percent")
               for key, (name, help_text) in RISK_COLUMNS.items()},
        })
        ui.download(table, f"risk_{asset_class.lower().replace(' ', '_')}")
        ui.explain(interpret.risk_table(table, asset_class, PERIOD))
