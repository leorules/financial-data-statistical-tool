from datetime import timedelta

import plotly.express as px
import streamlit as st

from market import benchmarks, filters, interpret, ui, universe

s = ui.settings()
inst = ui.instruments()
ui.require(inst)
HORIZONS = {"1D": "ret_1d", "1W": "ret_5d", "1M": "ret_1m", "3M": "ret_3m", "YTD": "ret_ytd", "1Y": "ret_1y"}
PERCENT = {**{v: k for k, v in HORIZONS.items()}, "vol_1y": "Vol 1Y", "max_dd_1y": "Max DD 1Y",
           "pct_from_52w_high": "vs 52w high"}
COLUMNS = ["name", "ticker", "trend", "close", *PERCENT, "asset_class", "exchange", "type"]

ui.header("Asset Classes", "Browse equities, bonds, rates, commodities, currencies and crypto by region, "
                           "and compare how they have performed.")
with st.container(border=True):
    selected, search = ui.instrument_filters("ac", keys=("asset_class", "exchange", "type"),
                                             defaults={"asset_class": ["Commodities"]})
    c1, c2 = st.columns([3, 2], vertical_alignment="bottom")
    horizon = c1.segmented_control("Performance over", list(HORIZONS), default="1M", required=True)
    members = c2.toggle("Include ASX 200 / S&P 500 stocks", help="Off shows indices, ETFs, futures, FX and crypto")

lists = None if members else [*universe.BUILT_IN, "custom"]
pool = filters.universe(inst, search, universe=lists, **selected)
ui.require(pool, "Nothing matches these filters. Refresh the lists in **Data Manager** if they are empty.")

snap = ui.snapshot(tuple(pool.ticker), s.end).merge(
    pool[["ticker", "name", "asset_class", "exchange", "type"]], on="ticker")
recent = ui.prices(tuple(pool.ticker), s.end - timedelta(days=183), s.end)
snap["trend"] = snap.ticker.map(recent.groupby("ticker")["close"].apply(list))
metric = HORIZONS[horizon]

bench = benchmarks.select(selected["asset_class"] or benchmarks.TABLE.asset_class.unique().tolist(), benchmarks.REGIONS)
bench = bench[bench.ticker.isin(set(inst.ticker))]
bench = bench.merge(ui.snapshot(tuple(bench.ticker), s.end), on="ticker")
if len(bench):
    with st.container(border=True):
        st.markdown("**Asset-class benchmarks**", help="The standard index for each asset class and region, or the ETF "
                                                        "that tracks it when the index itself is not available.")
        st.dataframe(bench[["asset_class", "region", "benchmark", "ticker", "series", *HORIZONS.values(), "vol_1y",
                            "max_dd_1y", "note"]], hide_index=True,
                     column_config={c: st.column_config.NumberColumn(label, format="percent")
                                    for c, label in PERCENT.items()} | {"asset_class": "Asset class", "region": "Region",
                                                                         "benchmark": st.column_config.TextColumn("Benchmark", width="medium")})

ranked = snap.dropna(subset=[metric]).sort_values(metric)
shown = ranked if len(ranked) <= 40 else ranked.iloc[list(range(20)) + list(range(-20, 0))]
with st.container(border=True):
    fig = px.bar(shown, x=metric, y="name", orientation="h", color=shown[metric].ge(0).map({True: "Up", False: "Down"}),
                 color_discrete_map={"Up": ui.UP, "Down": ui.DOWN}, hover_data=["ticker", "exchange"],
                 title=f"{horizon} return" + (" (top and bottom 20)" if len(ranked) > 40 else ""),
                 labels={metric: "", "name": ""})
    fig.update_xaxes(tickformat=".1%").update_yaxes(categoryorder="array", categoryarray=shown.name)
    fig.update_layout(showlegend=False)
    ui.chart(fig, height=max(300, 22 * len(shown) + 80))
    if "Rates" in set(snap.asset_class):
        st.caption("Rates show the % change in the yield itself, not the return on a bond.")

with st.container(border=True):
    table = snap[COLUMNS].sort_values(metric, ascending=False)
    st.dataframe(
        table, hide_index=True, height=min(640, 36 * (len(table) + 1)),
        column_config={c: st.column_config.NumberColumn(label, format="percent") for c, label in PERCENT.items()} | {
            "name": st.column_config.TextColumn("Name", width="medium"), "ticker": "Ticker",
            "trend": st.column_config.AreaChartColumn("6 months", width="small"),
            "close": st.column_config.NumberColumn("Last", format="%.4g"),
            "asset_class": "Asset class", "exchange": "Region", "type": "Type",
        })
    b1, b2, _ = st.columns([1, 1, 3])
    with b1:
        ui.download(table.drop(columns="trend"), "asset_classes")
    if b2.button("Use as basket", icon=":material/shopping_basket:", help="Send these tickers to the sidebar basket (max 50)"):
        ui.set_basket(table.ticker.head(50).tolist())

ui.explain(interpret.asset_classes(bench, table, horizon, metric))
