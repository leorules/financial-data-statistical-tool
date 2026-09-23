import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from market import interpret, portfolio, stats, ui

s = ui.settings()
inst = ui.instruments()
ui.require(inst)
tickers = inst.ticker.tolist()

ui.header("Portfolio", "Your holdings: value, contribution to return, risk decomposition and behaviour in past "
                       "stress periods.")
ui.adjustments_caption(s, "live_cash")

saved = portfolio.names()
with st.container(border=True):
    c1, c2, c3 = st.columns([2, 2, 1], vertical_alignment="bottom")
    name = c1.selectbox("Portfolio", saved or ["My portfolio"], index=0)
    new = c2.text_input("New portfolio", placeholder="Name and press enter to create")
    if new and new not in saved:
        portfolio.save(new, pd.DataFrame(columns=portfolio.COLUMNS), 0.0, "^AXJO")
        st.rerun()
    if c3.button("Delete", disabled=name not in saved, icon=":material/delete:"):
        portfolio.delete(name)
        st.rerun()

holdings, meta = portfolio.load(name)
with st.container(border=True):
    st.markdown("**Holdings**", help="Units held and the average price you paid. Add rows at the bottom of the table.")
    edited = st.data_editor(
        holdings if len(holdings) else pd.DataFrame(columns=portfolio.COLUMNS), num_rows="dynamic", hide_index=True,
        key=f"holdings_{name}", column_config={
            "ticker": st.column_config.SelectboxColumn("Ticker", options=tickers, required=True),
            "units": st.column_config.NumberColumn("Units", min_value=0.0, required=True),
            "cost_price": st.column_config.NumberColumn("Average cost", min_value=0.0, help="Price paid per unit")})
    c1, c2, c3 = st.columns([1, 2, 1], vertical_alignment="bottom")
    cash = c1.number_input("Cash", min_value=0.0, value=float(meta.get("cash") or 0.0), step=1000.0)
    benchmark = c2.selectbox("Benchmark", tickers, index=tickers.index(meta.get("benchmark") or "^AXJO")
                             if (meta.get("benchmark") or "^AXJO") in tickers else 0, format_func=ui.label)
    if c3.button("Save portfolio", type="primary", icon=":material/save:"):
        portfolio.save(name, edited, cash, benchmark)
        st.success(f"Saved {len(edited)} holdings.")

ui.require(edited.dropna(subset=["ticker"]), "Add at least one holding, then press Save portfolio.")
held = edited.dropna(subset=["ticker"]).query("units > 0")
prices = ui.price_matrix(held.ticker.tolist() + [benchmark], s)
missing = [t for t in held.ticker if t not in prices]
if missing:
    st.caption(f"No price data in this range for: {', '.join(missing)}")
values = portfolio.values(held, prices)
ui.require(values.columns, "None of these holdings have prices in the selected range.")
if len(values) and values.index[0] > prices.index[0]:
    limiting = portfolio.shortest_history(held, prices)
    st.caption(f"Measured from {values.index[0]:%d %b %Y}, the first date every holding has a price "
               f"({limiting} has the shortest history). Starting earlier would count its arrival as a gain.")

total = portfolio.series(values, cash)
table = portfolio.positions(held, values, cash)
returns = values.pct_change(fill_method=None)
portfolio_returns = total.pct_change(fill_method=None).dropna()
bench_returns = prices[benchmark].pct_change(fill_method=None).reindex(portfolio_returns.index)
periods = stats.periods_per_year(total.index)

k = st.columns(5)
k[0].metric("Market value", f"${total.iloc[-1]:,.0f}", f"{portfolio_returns.iloc[-1]:+.2%}", border=True)
k[1].metric("Cost", f"${table.cost.sum() + cash:,.0f}", border=True)
k[2].metric("Unrealised P&L", f"${table.profit.sum():,.0f}",
            f"{table.profit.sum() / table.cost.sum():+.1%}" if table.cost.sum() else None, border=True)
k[3].metric("Period return", f"{total.iloc[-1] / total.iloc[0] - 1:+.2%}", border=True,
            help="Change in market value over the sidebar date range (no contributions or withdrawals)")
k[4].metric(f"vs {benchmark}", f"{(total.iloc[-1] / total.iloc[0]) - (prices[benchmark].ffill().iloc[-1] / prices[benchmark].bfill().iloc[0]):+.2%}",
            border=True, help="Portfolio return minus benchmark return over the range")

left, right = st.columns([3, 2])
with left.container(border=True):
    growth = pd.DataFrame({"Portfolio": total / total.iloc[0] * 100,
                           benchmark: prices[benchmark].ffill() / prices[benchmark].ffill().bfill().iloc[0] * 100})
    fig = px.line(growth, title="Portfolio vs benchmark (indexed to 100)", labels={"value": "", "date": ""})
    ui.shade_stress(fig, ui.stress_picker("portfolio_stress"), s)
    ui.chart(fig, height=380)
with right.container(border=True):
    grouping = st.segmented_control("Weights by", ["Holding", "Asset class", "Sector"], default="Holding",
                                    required=True, key="weights_by")
    if grouping == "Holding":
        shares = table.weight.rename("weight").to_frame()
    else:
        shares = portfolio.by_group(table, inst, {"Asset class": "asset_class", "Sector": "sector"}[grouping])
    fig = px.bar(shares.sort_values("weight"), x="weight", y=shares.sort_values("weight").index, orientation="h",
                 labels={"weight": "", "y": ""}, text_auto=".1%")
    ui.chart(fig.update_layout(showlegend=False, xaxis_tickformat=".0%"), height=340)

conc = portfolio.concentration(table.weight)
with st.container(border=True):
    st.markdown("**Concentration**", help="How much of the portfolio depends on its largest positions.")
    for col, (label, value, help_text) in zip(st.columns(3), [
        ("Largest position", f"{conc['largest']:.1%}", "Weight of the single biggest holding"),
        ("Top 5 weight", f"{conc['top_5']:.1%}", "Combined weight of the five largest holdings"),
        ("Effective holdings", f"{conc['effective_holdings']:.1f}",
         f"Equivalent number of equally weighted positions, from {len(table)} actual holdings. "
         "Measured on the invested weights, so cash does not flatter it"),
    ]):
        col.metric(label, value, help=help_text)

with st.container(border=True):
    st.markdown("**Positions**")
    st.dataframe(table, column_config={
        "units": st.column_config.NumberColumn("Units", format="%.2f"),
        "cost_price": st.column_config.NumberColumn("Avg cost", format="%.2f"),
        "value": st.column_config.NumberColumn("Value", format="dollar"),
        "cost": st.column_config.NumberColumn("Cost", format="dollar"),
        "profit": st.column_config.NumberColumn("Unrealised P&L", format="dollar"),
        "return": st.column_config.NumberColumn("Return on cost", format="percent"),
        "weight": st.column_config.NumberColumn("Weight", format="percent")})
    d1, d2 = st.columns([1, 3])
    with d1:
        ui.download(table, f"portfolio_{name}")
    if d2.button("Use as basket", icon=":material/shopping_basket:",
                 help="Send these holdings to the sidebar basket for Compare, Correlation and Statistics"):
        ui.set_basket(table.index.tolist())

contribution = portfolio.contributions(values, cash)
risk_table = portfolio.risk_contributions(returns, table.weight)
left, right = st.columns(2)
with left.container(border=True):
    st.markdown("**Contribution to return**", help="Each holding's share of the portfolio's return over the range; "
                                                  "they add up to the total.")
    st.dataframe(contribution[["start_weight", "return", "contribution"]], column_config={
        c: st.column_config.NumberColumn(label, format="percent") for c, label in
        {"start_weight": "Start weight", "return": "Return", "contribution": "Contribution"}.items()})
with right.container(border=True):
    st.markdown("**Contribution to risk**", help="How much of the portfolio's volatility each holding accounts for, "
                                                 "including its correlation with the rest.")
    if len(risk_table):
        st.dataframe(risk_table, column_config={
            c: st.column_config.NumberColumn(label, format="percent") for c, label in
            {"weight": "Weight", "vol": "Volatility", "contribution": "Risk contribution",
             "share_of_risk": "Share of risk"}.items()} | {"marginal": st.column_config.NumberColumn("Marginal", format="%.3f")})

risk = stats.risk.metrics(portfolio_returns, bench_returns, periods, ui.risk_free(s, portfolio_returns.index))
with st.container(border=True):
    st.markdown(f"**:material/shield: Portfolio risk** · {len(portfolio_returns)} periods vs {benchmark}")
    if s.on("live_cash"):
        st.caption(ui.cash_label(s, portfolio_returns.index))
    tiles = [("Volatility", f"{risk.ann_vol:.1%}"), ("VaR 95%", f"{portfolio_returns.quantile(0.05):.2%}"),
             ("Max drawdown", f"{risk.max_drawdown:.1%}"), ("Sharpe", f"{risk.sharpe:.2f}"),
             ("Beta", f"{risk.get('beta', np.nan):.2f}"), ("Tracking error", f"{risk.get('tracking_error', np.nan):.1%}"),
             ("Information ratio", f"{risk.get('information_ratio', np.nan):.2f}"),
             ("Diversification", f"{(table.weight * risk_table.vol).sum() / risk.ann_vol:.2f}×" if len(risk_table) else "n/a")]
    for line in (tiles[:4], tiles[4:]):
        for col, (label, value) in zip(st.columns(4), line):
            col.metric(label, value)

history = ui.price_matrix(held.ticker.tolist(), ui.Settings(None, s.end, s.freq, s.kind, s.currency, s.basket))
stress_table = portfolio.stress_history(history, table.weight)
with st.container(border=True):
    st.markdown("**Today's portfolio through past stress periods**",
                help="Current weights applied to each historical window, held through it with no rebalancing.")
    if len(stress_table):
        st.dataframe(stress_table, column_config={
            "start": st.column_config.DateColumn("From", format="MMM YYYY"),
            "end": st.column_config.DateColumn("To", format="MMM YYYY"),
            "return": st.column_config.NumberColumn("Return", format="percent"),
            "worst": st.column_config.NumberColumn("Worst point", format="percent"),
            "covered": st.column_config.NumberColumn("Holdings with data", format="percent")})
    else:
        st.caption("None of these holdings have history covering a stress period.")

reading = interpret.portfolio(table, contribution, risk_table, risk, benchmark, total, cash)
reading.implications += interpret.portfolio_stress(stress_table, total.iloc[-1])
ui.explain(reading)
