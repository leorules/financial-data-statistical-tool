from dataclasses import replace
from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

from market import interpret, stress, ui

s = ui.settings()
inst = ui.instruments()
ui.require(inst)
CONTEXT = {"1 month": 31, "3 months": 92, "6 months": 183, "1 year": 365, "2 years": 730}
PCT_COLUMNS = {"event_return": "Event return", "trough_return": "Low vs start", "max_drawdown": "Max drawdown",
               "worst_day": "Worst day", "vol_before": "Vol (year before)", "vol_during": "Vol (during)"}

ui.header("Stress Periods", "How markets behaved through past crises and shocks, and what a repeat could mean for "
                            "a basket.")

catalogue = stress.catalogue()
labels = [p.label for p in catalogue]
with st.container(border=True):
    c1, c2 = st.columns([3, 2], vertical_alignment="bottom")
    default = next(i for i, p in enumerate(catalogue) if p.name == "COVID-19 crash")
    period = catalogue[labels.index(c1.selectbox("Stress period", labels, index=default))]
    horizon = c2.select_slider("Show around the period", list(CONTEXT), value="6 months")
    context = CONTEXT[horizon]
    tickers, groups = ui.subject_picker()
ui.require(tickers, "Pick at least one ticker or asset class.")

with st.expander("Add or remove your own stress periods"):
    with st.form("custom_event", clear_on_submit=True, border=False):
        c1, c2, c3 = st.columns([2, 1, 1])
        name = c1.text_input("Name", placeholder="e.g. 2024 AI sell-off")
        start = c2.date_input("Start", value=date.today() - timedelta(days=60))
        end = c3.date_input("End", value=date.today() - timedelta(days=30))
        description = st.text_input("Description", placeholder="What happened")
        if st.form_submit_button("Save", icon=":material/add:") and name and start < end:
            stress.save_custom(stress.StressPeriod(name, str(start), str(end), "Custom", description))
            st.rerun()
    own = [p.name for p in stress.custom()]
    removing = st.selectbox("Remove a custom stress period", own, index=None,
                            placeholder="None saved" if not own else "Choose…")
    if st.button("Remove", disabled=not removing, icon=":material/delete:"):
        stress.delete_custom(removing)
        st.rerun()

start, end = pd.Timestamp(period.start), pd.Timestamp(period.end)
with st.container(border=True):
    st.markdown(f"**:material/history: {period.name}** · {period.category} · {start:%d %b %Y} – {end:%d %b %Y} "
                f"({(end - start).days} days)")
    st.markdown(period.description)

history = replace(s, start=(start - timedelta(days=400)).date(), end=date.today())
prices = ui.price_matrix(tickers, history, groups=groups)
table = stress.impact(prices, period)
missing = [t for t in tickers if t not in table.index]
if missing:
    st.caption(f"No price history before the event start, so excluded: {', '.join(missing)}")
ui.require(table, "None of the selected series have data covering this event. Try indices, or an asset class.")

window = prices.loc[start - timedelta(days=context):end + timedelta(days=context), table.index]
rebased = window / prices.loc[:start, table.index].ffill().iloc[-1] * 100
with st.container(border=True):
    fig = px.line(rebased, title="Indexed to 100 at the start of the stress period", labels={"value": "", "date": ""})
    fig.add_vrect(x0=start, x1=end, fillcolor=ui.MUTED, opacity=0.15, line_width=0, layer="below",
                  annotation_text=period.name, annotation_position="top left")
    fig.add_hline(y=100, line_color=ui.MUTED, line_dash="dot")
    ui.chart(fig, height=460)

with st.container(border=True):
    st.markdown("**Impact**", help="Measured from the last close on or before the start date to the end date. "
                                         "Recovery = first date back at the pre-event level.")
    shown = table.sort_values("event_return")
    st.dataframe(shown, column_config={
        **{c: st.column_config.NumberColumn(label, format="percent") for c, label in PCT_COLUMNS.items()},
        "vol_ratio": st.column_config.NumberColumn("Vol ×", format="%.1f×"),
        "trough_date": st.column_config.DateColumn("Low on", format="D MMM YYYY"),
        "recovery_date": st.column_config.DateColumn("Recovered on", format="D MMM YYYY"),
        "days_to_recover": st.column_config.NumberColumn("Days to recover", format="%d"),
    })
    ui.download(shown, f"stress_{period.name.lower().replace(' ', '_')}")

corr_before, corr_during = stress.correlation_shift(prices[table.index], period)
basket = None
left, right = st.columns([2, 3])
with left.container(border=True, height="stretch"):
    st.markdown("**Correlation shift**", help="Average pairwise correlation of daily returns")
    if pd.notna(corr_before) and pd.notna(corr_during):
        c1, c2 = st.columns(2)
        c1.metric("Year before", f"{corr_before:.2f}")
        c2.metric("During period", f"{corr_during:.2f}", f"{corr_during - corr_before:+.2f}", delta_color="inverse")
    else:
        st.caption("Needs at least two series with overlapping data.")
with right.container(border=True):
    st.markdown(f"**Stress test: a buy-and-hold basket through the period and {horizon} after**")
    c1, c2 = st.columns([2, 1])
    weights = c1.data_editor(pd.DataFrame({"weight": 1.0}, index=table.index.rename("series")), key=f"w_{period.name}",
                             column_config={"weight": st.column_config.NumberColumn("Weight", min_value=0.0)})
    amount = c2.number_input("Amount ($)", min_value=0.0, value=10_000.0, step=1_000.0)
    if weights.weight.sum() > 0:
        basket = stress.stress_test(prices, weights.weight, period, after_days=context)
        for label, value in (("End of period", basket[:end].iloc[-1]), ("Lowest point", basket.min()),
                             (f"{horizon} after", basket.iloc[-1])):
            c2.metric(label, f"${amount * value:,.0f}", f"{value - 1:+.1%}")
        fig = px.area(basket * amount, labels={"value": "", "date": ""}).update_layout(showlegend=False)
        fig.add_vrect(x0=start, x1=end, fillcolor=ui.MUTED, opacity=0.15, line_width=0, layer="below")
        ui.chart(fig, height=240)

ui.explain(interpret.stress_period(period, table, corr_before, corr_during, basket, amount, horizon))
