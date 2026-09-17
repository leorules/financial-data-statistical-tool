import pandas as pd
import plotly.express as px
import streamlit as st

from market import filters, indicators, presets, ui

s = ui.settings()
inst = ui.instruments()
ui.require(inst)
ss = st.session_state
RULE_COLUMNS = ["left", "op", "right", "right2"]
DEFAULT_RULES = [{"left": "close", "op": ">", "right": "sma_200", "right2": None}]

ui.header("Screener", "Filter by asset class, region, sector and technical rules, then send the matches to your basket.")

# Presets load before the filter widgets render so they can set widget state.
saved = presets.load_all()
filters_box = st.container(border=True)
p1, p2, p3 = filters_box.columns([4, 1, 1], vertical_alignment="bottom")
chosen = p1.selectbox("Preset", list(saved), index=None, placeholder="Load a saved preset…")
if p2.button("Load", disabled=not chosen):
    preset = saved[chosen]
    for key in ui.FILTERS:
        ss[f"scr_{key}"] = preset.get(key, [])
    ss.scr_search = preset.get("search", "")
    ss.rules = pd.DataFrame(preset.get("rules", []), columns=RULE_COLUMNS)
    ss.pop("rules_editor", None)
if p3.button("Delete", disabled=not chosen):
    presets.delete(chosen)
    st.rerun()

with filters_box:
    selected, search = ui.instrument_filters("scr", defaults={"universe": ["asx200"]})
candidates = filters.universe(inst, search, **selected)

rules_box = st.container(border=True)
rules_box.markdown("**Rules**", help="Right side can be a number (0.05 or 5%) or another metric name. All rules must match.")
ss.setdefault("rules", pd.DataFrame(DEFAULT_RULES, columns=RULE_COLUMNS))
edited = rules_box.data_editor(ss.rules, key="rules_editor", num_rows="dynamic", hide_index=True, column_config={
    "left": st.column_config.SelectboxColumn("Metric", options=indicators.METRICS, required=True),
    "op": st.column_config.SelectboxColumn("Operator", options=list(filters.OPS), required=True),
    "right": st.column_config.TextColumn("Value or metric"),
    "right2": st.column_config.TextColumn("Upper (between)"),
})
with rules_box.expander("Available metrics"):
    st.write(", ".join(f"`{m}`" for m in indicators.METRICS))

ui.require(candidates, "No instruments match the universe filters.")
snap = ui.snapshot(tuple(candidates.ticker), s.end).merge(candidates[["ticker", "name", "asset_class", "sector"]], on="ticker")
rules = filters.rules_from_records(edited.to_dict("records"))
try:
    result = filters.apply(snap, rules)
except ValueError as e:
    st.error(str(e))
    st.stop()

results_box = st.container(border=True)
results_box.markdown(f"**{len(result)} of {len(snap)} match**")
table = result[["ticker", "name", "asset_class", "sector", *indicators.METRICS]].sort_values("ret_1m", ascending=False)
event = results_box.dataframe(
    table, hide_index=True, on_select="rerun",
    selection_mode="single-row", height=420,
    column_config=ui.percent(result, [m for m in indicators.METRICS if m.startswith(("ret_", "vol_", "max_dd", "pct_"))])
    | {"close": ui.NUM, "rsi_14": st.column_config.NumberColumn(format="%.0f")},
)

b1, b2, b3, b4 = results_box.columns([1, 1, 2, 1], vertical_alignment="bottom")
with b1:
    ui.download(table, "screener")
if b2.button("Use as basket", disabled=result.empty, help="Send matches to the sidebar basket (max 50)"):
    ui.set_basket(table.ticker.head(50).tolist())
name = b3.text_input("Preset name", placeholder="e.g. Miners above 200-day")
if b4.button("Save preset", disabled=not name):
    presets.save(name, selected | {"search": search, "rules": edited.to_dict("records")})
    st.toast(f"Saved preset '{name}'")

if rows := event.selection.rows:
    ticker = table.iloc[rows[0]].ticker
    close = ui.prices((ticker,), s.start, s.end).set_index("date")["close"]
    with st.container(border=True):
        ui.chart(px.line(close, title=ui.label(ticker), labels={"value": "", "date": ""}), height=320)
