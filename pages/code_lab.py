import matplotlib.figure
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_ace import st_ace

from market import sandbox, ui

s = ui.settings()
ss = st.session_state
ss.setdefault("code", sandbox.EXAMPLES["Correlation heatmap"])
ss.setdefault("editor_version", 0)


def load(code: str) -> None:
    ss.code, ss.editor_version = code, ss.editor_version + 1
    ss.pop("lab_run", None)


def render(obj) -> None:
    if isinstance(obj, go.Figure):
        ui.chart(obj)
    elif isinstance(obj, matplotlib.figure.Figure):
        st.pyplot(obj)
    elif isinstance(obj, (pd.DataFrame, pd.Series)):
        st.dataframe(obj)
    else:
        st.write(obj)


ui.header("Code Lab", "Write Python against your market data. The last expression, and anything passed to `show()`, is displayed.")

with st.sidebar:
    st.divider()
    st.markdown("**Snippets**")
    examples, saved = sandbox.EXAMPLES, sandbox.snippets()
    options = [f"Example: {k}" for k in examples] + list(saved)
    pick = st.selectbox("Open", options, index=None, placeholder="Choose a snippet…", label_visibility="collapsed")
    c1, c2 = st.columns(2)
    if c1.button("Open", disabled=not pick, width="stretch"):
        load(examples[pick.removeprefix("Example: ")] if pick.startswith("Example: ") else saved[pick])
        st.rerun()
    if c2.button("Delete", disabled=not pick or pick.startswith("Example: "), width="stretch"):
        sandbox.delete_snippet(pick)
        st.rerun()

editor = st.container(border=True)
with editor:
    code = st_ace(value=ss.code, language="python", theme="tomorrow_night" if ui.dark() else "chrome",
                  keybinding="vscode", font_size=14, min_lines=16, auto_update=True, key=f"ace_{ss.editor_version}")
ss.code = code

c1, c2, c3, c4 = editor.columns([1, 1, 2, 1], vertical_alignment="bottom")
if c1.button("Run", type="primary", icon=":material/play_arrow:", width="stretch"):
    ss.lab_run = sandbox.run(code, sandbox.namespace(s.basket))
if c2.button("Clear", width="stretch"):
    ss.pop("lab_run", None)
name = c3.text_input("Snippet name", placeholder="Save as…", label_visibility="collapsed")
if c4.button("Save", disabled=not name, width="stretch"):
    sandbox.save_snippet(name, code)
    st.toast(f"Saved '{name}'")

if result := ss.get("lab_run"):
    with st.container(border=True):
        st.caption(f"Output · ran in {result.seconds:.2f}s")
        if result.stdout:
            st.code(result.stdout, language="text")
        for obj in result.outputs:
            render(obj)
        if result.error:
            st.error(result.error.replace("\n", "  \n"), icon=":material/error:")

with st.expander("Reference"):
    st.markdown("""
| Name | Description |
|---|---|
| `prices(tickers, start=None, end=None, field="adj_close", freq="D")` | Wide price matrix; `field` can be open/high/low/close/adj_close/volume |
| `returns(tickers, freq="D", kind="simple", start=None, end=None)` | Aligned return matrix |
| `basket` | Tickers in the sidebar basket |
| `stats` | `descriptive`, `rolling`, `correlation`, `distribution`, `hypothesis`, `risk`, `regression`, `timeseries`, `multivariate` |
| `store` | Database access, e.g. `store.instruments()`, `store.query(sql)` |
| `benchmarks` | Asset-class benchmark table, e.g. `benchmarks.TABLE`, `benchmarks.select(["Equities"])` |
| `inflation` | CPI for the major hubs: `inflation.yoy("AUS")`, `inflation.deflate(series)`, `inflation.HUBS` |
| `factors` | Risk-factor proxies: `factors.returns(prices(factors.legs(["Size"])), ["Size"])` |
| `pd`, `np` | pandas, NumPy |
| `plt`, `sns` | matplotlib (pyplot), seaborn |
| `px`, `go` | Plotly Express, Plotly graph objects |
| `scipy`, `sm`, `smf` | SciPy (incl. `scipy.stats`), statsmodels API and formula API |
| `sklearn` | scikit-learn (import submodules, e.g. `from sklearn.cluster import KMeans`) |
| `math`, `dt` | math, datetime |
| `show(obj)` | Display a figure, DataFrame or any value |

Anything else installed in the project's environment can be imported as normal (`import seaborn.objects as so`, `from scipy.optimize import minimize`, …). Install more with `.venv/Scripts/python -m pip install <package>`.
""")
    st.caption("Code runs locally with your user permissions. Only run code you trust.")
