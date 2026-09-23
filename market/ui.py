"""Streamlit helpers shared by every page: sidebar settings, cached loaders, formatting."""
from dataclasses import dataclass, replace
from datetime import date, timedelta

import pandas as pd
import streamlit as st

from market import adjust, benchmarks, cash, descriptions, filters, indicators, resample, store, stress, universe
from market.config import MIN_OBS, RISK_FREE
from market import returns as rets

RANGES = ["1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y", "10Y", "Max", "Custom"]
DAYS = {"1M": 31, "3M": 92, "6M": 183, "1Y": 365, "3Y": 1096, "5Y": 1827, "10Y": 3653}
FREQS = {"D": "Daily", "W": "Weekly", "M": "Monthly"}
CURRENCIES = ["native", "AUD", "USD"]
DEFAULT_BASKET = ["^AXJO", "^GSPC", "BHP.AX", "RIO.AX", "FMG.AX", "CBA.AX", "SPY"]
PCT = st.column_config.NumberColumn(format="percent")
NUM = st.column_config.NumberColumn(format="%.3f")
UP, DOWN = "#0ca30c", "#d03b3b"
MUTED = "#7d8ca6"
MARKETS = ["^AXJO", "^GSPC", "^N225", "^FTSE", "GC=F", "CL=F", "^TNX", "AUDUSD=X"]
FILTERS = {"asset_class": "Asset class", "exchange": "Region", "universe": "List", "sector": "Sector", "type": "Type"}
SERIES = {"light": ["#0051ff", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
          "dark": ["#4b85fe", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]}
BENCHMARK_DEFAULT = ["Equities", "Fixed income", "Cash", "Commodities", "Real estate", "Infrastructure",
                     "Currencies"]
BLUES = ["#dde8fe", "#b5cefe", "#83adfe", "#4b85fe", "#0051ff", "#0036ba", "#001775"]


@dataclass(frozen=True)
class Settings:
    start: date | None
    end: date
    freq: str
    kind: str
    currency: str
    basket: tuple[str, ...]
    adjust: frozenset[str] = frozenset()

    def on(self, key: str) -> bool:
        return key in self.adjust


@st.cache_data(ttl=300)
def instruments() -> pd.DataFrame:
    loaded = store.query("SELECT DISTINCT ticker FROM prices")["ticker"]
    return universe.sort(store.instruments().merge(loaded, on="ticker"))


@st.cache_resource(ttl=300)
def names() -> dict[str, str]:
    return instruments().set_index("ticker")["name"].to_dict()


@st.cache_data(ttl=300)
def prices(tickers: tuple[str, ...], start=None, end=None) -> pd.DataFrame:
    return store.prices(list(tickers), start, end)


@st.cache_data(ttl=300)
def db(name: str):
    """Cached read from a no-argument `store` function; every refresh clears the cache anyway."""
    return getattr(store, name)()


@st.cache_data(ttl=300)
def freshness(ticker: str) -> pd.DataFrame:
    return store.query("SELECT last_date, last_run FROM ingest_log WHERE ticker = ?", [ticker])


@st.cache_data(ttl=300)
def snapshot(tickers: tuple[str, ...], as_of: date, total_return: bool = False) -> pd.DataFrame:
    long = store.prices([*tickers, *indicators.BENCHMARKS], as_of - timedelta(days=420), as_of)
    return indicators.snapshot(long, total_return=total_return)


@st.cache_data(ttl=3600, show_spinner=False)
def description(ticker: str) -> dict:
    return descriptions.describe(instruments().set_index("ticker", drop=False).loc[ticker])


def clear_cache() -> None:
    st.cache_data.clear()
    st.cache_resource.clear()


def label(ticker: str) -> str:
    name = names().get(ticker)
    return f"{ticker} · {name}" if name and name != ticker else ticker


def _start(choice: str, end: date) -> date | None:
    return None if choice == "Max" else date(end.year, 1, 1) if choice == "YTD" else end - timedelta(days=DAYS[choice])


def sidebar() -> Settings:
    ss, qp = st.session_state, st.query_params
    tickers = instruments()["ticker"].tolist()
    ss.setdefault("range", qp.get("range") if qp.get("range") in RANGES else "5Y")
    ss.setdefault("freq", qp.get("freq") if qp.get("freq") in FREQS else "D")
    ss.setdefault("kind", "simple")
    ss.setdefault("currency", "native")
    ss.setdefault("basket", qp.get("basket", ",".join(DEFAULT_BASKET)).split(","))
    if "pending_basket" in ss:
        ss.basket = ss.pop("pending_basket")
    ss.basket = [t for t in ss.basket if t in tickers]

    with st.sidebar:
        st.caption("SETTINGS")
        st.selectbox("Date range", RANGES, key="range")
        end = date.today()
        start = None if ss.range == "Custom" else _start(ss.range, end)
        if ss.range == "Custom":
            picked = st.date_input("From / to", value=(end - timedelta(days=365), end))
            start, end = (picked[0], picked[-1]) if len(picked) else (None, end)
        st.segmented_control("Interval", list(FREQS), key="freq", format_func=FREQS.get, required=True)
        st.segmented_control("Returns", ["simple", "log"], key="kind", required=True)
        st.selectbox("Currency", CURRENCIES, key="currency",
                     help="Convert prices with AUDUSD=X. 'native' keeps each instrument's own currency.")
        st.multiselect("Basket", tickers, key="basket", format_func=label,
                       help="Default tickers for Compare, Correlation, Statistics and Code Lab.")
        enabled = _adjustments(qp)

    qp.update(range=ss.range, freq=ss.freq, basket=",".join(ss.basket), adjust=",".join(sorted(enabled)))
    ss.settings = Settings(start, end, ss.freq, ss.kind, ss.currency, tuple(ss.basket), frozenset(enabled))
    return ss.settings


def _adjustments(qp) -> set[str]:
    """Opt-in methodology corrections. Everything is off by default so the pages report the data as it is."""
    ss = st.session_state
    for key in adjust.ADJUSTMENTS:
        ss.setdefault(f"adj_{key}", key in qp.get("adjust", "").split(","))
    ss.setdefault("adj_open", False)
    on = {key for key in adjust.ADJUSTMENTS if ss[f"adj_{key}"]}
    # Ticking a box reruns the script, which would otherwise collapse the panel again.
    with st.expander(f"Adjustments ({len(on)} on)" if on else "Adjustments", expanded=ss.adj_open):
        st.caption("Off by default: results use the data exactly as it is.")
        for key, (name, why) in adjust.ADJUSTMENTS.items():
            st.checkbox(name, key=f"adj_{key}", help=why, on_change=_keep_panel_open)
    return {key for key in adjust.ADJUSTMENTS if ss[f"adj_{key}"]}


def _keep_panel_open() -> None:
    st.session_state.adj_open = True


def adjustments_caption(s: Settings, *keys: str) -> None:
    """Name the adjustments affecting this page, so a screenshot says what produced it."""
    if names_on := adjust.active(s.adjust, *keys):
        st.caption(f":material/tune: Adjusted: {', '.join(names_on)}")


def risk_free(s: Settings, index) -> float | pd.Series:
    """Flat assumption unless the live cash rate is switched on."""
    return cash.annual_rate(index, _cash_currency(s)) if s.on("live_cash") else RISK_FREE


def cash_label(s: Settings, index) -> str:
    return cash.label(index, _cash_currency(s))


def _cash_currency(s: Settings) -> str:
    return s.currency if s.currency in cash.SOURCES else "AUD"


def settings() -> Settings:
    return st.session_state.settings


def set_basket(tickers: list[str]) -> None:
    st.session_state.pending_basket = list(tickers)
    st.rerun()


def require(items, message: str = "No data yet. Load some in **Data Manager**.") -> None:
    if not len(items):
        st.info(message)
        st.stop()


def _sources_present(src, wide: pd.DataFrame) -> bool:
    return src in wide if isinstance(src, str) else all(t in wide for t in src)


def price_matrix(tickers, s: Settings, field: str = "adj_close", groups: dict | None = None) -> pd.DataFrame:
    """Wide price matrix. A `groups` value is either a ticker or a {ticker: weight} blend, and the
    resulting series is labelled with the group's name."""
    if groups:
        needed = [t for src in groups.values() for t in ([src] if isinstance(src, str) else src)]
        wide = price_matrix(needed, s, field)
        built = {name: wide[src] if isinstance(src, str) else rets.blend(wide, src)
                 for name, src in groups.items() if _sources_present(src, wide)}
        return pd.DataFrame(built)
    tickers = list(dict.fromkeys(tickers))
    convert = s.currency != "native" and field != "volume"
    long = prices(tuple(tickers + ["AUDUSD=X"] * convert), s.start, s.end)
    wide = rets.price_matrix(long, field)
    if convert and "AUDUSD=X" in wide:
        currencies = instruments().set_index("ticker")["currency"].to_dict()
        wide = rets.to_currency(wide, currencies, wide["AUDUSD=X"], s.currency)
    return wide[[t for t in tickers if t in wide]]


def return_matrix(tickers, s: Settings, align: str = "inner", groups: dict | None = None) -> pd.DataFrame:
    prices = price_matrix(tickers, s, groups=groups)
    minimum = min(MIN_OBS, max(5, len(prices) // 2))  # short stress windows need a lower bar than a 5-year range
    r = rets.compute(prices, s.freq, s.kind, align, minimum)
    if r.attrs.get("dropped"):
        st.caption(f"Skipped (not enough data in range): {', '.join(r.attrs['dropped'])}")
    return r


def series_matrix(tickers, s: Settings, series: str, groups: dict | None = None) -> pd.DataFrame:
    """'Returns', 'Price' or 'Volume' matrix at the sidebar interval."""
    if series == "Returns":
        return return_matrix(tickers, s, groups=groups)
    if series == "Volume":
        return resample.total(price_matrix(tickers, s, "volume", groups), s.freq)
    return resample.last(price_matrix(tickers, s, groups=groups), s.freq)


def subject_picker() -> tuple[list[str], dict | None]:
    """Pick what to analyse: individual tickers, or asset classes represented by their standard benchmarks."""
    inst, s = instruments(), settings()
    mode = st.segmented_control("Analyse", ["Tickers", "Asset classes"], default="Tickers", required=True)
    if mode == "Tickers":
        return st.multiselect("Tickers", inst.ticker, default=list(s.basket), format_func=label), None

    basis = benchmark_basis()
    available = [c for c in universe.ASSET_CLASSES if c in benchmarks.classes(basis)]
    c1, c2 = st.columns(2)
    classes = c1.multiselect("Asset classes", available,
                             default=[c for c in BENCHMARK_DEFAULT if c in available] or available[:1])
    regions = c2.multiselect("Regions", benchmarks.REGIONS[basis], placeholder="Headline benchmark for each class")
    chosen = benchmark_table(benchmarks.select(classes, regions, basis), inst)
    return chosen.label.tolist(), dict(zip(chosen.label, chosen.source))


def benchmark_basis(container=None) -> str:
    """Which benchmark category represents each asset class."""
    return (container or st).segmented_control(
        "Benchmarks", benchmarks.BASES, key="bench_basis", default=benchmarks.BASES[0], required=True,
        help="Standard: the index most widely quoted for each asset class. "
             "APRA: the indices prescribed for the Australian superannuation performance test.")


def benchmark_table(chosen: pd.DataFrame, inst: pd.DataFrame) -> pd.DataFrame:
    """Show what stands in for each benchmark, and drop the ones with nothing behind them."""
    held = set(inst.ticker)
    chosen = chosen.assign(source=chosen.apply(benchmarks.source, axis=1))
    usable = chosen.source.map(lambda src: _sources_held(src, held))
    missing, chosen = chosen[~usable], chosen[usable]
    with st.expander(f"Benchmarks used ({len(chosen)})"):
        st.dataframe(chosen[["label", "benchmark", "provider", "ticker", "code", "series", "note"]], hide_index=True,
                     column_config={"label": "Analysed as", "code": "APRA code", "series": st.column_config.TextColumn(
                         "Series", help="index: the benchmark itself · tracker: an ETF/ETN tracking it · "
                                        "proxy: closest stand-in · composite: a weighted blend of other rows")})
        for _, row in chosen[chosen.series == "composite"].iterrows():
            st.caption(f"**{row.label}** = " + " + ".join(f"{w:.1%} {t}" for t, w in row.blend.items()))
        if len(missing):
            st.caption(f"No series available for {len(missing)}: "
                       + "; ".join(f"**{r.label}** — {r.note}" for _, r in missing.iterrows()))
    return chosen


def _sources_held(src, held: set) -> bool:
    if src is None:
        return False
    return src in held if isinstance(src, str) else all(t in held for t in src)


SECTIONS = (("findings", ":material/query_stats: What the numbers show"),
            ("implications", ":material/insights: Why it matters"),
            ("caveats", ":material/warning: Keep in mind"))


def bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def explain(reading) -> None:
    """'What this means' card: headline, findings, implications, caveats and definitions."""
    if reading is None:
        return
    with st.container(border=True):
        st.markdown("**:material/lightbulb: What this means**")
        st.markdown(f"##### {reading.headline}")
        for attr, title in SECTIONS:
            if items := getattr(reading, attr):
                st.markdown(f"**{title}**")
                st.markdown(bullets(items))
        if definitions := reading.definitions():
            with st.expander("Definitions"):
                st.markdown(bullets([f"**{term}:** {text}" for term, text in definitions.items()]))
        st.caption("Automatic interpretation from fixed statistical rules. Educational, not financial advice.")


def dark() -> bool:
    return st.context.theme.type == "dark"


def series_color(slot: int) -> str:
    """Categorical palette colour by fixed slot (1-8), stepped for the current theme."""
    return SERIES["dark" if dark() else "light"][slot - 1]


def diverging() -> list:
    """Red (negative) → neutral gray → blue (positive), for correlations and returns."""
    return [[0, "#e34948"], [0.5, "#1e3554" if dark() else "#eef3fb"], [1, "#4b85fe" if dark() else "#0051ff"]]


def sequential() -> list[str]:
    """Single-hue blue ramp; the low end recedes toward the page surface."""
    return BLUES[::-1] if dark() else BLUES


def header(title: str, caption: str) -> None:
    st.title(title)
    st.caption(caption)


def chart(fig, container=None, height: int | None = None) -> None:
    """Apply shared layout (legend, hover, margins, thin lines) and render a Plotly figure."""
    pointwise = any(t.type in ("heatmap", "box") or (t.type == "scatter" and "markers" in (t.mode or ""))
                    for t in fig.data)
    fig.update_layout(
        height=height or fig.layout.height or 380, margin=dict(l=8, r=16, t=56 if fig.layout.title.text else 12, b=8),
        title=dict(font_size=14, x=0, xanchor="left"), bargap=0.25,
        showlegend=len(fig.data) > 1 if fig.layout.showlegend is None else fig.layout.showlegend,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1, title_text=""),
        hovermode="closest" if pointwise else "x unified",
    )
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    fig.update_traces(line_width=2, selector=dict(type="scatter", mode="lines"))
    fig.update_traces(marker_size=7, selector=dict(type="scatter", mode="markers"))
    (container or st).plotly_chart(fig)


def stress_picker(key: str, container=None) -> list[str]:
    """Choose which stress periods to shade on this page's charts."""
    return (container or st).multiselect(
        "Stress periods on chart", [p.name for p in stress.catalogue()], key=key, placeholder="None",
        help="Shade these crisis and shock windows on the charts below")


def stress_window(key: str, s: Settings, container=None):
    """Optionally narrow a page's analysis to one stress period instead of the sidebar date range."""
    periods = stress.catalogue()
    labels = ["Sidebar date range"] + [p.label for p in periods]
    choice = (container or st).selectbox("Analysis window", labels, key=key,
                                         help="Run this page over a past crisis or shock instead of the sidebar range")
    if choice == labels[0]:
        return s, None
    period = periods[labels.index(choice) - 1]
    return replace(s, start=pd.Timestamp(period.start).date(), end=pd.Timestamp(period.end).date()), period


def stress_caption(period) -> None:
    if period:
        st.caption(f"Analysing the {period.name} window ({period.start} to {period.end}): {period.description}")


def shade_stress(fig, names, s: Settings, subplots: bool = False) -> None:
    """Shade the chosen stress periods where they fall inside the chart's date range."""
    visible = [p for p in stress.overlapping(s.start, s.end) if p.name in set(names or ())]
    for period in visible:
        fig.add_vrect(x0=period.start, x1=period.end, fillcolor=MUTED, opacity=0.14, line_width=0, layer="below",
                      annotation_text=period.name if len(visible) <= 8 else None, annotation_position="top left",
                      annotation_font_size=10, **({"row": "all", "col": 1} if subplots else {}))


def market_tiles(tickers: list[str] = MARKETS) -> None:
    """Row of bordered tiles: last level, daily change and a 3-month sparkline."""
    start = date.today() - timedelta(days=120)
    long = prices(tuple(tickers), start)
    shown = [t for t in tickers if t in set(long.ticker)]
    with st.container(key="tiles"):
        cols = [c for _ in range(0, len(shown), 4) for c in st.columns(4)]
        for col, ticker in zip(cols, shown):
            close = long.loc[long.ticker == ticker, "close"].tail(63)
            last = close.iloc[-1]
            col.metric(names().get(ticker, ticker), f"{last:,.0f}" if last >= 1000 else f"{last:,.4g}",
                       f"{last / close.iloc[-2] - 1:+.2%}", chart_data=close.round(4).tolist(), chart_type="area",
                       border=True)


def _ordered(key: str, values) -> list:
    order = {"asset_class": universe.ASSET_CLASSES, "universe": universe.PICKER_ORDER}.get(key)
    return sorted(values, key=lambda v: (order.index(v) if v in order else len(order), v)) if order else sorted(values)


def instrument_filters(prefix: str, keys=tuple(FILTERS), defaults: dict | None = None) -> tuple[dict, str]:
    """Cascading multiselects: each only offers values still present after the earlier selections."""
    inst, ss, selected = instruments(), st.session_state, {}
    for key, value in (defaults or {}).items():
        ss.setdefault(f"{prefix}_{key}", value)
    cols = st.columns(len(keys) + 1)
    for col, key in zip(cols, keys):
        state = f"{prefix}_{key}"
        options = _ordered(key, filters.universe(inst, **selected)[key].dropna().unique())
        ss[state] = [v for v in ss.get(state, []) if v in options]
        selected[key] = col.multiselect(FILTERS[key], options, key=state, placeholder="All",
                                        format_func=lambda v: universe.LABELS.get(v, v))
    search = cols[-1].text_input("Search", key=f"{prefix}_search", placeholder="ticker or name")
    return selected, search


def percent(df: pd.DataFrame, cols) -> dict:
    return {c: PCT for c in cols if c in df}


def download(df: pd.DataFrame, name: str) -> None:
    st.download_button("Download CSV", df.to_csv().encode(), f"{name}.csv", "text/csv", icon=":material/download:")


def conclusions(df: pd.DataFrame) -> None:
    """Show a test-results table with numeric formatting."""
    st.dataframe(df, hide_index=True, column_config={"statistic": NUM, "p_value": st.column_config.NumberColumn(
        format="%.4g"), "conclusion": st.column_config.TextColumn(width="large")})
