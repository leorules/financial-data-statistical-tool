import streamlit as st

from market import ingest, store, universe, ui

ui.header("Data Manager", "Download and refresh prices, manage custom tickers, and check the ingest log.")


def refresh(tickers: list[str]) -> None:
    bar = st.progress(0.0, text=f"Refreshing {len(tickers)} tickers…")
    log = ingest.refresh(tickers, on_progress=lambda p, msg: bar.progress(p, text=msg))
    ui.clear_cache()
    failed = log[log.status != "ok"]
    st.success(f"{len(log) - len(failed)}/{len(log)} tickers refreshed.", icon=":material/check_circle:")
    if len(failed):
        st.warning("Failed: " + ", ".join(failed.ticker), icon=":material/warning:")


info = store.stats()
k = st.columns(4)
k[0].metric("Tickers with data", f"{info['tickers']:,}", border=True)
k[1].metric("Price rows", f"{info['rows']:,}", border=True)
k[2].metric("Database", f"{info['size_mb']:.1f} MB", border=True)
k[3].metric("Last refresh", "—" if info["last_run"] is None or str(info["last_run"]) == "NaT"
            else f"{info['last_run']:%d %b %H:%M}", border=True)

with st.container(border=True):
    st.markdown("**Refresh**")
    st.caption("New tickers download full history; existing ones fetch only recent days.")
    with st.container(horizontal=True):
        for name in [*universe.NAMES, "all"]:
            if st.button(universe.LABELS.get(name, "All"), icon=":material/refresh:",
                         type="primary" if name == "indices" else "secondary"):
                names = universe.NAMES if name == "all" else [name]
                refresh([t for n in names for t in universe.sync(n)["ticker"]])

left, right = st.columns(2)
with left.container(border=True, height="stretch"):
    st.markdown("**Custom tickers**")
    new = st.text_input("Add tickers", placeholder="e.g. WDS.AX, NVDA, GLD",
                        help="Yahoo symbols: ASX stocks end in .AX, indices start with ^")
    if st.button("Add & download", icon=":material/add:", disabled=not new):
        refresh(universe.add_custom(new.replace(",", " ").split()).ticker.tolist())
    remove = st.multiselect("Remove custom tickers", store.instruments(["custom"]).ticker)
    if st.button("Remove", icon=":material/delete:", disabled=not remove):
        store.delete(remove)
        ui.clear_cache()
        st.rerun()

with right.container(border=True, height="stretch"):
    st.markdown("**Index constituents**")
    st.caption("ASX 200 and S&P 500 members are scraped from Wikipedia and cached as CSV.")
    target = st.selectbox("Universe", ["asx200", "sp500"], format_func=universe.LABELS.get)
    if st.button("Re-scrape constituents", icon=":material/download:"):
        members = universe.sync(target, rescrape=True)
        ui.clear_cache()
        st.success(f"{len(members)} members saved. Refresh {universe.LABELS[target]} to download their prices.")

with st.container(border=True):
    log = store.ingest_log()
    failed = log[log.status != "ok"]
    c1, c2 = st.columns([4, 1], vertical_alignment="center")
    c1.markdown("**Ingest log**")
    show_failed = c2.toggle("Failures only", value=False)
    if len(failed):
        st.warning(f"{len(failed)} tickers failed on their last refresh (often delisted or renamed).",
                   icon=":material/warning:")
    st.dataframe(failed if show_failed else log, hide_index=True, height=420)
