import time

import pandas as pd
import streamlit as st

from market import inflation, ingest, store, universe, ui

ui.header("Data Manager", "Download and refresh prices, manage custom tickers, and check the ingest log.")


def refresh(tickers: list[str]) -> None:
    bar = st.progress(0.0, text=f"Refreshing {len(tickers)} tickers…")
    started = time.perf_counter()

    def report(done: float, msg: str) -> None:
        elapsed = time.perf_counter() - started
        left = elapsed / done - elapsed if done else 0
        bar.progress(done, text=f"{msg} · {elapsed / 60:.1f} min elapsed, about {left / 60:.1f} min left")

    log = ingest.refresh(tickers, on_progress=report)
    ui.clear_cache()
    failed = log[log.status != "ok"]
    st.success(f"{len(log) - len(failed)}/{len(log)} tickers refreshed.", icon=":material/check_circle:")
    if len(failed):
        st.warning("Failed: " + ", ".join(failed.ticker), icon=":material/warning:")


info = ui.db("stats")
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

with st.container(border=True):
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    c1.markdown("**Inflation**")
    c1.caption("Consumer price indices for the major financial centres, from the OECD. Australia is quarterly "
               "because the ABS publishes it that way; the rest are monthly.")
    if c2.button("Download inflation", icon=":material/download:"):
        bar = st.progress(0.0, text="Fetching…")
        log = inflation.refresh(on_progress=lambda p, msg: bar.progress(p, text=msg))
        ui.clear_cache()
        failed = log[log.error.notna()]
        st.success(f"{len(log) - len(failed)}/{len(log)} regions updated.", icon=":material/check_circle:")
        if len(failed):
            st.warning("Failed: " + ", ".join(f"{r.region} ({r.error})" for _, r in failed.iterrows()))
    cover_cpi = ui.db("inflation_coverage")
    st.dataframe(cover_cpi, hide_index=True, column_config={
        "region": None, "hub": st.column_config.TextColumn("Hub", width="medium"), "frequency": "Published",
        "observations": st.column_config.NumberColumn("Readings", format="%d"),
        "first": st.column_config.DateColumn("From", format="MMM YYYY"),
        "latest": st.column_config.DateColumn("To", format="MMM YYYY")})
    stale = cover_cpi[cover_cpi.latest.notna() & (cover_cpi.latest < pd.Timestamp.today() - pd.Timedelta(days=270))]
    if len(stale):
        st.caption("The OECD series for " + ", ".join(stale.hub.str.split(" · ").str[1])
                   + " has not been updated recently; the readings above are the latest published.")

left, right = st.columns(2)
with left.container(border=True, height="stretch"):
    st.markdown("**Custom tickers**")
    new = st.text_input("Add tickers", placeholder="e.g. WDS.AX, NVDA, GLD",
                        help="Yahoo symbols: ASX stocks end in .AX, indices start with ^")
    if st.button("Add & download", icon=":material/add:", disabled=not new):
        refresh(universe.add_custom(new.replace(",", " ").split()).ticker.tolist())
    remove = st.multiselect("Remove custom tickers", ui.db("custom"))
    if st.button("Remove", icon=":material/delete:", disabled=not remove):
        store.delete(remove)
        ui.clear_cache()
        st.rerun()

with right.container(border=True, height="stretch"):
    st.markdown("**Index constituents**")
    st.caption("ASX 200 and S&P 500 members come from Wikipedia; the full ASX list comes from the exchange "
               "directory. All are cached as CSV.")
    target = st.selectbox("Universe", ["asx200", "sp500", "asx_listed"], format_func=universe.LABELS.get)
    if st.button("Re-scrape constituents", icon=":material/download:"):
        members = universe.sync(target, rescrape=True)
        ui.clear_cache()
        st.success(f"{len(members)} members saved. Refresh {universe.LABELS[target]} to download their prices.")

cover = ui.db("coverage")
issues = ui.db("quality")
left, right = st.columns([3, 4])
with left.container(border=True, height="stretch"):
    st.markdown("**Coverage by list**", help="Instruments held versus those with prices downloaded.")
    st.dataframe(cover.merge(universe.sources()[["universe", "source"]], on="universe", how="left"),
                 hide_index=True, column_config={
        "universe": st.column_config.TextColumn("List"), "instruments": "Held", "with_data": "With prices",
        "source": st.column_config.TextColumn("Members from", width="medium"),
        "last_date": st.column_config.DateColumn("Latest bar", format="DD MMM YYYY")})
    behind = cover[cover.with_data < cover.instruments]
    if len(behind):
        st.caption("Not fully downloaded: "
                   + ", ".join(f"**{universe.LABELS.get(r.universe, r.universe)}** "
                               f"{r.instruments - r.with_data} missing" for _, r in behind.iterrows())
                   + ". Refresh the list above to fetch them.")

with right.container(border=True, height="stretch"):
    st.markdown("**Data quality**", help="Problems that would quietly distort analysis if left unnoticed.")
    st.dataframe(issues[issues.tickers > 0], hide_index=True, column_config={
        "issue": st.column_config.TextColumn("Issue", width="medium"),
        "tickers": st.column_config.NumberColumn("Tickers"),
        "examples": st.column_config.TextColumn("Examples", width="large")})
    st.caption("Zero prices come from Yahoo for suspended microcaps; they are treated as missing data, so those "
               "instruments are measured only over the days they actually traded.")

with st.container(border=True):
    st.markdown("**Where the data comes from**",
                help="Every dataset, its origin and how it reaches the database.")
    st.dataframe(pd.DataFrame([
        {"Dataset": "Prices and volumes", "Source": "Yahoo Finance (yfinance)",
         "Detail": "Daily OHLCV. Adjusted close carries dividends and splits; close does not.",
         "Link": "https://finance.yahoo.com"},
        {"Dataset": "Index members (ASX 200, S&P 500)", "Source": "Wikipedia",
         "Detail": "Current constituents only, scraped and cached as CSV — historical membership is not available.",
         "Link": universe.WIKI["asx200"][0]},
        {"Dataset": "Every ASX listing", "Source": "ASX company directory",
         "Detail": "Codes, names and GICS industry groups for about 1,800 companies.",
         "Link": universe.ASX_FALLBACK},
        {"Dataset": "Curated lists", "Source": "market/universe.py",
         "Detail": "Indices, ETFs, commodities, rates, FX and crypto are hand-picked ticker tables.", "Link": ""},
        {"Dataset": "Asset-class benchmarks", "Source": "market/benchmarks.py",
         "Detail": "Standard and APRA categories, each priced from a Yahoo series or a blend of them.", "Link": ""},
        {"Dataset": "Cash rates", "Source": "Yahoo Finance",
         "Detail": "^IRX for US dollars and BILL.AX for Australian dollars, used by the live cash rate adjustment.",
         "Link": ""},
        {"Dataset": "Inflation", "Source": "OECD SDMX",
         "Detail": "CPI for eight financial hubs. FRED is the usual source and is unreachable from here.",
         "Link": "https://sdmx.oecd.org"},
    ]), hide_index=True, column_config={
        "Dataset": st.column_config.TextColumn(width="medium"),
        "Detail": st.column_config.TextColumn(width="large"),
        "Link": st.column_config.LinkColumn("Link", display_text="open")})
    st.caption("Everything is free and public. Nothing here is survivorship-adjusted: index lists are today's "
               "members applied to history, which flatters results before the present composition existed.")

with st.container(border=True):
    log = ui.db("ingest_log")
    failed = log[log.status != "ok"]
    c1, c2 = st.columns([4, 1], vertical_alignment="center")
    c1.markdown("**Ingest log**")
    show_failed = c2.toggle("Failures only", value=False)
    if len(failed):
        st.warning(f"{len(failed)} tickers failed on their last refresh (often delisted or renamed).",
                   icon=":material/warning:")
    st.dataframe(failed if show_failed else log, hide_index=True, height=420)
