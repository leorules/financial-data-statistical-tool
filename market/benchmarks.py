"""Benchmarks per asset class, in two categories.

Standard: the index most widely quoted for each asset class internationally.
APRA: the indices prescribed for the Australian superannuation performance test. Most of the
prescribed series are licensed, unlisted or quarterly, so each is represented by the closest AUD
listed proxy; rows with no proxy at all carry no ticker and are reported as unavailable.
"""
import pandas as pd

BASES = ["Standard", "APRA"]
REGIONS = {"Standard": ["Global", "United States", "Australia"], "APRA": ["Australia", "International"]}
COLUMNS = ["basis", "asset_class", "region", "variant", "benchmark", "provider", "ticker", "code", "series",
           "headline", "note"]

# series: "index" = the benchmark itself; "tracker" = an ETF/ETN that tracks it; "proxy" = closest stand-in.
STANDARD = [
    ("Equities", "Global", "", "MSCI ACWI", "MSCI", "^892400-USD-STRD", "index", True, "Price index, USD"),
    ("Equities", "United States", "", "S&P 500", "S&P Dow Jones Indices", "^GSPC", "index", False, "Price index"),
    ("Equities", "Australia", "", "S&P/ASX 200", "S&P Dow Jones Indices", "^AXJO", "index", False, "Price index"),
    ("Fixed income", "Global", "", "Bloomberg Global Aggregate", "Bloomberg", "AGGG.L", "tracker", True,
     "iShares ETF tracking the index, USD unhedged"),
    ("Fixed income", "United States", "", "Bloomberg US Aggregate", "Bloomberg", "AGG", "tracker", False,
     "iShares Core US Aggregate Bond ETF"),
    ("Fixed income", "Australia", "", "Bloomberg AusBond Composite 0+ Yr", "Bloomberg", "IAF.AX", "tracker", False,
     "iShares Core Composite Bond ETF"),
    ("Cash", "United States", "", "ICE BofA 3-Month US Treasury Bill", "ICE", "BIL", "proxy", True,
     "SPDR ETF tracking Bloomberg 1-3 Month T-Bill Index"),
    ("Cash", "Australia", "", "Bloomberg AusBond Bank Bill", "Bloomberg", "BILL.AX", "proxy", False,
     "iShares Core Cash ETF"),
    ("Commodities", "Global", "", "Bloomberg Commodity Index", "Bloomberg", "DJP", "tracker", True,
     "iPath total-return ETN; S&P GSCI (^SPGSCI) is the main alternative"),
    ("Real estate", "Global", "", "FTSE EPRA Nareit Global REITs", "FTSE Russell", "REET", "tracker", True,
     "iShares Global REIT ETF"),
    ("Real estate", "United States", "", "FTSE Nareit Equity REITs", "FTSE Russell / Nareit", "USRT", "tracker", False,
     "iShares Core US REIT ETF"),
    ("Real estate", "Australia", "", "S&P/ASX 200 A-REIT", "S&P Dow Jones Indices", "^AXPJ", "index", False,
     "Price index"),
    ("Infrastructure", "Global", "", "S&P Global Infrastructure", "S&P Dow Jones Indices", "IGF", "proxy", True,
     "iShares ETF tracking the index. Listed infrastructure stands in for unlisted exposure: MSCI's Global "
     "Quarterly Infrastructure Asset Index is the benchmark super funds use, but it is licensed, quarterly "
     "and appraisal-based, so it has no public series"),
    ("Infrastructure", "Australia", "", "FTSE Developed Core Infrastructure 50/50", "FTSE Russell", "IFRA.AX",
     "proxy", False, "VanEck AUD-hedged ETF; the MSCI Australia Quarterly Private Infrastructure Fund Index is "
     "not public"),
    ("Currencies", "Global", "", "ICE US Dollar Index (DXY)", "ICE", "DX-Y.NYB", "index", True,
     "USD vs six major currencies"),
    ("Currencies", "Australia", "", "AUD/USD", "WM/Reuters", "AUDUSD=X", "proxy", False,
     "Stand-in for the RBA trade-weighted index, which Yahoo does not carry"),
    ("Crypto", "Global", "", "Bloomberg Galaxy Crypto Index", "Bloomberg / Galaxy", "BTC-USD", "proxy", True,
     "Bitcoin used as a stand-in; crypto indices are not on Yahoo"),
    ("Alternatives", "Global", "", "HFRI Fund Weighted Composite", "HFR", "QAI", "proxy", True,
     "Hedge-fund replication ETF; HFRI itself is not on Yahoo"),
]

# Prescribed for the superannuation performance test, with APRA's own index code. A ticker of None means
# no public series exists: the index is licensed, appraisal-based, or defined as a blend of other rows.
APRA = [
    ("Equities", "Australia", "", "S&P/ASX 300 Total Return", "S&P Dow Jones Indices", "VAS.AX", "ASA52", True,
     "Vanguard ETF tracking the ASX 300; dividend-adjusted, so total return"),
    ("Equities", "International", "hedged", "MSCI ACWI ex-Australia with Special Tax (100% hedged to AUD)", "MSCI",
     "VGAD.AX", "DE725341", False, "Developed-markets ETF: no AUD-hedged ACWI ex-Australia fund exists, so "
     "emerging markets are excluded"),
    ("Equities", "International", "unhedged", "MSCI ACWI ex-Australia with Special Tax (unhedged in AUD)", "MSCI",
     "VGS.AX", "DN714533", False, "Developed-markets ETF: emerging markets excluded"),
    ("Equities", "International", "developed, hedged", "MSCI World ex Australia with Special Tax (100% hedged to "
     "AUD)", "MSCI", "VGAD.AX", "DA750700", False, "Same index family as the prescribed benchmark"),
    ("Equities", "International", "developed, unhedged", "MSCI World ex Australia with Special Tax (unhedged in "
     "AUD)", "MSCI", "VGS.AX", "NA714532", False, "Same index family as the prescribed benchmark"),
    ("Equities", "International", "emerging, unhedged", "MSCI Emerging Markets with Special Tax (unhedged in AUD)",
     "MSCI", "VGE.AX", "NA714531", False, "Vanguard FTSE Emerging Markets ETF; FTSE rather than MSCI coverage"),
    ("Equities", "International", "emerging, hedged", "MSCI Emerging Markets with Special Tax (100% hedged to AUD)",
     "MSCI", None, "DA725342", False, "No AUD-hedged emerging-markets ETF is listed in Australia"),
    ("Real estate", "Australia", "listed", "S&P/ASX 300 A-REIT Total Return", "S&P Dow Jones Indices", "VAP.AX",
     "ASA6PROP", True, "Vanguard Australian Property Securities ETF tracks the ASX 300 A-REIT index"),
    ("Real estate", "International", "listed", "FTSE EPRA Nareit Developed ex Aus Rental 100% Hedged to AUD Net Tax "
     "(Super)", "FTSE Russell", "REIT.AX", "RAHRSAH", False, "VanEck AUD-hedged international property ETF"),
    ("Real estate", "Australia", "unlisted", "MSCI/Mercer Australia Core Wholesale Monthly Property Fund Index — "
     "NAV-weighted post-fee total return", "MSCI / Mercer", None, None, False,
     "Licensed, monthly, appraisal-based: no public series"),
    ("Real estate", "International", "unlisted", "MSCI Global (excl. Pan-Europe and Pan-Asia Funds) Quarterly "
     "Property Fund Index (Unfrozen), net total return, AUD fixed", "MSCI", None, None, False,
     "Licensed, quarterly, appraisal-based: no public series"),
    ("Infrastructure", "Australia", "listed", "FTSE Developed Core Infrastructure 50/50 100% Hedged to AUD Net Tax "
     "(Super)", "FTSE Russell", "IFRA.AX", "FDCICSAH", True,
     "VanEck ETF on exactly this index. APRA prescribes the same index for Australian and international"),
    ("Infrastructure", "International", "listed", "FTSE Developed Core Infrastructure 50/50 100% Hedged to AUD Net "
     "Tax (Super)", "FTSE Russell", "IFRA.AX", "FDCICSAH", False, "Same index as the Australian row"),
    ("Infrastructure", "Australia", "unlisted", "MSCI Australia Quarterly Private Infrastructure Fund Index "
     "(Unfrozen) — 50th percentile post-fee total return", "MSCI", None, None, False,
     "Licensed, quarterly, appraisal-based: no public series"),
    ("Infrastructure", "International", "unlisted", "MSCI Australia Quarterly Private Infrastructure Fund Index "
     "(Unfrozen) — 50th percentile post-fee total return", "MSCI", None, None, False,
     "APRA prescribes the Australian index for international unlisted infrastructure as well"),
    ("Fixed income", "Australia", "", "Bloomberg AusBond Composite 0+ Yr", "Bloomberg", "IAF.AX", "BACM0", True,
     "iShares Core Composite Bond ETF tracks this index"),
    ("Fixed income", "Australia", "ex-credit", "Bloomberg AusBond Govt 0+ Yr", "Bloomberg", "VGB.AX", "BAGV0", False,
     "Vanguard Australian Government Bond Index ETF"),
    ("Fixed income", "Australia", "credit", "Bloomberg AusBond Credit 0+ Yr", "Bloomberg", "CRED.AX", "BACR0", False,
     "BetaShares investment-grade corporate bond ETF; narrower than the AusBond Credit index"),
    ("Fixed income", "International", "", "Bloomberg Global Aggregate (hedged AUD)", "Bloomberg", "VBND.AX",
     "LEGATRAH", False, "Vanguard Global Aggregate Bond Index (Hedged) ETF tracks this index"),
    ("Fixed income", "International", "ex-credit", "Bloomberg Global Treasury (hedged AUD)", "Bloomberg", "VIF.AX",
     "BTSYTRAH", False, "Vanguard International Fixed Interest Index (Hedged) ETF"),
    ("Fixed income", "International", "credit", "Bloomberg Global Aggregate Corporate (hedged AUD)", "Bloomberg",
     "VCF.AX", "LGCPTRAH", False, "Vanguard International Credit Securities Index (Hedged) ETF"),
    ("Cash", "Australia", "", "Bloomberg AusBond Bank Bill", "Bloomberg", "BILL.AX", "BAUBIL", True,
     "iShares Core Cash ETF. APRA prescribes this index for international cash as well"),
    ("Cash", "International", "", "Bloomberg AusBond Bank Bill", "Bloomberg", "BILL.AX", "BAUBIL", False,
     "Same index as Australian cash"),
    ("Alternatives", "International", "", "25% international equity hedged, 25% unhedged, 50% international fixed "
     "income", "APRA", None, None, True, "A composite of the rows above, not a published index"),
    ("Alternatives", "International", "defensive", "12.5% international equity hedged, 12.5% unhedged, 75% "
     "international fixed income", "APRA", None, None, False, "A composite of the rows above"),
    ("Alternatives", "International", "growth", "37.5% international equity hedged, 37.5% unhedged, 25% "
     "international fixed income", "APRA", None, None, False,
     "A composite of the rows above. Private equity is measured against this, having no benchmark of its own"),
]


def _frame(basis: str, rows: list[tuple]) -> pd.DataFrame:
    """Standard rows carry a `series` kind; APRA rows carry an index code and are all proxies."""
    if basis == "Standard":
        df = pd.DataFrame(rows, columns=["asset_class", "region", "variant", "benchmark", "provider", "ticker",
                                         "series", "headline", "note"]).assign(code=None)
    else:
        df = pd.DataFrame(rows, columns=["asset_class", "region", "variant", "benchmark", "provider", "ticker",
                                         "code", "headline", "note"])
        df["series"] = df.ticker.map(lambda t: "proxy" if pd.notna(t) else "unavailable")
    return df.assign(basis=basis)[COLUMNS]


TABLE = pd.concat([_frame("Standard", STANDARD), _frame("APRA", APRA)], ignore_index=True)


def classes(basis: str = "Standard") -> list[str]:
    return TABLE[TABLE.basis == basis].asset_class.unique().tolist()


def select(asset_classes: list[str], regions: list[str] | None = None, basis: str = "Standard") -> pd.DataFrame:
    """Benchmarks for the chosen classes: each class's headline benchmark, or its regional ones when given."""
    rows = TABLE[(TABLE.basis == basis) & TABLE.asset_class.isin(asset_classes)]
    if not regions:
        rows = rows[rows.headline]
        return rows.assign(label=rows.asset_class).reset_index(drop=True)
    rows = rows[rows.region.isin(regions)]
    label = rows.asset_class + " · " + rows.region + rows.variant.map(lambda v: f" · {v}" if v else "")
    return rows.assign(label=label).reset_index(drop=True)


def describe(row) -> str:
    """'MSCI ACWI', or 'Bloomberg Global Aggregate via AGGG.L' when a tracker or proxy stands in for the index."""
    return row.benchmark if row.series == "index" else f"{row.benchmark} via {row.ticker}"
