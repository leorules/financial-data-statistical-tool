"""Standard benchmark per asset class and region, and the Yahoo series used to represent each one."""
import pandas as pd

REGIONS = ["Global", "United States", "Australia"]
COLUMNS = ["asset_class", "region", "benchmark", "provider", "ticker", "series", "headline", "note"]

# series: "index" = the benchmark itself; "tracker" = an ETF/ETN that tracks it; "proxy" = closest available stand-in.
TABLE = pd.DataFrame([
    ("Equities", "Global", "MSCI ACWI", "MSCI", "^892400-USD-STRD", "index", True, "Price index, USD"),
    ("Equities", "United States", "S&P 500", "S&P Dow Jones Indices", "^GSPC", "index", False, "Price index"),
    ("Equities", "Australia", "S&P/ASX 200", "S&P Dow Jones Indices", "^AXJO", "index", False, "Price index"),
    ("Fixed income", "Global", "Bloomberg Global Aggregate", "Bloomberg", "AGGG.L", "tracker", True,
     "iShares ETF tracking the index, USD unhedged"),
    ("Fixed income", "United States", "Bloomberg US Aggregate", "Bloomberg", "AGG", "tracker", False,
     "iShares Core US Aggregate Bond ETF"),
    ("Fixed income", "Australia", "Bloomberg AusBond Composite 0+ Yr", "Bloomberg", "IAF.AX", "tracker", False,
     "iShares Core Composite Bond ETF"),
    ("Cash", "United States", "ICE BofA 3-Month US Treasury Bill", "ICE", "BIL", "proxy", True,
     "SPDR ETF tracking Bloomberg 1-3 Month T-Bill Index"),
    ("Cash", "Australia", "Bloomberg AusBond Bank Bill", "Bloomberg", "BILL.AX", "proxy", False,
     "iShares Core Cash ETF"),
    ("Commodities", "Global", "Bloomberg Commodity Index", "Bloomberg", "DJP", "tracker", True,
     "iPath total-return ETN; S&P GSCI (^SPGSCI) is the main alternative"),
    ("Real estate", "Global", "FTSE EPRA Nareit Global REITs", "FTSE Russell", "REET", "tracker", True,
     "iShares Global REIT ETF"),
    ("Real estate", "United States", "FTSE Nareit Equity REITs", "FTSE Russell / Nareit", "USRT", "tracker", False,
     "iShares Core US REIT ETF"),
    ("Real estate", "Australia", "S&P/ASX 200 A-REIT", "S&P Dow Jones Indices", "^AXPJ", "index", False, "Price index"),
    ("Currencies", "Global", "ICE US Dollar Index (DXY)", "ICE", "DX-Y.NYB", "index", True, "USD vs six major currencies"),
    ("Currencies", "Australia", "AUD/USD", "WM/Reuters", "AUDUSD=X", "proxy", False,
     "Stand-in for the RBA trade-weighted index, which Yahoo does not carry"),
    ("Crypto", "Global", "Bloomberg Galaxy Crypto Index", "Bloomberg / Galaxy", "BTC-USD", "proxy", True,
     "Bitcoin used as a stand-in; crypto indices are not on Yahoo"),
    ("Alternatives", "Global", "HFRI Fund Weighted Composite", "HFR", "QAI", "proxy", True,
     "Hedge-fund replication ETF; HFRI itself is not on Yahoo"),
], columns=COLUMNS)


def select(asset_classes: list[str], regions: list[str] | None = None) -> pd.DataFrame:
    """Benchmarks for the chosen classes: each class's headline benchmark, or its regional ones when regions are given."""
    rows = TABLE[TABLE.asset_class.isin(asset_classes)]
    if regions:
        rows = rows[rows.region.isin(regions)]
        label = rows.asset_class + " · " + rows.region
    else:
        rows = rows[rows.headline]
        label = rows.asset_class
    return rows.assign(label=label).reset_index(drop=True)


def describe(row) -> str:
    """'MSCI ACWI', or 'Bloomberg Global Aggregate via AGGG.L' when a tracker or proxy stands in for the index."""
    return row.benchmark if row.series == "index" else f"{row.benchmark} via {row.ticker}"
