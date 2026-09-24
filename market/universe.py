import io

import pandas as pd
import requests
import yfinance as yf

from market import store
from market.config import UNIVERSE_DIR

NAMES = ["indices", "etfs", "commodities", "rates", "fx", "crypto", "asx200", "asx_listed", "sp500", "custom"]
LABELS = {"indices": "Indices", "etfs": "ETFs", "commodities": "Commodities", "rates": "Rates", "fx": "FX",
          "crypto": "Crypto", "asx200": "ASX 200", "asx_listed": "ASX (all listed)", "sp500": "S&P 500",
          "custom": "Custom"}
HEADERS = {"User-Agent": "Mozilla/5.0 market-dashboard"}
ASSET_CLASSES = ["Equities", "Fixed income", "Cash", "Commodities", "Real estate", "Infrastructure", "Currencies",
                 "Crypto", "Alternatives", "Rates", "Volatility", "Other"]
COLUMNS = ["ticker", "name", "type", "exchange", "currency", "sector", "universe", "asset_class"]
# Instrument types grouped into the kinds people pick between, in the order they are offered.
KINDS = {"Indices": ("index",), "Sector indices": ("sector",), "ETFs": ("etf", "sector etf", "etn"),
         "Individual stocks": ("equity",),
         "Futures": ("energy", "precious metal", "industrial metal", "agriculture", "livestock"),
         "FX pairs": ("fx",), "Yields": ("yield",), "Crypto": ("crypto",)}
REGIONS = {"US": "United States", "AU": "Australia", "GB": "United Kingdom", "CA": "Canada", "JP": "Japan"}
QUOTE_TYPES = {"EQUITY": "Equities", "INDEX": "Equities", "FUTURE": "Commodities", "CURRENCY": "Currencies",
               "CRYPTOCURRENCY": "Crypto"}


def _table(universe: str, asset_class: str, rows: list[tuple]) -> pd.DataFrame:
    """Rows are (ticker, name, type, region, currency[, asset_class]); a 6th value overrides the default class."""
    rows = [(*r, asset_class)[:6] for r in rows]
    df = pd.DataFrame(rows, columns=["ticker", "name", "type", "exchange", "currency", "asset_class"])
    return df.assign(sector=None, universe=universe)[COLUMNS]


# Curated order: instruments appear in pickers in this order. `exchange` holds the region.
INDICES = _table("indices", "Equities", [
    ("^AXJO", "S&P/ASX 200", "index", "Australia", "AUD"),
    ("^AORD", "All Ordinaries", "index", "Australia", "AUD"),
    ("^ATLI", "S&P/ASX 20", "index", "Australia", "AUD"),
    ("^AFLI", "S&P/ASX 50", "index", "Australia", "AUD"),
    ("^ATOI", "S&P/ASX 100", "index", "Australia", "AUD"),
    ("^AXKO", "S&P/ASX 300", "index", "Australia", "AUD"),
    ("^AXSO", "S&P/ASX Small Ordinaries", "index", "Australia", "AUD"),
    ("^NZ50", "NZX 50", "index", "New Zealand", "NZD"),
    ("^GSPC", "S&P 500", "index", "United States", "USD"),
    ("^SP500TR", "S&P 500 Total Return", "index", "United States", "USD"),
    ("^DJI", "Dow Jones Industrial Average", "index", "United States", "USD"),
    ("^IXIC", "Nasdaq Composite", "index", "United States", "USD"),
    ("^NDX", "Nasdaq 100", "index", "United States", "USD"),
    ("^RUT", "Russell 2000", "index", "United States", "USD"),
    ("^NYA", "NYSE Composite", "index", "United States", "USD"),
    ("^VIX", "CBOE Volatility Index", "index", "United States", "USD", "Volatility"),
    ("^GSPTSE", "S&P/TSX Composite", "index", "Canada", "CAD"),
    ("^BVSP", "Ibovespa", "index", "Brazil", "BRL"),
    ("^MXX", "S&P/BMV IPC", "index", "Mexico", "MXN"),
    ("^892400-USD-STRD", "MSCI ACWI", "index", "Global", "USD"),
    ("^990100-USD-STRD", "MSCI World", "index", "Global", "USD"),
    ("^STOXX50E", "Euro Stoxx 50", "index", "Europe", "EUR"),
    ("^STOXX", "STOXX Europe 600", "index", "Europe", "EUR"),
    ("^FTSE", "FTSE 100", "index", "United Kingdom", "GBP"),
    ("^GDAXI", "DAX", "index", "Germany", "EUR"),
    ("^FCHI", "CAC 40", "index", "France", "EUR"),
    ("^IBEX", "IBEX 35", "index", "Spain", "EUR"),
    ("FTSEMIB.MI", "FTSE MIB", "index", "Italy", "EUR"),
    ("^AEX", "AEX", "index", "Netherlands", "EUR"),
    ("^SSMI", "Swiss Market Index", "index", "Switzerland", "CHF"),
    ("^N225", "Nikkei 225", "index", "Japan", "JPY"),
    ("^HSI", "Hang Seng", "index", "Hong Kong", "HKD"),
    ("000001.SS", "SSE Composite", "index", "China", "CNY"),
    ("^KS11", "KOSPI", "index", "South Korea", "KRW"),
    ("^TWII", "Taiwan Weighted", "index", "Taiwan", "TWD"),
    ("^BSESN", "BSE Sensex", "index", "India", "INR"),
    ("^NSEI", "Nifty 50", "index", "India", "INR"),
    ("^STI", "Straits Times Index", "index", "Singapore", "SGD"),
    ("^JKSE", "Jakarta Composite", "index", "Indonesia", "IDR"),
    ("^KLSE", "FTSE Bursa Malaysia KLCI", "index", "Malaysia", "MYR"),
    ("^AXEJ", "ASX 200 Energy", "sector", "Australia", "AUD"),
    ("^AXMJ", "ASX 200 Materials", "sector", "Australia", "AUD"),
    ("^AXNJ", "ASX 200 Industrials", "sector", "Australia", "AUD"),
    ("^AXDJ", "ASX 200 Consumer Discretionary", "sector", "Australia", "AUD"),
    ("^AXSJ", "ASX 200 Consumer Staples", "sector", "Australia", "AUD"),
    ("^AXHJ", "ASX 200 Health Care", "sector", "Australia", "AUD"),
    ("^AXFJ", "ASX 200 Financials", "sector", "Australia", "AUD"),
    ("^AXIJ", "ASX 200 Information Technology", "sector", "Australia", "AUD"),
    ("^AXTJ", "ASX 200 Communication Services", "sector", "Australia", "AUD"),
    ("^AXUJ", "ASX 200 Utilities", "sector", "Australia", "AUD"),
    ("^AXPJ", "ASX 200 A-REIT", "sector", "Australia", "AUD", "Real estate"),
])

ETFS = _table("etfs", "Equities", [
    ("SPY", "SPDR S&P 500 ETF", "etf", "United States", "USD"),
    ("QQQ", "Invesco QQQ Trust", "etf", "United States", "USD"),
    ("DIA", "SPDR Dow Jones Industrial Average ETF", "etf", "United States", "USD"),
    ("IWM", "iShares Russell 2000 ETF", "etf", "United States", "USD"),
    ("VTI", "Vanguard Total Stock Market ETF", "etf", "United States", "USD"),
    ("XLK", "Technology Select Sector SPDR", "sector etf", "United States", "USD"),
    ("XLF", "Financial Select Sector SPDR", "sector etf", "United States", "USD"),
    ("XLE", "Energy Select Sector SPDR", "sector etf", "United States", "USD"),
    ("XLV", "Health Care Select Sector SPDR", "sector etf", "United States", "USD"),
    ("XLI", "Industrial Select Sector SPDR", "sector etf", "United States", "USD"),
    ("XLY", "Consumer Discretionary Select Sector SPDR", "sector etf", "United States", "USD"),
    ("XLP", "Consumer Staples Select Sector SPDR", "sector etf", "United States", "USD"),
    ("XLU", "Utilities Select Sector SPDR", "sector etf", "United States", "USD"),
    ("XLB", "Materials Select Sector SPDR", "sector etf", "United States", "USD"),
    ("XLC", "Communication Services Select Sector SPDR", "sector etf", "United States", "USD"),
    ("XLRE", "Real Estate Select Sector SPDR", "sector etf", "United States", "USD", "Real estate"),
    ("VNQ", "Vanguard Real Estate ETF", "etf", "United States", "USD", "Real estate"),
    ("EFA", "iShares MSCI EAFE ETF", "etf", "Global", "USD"),
    ("EEM", "iShares MSCI Emerging Markets ETF", "etf", "Global", "USD"),
    ("ACWI", "iShares MSCI ACWI ETF", "etf", "Global", "USD"),
    ("STW.AX", "SPDR S&P/ASX 200 Fund", "etf", "Australia", "AUD"),
    ("IOZ.AX", "iShares Core S&P/ASX 200 ETF", "etf", "Australia", "AUD"),
    ("VAS.AX", "Vanguard Australian Shares ETF", "etf", "Australia", "AUD"),
    ("A200.AX", "Betashares Australia 200 ETF", "etf", "Australia", "AUD"),
    ("VGS.AX", "Vanguard MSCI Index International Shares ETF", "etf", "Global", "AUD"),
    ("VAP.AX", "Vanguard Australian Property Securities ETF", "etf", "Australia", "AUD", "Real estate"),
    ("SHY", "iShares 1-3 Year Treasury Bond ETF", "etf", "United States", "USD", "Fixed income"),
    ("IEF", "iShares 7-10 Year Treasury Bond ETF", "etf", "United States", "USD", "Fixed income"),
    ("TLT", "iShares 20+ Year Treasury Bond ETF", "etf", "United States", "USD", "Fixed income"),
    ("TIP", "iShares TIPS Bond ETF", "etf", "United States", "USD", "Fixed income"),
    ("AGG", "iShares Core US Aggregate Bond ETF", "etf", "United States", "USD", "Fixed income"),
    ("BND", "Vanguard Total Bond Market ETF", "etf", "United States", "USD", "Fixed income"),
    ("LQD", "iShares iBoxx Investment Grade Corporate Bond ETF", "etf", "United States", "USD", "Fixed income"),
    ("HYG", "iShares iBoxx High Yield Corporate Bond ETF", "etf", "United States", "USD", "Fixed income"),
    ("EMB", "iShares JP Morgan USD Emerging Markets Bond ETF", "etf", "Global", "USD", "Fixed income"),
    ("AGGG.L", "iShares Core Global Aggregate Bond UCITS ETF", "etf", "Global", "USD", "Fixed income"),
    ("VAF.AX", "Vanguard Australian Fixed Interest ETF", "etf", "Australia", "AUD", "Fixed income"),
    ("IAF.AX", "iShares Core Composite Bond ETF", "etf", "Australia", "AUD", "Fixed income"),
    # AUD proxies for the indices APRA prescribes for the superannuation performance test.
    ("VGAD.AX", "Vanguard MSCI Intl Shares (Hedged) ETF", "etf", "Global", "AUD"),
    ("VGE.AX", "Vanguard FTSE Emerging Markets Shares ETF", "etf", "Global", "AUD"),
    ("REIT.AX", "VanEck FTSE International Property (Hedged) ETF", "etf", "Global", "AUD", "Real estate"),
    ("VGB.AX", "Vanguard Australian Government Bond Index ETF", "etf", "Australia", "AUD", "Fixed income"),
    ("CRED.AX", "BetaShares Australian Investment Grade Corporate Bond ETF", "etf", "Australia", "AUD", "Fixed income"),
    ("VBND.AX", "Vanguard Global Aggregate Bond Index (Hedged) ETF", "etf", "Global", "AUD", "Fixed income"),
    ("VIF.AX", "Vanguard International Fixed Interest Index (Hedged) ETF", "etf", "Global", "AUD", "Fixed income"),
    ("VCF.AX", "Vanguard International Credit Securities Index (Hedged) ETF", "etf", "Global", "AUD", "Fixed income"),
    ("AAA.AX", "BetaShares Australian High Interest Cash ETF", "etf", "Australia", "AUD", "Cash"),
    ("BIL", "SPDR Bloomberg 1-3 Month T-Bill ETF", "etf", "United States", "USD", "Cash"),
    ("BILL.AX", "iShares Core Cash ETF", "etf", "Australia", "AUD", "Cash"),
    ("DJP", "iPath Bloomberg Commodity Index Total Return ETN", "etn", "Global", "USD", "Commodities"),
    ("REET", "iShares Global REIT ETF", "etf", "Global", "USD", "Real estate"),
    ("USRT", "iShares Core U.S. REIT ETF", "etf", "United States", "USD", "Real estate"),
    ("IGF", "iShares Global Infrastructure ETF", "etf", "Global", "USD", "Infrastructure"),
    ("GII", "SPDR S&P Global Infrastructure ETF", "etf", "Global", "USD", "Infrastructure"),
    ("NFRA", "FlexShares STOXX Global Broad Infrastructure ETF", "etf", "Global", "USD", "Infrastructure"),
    ("TOLZ", "ProShares DJ Brookfield Global Infrastructure ETF", "etf", "Global", "USD", "Infrastructure"),
    ("IFRA.AX", "VanEck FTSE Global Infrastructure (Hedged) ETF", "etf", "Australia", "AUD", "Infrastructure"),
    ("VBLD.AX", "Vanguard Global Infrastructure Index ETF", "etf", "Australia", "AUD", "Infrastructure"),
    ("QAI", "NYLI Hedge Multi-Strategy Tracker ETF", "etf", "Global", "USD", "Alternatives"),
    ("GLD", "SPDR Gold Shares", "etf", "United States", "USD", "Commodities"),
    ("SLV", "iShares Silver Trust", "etf", "United States", "USD", "Commodities"),
    ("USO", "United States Oil Fund", "etf", "United States", "USD", "Commodities"),
    ("DBC", "Invesco DB Commodity Index Tracking Fund", "etf", "United States", "USD", "Commodities"),
    ("GOLD.AX", "Global X Physical Gold", "etf", "Australia", "AUD", "Commodities"),
    ("QAU.AX", "Betashares Gold Bullion ETF (AUD hedged)", "etf", "Australia", "AUD", "Commodities"),
])

COMMODITIES = _table("commodities", "Commodities", [
    ("^SPGSCI", "S&P GSCI Commodity Index", "index", "Global", "USD"),
    ("CL=F", "WTI Crude Oil", "energy", "United States", "USD"),
    ("BZ=F", "Brent Crude Oil", "energy", "Global", "USD"),
    ("NG=F", "Natural Gas (Henry Hub)", "energy", "United States", "USD"),
    ("RB=F", "RBOB Gasoline", "energy", "United States", "USD"),
    ("HO=F", "Heating Oil", "energy", "United States", "USD"),
    ("GC=F", "Gold", "precious metal", "United States", "USD"),
    ("SI=F", "Silver", "precious metal", "United States", "USD"),
    ("PL=F", "Platinum", "precious metal", "United States", "USD"),
    ("PA=F", "Palladium", "precious metal", "United States", "USD"),
    ("HG=F", "Copper", "industrial metal", "United States", "USD"),
    ("ALI=F", "Aluminium", "industrial metal", "United States", "USD"),
    ("TIO=F", "Iron Ore 62% Fe", "industrial metal", "Global", "USD"),
    ("ZC=F", "Corn", "agriculture", "United States", "USD"),
    ("ZW=F", "Wheat", "agriculture", "United States", "USD"),
    ("ZS=F", "Soybeans", "agriculture", "United States", "USD"),
    ("KC=F", "Coffee", "agriculture", "United States", "USD"),
    ("SB=F", "Sugar", "agriculture", "United States", "USD"),
    ("CC=F", "Cocoa", "agriculture", "United States", "USD"),
    ("CT=F", "Cotton", "agriculture", "United States", "USD"),
    ("LE=F", "Live Cattle", "livestock", "United States", "USD"),
    ("HE=F", "Lean Hogs", "livestock", "United States", "USD"),
])

RATES = _table("rates", "Rates", [
    ("^IRX", "US 13-Week T-Bill Yield", "yield", "United States", "USD"),
    ("^FVX", "US 5-Year Treasury Yield", "yield", "United States", "USD"),
    ("^TNX", "US 10-Year Treasury Yield", "yield", "United States", "USD"),
    ("^TYX", "US 30-Year Treasury Yield", "yield", "United States", "USD"),
])

FX = _table("fx", "Currencies", [
    ("AUDUSD=X", "AUD/USD", "fx", "Australia", "USD"),
    ("DX-Y.NYB", "US Dollar Index", "index", "United States", "USD"),
    ("EURUSD=X", "EUR/USD", "fx", "Europe", "USD"),
    ("GBPUSD=X", "GBP/USD", "fx", "United Kingdom", "USD"),
    ("USDJPY=X", "USD/JPY", "fx", "Japan", "JPY"),
    ("USDCNY=X", "USD/CNY", "fx", "China", "CNY"),
    ("USDCAD=X", "USD/CAD", "fx", "Canada", "CAD"),
    ("USDCHF=X", "USD/CHF", "fx", "Switzerland", "CHF"),
    ("NZDUSD=X", "NZD/USD", "fx", "New Zealand", "USD"),
    ("AUDJPY=X", "AUD/JPY", "fx", "Australia", "JPY"),
    ("AUDNZD=X", "AUD/NZD", "fx", "Australia", "NZD"),
    ("EURAUD=X", "EUR/AUD", "fx", "Australia", "AUD"),
])

CRYPTO = _table("crypto", "Crypto", [
    ("BTC-USD", "Bitcoin", "crypto", "Global", "USD"),
    ("ETH-USD", "Ethereum", "crypto", "Global", "USD"),
    ("SOL-USD", "Solana", "crypto", "Global", "USD"),
    ("XRP-USD", "XRP", "crypto", "Global", "USD"),
    ("BNB-USD", "BNB", "crypto", "Global", "USD"),
    ("ADA-USD", "Cardano", "crypto", "Global", "USD"),
    ("DOGE-USD", "Dogecoin", "crypto", "Global", "USD"),
])

BUILT_IN = {"indices": INDICES, "etfs": ETFS, "commodities": COMMODITIES, "rates": RATES, "fx": FX, "crypto": CRYPTO}
PICKER_ORDER = ["indices", "commodities", "rates", "fx", "crypto", "etfs", "custom", "asx200", "sp500",
                "asx_listed"]


def kinds(types) -> list[str]:
    """Kinds present among these instrument types, in KINDS order."""
    present = set(types.dropna())
    return [kind for kind, members in KINDS.items() if present & set(members)]


def sources() -> pd.DataFrame:
    """Where each list of instruments comes from, read from the URLs the code actually fetches."""
    curated = "Curated in market/universe.py"
    rows = [(name, curated, "") for name in ("indices", "etfs", "commodities", "rates", "fx", "crypto")]
    rows += [("asx200", "Wikipedia", WIKI["asx200"][0]), ("sp500", "Wikipedia", WIKI["sp500"][0]),
             ("asx_listed", "ASX company directory", ASX_FALLBACK),
             ("custom", "Tickers you added by hand", "")]
    return pd.DataFrame(rows, columns=["universe", "source", "url"])


def sort(inst: pd.DataFrame) -> pd.DataFrame:
    """Order instruments for pickers: built-in lists in curated order, then custom, then index members."""
    group = {name: i for i, name in enumerate(PICKER_ORDER)}
    curated = {t: i for i, t in enumerate(pd.concat(BUILT_IN.values()).ticker)}
    keys = inst.assign(_group=inst.universe.map(group).fillna(len(group)), _rank=inst.ticker.map(curated))
    return keys.sort_values(["_group", "_rank", "ticker"]).drop(columns=["_group", "_rank"]).reset_index(drop=True)


WIKI = {
    "asx200": ("https://en.wikipedia.org/wiki/S%26P/ASX_200", "Code",
               lambda t: pd.DataFrame({"ticker": t["Code"] + ".AX", "name": t["Company"], "sector": t["Sector"]})),
    "sp500": ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "Symbol",
              lambda t: pd.DataFrame({"ticker": t["Symbol"].str.replace(".", "-"), "name": t["Security"],
                                      "sector": t["GICS Sector"]})),
}
MARKETS = {"asx200": ("Australia", "AUD"), "asx_listed": ("Australia", "AUD"), "sp500": ("United States", "USD")}
ASX_DIRECTORY = ("https://asx.api.markitdigital.com/asx-research/1.0/companies/directory/file"
                 "?access_token=83ff96335c2d45a094df02a206a39ff4")
ASX_FALLBACK = "https://www.asx.com.au/asx/research/ASXListedCompanies.csv"
# The ASX directory reports GICS industry groups; map them to the 11 GICS sectors the index lists use.
GICS_SECTORS = {
    "Energy": "Energy", "Materials": "Materials",
    "Capital Goods": "Industrials", "Commercial & Professional Services": "Industrials",
    "Transportation": "Industrials",
    "Automobiles & Components": "Consumer Discretionary", "Consumer Durables & Apparel": "Consumer Discretionary",
    "Consumer Services": "Consumer Discretionary", "Retailing": "Consumer Discretionary",
    "Consumer Discretionary Distribution & Retail": "Consumer Discretionary",
    "Food, Beverage & Tobacco": "Consumer Staples", "Household & Personal Products": "Consumer Staples",
    "Food & Staples Retailing": "Consumer Staples", "Consumer Staples Distribution & Retail": "Consumer Staples",
    "Health Care Equipment & Services": "Health Care",
    "Pharmaceuticals, Biotechnology & Life Sciences": "Health Care",
    "Banks": "Financials", "Financial Services": "Financials", "Diversified Financials": "Financials",
    "Insurance": "Financials",
    "Software & Services": "Information Technology", "Technology Hardware & Equipment": "Information Technology",
    "Semiconductors & Semiconductor Equipment": "Information Technology",
    "Telecommunication Services": "Communication Services", "Media & Entertainment": "Communication Services",
    "Utilities": "Utilities",
    "Equity Real Estate Investment Trusts (REITs)": "Real Estate",
    "Real Estate Management & Development": "Real Estate", "Real Estate": "Real Estate",
    # Spelling variants seen in the index lists
    "Healthcare": "Health Care", "Health care": "Health Care", "Consumer discretionary": "Consumer Discretionary",
    "Information technology": "Information Technology", "Communication services": "Communication Services",
}


def _members(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Index or exchange members as instrument rows, with sectors normalised to the 11 GICS sectors."""
    region, currency = MARKETS[name]
    sector = df.sector.map(lambda s: GICS_SECTORS.get(s, s if s in set(GICS_SECTORS.values()) else None))
    return df.assign(sector=sector, type="equity", exchange=region, currency=currency, universe=name,
                     asset_class="Equities")[COLUMNS]


def asx_listed() -> pd.DataFrame:
    """Every company currently listed on the ASX, from the exchange's own directory."""
    try:
        raw = pd.read_csv(io.StringIO(requests.get(ASX_DIRECTORY, headers=HEADERS, timeout=60).text))
        code, company, industry = "ASX code", "Company name", "GICs industry group"
    except Exception:
        raw = pd.read_csv(io.StringIO(requests.get(ASX_FALLBACK, headers=HEADERS, timeout=60).text), skiprows=1)
        code, company, industry = "ASX code", "Company name", "GICS industry group"
    return pd.DataFrame({"ticker": raw[code].str.strip() + ".AX", "name": raw[company].str.title(),
                         "sector": raw[industry]}).dropna(subset=["ticker"])


def scrape(name: str) -> pd.DataFrame:
    if name == "asx_listed":
        df = _members(name, asx_listed())
    else:
        url, key, parse = WIKI[name]
        html = requests.get(url, headers=HEADERS, timeout=30).text
        table = next(t for t in pd.read_html(io.StringIO(html)) if key in t.columns)
        df = _members(name, parse(table))
    UNIVERSE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(UNIVERSE_DIR / f"{name}.csv", index=False)
    return df


def load(name: str, rescrape: bool = False) -> pd.DataFrame:
    if name in BUILT_IN:
        return BUILT_IN[name]
    if name == "custom":
        return store.instruments(["custom"])
    cached = UNIVERSE_DIR / f"{name}.csv"
    if rescrape or not cached.exists():
        try:
            return scrape(name)
        except Exception:
            if not cached.exists():
                raise
    return _members(name, pd.read_csv(cached))


def sync(name: str, rescrape: bool = False) -> pd.DataFrame:
    """Refresh a list's metadata. Tickers already held in another list (index members, ETFs) keep their own entry."""
    df = load(name, rescrape)
    if name == "asx_listed":
        others = set(store.instruments().ticker) - set(store.instruments(["asx_listed"]).ticker)
        df = df[~df.ticker.isin(others)]
    if name != "custom":
        store.upsert("instruments", df)
    return df


def custom_row(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).get_info()
    except Exception:
        info = {}
    asx = ticker.endswith(".AX")
    quote_type = str(info.get("quoteType", "EQUITY")).upper()
    return {"ticker": ticker, "name": info.get("shortName", ticker), "type": quote_type.lower(),
            "exchange": "Australia" if asx else REGIONS.get(info.get("region"), "Other"),
            "currency": info.get("currency", "AUD" if asx else "USD"), "sector": info.get("sector"),
            "universe": "custom", "asset_class": QUOTE_TYPES.get(quote_type, "Other")}


def add_custom(tickers: list[str]) -> pd.DataFrame:
    rows = [custom_row(t.strip().upper()) for t in tickers if t.strip()]
    df = pd.DataFrame(rows, columns=COLUMNS)
    store.upsert("instruments", df)
    return df
