"""Plain-English description of an instrument: curated for built-in lists, Yahoo company profiles otherwise."""
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from market import benchmarks, store

CURATED = {
    # Equity indices
    "^AXJO": "Australia's benchmark share index: the 200 largest ASX-listed companies, weighted by float-adjusted "
             "market capitalisation and maintained by S&P Dow Jones Indices. It covers the large majority of the "
             "Australian market's value and is dominated by banks and miners.",
    "^AORD": "The All Ordinaries: the 500 largest companies listed on the ASX by market capitalisation. Launched in "
             "1980, it is Australia's oldest share index.",
    "^ATLI": "The 20 largest ASX-listed companies by float-adjusted market capitalisation: Australia's blue chips.",
    "^AFLI": "The 50 largest ASX-listed companies by float-adjusted market capitalisation.",
    "^ATOI": "The 100 largest ASX-listed companies by float-adjusted market capitalisation.",
    "^AXKO": "The S&P/ASX 300: the ASX 200 plus 100 smaller companies, a broad measure of the Australian market used "
             "as a benchmark by many fund managers.",
    "^AXSO": "The S&P/ASX Small Ordinaries: companies in the ASX 300 but outside the ASX 100, i.e. Australian "
             "small caps.",
    "^NZ50": "The NZX 50: the 50 largest companies on New Zealand's stock exchange, calculated as a gross index "
             "(dividends reinvested).",
    "^GSPC": "The S&P 500: 500 leading US large-cap companies weighted by float-adjusted market capitalisation. It is "
             "the most widely followed US equity benchmark.",
    "^DJI": "The Dow Jones Industrial Average: 30 large US blue-chip companies, weighted by share price rather than "
            "market capitalisation. First published in 1896.",
    "^IXIC": "The Nasdaq Composite: nearly all common stocks listed on the Nasdaq exchange, heavily weighted towards "
             "technology companies.",
    "^NDX": "The Nasdaq-100: the 100 largest non-financial companies listed on Nasdaq; the index tracked by QQQ.",
    "^RUT": "The Russell 2000: about 2,000 small-cap US companies and the standard US small-cap benchmark.",
    "^NYA": "The NYSE Composite: all common stocks listed on the New York Stock Exchange.",
    "^VIX": "The CBOE Volatility Index: the market's expectation of S&P 500 volatility over the next 30 days, derived "
            "from option prices. Often called the 'fear gauge', it tends to spike when stocks fall and is not "
            "directly investable.",
    "^GSPTSE": "The S&P/TSX Composite: the headline index of the Toronto Stock Exchange, covering Canada's largest "
               "companies and weighted towards financials, energy and materials.",
    "^BVSP": "The Ibovespa: the benchmark index of Brazil's B3 exchange, made up of its most traded stocks.",
    "^MXX": "The S&P/BMV IPC: the main index of the Mexican Stock Exchange, tracking its largest and most liquid stocks.",
    "^892400-USD-STRD": "MSCI ACWI (All Country World Index): large and mid-cap stocks across developed and emerging "
                        "markets in USD. It is the most widely used benchmark for global equities.",
    "^990100-USD-STRD": "MSCI World: large and mid-cap stocks across developed markets only, in USD; the US makes up "
                        "the majority of its weight.",
    "^STOXX50E": "The Euro Stoxx 50: 50 of the largest blue-chip companies in the eurozone.",
    "^STOXX": "The STOXX Europe 600: 600 large, mid and small companies across Europe, including the UK and Switzerland.",
    "^FTSE": "The FTSE 100: the 100 largest companies on the London Stock Exchange. Many earn most of their revenue "
             "outside the UK.",
    "^GDAXI": "The DAX: 40 major German companies on the Frankfurt Stock Exchange. It is a performance index, so "
              "dividends are reinvested.",
    "^FCHI": "The CAC 40: 40 of the largest companies listed on Euronext Paris.",
    "^IBEX": "The IBEX 35: the 35 most liquid Spanish stocks traded in Madrid.",
    "FTSEMIB.MI": "The FTSE MIB: the 40 most traded shares on Borsa Italiana, Italy's main index.",
    "^AEX": "The AEX: the leading Dutch index of the largest companies on Euronext Amsterdam.",
    "^SSMI": "The Swiss Market Index: the 20 largest Swiss companies, dominated by Nestlé, Novartis and Roche.",
    "^N225": "The Nikkei 225: 225 large Japanese companies on the Tokyo Stock Exchange, weighted by share price.",
    "^HSI": "The Hang Seng Index: the largest companies listed in Hong Kong, including many mainland Chinese firms.",
    "000001.SS": "The SSE Composite: all stocks listed on the Shanghai Stock Exchange.",
    "^KS11": "KOSPI: all common stocks on the Korea Exchange's main board, dominated by Samsung Electronics.",
    "^TWII": "The TAIEX (Taiwan Weighted): stocks listed on the Taiwan Stock Exchange, heavily weighted to TSMC.",
    "^BSESN": "The BSE Sensex: 30 large, well-established companies on the Bombay Stock Exchange.",
    "^NSEI": "The Nifty 50: 50 large Indian companies listed on the National Stock Exchange of India.",
    "^STI": "The Straits Times Index: the 30 largest companies on the Singapore Exchange.",
    "^JKSE": "The Jakarta Composite Index: all stocks listed on the Indonesia Stock Exchange.",
    "^KLSE": "The FTSE Bursa Malaysia KLCI: the 30 largest companies on Bursa Malaysia.",
    # ETFs
    "SPY": "SPDR S&P 500 ETF Trust: the oldest and largest US-listed ETF (1993), tracking the S&P 500.",
    "QQQ": "Invesco QQQ Trust: an ETF tracking the Nasdaq-100.",
    "DIA": "SPDR Dow Jones Industrial Average ETF: tracks the 30-stock Dow.",
    "IWM": "iShares Russell 2000 ETF: tracks US small-cap companies.",
    "VTI": "Vanguard Total Stock Market ETF: the entire investable US share market, from mega to micro caps.",
    "VNQ": "Vanguard Real Estate ETF: US real estate investment trusts and property companies.",
    "EFA": "iShares MSCI EAFE ETF: developed-market shares outside the US and Canada (Europe, Australasia, Far East).",
    "EEM": "iShares MSCI Emerging Markets ETF: large and mid-cap shares in emerging markets such as China, India, "
           "Taiwan and Brazil.",
    "ACWI": "iShares MSCI ACWI ETF: tracks the MSCI ACWI global equity index across developed and emerging markets.",
    "STW.AX": "SPDR S&P/ASX 200 Fund: Australia's first ETF (2001), tracking the S&P/ASX 200.",
    "IOZ.AX": "iShares Core S&P/ASX 200 ETF: a low-cost fund tracking the S&P/ASX 200.",
    "VAS.AX": "Vanguard Australian Shares Index ETF: tracks the S&P/ASX 300.",
    "A200.AX": "Betashares Australia 200 ETF: tracks the 200 largest ASX companies (Solactive Australia 200 index).",
    "VGS.AX": "Vanguard MSCI Index International Shares ETF: developed-market shares outside Australia, priced in "
              "AUD without currency hedging.",
    "VAP.AX": "Vanguard Australian Property Securities Index ETF: Australian REITs (S&P/ASX 300 A-REIT index).",
    "SHY": "iShares 1-3 Year Treasury Bond ETF: short-dated US government bonds with little interest-rate risk.",
    "IEF": "iShares 7-10 Year Treasury Bond ETF: intermediate US government bonds.",
    "TLT": "iShares 20+ Year Treasury Bond ETF: long-dated US government bonds, highly sensitive to interest rates.",
    "TIP": "iShares TIPS Bond ETF: US Treasury Inflation-Protected Securities, whose principal rises with US CPI.",
    "AGG": "iShares Core US Aggregate Bond ETF: tracks the Bloomberg US Aggregate, the broad US investment-grade bond "
           "market (Treasuries, corporates and mortgage-backed securities).",
    "BND": "Vanguard Total Bond Market ETF: the broad US investment-grade bond market.",
    "LQD": "iShares iBoxx $ Investment Grade Corporate Bond ETF: US dollar investment-grade corporate bonds.",
    "HYG": "iShares iBoxx $ High Yield Corporate Bond ETF: US dollar high-yield ('junk') corporate bonds.",
    "EMB": "iShares J.P. Morgan USD Emerging Markets Bond ETF: US dollar government bonds issued by emerging-market "
           "countries.",
    "AGGG.L": "iShares Core Global Aggregate Bond UCITS ETF (London, USD): tracks the Bloomberg Global Aggregate, "
              "investment-grade government, corporate and securitised bonds from developed and emerging markets.",
    "VAF.AX": "Vanguard Australian Fixed Interest Index ETF: Australian government, semi-government and corporate "
              "bonds (Bloomberg AusBond Composite 0+ Yr).",
    "IAF.AX": "iShares Core Composite Bond ETF: tracks the Bloomberg AusBond Composite 0+ Yr, the broad Australian "
              "investment-grade bond market.",
    "BIL": "SPDR Bloomberg 1-3 Month T-Bill ETF: very short-dated US Treasury bills; it behaves like US dollar cash.",
    "BILL.AX": "iShares Core Cash ETF: Australian dollar cash and short-term money-market investments.",
    "DJP": "iPath Bloomberg Commodity Index Total Return ETN: an exchange-traded note tracking the Bloomberg Commodity "
           "Index including collateral interest. As a note, it also carries the issuer's credit risk.",
    "REET": "iShares Global REIT ETF: listed real estate investment trusts in developed and emerging markets.",
    "USRT": "iShares Core U.S. REIT ETF: US equity REITs (FTSE Nareit Equity REITs).",
    "QAI": "NYLI Hedge Multi-Strategy Tracker ETF: aims to replicate the combined returns of hedge fund strategies "
           "using liquid ETFs.",
    "GLD": "SPDR Gold Shares: an ETF backed by physical gold bullion.",
    "SLV": "iShares Silver Trust: an ETF backed by physical silver.",
    "USO": "United States Oil Fund: holds near-month WTI crude oil futures; the cost of rolling contracts can make it "
           "lag the oil price over time.",
    "DBC": "Invesco DB Commodity Index Tracking Fund: futures on a basket of heavily traded energy, metal and "
           "agricultural commodities.",
    "GOLD.AX": "Global X Physical Gold: ASX-listed ETF backed by physical gold, priced in AUD without currency hedging.",
    "QAU.AX": "Betashares Gold Bullion ETF (Currency Hedged): physical gold with the US dollar exposure hedged back "
              "to AUD.",
    # Commodities
    "^SPGSCI": "The S&P GSCI: a production-weighted index of commodity futures, typically dominated by energy.",
    "CL=F": "WTI crude oil futures (NYMEX): the US benchmark oil grade, in USD per barrel.",
    "BZ=F": "Brent crude oil futures (ICE): the international oil price benchmark, in USD per barrel.",
    "NG=F": "Henry Hub natural gas futures (NYMEX): the US gas benchmark, in USD per MMBtu; notoriously volatile.",
    "RB=F": "RBOB gasoline futures (NYMEX), in USD per gallon.",
    "HO=F": "NY Harbor ultra-low-sulphur diesel (heating oil) futures (NYMEX), in USD per gallon.",
    "GC=F": "Gold futures (COMEX), in USD per troy ounce. Gold is traditionally held as a store of value and a hedge "
            "against inflation and crises.",
    "SI=F": "Silver futures (COMEX), in USD per troy ounce; both a precious and an industrial metal.",
    "PL=F": "Platinum futures (NYMEX), in USD per troy ounce; used heavily in autocatalysts and jewellery.",
    "PA=F": "Palladium futures (NYMEX), in USD per troy ounce; used mainly in petrol-engine catalytic converters.",
    "HG=F": "Copper futures (COMEX), in USD per pound. Copper is widely watched as a barometer of global industrial "
            "activity.",
    "ALI=F": "Aluminium futures (COMEX), in USD per metric tonne.",
    "TIO=F": "Iron ore 62% Fe futures (China delivery), in USD per dry metric tonne; a key price for Australian miners "
             "such as BHP, Rio Tinto and Fortescue.",
    "ZC=F": "Corn futures (CBOT), in US cents per bushel.",
    "ZW=F": "Wheat futures (CBOT), in US cents per bushel.",
    "ZS=F": "Soybean futures (CBOT), in US cents per bushel.",
    "KC=F": "Arabica coffee futures (ICE), in US cents per pound.",
    "SB=F": "Raw sugar No. 11 futures (ICE), in US cents per pound.",
    "CC=F": "Cocoa futures (ICE), in USD per metric tonne.",
    "CT=F": "Cotton No. 2 futures (ICE), in US cents per pound.",
    "LE=F": "Live cattle futures (CME), in US cents per pound.",
    "HE=F": "Lean hog futures (CME), in US cents per pound.",
    # Rates
    "^IRX": "Yield on 13-week US Treasury bills, in percent: a close proxy for US short-term interest rates.",
    "^FVX": "Yield on the 5-year US Treasury note, in percent.",
    "^TNX": "Yield on the 10-year US Treasury note, in percent: the key reference for long-term borrowing costs, "
            "mortgage rates and equity valuations worldwide.",
    "^TYX": "Yield on the 30-year US Treasury bond, in percent.",
    # Currencies
    "AUDUSD=X": "US dollars per Australian dollar. It rises when the AUD strengthens and is closely linked to "
                "commodity prices and global risk appetite.",
    "DX-Y.NYB": "The ICE US Dollar Index (DXY): the US dollar against six major currencies, dominated by the euro, "
                "with the yen, pound, Canadian dollar, Swedish krona and Swiss franc.",
    "EURUSD=X": "US dollars per euro: the world's most traded currency pair.",
    "GBPUSD=X": "US dollars per British pound ('cable').",
    "USDJPY=X": "Japanese yen per US dollar. It rises when the yen weakens.",
    "USDCNY=X": "Chinese yuan per US dollar (onshore rate, managed by the People's Bank of China).",
    "USDCAD=X": "Canadian dollars per US dollar. It rises when the Canadian dollar weakens.",
    "USDCHF=X": "Swiss francs per US dollar; the franc is a traditional safe-haven currency.",
    "NZDUSD=X": "US dollars per New Zealand dollar ('kiwi').",
    "AUDJPY=X": "Japanese yen per Australian dollar, a popular gauge of global risk appetite.",
    "AUDNZD=X": "New Zealand dollars per Australian dollar.",
    "EURAUD=X": "Australian dollars per euro. It rises when the AUD weakens against the euro.",
    # Crypto
    "BTC-USD": "Bitcoin: the first and largest cryptocurrency (2009), with supply capped at 21 million coins. It "
               "trades around the clock.",
    "ETH-USD": "Ether: the native token of the Ethereum smart-contract blockchain and the second-largest cryptocurrency.",
    "SOL-USD": "Solana: the token of a high-throughput smart-contract blockchain.",
    "XRP-USD": "XRP: the token of the XRP Ledger, designed for payments and cross-border settlement.",
    "BNB-USD": "BNB: the token of the BNB Chain ecosystem, originally launched by the Binance exchange.",
    "ADA-USD": "Cardano: the token of a proof-of-stake smart-contract blockchain.",
    "DOGE-USD": "Dogecoin: a cryptocurrency launched as a joke in 2013 that became widely traded.",
}

SECTORS = {
    "Energy": "oil, gas, coal and uranium producers and energy services",
    "Materials": "miners, chemicals, construction materials and packaging",
    "Industrials": "transport, construction, engineering and commercial services",
    "Consumer Discretionary": "retail, travel and leisure, gaming and household durables",
    "Consumer Staples": "supermarkets, food, beverages and household products",
    "Health Care": "healthcare providers, medical devices, pharmaceuticals and biotechnology",
    "Financials": "banks, insurers and diversified financial companies",
    "Information Technology": "software, IT services and technology hardware",
    "Communication Services": "telecommunications, media and interactive media",
    "Utilities": "electricity, gas and water utilities",
    "A-REIT": "Australian real estate investment trusts and property groups",
    "Real Estate": "real estate investment trusts and property companies",
}

TYPE_NOTES = {
    "energy": "Yahoo's futures series rolls between contracts, so its returns differ from what a futures investor earns.",
    "precious metal": "Yahoo's futures series rolls between contracts, so its returns differ from what a futures investor earns.",
    "industrial metal": "Yahoo's futures series rolls between contracts, so its returns differ from what a futures investor earns.",
    "agriculture": "Yahoo's futures series rolls between contracts, so its returns differ from what a futures investor earns.",
    "livestock": "Yahoo's futures series rolls between contracts, so its returns differ from what a futures investor earns.",
    "yield": "This is a yield, not a price: when yields rise, bond prices fall.",
}
PROFILE_MAX_AGE = timedelta(days=30)


SPDR_ALIASES = {"Technology": "Information Technology", "Financial": "Financials", "Industrial": "Industrials"}


def sector_text(name: str) -> str | None:
    """Descriptions for ASX 200 sector indices and the Select Sector SPDR ETFs, generated from the sector name."""
    first_word = name.split(" ")[0]
    if name.endswith("Select Sector SPDR") and first_word in SPDR_ALIASES:
        name = name.replace(first_word, SPDR_ALIASES[first_word], 1)
    for sector, holdings in SECTORS.items():
        if name == f"ASX 200 {sector}":
            return f"The {sector} sector of the S&P/ASX 200: {holdings}."
        if name.endswith("Select Sector SPDR") and sector in name:
            return f"Select Sector SPDR ETF holding the S&P 500's {sector.lower()} companies: {holdings}."
    return None


def yahoo_profile(ticker: str) -> dict:
    """Company or fund profile from Yahoo, cached in the local database for 30 days."""
    cached = store.query("SELECT * FROM profiles WHERE ticker = ?", [ticker])
    if len(cached) and datetime.now() - cached.fetched[0] < PROFILE_MAX_AGE:
        return cached.iloc[0].to_dict()
    try:
        info = yf.Ticker(ticker).get_info()
    except Exception:
        return cached.iloc[0].to_dict() if len(cached) else {}
    profile = {"ticker": ticker, "summary": info.get("longBusinessSummary") or info.get("description"),
               "sector": info.get("sector"), "industry": info.get("industry"), "website": info.get("website"),
               "fetched": datetime.now()}
    store.upsert("profiles", pd.DataFrame([profile]))
    return profile


def describe(row: pd.Series) -> dict:
    """Description text, where it came from, and optional sector/industry/website details."""
    ticker = row.ticker
    bench = benchmarks.TABLE[benchmarks.TABLE.ticker == ticker]
    bench_note = [f"Used as the {r.region} {r.asset_class.lower()} benchmark ({benchmarks.describe(r)})."
                  for _, r in bench.iterrows()]
    notes = bench_note + ([TYPE_NOTES[row.type]] if row.type in TYPE_NOTES else [])
    if text := CURATED.get(ticker) or sector_text(row["name"]):
        return {"text": text, "notes": notes, "source": "Curated description"}
    profile = yahoo_profile(ticker)
    if profile.get("summary"):
        return {"text": profile["summary"], "notes": notes, "source": "Yahoo Finance profile",
                "sector": profile.get("sector") or row.sector, "industry": profile.get("industry"),
                "website": profile.get("website")}
    return {"text": f"{row['name']} is a {row.type} in the {str(row.asset_class).lower()} asset class, listed in "
                    f"{row.exchange} and priced in {row.currency}. No further description is available.",
            "notes": notes, "source": "Instrument details", "sector": row.sector}
