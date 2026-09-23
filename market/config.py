from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DB_PATH = DATA / "market.duckdb"
UNIVERSE_DIR = DATA / "universe"
PRESETS_PATH = DATA / "presets.json"
SNIPPETS_DIR = DATA / "snippets"

BATCH_SIZE = 50
BATCH_PAUSE = 1.0
RETRIES = 3
REFETCH_DAYS = 5
MIN_OBS = 30
RISK_FREE = 0.04
PERIODS = {"D": 252, "W": 52, "M": 12}
ASX_INDICES = {"^AORD", "^ATLI", "^AFLI", "^ATOI"}
# Accumulation series standing in for each price index, for beta and alpha measured on total return.
TOTAL_RETURN = {"^AXJO": "STW.AX", "^GSPC": "^SP500TR"}


def benchmark_for(ticker: str, total_return: bool = False) -> str:
    asx = ticker.endswith(".AX") or ticker.startswith("^AX") or ticker in ASX_INDICES
    index = "^AXJO" if asx else "^GSPC"
    return TOTAL_RETURN[index] if total_return else index
