"""Refresh market data, e.g. python scripts/refresh.py --universe asx200"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from market import ingest, store, universe  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", nargs="+", choices=[*universe.NAMES, "all"], default=list(universe.BUILT_IN))
    parser.add_argument("--tickers", nargs="*", default=[], help="extra tickers, added to the custom universe")
    args = parser.parse_args()

    names = universe.NAMES if "all" in args.universe else args.universe
    tickers = [t for name in names for t in universe.sync(name)["ticker"]]
    if args.tickers:
        tickers += universe.add_custom(args.tickers)["ticker"].tolist()

    log = ingest.refresh(tickers, on_progress=lambda p, msg: print(f"{p:6.1%}  {msg}"))
    failed = log[log.status != "ok"]
    print(f"\n{len(log) - len(failed)}/{len(log)} tickers ok, {store.stats()['rows']:,} price rows")
    if len(failed):
        print(failed[["ticker", "error"]].to_string(index=False))


if __name__ == "__main__":
    main()
