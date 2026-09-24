"""A single self-contained HTML report, so a basket or portfolio can leave the app in one piece.

HTML rather than a spreadsheet: no extra dependency, opens anywhere, and prints to PDF. The header
records the window, interval and adjustments in force, so a saved report says how it was produced.
"""
from datetime import datetime

import pandas as pd

STYLE = """
body{font:14px/1.5 Inter,Segoe UI,sans-serif;color:#1c355e;background:#fff;margin:0;padding:32px;max-width:1100px}
h1{font-size:24px;margin:0 0 4px;letter-spacing:-.02em}h2{font-size:16px;margin:28px 0 8px;color:#0051ff}
.meta{color:#5b6b8c;font-size:13px;margin-bottom:4px}
table{border-collapse:collapse;width:100%;margin:6px 0 2px;font-size:13px}
th,td{border-bottom:1px solid #d8e4f7;padding:6px 10px;text-align:right}
th{background:#f1f6ff;font-weight:600;text-align:right}th:first-child,td:first-child{text-align:left}
.note{color:#5b6b8c;font-size:12px;margin:4px 0 0}
footer{margin-top:36px;color:#5b6b8c;font-size:12px;border-top:1px solid #d8e4f7;padding-top:10px}
"""


def _table(df: pd.DataFrame) -> str:
    numeric = df.select_dtypes("number").columns
    return df.style.format({c: "{:,.4g}" for c in numeric}).to_html(border=0)


def build(title: str, meta: dict, sections: list[tuple[str, object]], notes: list[str] | None = None) -> str:
    """`sections` are (heading, DataFrame or text); anything empty is left out."""
    head = "".join(f'<div class="meta"><b>{k}:</b> {v}</div>' for k, v in meta.items() if v)
    body = []
    for heading, content in sections:
        if content is None or (hasattr(content, "empty") and content.empty):
            continue
        rendered = _table(content) if isinstance(content, pd.DataFrame) else f"<p>{content}</p>"
        body.append(f"<h2>{heading}</h2>{rendered}")
    footnotes = "".join(f'<div class="note">{n}</div>' for n in notes or [])
    stamp = datetime.now().strftime("%d %b %Y %H:%M")
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title><style>{STYLE}</style></head>"
            f"<body><h1>{title}</h1>{head}{''.join(body)}{footnotes}"
            f"<footer>Generated {stamp} by Financial Data Statistical Tool. Past performance is not a guide to "
            f"future returns.</footer></body></html>")


def window(s, adjustments: list[str]) -> dict:
    """The settings a reader needs to reproduce the numbers."""
    return {"Period": f"{s.start or 'earliest available'} to {s.end}",
            "Interval": {"D": "Daily", "W": "Weekly", "M": "Monthly"}[s.freq],
            "Returns": s.kind, "Currency": s.currency,
            "Adjustments": ", ".join(adjustments) if adjustments else "none (data as reported)"}
