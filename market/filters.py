import operator
from dataclasses import dataclass

import numpy as np
import pandas as pd

OPS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le, "between": None}


def parse_value(value) -> float | str:
    """'0.05' → 0.05, '5%' → 0.05, 'sma_200' → 'sma_200'."""
    text = str(value).strip()
    try:
        return float(text[:-1]) / 100 if text.endswith("%") else float(text)
    except ValueError:
        return text


@dataclass
class Rule:
    left: str
    op: str
    right: float | str
    right2: float | str | None = None

    def mask(self, df: pd.DataFrame) -> pd.Series:
        for name in (self.left, self.right, self.right2):
            if isinstance(name, str) and name not in df:
                raise ValueError(f"Unknown metric '{name}'")
        side = lambda v: df[v] if isinstance(v, str) else v  # noqa: E731
        if self.op == "between":
            return df[self.left].between(side(self.right), side(self.right2))
        return OPS[self.op](df[self.left], side(self.right))

    def __str__(self) -> str:
        return f"{self.left} between {self.right} and {self.right2}" if self.op == "between" else \
            f"{self.left} {self.op} {self.right}"


def rules_from_records(records: list[dict]) -> list[Rule]:
    """Build rules from editor rows, skipping incomplete ones."""
    rules = []
    for r in records:
        if not r.get("left") or not r.get("op") or pd.isna(r.get("right")) or r.get("right") == "":
            continue
        right2 = r.get("right2")
        rules.append(Rule(r["left"], r["op"], parse_value(r["right"]),
                          None if right2 is None or pd.isna(right2) or right2 == "" else parse_value(right2)))
    return rules


def universe(inst: pd.DataFrame, search: str = "", **selected) -> pd.DataFrame:
    """Keep instruments matching every non-empty selection, e.g. asset_class=["Commodities"], exchange=["Australia"]."""
    mask = pd.Series(True, index=inst.index)
    for col, values in selected.items():
        if values:
            mask &= inst[col].isin(values)
    if search:
        mask &= inst.ticker.str.contains(search, case=False, regex=False) | \
                inst.name.fillna("").str.contains(search, case=False, regex=False)
    return inst[mask]


def apply(snapshot: pd.DataFrame, rules: list[Rule]) -> pd.DataFrame:
    if not rules:
        return snapshot
    return snapshot[np.logical_and.reduce([r.mask(snapshot).fillna(False) for r in rules])]
