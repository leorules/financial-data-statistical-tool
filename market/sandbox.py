"""Code Lab: run user Python against market data with a preloaded namespace."""
import ast
import datetime as dt
import io
import math
import time
import traceback
from contextlib import redirect_stdout
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import scipy  # noqa: E402
import scipy.stats  # noqa: E402,F401  (makes scipy.stats available as an attribute)
import seaborn as sns  # noqa: E402
import sklearn  # noqa: E402
import statsmodels.api as sm  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

from market import benchmarks, factors, inflation, resample, stats, store  # noqa: E402
from market import returns as rets  # noqa: E402
from market.config import SNIPPETS_DIR  # noqa: E402

FILENAME = "<code-lab>"

EXAMPLES = {
    "Correlation heatmap": '''r = returns(["BHP.AX", "RIO.AX", "CBA.AX", "SPY"], freq="W")
show(px.imshow(r.corr(), text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1))
stats.risk.summary(r)''',
    "Rolling beta": '''r = returns(["BHP.AX", "^AXJO"], start="2020-01-01")
beta = stats.rolling.beta(r["BHP.AX"], r["^AXJO"], window=126)
px.line(beta, title="BHP 6-month rolling beta vs ASX 200")''',
    "Return distribution (matplotlib)": '''r = returns("^AXJO")["^AXJO"]
fig, ax = plt.subplots(figsize=(8, 4))
r.hist(bins=100, ax=ax, density=True)
ax.set_title(f"ASX 200 daily returns  skew={r.skew():.2f}  kurt={r.kurt():.1f}")
print(stats.distribution.normality(r)[["test", "p_value"]])
fig''',
    "Seaborn pair plot": '''r = returns(["^GSPC", "^AXJO", "GC=F", "AGG"], freq="W", start="2015-01-01")
grid = sns.pairplot(r, corner=True, plot_kws={"alpha": 0.4, "s": 12})
grid.figure''',
    "Matplotlib drawdown chart": '''p = prices(["^GSPC", "^AXJO"], start="2007-01-01")
drawdown = p / p.cummax() - 1
fig, ax = plt.subplots(figsize=(10, 4))
drawdown.plot(ax=ax, linewidth=1)
ax.fill_between(drawdown.index, drawdown["^GSPC"], 0, alpha=0.15)
ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
ax.set_title("Drawdown from peak")
fig''',
    "Regression with statsmodels": '''r = returns(["BHP.AX", "^AXJO", "HG=F"], freq="W", start="2015-01-01")
model = smf.ols("Q('BHP.AX') ~ Q('^AXJO') + Q('HG=F')", data=r).fit()
print(model.summary())
model.params''',
    "scikit-learn clustering": '''from sklearn.cluster import KMeans
r = returns(basket, freq="W", start="2018-01-01")
features = pd.DataFrame({"return": r.mean() * 52, "volatility": r.std() * 52 ** 0.5})
features["cluster"] = KMeans(n_clusters=min(3, len(features)), n_init=10, random_state=0).fit_predict(features)
show(px.scatter(features, x="volatility", y="return", color=features.cluster.astype(str), text=features.index))
features''',
    "Basket growth of $10k": '''p = prices(basket, start="2021-01-01")
growth = 10_000 * p / p.bfill().iloc[0]
px.line(growth, title="Growth of $10,000")''',
}


@dataclass
class Run:
    outputs: list = field(default_factory=list)
    stdout: str = ""
    error: str | None = None
    seconds: float = 0.0


def _list(tickers) -> list[str]:
    return [tickers] if isinstance(tickers, str) else list(tickers)


def prices(tickers, start=None, end=None, field: str = "adj_close", freq: str = "D") -> pd.DataFrame:
    """Wide price matrix (dates × tickers) from the local database."""
    wide = rets.price_matrix(store.prices(_list(tickers), start, end), field)
    return resample.total(wide, freq) if field == "volume" else resample.last(wide, freq)


def returns(tickers, freq: str = "D", kind: str = "simple", start=None, end=None) -> pd.DataFrame:
    """Aligned return matrix (common trading dates only)."""
    return rets.compute(prices(tickers, start, end), freq, kind)


def namespace(basket=()) -> dict:
    return {"prices": prices, "returns": returns, "basket": list(basket), "stats": stats, "store": store,
            "benchmarks": benchmarks, "factors": factors, "inflation": inflation,
            "pd": pd, "np": np, "plt": plt, "sns": sns, "px": px, "go": go,
            "scipy": scipy, "sm": sm, "smf": smf, "sklearn": sklearn, "math": math, "dt": dt}


def format_error(exc: BaseException, code: str) -> str:
    lines = code.splitlines()
    if isinstance(exc, SyntaxError):
        return f"Line {exc.lineno}: {(exc.text or '').strip()}\nSyntaxError: {exc.msg}"
    frames = [f for f in traceback.extract_tb(exc.__traceback__) if f.filename == FILENAME]
    where = "\n".join(f"Line {f.lineno}: {lines[f.lineno - 1].strip()}" for f in frames if f.lineno)
    return f"{where}\n{type(exc).__name__}: {exc}".strip()


def _collect(outputs: list) -> list:
    """Turn Axes into figures, add any open matplotlib figures, and drop duplicates."""
    items = [o.figure if isinstance(o, Axes) else o for o in outputs]
    items += [plt.figure(n) for n in plt.get_fignums()]
    unique, seen = [], set()
    for item in items:
        if id(item) not in seen:
            seen.add(id(item))
            unique.append(item)
    return unique


def run(code: str, env: dict) -> Run:
    out = Run()
    ns = {**env, "show": out.outputs.append}
    buffer, start = io.StringIO(), time.perf_counter()
    plt.close("all")
    try:
        tree = ast.parse(code, FILENAME)
        last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
        with redirect_stdout(buffer):
            exec(compile(tree, FILENAME, "exec"), ns)
            if last:
                value = eval(compile(ast.Expression(last.value), FILENAME, "eval"), ns)
                if value is not None:
                    out.outputs.append(value)
    except Exception as e:
        out.error = format_error(e, code)
    out.outputs = _collect(out.outputs)
    out.stdout, out.seconds = buffer.getvalue(), time.perf_counter() - start
    return out


def snippets() -> dict[str, str]:
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted(SNIPPETS_DIR.glob("*.py"))}


def save_snippet(name: str, code: str) -> None:
    SNIPPETS_DIR.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in name).strip()
    (SNIPPETS_DIR / f"{safe}.py").write_text(code, encoding="utf-8")


def delete_snippet(name: str) -> None:
    (SNIPPETS_DIR / f"{name}.py").unlink(missing_ok=True)
