from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats as st

from market.stats.core import result


def mean_zero(s: pd.Series, level: float = 0.95) -> pd.Series:
    s = s.dropna()
    t = st.ttest_1samp(s, 0)
    low, high = st.t.interval(level, len(s) - 1, loc=s.mean(), scale=s.sem())
    return result("One-sample t (mean = 0)", t.statistic, t.pvalue, "mean = 0", mean=s.mean(), ci_low=low, ci_high=high)


def welch(a: pd.Series, b: pd.Series) -> pd.Series:
    t = st.ttest_ind(a.dropna(), b.dropna(), equal_var=False)
    return result("Welch t (equal means)", t.statistic, t.pvalue, "means are equal")


def mann_whitney(a: pd.Series, b: pd.Series) -> pd.Series:
    u = st.mannwhitneyu(a.dropna(), b.dropna())
    return result("Mann–Whitney U", u.statistic, u.pvalue, "same distribution location")


def f_test(a: pd.Series, b: pd.Series) -> pd.Series:
    a, b = a.dropna(), b.dropna()
    f = a.var() / b.var()
    cdf = st.f.cdf(f, len(a) - 1, len(b) - 1)
    return result("F-test (equal variance)", f, 2 * min(cdf, 1 - cdf), "variances are equal")


def levene(a: pd.Series, b: pd.Series) -> pd.Series:
    lv = st.levene(a.dropna(), b.dropna())
    return result("Levene (equal variance)", lv.statistic, lv.pvalue, "variances are equal")


def ks_2samp(a: pd.Series, b: pd.Series) -> pd.Series:
    ks = st.ks_2samp(a.dropna(), b.dropna())
    return result("Two-sample KS", ks.statistic, ks.pvalue, "same distribution")


def compare(a: pd.Series, b: pd.Series) -> pd.DataFrame:
    return pd.DataFrame([f(a, b) for f in (welch, mann_whitney, f_test, levene, ks_2samp)])


def bootstrap(data: pd.Series | pd.DataFrame, fn: Callable[[np.ndarray], float], n: int = 2000,
              level: float = 0.95, seed: int = 0) -> tuple[pd.Series, np.ndarray]:
    """Percentile bootstrap CI; fn receives resampled rows as a numpy array."""
    x = data.dropna().to_numpy()
    rng = np.random.default_rng(seed)
    samples = np.array([fn(x[rng.integers(0, len(x), len(x))]) for _ in range(n)])
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(samples, [tail, 100 - tail])
    return pd.Series({"estimate": fn(x), "ci_low": low, "ci_high": high, "level": level}), samples


def boot_mean(x: np.ndarray) -> float:
    return x.mean()


def boot_sharpe(periods: int) -> Callable[[np.ndarray], float]:
    return lambda x: x.mean() / x.std(ddof=1) * np.sqrt(periods)


def boot_corr(x: np.ndarray) -> float:
    return np.corrcoef(x[:, 0], x[:, 1])[0, 1]
