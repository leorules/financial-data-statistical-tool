"""Rule-based, plain-English interpretation of statistical results. Free, instant and offline."""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm

from market import benchmarks
from market.config import PERIODS, RISK_FREE
from market.stats.core import periods_per_year

PERIOD = {"D": "day", "W": "week", "M": "month"}
STAKE = 10_000

GLOSSARY = {
    "CAGR": "Compound annual growth rate: the constant yearly return that turns the starting value into the ending value.",
    "Standard deviation": "How widely returns scatter around their average. Annualised, it is called volatility.",
    "Skewness": "Asymmetry of returns. Negative means a longer tail of large losses; positive, a longer tail of large gains.",
    "Kurtosis": "Tail heaviness (excess over a normal distribution). Above 0 means extreme moves are more frequent than normal.",
    "Payoff ratio": "Average gain on up periods divided by average loss on down periods.",
    "z-score": "How many standard deviations a value sits from its average.",
    "Rolling window": "A statistic recalculated on the most recent N periods, so you can see it change through time.",
    "EWMA volatility": "Volatility that weights recent returns more heavily (RiskMetrics λ = 0.94), so it reacts faster.",
    "Sharpe ratio": "Annual return above the cash rate per unit of volatility, using the arithmetic mean "
                    "excess return. Above 1 is good. Annualised return compounds instead, so the two differ "
                    "slightly for volatile series.",
    "Sortino ratio": "Like Sharpe, but only penalises downside volatility.",
    "Max drawdown": "The largest fall from a previous peak before a new high was made.",
    "Calmar ratio": "Annual return divided by the size of the maximum drawdown.",
    "Ulcer index": "Measures the depth and duration of drawdowns; lower means less painful to hold.",
    "Beta": "Sensitivity to the benchmark: a beta of 1.2 has meant moving about 1.2% for each 1% benchmark move.",
    "Up/down capture": "Average return in the benchmark's up (down) periods as a share of the benchmark's own move.",
    "Tracking error": "Volatility of the return difference versus the benchmark.",
    "Information ratio": "Annual excess return over the benchmark divided by tracking error.",
    "p-value": "The chance of seeing a result at least this extreme if there were really no effect. Below 0.05 is the usual bar.",
    "Confidence interval": "A range of values consistent with the data; a 95% interval would contain the true value 95% of the time.",
    "Effect size": "The size of an effect in standard-deviation units, independent of sample size.",
    "Bootstrap": "Resampling the observed data thousands of times to see how much a statistic could vary.",
    "Welch t-test": "Tests whether two averages differ, without assuming equal volatility.",
    "Mann–Whitney U": "Rank-based test of whether one series tends to produce higher values than the other.",
    "F-test / Levene": "Test whether two series have different volatility; Levene is robust to fat tails.",
    "Kolmogorov–Smirnov": "Tests whether two samples come from the same overall distribution.",
    "Normality tests": "Jarque–Bera, Shapiro–Wilk, Lilliefors and Anderson–Darling each test whether data could be normal.",
    "VaR": "Value at Risk: the loss threshold exceeded only in the worst 5% (or 1%) of periods.",
    "CVaR": "Conditional VaR (expected shortfall): the average loss in those worst periods.",
    "Student-t distribution": "A bell curve with heavier tails; fewer degrees of freedom means fatter tails.",
    "R²": "Share of the variation in y explained by the regression.",
    "Alpha": "The regression intercept: return not explained by the factors.",
    "Durbin–Watson": "Checks residual autocorrelation; about 2 is ideal.",
    "Breusch–Pagan": "Tests whether the size of regression errors changes with the factors (heteroskedasticity).",
    "ADF": "Augmented Dickey–Fuller: null hypothesis is a unit root (random-walk-like, no fixed level).",
    "KPSS": "Null hypothesis is stationarity, the reverse of ADF, so the two are read together.",
    "Ljung–Box": "Tests whether returns are correlated with their own past values.",
    "Variance ratio": "Compares multi-period variance with single-period variance: above 1 suggests momentum, below 1 mean reversion.",
    "Hurst exponent": "Long-memory measure: 0.5 random walk, above 0.5 trending, below 0.5 mean-reverting.",
    "Cointegration": "Two prices share a long-run equilibrium, so the gap between them tends to close.",
    "Half-life": "Typical time for half of a gap from equilibrium to close.",
    "Granger causality": "Whether past values of one series improve forecasts of another. Predictive, not truly causal.",
    "Correlation": "Co-movement from −1 (opposite) to +1 (together).",
    "PCA": "Principal component analysis: finds the common factors that drive a group of series.",
    "Effective number of bets": "How many independent assets the basket behaves like, based on its PCA.",
    "ANOVA / Kruskal–Wallis": "Test whether average returns differ across groups such as calendar months.",
}


@dataclass
class Reading:
    headline: str
    findings: list[str] = field(default_factory=list)
    implications: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    terms: list[str] = field(default_factory=list)

    def definitions(self) -> dict[str, str]:
        return {t: GLOSSARY[t] for t in dict.fromkeys(self.terms) if t in GLOSSARY}


# --- wording helpers -------------------------------------------------------------------------------------------

def pct(x: float, digits: int = 1, sign: bool = False) -> str:
    return "n/a" if pd.isna(x) else f"{x:+.{digits}%}" if sign else f"{x:.{digits}%}"


def owner(name: str) -> str:
    return f"{name}'" if name.endswith("s") else f"{name}'s"


def money(x: float) -> str:
    return f"${x:,.0f}"


def evidence(p: float) -> str:
    if pd.isna(p):
        return "no p-value available"
    level = ("very strong" if p < 0.001 else "strong" if p < 0.01 else "moderate" if p < 0.05
             else "weak" if p < 0.10 else "no real")
    return f"{level} evidence (p = {p:.3g})"


def significant(p: float, alpha: float = 0.05) -> bool:
    return not pd.isna(p) and p < alpha


def corr_strength(r: float) -> str:
    size = abs(r)
    word = ("very weak" if size < 0.2 else "weak" if size < 0.4 else "moderate" if size < 0.6
            else "strong" if size < 0.8 else "very strong")
    return f"{word} {'positive' if r >= 0 else 'negative'}"


def vol_level(v: float) -> str:
    return "low" if v < 0.10 else "moderate" if v < 0.20 else "high" if v < 0.35 else "very high"


def sharpe_level(s: float) -> str:
    return ("negative (below cash)" if s < 0 else "weak" if s < 0.5 else "reasonable" if s < 1
            else "good" if s < 2 else "excellent")


def tail_findings(skew: float, kurt: float, mean: float | None = None, median: float | None = None) -> list[str]:
    notes = []
    gap = "" if mean is None else f" The mean ({pct(mean, 2, True)}) sits {'below' if mean < median else 'above'} the median ({pct(median, 2, True)}), which is what that asymmetry does."
    if skew < -0.5:
        notes.append(f"**Negative skew ({skew:.2f}):** the left tail is longer; sharp falls outweigh equally sharp rallies.{gap}")
    elif skew > 0.5:
        notes.append(f"**Positive skew ({skew:.2f}):** occasional large gains stretch the right tail.{gap}")
    else:
        notes.append(f"**Roughly symmetric ({skew:.2f} skew):** large gains and large losses have been similarly common.")
    if kurt > 1:
        notes.append(f"**Fat tails (excess kurtosis {kurt:.1f}):** extreme moves occur far more often than a bell curve "
                     "implies, so standard-deviation ranges understate how bad bad periods get.")
    return notes


def regime(r: pd.Series, ppy: int) -> str:
    first, second = r.iloc[: len(r) // 2], r.iloc[len(r) // 2:]
    v1, v2 = first.std() * np.sqrt(ppy), second.std() * np.sqrt(ppy)
    change = v2 / v1 - 1
    if abs(change) < 0.25:
        return f"Volatility has been fairly stable: {pct(v1)} in the first half of the period and {pct(v2)} in the second."
    return (f"The risk profile has **shifted**: volatility was {pct(v1)} in the first half and {pct(v2)} in the second "
            f"({pct(change, 0, True)}), so full-period averages may not describe current conditions well.")


def sample_caveat(n: int, period: str) -> list[str]:
    return [f"Only {n} {period}s of data: estimates like averages and correlations are noisy on samples this short."] \
        if n < 60 else []


PAST = "These statistics describe the past; they are not a forecast or a recommendation."


# --- analyses --------------------------------------------------------------------------------------------------

def descriptive(table: pd.DataFrame, focus: str, returns: bool, freq: str, series: pd.Series) -> Reading:
    period, row = PERIOD[freq], table.loc[focus]
    if not returns:
        position = ("well above" if row.last_z > 1.5 else "well below" if row.last_z < -1.5 else "close to")
        return Reading(
            f"{focus} ranged from {row['min']:,.2f} to {row['max']:,.2f} and now sits {position} its average level.",
            findings=[f"The average level was {row['mean']:,.2f} (median {row['median']:,.2f}); half of the time it "
                      f"traded between {row.q1:,.2f} and {row.q3:,.2f}.",
                      f"The latest value, {row['last']:,.2f}, is {row.last_z:+.1f} standard deviations from that average."],
            implications=["Prices trend, so their averages and standard deviations depend heavily on the date range "
                          "chosen. Switch the series to Returns for statistics that compare fairly across time and assets."],
            caveats=[PAST], terms=["Standard deviation", "z-score"])

    r, ppy = series.dropna(), periods_per_year(series.index)
    gains, losses = r[r > 0], r[r < 0]
    ann_vol, ann_mean = row["std"] * np.sqrt(ppy), row["mean"] * ppy
    payoff = gains.mean() / -losses.mean()
    extremes, expected = int(row.outliers_z3), len(r) * 0.0027
    short = len(r) < 60
    headline = (f"Over these {len(r)} {period}s {focus} returned {pct(row.cum_return, 1, True)} with "
                f"{vol_level(ann_vol)} volatility ({pct(ann_vol)} annualised)" if short else
                f"{focus} compounded {pct(row.cagr, sign=True)} a year with {vol_level(ann_vol)} volatility "
                f"({pct(ann_vol)} annualised)")
    reading = Reading(
        headline + (", and its returns have fat tails." if row["kurtosis"] > 1 else "."),
        findings=[
            f"**Growth:** {money(STAKE)} at the start would now be about {money(STAKE * (1 + row.cum_return))} "
            f"({pct(row.cum_return, 0, True)} over {len(r)} {period}s).",
            f"**Hit rate and payoff:** {pct(row.pct_positive, 0)} of {period}s were positive. The average up {period} "
            f"gained {pct(gains.mean(), 2)} and the average down {period} lost {pct(-losses.mean(), 2)}, a payoff ratio "
            f"of {payoff:.2f}. " + ("Winners were bigger than losers, so it did not need a high hit rate."
                                     if payoff > 1 else "Losers were bigger than winners, so returns relied on winning more often."),
            f"**Typical range:** about two-thirds of {period}s fell between {pct(row['mean'] - row['std'], 2, True)} and "
            f"{pct(row['mean'] + row['std'], 2, True)}, and 90% between {pct(row.p5, 2, True)} and {pct(row.p95, 2, True)}.",
            f"**Extremes:** best {period} {pct(row.best, 1, True)} ({r.idxmax():%d %b %Y}), worst {pct(row.worst, 1, True)} "
            f"({r.idxmin():%d %b %Y}). Longest winning streak {int(row.win_streak)} {period}s; longest losing streak "
            f"{int(row.loss_streak)}.",
            *tail_findings(row["skew"], row["kurtosis"], row["mean"], row["median"]),
        ],
        implications=[
            f"On an annual basis the average return of {pct(ann_mean, 1, True)} against volatility of {pct(ann_vol)} means "
            f"a ±1 standard-deviation year spans roughly {pct(ann_mean - ann_vol, 0, True)} to {pct(ann_mean + ann_vol, 0, True)}; "
            f"a normal approximation puts the chance of a losing year near {pct(norm.cdf(-ann_mean / ann_vol), 0)}.",
            regime(r, ppy),
        ],
        caveats=sample_caveat(len(r), period) + [PAST],
        terms=["CAGR", "Standard deviation", "Payoff ratio", "Skewness", "Kurtosis"],
    )
    if extremes > max(2 * expected, 1):
        reading.findings.append(f"**Outliers:** {extremes} moves exceeded 3 standard deviations, about "
                                f"{extremes / expected:.0f}× what a normal distribution predicts ({expected:.1f}).")
    if abs(row.last_z) > 2:
        reading.findings.append(f"**Latest {period}:** {pct(row['last'], 2, True)} is unusually large (z = {row.last_z:.1f}).")
    if len(table) > 1:
        efficiency = (table["mean"] / table["std"]).sort_values()
        reading.implications.append(
            f"Across the selection, **{table.cagr.idxmax()}** grew fastest and **{table['std'].idxmax()}** was the most "
            f"volatile, but per unit of volatility **{efficiency.index[-1]}** delivered the most return and "
            f"**{efficiency.index[0]}** the least.")
    return reading


def rolling(risk: pd.DataFrame, window: int, freq: str, focus: str) -> Reading | None:
    period = PERIOD[freq]
    vol = risk["rolling vol"].dropna()
    if vol.empty:
        return None
    rank = (vol <= vol.iloc[-1]).mean()
    reading = Reading(
        f"{owner(focus)} volatility is {pct(vol.iloc[-1])}, "
        + (f"higher than {pct(rank, 0)}" if rank >= 0.5 else f"lower than {pct(1 - rank, 0)}")
        + " of its readings over this period.",
        terms=["Rolling window", "EWMA volatility", "Sharpe ratio", "Beta", "Correlation"],
        caveats=[f"A {window}-{period} window trades responsiveness for noise: shorter windows react faster but jump "
                 "around more. Neighbouring readings overlap, so they are not independent observations.", PAST],
    )
    for col in risk:
        s = risk[col].dropna()
        if len(s) <= window:
            continue
        fmt = pct if "vol" in col else (lambda v: f"{v:.2f}")
        change = s.iloc[-1] - s.iloc[-window]
        direction = "rising" if change > 0 else "falling"
        reading.findings.append(
            f"**{col}:** {fmt(s.iloc[-1])} now (median {fmt(s.median())}, range {fmt(s.min())} to {fmt(s.max())}), "
            f"{direction} from {fmt(s.iloc[-window])} one window ago.")
    ewma, roll = risk["EWMA vol (λ=0.94)"].dropna(), vol
    if len(ewma) and ewma.iloc[-1] > roll.iloc[-1] * 1.15:
        reading.implications.append("The fast-reacting EWMA volatility is above the rolling figure, so risk has picked "
                                    "up recently and may not yet be fully reflected in the window average.")
    elif len(ewma) and ewma.iloc[-1] < roll.iloc[-1] * 0.85:
        reading.implications.append("EWMA volatility is below the rolling figure: markets have calmed recently.")
    reading.implications.append("Volatility clusters: calm and turbulent spells tend to persist, so the current "
                                "reading is usually a better near-term guide than the long-run average.")
    sharpe = risk["rolling Sharpe"].dropna()
    if len(sharpe):
        reading.implications.append(f"The rolling Sharpe ratio was positive {pct((sharpe > 0).mean(), 0)} of the time, "
                                    "showing how often a buyer over any single window would have beaten cash.")
    for col in [c for c in risk if c.startswith(("beta", "corr"))]:
        s = risk[col].dropna()
        if not len(s):
            continue
        share = (s > 1).mean() if col.startswith("beta") else (s < 0).mean()
        label = "above 1 (amplifying the benchmark)" if col.startswith("beta") else "negative (a hedge)"
        reading.implications.append(
            f"The {col} ranged {s.min():.2f} to {s.max():.2f} and was {label} {pct(share, 0)} of the time"
            + (": the relationship is unstable, so a single full-period estimate can mislead." if s.max() - s.min() > 0.5 else "."))
    return reading


def distribution(summary: pd.Series, normality: pd.DataFrame, fit: pd.DataFrame, var: pd.DataFrame, freq: str,
                 series: pd.Series) -> Reading:
    period, r = PERIOD[freq], series.dropna()
    ppy = periods_per_year(r.index)
    rejected = int(normality.p_value.lt(0.05).sum())
    hist, param = (var[var.method == m].set_index("level") for m in ("historical", "parametric"))
    z = (r - r.mean()) / r.std()
    observed, expected = int((z.abs() > 3).sum()), len(r) * 0.0027
    left, right = abs(summary.p5 - summary["mean"]), abs(summary.p95 - summary["mean"])
    reading = Reading(
        ("Returns are **not normally distributed**" if rejected >= 2 else "Returns are close to normal")
        + f": in a bad 1-in-20 {period} expect {pct(hist.VaR[0.95], 1, True)} or worse, averaging {pct(hist.CVaR[0.95], 1, True)}.",
        findings=[
            f"**Normality:** {rejected} of {len(normality)} tests reject a normal distribution"
            + (f" (strongest: {normality.loc[normality.p_value.idxmin(), 'test']}, p = {normality.p_value.min():.2g})." if rejected else "."),
            *tail_findings(summary["skew"], summary["kurtosis"]),
            f"**Extreme moves:** {observed} {period}s moved more than 3 standard deviations, versus about {expected:.1f} "
            f"expected under normality" + (f", {observed / expected:.0f}× more often." if observed > expected * 1.5 else "."),
            f"**Tail balance:** the 5th percentile sits {pct(left, 2)} below the mean and the 95th percentile {pct(right, 2)} "
            f"above it; the {'downside' if left > right else 'upside'} tail is {'longer' if abs(left - right) > 0.1 * max(left, right) else 'only slightly longer'}.",
            f"**Value at Risk:** 95% VaR {pct(hist.VaR[0.95], 2, True)} (CVaR {pct(hist.CVaR[0.95], 2, True)}); 99% VaR "
            f"{pct(hist.VaR[0.99], 2, True)} (CVaR {pct(hist.CVaR[0.99], 2, True)}).",
        ],
        implications=[
            f"For a {money(STAKE)} position, a 1-in-20 {period} would lose about {money(-STAKE * hist.VaR[0.95])} or more, "
            f"and the average loss in those {period}s would be about {money(-STAKE * hist.CVaR[0.95])}.",
            f"Expect the 99% VaR to be breached roughly {0.01 * ppy:.1f} times a year and the 95% VaR about {0.05 * ppy:.0f} times.",
        ],
        caveats=["VaR says how often a threshold is crossed, not how far beyond it losses go; CVaR covers that.",
                 "Historical VaR only knows the crises inside the chosen date range.", PAST],
        terms=["Normality tests", "Skewness", "Kurtosis", "VaR", "CVaR", "Student-t distribution"],
    )
    understatement = hist.VaR[0.99] / param.VaR[0.99] - 1
    if understatement > 0.1:
        reading.implications.append(f"At the 99% level the normal model understates the historical loss by "
                                    f"{pct(understatement, 0)}; use historical or Cornish–Fisher estimates for tail risk.")
    best = fit.iloc[0]
    if best.distribution == "student_t":
        dof = best.params[0]
        reading.findings.append(f"**Best fit:** Student-t with {dof:.1f} degrees of freedom ("
                                + ("very heavy tails" if dof < 5 else "moderately heavy tails" if dof < 30 else "near-normal tails")
                                + f"), with a lower AIC than the normal fit by {fit.aic.iloc[1] - best.aic:.0f}.")
    return reading


def hypothesis(mean_test: pd.Series, comparison: pd.DataFrame, ci: pd.Series, stat: str, focus: str, other: str,
               freq: str, x: pd.Series, y: pd.Series) -> Reading:
    period, x, y = PERIOD[freq], x.dropna(), y.dropna()
    ppy = periods_per_year(x.index)
    effect = x.mean() / x.std()
    needed = (1.96 / effect) ** 2 if effect else np.inf
    reading = Reading(
        f"{owner(focus)} average return is {'statistically distinguishable' if significant(mean_test.p_value) else 'not statistically distinguishable'} "
        f"from zero, and it {'differs' if significant(comparison.p_value.min()) else 'does not clearly differ'} from {other} on at least one test.",
        findings=[
            f"**Average:** {pct(x.mean(), 3, True)} per {period} (≈ {pct(x.mean() * ppy, 1, True)} a year), with a 95% "
            f"confidence interval of {pct(mean_test.ci_low, 3, True)} to {pct(mean_test.ci_high, 3, True)}; "
            f"{evidence(mean_test.p_value)} against a true average of zero.",
            f"**Effect size:** the average is {abs(effect):.3f} standard deviations per {period}. At that size you would "
            f"need about {needed:,.0f} {period}s ({needed / ppy:,.1f} years) of data for it to reach 5% significance, "
            f"versus the {len(x)} available.",
        ],
        caveats=["Returns are not independent draws (volatility clusters, trends), which makes p-values somewhat optimistic.",
                 f"Running {len(comparison) + 1} tests at once raises the chance that one looks significant by luck.", PAST],
        terms=["p-value", "Confidence interval", "Effect size", "Welch t-test", "Mann–Whitney U", "F-test / Levene",
               "Kolmogorov–Smirnov", "Bootstrap"],
    )
    diffs = {"Welch": f"average returns (difference {pct((x.mean() - y.mean()) * ppy, 1, True)} a year)",
             "Mann": "typical returns (rank-based, robust to outliers)",
             "F-test": f"volatility ({focus} is {x.std() / y.std():.2f}× as volatile)",
             "Levene": "volatility, robust to fat tails", "Two-sample KS": "the overall shape of their returns"}
    for _, row in comparison.iterrows():
        what = next(v for k, v in diffs.items() if row.test.startswith(k))
        reading.findings.append(f"**{row.test}:** {'a significant difference' if significant(row.p_value) else 'no significant difference'} "
                                f"in {what}; {evidence(row.p_value)}.")
    tests = comparison.set_index("test").p_value
    f_sig, levene_sig = (significant(tests.filter(like=k).iloc[0]) for k in ("F-test", "Levene"))
    if f_sig and not levene_sig:
        reading.implications.append("The F-test finds a volatility difference but Levene does not, which usually means a "
                                    "few extreme moves drive the gap rather than a consistently different level of risk.")
    reading.implications.append(
        "Statistical significance is not economic significance: a tiny but real edge can be swamped by costs, and a "
        "large average can be insignificant simply because returns are noisy." if not significant(mean_test.p_value) else
        "The average return is large relative to its noise, but check that it is not driven by one short spell.")
    includes_zero = ci.ci_low <= 0 <= ci.ci_high
    reading.findings.append(f"**Bootstrap ({ci.level:.0%}):** the {stat} ranges {ci.ci_low:.4g} to {ci.ci_high:.4g} "
                            f"around an estimate of {ci.estimate:.4g}; "
                            + ("the range spans zero, so even its sign is uncertain." if includes_zero
                               else "the range excludes zero, so the sign is reliable."))
    return reading


def risk(table: pd.DataFrame, focus: str, bench: str | None, series: pd.Series,
         cash_rate: float | None = None) -> Reading:
    """`cash_rate` is the average rate actually subtracted; None means the flat assumption."""
    rate = RISK_FREE if cash_rate is None else cash_rate
    row = table.loc[focus]
    short = len(series.dropna()) < 60
    current_dd = float((1 + series.dropna()).cumprod().pipe(lambda w: w.iloc[-1] / w.max() - 1))
    recovery_gain = -row.max_drawdown / (1 + row.max_drawdown)
    reading = Reading(
        f"{focus} earned {pct(row.ann_return, sign=True)} a year at {pct(row.ann_vol)} volatility: a "
        f"{sharpe_level(row.sharpe)} risk-adjusted result (Sharpe {row.sharpe:.2f}).",
        findings=[
            f"**Return vs risk:** after subtracting the {rate:.2%} cash rate, each unit of volatility earned "
            f"{row.sharpe:.2f} units of return. On a normal approximation that implies about a "
            f"{pct(norm.cdf(row.sharpe), 0)} chance of beating cash in any given year.",
            f"**Downside:** downside deviation is {pct(row.downside_dev)}; the Sortino ratio of {row.sortino:.2f} "
            + ("is well above the Sharpe ratio, so much of the volatility came from gains." if row.sortino > row.sharpe * 1.3
               else "is close to the Sharpe ratio, so volatility was split fairly evenly between gains and losses."),
            f"**Worst drawdown:** {pct(row.max_drawdown)} from {pd.Timestamp(row.peak):%b %Y} to "
            f"{pd.Timestamp(row.trough):%b %Y}, lasting {int(row.dd_length)} periods from peak "
            + (f"to recovery in {pd.Timestamp(row.recovery):%b %Y}." if pd.notna(row.recovery) else "and **still not recovered**.")
            + f" It is currently {pct(-current_dd)} below its peak.",
            f"**Pain measures:** Calmar ratio {row.calmar:.2f} (annual return per unit of worst drawdown) and ulcer index "
            f"{pct(row.ulcer_index)} ({'deep or prolonged' if row.ulcer_index > 0.1 else 'moderate' if row.ulcer_index > 0.05 else 'shallow'} drawdowns overall).",
        ],
        implications=[f"Losses compound asymmetrically: recovering from a {pct(row.max_drawdown, 0)} fall needs a "
                      f"{pct(recovery_gain, 0, True)} gain."],
        caveats=(["Annualised figures from fewer than 60 observations are arithmetic extrapolations of a short "
                  "window, not an expectation for a full year."] if short else [])
                + ["Sharpe and Sortino assume a stable return distribution; fat tails and regime changes make them look "
                   "better than the risk really is."
                   + (f" The cash rate is a fixed {RISK_FREE:.0%} assumption." if cash_rate is None else
                      " The cash rate is the one that actually applied on each date."), PAST],
        terms=["Sharpe ratio", "Sortino ratio", "Max drawdown", "Calmar ratio", "Ulcer index"],
    )
    if bench and pd.notna(row.get("beta")):
        style = ("moves against" if row.beta < 0 else "is defensive relative to" if row.beta < 0.8
                 else "moves roughly in line with" if row.beta <= 1.2 else "amplifies")
        reading.findings.append(
            f"**Versus {bench}:** beta {row.beta:.2f} ({style} the benchmark), tracking error {pct(row.tracking_error)} and "
            f"information ratio {row.information_ratio:.2f}"
            + (" (consistent outperformance)." if row.information_ratio > 0.5 else " (little consistent excess return)."
               if row.information_ratio > -0.5 else " (consistent underperformance)."))
        reading.findings.append(f"**Capture:** {pct(row.up_capture, 0)} of the benchmark's up periods and "
                                f"{pct(row.down_capture, 0)} of its down periods.")
        reading.implications.append(
            f"Historically, a 10% fall in {bench} has come with roughly a {pct(0.10 * row.down_capture, 1)} fall here "
            f"(down capture), and a 10% rise with about {pct(0.10 * row.up_capture, 1)}"
            + (": an asymmetric, favourable profile." if row.up_capture > row.down_capture else ": losses have been captured more than gains."))
        reading.terms += ["Beta", "Up/down capture", "Tracking error", "Information ratio"]
    if len(table) > 1:
        ranked = table.sharpe.sort_values(ascending=False)
        reading.implications.append("Ranked by Sharpe: " + ", ".join(f"{t} ({v:.2f})" for t, v in ranked.items()) + ". "
                                    f"Shallowest worst drawdown: **{table.max_drawdown.idxmax()}** ({pct(table.max_drawdown.max())}).")
    return reading


def regression(coefficients: pd.DataFrame, diagnostics: pd.DataFrame, trend: pd.Series, y: str, returns: bool,
               freq: str, price: pd.Series, fitted: pd.Series) -> Reading:
    diag, ppy = diagnostics.set_index("metric")["value"], PERIODS[freq]
    slopes = coefficients.drop(index="const", errors="ignore")
    key = slopes.t.abs().idxmax()
    reading = Reading(
        f"The factors explain {pct(diag['R²'], 0)} of {owner(y)} movements; **{key}** is the most important "
        f"(t = {slopes.t[key]:.1f}).",
        findings=[],
        implications=[],
        caveats=["Regression shows association, not cause; coefficients can shift across market regimes.", PAST],
        terms=["R²", "Alpha", "Beta", "Confidence interval", "Durbin–Watson", "Breusch–Pagan"],
    )
    for name, row in slopes.iterrows():
        link = (f"each 1% move in {name} has come with {row.coef:+.2f}% in {y} (95% range {row.ci_low:+.2f} to {row.ci_high:+.2f})"
                if returns else f"each unit of {name} has come with {row.coef:+.3g} units of {y}")
        reading.findings.append(f"**{name}:** {link}; {evidence(row.p_value)}.")
    if returns and "const" in coefficients.index:
        const = coefficients.loc["const"]
        reading.findings.append(f"**Alpha:** {pct(const.coef * ppy, 1, True)} a year beyond what the factors explain; "
                                f"{evidence(const.p_value)} that it is real.")
    if returns:
        systematic, idio = diag["R²"], 1 - diag["R²"]
        reading.implications.append(
            f"{pct(systematic, 0)} of {y}'s variance is systematic (shared with the factors) and {pct(idio, 0)} is specific "
            f"to {y}, with residual volatility of about {pct(diag['Residual std'] * np.sqrt(ppy))} a year. "
            + ("Most of its risk is its own, so these factors are a weak hedge." if idio > 0.6
               else "Most of its risk comes from the factors, so they would hedge it effectively."))
    if diag["Durbin–Watson"] < 1.5 or diag["Durbin–Watson"] > 2.5:
        reading.caveats.insert(0, f"Durbin–Watson is {diag['Durbin–Watson']:.2f}, so residuals are autocorrelated and "
                                  "the p-values are overstated.")
    if significant(diag["Breusch–Pagan p-value"]):
        reading.caveats.insert(0, "Breusch–Pagan finds heteroskedasticity (error size varies), so standard errors and "
                                  "confidence ranges are less reliable than they look.")
    doubling = np.log(2) / np.log1p(trend.annual_growth) if trend.annual_growth > 0 else np.inf
    gap = price.dropna().iloc[-1] / fitted.iloc[-1] - 1
    reading.findings.append(
        f"**Long-run trend:** {pct(trend.annual_growth, 1, True)} a year"
        + (f" (doubling roughly every {doubling:.0f} years)" if np.isfinite(doubling) else "")
        + f", R² {trend.r2:.2f} ({'a steady path' if trend.r2 > 0.8 else 'a bumpy path'}). The latest price is "
        f"{pct(abs(gap), 0)} {'above' if gap > 0 else 'below'} that trend line.")
    return reading


def timeseries(tests: pd.DataFrame, hurst: float, coint: pd.Series, granger: pd.DataFrame, focus: str, other: str,
               freq: str, ac: pd.DataFrame, spread_z: float) -> Reading:
    period = PERIOD[freq]
    p = {(s, t): v for s, t, v in zip(tests.series, tests.test.str.split(" ").str[0], tests.p_value)}
    price_random = not significant(p[("log price", "ADF")]) and significant(p[("log price", "KPSS")])
    returns_stationary = significant(p[("returns", "ADF")])
    lags = ac[(ac.lag > 0) & (ac.acf.abs() > ac.bound)]
    lag1 = ac.acf[ac.lag == 1].iloc[0]
    vr = tests[tests.test.str.startswith("Variance")].iloc[0]
    reading = Reading(
        f"{focus} " + ("behaves like a random walk" if price_random else "shows some structure beyond a pure random walk")
        + f"; Hurst {hurst:.2f} and a variance ratio of {vr.statistic:.2f} "
        + ("lean towards momentum." if hurst > 0.55 and vr.statistic > 1 else "lean towards mean reversion."
           if hurst < 0.45 and vr.statistic < 1 else "show no consistent persistence either way."),
        findings=[
            f"**Stationarity:** on log prices, ADF p = {p[('log price', 'ADF')]:.2g} and KPSS p = {p[('log price', 'KPSS')]:.2g}; "
            + ("both agree the price has no fixed level to return to." if price_random else "the tests do not cleanly agree on a random walk.")
            + f" Returns are {'stationary' if returns_stationary else 'not clearly stationary'} (ADF p = {p[('returns', 'ADF')]:.2g}), "
            "which is what most statistics on this page assume.",
            f"**Autocorrelation:** lag-1 autocorrelation is {lag1:+.2f}"
            + (f"; {len(lags)} of 20 lags {'exceeds' if len(lags) == 1 else 'exceed'} the 95% band "
               f"(lag {', '.join(map(str, lags.lag.head(6)))}; about 1 would be expected by chance)." if len(lags)
               else "; no lag exceeds the 95% significance band.")
            + f" Ljung–Box finds {evidence(p[('returns', 'Ljung–Box')])} of autocorrelation overall.",
            f"**Variance ratio:** {vr.statistic:.2f} "
            + ("(above 1: multi-period moves have been larger than single moves imply, a momentum signature)"
               if vr.statistic > 1 else "(below 1: moves have partly reversed over several periods)")
            + f"; {evidence(vr.p_value)}.",
            f"**Hurst exponent:** {hurst:.2f} "
            + ("(trending: moves tend to persist)." if hurst > 0.55 else "(mean-reverting: moves tend to reverse)."
               if hurst < 0.45 else "(close to 0.5: little long memory)."),
        ],
        implications=[],
        caveats=["These tests have low power on short samples and assume one regime; a structural break can mimic "
                 "trending or mean reversion.", PAST],
        terms=["ADF", "KPSS", "Ljung–Box", "Variance ratio", "Hurst exponent", "Cointegration", "Half-life",
               "Granger causality"],
    )
    if lag1 > 0.05:
        reading.implications.append(f"Positive short-term autocorrelation means a strong {period} has tended to be "
                                    f"followed by another in the same direction.")
    elif lag1 < -0.05:
        reading.implications.append(f"Negative lag-1 autocorrelation means moves have tended to partly reverse the next "
                                    f"{period} (common in less liquid markets).")
    else:
        reading.implications.append(f"With near-zero autocorrelation, one {period}'s return has said almost nothing about the next.")
    if significant(coint.p_value):
        reading.findings.append(
            f"**Pair ({focus} & {other}):** cointegrated, {evidence(coint.p_value)}. A hedge ratio of {coint.hedge_ratio:.2f} "
            f"links their log prices, and gaps have closed with a half-life of about {coint.half_life:.0f} {period}s.")
        reading.implications.append(
            f"The spread is currently {spread_z:+.1f} standard deviations from its 60-{period} average; historically such "
            f"gaps have halved in roughly {coint.half_life:.0f} {period}s.")
    else:
        reading.findings.append(f"**Pair ({focus} & {other}):** not cointegrated (p = {coint.p_value:.2f}), so there is no "
                                f"evidence of a stable long-run link; the spread (now {spread_z:+.1f}σ) need not revert.")
    for direction, rows in granger.groupby("direction"):
        best = rows.loc[rows.p_value.idxmin()]
        reading.findings.append(f"**Granger, {direction}:** " + (
            f"past values improve forecasts at lag {int(best.lag)}, {evidence(best.p_value)}."
            if significant(best.p_value) else f"no forecasting value at lags 1–{int(rows.lag.max())} (best p = {best.p_value:.2f})."))
    return reading


def seasonality(parts: list[tuple[str, pd.DataFrame, pd.DataFrame]], monthly: pd.DataFrame, focus: str) -> Reading:
    years = len(monthly)
    reading = Reading("", caveats=[], terms=["ANOVA / Kruskal–Wallis", "p-value"])
    label = {"month": lambda m: pd.Timestamp(2000, int(m), 1).strftime("%B"), "weekday": str}
    for by, summary, tests in parts:
        best, worst = summary["mean"].idxmax(), summary["mean"].idxmin()
        p = tests.set_index("test").p_value
        spread = summary["mean"][best] - summary["mean"][worst]
        noise = summary["std"].mean()
        reading.findings.append(
            f"**By {by}:** best **{label[by](best)}** ({pct(summary['mean'][best], 2, True)} on average, positive "
            f"{pct(summary.pct_positive[best], 0)} of the time); worst **{label[by](worst)}** "
            f"({pct(summary['mean'][worst], 2, True)}, positive {pct(summary.pct_positive[worst], 0)}). "
            f"{int((summary['mean'] > 0).sum())} of {len(summary)} {by}s averaged a gain.")
        reading.findings.append(
            f"**Is it real?** The best–worst gap of {pct(spread, 2)} compares with a typical {by}-to-{by} swing of "
            f"{pct(noise, 2)} (only {spread / noise:.2f}× the noise). ANOVA p = {p['ANOVA']:.2f}, Kruskal–Wallis "
            f"p = {p['Kruskal–Wallis']:.2f}: " + ("the differences are statistically significant."
                                                 if significant(p["Kruskal–Wallis"]) else "the differences are consistent with chance."))
        if not reading.headline:
            reading.headline = (f"{focus} shows " + ("a statistically significant" if significant(p["Kruskal–Wallis"]) else "no reliable")
                                + f" {by}ly pattern; {label[by](best)} has been strongest and {label[by](worst)} weakest.")
    if parts and parts[0][0] == "month":
        best_month = parts[0][1]["mean"].idxmax()
        column = monthly.get(best_month)
        if column is not None:
            hits = int((column.dropna() > 0).sum())
            reading.implications.append(f"{label['month'](best_month)} was positive in {hits} of {column.notna().sum()} years; "
                                        "consistency across years matters more than one big average.")
    reading.implications.append("With 12 months (or 5 weekdays) compared at once, one will usually look special by chance, "
                                "so a pattern needs a clear economic reason and statistical support before it means much.")
    reading.caveats += ([f"Only {years} years of data, so each month's average rests on just {years} observations."]
                        if years < 15 else []) + [PAST]
    return reading


def correlation(corr: pd.DataFrame, pvals: pd.DataFrame, pairs: pd.DataFrame, explained: pd.DataFrame, method: str,
                returns: pd.DataFrame) -> Reading:
    n = len(corr)
    values = pairs["corr"]
    shares = explained.explained.to_numpy()
    bets = 1 / np.sum(shares ** 2)
    upper = pvals.where(np.triu(np.ones(pvals.shape, dtype=bool), k=1)).stack().dropna()
    portfolio_vol = returns.mean(axis=1).std()
    member_vol = returns.std().mean()
    half = len(returns) // 2
    early, late = (pairs_mean(returns.iloc[s].corr(method="pearson")) for s in (slice(None, half), slice(half, None)))
    top, bottom = pairs.iloc[0], pairs.iloc[-1]
    reading = Reading(
        f"The {n} series have an average correlation of {values.mean():.2f} and behave like about {bets:.1f} "
        f"independent bets.",
        findings=[
            f"**Spread of correlations:** {pct((values > 0.7).mean(), 0)} of the {len(values)} pairs are above 0.7, "
            f"{pct((values.abs() < 0.3).mean(), 0)} are between −0.3 and 0.3, and {pct((values < 0).mean(), 0)} are negative.",
            f"**Strongest pair:** {top.a} & {top.b} ({top['corr']:.2f}, {corr_strength(top['corr'])}). **Weakest:** "
            f"{bottom.a} & {bottom.b} ({bottom['corr']:.2f}, {corr_strength(bottom['corr'])}).",
            f"**Significance:** {int(upper.lt(0.05).sum())} of {len(upper)} pairs are significant at 5% on "
            f"{len(returns)} observations.",
            f"**Common factor:** the first principal component explains {pct(shares[0], 0)} of total variance"
            + (f" and the first two {pct(shares[:2].sum(), 0)}." if n > 2 else "."),
        ],
        implications=[
            f"An equal-weight mix of these series would have had about {pct(portfolio_vol / member_vol, 0)} of the "
            f"volatility of the average member: {'a large' if portfolio_vol / member_vol < 0.7 else 'a modest'} "
            "diversification benefit.",
            ("One common factor dominates, so holding more of these adds little diversification."
             if shares[0] > 0.6 else "No single factor dominates, so the series genuinely diversify one another."),
        ],
        caveats=["Correlations tend to rise in sell-offs, exactly when diversification is needed most.", PAST],
        terms=["Correlation", "PCA", "Effective number of bets", "p-value"],
    )
    partners = {c: corr[c].drop(c).idxmax() for c in corr.columns[:8]}
    reading.findings.append("**Closest partner:** " + "; ".join(f"{c} → {p} ({corr.loc[c, p]:.2f})" for c, p in partners.items()) + ".")
    if abs(late - early) > 0.1:
        reading.implications.append(f"Correlations have {'risen' if late > early else 'fallen'} from {early:.2f} in the "
                                    f"first half of the period to {late:.2f} in the second, so the relationships are not stable.")
    if bottom["corr"] < 0:
        reading.implications.append(f"{bottom.a} and {bottom.b} have tended to move in opposite directions, the most "
                                    "effective offsetting pair here.")
    reading.caveats.insert(0, {
        "spearman": "Spearman uses ranks, so a few extreme days influence it less than Pearson.",
        "kendall": "Kendall's tau is rank-based and naturally smaller in magnitude than Pearson; compare like with like.",
        "partial": "Partial correlations strip out the influence of every other selected series; they can differ sharply from plain correlations.",
        "covariance": "Covariance mixes correlation with volatility, so large values may just reflect volatile series.",
    }.get(method, "Pearson captures linear co-movement and is sensitive to extreme days."))
    return reading


def pairs_mean(corr: pd.DataFrame) -> float:
    return float(corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack().dropna().mean())


def compare(table: pd.DataFrame) -> Reading:
    ranked = table.sort_values("total_return", ascending=False)
    efficiency = (table.ann_return / table.ann_vol).sort_values(ascending=False)
    unrecovered = table.index[table.recovery.isna()].tolist()
    best, worst = ranked.index[0], ranked.index[-1]
    reading = Reading(
        f"**{best}** returned the most ({pct(table.total_return[best], 0, True)}) and **{worst}** the least "
        f"({pct(table.total_return[worst], 0, True)}); on a risk-adjusted basis **{table.sharpe.idxmax()}** led.",
        findings=[
            f"**{money(STAKE)} grew to:** " + ", ".join(f"{t} {money(STAKE * (1 + v))}" for t, v in ranked.total_return.items()) + ".",
            "**Return per unit of volatility:** " + ", ".join(f"{t} {v:.2f}" for t, v in efficiency.items()) + ".",
            f"**Risk:** most volatile **{table.ann_vol.idxmax()}** ({pct(table.ann_vol.max())} a year), calmest "
            f"**{table.ann_vol.idxmin()}** ({pct(table.ann_vol.min())}). Deepest drawdown **{table.max_drawdown.idxmin()}** "
            f"({pct(table.max_drawdown.min())}), shallowest **{table.max_drawdown.idxmax()}** ({pct(table.max_drawdown.max())}).",
        ],
        implications=[f"The gap between best and worst is {pct(table.total_return.max() - table.total_return.min(), 0)}: "
                      + ("choice of asset mattered enormously over this period." if table.total_return.max() - table.total_return.min() > 0.5
                         else "outcomes were relatively bunched.")],
        caveats=["Total returns depend heavily on start and end dates; try other date ranges to see how robust the ranking is.",
                 "Instruments priced in different currencies are compared in native terms unless a currency is chosen.", PAST],
        terms=["Sharpe ratio", "Max drawdown", "CAGR"],
    )
    if table.sharpe.idxmax() != best:
        reading.implications.append(f"{best}'s higher return came with more risk; per unit of volatility "
                                    f"{table.sharpe.idxmax()} was more efficient.")
    if unrecovered:
        reading.implications.append(f"Still below their worst-drawdown peak: {', '.join(unrecovered)}.")
    return reading


def asset_classes(bench: pd.DataFrame, table: pd.DataFrame, horizon: str, metric: str) -> Reading | None:
    ranked = table.dropna(subset=[metric]).sort_values(metric, ascending=False)
    if ranked.empty:
        return None
    values = ranked[metric]
    q1, q3 = values.quantile([0.25, 0.75])
    consistent = ranked[(ranked.ret_1m > 0) & (ranked.ret_3m > 0) & (ranked.ret_1y > 0)]
    reversals = ranked[(ranked.ret_1y > 0.15) & (ranked.ret_1m < -0.03)]
    efficient = (ranked.ret_1y / ranked.vol_1y).dropna().sort_values(ascending=False)
    bear = ranked[ranked.pct_from_52w_high < -0.2]
    reading = Reading(
        f"{pct(values.gt(0).mean(), 0)} of {len(ranked)} instruments are up over {horizon} (median "
        f"{pct(values.median(), 1, True)}), led by **{ranked['name'].iloc[0]}** and trailed by **{ranked['name'].iloc[-1]}**.",
        findings=[
            "**Leaders:** " + ", ".join(f"{r['name']} {pct(r[metric], 1, True)}" for _, r in ranked.head(3).iterrows()) + ".",
            "**Laggards:** " + ", ".join(f"{r['name']} {pct(r[metric], 1, True)}" for _, r in ranked.tail(3).iloc[::-1].iterrows()) + ".",
            f"**Dispersion:** the middle half of returns spans {pct(q1, 1, True)} to {pct(q3, 1, True)}; best-to-worst "
            f"spread {pct(values.max() - values.min())}.",
        ],
        implications=[],
        caveats=["Futures prices roll between contracts and yields are rates, not prices, so their 'returns' differ from "
                 "what an investor would earn.", PAST],
        terms=["Standard deviation", "Sharpe ratio"],
    )
    headline = bench[bench.region == "Global"].dropna(subset=[metric]).sort_values(metric, ascending=False)
    if len(headline):
        reading.findings.append("**Global benchmarks:** " + ", ".join(
            f"{r.asset_class} ({benchmarks.describe(r)}) {pct(r[metric], 1, True)}" for _, r in headline.iterrows()) + ".")
    for region in ("United States", "Australia"):
        local = bench[bench.region == region].dropna(subset=[metric])
        if len(local):
            reading.findings.append(f"**{region}:** " + ", ".join(
                f"{benchmarks.describe(r)} {pct(r[metric], 1, True)}" for _, r in local.iterrows()) + ".")
    reading.implications.append(
        f"**Broad momentum:** {len(consistent)} instruments are up over 1 month, 3 months and 1 year"
        + (f", e.g. {', '.join(consistent['name'].head(4))}." if len(consistent) else "."))
    if len(reversals):
        reading.implications.append(f"**Short-term pullbacks in long-term winners** (1Y > +15%, 1M < −3%): "
                                    f"{', '.join(reversals['name'].head(4))}.")
    if len(efficient):
        reading.implications.append("**Best 1-year return per unit of volatility:** "
                                    f"{', '.join(ranked.loc[efficient.index[:3], 'name'])}.")
    if len(bear):
        reading.implications.append(f"**More than 20% below their 52-week high** (often called bear-market territory): "
                                    f"{', '.join(bear['name'].head(5))}{'…' if len(bear) > 5 else ''}.")
    return reading


def objective(table: pd.DataFrame, subject: str, margin: float, years: int, region: str) -> Reading | None:
    """Reading for a CPI + margin objective measured over rolling windows."""
    if len(table) < 2:
        return None
    from market.inflation import HUBS, met_rate
    latest, hit, where = table.iloc[-1], met_rate(table), HUBS[region][0]
    run = (table.excess > 0).astype(int)
    streak = int(run.iloc[::-1].cumsum().eq(range(1, len(run) + 1)).sum()) if latest.excess > 0 else 0
    verdict = "clears" if latest.excess > 0 else "falls short of"
    reading = Reading(
        f"{subject} {verdict} CPI + {pct(margin, 1)} over the {years} years to {table.index[-1]:%b %Y}, returning "
        f"{pct(latest.achieved)} a year against a {pct(latest.target)} target.",
        findings=[
            f"**The target moves with prices:** {where} inflation ran {pct(latest.inflation)} a year over this "
            f"window, so the objective was {pct(latest.target)}, not a fixed number. A fund meets it by beating "
            "inflation, which is harder in the windows where inflation itself was high.",
            f"**Consistency:** the objective was met in {pct(hit, 0)} of {len(table)} rolling {years}-year windows "
            f"since {table.index[0]:%b %Y}"
            + (f", the most recent {streak} of them consecutively." if streak > 1 else "."),
            f"**Best and worst windows:** {pct(table.excess.max(), 1, True)} above target at its best and "
            f"{pct(table.excess.min(), 1, True)} at its worst.",
        ],
        implications=[
            f"An objective is a long-horizon promise: a {years}-year window smooths everything shorter, so a bad "
            "year shows up slowly and leaves slowly."
            if hit > 0.5 else
            f"Missing in {pct(1 - hit, 0)} of windows means the shortfall is the normal case here, not bad luck."],
        caveats=["Returns here are gross. Published objectives are after fees and tax, so a fund charging 0.6% a "
                 "year needs to beat this by that much to report the same result.",
                 f"Windows overlap, so {len(table)} readings are far fewer than {len(table)} independent tests.",
                 PAST],
        terms=["CAGR"],
    )
    return reading


def risk_table(table: pd.DataFrame, group: str, period: str) -> Reading | None:
    """Reading for the cross-sectional risk table of many instruments."""
    if len(table) < 2:
        return None
    name, vol = table["name"], table.vol_ann
    tail_ratio = (table.cvar_95 / table.var_95).sort_values(ascending=False)
    rank_corr = vol.rank().corr(table.max_drawdown.rank(ascending=False))
    reading = Reading(
        f"Across {len(table)} instruments in {group}, annualised volatility ranges from {pct(vol.min())} "
        f"({name[vol.idxmin()]}) to {pct(vol.max())} ({name[vol.idxmax()]}), with a median of {pct(vol.median())}.",
        findings=[
            f"**Value at Risk:** the median 95% VaR is {pct(table.var_95.median(), 2, True)} per {period}. The most exposed "
            f"is {name[table.var_95.idxmin()]}, where 1 {period} in 20 lost {pct(-table.var_95.min(), 2)} or more.",
            f"**Worst single {period}:** {name[table.worst.idxmin()]} at {pct(table.worst.min(), 1, True)}.",
            f"**Drawdowns:** median maximum drawdown {pct(table.max_drawdown.median())}; deepest "
            f"{name[table.max_drawdown.idxmin()]} ({pct(table.max_drawdown.min())}), shallowest "
            f"{name[table.max_drawdown.idxmax()]} ({pct(table.max_drawdown.max())}).",
            f"**Risk-adjusted:** best Sharpe {name[table.sharpe.idxmax()]} ({table.sharpe.max():.2f}), worst "
            f"{name[table.sharpe.idxmin()]} ({table.sharpe.min():.2f}); {pct((table.sharpe > 0).mean(), 0)} beat cash.",
        ],
        implications=[
            f"Higher volatility has {'closely ' if rank_corr > 0.7 else 'loosely ' if rank_corr > 0.3 else 'barely '}"
            f"gone with deeper drawdowns (rank correlation {rank_corr:.2f}), so volatility alone "
            + ("is a fair guide to how much you could lose." if rank_corr > 0.7 else "does not tell the whole story."),
            "Heaviest tails relative to their VaR (average loss beyond VaR ÷ VaR): "
            + ", ".join(f"{name[t]} {v:.2f}×" for t, v in tail_ratio.head(3).items())
            + ". Above about 1.5× means losses beyond the VaR line have tended to be much larger than the line itself.",
        ],
        caveats=[PAST],
        terms=["Standard deviation", "VaR", "CVaR", "Max drawdown", "Sharpe ratio", "Sortino ratio"],
    )
    if table.observations.min() < 0.8 * table.observations.max():
        reading.caveats.insert(0, "Some instruments have much shorter histories in this range (see Obs), which makes "
                                  "their figures less comparable.")
    if group in ("Rates", "Currencies"):
        reading.caveats.insert(0, "For yields and exchange rates these are changes in the rate itself, not investment returns.")
    return reading


def duration(days: float) -> str:
    return "n/a" if pd.isna(days) else f"{days:.0f} days" if days < 60 else f"{days / 30.4:.0f} months" \
        if days < 730 else f"{days / 365.25:.1f} years"


def stress_period(period, table: pd.DataFrame, corr_before: float, corr_during: float, basket: pd.Series | None,
                  amount: float, horizon: str) -> Reading | None:
    """Reading for how a set of series behaved through a stress period."""
    if table.empty:
        return None
    moves = table.event_return
    worst, best = moves.idxmin(), moves.idxmax()
    deepest = table.trough_return.idxmin()
    recovered, pending = table[table.recovery_date.notna()], table[table.recovery_date.isna()]
    headline = (f"Through the {period.name} ({period.span}), **{worst}** moved {pct(moves.iloc[0], 1, True)}."
                if len(table) == 1 else
                f"Through the {period.name} ({period.span}), **{worst}** fell the most ({pct(moves.min(), 1, True)}) "
                f"and **{best}** held up best ({pct(moves.max(), 1, True)}).")
    reading = Reading(
        headline,
        findings=[
            f"**Typical move:** median {pct(moves.median(), 1, True)} across {len(table)} series; "
            f"{pct((moves < 0).mean(), 0)} ended the window lower.",
            f"**Deepest point:** {deepest} was down {pct(-table.trough_return.min())} at its low on "
            f"{pd.Timestamp(table.trough_date[deepest]):%d %b %Y}.",
            f"**Volatility:** median annualised volatility went from {pct(table.vol_before.median())} in the year before "
            f"to {pct(table.vol_during.median())} during the stress period ({table.vol_ratio.median():.1f}×).",
            f"**Worst single day:** {table.worst_day.idxmin()} at {pct(table.worst_day.min(), 1, True)}.",
        ],
        implications=[],
        caveats=["Stress-period windows are defined on the S&P 500's peak and trough; other markets turned on different days.",
                 "Series without price history before the period start are excluded.",
                 "Every crisis has different causes, so past behaviour is a stress test, not a forecast.", PAST],
        terms=["Max drawdown", "Standard deviation", "Correlation"],
    )
    if len(recovered):
        slowest = recovered.days_to_recover.idxmax()
        reading.findings.append(f"**Recovery:** {len(recovered)} of {len(table)} regained their pre-event level; the "
                                f"median took {duration(recovered.days_to_recover.median())} from the event start "
                                f"and the slowest, {slowest}, {duration(recovered.days_to_recover.max())}.")
    if len(pending):
        reading.findings.append(f"**Not yet recovered:** {', '.join(pending.index[:6])}"
                                f"{'…' if len(pending) > 6 else ''} remain below their pre-event level.")
    gainers = moves[moves > 0]
    reading.implications.append(
        f"{', '.join(gainers.index[:5])} rose during the stress period, the kind of holdings that cushioned losses elsewhere."
        if len(gainers) else "Nothing in this selection rose during the stress period, so it offered no true shelter.")
    if pd.notna(corr_before) and pd.notna(corr_during):
        shift = corr_during - corr_before
        reading.implications.append(
            f"Average correlation moved from {corr_before:.2f} in the year before to {corr_during:.2f} during the stress period"
            + (": assets moved together far more, so diversification weakened just when it was needed."
               if shift > 0.15 else ": diversification broadly held up." if shift > -0.15
               else ": assets moved more independently than usual."))
    if basket is not None and len(basket):
        at_end = basket[:pd.Timestamp(period.end)].iloc[-1]
        reading.implications.append(
            f"**Stress test:** a {money(amount)} buy-and-hold basket would have been worth {money(amount * at_end)} "
            f"at the end of the period ({pct(at_end - 1, 1, True)}), {money(amount * basket.min())} at its lowest "
            f"({pct(basket.min() - 1, 1, True)}) and {money(amount * basket.iloc[-1])} {horizon} later "
            f"({pct(basket.iloc[-1] - 1, 1, True)}). If a similar stress period repeated, that is the kind of swing to be "
            "prepared for.")
    return reading


def portfolio(positions: pd.DataFrame, contribution: pd.DataFrame, risk_table: pd.DataFrame, risk: pd.Series,
              benchmark: str, value: pd.Series, cash: float) -> Reading:
    """Reading for a portfolio: concentration, what drove returns, where the risk sits, and stress behaviour."""
    top, bottom = contribution.iloc[0], contribution.iloc[-1]
    biggest = positions.weight.idxmax()
    cash_weight = cash / value.iloc[-1] if value.iloc[-1] else 0
    period_return = value.iloc[-1] / value.iloc[0] - 1
    reading = Reading(
        f"The portfolio is worth {money(value.iloc[-1])} across {len(positions)} holdings and returned "
        f"{pct(period_return, 1, True)} over this range, with {pct(risk.ann_vol)} volatility.",
        findings=[
            f"**Concentration:** the largest holding is {biggest} at {pct(positions.weight.max(), 0)}; the top three "
            f"are {pct(positions.weight.nlargest(3).sum(), 0)} of the portfolio"
            + (f", and cash is {pct(cash_weight, 0)}." if cash_weight > 0.005 else "."),
            f"**What drove the return:** {top.name} added {pct(top.contribution, 1, True)} and {bottom.name} "
            f"{pct(bottom.contribution, 1, True)}. Contributions combine each holding's return with how much of the "
            "portfolio it was.",
            f"**Unrealised profit:** {money(positions.profit.sum())} against a cost base of "
            f"{money(positions.cost.sum())} ({pct(positions.profit.sum() / positions.cost.sum(), 1, True)}), which is "
            "a different figure from the period return because it runs from when you bought.",
        ],
        implications=[],
        caveats=["Returns here follow market value only: contributions, withdrawals, dividends, brokerage and tax are "
                 "not included, so this is not a true time-weighted or money-weighted return.",
                 "Holdings priced in different currencies are added together as they are unless a currency is chosen "
                 "in the sidebar.", PAST],
        terms=["Standard deviation", "Sharpe ratio", "Beta", "Tracking error", "Information ratio", "Max drawdown",
               "VaR", "Correlation"],
    )
    if len(risk_table):
        riskiest = risk_table.index[0]
        gap = risk_table.share_of_risk[riskiest] - risk_table.weight[riskiest]
        reading.findings.append(
            f"**Where the risk sits:** {riskiest} is {pct(risk_table.weight[riskiest], 0)} of the money but "
            f"{pct(risk_table.share_of_risk[riskiest], 0)} of the volatility"
            + (", so it carries more risk than its size suggests." if gap > 0.05 else "."))
    if pd.notna(risk.get("beta")):
        reading.implications.append(
            f"Against {benchmark}, beta is {risk.beta:.2f} and tracking error {pct(risk.tracking_error)}, giving an "
            f"information ratio of {risk.information_ratio:.2f}: "
            + ("consistent value added versus simply holding the benchmark." if risk.information_ratio > 0.5
               else "little consistent difference from the benchmark after allowing for the extra risk."
               if risk.information_ratio > -0.5 else "consistent underperformance against the benchmark."))
    reading.implications.append(
        f"The worst fall from a peak in this range was {pct(risk.max_drawdown)}"
        + (", and it has since recovered." if pd.notna(risk.recovery) else ", and it has not yet recovered."))
    return reading


def portfolio_stress(stress_table: pd.DataFrame, value: float) -> list[str]:
    """Extra lines about how today's portfolio would have behaved in past stress periods."""
    if stress_table.empty:
        return []
    worst = stress_table.iloc[0]
    partial = stress_table[stress_table.covered < 0.999]
    lines = [f"**Stress history:** applying today's weights to past crises, the worst was {worst.name} "
             f"({pct(worst['return'], 1, True)}, low {pct(worst.worst, 1, True)}), which on {money(value)} today would "
             f"be about {money(value * (1 + worst.worst))} at the bottom.",
             f"The median stress period cost {pct(stress_table['return'].median(), 1, True)}, and "
             f"{pct((stress_table['return'] > 0).mean(), 0)} of them ended positive."]
    if len(partial):
        lines.append(f"{len(partial)} periods only had data for part of the portfolio, so those figures cover the "
                     "holdings that existed at the time.")
    return lines
