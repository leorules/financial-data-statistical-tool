import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from plotly.subplots import make_subplots

from market import factors, interpret, stats, ui
from market import returns as rets
from market import resample
from market.config import benchmark_for

s = ui.settings()
inst = ui.instruments()
ui.require(inst)
SECTIONS = {"Descriptive": ":material/table_chart:", "Rolling": ":material/show_chart:",
            "Distribution": ":material/bar_chart:", "Hypothesis": ":material/science:", "Risk": ":material/shield:",
            "Regression": ":material/trending_up:", "Factors": ":material/scatter_plot:",
            "Time Series": ":material/timeline:",
            "Seasonality": ":material/calendar_month:"}
SEASON_PERIOD = {"D": 21, "W": 52, "M": 12}
NO_AXIS = {"value": "", "date": "", "index": ""}

ui.header("Statistics", "Descriptive statistics, distributions, significance tests, risk, regression and time-series "
                        "analysis for tickers or whole asset classes.")
ui.adjustments_caption(s, "total_return", "live_cash", "hac")
with st.container(border=True):
    tickers, groups = ui.subject_picker()
    c1, c2 = st.columns([1, 2], vertical_alignment="bottom")
    series = c1.segmented_control("Series", ["Returns", "Price"] + ([] if groups else ["Volume"]), default="Returns",
                                  required=True)
    s, period = ui.stress_window("statistics_window", s, c2)
    ui.stress_caption(period)
    section = st.segmented_control("Analysis", list(SECTIONS), default="Descriptive", required=True,
                                   format_func=lambda x: f"{SECTIONS[x]} {x}")
ui.require(tickers, "Pick at least one ticker or asset class.")

data = ui.series_matrix(tickers, s, series, groups)
ui.require(data.columns, f"No overlapping data for these series in the {period.name} window." if period else
           "No data in the selected range.")
if len(data) < 20:
    st.warning(f"Only {len(data)} observations in this window. Tests and rolling statistics need more data than this, "
               "so treat anything below as indicative.", icon=":material/warning:")
returns = data if series == "Returns" else ui.return_matrix(tickers, s, groups=groups)
ui.close_time_caption(returns.columns, s, groups)
prices = ui.series_matrix(tickers, s, "Price", groups)
cols = list(data.columns)
loaded = list(inst.ticker)

c1, c2, c3 = st.columns(3)
focus = c1.selectbox("Focus", cols, format_func=ui.label)
other = c2.selectbox("Compare with", [c for c in cols if c != focus] or cols, format_func=ui.label)
default_bench = benchmark_for(focus, s.on("total_return"))
bench_options = list(dict.fromkeys([default_bench, *loaded])) if default_bench in loaded else loaded
bench = c3.selectbox("Benchmark", bench_options, format_func=ui.label)
bench_r = ui.return_matrix([bench], s, align="ffill").get(bench)
x = data[focus].dropna()
reading = None


def card(title: str | None = None):
    box = st.container(border=True)
    if title:
        box.markdown(f"**{title}**")
    return box


def line(df, title="", height=320, container=None, **kw):
    ui.chart(px.line(df, title=title, labels=NO_AXIS, **kw), container, height)


if section == "Descriptive":
    table = stats.descriptive.summary(data, returns=series == "Returns", log=s.kind == "log")
    with card("Summary statistics"):
        st.dataframe(table.T, height=min(900, 36 * (len(table.columns) + 1)))
        ui.download(table, "descriptive")
    with card():
        ui.chart(px.box(data.melt(var_name="ticker"), x="ticker", y="value", points="outliers",
                        title=f"{series} distribution", labels={"value": "", "ticker": ""}), height=420)
    outliers = stats.descriptive.outliers(x)
    with st.expander(f"Outliers in {focus}"):
        st.dataframe(outliers)
    reading = interpret.descriptive(table, focus, series == "Returns", s.freq, x)

elif section == "Rolling":
    longest = max(5, min(252, len(x) // 2))
    if longest < 10:
        st.info(f"Only {len(x)} observations in this window: too few for rolling statistics.")
    with card():
        c1, c2 = st.columns([1, 2])
        window = c1.slider("Window", 5, longest, min(63 if s.freq == "D" else 26, longest))
        moments = stats.rolling.moments(x, window)
        picked = c2.multiselect("Statistics", list(moments.columns), default=["mean", "std"])
        line(moments[picked], f"Rolling {window}-period statistics: {focus} ({series.lower()})")
        if series == "Price":
            line(stats.rolling.bollinger(x, window), f"Bollinger bands ({window}, 2σ)")
    rf = returns[focus]
    rolling = pd.DataFrame({"rolling vol": stats.rolling.moments(rf, window)["vol_ann"],
                            "EWMA vol (λ=0.94)": stats.rolling.ewma_vol(rf),
                            "rolling Sharpe": stats.rolling.sharpe(rf, window,
                                                                  rf=ui.risk_free(s, rf.index))})
    left, right = st.columns(2)
    with left.container(border=True):
        line(rolling[["rolling vol", "EWMA vol (λ=0.94)"]], "Annualised volatility")
        line(rolling["rolling Sharpe"], "Rolling Sharpe ratio")
    with right.container(border=True):
        if bench_r is not None:
            rolling[f"beta vs {bench}"] = stats.rolling.beta(rf, bench_r, window)
            line(rolling[f"beta vs {bench}"], f"Rolling beta vs {bench}")
        if other in returns and other != focus:
            rolling[f"corr with {other}"] = stats.rolling.corr(rf, returns[other], window)
            line(rolling[f"corr with {other}"], f"Rolling correlation with {other}")
    reading = interpret.rolling(rolling, window, s.freq, focus)

elif section == "Distribution":
    kde = stats.distribution.kde(x)
    var = stats.distribution.var(x)
    left, right = st.columns(2)
    with left.container(border=True):
        fig = px.histogram(x, nbins=80, histnorm="probability density", opacity=0.55, title=f"{focus} histogram",
                           labels=NO_AXIS)
        fig.add_scatter(x=kde.x, y=kde.kde, name="KDE", mode="lines")
        fig.add_scatter(x=kde.x, y=kde.normal, name="Normal", mode="lines", line_dash="dot")
        for _, row in var[var.method == "historical"].iterrows():
            fig.add_vline(x=row.VaR, line_color=ui.MUTED, annotation_text=f"VaR {row.level:.0%}")
        ui.chart(fig, height=420)
    with right.container(border=True):
        qq = stats.distribution.qq(x)
        fig = px.scatter(qq, x="theoretical", y="sample", opacity=0.45, title="Q–Q plot vs normal")
        fig.add_scatter(x=qq.theoretical, y=qq.fit, mode="lines", name="Normal fit")
        ui.chart(fig, height=420)
    normality, fit = stats.distribution.normality(x), stats.distribution.fit(x)
    with card("Normality tests"):
        ui.conclusions(normality)
    left, right = st.columns(2)
    with left.container(border=True):
        st.markdown("**Distribution fit** (lower AIC is better)")
        st.dataframe(fit, hide_index=True)
    with right.container(border=True):
        st.markdown("**Value at Risk** (return threshold per period)")
        st.dataframe(var, hide_index=True, column_config=ui.percent(var, ["level", "VaR", "CVaR"]))
    if series == "Returns":
        reading = interpret.distribution(stats.descriptive.describe(x), normality, fit, var, s.freq, x)

elif section == "Hypothesis":
    paired = other != focus
    mean_test = stats.hypothesis.mean_zero(x).to_frame().T
    comparison = stats.hypothesis.compare(x, data[other]) if paired else None
    with card(f"Is the mean of {focus} different from zero?"):
        ui.conclusions(mean_test)
    with card(f"{focus} vs {other}: same mean, variance and distribution?" if paired else "Comparing two series"):
        if paired:
            ui.conclusions(comparison)
        else:
            st.caption("Only one series is selected, so there is nothing to compare it against. Add a second ticker "
                       "or asset class above.")
    with card("Bootstrap confidence interval"):
        c1, c2, c3 = st.columns(3)
        stat = c1.selectbox("Statistic", ["mean", "Sharpe"] + ([f"correlation with {other}"] if paired else []))
        n = c2.select_slider("Resamples", [500, 1000, 2000, 5000], value=2000)
        level = c3.select_slider("Confidence", [0.9, 0.95, 0.99], value=0.95)
        rf = returns[focus]
        if stat == "mean":
            ci, samples = stats.hypothesis.bootstrap(x, stats.hypothesis.boot_mean, n, level)
        elif stat == "Sharpe":
            sharpe = stats.hypothesis.boot_sharpe(stats.periods_per_year(rf.index))
            ci, samples = stats.hypothesis.bootstrap(rf, sharpe, n, level)
        else:
            ci, samples = stats.hypothesis.bootstrap(data[[focus, other]], stats.hypothesis.boot_corr, n, level)
        st.dataframe(ci.to_frame().T, hide_index=True)
        fig = px.histogram(samples, nbins=60, title=f"Bootstrap distribution of {stat}", labels=NO_AXIS)
        for v in (ci.ci_low, ci.ci_high):
            fig.add_vline(x=v, line_color=ui.MUTED)
        ui.chart(fig, height=320)
    if series == "Returns" and paired:
        reading = interpret.hypothesis(mean_test.iloc[0], comparison, ci, stat, focus, other, s.freq, x, data[other])

elif section == "Risk":
    table = stats.risk.summary(returns, bench_r, rf=ui.risk_free(s, returns.index))
    pct_cols = ["ann_return", "ann_vol", "downside_dev", "max_drawdown", "ulcer_index", "tracking_error"]
    with card(f"Risk & performance vs {bench}" if bench_r is not None else "Risk & performance"):
        st.dataframe(table, column_config=ui.percent(table, pct_cols))
        ui.download(table, "risk")
    left, right = st.columns(2)
    with left.container(border=True):
        fig = px.line(stats.risk.drawdown(returns[focus]), title=f"Underwater chart: {focus}", labels=NO_AXIS)
        ui.chart(fig.update_traces(fill="tozeroy").update_yaxes(tickformat=".0%"), height=360)
    if bench_r is not None:
        with right.container(border=True):
            capture = table[["up_capture", "down_capture"]].reset_index(names="ticker").melt("ticker")
            fig = px.bar(capture, x="ticker", y="value", color="variable", barmode="group",
                         title=f"Up / down capture vs {bench}", labels={"value": "", "ticker": ""})
            ui.chart(fig.update_yaxes(tickformat=".0%"), height=360)
    cash_rate = ui.risk_free(s, returns.index)
    reading = interpret.risk(table, focus, bench if bench_r is not None else None, returns[focus],
                             float(cash_rate.mean()) if s.on("live_cash") else None)

elif section == "Regression":
    with card():
        c1, c2 = st.columns([1, 2])
        y_name = c1.selectbox("Dependent (y)", cols, index=cols.index(focus), format_func=ui.label)
        x_names = c2.multiselect("Independent (X)", list(dict.fromkeys([bench, *loaded])), default=[bench],
                                 format_func=ui.label)
    ui.require(x_names, "Pick at least one independent variable.")
    regressors = ui.series_matrix(x_names, s, series)
    involved = pd.concat([data[[y_name]], regressors], axis=1)
    drifting = stats.timeseries.non_stationary(involved.loc[:, ~involved.columns.duplicated()])
    if len(drifting) > 1:
        st.warning(f"{', '.join(drifting)} each fail the ADF test for stationarity. Regressing one trending series "
                   "on another finds a relationship where none exists — on independent random walks this reports a "
                   "significant slope about 90% of the time, and robust standard errors barely help. Use **Returns** "
                   "above, or the cointegration test in **Time Series**, to ask whether they really move together.",
                   icon=":material/warning:")
    res = stats.regression.fit(data[y_name], regressors, hac=s.on("hac"))
    coefficients, diagnostics = stats.regression.coefficients(res), stats.regression.diagnostics(res)
    left, right = st.columns(2)
    with left.container(border=True):
        st.markdown("**Coefficients**")
        st.dataframe(coefficients, column_config={c: ui.NUM for c in ("coef", "std_err", "t", "ci_low", "ci_high")})
    with right.container(border=True):
        st.markdown("**Diagnostics**")
        st.dataframe(diagnostics, hide_index=True, column_config={"value": ui.NUM})
    fitted = pd.DataFrame({"actual": res.model.endog, "fitted": res.fittedvalues, "residual": res.resid})
    left, right = st.columns(2)
    with left.container(border=True):
        ui.chart(px.scatter(fitted, x="fitted", y="actual", opacity=0.45, title="Actual vs fitted"), height=360)
    with right.container(border=True):
        fig = px.scatter(fitted, x="fitted", y="residual", opacity=0.45, title="Residuals")
        ui.chart(fig.add_hline(y=0, line_color=ui.MUTED), height=360)
    trend, line_fit = stats.regression.trend(prices[focus])
    with card(f"Log-price trend: {focus} grows {trend.annual_growth:+.2%} a year (R² {trend.r2:.2f})"):
        line(pd.DataFrame({"Price": prices[focus], "Trend": line_fit}), log_y=True)
    reading = interpret.regression(coefficients, diagnostics, trend, y_name, series == "Returns", s.freq,
                                   prices[focus], line_fit)

elif section == "Factors":
    rf = returns[focus]
    with card():
        picked = st.multiselect("Factors", list(factors.FACTORS), default=["Market (AU)", "Market (US)", "Duration"],
                                help="Each is a listed proxy; spreads isolate the factor from market direction")
    ui.require(picked, "Pick at least one factor.")
    factor_prices = ui.price_matrix(factors.legs(picked), s)
    if s.on("align_closes") and s.freq == "D":
        factor_prices = rets.align_closes(factor_prices, ui.regions())
    f = factors.returns(resample.last(factor_prices, s.freq), picked)
    missing = [n for n in picked if n not in f]
    if missing:
        st.caption(f"Not loaded, so skipped: {', '.join(missing)}. Refresh the built-in lists in Data Manager.")
    ui.require(f.columns, "None of the chosen factors have data. Refresh the built-in lists in Data Manager.")
    pair = pd.concat([rf.rename(focus), f], axis=1).dropna()
    if len(pair) < 30:
        st.info(f"Only {len(pair)} overlapping observations; a factor model needs more than this.")
    else:
        res = stats.regression.fit(pair[focus], pair[f.columns], hac=s.on("hac"))
        periods = stats.periods_per_year(pair.index)
        table = factors.exposures(res, periods)
        left, right = st.columns([3, 2])
        with left.container(border=True):
            st.markdown(f"**What drives {focus}**", help="Beta is the return per 1% factor move; alpha is what "
                                                        "none of the factors explain.")
            st.dataframe(table, column_config={c: ui.NUM for c in ("beta", "t", "p_value")})
            st.caption(f"R² {res.rsquared:.2f} — the share of {focus}'s variation these factors account for. "
                       f"Alpha {table.loc['Alpha (annual)', 'beta']:+.2%} a year "
                       f"(p = {table.loc['Alpha (annual)', 'p_value']:.3f}).")
        with right.container(border=True):
            betas = table.drop("Alpha (annual)")
            fig = px.bar(betas.sort_values("beta"), x="beta", y=betas.sort_values("beta").index,
                         orientation="h", labels={"beta": "", "y": ""}, text_auto=".2f")
            ui.chart(fig.update_layout(showlegend=False), height=max(240, 60 * len(betas)))
        with card("What each factor is"):
            st.dataframe(pd.DataFrame({"factor": list(f.columns),
                                       "built from": [factors.describe(n) for n in f.columns]}), hide_index=True)

elif section == "Time Series" and len(x) < 40:
    st.info(f"Only {len(x)} observations in this window. Stationarity, autocorrelation, cointegration and Granger "
            "tests need roughly 40 or more, so they are not shown here.")

elif section == "Time Series":
    level = np.log(prices[focus].dropna())
    rf = returns[focus]
    with card(f"Stationarity & randomness: {focus}"):
        tests = pd.concat([
            stats.timeseries.stationarity(level).assign(series="log price"),
            stats.timeseries.stationarity(rf).assign(series="returns"),
            pd.DataFrame([stats.timeseries.ljung_box(rf).to_dict() | {"series": "returns"},
                          stats.timeseries.variance_ratio(rf).to_dict() | {"series": "returns"}]),
        ])[["series", "test", "statistic", "p_value", "conclusion"]]
        ui.conclusions(tests)
        h = stats.timeseries.hurst(level)
        kind = "mean-reverting" if h < 0.45 else "trending" if h > 0.55 else "close to a random walk"
        st.metric("Hurst exponent (log price)", f"{h:.3f}", kind, delta_color="off", delta_arrow="off",
                  border=True, width=260, help="Below 0.5 mean-reverting · 0.5 random walk · above 0.5 trending")
    lags = min(20, len(x) // 2 - 1)
    ac = stats.timeseries.autocorr(x, lags)
    with card():
        fig = make_subplots(rows=1, cols=2, subplot_titles=["ACF", "PACF"])
        for i, col in enumerate(["acf", "pacf"], start=1):
            fig.add_bar(x=ac.lag[1:], y=ac[col][1:], row=1, col=i, marker_color=ui.series_color(1))
            for sign in (1, -1):
                fig.add_hline(y=sign * ac.bound[0], line_dash="dot", line_color=ui.MUTED, row=1, col=i)
        ui.chart(fig.update_layout(title=f"Autocorrelation: {focus} ({series.lower()}), {lags} lags"), height=320)
    with card():
        season = st.number_input("STL seasonal period", 2, max(3, len(x) // 2), min(SEASON_PERIOD[s.freq], len(x) // 2))
        parts = stats.timeseries.decompose(x, int(season))
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=list(parts.columns), vertical_spacing=0.05)
        for i, col in enumerate(parts.columns, start=1):
            fig.add_scatter(x=parts.index, y=parts[col], row=i, col=1, mode="lines", line_color=ui.series_color(1))
        ui.chart(fig.update_layout(title="STL decomposition"), height=680)
    with card(f"Pair: {focus} & {other}"):
        log_a, log_b = np.log(prices[focus]), np.log(prices[other])
        coint = stats.timeseries.cointegration(log_a, log_b).to_frame().T
        ui.conclusions(coint)
        spread = stats.timeseries.spread(log_a, log_b)["zscore"]
        fig = px.line(spread, title="Spread z-score (60)", labels=NO_AXIS)
        for y in (-2, 2):
            fig.add_hline(y=y, line_dash="dot", line_color=ui.MUTED)
        ui.chart(fig, height=300)
        granger = pd.concat([
            stats.timeseries.granger(returns[focus], returns[other]).assign(direction=f"{other} → {focus}"),
            stats.timeseries.granger(returns[other], returns[focus]).assign(direction=f"{focus} → {other}"),
        ])
        st.dataframe(granger, hide_index=True, column_config={"F": ui.NUM, "p_value": ui.NUM,
                                                              "conclusion": st.column_config.TextColumn(width="large")})
    reading = interpret.timeseries(tests, h, coint.iloc[0], granger, focus, other, s.freq, ac,
                                   spread.dropna().iloc[-1])

elif section == "Seasonality":
    rf = returns[focus]
    table = stats.timeseries.monthly_table(rf)
    limit = float(table.abs().max().max())
    with card():
        fig = px.imshow(table, text_auto=".1%", aspect="auto", color_continuous_scale=ui.diverging(),
                        zmin=-limit, zmax=limit, title=f"Monthly returns: {focus}")
        ui.chart(fig.update_layout(coloraxis_showscale=False), height=max(360, 26 * len(table)))
    parts = []
    for by in ["month", "weekday"] if s.freq == "D" else ["month"]:
        summary, tests = stats.timeseries.seasonality(rf, by)
        left, right = st.columns([2, 3])
        with left.container(border=True):
            ui.chart(px.bar(summary, y="mean", title=f"Average return by {by}", labels=NO_AXIS)
                     .update_yaxes(tickformat=".2%"), height=320)
        with right.container(border=True):
            st.dataframe(summary, column_config=ui.percent(summary, ["mean", "median", "std", "pct_positive"]))
            ui.conclusions(tests)
        parts.append((by, summary, tests))
    reading = interpret.seasonality(parts, table, focus)

ui.explain(reading)
