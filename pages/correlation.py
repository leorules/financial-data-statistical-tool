import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from market import interpret, stats, ui

s = ui.settings()
inst = ui.instruments()
ui.require(inst)
corr_mod, mv = stats.correlation, stats.multivariate

ui.header("Correlation & Covariance", "How tickers or asset classes move together: matrices, pairs, common factors "
                                     "and portfolios.")
ui.adjustments_caption(s, "fdr", "hac", "shrinkage", "live_cash")
with st.container(border=True):
    tickers, groups = ui.subject_picker()
    c1, c2, c3 = st.columns([3, 2, 1], vertical_alignment="bottom")
    method = c1.segmented_control("Matrix", ["pearson", "spearman", "kendall", "partial", "covariance"],
                                  default="pearson", required=True)
    s, period = ui.stress_window("correlation_window", s, c2)
    clustered = c3.toggle("Cluster", value=True, help="Order columns so similar ones sit together")
    ui.stress_caption(period)
ui.require(tickers[1:], "Pick at least two tickers or asset classes.")
r = ui.return_matrix(tickers, s, groups=groups)
ui.require(r.columns[1:], f"No overlapping data for these series in the {period.name} window." if period else
           "Need at least two tickers with enough overlapping data.")
if len(r) < 10:
    st.warning(f"Only {len(r)} return observations in this window; correlations will be very unreliable.",
               icon=":material/warning:")

corr = corr_mod.matrix(r, "pearson" if method in ("partial", "covariance") else method)
matrix = {"partial": corr_mod.partial, "covariance": corr_mod.covariance}.get(method, lambda x: corr)(r)
order = corr_mod.cluster_order(corr) if clustered else list(r.columns)
matrix = matrix.loc[order, order]
limit = 1 if method != "covariance" else matrix.abs().max().max()

with st.container(border=True):
    st.caption(f"{len(r)} {ui.FREQS[s.freq].lower()} {s.kind} returns · {r.index[0]:%d %b %Y} – {r.index[-1]:%d %b %Y}")
    ui.close_time_caption(r.columns, s, groups)
    fig = px.imshow(matrix, text_auto=".2f" if len(order) <= 20 else False, aspect="auto",
                    color_continuous_scale=ui.diverging(), zmin=-limit, zmax=limit)
    ui.chart(fig.update_layout(coloraxis_colorbar=dict(thickness=10)), height=max(420, 30 * len(order)))
    ui.download(matrix, f"{method}_matrix")

raw_p = corr_mod.pvalues(r, "pearson" if method in ("partial", "covariance") else method).loc[order, order]
pvals = corr_mod.correct(raw_p) if s.on("fdr") else raw_p
explained, loadings = mv.pca(r)
pairs = corr_mod.pairs(corr)
pairs_tab, time_tab, pvalue_tab, pair_tab, pca_tab, port_tab = st.tabs(
    ["Top pairs", "Through time", "Significance", "Pair analysis", "PCA", "Portfolios"])

with pairs_tab:
    left, right = st.columns(2)
    left.markdown("**Most correlated**")
    left.dataframe(pairs.head(10), hide_index=True, column_config={"corr": ui.NUM})
    right.markdown("**Least correlated**")
    right.dataframe(pairs.tail(10).iloc[::-1], hide_index=True, column_config={"corr": ui.NUM})

with time_tab:
    longest = max(20, min(252, len(r) // 2))
    window = st.slider("Window", 20, longest, min(126 if s.freq == "D" else 26, longest),
                       help="Average correlation across every pair in the basket, recalculated over this "
                            "trailing window")
    average = corr_mod.rolling_average(r, window)
    if len(average) < 2:
        st.info("Not enough observations in this window to track correlation through time.")
    else:
        whole = corr.to_numpy()[np.triu_indices(len(corr), 1)].mean()
        fig = px.line(average.rename("Average pairwise correlation"), labels={"value": "", "date": ""},
                      title="Average pairwise correlation")
        fig.add_hline(y=whole, line_dash="dot", line_color=ui.MUTED,
                      annotation_text=f"full period {whole:.2f}", annotation_position="top right")
        ui.chart(fig.update_yaxes(range=[min(-0.1, average.min() - 0.05), 1]), height=340)
        c1, c2, c3 = st.columns(3)
        c1.metric("Latest", f"{average.iloc[-1]:.2f}", f"{average.iloc[-1] - whole:+.2f} vs full period")
        c2.metric("Highest", f"{average.max():.2f}", help=f"Reached {average.idxmax():%d %b %Y}")
        c3.metric("Lowest", f"{average.min():.2f}", help=f"Reached {average.idxmin():%d %b %Y}")
        st.caption("Correlations rise together in a sell-off, so diversification is weakest exactly when it is "
                   "most needed. A line well above its full-period average means the basket is currently "
                   "behaving as one position.")

with pvalue_tab:
    st.caption("P-values for H0: no correlation. Darker cells (p < 0.05) are statistically significant.")
    upper = np.triu_indices(len(pvals), k=1)
    significant, tested = (pvals.to_numpy()[upper] < 0.05).sum(), len(upper[0])
    if s.on("fdr"):
        before = (raw_p.to_numpy()[upper] < 0.05).sum()
        st.caption(f"{tested} pairs tested at once: {before} significant before the false-discovery correction, "
                   f"{significant} after.")
    else:
        st.caption(f"{significant} of {tested} pairs are significant. With this many simultaneous tests, about "
                   f"{0.05 * tested:.0f} would look significant by chance alone — switch on the false-discovery "
                   "correction in the sidebar to allow for that.")
    fig = px.imshow(pvals, text_auto=".3f" if len(order) <= 20 else False, aspect="auto", zmin=0, zmax=0.1,
                    color_continuous_scale=ui.sequential()[::-1])
    ui.chart(fig.update_layout(coloraxis_colorbar=dict(thickness=10)), height=max(420, 30 * len(order)))

with pair_tab:
    c1, c2, c3 = st.columns(3)
    a = c1.selectbox("A", r.columns, index=0, format_func=ui.label)
    b = c2.selectbox("B", r.columns, index=1, format_func=ui.label)
    window = c3.slider("Rolling window", 10, 252, 63 if s.freq == "D" else 26)
    rolling = stats.rolling.corr(r[a], r[b], window).rename(f"{window}-period correlation")
    fig = px.line(rolling, title=f"Rolling correlation: {a} vs {b}", labels={"value": "", "date": ""})
    fig.add_hline(y=corr.loc[a, b], line_color=ui.MUTED, annotation_text=f"full period {corr.loc[a, b]:.2f}")
    ui.chart(fig, height=300)
    left, right = st.columns(2)
    ui.chart(px.scatter(r, x=b, y=a, trendline="ols", opacity=0.45, title="Return scatter with OLS fit"), left)
    lag = corr_mod.lead_lag(r[a], r[b], 10)
    ui.chart(px.bar(lag, x="lag", y="corr", title="Lead–lag (positive lag: B leads A)"), right)
    st.dataframe(stats.regression.capm(r[a], r[b], hac=s.on("hac")).to_frame(f"{a} ~ {b}").T, hide_index=True)

with pca_tab:
    fig = go.Figure([go.Bar(x=explained.component, y=explained.explained, name="Explained"),
                     go.Scatter(x=explained.component, y=explained.cumulative, name="Cumulative", mode="lines+markers")])
    ui.chart(fig.update_layout(title="Explained variance", yaxis_tickformat=".0%"), height=320)
    st.caption(f"PC1 explains {explained.explained.iloc[0]:.0%} of the basket's variance. A high share means the "
               "tickers mostly move together with one common factor.")
    k = min(5, len(loadings.columns))
    fig = px.imshow(loadings.iloc[:, :k], text_auto=".2f", aspect="auto", color_continuous_scale=ui.diverging(),
                    zmin=-1, zmax=1, title="Loadings")
    ui.chart(fig, height=max(320, 28 * len(loadings)))

with port_tab:
    table, weights = mv.portfolios(r, ui.risk_free(s, r.index), shrink=s.on("shrinkage"))
    if s.on("shrinkage"):
        st.caption(f"Ledoit–Wolf shrinkage intensity {mv.shrinkage_intensity(r):.0%} towards the structured "
                   "target; higher means a noisier sample covariance.")
    st.dataframe(table, column_config=ui.percent(table, ["ann_return", "ann_vol"])
                 | {"sharpe": ui.NUM, "diversification_ratio": ui.NUM})
    fig = px.bar(weights, barmode="group", title="Weights", labels={"value": "", "index": ""})
    ui.chart(fig.update_yaxes(tickformat=".0%"), height=360)

ui.explain(interpret.correlation(corr, pvals, pairs, explained, method, r))
