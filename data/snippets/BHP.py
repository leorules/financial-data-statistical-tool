r = returns("BHP.AX", start="2021-01-01")["BHP.AX"]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax1.bar(r.index, r, width=1, color=np.where(r >= 0, "#0ca30c", "#d03b3b"))
ax1.set_title("BHP.AX daily returns")
ax1.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))

growth = (1 + r).cumprod() - 1
ax2.plot(growth.index, growth, linewidth=1.5)
ax2.set_title("Cumulative return")
ax2.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
fig.tight_layout()

print(f"Average daily return {r.mean():.3%}, volatility {r.std() * 252 ** 0.5:.1%} a year, total {growth.iloc[-1]:+.1%}")
fig