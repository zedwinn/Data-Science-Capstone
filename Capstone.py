import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from arch import arch_model
import matplotlib.pyplot as plt

# download data
tickers = ["AAPL", "MSFT", "NVDA", "JNJ", "SPY"]
prices = yf.download(tickers, start="2010-01-01", end="2026-03-26")["Close"].dropna()
log_returns = np.log(prices / prices.shift(1)).dropna()

# build features
def build_features(series):
    df = pd.DataFrame({"ret": series})
    df["lag1"] = df["ret"].shift(1)
    df["lag2"] = df["ret"].shift(2)
    df["lag5"] = df["ret"].shift(5)
    df["roll_mean_5"] = df["ret"].rolling(5).mean().shift(1)
    df["roll_mean_20"] = df["ret"].rolling(20).mean().shift(1)
    df["roll_std_5"] = df["ret"].rolling(5).std().shift(1)
    df["roll_std_20"] = df["ret"].rolling(20).std().shift(1)
    df["momentum_20"] = df["ret"].rolling(20).sum().shift(1)
    df["target"] = (df["ret"] > 0).astype(int)
    return df.dropna()

features = ["lag1", "lag2", "lag5", "roll_mean_5", "roll_mean_20",
            "roll_std_5", "roll_std_20", "momentum_20"]

TRAIN_END = "2024-03-24"
TEST_START = "2024-03-25"

results = {}

for ticker in tickers:
    df = build_features(log_returns[ticker])
    train = df.loc[:TRAIN_END]
    test = df.loc[TEST_START:]

    # logistic regression
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(train[features], train["target"])
    proba = model.predict_proba(test[features])[:, 1]
    preds = (proba >= 0.5).astype(int)

    acc = accuracy_score(test["target"], preds)
    auc = roc_auc_score(test["target"], proba)

    print(f"\n--- Logistic Regression summary ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC:  {auc:.4f}")
    print(classification_report(test["target"], preds, target_names=["Down", "Up"], zero_division=0))

    # garch
    test_dates = test.index
    ret_scaled = log_returns[ticker] * 100
    garch_vol = pd.Series(index=test_dates, dtype=float)

    for day in test_dates:
        data = ret_scaled.loc[:day].iloc[:-1]
        gm = arch_model(data, vol="Garch", p=2, q=1, mean="Constant", dist="normal")
        res = gm.fit(disp="off", show_warning=False)
        fcast = res.forecast(horizon=1)
        garch_vol.loc[day] = np.sqrt(fcast.variance.values[-1, 0])

    # trading strategy
    actual_ret = log_returns[ticker].loc[test_dates]
    signal = pd.Series(0.0, index=test_dates)
    signal[proba > 0.52] = 1.0
    signal[proba < 0.48] = -1.0

    inv_vol = 1.0 / garch_vol
    weight = inv_vol / inv_vol.mean()

    strategy_ret = signal * weight * actual_ret
    cum_strategy = strategy_ret.cumsum().apply(np.exp) - 1
    cum_buyhold = actual_ret.cumsum().apply(np.exp) - 1

    tp = prices[ticker].loc[TEST_START:]
    bh_return = (tp.iloc[-1] - tp.iloc[0]) / tp.iloc[0]

    results[ticker] = {
        "acc": acc, "auc": auc,
        "strat_return": cum_strategy.iloc[-1],
        "bh_return": bh_return,
        "cum_strategy": cum_strategy,
        "cum_buyhold": cum_buyhold,
        "alpha1": res.params['alpha[1]'],
        "alpha2": res.params['alpha[2]'],
        "beta": res.params['beta[1]'],
        "mean_vol": garch_vol.mean(),}

# plot
fig, axes = plt.subplots(3, 2, figsize=(14, 10))
axes = axes.flatten()

for i, ticker in enumerate(tickers):
    ax = axes[i]
    r = results[ticker]
    ax.plot(r["cum_strategy"] * 100, label="Strategy")
    ax.plot(r["cum_buyhold"] * 100, label="Buy & Hold", alpha=0.7)
    ax.set_title(f"{ticker}  (Acc: {r['acc']:.2f}, AUC: {r['auc']:.2f})")
    ax.set_ylabel("Return (%)")
    ax.legend(fontsize=8)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.grid(True, alpha=0.3)

axes[-1].axis("off")
fig.suptitle("Strategy vs Buy-and-Hold (Mar 2024 – Mar 2026)", fontsize=14)
plt.tight_layout()
plt.savefig("results.png", dpi=150, bbox_inches="tight")
print("\nSaved results.png")

# visualizaion 1
plt.figure()
plt.bar(tickers, [results[t]["strat_return"]*100 for t in tickers])
plt.ylabel("Return (%)")
plt.title("Strategy Returns")
plt.savefig("returns_comparison.png", dpi=150)

# visualization 2
plt.figure()
plt.bar(tickers, [results[t]["acc"] for t in tickers])
plt.ylabel("Accuracy")
plt.title("Model Accuracy")
plt.savefig("model_accuracy.png", dpi=150)

# garch summary
print("\n--- Garch summary ---")
print(f"\n{'Ticker':<8} {'alpha1':>8} {'alpha2':>8} {'beta':>8} {'Persist':>8} {'Mean Vol':>10}")
for t in tickers:
    r = results[t]
    persistence = r['alpha1'] + r['alpha2'] + r['beta']
    print(f"{t:<8} {r['alpha1']:>8.4f} {r['alpha2']:>8.4f} {r['beta']:>8.4f} {persistence:>8.4f} {r['mean_vol']:>8.2f}%")

# summary table
print("\n--- Summary table ---")
print(f"{'Ticker':<8} {'Accuracy':>8} {'AUC':>8} {'Strategy':>10} {'Buy&Hold':>10}")
for t in tickers:
    r = results[t]
    print(f"{t:<8} {r['acc']:>8.3f} {r['auc']:>8.3f} {r['strat_return']*100:>+9.1f}% {r['bh_return']*100:>+9.1f}%")