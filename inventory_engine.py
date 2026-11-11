"""Data loading and ML demand forecasting for the inventory dashboard.

The browser dashboard recalculates safety stock, Monte Carlo, Bayesian optimisation
and scenarios itself; this module supplies the parts that need Python: cleaning the
CSV and training the gradient-boosting demand forecasts.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

# Supply assumptions per product (the CSV holds demand only).
# L = lead time (days), sigma_L = lead-time std (days), h = holding cost / unit / day
SUPPLY = {
    "Bread & Bakery": {"L": 2, "sigma_L": 0.5, "h": 1.8},
    "Dairy":          {"L": 3, "sigma_L": 0.8, "h": 1.3},
    "Cleaning":       {"L": 4, "sigma_L": 1.0, "h": 1.1},
}
DEFAULT_SUPPLY = {"L": 3, "sigma_L": 0.8, "h": 1.3}
GLOBAL_DEFAULTS = {"sl": 0.95, "oc": 2000, "so": 15}   # service level, cost per order, cost per lost unit
HORIZON = 7
MIN_DAYS = 150


def load_data(src):
    """Read a CSV (path or file-like) with columns date, product, demand [, promo, other numeric drivers].
    Returns one row per product per day with gaps filled, and the list of driver columns."""
    df = pd.read_csv(src)
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = {"date", "product", "demand"} - set(df.columns)
    if missing:
        raise ValueError(f"The CSV needs the columns date, product and demand. Missing: {', '.join(sorted(missing))}.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["product"] = df["product"].astype(str).str.strip()
    if "promo" not in df.columns:
        df["promo"] = 0
    factors = ["promo"] + [c for c in df.columns if c not in ("date", "product", "demand", "promo")
                           and pd.api.types.is_numeric_dtype(df[c])]
    for c in ["demand"] + factors:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date", "demand"])
    if df.empty:
        raise ValueError("No usable rows: check that the date and demand columns contain valid values.")
    clean = []
    for name, g in df.groupby("product", sort=False):
        g = g.groupby("date").agg({"demand": "sum", **{c: "mean" for c in factors}})
        g = g.reindex(pd.date_range(g.index.min(), g.index.max()))
        g["demand"] = g["demand"].fillna(0).clip(lower=0)
        g["promo"] = g["promo"].fillna(0)
        g[factors] = g[factors].ffill().bfill().fillna(0)
        clean.append(g.rename_axis("date").reset_index().assign(product=name))
    return pd.concat(clean, ignore_index=True), factors


def add_features(df):
    df = df.copy()
    df["level"] = df["demand"].shift(8).rolling(28).mean()          # recent level, known 7 days ahead
    df["dow"] = df["date"].dt.dayofweek
    doy = df["date"].dt.dayofyear
    df["sin"], df["cos"] = np.sin(2 * np.pi * doy / 365), np.cos(2 * np.pi * doy / 365)
    return df


def fit(df, features):
    df = df.dropna(subset=["level", "demand"])
    return GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                     subsample=0.8, random_state=0).fit(df[features], df["demand"])


def forecast_demand(hist, features, factors, refit_every=7):
    """Back-test (static model vs. model re-trained weekly), then forecast the next 7 days."""
    df = add_features(hist)
    n = len(df)
    test_days = min(90, n // 4)
    predict = lambda model, rows: np.maximum(0, model.predict(rows[features]))
    static = predict(fit(df.iloc[:n - test_days], features), df.iloc[n - test_days:])
    updated = np.concatenate([predict(fit(df.iloc[:s], features), df.iloc[s:min(s + refit_every, n)])
                              for s in range(n - test_days, n, refit_every)])
    actual = df["demand"].values[n - test_days:]
    future = pd.DataFrame({"date": pd.date_range(df["date"].iloc[-1] + pd.Timedelta(days=1), periods=HORIZON), "demand": np.nan})
    for c in factors:                                      # future drivers assumed at recent levels
        future[c] = df[c].tail(14).median() if c == "promo" else df[c].tail(7).mean()
    fut = add_features(pd.concat([df[["date", "demand"] + factors], future], ignore_index=True)).tail(HORIZON)
    error = lambda pred: np.abs(actual - pred).sum() / max(actual.sum(), 1e-9) * 100
    model = fit(df, features)
    return {"dates": df["date"].values[n - test_days:], "actual": actual, "backtest": updated,
            "err_static": error(static), "err_updated": error(updated),
            "sigma_raw": float(actual.std()), "sigma_D": max(float(np.std(actual - updated)), 1e-6),
            "forecast": predict(model, fut), "importance": dict(zip(features, model.feature_importances_ * 100))}


def run_analysis(full, factors, new_days=14, max_products=3, progress=None):
    """Train forecasts on data up to the cutoff (last `new_days` held back) and again on all data."""
    names = [n for n in full["product"].unique() if (full["product"] == n).sum() >= MIN_DAYS][:max_products]
    if not names:
        raise ValueError(f"Each product needs at least {MIN_DAYS} days of history.")
    cutoff = full["date"].max() - pd.Timedelta(days=new_days)
    data = full[full["date"] <= cutoff]
    features = ["level", "dow", "sin", "cos"] + factors
    results = {}
    for i, n in enumerate(names):
        if progress:
            progress(i / len(names), f"Training forecasting models for {n}…")
        hist = data[data["product"] == n].sort_values("date").reset_index(drop=True)
        allp = full[full["product"] == n].sort_values("date").reset_index(drop=True)
        if len(hist) < MIN_DAYS - new_days:
            raise ValueError(f"'{n}' has too little history before the held-back period.")
        r0 = forecast_demand(hist, features, factors)
        r1 = forecast_demand(allp, features, factors)
        supply = {**DEFAULT_SUPPLY, **SUPPLY.get(n, {})}
        D = max(float(np.mean(r0["forecast"][:int(np.ceil(supply["L"]))])), 1.0)
        results[n] = {"r0": r0, "r1": r1, "supply": supply, "D": D}
    if progress:
        progress(1.0, "Building dashboard…")
    return {"names": names, "cutoff": cutoff, "full": full, "factors": factors, "features": features,
            "new_days": new_days, "results": results}
