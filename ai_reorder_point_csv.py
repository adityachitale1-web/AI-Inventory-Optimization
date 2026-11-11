import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

# -----------------------------
# Settings (adjust as needed)
# -----------------------------
DATA_FILE = "retail_demand_data.csv"   # CSV columns: date, product, demand, promo + any numeric external factors

# Supply assumptions per product (the CSV holds demand only, so these are yours to set):
# L = lead time (days), sigma_L = lead time std (days), h = holding cost/unit/day,
# optional "cap" = warehouse capacity in units (if left out it is sized from demand)
supply = {
    "Bread & Bakery": {"L": 2, "sigma_L": 0.5, "h": 1.8},
    "Dairy":          {"L": 3, "sigma_L": 0.8, "h": 1.3},
    "Cleaning":       {"L": 4, "sigma_L": 1.0, "h": 1.1},
}
DEFAULT_SUPPLY = {"L": 3, "sigma_L": 0.8, "h": 1.3}   # used for any product not listed above
SERVICE_LEVEL = 0.95        # target used for Z, the Monte Carlo percentile and the fill-rate constraint
ORDER_COST = 2000           # fixed cost per replenishment order
STOCKOUT_COST = 15          # cost per unit of lost sales
NEW_DAYS = 14               # the last 14 days of the file are held back, then added as "new data"
MAX_PRODUCTS = 3            # products analysed and plotted (the first ones found in the file)
rng = np.random.default_rng(42)
warnings.filterwarnings("ignore")
z = norm.ppf(SERVICE_LEVEL)

# -----------------------------
# Load the CSV (asks for an upload in Colab if the file is not there yet) and clean it
# -----------------------------
def load_data(path):
    if not os.path.exists(path):
        try:
            from google.colab import files
            print(f"'{path}' not found - please choose the CSV file to upload:")
            uploaded = files.upload()
            path = list(uploaded.keys())[0]
        except ImportError:
            raise FileNotFoundError(f"Put '{path}' in the same folder as this script, or change DATA_FILE.")
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = {"date", "product", "demand"} - set(df.columns)
    if missing:
        raise ValueError(f"The CSV needs the columns date, product, demand. Missing: {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["product"] = df["product"].astype(str).str.strip()
    if "promo" not in df.columns:
        df["promo"] = 0
    factors = ["promo"] + [c for c in df.columns if c not in ("date", "product", "demand", "promo")
                           and pd.api.types.is_numeric_dtype(df[c])]            # external factors found in the file
    for c in ["demand"] + factors:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date", "demand"])
    clean = []
    for name, g in df.groupby("product", sort=False):                          # one row per day per product, no gaps
        g = g.groupby("date").agg({"demand": "sum", **{c: "mean" for c in factors}})
        g = g.reindex(pd.date_range(g.index.min(), g.index.max()))
        g["demand"] = g["demand"].fillna(0).clip(lower=0)                      # missing day = closed, no sales
        g["promo"] = g["promo"].fillna(0)
        g[factors] = g[factors].ffill().bfill().fillna(0)
        clean.append(g.rename_axis("date").reset_index().assign(product=name))
    return pd.concat(clean, ignore_index=True), factors

full, FACTORS = load_data(DATA_FILE)
names = [n for n in full["product"].unique() if (full["product"] == n).sum() >= 150][:MAX_PRODUCTS]
if not names:
    raise ValueError("Each product needs at least 150 days of history.")
products = {n: {**DEFAULT_SUPPLY, **supply.get(n, {})} for n in names}
cutoff = full["date"].max() - pd.Timedelta(days=NEW_DAYS)
data, new_data = full[full["date"] <= cutoff], full[full["date"] > cutoff]
print(f"Loaded {len(full)} rows | products: {', '.join(names)} | {full['date'].min().date()} to {full['date'].max().date()}")
print(f"Demand drivers used by the forecast: weekday, seasonality, recent level, {', '.join(FACTORS)}")

def add_new_data(data, new_rows):
    """Append newly arrived rows (same columns); every step below is then recomputed from the updated data.
    Example with a second file:  data = add_new_data(data, load_data("new_days.csv")[0])"""
    return pd.concat([data, new_rows]).drop_duplicates(["date", "product"], keep="last").sort_values(["product", "date"])

# -----------------------------
# 1. Demand forecasting (gradient-boosting regression, refreshed as data arrives)
# -----------------------------
FEATURES = ["level", "dow", "sin", "cos"] + FACTORS
HORIZON = 7

def add_features(df):
    df = df.copy()
    df["level"] = df["demand"].shift(8).rolling(28).mean()          # recent demand level (known 7 days ahead)
    df["dow"] = df["date"].dt.dayofweek
    df["sin"], df["cos"] = np.sin(2 * np.pi * df["date"].dt.dayofyear / 365), np.cos(2 * np.pi * df["date"].dt.dayofyear / 365)
    return df

def fit(df):
    df = df.dropna(subset=["level", "demand"])
    return GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0).fit(df[FEATURES], df["demand"])

def forecast_demand(hist, refit_every=7):
    """Back-test on the most recent days (static model vs. model re-trained weekly), then forecast the next 7 days."""
    df = add_features(hist)
    n = len(df)
    test_days = min(90, n // 4)
    predict = lambda model, rows: np.maximum(0, model.predict(rows[FEATURES]))
    static = predict(fit(df.iloc[:n - test_days]), df.iloc[n - test_days:])
    updated = np.concatenate([predict(fit(df.iloc[:s]), df.iloc[s:min(s + refit_every, n)]) for s in range(n - test_days, n, refit_every)])
    actual = df["demand"].values[n - test_days:]
    future = pd.DataFrame({"date": pd.date_range(df["date"].iloc[-1] + pd.Timedelta(days=1), periods=HORIZON), "demand": np.nan})
    for c in FACTORS:                                                # assumed future drivers: replace with real plans if known
        future[c] = df[c].tail(14).median() if c == "promo" else df[c].tail(7).mean()
    fut = add_features(pd.concat([df[["date", "demand"] + FACTORS], future], ignore_index=True)).tail(HORIZON)
    error = lambda pred: np.abs(actual - pred).sum() / max(actual.sum(), 1e-9) * 100    # % of volume mis-forecast
    return {"dates": df["date"].values[n - test_days:], "actual": actual, "backtest": updated,
            "err_static": error(static), "err_updated": error(updated),
            "sigma_raw": actual.std(), "sigma_D": max(np.std(actual - updated), 1e-6),   # forecast error = remaining uncertainty
            "forecast": predict(fit(df), fut)}

# -----------------------------
# 2. Dynamic safety stock: formula + Monte Carlo
#    SS = Z * sqrt(sigma_D^2 * L + D^2 * sigma_L^2)
# -----------------------------
def safety_stock(D, sigma_D, L, sigma_L, n=50000):
    ss_formula = z * np.sqrt(sigma_D**2 * L + D**2 * sigma_L**2)
    lead = rng.gamma((L / sigma_L)**2, sigma_L**2 / L, n)                    # random lead times (always positive, late-skewed)
    lt_demand = rng.normal(D * lead, sigma_D * np.sqrt(lead))                # demand during each simulated lead time
    ss_mc = np.quantile(lt_demand, SERVICE_LEVEL) - lt_demand.mean()
    return ss_formula, ss_mc, lt_demand

# -----------------------------
# 3. AI-driven sensitivity analysis: Bayesian optimisation of the reorder point
#    Objective: daily cost (holding + stockout + ordering) from an inventory simulation
#    Constraints: fill rate >= SERVICE_LEVEL, reorder point + order quantity <= warehouse capacity
# -----------------------------
def simulate_policy(rop, Q, D, sigma_D, L, sigma_L, h, days=120, paths=300, seed=1):
    r = np.random.default_rng(seed)                                          # same random scenarios for every reorder point
    dem = np.maximum(0, r.normal(D, sigma_D, (paths, days)))
    lead = np.maximum(1, np.round(r.gamma((L / sigma_L)**2, sigma_L**2 / L, (paths, days)))).astype(int)
    stock, on_order, arrivals = np.full(paths, rop + Q, float), np.zeros(paths), np.zeros((paths, days + lead.max() + 1))
    holding = lost = orders = 0.0
    for t in range(days):
        stock += arrivals[:, t]; on_order -= arrivals[:, t]
        sold = np.minimum(stock, dem[:, t]); stock -= sold; lost += (dem[:, t] - sold).sum()
        idx = np.where(stock + on_order <= rop)[0]                           # reorder when inventory position hits the reorder point
        arrivals[idx, t + lead[idx, t]] += Q; on_order[idx] += Q; orders += len(idx)
        holding += stock.sum()
    cost = (h * holding + STOCKOUT_COST * lost + ORDER_COST * orders) / (paths * days)
    return cost, 1 - lost / max(dem.sum(), 1e-9)

def bayesian_optimise(objective, lo, hi, n_init=5, n_iter=15):
    X = list(np.linspace(lo, hi, n_init)); y = [objective(x) for x in X]
    grid = np.linspace(0, 1, 400)[:, None]
    for _ in range(n_iter):
        gp = GaussianProcessRegressor(ConstantKernel() * Matern(0.3, (0.05, 2), nu=2.5) + WhiteKernel(1e-3, (1e-6, 0.1)), normalize_y=True, random_state=0)
        gp.fit((np.array(X)[:, None] - lo) / (hi - lo), y)                   # surrogate model of cost vs reorder point
        mu, sd = gp.predict(grid, return_std=True)
        imp = min(y) - mu
        ei = imp * norm.cdf(imp / (sd + 1e-9)) + sd * norm.pdf(imp / (sd + 1e-9))   # expected improvement
        x_next = lo + grid[np.argmax(ei), 0] * (hi - lo)
        X.append(x_next); y.append(objective(x_next))
    return np.array(X), np.array(y)

# -----------------------------
# Full pipeline for one product (re-run whenever data is added)
# -----------------------------
def analyse(name, data):
    p = products[name]
    f = forecast_demand(data[data["product"] == name].sort_values("date").reset_index(drop=True))
    D = max(f["forecast"][:int(np.ceil(p["L"]))].mean(), 1.0)                # forecast demand over the coming lead time
    ss_formula, ss_mc, lt_demand = safety_stock(D, f["sigma_D"], p["L"], p["sigma_L"])
    Q = np.sqrt(2 * D * ORDER_COST / p["h"])                                 # economic order quantity
    cap = p.get("cap", D * (2.5 * p["L"] + 3))                               # warehouse capacity
    sim = lambda rop: simulate_policy(rop, Q, D, f["sigma_D"], p["L"], p["sigma_L"], p["h"])
    def penalised(rop):                                                      # cost + penalty if the fill-rate constraint is broken
        cost, fill = sim(rop)
        return cost + 50 * D * max(0, SERVICE_LEVEL - fill)
    lo = 0.5 * D * p["L"]
    hi = max(cap - Q, 1.5 * lo)                                              # capacity constraint = upper bound
    X, y = bayesian_optimise(penalised, lo, hi)
    rop_bo = X[np.argmin(y)]
    return {**f, "D": D, "Q": Q, "ss_formula": ss_formula, "ss_mc": ss_mc, "lt_demand": lt_demand,
            "rop_formula": D * p["L"] + ss_formula, "rop_mc": D * p["L"] + ss_mc, "rop_bo": rop_bo,
            "bo_X": X, "bo_y": y, "sim": sim, "penalised": penalised, "lo": lo, "hi": hi}

results = {name: analyse(name, data) for name in products}

# -----------------------------
# Results table
# -----------------------------
def report(results, title):
    rows = []
    for name, r in results.items():
        cost_f, fill_f = r["sim"](r["rop_formula"]); cost_b, fill_b = r["sim"](r["rop_bo"])
        rows.append({"Product": name, "Forecast error % (static)": r["err_static"], "Forecast error % (updated)": r["err_updated"],
                     "Sigma raw": r["sigma_raw"], "Sigma forecast": r["sigma_D"], "Forecast D/day": r["D"],
                     "SS formula": r["ss_formula"], "SS Monte Carlo": r["ss_mc"], "ROP formula": r["rop_formula"], "ROP Monte Carlo": r["rop_mc"],
                     "ROP Bayesian": r["rop_bo"], "Fill formula %": fill_f * 100, "Fill Bayesian %": fill_b * 100,
                     "Cost/day formula": cost_f, "Cost/day Bayesian": cost_b})
    print("\n" + title); print(pd.DataFrame(rows).set_index("Product").T.round(1).to_string())

report(results, f"RESULTS (data up to {data['date'].max().date()})")
if len(new_data):                                                            # new data arrives -> everything is recomputed
    data = add_new_data(data, new_data)
    report({name: analyse(name, data) for name in products}, f"RESULTS (after adding {new_data['date'].nunique()} new days, up to {data['date'].max().date()})")

# -----------------------------
# Plot setup
# -----------------------------
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": True, "axes.spines.right": True, "axes.grid": True, "grid.alpha": 0.3})
fig, axes = plt.subplots(3, len(results), figsize=(14 * max(len(results), 2) / 3, 11), squeeze=False)
fig.suptitle("AI-Enhanced Reorder Point & Inventory Optimization", fontsize=13, fontweight='bold')

for col, (name, r) in enumerate(results.items()):
    # Row 1: demand forecast back-test
    ax = axes[0, col]
    ax.plot(r["dates"], r["actual"], color="tab:blue", linewidth=1.5, label="Actual Demand")
    ax.plot(r["dates"], r["backtest"], color="tab:green", linewidth=1.5, label="ML Forecast (updated weekly)")
    ax.set_title(f"{name}\nForecast error {r['err_updated']:.1f}% of volume", fontsize=11, fontweight='bold')
    ax.tick_params(axis="x", rotation=30, labelsize=8)

    # Row 2: Monte Carlo lead-time demand and reorder points
    ax = axes[1, col]
    ax.hist(r["lt_demand"], bins=60, color="tab:blue", alpha=0.5, label="Simulated Lead-Time Demand")
    ax.axvline(r["rop_formula"], color="tab:green", linewidth=2, label="Reorder Point (Formula)")
    ax.axvline(r["rop_mc"], color="tab:red", linewidth=2, linestyle="--", label="Reorder Point (Monte Carlo)")
    ax.set_title(f"Safety stock: formula {r['ss_formula']:.0f} | Monte Carlo {r['ss_mc']:.0f}", fontsize=10, fontweight='bold')
    ax.set_xlabel("Units Demanded During Lead Time")

    # Row 3: Bayesian optimisation vs. a manual parameter sweep
    ax = axes[2, col]
    sweep = np.linspace(r["lo"], r["hi"], 60)
    ax.plot(sweep, [r["penalised"](x) for x in sweep], color="tab:blue", linewidth=2, label="Manual Sweep (60 runs)")
    ax.scatter(r["bo_X"], r["bo_y"], color="tab:green", s=22, zorder=3, label=f"Bayesian Search ({len(r['bo_X'])} runs)")
    ax.axvline(r["rop_bo"], color="tab:red", linewidth=2, linestyle="--", label="Optimal Reorder Point")
    ax.set_title(f"Optimal reorder point {r['rop_bo']:.0f}", fontsize=10, fontweight='bold')
    ax.set_xlabel("Reorder Point (units)")

for ax in axes.ravel():
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
axes[0, 0].set_ylabel("Units per Day", fontsize=10, fontweight='bold')
axes[1, 0].set_ylabel("Simulated Scenarios", fontsize=10, fontweight='bold')
axes[2, 0].set_ylabel("Cost per Day (incl. penalty)", fontsize=10, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("reorder_point_results.png")
plt.show()
