# AI Inventory Optimization

Demand forecasting, Monte Carlo safety stock and Bayesian reorder-point optimisation on real retail data, delivered as an interactive Streamlit app.

## Run it on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **Create app** → **Deploy a public app from GitHub**.
3. Choose this repository, branch `main`, main file path **`app.py`**.
4. Under **Advanced settings**, pick Python **3.12** (3.10+ works).
5. Click **Deploy**. The first build installs `requirements.txt` (a few minutes); the first page load trains the forecasting models (about 20–30 seconds), after which results are cached.

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Using the app

- **Sidebar:** use the sample data or upload your own CSV, choose how many recent days to hold back as "new data", how many products to analyse, and the dashboard height.
- **Tabs:** Decision Cockpit · Demand Forecast · Safety Stock & Monte Carlo · Sensitivity & Bayesian Optimisation · Data & Methodology. Product buttons switch the charts and product tables.
- **Edit inputs:** the blue **Edit inputs & data** tab on the right edge of the dashboard opens a panel to change the service level, costs, each product's lead time, lead-time variability, holding cost, capacity, forecast-error std and the 7 daily demand data points. **Recalculate** re-runs safety stock, Monte Carlo, Bayesian optimisation and the scenarios in about 1–2 seconds; **Reset all** restores the model values.
- **Download:** the button under the dashboard saves a standalone HTML copy that opens offline in any browser.

### Uploading your own data

| Column | Required | Notes |
|---|---|---|
| `date` | yes | Daily dates |
| `product` | yes | Product or category name |
| `demand` | yes | Units sold that day |
| `promo` | no | Promotion indicator or count |
| any other numeric column | no | Used as an extra demand driver (e.g. `oil_price`, `holiday`, `temperature`) |

At least 150 days per product. Header names are not case-sensitive. Products not in the built-in list get default supply assumptions (lead time 3 days, std 0.8, holding cost 1.30), editable in the panel.

## How it works

| Step | Where | Method |
|---|---|---|
| 1. Demand forecasting | Python | Gradient-boosting regression on recent demand level, weekday, seasonality and the drivers in the data; re-trained weekly in a rolling 90-day back-test. |
| 2. Dynamic safety stock | Browser | `SS = Z · √(σD²·L + D²·σL²)` using the forecast error as σD, plus a 50,000-scenario Monte Carlo with random (gamma) lead times. |
| 3. Bayesian optimisation | Browser | A Gaussian-process surrogate searched by expected improvement finds the reorder point that minimises daily cost (holding + stockout + ordering) under a fill-rate target and capacity, in 20 runs instead of a 60-run sweep. |
| 4. Sensitivity analysis | Browser | The optimisation is re-run under changed lead time, lead-time variability, demand, holding cost, stockout cost and service target. |

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit app (entry point). |
| `inventory_engine.py` | CSV loading/cleaning and the ML demand forecasts. |
| `dashboard_builder.py` | Builds the interactive dashboard page, including the in-browser maths. |
| `assets/chart.umd.js` | Chart.js 4.4.4 (MIT licence, see `assets/CHARTJS_LICENSE.md`), embedded so the dashboard works offline. |
| `retail_demand_data.csv` | Sample data: daily demand for Bread & Bakery, Cleaning and Dairy, store 1, 16 Aug 2015 – 15 Aug 2017. |
| `inventory_dashboard.html` | Pre-built standalone copy of the dashboard; double-click to open, no installation needed. |
| `ai_reorder_point_csv.py` | Stand-alone analysis script for Google Colab or the command line (prints tables and a 3×3 chart). |
| `requirements.txt`, `.streamlit/config.toml` | Dependencies and theme for Streamlit. |

## Data

Corporación Favorita (Ecuador) grocery sales, from the public Kaggle *Store Sales – Time Series Forecasting* dataset. External drivers: national/local holidays and daily WTI oil price.

## Assumptions and limitations

- The dataset contains demand only. Lead times, lead-time variability and costs are assumptions, editable in the dashboard.
- The textbook formula targets a 95% *cycle* service level (≈99.7% fill rate in simulation); the Bayesian search targets a 95% *fill rate*, a looser target, so part of its cost saving reflects that.
- Editing inputs re-runs the inventory maths in the browser; the ML forecast is re-trained in Python when data or sidebar settings change.
- Results are for one store.
