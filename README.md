# AI Inventory Optimization

Demand forecasting, Monte Carlo safety stock and Bayesian reorder-point optimisation on real retail data, with an interactive dashboard.

## Open the dashboard

Download **`inventory_dashboard.html`** and double-click it. It opens in any web browser (Chrome, Edge, Safari, Firefox) with no installation and no internet connection; all charts, data and the charting library are embedded in the single file.

**Tabs:** Decision Cockpit · Demand Forecast · Safety Stock & Monte Carlo · Sensitivity & Bayesian Optimisation · Data & Methodology. Use the product buttons to switch the charts and product tables.

**Editing inputs:** click the blue **"Edit inputs & data"** tab on the right edge. Change the service level, costs, each product's lead time, lead-time variability, holding cost, capacity, forecast-error std and the 7 daily demand data points, then press **Recalculate**. Safety stock, Monte Carlo, Bayesian optimisation and the scenarios re-run in the browser (about 1–2 seconds). **Reset all** restores the model values.

## What it does

| Step | Method |
|---|---|
| 1. Demand forecasting | Gradient-boosting regression on recent demand level, weekday, seasonality, promotions, holidays and oil price; re-trained weekly in a rolling 90-day back-test. New data can be appended and the pipeline recomputes. |
| 2. Dynamic safety stock | `SS = Z · √(σD²·L + D²·σL²)` using the forecast error as σD, alongside a 50,000-scenario Monte Carlo that samples random (gamma) lead times and demand. |
| 3. Bayesian optimisation | A Gaussian-process surrogate searched by expected improvement finds the reorder point that minimises daily cost (holding + stockout + ordering) under a fill-rate target and warehouse capacity, in 20 runs instead of a 60-run manual sweep. |
| 4. Sensitivity analysis | The optimisation is re-run under changed lead time, lead-time variability, demand, holding cost, stockout cost and service target. |

## Files

| File | Purpose |
|---|---|
| `inventory_dashboard.html` | The self-contained interactive dashboard. |
| `retail_demand_data.csv` | Daily demand for three product families (Bread & Bakery, Cleaning, Dairy), store 1, 16 Aug 2015 – 15 Aug 2017. Columns: `date, product, demand, promo, oil_price, holiday`. |
| `ai_reorder_point_csv.py` | The analysis: forecasting, safety stock, Monte Carlo, Bayesian optimisation. Prints results tables and a 3×3 chart. |
| `build_dashboard.py` | Regenerates `inventory_dashboard.html` from the analysis. |

## Run the analysis

Works in Google Colab or locally (Python 3.10+).

```bash
pip install numpy pandas scipy scikit-learn matplotlib
python ai_reorder_point_csv.py          # reads retail_demand_data.csv from the same folder
```

In Colab, paste the script into a cell and run it; if the CSV is not in the session it asks you to upload it.

To rebuild the dashboard:

```bash
npm pack chart.js@4.4.4 && tar xzf chart.js-4.4.4.tgz
python build_dashboard.py package/dist/chart.umd.js
```

## Data

Corporación Favorita (Ecuador) grocery sales, from the public Kaggle *Store Sales – Time Series Forecasting* dataset. External drivers: national/local holidays and daily WTI oil price.

## Assumptions and limitations

- The dataset contains demand only. Lead times, lead-time variability and costs (holding, ordering, stockout) are configurable assumptions, not derived from the data.
- The textbook formula targets a 95% *cycle* service level (≈99.7% fill rate in simulation); the Bayesian search targets a 95% *fill rate*, a looser target, so part of its cost saving reflects that.
- In the dashboard, editing inputs re-runs the inventory maths in the browser; the ML forecast itself is only re-trained in Python.
- Results are for one store.
