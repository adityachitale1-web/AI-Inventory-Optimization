"""Builds the self-contained interactive dashboard (one HTML string) from inventory_engine results.

The page embeds Chart.js and all data, so it works inside Streamlit and as a standalone,
offline HTML file. Safety stock, Monte Carlo, inventory simulation, Bayesian optimisation and
scenarios are recalculated in the browser whenever inputs are edited in the side panel.
"""
import datetime
import html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from inventory_engine import GLOBAL_DEFAULTS, HORIZON

CHART_JS_PATH = Path(__file__).parent / "assets" / "chart.umd.js"
esc = html.escape


def _f(x, d=0): return f"{x:,.{d}f}"
def _pct(x, d=1): return f"{x:.{d}f}%"
def _vis(c): return re.sub(r"<[^>]+>", "", str(c)).replace("&minus;", "-").replace("&mdash;", "").strip()
def _isnum(c): return _vis(c) == "" or bool(re.fullmatch(r"[-+~]?[\d,]+(\.\d+)?\s*(%|pts)?( / [\d,]+)?", _vis(c)))


def _table(headers, rows):
    cl = []
    for j in range(len(headers)):
        col = [r[j] for r in rows]
        num = j > 0 and np.mean([_isnum(c) for c in col]) >= 0.5
        wrap = np.mean([len(_vis(c)) for c in col]) > 34
        cl.append(("" if num else "l") + (" w" if wrap else ""))
    th = "".join(f'<th class="{cl[j]}">{h}</th>' for j, h in enumerate(headers))
    tr = "".join("<tr>" + "".join(f'<td class="{cl[j]}">{c}</td>' for j, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="tbl-wrap"><table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


FEATURE_NAMES = {"level": "Recent demand level (28-day avg)", "dow": "Day of week", "sin": "Seasonality (sin)",
                 "cos": "Seasonality (cos)", "promo": "Promotions", "oil_price": "Oil price (external)",
                 "holiday": "Holiday (external)"}


def build_dashboard_html(A, data_label, data_desc, chart_js=None):
    """A = inventory_engine.run_analysis(...) output. Returns the full HTML page as a string."""
    chart_js = chart_js if chart_js is not None else CHART_JS_PATH.read_text(encoding="utf-8")
    names, full, factors, cutoff = A["names"], A["full"], A["factors"], A["cutoff"]
    R = A["results"]

    prod_payload, profile = {}, {}
    for n in names:
        r0, r1, sup = R[n]["r0"], R[n]["r1"], R[n]["supply"]
        allp = full[full["product"] == n].set_index("date").sort_index()
        d = allp["demand"]
        fc_dates = pd.date_range(cutoff + pd.Timedelta(days=1), periods=HORIZON)
        fc_actual = d.reindex(fc_dates)
        w = d.resample("W").sum().iloc[1:-1]
        prof = {"Days of data": len(d), "First date": d.index.min().strftime("%d %b %Y"),
                "Last date": d.index.max().strftime("%d %b %Y"), "Mean demand / day": _f(d.mean()),
                "Std deviation": _f(d.std()), "Min / Max": f"{_f(d.min())} / {_f(d.max())}",
                "Coefficient of variation": _pct(d.std() / max(d.mean(), 1e-9) * 100),
                "Zero-sales days": int((d == 0).sum()), "Days with promotions": _pct((allp["promo"] > 0).mean() * 100, 0)}
        for c in factors:
            if c == "promo":
                continue
            vals = allp[c]
            if set(vals.dropna().unique()) <= {0, 1}:
                prof[f"{esc(c.replace('_', ' ').capitalize())} days"] = int(vals.sum())
            else:
                prof[f"Avg {esc(c.replace('_', ' '))}"] = _f(vals.mean(), 1)
        for c in factors:
            cor = d.corr(allp[c]) if allp[c].std() > 0 else float("nan")
            prof[f"Correlation: demand vs {esc(c.replace('_', ' '))}"] = "&mdash;" if np.isnan(cor) else _f(cor, 2)
        profile[n] = prof
        cap = sup.get("cap", R[n]["D"] * (2.5 * sup["L"] + 3))
        prod_payload[n] = {
            "defaults": {"L": int(round(sup["L"])), "sL": float(sup["sigma_L"]), "h": float(sup["h"]),
                         "cap": int(round(cap)), "sd": round(float(r0["sigma_D"]), 1),
                         "fc": [int(round(x)) for x in r0["forecast"]]},
            "sigma_raw": r0["sigma_raw"], "err_static": float(r0["err_static"]), "err_updated": float(r0["err_updated"]),
            "fc_dates": fc_dates.strftime("%a %d %b").tolist(),
            "fc_actual": [None if pd.isna(v) else float(round(v)) for v in fc_actual],
            "after": {"fc": [float(x) for x in r1["forecast"]], "sd": float(r1["sigma_D"]), "err": float(r1["err_updated"])},
            "bt_labels": pd.to_datetime(r0["dates"]).strftime("%d %b").tolist(),
            "bt_actual": np.round(r0["actual"], 0).tolist(), "bt_fc": np.round(r0["backtest"], 0).tolist(),
            "weekly": {"labels": w.index.strftime("%Y-%m-%d").tolist(), "values": w.round(0).tolist()},
            "dow": d.groupby(d.index.dayofweek).mean().reindex(range(7)).fillna(0).round(0).tolist(),
        }
    payload = {"names": names, "newDays": A["new_days"], "G": dict(GLOBAL_DEFAULTS), "P": prod_payload}

    acc_rows = []
    for n in names:
        r = R[n]["r0"]
        imp = r["err_static"] - r["err_updated"]
        acc_rows.append([f"<b>{esc(n)}</b>", _pct(r["err_static"]), _pct(r["err_updated"]),
                         f'<span class="good">&minus;{imp:.1f} pts</span>' if imp >= 0 else f"+{-imp:.1f} pts",
                         _f(r["sigma_raw"]), _f(r["sigma_D"]), _pct((1 - r["sigma_D"] / max(r["sigma_raw"], 1e-9)) * 100, 0)])
    acc = _table(["Product", "Error: static model", "Error: re-trained weekly", "Improvement", "Raw demand std",
                  "Forecast-error std (&sigma;<sub>D</sub>)", "Uncertainty removed"], acc_rows)
    fi_rows = [[FEATURE_NAMES.get(ft, esc(ft.replace("_", " ").capitalize()))] + [_pct(R[n]["r0"]["importance"][ft]) for n in names]
               for ft in A["features"]]
    fi_rows.sort(key=lambda r: -np.mean([float(x.rstrip("%")) for x in r[1:]]))
    fi = _table(["Demand driver"] + [esc(n) for n in names], fi_rows)
    pkeys = list(dict.fromkeys(k for n in names for k in profile[n]))
    prof = _table(["Metric"] + [esc(n) for n in names], [[k] + [str(profile[n].get(k, "&mdash;")) for n in names] for k in pkeys])
    gloss = _table(["Term", "Meaning"], [
        ["Reorder point (ROP)", "Inventory position (on hand + on order) at which a new order is placed."],
        ["Safety stock (SS)", "Buffer above expected lead-time demand that protects against uncertainty."],
        ["EOQ", "Economic order quantity, &radic;(2&middot;D&middot;order cost / holding cost): balances ordering and holding cost."],
        ["Fill rate", "Share of customer demand served from stock."],
        ["Cycle service level", "Probability of no stockout during a replenishment cycle (what Z targets)."],
        ["&sigma;<sub>D</sub>", "Std of the forecast error: the demand uncertainty left after forecasting."],
        ["Expected improvement", "Rule the Bayesian search uses to pick the next reorder point to test."]])
    drivers = ", ".join(FEATURE_NAMES.get(c, c.replace("_", " ")).split(" (")[0].lower() for c in factors)

    body = (BODY.replace("%%RANGE%%", f'{full["date"].min():%d %b %Y} &ndash; {full["date"].max():%d %b %Y}')
            .replace("%%PRODS%%", ", ".join(esc(n) for n in names))
            .replace("%%GEN%%", datetime.datetime.now().strftime("%d %b %Y"))
            .replace("%%DATALABEL%%", esc(data_label)).replace("%%DATADESC%%", data_desc)
            .replace("%%DRIVERS%%", esc(drivers))
            .replace("%%ACC%%", acc).replace("%%FI%%", fi).replace("%%PROF%%", prof).replace("%%GLOSS%%", gloss))
    return ("<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            "<title>AI Inventory Optimization Dashboard</title><style>" + CSS + "</style></head><body>" + body
            + "<script>" + chart_js + "</script><script>"
            + JS.replace("__DATA__", json.dumps(payload).replace("</", "<\\/")) + "</script></body></html>")


# ---------------------------------------------------------------------------
# Page templates
# ---------------------------------------------------------------------------
CSS = r"""
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;background:#f4f6fa;color:#1f2937;font-size:14px;line-height:1.55}
.wrap{max-width:1240px;margin:0 auto;padding:24px 22px 50px;transition:margin .25s}
@media(min-width:1500px){body.dopen .wrap{margin-right:400px;margin-left:12px}}
header{background:#0f2a5c;color:#fff;border-radius:14px;padding:24px 28px}
header h1{font-size:22px;font-weight:700;letter-spacing:-.2px}
header p{opacity:.85;margin-top:4px;font-size:13.5px}
.chips{margin-top:14px;display:flex;flex-wrap:wrap;gap:8px}
.chips span{background:rgba(255,255,255,.12);border-radius:16px;padding:4px 12px;font-size:12px}
.banner{display:none;margin-top:14px;background:#fffbeb;border:1px solid #fcd34d;color:#78350f;border-radius:10px;padding:10px 14px;font-size:13px;align-items:center;gap:12px}
.banner.on{display:flex}
.banner button{margin-left:auto;background:#fff;border:1px solid #f59e0b;color:#92400e;border-radius:8px;padding:5px 12px;cursor:pointer;font-weight:600}
nav.tabs{display:flex;gap:4px;margin-top:18px;border-bottom:2px solid #dbe1ea;flex-wrap:wrap}
nav.tabs button{background:none;border:none;padding:11px 16px;font-size:14px;font-weight:600;color:#64748b;cursor:pointer;border-bottom:3px solid transparent;margin-bottom:-2px}
nav.tabs button:hover{color:#0f2a5c}
nav.tabs button.on{color:#0f2a5c;border-bottom-color:#2563eb}
.pbar{display:flex;align-items:center;gap:8px;margin:16px 0 4px;flex-wrap:wrap}
.pbar .lab{font-size:12.5px;color:#64748b;margin-right:4px}
.pills button{border:1px solid #cbd5e1;background:#fff;border-radius:18px;padding:6px 14px;font-size:13px;cursor:pointer;color:#334155}
.pills button.on{background:#2563eb;border-color:#2563eb;color:#fff}
.pills{display:flex;gap:8px;flex-wrap:wrap}
section{display:none;padding-top:10px}
section.on{display:block}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:6px 0 16px}
.kpi{background:#fff;border-radius:12px;padding:16px 18px;border:1px solid #e5e9f0}
.kv{font-size:26px;font-weight:700;color:#0f2a5c}
.kl{font-size:12.5px;color:#64748b;margin-top:2px}
.card{background:#fff;border:1px solid #e5e9f0;border-radius:12px;padding:18px 20px;margin-bottom:16px;min-width:0}
.card h2{font-size:15px;font-weight:700;color:#0f2a5c}
.sub{font-size:12.5px;color:#64748b;margin:3px 0 12px}
.grid2,.g23,.g32{display:grid;gap:16px}
.grid2{grid-template-columns:1fr 1fr}.g23{grid-template-columns:2fr 3fr}.g32{grid-template-columns:3fr 2fr}
.ch{position:relative;height:300px}.ch.tall{height:340px}.ch.short{height:200px;margin-top:14px}
.tbl-wrap{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:13px}
th{background:#f1f4f9;color:#334155;font-weight:600;text-align:right;padding:8px 10px;border-bottom:1px solid #dbe1ea;white-space:normal;vertical-align:bottom;line-height:1.3}
td{padding:7px 10px;text-align:right;border-bottom:1px solid #eef1f5;font-variant-numeric:tabular-nums;white-space:nowrap}
.l{text-align:left}.w{white-space:normal;min-width:240px}
tbody tr:nth-child(even){background:#fafbfd}
tbody tr:hover{background:#eef4ff}
th small{font-weight:400;color:#64748b}
.good{color:#15803d;font-weight:600}.hl{color:#1d4ed8}.edv{color:#b45309;font-weight:700}
.formula{font-family:Georgia,serif;font-style:italic;background:#f1f4f9;padding:1px 6px;border-radius:4px}
.note{font-size:12.5px;color:#475569;background:#fffbeb;border:1px solid #fde68a;border-radius:8px;padding:9px 12px;margin-top:12px}
.steps{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:10px}
.step{display:flex;gap:12px;background:#f8fafc;border:1px solid #e5e9f0;border-radius:10px;padding:12px 14px;font-size:13px}
.sn{flex:0 0 28px;height:28px;border-radius:50%;background:#2563eb;color:#fff;font-weight:700;display:flex;align-items:center;justify-content:center}
ul.lim{margin:8px 0 0 18px;font-size:13px}ul.lim li{margin:4px 0}
footer{text-align:center;color:#94a3b8;font-size:12px;margin-top:20px}
/* ---- side panel ---- */
.dtab{position:fixed;right:0;top:38%;z-index:40;writing-mode:vertical-rl;background:#2563eb;color:#fff;border:none;border-radius:10px 0 0 10px;padding:16px 10px;font-size:13.5px;font-weight:700;letter-spacing:.3px;cursor:pointer;box-shadow:-4px 4px 16px rgba(37,99,235,.35)}
.dtab:hover{background:#1d4ed8}
body.dopen .dtab{display:none}
.drawer{position:fixed;top:0;right:0;height:100vh;width:390px;max-width:94vw;background:#fff;z-index:50;display:flex;flex-direction:column;box-shadow:-10px 0 30px rgba(15,42,92,.18);transform:translateX(105%);transition:transform .25s}
body.dopen .drawer{transform:none}
.dhead{display:flex;align-items:center;justify-content:space-between;padding:16px 18px;background:#0f2a5c;color:#fff}
.dhead b{font-size:15px}.dhead small{display:block;opacity:.8;font-size:11.5px;font-weight:400;margin-top:2px}
.dhead button{background:none;border:none;color:#fff;font-size:24px;line-height:1;cursor:pointer;padding:0 4px}
.dbody{flex:1;overflow-y:auto;padding:6px 18px 18px}
.dsec{border-bottom:1px solid #e5e9f0;padding:14px 0}
.dsec:last-child{border-bottom:none}
.dsec h4{font-size:13px;text-transform:uppercase;letter-spacing:.5px;color:#0f2a5c;margin-bottom:10px}
.dsec h5{font-size:12.5px;color:#334155;margin:14px 0 4px}
.fld{display:flex;align-items:center;justify-content:space-between;gap:10px;margin:7px 0}
.fld span{font-size:13px;color:#334155}
.fld span small{display:block;font-size:11px;color:#94a3b8}
.fld input,.dp input{width:104px;padding:6px 8px;border:1px solid #cbd5e1;border-radius:7px;font-size:13px;text-align:right;font-variant-numeric:tabular-nums;background:#fff}
.dp input{width:84px}
input.ed{border-color:#f59e0b;background:#fffbeb}
input.bad{border-color:#dc2626;background:#fef2f2}
.dp{width:100%;font-size:12.5px;margin-top:4px}
.dp td,.dp th{padding:4px 6px}
.dp th{background:#f8fafc}
.dhint{font-size:11.5px;color:#64748b;margin-top:6px}
.dfoot{border-bottom:1px solid #e5e9f0;padding:12px 18px;background:#f8fafc}
.dstat{font-size:12px;color:#64748b;margin-top:9px;min-height:16px;order:2}
.dfoot{display:flex;flex-direction:column}.dbtns{order:1}
.dstat.err{color:#dc2626;font-weight:600}
.dbtns{display:flex;gap:8px}
.btn{border:1px solid #cbd5e1;background:#fff;border-radius:8px;padding:8px 12px;font-size:13px;font-weight:600;cursor:pointer;color:#334155}
.btn.pri{background:#2563eb;border-color:#2563eb;color:#fff;flex:1}
.btn.pri:disabled{background:#93c5fd;border-color:#93c5fd;cursor:not-allowed}
.busy{position:fixed;inset:0;background:rgba(244,246,250,.55);z-index:60;display:none;align-items:center;justify-content:center}
.busy.on{display:flex}
.busy div{background:#0f2a5c;color:#fff;padding:12px 20px;border-radius:10px;font-weight:600;font-size:14px;box-shadow:0 8px 24px rgba(0,0,0,.2)}
@media(max-width:900px){.grid2,.g23,.g32,.steps{grid-template-columns:1fr}.kpis{grid-template-columns:repeat(2,1fr)}}
"""

BODY = r"""
<div class="wrap">
<header><h1>AI-Enhanced Reorder Point &amp; Inventory Optimization</h1>
<p>Predictive demand forecasting &middot; dynamic safety stock &middot; Bayesian reorder-point optimisation</p>
<div class="chips"><span>Data: %%DATALABEL%%</span><span>%%RANGE%%</span>
<span>Products: %%PRODS%%</span><span>Service level: <b class="v_sl"></b></span><span>Generated %%GEN%%</span></div></header>
<div class="banner" id="banner"><span>&#9998; Results reflect <b id="nedits"></b> edited input(s). Python-model values are shown in the side panel for comparison.</span><button id="bannerReset">Reset all</button></div>
<nav class="tabs">
 <button data-t="cockpit" class="on">Decision Cockpit</button><button data-t="forecast">Demand Forecast</button>
 <button data-t="safety">Safety Stock &amp; Monte Carlo</button><button data-t="bayes">Sensitivity &amp; Bayesian Optimisation</button>
 <button data-t="data">Data &amp; Methodology</button></nav>
<div class="pbar" id="pbar"><span class="lab">Product shown in charts &amp; product tables:</span><div class="pills" id="pills"></div></div>

<section id="cockpit" class="on" data-pp="0">
 <div class="kpis" id="kpis"></div>
 <div class="card"><h2>Recommended reorder policy</h2><p class="sub">What to do now, per product, using the AI-optimised reorder point.</p><div id="t_rec"></div></div>
 <div class="g23">
  <div class="card"><h2>Daily cost by policy</h2><p class="sub">Holding + stockout + ordering cost per day, simulated over 300 scenarios &times; 120 days.</p><div class="ch"><canvas id="c_cost"></canvas></div></div>
  <div class="card"><h2>Policy comparison</h2><p class="sub">Lowest-cost policy per product highlighted.</p><div id="t_cmp"></div></div>
 </div>
 <div class="card"><h2>Adaptive update: impact of <span class="v_nd"></span> new days of data</h2><p class="sub">The last <span class="v_nd"></span> days were held back, then appended; the ML forecast was re-run and the reorder point re-optimised with the current supply inputs.</p><div id="t_upd"></div></div>
</section>

<section id="forecast" data-pp="1">
 <div class="card"><h2>Forecast accuracy</h2><p class="sub">Back-test on the last 90 days (from the Python ML model). Error = absolute forecast error as a share of actual volume. The forecast error, not raw demand variability, drives the safety stock.</p>%%ACC%%</div>
 <div class="card"><h2>Actual vs forecast &mdash; <span class="pname"></span></h2><p class="sub">90-day back-test; the model is re-trained every 7 days as new data arrives.</p><div class="ch tall"><canvas id="c_bt"></canvas></div></div>
 <div class="grid2">
  <div class="card"><h2>Next 7 days: forecast vs actual &mdash; <span class="pname"></span></h2><p class="sub">Forecast made at the cutoff date vs the held-back days the model never saw. &ldquo;Value used&rdquo; is what drives the calculations (editable in the side panel).</p><div id="t_fc7"></div><div class="ch short"><canvas id="c_fc7"></canvas></div></div>
  <div class="card"><h2>What drives demand</h2><p class="sub">Feature importance from the gradient-boosting model (share of predictive power).</p>%%FI%%</div>
 </div>
</section>

<section id="safety" data-pp="1">
 <div class="card"><h2>Safety stock calculation</h2><p class="sub">Formula: <span class="formula">SS = Z &middot; &radic;(&sigma;<sub>D</sub>&sup2;&middot;L + D&sup2;&middot;&sigma;<sub>L</sub>&sup2;)</span> &nbsp;vs a 50,000-scenario Monte Carlo that samples random lead times (gamma, skewed toward delays) and demand during each.</p><div id="t_ss"></div></div>
 <div class="grid2">
  <div class="card"><h2>Simulated lead-time demand &mdash; <span class="pname"></span></h2><p class="sub">Blue: covered by the formula reorder point &middot; amber: extra cover added by Monte Carlo &middot; red: stockout risk beyond both.</p><div class="ch"><canvas id="c_hist"></canvas></div></div>
  <div class="card"><h2>Stockout risk by reorder point &mdash; <span class="pname"></span></h2><p class="sub">From the 50,000 Monte Carlo scenarios.</p><div id="t_risk"></div></div>
 </div>
 <div class="card"><h2>Safety stock vs service level</h2><p class="sub">How much buffer each service level costs. The gap between formula and Monte Carlo widens at high service levels, where late deliveries dominate.</p><div id="t_sl"></div></div>
</section>

<section id="bayes" data-pp="1">
 <div class="card"><h2>Bayesian optimisation vs manual parameter sweep</h2><p class="sub">Objective: minimise daily cost (holding + stockout + ordering) subject to a <span class="v_sl"></span> fill rate and warehouse capacity. Each run is a 300 &times; 120-day inventory simulation.</p><div id="t_eff"></div></div>
 <div class="grid2">
  <div class="card"><h2>Cost curve and search path &mdash; <span class="pname"></span></h2><p class="sub">Grey line: 60-run manual sweep. Blue dots: the 20 points the Bayesian search chose. Green: optimum.</p><div class="ch"><canvas id="c_bo"></canvas></div></div>
  <div class="card"><h2>Sensitivity of the optimal reorder point &mdash; <span class="pname"></span></h2><p class="sub">Change vs base case, Bayesian optimisation re-run for each scenario.</p><div class="ch"><canvas id="c_sens"></canvas></div></div>
 </div>
 <div class="g32">
  <div class="card"><h2>Scenario analysis &mdash; <span class="pname"></span></h2><p class="sub">Each row is a fresh optimisation under the changed assumption.</p><div id="t_sens"></div></div>
  <div class="card"><h2>Bayesian search log &mdash; <span class="pname"></span></h2><p class="sub">5 evenly spaced starting points, then 15 points chosen by expected improvement on a Gaussian-process model.</p><div id="t_log"></div><p class="sub" style="margin-top:8px">* includes the penalty when the fill-rate target is missed.</p></div>
 </div>
</section>

<section id="data" data-pp="1">
 <div class="card"><h2>Dataset</h2><p class="sub">%%DATADESC%%</p>%%PROF%%</div>
 <div class="grid2">
  <div class="card"><h2>Weekly demand history &mdash; <span class="pname"></span></h2><div class="ch"><canvas id="c_weekly"></canvas></div></div>
  <div class="card"><h2>Average demand by weekday</h2><div class="ch"><canvas id="c_dow"></canvas></div></div>
 </div>
 <div class="card"><h2>Methodology</h2>
 <div class="steps">
  <div class="step"><div class="sn">1</div><div><b>Demand forecasting (Python).</b> Gradient-boosting regression on recent demand level, weekday, yearly seasonality and %%DRIVERS%%. Re-trained weekly in a rolling back-test; new data can be appended and the pipeline recomputes.</div></div>
  <div class="step"><div class="sn">2</div><div><b>Dynamic safety stock.</b> The forecast error &sigma;<sub>D</sub> feeds <span class="formula">SS = Z&middot;&radic;(&sigma;<sub>D</sub>&sup2;L + D&sup2;&sigma;<sub>L</sub>&sup2;)</span>. A 50,000-scenario Monte Carlo samples random lead times and demand to read the buffer from the simulated distribution.</div></div>
  <div class="step"><div class="sn">3</div><div><b>Bayesian optimisation.</b> A Gaussian-process surrogate of cost vs reorder point, searched by expected improvement, replaces a manual parameter sweep. Constraints: fill rate &ge; <span class="v_sl"></span> (penalty) and capacity (search bound).</div></div>
  <div class="step"><div class="sn">4</div><div><b>Interactive recalculation.</b> Steps 2&ndash;3 and the scenario analysis re-run in your browser whenever inputs are edited in the side panel, so the dashboard responds without Python.</div></div>
 </div></div>
 <div class="card"><h2>Supply assumptions per product</h2><p class="sub">The dataset holds demand only, so these are inputs &mdash; edit them in the side panel.</p><div id="t_assum"></div></div>
 <div class="grid2">
  <div class="card"><h2>Global settings</h2><div id="t_glob"></div></div>
  <div class="card"><h2>Glossary</h2>%%GLOSS%%
   <h2 style="margin-top:18px">Limitations</h2><ul class="lim">
   <li>The dataset contains demand only; lead times and costs are assumptions.</li>
   <li>The formula targets a <span class="v_sl"></span> <i>cycle</i> service level; the Bayesian search targets a <span class="v_sl"></span> <i>fill rate</i>, a looser target, so part of its saving reflects that.</li>
   <li>Editing inputs re-runs the inventory maths in the browser; the ML forecast itself is only re-trained in Python.</li>
   <li>Results are for one store; other stores would need their own run.</li></ul></div>
 </div>
</section>
<footer>Self-contained dashboard &mdash; opens in any web browser, no installation or internet needed. Forecasting in Python (scikit-learn); interactive maths and charts in the browser (Chart.js).</footer>
</div>

<button class="dtab" id="dtab">&#9998; Edit inputs &amp; data</button>
<aside class="drawer" id="drawer">
 <div class="dhead"><div><b>Edit inputs &amp; data points</b><small>Change values, then press Recalculate</small></div><button id="dclose" title="Close">&times;</button></div>
 <div class="dfoot"><div class="dstat" id="dstat"></div>
  <div class="dbtns"><button class="btn pri" id="recalc">Recalculate</button><button class="btn" id="resetP">Reset product</button><button class="btn" id="resetA">Reset all</button></div></div>
 <div class="dbody">
  <div class="dsec"><h4>Global settings</h4>
   <label class="fld"><span>Service level target (%)<small id="d_g_sl"></small></span><input type="number" step="0.5" min="50" max="99.9" data-g="sl"></label>
   <label class="fld"><span>Fixed cost per order<small id="d_g_oc"></small></span><input type="number" step="50" min="1" data-g="oc"></label>
   <label class="fld"><span>Cost per unit of lost sales<small id="d_g_so"></small></span><input type="number" step="1" min="0" data-g="so"></label>
  </div>
  <div class="dsec"><h4>Product inputs</h4>
   <div class="pills" id="dpills"></div>
   <label class="fld" style="margin-top:12px"><span>Lead time (days, 1&ndash;6)<small data-dd="L"></small></span><input type="number" step="1" min="1" max="6" data-k="L"></label>
   <label class="fld"><span>Lead-time std (days)<small data-dd="sL"></small></span><input type="number" step="0.1" min="0" max="5" data-k="sL"></label>
   <label class="fld"><span>Holding cost / unit / day<small data-dd="h"></small></span><input type="number" step="0.05" min="0.01" data-k="h"></label>
   <label class="fld"><span>Warehouse capacity (units)<small data-dd="cap"></small></span><input type="number" step="50" min="1" data-k="cap"></label>
   <label class="fld"><span>Forecast-error std &sigma;<sub>D</sub><small data-dd="sd"></small></span><input type="number" step="1" min="0" data-k="sd"></label>
   <h5>Demand data points &mdash; forecast units/day</h5>
   <table class="dp"><thead><tr><th class="l">Date</th><th>ML model</th><th>Your value</th><th>Actual</th></tr></thead><tbody id="dp_rows"></tbody></table>
   <p class="dhint">The average of the first L days (marked <span style="color:#2563eb">&bull;</span>) sets the demand rate D used by every calculation. &ldquo;Actual&rdquo; is the held-back real demand, shown for reference.</p>
  </div>
 </div>
</aside>
<div class="busy" id="busy"><div>Recalculating&hellip;</div></div>
"""

JS = r"""
const PD = __DATA__;
const NAMES = PD.names;
const C = {blue:'#2563eb', lblue:'#93c5fd', green:'#16a34a', red:'#dc2626', amber:'#f59e0b', grey:'#94a3b8', navy:'#0f2a5c'};
const clone = o => JSON.parse(JSON.stringify(o));
const DEF = {G: clone(PD.G), S: Object.fromEntries(NAMES.map(n => [n, clone(PD.P[n].defaults)]))};
let ST = clone(DEF);          // current inputs
let RES = {};                 // computed results per product
let cur = 0, curTab = 'cockpit';
const KEY = 'inv-dashboard-inputs-v1';
try { const s = JSON.parse(localStorage.getItem(KEY) || 'null');
  if (s && s.G && s.S && NAMES.every(n => s.S[n] && s.S[n].fc && s.S[n].fc.length === DEF.S[n].fc.length)) ST = s; } catch (e) {}

// ================= maths =================
function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;}}
function makeRng(seed){
  const u = mulberry32(seed); let spare = null;
  const rnd = () => { let x; do { x = u(); } while (x === 0); return x; };
  const normal = () => { if (spare !== null) { const s = spare; spare = null; return s; }
    const r = Math.sqrt(-2 * Math.log(rnd())), th = 2 * Math.PI * u(); spare = r * Math.sin(th); return r * Math.cos(th); };
  const gamma = k => { if (k < 1) return gamma(k + 1) * Math.pow(rnd(), 1 / k);
    const d = k - 1 / 3, c = 1 / Math.sqrt(9 * d);
    for (;;) { let x, v; do { x = normal(); v = 1 + c * x; } while (v <= 0); v = v * v * v; const uu = rnd();
      if (uu < 1 - 0.0331 * x * x * x * x || Math.log(uu) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v; } };
  return {normal, gamma};
}
const leadSample = (g, L, sL) => sL <= 0 ? L : g.gamma((L / sL) ** 2) * (sL * sL / L);
function normInv(p){const a=[-39.69683028665376,220.9460984245205,-275.9285104469687,138.357751867269,-30.66479806614716,2.506628277459239],b=[-54.47609879822406,161.5858368580409,-155.6989798598866,66.80131188771972,-13.28068155288572],c=[-0.007784894002430293,-0.3223964580411365,-2.400758277161838,-2.549732539343734,4.374664141464968,2.938163982698783],d=[0.007784695709041462,0.3224671290700398,2.445134137142996,3.754408661907416],pl=0.02425;let q,r;
 if(p<pl){q=Math.sqrt(-2*Math.log(p));return(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)}
 if(p>1-pl){q=Math.sqrt(-2*Math.log(1-p));return-(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)}
 q=p-0.5;r=q*q;return(((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q/(((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)}
const normPdf = x => Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
function normCdf(x){const t=1/(1+0.2316419*Math.abs(x)),y=t*(0.319381530+t*(-0.356563782+t*(1.781477937+t*(-1.821255978+t*1.330274429))));const p=1-normPdf(x)*y;return x>=0?p:1-p}
function quantile(s, q){ const pos = (s.length - 1) * q, lo = Math.floor(pos), hi = Math.ceil(pos); return s[lo] + (s[hi] - s[lo]) * (pos - lo); }
function monteCarlo(D, sd, L, sL, n){ const g = makeRng(7), s = new Float64Array(n); let m = 0;
  for (let i = 0; i < n; i++) { const lead = leadSample(g, L, sL); s[i] = D * lead + sd * Math.sqrt(lead) * g.normal(); m += s[i]; }
  s.sort(); return {s, mean: m / n}; }
function tailStats(s, rop){ let lo = 0, hi = s.length; while (lo < hi) { const mid = (lo + hi) >> 1; if (s[mid] <= rop) lo = mid + 1; else hi = mid; }
  let short = 0; for (let i = lo; i < s.length; i++) short += s[i] - rop; return {p: (s.length - lo) / s.length, short: short / s.length}; }

const PATHS = 300, DAYS = 120;
function simArrays(D, sd, L, sL){ const g = makeRng(1), n = PATHS * DAYS, dem = new Float64Array(n), lead = new Int32Array(n); let maxL = 1, tot = 0;
  for (let i = 0; i < n; i++) { const v = D + sd * g.normal(); dem[i] = v > 0 ? v : 0; tot += dem[i]; }
  for (let i = 0; i < n; i++) { const l = Math.max(1, Math.round(leadSample(g, L, sL))); lead[i] = l; if (l > maxL) maxL = l; }
  return {dem, lead, maxL, tot}; }
function simulate(rop, Q, A, h, so, oc){ let hold = 0, lost = 0, orders = 0; const buf = new Float64Array(DAYS + A.maxL + 1);
  for (let p = 0; p < PATHS; p++) { buf.fill(0); let stock = rop + Q, on = 0; const o = p * DAYS;
    for (let t = 0; t < DAYS; t++) { const a = buf[t]; stock += a; on -= a; const d = A.dem[o + t], sold = stock < d ? stock : d;
      stock -= sold; lost += d - sold; if (stock + on <= rop) { buf[t + A.lead[o + t]] += Q; on += Q; orders++; } hold += stock; } }
  return {cost: (h * hold + so * lost + oc * orders) / (PATHS * DAYS), fill: 1 - lost / Math.max(A.tot, 1e-9)}; }

// Gaussian process (Matern 5/2 + noise; hyper-parameters by marginal likelihood) + expected improvement
const matern = (r, l) => { const s = Math.sqrt(5) * r / l; return (1 + s + s * s / 3) * Math.exp(-s); };
function chol(K, n){ const L = new Float64Array(n * n); for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) { let s = K[i*n+j];
  for (let k = 0; k < j; k++) s -= L[i*n+k] * L[j*n+k]; if (i === j) { if (s <= 0) return null; L[i*n+i] = Math.sqrt(s); } else L[i*n+j] = s / L[j*n+j]; } return L; }
function solveL(L, n, b){ const x = new Float64Array(n); for (let i = 0; i < n; i++) { let s = b[i]; for (let k = 0; k < i; k++) s -= L[i*n+k] * x[k]; x[i] = s / L[i*n+i]; } return x; }
function solveLT(L, n, b){ const x = new Float64Array(n); for (let i = n - 1; i >= 0; i--) { let s = b[i]; for (let k = i + 1; k < n; k++) s -= L[k*n+i] * x[k]; x[i] = s / L[i*n+i]; } return x; }
function gpFit(xs, ys){ const n = xs.length, ym = ys.reduce((a, b) => a + b, 0) / n;
  const sd = Math.sqrt(ys.reduce((a, v) => a + (v - ym) ** 2, 0) / n) || 1, y = ys.map(v => (v - ym) / sd); let best = null;
  for (const l of [0.05,0.08,0.12,0.18,0.27,0.4,0.6,0.9,1.3,2]) for (const nz of [1e-6,1e-4,1e-3,1e-2,0.1]) for (const Cc of [0.5,1,2]) {
    const K = new Float64Array(n * n); for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) K[i*n+j] = Cc * matern(Math.abs(xs[i] - xs[j]), l) + (i === j ? nz + 1e-10 : 0);
    const Lc = chol(K, n); if (!Lc) continue; const alpha = solveLT(Lc, n, solveL(Lc, n, y)); let lml = 0;
    for (let i = 0; i < n; i++) lml += -0.5 * y[i] * alpha[i] - Math.log(Lc[i*n+i]);
    if (!best || lml > best.lml) best = {lml, l, nz, Cc, Lc, alpha}; }
  return Object.assign(best, {xs, n, ym, sd}); }
function gpPredict(g, x){ const k = new Float64Array(g.n); for (let i = 0; i < g.n; i++) k[i] = g.Cc * matern(Math.abs(x - g.xs[i]), g.l);
  let mu = 0; for (let i = 0; i < g.n; i++) mu += k[i] * g.alpha[i]; const v = solveL(g.Lc, g.n, k); let vv = 0; for (const a of v) vv += a * a;
  return {mu: mu * g.sd + g.ym, sd: Math.sqrt(Math.max(g.Cc + g.nz - vv, 1e-12)) * g.sd}; }
function bayesOpt(obj, lo, hi){ const X = [], Y = [];
  for (let i = 0; i < 5; i++) { const x = lo + (hi - lo) * i / 4; X.push(x); Y.push(obj(x)); }
  for (let it = 0; it < 15; it++) { const g = gpFit(X.map(x => (x - lo) / (hi - lo)), Y), yb = Math.min(...Y); let be = -Infinity, bx = lo;
    for (let k = 0; k < 400; k++) { const u = k / 399, {mu, sd} = gpPredict(g, u), imp = yb - mu, z = imp / (sd + 1e-9), ei = imp * normCdf(z) + sd * normPdf(z);
      if (ei > be) { be = ei; bx = lo + u * (hi - lo); } }
    X.push(bx); Y.push(obj(bx)); }
  return {X, Y}; }

// ================= model per product =================
function evalProduct(n, mods = {}, full = false){
  const p = ST.S[n], G = ST.G;
  const L = mods.L ?? p.L, sL = mods.sL ?? p.sL, h = mods.h ?? p.h, so = mods.so ?? G.so, sl = mods.sl ?? G.sl, dm = mods.dm ?? 1;
  const fc = mods.fc ?? p.fc, nd = Math.max(1, Math.min(fc.length, Math.ceil(L)));
  const D = Math.max(fc.slice(0, nd).reduce((a, b) => a + b, 0) / nd, 1) * dm, sd = (mods.sd ?? p.sd) * dm;
  const Q = Math.sqrt(2 * D * G.oc / h), cap = p.cap * (mods.capMult ?? 1);
  const A = simArrays(D, sd, L, sL), sim = r => simulate(r, Q, A, h, so, G.oc);
  const pen = r => { const s = sim(r); return s.cost + 50 * D * Math.max(0, sl - s.fill); };
  const lo = 0.5 * D * L, hi = Math.max(cap - Q, 1.5 * lo);
  const bo = bayesOpt(pen, lo, hi); let bi = 0; bo.Y.forEach((v, i) => { if (v < bo.Y[bi]) bi = i; });
  const r = {D, sd, L, sL, h, Q, cap, lo, hi, rop: bo.X[bi], bo, polB: sim(bo.X[bi]), capLimited: cap - Q < 1.5 * lo};
  if (!full) return r;
  r.z = normInv(G.sl); r.ssF = r.z * Math.sqrt(sd * sd * L + D * D * sL * sL);
  r.mc = monteCarlo(D, sd, L, sL, 50000); r.ssMC = quantile(r.mc.s, G.sl) - r.mc.mean;
  r.ropF = D * L + r.ssF; r.ropMC = D * L + r.ssMC; r.polF = sim(r.ropF); r.polMC = sim(r.ropMC);
  r.sweep = []; for (let i = 0; i < 60; i++) { const x = lo + (hi - lo) * i / 59; r.sweep.push({x, y: pen(x)}); }
  const p0 = ST.S[n], slUp = Math.min(G.sl + 0.03, 0.999);
  r.scen = [["Base case", r.rop, r.polB.cost, r.polB.fill]];
  for (const [lab, md] of [["Lead time +1 day", {L: p0.L + 1, capMult: (p0.L + 1) / p0.L}], ["Lead-time variability ×2", {sL: Math.max(p0.sL * 2, 0.2)}],
      ["Demand +20%", {dm: 1.2, capMult: 1.2}], ["Holding cost +50%", {h: p0.h * 1.5}], ["Stockout cost ×2", {so: G.so * 2}],
      [`Fill-rate target ${(slUp * 100).toFixed(1).replace('.0', '')}%`, {sl: slUp}]]) {
    const s = evalProduct(n, md); r.scen.push([lab, s.rop, s.polB.cost, s.polB.fill]); }
  const af = PD.P[n].after; r.after = evalProduct(n, {fc: af.fc, sd: af.sd});
  return r;
}
function computeAll(){ for (const n of NAMES) RES[n] = evalProduct(n, {}, true); }

// ================= formatting & tables =================
const f = (x, d = 0) => (x === null || x === undefined || !isFinite(x)) ? '—' : x.toLocaleString('en-US', {minimumFractionDigits: d, maximumFractionDigits: d});
const pct = (x, d = 1) => f(x, d) + '%';
const sgn = (x, d = 1) => (x > 0 ? '+' : '') + f(x, d) + '%';
const vis = c => String(c).replace(/<[^>]+>/g, '').replace(/&[a-z]+;/g, '').trim();
const isNum = c => { const t = vis(c); return t === '' || /^[-+~−]?[\d,]+(\.\d+)?\s*(%|pts)?( \/ [\d,]+)?$/.test(t); };
function tbl(head, rows){ const cl = head.map((_, j) => { const col = rows.map(r => r[j]);
    const num = j > 0 && col.filter(isNum).length >= col.length / 2; const wrap = col.reduce((a, c) => a + vis(c).length, 0) / Math.max(col.length, 1) > 34;
    return (num ? '' : 'l') + (wrap ? ' w' : ''); });
  return '<div class="tbl-wrap"><table><thead><tr>' + head.map((h, j) => `<th class="${cl[j]}">${h}</th>`).join('') + '</tr></thead><tbody>' +
    rows.map(r => '<tr>' + r.map((c, j) => `<td class="${cl[j]}">${c}</td>`).join('') + '</tr>').join('') + '</tbody></table></div>'; }
const set = (id, h) => { const e = document.getElementById(id); if (e) e.innerHTML = h; };
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

// ================= rendering =================
function renderGlobal(){
  const G = ST.G;
  document.querySelectorAll('.v_sl').forEach(e => e.textContent = pct(G.sl * 100, G.sl * 1000 % 10 ? 1 : 0));
  document.querySelectorAll('.v_nd').forEach(e => e.textContent = PD.newDays);
  const nE = countEdits(); document.getElementById('banner').classList.toggle('on', nE > 0); set('nedits', nE);
  // KPIs
  const sav = NAMES.map(n => (1 - RES[n].polB.cost / RES[n].polF.cost) * 100);
  const tot = NAMES.reduce((a, n) => a + RES[n].polF.cost - RES[n].polB.cost, 0);
  const err = NAMES.reduce((a, n) => a + PD.P[n].err_updated, 0) / NAMES.length;
  const unc = NAMES.reduce((a, n) => a + (1 - RES[n].sd / PD.P[n].sigma_raw) * 100, 0) / NAMES.length;
  set('kpis', [[pct(sav.reduce((a, b) => a + b, 0) / NAMES.length, 0), 'Avg daily-cost reduction vs textbook policy'],
    [f(tot), 'Total cost saved per day (all products)'], [pct(err), 'Avg forecast error (share of volume)'],
    [pct(unc, 0), 'Demand uncertainty removed by forecasting']].map(([v, l]) => `<div class="kpi"><div class="kv">${v}</div><div class="kl">${l}</div></div>`).join(''));
  // Cockpit tables
  set('t_rec', tbl(['Product', 'Demand/day used', 'Order qty (EOQ)', 'Safety stock (MC)', 'Reorder point', 'Expected fill rate', 'Cost/day', 'Saving', 'Action'],
    NAMES.map(n => { const r = RES[n], s = (1 - r.polB.cost / r.polF.cost) * 100;
      return [`<b>${esc(n)}</b>`, f(r.D), f(r.Q), f(r.ssMC), `<b class="hl">${f(r.rop)}</b>`, pct(r.polB.fill * 100), f(r.polB.cost),
        `<span class="${s >= 0 ? 'good' : ''}">${pct(s, 0)}</span>`, `Reorder <b>${f(r.Q)}</b> units when stock on hand + on order &le; <b>${f(r.rop)}</b>`]; })));
  const cmp = []; NAMES.forEach(n => { const r = RES[n], best = Math.min(r.polF.cost, r.polMC.cost, r.polB.cost);
    [['Textbook formula', r.ropF, r.polF], ['Monte Carlo', r.ropMC, r.polMC], ['Bayesian optimisation', r.rop, r.polB]].forEach(([m, rop, pl], i) => {
      const b = pl.cost === best; cmp.push([i === 0 ? `<b>${esc(n)}</b>` : '', m, f(rop), pct(pl.fill * 100), `<span class="${b ? 'good' : ''}">${f(pl.cost)}</span>`, b ? '<span class="good">&#10003; best</span>' : '']); }); });
  set('t_cmp', tbl(['Product', 'Policy', 'Reorder point', 'Fill rate', 'Cost/day', ''], cmp));
  set('t_upd', tbl(['Product', 'Demand/day (before)', 'Demand/day (after)', 'Reorder point (before)', 'Reorder point (after)', 'Change', 'Forecast error (before)', 'Forecast error (after)'],
    NAMES.map(n => { const a = RES[n], b = a.after; return [esc(n), f(a.D), f(b.D), f(a.rop), f(b.rop), sgn((b.rop / a.rop - 1) * 100), pct(PD.P[n].err_updated), pct(PD.P[n].after.err)]; })));
  // Safety stock table (all products)
  const rows = [['Demand rate used (D)', r => f(r.D)], ['Forecast-error std (&sigma;<sub>D</sub>)', r => f(r.sd, 1)], ['Lead time, days (L)', r => f(r.L)],
    ['Lead-time std, days (&sigma;<sub>L</sub>)', r => f(r.sL, 1)], [`Service factor Z (${pct(G.sl * 100, 1)})`, r => f(r.z, 3)],
    ['Demand-variability term &sigma;<sub>D</sub>&sup2;&middot;L', r => f(r.sd * r.sd * r.L)], ['Lead-time term D&sup2;&middot;&sigma;<sub>L</sub>&sup2;', r => f(r.D * r.D * r.sL * r.sL)],
    ['Share of risk from lead time', r => { const t1 = r.sd * r.sd * r.L, t2 = r.D * r.D * r.sL * r.sL; return pct(t2 / Math.max(t1 + t2, 1e-9) * 100, 0); }],
    ['<b>Safety stock &mdash; formula</b>', r => `<b>${f(r.ssF)}</b>`], ['<b>Safety stock &mdash; Monte Carlo</b>', r => `<b>${f(r.ssMC)}</b>`],
    ['Monte Carlo vs formula', r => r.ssF > 0 ? sgn((r.ssMC / r.ssF - 1) * 100) : '—'],
    ['Reorder point &mdash; formula (D&middot;L + SS)', r => f(r.ropF)], ['Reorder point &mdash; Monte Carlo', r => f(r.ropMC)]];
  set('t_ss', tbl(['Metric'].concat(NAMES.map(esc)), rows.map(([l, fn]) => [l].concat(NAMES.map(n => fn(RES[n]))))));
  set('t_sl', tbl(['Service level', 'Z'].concat(...NAMES.map(n => [`${esc(n)}<br><small>formula</small>`, `${esc(n)}<br><small>Monte Carlo</small>`])),
    [0.90, 0.95, 0.975, 0.99, 0.995].map(s => [pct(s * 100), f(normInv(s), 2)].concat(...NAMES.map(n => { const r = RES[n];
      return [f(normInv(s) * Math.sqrt(r.sd * r.sd * r.L + r.D * r.D * r.sL * r.sL)), f(quantile(r.mc.s, s) - r.mc.mean)]; })))));
  set('t_eff', tbl(['Product', 'Manual sweep: best ROP', 'Sweep cost/day', 'Sweep runs', 'Bayesian: best ROP', 'Bayesian cost/day', 'Bayesian runs', 'Simulations saved'],
    NAMES.map(n => { const r = RES[n], sb = r.sweep.reduce((a, p) => p.y < a.y ? p : a);
      return [`<b>${esc(n)}</b>`, f(sb.x), f(sb.y), '60', f(r.rop), f(Math.min(...r.bo.Y)), '20', '<span class="good">67% fewer</span>']; })));
  set('t_assum', tbl(['Product', 'Lead time (days)', 'Lead-time std (days)', 'Holding cost /unit/day', 'Order qty (EOQ)', 'Warehouse capacity', 'Forecast-error std &sigma;<sub>D</sub>'],
    NAMES.map(n => { const p = ST.S[n], d = DEF.S[n], ed = (a, b, t) => Math.abs(a - b) > 1e-9 ? `<span class="edv">${t}</span>` : t;
      return [esc(n), ed(p.L, d.L, f(p.L)), ed(p.sL, d.sL, f(p.sL, 1)), ed(p.h, d.h, f(p.h, 2)), f(RES[n].Q), ed(p.cap, d.cap, f(p.cap)), ed(p.sd, d.sd, f(p.sd, 1))]; })));
  const ed = (a, b, t) => Math.abs(a - b) > 1e-9 ? `<span class="edv">${t}</span>` : t;
  set('t_glob', tbl(['Parameter', 'Value', 'Source'], [
    ['Service level target', ed(G.sl, DEF.G.sl, pct(G.sl * 100, 1)), 'Input'], ['Fixed cost per order', ed(G.oc, DEF.G.oc, f(G.oc)), 'Input'],
    ['Cost per unit of lost sales', ed(G.so, DEF.G.so, f(G.so)), 'Input'], ['Monte Carlo scenarios', '50,000', 'Model'],
    ['Inventory simulation', '300 &times; 120 days', 'Model'], ['Bayesian search budget', '20 (5 initial + 15 guided)', 'Model'],
    ['Back-test window', '90 days, refit weekly', 'Model'], ['Held-back new data', PD.newDays + ' days', 'Model'], ['Demand, promo, oil, holidays', 'Daily', 'Data']]));
}
function renderProduct(){
  const n = NAMES[cur], r = RES[n], P = PD.P[n], p = ST.S[n];
  document.querySelectorAll('.pname').forEach(e => e.textContent = n);
  document.querySelectorAll('#pills button,#dpills button').forEach(b => b.classList.toggle('on', +b.dataset.i === cur));
  const nd = Math.max(1, Math.min(p.fc.length, Math.ceil(p.L)));
  set('t_fc7', tbl(['Date', 'ML model', 'Value used', 'Actual', 'Error'],
    P.fc_dates.map((d, i) => { const used = p.fc[i], md = DEF.S[n].fc[i], a = P.fc_actual[i];
      return [d + (i < nd ? ' <span style="color:#2563eb" title="Sets demand rate D">&bull;</span>' : ''), f(md), used !== md ? `<span class="edv">${f(used)}</span>` : f(used), f(a), a == null ? '\u2014' : pct(Math.abs(used - a) / Math.max(a, 1) * 100)]; })) +
    '<p class="sub" style="margin-top:6px"><span style="color:#2563eb">&bull;</span> days whose average sets the demand rate D (first L days).</p>');
  const rk = [['No safety stock', r.mc.mean], ['Textbook formula', r.ropF], ['Monte Carlo', r.ropMC], ['Bayesian optimum', r.rop]].map(([l, rop]) => { const t = tailStats(r.mc.s, rop); return [l, f(rop), pct(t.p * 100), f(t.short, 1)]; });
  const tb = tailStats(r.mc.s, r.rop);
  set('t_risk', tbl(['Set by', 'Reorder point', 'Stockout risk / cycle', 'Units short / cycle'], rk) +
    (tb.p > 0.25 ? `<div class="note">The Bayesian optimum accepts frequent but <b>small</b> end-of-cycle shortages: a ${pct(tb.p * 100, 0)} chance of running short each cycle, yet a ${pct(r.polB.fill * 100)} fill rate, because each order of ${f(r.Q)} units covers about ${f(r.Q / r.D, 1)} days of demand. It minimises cost under a fill-rate target, not a cycle-service target.</div>` : '') +
    (r.capLimited ? `<div class="note">Warehouse capacity (${f(r.cap)}) is tight for this order quantity, so the search range was widened to keep it valid. Consider raising capacity.</div>` : ''));
  const base = r.scen[0];
  set('t_sens', tbl(['Scenario', 'Optimal ROP', 'Change', 'Cost/day', 'Cost chg.', 'Fill rate'],
    r.scen.map((s, i) => [i ? s[0] : '<b>Base case</b>', f(s[1]), i ? sgn((s[1] / base[1] - 1) * 100) : '—', f(s[2]), i ? sgn((s[2] / base[2] - 1) * 100) : '—', pct(s[3] * 100)])));
  let best = Infinity;
  set('t_log', tbl(['Run', 'Chosen by', 'ROP tried', 'Cost/day*', 'Best so far'],
    r.bo.X.map((x, k) => { const y = r.bo.Y[k], nb = y < best; if (nb) best = y; return [String(k + 1), k < 5 ? 'Initial design' : 'Guided (EI)', f(x), f(y), nb ? `<span class="good">${f(best)}</span>` : f(best)]; })));
  charts();
}

// ================= charts =================
Chart.defaults.font.family = "-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif";
Chart.defaults.font.size = 12; Chart.defaults.color = '#475569'; Chart.defaults.plugins.legend.labels.boxWidth = 12;
const made = {};
function mk(id, cfg){ if (made[id]) { made[id].destroy(); delete made[id]; } const el = document.getElementById(id); if (!el || !el.offsetParent) return; made[id] = new Chart(el, cfg); }
const axes = (x, y, xo = {}) => ({x: Object.assign({title: {display: !!x, text: x}, grid: {display: false}}, xo), y: {title: {display: !!y, text: y}, grid: {color: '#eef1f5'}}});
function charts(){
  const n = NAMES[cur], r = RES[n], P = PD.P[n], p = ST.S[n];
  mk('c_cost', {type: 'bar', data: {labels: NAMES, datasets: [
    {label: 'Textbook formula', data: NAMES.map(m => Math.round(RES[m].polF.cost)), backgroundColor: C.grey},
    {label: 'Monte Carlo', data: NAMES.map(m => Math.round(RES[m].polMC.cost)), backgroundColor: C.lblue},
    {label: 'Bayesian optimisation', data: NAMES.map(m => Math.round(RES[m].polB.cost)), backgroundColor: C.blue}]},
    options: {responsive: true, maintainAspectRatio: false, scales: axes('', 'Cost per day')}});
  mk('c_bt', {type: 'line', data: {labels: P.bt_labels, datasets: [
    {label: 'Actual demand', data: P.bt_actual, borderColor: C.blue, backgroundColor: C.blue, borderWidth: 1.6, pointRadius: 0, tension: .15},
    {label: 'ML forecast (re-trained weekly)', data: P.bt_fc, borderColor: C.green, backgroundColor: C.green, borderWidth: 1.6, pointRadius: 0, tension: .15, borderDash: [5, 3]}]},
    options: {responsive: true, maintainAspectRatio: false, interaction: {mode: 'index', intersect: false}, scales: axes('', 'Units per day', {ticks: {maxTicksLimit: 12}})}});
  const ds7 = [{label: 'ML model', data: DEF.S[n].fc, backgroundColor: C.green}];
  if (p.fc.some((v, i) => v !== DEF.S[n].fc[i])) ds7.push({label: 'Value used (edited)', data: p.fc, backgroundColor: C.amber});
  ds7.push({label: 'Actual', data: P.fc_actual, backgroundColor: C.blue});
  mk('c_fc7', {type: 'bar', data: {labels: P.fc_dates, datasets: ds7}, options: {responsive: true, maintainAspectRatio: false, scales: axes('', 'Units')}});
  const s = r.mc.s, lo = s[Math.floor(s.length * 0.0005)], hi = s[Math.floor(s.length * 0.9995)], nb = 45, w = (hi - lo) / nb, cnt = new Array(nb).fill(0);
  for (const v of s) { const k = Math.floor((v - lo) / w); if (k >= 0 && k < nb) cnt[k]++; }
  const mids = cnt.map((_, k) => lo + (k + 0.5) * w);
  mk('c_hist', {type: 'bar', data: {labels: mids.map(x => f(x)), datasets: [{label: 'Scenarios', data: cnt, barPercentage: 1, categoryPercentage: 1,
    backgroundColor: mids.map(x => x <= r.ropF ? C.lblue : (x <= r.ropMC ? C.amber : C.red))}]},
    options: {responsive: true, maintainAspectRatio: false, plugins: {legend: {display: false}, tooltip: {callbacks: {title: t => '~' + t[0].label + ' units'}}},
      scales: axes(`Units demanded during lead time (reorder point: formula ${f(r.ropF)}, Monte Carlo ${f(r.ropMC)})`, 'Number of scenarios', {ticks: {maxTicksLimit: 10}})}});
  mk('c_bo', {type: 'scatter', data: {datasets: [
    {type: 'line', label: 'Manual sweep (60 runs)', data: r.sweep, borderColor: C.grey, borderWidth: 2, pointRadius: 0, showLine: true},
    {label: 'Bayesian search (20 runs)', data: r.bo.X.map((x, i) => ({x, y: r.bo.Y[i]})), backgroundColor: C.blue, pointRadius: 5},
    {label: 'Optimum', data: [{x: r.rop, y: Math.min(...r.bo.Y)}], backgroundColor: C.green, borderColor: '#fff', borderWidth: 2, pointRadius: 9, pointStyle: 'rectRot'}]},
    options: {responsive: true, maintainAspectRatio: false, plugins: {tooltip: {callbacks: {label: t => t.dataset.label + ': ROP ' + f(t.parsed.x) + ', cost ' + f(t.parsed.y)}}},
      scales: {x: {type: 'linear', title: {display: true, text: 'Reorder point (units)'}, grid: {color: '#f3f5f8'}}, y: {title: {display: true, text: 'Cost per day (incl. penalty)'}, grid: {color: '#eef1f5'}}}}});
  const sd = r.scen.slice(1).map(x => +((x[1] / r.scen[0][1] - 1) * 100).toFixed(1));
  mk('c_sens', {type: 'bar', data: {labels: r.scen.slice(1).map(x => x[0]), datasets: [{label: 'Change in optimal reorder point (%)', data: sd, backgroundColor: sd.map(v => v >= 0 ? C.red : C.green)}]},
    options: {indexAxis: 'y', responsive: true, maintainAspectRatio: false, plugins: {legend: {display: false}, tooltip: {callbacks: {label: t => sgn(t.parsed.x) + ' vs base'}}},
      scales: {x: {title: {display: true, text: '% change vs base-case optimal reorder point'}, grid: {color: '#eef1f5'}}, y: {grid: {display: false}}}}});
  mk('c_weekly', {type: 'line', data: {labels: P.weekly.labels, datasets: [{label: 'Weekly demand', data: P.weekly.values, borderColor: C.blue,
    backgroundColor: 'rgba(37,99,235,.08)', fill: true, borderWidth: 1.6, pointRadius: 0, tension: .2}]},
    options: {responsive: true, maintainAspectRatio: false, plugins: {legend: {display: false}}, scales: axes('', 'Units per week', {ticks: {maxTicksLimit: 10}})}});
  mk('c_dow', {type: 'bar', data: {labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
    datasets: NAMES.map((m, i) => ({label: m, data: PD.P[m].dow, backgroundColor: [C.blue, C.amber, C.green][i % 3]}))},
    options: {responsive: true, maintainAspectRatio: false, scales: axes('', 'Avg units per day')}});
}

// ================= side panel =================
function countEdits(){ let k = 0; for (const g of ['sl', 'oc', 'so']) if (Math.abs(ST.G[g] - DEF.G[g]) > 1e-9) k++;
  for (const n of NAMES) { const p = ST.S[n], d = DEF.S[n]; for (const q of ['L', 'sL', 'h', 'cap', 'sd']) if (Math.abs(p[q] - d[q]) > 1e-9) k++;
    p.fc.forEach((v, i) => { if (v !== d.fc[i]) k++; }); } return k; }
const RULES = {sl: [50, 99.9], oc: [1, 1e9], so: [0, 1e9], L: [1, 6], sL: [0, 5], h: [0.01, 1e6], cap: [1, 1e9], sd: [0, 1e6], fc: [0, 1e9]};
let pending = false;
function fillPanel(){
  const n = NAMES[cur], p = ST.S[n], d = DEF.S[n];
  document.querySelectorAll('[data-g]').forEach(i => { const k = i.dataset.g, v = k === 'sl' ? +(ST.G.sl * 100).toFixed(2) : ST.G[k]; i.value = v;
    i.classList.toggle('ed', Math.abs(ST.G[k] - DEF.G[k]) > 1e-9); i.classList.remove('bad'); });
  set('d_g_sl', 'default ' + pct(DEF.G.sl * 100, 0)); set('d_g_oc', 'default ' + f(DEF.G.oc)); set('d_g_so', 'default ' + f(DEF.G.so));
  document.querySelectorAll('[data-k]').forEach(i => { const k = i.dataset.k; i.value = p[k]; i.classList.toggle('ed', Math.abs(p[k] - d[k]) > 1e-9); i.classList.remove('bad'); });
  document.querySelectorAll('[data-dd]').forEach(e => { const k = e.dataset.dd; e.textContent = 'model / default ' + (k === 'h' ? f(d[k], 2) : k === 'sL' || k === 'sd' ? f(d[k], 1) : f(d[k])); });
  const P = PD.P[n];
  const ndp = Math.max(1, Math.min(p.fc.length, Math.ceil(p.L)));
  set('dp_rows', P.fc_dates.map((dt, i) => `<tr><td class="l">${dt}${i < ndp ? ' <span style="color:#2563eb" title="Sets demand rate D">&bull;</span>' : ''}</td><td>${f(d.fc[i])}</td><td><input type="number" step="1" min="0" data-fc="${i}" value="${p.fc[i]}" class="${p.fc[i] !== d.fc[i] ? 'ed' : ''}"></td><td>${f(P.fc_actual[i])}</td></tr>`).join(''));
  document.querySelectorAll('[data-fc]').forEach(bindInput);
  status();
}
function status(msg, err){ const e = document.getElementById('dstat'); e.classList.toggle('err', !!err);
  e.textContent = msg || (pending ? 'Inputs changed — press Recalculate to update the dashboard.' : `${countEdits()} input(s) differ from the model defaults.`); }
function bindInput(i){
  i.oninput = () => { const g = i.dataset.g, k = i.dataset.k, fc = i.dataset.fc; const key = g || k || 'fc'; const raw = i.value.trim(); let v = Number(raw);
    const [mn, mx] = RULES[key]; const ok = raw !== '' && isFinite(v) && v >= mn && v <= mx && (key !== 'L' || Number.isInteger(v));
    i.classList.toggle('bad', !ok);
    if (!ok) { status(`Enter a value between ${f(mn, key === 'sl' || key === 'sL' ? 1 : 0)} and ${mx >= 1e6 ? 'any' : f(mx, key === 'sl' ? 1 : (key === 'sL' ? 0 : 0))}${key === 'L' ? ' (whole days)' : ''}.`, true); refreshBtn(); return; }
    const n = NAMES[cur];
    if (g) { ST.G[g] = g === 'sl' ? v / 100 : v; i.classList.toggle('ed', Math.abs(ST.G[g] - DEF.G[g]) > 1e-9); }
    else if (k) { ST.S[n][k] = v; i.classList.toggle('ed', Math.abs(v - DEF.S[n][k]) > 1e-9); if (k === 'L') { const nd = Math.max(1, Math.min(6, Math.ceil(v))); document.querySelectorAll('#dp_rows tr').forEach((tr, j) => { const c = tr.firstElementChild; c.innerHTML = PD.P[n].fc_dates[j] + (j < nd ? ' <span style="color:#2563eb" title="Sets demand rate D">&bull;</span>' : ''); }); } }
    else { v = Math.round(v); ST.S[n].fc[+fc] = v; i.classList.toggle('ed', v !== DEF.S[n].fc[+fc]); }
    pending = true; status(); refreshBtn(); };
  i.onkeydown = e => { if (e.key === 'Enter' && !document.getElementById('recalc').disabled) recalc(); };
}
function refreshBtn(){ document.getElementById('recalc').disabled = document.querySelectorAll('.drawer input.bad').length > 0; }
function recalc(){
  const b = document.getElementById('busy'); b.classList.add('on');
  setTimeout(() => { const t0 = performance.now(); computeAll(); renderGlobal(); renderProduct(); pending = false;
    try { localStorage.setItem(KEY, JSON.stringify(ST)); } catch (e) {}
    b.classList.remove('on'); fillPanel(); status(`Updated in ${((performance.now() - t0) / 1000).toFixed(1)} s — ${countEdits()} input(s) differ from the model defaults.`); }, 30);
}
function resetAll(){ ST = clone(DEF); try { localStorage.removeItem(KEY); } catch (e) {} fillPanel(); recalc(); }
document.getElementById('dtab').onclick = () => { document.body.classList.add('dopen'); fillPanel(); setTimeout(charts, 280); };
document.getElementById('dclose').onclick = () => { document.body.classList.remove('dopen'); setTimeout(charts, 280); };
document.getElementById('recalc').onclick = recalc;
document.getElementById('resetP').onclick = () => { const n = NAMES[cur]; ST.S[n] = clone(DEF.S[n]); fillPanel(); recalc(); };
document.getElementById('resetA').onclick = resetAll;
document.getElementById('bannerReset').onclick = resetAll;
document.querySelectorAll('.drawer [data-g],.drawer [data-k]').forEach(bindInput);

// ================= navigation =================
const pillHtml = NAMES.map((n, i) => `<button data-i="${i}">${esc(n)}</button>`).join('');
set('pills', pillHtml); set('dpills', pillHtml);
document.querySelectorAll('#pills button,#dpills button').forEach(b => b.onclick = () => { cur = +b.dataset.i; renderProduct(); if (document.body.classList.contains('dopen')) fillPanel(); });
function showTab(k){ curTab = k;
  document.querySelectorAll('nav.tabs button').forEach(b => b.classList.toggle('on', b.dataset.t === k));
  document.querySelectorAll('section').forEach(s => s.classList.toggle('on', s.id === k));
  document.getElementById('pbar').style.display = document.getElementById(k).dataset.pp === '1' ? 'flex' : 'none';
  charts(); }
document.querySelectorAll('nav.tabs button').forEach(b => b.onclick = () => showTab(b.dataset.t));
computeAll(); renderGlobal(); renderProduct(); showTab('cockpit');
"""
