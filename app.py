"""Streamlit app: AI-enhanced reorder point & inventory optimization.

Run locally:   streamlit run app.py
Deploy:        share.streamlit.io -> New app -> this repo, main branch, app.py
"""
import hashlib
import html
import io
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from dashboard_builder import build_dashboard_html
from inventory_engine import MIN_DAYS, load_data, run_analysis

HERE = Path(__file__).parent
SAMPLE_CSV = HERE / "retail_demand_data.csv"
SAMPLE_LABEL = "Corporación Favorita (Ecuador), store 1"
SAMPLE_DESC = ("Corporaci&oacute;n Favorita (Ecuador) grocery sales, public Kaggle &ldquo;Store Sales &ndash; Time Series "
               "Forecasting&rdquo; dataset; store 1, three product families, daily. External drivers: national/local "
               "holidays and daily oil price (WTI).")

st.set_page_config(page_title="AI Inventory Optimization", page_icon="📦", layout="wide")
st.markdown("""<style>
.block-container{padding-top:1.2rem;padding-bottom:0;max-width:100%}
header[data-testid="stHeader"]{height:0}
</style>""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False, max_entries=8)
def build(csv_bytes: bytes, new_days: int, max_products: int, label: str, desc: str) -> str:
    full, factors = load_data(io.BytesIO(csv_bytes))
    bar = st.progress(0.0, "Loading data…")
    try:
        A = run_analysis(full, factors, new_days=new_days, max_products=max_products,
                         progress=lambda p, msg: bar.progress(min(p, 1.0), msg))
        return build_dashboard_html(A, label, desc)
    finally:
        bar.empty()


# ---------------- sidebar ----------------
with st.sidebar:
    st.header("📦 Data & settings")
    source = st.radio("Demand data", ["Sample data (Corporación Favorita)", "Upload my own CSV"])
    upload = None
    if source == "Upload my own CSV":
        upload = st.file_uploader("CSV file", type=["csv"])
        with st.expander("Required CSV format"):
            st.markdown(f"""
- **Required columns:** `date`, `product`, `demand`
- **Optional:** `promo` and any other numeric columns (e.g. `oil_price`, `holiday`). They are used as demand drivers.
- One row per product per day; at least **{MIN_DAYS} days** per product.
- Header names are not case-sensitive.
""")
    new_days = st.slider("Days held back as 'new data'", 7, 28, 14,
                         help="The most recent days are hidden from the first model, then added, to show the forecast "
                              "and reorder points updating. The first 7 are also used to check the 7-day forecast.")
    max_products = st.slider("Products to analyse", 1, 5, 3, help="The first products found in the file.")
    frame_h = st.slider("Dashboard height (px)", 650, 1600, 900, 50,
                        help="Match this to your screen so the edit panel's buttons stay visible.")
    st.divider()
    st.caption("Edit lead times, costs, service level and demand data points with the blue **Edit inputs & data** "
               "tab on the right edge of the dashboard. Those changes recalculate instantly in the browser.")

# ---------------- data ----------------
if source == "Upload my own CSV":
    if upload is None:
        st.info("Upload a CSV in the sidebar to analyse your own data, or switch back to the sample data.")
        st.stop()
    csv_bytes = upload.getvalue()
    label = Path(upload.name).stem.replace("_", " ")
    desc = f"Uploaded file <b>{html.escape(upload.name)}</b>. Demand drivers detected from the numeric columns in the file."
else:
    csv_bytes = SAMPLE_CSV.read_bytes()
    label, desc = SAMPLE_LABEL, SAMPLE_DESC

try:
    with st.spinner("Training the demand-forecasting models (first run takes about 30 seconds)…"):
        page = build(csv_bytes, new_days, max_products, label, desc)
except ValueError as e:
    st.error(f"Could not analyse this file: {e}")
    st.stop()
except Exception as e:  # unexpected file problems
    st.error(f"Something went wrong while reading the data: {e}")
    st.stop()

# ---------------- dashboard ----------------
# Fixed height keeps the dashboard's side panel anchored to the visible frame; the dashboard scrolls inside it.
if hasattr(st, "iframe"):
    st.iframe(page, height=frame_h)
else:  # Streamlit versions before st.iframe existed
    components.html(page, height=frame_h, scrolling=True)

c1, c2 = st.columns([1, 3])
with c1:
    st.download_button("⬇️ Download dashboard (HTML)", page, file_name="inventory_dashboard.html", mime="text/html",
                       help="A standalone copy that opens offline in any browser.")
with c2:
    st.caption(f"Analysis id {hashlib.md5(csv_bytes).hexdigest()[:8]} · Forecasting: scikit-learn gradient boosting "
               "(re-trained in Python) · Safety stock, Monte Carlo and Bayesian optimisation: recalculated in the browser.")
