"""
Harmonized EMA 20/40 + Price Action AI Trading Mentor — Streamlit UI

Upload your 1H, 15M, and 5M screenshots, get a stage-by-stage verdict,
and log the outcome later so you can track calibration.

SETUP:
    pip install streamlit anthropic --break-system-packages
    export ANTHROPIC_API_KEY=your_key_here

RUN:
    streamlit run app.py

Decision-support / checklist tool, not financial advice.
"""

import csv
import os
from datetime import datetime

import streamlit as st
import anthropic

from scalp_mentor import run_scalp_analysis, get_sl_config, DEFAULT_MIN_RR

LOG_PATH = "trade_log.csv"
LOG_FIELDS = [
    "timestamp", "pair", "environment", "verdict",
    "t_pass", "p_pass", "e_pass", "failing_stages", "outcome", "notes",
]

st.set_page_config(page_title="EMA 20/40 AI Mentor", layout="wide")
st.title("Harmonized EMA 20/40 + Price Action Mentor")
st.caption("Decision support, not financial advice — verify against your own chart reading.")

with st.sidebar:
    st.header("Setup")
    pair = st.selectbox("Instrument", ["XAUUSD", "NAS100", "EURUSD", "GBPUSD", "Other"])
    if pair == "Other":
        pair = st.text_input("Enter instrument symbol", "")
    sl_config = get_sl_config(pair) if pair else {}

    st.subheader("Stop-loss / Take-profit")
    sl_unit = st.text_input(
        "Stop-loss unit",
        value=sl_config.get("unit", ""),
        help="e.g. 'pips', 'USD (price points)', 'index points'",
    )
    min_rr = st.text_input(
        "Minimum risk:reward",
        value=DEFAULT_MIN_RR,
        help="Strategy minimum — a setup failing this bar is AVOID even if everything else looks clean",
    )

    st.divider()
    st.caption(
        "Risk rules (reminder, not enforced by this app): "
        "0.25% risk per trade · 0.75% max daily loss · stop after 2 losses in a day · "
        "never widen a stop to avoid a loss."
    )

st.subheader("1. Upload charts")
col1, col2, col3 = st.columns(3)
with col1:
    h1_file = st.file_uploader("1H chart — Trend Filter (EMA 20/40)", type=["png", "jpg", "jpeg"], key="h1")
with col2:
    m15_file = st.file_uploader("15M chart — Pullback / POI / Liquidity", type=["png", "jpg", "jpeg"], key="m15")
with col3:
    m5_file = st.file_uploader("5M chart — Execution (MSS) / Risk", type=["png", "jpg", "jpeg"], key="m5")

for label, f in [("1H", h1_file), ("15M", m15_file), ("5M", m5_file)]:
    if f is not None:
        st.image(f, caption=label, width=250)

st.subheader("2. Run analysis")

if st.button("Analyze setup", type="primary", disabled=not (h1_file and m15_file and m5_file and pair)):
    api_key = st.secrets.get("ANTHROPIC_API_KEY", None) if hasattr(st, "secrets") else None
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        st.error(
            "No API key found. Set ANTHROPIC_API_KEY as an environment variable, "
            "or create a .streamlit/secrets.toml file with ANTHROPIC_API_KEY = \"your-key\"."
        )
    else:
        tmp_paths = {}
        for label, f in [("h1", h1_file), ("m15", m15_file), ("m5", m5_file)]:
            tmp_path = f"_tmp_{label}_{f.name}"
            with open(tmp_path, "wb") as out:
                out.write(f.getbuffer())
            tmp_paths[label] = tmp_path

        client = anthropic.Anthropic(api_key=api_key)
        with st.spinner("Running Trend Filter -> Pullback/POI/Liquidity -> Execution/Risk..."):
            result = run_scalp_analysis(
                client, tmp_paths["h1"], tmp_paths["m15"], tmp_paths["m5"], pair,
                sl_unit=sl_unit, min_rr=min_rr,
            )

        for path in tmp_paths.values():
            os.remove(path)

        st.session_state["last_result"] = result

if "last_result" in st.session_state:
    result = st.session_state["last_result"]
    st.subheader("3. Result")
    if result["final_verdict"] == "EXECUTE":
        st.success(f"VERDICT: EXECUTE — {pair} ({result['environment']})")
    else:
        st.warning(f"VERDICT: AVOID — {pair} (1H environment: {result['environment']})")
        st.write(f"Failing stages: {', '.join(result['failing_stages'])}")

    stage_labels = {"T": "Trend Filter (1H)", "P": "Pullback/POI/Liquidity (15M)", "E": "Execution/Risk (5M)"}
    for key, data in result["stages"].items():
        icon = "PASS" if data["passed"] else "FAIL"
        with st.expander(f"[{stage_labels.get(key, key)}] {icon} — {data['reasoning'][:80]}"):
            st.json(data["details"])

    st.subheader("4. Log the outcome (fill in after the trade closes)")
    with st.form("log_form"):
        outcome = st.selectbox("Outcome", ["pending", "win", "loss", "skipped (did not take)"])
        notes = st.text_input("Notes (optional)")
        submitted = st.form_submit_button("Save to log")
        if submitted:
            file_exists = os.path.exists(LOG_PATH)
            with open(LOG_PATH, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
                if not file_exists:
                    writer.writeheader()
                writer.writerow({
                    "timestamp": datetime.utcnow().isoformat(),
                    "pair": pair,
                    "environment": result["environment"],
                    "verdict": result["final_verdict"],
                    "t_pass": result["stages"]["T"]["passed"],
                    "p_pass": result["stages"]["P"]["passed"],
                    "e_pass": result["stages"]["E"]["passed"],
                    "failing_stages": "; ".join(result["failing_stages"]),
                    "outcome": outcome,
                    "notes": notes,
                })
            st.success("Logged.")

if os.path.exists(LOG_PATH):
    st.subheader("Log summary")
    with open(LOG_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    total = len(rows)
    executed = [r for r in rows if r["verdict"] == "EXECUTE"]
    wins = [r for r in executed if r["outcome"] == "win"]
    losses = [r for r in executed if r["outcome"] == "loss"]
    decided = len(wins) + len(losses)
    win_rate = (len(wins) / decided * 100) if decided else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total logged", total)
    c2.metric("EXECUTE verdicts", len(executed))
    c3.metric("Decided (win/loss)", decided)
    c4.metric("Win rate", f"{win_rate:.0f}%")
