"""
S.C.A.L.P. AI Trading Mentor — Streamlit UI

A local web app around scalp_mentor.py: upload your 1H/M5/M1
screenshots, get a stage-by-stage verdict, and log the outcome later
so you can track calibration against your KPI targets.

SETUP:
    pip install streamlit anthropic --break-system-packages
    export ANTHROPIC_API_KEY=your_key_here

RUN:
    streamlit run app.py

This is a decision-support / checklist tool, not financial advice —
treat its output as a second opinion to check your own read of the
chart against, not a signal to act on automatically.
"""

import csv
import os
from datetime import datetime, date

import streamlit as st
import anthropic

from scalp_mentor import run_scalp_analysis, get_sl_config

LOG_PATH = "trade_log.csv"
LOG_FIELDS = [
    "timestamp", "pair", "verdict", "s_pass", "c_pass", "a_pass",
    "l_pass", "p_pass", "failing_stages", "outcome", "notes",
]

st.set_page_config(page_title="SCALP AI Mentor", layout="wide")
st.title("S.C.A.L.P. AI Trading Mentor")
st.caption("Decision support, not financial advice — verify against your own chart reading.")

# --- Sidebar: inputs -------------------------------------------------------
with st.sidebar:
    st.header("Setup")
    pair = st.selectbox("Instrument", ["XAUUSD", "NAS100", "EURUSD", "GBPUSD", "Other"])
    if pair == "Other":
        pair = st.text_input("Enter instrument symbol", "")
    current_time = st.text_input("Current time (GMT, HH:MM)", datetime.utcnow().strftime("%H:%M"))
    sl_config = get_sl_config(pair) if pair else {}
    if sl_config:
        st.caption(f"Stop-loss reference for {pair}: {sl_config['range']} ({sl_config['unit']})")

st.subheader("1. Upload charts")
col1, col2, col3 = st.columns(3)
with col1:
    h1_file = st.file_uploader("1H chart (Spot Impulse / Premium-Discount)", type=["png", "jpg", "jpeg"], key="h1")
with col2:
    m5_file = st.file_uploader("M5 chart (Assess POI / Liquidity Grab)", type=["png", "jpg", "jpeg"], key="m5")
with col3:
    m1_file = st.file_uploader("M1 chart (Position Entry)", type=["png", "jpg", "jpeg"], key="m1")

for label, f in [("1H", h1_file), ("M5", m5_file), ("M1", m1_file)]:
    if f is not None:
        st.image(f, caption=label, width=250)

# --- Run analysis ------------------------------------------------------------
st.subheader("2. Run analysis")

if st.button("Analyze setup", type="primary", disabled=not (h1_file and m5_file and m1_file and pair)):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        st.error("ANTHROPIC_API_KEY environment variable is not set.")
    else:
        # Save uploads to temp paths since the analysis functions expect file paths
        tmp_paths = {}
        for label, f in [("h1", h1_file), ("m5", m5_file), ("m1", m1_file)]:
            tmp_path = f"_tmp_{label}_{f.name}"
            with open(tmp_path, "wb") as out:
                out.write(f.getbuffer())
            tmp_paths[label] = tmp_path

        client = anthropic.Anthropic(api_key=api_key)
        with st.spinner("Running SCALP stages..."):
            result = run_scalp_analysis(
                client, tmp_paths["h1"], tmp_paths["m5"], tmp_paths["m1"], pair, current_time
            )

        for path in tmp_paths.values():
            os.remove(path)

        st.session_state["last_result"] = result

# --- Display results ---------------------------------------------------------
if "last_result" in st.session_state:
    result = st.session_state["last_result"]
    st.subheader("3. Result")

    if result["final_verdict"] == "EXECUTE":
        st.success(f"**VERDICT: EXECUTE** — {pair}")
    else:
        st.warning(f"**VERDICT: AVOID** — {pair}")
        st.write(f"Failing stages: {', '.join(result['failing_stages'])}")

    for key, data in result["stages"].items():
        icon = "PASS" if data["passed"] else "FAIL"
        with st.expander(f"[{key}] {icon} — {data['reasoning'][:80]}"):
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
                    "verdict": result["final_verdict"],
                    "s_pass": result["stages"]["S"]["passed"],
                    "c_pass": result["stages"]["C"]["passed"],
                    "a_pass": result["stages"]["A"]["passed"],
                    "l_pass": result["stages"]["L"]["passed"],
                    "p_pass": result["stages"]["P"]["passed"],
                    "failing_stages": "; ".join(result["failing_stages"]),
                    "outcome": outcome,
                    "notes": notes,
                })
            st.success("Logged.")

# --- Stats from log ------------------------------------------------------------
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
    c4.metric("Win rate", f"{win_rate:.0f}%", help="Target: 50%")
