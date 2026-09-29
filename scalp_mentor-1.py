"""
Harmonized EMA 20/40 + Price Action AI Trading Mentor — Prototype

Implements the user's Harmonized EMA 20/40 strategy:
  1H  -> Trend Filter (EMA 20/40 relationship + slope + market structure)
  15M -> Pullback + POI + Liquidity sweep + rejection/engulfing/displacement
  5M  -> MSS/Shift execution confirmation + entry/SL/TP with minimum R:R

Each stage is a separate Claude vision call with structured JSON output.
Later stages receive context from earlier ones. Final verdict is EXECUTE
only if all three stages pass AND the resulting risk:reward meets the
strategy's minimum target (default 1:4).

SETUP:
    pip install anthropic --break-system-packages
    export ANTHROPIC_API_KEY=your_key_here

USAGE:
    python scalp_mentor.py \
        --h1 chart_1h.png \
        --m15 chart_15m.png \
        --m5 chart_5m.png \
        --pair XAUUSD \
        --min-rr 1:4

NOTE: This is a decision-support / checklist tool, not financial advice,
and it does not place trades. Treat its output as a second opinion to
check your own chart reading against. Vision models can misread candle
patterns and chart details, and past checklist performance is no
guarantee of future results.
"""

import argparse
import base64
import json
import os
import sys
from dataclasses import dataclass
from typing import Optional

import anthropic

MODEL = "claude-sonnet-5"
MAX_TOKENS = 1500


# ---------------------------------------------------------------------------
# Instrument-specific stop-loss conventions (used only as a fallback
# reference shown to the model — the strategy itself places SL beyond
# the invalidating structure, not at a fixed distance).
# ---------------------------------------------------------------------------

INSTRUMENT_SL_CONFIG = {
    "XAUUSD": {"unit": "USD (price points)", "range": "structure-dependent, typically $3-15"},
    "NAS100": {"unit": "index points", "range": "structure-dependent, typically 15-60 points"},
    "EURUSD": {"unit": "pips", "range": "structure-dependent, typically 5-15 pips"},
    "GBPUSD": {"unit": "pips", "range": "structure-dependent, typically 5-15 pips"},
}
DEFAULT_SL_CONFIG = {"unit": "price units", "range": "structure-dependent (not yet configured for this instrument)"}
DEFAULT_MIN_RR = "1:4"


def get_sl_config(pair: str) -> dict:
    return INSTRUMENT_SL_CONFIG.get(pair.upper(), DEFAULT_SL_CONFIG)


# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------

@dataclass
class StageResult:
    stage: str
    passed: bool
    details: dict
    reasoning: str


# NOTE ON CURLY BRACES: stages that use .format() have every literal
# brace in the JSON example doubled ({{ }}) so Python's string
# formatting doesn't try to interpret the example as a field name.
# Stages with no injected variables are left as plain strings and are
# NOT passed through .format() at all.

STAGE_PROMPTS = {
    "trend_filter": """
You are checking the 1H Trend Filter stage of a Harmonized EMA 20/40
+ Price Action strategy.

Look at this 1-hour chart, which should show the 20-period EMA and
40-period EMA. Determine:
1. Is EMA 20 above EMA 40 (bullish) or below (bearish)?
2. Are both EMAs sloping clearly in that direction (not flat)?
3. Does market structure agree (higher-highs/higher-lows for bullish,
   lower-highs/lower-lows for bearish)?
4. Is price excessively extended away from the EMAs (overextended,
   poor risk:reward from here)?
5. Classify the environment: "bullish", "bearish", or "no_trade" (if
   EMAs are repeatedly crossing, flat, or price is choppy/ranging).

Respond ONLY with JSON, no other text, in this exact shape:
{
  "passed": true/false,
  "environment": "bullish" | "bearish" | "no_trade",
  "ema_alignment_clear": true/false,
  "structure_agrees": true/false,
  "overextended": true/false,
  "reasoning": "<1-2 sentence explanation>"
}
""",
    "pullback_poi_liquidity": """
You are checking the 15M Pullback + POI + Liquidity stage of a
Harmonized EMA 20/40 + Price Action strategy.
Context: the 1H environment is "{environment}" (from the trend filter
stage). Look at this 15-minute chart and determine:
1. Has price pulled back toward the EMA 20-40 zone (without
   necessarily crossing both EMAs)?
2. Is there a meaningful Point of Interest at/near that zone — e.g.
   prior support/resistance, order block, fair value gap, a previous
   swing area, or a liquidity area? What type is it?
3. Has price swept a recent obvious high (for a bearish setup) or low
   (for a bullish setup) and then rejected?
4. Is there price-action confirmation matching the "{environment}"
   direction — a rejection candle, an engulfing candle, or clear
   displacement?

Respond ONLY with JSON:
{
  "passed": true/false,
  "pullback_to_ema_zone": true/false,
  "poi_type": "<short description or none>",
  "liquidity_swept": true/false,
  "confirmation_type": "rejection" | "engulfing" | "displacement" | "none",
  "reasoning": "<1-2 sentence explanation>"
}
""",
    "execution_and_risk": """
You are checking the 5M Execution + Risk/Target stage of a Harmonized
EMA 20/40 + Price Action strategy.

Context: 1H environment is "{environment}". Instrument: {pair}.
Stop-loss unit for this instrument: {sl_unit}. Minimum acceptable
risk:reward ratio for this strategy: {min_rr}.

Look at this 5-minute chart and determine:
1. Is there a valid Market Structure Shift (MSS) or "Change of
   Character" confirming the "{environment}" direction on this
   timeframe?
2. What would the entry price be, based on that shift/confirmation?
3. Where would the stop-loss go — it must sit beyond the structure
   that would invalidate this setup (not a fixed arbitrary distance).
   State the stop-loss price and the distance in {sl_unit}.
4. What is the next meaningful opposing liquidity/POI/structure level
   that could serve as the take-profit target?
5. Calculate the resulting risk:reward ratio from entry/SL/TP. Does
   it meet or exceed the minimum ({min_rr})? If not, this stage fails
   regardless of how clean the setup looks, per the strategy's rule
   that a trade is only valid if it clears the minimum R:R.

Respond ONLY with JSON:
{
  "passed": true/false,
  "mss_confirmed": true/false,
  "entry_price": <number or null>,
  "stop_loss_price": <number or null>,
  "stop_loss_distance": <number or null>,
  "take_profit_price": <number or null>,
  "calculated_rr": "<e.g. '1:4.2' or null>",
  "meets_min_rr": true/false,
  "reasoning": "<1-2 sentence explanation>"
}
""",
}


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def encode_image(path: str) -> tuple[str, str]:
    """Return (base64_data, media_type) for an image file."""
    ext = os.path.splitext(path)[1].lower()
    media_type = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(ext, "image/png")
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return data, media_type


def call_stage(client: anthropic.Anthropic, prompt: str, image_path: str) -> dict:
    """Call Claude with a single image + stage prompt, parse JSON response."""
    img_data, media_type = encode_image(image_path)

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": img_data,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    text = text.strip()
    if text.startswith("
        text = text.split("
")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"passed": False, "reasoning": f"Could not parse model output: {text[:200]}"}


def run_scalp_analysis(
    client: anthropic.Anthropic,
    h1_path: str,
    m15_path: str,
    m5_path: str,
    pair: str,
    sl_unit: str = None,
    min_rr: str = DEFAULT_MIN_RR,
) -> dict:
    """
    Runs the 3-stage Harmonized EMA 20/40 pipeline:
      1H  trend_filter
      15M pullback_poi_liquidity
      5M  execution_and_risk
    """
    results: dict[str, StageResult] = {}
    default_sl_config = get_sl_config(pair)
    effective_sl_unit = sl_unit or default_sl_config["unit"]

    # Stage 1 — Trend Filter (1H chart) — no injected variables, call directly
    t = call_stage(client, STAGE_PROMPTS["trend_filter"], h1_path)
    results["T"] = StageResult("Trend Filter (1H)", t.get("passed", False), t, t.get("reasoning", ""))
    environment = t.get("environment", "no_trade")

    # Stage 2 — Pullback + POI + Liquidity (15M chart)
    p_prompt = STAGE_PROMPTS["pullback_poi_liquidity"].format(environment=environment)
    p = call_stage(client, p_prompt, m15_path)
    results["P"] = StageResult("Pullback/POI/Liquidity (15M)", p.get("passed", False), p, p.get("reasoning", ""))

    # Stage 3 — Execution + Risk/Target (5M chart)
    e_prompt = STAGE_PROMPTS["execution_and_risk"].format(
        environment=environment, pair=pair, sl_unit=effective_sl_unit, min_rr=min_rr
    )
    e = call_stage(client, e_prompt, m5_path)
    results["E"] = StageResult("Execution/Risk (5M)", e.get("passed", False), e, e.get("reasoning", ""))

    all_passed = all(r.passed for r in results.values()) and environment != "no_trade"
    failing_stages = [f"{k} ({r.stage})" for k, r in results.items() if not r.passed]
    if environment == "no_trade":
        failing_stages.insert(0, "T (1H environment classified as no_trade)")

    verdict = {
        "pair": pair,
        "environment": environment,
        "final_verdict": "EXECUTE" if all_passed else "AVOID",
        "stages": {k: {"passed": r.passed, "details": r.details, "reasoning": r.reasoning} for k, r in results.items()},
        "failing_stages": failing_stages,
    }
    return verdict


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Harmonized EMA 20/40 AI Trading Mentor")
    parser.add_argument("--h1", required=True, help="Path to 1H chart screenshot (with EMA 20/40)")
    parser.add_argument("--m15", required=True, help="Path to 15M chart screenshot")
    parser.add_argument("--m5", required=True, help="Path to 5M chart screenshot")
    parser.add_argument("--pair", required=True, help="e.g. XAUUSD")
    parser.add_argument("--sl-unit", default=None, help="Override stop-loss unit, e.g. 'pips' or 'USD'")
    parser.add_argument("--min-rr", default=DEFAULT_MIN_RR, help="Minimum acceptable risk:reward, e.g. '1:4'")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: set ANTHROPIC_API_KEY environment variable first.", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    print(f"Running Harmonized EMA 20/40 analysis for {args.pair}...\n")
    result = run_scalp_analysis(
        client, args.h1, args.m15, args.m5, args.pair,
        sl_unit=args.sl_unit, min_rr=args.min_rr,
    )
print("=" * 60)
    print(f"VERDICT: {result['final_verdict']}  (1H environment: {result['environment']})")
    print("=" * 60)
    for stage_key, stage_data in result["stages"].items():
        status = "PASS" if stage_data["passed"] else "FAIL"
        print(f"\n[{stage_key}] {status}")
        print(f"  {stage_data['reasoning']}")

    if result["failing_stages"]:
        print(f"\nFailing stages: {', '.join(result['failing_stages'])}")

    print("\n(This is a checklist tool, not financial advice — verify against your own reading of the chart.)")


if name == "main":
    main()
