"""
S.C.A.L.P. AI Trading Mentor — Prototype

Takes chart screenshots (1H, M5, M1) plus a few structured inputs
(current time, pip value) and runs each SCALP stage as a separate
Claude vision call with structured JSON output. Combines the stage
results into a final execute/avoid verdict with reasoning.

SETUP:
    pip install anthropic --break-system-packages
    export ANTHROPIC_API_KEY=your_key_here

USAGE:
    python scalp_mentor.py \
        --h1 chart_1h.png \
        --m5 chart_5m.png \
        --m1 chart_1m.png \
        --pair GBPUSD \
        --current-time-gmt "09:15" \
        --pip-value 0.0001

NOTE: This is a prototype / decision-support tool, not financial
advice, and it does not place trades. It only reasons over the
SCALP checklist you defined and reports where the setup passes or
fails. Treat its output as a second opinion to check your own
analysis against, not a signal to blindly follow — vision models
can misread candle patterns and chart details, and past checklist
performance is no guarantee of future results.
"""

import argparse
import base64
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import anthropic

MODEL = "claude-sonnet-5"  # good balance of cost and reasoning quality for this checklist task
MAX_TOKENS = 1500


# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------

@dataclass
class StageResult:
    stage: str
    passed: bool
    details: dict
    reasoning: str


STAGE_PROMPTS = {
    "spot_impulse": """
You are checking Stage S (Spot the Impulse) of a SCALP trading checklist.

Look at this 1-hour timeframe chart. Determine:
1. Is there a clear break of structure (BOS) — a new higher-high (bullish)
   or lower-low (bearish)?
2. What is the impulse direction (bullish/bearish/none)?
3. What price levels mark the External High and External Low of the
   current range?
4. Is a clean trading range defined?

Respond ONLY with JSON, no other text, in this exact shape:
{
  "passed": true/false,
  "direction": "bullish" | "bearish" | "none",
  "external_high": <number or null>,
  "external_low": <number or null>,
  "reasoning": "<1-2 sentence explanation>"
}
""",
    "premium_discount": """
You are checking Stage C (Calculate Premium/Discount) of a SCALP checklist.

Given the External High and External Low from Stage S ({external_high},
{external_low}) and direction "{direction}", look at this chart and
determine:
1. Where is the 50% (fibonacci midpoint) of that range?
2. Is current price in the premium zone (above 50%) or discount zone
   (below 50%)?
3. Does the zone match the rule: bearish setups need premium, bullish
   setups need discount?

Respond ONLY with JSON:
{
  "passed": true/false,
  "fifty_percent_level": <number or null>,
  "current_zone": "premium" | "discount" | "unclear",
  "reasoning": "<1-2 sentence explanation>"
}
""",
    "assess_poi": """
You are checking Stage A (Assess POIs) of a SCALP checklist.

Look at this chart (1H context and M5 detail if provided). Determine:
1. Is there a valid Point of Interest — an Extreme zone (origination
   order block / RIFC) or a Decisional zone (the area responsible for
   the BOS)?
2. On the lower timeframe, is there a visible buildup -> inducement ->
   push-out sequence at that POI?
3. Is the POI fresh (unmitigated) rather than already tapped through?

Respond ONLY with JSON:
{
  "passed": true/false,
  "poi_type": "extreme" | "decisional" | "none",
  "buildup_inducement_pushout_seen": true/false,
  "poi_fresh": true/false,
  "reasoning": "<1-2 sentence explanation>"
}
""",
    "liquidity_grab": """
You are checking Stage L (Liquidity Grab) of a SCALP checklist.

Context: current time is {current_time_gmt} GMT. The key time window
for this strategy is around London open (09:00 GMT).

Look at this chart. Determine:
1. Has price swept a liquidity pool at/near the POI (e.g. prior
   session high/low, trendline liquidity, equal highs/lows, or an
   SMC-style trap)?
2. Is there inducement visible within roughly 30 minutes of the key
   time window?
3. Given the supplied current time, are we inside a reasonable window
   of the key session time?

Respond ONLY with JSON:
{
  "passed": true/false,
  "liquidity_pool_type": "<short description or none>",
  "inducement_near_key_time": true/false,
  "within_key_time_window": true/false,
  "reasoning": "<1-2 sentence explanation>"
}
""",
    "position_entry": """
You are checking Stage P (Position Entry) of a SCALP checklist.

Instrument: {pair}. Stop-loss unit for this instrument: {sl_unit}.
Reference stop-loss range for this instrument: {sl_range}.

Look at this M1 chart. Determine:
1. Is there a valid entry trigger — BOS, 2-Leg pullback, or a
   Buildup-Inducement-BOS pattern on M1?
2. What price would the entry be at, based on a refined IFC
   (inefficiency/imbalance close) point?
3. Based on nearby structure, what stop-loss distance (in {sl_unit})
   would this require? Flag if the visible structure would require
   a stop meaningfully outside the reference range above.
4. What would the take-profit be at a fixed 1:3 risk:reward from
   that entry/stop?

Respond ONLY with JSON:
{
  "passed": true/false,
  "entry_trigger": "<description or none>",
  "entry_price": <number or null>,
  "stop_loss_distance": <number or null>,
  "stop_loss_unit": "{sl_unit}",
  "take_profit_price": <number or null>,
  "reasoning": "<1-2 sentence explanation>"
}
""",
}


# ---------------------------------------------------------------------------
# Instrument-specific stop-loss conventions
# ---------------------------------------------------------------------------

# SCALP's "3-7 pip" stop was defined for forex majors. Gold and indices
# don't use pips, so define the right unit + a reasonable reference range
# per instrument here. Adjust these to match your own backtested numbers.
INSTRUMENT_SL_CONFIG = {
    "XAUUSD": {"unit": "USD (price points)", "range": "$3-7"},
    "NAS100": {"unit": "index points", "range": "15-40 points"},
    "EURUSD": {"unit": "pips", "range": "3-7 pips"},
    "GBPUSD": {"unit": "pips", "range": "3-7 pips"},
}

DEFAULT_SL_CONFIG = {"unit": "price units", "range": "instrument-appropriate range (not yet configured)"}


def get_sl_config(pair: str) -> dict:
    return INSTRUMENT_SL_CONFIG.get(pair.upper(), DEFAULT_SL_CONFIG)


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
    # Strip markdown fences if the model added them despite instructions
    if text.startswith("```"):
        text = text.split("```")[1]
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
    m5_path: str,
    m1_path: str,
    pair: str,
    current_time_gmt: str,
) -> dict:
    results: dict[str, StageResult] = {}

    # Stage S — Spot the Impulse (1H chart)
    s = call_stage(client, STAGE_PROMPTS["spot_impulse"], h1_path)
    results["S"] = StageResult("Spot the Impulse", s.get("passed", False), s, s.get("reasoning", ""))

    # Stage C — Premium/Discount (1H chart, needs Stage S output)
    c_prompt = STAGE_PROMPTS["premium_discount"].format(
        external_high=s.get("external_high"),
        external_low=s.get("external_low"),
        direction=s.get("direction"),
    )
    c = call_stage(client, c_prompt, h1_path)
    results["C"] = StageResult("Premium/Discount", c.get("passed", False), c, c.get("reasoning", ""))

    # Stage A — Assess POIs (M5 chart)
    a = call_stage(client, STAGE_PROMPTS["assess_poi"], m5_path)
    results["A"] = StageResult("Assess POIs", a.get("passed", False), a, a.get("reasoning", ""))

    # Stage L — Liquidity Grab (M5 chart, needs current time)
    l_prompt = STAGE_PROMPTS["liquidity_grab"].format(current_time_gmt=current_time_gmt)
    l = call_stage(client, l_prompt, m5_path)
    results["L"] = StageResult("Liquidity Grab", l.get("passed", False), l, l.get("reasoning", ""))

    # Stage P — Position Entry (M1 chart, needs instrument-specific SL config)
    sl_config = get_sl_config(pair)
    p_prompt = STAGE_PROMPTS["position_entry"].format(
        pair=pair, sl_unit=sl_config["unit"], sl_range=sl_config["range"]
    )
    p = call_stage(client, p_prompt, m1_path)
    results["P"] = StageResult("Position Entry", p.get("passed", False), p, p.get("reasoning", ""))

    all_passed = all(r.passed for r in results.values())
    failing_stages = [f"{k} ({r.stage})" for k, r in results.items() if not r.passed]

    verdict = {
        "pair": pair,
        "final_verdict": "EXECUTE" if all_passed else "AVOID",
        "stages": {k: {"passed": r.passed, "details": r.details, "reasoning": r.reasoning} for k, r in results.items()},
        "failing_stages": failing_stages,
    }
    return verdict


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SCALP AI Trading Mentor")
    parser.add_argument("--h1", required=True, help="Path to 1H chart screenshot")
    parser.add_argument("--m5", required=True, help="Path to M5 chart screenshot")
    parser.add_argument("--m1", required=True, help="Path to M1 chart screenshot")
    parser.add_argument("--pair", required=True, help="e.g. GBPUSD")
    parser.add_argument("--current-time-gmt", required=True, help="e.g. 09:15")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: set ANTHROPIC_API_KEY environment variable first.", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    print(f"Running SCALP analysis for {args.pair}...\n")
    result = run_scalp_analysis(
        client,
        args.h1,
        args.m5,
        args.m1,
        args.pair,
        args.current_time_gmt,
    )

    print("=" * 60)
    print(f"VERDICT: {result['final_verdict']}")
    print("=" * 60)
    for stage_key, stage_data in result["stages"].items():
        status = "PASS" if stage_data["passed"] else "FAIL"
        print(f"\n[{stage_key}] {status}")
        print(f"  {stage_data['reasoning']}")

    if result["failing_stages"]:
        print(f"\nFailing stages: {', '.join(result['failing_stages'])}")

    print("\n(This is a checklist tool, not financial advice — verify against your own reading of the chart.)")


if __name__ == "__main__":
    main()
