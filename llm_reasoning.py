"""
llm_reasoning.py
-----------------
This module is the LLM layer in the pipeline: it takes the ANN's severity
prediction + the tabular claim data + the claimant's free-text description
and turns them into a human-readable adjuster summary and a suggested
payout. Per the hackathon rules, the LLM is used for REASONING /
SUMMARIZATION only -- the ANN in explain.py / train_model.py is the actual
predictive engine, never this module.

The payout NUMBER is always computed deterministically in suggested_payout()
below, in plain Python -- never by the LLM. The LLM only writes the
explanation text. This keeps the numeric result auditable and reproducible
even though the wording is LLM-generated.

Calls OpenRouter (https://openrouter.ai) with an OpenAI-compatible chat
completions request. The API key is read from the OPENROUTER_API_KEY
environment variable (see .env.example) -- never hardcode it here.

If no key is set, or the request fails for any reason (offline demo,
rate limit, free-model temporarily unavailable, etc.), this module
automatically falls back to a rule-based template so the app never breaks
mid-demo.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openrouter/free")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

PAYOUT_MULTIPLIER = {"minor": 0.9, "moderate": 1.0, "severe": 1.15}


def suggested_payout(claimed_amount: float, severity: str, confidence: float) -> float:
    """Deterministic payout calculation -- NEVER done by the LLM."""
    mult = PAYOUT_MULTIPLIER.get(severity, 1.0)
    # Low model confidence -> recommend paying closer to the claimed amount
    # and flagging for closer human review, rather than aggressively adjusting.
    confidence_adj = 1 - (1 - confidence) * 0.3
    return round(claimed_amount * mult * confidence_adj, 2)


def _rule_based_summary(description, severity, conf_pct, vehicle_age,
                         claimed_amount, policy_type, payout) -> str:
    """Offline fallback used if the LLM call is unavailable or fails."""
    confidence_note = (
        "The model is highly confident in this assessment."
        if conf_pct >= 75 else
        "Model confidence is moderate -- recommend closer manual review."
        if conf_pct >= 50 else
        "Model confidence is low -- this claim should be manually re-inspected."
    )
    return (
        f"Claimant reports: \"{description}\" The AI Co-Pilot classifies this as "
        f"{severity.upper()} damage with {conf_pct}% confidence. {confidence_note} "
        f"Given the vehicle age ({vehicle_age} yrs), policy tier ({policy_type}), and "
        f"claimed amount (${claimed_amount:,.2f}), the recommended payout is "
        f"${payout:,.2f}. Final approval requires adjuster sign-off below."
    )


def _build_prompt(description, severity, conf_pct, vehicle_age,
                   claimed_amount, policy_type, payout) -> str:
    return (
        "You are an assistant helping an insurance claims adjuster. You do NOT "
        "decide severity or payout amounts yourself -- those are computed by a "
        "separate deep learning model and a fixed formula, and are given to you "
        "below as facts. Your only job is to write a short, professional 3-4 "
        "sentence summary for the adjuster's dashboard, in plain English, using "
        "exactly the numbers provided (do not invent or change any numbers).\n\n"
        f"Claimant's description: \"{description}\"\n"
        f"AI-predicted damage severity: {severity.upper()}\n"
        f"Model confidence: {conf_pct}%\n"
        f"Vehicle age: {vehicle_age} years\n"
        f"Policy type: {policy_type}\n"
        f"Claimed amount: ${claimed_amount:,.2f}\n"
        f"Computed suggested payout: ${payout:,.2f}\n\n"
        "Write the summary now. Mention the severity, your confidence-based "
        "recommendation (approve quickly vs. manual review, based on the "
        "confidence level), and the suggested payout. Do not use markdown."
    )


def _call_openrouter(prompt: str):
    if not OPENROUTER_API_KEY:
        return None
    try:
        resp = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": OPENROUTER_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 250,
                "temperature": 0.4,
            },
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[llm_reasoning] OpenRouter call failed, using fallback summary: {e}")
        return None


def summarize(description: str, severity: str, confidence: float,
              vehicle_age: int, claimed_amount: float, policy_type: str):
    conf_pct = round(confidence * 100, 1)
    payout = suggested_payout(claimed_amount, severity, confidence)

    prompt = _build_prompt(description, severity, conf_pct, vehicle_age,
                            claimed_amount, policy_type, payout)
    summary = _call_openrouter(prompt)

    if not summary:
        summary = _rule_based_summary(description, severity, conf_pct,
                                       vehicle_age, claimed_amount,
                                       policy_type, payout)

    return summary, payout
