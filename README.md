Created By: Muhammad Saad Hassan

Live App Link : https://ai-co-pilot-production.up.railway.app

Youtube Video Link : https://youtu.be/bmH9pcsyKFc



---
title: Hackathon Ai Coplit
emoji: 🚗
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---








# AI Co-Pilot for Insurance Claim Processing

An AI Co-Pilot that speeds up auto-insurance claim triage: a claimant uploads a
damage photo + description + claim details, the system predicts damage
severity with a deep-learning model, explains *why*, and a human adjuster
approves/rejects/edits the recommendation before a PDF report is generated.

## Quick start

```bash
pip install -r requirements.txt

cp .env.example .env        # then edit .env and paste in your OpenRouter API key

python train_model.py       # trains the ANN (~10-20s) -- skip if model/ already has a saved model
python app.py                # starts the web app
```

Open **http://localhost:5000**, upload any image, fill the form, submit,
review the AI's recommendation, and click Approve/Reject/Edit to generate
the downloadable PDF report.

### Setting up the LLM (OpenRouter) API key

1. Get a free key at [openrouter.ai/keys](https://openrouter.ai/keys).
2. Copy `.env.example` to `.env` and paste your key into `OPENROUTER_API_KEY`.
3. `.env` is already listed in `.gitignore` -- **never commit it**. Only
   `.env.example` (no real key) goes to GitHub.
4. The free auto-router (`openrouter/free`) rotates between whichever free
   models are currently available. Free models can be rate-limited, so
   before your demo/deploy it's worth pinning `OPENROUTER_MODEL` in `.env`
   to one specific stable free model (e.g. `meta-llama/llama-3.3-70b-instruct:free`)
   -- check [openrouter.ai/models](https://openrouter.ai/models) for what's
   currently free.
5. If the API key is missing or the request fails for any reason (offline,
   rate-limited, model temporarily down), `llm_reasoning.py` automatically
   falls back to a rule-based summary so the app never crashes mid-demo --
   you'll see a `[llm_reasoning] OpenRouter call failed...` line in the
   terminal when that happens, and can still show judges the fallback and
   explain the design choice.

## Architecture

```
Claimant Upload (photo + text + tabular)
        │
        ▼
 ┌─────────────────────┐     ┌──────────────────────────┐
 │ ANN (deep learning)  │────▶│ Explainability (XAI)      │
 │ severity classifier  │     │ confidence + saliency map │
 │ train_model.py       │     │ explain.py                │
 └─────────────────────┘     └──────────────────────────┘
        │                                │
        ▼                                ▼
 ┌─────────────────────────────────────────────┐
 │ LLM reasoning layer (llm_reasoning.py)        │
 │ -> plain-English summary + suggested payout   │
 └─────────────────────────────────────────────┘
        │
        ▼
 ┌─────────────────────────────────────────────┐
 │ Human-in-the-Loop dashboard (review.html)     │
 │ Adjuster: Approve / Reject / Edit             │
 └─────────────────────────────────────────────┘
        │
        ▼
 ┌─────────────────────────────────────────────┐
 │ PDF report generator (report_gen.py)          │
 │ full audit trail, downloadable                │
 └─────────────────────────────────────────────┘
```

## How each rubric requirement is met

| Requirement | Where |
|---|---|
| 3+ data modalities | **Image** (damage photo), **Text** (claim description), **Tabular** (vehicle age, claimed amount, policy type) |
| Predictive deep learning model | `train_model.py` — ANN (MLPClassifier, 2 hidden layers) trained on image features, classifying severity |
| LLM used for reasoning only | `llm_reasoning.py` — calls a real LLM (OpenRouter) to turn prediction + data into a summary; the payout number itself is computed by a deterministic formula, never the LLM |
| Human-in-the-Loop | `review.html` + `/finalize` route — adjuster must Approve/Reject/Edit before anything is finalized |
| Explainable AI | `explain.py` — per-class confidence scores + occlusion-based saliency heatmap (Grad-CAM-style region importance) |
| Working web app | `app.py` (Flask) + `templates/` — upload → review → report download flow |
| Downloadable report | `report_gen.py` — PDF via reportlab, includes photo, heatmap, prediction, confidences, adjuster decision |
| Business model | See below |


## Demo login: 

Car Key CAR-001 (or any CAR-002–CAR-050) · Password 995565

Car Owner Login
Log in with a car key + password before filing a claim. Vehicle age and policy type come from cars_dataset.csv and are locked (not editable) — age is calculated live from purchase_year, so it auto-increases every year.

## File List : 

auth.py, cars_dataset.csv          → login + car data
templates/login.html, car_info.html → login pages

## Business Model & Commercialization

**Target customer**: Mid-size auto insurers and third-party claims
administrators (TPAs) who process high volumes of low-to-moderate severity
claims and want to cut manual triage time.

**Value proposition**: Reduces average claim triage time from ~2-3 days
(manual adjuster review + photo inspection) to minutes, while keeping a
human adjuster as the final decision-maker — so it's fast *and* compliant
with insurance regulations that require human sign-off on payouts.

**Revenue model**: B2B SaaS, priced per-processed-claim (e.g. tiered
per-claim fee) or per-adjuster-seat/month, with an enterprise tier for
white-labeled deployment integrated into the insurer's existing claims
system via API.

**Why it's defensible**: The value isn't just the ANN model (which is
replaceable) — it's the full workflow: multimodal intake, explainability
that builds adjuster trust (they can see *why* the AI flagged something),
audit-ready PDF reports for compliance/dispute records, and a HITL loop that
can be used to continuously retrain the model on adjuster corrections.

**Go-to-market**: Start with a pilot on a single claim type (e.g. minor
collision claims under $5k) at 1-2 mid-size insurers, prove turnaround-time
and cost savings, then expand to more claim types and larger insurers.

## Files

```
generate_data.py       Synthetic dataset generator (swap for real Kaggle data)
ingest_real_dataset.py Builds data/claims.csv from the real Kaggle car-damage dataset
train_model.py         ANN training script + feature extraction
explain.py              Prediction + confidence scores + saliency heatmap (XAI)
llm_reasoning.py        Real OpenRouter LLM call for the summary + deterministic payout formula
report_gen.py            PDF report generator
app.py                   Flask web app tying it all together
templates/                index.html (upload), review.html (HITL), result.html
data/                     Real car-damage images + claims.csv
model/                    Saved trained model + scaler
reports/                  Generated PDF reports land here
requirements.txt          Python dependencies
.env.example              Template for your OpenRouter API key (copy to .env)
```
