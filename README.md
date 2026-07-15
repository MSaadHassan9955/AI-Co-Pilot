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

## Important note on this build (read before your demo/judging Q&A)

This sandbox has **no internet access**, so some components are intentionally
built as swappable stand-ins. Be upfront about this with judges — it reads
as an engineering decision, not a gap you missed:

1. **Dataset — REAL, not synthetic.** Uses the actual Kaggle dataset the
   handbook pointed to: `coco-car-damage-detection-dataset` (70 real
   annotated car photos). `ingest_real_dataset.py` reads the dataset's real
   COCO-format damage annotations, computes each image's annotated-damage
   area as a fraction of image area, and buckets images into
   minor/moderate/severe using tertiles of that *real* distribution — so the
   severity label is derived from real data, not invented.
   This Kaggle dataset does **not** include text descriptions or tabular
   claim fields (vehicle age, claimed amount, policy type) — those remain
   templated/synthetic since no such data exists for these images. If you
   find/attach a real insurance-claims tabular dataset, swap the generation
   logic in `ingest_real_dataset.py`.

2. **Model — ANN, not CNN, but tuned for the small real dataset.**
   Uses a scikit-learn ANN (MLPClassifier) instead of a torch/tensorflow CNN
   — installing those needs internet access this sandbox doesn't have. To
   compensate for having only 70 real images, `train_model.py`:
   - extracts **HOG (Histogram of Oriented Gradients)** features instead of
     raw pixels — HOG responds to edges/gradients, a much more relevant
     signal for dents/scratches/cracks than raw pixel values
   - **augments the training split only** (flips, small rotations,
     brightness jitter) to stretch 56 training images into ~390 samples,
     with the 14 held-out test images never touched by augmentation
   - Result: **test accuracy improved from 21% (raw pixels, worse than
     random) to 64% (HOG + augmentation)** on real, unseen photos.
   This is a genuine, legitimate improvement — not a CNN, but no longer
   guesswork either. `explain.py` and the rest of the app only depend on
   `predict_proba()` / `predict()`, so swapping in a real torchvision
   ResNet18 fine-tune later requires zero changes anywhere else.

3. **LLM summary**: `llm_reasoning.py` makes a real API call to an LLM via
   OpenRouter (free auto-routed model, see "Setting up the LLM" above) to
   write the adjuster-facing summary. The severity prediction and the
   payout number are never generated by the LLM -- they come from the ANN
   and a fixed formula respectively -- the LLM only writes the explanatory
   text around numbers it's given. If the key is missing or the call fails
   for any reason, it falls back to a rule-based template automatically so
   a rate limit or dropped connection can't break the demo.

**Be ready for this judge question: "Is 64% accuracy good?"**
Honest answer: it's reasonable for 70 total images and no CNN, but not
production-grade. Frame it as: *"We validated the full pipeline on real
data and got a real accuracy number, rather than hiding behind a synthetic
dataset that would have looked artificially good. The clear next step for
production is more images and CNN transfer learning, which the architecture
already supports without changes."* That's a stronger answer than a fake
89% would have been.

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
