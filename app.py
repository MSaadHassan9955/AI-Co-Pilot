"""
app.py
-------
AI Co-Pilot for Insurance Claim Processing -- main web application.

Pipeline:
  1. User uploads photo + text description + tabular claim details.
  2. ANN (deep learning model) classifies damage severity from the photo.
  3. Explainability module produces confidence scores + a saliency heatmap.
  4. LLM-reasoning module turns everything into a plain-English summary +
     suggested payout (LLM used for reasoning only, NOT prediction).
  5. Adjuster reviews on a dashboard and Approves / Rejects / Edits (HITL).
  6. A downloadable PDF report is generated with the full audit trail.

Run:  python app.py   then open http://localhost:5000
"""

import os
import uuid
from flask import Flask, request, render_template, send_file, redirect, url_for
from PIL import Image

from explain import load_model, predict, saliency_map, overlay_heatmap
from llm_reasoning import summarize
from report_gen import generate_report

BASE = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE, "static", "uploads")
HEATMAP_DIR = os.path.join(BASE, "static", "heatmaps")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(HEATMAP_DIR, exist_ok=True)

app = Flask(__name__)

CLF, SCALER = load_model()

# In-memory store of in-flight / finalized claims (fine for a hackathon demo)
CLAIMS = {}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    photo = request.files["photo"]
    description = request.form["description"]
    vehicle_age = int(request.form["vehicle_age"])
    claimed_amount = float(request.form["claimed_amount"])
    policy_type = request.form["policy_type"]

    claim_id = str(uuid.uuid4())[:8]
    img = Image.open(photo.stream).convert("RGB")

    orig_path = os.path.join(UPLOAD_DIR, f"{claim_id}.png")
    img.save(orig_path)

    # --- Deep learning prediction + XAI ---
    heat, pred = saliency_map(img, CLF, SCALER)
    overlay = overlay_heatmap(img, heat)
    heat_path = os.path.join(HEATMAP_DIR, f"{claim_id}_heat.png")
    overlay.save(heat_path)

    # --- LLM-style reasoning / summarization ---
    summary, suggested_payout = summarize(
        description, pred["predicted_class"], pred["confidence"],
        vehicle_age, claimed_amount, policy_type
    )

    CLAIMS[claim_id] = {
        "claim_id": claim_id,
        "description": description,
        "vehicle_age": vehicle_age,
        "claimed_amount": claimed_amount,
        "policy_type": policy_type,
        "predicted_class": pred["predicted_class"],
        "confidence": pred["confidence"],
        "all_confidences": pred["all_confidences"],
        "summary": summary,
        "suggested_payout": suggested_payout,
        "original_image_path": orig_path,
        "heatmap_image_path": heat_path,
        "original_image_rel": os.path.relpath(orig_path, BASE),
        "heatmap_image_rel": os.path.relpath(heat_path, BASE),
    }

    return render_template("review.html", claim=CLAIMS[claim_id])


@app.route("/finalize", methods=["POST"])
def finalize_route():
    claim_id = request.form["claim_id"]
    decision = request.form["decision"]  # approved / rejected / edited
    final_payout = float(request.form["final_payout"])
    adjuster_notes = request.form.get("adjuster_notes", "")

    claim = CLAIMS[claim_id]
    claim["adjuster_decision"] = decision
    claim["adjuster_notes"] = adjuster_notes
    claim["final_payout"] = final_payout if decision != "rejected" else 0.0

    pdf_path = generate_report(claim)
    claim["pdf_path"] = pdf_path

    return render_template("result.html", claim_id=claim_id, decision=decision)


@app.route("/download/<claim_id>")
def download_route(claim_id):
    claim = CLAIMS.get(claim_id)
    if not claim or "pdf_path" not in claim:
        return "Report not found", 404
    return send_file(claim["pdf_path"], as_attachment=True,
                      download_name=f"claim_{claim_id}_report.pdf")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port)
