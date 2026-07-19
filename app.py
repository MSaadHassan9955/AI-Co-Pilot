"""
app.py
-------
AI Co-Pilot for Insurance Claim Processing -- main web application.

Flow:
  0. Car owner logs in with car_key + password (auth.py, cars_dataset.csv).
  1. Locked vehicle info (age, policy type, number plate) is shown, read-only.
  2. User uploads photo + text description + claimed amount (age/policy are
     pulled automatically from the logged-in car, not typed).
  3. ANN (deep learning model) classifies damage severity from the photo.
  4. Explainability module produces confidence scores + a saliency heatmap.
  5. LLM-reasoning module turns everything into a plain-English summary +
     suggested payout (LLM used for reasoning only, NOT prediction).
  6. Adjuster reviews on a dashboard and Approves / Rejects / Edits (HITL).
  7. A downloadable PDF report is generated with the full audit trail.

Run:  python app.py   then open http://localhost:5000
"""

import os
import uuid
from flask import Flask, request, render_template, send_file, redirect, url_for, session
from PIL import Image

from explain import load_model, predict, saliency_map, overlay_heatmap
from llm_reasoning import summarize
from report_gen import generate_report
from auth import verify_login, get_car_by_key

BASE = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE, "static", "uploads")
HEATMAP_DIR = os.path.join(BASE, "static", "heatmaps")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(HEATMAP_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-in-production")

CLF, SCALER = load_model()

# In-memory store of in-flight / finalized claims (fine for a hackathon demo)
CLAIMS = {}


def login_required(view):
    """Redirects to /login if no car is logged in for this session."""
    def wrapped(*args, **kwargs):
        if "car_key" not in session:
            return redirect(url_for("login_route"))
        return view(*args, **kwargs)
    wrapped.__name__ = view.__name__
    return wrapped


@app.route("/")
def index():
    if "car_key" in session:
        return redirect(url_for("car_info_route"))
    return redirect(url_for("login_route"))


@app.route("/login", methods=["GET", "POST"])
def login_route():
    if request.method == "GET":
        return render_template("login.html", error=None)

    car_key = request.form.get("car_key", "")
    password = request.form.get("password", "")
    car = verify_login(car_key, password)

    if car is None:
        return render_template("login.html", error="Key not registered.")

    session["car_key"] = car["car_key"]
    return redirect(url_for("car_info_route"))


@app.route("/car-info")
@login_required
def car_info_route():
    car = get_car_by_key(session["car_key"])
    if car is None:
        session.pop("car_key", None)
        return redirect(url_for("login_route"))
    return render_template("car_info.html", car=car)


@app.route("/logout")
def logout_route():
    session.pop("car_key", None)
    return redirect(url_for("login_route"))


@app.route("/claim")
@login_required
def claim_form_route():
    car = get_car_by_key(session["car_key"])
    return render_template("index.html", car=car)


@app.route("/predict", methods=["POST"])
@login_required
def predict_route():
    car = get_car_by_key(session["car_key"])
    if car is None:
        return redirect(url_for("login_route"))

    photo = request.files["photo"]
    description = request.form["description"]
    claimed_amount = float(request.form["claimed_amount"])

    # Locked fields come from the logged-in car, NOT the form
    vehicle_age = car["vehicle_age"]
    policy_type = car["policy_type"]

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
        "car_key": car["car_key"],
        "number_plate": car["number_plate"],
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
@login_required
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
@login_required
def download_route(claim_id):
    claim = CLAIMS.get(claim_id)
    if not claim or "pdf_path" not in claim:
        return "Report not found", 404
    return send_file(claim["pdf_path"], as_attachment=True,
                      download_name=f"claim_{claim_id}_report.pdf")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
