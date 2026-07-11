"""
generate_data.py
-----------------
Creates a synthetic 'car damage' dataset so the whole pipeline is self-contained
and trainable in seconds, with NO internet access required.

>>> SWAP-IN INSTRUCTIONS FOR THE REAL DATASET <<<
Replace this script's output with a real Kaggle car-damage dataset:
  1. Download e.g. https://www.kaggle.com/search?q=car+damage
  2. Put images into data/images/minor/, data/images/moderate/, data/images/severe/
  3. Update data/claims.csv with real claim_id, image_path, description, vehicle_age,
     claimed_amount, policy_type, severity_label columns.
  4. Re-run train_model.py -- no other code changes needed.
"""

import os
import random
import csv
import numpy as np
from PIL import Image, ImageDraw

random.seed(42)
np.random.seed(42)

BASE = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE, "data", "images")
CSV_PATH = os.path.join(BASE, "data", "claims.csv")
IMG_SIZE = 64

SEVERITIES = ["minor", "moderate", "severe"]

DESC_TEMPLATES = {
    "minor": [
        "Small scratch on the rear bumper from a parking lot incident.",
        "Light scuff mark on the front fender, no dents visible.",
        "Minor paint chip near the door handle after a shopping cart bump.",
    ],
    "moderate": [
        "Noticeable dent on the driver-side door after a low-speed collision.",
        "Cracked headlight and bent bumper panel from a rear-end tap.",
        "Multiple scratches and a dent on the trunk lid from a reversing accident.",
    ],
    "severe": [
        "Major front-end collision with crumpled hood and shattered headlight.",
        "Side panel caved in, airbags deployed, extensive structural damage.",
        "Vehicle rear totaled after highway collision, frame damage suspected.",
    ],
}

POLICY_TYPES = ["Basic", "Standard", "Premium"]


def make_damage_image(severity: str) -> Image.Image:
    """Procedurally draws a 'car body' with damage marks whose count/size scale with severity."""
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (180, 180, 190))  # car body base color
    draw = ImageDraw.Draw(img)
    # car body silhouette
    draw.rounded_rectangle([4, 20, 60, 50], radius=8, fill=(70, 90, 140))

    if severity == "minor":
        n_marks, size_range, color = random.randint(1, 2), (2, 5), (200, 200, 60)
    elif severity == "moderate":
        n_marks, size_range, color = random.randint(3, 5), (5, 10), (230, 140, 30)
    else:
        n_marks, size_range, color = random.randint(6, 10), (8, 16), (200, 30, 30)

    for _ in range(n_marks):
        x, y = random.randint(6, 55), random.randint(20, 48)
        s = random.randint(*size_range)
        shape = random.choice(["ellipse", "line"])
        if shape == "ellipse":
            draw.ellipse([x, y, x + s, y + s], fill=color)
        else:
            draw.line([x, y, x + s, y + random.randint(-s, s)], fill=color, width=2)

    # slight random noise for realism / model robustness
    arr = np.array(img).astype(np.int16)
    arr += np.random.randint(-8, 8, arr.shape)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def main():
    for s in SEVERITIES:
        os.makedirs(os.path.join(IMG_DIR, s), exist_ok=True)

    rows = []
    claim_id = 1000
    for severity in SEVERITIES:
        for i in range(60):  # 60 images per class = 180 total, fast to train
            img = make_damage_image(severity)
            fname = f"{severity}_{i}.png"
            fpath = os.path.join(IMG_DIR, severity, fname)
            img.save(fpath)

            claim_id += 1
            vehicle_age = int(np.clip(np.random.normal(5, 3), 0, 20))
            base_amount = {"minor": 300, "moderate": 2500, "severe": 9000}[severity]
            claimed_amount = round(base_amount + np.random.normal(0, base_amount * 0.2), 2)
            policy_type = random.choice(POLICY_TYPES)
            description = random.choice(DESC_TEMPLATES[severity])

            rows.append({
                "claim_id": claim_id,
                "image_path": os.path.relpath(fpath, BASE),
                "description": description,
                "vehicle_age": vehicle_age,
                "claimed_amount": claimed_amount,
                "policy_type": policy_type,
                "severity_label": severity,
            })

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} synthetic claims -> {CSV_PATH}")


if __name__ == "__main__":
    main()
