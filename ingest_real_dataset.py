"""
ingest_real_dataset.py
------------------------
Replaces the synthetic dataset with the REAL Kaggle dataset the handbook
pointed to: "coco-car-damage-detection-dataset" (COCO-format damage
annotations over real car photos).

IMPORTANT -- what's real here and what isn't:
  - Images: 100% real photos from the Kaggle dataset.
  - Severity label (minor/moderate/severe): DERIVED from real annotation
    data. The dataset gives per-image damage-region area (COCO 'area'
    field). We compute damage_ratio = total_annotated_damage_area /
    image_area for every image, then split into 3 tertiles across the
    actual distribution -- so "severe" really does mean "most heavily
    annotated damage in this dataset," not a guess.
  - Text description & tabular fields (vehicle_age, claimed_amount,
    policy_type): this Kaggle dataset does NOT include those, so they are
    still templated/synthetic, same as before. This is disclosed in the
    README -- if you find/attach a real insurance-claims tabular dataset,
    swap the generation logic below for a real join.

Run this once after placing the extracted Kaggle dataset (folders: img/,
train/, val/, test/) into raw_dataset/ next to this script.
"""

import os
import json
import shutil
import random
import csv
from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE, "raw_dataset")          # extracted Kaggle archive.zip goes here
IMG_DIR = os.path.join(BASE, "data", "images")
CSV_PATH = os.path.join(BASE, "data", "claims.csv")
SEVERITIES = ["minor", "moderate", "severe"]

random.seed(42)

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


def load_damage_areas(json_path):
    """Returns {file_name: damage_ratio} using the REAL COCO annotation areas."""
    d = json.load(open(json_path))
    img_map = {im["id"]: im for im in d["images"]}
    area_by_img = {}
    for ann in d["annotations"]:
        iid = ann["image_id"]
        area_by_img[iid] = area_by_img.get(iid, 0) + ann["area"]

    result = {}
    for iid, info in img_map.items():
        total_area = area_by_img.get(iid, 0)
        ratio = total_area / (info["width"] * info["height"])
        result[info["file_name"]] = ratio
    return result


def main():
    train_json = os.path.join(RAW_DIR, "train", "COCO_train_annos.json")
    val_json = os.path.join(RAW_DIR, "val", "COCO_val_annos.json")
    if not os.path.exists(train_json):
        raise FileNotFoundError(
            f"Expected the extracted Kaggle dataset at {RAW_DIR}/ "
            f"(with train/, val/, img/ subfolders). Extract archive.zip there first."
        )

    ratios = {}
    ratios.update(load_damage_areas(train_json))
    ratios.update(load_damage_areas(val_json))
    print(f"Loaded real damage-area ratios for {len(ratios)} images.")

    # Data-driven severity thresholds (tertiles of the REAL distribution)
    sorted_ratios = sorted(ratios.values())
    n = len(sorted_ratios)
    t1 = sorted_ratios[n // 3]
    t2 = sorted_ratios[2 * n // 3]
    print(f"Derived thresholds -> minor < {t1:.4f} <= moderate < {t2:.4f} <= severe")

    def bucket(r):
        if r < t1:
            return "minor"
        elif r < t2:
            return "moderate"
        else:
            return "severe"

    for s in SEVERITIES:
        os.makedirs(os.path.join(IMG_DIR, s), exist_ok=True)

    rows = []
    claim_id = 2000
    img_src_dir = os.path.join(RAW_DIR, "img")
    for fname, ratio in ratios.items():
        severity = bucket(ratio)
        src = os.path.join(img_src_dir, fname)
        if not os.path.exists(src):
            print(f"  [skip] missing image file: {fname}")
            continue

        dst = os.path.join(IMG_DIR, severity, fname)
        shutil.copyfile(src, dst)

        claim_id += 1
        vehicle_age = int(max(0, min(20, round(random.gauss(5, 3)))))
        base_amount = {"minor": 300, "moderate": 2500, "severe": 9000}[severity]
        claimed_amount = round(base_amount + random.gauss(0, base_amount * 0.2), 2)
        policy_type = random.choice(POLICY_TYPES)
        description = random.choice(DESC_TEMPLATES[severity])

        rows.append({
            "claim_id": claim_id,
            "image_path": os.path.relpath(dst, BASE),
            "description": description,
            "vehicle_age": vehicle_age,
            "claimed_amount": claimed_amount,
            "policy_type": policy_type,
            "severity_label": severity,
            "damage_area_ratio": round(ratio, 5),  # kept for transparency/audit
        })

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    counts = {s: sum(1 for r in rows if r["severity_label"] == s) for s in SEVERITIES}
    print(f"Wrote {len(rows)} real-image claims -> {CSV_PATH}")
    print(f"Class counts: {counts}")


if __name__ == "__main__":
    main()
