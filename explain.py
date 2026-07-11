"""
explain.py
----------
Explainable AI module.

Provides:
  1. Confidence scores per class (from the ANN's predict_proba)
  2. An occlusion-based saliency heatmap: slides a small mask over the image,
     re-runs the model on the masked version, and measures how much the
     predicted class's confidence drops. Large drop = that region mattered a
     lot to the model's decision. This is a classic, model-agnostic XAI
     technique in the same family as Grad-CAM (region-importance heatmap over
     the input image) and works without needing gradients from a torch model.

>>> UPGRADING TO REAL GRAD-CAM <<<
If you swap in a torch CNN (see train_model.py notes), replace `saliency_map()`
with the `pytorch-grad-cam` library's GradCAM class pointed at the last conv
layer -- the Flask app and report generator consume the returned heatmap the
same way either way (a 2D numpy array of importance values).
"""

import numpy as np
from PIL import Image
import joblib
import os

from train_model import extract_features, CLASSES

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "model")


def load_model():
    clf = joblib.load(os.path.join(MODEL_DIR, "severity_model.joblib"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    return clf, scaler


def predict(img: Image.Image, clf, scaler):
    feats = extract_features(img).reshape(1, -1)
    feats_s = scaler.transform(feats)
    proba = clf.predict_proba(feats_s)[0]
    pred_idx = int(np.argmax(proba))
    return {
        "predicted_class": CLASSES[pred_idx],
        "confidence": float(proba[pred_idx]),
        "all_confidences": {CLASSES[i]: float(proba[i]) for i in range(len(CLASSES))},
    }


def saliency_map(img: Image.Image, clf, scaler, patch=8, stride=8) -> np.ndarray:
    """Occlusion sensitivity map -- higher value = more important region."""
    img = img.resize((64, 64))
    base_pred = predict(img, clf, scaler)
    target_class = base_pred["predicted_class"]
    base_conf = base_pred["confidence"]

    W, H = img.size
    heat = np.zeros((H, W))
    counts = np.zeros((H, W))

    arr = np.array(img)
    gray_fill = int(arr.mean())

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            occluded = arr.copy()
            occluded[y:y + patch, x:x + patch] = gray_fill
            occ_img = Image.fromarray(occluded)
            pred = predict(occ_img, clf, scaler)
            drop = base_conf - pred["all_confidences"][target_class]
            heat[y:y + patch, x:x + patch] += max(drop, 0)
            counts[y:y + patch, x:x + patch] += 1

    counts[counts == 0] = 1
    heat = heat / counts
    if heat.max() > 0:
        heat = heat / heat.max()
    return heat, base_pred


def overlay_heatmap(img: Image.Image, heat: np.ndarray) -> Image.Image:
    """Blends the saliency heatmap (red = high importance) onto the original image."""
    img = img.resize((64, 64)).convert("RGB")
    base = np.array(img).astype(np.float32)

    heat_rgb = np.zeros_like(base)
    heat_rgb[:, :, 0] = heat * 255  # red channel

    blended = (0.6 * base + 0.4 * heat_rgb).clip(0, 255).astype(np.uint8)
    return Image.fromarray(blended).resize((256, 256), Image.NEAREST)
