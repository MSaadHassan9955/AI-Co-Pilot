"""
train_model.py
---------------
Trains the predictive DEEP LEARNING model (an Artificial Neural Network / MLP
with two hidden layers -- satisfies the hackathon's "ANN, CNN, or RNN" requirement)
that classifies car-damage severity from image pixel + color-histogram features.

>>> UPGRADING TO A REAL CNN <<<
If torch/tensorflow are available in your environment, swap `extract_features()`
and `MLPClassifier` for a torchvision ResNet18 fine-tune -- the rest of the
pipeline (Flask app, Grad-CAM-style explainability, HITL, report generation)
does not need to change, it just calls model.predict_proba() / model.predict().
"""

import os
import csv
import numpy as np
from PIL import Image, ImageEnhance
from skimage.feature import hog
from skimage.color import rgb2gray
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib

BASE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE, "data", "claims.csv")
MODEL_DIR = os.path.join(BASE, "model")
IMG_SIZE = 64
CLASSES = ["minor", "moderate", "severe"]


def extract_features(img: Image.Image) -> np.ndarray:
    """HOG (Histogram of Oriented Gradients) + color histogram features.

    HOG captures local edge/gradient patterns -- dents, scratches, and cracks
    show up as strong, oriented gradients -- which is a far more relevant
    signal for damage detection than raw pixel values. This is a classical
    (pre-deep-learning) computer vision feature extractor, used here because
    this environment has no internet access to install torch/torchvision for
    a real CNN. It is a genuine improvement over raw pixels, not a synthetic
    fix, but it is still not as strong as learned CNN features.
    """
    img = img.resize((128, 128))
    gray = rgb2gray(np.array(img))
    hog_feats = hog(
        gray, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
        block_norm="L2-Hys", feature_vector=True,
    )

    small = img.resize((32, 32))
    arr = np.array(small).astype(np.float32) / 255.0
    hist_feats = []
    for c in range(3):
        hist, _ = np.histogram(arr[:, :, c], bins=16, range=(0, 1))
        hist_feats.extend(hist / (hist.sum() + 1e-6))

    return np.concatenate([hog_feats, np.array(hist_feats)])


def augment_image(img: Image.Image, seed: int) -> Image.Image:
    """Cheap, label-preserving augmentations to stretch a small real dataset:
    horizontal flip, small rotation, brightness jitter. Applied ONLY to the
    training split (never to test data) to avoid leakage."""
    rng = np.random.RandomState(seed)
    out = img

    if rng.rand() > 0.5:
        out = out.transpose(Image.FLIP_LEFT_RIGHT)

    angle = rng.uniform(-12, 12)
    out = out.rotate(angle, fillcolor=(128, 128, 128))

    factor = rng.uniform(0.75, 1.25)
    out = ImageEnhance.Brightness(out).enhance(factor)

    return out


def load_rows():
    with open(CSV_PATH) as f:
        return list(csv.DictReader(f))


AUGMENTS_PER_IMAGE = 6  # stretches ~56 training images into ~390 samples


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("Loading dataset...")
    rows = load_rows()

    # Split FIRST, on raw rows, so augmentation never leaks into the test set.
    train_rows, test_rows = train_test_split(
        rows, test_size=0.2, random_state=42,
        stratify=[r["severity_label"] for r in rows],
    )
    print(f"Train images: {len(train_rows)}, Test images: {len(test_rows)}")

    print(f"Extracting HOG features + augmenting training set "
          f"({AUGMENTS_PER_IMAGE}x per image)...")
    X_train, y_train = [], []
    for r in train_rows:
        img = Image.open(os.path.join(BASE, r["image_path"])).convert("RGB")
        X_train.append(extract_features(img))
        y_train.append(r["severity_label"])
        for k in range(AUGMENTS_PER_IMAGE):
            aug = augment_image(img, seed=hash((r["claim_id"], k)) % (2**31))
            X_train.append(extract_features(aug))
            y_train.append(r["severity_label"])

    X_test, y_test = [], []
    for r in test_rows:
        img = Image.open(os.path.join(BASE, r["image_path"])).convert("RGB")
        X_test.append(extract_features(img))
        y_test.append(r["severity_label"])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)
    print(f"Final training samples (with augmentation): {len(X_train)}")
    print(f"Feature dim: {X_train.shape[1]}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print("Training ANN (MLPClassifier, hidden layers=(128,64)) on HOG features...")
    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        max_iter=600,
        random_state=42,
        early_stopping=False,
        alpha=1e-3,  # a bit more regularization given the small real dataset
    )
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy (on real, un-augmented held-out images): {acc:.3f}")
    print(classification_report(y_test, y_pred))

    joblib.dump(clf, os.path.join(MODEL_DIR, "severity_model.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    print(f"\nSaved model + scaler to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
