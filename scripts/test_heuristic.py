#!/usr/bin/env python3
"""Test heuristic garage door detection on sample images.

Computes multiple metrics on a central ROI and compares them
to determine if the garage door is open or closed.

Usage:
    python scripts/test_heuristic.py <image_path> <expected_status>
    python scripts/test_heuristic.py --batch

Batch mode tests all images defined in SAMPLES below.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

# ── Sample images with ground truth ──────────────────────────────────
BOX = Path.home() / "Downloads/box/converted"
DL = Path.home() / "Downloads/box/converted"

SAMPLES = [
    # Original test set
    (DL / "IMG_6518.JPG", "open"),   # IR night, open
    (DL / "IMG_6533.JPG", "open"),   # Day, open, no car
    (DL / "IMG_6532.JPG", "open"),   # Day, partially open
    (DL / "IMG_6531.JPG", "closed"), # IR night, closed
    (DL / "IMG_6530.JPG", "open"),   # Day, open, car
    (DL / "IMG_6529.JPG", "open"),   # Dusk, open
    (DL / "IMG_6528.JPG", "closed"), # IR night, closed, car

    # image_* series: IR night — closed(1-6), open(7-45), closed(46-49)
    *[(BOX / f"image_{i:02d}.jpg", "closed") for i in range(1, 7)],
    *[(BOX / f"image_{i:02d}.jpg", "open") for i in range(7, 46)],
    *[(BOX / f"image_{i:02d}.jpg", "closed") for i in range(46, 50)],

    # image2_* series: night with light — open(1-17), closed(18-23), open(24-39)
    *[(BOX / f"image2_{i}.jpg", "open") for i in range(1, 18)],
    *[(BOX / f"image2_{i}.jpg", "closed") for i in range(18, 24)],
    *[(BOX / f"image2_{i}.jpg", "open") for i in range(24, 40)],

    # Extra
    (BOX / "IMG_6535.JPG", "open"),
]

# ── ROI: central portion of the image where the door is ─────────────
# Expressed as fractions of image width/height.
ROI_X_START = 0.25
ROI_X_END = 0.65
ROI_Y_START = 0.20
ROI_Y_END = 0.85


def extract_roi(img: np.ndarray) -> np.ndarray:
    """Extract the central ROI from the image."""
    h, w = img.shape[:2]
    x1 = int(w * ROI_X_START)
    x2 = int(w * ROI_X_END)
    y1 = int(h * ROI_Y_START)
    y2 = int(h * ROI_Y_END)
    return img[y1:y2, x1:x2]


def compute_metrics(image_path: str) -> dict:
    """Compute heuristic metrics for a garage image."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")

    roi = extract_roi(img)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 1. Edge detection
    edges = cv2.Canny(gray, 30, 100)
    total_pixels = h * w
    edge_density = np.count_nonzero(edges) / total_pixels

    # 2. Horizontal vs diagonal vs vertical lines via Hough
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=30,
        minLineLength=w * 0.15, maxLineGap=15,
    )

    horizontal_count = 0
    diagonal_count = 0
    vertical_count = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                vertical_count += 1
                continue
            angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
            if angle < 20 or angle > 160:  # ~horizontal
                horizontal_count += 1
            elif 25 < angle < 65 or 115 < angle < 155:  # diagonal
                diagonal_count += 1
            elif 70 < angle < 110:  # ~vertical
                vertical_count += 1

    total_lines = horizontal_count + diagonal_count + vertical_count
    horizontal_ratio = horizontal_count / total_lines if total_lines > 0 else 0.5

    # 3. Texture variance (stddev of pixel intensities)
    pixel_stddev = float(np.std(gray))

    # 4. Entropy (information content)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    entropy = -float(np.sum(hist * np.log2(hist)))

    # 5. Vertical gradient strength (strong = panel joints present = closed)
    sobel_h = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # horizontal edges
    sobel_v = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # vertical edges
    h_strength = float(np.mean(np.abs(sobel_h)))
    v_strength = float(np.mean(np.abs(sobel_v)))
    hv_ratio = h_strength / v_strength if v_strength > 0 else 999

    return {
        "edge_density": edge_density,
        "horizontal_lines": horizontal_count,
        "diagonal_lines": diagonal_count,
        "vertical_count": vertical_count,
        "total_lines": total_lines,
        "horizontal_ratio": horizontal_ratio,
        "pixel_stddev": pixel_stddev,
        "entropy": entropy,
        "sobel_h_strength": h_strength,
        "sobel_v_strength": v_strength,
        "hv_ratio": hv_ratio,
    }


def classify(metrics: dict) -> str:
    """Rule-based classification using two regimes: day and IR/night.

    Day (high edge density): the outdoor scene creates much more edge detail
    than the smooth door panels. Edge density alone separates them cleanly.

    IR/night (low edge density): fewer edges overall, so we rely on line
    direction analysis — diagonal lines from the fence = open, horizontal
    panel joints = closed — plus the Sobel H/V ratio.
    """
    edge_density = metrics["edge_density"]

    # Day regime: edge density cleanly separates open (>0.10) vs closed (<0.08)
    # Open day images: 0.17-0.19. Closed day: 0.05-0.08.
    if edge_density > 0.10:
        return "open"

    # IR/night regime: low edge density, use line geometry + sobel
    score = 0  # positive = closed, negative = open

    # Diagonal lines dominate = fence visible = open
    h = metrics["horizontal_lines"]
    d = metrics["diagonal_lines"]
    if d > 0 and d > h * 2:
        score -= 2
    elif h > 0 and h >= d:
        score += 1

    # Sobel H/V ratio: closed panels have stronger horizontal gradients
    if metrics["hv_ratio"] > 1.25:
        score += 1
    elif metrics["hv_ratio"] < 1.0:
        score -= 1

    # Very low edge density + low entropy = smooth panels = closed
    if edge_density < 0.03 and metrics["entropy"] < 6.5:
        score += 1

    return "closed" if score > 0 else "open"


def analyze_image(image_path: str | Path, expected: str | None = None) -> None:
    """Analyze a single image and print results."""
    path = Path(image_path)
    metrics = compute_metrics(str(path))
    prediction = classify(metrics)

    match_str = ""
    if expected:
        match = prediction == expected
        match_str = f"  {'✅' if match else '❌ WRONG'} (expected: {expected})"

    print(f"\n{'='*60}")
    print(f"📷 {path.name}  →  {prediction.upper()}{match_str}")
    print(f"{'─'*60}")
    print(f"  edge_density:     {metrics['edge_density']:.4f}")
    print(f"  lines (H/D/V):    {metrics['horizontal_lines']}/{metrics['diagonal_lines']}/{metrics['vertical_count']}")
    print(f"  horizontal_ratio: {metrics['horizontal_ratio']:.3f}")
    print(f"  pixel_stddev:     {metrics['pixel_stddev']:.1f}")
    print(f"  entropy:          {metrics['entropy']:.2f}")
    print(f"  sobel H/V ratio:  {metrics['hv_ratio']:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Test garage door heuristic")
    parser.add_argument("image", nargs="?", help="Path to image")
    parser.add_argument("expected", nargs="?", choices=["open", "closed"])
    parser.add_argument("--batch", action="store_true", help="Run on all samples")
    args = parser.parse_args()

    if args.batch:
        correct = 0
        total = 0
        for path, expected in SAMPLES:
            if not path.exists():
                print(f"\n⚠️  Skipping {path.name} (not found)")
                continue
            analyze_image(path, expected)
            metrics = compute_metrics(str(path))
            if classify(metrics) == expected:
                correct += 1
            total += 1

        print(f"\n{'='*60}")
        print(f"📊 Results: {correct}/{total} correct ({correct/total*100:.0f}%)")
        print(f"{'='*60}")
    elif args.image:
        analyze_image(args.image, args.expected)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
