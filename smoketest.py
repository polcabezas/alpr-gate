"""Smoke test for the ALPR pipeline.

Runs the same detector+OCR stack the app uses, but against known-good test
images (not the webcam). If this fails, the model stack itself is broken —
which is the information we need to know before spending any more time
debugging the webcam-specific path.

Usage:
    source .venv/bin/activate
    python smoketest.py

The annotated output images land in ./smoketest-out/ so you can open them
in Finder and see exactly what the detector did or did not find.
"""
from __future__ import annotations

import shutil
import sys
import urllib.request
from pathlib import Path

import cv2


OUT_DIR = Path(__file__).parent / "smoketest-out"


def scalar_ocr_conf(c) -> float:
    """Fast-ALPR types OCR confidence as `float | list[float]` (per-char).
    Normalize to a scalar using min() — see detector.py for rationale."""
    if isinstance(c, (int, float)):
        return float(c)
    if isinstance(c, (list, tuple)) and c:
        return float(min(c))
    return 0.0

# Test inputs — the fast-alpr repo ships its own test image for exactly this
# purpose. If the detector can't find a plate in this one, the model stack
# is wrong and no amount of webcam tuning will help.
#
# Note the `master` branch — fast-alpr's default branch isn't `main`. This
# caught us once; if the primary URL breaks again, the loader below will
# try each candidate in order.
TEST_IMAGES: list[tuple[str, list[str]]] = [
    (
        "fast_alpr_official_test",
        [
            "https://raw.githubusercontent.com/ankandrew/fast-alpr/master/assets/test_image.png",
            "https://raw.githubusercontent.com/ankandrew/fast-alpr/main/assets/test_image.png",
        ],
    ),
]

# If the app's Save-frame button has been used, pick up that frame too.
WEBCAM_FRAME = Path(__file__).parent / "debug-frame.jpg"


def download(urls: list[str], dest: Path) -> bool:
    """Try each candidate URL; return True if any succeeded."""
    if dest.exists():
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    for url in urls:
        try:
            print(f"    trying  {url}")
            urllib.request.urlretrieve(url, str(dest))
            print(f"    ok      ({dest.stat().st_size} bytes)")
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"    failed  {exc}")
    return False


def annotate(img, results) -> None:
    for r in results:
        det = getattr(r, "detection", None)
        ocr = getattr(r, "ocr", None)
        if det is None:
            continue
        bb = det.bounding_box
        cv2.rectangle(img, (bb.x1, bb.y1), (bb.x2, bb.y2), (0, 230, 0), 3)
        text = ocr.text if ocr and ocr.text else "?"
        caption = f"{text}  det={det.confidence:.2f}"
        if ocr:
            caption += f"  ocr={scalar_ocr_conf(ocr.confidence):.2f}"
        cv2.putText(
            img, caption, (bb.x1, max(20, bb.y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 0), 2, cv2.LINE_AA,
        )


def run_one(name: str, img_path: Path, alpr) -> int:
    if not img_path.exists():
        print(f"  [skip] {img_path} does not exist")
        return -1
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  [error] cv2.imread could not decode {img_path}")
        return -1
    print(f"  image:  {img.shape[1]}x{img.shape[0]} {img.dtype}")

    results = alpr.predict(img)
    print(f"  detections: {len(results)}")
    for i, r in enumerate(results):
        det = getattr(r, "detection", None)
        ocr = getattr(r, "ocr", None)
        if det is None:
            continue
        bb = det.bounding_box
        text = ocr.text if ocr and ocr.text else "(none)"
        ocr_conf = scalar_ocr_conf(ocr.confidence) if ocr else 0.0
        print(
            f"    [{i}] bbox=({bb.x1},{bb.y1})-({bb.x2},{bb.y2})  "
            f"det={det.confidence:.2f}  text={text!r}  ocr={ocr_conf:.2f}"
        )

    annotate(img, results)
    out = OUT_DIR / f"{name}_annotated.jpg"
    cv2.imwrite(str(out), img)
    print(f"  annotated: {out}")
    return len(results)


def main() -> int:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    print("Loading ALPR (may download models on first run)…")
    from fast_alpr import ALPR  # noqa: PLC0415

    alpr = ALPR(
        detector_model="yolo-v9-t-640-license-plate-end2end",
        ocr_model="european-plates-mobile-vit-v2-model",
        detector_providers=["CPUExecutionProvider"],
        detector_conf_thresh=0.1,
    )
    print("ALPR loaded.\n")

    summary: list[tuple[str, int]] = []

    for name, urls in TEST_IMAGES:
        print(f"=== {name} ===")
        img_path = OUT_DIR / f"{name}_input.png"
        if not download(urls, img_path):
            print(f"  [error] all candidate URLs failed for {name}")
            summary.append((name, -1))
            print()
            continue
        count = run_one(name, img_path, alpr)
        summary.append((name, count))
        print()

    if WEBCAM_FRAME.exists():
        print(f"=== webcam (your saved frame: {WEBCAM_FRAME.name}) ===")
        count = run_one("webcam_debug_frame", WEBCAM_FRAME, alpr)
        summary.append(("webcam_debug_frame", count))
        print()
    else:
        print(f"(no webcam frame at {WEBCAM_FRAME} — click 'Save current frame' "
              "in the app while a plate is visible, then re-run.)\n")

    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for name, n in summary:
        status = "SKIP" if n < 0 else ("FAIL  (0 detections)" if n == 0 else f"OK    ({n} detection(s))")
        print(f"  {name:35s}  {status}")

    print()
    any_ran = any(n >= 0 for _, n in summary)
    any_ok = any(n > 0 for _, n in summary)
    any_fail_zero = any(n == 0 for _, n in summary)

    if not any_ran:
        print("DIAGNOSIS: Could not run any test — nothing reached the model.")
        print("           (Download failures. Check your network or the URLs in")
        print("           smoketest.py's TEST_IMAGES list.)")
        return 2

    if not any_ok:
        print("DIAGNOSIS: Model stack itself appears broken — not a single detection")
        print("           on a known-good image. Most likely causes:")
        print("             * Fast-ALPR / open-image-models package mismatch")
        print("             * Corrupted cached ONNX model file")
        print("             * end2end graph with hardcoded internal threshold")
        print("           Try: pip install -U fast-alpr  OR  clear cache & re-download.")
        return 1

    if any_ok and any_fail_zero:
        print("DIAGNOSIS: Pipeline works on the official test image but fails on")
        print("           your webcam frame. The detector has a blind spot on")
        print("           your specific input. Most likely: out-of-distribution")
        print("           content (rendered graphics on a screen vs real car).")
        print("           Try pointing the webcam at a photo of a real car.")
        return 0

    print("DIAGNOSIS: All inputs produced detections. The pipeline works.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
