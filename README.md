# ALPR Gate

A local, fully offline prototype of a license-plate-controlled gate. It reads a webcam feed, detects and reads license plates, and shows **GRANTED** or **DENIED** depending on whether the plate is on an allowed list.

Built with [Fast-ALPR](https://github.com/ankandrew/fast-alpr) (ONNX Runtime), OpenCV and PySide6. Developed and tested on macOS.

## Features

- Live webcam preview with plate detection overlays
- Threaded pipeline: separate capture and inference threads keep the preview smooth
- Fuzzy plate matching that tolerates common OCR confusions (O/0, I/1, B/8, Z/2, S/5, and others)
- Minimum OCR confidence required before access is granted
- Allowed-plate list managed in the GUI and stored in `plates.json`
- Colored, timestamped terminal log with a periodic heartbeat (FPS and worker state)
- One-click cleanup of downloaded models and generated data
- Smoke test that runs the model stack against known-good images

## Requirements

- Python 3.10+
- A webcam
- macOS camera permission for your terminal (System Settings → Privacy & Security → Camera)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python main.py
# verbose logging
LOG_LEVEL=DEBUG python main.py
```

On first run, the two ONNX models (about 25 MB in total) are downloaded. After that everything runs offline.

Add the plates you want to allow in the right-hand panel. When a readable plate matches one of them, the status turns green for a few seconds; otherwise it shows DENIED.

### Smoke test

To check the model stack independently of the webcam:

```bash
python smoketest.py
```

Annotated output images are written to `smoketest-out/`.

## How it works

| File | Role |
| --- | --- |
| `main.py` | Entry point; starts the Qt app and terminal heartbeat |
| `detector.py` | Capture and inference threads around Fast-ALPR (`yolo-v9-t-640-license-plate-end2end` detector, `european-plates-mobile-vit-v2-model` OCR) |
| `matcher.py` | Normalization and fuzzy matching (ambiguous-character folding, then similarity ratio with a 0.85 threshold) |
| `storage.py` | JSON-backed allowed-plate list |
| `gui.py` | PySide6 window: preview, plate manager, status |
| `tui.py` | Rich logging and heartbeat |
| `cache.py` | Inventory and removal of downloaded/generated files |
| `smoketest.py` | Model-stack sanity check |

## Notes

- The OCR model is trained on European plates, so accuracy on other formats may be lower.
- This is a prototype. Do not use it as a real security control: plates can be spoofed and fuzzy matching can produce false positives.
