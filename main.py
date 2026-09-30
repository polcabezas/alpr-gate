"""ALPR Gate Prototype - local webcam license-plate recognition.

First-time setup:
    cd alpr-gate
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

Run:
    python main.py

Verbose logging:
    LOG_LEVEL=DEBUG python main.py

On first launch macOS will ask for camera permission; grant it in
System Settings - Privacy & Security - Camera.

The two ONNX models (~25 MB total) download automatically the first time
you run inference. After that, everything runs fully offline.
"""
from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from gui import AppWindow
from tui import Heartbeat, configure_logging


def main() -> None:
    log = configure_logging()

    app = QApplication(sys.argv)
    window = AppWindow()
    window.show()

    heartbeat = Heartbeat(window.worker, plate_count=lambda: len(window.store.list()))
    heartbeat.start()

    try:
        exit_code = app.exec()
    finally:
        heartbeat.stop()
        log.info("[bold]shutdown complete[/bold]")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
