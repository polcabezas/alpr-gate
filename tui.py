"""Terminal log stream for real-time visibility.

Uses Rich for colored, timestamped output. Runs alongside the GUI — events
logged from any module show up in the terminal where you launched the app.

A Heartbeat thread additionally logs current FPS and worker state every few
seconds so the terminal never goes silent for long, even when nothing is
actively happening.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Callable

from rich.console import Console
from rich.logging import RichHandler

from detector import ALPRWorker


ROOT_LOGGER = "alpr"


def configure_logging() -> logging.Logger:
    """Install a colored, timestamped handler. Call once at program start."""
    # Respect LOG_LEVEL=DEBUG from the env for verbose mode; default to INFO.
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    handler = RichHandler(
        console=Console(stderr=False),
        rich_tracebacks=True,
        show_path=False,
        markup=True,
        show_time=True,
        show_level=True,
    )
    logging.basicConfig(
        level=level,
        format="%(name)s  %(message)s",
        datefmt="[%H:%M:%S]",
        handlers=[handler],
        force=True,  # override any config imported libraries may have set
    )
    # Silence the chattiest dependencies. open_image_models logs a warning on
    # *every* empty-scene inference when the CoreML EP had its dynamic-shape
    # issue; even after we pin the detector to CPU we keep it quiet so a
    # future model swap can't flood the terminal again.
    for noisy, level in (
        ("onnxruntime", logging.WARNING),
        ("huggingface_hub", logging.WARNING),
        ("PIL", logging.WARNING),
        ("open_image_models", logging.ERROR),
        ("open_image_models.detection.core.yolo_v9.inference", logging.ERROR),
    ):
        logging.getLogger(noisy).setLevel(level)

    # Silence the ORT C++ runtime's own stderr stream (it writes directly,
    # bypassing Python's logging). Severity 3 = errors only, 4 = fatal only.
    try:
        import onnxruntime as ort  # noqa: PLC0415

        ort.set_default_logger_severity(3)
    except ImportError:
        pass

    log = logging.getLogger(ROOT_LOGGER)
    log.info("[bold green]ALPR Gate starting[/bold green]  (log level=%s)", level_name)
    return log


class Heartbeat:
    """Periodic state dump so the terminal never goes silent for long."""

    def __init__(
        self,
        worker: ALPRWorker,
        plate_count: Callable[[], int],
        interval_sec: float = 8.0,
    ):
        self.worker = worker
        self.plate_count = plate_count
        self.interval = interval_sec
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._log = logging.getLogger(f"{ROOT_LOGGER}.beat")

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="heartbeat")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        # Event.wait returns True when set (stop requested), False on timeout —
        # so the loop exits cleanly the moment stop() is called.
        while not self._stop.wait(self.interval):
            snap = self.worker.snapshot()
            if snap.error and not snap.camera_ok:
                self._log.warning("camera unavailable — [red]%s[/red]", snap.error)
                continue
            if not snap.model_ok:
                self._log.info("[yellow]loading models…[/yellow]")
                continue
            inf_ms = f"{snap.processed.inference_ms:.0f}ms" if snap.processed else "—"
            # Live detection count from the most recent processed frame — this is
            # how we tell from the terminal alone whether the detector is firing.
            # If it stays at 0 with plates clearly in view, the detector has
            # a blind spot. Separate from the allowed-list count.
            dets = len(snap.processed.plates) if snap.processed else 0
            self._log.info(
                "cap=[cyan]%.1f[/cyan]fps  inf=[cyan]%.1f[/cyan]fps (%s)  "
                "detections=[cyan]%d[/cyan]  allowed=[cyan]%d[/cyan]",
                snap.capture_fps, snap.inference_fps, inf_ms,
                dets, self.plate_count(),
            )
