"""Threaded ALPR pipeline wrapping Fast-ALPR.

Two background threads:
  - capture: pulls frames from the webcam into a shared slot.
  - inference: reads the latest raw frame, runs detector + OCR, publishes results.

The UI thread reads raw frames at capture rate (smooth preview) and overlays
the most recent detections when they are still fresh (<800 ms old).
"""
from __future__ import annotations

import logging
import platform
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np


_log = logging.getLogger("alpr.detector")


def _scalar_ocr_conf(c) -> float:
    """Normalize Fast-ALPR's OcrResult.confidence to a single float.

    Fast-ALPR types this field as `float | list[float]`. When the OCR model
    emits per-character confidences, `c` is a list; when it aggregates, it's
    a scalar. We take min() of the list (the weakest character is the most
    likely to be misread) so the grant threshold stays conservative.
    """
    if isinstance(c, (int, float)):
        return float(c)
    if isinstance(c, (list, tuple)) and c:
        return float(min(c))
    return 0.0


@dataclass
class Plate:
    text: str
    ocr_confidence: float
    det_confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2


@dataclass
class Processed:
    image: np.ndarray
    plates: list[Plate]
    timestamp: float
    inference_ms: float


@dataclass
class WorkerState:
    camera_ok: bool = False
    model_ok: bool = False
    error: str | None = None
    capture_fps: float = 0.0
    inference_fps: float = 0.0
    raw_frame: np.ndarray | None = None
    processed: Processed | None = None


class ALPRWorker:
    """Owns the camera + ALPR lifecycle. Thread-safe reads for the UI."""

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self._state = WorkerState()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._cap_thread: threading.Thread | None = None
        self._inf_thread: threading.Thread | None = None

    def start(self) -> None:
        self._cap_thread = threading.Thread(target=self._capture_loop, daemon=True, name="capture")
        self._inf_thread = threading.Thread(target=self._inference_loop, daemon=True, name="inference")
        self._cap_thread.start()
        self._inf_thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Signal threads to exit and wait briefly for clean shutdown.

        Joining matters when the caller intends to delete the ONNX model files
        right after stopping — otherwise the inference thread might still hold
        open handles on them.
        """
        self._stop.set()
        for thread in (self._cap_thread, self._inf_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout=timeout)

    def snapshot(self) -> WorkerState:
        with self._lock:
            # Shallow copy with references to arrays — the arrays themselves are not mutated
            # in place after publication, so the UI can read them without further locking.
            return WorkerState(
                camera_ok=self._state.camera_ok,
                model_ok=self._state.model_ok,
                error=self._state.error,
                capture_fps=self._state.capture_fps,
                inference_fps=self._state.inference_fps,
                raw_frame=self._state.raw_frame,
                processed=self._state.processed,
            )

    def _open_camera(self) -> cv2.VideoCapture | None:
        # On macOS, AVFoundation is the right backend; fall back to default if unavailable.
        for backend in (cv2.CAP_AVFOUNDATION, cv2.CAP_ANY):
            cap = cv2.VideoCapture(self.camera_index, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                return cap
            cap.release()
        return None

    def _capture_loop(self) -> None:
        cap = self._open_camera()
        if cap is None:
            err = ("Could not open webcam. Grant camera access in "
                   "System Settings → Privacy & Security → Camera.")
            _log.error("[red]camera: %s[/red]", err)
            with self._lock:
                self._state.camera_ok = False
                self._state.error = err
            return

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        _log.info("[green]camera opened[/green] %dx%d", w, h)
        with self._lock:
            self._state.camera_ok = True

        last_tick = time.time()
        frames_since = 0
        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.02)
                continue
            with self._lock:
                self._state.raw_frame = frame
            frames_since += 1
            now = time.time()
            if now - last_tick >= 1.0:
                with self._lock:
                    self._state.capture_fps = frames_since / (now - last_tick)
                frames_since = 0
                last_tick = now
        cap.release()

    @staticmethod
    def _choose_providers() -> list[str]:
        try:
            import onnxruntime as ort  # noqa: PLC0415
        except ImportError:
            return ["CPUExecutionProvider"]
        available = set(ort.get_available_providers())
        if platform.system() == "Darwin" and "CoreMLExecutionProvider" in available:
            return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def _build_alpr(self):
        from fast_alpr import ALPR  # noqa: PLC0415

        # CoreML EP has a known incompatibility with this detector's end2end
        # graph: the post-NMS output has shape {-1} (dynamic number of
        # detections), and when the frame has zero plates the runtime shape
        # becomes {0}, which CoreML rejects. open_image_models swallows the
        # exception internally so our try/except never fires — instead we get
        # thousands of instant fast-fails that peg a CPU core. Detector must
        # stay on CPU; the OCR graph is static-shape and still benefits from
        # the Neural Engine via CoreML.
        accelerated = self._choose_providers()
        # Detector choices:
        #   * 640-input variant (vs the 384 default): more pixels on plate after
        #     letterboxing from a 1280×720 webcam. A plate occupying ~500 px of
        #     a 720p frame becomes ~150 px at 384 input but ~250 px at 640.
        #     Cost is ~3× more compute (~40 ms on M5 CPU vs ~17 ms), still
        #     comfortably under the 15-FPS cap.
        #   * conf_thresh=0.1 (vs the 0.4 default): very permissive. Anything
        #     plate-shaped surfaces as a yellow box so we can debug. The grant
        #     path in gui.py enforces a much stricter OCR confidence.
        return ALPR(
            detector_model="yolo-v9-t-640-license-plate-end2end",
            ocr_model="european-plates-mobile-vit-v2-model",
            detector_providers=["CPUExecutionProvider"],
            ocr_providers=accelerated,
            detector_conf_thresh=0.1,
        )

    def _inference_loop(self) -> None:
        accelerated = self._choose_providers()
        _log.info(
            "loading models…  detector=[cyan]CPU[/cyan]  ocr=[cyan]%s[/cyan]",
            ",".join(accelerated),
        )
        t0 = time.time()
        try:
            alpr = self._build_alpr()
        except Exception as exc:  # noqa: BLE001
            _log.exception("[red]model load failed[/red]")
            with self._lock:
                self._state.model_ok = False
                self._state.error = f"Model load failed: {exc}"
            return

        _log.info("[green]models loaded[/green] in %.1fs", time.time() - t0)
        with self._lock:
            self._state.model_ok = True

        # Hard cap on inference rate. Protects us from runaway loops if a future
        # library-level error path silently fast-fails (as CoreML+end2end did),
        # and also keeps the CPU cool for a gate that doesn't need 30 fps.
        min_interval_sec = 1.0 / 15.0

        last_tick = time.time()
        inferences_since = 0
        while not self._stop.is_set():
            loop_start = time.time()
            with self._lock:
                frame = self._state.raw_frame
            if frame is None:
                time.sleep(0.03)
                continue

            frame_copy = frame.copy()  # decouple from capture thread before inference
            t0 = time.time()
            try:
                results = alpr.predict(frame_copy)
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    self._state.error = f"Inference error: {exc}"
                time.sleep(0.1)
                continue
            inference_ms = (time.time() - t0) * 1000.0

            plates = _parse_results(results)

            with self._lock:
                self._state.processed = Processed(
                    image=frame_copy,
                    plates=plates,
                    timestamp=time.time(),
                    inference_ms=inference_ms,
                )
                inferences_since += 1
                now = time.time()
                if now - last_tick >= 1.0:
                    self._state.inference_fps = inferences_since / (now - last_tick)
                    inferences_since = 0
                    last_tick = now

            # Enforce the min interval: sleep if this iteration finished faster.
            elapsed = time.time() - loop_start
            if elapsed < min_interval_sec:
                time.sleep(min_interval_sec - elapsed)


def _parse_results(results) -> list[Plate]:
    parsed: list[Plate] = []
    for r in results or []:
        try:
            det = getattr(r, "detection", None)
            ocr = getattr(r, "ocr", None)
            if det is None:
                continue
            bbox = det.bounding_box
            text = (ocr.text if ocr and ocr.text else "").strip().upper()
            parsed.append(Plate(
                text=text,
                ocr_confidence=_scalar_ocr_conf(ocr.confidence) if ocr else 0.0,
                det_confidence=float(det.confidence),
                bbox=(float(bbox.x1), float(bbox.y1), float(bbox.x2), float(bbox.y2)),
            ))
        except Exception as exc:  # noqa: BLE001
            # Log instead of silently swallowing. A bare `pass` here previously
            # hid a TypeError on list-typed OCR confidence and cost us hours of
            # debugging. Always log; never let bugs hide behind an except.
            _log.warning("could not parse detection result: %r", exc)
            continue
    return parsed
