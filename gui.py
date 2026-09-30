"""PySide6 UI: camera preview on the left, plate manager + status on the right."""
from __future__ import annotations

import logging
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QCloseEvent, QImage, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

import cache
from detector import ALPRWorker, Plate
from matcher import best_match
from storage import PlateStore


_log = logging.getLogger("alpr.gui")


UI_REFRESH_MS = 33                     # ~30 fps UI redraw
STABILITY_FRAMES = 3                   # same plate must appear in N consecutive inferences
GRANT_DISPLAY_SEC = 3.0                # hold GRANTED on screen this long
DETECTION_FRESH_SEC = 0.8              # overlay boxes only while inference is this fresh
DISPLAY_MIN_OCR_CONFIDENCE = 0.25      # show a red/green box above this OCR confidence
GRANT_MIN_OCR_CONFIDENCE = 0.5         # only GRANT if OCR confidence is at least this


_STATUS_STYLE = """
    QLabel {{
        background-color: {bg};
        color: white;
        font-size: 22pt;
        font-weight: 700;
        padding: 14px;
        border-radius: 6px;
    }}
"""


class AppWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ALPR Gate Prototype")
        self.resize(1200, 720)
        self.setMinimumSize(960, 600)

        self.store = PlateStore(Path(__file__).parent / "plates.json")
        self.worker = ALPRWorker(camera_index=0)

        self._match_streak: deque[str] = deque(maxlen=STABILITY_FRAMES)
        self._granted_until: float = 0.0
        self._granted_plate: str | None = None
        self._prev_status: str | None = None
        self._last_detection_text: str | None = None

        self._build_ui()
        self.worker.start()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(UI_REFRESH_MS)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # Left: camera
        cam_frame = QFrame()
        cam_frame.setFrameShape(QFrame.StyledPanel)
        cam_layout = QVBoxLayout(cam_frame)
        cam_layout.setContentsMargins(4, 4, 4, 4)
        self.video_label = QLabel("Connecting to camera…")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: #888;")
        self.video_label.setMinimumSize(640, 480)
        cam_layout.addWidget(self.video_label)
        root.addWidget(cam_frame, 3)

        # Right: sidebar
        sidebar = QWidget()
        sidebar.setMinimumWidth(280)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(0, 0, 0, 0)
        root.addWidget(sidebar, 1)

        self.status_label = QLabel("STARTING")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet(_STATUS_STYLE.format(bg="#444"))
        side.addWidget(self.status_label)

        self.detail_label = QLabel("")
        self.detail_label.setStyleSheet("font-size: 12pt; color: #333;")
        self.detail_label.setWordWrap(True)
        side.addWidget(self.detail_label)

        side.addWidget(_hline())

        title = QLabel("Allowed plates")
        title.setStyleSheet("font-size: 13pt; font-weight: 600;")
        side.addWidget(title)

        self.plate_list = QListWidget()
        self.plate_list.setStyleSheet("font-family: Menlo, monospace; font-size: 13pt;")
        side.addWidget(self.plate_list, 1)
        self._refresh_list()

        add_row = QHBoxLayout()
        self.add_input = QLineEdit()
        self.add_input.setPlaceholderText("e.g. 1234ABC")
        self.add_input.returnPressed.connect(self._on_add)
        add_row.addWidget(self.add_input, 1)
        add_btn = QPushButton("Add")
        add_btn.clicked.connect(self._on_add)
        add_row.addWidget(add_btn)
        side.addLayout(add_row)

        remove_btn = QPushButton("Remove selected")
        remove_btn.clicked.connect(self._on_remove)
        side.addWidget(remove_btn)

        # Diagnostic: dump the current raw webcam frame to disk for offline
        # analysis (e.g. feeding it to smoketest.py to isolate whether the
        # frame itself is the reason detections are failing).
        self.save_frame_btn = QPushButton("Save current frame")
        self.save_frame_btn.setToolTip(
            "Save the exact frame the detector is currently seeing to "
            "debug-frame.jpg in this folder, for offline debugging."
        )
        self.save_frame_btn.clicked.connect(self._on_save_frame)
        side.addWidget(self.save_frame_btn)

        side.addWidget(_hline())

        self.meta_label = QLabel("starting up…")
        self.meta_label.setStyleSheet("font-family: Menlo, monospace; font-size: 10pt; color: #666;")
        self.meta_label.setWordWrap(True)
        side.addWidget(self.meta_label)

        # Destructive action pinned to the bottom of the sidebar.
        # Qt treats "&" in a button label as a mnemonic-marker, so we escape it as "&&".
        self.clear_quit_btn = QPushButton("Clear all data && quit")
        self.clear_quit_btn.setToolTip(
            "Delete all app-generated data: downloaded models, allowed-plate list, "
            "debug frames, and Python bytecode caches. "
            "Does not touch the virtual-environment or installed packages."
        )
        self.clear_quit_btn.setStyleSheet(
            "QPushButton { background-color: #a43636; color: white; "
            "padding: 8px; border-radius: 4px; font-weight: 600; margin-top: 8px; } "
            "QPushButton:hover { background-color: #bc4040; } "
            "QPushButton:pressed { background-color: #8c2e2e; }"
        )
        self.clear_quit_btn.clicked.connect(self._on_clear_and_quit)
        side.addWidget(self.clear_quit_btn)

    def _refresh_list(self) -> None:
        self.plate_list.clear()
        for plate in self.store.list():
            self.plate_list.addItem(plate)

    def _on_add(self) -> None:
        text = self.add_input.text().strip()
        if not text:
            return
        if self.store.add(text):
            _log.info("[green]+ plate[/green] %s", text.upper())
            self.add_input.clear()
            self._refresh_list()
        else:
            _log.debug("plate already in list: %s", text.upper())
            QMessageBox.information(
                self, "Already in list",
                f"'{text.upper()}' is already allowed.",
            )

    def _on_remove(self) -> None:
        for item in self.plate_list.selectedItems():
            _log.info("[yellow]- plate[/yellow] %s", item.text())
            self.store.remove(item.text())
        self._refresh_list()

    def _on_save_frame(self) -> None:
        snap = self.worker.snapshot()
        if snap.raw_frame is None:
            _log.warning("[yellow]save-frame:[/yellow] no frame available yet")
            return
        out = Path(__file__).parent / "debug-frame.jpg"
        # 95% JPEG quality — enough fidelity for the model to behave identically
        # to what it saw in-app, small enough to be convenient to open.
        ok = cv2.imwrite(str(out), snap.raw_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if ok:
            _log.info("[green]frame saved[/green] %s  (%dx%d)",
                      out, snap.raw_frame.shape[1], snap.raw_frame.shape[0])
        else:
            _log.error("cv2.imwrite failed for %s", out)

    def _on_clear_and_quit(self) -> None:
        _log.info("cache clear requested")
        items = cache.scan()
        if not items:
            reply = QMessageBox.question(
                self, "Nothing to clear",
                "No app data was found. Quit anyway?",
                QMessageBox.Yes | QMessageBox.Cancel,
            )
            if reply == QMessageBox.Yes:
                self.close()
            return

        total = sum(i.size_bytes for i in items)
        details = "\n".join(
            f"  • {i.label}  ({cache.human_size(i.size_bytes)})" for i in items
        )
        reply = QMessageBox.warning(
            self, "Clear all data and quit?",
            f"This will permanently delete {cache.human_size(total)}:\n\n{details}\n\n"
            "Everything listed above will be gone. "
            "The virtual-environment and installed packages are NOT affected.\n\n"
            "Continue?",
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        if reply != QMessageBox.Yes:
            return

        # Stop the worker first so inference threads release ONNX file handles
        # before we delete the model files. The join inside stop() covers the
        # race; without it we'd occasionally hit "file in use" on re-downloads.
        _log.info("[red]clearing[/red] %d items, %s", len(items), cache.human_size(total))
        self._timer.stop()
        self.worker.stop()

        errors = cache.clear(items)
        if errors:
            _log.error("cache clear: %d error(s)", len(errors))
            preview = "\n".join(errors[:10])
            if len(errors) > 10:
                preview += f"\n… and {len(errors) - 10} more"
            QMessageBox.warning(
                self, "Some items could not be deleted", preview,
            )
        else:
            _log.info("[green]cache cleared[/green] (%s)", cache.human_size(total))

        self.close()

    def closeEvent(self, event: QCloseEvent) -> None:
        _log.info("closing window")
        self._timer.stop()
        self.worker.stop()
        event.accept()

    def _tick(self) -> None:
        snap = self.worker.snapshot()

        if snap.error and not snap.camera_ok:
            self._set_status("ERROR", "#b0302f", snap.error)
            self.meta_label.setText("")
            return

        raw = snap.raw_frame
        if raw is None:
            self._set_status("CONNECTING…", "#444", "")
            return

        frame = raw.copy()
        now = time.time()

        # Show every detection the detector returned, regardless of OCR quality.
        # The per-plate loop below tiers them into green/red/yellow so we can
        # tell "detector didn't fire" from "detector fired but OCR couldn't read".
        all_plates: list[Plate] = []
        if snap.processed and (now - snap.processed.timestamp) < DETECTION_FRESH_SEC:
            all_plates = list(snap.processed.plates)

        allowed = self.store.list()

        # Debug log (visible with LOG_LEVEL=DEBUG) on distinct reads.
        if all_plates:
            text_now = all_plates[0].text or "?"
            if text_now != self._last_detection_text:
                _log.debug(
                    "detected [cyan]%s[/cyan] (det=%.0f%% ocr=%.0f%%)",
                    text_now,
                    all_plates[0].det_confidence * 100,
                    all_plates[0].ocr_confidence * 100,
                )
                self._last_detection_text = text_now
        else:
            self._last_detection_text = None

        matched_plate: str | None = None
        matched_read: Plate | None = None
        readable_plates: list[Plate] = []  # plates whose OCR passed DISPLAY threshold
        for p in all_plates:
            has_ocr_text = bool(p.text) and p.ocr_confidence >= DISPLAY_MIN_OCR_CONFIDENCE
            match = best_match(p.text, allowed) if has_ocr_text else None

            if match:
                color = (0, 200, 0)         # BGR green: match
            elif has_ocr_text:
                color = (0, 0, 230)         # BGR red: readable but no match
            else:
                color = (0, 200, 255)       # BGR yellow: detector-only; OCR too weak

            x1, y1, x2, y2 = (int(v) for v in p.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            if match:
                caption = f"{p.text} -> {match}  ({p.ocr_confidence:.0%})"
            elif has_ocr_text:
                caption = f"{p.text}  ({p.ocr_confidence:.0%})"
            elif p.text:
                caption = f"{p.text}?  ocr={p.ocr_confidence:.0%}"
            else:
                caption = f"?  det={p.det_confidence:.0%}"

            cv2.putText(
                frame, caption, (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA,
            )

            if has_ocr_text:
                readable_plates.append(p)
            if (match and matched_plate is None
                    and p.ocr_confidence >= GRANT_MIN_OCR_CONFIDENCE):
                matched_plate = match
                matched_read = p

        if matched_plate:
            self._match_streak.append(matched_plate)
            if (len(self._match_streak) >= STABILITY_FRAMES
                    and all(p == matched_plate for p in self._match_streak)):
                self._granted_until = now + GRANT_DISPLAY_SEC
                self._granted_plate = matched_plate
        else:
            self._match_streak.clear()

        if now < self._granted_until and self._granted_plate:
            self._set_status("GRANTED", "#1f8a3a", f"Plate: {self._granted_plate}")
        elif readable_plates:
            read = matched_read or readable_plates[0]
            self._set_status(
                "DENIED", "#b0302f",
                f"Read: {read.text}  ({read.ocr_confidence:.0%})",
            )
        elif all_plates:
            # Detector fired but OCR couldn't read. The yellow boxes tell the
            # user where — this detail is just the summary.
            best_det = max(p.det_confidence for p in all_plates)
            self._set_status(
                "WAITING", "#444",
                f"Detected {len(all_plates)} plate region(s) "
                f"(best {best_det:.0%}) — OCR couldn't read them.",
            )
        else:
            self._set_status("WAITING", "#444", "Show a plate to the camera.")

        accel = "NE (ocr) + CPU (det)" if snap.processed else "loading…"
        inf_ms = f"{snap.processed.inference_ms:.0f} ms" if snap.processed else "—"
        raw_count = len(snap.processed.plates) if snap.processed else 0
        self.meta_label.setText(
            f"capture {snap.capture_fps:.1f} fps\n"
            f"inference {snap.inference_fps:.1f} fps ({inf_ms})\n"
            f"detections: {raw_count}\n"
            f"accel: {accel}"
        )

        self._draw_to_label(frame)

    def _set_status(self, text: str, bg: str, detail: str) -> None:
        # Log only on transition — _set_status runs on every frame, and we don't
        # want 30 log lines per second when nothing has changed.
        if text != self._prev_status:
            color = {
                "GRANTED": "bold green",
                "DENIED": "red",
                "ERROR": "bold red",
                "WAITING": "dim",
            }.get(text, "yellow")
            _log.info("status → [%s]%s[/%s]  %s", color, text, color, detail or "")
            self._prev_status = text
        self.status_label.setText(text)
        self.status_label.setStyleSheet(_STATUS_STYLE.format(bg=bg))
        self.detail_label.setText(detail)

    def _draw_to_label(self, frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if not rgb.flags["C_CONTIGUOUS"]:
            rgb = np.ascontiguousarray(rgb)
        h, w, ch = rgb.shape
        # .copy() detaches QImage from the numpy buffer; without it we'd risk a
        # use-after-free when the temporary `rgb` is garbage-collected.
        qimage = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimage).scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.video_label.setPixmap(pixmap)


def _hline() -> QFrame:
    sep = QFrame()
    sep.setFrameShape(QFrame.HLine)
    sep.setFrameShadow(QFrame.Sunken)
    return sep
