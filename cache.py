"""Cache inventory and cleanup.

Identifies everything this app has downloaded or generated on disk so the
user can wipe it all on demand. Excludes only the virtual-environment tree
(.venv / venv) and installed third-party packages.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import NamedTuple


PROJECT_ROOT = Path(__file__).parent

# Directory names that indicate a virtualenv tree we must never recurse into —
# bytecode inside them belongs to installed packages, not to this app.
_VENV_DIR_NAMES = {"venv", ".venv", "env", ".env", "site-packages"}


class CacheItem(NamedTuple):
    path: Path
    size_bytes: int
    label: str


def _is_under_venv(path: Path, root: Path) -> bool:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return False
    return any(part in _VENV_DIR_NAMES for part in parts)


def _dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            continue
    return total



def scan() -> list[CacheItem]:
    """Return every app-generated / downloaded location we would delete."""
    items: list[CacheItem] = []

    # fast-alpr stores ONNX weights through two underlying libraries:
    #   • open-image-models  → ~/.cache/open-image-models  (YOLO detector)
    #   • fast-plate-ocr     → ~/.cache/fast-plate-ocr     (Mobile-ViT OCR)
    dot_cache = Path.home() / ".cache"
    for lib_dir_name, label_prefix in (
        ("open-image-models", "Detector model cache"),
        ("fast-plate-ocr",    "OCR model cache"),
    ):
        lib_dir = dot_cache / lib_dir_name
        if lib_dir.exists():
            items.append(CacheItem(
                path=lib_dir,
                size_bytes=_dir_size(lib_dir),
                label=f"{label_prefix} ({lib_dir_name}/)",
            ))

    # Python bytecode caches inside this project — but NOT inside any
    # virtualenv, where they belong to third-party packages and will just
    # regenerate noisily on the next import.
    for pycache in PROJECT_ROOT.rglob("__pycache__"):
        if _is_under_venv(pycache, PROJECT_ROOT):
            continue
        items.append(CacheItem(
            path=pycache,
            size_bytes=_dir_size(pycache),
            label=f"Python bytecode: {pycache.relative_to(PROJECT_ROOT)}",
        ))

    # Runtime-generated files and directories (user data + diagnostic artifacts).
    _add_if_exists(items, PROJECT_ROOT / "plates.json",    "Allowed-plate list (plates.json)")
    _add_if_exists(items, PROJECT_ROOT / "debug-frame.jpg", "Debug frame snapshot (debug-frame.jpg)")
    _add_if_exists(items, PROJECT_ROOT / ".cache",         "App cache dir (.cache/)")
    _add_if_exists(items, PROJECT_ROOT / "smoketest-out",  "Smoke-test output (smoketest-out/)")

    # Any loose ONNX weight files saved directly in the project root.
    for onnx_file in PROJECT_ROOT.glob("*.onnx"):
        if not _is_under_venv(onnx_file, PROJECT_ROOT):
            items.append(CacheItem(
                path=onnx_file,
                size_bytes=onnx_file.stat().st_size,
                label=f"ONNX model: {onnx_file.name}",
            ))

    return items


def _add_if_exists(items: list[CacheItem], path: Path, label: str) -> None:
    if not path.exists():
        return
    size = _dir_size(path) if path.is_dir() else path.stat().st_size
    items.append(CacheItem(path=path, size_bytes=size, label=label))


def clear(items: list[CacheItem]) -> list[str]:
    """Delete the given items. Returns a list of error strings (empty on success)."""
    errors: list[str] = []
    for item in items:
        try:
            if item.path.is_dir():
                shutil.rmtree(item.path)
            elif item.path.exists():
                item.path.unlink()
        except OSError as exc:
            errors.append(f"{item.path}: {exc}")
    return errors


def human_size(n_bytes: int) -> str:
    size = float(n_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"
