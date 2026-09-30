"""JSON-backed allowed-plate list."""
from __future__ import annotations

import json
from pathlib import Path


def _normalize(plate: str) -> str:
    return "".join(c for c in plate.upper() if c.isalnum())


class PlateStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._plates: list[str] = []
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self._plates = []
            return
        try:
            data = json.loads(self.path.read_text())
            raw = data.get("plates", []) if isinstance(data, dict) else []
            self._plates = sorted({_normalize(p) for p in raw if _normalize(p)})
        except (json.JSONDecodeError, OSError):
            self._plates = []

    def save(self) -> None:
        self.path.write_text(json.dumps({"plates": self._plates}, indent=2) + "\n")

    def list(self) -> list[str]:
        return list(self._plates)

    def add(self, plate: str) -> bool:
        norm = _normalize(plate)
        if not norm or norm in self._plates:
            return False
        self._plates.append(norm)
        self._plates.sort()
        self.save()
        return True

    def remove(self, plate: str) -> bool:
        norm = _normalize(plate)
        if norm not in self._plates:
            return False
        self._plates.remove(norm)
        self.save()
        return True
