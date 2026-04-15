"""BRAID gallery — read/write ``/workspace/database/braid_sys_db/``.

Layout:
    braid_sys_db/
        p_<n>.npz       # arrays: face_emb (512,), voice_emb (256,), meta (pickle fallback)
        p_<n>.json      # human-readable metadata
        p_<n>.png       # optional representative face image
        next_id.txt     # monotonically increasing id counter

Thread-safe via an internal ``threading.Lock`` (the gRPC servicer may run
concurrent ticks per session).
"""
from __future__ import annotations

import glob
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

logger = logging.getLogger("braid")


@dataclass
class GalleryEntry:
    person_id: str
    face_emb: Optional[np.ndarray]
    voice_emb: Optional[np.ndarray]
    face_quality: float
    created_at: float
    updated_at: float
    face_count: int = 1
    voice_count: int = 1


class BraidGallery:
    def __init__(self, db_dir: str | Path):
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._entries: List[GalleryEntry] = []
        self._load()

    # ---- IO ---------------------------------------------------------------

    def _load(self):
        self._entries.clear()
        for meta_path in sorted(self.db_dir.glob("*.json")):
            try:
                with open(meta_path, "r") as fh:
                    meta = json.load(fh)
                pid = meta["person_id"]
                npz_path = self.db_dir / f"{pid}.npz"
                face_emb = voice_emb = None
                if npz_path.exists():
                    data = np.load(npz_path, allow_pickle=False)
                    if "face_emb" in data.files and data["face_emb"].size > 0:
                        face_emb = data["face_emb"].astype(np.float32)
                    if "voice_emb" in data.files and data["voice_emb"].size > 0:
                        voice_emb = data["voice_emb"].astype(np.float32)
                self._entries.append(GalleryEntry(
                    person_id=pid,
                    face_emb=face_emb,
                    voice_emb=voice_emb,
                    face_quality=float(meta.get("face_quality", 0.0)),
                    created_at=float(meta.get("created_at", time.time())),
                    updated_at=float(meta.get("updated_at", time.time())),
                    face_count=int(meta.get("face_count", 1)),
                    voice_count=int(meta.get("voice_count", 1)),
                ))
            except Exception as e:
                logger.warning("[gallery] failed to load %s: %s", meta_path, e)
        logger.info("[gallery] loaded %d entries from %s",
                    len(self._entries), self.db_dir)

    def _write_entry(self, e: GalleryEntry):
        npz_path = self.db_dir / f"{e.person_id}.npz"
        meta_path = self.db_dir / f"{e.person_id}.json"
        np.savez(
            npz_path,
            face_emb=e.face_emb if e.face_emb is not None else np.zeros(0, dtype=np.float32),
            voice_emb=e.voice_emb if e.voice_emb is not None else np.zeros(0, dtype=np.float32),
        )
        with open(meta_path, "w") as fh:
            json.dump({
                "person_id": e.person_id,
                "face_quality": e.face_quality,
                "created_at": e.created_at,
                "updated_at": e.updated_at,
                "face_count": e.face_count,
                "voice_count": e.voice_count,
                "has_face": e.face_emb is not None,
                "has_voice": e.voice_emb is not None,
            }, fh, indent=2)

    # ---- public ------------------------------------------------------------

    def entries(self) -> List[GalleryEntry]:
        with self._lock:
            return list(self._entries)

    def get(self, person_id: str) -> Optional[GalleryEntry]:
        with self._lock:
            for e in self._entries:
                if e.person_id == person_id:
                    return e
        return None

    def _next_id(self) -> str:
        """Monotonic p_<n> ids; survives across runs."""
        counter_path = self.db_dir / "next_id.txt"
        n = 1
        if counter_path.exists():
            try:
                n = int(counter_path.read_text().strip()) + 1
            except Exception:
                n = 1
        # Avoid colliding with any existing entry
        while any(e.person_id == f"p_{n}" for e in self._entries):
            n += 1
        counter_path.write_text(str(n))
        return f"p_{n}"

    def enrol(self, face_emb: Optional[np.ndarray],
              voice_emb: Optional[np.ndarray],
              face_quality: float,
              representative_image: Optional[np.ndarray] = None) -> GalleryEntry:
        with self._lock:
            pid = self._next_id()
            now = time.time()
            e = GalleryEntry(
                person_id=pid,
                face_emb=face_emb.astype(np.float32) if face_emb is not None else None,
                voice_emb=voice_emb.astype(np.float32) if voice_emb is not None else None,
                face_quality=float(face_quality),
                created_at=now, updated_at=now,
                face_count=1 if face_emb is not None else 0,
                voice_count=1 if voice_emb is not None else 0,
            )
            self._entries.append(e)
            self._write_entry(e)
            if representative_image is not None:
                try:
                    import cv2
                    cv2.imwrite(str(self.db_dir / f"{pid}.png"), representative_image)
                except Exception:
                    pass
            modalities = []
            if face_emb is not None: modalities.append("face")
            if voice_emb is not None: modalities.append("voice")
            logger.info("[gallery] ENROL person_id=%s modalities=%s quality=%.2f",
                        pid, "+".join(modalities) or "none", face_quality)
        return e

    def update_ema(self, person_id: str,
                   face_emb: Optional[np.ndarray],
                   voice_emb: Optional[np.ndarray],
                   alpha: float,
                   face_quality: Optional[float] = None) -> Optional[GalleryEntry]:
        with self._lock:
            entry = None
            for e in self._entries:
                if e.person_id == person_id:
                    entry = e
                    break
            if entry is None:
                return None
            if face_emb is not None:
                if entry.face_emb is None:
                    entry.face_emb = face_emb.astype(np.float32)
                else:
                    entry.face_emb = ((1 - alpha) * entry.face_emb
                                      + alpha * face_emb.astype(np.float32)).astype(np.float32)
                entry.face_count += 1
            if voice_emb is not None:
                if entry.voice_emb is None:
                    entry.voice_emb = voice_emb.astype(np.float32)
                else:
                    entry.voice_emb = ((1 - alpha) * entry.voice_emb
                                       + alpha * voice_emb.astype(np.float32)).astype(np.float32)
                entry.voice_count += 1
            if face_quality is not None:
                entry.face_quality = float(face_quality)
            entry.updated_at = time.time()
            self._write_entry(entry)
            modalities = []
            if face_emb is not None: modalities.append("face")
            if voice_emb is not None: modalities.append("voice")
            logger.info("[gallery] UPDATE person_id=%s modalities=%s alpha=%.2f",
                        person_id, "+".join(modalities) or "none", alpha)
            return entry
