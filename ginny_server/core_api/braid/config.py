"""BRAID configuration loader.

Reads braid_config.yaml next to this module. All thresholds in Blueprint §3.6
live there; this module only provides typed access + sensible fallbacks.
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover — fallback if pyyaml missing
    yaml = None

logger = logging.getLogger("braid")

_DEFAULTS: Dict[str, Any] = {
    # decision
    "tau_recog": 0.70, "Q_recog": 1.5,
    "tau_confirm": 0.40, "Q_confirm": 0.8,
    "tau_enrol_unk": 0.60, "Q_enrol": 0.5,
    "Q_min": 0.3, "Q_min_enrol": 0.3,
    "H_explore": 1.5, "tau_bridge": 0.3,
    # priors / noise
    "p_new": 0.3,
    "sigma_0_deg": 15.0, "sigma_det_deg": 5.0, "sigma_drift_deg": 10.0,
    # embeddings
    "theta_face": 0.30, "beta_face": 0.05, "lambda_face": 0.20,
    "theta_voice": 0.25, "beta_voice": 0.06, "lambda_voice": 0.15,
    # re-assoc
    "reassoc_face_cos": 0.5, "reassoc_voice_cos": 0.5,
    "gallery_ema_alpha": 0.1,
    # quality
    "face_area_min_px": 6400, "lap_min": 40.0, "yaw_sigma_deg": 30.0,
    # infra
    "num_azimuth_bins": 36, "camera_hfov_deg": 57.0, "max_persons": 4,
    # action
    "rotate_min_deg": 10.0, "rotate_nudge_deg": 15.0, "move_forward_m": 0.3,
    # tick
    "tick_window_seconds": 30.0,
    # paths
    "gallery_dir": "/workspace/database/braid_sys_db",
    # asd fallback
    "asd_stub_default_alpha": 0.5,
}


@dataclass
class BraidConfig:
    data: Dict[str, Any] = field(default_factory=lambda: dict(_DEFAULTS))

    # ---- decision rules ----
    @property
    def tau_recog(self) -> float:     return float(self.data["tau_recog"])
    @property
    def Q_recog(self) -> float:       return float(self.data["Q_recog"])
    @property
    def tau_confirm(self) -> float:   return float(self.data["tau_confirm"])
    @property
    def Q_confirm(self) -> float:     return float(self.data["Q_confirm"])
    @property
    def tau_enrol_unk(self) -> float: return float(self.data["tau_enrol_unk"])
    @property
    def Q_enrol(self) -> float:       return float(self.data["Q_enrol"])
    @property
    def Q_min(self) -> float:         return float(self.data["Q_min"])
    @property
    def Q_min_enrol(self) -> float:   return float(self.data["Q_min_enrol"])
    @property
    def H_explore(self) -> float:     return float(self.data["H_explore"])
    @property
    def tau_bridge(self) -> float:    return float(self.data["tau_bridge"])

    # ---- priors ----
    @property
    def p_new(self) -> float:         return float(self.data["p_new"])

    # ---- radians conversions ----
    @property
    def sigma_0(self) -> float:       return math.radians(float(self.data["sigma_0_deg"]))
    @property
    def sigma_det(self) -> float:     return math.radians(float(self.data["sigma_det_deg"]))
    @property
    def sigma_drift(self) -> float:   return math.radians(float(self.data["sigma_drift_deg"]))
    @property
    def camera_hfov(self) -> float:   return math.radians(float(self.data["camera_hfov_deg"]))
    @property
    def rotate_min_rad(self) -> float:   return math.radians(float(self.data["rotate_min_deg"]))
    @property
    def rotate_nudge_rad(self) -> float: return math.radians(float(self.data["rotate_nudge_deg"]))
    @property
    def yaw_sigma_deg(self) -> float: return float(self.data["yaw_sigma_deg"])

    # ---- embedding operating points ----
    @property
    def theta_face(self) -> float:    return float(self.data["theta_face"])
    @property
    def beta_face(self) -> float:     return float(self.data["beta_face"])
    @property
    def lambda_face(self) -> float:   return float(self.data["lambda_face"])
    @property
    def theta_voice(self) -> float:   return float(self.data["theta_voice"])
    @property
    def beta_voice(self) -> float:    return float(self.data["beta_voice"])
    @property
    def lambda_voice(self) -> float:  return float(self.data["lambda_voice"])

    # ---- re-assoc ----
    @property
    def reassoc_face_cos(self) -> float: return float(self.data["reassoc_face_cos"])
    @property
    def reassoc_voice_cos(self) -> float: return float(self.data["reassoc_voice_cos"])
    @property
    def gallery_ema_alpha(self) -> float: return float(self.data["gallery_ema_alpha"])

    # ---- quality ----
    @property
    def face_area_min_px(self) -> float: return float(self.data["face_area_min_px"])
    @property
    def lap_min(self) -> float:          return float(self.data["lap_min"])

    # ---- infra ----
    @property
    def num_azimuth_bins(self) -> int:  return int(self.data["num_azimuth_bins"])
    @property
    def max_persons(self) -> int:       return int(self.data["max_persons"])
    @property
    def move_forward_m(self) -> float:  return float(self.data["move_forward_m"])
    @property
    def tick_window_seconds(self) -> float: return float(self.data["tick_window_seconds"])
    @property
    def gallery_dir(self) -> str:       return str(self.data["gallery_dir"])
    @property
    def asd_stub_default_alpha(self) -> float: return float(self.data["asd_stub_default_alpha"])


def load_config(path: str | None = None) -> BraidConfig:
    """Load BRAID config from YAML. Falls back to hardcoded defaults if
    pyyaml is unavailable or the file is missing. Unknown keys are kept so
    callers can introduce new thresholds without touching this module."""
    if path is None:
        path = str(Path(__file__).with_name("braid_config.yaml"))

    merged: Dict[str, Any] = dict(_DEFAULTS)
    if yaml is None:
        logger.warning("PyYAML not installed; using built-in BRAID defaults.")
        return BraidConfig(data=merged)
    if not os.path.exists(path):
        logger.warning("braid_config.yaml not found at %s; using defaults.", path)
        return BraidConfig(data=merged)
    with open(path, "r") as fh:
        loaded = yaml.safe_load(fh) or {}
    if not isinstance(loaded, dict):
        logger.warning("braid_config.yaml malformed; using defaults.")
        return BraidConfig(data=merged)
    merged.update(loaded)
    return BraidConfig(data=merged)
