"""Joint posterior P(ID, L | O) per Blueprint §3.2–§3.4.

Azimuth is discretised into ``cfg.num_azimuth_bins`` bins across [-π, π]. Each
person is scored against every gallery entry plus the "unknown" slot ∅.
Returns marginal identity posterior plus the key scalars used by decision.py:
p_best, p_second, p_unk, margin, entropy, Q_gate, modality_agreement.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .association import PersonObservation
from .config import BraidConfig
from .gallery import BraidGallery

logger = logging.getLogger("braid")

_EPS = 1e-9


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _cosine(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.0
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < _EPS or nb < _EPS:
        return 0.0
    return float(np.dot(a.reshape(-1), b.reshape(-1)) / (na * nb))


@dataclass
class IdentityPosterior:
    """Marginal identity posterior for one person hypothesis."""
    person_id: str
    # ordered ids: first M are gallery, last is ∅
    ids: List[str] = field(default_factory=list)
    probs: List[float] = field(default_factory=list)
    p_best: float = 0.0
    p_second: float = 0.0
    p_unk: float = 1.0
    j_best: Optional[str] = None           # gallery id (not ∅); None if best is unknown
    margin: float = 0.0
    entropy: float = 0.0
    Q_gate: float = 0.0
    modality_agreement: bool = False
    face_rank: int = -1                     # rank of j_best under face-only score
    voice_rank: int = -1                    # rank of j_best under voice-only score
    has_face: bool = False
    has_voice: bool = False


class PosteriorComputer:
    def __init__(self, cfg: BraidConfig):
        self.cfg = cfg
        self._bin_centers = np.linspace(
            -math.pi, math.pi, cfg.num_azimuth_bins, endpoint=False
        ) + (math.pi / cfg.num_azimuth_bins)

    # ------ likelihood terms -------------------------------------------------

    def _p_face(self, face_emb: Optional[np.ndarray],
                gallery_face: Optional[np.ndarray], visible: bool,
                M: int) -> float:
        """§3.3.1"""
        cfg = self.cfg
        if not visible:
            return 1.0 / (M + 1)
        if face_emb is None or gallery_face is None:
            # Visible but unknown slot
            return _sigmoid((cfg.lambda_face - cfg.theta_face) / max(_EPS, cfg.beta_face))
        cos = _cosine(face_emb, gallery_face)
        return _sigmoid((cos - cfg.theta_face) / max(_EPS, cfg.beta_face))

    def _p_face_unk(self) -> float:
        cfg = self.cfg
        return _sigmoid((cfg.lambda_face - cfg.theta_face) / max(_EPS, cfg.beta_face))

    def _p_voice(self, voice_emb: Optional[np.ndarray],
                 gallery_voice: Optional[np.ndarray], speaking: bool,
                 M: int) -> float:
        """§3.3.2 — marginalised outside."""
        cfg = self.cfg
        if not speaking:
            return 1.0 / (M + 1)
        if voice_emb is None or gallery_voice is None:
            return _sigmoid((cfg.lambda_voice - cfg.theta_voice) / max(_EPS, cfg.beta_voice))
        cos = _cosine(voice_emb, gallery_voice)
        return _sigmoid((cos - cfg.theta_voice) / max(_EPS, cfg.beta_voice))

    def _p_voice_unk(self) -> float:
        cfg = self.cfg
        return _sigmoid((cfg.lambda_voice - cfg.theta_voice) / max(_EPS, cfg.beta_voice))

    @staticmethod
    def _gaussian(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
        # wrapped-ish; we keep azimuths in [-π, π] and just use a plain Gaussian
        diff = x - mu
        diff = ((diff + math.pi) % (2 * math.pi)) - math.pi
        return np.exp(-0.5 * (diff / max(_EPS, sigma)) ** 2)

    def _p_loc(self, bbox_az: Optional[float], visible: bool) -> np.ndarray:
        if not visible or bbox_az is None:
            return np.ones_like(self._bin_centers) / len(self._bin_centers)
        p = self._gaussian(self._bin_centers, bbox_az, self.cfg.sigma_det)
        s = p.sum()
        return p / s if s > 0 else np.ones_like(p) / len(p)

    def _p_ssl(self, ssl_az: Optional[float], ssl_conf: float,
               speaking: bool) -> np.ndarray:
        if not speaking or ssl_az is None:
            return np.ones_like(self._bin_centers) / len(self._bin_centers)
        sigma = self.cfg.sigma_0 / math.sqrt(max(0.01, ssl_conf) + 0.01)
        p = self._gaussian(self._bin_centers, ssl_az, sigma)
        s = p.sum()
        return p / s if s > 0 else np.ones_like(p) / len(p)

    def _loc_prior(self, prior_az: Optional[float]) -> np.ndarray:
        if prior_az is None:
            return np.ones_like(self._bin_centers) / len(self._bin_centers)
        p = self._gaussian(self._bin_centers, prior_az, self.cfg.sigma_drift)
        s = p.sum()
        return p / s if s > 0 else np.ones_like(p) / len(p)

    # ------ main ------------------------------------------------------------

    def compute(
        self,
        person: PersonObservation,
        gallery: BraidGallery,
        identity_prior: Optional[Dict[str, float]] = None,
        location_prior_az: Optional[float] = None,
    ) -> IdentityPosterior:
        """Return identity posterior for one person."""
        cfg = self.cfg
        entries = gallery.entries()
        gallery_ids = [e.person_id for e in entries]
        M = len(gallery_ids)

        # ---- identity prior -------------------------------------------------
        if identity_prior:
            prior = [max(_EPS, float(identity_prior.get(gid, 0.0))) for gid in gallery_ids]
            prior.append(max(_EPS, float(identity_prior.get("__unk__", cfg.p_new))))
            s = sum(prior)
            prior = [p / s for p in prior]
        else:
            if M > 0:
                each = (1.0 - cfg.p_new) / M
                prior = [each] * M + [cfg.p_new]
            else:
                prior = [1.0]  # only unknown

        # ---- ASD / diar → P(S_i=1) -----------------------------------------
        p_asd_speak = person.mean_asd if person.visible else 0.5
        p_diar_speak = person.diar_delta
        # combine ASD + diar symmetrically; clamp
        p_s1 = min(1.0, max(0.0, 0.5 * (p_asd_speak + p_diar_speak)))
        p_s0 = 1.0 - p_s1

        ploc_vis = self._p_loc(person.face_azimuth_rad if person.visible else None,
                               person.visible)
        ploc_prior = self._loc_prior(location_prior_az)

        # Face / voice likelihood tables indexed by (j including ∅)
        face_lik = np.zeros(M + 1, dtype=np.float64)
        voice_lik_s1 = np.zeros(M + 1, dtype=np.float64)
        face_cos_tbl = np.full(M + 1, -1.0)    # for rank_face
        voice_cos_tbl = np.full(M + 1, -1.0)   # for rank_voice

        for j, entry in enumerate(entries):
            face_lik[j] = self._p_face(person.face_emb, entry.face_emb, person.visible, M)
            voice_lik_s1[j] = self._p_voice(person.voice_emb, entry.voice_emb, True, M)
            face_cos_tbl[j] = _cosine(person.face_emb, entry.face_emb)
            voice_cos_tbl[j] = _cosine(person.voice_emb, entry.voice_emb)
        # ∅ slot
        face_lik[M] = self._p_face_unk() if person.visible else 1.0 / (M + 1)
        voice_lik_s1[M] = self._p_voice_unk()

        # Quality-modulated face likelihood §3.3.1
        q = max(0.0, min(1.0, person.face_quality)) if person.visible else 0.0
        face_lik_eff = np.power(np.clip(face_lik, _EPS, 1.0), q if q > 0 else 0.0)
        if q <= 0:
            face_lik_eff = np.ones_like(face_lik) / (M + 1)

        # Accumulate joint over azimuth bins, then marginalise.
        # joint[j] = sum_theta P_face_eff(j) * P_loc_vis(theta) * P(j) * P_loc_prior(theta) * Gamma(j,theta)
        # Gamma = sum_S P_voice(j|S) * P(S|asd,V) * P_SSL(theta|ssl,S) * P(S|delta)
        bins = self._bin_centers
        ssl_s1 = self._p_ssl(person.ssl_azimuth_rad, person.ssl_confidence, True)
        ssl_s0 = np.ones_like(bins) / len(bins)

        p_s_asd_1 = p_asd_speak
        p_s_asd_0 = 1.0 - p_s_asd_1
        p_s_diar_1 = p_diar_speak
        p_s_diar_0 = 1.0 - p_s_diar_1

        # Per-bin location factor = P_loc_vis * P_loc_prior
        loc_factor = ploc_vis * ploc_prior
        loc_factor = loc_factor / (loc_factor.sum() + _EPS)

        joint = np.zeros(M + 1, dtype=np.float64)
        for j in range(M + 1):
            gamma_s1 = voice_lik_s1[j] * p_s_asd_1 * p_s_diar_1  # weight
            gamma_s0 = (1.0 / (M + 1)) * p_s_asd_0 * p_s_diar_0
            # integrate over azimuth:
            integrand = (gamma_s1 * ssl_s1 + gamma_s0 * ssl_s0) * loc_factor
            gamma_marginal = float(integrand.sum())
            joint[j] = face_lik_eff[j] * prior[j] * max(_EPS, gamma_marginal)

        Z = float(joint.sum())
        if Z <= 0:
            probs = np.ones(M + 1) / (M + 1)
        else:
            probs = joint / Z

        ids = gallery_ids + ["__unk__"]
        # --- marginal metrics ---
        order = np.argsort(-probs)
        p_best = float(probs[order[0]])
        p_second = float(probs[order[1]]) if len(order) > 1 else 0.0
        # "best gallery" excludes ∅
        gallery_scores = probs[:M] if M > 0 else np.array([])
        j_best = None
        if M > 0:
            g_order = np.argsort(-gallery_scores)
            j_best = gallery_ids[int(g_order[0])]
        p_unk = float(probs[M])
        margin = p_best - p_second
        H = -float(np.sum(probs * np.log(np.clip(probs, _EPS, 1.0))))
        Q_gate = margin * (M + 1)

        # face_rank / voice_rank for j_best (for modality agreement)
        face_rank = -1
        voice_rank = -1
        if j_best is not None and M > 0:
            face_rank = int(np.argsort(-face_cos_tbl[:M]).tolist().index(gallery_ids.index(j_best)))
            voice_rank = int(np.argsort(-voice_cos_tbl[:M]).tolist().index(gallery_ids.index(j_best)))
        # agreement ≡ both modalities rank j_best at position 0
        modality_agreement = (face_rank == 0 and voice_rank == 0) \
            if (person.face_emb is not None and person.voice_emb is not None) else False

        logger.info(
            "[posterior] person=%s M=%d p_s1=%.2f p_best=%.3f p_second=%.3f "
            "p_unk=%.3f j_best=%s face_rank=%d voice_rank=%d H=%.2f",
            person.person_id, M, p_s1, p_best, p_second, p_unk,
            j_best or "-", face_rank, voice_rank, H,
        )

        return IdentityPosterior(
            person_id=person.person_id,
            ids=ids,
            probs=probs.tolist(),
            p_best=p_best, p_second=p_second, p_unk=p_unk,
            j_best=j_best, margin=margin, entropy=H, Q_gate=Q_gate,
            modality_agreement=modality_agreement,
            face_rank=face_rank, voice_rank=voice_rank,
            has_face=person.face_emb is not None and person.visible,
            has_voice=person.voice_emb is not None and person.diar_delta > 0.3,
        )
