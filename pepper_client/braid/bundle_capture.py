"""Capture one 30s BRAID bundle from Pepper.

Coordinates:
  * ``ALAudioDevice`` (4-ch 16 kHz PCM) via a qi-registered service with a
    ``processRemote`` callback. The service **must** be registered on the
    qi session (see ``pepper_client/speaker_client.py::PepperAudioCapture``
    and ``pepper_client/pepper_middleware/pepper.py`` for the canonical
    pattern).
  * ``ALVideoDevice`` — RGB frames pulled via ``getImageRemote``.
  * ``ALSoundLocalization`` — azimuth / confidence stream via
    ``ALMemory['ALSoundLocalization/SoundLocated']``.

Usage:

    capture = BundleCapture(session, listen_ip="192.168.0.50", listen_port=52100)
    bundle  = capture.run(tick_id=..., session_id=..., duration_seconds=30.0,
                          robot_heading_rad=...)

``session.listen`` + ``session.registerService`` are called exactly once at
construction; the audio collector is re-used across ticks.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger("braid.client")


@dataclass
class ClientBundle:
    tick_id: int
    session_id: str
    window_start_ts: float
    robot_heading_rad: float
    audio_pcm: bytes
    audio_sample_rate: int
    audio_channels: int
    # frames: list of (ts, jpeg_bytes, width, height)
    frames: List[Tuple[float, bytes, int, int]] = field(default_factory=list)
    # ssl events: list of (ts, azimuth_rad, elevation_rad, confidence)
    ssl_events: List[Tuple[float, float, float, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# qi-registered audio collector. Mirrors the PepperAudioCapture pattern from
# speaker_client.py. This MUST be a top-level class (not a nested one) so
# qi's service introspection can bind its methods.
# ---------------------------------------------------------------------------
class BraidAudioCapture(object):
    """qi service: receives audio via ``processRemote``."""

    def __init__(self, sample_rate=16000, channels=4):
        self.module_name = "BraidAudioCapture"
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.audio_service = None
        self._buf = bytearray()
        self._lock = threading.Lock()
        self._active = False

    def init_service(self, session):
        self.audio_service = session.service("ALAudioDevice")

    def start(self):
        with self._lock:
            self._buf = bytearray()
        self._active = True
        # Pepper's ALAudioDevice: setClientPreferences(name, sr, channel_cfg, deinterleaved)
        # channel_cfg: 0=all 4 channels interleaved, 1..4=single mic, 5=front
        self.audio_service.setClientPreferences(
            self.module_name, self.sample_rate, 0, 0  # 0=all-4-channels interleaved
        )
        self.audio_service.subscribe(self.module_name)

    def stop(self):
        self._active = False
        try:
            self.audio_service.unsubscribe(self.module_name)
        except Exception as e:
            logger.warning("[capture] audio unsubscribe failed: %s", e)

    def drain(self) -> bytes:
        with self._lock:
            data = bytes(self._buf)
            self._buf = bytearray()
        return data

    # Called by NAOqi:
    def processRemote(self, nbOfChannels, nbOfSamplesByChannel, timeStamp, inputBuffer):
        if not self._active:
            return
        with self._lock:
            self._buf.extend(bytes(inputBuffer))


class BundleCapture:
    def __init__(self, session,
                 listen_ip: str = "0.0.0.0",
                 listen_port: int = 52100,
                 sample_rate: int = 16000,
                 channels: int = 4,
                 camera_resolution: int = 2,
                 camera_colorspace: int = 11,  # RGB
                 camera_fps: int = 10):
        self.session = session
        self.sample_rate = sample_rate
        self.channels = channels
        self.camera_resolution = camera_resolution
        self.camera_colorspace = camera_colorspace
        self.camera_fps = camera_fps

        # ---- services ----------------------------------------------------
        self._video = session.service("ALVideoDevice")
        self._memory = session.service("ALMemory")
        self._ssl = session.service("ALSoundLocalization")

        self._video_client: Optional[str] = None
        self._ssl_subscribed = False

        # ---- audio collector: register ONCE per session ------------------
        self._audio_collector = BraidAudioCapture(
            sample_rate=sample_rate, channels=channels,
        )
        listen_url = f"tcp://{listen_ip}:{listen_port}"
        try:
            self.session.listen(listen_url)
        except Exception as e:
            logger.warning("[capture] session.listen(%s) failed (possibly already listening): %s",
                           listen_url, e)
        try:
            self.session.registerService(
                self._audio_collector.module_name, self._audio_collector,
            )
            logger.info("[capture] registered qi service '%s' (listen=%s)",
                        self._audio_collector.module_name, listen_url)
        except Exception as e:
            # Already registered from a previous run of the same process is fine.
            logger.warning("[capture] registerService failed (continuing): %s", e)
        self._audio_collector.init_service(self.session)

    # ---------- subscription helpers ----------
    def _subscribe_camera(self):
        try:
            self._video_client = self._video.subscribeCamera(
                "braid_cam", 0, self.camera_resolution,
                self.camera_colorspace, self.camera_fps,
            )
        except Exception as e:
            logger.warning("[capture] subscribeCamera failed (%s); using subscribe()", e)
            self._video_client = self._video.subscribe(
                "braid_cam", self.camera_resolution, self.camera_colorspace,
                self.camera_fps,
            )

    def _unsubscribe_camera(self):
        try:
            if self._video_client:
                self._video.unsubscribe(self._video_client)
        except Exception as e:
            logger.warning("[capture] camera unsubscribe failed: %s", e)
        self._video_client = None

    def _subscribe_ssl(self):
        try:
            self._ssl.setParameter("Sensibility", 0.8)
        except Exception:
            pass
        try:
            self._ssl.subscribe("braid_ssl")
            self._ssl_subscribed = True
        except Exception as e:
            logger.warning("[capture] SSL subscribe failed: %s", e)

    def _unsubscribe_ssl(self):
        if self._ssl_subscribed:
            try:
                self._ssl.unsubscribe("braid_ssl")
            except Exception:
                pass
            self._ssl_subscribed = False

    # ---------- drainers ----------
    def _camera_loop(self, out_frames, stop_flag, t0):
        import numpy as np
        try:
            import cv2
        except Exception:
            cv2 = None  # type: ignore
        next_capture = t0
        period = 1.0 / max(1, self.camera_fps)
        while not stop_flag["stop"]:
            now = time.time()
            if now < next_capture:
                time.sleep(min(0.01, next_capture - now))
                continue
            next_capture = now + period
            try:
                img = self._video.getImageRemote(self._video_client)
                if img is None:
                    continue
                w, h = int(img[0]), int(img[1])
                raw = img[6]
                try:
                    self._video.releaseImage(self._video_client)
                except Exception:
                    pass
                if cv2 is None:
                    continue
                arr = np.frombuffer(bytes(raw), dtype=np.uint8).reshape(h, w, 3)
                bgr = arr[:, :, ::-1].copy()
                ok, jpg = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok:
                    out_frames.append((time.time(), jpg.tobytes(), w, h))
            except Exception as e:
                logger.warning("[capture] frame error: %s", e)

    def _ssl_loop(self, out_events, stop_flag):
        while not stop_flag["stop"]:
            try:
                data = self._memory.getData("ALSoundLocalization/SoundLocated")
                if data and len(data) >= 2:
                    az = float(data[1][0])
                    el = float(data[1][1])
                    conf = float(data[1][2])
                    out_events.append((time.time(), az, el, conf))
            except Exception:
                pass
            time.sleep(0.1)

    # ---------- public ----------
    def run(self, tick_id: int, session_id: str,
            duration_seconds: float, robot_heading_rad: float) -> ClientBundle:
        logger.info("[capture] BEGIN tick=%d duration=%.1fs sr=%d ch=%d fps=%d",
                    tick_id, duration_seconds, self.sample_rate,
                    self.channels, self.camera_fps)
        self._subscribe_camera()
        self._subscribe_ssl()
        self._audio_collector.start()
        logger.info("[capture] subscribed camera+ssl+audio; capturing…")
        try:
            t0 = time.time()
            stop_flag = {"stop": False}
            frames: List[Tuple[float, bytes, int, int]] = []
            ssl_events: List[Tuple[float, float, float, float]] = []

            thr_cam = threading.Thread(target=self._camera_loop,
                                       args=(frames, stop_flag, t0), daemon=True)
            thr_ssl = threading.Thread(target=self._ssl_loop,
                                       args=(ssl_events, stop_flag), daemon=True)
            thr_cam.start(); thr_ssl.start()

            t_end = t0 + duration_seconds
            while time.time() < t_end:
                time.sleep(0.1)

            stop_flag["stop"] = True
            thr_cam.join(timeout=2.0)
            thr_ssl.join(timeout=2.0)

            # Stop audio and drain. stop() unsubscribes; drain() returns bytes.
            self._audio_collector.stop()
            audio_pcm = self._audio_collector.drain()

            bundle = ClientBundle(
                tick_id=tick_id,
                session_id=session_id,
                window_start_ts=t0,
                robot_heading_rad=float(robot_heading_rad),
                audio_pcm=audio_pcm,
                audio_sample_rate=self.sample_rate,
                audio_channels=self.channels,
                frames=frames,
                ssl_events=ssl_events,
            )
            logger.info(
                "[capture] tick=%d audio=%dB frames=%d ssl=%d wall=%.2fs",
                tick_id, len(bundle.audio_pcm), len(frames), len(ssl_events),
                time.time() - t0,
            )
            return bundle
        finally:
            self._unsubscribe_camera()
            self._unsubscribe_ssl()
