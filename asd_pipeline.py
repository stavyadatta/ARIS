#!/usr/bin/env python3
"""
Active Speaker Detection Pipeline

Combines:
  - S3FD face detection (from Light-ASD)
  - Light-ASD active speaker detection model
  - DiariZen diarization + WavLM speaker recognition (from ginny_server)

Takes a video, detects faces, determines who is actively speaking,
identifies speakers via voice embeddings, and outputs an annotated video
with bounding boxes on active speakers.

Usage:
    python asd_pipeline.py <video_path> [--duration 60] [--device cuda]

Example:
    python asd_pipeline.py marking_work/Meeting01.mp4 --duration 120
"""

import sys
import os

# ── Path setup (must happen before model imports) ──────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LIGHT_ASD_DIR = os.path.join(SCRIPT_DIR, "Light-ASD")
LASER_ASD_DIR = os.path.join(SCRIPT_DIR, "LASER_ASD", "LoCoNet")
sys.path.insert(0, LIGHT_ASD_DIR)
sys.path.insert(0, SCRIPT_DIR)

import argparse
import cv2
import torch
import numpy as np
import subprocess
import tempfile
import shutil
import math
import glob
import wave
import time
import python_speech_features
from scipy import signal
from scipy.io import wavfile as scipy_wavfile
from scipy.interpolate import interp1d

# ── Light-ASD imports (requires sys.path to include Light-ASD dir) ─────
# S3FD weight loading uses os.getcwd(), so we chdir temporarily
_orig_cwd = os.getcwd()
os.chdir(LIGHT_ASD_DIR)
from model.faceDetector.s3fd import S3FD as _S3FD_cls
os.chdir(_orig_cwd)

from ASD import ASD as _ASD_cls
from loss import lossAV as _lossAV_cls
from loss import lossV as _lossV_cls

# ── Constants ──────────────────────────────────────────────────────────
VIDEO_FPS = 25          # Light-ASD expects 25fps
AUDIO_SR = 16000        # 16kHz mono
FACE_DET_SCALE = 0.25   # Scale frames for face detection
FACE_DET_CONF = 0.9     # Face detection confidence threshold
MIN_TRACK_LEN = 10      # Minimum frames for a valid face track
IOU_TRACK_THRESH = 0.5  # IOU threshold for face tracking
NUM_FAILED_DET = 10     # Max missed detections before dropping track
CROP_SCALE = 0.40       # Padding around face crops
ASD_DURATION_SET = [1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6]  # Multi-scale scoring
ASD_SPEAKING_THRESH = 0.0  # Score > 0 means speaking

# ── LASER/LoCoNet weight downloads (Google Drive file IDs) ────────────
LASER_WEIGHTS = {
    "loconet_laser": {
        "gdrive_id": "1IrntlKqzw5EYAVbyDupr5tk-H3q9kkoW",
        "filename": "loconet_laser_ava.model",
    },
    "loconet": {
        "gdrive_id": "1NVIkksrD3zbxbDuDbPc_846bLfPSZcZm",
        "filename": "loconet_ava.model",
    },
}
LASER_WEIGHT_DIR = os.path.join(LASER_ASD_DIR, "weight")


def _ensure_laser_weights(model_key="loconet_laser"):
    """Download LASER/LoCoNet weights from Google Drive if not present."""
    info = LASER_WEIGHTS[model_key]
    weight_path = os.path.join(LASER_WEIGHT_DIR, info["filename"])
    if os.path.isfile(weight_path):
        return weight_path
    os.makedirs(LASER_WEIGHT_DIR, exist_ok=True)
    print(f"    Downloading {info['filename']} from Google Drive...")
    cmd = f"gdown --id {info['gdrive_id']} -O {weight_path}"
    subprocess.call(cmd, shell=True)
    if not os.path.isfile(weight_path):
        raise RuntimeError(
            f"Failed to download LASER weights. "
            f"Please manually download from Google Drive ID {info['gdrive_id']} "
            f"and place at {weight_path}"
        )
    print(f"    Downloaded: {weight_path}")
    return weight_path

SPEAKER_COLORS = [
    (0, 220, 0),     # Green
    (0, 180, 255),   # Orange
    (255, 100, 100), # Light blue
    (0, 255, 255),   # Yellow
    (255, 0, 255),   # Magenta
    (255, 255, 0),   # Cyan
    (100, 200, 100), # Muted green
    (200, 150, 255), # Light purple
]
INACTIVE_COLOR = (100, 100, 100)


# ═══════════════════════════════════════════════════════════════════════
# Step 1: Media extraction
# ═══════════════════════════════════════════════════════════════════════

def extract_media(video_path, work_dir, duration=0):
    """Extract video at 25fps and audio at 16kHz from input."""
    video_out = os.path.join(work_dir, "video.avi")
    audio_out = os.path.join(work_dir, "audio.wav")

    dur_args = ["-t", str(duration)] if duration > 0 else []
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path] + dur_args +
        ["-qscale:v", "2", "-async", "1", "-r", str(VIDEO_FPS),
         video_out, "-loglevel", "warning"],
        check=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_out,
         "-ac", "1", "-ar", str(AUDIO_SR), "-vn",
         "-acodec", "pcm_s16le", audio_out, "-loglevel", "warning"],
        check=True,
    )
    return video_out, audio_out


def extract_frames(video_path, frames_dir):
    """Extract individual frames as JPEGs."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path,
         "-qscale:v", "2", "-f", "image2",
         os.path.join(frames_dir, "%06d.jpg"), "-loglevel", "warning"],
        check=True,
    )
    flist = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    return flist


# ═══════════════════════════════════════════════════════════════════════
# Step 2: Face detection (S3FD)
# ═══════════════════════════════════════════════════════════════════════

def detect_faces(frames_dir, device="cuda"):
    """Run S3FD face detection on every frame."""
    _cwd = os.getcwd()
    os.chdir(LIGHT_ASD_DIR)
    detector = _S3FD_cls(device=device)
    os.chdir(_cwd)

    flist = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    all_dets = []
    t0 = time.time()
    for fidx, fname in enumerate(flist):
        image = cv2.imread(fname)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = detector.detect_faces(image_rgb, conf_th=FACE_DET_CONF,
                                       scales=[FACE_DET_SCALE])
        frame_dets = []
        for bbox in bboxes:
            frame_dets.append({
                "frame": fidx,
                "bbox": bbox[:4].tolist(),
                "conf": float(bbox[4]),
            })
        all_dets.append(frame_dets)
        if (fidx + 1) % 200 == 0 or fidx == len(flist) - 1:
            elapsed = time.time() - t0
            print(f"    Face detection: {fidx+1}/{len(flist)} frames "
                  f"({elapsed:.1f}s)")

    return all_dets


# ═══════════════════════════════════════════════════════════════════════
# Step 3: Face tracking (IOU-based)
# ═══════════════════════════════════════════════════════════════════════

def _bb_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter + 1e-6)


def track_faces(all_dets):
    """IOU-based face tracking across frames. Returns interpolated tracks."""
    tracks = []
    while True:
        track = []
        for frame_faces in all_dets:
            for face in frame_faces:
                if not track:
                    track.append(face)
                    frame_faces.remove(face)
                elif face["frame"] - track[-1]["frame"] <= NUM_FAILED_DET:
                    if _bb_iou(face["bbox"], track[-1]["bbox"]) > IOU_TRACK_THRESH:
                        track.append(face)
                        frame_faces.remove(face)
                        continue
                else:
                    break
        if not track:
            break
        if len(track) <= MIN_TRACK_LEN:
            continue

        # Interpolate bounding boxes for missing frames
        frame_nums = np.array([f["frame"] for f in track])
        bboxes = np.array([f["bbox"] for f in track])
        frame_range = np.arange(frame_nums[0], frame_nums[-1] + 1)
        bboxes_interp = []
        for dim in range(4):
            fn = interp1d(frame_nums, bboxes[:, dim])
            bboxes_interp.append(fn(frame_range))
        bboxes_interp = np.stack(bboxes_interp, axis=1)

        # Filter out tiny faces
        mean_w = np.mean(bboxes_interp[:, 2] - bboxes_interp[:, 0])
        mean_h = np.mean(bboxes_interp[:, 3] - bboxes_interp[:, 1])
        if max(mean_w, mean_h) > 1:
            tracks.append({"frame": frame_range, "bbox": bboxes_interp})

    print(f"    Found {len(tracks)} face tracks")
    return tracks


# ═══════════════════════════════════════════════════════════════════════
# Step 4: Crop face clips + extract MFCC features for ASD
# ═══════════════════════════════════════════════════════════════════════

def crop_tracks(tracks, frames_dir, crop_dir, audio_path):
    """Crop 224x224 face clips and corresponding audio for each track."""
    flist = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    processed = []

    for tidx, track in enumerate(tracks):
        crop_base = os.path.join(crop_dir, f"{tidx:05d}")

        # Compute smoothed face centers/sizes
        cs = CROP_SCALE
        dets = {"x": [], "y": [], "s": []}
        for bbox in track["bbox"]:
            dets["s"].append(max(bbox[3] - bbox[1], bbox[2] - bbox[0]) / 2)
            dets["y"].append((bbox[1] + bbox[3]) / 2)
            dets["x"].append((bbox[0] + bbox[2]) / 2)
        ks = min(13, len(dets["s"]) | 1)  # kernel must be odd and <= length
        dets["s"] = signal.medfilt(dets["s"], kernel_size=ks)
        dets["x"] = signal.medfilt(dets["x"], kernel_size=ks)
        dets["y"] = signal.medfilt(dets["y"], kernel_size=ks)

        # Write face video
        vOut = cv2.VideoWriter(
            crop_base + "_tmp.avi",
            cv2.VideoWriter_fourcc(*"XVID"), VIDEO_FPS, (224, 224),
        )
        for fidx, frame_idx in enumerate(track["frame"]):
            bs = dets["s"][fidx]
            bsi = int(bs * (1 + 2 * cs))
            image = cv2.imread(flist[frame_idx])
            padded = np.pad(image, ((bsi, bsi), (bsi, bsi), (0, 0)),
                            "constant", constant_values=(110, 110))
            my = dets["y"][fidx] + bsi
            mx = dets["x"][fidx] + bsi
            face = padded[
                int(my - bs):int(my + bs * (1 + 2 * cs)),
                int(mx - bs * (1 + cs)):int(mx + bs * (1 + cs)),
            ]
            vOut.write(cv2.resize(face, (224, 224)))
        vOut.release()

        # Extract corresponding audio segment
        audio_start = track["frame"][0] / VIDEO_FPS
        audio_end = (track["frame"][-1] + 1) / VIDEO_FPS
        audio_tmp = crop_base + ".wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path,
             "-async", "1", "-ac", "1", "-vn", "-acodec", "pcm_s16le",
             "-ar", str(AUDIO_SR),
             "-ss", f"{audio_start:.3f}", "-to", f"{audio_end:.3f}",
             audio_tmp, "-loglevel", "warning"],
            check=False,
        )

        # Merge video + audio
        subprocess.run(
            ["ffmpeg", "-y",
             "-i", crop_base + "_tmp.avi", "-i", audio_tmp,
             "-c:v", "copy", "-c:a", "copy",
             crop_base + ".avi", "-loglevel", "warning"],
            check=False,
        )
        try:
            os.remove(crop_base + "_tmp.avi")
        except OSError:
            pass

        processed.append({"track": track, "proc_track": dets})

    print(f"    Cropped {len(processed)} face tracks")
    return processed


# ═══════════════════════════════════════════════════════════════════════
# Step 5: Light-ASD active speaker scoring
# ═══════════════════════════════════════════════════════════════════════

def run_asd_scoring(crop_dir, pretrain_model=None, device="cuda"):
    """Run Light-ASD model on each cropped face track."""
    _cwd = os.getcwd()
    os.chdir(LIGHT_ASD_DIR)

    if pretrain_model is None:
        pretrain_model = os.path.join(LIGHT_ASD_DIR, "weight", "finetuning_TalkSet.model")
    model = _ASD_cls()
    model.loadParameters(pretrain_model)
    model.eval()
    print(f"    Loaded Light-ASD model: {pretrain_model}")

    os.chdir(_cwd)

    avi_files = sorted(glob.glob(os.path.join(crop_dir, "*.avi")))
    all_scores = []

    for fpath in avi_files:
        basename = os.path.splitext(os.path.basename(fpath))[0]
        wav_path = os.path.join(crop_dir, basename + ".wav")

        if not os.path.exists(wav_path):
            all_scores.append(np.array([0.0]))
            continue

        # Load audio features (MFCC)
        try:
            _, audio = scipy_wavfile.read(wav_path)
        except Exception:
            all_scores.append(np.array([0.0]))
            continue
        if len(audio) == 0:
            all_scores.append(np.array([0.0]))
            continue
        audio_feat = python_speech_features.mfcc(
            audio, AUDIO_SR, numcep=13, winlen=0.025, winstep=0.010,
        )

        # Load video features (grayscale 112x112 center crops)
        cap = cv2.VideoCapture(fpath)
        video_feat = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (224, 224))
            gray = gray[56:168, 56:168]  # center 112x112
            video_feat.append(gray)
        cap.release()
        video_feat = np.array(video_feat)

        if video_feat.shape[0] == 0 or audio_feat.shape[0] == 0:
            all_scores.append(np.array([0.0]))
            continue

        # Align audio/video lengths (in seconds)
        length = min(
            (audio_feat.shape[0] - audio_feat.shape[0] % 4) / 100,
            video_feat.shape[0] / VIDEO_FPS,
        )
        if length <= 0:
            all_scores.append(np.array([0.0]))
            continue

        audio_feat = audio_feat[:int(round(length * 100)), :]
        video_feat = video_feat[:int(round(length * VIDEO_FPS)), :, :]

        # Multi-scale scoring for robustness
        multi_scores = []
        for dur in ASD_DURATION_SET:
            n_batches = int(math.ceil(length / dur))
            scores = []
            with torch.no_grad():
                for i in range(n_batches):
                    a_slice = audio_feat[i * dur * 100:(i + 1) * dur * 100, :]
                    v_slice = video_feat[i * dur * VIDEO_FPS:(i + 1) * dur * VIDEO_FPS, :, :]
                    if a_slice.shape[0] == 0 or v_slice.shape[0] == 0:
                        continue
                    inputA = torch.FloatTensor(a_slice).unsqueeze(0).cuda()
                    inputV = torch.FloatTensor(v_slice).unsqueeze(0).cuda()
                    embedA = model.model.forward_audio_frontend(inputA)
                    embedV = model.model.forward_visual_frontend(inputV)
                    out = model.model.forward_audio_visual_backend(embedA, embedV)
                    score = model.lossAV.forward(out, labels=None)
                    scores.extend(score)
            if scores:
                multi_scores.append(scores)

        if multi_scores:
            final = np.round(
                np.mean(np.array(multi_scores), axis=0), 1
            ).astype(float)
        else:
            final = np.array([0.0])

        all_scores.append(final)

    print(f"    Scored {len(all_scores)} face tracks")
    return all_scores


def run_asd_scoring_visual_only(crop_dir, pretrain_model=None, device="cuda"):
    """Run Light-ASD using only the visual encoder (no audio fusion)."""
    _cwd = os.getcwd()
    os.chdir(LIGHT_ASD_DIR)

    if pretrain_model is None:
        pretrain_model = os.path.join(LIGHT_ASD_DIR, "weight", "finetuning_TalkSet.model")
    model = _ASD_cls()
    model.loadParameters(pretrain_model)
    model.eval()
    print(f"    Loaded Light-ASD model (visual-only): {pretrain_model}")

    os.chdir(_cwd)

    avi_files = sorted(glob.glob(os.path.join(crop_dir, "*.avi")))
    all_scores = []

    for fpath in avi_files:
        # Load video features (grayscale 112x112 center crops)
        cap = cv2.VideoCapture(fpath)
        video_feat = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (224, 224))
            gray = gray[56:168, 56:168]  # center 112x112
            video_feat.append(gray)
        cap.release()
        video_feat = np.array(video_feat)

        if video_feat.shape[0] == 0:
            all_scores.append(np.array([0.0]))
            continue

        length = video_feat.shape[0] / VIDEO_FPS
        if length <= 0:
            all_scores.append(np.array([0.0]))
            continue

        # Multi-scale scoring (visual only)
        multi_scores = []
        for dur in ASD_DURATION_SET:
            n_batches = int(math.ceil(length / dur))
            scores = []
            with torch.no_grad():
                for i in range(n_batches):
                    v_slice = video_feat[i * dur * VIDEO_FPS:(i + 1) * dur * VIDEO_FPS, :, :]
                    if v_slice.shape[0] == 0:
                        continue
                    inputV = torch.FloatTensor(v_slice).unsqueeze(0).to(device)
                    embedV = model.model.forward_visual_frontend(inputV)
                    out = model.model.forward_visual_backend(embedV)
                    # lossV has no labels=None inference path, so
                    # manually run through its FC and extract speaking logit
                    logits = model.lossV.FC(out)
                    score = logits[:, 1].detach().cpu().numpy()
                    scores.extend(score)
            if scores:
                multi_scores.append(scores)

        if multi_scores:
            final = np.round(
                np.mean(np.array(multi_scores), axis=0), 1
            ).astype(float)
        else:
            final = np.array([0.0])

        all_scores.append(final)

    print(f"    Scored {len(all_scores)} face tracks (visual-only)")
    return all_scores


# ═══════════════════════════════════════════════════════════════════════
# Step 5b: LASER / LoCoNet active speaker scoring
# ═══════════════════════════════════════════════════════════════════════

def _load_laser_model(weight_path, device="cuda"):
    """Load LoCoNet model for inference (non-distributed, rank=None path).

    Uses subprocess to avoid sys.path/sys.modules conflicts between
    Light-ASD and LASER — both have 'utils', 'model', 'loss' top-level
    packages that collide.  Instead we manipulate the import environment
    in an isolated way.
    """
    import importlib, importlib.util
    _saved_argv = sys.argv
    _saved_cwd = os.getcwd()
    _saved_path = list(sys.path)

    os.chdir(LASER_ASD_DIR)
    sys.argv = [sys.argv[0], "--cfg", os.path.join(LASER_ASD_DIR, "configs", "multi.yaml")]

    # 1. Evict ALL conflicting cached modules
    _conflict_prefixes = {"utils", "loss_multi", "loconet", "dlhammer", "model",
                          "loss", "ASD", "torchvggish", "builder", "metrics"}
    _saved_mods = {}
    for k in list(sys.modules.keys()):
        top = k.split(".")[0]
        if top in _conflict_prefixes:
            _saved_mods[k] = sys.modules.pop(k)

    # 2. Rebuild sys.path: LASER first, remove Light-ASD to prevent collisions
    sys.path = [LASER_ASD_DIR] + [
        p for p in _saved_path
        if p != LASER_ASD_DIR and "Light-ASD" not in p
    ]

    try:
        from dlhammer.dlhammer import bootstrap as _bootstrap
        from loconet import loconet as _loconet_cls

        cfg = _bootstrap(print_cfg=False)
        # loconet.__init__ hardcodes .cuda() which defaults to cuda:0.
        # Override the default CUDA device so all tensors land on the right GPU.
        target_dev = torch.device(device)
        if target_dev.type == "cuda":
            with torch.cuda.device(target_dev):
                model = _loconet_cls(cfg)
                model.loadParameters(weight_path)
        else:
            model = _loconet_cls(cfg)
            model.loadParameters(weight_path)
        model = model.to(device=device)
        model.eval()
    finally:
        # 3. Restore original state
        os.chdir(_saved_cwd)
        sys.argv = _saved_argv
        sys.path = _saved_path

    return model, cfg


def run_asd_scoring_laser(vid_tracks, frames_dir, audio_path, crop_dir,
                          weight_path=None, device="cuda", visual_only=False):
    """Run LASER/LoCoNet ASD scoring.

    Key difference from Light-ASD: LoCoNet scores faces *jointly*, stacking
    the target speaker's face with up to 2 co-visible faces (NUM_SPEAKERS=3).
    Audio uses VGGish log-mel spectrograms instead of MFCC.
    """
    if weight_path is None:
        weight_path = _ensure_laser_weights("loconet_laser")

    model, cfg = _load_laser_model(weight_path, device)
    print(f"    Loaded LASER/LoCoNet model: {weight_path}")

    # Import VGGish input helper
    sys.path.insert(0, LASER_ASD_DIR)
    from torchvggish import vggish_input as _vggish_input
    from scipy.io import wavfile as _wavfile

    num_speakers = cfg.MODEL.NUM_SPEAKERS  # typically 3

    # ── Build per-track visual features (grayscale 112x112) ──────────
    flist = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    visual_info = []
    for tidx, tinfo in enumerate(vid_tracks):
        track = tinfo["track"]
        crops = []
        for fidx, frame_num in enumerate(track["frame"]):
            frame = cv2.imread(flist[frame_num])
            bbox = track["bbox"][fidx]
            # Use LoCoNet's crop_thumbnail style: padding=0.775, size=112
            face, _ = _crop_thumbnail_laser(frame, bbox, padding=0.775, size=112)
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            crops.append(torch.from_numpy(face))
        visual_info.append({
            "frame": list(track["frame"]),
            "faceCrop": torch.stack(crops, dim=0),
        })

    all_scores = []

    for person_id in range(len(visual_info)):
        # ── Stack multi-person visual features ───────────────────────
        # Find co-visible faces (overlap >= 50% of frames)
        candidate = []
        for i in range(len(visual_info)):
            if i == person_id:
                continue
            intersect = set(visual_info[i]["frame"]).intersection(
                set(visual_info[person_id]["frame"]))
            if len(intersect) >= len(visual_info[person_id]["frame"]) / 2:
                candidate.append({
                    "id": i,
                    "start": visual_info[i]["frame"][0],
                    "end": visual_info[i]["frame"][-1],
                })

        my_crop = visual_info[person_id]["faceCrop"]
        my_start = visual_info[person_id]["frame"][0]
        my_end = visual_info[person_id]["frame"][-1]

        if len(candidate) == 0:
            # No co-visible faces — duplicate self to fill slots
            vis_feat = torch.stack([my_crop, my_crop, my_crop], dim=0)
        elif len(candidate) == 1:
            ctx = _pad_track_tensor(
                visual_info[candidate[0]["id"]]["faceCrop"],
                candidate[0]["start"], candidate[0]["end"],
                my_start, my_end,
            )
            vis_feat = torch.stack([my_crop, ctx, my_crop], dim=0)
        else:
            ctx1 = _pad_track_tensor(
                visual_info[candidate[0]["id"]]["faceCrop"],
                candidate[0]["start"], candidate[0]["end"],
                my_start, my_end,
            )
            ctx2 = _pad_track_tensor(
                visual_info[candidate[-1]["id"]]["faceCrop"],
                candidate[-1]["start"], candidate[-1]["end"],
                my_start, my_end,
            )
            vis_feat = torch.stack([my_crop, ctx1, ctx2], dim=0)

        # vis_feat shape: (S, T, 112, 112)  — S=num_speakers
        vis_feat = vis_feat.unsqueeze(0)  # (1, S, T, 112, 112)

        if visual_only:
            # Visual-only path: skip audio entirely
            scores = _laser_score_visual_only(model, vis_feat, device)
        else:
            # ── Extract audio for this track ─────────────────────────
            track = vid_tracks[person_id]["track"]
            audio_start = track["frame"][0] / VIDEO_FPS
            audio_end = (track["frame"][-1] + 1) / VIDEO_FPS
            audio_tmp = os.path.join(crop_dir, f"laser_audio_{person_id:05d}.wav")
            subprocess.run(
                ["ffmpeg", "-y", "-i", audio_path,
                 "-async", "1", "-ac", "1", "-vn", "-acodec", "pcm_s16le",
                 "-ar", str(AUDIO_SR), "-ss", f"{audio_start:.3f}",
                 "-to", f"{audio_end:.3f}", audio_tmp, "-loglevel", "warning"],
                check=False,
            )

            # VGGish log-mel spectrogram
            sr, wav_data = _wavfile.read(audio_tmp)
            num_video_frames = vis_feat.shape[2]
            audio_feat = _vggish_input.waveform_to_examples(
                wav_data, sr, num_video_frames, VIDEO_FPS, return_tensor=False,
            )
            # audio_feat shape: (4*T, 64) — 4 audio frames per video frame
            audio_feat = torch.from_numpy(audio_feat).unsqueeze(0).unsqueeze(0).float()
            # (1, 1, 4*T, 64)

            scores = _laser_score_av(model, vis_feat, audio_feat, device)

        all_scores.append(scores)

    print(f"    Scored {len(all_scores)} face tracks (LASER/LoCoNet)")
    return all_scores


LASER_CHUNK_SEC = 4  # Process LASER in 4-second chunks to avoid OOM


def _laser_score_av(model, vis_feat, audio_feat, device):
    """Run full AV inference through LoCoNet in temporal chunks."""
    b, s, t = vis_feat.shape[0], vis_feat.shape[1], vis_feat.shape[2]
    chunk_frames = LASER_CHUNK_SEC * VIDEO_FPS  # 100 frames per chunk
    all_scores = []

    with torch.no_grad():
        for start in range(0, t, chunk_frames):
            end = min(start + chunk_frames, t)
            t_chunk = end - start

            vis_chunk = vis_feat[:, :, start:end, :, :].to(device)
            # Audio: 4 frames per video frame
            a_start, a_end = start * 4, end * 4
            aud_chunk = audio_feat[:, :, a_start:a_end, :].to(device, dtype=torch.float)

            audioEmbed = model.model.forward_audio_frontend(aud_chunk)
            visualEmbed = model.model.forward_visual_frontend(
                vis_chunk.view(b * s, *vis_chunk.shape[2:]))
            audioEmbed = audioEmbed.repeat(s, 1, 1)
            audioEmbed, visualEmbed = model.model.forward_cross_attention(
                audioEmbed, visualEmbed)
            outsAV = model.model.forward_audio_visual_backend(
                audioEmbed, visualEmbed, b, s)
            outsAV = outsAV.reshape(b, s, t_chunk, -1)[:, 0, :, :].reshape(b * t_chunk, -1)
            predScore = model.lossAV(outsAV)
            all_scores.append(predScore.detach().cpu().numpy())

    return np.concatenate(all_scores)


def _laser_score_visual_only(model, vis_feat, device):
    """Run visual-only inference through LoCoNet in temporal chunks."""
    b, s, t = vis_feat.shape[0], vis_feat.shape[1], vis_feat.shape[2]
    chunk_frames = LASER_CHUNK_SEC * VIDEO_FPS
    all_scores = []

    with torch.no_grad():
        for start in range(0, t, chunk_frames):
            end = min(start + chunk_frames, t)
            t_chunk = end - start

            vis_chunk = vis_feat[:, :, start:end, :, :].to(device)
            visualEmbed = model.model.forward_visual_frontend(
                vis_chunk.view(b * s, *vis_chunk.shape[2:]))
            outsV = model.model.forward_visual_backend(visualEmbed)
            logits = model.lossV.FC(outsV)
            logits = logits.view(b, s, t_chunk, -1)[:, 0, :, :].view(b * t_chunk, -1)
            all_scores.append(logits[:, 1].detach().cpu().numpy())

    return np.concatenate(all_scores)


def _crop_thumbnail_laser(image, bbox, padding=0.775, size=112):
    """Crop face thumbnail with padding (LoCoNet style)."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    r = max(x2 - x1, y2 - y1) * padding

    p1x, p1y = int(cx - r), int(cy - r)
    p2x, p2y = int(cx + r), int(cy + r)

    img = image.copy()
    if p1x < 0:
        img = cv2.copyMakeBorder(img, 0, 0, -p1x, 0,
                                 cv2.BORDER_CONSTANT, value=[0, 0, 0])
        p2x -= p1x; p1x = 0
    if p1y < 0:
        img = cv2.copyMakeBorder(img, -p1y, 0, 0, 0,
                                 cv2.BORDER_CONSTANT, value=[0, 0, 0])
        p2y -= p1y; p1y = 0
    if p2x > img.shape[1]:
        img = cv2.copyMakeBorder(img, 0, 0, 0, p2x - img.shape[1],
                                 cv2.BORDER_CONSTANT, value=[0, 0, 0])
    if p2y > img.shape[0]:
        img = cv2.copyMakeBorder(img, 0, p2y - img.shape[0], 0, 0,
                                 cv2.BORDER_CONSTANT, value=[0, 0, 0])

    output = img[p1y:p2y, p1x:p2x]
    output = cv2.resize(output, (size, size), interpolation=cv2.INTER_LINEAR)

    s_x = size / max(p2x - p1x, 1)
    s_y = size / max(p2y - p1y, 1)
    new_bbox = [(x1 - p1x) * s_x, (y1 - p1y) * s_y,
                (x2 - p1x) * s_x, (y2 - p1y) * s_y]
    return output, new_bbox


def _pad_track_tensor(tensor, start_src, end_src, start_tgt, end_tgt):
    """Pad/slice a track tensor to align with target track's frame range."""
    # Slice the overlapping portion
    begin = max(start_src, start_tgt) - start_src
    end = min(end_src, end_tgt) - start_src + 1
    result = tensor[begin:end]

    # Pad left if source starts after target
    if start_src > start_tgt:
        pad_left = start_src - start_tgt
        zeros = torch.zeros_like(tensor[0]).unsqueeze(0).expand(pad_left, -1, -1)
        result = torch.cat((zeros, result), dim=0)

    # Pad right if source ends before target
    if end_src < end_tgt:
        pad_right = end_tgt - end_src
        zeros = torch.zeros_like(tensor[0]).unsqueeze(0).expand(pad_right, -1, -1)
        result = torch.cat((result, zeros), dim=0)

    return result


# ═══════════════════════════════════════════════════════════════════════
# Step 6: Diarization + Speaker Recognition
# ═══════════════════════════════════════════════════════════════════════

def _patch_torch_load():
    """Patch torch.load to default weights_only=False for PyTorch >= 2.6
    which changed the default to True, breaking older model checkpoints."""
    import functools
    _original = torch.load
    @functools.wraps(_original)
    def _patched(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _original(*args, **kwargs)
    torch.load = _patched


def run_speaker_identification(audio_path):
    """
    Run diarization and speaker recognition on the audio track.

    Returns:
        speaker_timeline: [(start_s, end_s, voice_id), ...]
        speaker_voices:   {diar_label: voice_id}
    """
    # Import directly from file paths to avoid ginny_server/core_api/__init__.py
    # which loads ERes2NetV2, InsightFace, and other unneeded singletons.
    import importlib.util

    _diar_spec = importlib.util.spec_from_file_location(
        "_diar_mod",
        os.path.join(SCRIPT_DIR, "ginny_server", "core_api",
                     "diarization", "diarization.py"),
    )
    _diar_mod = importlib.util.module_from_spec(_diar_spec)
    _diar_spec.loader.exec_module(_diar_mod)
    _Diarization = _diar_mod._Diarization

    _sr_spec = importlib.util.spec_from_file_location(
        "_sr_mod",
        os.path.join(SCRIPT_DIR, "ginny_server", "core_api",
                     "speaker_recognition", "speaker_recognition.py"),
    )
    _sr_mod = importlib.util.module_from_spec(_sr_spec)
    _sr_spec.loader.exec_module(_sr_mod)
    _SpeakerRecognition = _sr_mod._SpeakerRecognition

    with wave.open(audio_path, "rb") as wf:
        sr = wf.getframerate()
        audio_bytes = wf.readframes(wf.getnframes())
    audio_dur = (len(audio_bytes) // 2) / sr

    if audio_dur < 20:
        print("    Audio too short for diarization (< 20s), skipping speaker ID")
        return [], {}

    print(f"    Initialising diarization + speaker recognition (WavLM)...")
    _patch_torch_load()
    diarization = _Diarization()
    speaker_rec = _SpeakerRecognition(model_name="wavlm_ssl")
    speaker_rec.init_enrollment_buffer()

    print(f"    Running diarization on {audio_dur:.1f}s audio...")
    segments = diarization.diarize(audio_bytes, sr)
    print(f"    Diarization found {len(segments)} segments")

    # Group segments by speaker label
    clusters = {}
    for seg in segments:
        label = seg["speaker"]
        if label not in clusters:
            clusters[label] = {"audio": bytearray(), "segments": []}
        clusters[label]["audio"].extend(seg["audio"])
        clusters[label]["segments"].append((seg["start"], seg["end"]))

    speaker_voices = {}
    for label, cluster in clusters.items():
        cluster_audio = bytes(cluster["audio"])
        cluster_dur = (len(cluster_audio) // 2) / sr
        if cluster_dur < 1.0:
            speaker_voices[label] = f"Speaker_{label}"
            continue
        try:
            embedding = speaker_rec.extract_embedding(cluster_audio, sr)
            result = speaker_rec.match_or_buffer(
                embedding, cluster_audio, sr, min_samples=1,
            )
            vid = (result.get("voice_id")
                   or result.get("pending_id")
                   or f"Speaker_{label}")
            speaker_voices[label] = vid
            conf = result.get("confidence", 0)
            print(f"      {label} -> {vid} (conf={conf:.2f})")
        except Exception as e:
            print(f"      {label} -> identification failed: {e}")
            speaker_voices[label] = f"Speaker_{label}"

    # Flush pending enrollments
    try:
        flushed = speaker_rec.flush_all_pending(sr, min_samples=1)
        for r in flushed:
            # Update any pending references
            for label, vid in speaker_voices.items():
                if vid == r.get("pending_id"):
                    speaker_voices[label] = r["voice_id"]
            print(f"      Enrolled: {r['voice_id']}")
    except Exception:
        pass

    # Build timeline
    timeline = []
    for label, cluster in clusters.items():
        vid = speaker_voices[label]
        for start, end in cluster["segments"]:
            timeline.append((start, end, vid))
    timeline.sort(key=lambda x: x[0])

    return timeline, speaker_voices


# ═══════════════════════════════════════════════════════════════════════
# Step 7: Map speakers to face tracks
# ═══════════════════════════════════════════════════════════════════════

def map_speakers_to_tracks(vid_tracks, asd_scores, speaker_timeline):
    """
    Map identified voice IDs to face tracks by correlating
    ASD speaking times with diarization speaker segments.
    """
    track_labels = {}  # track_idx -> voice_id

    if not speaker_timeline:
        # No speaker ID available; label tracks by index
        for tidx in range(len(vid_tracks)):
            track_labels[tidx] = f"Person_{tidx + 1}"
        return track_labels

    for tidx, tinfo in enumerate(vid_tracks):
        track = tinfo["track"]
        scores = asd_scores[tidx] if tidx < len(asd_scores) else np.array([0])

        # Collect timestamps where this face is speaking
        speaking_times = []
        for fidx, frame_idx in enumerate(track["frame"]):
            sidx = min(fidx, len(scores) - 1)
            if scores[sidx] > ASD_SPEAKING_THRESH:
                speaking_times.append(frame_idx / VIDEO_FPS)

        if not speaking_times:
            track_labels[tidx] = f"Person_{tidx + 1}"
            continue

        # Count overlap with each speaker's segments
        overlap = {}
        for start, end, vid in speaker_timeline:
            n = sum(1 for t in speaking_times if start <= t <= end)
            if n > 0:
                overlap[vid] = overlap.get(vid, 0) + n

        if overlap:
            track_labels[tidx] = max(overlap, key=overlap.get)
        else:
            track_labels[tidx] = f"Person_{tidx + 1}"

    return track_labels


# ═══════════════════════════════════════════════════════════════════════
# Step 8: Render annotated output video
# ═══════════════════════════════════════════════════════════════════════

def render_output(vid_tracks, asd_scores, track_labels,
                  frames_dir, audio_path, output_path):
    """Draw bounding boxes on active speakers and write output video."""
    flist = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    num_frames = len(flist)

    # Pre-build per-frame face list for fast lookup
    faces_per_frame = [[] for _ in range(num_frames)]
    for tidx, tinfo in enumerate(vid_tracks):
        track = tinfo["track"]
        proc = tinfo["proc_track"]
        scores = asd_scores[tidx] if tidx < len(asd_scores) else np.array([0])
        label = track_labels.get(tidx, f"Person_{tidx + 1}")

        for fidx, frame_idx in enumerate(track["frame"].tolist()):
            # Temporal smoothing of scores (window of 5)
            lo = max(fidx - 2, 0)
            hi = min(fidx + 3, len(scores))
            s = np.mean(scores[lo:hi]) if hi > lo else 0
            faces_per_frame[frame_idx].append({
                "score": float(s),
                "s": proc["s"][fidx],
                "x": proc["x"][fidx],
                "y": proc["y"][fidx],
                "label": label,
                "tidx": tidx,
            })

    # Assign colors per speaker label
    label_colors = {}
    for tidx, label in track_labels.items():
        if label not in label_colors:
            label_colors[label] = SPEAKER_COLORS[
                len(label_colors) % len(SPEAKER_COLORS)
            ]

    # Read first frame for dimensions
    first = cv2.imread(flist[0])
    fh, fw = first.shape[:2]

    tmp_video = output_path.rsplit(".", 1)[0] + "_tmp.avi"
    writer = cv2.VideoWriter(
        tmp_video, cv2.VideoWriter_fourcc(*"XVID"), VIDEO_FPS, (fw, fh),
    )

    t0 = time.time()
    for fidx, fname in enumerate(flist):
        image = cv2.imread(fname)
        active_labels = []

        for face in faces_per_frame[fidx]:
            x = int(face["x"])
            y = int(face["y"])
            s = int(face["s"])
            is_speaking = face["score"] > ASD_SPEAKING_THRESH
            label = face["label"]
            color = label_colors.get(label, INACTIVE_COLOR)

            if is_speaking:
                # Thick colored bounding box
                cv2.rectangle(image, (x - s, y - s), (x + s, y + s),
                              color, 3)

                # Label with background
                disp = f"{label} (SPEAKING)"
                (tw, th), baseline = cv2.getTextSize(
                    disp, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2,
                )
                # Background rect above the box
                cv2.rectangle(
                    image,
                    (x - s, y - s - th - 12),
                    (x - s + tw + 8, y - s - 2),
                    color, -1,
                )
                cv2.putText(
                    image, disp,
                    (x - s + 4, y - s - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2,
                )

                # ASD confidence score below box
                cv2.putText(
                    image, f"{face['score']:.1f}",
                    (x - s, y + s + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
                )
                active_labels.append(label)
            else:
                # Thin gray bounding box for non-speaking faces
                cv2.rectangle(image, (x - s, y - s), (x + s, y + s),
                              INACTIVE_COLOR, 1)

        # Bottom status bar
        overlay = image.copy()
        cv2.rectangle(overlay, (0, fh - 40), (fw, fh), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

        if active_labels:
            unique = list(dict.fromkeys(active_labels))  # preserve order, dedup
            status = "Speaking: " + ", ".join(unique)
            cv2.putText(image, status, (10, fh - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)
        else:
            cv2.putText(image, "No active speaker", (10, fh - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, INACTIVE_COLOR, 1)

        writer.write(image)

        if (fidx + 1) % 500 == 0 or fidx == num_frames - 1:
            elapsed = time.time() - t0
            print(f"    Rendering: {fidx+1}/{num_frames} frames ({elapsed:.1f}s)")

    writer.release()

    # Mux with original audio
    print(f"    Temp video: {tmp_video} ({os.path.getsize(tmp_video)} bytes)")
    result = subprocess.run(
        ["ffmpeg", "-y",
         "-i", tmp_video, "-i", audio_path,
         "-c:v", "mpeg4", "-q:v", "5",
         "-c:a", "aac", "-b:a", "128k",
         "-shortest", output_path],
        capture_output=True,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")
        print(f"    FFmpeg mux failed (rc={result.returncode}): {stderr[-500:]}")
        # Fallback: try without re-encoding, just copy streams
        result2 = subprocess.run(
            ["ffmpeg", "-y",
             "-i", tmp_video, "-i", audio_path,
             "-c", "copy", "-shortest", output_path],
            capture_output=True,
        )
        if result2.returncode != 0:
            # Last resort: just copy the temp video without audio
            print(f"    Fallback: saving video without audio")
            shutil.copy2(tmp_video, output_path)

    try:
        os.remove(tmp_video)
    except OSError:
        pass

    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"    Output saved: {output_path} ({size_mb:.1f} MB)")
    else:
        print(f"    ERROR: Output file was not created!")


# ═══════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Active Speaker Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python asd_pipeline.py marking_work/Meeting01.mp4
  python asd_pipeline.py marking_work/Meeting01.mp4 --duration 60
  python asd_pipeline.py marking_work/Meeting01.mp4 --no-speaker-id
        """,
    )
    parser.add_argument("video", help="Path to input video")
    parser.add_argument("--duration", type=int, default=0,
                        help="Process only first N seconds (0 = full)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Torch device for face detection and Light-ASD (default: cuda:0)")
    parser.add_argument("--asd-device", type=str, default=None,
                        help="Torch device for ASD model, useful to put LASER on a "
                             "separate GPU (default: same as --device)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: <input>_asd.mp4)")
    parser.add_argument("--no-speaker-id", action="store_true",
                        help="Skip diarization/speaker recognition "
                             "(only show active speaker detection)")
    parser.add_argument("--asd-model", type=str, default="talkset",
                        choices=["ava", "talkset"],
                        help="Light-ASD model: 'ava' (AVA-CVPR pretrain) "
                             "or 'talkset' (TalkSet fine-tuned, default)")
    parser.add_argument("--asd-engine", type=str, default="light",
                        choices=["light", "laser"],
                        help="ASD engine: 'light' (Light-ASD, default) "
                             "or 'laser' (LASER/LoCoNet, heavier but more accurate)")
    parser.add_argument("--visual-only", action="store_true",
                        help="Use only the visual encoder for ASD "
                             "(no audio-visual fusion, less accurate)")
    args = parser.parse_args()

    if not os.path.isfile(args.video):
        print(f"Error: Video not found: {args.video}")
        sys.exit(1)

    if args.asd_device is None:
        args.asd_device = args.device

    if args.output is None:
        base, ext = os.path.splitext(args.video)
        args.output = base + "_asd.mp4"

    print("=" * 65)
    print("  Active Speaker Detection Pipeline")
    print(f"  Input:    {args.video}")
    print(f"  Output:   {args.output}")
    print(f"  Duration: {'full video' if args.duration == 0 else f'{args.duration}s'}")
    print(f"  Device:   {args.device}")
    print(f"  ASD engine: {args.asd_engine}")
    print(f"  ASD model: {args.asd_model}")
    print(f"  ASD mode:  {'visual-only' if args.visual_only else 'audio-visual'}")
    print(f"  Speaker ID: {'disabled' if args.no_speaker_id else 'enabled (WavLM)'}")
    print("=" * 65)

    work_dir = tempfile.mkdtemp(prefix="asd_pipeline_")
    frames_dir = os.path.join(work_dir, "frames")
    crop_dir = os.path.join(work_dir, "crops")
    os.makedirs(frames_dir)
    os.makedirs(crop_dir)

    pipeline_t0 = time.time()

    try:
        # ── 1. Extract media ──────────────────────────────────────────
        print("\n[1/7] Extracting video and audio...")
        video_path, audio_path = extract_media(
            args.video, work_dir, args.duration,
        )

        # ── 2. Extract frames ─────────────────────────────────────────
        print("\n[2/7] Extracting frames...")
        flist = extract_frames(video_path, frames_dir)
        print(f"    {len(flist)} frames extracted")

        # ── 3. Face detection ─────────────────────────────────────────
        print("\n[3/7] Detecting faces (S3FD)...")
        face_dets = detect_faces(frames_dir, device=args.device)

        # ── 4. Face tracking ──────────────────────────────────────────
        print("\n[4/7] Tracking faces...")
        tracks = track_faces(face_dets)

        if not tracks:
            print("    No face tracks found. Cannot produce output.")
            return

        # ── 5. Crop + ASD scoring ─────────────────────────────────────
        vid_tracks = crop_tracks(tracks, frames_dir, crop_dir, audio_path)

        if args.asd_engine == "laser":
            print("\n[5/7] Running LASER/LoCoNet ASD...")
            laser_weight = _ensure_laser_weights("loconet_laser")
            asd_scores = run_asd_scoring_laser(
                vid_tracks, frames_dir, audio_path, crop_dir,
                weight_path=laser_weight, device=args.asd_device,
                visual_only=args.visual_only,
            )
        else:
            print("\n[5/7] Running Light-ASD...")
            asd_model_files = {
                "ava": "pretrain_AVA_CVPR.model",
                "talkset": "finetuning_TalkSet.model",
            }
            pretrain_model = os.path.join(
                LIGHT_ASD_DIR, "weight", asd_model_files[args.asd_model],
            )
            if args.visual_only:
                asd_scores = run_asd_scoring_visual_only(crop_dir, pretrain_model, device=args.asd_device)
            else:
                asd_scores = run_asd_scoring(crop_dir, pretrain_model, device=args.asd_device)

        # Print per-track summary
        for tidx, (tinfo, scores) in enumerate(zip(vid_tracks, asd_scores)):
            t_start = tinfo["track"]["frame"][0] / VIDEO_FPS
            t_end = (tinfo["track"]["frame"][-1] + 1) / VIDEO_FPS
            speaking_pct = np.mean(scores > ASD_SPEAKING_THRESH) * 100
            print(f"    Track {tidx}: [{t_start:.1f}s - {t_end:.1f}s] "
                  f"speaking {speaking_pct:.0f}% of frames")

        # ── 6. Speaker identification ─────────────────────────────────
        if args.no_speaker_id:
            print("\n[6/7] Speaker identification skipped (--no-speaker-id)")
            speaker_timeline = []
        else:
            print("\n[6/7] Running diarization + speaker recognition...")
            speaker_timeline, _ = run_speaker_identification(audio_path)

        # ── 7. Map + render ───────────────────────────────────────────
        print("\n[7/7] Mapping speakers to faces and rendering output...")
        track_labels = map_speakers_to_tracks(
            vid_tracks, asd_scores, speaker_timeline,
        )
        for tidx, label in track_labels.items():
            t = vid_tracks[tidx]["track"]
            print(f"    Track {tidx} -> {label}")

        render_output(
            vid_tracks, asd_scores, track_labels,
            frames_dir, audio_path, args.output,
        )

        elapsed = time.time() - pipeline_t0
        print(f"\n{'=' * 65}")
        print(f"  Pipeline complete in {elapsed:.1f}s")
        print(f"  Output: {args.output}")
        print(f"{'=' * 65}")

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
