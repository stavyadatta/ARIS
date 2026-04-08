"""
Batch video processor — sliding window diarization + multi-sample voice ReID.

Pipeline (simulates real-time):
1. Split audio into 30s windows with 10s overlap
2. DiariZen clusters each window internally (wespeaker)
3. Concat audio per cluster → speaker embedding (model-dependent)
4. Match against voice DB or enrollment buffer
5. 3-sample enrollment: average embeddings before saving

Face recognition runs independently on frames (decoupled).
"""
import os
import cv2
import wave
import logging
import tempfile
import datetime
import subprocess
import multiprocessing as mp
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("speaker_recognition")

CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
BLUE = "\033[94m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _ts():
    return datetime.datetime.now().strftime("%H:%M:%S")


def _vt(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


FACE_COLORS = [
    (255, 100, 50), (200, 50, 50), (255, 150, 0),
    (180, 80, 120), (255, 200, 100),
]
VOICE_COLORS = [
    (0, 255, 0), (0, 200, 100), (0, 255, 200),
    (100, 255, 0), (0, 180, 0),
]
DIAR_COLORS = [
    (0, 200, 255), (255, 100, 200), (100, 255, 255),
    (200, 100, 255),
]


def _get_color(identity, color_map, palette):
    if identity not in color_map:
        color_map[identity] = palette[len(color_map) % len(palette)]
    return color_map[identity]


def _annotate_chunk(args):
    """
    Worker process: annotate a contiguous slice of frames and write a chunk .mp4.

    Visuals replicate the original layout exactly, minus the timestamp.
    Receives precomputed per-frame voice/diar entries (no timeline scans).
    """
    (src_path, start_frame, end_frame, fps, frame_w, frame_h,
     voice_slice, diar_slice, num_diar_speakers, out_path) = args

    # Each worker is single-threaded; we parallelize at the process level.
    cv2.setNumThreads(1)

    cap = cv2.VideoCapture(src_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (frame_w, frame_h))

    n = end_frame - start_frame
    written = 0
    for i in range(n):
        ret, frame = cap.read()
        if not ret:
            break

        # ---- VOICE: bottom bar (same look as original) ----
        v_entry = voice_slice[i]
        if v_entry is not None:
            vid, vconf, vcolor = v_entry
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, frame_h - 70), (frame_w, frame_h),
                          (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            cv2.putText(frame, f"VOICE: {vid}", (10, frame_h - 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, vcolor, 2)
            bar_w = int(vconf * 200)
            cv2.rectangle(frame, (10, frame_h - 25),
                          (10 + bar_w, frame_h - 15), vcolor, -1)
            cv2.rectangle(frame, (10, frame_h - 25),
                          (210, frame_h - 15), (100, 100, 100), 1)
            cv2.putText(frame, f"{vconf:.2f}", (220, frame_h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.circle(frame, (frame_w - 30, 30), 10, vcolor, -1)
        else:
            cv2.circle(frame, (frame_w - 30, 30), 10, (80, 80, 80), -1)

        # ---- DIARIZATION: top-right panel ----
        cv2.putText(frame, f"DIAR: {num_diar_speakers} spk",
                    (frame_w - 180, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        diar_y = 55
        d_active = diar_slice[i]
        if d_active:
            for dlabel, dvid, dcolor in d_active:
                cv2.circle(frame, (frame_w - 175, diar_y - 5), 6, dcolor, -1)
                txt = f"{dlabel}"
                dvid_s = str(dvid) if dvid is not None else ""
                if dvid_s and dvid_s != "?" and not dvid_s.startswith("pending"):
                    txt += f" = {dvid_s}"
                cv2.putText(frame, txt, (frame_w - 163, diar_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, dcolor, 1)
                diar_y += 20
        else:
            cv2.putText(frame, "(silence)", (frame_w - 163, diar_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

        out.write(frame)
        written += 1

    cap.release()
    out.release()
    return out_path, written


def process_video(video_path, max_duration, speaker_recognition, diarization):
    """
    Process video with sliding-window diarization + multi-sample voice ReID.
    Face recognition has been removed from this path to isolate annotation cost.
    """
    logger.info(f"  {DIM}{_ts()}{RESET}  {CYAN}.. VIDEO{RESET}        Starting sliding-window pipeline")

    # Trim if needed
    if max_duration > 0:
        temp_trimmed = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_trimmed.close()
        result = subprocess.run([
            "ffmpeg", "-y", "-i", video_path,
            "-t", str(max_duration), "-c", "copy", temp_trimmed.name
        ], capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"Trim failed: {result.stderr.decode(errors='replace')[-200:]}")
        working_video = temp_trimmed.name
    else:
        working_video = video_path

    # Extract audio
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_wav.close()
    subprocess.run([
        "ffmpeg", "-y", "-i", working_video,
        "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", temp_wav.name
    ], capture_output=True, check=True)

    with wave.open(temp_wav.name, 'rb') as wf:
        sample_rate = wf.getframerate()
        all_audio = wf.readframes(wf.getnframes())
    os.remove(temp_wav.name)

    audio_duration = (len(all_audio) // 2) / sample_rate
    logger.info(f"  {DIM}{_ts()}{RESET}  {GREEN}OK AUDIO{RESET}       {audio_duration:.1f}s extracted")

    # Open video
    cap = cv2.VideoCapture(working_video)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info(f"  {DIM}{_ts()}{RESET}  {GREEN}OK VIDEO{RESET}       {frame_w}x{frame_h} @ {video_fps:.0f}fps, {total_frames} frames")

    # ===================== SLIDING WINDOW DIARIZATION + ReID =====================
    WINDOW_SIZE = 30.0  # seconds
    WINDOW_OVERLAP = 10.0  # seconds
    WINDOW_STEP = WINDOW_SIZE - WINDOW_OVERLAP  # 20s step
    MIN_CLUSTER_AUDIO = 3.0  # seconds — skip clusters shorter than this
    MIN_ENROLL_SAMPLES = 3

    # Initialize enrollment buffer
    speaker_recognition.init_enrollment_buffer()

    # Build timelines
    voice_timeline = []     # [(start, end, voice_id, confidence, status)]
    diar_timeline = []      # [(start, end, diar_label, voice_id, window_num)]
    voice_color_map = {}
    diar_color_map = {}

    num_windows = max(1, int((audio_duration - WINDOW_SIZE) / WINDOW_STEP) + 1)
    if audio_duration < WINDOW_SIZE:
        num_windows = 1

    logger.info(f"  {DIM}{_ts()}{RESET}  {CYAN}.. WINDOWS{RESET}     {num_windows} windows ({WINDOW_SIZE:.0f}s each, {WINDOW_OVERLAP:.0f}s overlap)")

    for win_idx in range(num_windows):
        win_start = win_idx * WINDOW_STEP
        win_end = min(win_start + WINDOW_SIZE, audio_duration)
        win_dur = win_end - win_start

        # Skip windows shorter than diarization minimum
        if win_dur < diarization.MIN_DIARIZATION_DURATION:
            logger.info(f"  {DIM}{_ts()}{RESET}  {YELLOW}!! WIN {win_idx+1}{RESET}      too short ({win_dur:.1f}s), skipping")
            continue

        yield f"Window {win_idx+1}/{num_windows} [{_vt(win_start)}-{_vt(win_end)}]...", None

        # Extract window audio bytes
        start_sample = int(win_start * sample_rate) * 2  # PCM_16 = 2 bytes/sample
        end_sample = int(win_end * sample_rate) * 2
        window_audio = all_audio[start_sample:end_sample]

        # Step 1: Diarize this window
        try:
            diar_segments = diarization.diarize(window_audio, sample_rate)
        except Exception as e:
            logger.error(f"  {DIM}{_ts()}{RESET}  {YELLOW}!! DIAR{RESET}        Window {win_idx+1} failed: {e}")
            continue

        # Group segments by diarization speaker label
        clusters = {}  # diar_label → {"audio": bytearray, "segments": [(start, end)]}
        for seg in diar_segments:
            label = seg["speaker"]
            if label not in clusters:
                clusters[label] = {"audio": bytearray(), "segments": []}
            clusters[label]["audio"].extend(seg["audio"])
            clusters[label]["segments"].append((seg["start"] + win_start, seg["end"] + win_start))

        cluster_summary = ', '.join(
            "{}({:.1f}s)".format(k, len(v["audio"]) // 2 / sample_rate)
            for k, v in clusters.items()
        )
        logger.info(
            f"  {DIM}{_ts()}{RESET}  {GREEN}  WIN {win_idx+1:<3}{RESET}     "
            f"[{_vt(win_start)}-{_vt(win_end)}]  "
            f"{len(clusters)} speakers: {cluster_summary}"
        )

        # Step 2: For each cluster, extract ERes2NetV2 embedding and match/buffer
        for diar_label, cluster in clusters.items():
            cluster_audio = bytes(cluster["audio"])
            cluster_dur = (len(cluster_audio) // 2) / sample_rate

            _get_color(diar_label, diar_color_map, DIAR_COLORS)

            if cluster_dur < MIN_CLUSTER_AUDIO:
                logger.info(f"  {DIM}{_ts()}{RESET}  {DIM}    {diar_label}: {cluster_dur:.1f}s (too short, skipping){RESET}")
                for seg_start, seg_end in cluster["segments"]:
                    diar_timeline.append((seg_start, seg_end, diar_label, "?", win_idx + 1))
                continue

            # Extract embedding from concatenated cluster audio
            try:
                embedding = speaker_recognition.extract_embedding(cluster_audio, sample_rate)
            except Exception as e:
                logger.error(f"  {DIM}{_ts()}{RESET}  {YELLOW}    {diar_label}: embedding failed: {e}{RESET}")
                for seg_start, seg_end in cluster["segments"]:
                    diar_timeline.append((seg_start, seg_end, diar_label, "?", win_idx + 1))
                continue

            # Match or buffer for enrollment
            result = speaker_recognition.match_or_buffer(
                embedding, cluster_audio, sample_rate,
                min_samples=MIN_ENROLL_SAMPLES
            )

            voice_id = result["voice_id"] or result.get("pending_id", "?")
            confidence = result["confidence"]

            if result["voice_id"]:
                _get_color(result["voice_id"], voice_color_map, VOICE_COLORS)

            # Add to timelines
            for seg_start, seg_end in cluster["segments"]:
                diar_timeline.append((seg_start, seg_end, diar_label, voice_id, win_idx + 1))
                if result["voice_id"]:
                    voice_timeline.append((seg_start, seg_end, result["voice_id"], confidence, result["status"]))

            logger.info(
                f"  {DIM}{_ts()}{RESET}  {GREEN}    {diar_label}{RESET}: "
                f"{cluster_dur:.1f}s → {BOLD}{voice_id}{RESET}  "
                f"conf={confidence:.2f}  {DIM}{result['status']}{RESET}"
            )

    # Flush remaining pending enrollments (allow entries with 2+ samples at end)
    logger.info(f"  {DIM}{_ts()}{RESET}  {CYAN}.. FLUSH{RESET}       Enrolling remaining pending voices...")
    flushed = speaker_recognition.flush_all_pending(sample_rate, min_samples=2)
    for r in flushed:
        _get_color(r["voice_id"], voice_color_map, VOICE_COLORS)

    num_diar_speakers = len(diar_color_map)
    num_voices = len(voice_color_map)
    logger.info(f"  {DIM}{_ts()}{RESET}  {GREEN}OK PIPELINE{RESET}    {num_diar_speakers} diar speakers, {num_voices} enrolled voices")

    # ===================== ANNOTATE VIDEO (PARALLEL) =====================
    # Strategy:
    #   1. Resolve all colors NOW (main thread) so workers don't mutate
    #      shared color_map dicts.
    #   2. Build per-frame lookup arrays voice_per_frame / diar_per_frame.
    #      Each frame index resolves to a tiny picklable tuple — no
    #      timeline scans inside the worker.
    #   3. Split frames into N == cpu_count chunks. Each worker opens its
    #      own VideoCapture, seeks to its start frame, annotates, writes
    #      its own segment .mp4.
    #   4. ffmpeg concat all segments + mux original audio in one call.
    cap.release()  # release main-thread capture before workers spawn

    yield "Annotating (parallel) ...", None

    # ---- 1. Pre-resolve colors for every voice/diar id we'll draw ----
    for _, _, vid, _, _ in voice_timeline:
        if vid:
            _get_color(vid, voice_color_map, VOICE_COLORS)
    for _, _, dlabel, _, _ in diar_timeline:
        _get_color(dlabel, diar_color_map, DIAR_COLORS)

    # ---- 2. Build per-frame lookup arrays ----
    voice_per_frame = [None] * total_frames
    diar_per_frame = [[] for _ in range(total_frames)]

    for s, e, vid, conf, _status in voice_timeline:
        if not vid:
            continue
        vcolor = voice_color_map[vid]
        f0 = max(0, int(s * video_fps))
        f1 = min(total_frames, int(e * video_fps) + 1)
        for f in range(f0, f1):
            # Last writer wins on overlap (matches original linear-scan behavior)
            voice_per_frame[f] = (vid, conf, vcolor)

    for s, e, dlabel, vid, _wnum in diar_timeline:
        dcolor = diar_color_map[dlabel]
        entry = (dlabel, vid, dcolor)
        f0 = max(0, int(s * video_fps))
        f1 = min(total_frames, int(e * video_fps) + 1)
        for f in range(f0, f1):
            diar_per_frame[f].append(entry)

    # ---- 3. Split into chunks and run pool ----
    n_workers = max(1, mp.cpu_count())
    if total_frames < n_workers:
        n_workers = max(1, total_frames)
    chunk_size = (total_frames + n_workers - 1) // n_workers

    chunk_paths = []
    chunk_args = []
    for w in range(n_workers):
        f_start = w * chunk_size
        f_end = min(f_start + chunk_size, total_frames)
        if f_start >= f_end:
            break
        chunk_path = tempfile.NamedTemporaryFile(
            delete=False, suffix=f"_chunk{w:02d}.mp4"
        ).name
        chunk_paths.append(chunk_path)
        chunk_args.append((
            working_video, f_start, f_end, video_fps, frame_w, frame_h,
            voice_per_frame[f_start:f_end],
            diar_per_frame[f_start:f_end],
            num_diar_speakers,
            chunk_path,
        ))

    logger.info(
        f"  {DIM}{_ts()}{RESET}  {CYAN}.. PARALLEL{RESET}    "
        f"{total_frames} frames across {len(chunk_args)} workers "
        f"(~{chunk_size} frames/worker)"
    )

    # Use spawn context to avoid potential cv2/fork interactions in a server.
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(chunk_args)) as pool:
        results = pool.map(_annotate_chunk, chunk_args)

    total_written = sum(n for _, n in results)
    logger.info(
        f"  {DIM}{_ts()}{RESET}  {GREEN}{BOLD}OK ANNOTATED{RESET}   "
        f"{total_written} frames in {len(chunk_args)} chunks"
    )

    # ---- 4. ffmpeg concat segments + mux original audio in one call ----
    concat_list_file = tempfile.NamedTemporaryFile(
        delete=False, suffix=".txt", mode="w"
    )
    for cp in chunk_paths:
        concat_list_file.write(f"file '{cp}'\n")
    concat_list_file.close()

    final_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", concat_list_file.name,
            "-i", working_video,
            "-c:v", "copy",
            "-map", "0:v:0", "-map", "1:a:0",
            "-shortest", final_output
        ], capture_output=True, check=True)
        logger.info(
            f"  {DIM}{_ts()}{RESET}  {GREEN}OK MUX{RESET}         "
            f"Concatenated {len(chunk_paths)} chunks + audio"
        )
    except subprocess.CalledProcessError as e:
        logger.warning(
            f"  {DIM}{_ts()}{RESET}  {YELLOW}!! MUX{RESET}         "
            f"{e.stderr.decode(errors='replace')[-200:] if e.stderr else e}"
        )
        # Fallback: try concat without mux (audio missing) so user still
        # gets a video back rather than nothing.
        try:
            subprocess.run([
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0", "-i", concat_list_file.name,
                "-c", "copy", final_output
            ], capture_output=True, check=True)
        except Exception:
            final_output = chunk_paths[0] if chunk_paths else None

    # Cleanup chunks + concat list
    for cp in chunk_paths:
        try:
            os.remove(cp)
        except OSError:
            pass
    try:
        os.remove(concat_list_file.name)
    except OSError:
        pass

    if working_video != video_path and os.path.exists(working_video):
        os.remove(working_video)

    # ---- SUMMARY ----
    print(f"""
{CYAN}{'=' * 65}
   VIDEO PROCESSING SUMMARY
   Sliding Window Diarization + Multi-Sample Voice ReID
{'=' * 65}{RESET}

{BOLD}  Pipeline:{RESET}
    Windows       {num_windows} x {WINDOW_SIZE:.0f}s (overlap {WINDOW_OVERLAP:.0f}s)
    Diar speakers  {num_diar_speakers} (DiariZen clustering per window)
    Voice IDs      {num_voices} (ERes2NetV2, {MIN_ENROLL_SAMPLES}-sample enrollment)

{GREEN}{BOLD}  Voice ReID Timeline:{RESET}""")
    last_voice = None
    for seg_start, seg_end, vid, conf, status in voice_timeline:
        if vid != last_voice:
            arrow = f"  {MAGENTA}>>{RESET}" if last_voice else f"  {GREEN}>>{RESET}"
            print(f"{arrow}  [{_vt(seg_start)} - {_vt(seg_end)}]  {BOLD}{vid:<14}{RESET}  conf={conf:.2f}")
            last_voice = vid
        else:
            print(f"      [{_vt(seg_start)} - {_vt(seg_end)}]  {DIM}{vid:<14}  conf={conf:.2f}{RESET}")

    if diar_timeline:
        print(f"""
{CYAN}{BOLD}  Diarization Timeline:{RESET}""")
        last_diar = None
        for seg_start, seg_end, dlabel, vid, wnum in diar_timeline:
            link = f"= {vid}" if vid and vid != "?" and not vid.startswith("pending") else ""
            if dlabel != last_diar:
                arrow = f"  {MAGENTA}>>{RESET}" if last_diar else f"  {CYAN}>>{RESET}"
                print(f"{arrow}  [{_vt(seg_start)} - {_vt(seg_end)}]  {BOLD}{dlabel:<14}{RESET}  {link}  {DIM}win {wnum}{RESET}")
                last_diar = dlabel
            else:
                print(f"      [{_vt(seg_start)} - {_vt(seg_end)}]  {DIM}{dlabel:<14}  {link}  win {wnum}{RESET}")

    print(f"""
{CYAN}{'=' * 65}{RESET}
""")

    yield None, final_output
