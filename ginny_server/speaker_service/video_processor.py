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
import time
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
     voice_slice, diar_slice, num_diar_speakers,
     summary_text, summary_until_frame, out_path) = args

    # Each worker is single-threaded; we parallelize at the process level.
    cv2.setNumThreads(1)

    cap = cv2.VideoCapture(src_path)

    # *** FRAME-ACCURATE SEEK (do NOT use cap.set(CAP_PROP_POS_FRAMES, N)) ***
    # cv2's POS_FRAMES seek is NOT precise for H.264 — it lands on the nearest
    # keyframe at or BEFORE the requested frame. For a typical 2-5s keyframe
    # interval, worker K that asks for frame `start_frame` can actually land
    # 60-150 frames early. Each worker then reads a different absolute range
    # than it was assigned, causing frame duplication + gaps when ffmpeg
    # concats the chunks, which produces audio/video desync in the final
    # muxed output.
    #
    # Fix: read-and-discard from frame 0 up to start_frame using cap.grab(),
    # which decodes but doesn't copy the frame (cheap). Every worker starts
    # at EXACTLY the right absolute frame.
    for _ in range(start_frame):
        if not cap.grab():
            break

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

        # ---- TOP-LEFT SUMMARY BANNER (first 60s only) ----
        # Absolute frame index for this iteration = start_frame + i
        if (start_frame + i) < summary_until_frame:
            # Two lines: header + the counts
            cv2.putText(frame, "PIPELINE", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
            cv2.putText(frame, summary_text, (10, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

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
    # Module-level import for the per-frame gate helpers and constants.
    # Hoisted out of the per-window loop (perf: avoid repeat import overhead).
    from ginny_server.core_api.speaker_recognition import speaker_recognition as sr_mod
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
    MIN_CLUSTER_AUDIO = 0.0  # replaced by per-frame enrollment gate (kept for backward compat)
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

        # Step 1: Diarize this window AND get soft posteriors for the gate
        try:
            diar_segments, soft_data, sw, class_map, frame_rate_hz = \
                diarization.diarize_with_posteriors(window_audio, sample_rate)
        except Exception as e:
            logger.error(f"  {DIM}{_ts()}{RESET}  {YELLOW}!! DIAR{RESET}        Window {win_idx+1} failed: {e}")
            continue

        # Group segments by diarization speaker label.
        # Track per-segment byte lengths (parallel to "segments") for the
        # per-frame gate's byte-aligned timeline computation.
        clusters = {}  # diar_label → {"audio": bytearray, "segments": [(start, end)], "segment_byte_lens": [int]}
        for seg in diar_segments:
            label = seg["speaker"]
            if label not in clusters:
                clusters[label] = {"audio": bytearray(), "segments": [], "segment_byte_lens": []}
            clusters[label]["audio"].extend(seg["audio"])
            clusters[label]["segments"].append((seg["start"] + win_start, seg["end"] + win_start))
            clusters[label]["segment_byte_lens"].append(len(seg["audio"]))

        cluster_summary = ', '.join(
            "{}({:.1f}s)".format(k, len(v["audio"]) // 2 / sample_rate)
            for k, v in clusters.items()
        )
        logger.info(
            f"  {DIM}{_ts()}{RESET}  {GREEN}  WIN {win_idx+1:<3}{RESET}     "
            f"[{_vt(win_start)}-{_vt(win_end)}]  "
            f"{len(clusters)} speakers: {cluster_summary}"
        )

        # Per-frame gate constants (sr_mod imported once at function start)
        samples_per_frame = round(sample_rate / frame_rate_hz)
        bytes_per_frame = samples_per_frame * 2

        # Step 2: For each cluster, apply the three-tier per-frame gate
        for diar_label, cluster in clusters.items():
            cluster_audio = bytes(cluster["audio"])
            cluster_segments = cluster["segments"]
            cluster_segment_byte_lens = cluster["segment_byte_lens"]
            cluster_dur = (len(cluster_audio) // 2) / sample_rate

            _get_color(diar_label, diar_color_map, DIAR_COLORS)

            # Hard invariant check (not assert): byte_lens MUST sum to the
            # concatenated audio length. Violation would silently misalign
            # the per-frame gate and could poison the voice DB.
            if sum(cluster_segment_byte_lens) != len(cluster_audio):
                raise RuntimeError(
                    f"byte_lens mismatch: {sum(cluster_segment_byte_lens)} "
                    f"vs {len(cluster_audio)} for {diar_label}"
                )

            _t0 = time.perf_counter()
            try:
                alone_timeline = sr_mod._compute_alone_timeline(
                    cluster_segments, cluster_segment_byte_lens,
                    soft_data, sw, class_map,
                    frame_rate_hz, sample_rate,
                    window_offset_s=win_start,
                )
            except Exception as e:
                logger.error(f"  {DIM}{_ts()}{RESET}  {YELLOW}    {diar_label}: alone_timeline failed: {e}{RESET}")
                for seg_start, seg_end in cluster_segments:
                    diar_timeline.append((seg_start, seg_end, diar_label, "?", win_idx + 1))
                continue

            # === Strict pass: enrollment threshold (0.8) ===
            enroll_audio, enroll_frames, total_clean_strict = sr_mod._lccs_from_timeline(
                alone_timeline,
                sr_mod.PER_FRAME_ALONE_THRESHOLD_ENROLL,
                cluster_audio, bytes_per_frame
            )
            enroll_lccs_s = enroll_frames / frame_rate_hz
            total_clean_strict_s = total_clean_strict / frame_rate_hz
            frag_strict = (total_clean_strict / enroll_frames) if enroll_frames > 0 else float('inf')

            if enroll_lccs_s >= sr_mod.MIN_CLEAN_DURATION_ENROLL:
                # === FULL tier ===
                try:
                    embedding = speaker_recognition.extract_embedding(enroll_audio, sample_rate)
                    result = speaker_recognition.match_or_buffer(
                        embedding, enroll_audio, sample_rate, min_samples=MIN_ENROLL_SAMPLES
                    )
                except Exception as e:
                    logger.error(f"  {DIM}{_ts()}{RESET}  {YELLOW}    FULL {diar_label}: embed/match failed: {e}{RESET}")
                    for seg_start, seg_end in cluster_segments:
                        diar_timeline.append((seg_start, seg_end, diar_label, "?", win_idx + 1))
                    continue

                voice_id = result["voice_id"] or result.get("pending_id", "?")
                confidence = result["confidence"]
                if result["voice_id"]:
                    _get_color(result["voice_id"], voice_color_map, VOICE_COLORS)

                _filter_ms = (time.perf_counter() - _t0) * 1000
                logger.info(
                    f"  {DIM}{_ts()}{RESET}  {GREEN}    FULL {diar_label}{RESET}: "
                    f"raw={cluster_dur:.1f}s lccs@0.8={enroll_lccs_s:.1f}s "
                    f"tot_clean={total_clean_strict_s:.1f}s (frag={frag_strict:.2f}) "
                    f"filter={_filter_ms:.1f}ms → {BOLD}{voice_id}{RESET}  "
                    f"conf={confidence:.2f}  {DIM}{result['status']}{RESET}"
                )
                for seg_start, seg_end in cluster_segments:
                    diar_timeline.append((seg_start, seg_end, diar_label, voice_id, win_idx + 1))
                    if result["voice_id"]:
                        voice_timeline.append((seg_start, seg_end, result["voice_id"], confidence, result["status"]))
                continue

            # === Permissive pass: quick-match only (0.6) ===
            quick_audio, quick_frames, total_clean_perm = sr_mod._lccs_from_timeline(
                alone_timeline,
                sr_mod.PER_FRAME_ALONE_THRESHOLD_QUICK,
                cluster_audio, bytes_per_frame
            )
            quick_lccs_s = quick_frames / frame_rate_hz

            if quick_lccs_s < sr_mod.MIN_CLEAN_DURATION_QUICKMATCH:
                # === SKIP tier ===
                _filter_ms = (time.perf_counter() - _t0) * 1000
                logger.info(
                    f"  {DIM}{_ts()}{RESET}  {DIM}    SKIP {diar_label}: "
                    f"raw={cluster_dur:.1f}s lccs@0.8={enroll_lccs_s:.1f}s "
                    f"lccs@0.6={quick_lccs_s:.1f}s filter={_filter_ms:.1f}ms TOO CONTAMINATED{RESET}"
                )
                for seg_start, seg_end in cluster_segments:
                    diar_timeline.append((seg_start, seg_end, diar_label, "?", win_idx + 1))
                continue

            # === QUICK-ONLY tier ===
            try:
                embedding = speaker_recognition.extract_embedding(quick_audio, sample_rate)
                matched_id, match_conf = speaker_recognition._match_voice(embedding)
            except Exception as e:
                logger.error(f"  {DIM}{_ts()}{RESET}  {YELLOW}    QUICK {diar_label}: embed/match failed: {e}{RESET}")
                for seg_start, seg_end in cluster_segments:
                    diar_timeline.append((seg_start, seg_end, diar_label, "?", win_idx + 1))
                continue

            _filter_ms = (time.perf_counter() - _t0) * 1000
            if matched_id is not None:
                _get_color(matched_id, voice_color_map, VOICE_COLORS)
                logger.info(
                    f"  {DIM}{_ts()}{RESET}  {CYAN}    QUICK-ONLY {diar_label}{RESET}: "
                    f"raw={cluster_dur:.1f}s lccs@0.6={quick_lccs_s:.1f}s "
                    f"filter={_filter_ms:.1f}ms → {BOLD}{matched_id}{RESET}  "
                    f"conf={match_conf:.2f}  {DIM}quick-matched{RESET}"
                )
                for seg_start, seg_end in cluster_segments:
                    voice_timeline.append((seg_start, seg_end, matched_id, match_conf, "quick-matched"))
                    diar_timeline.append((seg_start, seg_end, diar_label, matched_id, win_idx + 1))
            else:
                logger.info(
                    f"  {DIM}{_ts()}{RESET}  {DIM}    QUICK-ONLY {diar_label}: "
                    f"raw={cluster_dur:.1f}s lccs@0.6={quick_lccs_s:.1f}s "
                    f"filter={_filter_ms:.1f}ms NO MATCH, DROPPED (no enrollment){RESET}"
                )
                for seg_start, seg_end in cluster_segments:
                    diar_timeline.append((seg_start, seg_end, diar_label, "?", win_idx + 1))

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

    # Summary banner for the first 60 seconds of the annotated video.
    # Mirrors the `OK PIPELINE` log line.
    summary_text = f"{num_voices} enrolled voices"
    SUMMARY_BANNER_DURATION_S = 60.0
    summary_until_frame = int(SUMMARY_BANNER_DURATION_S * video_fps)

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
            summary_text,
            summary_until_frame,
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

    # *** A/V DESYNC GUARD ***
    # If total_written != total_frames, the concatenated video will have a
    # different duration than the source audio, and ffmpeg's -shortest flag
    # will clip whichever stream ends first. Warn loudly so the operator
    # knows the annotation pipeline dropped/duplicated frames.
    if total_written != total_frames:
        delta = total_written - total_frames
        drift_s = delta / video_fps
        logger.warning(
            f"  {DIM}{_ts()}{RESET}  {YELLOW}!! FRAME COUNT{RESET}  "
            f"written={total_written} expected={total_frames} "
            f"delta={delta:+d} frames ({drift_s:+.2f}s). "
            f"A/V desync may be visible in the final muxed video."
        )
        # Per-chunk breakdown for debugging
        for idx, (path, nw) in enumerate(results):
            expected_n = chunk_args[idx][2] - chunk_args[idx][1]  # end - start
            if nw != expected_n:
                logger.warning(
                    f"  {DIM}{_ts()}{RESET}  {YELLOW}   chunk {idx}{RESET}: "
                    f"wrote {nw}/{expected_n} frames "
                    f"(range [{chunk_args[idx][1]}, {chunk_args[idx][2]}))"
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
