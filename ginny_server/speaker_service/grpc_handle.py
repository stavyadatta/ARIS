import cv2
import time
import logging
import datetime
import numpy as np
import traceback
import grpc_communication.grpc_pb2 as pb2
import grpc_communication.grpc_pb2_grpc as pb2_grpc

logger = logging.getLogger("speaker_recognition")

_CYAN = "\033[96m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_MAGENTA = "\033[95m"
_BLUE = "\033[94m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_RESET = "\033[0m"

def _log(icon, label, detail, color=_RESET):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    logger.info(f"  {_DIM}{ts}{_RESET}  {color}{icon} {label:<14}{_RESET} {detail}")


class SpeakerRecognitionManager(pb2_grpc.SpeakerRecognitionServiceServicer):
    """
    gRPC servicer for combined Face + Speaker Recognition.

    For each incoming segment:
    1. If image_data is present → run face recognition → get face_id
    2. Run speaker recognition with that face_id context
    3. Return combined result (face_id + voice confidence + status)

    Dual-path audio:
    - Short path: immediate single-speaker embedding per segment
    - Long path: buffered diarization when >=10s accumulated
    """

    def __init__(self):
        super().__init__()
        from core_api import SpeakerRecognition, Diarization, FaceRecognition
        self.speaker_recognition = SpeakerRecognition
        self.diarization = Diarization
        self.face_recognition = FaceRecognition
        _log("OK", "SERVICER", "Face + Voice pipeline ready", _GREEN)

    def _extract_face_id(self, image_data, image_width, image_height):
        """
        Run face recognition on the provided image frame.
        Returns face_id (str) or None if no face detected.
        """
        if not image_data:
            return None

        try:
            # Decode JPEG bytes → numpy array
            img_array = np.frombuffer(image_data, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if img is None:
                logger.warning("Failed to decode image frame")
                return None

            # Use face recognition — recognize without enrolling first
            face_id, embedding = self.face_recognition.recognize_face_no_enroll(
                img, skip_validation=True
            )

            if face_id is not None:
                _log("^^", "FACE FOUND", f"{_BOLD}{face_id}{_RESET}", _BLUE)
                return face_id
            else:
                new_id = self.face_recognition.enroll_face(embedding, img)
                _log("**", "FACE NEW", f"{_BOLD}{new_id}{_RESET}  (enrolled)", _MAGENTA)
                return new_id

        except ValueError as e:
            _log("--", "NO FACE", str(e), _DIM)
            return None
        except Exception as e:
            _log("!!", "FACE ERROR", str(e), _RED)
            return None

    def _render_session_video(self, session_id, frames, full_audio, sample_rate,
                              voice_results, diar_results):
        """Render an annotated video from a completed real-time session."""
        import os
        import wave
        import subprocess
        import tempfile
        from sklearn.metrics.pairwise import cosine_similarity

        output_dir = "/workspace/database/recordings"
        os.makedirs(output_dir, exist_ok=True)

        _log(">>", "RENDER VIDEO", f"session={session_id}, {len(frames)} frames", _CYAN)

        if not frames:
            return

        # Determine fps from frame timestamps
        if len(frames) >= 2:
            total_time = frames[-1][0] - frames[0][0]
            fps = len(frames) / max(total_time, 0.1)
            fps = min(max(fps, 1.0), 30.0)  # clamp to 1-30
        else:
            fps = 10.0

        # Decode first frame to get dimensions
        first_jpeg = frames[0][1]
        first_img = cv2.imdecode(np.frombuffer(first_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if first_img is None:
            _log("!!", "RENDER", "Could not decode first frame", _RED)
            return
        frame_h, frame_w = first_img.shape[:2]

        # Voice/diar color maps
        voice_colors = {}
        diar_colors = {}
        VCOLORS = [(0,255,0),(0,200,100),(0,255,200),(100,255,0),(0,180,0)]
        DCOLORS = [(0,200,255),(255,100,200),(100,255,255),(200,100,255)]
        FCOLORS = [(255,100,50),(200,50,50),(255,150,0),(180,80,120)]
        face_colors = {}

        def _gc(identity, cmap, palette):
            if identity not in cmap:
                cmap[identity] = palette[len(cmap) % len(palette)]
            return cmap[identity]

        def _vt_local(s):
            m, sec = divmod(int(s), 60)
            return f"{m:02d}:{sec:02d}"

        def get_voice_at(t):
            for s, e, vid, conf, status in voice_results:
                if s <= t <= e:
                    return vid, conf
            return None, 0.0

        def get_diar_at(t):
            active = []
            for s, e, dlabel, vid in diar_results:
                if s <= t <= e:
                    active.append((dlabel, vid))
            return active

        # Write video frames
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video.close()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video.name, fourcc, fps, (frame_w, frame_h))

        num_diar_speakers = len(set(dl for _, _, dl, _ in diar_results))

        for i, (ts, jpeg, w, h, face_id) in enumerate(frames):
            frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # Resize if needed
            if frame.shape[1] != frame_w or frame.shape[0] != frame_h:
                frame = cv2.resize(frame, (frame_w, frame_h))

            # Face label (from recorded face_id)
            if face_id and face_id != "unknown":
                fc = _gc(face_id, face_colors, FCOLORS)
                cv2.putText(frame, f"FACE: {face_id}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, fc, 2)

            # Timestamp
            cv2.putText(frame, _vt_local(ts), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Voice bar (bottom)
            active_voice, vconf = get_voice_at(ts)
            if active_voice:
                vc = _gc(active_voice, voice_colors, VCOLORS)
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, frame_h - 70), (frame_w, frame_h), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                cv2.putText(frame, f"VOICE: {active_voice}", (10, frame_h - 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, vc, 2)
                bar_w = int(vconf * 200)
                cv2.rectangle(frame, (10, frame_h-25), (10+bar_w, frame_h-15), vc, -1)
                cv2.rectangle(frame, (10, frame_h-25), (210, frame_h-15), (100,100,100), 1)
                cv2.putText(frame, f"{vconf:.2f}", (220, frame_h-15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
                cv2.circle(frame, (frame_w-30, 30), 10, vc, -1)
            else:
                cv2.circle(frame, (frame_w-30, 30), 10, (80,80,80), -1)

            # Diarization panel (top-right)
            cv2.putText(frame, f"DIAR: {num_diar_speakers} spk", (frame_w-180, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
            active_diar = get_diar_at(ts)
            dy = 55
            for dlabel, vid in active_diar:
                dc = _gc(dlabel, diar_colors, DCOLORS)
                cv2.circle(frame, (frame_w-175, dy-5), 6, dc, -1)
                txt = dlabel
                if vid and not vid.startswith("pending"):
                    txt += f" = {vid}"
                cv2.putText(frame, txt, (frame_w-163, dy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, dc, 1)
                dy += 20

            out.write(frame)

        out.release()

        # Mux audio
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_wav.close()
        import wave as wave_mod
        with wave_mod.open(temp_wav.name, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(bytes(full_audio))

        final_path = os.path.join(output_dir, f"session_{session_id}.mp4")
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", temp_video.name, "-i", temp_wav.name,
                "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0",
                "-shortest", final_path
            ], capture_output=True, check=True)
            os.remove(temp_video.name)
        except Exception:
            # Fallback: video only
            import shutil
            shutil.move(temp_video.name, final_path)

        os.remove(temp_wav.name)

        _log("**", "VIDEO SAVED",
             f"{_BOLD}{final_path}{_RESET}  "
             f"({os.path.getsize(final_path) / 1024 / 1024:.1f}MB, "
             f"{len(frames)} frames, {len(voice_results)} voice segments)",
             _GREEN)

    def RecognizeSpeakers(self, request_iterator, context):
        """
        Bidirectional streaming RPC with sliding-window diarization + multi-sample ReID.

        Audio accumulates in a per-session buffer. When buffer reaches WINDOW_SIZE (30s):
        1. Run DiariZen on the buffer → speaker clusters
        2. Concat audio per cluster → ERes2NetV2 embedding
        3. match_or_buffer against voice DB → voice_id or pending enrollment
        4. Yield SpeakerResult for each cluster's segments
        5. Shift buffer forward (keep WINDOW_OVERLAP for continuity)

        Face recognition runs immediately on each image frame (decoupled).
        """
        WINDOW_SIZE = 30.0      # seconds
        WINDOW_OVERLAP = 10.0   # seconds
        MIN_CLUSTER_AUDIO = 0.0 # replaced by per-frame enrollment gate
        MIN_ENROLL_SAMPLES = 3

        # Hoisted import for the per-frame gate helpers and constants
        # (avoid repeated import overhead inside the per-window loop).
        from ginny_server.core_api.speaker_recognition import speaker_recognition as sr_mod

        session_id = None
        current_face_id = None
        sample_rate = 16000

        # Per-session audio buffer (timeline-aware)
        audio_buffer = bytearray()
        buffer_start_time = 0.0  # absolute time of buffer start
        buffer_end_time = 0.0    # expected end time of audio in buffer
        first_segment_seen = False
        window_count = 0

        # Recording: collect frames + audio + results for post-session video
        recorded_full_audio = bytearray()   # all audio received
        recorded_frames = []                 # [(timestamp, jpeg_bytes, w, h, face_id)]
        recorded_voice_results = []          # [(seg_start, seg_end, voice_id, confidence, status)]
        recorded_diar_results = []           # [(seg_start, seg_end, diar_label, voice_id)]
        session_start_time = None

        # Init enrollment buffer for this session
        self.speaker_recognition.init_enrollment_buffer()

        try:
            for request in request_iterator:
                audio_data = request.audio_data
                sample_rate = request.sample_rate or 16000
                session_id = request.session_id
                video_ts = request.video_timestamp

                if session_start_time is None:
                    session_start_time = __import__('time').time()

                # === FACE RECOGNITION (immediate, every frame) ===
                if request.image_data:
                    detected_face = self._extract_face_id(
                        request.image_data,
                        request.image_width,
                        request.image_height
                    )
                    if detected_face is not None:
                        current_face_id = detected_face

                    # Record frame for post-session video
                    elapsed = __import__('time').time() - session_start_time
                    recorded_frames.append((
                        elapsed, request.image_data,
                        request.image_width, request.image_height,
                        current_face_id or "unknown"
                    ))

                if current_face_id is None and request.face_id:
                    current_face_id = request.face_id

                # === QUICK MATCH: per-segment voice matching (immediate) ===
                seg_duration = (len(audio_data) // 2) / sample_rate
                if seg_duration >= 0.5:
                    try:
                        seg_embedding = self.speaker_recognition.extract_embedding(
                            audio_data, sample_rate
                        )
                        matched_id, match_conf = self.speaker_recognition._match_voice(
                            seg_embedding
                        )
                        seg_start = request.segment_start_time
                        seg_end = seg_start + seg_duration

                        if matched_id is not None:
                            _log("~~", "QUICK MATCH",
                                 f"{_BOLD}{matched_id}{_RESET}  conf={match_conf:.2f}  "
                                 f"seg={seg_duration:.1f}s", _GREEN)
                            yield pb2.SpeakerResult(
                                speaker_id=matched_id,
                                confidence=match_conf,
                                segment_start_time=seg_start,
                                segment_duration=seg_duration,
                                is_new_speaker=False,
                                session_id=session_id,
                                is_correction=False,
                                status=f"quick:{matched_id}",
                                video_timestamp=video_ts
                            )
                            recorded_voice_results.append((
                                seg_start, seg_end, matched_id, match_conf,
                                f"quick:{matched_id}"
                            ))
                        else:
                            _log("xx", "QUICK MISS",
                                 f"best={match_conf:.2f}  seg={seg_duration:.1f}s  "
                                 f"(will try diarization)", _YELLOW)
                    except Exception as e:
                        _log("!!", "QUICK ERR", str(e), _RED)

                # === ACCUMULATE AUDIO (timeline-aware for diarization) ===
                seg_start_t = request.segment_start_time
                seg_dur_t = (len(audio_data) // 2) / sample_rate

                if not first_segment_seen:
                    # First segment: set buffer origin
                    buffer_start_time = seg_start_t
                    buffer_end_time = seg_start_t
                    first_segment_seen = True

                # Insert silence for gaps between segments (max 5s to avoid bloat)
                gap = seg_start_t - buffer_end_time
                if gap > 0.05:  # ignore tiny gaps (<50ms)
                    silence_duration = min(gap, 5.0)
                    silence_bytes = int(silence_duration * sample_rate * 2)
                    audio_buffer.extend(b'\x00' * silence_bytes)
                    if gap > 5.0:
                        _log("..", "GAP CLAMP",
                             f"gap={gap:.1f}s clamped to 5.0s silence", _DIM)

                audio_buffer.extend(audio_data)
                recorded_full_audio.extend(audio_data)
                buffer_end_time = seg_start_t + seg_dur_t
                buffer_duration = (len(audio_buffer) // 2) / sample_rate

                # === WINDOW READY? Run diarization + ReID ===
                if buffer_duration >= WINDOW_SIZE:
                    window_count += 1
                    # Ensure even byte count (int16 = 2 bytes per sample)
                    if len(audio_buffer) % 2 != 0:
                        audio_buffer = audio_buffer[:-1]
                    window_audio = bytes(audio_buffer)
                    win_start = buffer_start_time
                    win_end = win_start + buffer_duration

                    _log(">>", f"WINDOW {window_count}",
                         f"[{win_start:.0f}s - {win_end:.0f}s]  {buffer_duration:.1f}s audio",
                         _CYAN)

                    # Run diarization on the window AND get soft posteriors for the gate
                    try:
                        diar_segments, soft_data, sw, class_map, frame_rate_hz = \
                            self.diarization.diarize_with_posteriors(window_audio, sample_rate)
                    except Exception as e:
                        _log("!!", "DIAR ERROR", str(e), _RED)
                        # Shift buffer and continue
                        shift_bytes = int(WINDOW_SIZE - WINDOW_OVERLAP) * sample_rate * 2
                        audio_buffer = audio_buffer[shift_bytes:]
                        buffer_start_time += (WINDOW_SIZE - WINDOW_OVERLAP)
                        continue

                    # Group diarized segments by speaker label.
                    # Track per-segment byte lengths for the per-frame gate.
                    clusters = {}
                    for seg in diar_segments:
                        label = seg["speaker"]
                        if label not in clusters:
                            clusters[label] = {"audio": bytearray(), "segments": [], "segment_byte_lens": []}
                        clusters[label]["audio"].extend(seg["audio"])
                        clusters[label]["segments"].append(
                            (seg["start"] + win_start, seg["end"] + win_start)
                        )
                        clusters[label]["segment_byte_lens"].append(len(seg["audio"]))

                    # DIAGNOSTIC: raw segment count + per-cluster breakdown
                    n_raw_segs = len(diar_segments)
                    _log("..", "CLUSTERS",
                         f"{len(clusters)} labels / {n_raw_segs} raw segs in window {window_count}",
                         _GREEN)
                    for _diag_label, _diag_cluster in clusters.items():
                        _diag_audio_dur = (len(_diag_cluster["audio"]) // 2) / sample_rate
                        _diag_n_segs = len(_diag_cluster["segments"])
                        _diag_span_start = min(s for s, _ in _diag_cluster["segments"])
                        _diag_span_end = max(e for _, e in _diag_cluster["segments"])
                        _diag_span = _diag_span_end - _diag_span_start
                        _log("  ", f"  {_diag_label}",
                             f"{_diag_n_segs} segs, {_diag_audio_dur:.1f}s audio, "
                             f"span {_diag_span:.1f}s [{_diag_span_start:.1f}-{_diag_span_end:.1f}]",
                             _DIM)

                    # Per-frame gate constants (sr_mod imported once at method start)
                    samples_per_frame = round(sample_rate / frame_rate_hz)
                    bytes_per_frame = samples_per_frame * 2

                    # For each cluster: three-tier per-frame enrollment gate
                    for diar_label, cluster in clusters.items():
                        cluster_audio = bytes(cluster["audio"])
                        cluster_segments = cluster["segments"]
                        cluster_segment_byte_lens = cluster["segment_byte_lens"]
                        cluster_dur = (len(cluster_audio) // 2) / sample_rate

                        # Hard invariant check (not assert): byte_lens MUST sum to
                        # cluster_audio length. Violation would silently misalign
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
                            _log("!!", "TIMELINE ERR", f"{diar_label}: {e}", _RED)
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
                                embedding = self.speaker_recognition.extract_embedding(
                                    enroll_audio, sample_rate
                                )
                                result = self.speaker_recognition.match_or_buffer(
                                    embedding, enroll_audio, sample_rate,
                                    min_samples=MIN_ENROLL_SAMPLES
                                )
                            except Exception as e:
                                _log("!!", "FULL ERR", f"{diar_label}: {e}", _RED)
                                continue

                            voice_id = result["voice_id"] or result.get("pending_id", "")
                            confidence = result["confidence"]
                            _filter_ms = (time.perf_counter() - _t0) * 1000

                            for seg_start, seg_end in cluster_segments:
                                yield pb2.SpeakerResult(
                                    speaker_id=voice_id,
                                    confidence=confidence,
                                    segment_start_time=seg_start,
                                    segment_duration=seg_end - seg_start,
                                    is_new_speaker=result["is_new"],
                                    session_id=session_id,
                                    is_correction=False,
                                    status=result["status"],
                                    video_timestamp=video_ts
                                )
                                recorded_voice_results.append((
                                    seg_start, seg_end, voice_id, confidence, result["status"]
                                ))
                                recorded_diar_results.append((
                                    seg_start, seg_end, diar_label, voice_id
                                ))

                            _log("..", f"FULL {diar_label}",
                                 f"raw={cluster_dur:.1f}s lccs@0.8={enroll_lccs_s:.1f}s "
                                 f"tot_clean={total_clean_strict_s:.1f}s (frag={frag_strict:.2f}) "
                                 f"filter={_filter_ms:.1f}ms → {_BOLD}{voice_id}{_RESET} "
                                 f"conf={confidence:.2f} {_DIM}{result['status']}{_RESET}",
                                 _GREEN)
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
                            _log("--", f"SKIP {diar_label}",
                                 f"raw={cluster_dur:.1f}s lccs@0.8={enroll_lccs_s:.1f}s "
                                 f"lccs@0.6={quick_lccs_s:.1f}s filter={_filter_ms:.1f}ms TOO CONTAMINATED",
                                 _DIM)
                            continue

                        # === QUICK-ONLY tier ===
                        try:
                            embedding = self.speaker_recognition.extract_embedding(
                                quick_audio, sample_rate
                            )
                            matched_id, match_conf = self.speaker_recognition._match_voice(embedding)
                        except Exception as e:
                            _log("!!", "QUICK ERR", f"{diar_label}: {e}", _RED)
                            continue

                        _filter_ms = (time.perf_counter() - _t0) * 1000
                        if matched_id is not None:
                            for seg_start, seg_end in cluster_segments:
                                yield pb2.SpeakerResult(
                                    speaker_id=matched_id,
                                    confidence=match_conf,
                                    segment_start_time=seg_start,
                                    segment_duration=seg_end - seg_start,
                                    is_new_speaker=False,
                                    session_id=session_id,
                                    is_correction=False,
                                    status="quick-matched",
                                    video_timestamp=video_ts
                                )
                                recorded_voice_results.append((
                                    seg_start, seg_end, matched_id, match_conf, "quick-matched"
                                ))
                                recorded_diar_results.append((
                                    seg_start, seg_end, diar_label, matched_id
                                ))
                            _log("..", f"QUICK-ONLY {diar_label}",
                                 f"raw={cluster_dur:.1f}s lccs@0.6={quick_lccs_s:.1f}s "
                                 f"filter={_filter_ms:.1f}ms → {_BOLD}{matched_id}{_RESET} "
                                 f"conf={match_conf:.2f} {_DIM}quick-matched{_RESET}",
                                 _CYAN)
                        else:
                            _log("--", f"QUICK-ONLY {diar_label}",
                                 f"raw={cluster_dur:.1f}s lccs@0.6={quick_lccs_s:.1f}s "
                                 f"filter={_filter_ms:.1f}ms NO MATCH, DROPPED (no enrollment)",
                                 _DIM)

                    # Shift buffer: keep last WINDOW_OVERLAP seconds
                    shift_bytes = int((WINDOW_SIZE - WINDOW_OVERLAP) * sample_rate * 2)
                    audio_buffer = audio_buffer[shift_bytes:]
                    buffer_start_time += (WINDOW_SIZE - WINDOW_OVERLAP)

        except Exception as e:
            _log("!!", "FATAL ERROR", str(e), _RED)
            traceback.print_exc()
        finally:
            # Flush pending enrollments at end of session
            if hasattr(self.speaker_recognition, '_pending'):
                flushed = self.speaker_recognition.flush_all_pending(sample_rate, min_samples=2)
                for r in flushed:
                    _log("**", "FLUSH", f"Enrolled {r['voice_id']} at session end", _GREEN)

            if session_id is not None:
                self.diarization.clear_session(session_id)
            _log("--", "SESSION END", f"{session_id}  windows={window_count}", _DIM)

            # === RENDER POST-SESSION VIDEO ===
            if recorded_frames and len(recorded_frames) > 10:
                try:
                    self._render_session_video(
                        session_id or "unknown",
                        recorded_frames, recorded_full_audio, sample_rate,
                        recorded_voice_results, recorded_diar_results
                    )
                except Exception as e:
                    _log("!!", "VIDEO ERROR", f"Failed to render session video: {e}", _RED)
                    traceback.print_exc()
            else:
                _log("--", "NO VIDEO", f"Only {len(recorded_frames)} frames, skipping video render", _DIM)

    def ProcessVideo(self, request_iterator, context):
        """
        Batch video processing RPC.
        Client uploads video → server annotates with face+speaker labels → sends back.
        """
        import os
        import tempfile
        from speaker_service.video_processor import process_video

        # Step 1: Receive video file chunks
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        max_duration = 0.0
        filename = "input.mp4"

        for chunk in request_iterator:
            if chunk.filename:
                filename = chunk.filename
            if chunk.max_duration_seconds > 0:
                max_duration = chunk.max_duration_seconds
            temp_input.write(chunk.data)

        temp_input.close()
        _log("<<", "VIDEO RECV", f"{filename} ({os.path.getsize(temp_input.name) / 1024 / 1024:.1f}MB), max_dur={max_duration}s", _CYAN)

        # Step 2: Process video
        output_path = None
        try:
            for progress, result_path in process_video(
                temp_input.name, max_duration,
                self.speaker_recognition, self.diarization
            ):
                if progress:
                    yield pb2.VideoDownloadChunk(
                        data=b"", filename="", is_last=False, progress=progress
                    )
                if result_path:
                    output_path = result_path
        except Exception as e:
            _log("!!", "VIDEO ERROR", str(e), _RED)
            traceback.print_exc()
            os.remove(temp_input.name)
            return

        os.remove(temp_input.name)

        if not output_path or not os.path.exists(output_path):
            _log("!!", "VIDEO ERROR", "No output produced", _RED)
            return

        # Step 3: Stream annotated video back to client
        out_filename = f"annotated_{filename}"
        file_size = os.path.getsize(output_path)
        _log(">>", "VIDEO SEND", f"{out_filename} ({file_size / 1024 / 1024:.1f}MB)", _GREEN)

        CHUNK_SIZE = 1024 * 1024  # 1MB chunks
        with open(output_path, 'rb') as f:
            first = True
            while True:
                data = f.read(CHUNK_SIZE)
                if not data:
                    break
                yield pb2.VideoDownloadChunk(
                    data=data,
                    filename=out_filename if first else "",
                    is_last=False,
                    progress=""
                )
                first = False

        yield pb2.VideoDownloadChunk(data=b"", filename="", is_last=True, progress="Done")
        os.remove(output_path)
        _log("OK", "VIDEO DONE", f"Sent {out_filename} to client", _GREEN)
