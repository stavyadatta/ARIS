"""
Batch video processor — receives a video file, runs face + speaker recognition
on every frame/audio segment, annotates the video with labels, and returns it.
"""
import os
import cv2
import wave
import logging
import tempfile
import datetime
import subprocess
import numpy as np

logger = logging.getLogger("speaker_recognition")

CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _ts():
    return datetime.datetime.now().strftime("%H:%M:%S")


def _vt(seconds):
    """Format seconds as MM:SS."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


# Colors for different speakers (BGR for OpenCV)
SPEAKER_COLORS = [
    (0, 255, 0),    # green
    (255, 165, 0),  # orange
    (255, 0, 0),    # blue
    (0, 255, 255),  # yellow
    (255, 0, 255),  # magenta
    (0, 165, 255),  # orange-red
    (255, 255, 0),  # cyan
    (128, 0, 255),  # purple
]


def get_speaker_color(face_id, color_map):
    """Assign a consistent color to each speaker."""
    if face_id not in color_map:
        color_map[face_id] = SPEAKER_COLORS[len(color_map) % len(SPEAKER_COLORS)]
    return color_map[face_id]


def process_video(video_path, max_duration, face_recognition, speaker_recognition, diarization):
    """
    Process a video file end-to-end:
    1. Extract audio
    2. Run VAD to find speech segments
    3. For each speech segment: identify speaker (face + voice)
    4. Annotate video frames with speaker labels
    5. Write annotated video

    Args:
        video_path: path to input video
        max_duration: max seconds to process (0 = all)
        face_recognition: _FaceRecognition singleton
        speaker_recognition: _SpeakerRecognition singleton
        diarization: _Diarization singleton

    Yields:
        (progress_message, None) during processing
        (None, output_path) when done
    """
    logger.info(f"  {DIM}{_ts()}{RESET}  {CYAN}.. VIDEO{RESET}        Starting batch video processing")

    # Step 1: Extract audio
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_wav.close()

    ffmpeg_cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
    ]
    if max_duration > 0:
        ffmpeg_cmd.extend(["-t", str(max_duration)])
    ffmpeg_cmd.append(temp_wav.name)

    subprocess.run(ffmpeg_cmd, capture_output=True, check=True)

    with wave.open(temp_wav.name, 'rb') as wf:
        sample_rate = wf.getframerate()
        all_audio = wf.readframes(wf.getnframes())
    os.remove(temp_wav.name)

    audio_duration = (len(all_audio) // 2) / sample_rate
    logger.info(f"  {DIM}{_ts()}{RESET}  {GREEN}OK AUDIO{RESET}       {audio_duration:.1f}s extracted")

    # Step 2: Open video
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if max_duration > 0:
        max_frames = int(max_duration * video_fps)
    else:
        max_frames = total_frames

    logger.info(f"  {DIM}{_ts()}{RESET}  {GREEN}OK VIDEO{RESET}       {frame_w}x{frame_h} @ {video_fps:.0f}fps, processing {max_frames} frames")

    # Step 3: VAD — find speech segments with timestamps
    yield f"Extracting speech segments...", None

    ENERGY_THRESHOLD = 500
    CHUNK_SAMPLES = 1024
    CHUNK_BYTES = CHUNK_SAMPLES * 2
    max_silence = 8

    speech_segments = []  # [(start_sec, end_sec, audio_bytes), ...]
    segment_buf = bytearray()
    is_in_speech = False
    silence_count = 0
    speech_start = 0.0
    offset = 0

    while offset < len(all_audio):
        chunk = all_audio[offset:offset + CHUNK_BYTES]
        offset += CHUNK_BYTES
        if len(chunk) < CHUNK_BYTES:
            break

        audio_np = np.frombuffer(chunk, dtype=np.int16)
        rms = np.sqrt(np.mean(audio_np.astype(np.float32) ** 2))
        current_time = (offset // 2) / sample_rate

        if rms > ENERGY_THRESHOLD:
            silence_count = 0
            if not is_in_speech:
                is_in_speech = True
                segment_buf = bytearray()
                speech_start = current_time
            segment_buf.extend(chunk)
        else:
            if is_in_speech:
                segment_buf.extend(chunk)
                silence_count += 1
                if silence_count >= max_silence:
                    is_in_speech = False
                    silence_count = 0
                    duration = (len(segment_buf) // 2) / sample_rate
                    if duration >= 0.5:
                        speech_segments.append((speech_start, current_time, bytes(segment_buf)))

    # Flush remaining
    if is_in_speech:
        duration = (len(segment_buf) // 2) / sample_rate
        current_time = (offset // 2) / sample_rate
        if duration >= 0.5:
            speech_segments.append((speech_start, current_time, bytes(segment_buf)))

    logger.info(f"  {DIM}{_ts()}{RESET}  {GREEN}OK VAD{RESET}         {len(speech_segments)} speech segments found")

    # Step 4: Process each speech segment — face + voice recognition
    # Build a timeline: {time_range -> (face_id, speaker_status, confidence)}
    timeline = []  # [(start, end, face_id, status, confidence), ...]
    speaker_color_map = {}

    for i, (seg_start, seg_end, seg_audio) in enumerate(speech_segments):
        yield f"Identifying speaker {i+1}/{len(speech_segments)} [{_vt(seg_start)}]...", None

        # Get frame at segment midpoint for face recognition
        mid_time = (seg_start + seg_end) / 2
        frame_idx = int(mid_time * video_fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        current_face_id = None
        if ret:
            try:
                face_id, emb = face_recognition.recognize_face_no_enroll(frame, skip_validation=True)
                if face_id is not None:
                    current_face_id = face_id
                else:
                    current_face_id = face_recognition.enroll_face(emb, frame)
            except ValueError:
                pass

        # Speaker recognition
        try:
            result = speaker_recognition.identify_speaker(
                seg_audio, sample_rate, current_face_id=current_face_id
            )
            rid = result["face_id"] or "unknown"
            status = result["status"]
            conf = result["confidence"]
            is_new = result["is_new"]

            if is_new:
                label = f"NEW: {rid}"
            elif rid != "unknown":
                label = rid
            else:
                label = f"unknown ({status.split(':')[-1]})"

            timeline.append((seg_start, seg_end, rid, label, conf))
            get_speaker_color(rid, speaker_color_map)

            logger.info(
                f"  {DIM}{_ts()}{RESET}  {GREEN}  [{_vt(seg_start)}-{_vt(seg_end)}]{RESET}  "
                f"{BOLD}{label:<20}{RESET}  conf={conf:.2f}"
            )
        except Exception as e:
            logger.error(f"  {DIM}{_ts()}{RESET}  Error at {_vt(seg_start)}: {e}")
            timeline.append((seg_start, seg_end, "error", f"error: {e}", 0.0))

    logger.info(f"  {DIM}{_ts()}{RESET}  {GREEN}OK IDENTIFY{RESET}    All segments processed")

    # Step 5: Annotate video frames and write output
    yield f"Writing annotated video...", None

    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, video_fps, (frame_w, frame_h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_num = 0

    # Pre-compute: for each frame, which speaker is active?
    def get_active_speaker(t):
        for seg_start, seg_end, face_id, label, conf in timeline:
            if seg_start <= t <= seg_end:
                return face_id, label, conf
        return None, None, 0.0

    while frame_num < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        video_time = frame_num / video_fps

        if frame_num % 500 == 0:
            yield f"Annotating frame {frame_num}/{max_frames} [{_vt(video_time)}]...", None

        active_id, active_label, active_conf = get_active_speaker(video_time)

        # Draw timestamp
        ts_text = f"{_vt(video_time)}"
        cv2.putText(frame, ts_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw speaker label
        if active_id and active_id != "error":
            color = get_speaker_color(active_id, speaker_color_map)

            # Speaker bar at bottom
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, frame_h - 60), (frame_w, frame_h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            # Speaker name
            cv2.putText(frame, f"SPEAKING: {active_label}", (10, frame_h - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Confidence bar
            bar_w = int(active_conf * 200)
            cv2.rectangle(frame, (10, frame_h - 20), (10 + bar_w, frame_h - 10), color, -1)
            cv2.rectangle(frame, (10, frame_h - 20), (210, frame_h - 10), (100, 100, 100), 1)
            cv2.putText(frame, f"{active_conf:.2f}", (220, frame_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            # Speaking indicator dot
            cv2.circle(frame, (frame_w - 30, 30), 10, color, -1)
        else:
            # No one speaking — dim indicator
            cv2.circle(frame, (frame_w - 30, 30), 10, (80, 80, 80), -1)

        out.write(frame)

    out.release()
    cap.release()

    logger.info(f"  {DIM}{_ts()}{RESET}  {GREEN}{BOLD}OK DONE{RESET}        Annotated video: {output_path}")

    # Step 6: Mux audio back into annotated video
    final_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    mux_cmd = ["ffmpeg", "-y", "-i", output_path, "-i", video_path]
    if max_duration > 0:
        mux_cmd.extend(["-t", str(max_duration)])
    mux_cmd.extend([
        "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0",
        "-shortest", final_output
    ])

    try:
        subprocess.run(mux_cmd, capture_output=True, check=True)
        os.remove(output_path)
        logger.info(f"  {DIM}{_ts()}{RESET}  {GREEN}OK MUX{RESET}         Audio muxed into annotated video")
    except Exception as e:
        logger.warning(f"  {DIM}{_ts()}{RESET}  {YELLOW}!! MUX{RESET}         Failed to mux audio: {e}. Video-only output.")
        final_output = output_path

    # Print summary
    print(f"""
{CYAN}{'=' * 55}
   VIDEO PROCESSING SUMMARY
{'=' * 55}{RESET}

  Segments: {len(speech_segments)}
  Speakers: {len(speaker_color_map)}

{BOLD}  Timeline:{RESET}""")

    last_speaker = None
    for seg_start, seg_end, face_id, label, conf in timeline:
        if face_id != last_speaker:
            arrow = f"  {MAGENTA}>>{RESET}" if last_speaker else f"  {GREEN}>>{RESET}"
            print(f"{arrow}  [{_vt(seg_start)} - {_vt(seg_end)}]  {BOLD}{label:<20}{RESET}  conf={conf:.2f}")
            last_speaker = face_id
        else:
            print(f"      [{_vt(seg_start)} - {_vt(seg_end)}]  {DIM}{label:<20}  conf={conf:.2f}{RESET}")

    print(f"""
{CYAN}{'=' * 55}{RESET}
""")

    yield None, final_output
