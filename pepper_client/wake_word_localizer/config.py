import os

# Pepper connection
PEPPER_IP = "192.168.0.52"
PEPPER_PORT = 9559
PI_LISTEN_IP = "192.168.0.50"
PI_LISTEN_PORT = 9559

# Audio
SAMPLE_RATE = 16000
CHANNELS = 1

# Vosk model path (download vosk-model-small-en-us-0.15 into this directory)
VOSK_MODEL_PATH = os.path.join(os.path.dirname(__file__), "model")

# Wake word variants (all lowercase)
WAKE_WORDS = {
    "ginny", "jeanie", "jenny", "genie", "jeannie", "jinny",
    "ginnie", "genny", "jenna", "gina", "jimmy", "gimmy",
}

# Fuzzy matching threshold (0-100). 75 catches Vosk transcription variants
# while rejecting unrelated words.
FUZZY_THRESHOLD = 75

# SRP-PHAT-HSDA sound localization
SOUND_LOC_EXPIRY = 5.0       # ignore sound locations older than this (seconds)
SRP_NFFT = 1024              # FFT window size (samples at 48kHz, ~21ms per frame)
SRP_COARSE_STEP = 10         # coarse grid resolution (degrees)
SRP_FINE_STEP = 1            # fine grid resolution (degrees)
SRP_REFINE_RADIUS = 15       # fine scan half-width around best coarse direction (degrees)
SRP_MIC_ACCEPTANCE = 150     # directivity model: mic acceptance half-angle (degrees)
SRP_MSW_DELTA = None         # MSW half-width in samples (None = auto from mic geometry)
SRP_ENABLE_MSW = False       # disable MSW to save CPU on Pi (minor accuracy trade-off)
SRP_MAX_FRAMES = 2           # max time-frames per buffer (2 frames = 36 FFTs)
SRP_MIN_CONFIDENCE = 1.5     # minimum peak-to-mean ratio to accept a direction

# Movement
MOVEMENT_COOLDOWN = 5.0      # seconds between consecutive wake-word movements
MOVEMENT_MODE = "body"           # "head", "body", or "head_and_body"
HEAD_SPEED = 0.15            # fraction of max speed for head interpolation
