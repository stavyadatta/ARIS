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
    "ginnie", "genny", "jenna", "gina",
}

# Fuzzy matching threshold (0-100). 75 catches Vosk transcription variants
# while rejecting unrelated words.
FUZZY_THRESHOLD = 75

# Sound localization
SOUND_POLL_INTERVAL = 0.0   # seconds between ALSoundLocalization polls
SOUND_LOC_EXPIRY = 3.0      # ignore sound locations older than this (seconds)
MIN_CONFIDENCE = 0.0         # minimum confidence to accept a sound direction
SOUND_SENSITIVITY = 0.8      # ALSoundLocalization sensitivity parameter

# Movement
MOVEMENT_COOLDOWN = 5.0      # seconds between consecutive wake-word movements
MOVEMENT_MODE = "head_and_body"  # "head", "body", or "head_and_body"
HEAD_SPEED = 0.15            # fraction of max speed for head interpolation
