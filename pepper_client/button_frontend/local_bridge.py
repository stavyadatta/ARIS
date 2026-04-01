import time
import requests
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from .button import Buttons_vals, Mic_UI, FaceArea_UI, Telemetry, Volume

# Set CLOUD_URL env var, or use default custom domain
CLOUD_URL = os.getenv("CLOUD_URL", "https://ginny.stavyadatta.com").rstrip('/')
API_KEY = os.getenv("FLAGS_API_KEY", "").strip()
API_HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}

def poll_commands():
    try:
        resp = requests.get(f"{CLOUD_URL}/api/poll_commands", headers=API_HEADERS, timeout=1)
        if resp.status_code == 200:
            data = resp.json()
            commands = data.get("commands", [])
            for cmd in commands:
                kind = cmd.get("kind", "")
                print(f"[bridge] Received command: {kind}")

                if kind == "birthday":
                    Buttons_vals.set_birthday()
                elif kind == "stop_recording":
                    Buttons_vals.set_stop_recording()
                elif kind == "dance":
                    Buttons_vals.set_dance()
                elif kind == "raise_hand":
                    Buttons_vals.set_raise_hand()
                elif kind == "ask_question":
                    Buttons_vals.set_ask_question()
                elif kind == "say_thanks":
                    Buttons_vals.set_say_thanks()
                elif kind == "cycle_volume":
                    new_vol = Volume.cycle()
                    print(f"[bridge] Cycled volume to: {new_vol}")
                elif kind.startswith("set_mic_threshold:"):
                    try:
                        val = int(kind.split(":")[1])
                        Mic_UI.set_mic_threshold(val)
                        print(f"[bridge] Set mic threshold to: {val}")
                    except:
                        print(f"[bridge] Invalid threshold cmd: {kind}")
                elif kind.startswith("set_face_min_area:"):
                    try:
                        val = int(kind.split(":")[1])
                        FaceArea_UI.set_face_min_area(val)
                        print(f"[bridge] Set face min area to: {val}")
                    except:
                        print(f"[bridge] Invalid face area cmd: {kind}")
                else:
                    print(f"[bridge] Unknown command: {kind}")
    except Exception as e:
        print(f"[bridge] Poll error: {e}")

def push_telemetry():
    try:
        payload = {
            "front_energy": Telemetry.peek_front_mic_energy(),
            "volume": Volume.peek_volume(),
            "mic_threshold": Mic_UI.peek_mic_threshold(),
            "face_min_area": FaceArea_UI.peek_face_min_area()
        }
        requests.post(f"{CLOUD_URL}/api/telemetry", json=payload, headers=API_HEADERS, timeout=0.5)
    except Exception as e:
        # Don't spam logs on connection error
        pass

def run_bridge():
    print(f"Starting Local Bridge -> {CLOUD_URL}")
    while True:
        poll_commands()
        push_telemetry()
        time.sleep(0.1) # 100ms poll interval

if __name__ == "__main__":
    # For standalone testing: python -m button_frontend.local_bridge
    run_bridge()
