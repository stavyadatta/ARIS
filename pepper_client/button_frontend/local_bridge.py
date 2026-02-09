import time
import requests
import os
import sys

# Import shared classes from button.py
# ensuring we can import from current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from button import Buttons_vals, Mic_UI, Telemetry, Volume

CLOUD_URL = os.getenv("CLOUD_URL", "http://localhost:8004").rstrip('/')

def poll_commands():
    try:
        resp = requests.get(f"{CLOUD_URL}/api/poll_commands", timeout=1)
        if resp.status_code == 200:
            data = resp.json()
            commands = data.get("commands", [])
            for cmd in commands:
                kind = cmd.get("kind", "")
                print(f"Received command: {kind}")
                
                if kind == "first_source":
                    Buttons_vals.set_first_source()
                elif kind == "stop_recording":
                    Buttons_vals.set_stop_recording()
                elif kind == "dance":
                    Buttons_vals.set_dance()
                elif kind == "cycle_volume":
                    # Cycle volume locally
                    new_vol = Volume.cycle()
                    print(f"Cycled volume to: {new_vol}")
                elif kind.startswith("set_mic_threshold:"):
                    try:
                        val = int(kind.split(":")[1])
                        Mic_UI.set_mic_threshold(val)
                        print(f"Set mic threshold to: {val}")
                    except:
                        print(f"Invalid threshold cmd: {kind}")
    except Exception as e:
        print(f"Poll error: {e}")

def push_telemetry():
    try:
        payload = {
            "front_energy": Telemetry.peek_front_mic_energy(),
            "volume": Volume.peek_volume(),
            "mic_threshold": Mic_UI.peek_mic_threshold()
        }
        requests.post(f"{CLOUD_URL}/api/telemetry", json=payload, timeout=0.5)
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
    run_bridge()
