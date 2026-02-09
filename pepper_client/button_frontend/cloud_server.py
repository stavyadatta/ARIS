import os
import time
from collections import deque
from threading import Lock
from flask import Flask, request, jsonify, render_template_string

# -------------------------------------------------
# Cloud State
# -------------------------------------------------
class CloudState:
    # Command queue: (command_name, timestamp)
    _command_queue = deque()
    _queue_lock = Lock()
    
    # Telemetry cache
    _telemetry = {
        "front_energy": 0.0,
        "volume": 40,
        "mic_threshold": 370
    }
    _telemetry_lock = Lock()

    @classmethod
    def add_command(cls, kind: str):
        with cls._queue_lock:
            # Add command with timestamp
            cls._command_queue.append({
                "kind": kind,
                "ts": time.time()
            })

    @classmethod
    def get_pending_commands(cls):
        """Return all pending commands and clear the queue."""
        with cls._queue_lock:
            cmds = list(cls._command_queue)
            cls._command_queue.clear()
            return cmds

    @classmethod
    def update_telemetry(cls, data: dict):
        with cls._telemetry_lock:
            if "front_energy" in data:
                cls._telemetry["front_energy"] = float(data["front_energy"])
            if "volume" in data:
                cls._telemetry["volume"] = int(data["volume"])
            if "mic_threshold" in data:
                cls._telemetry["mic_threshold"] = int(data["mic_threshold"])

    @classmethod
    def get_telemetry(cls):
        with cls._telemetry_lock:
            return cls._telemetry.copy()

# -------------------------------------------------
# Flask app
# -------------------------------------------------
def create_app():
    app = Flask(__name__)
    API_KEY = os.getenv("FLAGS_API_KEY", "").strip()

    def require_api_key():
        if not API_KEY:
            return True
        return request.headers.get("X-API-Key") == API_KEY

    # ---------- Fancy UI (Same as button.py) ----------
    INDEX_HTML = """
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>GINNY control (Cloud)</title>
      <script src="https://cdn.tailwindcss.com"></script>
      <style>
        .glass{background:rgba(17,24,39,.55);backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);border:1px solid rgba(255,255,255,.06)}
        .btn-press:active{transform:translateY(1px) scale(.99)}
        .pulse::after{content:"";position:absolute;inset:0;border-radius:1rem;box-shadow:0 0 0 0 rgba(255,255,255,.25);animation:pulse 1.8s ease-out infinite}
        @keyframes pulse{0%{box-shadow:0 0 0 0 rgba(255,255,255,.25)}100%{box-shadow:0 0 0 24px rgba(255,255,255,0)}}
        input[type=range]{-webkit-appearance:none;width:100%;height:6px;border-radius:9999px;background:linear-gradient(90deg,#38bdf8,#a78bfa);outline:none}
        input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;width:20px;height:20px;border-radius:9999px;background:white;border:2px solid rgba(0,0,0,.15);box-shadow:0 2px 10px rgba(0,0,0,.25)}
        input[type=range]::-moz-range-thumb{width:20px;height:20px;border-radius:9999px;background:white;border:2px solid rgba(0,0,0,.15)}
        /* Volume state classes */
        .vol-40 { background: linear-gradient(to bottom right, #f59e0b, #fbbf24); }   /* amber */
        .vol-90 { background: linear-gradient(to bottom right, #10b981, #34d399); }  /* emerald */
        .vol-0  { background: linear-gradient(to bottom right, #475569, #334155); }  /* slate */
      </style>
    </head>
    <body class="min-h-screen text-slate-100" style="background: radial-gradient(1200px 600px at 10% -10%, #1f3a8a, transparent), radial-gradient(1000px 600px at 110% 10%, #7c3aed, transparent), linear-gradient(180deg, #0b1020, #0a0f1e 55%, #0b1324);">
      <div class="max-w-2xl mx-auto px-4 py-10">
        <header class="mb-8 text-center">
          <h1 class="text-3xl md:text-4xl font-extrabold tracking-wide">
            <span class="bg-gradient-to-r from-indigo-300 via-sky-200 to-violet-300 bg-clip-text text-transparent">GINNY control</span>
          </h1>
          <p class="text-slate-300/80 mt-2 text-sm">Cloud Interface. Commands are queued for device.</p>
        </header>

        <main class="glass rounded-2xl p-5 md:p-7 shadow-2xl">
        <!-- Live Mic (moved up) -->
            <div class="mb-5">
            <div class="rounded-2xl ring-1 ring-white/10 bg-gradient-to-br from-slate-800/70 to-slate-900/60 glass p-4 flex items-center justify-between">
                <div class="text-xs uppercase tracking-wider text-slate-300">Live Mic (Cached)</div>
                <div id="energy-value"
                    class="font-extrabold"
                    style="font-size: clamp(2rem, 7vw, 3.75rem); line-height: 1; letter-spacing: 0.01em;">
                0.0
                </div>
            </div>
            </div>

          <div class="grid gap-4">
            <!-- Speak -->
            <button id="btn-speak"
              class="relative pulse btn-press rounded-xl px-5 py-5 text-lg font-semibold shadow-lg ring-1 ring-white/10
                     bg-gradient-to-br from-indigo-500 to-sky-500 hover:from-indigo-400 hover:to-sky-400
                     focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-sky-300 focus:ring-offset-transparent flex items-center justify-center gap-3">
              <svg xmlns="http://www.w3.org/2000/svg" class="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.6" d="M12 18v3m0-3a6 6 0 0 0 6-6m-6 6a6 6 0 0 1-6-6m6-9v9m0 0a3 3 0 1 0 0-6 3 3 0 0 0 0 6" />
              </svg>
              Speak First Source
            </button>

            <!-- Stop recording -->
            <button id="btn-stop"
              class="btn-press rounded-xl px-5 py-5 text-lg font-semibold shadow-lg ring-1 ring-white/10
                     bg-gradient-to-br from-rose-500 to-red-600 hover:from-rose-400 hover:to-red-500
                     focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-rose-300 focus:ring-offset-transparent flex items-center justify-center gap-3">
              <svg xmlns="http://www.w3.org/2000/svg" class="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                <path d="M6 7a1 1 0 0 1 1-1h10a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H7a1 1 0 0 1-1-1z"/>
              </svg>
              Stop Recording
            </button>

            <!-- Dance -->
            <button id="btn-dance"
              class="relative pulse btn-press rounded-xl px-5 py-5 text-lg font-semibold shadow-lg ring-1 ring-white/10
                     bg-gradient-to-br from-fuchsia-500 to-purple-600 hover:from-fuchsia-400 hover:to-purple-500
                     focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-fuchsia-300 focus:ring-offset-transparent flex items-center justify-center gap-3">
              <svg xmlns="http://www.w3.org/2000/svg" class="w-6 h-6" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 3a2 2 0 110 4 2 2 0 010-4zm-2 6h4l2 3-2 2 1 6h-2l-1-4-1 4H9l1-6-2-2 2-3z"/>
              </svg>
              Dance
            </button>

            <!-- Volume (cycles 40 -> 90 -> 0) -->
            <button id="btn-volume"
              class="btn-press rounded-xl px-5 py-5 text-lg font-semibold shadow-lg ring-1 ring-white/10 vol-40
                     focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-amber-300 focus:ring-offset-transparent
                     flex items-center justify-between gap-3">
              <div class="flex items-center gap-3">
                <svg xmlns="http://www.w3.org/2000/svg" class="w-6 h-6" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M5 9v6h4l5 4V5l-5 4H5z"/>
                </svg>
                <span>Volume</span>
              </div>
              <span id="vol-label" class="text-sm font-bold">40</span>
            </button>
          </div>

          <!-- Mic Threshold Slider -->
          <div class="mt-6 p-4 rounded-xl ring-1 ring-white/10 bg-black/20">
            <div class="flex items-center justify-between mb-3">
              <label for="mic-slider" class="text-sm uppercase tracking-wider text-slate-300">Mic Threshold</label>
              <span id="mic-value" class="text-sm font-semibold text-sky-200">—</span>
            </div>
            <input id="mic-slider" type="range" min="12" max="8000" step="20" value="370"/>
            <p class="mt-2 text-xs text-slate-400">Min 12 • Max 8000 • Step 20</p>
          </div>

          <div id="status" class="mt-4 text-sm text-slate-300/90"></div>
        </main>
      </div>

      <div id="toast" class="fixed bottom-5 left-1/2 -translate-x-1/2 hidden">
        <div class="glass rounded-xl px-4 py-3 shadow-xl text-sm">
          <span id="toast-text">Flag set</span>
        </div>
      </div>

      <script>
        const statusEl = document.getElementById('status');
        const toastEl = document.getElementById('toast');
        const toastText = document.getElementById('toast-text');
        const micSlider = document.getElementById('mic-slider');
        const micValue = document.getElementById('mic-value');
        const volBtn = document.getElementById('btn-volume');
        const volLabel = document.getElementById('vol-label');

        function vibrate(ms=20){ if (navigator.vibrate) navigator.vibrate(ms); }
        function showToast(msg){
          toastText.textContent = msg;
          toastEl.classList.remove('hidden','opacity-0');
          toastEl.classList.add('opacity-100');
          setTimeout(()=>toastEl.classList.add('opacity-0'), 1200);
          setTimeout(()=>toastEl.classList.add('hidden'), 1600);
        }

        function setVolButtonStyle(val){
          volBtn.classList.remove('vol-40','vol-90','vol-0');
          if(val===40){ volBtn.classList.add('vol-40'); }
          else if(val===90){ volBtn.classList.add('vol-90'); }
          else { volBtn.classList.add('vol-0'); }
          volLabel.textContent = String(val);
        }

        async function fetchVolume(){
          try{
            const res = await fetch('/api/volume'); 
            const data = await res.json();
            if(res.ok && typeof data.value === 'number'){
              setVolButtonStyle(data.value);
            }
          }catch{}
        }

        async function cycleVolume(){
          statusEl.textContent = "Cycling volume...";
          vibrate();
          try{
            const res = await fetch('/volume/cycle', { method: 'POST' });
            const data = await res.json();
            if(res.ok){
              // Optimistic update or wait for poll? 
              // For now, we just rely on the response which confirms the command was queued.
              // The actual volume update will come via telemetry polling.
              statusEl.textContent = "Volume cycle queued";
              showToast("Volume cycle queued");
            }else{
              statusEl.textContent = data.error || 'Error';
            }
          }catch{
            statusEl.textContent = 'Network error';
          }
        }

        async function setFlag(kind){
          statusEl.textContent = "Sending...";
          vibrate();
          try{
            const res = await fetch('/flag/set', {
              method: 'POST',
              headers: {'Content-Type':'application/json'},
              body: JSON.stringify({ kind })
            });
            const data = await res.json();
            if(res.ok){
              const msg = `Queued: ${data.kind}`;
              statusEl.textContent = msg;
              showToast(msg);
            }else{
              statusEl.textContent = data.error || 'Error';
            }
          }catch{ statusEl.textContent = 'Network error'; }
        }

        // --- Mic threshold helpers ---
        function snapToStep(x, min=12, step=20){
          const k = Math.round((x - min) / step);
          return min + k * step;
        }

        async function loadMic(){
          try{
            const res = await fetch('/api/threshold');
            const data = await res.json();
            if(res.ok && typeof data.value === 'number'){
              micSlider.value = data.value;
              micValue.textContent = data.value;
            } else {
              micValue.textContent = micSlider.value;
            }
          }catch{
            micValue.textContent = micSlider.value;
          }
        }

        async function saveMic(val){
          try{
            const res = await fetch('/threshold/set', {
              method: 'POST',
              headers: {'Content-Type':'application/json'},
              body: JSON.stringify({ value: val })
            });
            const data = await res.json();
            if(res.ok){
              micValue.textContent = data.value;
              showToast(`Mic threshold queued: ${data.value}`);
            } else {
              showToast(data.error || 'Error');
            }
          }catch{
            showToast('Network error');
          }
        }

        document.getElementById('btn-speak').addEventListener('click', ()=>setFlag('first_source'));
        document.getElementById('btn-stop').addEventListener('click', ()=>setFlag('stop_recording'));
        document.getElementById('btn-dance').addEventListener('click', ()=>setFlag('dance'));
        volBtn.addEventListener('click', cycleVolume);

        document.addEventListener('keydown', (e)=>{
          if(e.key.toLowerCase()==='s') setFlag('first_source');
          if(e.key.toLowerCase()==='x') setFlag('stop_recording');
          if(e.key.toLowerCase()==='d') setFlag('dance');
        });

        // Slider events (snap + save)
        micSlider.addEventListener('input', (e)=>{
          const snapped = snapToStep(parseInt(e.target.value,10));
          if(snapped != e.target.value){
            e.target.value = snapped;
          }
          micValue.textContent = e.target.value;
        });
        micSlider.addEventListener('change', (e)=>{
          const snapped = snapToStep(parseInt(e.target.value,10));
          e.target.value = snapped;
          saveMic(snapped);
        });

        async function pollTelemetry(){
          try{
            const res = await fetch('/front_energy', { cache: 'no-store' });
            const data = await res.json();
            if(res.ok){
                // Energy
                if (typeof data.value !== 'undefined') {
                    const v = Number(data.value);
                    document.getElementById('energy-value').textContent =
                        Number.isFinite(v) ? v.toFixed(1) : '—';
                }
                // Volume sync
                if (typeof data.volume !== 'undefined') {
                    setVolButtonStyle(data.volume);
                }
                // Mic threshold sync (optional, if we want to keep it in sync with device)
                if (typeof data.mic_threshold !== 'undefined') {
                    // Only update if user is not dragging? For now, let's just update text
                    // micValue.textContent = data.mic_threshold;
                }
            }
          }catch{}
        }

        // Init
        fetchVolume();   
        setInterval(pollTelemetry, 200); // Poll cloud cache every 200ms
        pollTelemetry();
        loadMic();
      </script>
    </body>
    </html>
    """

    @app.get("/")
    def index():
        return render_template_string(INDEX_HTML)

    # ---------- UI Endpoints (Queue Commands) ----------
    @app.post("/flag/set")
    def flag_set():
        payload = request.get_json(silent=True) or {}
        kind = str(payload.get("kind", "")).strip().lower()
        if kind in ["first_source", "stop_recording", "dance"]:
            CloudState.add_command(kind)
            return jsonify({"ok": True, "kind": kind})
        return jsonify({"error": "unknown flag kind"}), 400

    @app.post("/threshold/set")
    def threshold_set():
        payload = request.get_json(silent=True) or {}
        try:
            val = int(payload.get("value"))
            CloudState.add_command(f"set_mic_threshold:{val}")
            return jsonify({"ok": True, "value": val})
        except Exception:
            return jsonify({"error": "invalid value"}), 400

    @app.post("/volume/cycle")
    def volume_cycle():
        CloudState.add_command("cycle_volume")
        return jsonify({"ok": True})

    @app.get("/api/volume")
    def api_volume_get():
        telemetry = CloudState.get_telemetry()
        return jsonify({"ok": True, "value": telemetry["volume"]})

    @app.get("/api/threshold")
    def api_threshold_get():
        telemetry = CloudState.get_telemetry()
        return jsonify({"ok": True, "value": telemetry["mic_threshold"]})

    @app.get("/front_energy")
    def front_energy_public():
        telemetry = CloudState.get_telemetry()
        return jsonify({
            "ok": True, 
            "value": telemetry["front_energy"],
            "volume": telemetry["volume"],
            "mic_threshold": telemetry["mic_threshold"]
        })

    # ---------- Device Endpoints (Poll & Push) ----------
    @app.get("/api/poll_commands")
    def poll_commands():
        # Device calls this to get pending commands
        cmds = CloudState.get_pending_commands()
        return jsonify({"ok": True, "commands": cmds})

    @app.post("/api/telemetry")
    def push_telemetry():
        # Device calls this to update state
        payload = request.get_json(silent=True) or {}
        CloudState.update_telemetry(payload)
        return jsonify({"ok": True})

    return app

def run_cloud_server(host="0.0.0.0", port=8004):
    app = create_app()
    app.run(host=host, port=port)

if __name__ == "__main__":
    run_cloud_server()
