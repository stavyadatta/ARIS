import os
from threading import Lock
from flask import Flask, request, jsonify, render_template_string

# -------------------------------------------------
# One-shot latch flags (process-wide, thread-safe)
# -------------------------------------------------
class _Latch:
    def __init__(self):
        self._v = False
        self._lock = Lock()
    def set(self, value: bool = True):
        with self._lock:
            self._v = bool(value)
    def peek(self) -> bool:
        with self._lock:
            return self._v
    def consume(self) -> bool:
        """Return current value and immediately reset to False."""
        with self._lock:
            curr = self._v
            self._v = False
            return curr

# -------------------------------------------------
# Persistent (non-latching) integer setting
# -------------------------------------------------
class _IntSetting:
    def __init__(self, initial: int, min_v: int, max_v: int, step: int):
        self._v = initial
        self._min = min_v
        self._max = max_v
        self._step = step
        self._lock = Lock()

    def _snap(self, x: int) -> int:
        if x < self._min: x = self._min
        if x > self._max: x = self._max
        k = round((x - self._min) / self._step)
        return self._min + k * self._step

    def get(self) -> int:
        with self._lock:
            return self._v

    def set(self, x: int) -> int:
        snapped = self._snap(int(x))
        with self._lock:
            self._v = snapped
            return self._v

# ---- Float setting (no snapping) ----
class _FloatVar:
    def __init__(self, initial: float = 0.0):
        self._v = float(initial)
        self._lock = Lock()
    def get(self) -> float:
        with self._lock:
            return self._v
    def set(self, x: float) -> float:
        with self._lock:
            self._v = float(x)
            return self._v

# -------------------------------------------------
# Global shared Flags and Settings
# -------------------------------------------------
class Buttons_vals:
    """Import this class in other Python code to interact programmatically."""
    first_source = _Latch()
    stop_recording = _Latch()
    dance = _Latch()  # NEW

    @classmethod
    def set_first_source(cls): cls.first_source.set(True)
    @classmethod
    def set_stop_recording(cls): cls.stop_recording.set(True)
    @classmethod
    def set_dance(cls): cls.dance.set(True)  # NEW

    @classmethod
    def peek_first_source(cls) -> bool: return cls.first_source.peek()
    @classmethod
    def peek_stop_recording(cls) -> bool: return cls.stop_recording.peek()
    @classmethod
    def peek_dance(cls) -> bool: return cls.dance.peek()  # NEW

    @classmethod
    def consume_first_source(cls) -> bool: return cls.first_source.consume()
    @classmethod
    def consume_stop_recording(cls) -> bool: return cls.stop_recording.consume()
    @classmethod
    def consume_dance(cls) -> bool: return cls.dance.consume()  # NEW

class Mic_UI:
    """Thread-safe runtime settings."""
    _mic_threshold = _IntSetting(initial=370, min_v=12, max_v=8000, step=20)

    @classmethod
    def peek_mic_threshold(cls) -> int:
        return cls._mic_threshold.get()

    @classmethod
    def set_mic_threshold(cls, value: int) -> int:
        return cls._mic_threshold.set(value)

class Telemetry:
    """Live telemetry pushed by other modules (e.g., audio)."""
    _front_mic_energy = _FloatVar(0.0)

    @classmethod
    def peek_front_mic_energy(cls) -> float:
        return cls._front_mic_energy.get()

    @classmethod
    def set_front_mic_energy(cls, value: float) -> float:
        return cls._front_mic_energy.set(value)

# -------------------------------------------------
# Volume control (cycle via server only; public peek)
# -------------------------------------------------
class Volume:
    """Process-wide volume with cycle(40->90->0). External code gets peek only."""
    _values = [40, 90, 0]
    _idx = 0
    _lock = Lock()

    @classmethod
    def _set_idx(cls, idx: int) -> int:
        with cls._lock:
            cls._idx = idx % len(cls._values)
            return cls._values[cls._idx]

    @classmethod
    def cycle(cls) -> int:
        """Advance to next value and return it. Used by server endpoint only."""
        with cls._lock:
            cls._idx = (cls._idx + 1) % len(cls._values)
            return cls._values[cls._idx]

    @classmethod
    def peek_volume(cls) -> int:
        with cls._lock:
            return cls._values[cls._idx]

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

    # ---------- Fancy UI ----------
    INDEX_HTML = """
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>GINNY control</title>
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
          <p class="text-slate-300/80 mt-2 text-sm">Tap a control. Flags are one-shot and reset when read.</p>
        </header>

        <main class="glass rounded-2xl p-5 md:p-7 shadow-2xl">
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

      <!-- Live Mic Energy -->
      <div class="mt-6 p-5 rounded-2xl ring-1 ring-white/10 bg-black/20 text-center">
        <div class="text-xs uppercase tracking-wider text-slate-400">Live Mic Energy</div>
        <div id="energy-value" class="mt-2 font-extrabold"
            style="font-size: clamp(2.5rem, 8vw, 5rem); line-height: 1; letter-spacing: 0.02em;">
            0.0
        </div>
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
            const res = await fetch('/api/volume'); // protected only if API key set
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
              setVolButtonStyle(data.value);
              statusEl.textContent = "Volume: " + data.value;
              showToast("Volume " + data.value);
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
              const msg = `Set: ${data.kind}`;
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
              showToast(`Mic threshold: ${data.value}`);
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

        async function pollEnergy(){
          try{
            const res = await fetch('/front_energy', { cache: 'no-store' });
            const data = await res.json();
            if(res.ok && typeof data.value !== 'undefined'){
              const v = Number(data.value);
              document.getElementById('energy-value').textContent =
                  Number.isFinite(v) ? v.toFixed(1) : '—';
            }
          }catch{}
        }

        // Init
        fetchVolume();   // get initial volume (defaults to 40 server-side)
        setInterval(pollEnergy, 50);
        pollEnergy();
        loadMic();
      </script>
    </body>
    </html>
    """

    @app.get("/")
    def index():
        return render_template_string(INDEX_HTML)

    # ---------- Flag endpoints ----------
    @app.post("/flag/set")
    def flag_set():
        payload = request.get_json(silent=True) or {}
        kind = str(payload.get("kind", "")).strip().lower()
        if kind == "first_source":
            Buttons_vals.set_first_source()
        elif kind == "stop_recording":
            Buttons_vals.set_stop_recording()
        elif kind == "dance":  # NEW
            Buttons_vals.set_dance()
        else:
            return jsonify({"error": "unknown flag kind"}), 400
        return jsonify({"ok": True, "kind": kind})

    # ---------- Threshold endpoints ----------
    @app.post("/threshold/set")
    def threshold_set():
        payload = request.get_json(silent=True) or {}
        try:
            val = int(payload.get("value"))
        except Exception:
            return jsonify({"error": "invalid value"}), 400
        new_val = Mic_UI.set_mic_threshold(val)
        return jsonify({"ok": True, "value": new_val})

    @app.get("/api/threshold")
    def api_threshold_get():
        if not require_api_key():
            return jsonify({"error": "Unauthorized"}), 401
        return jsonify({"ok": True, "value": Mic_UI.peek_mic_threshold()})

    @app.post("/api/threshold")
    def api_threshold_post():
        if not require_api_key():
            return jsonify({"error": "Unauthorized"}), 401
        payload = request.get_json(silent=True) or {}
        try:
            val = int(payload.get("value"))
        except Exception:
            return jsonify({"error": "invalid value"}), 400
        new_val = Mic_UI.set_mic_threshold(val)
        return jsonify({"ok": True, "value": new_val})

    # ---------- Health + flag API ----------
    @app.get("/api/flags/peek")
    def api_peek():
        if not require_api_key():
            return jsonify({"error": "Unauthorized"}), 401
        return jsonify({
            "ok": True,
            "first_source": Buttons_vals.peek_first_source(),
            "stop_recording": Buttons_vals.peek_stop_recording(),
            "dance": Buttons_vals.peek_dance(),  # NEW
        })

    @app.get("/api/flags/consume/first_source")
    def api_consume_first():
        if not require_api_key():
            return jsonify({"error": "Unauthorized"}), 401
        return jsonify({"ok": True, "value": Buttons_vals.consume_first_source()})

    @app.get("/api/flags/consume/stop_recording")
    def api_consume_stop():
        if not require_api_key():
            return jsonify({"error": "Unauthorized"}), 401
        return jsonify({"ok": True, "value": Buttons_vals.consume_stop_recording()})

    @app.get("/api/flags/consume/dance")
    def api_consume_dance():  # NEW
        if not require_api_key():
            return jsonify({"error": "Unauthorized"}), 401
        return jsonify({"ok": True, "value": Buttons_vals.consume_dance()})

    @app.get("/api/health")
    def api_health():
        return jsonify({"ok": True})

    # --- Live energy (unprotected UI poll) ---
    @app.get("/front_energy")
    def front_energy_public():
        return jsonify({"ok": True, "value": Telemetry.peek_front_mic_energy()})

    # --- Live energy (API; honours FLAGS_API_KEY if set) ---
    @app.get("/api/front_energy")
    def front_energy_api():
        if not require_api_key():
            return jsonify({"error": "Unauthorized"}), 401
        return jsonify({"ok": True, "value": Telemetry.peek_front_mic_energy()})

    # --- Volume endpoints ---
    @app.post("/volume/cycle")
    def volume_cycle():
        """Cycle volume 40 -> 90 -> 0. UI only."""
        val = Volume.cycle()
        return jsonify({"ok": True, "value": val})

    @app.get("/api/volume")
    def api_volume_get():
        """Peek-only API (protected if API key set)."""
        if not require_api_key():
            return jsonify({"error": "Unauthorized"}), 401
        return jsonify({"ok": True, "value": Volume.peek_volume()})

    # Ensure default volume is 40 on server start
    Volume._set_idx(0)

    return app


def run_button_server(host="0.0.0.0", port=8004):
    app = create_app()
    app.run(host=host, port=port)


if __name__ == "__main__":
    run_button_server()

