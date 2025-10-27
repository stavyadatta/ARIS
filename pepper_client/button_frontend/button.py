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

class Flags:
    """Import this class in other Python code to interact programmatically."""
    first_source = _Latch()
    stop_recording = _Latch()

    @classmethod
    def set_first_source(cls): cls.first_source.set(True)
    @classmethod
    def set_stop_recording(cls): cls.stop_recording.set(True)

    @classmethod
    def peek_first_source(cls) -> bool: return cls.first_source.peek()
    @classmethod
    def peek_stop_recording(cls) -> bool: return cls.stop_recording.peek()

    @classmethod
    def consume_first_source(cls) -> bool: return cls.first_source.consume()
    @classmethod
    def consume_stop_recording(cls) -> bool: return cls.stop_recording.consume()

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
            <button id="btn-speak"
              class="relative pulse btn-press rounded-xl px-5 py-5 text-lg font-semibold shadow-lg ring-1 ring-white/10
                     bg-gradient-to-br from-indigo-500 to-sky-500 hover:from-indigo-400 hover:to-sky-400
                     focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-sky-300 focus:ring-offset-transparent flex items-center justify-center gap-3">
              <svg xmlns="http://www.w3.org/2000/svg" class="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.6" d="M12 18v3m0-3a6 6 0 0 0 6-6m-6 6a6 6 0 0 1-6-6m6-9v9m0 0a3 3 0 1 0 0-6 3 3 0 0 0 0 6" />
              </svg>
              Speak First Source
            </button>

            <button id="btn-stop"
              class="btn-press rounded-xl px-5 py-5 text-lg font-semibold shadow-lg ring-1 ring-white/10
                     bg-gradient-to-br from-rose-500 to-red-600 hover:from-rose-400 hover:to-red-500
                     focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-rose-300 focus:ring-offset-transparent flex items-center justify-center gap-3">
              <svg xmlns="http://www.w3.org/2000/svg" class="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                <path d="M6 7a1 1 0 0 1 1-1h10a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H7a1 1 0 0 1-1-1z"/>
              </svg>
              Stop Recording
            </button>
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

        function vibrate(ms=20){ if (navigator.vibrate) navigator.vibrate(ms); }
        function showToast(msg){
          toastText.textContent = msg;
          toastEl.classList.remove('hidden','opacity-0');
          toastEl.classList.add('opacity-100');
          setTimeout(()=>toastEl.classList.add('opacity-0'), 1200);
          setTimeout(()=>toastEl.classList.add('hidden'), 1600);
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

        document.getElementById('btn-speak').addEventListener('click', ()=>setFlag('first_source'));
        document.getElementById('btn-stop').addEventListener('click', ()=>setFlag('stop_recording'));

        document.addEventListener('keydown', (e)=>{
          if(e.key.toLowerCase()==='s') setFlag('first_source');
          if(e.key.toLowerCase()==='x') setFlag('stop_recording');
        });
      </script>
    </body>
    </html>
    """

    @app.get("/")
    def index():
        return render_template_string(INDEX_HTML)

    # --------- Web UI -> set a flag ---------
    @app.post("/flag/set")
    def flag_set():
        payload = request.get_json(silent=True) or {}
        kind = str(payload.get("kind","")).strip().lower()
        if kind == "first_source":
            Flags.set_first_source()
        elif kind == "stop_recording":
            Flags.set_stop_recording()
        else:
            return jsonify({"error":"unknown flag kind"}), 400
        return jsonify({"ok": True, "kind": kind})

    # --------- API for client programs ---------
    @app.get("/api/flags/peek")
    def api_peek():
        if not require_api_key():
            return jsonify({"error":"Unauthorized"}), 401
        return jsonify({
            "ok": True,
            "first_source": Flags.peek_first_source(),
            "stop_recording": Flags.peek_stop_recording(),
        })

    @app.get("/api/flags/consume/first_source")
    def api_consume_first():
        if not require_api_key():
            return jsonify({"error":"Unauthorized"}), 401
        return jsonify({"ok": True, "value": Flags.consume_first_source()})

    @app.get("/api/flags/consume/stop_recording")
    def api_consume_stop():
        if not require_api_key():
            return jsonify({"error":"Unauthorized"}), 401
        return jsonify({"ok": True, "value": Flags.consume_stop_recording()})

    @app.get("/api/health")
    def api_health():
        return jsonify({"ok": True})

    return app


def run_button_server(host: str = "0.0.0.0", port: int = 8004):
    """Starts the GINNY control web server."""
    app = create_app()
    app.run(host=host, port=port)


if __name__ == "__main__":
    # Optional: export FLAGS_API_KEY=dev123 to require X-API-Key on /api/* routes.
    run_button_server()

