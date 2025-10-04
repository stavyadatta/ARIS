# analysis_and_plots_by_message_len.py
# Requirements: pandas, matplotlib

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- Config ----------
CSV_PATH = "/workspace/7000s_rag_benchmark_results_repitition_filtered.csv"   # change if needed
SAVE_DIR = Path("./7000s_rag_results_repitition_filtered/")
TIME_METRICS_MS = ["speaking_ms", "exec_init_ms", "stream_ms", "total_ms"]
OTHER_METRICS = ["printed_chars"]  # added per your request
METRICS_TO_PLOT = TIME_METRICS_MS + OTHER_METRICS
SHOW_PLOTS = True  # False => only save
# ---------------------------

SAVE_DIR.mkdir(parents=True, exist_ok=True)

# 1) Load & tidy
df = pd.read_csv(CSV_PATH)
df["message_len"] = pd.to_numeric(df["message_len"], errors="coerce")

# Normalise is_rag column to booleans
if "is_rag" in df.columns and df["is_rag"].dtype == object:
    df["is_rag"] = (
        df["is_rag"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False})
    )

# Drop invalid rows
df = df.dropna(subset=["message_len", "is_rag"])

# 2) Aggregate: average across all questions & faces per (message_len, is_rag)
metrics_avgs = {m: "mean" for m in METRICS_TO_PLOT}
agg = (
    df.groupby(["message_len", "is_rag"], dropna=False)
      .agg(metrics_avgs)
      .reset_index()
)

if agg.empty:
    raise ValueError("No data after aggregation. Check CSV filters/columns.")

# Global X range
x_min, x_max = agg["message_len"].min(), agg["message_len"].max()

def nice_ylabel(metric: str) -> str:
    if metric in TIME_METRICS_MS:
        return f"{metric} (milliseconds)"
    return metric

def plot_metric(metric: str):
    """One figure per metric, two lines (RAG ON/OFF) averaged across faces/questions"""
    sub = agg[["message_len", "is_rag", metric]].dropna()
    if sub.empty:
        return None

    rag_on  = sub[sub["is_rag"] == True].sort_values("message_len")
    rag_off = sub[sub["is_rag"] == False].sort_values("message_len")

    fig, ax = plt.subplots(figsize=(8, 5))

    if not rag_on.empty:
        ax.plot(rag_on["message_len"], rag_on[metric], marker="o", label="RAG: ON")
    if not rag_off.empty:
        ax.plot(rag_off["message_len"], rag_off[metric], marker="o", label="RAG: OFF")

    ax.set_title(f"{metric} vs Number of Messages (avg over all user ids & questions)")
    ax.set_xlabel("Number of Messages")
    ax.set_ylabel(nice_ylabel(metric))
    ax.set_xlim(x_min, x_max)
    ax.grid(True, alpha=0.3)
    ax.legend()

    out_path = SAVE_DIR / f"{metric}_vs_message_len.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)
    return out_path

# 3) Generate one figure per metric (includes printed_chars now)
outputs = []
for metric in METRICS_TO_PLOT:
    p = plot_metric(metric)
    if p is not None:
        outputs.append(p)

print(f"Saved {len(outputs)} figures (one per metric) to: {SAVE_DIR.resolve()}")

