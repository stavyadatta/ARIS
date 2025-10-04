# === Config ===
INPUT_CSV  = "/workspace/7000s_rag_benchmark_results_repitition.csv"
OUTPUT_CSV = "7000s_rag_benchmark_results_repitition_similarity.csv"

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

REQUIRED_COLS = ["face_id", "question_idx", "is_rag", "response_text", "repitition"]

def to_bool_is_rag(x):
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    return s in {"true", "t", "1", "yes", "y"}

def main():
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # Load
    df = pd.read_csv(INPUT_CSV)

    # Basic checks
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # Normalise is_rag
    df["is_rag_bool"] = df["is_rag"].apply(to_bool_is_rag)

    # Split RAG / Non-RAG
    rag_df = (
        df[df["is_rag_bool"] == True]
        [["face_id", "question_idx", "repitition", "response_text"]]
        .rename(columns={"response_text": "rag_text",
                         "repitition": "rag_repitition"})
    )
    nrag_df = (
        df[df["is_rag_bool"] == False]
        [["face_id", "question_idx", "repitition", "response_text"]]
        .rename(columns={"response_text": "nonrag_text",
                         "repitition": "nonrag_repitition"})
    )

    # 3×3 Cartesian within same (face_id, question_idx)
    pair_df = rag_df.merge(
        nrag_df,
        on=["face_id", "question_idx"],
        how="inner",
        suffixes=("_rag", "_nonrag")
    )

    if pair_df.empty:
        out = df.copy()
        out["rag_nonrag_similarity"] = pd.NA
        out.to_csv(OUTPUT_CSV, index=False)
        print("No RAG/Non-RAG pairs found. Wrote file with empty similarity column.")
        return

    # Load model
    model = SentenceTransformer("google/embeddinggemma-300m", device=DEVICE)

    # Embed texts for row-wise pairs
    rag_emb  = model.encode(pair_df["rag_text"].tolist(),
                            convert_to_tensor=True, device=DEVICE)
    nrag_emb = model.encode(pair_df["nonrag_text"].tolist(),
                            convert_to_tensor=True, device=DEVICE)

    sims = model.similarity(rag_emb, nrag_emb).diag().detach().cpu().numpy()
    pair_df["pair_similarity"] = sims

    # For each (face_id, question_idx), average over all 3×3 = 9 sims
    mean_sim = (
        pair_df.groupby(["face_id", "question_idx"], as_index=False)["pair_similarity"]
               .mean()
               .rename(columns={"pair_similarity": "rag_nonrag_similarity"})
    )

    # Merge mean similarity back to original rows
    out = df.merge(mean_sim, on=["face_id", "question_idx"], how="left")

    # Save
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV}")
    # Diagnostics
    per_key_counts = pair_df.groupby(["face_id", "question_idx"]).size().reset_index(name="num_pairs")
    total_pairs = int(per_key_counts["num_pairs"].sum())
    print(f"Total pair rows compared: {total_pairs}")
    print(f"Unique keys with similarity: {mean_sim.shape[0]}")
    # Optional sanity: how many keys hit full 9 comparisons?
    full9 = (per_key_counts["num_pairs"] == 9).sum()
    print(f"Keys with full 3×3 coverage (9 pairs): {full9}")

if __name__ == "__main__":
    main()
