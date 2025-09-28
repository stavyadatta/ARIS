# === Config ===
INPUT_CSV  = "/workspace/7000s_rag_benchmark_results.csv"
OUTPUT_CSV = "7000s_rag_benchmark_results_with_similarity.csv"

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

def to_bool_is_rag(x):
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    return s in {"true", "t", "1", "yes", "y"}

def main():
    # Pick device (GPU if available)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # Load
    df = pd.read_csv(INPUT_CSV)

    # Basic checks
    for col in ["face_id", "question_idx", "is_rag", "response_text"]:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    # Normalise is_rag
    df["is_rag_bool"] = df["is_rag"].apply(to_bool_is_rag)

    # Split into RAG and non-RAG tables
    rag_df = (
        df[df["is_rag_bool"] == True]
        [["face_id", "question_idx", "response_text"]]
        .rename(columns={"response_text": "rag_text"})
    )
    nrag_df = (
        df[df["is_rag_bool"] == False]
        [["face_id", "question_idx", "response_text"]]
        .rename(columns={"response_text": "nonrag_text"})
    )

    # Cartesian pair all RAG with all non-RAG for the same (face_id, question_idx)
    pair_df = rag_df.merge(nrag_df, on=["face_id", "question_idx"], how="inner")

    if pair_df.empty:
        out = df.copy()
        out["rag_nonrag_similarity"] = pd.NA
        out.to_csv(OUTPUT_CSV, index=False)
        print("No RAG/Non-RAG pairs found. Wrote file with empty similarity column.")
        return

    # Load model on GPU if available
    model = SentenceTransformer("google/embeddinggemma-300m", device=DEVICE)

    # Embed and compute similarity (tensors live on DEVICE)
    rag_emb  = model.encode(pair_df["rag_text"].tolist(), convert_to_tensor=True, device=DEVICE)
    nrag_emb = model.encode(pair_df["nonrag_text"].tolist(), convert_to_tensor=True, device=DEVICE)

    sims_diag = model.similarity(rag_emb, nrag_emb).diag().detach().cpu().numpy()
    pair_df["rag_nonrag_similarity_row"] = sims_diag

    # If multiple pairs per (face_id, question_idx), keep max similarity
    best_sim = (
        pair_df.groupby(["face_id", "question_idx"], as_index=False)["rag_nonrag_similarity_row"]
               .max()
               .rename(columns={"rag_nonrag_similarity_row": "rag_nonrag_similarity"})
    )

    # Merge similarity back to all original rows
    out = df.merge(best_sim, on=["face_id", "question_idx"], how="left")

    # Save
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV}")
    print(f"Pairs compared: {len(pair_df)}")
    print(f"Unique (face_id, question_idx) with similarity: {best_sim.shape[0]}")

if __name__ == "__main__":
    main()

