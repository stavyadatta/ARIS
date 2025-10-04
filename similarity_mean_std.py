# === Config ===
INPUT_CSV  = "./7000s_rag_benchmark_results_repitition_similarity.csv"

import pandas as pd

def main():
    # Load
    df = pd.read_csv(INPUT_CSV)

    # Check required columns
    for col in ["message_len", "rag_nonrag_similarity"]:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    # Drop rows without similarity score
    df_valid = df.dropna(subset=["rag_nonrag_similarity"]).copy()

    # Group by message_len and calculate mean + std
    summary = (
        df_valid.groupby("message_len")["rag_nonrag_similarity"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "avg_similarity", "std": "std_similarity"})
    )

    # Compute overall average + std across all rows
    overall_mean = df_valid["rag_nonrag_similarity"].mean()
    overall_std  = df_valid["rag_nonrag_similarity"].std()

    # Append as a new row
    overall_row = pd.DataFrame({
        "message_len": ["ALL"],
        "avg_similarity": [overall_mean],
        "std_similarity": [overall_std]
    })
    print("The len of dataframe is ", len(summary))
    summary = pd.concat([summary, overall_row], ignore_index=True)

    # Print neatly to terminal
    pd.set_option("display.float_format", "{:.4f}".format)
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()

