from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd


def create_embeddings(df_name):
    df_input = f"4_{df_name}_chunked.csv"
    df_output = f"5_{df_name}_embedded.csv"

    df = pd.read_csv(df_input).dropna(subset = ["text"])
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    if df_name in ["kpru", "ukpravda", "rt", "liganet"]:
        filters = ["Russia Ukraine conflict", "Ukraine Russia relations", "Eastern Europe crisis"]
    elif df_name in ["jpost", "alquds", "ynet", "ynet_global"]:
        filters = ["Israel Palestine conflict", "Gaza West Bank situation", "Middle East political tensions"]

    filter_embeddings = model.encode(filters, normalize_embeddings=True)

    print(f"🛰️ Generating embeddings for {len(df)} chunks...")
    embeddings = model.encode(df["text"].tolist(), show_progress_bar=True, normalize_embeddings=True)

    df["filter1"] = cosine_similarity(embeddings, filter_embeddings[0].reshape(1, -1)).flatten().round(3)
    df["filter2"] = cosine_similarity(embeddings, filter_embeddings[1].reshape(1, -1)).flatten().round(3)
    df["filter3"] = cosine_similarity(embeddings, filter_embeddings[2].reshape(1, -1)).flatten().round(3)

    df["embedding"] = embeddings.tolist()
    df.to_csv(df_output, index=False)
    print(f"✅ Embeddings saved to {df_output}")


def apply_narrative_filter(website_name):
    df = pd.read_csv(f"5_{website_name}_embedded.csv")
    threshold = 0.4
    filtered_df = df[(df["filter1"] > threshold) | (df["filter2"] > threshold) | (df["filter3"] > threshold)].copy()

    output_file = f"6_{website_name}_filtered.csv"
    filtered_df.to_csv(output_file, index=False)

    retention = (len(filtered_df) / len(df)) * 100
    print(f"🎯 Narrative Filter: Kept {len(filtered_df)} chunks ({retention:.2f}% of total).")
    return filtered_df



# --- EXECUTION ---

website = "alquds"
create_embeddings(website)
df_filtered = apply_narrative_filter(website)