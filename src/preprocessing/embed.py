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


# --- EXECUTION ---

create_embeddings(website)