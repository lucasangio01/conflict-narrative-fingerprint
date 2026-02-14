from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from src.utils.constants import Embedding, Websites


def create_embeddings(df_name):
    df_input = f"{df_name}_chunked.csv"
    df_output = f"{df_name}_embedded.csv"

    df = pd.read_csv(df_input)
    df = df.dropna(subset = ["text"])
    df["text"] = df["text"].astype(str)

    if df_name in Websites.WEBSITES_UKRAINE_RUSSIA:
        filters = Embedding.FILTERS_UKRAINE_RUSSIA
    elif df_name in Websites.WEBSITES_PALESTINE_ISRAEL:
        filters = Embedding.FILTERS_PALESTINE_ISRAEL
    
    filter_embeddings = Embedding.SENTENCE_EMBEDDER.encode(filters, normalize_embeddings=True).reshape(3, 384)
    text_embeddings = Embedding.SENTENCE_EMBEDDER.encode(df["text"].tolist(), show_progress_bar = True, normalize_embeddings = True)

    df["filter1"] = cosine_similarity(text_embeddings, filter_embeddings[0].reshape(1, -1)).flatten().round(3)
    df["filter2"] = cosine_similarity(text_embeddings, filter_embeddings[1].reshape(1, -1)).flatten().round(3)
    df["filter3"] = cosine_similarity(text_embeddings, filter_embeddings[2].reshape(1, -1)).flatten().round(3)

    df["embedding"] = text_embeddings.tolist()
    df.drop(columns=["text"]).to_csv(df_output, index = False)


create_embeddings(df_name="ukpravda")
