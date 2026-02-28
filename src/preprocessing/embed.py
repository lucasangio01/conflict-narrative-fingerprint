import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
from src.utils.constants import PretrainedModels, Websites



class EmbedText:
    def __init__(self, website):
        self.website = website
        if website in Websites.WEBSITES_PALESTINE_ISRAEL:
            self.filters = PretrainedModels.FILTERS_UKRAINE_RUSSIA
            self.df_chunked_path = f"../data/pal_isr/{website}_chunked.csv"
            self.df_embedded_path = f"../data/pal_isr/{website}_embedded.csv"
            self.df_filtered_path = f"../data/pal_isr/{website}_filtered.csv"
            self.df_toxicity_path = f"../data/pal_isr/{website}_toxicity.csv"
        else:
            self.filters = PretrainedModels.FILTERS_PALESTINE_ISRAEL
            self.df_chunked_path = f"../data/ukr_rus/{website}_chunked.csv"
            self.df_embedded_path = f"../data/ukr_rus/{website}_embedded.csv"
            self.df_filtered_path = f"../data/ukr_rus/{website}_filtered.csv"
            self.df_toxicity_path = f"../data/ukr_rus/{website}_toxicity.csv"


    def create_embeddings(self):
        self.df_chunked = pd.read_csv(self.df_chunked_path).dropna(subset = ["text"])
        self.df_chunked["text"] = self.df_chunked["text"].astype(str)
        filter_embeddings = PretrainedModels.SENTENCE_EMBEDDER.encode(self.filters, normalize_embeddings=True).reshape(3, 384)
        text_embeddings = PretrainedModels.SENTENCE_EMBEDDER.encode(self.df_chunked["text"].tolist(), show_progress_bar = True, normalize_embeddings = True)
        self.df_chunked["filter1"] = cosine_similarity(text_embeddings, filter_embeddings[0].reshape(1, -1)).flatten().round(3)
        self.df_chunked["filter2"] = cosine_similarity(text_embeddings, filter_embeddings[1].reshape(1, -1)).flatten().round(3)
        self.df_chunked["filter3"] = cosine_similarity(text_embeddings, filter_embeddings[2].reshape(1, -1)).flatten().round(3)
        self.df_chunked["embedding"] = text_embeddings.tolist()
        self.df_embedded = self.df_chunked.copy()
        self.df_embedded.to_csv(self.df_embedded_path, index = False)


    def filter_similarity(self):
        self.df_filtered = self.df_embedded[(self.df_embedded["filter1"] > 0.4) | (self.df_embedded["filter2"] > 0.4) | (self.df_embedded["filter3"] > 0.4)].copy()
        self.df_filtered.to_csv(self.df_filtered_path, index = False)

    
    def calculate_text_toxicity(self):
        text = self.df_filtered["text"].tolist()
        self.df_filtered["toxicity"] = PretrainedModels.TOXICITY_DETECTOR.predict(text)
        self.df_filtered.to_csv(self.df_toxicity_path, index = False)


if __name__ == "__main__":
    embed_text = EmbedText(website = "ukpravda")
    embed_text.create_embeddings()
    embed_text.filter_similarity()