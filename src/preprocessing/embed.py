from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from src.utils.constants import Websites, PretrainedModels, PreprocessingConfig
from src.utils.logging_config import get_logger

logger = get_logger("PREPROCESSING")


def create_embeddings(df_name):
    df_input = PreprocessingConfig.STAGE_CHUNKED.format(website=df_name)
    df_output = PreprocessingConfig.STAGE_EMBEDDED.format(website=df_name)

    df = pd.read_csv(df_input).dropna(subset=["text"])
    model = PretrainedModels.sentence_embedder()

    if df_name in Websites.WEBSITES_UKRAINE_RUSSIA:
        filters = PretrainedModels.FILTERS_UKRAINE_RUSSIA
    elif df_name in Websites.WEBSITES_PALESTINE_ISRAEL:
        filters = PretrainedModels.FILTERS_PALESTINE_ISRAEL

    filter_embeddings = model.encode(filters, normalize_embeddings=True)

    logger.info(f"Generating embeddings for {len(df)} chunks...")
    embeddings = model.encode(df["text"].tolist(), show_progress_bar=True, normalize_embeddings=True)

    df["filter1"] = cosine_similarity(embeddings, filter_embeddings[0].reshape(1, -1)).flatten().round(3)
    df["filter2"] = cosine_similarity(embeddings, filter_embeddings[1].reshape(1, -1)).flatten().round(3)
    df["filter3"] = cosine_similarity(embeddings, filter_embeddings[2].reshape(1, -1)).flatten().round(3)

    df["embedding"] = embeddings.tolist()
    df.to_csv(df_output, index=False)
    logger.info(f"Embeddings saved to {df_output}")


def main(website="alquds"):
    create_embeddings(website)


if __name__ == "__main__":
    main()
