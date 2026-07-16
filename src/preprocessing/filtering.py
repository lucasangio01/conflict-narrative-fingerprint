import pandas as pd
from src.utils.constants import PreprocessingConfig
from src.utils.logging_config import get_logger

logger = get_logger("PREPROCESSING")


def apply_narrative_filter(website_name):
    df = pd.read_csv(PreprocessingConfig.STAGE_EMBEDDED.format(website=website_name))
    threshold = PreprocessingConfig.FILTER_THRESHOLD
    filtered_df = df[(df["filter1"] > threshold) | (df["filter2"] > threshold) | (df["filter3"] > threshold)].copy()

    output_file = PreprocessingConfig.STAGE_FILTERED.format(website=website_name)
    filtered_df.to_csv(output_file, index=False)

    retention = (len(filtered_df) / len(df)) * 100
    logger.info(f"Narrative filter: kept {len(filtered_df)} chunks ({retention:.2f}% of total).")
    return filtered_df


def main(website="alquds"):
    return apply_narrative_filter(website)


if __name__ == "__main__":
    main()
