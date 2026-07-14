import pandas as pd
from src.utils.constants import PreprocessingConfig


def apply_narrative_filter(website_name):
    df = pd.read_csv(f"5_{website_name}_embedded.csv")
    threshold = PreprocessingConfig.FILTER_THRESHOLD
    filtered_df = df[(df["filter1"] > threshold) | (df["filter2"] > threshold) | (df["filter3"] > threshold)].copy()

    output_file = f"6_{website_name}_filtered.csv"
    filtered_df.to_csv(output_file, index=False)

    retention = (len(filtered_df) / len(df)) * 100
    print(f"🎯 Narrative Filter: Kept {len(filtered_df)} chunks ({retention:.2f}% of total).")
    return filtered_df


def main(website="alquds"):
    return apply_narrative_filter(website)


if __name__ == "__main__":
    main()
