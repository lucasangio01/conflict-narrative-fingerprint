import pandas as pd
import re
from src.utils.constants import PreprocessingConfig
from src.utils.logging_config import get_logger

logger = get_logger("PREPROCESSING")


def chunk_resolved_data(website_name, max_chars=PreprocessingConfig.MAX_CHUNK_CHARS):
    input_file = PreprocessingConfig.STAGE_RESOLVED.format(website=website_name)
    df = pd.read_csv(input_file)

    rows = []
    stop_marker = PreprocessingConfig.STOP_MARKER

    logger.info(f"Chunking resolved text for {len(df)} articles...")

    for _, row in df.iterrows():
        title, date = row["title"], row["date"]
        text = str(row["resolved_text"])

        if stop_marker in text:
            text = text.split(stop_marker)[0]

        sentences = re.split(PreprocessingConfig.SENTENCE_SPLIT_REGEX, text.strip())
        current_chunk = ""
        for sent in sentences:
            if not sent.strip(): continue

            if len(current_chunk) + len(sent) > max_chars and current_chunk:
                rows.append({"title": title, "date": date, "text": current_chunk.strip()})
                current_chunk = sent
            else:
                current_chunk += " " + sent

        if current_chunk.strip():
            rows.append({"title": title, "date": date, "text": current_chunk.strip()})

    chunked_df = pd.DataFrame(rows)
    output_file = PreprocessingConfig.STAGE_CHUNKED.format(website=website_name)
    chunked_df.to_csv(output_file, index=False)
    logger.info(f"Created {len(chunked_df)} chunks. Saved to {output_file}")
    return chunked_df


def main(website="alquds"):
    return chunk_resolved_data(website)


if __name__ == "__main__":
    main()
