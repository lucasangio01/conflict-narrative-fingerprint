import pandas as pd
import re


def chunk_resolved_data(website_name, max_chars=800):
    input_file = f"3_{website_name}_resolved.csv"
    df = pd.read_csv(input_file)

    rows = []
    stop_marker = "The use of site materials is allowed only"

    print(f"✂️ Chunking resolved text for {len(df)} articles...")

    for _, row in df.iterrows():
        title, date = row["title"], row["date"]
        text = str(row["resolved_text"])

        if stop_marker in text:
            text = text.split(stop_marker)[0]

        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
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
    output_file = f"4_{website_name}_chunked.csv"
    chunked_df.to_csv(output_file, index=False)
    print(f"✅ Created {len(chunked_df)} chunks. Saved to {output_file}")
    return chunked_df


# --- EXECUTION ---

website = "alquds"
df_chunked = chunk_resolved_data(website)