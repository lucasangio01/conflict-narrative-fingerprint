from detoxify import Detoxify
import torch
import pandas as pd


def run_toxicity_scoring(website_name):
    detector = Detoxify(model_type='original', device='cuda')
    df = pd.read_csv(f"6_{website_name}_filtered.csv")

    texts = df["text"].astype(str).tolist()
    batch_size = 32
    tox_scores = []

    print(f"🧪 Scoring toxicity for {len(texts)} relevant chunks...")
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        preds = detector.predict(batch)["toxicity"]
        tox_scores.extend(preds)
        torch.cuda.empty_cache()

    df["toxicity"] = [round(s, 3) for s in tox_scores]

    df = df.reset_index().rename(columns={"index": "id"})
    desired_order = ["id", "title", "date", "filter1", "filter2", "filter3", "toxicity", "text", "embedding"]
    df = df[desired_order].sort_values(by="date", ascending=False)

    final_output = f"{website_name}_final.csv"
    df.to_csv(final_output, index=False)
    print(f"🏁 PIPELINE COMPLETE! Final file: {final_output}")


# --- EXECUTION ---

website = "alquds"
run_toxicity_scoring(website)