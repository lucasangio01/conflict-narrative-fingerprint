from detoxify import Detoxify
import torch
import pandas as pd
from src.utils.constants import PreprocessingConfig


def run_toxicity_scoring(website_name):
    detector = Detoxify(model_type='original', device='cuda')
    df = pd.read_csv(PreprocessingConfig.STAGE_FILTERED.format(website=website_name))

    texts = df["text"].astype(str).tolist()
    batch_size = PreprocessingConfig.TOXICITY_BATCH_SIZE
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

    final_output = PreprocessingConfig.STAGE_FINAL.format(website=website_name)
    df.to_csv(final_output, index=False)
    print(f"🏁 PIPELINE COMPLETE! Final file: {final_output}")


def main(website="alquds"):
    run_toxicity_scoring(website)


if __name__ == "__main__":
    main()