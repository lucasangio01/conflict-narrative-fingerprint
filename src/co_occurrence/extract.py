import pandas as pd
import spacy
import numpy as np
from tqdm import tqdm
from collections import Counter
from itertools import combinations
import warnings
from src.utils.constants import NamesDicts, Websites, CoOccurrenceConfig, PretrainedModels, PreprocessingConfig


def main(website="alquds"):
    warnings.filterwarnings('ignore')

    clean_data_file = PreprocessingConfig.STAGE_FINAL.format(website=website)

    # en_core_web_lg ships with a dependency parser that handles sentence segmentation.
    # Sentencizer is NOT added here -- it would conflict with the parser and is redundant.
    nlp = spacy.load(PretrainedModels.SPACY_MODEL_LG)

    is_il_pa        = website in Websites.WEBSITES_PALESTINE_ISRAEL
    active_entities = NamesDicts.IZ_PA_BASE if is_il_pa else NamesDicts.RU_UK_BASE
    theater_name    = Websites.THEATER_IL_PA if is_il_pa else Websites.THEATER_RU_UK

    # Sort by length descending so multi-word keys are matched before their substrings
    search_keys = sorted(list(set(list(NamesDicts.SYNONYM_MAP.keys()) + list(active_entities.keys()))), key=len, reverse=True)

    print(f"🌍 Theater Detected: {theater_name} — website: {website}")

    def get_entity_from_token(token, doc):
        """
        Matches a token to a known entity using two strategies in order:
          1. Full noun chunk the token belongs to  (best for multi-word entities)
          2. ±3 token window fallback              (catches tokens outside noun chunks)
        Returns the canonical entity key (str) or None.
        The single-token fallback is intentionally omitted — it is always a strict
        subset of the window and never fires independently.
        """
        chunk_text = next((chunk.text.lower() for chunk in doc.noun_chunks if token.i in range(chunk.start, chunk.end)), None)
        window_text = " ".join(t.text.lower() for t in doc[max(0, token.i - 3): min(len(doc), token.i + 4)])

        for candidate_text in filter(None, [chunk_text, window_text]):
            match = next((k for k in search_keys if k in candidate_text), None)
            if match:
                clean = NamesDicts.SYNONYM_MAP.get(match, match)
                if clean in active_entities:
                    return clean

        return None

    df = pd.read_csv(clean_data_file)
    texts_list = df['text'].dropna().astype(str).tolist()

    pair_counts       = Counter()   # (ent_a, ent_b) → sentence co-occurrence count
    entity_freq       = Counter()   # entity → number of sentences it appears in
    raw_audit_counter = Counter()   # raw NER text → freq (for audit only)
    total_sentences   = 0           # true sentence count — required for correct PMI

    print(f"🚀 Processing {len(texts_list)} articles for {website}...")

    for doc in tqdm(nlp.pipe(texts_list, batch_size=32), total=len(texts_list)):

        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE"]:
                raw_audit_counter[ent.text.lower()] += 1

        for sent in doc.sents:
            total_sentences += 1
            found_in_sent = set()

            for token in sent:
                entity = get_entity_from_token(token, doc)
                if entity:
                    found_in_sent.add(entity)

            entity_freq.update(found_in_sent)

            if len(found_in_sent) >= 2:
                for pair in combinations(sorted(found_in_sent), 2):
                    pair_counts[pair] += 1

    print(f"   Processed {total_sentences:,} sentences across {len(texts_list):,} articles")

    results = []
    for (ent_a, ent_b), count in pair_counts.items():

        if count < CoOccurrenceConfig.MIN_COUNT:
            continue

        freq_a = entity_freq[ent_a]
        freq_b = entity_freq[ent_b]

        # Jaccard: intersection / union — how exclusively the two entities share
        # sentence appearances
        union   = freq_a + freq_b - count
        jaccard = round(count / union, 4) if union > 0 else 0.0

        # PMI: log2( P(a,b) / (P(a) * P(b)) ) = log2( count(a,b) * total_sentences / (freq_a * freq_b) )
        # Uses total_sentences as the correct denominator so absolute values are
        # meaningful and comparable across corpora with different sizes.
        pmi = round(np.log2((count * total_sentences) / (freq_a * freq_b + 1e-12)), 4)

        results.append({
            "entity_a": ent_a,
            "label_a":  active_entities[ent_a],
            "entity_b": ent_b,
            "label_b":  active_entities[ent_b],
            "count":    count,
            "freq_a":   freq_a,
            "freq_b":   freq_b,
            "jaccard":  jaccard,
            "pmi":      pmi,
        })

    df_results = pd.DataFrame(results).sort_values(by="count", ascending=False)

    label_results = (
        df_results
        .groupby(['label_a', 'label_b'])['count']
        .sum()
        .reset_index()
        .sort_values(by='count', ascending=False)
    )

    print(f"\n--- TOP 15 ENTITY PAIRS BY RAW COUNT: {website} ---")
    print(df_results[['entity_a', 'entity_b', 'count', 'jaccard', 'pmi']].head(15).to_string(index=False))

    print(f"\n--- TOP 15 ENTITY PAIRS BY PMI (surprising co-occurrences): {website} ---")
    pmi_df = df_results.sort_values('pmi', ascending=False)
    print(pmi_df[['entity_a', 'entity_b', 'count', 'pmi']].head(15).to_string(index=False))

    print(f"\n--- TOP 10 LABEL CATEGORY RELATIONSHIPS: {website} ---")
    print(label_results.head(10).to_string(index=False))

    def is_recognized(name):
        clean = NamesDicts.SYNONYM_MAP.get(name, name)
        return clean in active_entities

    audit_df = (pd.DataFrame(raw_audit_counter.items(), columns=['Name', 'Freq']).sort_values('Freq', ascending=False))
    audit_df['Recognized'] = audit_df['Name'].apply(is_recognized)
    unrecognized = audit_df[audit_df['Recognized'] == False]

    print(f"\n--- AUDIT: TOP 20 UNRECOGNIZED NAMED ENTITIES ---")
    print("(Frequent names not in your active entity dictionary)")
    print("(Use this list to expand NamesDicts.SYNONYM_MAP in src/utils/constants.py)\n")
    print(unrecognized[['Name', 'Freq']].head(20).to_string(index=False))

    pairs_csv = CoOccurrenceConfig.PAIRS_CSV_PATTERN.format(website=website)
    labels_csv = CoOccurrenceConfig.LABELS_CSV_PATTERN.format(website=website)
    audit_csv = f"{website}_audit_names.csv"

    df_results.to_csv(pairs_csv, index=False)
    label_results.to_csv(labels_csv, index=False)
    audit_df.to_csv(audit_csv, index=False)

    print(f"\n✅ Saved:")
    print(f"   {pairs_csv}   — full pair matrix with Jaccard + PMI")
    print(f"   {labels_csv}  — label-level aggregation")
    print(f"   {audit_csv}          — full NER audit for dictionary expansion")

    return df_results, label_results, audit_df


if __name__ == "__main__":
    main()
