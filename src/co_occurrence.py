import pandas as pd
import spacy
import numpy as np
from tqdm import tqdm
from collections import Counter
from itertools import combinations
import warnings

warnings.filterwarnings('ignore')


website      = "alquds"
clean_data_file = f"{website}_final.csv"
MIN_COUNT    = 3

nlp = spacy.load("en_core_web_lg")

SYNONYM_MAP = {
    # --- USA / WEST ---
    "the united states": "usa", "u.s.": "usa", "united states": "usa",
    "washington": "usa", "america": "usa", "state department": "usa",
    "the white house": "usa", "brussels": "eu", "european union": "eu",

    # --- UKRAINE / RUSSIA ---
    "kyiv": "kiev",
    "vladimir zelensky": "zelensky", "volodymyr zelensky": "zelensky",
    "andriy yermak": "yermak", "andriy yermak's": "yermak",
    "the office of the president": "ukraine", "office of the president": "ukraine",
    "vladimir putin": "putin",
    "russian federation": "russia", "the kremlin": "kremlin", "ussr": "russia",
    "warsaw": "poland", "berlin": "germany", "paris": "france",

    # --- ISRAEL / PALESTINE ---
    "the state of israel": "israel", "the knesset": "israel",
    "tel aviv": "israel", "judea and samaria": "israel",
    "the zionist entity": "israel",
    "benjamin netanyahu": "netanyahu", "bibi": "netanyahu",
    "israeli defense forces": "idf", "iaf": "idf",
    "israeli army": "idf", "occupation forces": "idf",
    "the gaza strip": "gaza", "gaza strip": "gaza", "the strip": "gaza",
    "mahmoud abbas": "abbas", "abu mazen": "abbas",
    "mansour abbas": "mansour_abbas",       # Israeli-Arab politician — kept distinct from Mahmoud Abbas
    "palestinian authority": "pa", "the palestinian authority": "pa",
    "hamas movement": "hamas",
    "yahya sinwar": "sinwar", "ismail haniyeh": "haniyeh",
    "al-qassam brigades": "qassam", "izz ad-din al-qassam": "qassam",
    "the resistance": "militants", "armed groups": "militants",
    "the west bank": "palestine", "west bank": "palestine",
    "the state of palestine": "palestine", "east jerusalem": "palestine",
    "oslo accords": "pa",                   # specific form only — not bare "oslo"
    "united nations relief": "unrwa", "unrwa": "un",
    "international court of justice": "icj",
    "international court": "icj",           # specific — not bare "court"
    "türkiye": "turkey",

    # --- IRAN AXIS ---
    "the islamic republic": "iran", "tehran": "iran",
    "irgc": "iran", "khamenei": "iran",
}


RU_UK_BASE = {
    # Russian political/state
    "russia": "RU_POLITICAL", "putin": "RU_POLITICAL",
    "moscow": "RU_POLITICAL", "kremlin": "RU_POLITICAL",

    # Ukrainian political/state
    "ukraine": "UKR_POLITICAL", "zelensky": "UKR_POLITICAL",
    "kiev": "UKR_POLITICAL", "yermak": "UKR_POLITICAL",
    "zaporizhzhia": "UKR_POLITICAL", "crimea": "UKR_POLITICAL",
    "kharkiv": "UKR_POLITICAL", "donetsk": "UKR_POLITICAL",

    # Western actors
    "usa": "WEST_ACTORS", "nato": "WEST_ACTORS", "eu": "WEST_ACTORS",
    "biden": "WEST_ACTORS", "trump": "WEST_ACTORS",
    "poland": "WEST_ACTORS", "germany": "WEST_ACTORS", "france": "WEST_ACTORS",

    # Military (specific named units only — no generic "army")
    "wagner": "RU_MILITARY",
    "afu": "UKR_MILITARY", "azov": "UKR_MILITARY",

    # International
    "china": "INTL_ACTORS", "un": "INTL_ACTORS",
    "iaea": "INTL_ACTORS", "turkey": "INTL_ACTORS",

    "civilians": "CIVILIANS"
}

IZ_PA_BASE = {
    # Israeli political/state
    "netanyahu": "ISR_POLITICAL", "israel": "ISR_POLITICAL",
    "knesset": "ISR_POLITICAL", "mansour_abbas": "ISR_POLITICAL",
    "gallant": "ISR_POLITICAL", "gantz": "ISR_POLITICAL",

    # Israeli military/security
    "idf": "ISR_MILITARY", "mossad": "ISR_MILITARY", "shin bet": "ISR_MILITARY",

    # Palestinian political
    "abbas": "PAL_POLITICAL", "pa": "PAL_POLITICAL", "palestine": "PAL_POLITICAL",

    # Palestinian organisations
    "hamas": "PAL_ORG", "haniyeh": "PAL_ORG", "sinwar": "PAL_ORG", "pij": "PAL_ORG",

    # Palestinian resistance / territory
    "militants": "PAL_RESISTANCE", "qassam": "PAL_RESISTANCE", "gaza": "PAL_RESISTANCE",

    # International actors
    "usa": "INTL_ACTORS", "un": "INTL_ACTORS", "icj": "INTL_ACTORS",
    "biden": "INTL_ACTORS", "trump": "INTL_ACTORS",
    "iran": "INTL_ACTORS", "lebanon": "INTL_ACTORS", "turkey": "INTL_ACTORS",
    "syria": "INTL_ACTORS", "hezbollah": "INTL_ACTORS",
    "saudi arabia": "INTL_ACTORS", "qatar": "INTL_ACTORS", "egypt": "INTL_ACTORS",

    # Civil
    "civilians": "CIVILIANS", "hostages": "CIVILIANS", "settlers": "SETTLERS"
}


is_il_pa = website in ["ynet", "ynet_global", "alquds", "jpost", "aljazeera"]
active_entities = IZ_PA_BASE if is_il_pa else RU_UK_BASE
theater_name    = "Israel-Palestine" if is_il_pa else "Russia-Ukraine"

search_keys = sorted(list(set(list(SYNONYM_MAP.keys()) + list(active_entities.keys()))), key=len, reverse=True)

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
            clean = SYNONYM_MAP.get(match, match)
            if clean in active_entities:
                return clean

    return None


df = pd.read_csv(clean_data_file)
texts_list = df['text'].dropna().astype(str).tolist()

pair_counts       = Counter()
entity_freq       = Counter()
raw_audit_counter = Counter()
total_sentences   = 0

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

    if count < MIN_COUNT:
        continue

    freq_a = entity_freq[ent_a]
    freq_b = entity_freq[ent_b]

    union   = freq_a + freq_b - count
    jaccard = round(count / union, 4) if union > 0 else 0.0

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
        "pmi":      pmi
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
    clean = SYNONYM_MAP.get(name, name)
    return clean in active_entities

audit_df = (pd.DataFrame(raw_audit_counter.items(), columns=['Name', 'Freq']).sort_values('Freq', ascending=False))
audit_df['Recognized'] = audit_df['Name'].apply(is_recognized)
unrecognized = audit_df[audit_df['Recognized'] == False]

print(f"\n--- AUDIT: TOP 20 UNRECOGNIZED NAMED ENTITIES ---")
print("(Frequent names not in your active entity dictionary)")
print("(Use this list to expand SYNONYM_MAP)\n")
print(unrecognized[['Name', 'Freq']].head(20).to_string(index=False))


df_results.to_csv(f"{website}_cooccurrence_pairs.csv", index=False)
label_results.to_csv(f"{website}_cooccurrence_labels.csv", index=False)
audit_df.to_csv(f"{website}_audit_names.csv", index=False)

print(f"\n✅ Saved:")
print(f"   {website}_cooccurrence_pairs.csv   — full pair matrix with Jaccard + PMI")
print(f"   {website}_cooccurrence_labels.csv  — label-level aggregation")
print(f"   {website}_audit_names.csv          — full NER audit for dictionary expansion")
