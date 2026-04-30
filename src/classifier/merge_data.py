import pandas as pd
import numpy as np
import spacy
import torch
import warnings
from tqdm import tqdm
from transformers import pipeline
from emfdscore.scoring import score_docs

warnings.filterwarnings('ignore')


SOURCE_LABELS = {"kpru": 0, "rt": 0, "jpost": 0, "ynet": 0, "ynet_global": 0, "ukpravda": 1, "liganet": 1, "alquds": 1}
THEATER_MAP = {"kpru": "RU_UK", "rt": "RU_UK", "ukpravda": "RU_UK", "liganet": "RU_UK", "jpost": "IL_PA", "ynet": "IL_PA", "ynet_global": "IL_PA", "alquds": "IL_PA"}

SOURCE_INGROUP = {
    "kpru":        lambda l: "INGROUP"  if l.startswith("RU_")  else ("OUTGROUP" if l.startswith("UKR_") else "OTHER"),
    "rt":          lambda l: "INGROUP"  if l.startswith("RU_")  else ("OUTGROUP" if l.startswith("UKR_") else "OTHER"),
    "ukpravda":    lambda l: "INGROUP"  if l.startswith("UKR_") else ("OUTGROUP" if l.startswith("RU_")  else "OTHER"),
    "liganet":     lambda l: "INGROUP"  if l.startswith("UKR_") else ("OUTGROUP" if l.startswith("RU_")  else "OTHER"),
    "jpost":       lambda l: "INGROUP"  if l.startswith("ISR_") else ("OUTGROUP" if l.startswith("PAL_") else "OTHER"),
    "ynet":        lambda l: "INGROUP"  if l.startswith("ISR_") else ("OUTGROUP" if l.startswith("PAL_") else "OTHER"),
    "ynet_global": lambda l: "INGROUP"  if l.startswith("ISR_") else ("OUTGROUP" if l.startswith("PAL_") else "OTHER"),
    "alquds":      lambda l: "INGROUP"  if l.startswith("PAL_") else ("OUTGROUP" if l.startswith("ISR_") else "OTHER"),
}

RU_UK_BASE = {
    "russia": "RU_POLITICAL", "putin": "RU_POLITICAL", "kremlin": "RU_POLITICAL", "moscow": "RU_POLITICAL",
    "ukraine": "UKR_POLITICAL", "zelensky": "UKR_POLITICAL", "kiev": "UKR_POLITICAL", "kyiv": "UKR_POLITICAL",
    "usa": "WEST_ACTORS", "nato": "WEST_ACTORS", "eu": "WEST_ACTORS",
    "army": "RU_MILITARY", "wagner": "RU_MILITARY",
    "afu": "UKR_MILITARY", "azov": "UKR_MILITARY",
    "civilians": "CIVILIANS",
}

IZ_PA_BASE = {
    "netanyahu": "ISR_POLITICAL", "israel": "ISR_POLITICAL", "knesset": "ISR_POLITICAL",
    "idf": "ISR_MILITARY", "mossad": "ISR_MILITARY",
    "abbas": "PAL_POLITICAL", "pa": "PAL_POLITICAL", "palestine": "PAL_POLITICAL",
    "hamas": "PAL_ORG", "sinwar": "PAL_ORG",
    "militants": "PAL_RESISTANCE", "qassam": "PAL_RESISTANCE", "gaza": "PAL_RESISTANCE",
    "usa": "INTL_ACTORS", "un": "INTL_ACTORS",
    "civilians": "CIVILIANS", "hostages": "CIVILIANS",
}

SYNONYM_MAP = {
    "the united states": "usa", "u.s.": "usa", "washington": "usa", "america": "usa",
    "kyiv": "kiev", "vladimir zelensky": "zelensky", "volodymyr zelensky": "zelensky",
    "vladimir putin": "putin", "russian federation": "russia", "the kremlin": "kremlin",
    "benjamin netanyahu": "netanyahu", "bibi": "netanyahu",
    "israeli defense forces": "idf", "israeli army": "idf", "occupation forces": "idf",
    "the gaza strip": "gaza", "gaza strip": "gaza",
    "palestinian authority": "pa", "hamas movement": "hamas",
    "the resistance": "militants", "fighters": "militants",
    "the west bank": "palestine",
}

VIOLENT_VERBS = {"attack", "bomb", "kill", "destroy", "invade", "strike", "shell", "target", "raid", "fire", "massacre", "execute", "wound", "siege"}
MODAL_VERBS   = {"may", "might", "could", "would", "should", "can"}
HEDGES        = {"seem", "appear", "allege", "probable", "suggest", "warn", "claim", "reportedly", "allegedly"}
NEGATION_TOKENS = {"not", "never", "no", "neither", "nor", "deny", "refuse", "reject", "false", "untrue", "deny"}
DEHUMAN_WORDS = {
    "rat", "vermin", "beast", "animal", "hyena", "wolf", "parasite",
    "insect", "viper", "pig", "monkey", "cancer", "plague", "virus",
    "tumor", "toxic", "poison", "barbaric", "monster", "scum",
    "demon", "savage", "primitive", "uncivilized",
}

AXIS_SEEDS = {
    "competence": {
        "high": ["powerful", "strategic", "efficient", "capable", "advanced", "strong"],
        "low":  ["weak", "failing", "disorganized", "incompetent", "chaotic", "ineffective"],
    },
    "morality": {
        "high": ["righteous", "just", "innocent", "heroic", "moral", "civilized", "peaceful"],
        "low":  ["cruel", "evil", "terrorist", "murderous", "corrupt", "barbaric", "aggressive"],
    },
}

MIN_ENTITY_HITS = 2
MAX_CHUNKS_PER_SOURCE = 3000


print("🔧 Loading models...")
nlp = spacy.load("en_core_web_lg")
sentiment_task = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", device=0 if torch.cuda.is_available() else -1)

def get_axis_vector(seeds):
    h = np.mean([nlp(w).vector for w in seeds["high"] if nlp(w).has_vector], axis=0)
    l = np.mean([nlp(w).vector for w in seeds["low"]  if nlp(w).has_vector], axis=0)
    return h - l

comp_vec  = get_axis_vector(AXIS_SEEDS["competence"])
moral_vec = get_axis_vector(AXIS_SEEDS["morality"])

def axis_projection(text, vec):
    doc = nlp(text)
    if not doc.has_vector or doc.vector_norm == 0: return 0.0
    return float(np.dot(doc.vector, vec) / np.linalg.norm(vec))


def soft_mask_text(text, active_entities, search_keys):
    """Replaces names with [TAGS]"""
    doc = nlp(text)
    tokens = [t.text for t in doc]
    for i, token in enumerate(doc):
        window = " ".join([t.text.lower() for t in doc[max(0, i - 1): min(len(doc), i + 2)]])
        match = next((k for k in search_keys if k in window), None)
        if match:
            clean = SYNONYM_MAP.get(match, match)
            if clean in active_entities:
                tokens[i] = f"[{active_entities[clean]}]"
    return " ".join(tokens)


def extract_structural_features(text, tag_map, perspective_fn):
    """
    Looks specifically for [TAGS] in the masked text.
    """
    doc = nlp(text)
    ingroup_agent = ingroup_patient = outgroup_agent = outgroup_patient = 0
    passive_count = total_verbs = violence_ingroup = violence_outgroup = entity_hit_count = 0
    negation_outgroup_sents = outgroup_sents_total = dehuman_hits = adj_total = 0
    sent_ingroup_idx, sent_outgroup_idx, uncertainty_scores = [], [], []
    adj_comp_in, adj_comp_out, adj_moral_in, adj_moral_out = [], [], [], []

    for s_idx, sent in enumerate(doc.sents):
        h_count = sum(1 for t in sent if t.lemma_.lower() in MODAL_VERBS or t.lemma_.lower() in HEDGES)
        uncertainty_scores.append(h_count / max(len(sent), 1))

        has_ig = has_og = False
        has_neg = any(t.dep_ == "neg" or t.lemma_.lower() in NEGATION_TOKENS for t in sent)

        for token in sent:
            clean_token = token.text.replace("[", "").replace("]", "")
            if clean_token in tag_map:
                entity_label = tag_map[clean_token]
                role = perspective_fn(entity_label)
                entity_hit_count += 1

                if role == "INGROUP": has_ig = True
                elif role == "OUTGROUP": has_og = True

                # Agency
                if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
                    total_verbs += 1
                    verb = token.head.lemma_.lower()
                    if role == "INGROUP":
                        ingroup_agent += 1
                        if verb in VIOLENT_VERBS: violence_ingroup += 1
                    elif role == "OUTGROUP":
                        outgroup_agent += 1
                        if verb in VIOLENT_VERBS: violence_outgroup += 1
                elif token.dep_ in ("dobj", "nsubjpass") and token.head.pos_ == "VERB":
                    if token.dep_ == "nsubjpass": passive_count += 1
                    if role == "INGROUP": ingroup_patient += 1
                    elif role == "OUTGROUP": outgroup_patient += 1

                # Adjectives
                for child in token.children:
                    if child.pos_ == "ADJ":
                        adj_total += 1
                        c_s = axis_projection(child.text, comp_vec)
                        m_s = axis_projection(child.text, moral_vec)
                        if role == "INGROUP":
                            adj_comp_in.append(c_s); adj_moral_in.append(m_s)
                        elif role == "OUTGROUP":
                            adj_comp_out.append(c_s); adj_moral_out.append(m_s)
                        if child.lemma_.lower() in DEHUMAN_WORDS: dehuman_hits += 1

        if has_ig: sent_ingroup_idx.append(s_idx)
        if has_og:
            sent_outgroup_idx.append(s_idx)
            outgroup_sents_total += 1
            if has_neg: negation_outgroup_sents += 1

    # Ratio Math
    ig_ar = ingroup_agent / max(ingroup_agent + ingroup_patient, 1)
    og_ar = outgroup_agent / max(outgroup_agent + outgroup_patient, 1)

    return {
        "agency_asymmetry": ig_ar - og_ar,
        "passive_voice_rate": passive_count / max(total_verbs, 1),
        "violence_asymmetry": (violence_outgroup/max(outgroup_agent,1)) - (violence_ingroup/max(ingroup_agent,1)),
        "uncertainty_score": np.mean(uncertainty_scores) if uncertainty_scores else 0.0,
        "negation_rate": negation_outgroup_sents / max(outgroup_sents_total, 1),
        "competence_asymmetry": np.mean(adj_comp_in or [0]) - np.mean(adj_comp_out or [0]),
        "morality_asymmetry": np.mean(adj_moral_in or [0]) - np.mean(adj_moral_out or [0]),
        "dehumanization_index": dehuman_hits / max(adj_total, 1),
        "ingroup_outgroup_ratio": len(sent_ingroup_idx) / max(len(sent_outgroup_idx), 1),
        "_sent_ingroup_idx": sent_ingroup_idx, "_sent_outgroup_idx": sent_outgroup_idx,
        "_sents": [s.text for s in doc.sents], "_entity_hit_count": entity_hit_count
    }


all_rows = []

for website, label in SOURCE_LABELS.items():
    print(f"\n📂 [{website}] label={label}")
    try:
        df = pd.read_csv(f"{website}_final.csv").dropna(subset=["text"])
    except: continue

    if len(df) > MAX_CHUNKS_PER_SOURCE: df = df.sample(n=MAX_CHUNKS_PER_SOURCE, random_state=42)

    theater = THEATER_MAP[website]
    active_ents = IZ_PA_BASE if theater == "IL_PA" else RU_UK_BASE
    perspective = SOURCE_INGROUP[website]

    tag_map = {v: v for v in set(active_ents.values())}

    search_keys = sorted(list(set(list(SYNONYM_MAP.keys()) + list(active_ents.keys()))), key=len, reverse=True)
    raw_texts = df["text"].astype(str).tolist()

    print("🧼 Masking entities...")
    masked_texts = [soft_mask_text(t, active_ents, search_keys) for t in tqdm(raw_texts)]

    print(f"🔬 Structural extraction...")
    structural = [extract_structural_features(t, tag_map, perspective) for t in tqdm(masked_texts)]

    valid_idx = [i for i, f in enumerate(structural) if f["_entity_hit_count"] >= MIN_ENTITY_HITS]
    print(f"🔎 Kept {len(valid_idx)}/{len(raw_texts)} chunks.")

    if not valid_idx: continue

    structural = [structural[i] for i in valid_idx]
    texts_to_score = [masked_texts[i] for i in valid_idx]
    df_filtered = df.iloc[valid_idx]

    print("🤖 Sentiment & MFT...")
    chunk_sents = [p['score'] if p['label']=='positive' else -p['score'] for p in sentiment_task([t[:512] for t in texts_to_score])]
    mft_df = score_docs(pd.DataFrame({"text": texts_to_score}), 'emfd', 'all', 'bow', 'sentiment', len(texts_to_score))

    for i, feat in enumerate(structural):
        feat.update({
            "label": label, "source": website, "theater": theater,
            "chunk_sentiment": chunk_sents[i],
            "toxicity_score": float(df_filtered.iloc[i].get("toxicity", 0.0))
        })

        for col in mft_df.columns:
            if any(f in col for f in ['care', 'fair', 'loy', 'auth', 'sanc']):
                feat[f"mft_{col}"] = mft_df.iloc[i][col]
        all_rows.append(feat)

master_df = pd.DataFrame(all_rows).drop(columns=["_sent_ingroup_idx", "_sent_outgroup_idx", "_sents", "_entity_hit_count"])
master_df.to_csv("classification_data.csv", index=False)
print("\n✅ DONE. File saved.")