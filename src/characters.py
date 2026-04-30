import pandas as pd
import spacy
from tqdm import tqdm
from transformers import pipeline
from emfdscore.scoring import score_docs
import numpy as np
import warnings
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings('ignore')


website = "alquds"
clean_data_file = f"{website}_final.csv"
MIN_OCCURRENCES = 1   

nlp = spacy.load("en_core_web_lg")

print("✨ Initializing RoBERTa Sentiment Transformer...")
try:
    sentiment_task = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", device=0, batch_size=16)
    print("   ✅ Running on GPU")
except Exception as e:
    print(f"   ⚠️  GPU unavailable ({e}), falling back to CPU")
    sentiment_task = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", device=-1)


SYNONYM_MAP = {
    # --- USA / WEST ---
    "the united states": "usa", "u.s.": "usa", "united states": "usa",
    "washington": "usa", "america": "usa", "state department": "usa",
    "the white house": "usa", "brussels": "eu", "european union": "eu",

    # --- UKRAINE / RUSSIA ---
    "kyiv": "kiev",
    "vladimir zelensky": "zelensky", "volodymyr zelensky": "zelensky",
    "andriy yermak": "yermak",
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
    "mansour abbas": "mansour_abbas",
    "palestinian authority": "pa", "the palestinian authority": "pa",
    "hamas movement": "hamas",
    "yahya sinwar": "sinwar", "ismail haniyeh": "haniyeh",
    "al-qassam brigades": "qassam", "izz ad-din al-qassam": "qassam",
    "the resistance": "militants", "armed groups": "militants",
    "the west bank": "palestine", "west bank": "palestine",
    "the state of palestine": "palestine", "east jerusalem": "palestine",
    "oslo accords": "pa",
    "unrwa": "un", "united nations relief": "un",
    "international court of justice": "icj", "international court": "icj",
    "türkiye": "turkey",

    # --- IRAN AXIS ---
    "the islamic republic": "iran", "tehran": "iran",
    "irgc": "iran", "khamenei": "iran",
}


RU_UK_BASE = {
    "russia": "RU_POLITICAL", "putin": "RU_POLITICAL",
    "moscow": "RU_POLITICAL", "kremlin": "RU_POLITICAL", "lavrov": "RU_POLITICAL",

    "ukraine": "UKR_POLITICAL", "zelensky": "UKR_POLITICAL",
    "kiev": "UKR_POLITICAL", "yermak": "UKR_POLITICAL",
    "zaporizhzhia": "UKR_POLITICAL", "crimea": "UKR_POLITICAL",
    "kharkiv": "UKR_POLITICAL", "donetsk": "UKR_POLITICAL",

    "wagner": "RU_MILITARY",
    "afu": "UKR_MILITARY", "azov": "UKR_MILITARY", "syrskyi": "UKR_MILITARY",

    "usa": "WEST_ACTORS", "nato": "WEST_ACTORS", "eu": "WEST_ACTORS",
    "biden": "WEST_ACTORS", "trump": "WEST_ACTORS",
    "poland": "WEST_ACTORS", "germany": "WEST_ACTORS", "france": "WEST_ACTORS",

    "china": "INTL_ACTORS", "un": "INTL_ACTORS",
    "iaea": "INTL_ACTORS", "turkey": "INTL_ACTORS",

    "civilians": "CIVILIANS", "refugees": "CIVILIANS",
}

IZ_PA_BASE = {
    "netanyahu": "ISR_POLITICAL", "israel": "ISR_POLITICAL",
    "knesset": "ISR_POLITICAL", "mansour_abbas": "ISR_POLITICAL",
    "gallant": "ISR_POLITICAL", "gantz": "ISR_POLITICAL",

    "idf": "ISR_MILITARY", "mossad": "ISR_MILITARY", "shin bet": "ISR_MILITARY",

    "abbas": "PAL_POLITICAL", "pa": "PAL_POLITICAL", "palestine": "PAL_POLITICAL",

    "hamas": "PAL_ORG", "haniyeh": "PAL_ORG", "sinwar": "PAL_ORG", "pij": "PAL_ORG",

    "militants": "PAL_RESISTANCE", "qassam": "PAL_RESISTANCE", "gaza": "PAL_RESISTANCE",

    "usa": "INTL_ACTORS", "un": "INTL_ACTORS", "icj": "INTL_ACTORS",
    "biden": "INTL_ACTORS", "trump": "INTL_ACTORS",
    "iran": "INTL_ACTORS", "lebanon": "INTL_ACTORS", "turkey": "INTL_ACTORS",
    "syria": "INTL_ACTORS", "hezbollah": "INTL_ACTORS",
    "saudi arabia": "INTL_ACTORS", "qatar": "INTL_ACTORS", "egypt": "INTL_ACTORS",

    "civilians": "CIVILIANS", "hostages": "CIVILIANS", "settlers": "SETTLERS",
}


is_il_pa = website in ["ynet", "ynet_global", "alquds", "jpost", "aljazeera"]
active_entities = IZ_PA_BASE if is_il_pa else RU_UK_BASE
theater_name    = "Israel-Palestine" if is_il_pa else "Russia-Ukraine"
target_labels   = list(set(active_entities.values()))

search_keys = sorted(list(set(list(SYNONYM_MAP.keys()) + list(active_entities.keys()))), key=len, reverse=True)

print(f"🌍 Theater Detected: {theater_name} — website: {website}")


DEHUMAN_LEXICON = {
    "animal":   ["rat", "vermin", "beast", "animal", "hyena", "wolf",
                 "parasite", "insect", "viper", "pig", "monkey"],
    "disease":  ["cancer", "plague", "virus", "infection", "tumor",
                 "disease", "toxic", "poison", "pestilence"],
    "subhuman": ["barbaric", "monster", "scum", "demon", "savage",
                 "primitive", "beastly", "uncivilized", "evil"]
}

DEHUMAN_KEYWORDS = set(word for words in DEHUMAN_LEXICON.values() for word in words)
DEHUMAN_CATEGORY = {word: cat for cat, words in DEHUMAN_LEXICON.items() for word in words}

axis_seeds = {
    "competence": {
        "high": ["powerful", "strategic", "efficient", "capable", "advanced", "strong", "organized"],
        "low":  ["weak", "failing", "disorganized", "incompetent", "chaotic", "vulnerable", "ineffective"]
    },
    "morality": {
        "high": ["righteous", "just", "innocent", "heroic", "moral", "civilized", "peaceful"],
        "low":  ["cruel", "evil", "terrorist", "murderous", "corrupt", "barbaric", "aggressive"]
    }
}


def build_axis_vector(seeds_dict):
    """
    Constructs a semantic axis as the vector difference between the mean
    of high-pole word vectors and the mean of low-pole word vectors.
    Prints a sanity check: if high/low cosine similarity is >= 0, the axis
    may be degenerate (poles not meaningfully opposed in this vector space).
    """
    high_vecs = [nlp(w).vector for w in seeds_dict["high"] if nlp(w).has_vector]
    low_vecs  = [nlp(w).vector for w in seeds_dict["low"]  if nlp(w).has_vector]
    h = np.mean(high_vecs, axis=0)
    l = np.mean(low_vecs,  axis=0)

    cos_sim = np.dot(h, l) / (np.linalg.norm(h) * np.linalg.norm(l) + 1e-12)
    polarity = "✅ opposed" if cos_sim < 0 else "⚠️  not well-opposed"
    print(f"   Axis sanity — high/low cosine similarity: {cos_sim:.3f} {polarity}")

    return h - l


print("📐 Building semantic axes...")
comp_axis_vec  = build_axis_vector(axis_seeds["competence"])
moral_axis_vec = build_axis_vector(axis_seeds["morality"])


def get_projection_score(adj_text, axis_vec):
    """Projects an adjective's word vector onto the semantic axis."""
    doc = nlp(adj_text)
    if not doc.has_vector or doc.vector_norm == 0:
        return 0.0
    return float(np.dot(doc.vector, axis_vec) / (np.linalg.norm(axis_vec) + 1e-12))


def get_entity_from_token(token, doc):
    """
    Matches a token to a known entity using:
      1. Full noun chunk the token belongs to
      2. ±3 token window fallback
    Returns (canonical_name, label) or (None, None).
    """
    chunk_text = next((chunk.text.lower() for chunk in doc.noun_chunks if token.i in range(chunk.start, chunk.end)), None)
    window_text = " ".join(t.text.lower() for t in doc[max(0, token.i - 3): min(len(doc), token.i + 4)])
    for candidate_text in filter(None, [chunk_text, window_text]):
        match = next((k for k in search_keys if k in candidate_text), None)
        if match:
            clean = SYNONYM_MAP.get(match, match)
            if clean in active_entities:
                return clean, active_entities[clean]
    return None, None


ADJ_STOPLIST = {
    # --- Temporal / ordinal ---
    "recent", "new", "past", "current", "last", "first", "second", "third",
    "fourth", "fifth", "former", "late", "early", "annual", "daily",
    "next", "previous", "prior", "final", "initial", "ongoing", "constant",
    "pending", "upcoming", "scheduled", "planned", "active", "future",

    # --- Generic / scope ---
    "other", "same", "full", "able", "certain", "main", "overall", "general",
    "specific", "particular", "various", "several", "many", "few", "enough",
    "only", "clear", "direct", "open", "close", "hard", "heavy", "such",
    "actual", "sufficient", "difficult", "formal", "physical", "quiet",
    "different", "fundamental", "broad", "total", "comparable", "comparative",
    "bottom", "own", "possible", "complete", "critical", "positive",
    "multiple", "single", "available", "necessary", "relevant", "appropriate",
    "effective", "successful", "viable", "relative", "absolute", "universal",
    "extreme", "severe", "intense", "moderate",

    # --- Size / degree ---
    "old", "large", "small", "high", "low", "long", "short", "great",
    "big", "little", "wide", "narrow", "deep", "strong", "weak",

    # --- Colors (appear in "red line", "white phosphorus", "black market") ---
    "red", "blue", "green", "white", "black", "gray", "grey",
    "orange", "yellow", "purple", "brown", "dark", "light",

    # --- Demographic ---
    "young", "adult", "elderly", "female", "male",

    # --- Evaluative but content-free ---
    "good", "bad", "important", "major", "minor", "significant",
    "key", "central", "senior", "junior", "top", "leading", "historic",
    "chief",

    # --- Domain / relational (describe topic, not character) ---
    "military", "political", "diplomatic", "economic", "nuclear",
    "foreign", "regional", "international", "internal", "global",
    "national", "local", "federal", "official", "legal", "civil",
    "armed", "northern", "southern", "eastern", "western", "central",
    "strategic", "operational", "ballistic", "kinetic", "supreme",
    "scientific", "academic", "humanitarian", "ideological",
    "royal", "multinational", "intensive", "civilian", "covert",
    # Military domain
    "tactical", "conventional", "aerial", "naval", "ground",
    # Financial domain
    "financial", "fiscal", "monetary", "budgetary",
    # Religious/social domain
    "religious", "secular",
    # Medical domain
    "medical", "psychological", "psychiatric",
    # Administrative/legal domain
    "administrative", "governmental", "constitutional", "provisional", "interim",
    # Structural domain
    "structural", "systematic", "systemic", "institutional",
    # Diplomatic scope
    "bilateral", "unilateral", "multilateral",

    # --- Status descriptors ---
    "underway",

    # --- Scope / logical ---
    "potential", "partial", "immediate", "alternative", "comprehensive",
    "classic", "cognitive", "additional", "separate", "joint", "mutual",
    "common", "public", "private", "real", "true", "false",

    # --- Morphological prefixes spaCy tags as ADJ ---
    "anti", "pro", "post", "non", "pre", "neo", "ex",
}


def collect_adjs(token):
    """
    Recursively collects adjectives from a token, including:
    - The token itself (if ADJ, alphabetic, not a stoplist/demonym term)
    - Negation: checks dep=="neg" on the adjective's own children,
      and on the head's children when the adjective is a predicative complement
    - Conjuncts: "brutal and aggressive" → both collected
    Returns list of (lemma, is_negated) tuples.
    """
    results = []

    if token.pos_ != "ADJ":
        return results

    if not token.is_alpha:
        return results

    if token.text[0].isupper() or token.ent_type_ in ("NORP", "GPE"):
        return results

    if token.lemma_.lower() in ADJ_STOPLIST:
        return results

    neg = any(c.dep_ == "neg" for c in token.children)
    if token.dep_ == "acomp":
        neg = neg or any(c.dep_ == "neg" for c in token.head.children)

    results.append((token.lemma_.lower(), neg))

    for child in token.children:
        if child.dep_ == "conj" and child.pos_ == "ADJ":
            results.extend(collect_adjs(child))

    return results


def get_adjs_for_token(token):
    """
    Collects all adjectives associated with an entity token, covering:
      1. Attributive modifiers (amod): "the brutal attack"
         — demonym amod children skipped (e.g. "Israeli" in "Israeli forces")
      2. Predicative subject complements (acomp/xcomp on nsubj/nsubjpass):
         "Hamas is brutal" / "Hamas was declared illegal"
      3. Predicative object complements (xcomp/acomp on dobj/obj):
         "they called Hamas barbaric"
    Returns list of (lemma, is_negated) tuples.
    """
    found = []

    for child in token.children:
        if child.dep_ == "amod":
            found.extend(collect_adjs(child))

    if token.dep_ in ("nsubj", "nsubjpass"):
        for sibling in token.head.children:
            if sibling.dep_ in ("acomp", "xcomp") and sibling.pos_ == "ADJ":
                found.extend(collect_adjs(sibling))

    if token.dep_ in ("dobj", "obj"):
        for sibling in token.head.children:
            if sibling.dep_ in ("acomp", "xcomp", "oprd") and sibling.pos_ == "ADJ":
                found.extend(collect_adjs(sibling))
    return found


def build_sentiment_cache(texts_list):
    print("📦 Collecting unique sentences for RoBERTa batch scoring...")
    unique_sents = set()
    for doc in nlp.pipe(texts_list, batch_size=32, disable=["ner"]):
        for sent in doc.sents:
            unique_sents.add(sent.text.strip())

    unique_sents = list(unique_sents)
    polarities   = {}
    batch_size   = 64

    print(f"🤖 Scoring {len(unique_sents):,} unique sentences with RoBERTa...")
    for i in tqdm(range(0, len(unique_sents), batch_size)):
        batch = unique_sents[i: i + batch_size]
        try:
            results = sentiment_task([s[:512] for s in batch])
            for sent_text, result in zip(batch, results):
                label, score = result['label'], result['score']
                if label == 'positive':   polarities[sent_text] = score
                elif label == 'negative': polarities[sent_text] = -score
                else:                     polarities[sent_text] = 0.0
        except Exception:
            for sent_text in batch:
                polarities[sent_text] = 0.0

    return polarities


df = pd.read_csv(clean_data_file)
texts_list = df['text'].dropna().astype(str).tolist()

sentiment_cache       = build_sentiment_cache(texts_list)
adj_data_rows         = []
entity_texts_for_mft  = {l: [] for l in target_labels}

print(f"🚀 Processing {len(texts_list)} articles for {website}...")

# NER disabled — all entity matching is dictionary-based
for doc in tqdm(nlp.pipe(texts_list, batch_size=32, disable=["ner"]), total=len(texts_list)):
    for token in doc:
        ent_name, ent_label = get_entity_from_token(token, doc)
        if not ent_name:
            continue

        sent_text = token.sent.text.strip()
        entity_texts_for_mft[ent_label].append(sent_text)
        found_adjs = get_adjs_for_token(token)

        for adj_lemma, is_neg in found_adjs:
            adj_data_rows.append({
                "label":        ent_label,
                "entity":       ent_name,
                "adj":          adj_lemma,
                "is_neg":       is_neg,
                "context_sent": sent_text,
            })


print("\n📖 Scoring Moral Foundations Theory...")
mft_results = []
for label in tqdm(target_labels, desc="MFT"):
    texts = entity_texts_for_mft.get(label, [])
    if texts:
        temp_df  = pd.DataFrame(texts, columns=['text'])
        m_scores = score_docs(temp_df, 'emfd', 'all', 'bow', 'sentiment', len(temp_df))
        if not m_scores.empty:
            avg_scores          = m_scores.mean(numeric_only=True)
            avg_scores['label'] = label
            mft_results.append(avg_scores)

mft_df = pd.DataFrame(mft_results) if mft_results else pd.DataFrame()


print("\n🔬 Computing character projections...")
processed_rows = []

for item in tqdm(adj_data_rows, desc="Projecting adjectives"):
    adj    = item["adj"]
    is_neg = item["is_neg"]

    s_score = sentiment_cache.get(item["context_sent"], 0.0)

    comp  = get_projection_score(adj, comp_axis_vec)
    moral = get_projection_score(adj, moral_axis_vec)

    is_dehumanizing = 1 if adj in DEHUMAN_KEYWORDS else 0
    dehuman_category = DEHUMAN_CATEGORY.get(adj, None)

    if is_neg:
        s_score = -s_score
        comp    = -comp
        moral   = -moral

    processed_rows.append({
        "label":               item["label"],
        "entity":              item["entity"],
        "adjective":           f"not_{adj}" if is_neg else adj,
        "is_negated":          is_neg,
        "sentiment":           round(s_score, 4),
        "competence":          round(comp,    4),
        "morality":            round(moral,   4),
        "dehuman_flag":        is_dehumanizing,
        "negated_dehuman_flag": 1 if (is_neg and is_dehumanizing) else 0,
        "dehuman_category":    dehuman_category,
    })

df_adjectives = pd.DataFrame(processed_rows)



if df_adjectives.empty:
    print("⚠️  No adjectives extracted. Check entity dictionary and input data.")
else:
    entity_counts = df_adjectives['entity'].value_counts()
    valid_entities = entity_counts[entity_counts >= MIN_OCCURRENCES].index
    df_adjectives  = df_adjectives[df_adjectives['entity'].isin(valid_entities)]

    label_summary = df_adjectives.groupby("label").agg(
        competence          = ("competence",           "mean"),
        morality            = ("morality",             "mean"),
        sentiment           = ("sentiment",            "mean"),
        dehumanization_rate = ("dehuman_flag",         "mean"),
        negated_dehuman_rate= ("negated_dehuman_flag", "mean"),
        n_adjectives        = ("adjective",            "count"),
    ).round(4)

    if not mft_df.empty and 'label' in mft_df.columns:
        final_report = label_summary.merge(mft_df.set_index('label'), left_index=True, right_index=True, how='left')
    else:
        final_report = label_summary

    print(f"\n{'='*60}")
    print(f"  CHARACTER ANALYSIS REPORT: {website} ({theater_name})")
    print(f"{'='*60}")
    print(final_report.sort_values("dehumanization_rate", ascending=False).to_string())

    dehuman_rows = df_adjectives[df_adjectives['dehuman_flag'] == 1]
    if not dehuman_rows.empty:
        print("\n--- DEHUMANIZATION CATEGORY BREAKDOWN ---")
        cat_breakdown = (
            dehuman_rows.groupby(['label', 'dehuman_category'])
            .size()
            .reset_index(name='count')
            .sort_values('count', ascending=False)
        )
        print(cat_breakdown.to_string(index=False))

    print("\n--- TOP 10 ADJECTIVES BY LABEL ---")
    top_adjs = (
        df_adjectives.groupby(['label', 'adjective'])
        .size()
        .reset_index(name='count')
        .sort_values(['label', 'count'], ascending=[True, False])
        .groupby('label')
        .head(10)
    )
    print(top_adjs.to_string(index=False))


df_adjectives.to_csv(f"{website}_character_adjectives.csv", index=False)
print(f"\n✅ Saved: {website}_character_adjectives.csv — {len(df_adjectives):,} adjective records")