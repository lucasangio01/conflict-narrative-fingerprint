import pandas as pd
import numpy as np
import spacy
import math
import matplotlib.pyplot as plt
from collections import Counter
from gensim.models import Word2Vec
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import cosine
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from tqdm import tqdm

website1 = "ynet"
website2 = "alquds"

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
    "zaporizhzhia": "zaporizhzhia", "crimea": "crimea",
    "kharkiv": "kharkiv", "donetsk": "donetsk",
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
    "al-qassam brigades": "qassam",
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


# --- 3. THEATER DICTIONARIES (no generic keys) ---

RU_UK_BASE = {
    "russia": "RU_POLITICAL", "putin": "RU_POLITICAL",
    "moscow": "RU_POLITICAL", "kremlin": "RU_POLITICAL",
    "ukraine": "UKR_POLITICAL", "zelensky": "UKR_POLITICAL",
    "kiev": "UKR_POLITICAL", "yermak": "UKR_POLITICAL",
    "zaporizhzhia": "UKR_POLITICAL", "crimea": "UKR_POLITICAL",
    "kharkiv": "UKR_POLITICAL", "donetsk": "UKR_POLITICAL",
    "usa": "WEST_ACTORS", "nato": "WEST_ACTORS", "eu": "WEST_ACTORS",
    "biden": "WEST_ACTORS", "trump": "WEST_ACTORS",
    "poland": "WEST_ACTORS", "germany": "WEST_ACTORS", "france": "WEST_ACTORS",
    "wagner": "RU_MILITARY",
    "afu": "UKR_MILITARY", "azov": "UKR_MILITARY",
    "china": "INTL_ACTORS", "un": "INTL_ACTORS",
    "iaea": "INTL_ACTORS", "turkey": "INTL_ACTORS",
    "civilians": "CIVILIANS",
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


IL_PA_SITES = {"ynet", "ynet_global", "alquds", "jpost", "aljazeera"}
is_il_pa    = website1 in IL_PA_SITES or website2 in IL_PA_SITES
active_entities = IZ_PA_BASE if is_il_pa else RU_UK_BASE
theater_name    = "Israel-Palestine" if is_il_pa else "Russia-Ukraine"

if is_il_pa:
    concepts = ["security", "peace", "justice", "war", "civilian",
                "israel", "palestine", "hamas", "idf", "netanyahu"]
else:
    concepts = ["security", "peace", "justice", "war", "civilian",
                "russia", "ukraine", "putin", "zelensky", "nato"]

print(f"🌍 Theater Detected: {theater_name}")
print(f"🎯 Tracking Concepts: {', '.join(concepts)}")


def preprocess_corpus(df_subset):
    """
    Tokenizes pre-cleaned text using spaCy with sentence segmentation.
    Parser is kept active so sentence boundaries are respected — each sentence
    becomes one training unit for Word2Vec, giving meaningful context windows.
    NER is disabled (dictionary-based matching used elsewhere).
    Returns:
        flat_tokens: list of all tokens (for frequency analysis)
        sentences:   list of sentences, each a list of tokens (for Word2Vec)
    """
    flat_tokens, sentences = [], []
    texts = df_subset['text'].dropna().astype(str).tolist()

    print("   ✨ Running spaCy pipeline with sentence segmentation...")
    for doc in tqdm(nlp.pipe(texts, batch_size=32, disable=["ner"]),
                    total=len(texts)):
        for sent in doc.sents:
            sent_tokens = [
                t.lemma_.lower() for t in sent
                if not t.is_stop
                and t.is_alpha
                and len(t.lemma_) > 2
                and t.pos_ in ("NOUN", "ADJ", "VERB", "PROPN")
            ]
            if sent_tokens:
                flat_tokens.extend(sent_tokens)
                sentences.append(sent_tokens)

    return flat_tokens, sentences


def balance_by_tokens(tokens1, sents1, tokens2, sents2, random_state=42):
    """
    Balances two corpora by token count rather than article count.
    Randomly drops sentences from the larger corpus until both have
    approximately equal token counts, making Word2Vec spaces comparable.
    """
    rng = np.random.default_rng(random_state)
    n1, n2 = len(tokens1), len(tokens2)

    if n1 == n2:
        return tokens1, sents1, tokens2, sents2

    if n1 > n2:
        target_ratio = n2 / n1
        keep = rng.random(len(sents1)) < target_ratio
        sents1 = [s for s, k in zip(sents1, keep) if k]
        tokens1 = [t for s in sents1 for t in s]
    else:
        target_ratio = n1 / n2
        keep = rng.random(len(sents2)) < target_ratio
        sents2 = [s for s, k in zip(sents2, keep) if k]
        tokens2 = [t for s in sents2 for t in s]

    print(f"   Balanced token counts: {len(tokens1):,} vs {len(tokens2):,}")
    return tokens1, sents1, tokens2, sents2


def calculate_log_odds_zscore(tokens1, tokens2, prior=0.1, threshold=1.96):
    """
    Computes log-odds ratio with Dirichlet smoothing (prior) and z-score
    significance for each word in the combined vocabulary.

    Formula follows Monroe, Colaresi & Quinn (2008):
        log-odds(w) = log[ (c1_w + prior) / (n1 + prior - c1_w) ]
                    - log[ (c2_w + prior) / (n2 + prior - c2_w) ]
        variance    = 1/(c1_w + prior) + 1/(c2_w + prior)
        z-score     = log-odds / sqrt(variance)

    prior=0.1 is a weak uninformative prior that prevents log(0).
    Positive z-scores indicate overrepresentation in corpus 1.
    """
    counts1, counts2 = Counter(tokens1), Counter(tokens2)
    vocab = set(counts1.keys()) | set(counts2.keys())
    n1, n2 = sum(counts1.values()), sum(counts2.values())

    results = []
    for word in tqdm(vocab, desc="Computing log-odds z-scores"):
        c1, c2 = counts1.get(word, 0), counts2.get(word, 0)

        log_odds = (math.log((c1 + prior) / (n1 + prior - c1)) - math.log((c2 + prior) / (n2 + prior - c2)))
        variance = (1 / (c1 + prior)) + (1 / (c2 + prior))
        z_score  = log_odds / math.sqrt(variance)

        results.append({
            "word":           word,
            "log_odds":       round(log_odds, 4),
            "z_score":        round(z_score, 4),
            "count_site1":    c1,
            "count_site2":    c2,
            "count_total":    c1 + c2,
            "is_significant": abs(z_score) > threshold,
        })

    return pd.DataFrame(results).sort_values("z_score", ascending=False)


def calculate_narrative_entropy(tokens1, tokens2, top_n=500):
    """
    Computes narrative entropy for both corpora over a SHARED distinctive
    vocabulary — the union of each site's top_n most distinctive words.
    Using a shared vocabulary makes the two entropy scores directly comparable.

    Returns entropy and normalized entropy for each corpus.
    """
    marker_df = calculate_log_odds_zscore(tokens1, tokens2)
    significant = marker_df[marker_df["is_significant"]]

    shared_lex = (set(significant.head(top_n)["word"]) | set(significant.tail(top_n)["word"]))

    def _entropy(tokens, lex):
        filtered = [w for w in tokens if w in lex]
        if not filtered:
            return 0.0, 0.0
        counts = Counter(filtered)
        total  = len(filtered)
        h = -sum((c / total) * math.log2(c / total) for c in counts.values())
        nh = h / math.log2(len(lex)) if len(lex) > 1 else 0.0
        return round(h, 4), round(nh, 4)

    h1, nh1 = _entropy(tokens1, shared_lex)
    h2, nh2 = _entropy(tokens2, shared_lex)
    return h1, nh1, h2, nh2, shared_lex


def align_and_drift(model1, model2):
    """
    Aligns model2 into model1's vector space using Procrustes rotation,
    then computes per-word semantic drift as 1 - cosine_similarity.

    orthogonal_procrustes(v2, v1) returns R such that v2 @ R ≈ v1,
    so model2.wv[w] @ R maps model2 words into model1's space.
    """
    common = list(set(model1.wv.index_to_key) & set(model2.wv.index_to_key))
    v1 = np.array([model1.wv[w] for w in common])
    v2 = np.array([model2.wv[w] for w in common])

    R, _ = orthogonal_procrustes(v2, v1)

    drift = []
    for w in common:
        w1 = model1.wv[w]
        w2 = model2.wv[w] @ R
        sim = np.dot(w1, w2) / (np.linalg.norm(w1) * np.linalg.norm(w2) + 1e-12)
        drift.append({"word": w, "drift": round(1 - sim, 4)})

    return pd.DataFrame(drift).sort_values("drift", ascending=False), R


def analyze_neighborhood(word, m1, m2, R, top_k=15):
    """
    Compares the semantic neighborhoods of a word across two aligned models.
    Jaccard distance measures neighborhood overlap in each model's own space
    (correct — Jaccard is about which words are nearby, not cross-model distance).
    Centroid drift uses the rotation to compare neighborhood centers cross-model.
    """
    if word not in m1.wv or word not in m2.wv:
        return None

    n1 = [w for w, _ in m1.wv.most_similar(word, topn=top_k)]
    n2 = [w for w, _ in m2.wv.most_similar(word, topn=top_k)]
    set1, set2 = set(n1), set(n2)

    jaccard  = 1 - (len(set1 & set2) / len(set1 | set2))
    c1       = np.mean([m1.wv[w] for w in n1], axis=0)
    c2       = np.mean([m2.wv[w] for w in n2], axis=0) @ R
    c_drift  = cosine(c1, c2)

    return {
        "word":    word,
        "jaccard": round(jaccard, 4),
        "c_drift": round(c_drift, 4),
        "excl1":   sorted(set1 - set2),
        "excl2":   sorted(set2 - set1),
        "shared":  sorted(set1 & set2),
    }


def run_stability_test(sents1, sents2, concepts, iterations=5, subsample=0.8):
    """
    Bootstrap stability test: trains Word2Vec on random subsamples of each
    corpus and measures variance in Jaccard neighborhood divergence.
    High std dev → embedding is unstable for that concept → interpret with caution.
    """
    stability_results = {c: [] for c in concepts}

    for i in range(iterations):
        print(f"   Stability iteration {i+1}/{iterations}...")
        idx1 = np.random.choice(len(sents1), int(len(sents1) * subsample), replace=False)
        idx2 = np.random.choice(len(sents2), int(len(sents2) * subsample), replace=False)
        sub1 = [sents1[j] for j in idx1]
        sub2 = [sents2[j] for j in idx2]

        ms1 = Word2Vec(sub1, vector_size=100, min_count=5, workers=4, seed=i)
        ms2 = Word2Vec(sub2, vector_size=100, min_count=5, workers=4, seed=i)

        if ms1.wv.index_to_key and ms2.wv.index_to_key:
            _, R_s = align_and_drift(ms1, ms2)
            for c in concepts:
                res = analyze_neighborhood(c, ms1, ms2, R_s)
                if res:
                    stability_results[c].append(res["jaccard"])

    return {c: (round(np.mean(v), 4), round(np.std(v), 4)) for c, v in stability_results.items() if v}


def calculate_null_baseline(sents1, sents2, word, iterations=50):
    """
    Estimates the null distribution of Jaccard divergence for a word by
    randomly shuffling sentences between the two corpora and retraining.
    Returns the full null distribution and a p-value:
        p = proportion of null scores >= observed score.
    A low p-value (< 0.05) means the observed divergence is unlikely by chance.
    """
    combined = sents1 + sents2
    split    = len(sents1)
    null_scores = []

    for _ in range(iterations):
        shuffled = combined.copy()
        np.random.shuffle(shuffled)
        nm1 = Word2Vec(shuffled[:split], vector_size=100, min_count=10, workers=4)
        nm2 = Word2Vec(shuffled[split:], vector_size=100, min_count=10, workers=4)

        if word in nm1.wv and word in nm2.wv:
            n1 = [w for w, _ in nm1.wv.most_similar(word, topn=15)]
            n2 = [w for w, _ in nm2.wv.most_similar(word, topn=15)]
            null_scores.append(1 - len(set(n1) & set(n2)) / len(set(n1) | set(n2)))

    return null_scores


def get_glove_vector(word):
    """Returns the GloVe vector for a word from en_core_web_lg, or None."""
    token = nlp.vocab[word]
    if token.has_vector:
        return token.vector
    return None


def centroid_nearest_neighbors(centroid_vec, top_k=8, exclude=None):
    """
    Finds the top_k vocabulary words whose GloVe vectors are closest to a
    given centroid vector. This reveals the latent semantic region that a
    corpus's usage of a concept gravitates toward in GloVe space.

    For example, if YNET's 'peace' centroid is nearest to
    {deal, agreement, negotiation} and ALQUDS's is nearest to
    {rights, liberation, dignity}, that is the "smoking gun" of semantic
    divergence — not just different neighbors, but different conceptual frames.

    exclude: set of words to suppress (e.g. the concept itself, stopwords)
    """
    exclude = exclude or set()
    centroid_norm = centroid_vec / (np.linalg.norm(centroid_vec) + 1e-12)

    scored = []
    for word in nlp.vocab:
        if not word.has_vector or not word.is_alpha or word.is_stop:
            continue
        if word.text.lower() in exclude or len(word.text) < 3:
            continue
        vec  = word.vector / (np.linalg.norm(word.vector) + 1e-12)
        sim  = float(np.dot(centroid_norm, vec))
        scored.append((word.text.lower(), sim))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [w for w, _ in scored[:top_k]]


def get_cooccurrence_neighbors(concept, sentences, top_k=20, window=None):
    """
    Finds the top_k most frequent content words that co-occur with `concept`
    in the same sentence. `window=None` means full sentence scope.
    Returns a Counter of {word: frequency}.
    """
    neighbor_counts = Counter()
    for sent in sentences:
        if concept in sent:
            for word in sent:
                if word != concept:
                    neighbor_counts[word] += 1
    return neighbor_counts.most_common(top_k)


def analyze_glove_neighborhood(concept, sents1, sents2, top_k=20):
    """
    Compares the GloVe-space semantic context of a concept across two corpora.
    Returns the same structure as analyze_neighborhood() for direct comparison.
    """
    if get_glove_vector(concept) is None:
        return None

    neighbors1 = get_cooccurrence_neighbors(concept, sents1, top_k=top_k)
    neighbors2 = get_cooccurrence_neighbors(concept, sents2, top_k=top_k)

    if not neighbors1 or not neighbors2:
        return None

    words1 = [w for w, _ in neighbors1 if get_glove_vector(w) is not None]
    words2 = [w for w, _ in neighbors2 if get_glove_vector(w) is not None]

    if not words1 or not words2:
        return None

    set1, set2 = set(words1), set(words2)
    jaccard = round(1 - len(set1 & set2) / len(set1 | set2), 4)

    freqs1  = {w: f for w, f in neighbors1}
    freqs2  = {w: f for w, f in neighbors2}
    total1  = sum(freqs1.get(w, 1) for w in words1)
    total2  = sum(freqs2.get(w, 1) for w in words2)

    c1 = np.sum([get_glove_vector(w) * (freqs1.get(w, 1) / total1) for w in words1], axis=0)
    c2 = np.sum([get_glove_vector(w) * (freqs2.get(w, 1) / total2) for w in words2], axis=0)

    c_drift = round(float(cosine(c1, c2)), 4)

    exclude_words = {concept} | set(words1) | set(words2)
    nn_c1 = centroid_nearest_neighbors(c1, top_k=8, exclude=exclude_words)
    nn_c2 = centroid_nearest_neighbors(c2, top_k=8, exclude=exclude_words)

    return {
        "word":      concept,
        "jaccard":   jaccard,
        "c_drift":   c_drift,
        "excl1":     sorted(set1 - set2),
        "excl2":     sorted(set2 - set1),
        "shared":    sorted(set1 & set2),
        "centroid1": c1,
        "centroid2": c2,
        "nn_c1":     nn_c1,
        "nn_c2":     nn_c2,
    }


def plot_glove_semantic_map(concept, sents1, sents2, label1, label2, top_k=10):
    """
    Plots the GloVe-space semantic neighborhood of a concept for both corpora.
    Uses PCA on the GloVe vectors of the concept + its top co-occurrence neighbors.
    No alignment step needed — both corpora are already in the same GloVe space.
    """
    res = analyze_glove_neighborhood(concept, sents1, sents2, top_k=top_k)
    if res is None:
        print(f"   '{concept}' — insufficient GloVe coverage, skipping.")
        return

    n1 = [w for w in res["excl1"][:top_k//2] + res["shared"][:top_k//2]
          if get_glove_vector(w) is not None][:top_k]
    n2 = [w for w in res["excl2"][:top_k//2] + res["shared"][:top_k//2]
          if get_glove_vector(w) is not None][:top_k]

    concept_vec = get_glove_vector(concept)
    all_vecs = np.vstack([concept_vec, concept_vec] + [get_glove_vector(w) for w in n1] + [get_glove_vector(w) for w in n2])
    coords = PCA(n_components=2).fit_transform(all_vecs)

    plt.figure(figsize=(10, 7))

    def draw_cloud(pts, color, label):
        if len(pts) >= 3:
            try:
                hull = ConvexHull(pts)
                plt.fill(
                    pts[hull.vertices, 0], pts[hull.vertices, 1],
                    color=color, alpha=0.1, label=f"{label} Context"
                )
            except Exception:
                pass

    offset = 2
    draw_cloud(coords[offset:offset + len(n1)],          "skyblue", label1.upper())
    draw_cloud(coords[offset + len(n1):],                 "salmon",  label2.upper())

    plt.scatter(coords[offset:offset+len(n1), 0], coords[offset:offset+len(n1), 1], color="skyblue", s=40, alpha=0.6)
    plt.scatter(coords[offset+len(n1):, 0], coords[offset+len(n1):, 1], color="salmon",  s=40, alpha=0.6)
    plt.scatter(coords[0, 0], coords[0, 1], color="skyblue", s=500, marker="*",
                label=f"{label1.upper()} Pivot", edgecolors="navy", zorder=10)
    plt.scatter(coords[1, 0], coords[1, 1], color="salmon",  s=400, marker="o",
                label=f"{label2.upper()} Pivot", edgecolors="darkred", zorder=10)

    for i, w in enumerate(n1):
        plt.annotate(w, (coords[offset+i, 0], coords[offset+i, 1]),
                     color="navy", fontsize=9, alpha=0.8,
                     xytext=(3, 3), textcoords="offset points")
    for i, w in enumerate(n2):
        plt.annotate(w, (coords[offset+len(n1)+i, 0], coords[offset+len(n1)+i, 1]),
                     color="darkred", fontsize=9, alpha=0.8,
                     xytext=(3, 3), textcoords="offset points")

    plt.title(f"[GloVe] Semantic Map: '{concept.upper()}' | Jaccard: {res['jaccard']:.3f} | Centroid drift: {res['c_drift']:.3f}", fontsize=12, fontweight="bold")
    plt.legend(fontsize=9)
    plt.grid(alpha=0.1)
    plt.tight_layout()
    plt.show()


def plot_polarization(marker_df, label1, label2, top_n=15):
    significant = marker_df[marker_df["is_significant"]]
    plot_df = pd.concat([significant.head(top_n), significant.tail(top_n)])
    plt.figure(figsize=(10, 8))
    colors = ["skyblue" if x > 0 else "salmon" for x in plot_df["log_odds"]]
    plt.barh(plot_df["word"], plot_df["log_odds"], color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title(f"Polarized Lexicons: {label1.upper()} (+) vs {label2.upper()} (−)", fontsize=14, fontweight="bold")
    plt.xlabel("Log-Odds Ratio (z-score significant only)")
    plt.grid(alpha=0.1)
    plt.tight_layout()
    plt.show()


def plot_semantic_map(word, m1, m2, R, label1, label2, top_k=10):
    if word not in m1.wv or word not in m2.wv:
        print(f"   '{word}' not in one or both models — skipping.")
        return

    metrics = analyze_neighborhood(word, m1, m2, R, top_k=top_k)
    n1 = [w for w, _ in m1.wv.most_similar(word, topn=top_k)]
    n2 = [w for w, _ in m2.wv.most_similar(word, topn=top_k)]

    all_vecs = np.vstack(
        [m1.wv[word], m2.wv[word] @ R] +
        [m1.wv[w] for w in n1] +
        [m2.wv[w] @ R for w in n2])
    coords = PCA(n_components=2).fit_transform(all_vecs)

    plt.figure(figsize=(10, 7))

    def draw_cloud(pts, color, label):
        if len(pts) >= 3:
            try:
                hull = ConvexHull(pts)
                plt.fill(pts[hull.vertices, 0], pts[hull.vertices, 1], color=color, alpha=0.1, label=f"{label} Context")
            except Exception:
                pass

    draw_cloud(coords[2:2 + top_k],            "skyblue", label1.upper())
    draw_cloud(coords[2 + top_k:],              "salmon",  label2.upper())

    plt.scatter(coords[2:2+top_k, 0],         coords[2:2+top_k, 1],         color="skyblue", s=40, alpha=0.6)
    plt.scatter(coords[2+top_k:, 0],           coords[2+top_k:, 1],           color="salmon",  s=40, alpha=0.6)
    plt.scatter(coords[0, 0], coords[0, 1], color="skyblue", s=500, marker="*",
                label=f"{label1.upper()} Pivot", edgecolors="navy", zorder=10)
    plt.scatter(coords[1, 0], coords[1, 1], color="salmon",  s=400, marker="o",
                label=f"{label2.upper()} Pivot", edgecolors="darkred", zorder=10)

    for i in range(top_k):
        plt.annotate(n1[i], (coords[i+2, 0],         coords[i+2, 1]),         color="navy",    fontsize=9, alpha=0.8, xytext=(3, 3), textcoords="offset points")
        plt.annotate(n2[i], (coords[i+2+top_k, 0],   coords[i+2+top_k, 1]),   color="darkred", fontsize=9, alpha=0.8, xytext=(3, 3), textcoords="offset points")

    plt.title(f"Semantic Map: '{word.upper()}' | Jaccard: {metrics['jaccard']:.3f} | Shared neighbors: {len(metrics['shared'])}", fontsize=12, fontweight="bold")
    plt.legend(fontsize=9)
    plt.grid(alpha=0.1)
    plt.tight_layout()
    plt.show()


print("\n🚀 Loading data...")
df1 = pd.read_csv(f"{website1}_final.csv")
df2 = pd.read_csv(f"{website2}_final.csv")
print(f"   {website1}: {len(df1):,} articles | {website2}: {len(df2):,} articles")

print("\n📝 Preprocessing corpora (sentence-aware)...")
print(f"   {website1}:")
t1, s1 = preprocess_corpus(df1)
print(f"   {website2}:")
t2, s2 = preprocess_corpus(df2)
print(f"   Raw token counts: {len(t1):,} ({website1}) vs {len(t2):,} ({website2})")

print("\n⚖️  Balancing corpora by token count...")
t1, s1, t2, s2 = balance_by_tokens(t1, s1, t2, s2)
counts1, counts2 = Counter(t1), Counter(t2)

print("\n🤖 Training Word2Vec models...")
W2V_MIN_COUNT = 3
W2V_VOCAB_THRESHOLD = 2000

m1 = Word2Vec(sentences=s1, vector_size=100, window=5, min_count=W2V_MIN_COUNT, workers=4, seed=42)
m2 = Word2Vec(sentences=s2, vector_size=100, window=5, min_count=W2V_MIN_COUNT, workers=4, seed=42)

vocab1, vocab2 = len(m1.wv.index_to_key), len(m2.wv.index_to_key)
print(f"   Vocab sizes: {vocab1:,} ({website1}) | {vocab2:,} ({website2})")

w2v_reliable = vocab1 >= W2V_VOCAB_THRESHOLD and vocab2 >= W2V_VOCAB_THRESHOLD
if not w2v_reliable:
    print(f"\n   ⚠️  WARNING: One or both Word2Vec vocabularies are below {W2V_VOCAB_THRESHOLD:,} types.")
    print(f"   W2V embeddings are likely degenerate on this corpus size.")
    print(f"   Neighborhood and drift results will be reported but should NOT be interpreted.")
    print(f"   Treat GloVe results as the primary analysis for this corpus pair.")


print("\n🔗 Aligning embedding spaces (Procrustes)...")
drift_df, rotation_matrix = align_and_drift(m1, m2)

if w2v_reliable:
    print("\n🔁 Running stability bootstrap tests...")
    stability_stats = run_stability_test(s1, s2, concepts, iterations=5)
else:
    print("\n⏭️  Skipping stability bootstrap — W2V vocabulary too small for reliable results.")
    stability_stats = {}


print("\n⚖️  Computing log-odds z-scores...")
marker_df = calculate_log_odds_zscore(t1, t2)
significant = marker_df[marker_df["is_significant"]]

print(f"\n--- STATISTICALLY SIGNIFICANT ANCHORS: {website1.upper()} ---")
print(significant.head(10)[["word", "z_score", "count_site1", "count_site2"]].to_string(index=False))

print(f"\n--- STATISTICALLY SIGNIFICANT ANCHORS: {website2.upper()} ---")
print(significant.tail(10)[["word", "z_score", "count_site1", "count_site2"]].to_string(index=False))

print("\n📐 Computing narrative entropy over shared distinctive vocabulary...")
h1, nh1, h2, nh2, shared_lex = calculate_narrative_entropy(t1, t2)
print(f"\n--- NARRATIVE ENTROPY (shared vocabulary of {len(shared_lex):,} words) ---")
print(f"{'Site':<15} | {'H (raw)':<10} | {'H (normalized)'}")
print("-" * 45)
print(f"{website1.upper():<15} | {h1:<10} | {nh1}")
print(f"{website2.upper():<15} | {h2:<10} | {nh2}")
print("Higher normalized entropy → more dispersed use of distinctive vocabulary")

print(f"\n--- SEMANTIC DRIFT: TOP 15 MOST DRIFTED WORDS ---")
print(drift_df.head(15).to_string(index=False))

print(f"\n--- COMPREHENSIVE NARRATIVE DIVERGENCE REPORT (Word2Vec) ---")
if not w2v_reliable:
    print(f"    ⚠️  UNRELIABLE — vocab sizes ({vocab1}, {vocab2}) below threshold ({W2V_VOCAB_THRESHOLD})")
    print(f"    Results shown for completeness only. Use GloVe section below.\n")

print(f"{'CONCEPT':<12} | {'FREQ1':<6} | {'FREQ2':<6} | {'JACCARD μ':<10} | {'STD':<7} | {'NULL μ':<8} | {'P-VAL':<7} | SIG")
print("-" * 85)

for c in concepts:
    if c not in stability_stats:
        if w2v_reliable:
            print(f"{c.upper():<12} | {'N/A':}")
        else:
            print(f"{c.upper():<12} | {counts1.get(c,0):<6} | {counts2.get(c,0):<6} | {'skipped (unreliable)'}")
        continue

    f1 = counts1.get(c, 0)
    f2 = counts2.get(c, 0)
    mean_j, std_j = stability_stats[c]

    if w2v_reliable:
        null_scores = calculate_null_baseline(s1, s2, c, iterations=50)
        null_mean   = round(np.mean(null_scores), 4) if null_scores else 0.0
        p_value     = round(np.mean([s >= mean_j for s in null_scores]), 3) if null_scores else 1.0
        is_sig      = "⭐" if p_value < 0.05 else ""
    else:
        null_mean, p_value, is_sig = "N/A", "N/A", ""

    print(f"{c.upper():<12} | {f1:<6} | {f2:<6} | {mean_j:<10} | {std_j:<7} | {null_mean:<8} | {p_value:<7} | {is_sig}")

print("\n--- NEIGHBORHOOD ANALYSIS PER CONCEPT (Word2Vec) ---")
if not w2v_reliable:
    print("    ⚠️  UNRELIABLE — shown for reference only\n")
for c in concepts:
    res = analyze_neighborhood(c, m1, m2, rotation_matrix)
    if res:
        print(f"\n  {c.upper()} | Jaccard: {res['jaccard']} | Centroid drift: {res['c_drift']}")
        print(f"    {website1} only: {res['excl1'][:8]}")
        print(f"    {website2} only: {res['excl2'][:8]}")
        print(f"    Shared:         {res['shared'][:8]}")


print("\n" + "=" * 70)
print("  GLOVE ANALYSIS (pretrained, stable — Common Crawl 840B tokens)")
print("  No alignment needed — both corpora share the same vector space.")
print("=" * 70)

print(f"\n--- GLOVE NEIGHBORHOOD DIVERGENCE PER CONCEPT ---")
print(f"{'CONCEPT':<12} | {'JACCARD':<9} | {'CENTROID DRIFT'}")
print("-" * 40)

glove_results = {}
for c in concepts:
    res = analyze_glove_neighborhood(c, s1, s2, top_k=20)
    if res:
        glove_results[c] = res
        print(f"{c.upper():<12} | {res['jaccard']:<9} | {res['c_drift']}")
    else:
        print(f"{c.upper():<12} | {'N/A':<9} | N/A")

print(f"\n--- GLOVE NEIGHBORHOOD DETAILS PER CONCEPT ---")
for c, res in glove_results.items():
    print(f"\n  {c.upper()} | Jaccard: {res['jaccard']} | Centroid drift: {res['c_drift']}")
    print(f"    {website1} only: {res['excl1'][:8]}")
    print(f"    {website2} only: {res['excl2'][:8]}")
    print(f"    Shared:         {res['shared'][:8]}")
    print(f"    ── Centroid semantic region ──")
    print(f"    {website1} centroid → {res['nn_c1']}")
    print(f"    {website2} centroid → {res['nn_c2']}")

print(f"\n--- METHOD COMPARISON: Word2Vec vs GloVe ---")
if not w2v_reliable:
    print(f"    ⚠️  W2V results unreliable (vocab: {vocab1}/{vocab2}). GloVe is primary analysis.")
print(f"{'CONCEPT':<12} | {'W2V Jaccard':<13} | {'W2V C-drift':<13} | {'GLoVe Jaccard':<15} | {'GLoVe C-drift'}")
print("-" * 75)
for c in concepts:
    w2v_res   = analyze_neighborhood(c, m1, m2, rotation_matrix)
    glove_res = glove_results.get(c)
    w2v_j     = f"{w2v_res['jaccard']:.4f}"   if w2v_res   else "N/A"
    w2v_cd    = f"{w2v_res['c_drift']:.4f}"   if w2v_res   else "N/A"
    glv_j     = f"{glove_res['jaccard']:.4f}" if glove_res else "N/A"
    glv_cd    = f"{glove_res['c_drift']:.4f}" if glove_res else "N/A"
    print(f"{c.upper():<12} | {w2v_j:<13} | {w2v_cd:<13} | {glv_j:<15} | {glv_cd}")

print("""
Interpretation guide:
  High W2V / Low GloVe  → outlet-specific framing not captured by pretrained vectors
                           (corpus-idiosyncratic usage, possible domain-specific sense)
  High GloVe / Low W2V  → divergence in general associations, W2V unstable on this word
                           (likely corpus size issue — treat W2V result with caution)
  Both high             → robust divergence confirmed by both methods ✅
  Both low              → concept used similarly across both outlets
""")


print("\n📊 Generating visualizations...")
plot_polarization(marker_df, website1, website2)

print("\n  Word2Vec semantic maps:")
for concept in concepts:
    plot_semantic_map(concept, m1, m2, rotation_matrix, website1, website2)

print("\n  GloVe semantic maps:")
for concept in concepts:
    plot_glove_semantic_map(concept, s1, s2, website1, website2)


marker_df.to_csv(f"{website1}_vs_{website2}_logodds.csv", index=False)
drift_df.to_csv(f"{website1}_vs_{website2}_w2v_drift.csv", index=False)

if glove_results:
    glove_df = pd.DataFrame([
        {"concept":        c,
         "jaccard":        r["jaccard"],
         "centroid_drift": r["c_drift"],
         "excl_site1":     str(r["excl1"][:10]),
         "excl_site2":     str(r["excl2"][:10]),
         "shared":         str(r["shared"][:10]),
         "centroid_nn_site1": str(r["nn_c1"]),
         "centroid_nn_site2": str(r["nn_c2"]),
        }
        for c, r in glove_results.items()
    ])
    glove_df.to_csv(f"{website1}_vs_{website2}_glove_divergence.csv", index=False)

print(f"\n✅ Saved:")
print(f"   {website1}_vs_{website2}_logodds.csv          — log-odds z-score table")
print(f"   {website1}_vs_{website2}_w2v_drift.csv        — Word2Vec per-word semantic drift")
print(f"   {website1}_vs_{website2}_glove_divergence.csv — GloVe neighborhood divergence")