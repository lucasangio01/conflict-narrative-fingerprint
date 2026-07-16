import pandas as pd
import numpy as np
import math
import pickle
import matplotlib.pyplot as plt
from collections import Counter
from gensim.models import Word2Vec
from scipy.linalg import orthogonal_procrustes
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from tqdm import tqdm
from src.utils.constants import Websites, SemanticDivergenceConfig, PreprocessingConfig
from src.semantic_divergence.common import (
    get_nlp, get_glove_vector, analyze_glove_neighborhood, analyze_neighborhood,
)
from src.utils.logging_config import get_logger

logger = get_logger("SEMANTIC DIVERGENCE")


def main(website1="kpru", website2="ukpravda"):
    nlp = get_nlp()

    is_il_pa     = website1 in Websites.WEBSITES_PALESTINE_ISRAEL or website2 in Websites.WEBSITES_PALESTINE_ISRAEL
    theater_name = Websites.THEATER_IL_PA if is_il_pa else Websites.THEATER_RU_UK
    concepts     = SemanticDivergenceConfig.CONCEPTS_IL_PA if is_il_pa else SemanticDivergenceConfig.CONCEPTS_RU_UK

    logger.info(f"Theater detected: {theater_name}")
    logger.info(f"Tracking concepts: {', '.join(concepts)}")

    def preprocess_corpus(df_subset):
        """
        Tokenizes pre-cleaned text using spaCy with sentence segmentation.
        Parser is kept active so sentence boundaries are respected -- each sentence
        becomes one training unit for Word2Vec, giving meaningful context windows.
        NER is disabled (dictionary-based matching used elsewhere).
        Returns:
            flat_tokens: list of all tokens (for frequency analysis)
            sentences:   list of sentences, each a list of tokens (for Word2Vec)
        """
        flat_tokens, sentences = [], []
        texts = df_subset['text'].dropna().astype(str).tolist()

        logger.info("Running spaCy pipeline with sentence segmentation...")
        for doc in tqdm(nlp.pipe(texts, batch_size=32, disable=["ner"]), total=len(texts)):
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

        logger.info(f"Balanced token counts: {len(tokens1):,} vs {len(tokens2):,}")
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
        vocabulary -- the union of each site's top_n most distinctive words.
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

    def run_stability_test(sents1, sents2, concepts, iterations=5, subsample=0.8):
        """
        Bootstrap stability test: trains Word2Vec on random subsamples of each
        corpus and measures variance in Jaccard neighborhood divergence.
        High std dev → embedding is unstable for that concept → interpret with caution.
        """
        stability_results = {c: [] for c in concepts}

        for i in range(iterations):
            logger.info(f"Stability iteration {i+1}/{iterations}...")
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
        Returns the full null distribution; p = proportion of null scores >=
        observed score. A low p-value (< 0.05) means the observed divergence is
        unlikely by chance.
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

    def plot_polarization(marker_df, label1, label2, top_n=15):
        significant = marker_df[marker_df["is_significant"]]
        plot_df = pd.concat([significant.head(top_n), significant.tail(top_n)])
        plt.figure(figsize=(10, 8))
        colors = ["skyblue" if x > 0 else "salmon" for x in plot_df["log_odds"]]
        plt.barh(plot_df["word"], plot_df["log_odds"], color=colors)
        plt.axvline(0, color="black", linewidth=0.8)
        plt.title(f"Polarized Lexicons: {label1.upper()} (+) vs {label2.upper()} (−)", fontsize=14, fontweight="bold")
        plt.xlabel("\nLog-odds ratio (z-score significant only)")
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=11)
        plt.grid(alpha=0.1)
        plt.tight_layout()
        plt.savefig(f"{label1}_vs_{label2}_polarization.png", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

    def plot_semantic_map(word, m1, m2, R, label1, label2, top_k=10):
        if word not in m1.wv or word not in m2.wv:
            logger.warning(f"'{word}' not in one or both models — skipping.")
            return

        metrics = analyze_neighborhood(word, m1, m2, R, top_k=top_k)
        n1 = [w for w, _ in m1.wv.most_similar(word, topn=top_k)]
        n2 = [w for w, _ in m2.wv.most_similar(word, topn=top_k)]

        all_vecs = np.vstack([m1.wv[word], m2.wv[word] @ R] + [m1.wv[w] for w in n1] + [m2.wv[w] @ R for w in n2])
        coords = PCA(n_components=2).fit_transform(all_vecs)

        plt.figure(figsize=(10, 7))

        def draw_cloud(pts, color, label):
            if len(pts) >= 3:
                try:
                    hull = ConvexHull(pts)
                    plt.fill(pts[hull.vertices, 0], pts[hull.vertices, 1], color=color, alpha=0.1, label=f"{label} context")
                except Exception:
                    pass

        draw_cloud(coords[2:2 + top_k], "skyblue", label1.upper())
        draw_cloud(coords[2 + top_k:],  "salmon",  label2.upper())

        plt.scatter(coords[2:2+top_k, 0], coords[2:2+top_k, 1], color="skyblue", s=40, alpha=0.6)
        plt.scatter(coords[2+top_k:, 0],  coords[2+top_k:, 1],  color="salmon",  s=40, alpha=0.6)
        plt.scatter(coords[0, 0], coords[0, 1], color="skyblue", s=500, marker="*", label=f"{label1.upper()} Pivot", edgecolors="navy", zorder=10)
        plt.scatter(coords[1, 0], coords[1, 1], color="salmon",  s=400, marker="o", label=f"{label2.upper()} Pivot", edgecolors="darkred", zorder=10)

        for i in range(top_k):
            plt.annotate(n1[i], (coords[i+2, 0], coords[i+2, 1]), color="navy", fontsize=9, alpha=0.8, xytext=(3, 3), textcoords="offset points")
            plt.annotate(n2[i], (coords[i+2+top_k, 0], coords[i+2+top_k, 1]), color="darkred", fontsize=9, alpha=0.8, xytext=(3, 3), textcoords="offset points")

        plt.title(f"Semantic Map: '{word.upper()}' | Jaccard: {metrics['jaccard']:.3f} | Shared neighbors: {len(metrics['shared'])}", fontsize=12, fontweight="bold")
        plt.legend(fontsize=9)
        plt.grid(alpha=0.1)
        plt.tight_layout()
        plt.savefig(f"{label1}_vs_{label2}_w2v_{word}.png", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

    def plot_glove_semantic_map(concept, sents1, sents2, label1, label2, top_k=10):
        """
        Plots the GloVe-space semantic neighborhood of a concept for both corpora.
        Uses PCA on the GloVe vectors of the concept + its top co-occurrence neighbors.
        No alignment step needed -- both corpora are already in the same GloVe space.
        """
        res = analyze_glove_neighborhood(concept, sents1, sents2, top_k=top_k)
        if res is None:
            logger.warning(f"'{concept}' — insufficient GloVe coverage, skipping.")
            return

        n1 = [w for w in res["excl1"][:top_k//2] + res["shared"][:top_k//2] if get_glove_vector(w) is not None][:top_k]
        n2 = [w for w in res["excl2"][:top_k//2] + res["shared"][:top_k//2] if get_glove_vector(w) is not None][:top_k]

        concept_vec = get_glove_vector(concept)
        all_vecs = np.vstack([concept_vec, concept_vec] + [get_glove_vector(w) for w in n1] + [get_glove_vector(w) for w in n2])
        coords = PCA(n_components=2).fit_transform(all_vecs)

        plt.figure(figsize=(10, 7))

        def draw_cloud(pts, color, label):
            if len(pts) >= 3:
                try:
                    hull = ConvexHull(pts)
                    plt.fill(pts[hull.vertices, 0], pts[hull.vertices, 1], color=color, alpha=0.1, label=f"{label} Context")
                except Exception:
                    pass

        offset = 2
        draw_cloud(coords[offset:offset + len(n1)], "skyblue", Websites.DISPLAY_NAMES.get(label1, label1))
        draw_cloud(coords[offset + len(n1):],       "salmon",  Websites.DISPLAY_NAMES.get(label2, label2))

        plt.scatter(coords[offset:offset+len(n1), 0], coords[offset:offset+len(n1), 1], color="skyblue", s=40, alpha=0.6)
        plt.scatter(coords[offset+len(n1):, 0],       coords[offset+len(n1):, 1],       color="salmon",  s=40, alpha=0.6)
        plt.scatter(coords[0, 0], coords[0, 1], color="skyblue", s=500, marker="*", label=f"{Websites.DISPLAY_NAMES.get(label1, label1)} pivot", edgecolors="navy", zorder=10)
        plt.scatter(coords[1, 0], coords[1, 1], color="salmon",  s=400, marker="o", label=f"{Websites.DISPLAY_NAMES.get(label2, label2)} pivot", edgecolors="darkred", zorder=10)

        for i, w in enumerate(n1):
            plt.annotate(w, (coords[offset+i, 0], coords[offset+i, 1]), color="navy", fontsize=9, alpha=0.8, xytext=(3, 3), textcoords="offset points")
        for i, w in enumerate(n2):
            plt.annotate(w, (coords[offset+len(n1)+i, 0], coords[offset+len(n1)+i, 1]), color="darkred", fontsize=9, alpha=0.8, xytext=(3, 3), textcoords="offset points")

        plt.title(f"[GloVe] Semantic Map: '{concept.upper()}' | Jaccard: {res['jaccard']:.3f} | Centroid drift: {res['c_drift']:.3f}", fontsize=12, fontweight="bold")
        plt.legend(fontsize=9)
        plt.grid(alpha=0.1)
        plt.tight_layout()
        plt.savefig(f"{label1}_vs_{label2}_glove_{concept}.png", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

    logger.info("Loading data...")
    df1 = pd.read_csv(PreprocessingConfig.STAGE_FINAL.format(website=website1))
    df2 = pd.read_csv(PreprocessingConfig.STAGE_FINAL.format(website=website2))
    logger.info(f"{website1}: {len(df1):,} articles | {website2}: {len(df2):,} articles")

    logger.info("Preprocessing corpora (sentence-aware)...")
    logger.info(f"{website1}:")
    t1, s1 = preprocess_corpus(df1)
    logger.info(f"{website2}:")
    t2, s2 = preprocess_corpus(df2)
    logger.info(f"Raw token counts: {len(t1):,} ({website1}) vs {len(t2):,} ({website2})")

    logger.info("Balancing corpora by token count...")
    t1, s1, t2, s2 = balance_by_tokens(t1, s1, t2, s2)
    counts1, counts2 = Counter(t1), Counter(t2)

    logger.info("Training Word2Vec models...")
    m1 = Word2Vec(sentences=s1, vector_size=100, window=5, min_count=SemanticDivergenceConfig.W2V_MIN_COUNT, workers=4, seed=42)
    m2 = Word2Vec(sentences=s2, vector_size=100, window=5, min_count=SemanticDivergenceConfig.W2V_MIN_COUNT, workers=4, seed=42)

    vocab1, vocab2 = len(m1.wv.index_to_key), len(m2.wv.index_to_key)
    logger.info(f"Vocab sizes: {vocab1:,} ({website1}) | {vocab2:,} ({website2})")

    w2v_reliable = vocab1 >= SemanticDivergenceConfig.W2V_VOCAB_THRESHOLD and vocab2 >= SemanticDivergenceConfig.W2V_VOCAB_THRESHOLD
    if not w2v_reliable:
        logger.warning(
            f"One or both Word2Vec vocabularies are below {SemanticDivergenceConfig.W2V_VOCAB_THRESHOLD:,} types. "
            "W2V embeddings are likely degenerate on this corpus size. Neighborhood and drift results will be "
            "reported but should NOT be interpreted. Treat GloVe results as the primary analysis for this corpus pair."
        )

    logger.info("Aligning embedding spaces (Procrustes)...")
    drift_df, rotation_matrix = align_and_drift(m1, m2)

    if w2v_reliable:
        logger.info("Running stability bootstrap tests...")
        stability_stats = run_stability_test(s1, s2, concepts, iterations=5)
    else:
        logger.warning("Skipping stability bootstrap — W2V vocabulary too small for reliable results.")
        stability_stats = {}

    logger.info("Computing log-odds z-scores...")
    marker_df = calculate_log_odds_zscore(t1, t2)
    significant = marker_df[marker_df["is_significant"]]

    logger.info(
        f"--- STATISTICALLY SIGNIFICANT ANCHORS: {website1.upper()} ---\n"
        + significant.head(10)[["word", "z_score", "count_site1", "count_site2"]].to_string(index=False)
    )
    logger.info(
        f"--- STATISTICALLY SIGNIFICANT ANCHORS: {website2.upper()} ---\n"
        + significant.tail(10)[["word", "z_score", "count_site1", "count_site2"]].to_string(index=False)
    )

    logger.info("Computing narrative entropy over shared distinctive vocabulary...")
    h1, nh1, h2, nh2, shared_lex = calculate_narrative_entropy(t1, t2)
    entropy_lines = [
        f"--- NARRATIVE ENTROPY (shared vocabulary of {len(shared_lex):,} words) ---",
        f"{'Site':<15} | {'H (raw)':<10} | {'H (normalized)'}",
        "-" * 45,
        f"{website1.upper():<15} | {h1:<10} | {nh1}",
        f"{website2.upper():<15} | {h2:<10} | {nh2}",
        "Higher normalized entropy → more dispersed use of distinctive vocabulary",
    ]
    logger.info("\n".join(entropy_lines))

    logger.info(f"--- SEMANTIC DRIFT: TOP 15 MOST DRIFTED WORDS ---\n" + drift_df.head(15).to_string(index=False))

    divergence_lines = ["--- COMPREHENSIVE NARRATIVE DIVERGENCE REPORT (Word2Vec) ---"]
    if not w2v_reliable:
        divergence_lines.append(f"UNRELIABLE — vocab sizes ({vocab1}, {vocab2}) below threshold ({SemanticDivergenceConfig.W2V_VOCAB_THRESHOLD})")
        divergence_lines.append("Results shown for completeness only. Use GloVe section below.")

    divergence_lines.append(f"{'CONCEPT':<12} | {'FREQ1':<6} | {'FREQ2':<6} | {'JACCARD μ':<10} | {'STD':<7} | {'NULL μ':<8} | {'P-VAL':<7} | SIG")
    divergence_lines.append("-" * 85)

    for c in concepts:
        if c not in stability_stats:
            if w2v_reliable:
                divergence_lines.append(f"{c.upper():<12} | {'N/A':}")
            else:
                divergence_lines.append(f"{c.upper():<12} | {counts1.get(c,0):<6} | {counts2.get(c,0):<6} | {'skipped (unreliable)'}")
            continue

        f1 = counts1.get(c, 0)
        f2 = counts2.get(c, 0)
        mean_j, std_j = stability_stats[c]

        if w2v_reliable:
            null_scores = calculate_null_baseline(s1, s2, c, iterations=50)
            null_mean   = round(np.mean(null_scores), 4) if null_scores else 0.0
            p_value     = round(np.mean([s >= mean_j for s in null_scores]), 3) if null_scores else 1.0
            is_sig      = "significant" if p_value < 0.05 else ""
        else:
            null_mean, p_value, is_sig = "N/A", "N/A", ""

        divergence_lines.append(f"{c.upper():<12} | {f1:<6} | {f2:<6} | {mean_j:<10} | {std_j:<7} | {null_mean:<8} | {p_value:<7} | {is_sig}")
    logger.info("\n".join(divergence_lines))

    neighborhood_lines = ["--- NEIGHBORHOOD ANALYSIS PER CONCEPT (Word2Vec) ---"]
    if not w2v_reliable:
        neighborhood_lines.append("UNRELIABLE — shown for reference only")
    for c in concepts:
        res = analyze_neighborhood(c, m1, m2, rotation_matrix)
        if res:
            neighborhood_lines.append(f"{c.upper()} | Jaccard: {res['jaccard']} | Centroid drift: {res['c_drift']}")
            neighborhood_lines.append(f"    {website1} only: {res['excl1'][:8]}")
            neighborhood_lines.append(f"    {website2} only: {res['excl2'][:8]}")
            neighborhood_lines.append(f"    Shared:         {res['shared'][:8]}")
    logger.info("\n".join(neighborhood_lines))

    logger.info(
        f"{'='*70}\n"
        "  GLOVE ANALYSIS (pretrained, stable — Common Crawl 840B tokens)\n"
        "  No alignment needed — both corpora share the same vector space.\n"
        f"{'='*70}"
    )

    glove_div_lines = [
        "--- GLOVE NEIGHBORHOOD DIVERGENCE PER CONCEPT ---",
        f"{'CONCEPT':<12} | {'JACCARD':<9} | {'CENTROID DRIFT'}",
        "-" * 40,
    ]
    glove_results = {}
    for c in concepts:
        res = analyze_glove_neighborhood(c, s1, s2, top_k=20)
        if res:
            glove_results[c] = res
            glove_div_lines.append(f"{c.upper():<12} | {res['jaccard']:<9} | {res['c_drift']}")
        else:
            glove_div_lines.append(f"{c.upper():<12} | {'N/A':<9} | N/A")
    logger.info("\n".join(glove_div_lines))

    glove_detail_lines = ["--- GLOVE NEIGHBORHOOD DETAILS PER CONCEPT ---"]
    for c, res in glove_results.items():
        glove_detail_lines.append(f"{c.upper()} | Jaccard: {res['jaccard']} | Centroid drift: {res['c_drift']}")
        glove_detail_lines.append(f"    {website1} only: {res['excl1'][:8]}")
        glove_detail_lines.append(f"    {website2} only: {res['excl2'][:8]}")
        glove_detail_lines.append(f"    Shared:         {res['shared'][:8]}")
        glove_detail_lines.append("    Centroid semantic region:")
        glove_detail_lines.append(f"    {website1} centroid → {res['nn_c1']}")
        glove_detail_lines.append(f"    {website2} centroid → {res['nn_c2']}")
    logger.info("\n".join(glove_detail_lines))

    comparison_lines = ["--- METHOD COMPARISON: Word2Vec vs GloVe ---"]
    if not w2v_reliable:
        comparison_lines.append(f"W2V results unreliable (vocab: {vocab1}/{vocab2}). GloVe is primary analysis.")
    comparison_lines.append(f"{'CONCEPT':<12} | {'W2V Jaccard':<13} | {'W2V C-drift':<13} | {'GLoVe Jaccard':<15} | {'GLoVe C-drift'}")
    comparison_lines.append("-" * 75)
    for c in concepts:
        w2v_res   = analyze_neighborhood(c, m1, m2, rotation_matrix)
        glove_res = glove_results.get(c)
        w2v_j     = f"{w2v_res['jaccard']:.4f}"   if w2v_res   else "N/A"
        w2v_cd    = f"{w2v_res['c_drift']:.4f}"   if w2v_res   else "N/A"
        glv_j     = f"{glove_res['jaccard']:.4f}" if glove_res else "N/A"
        glv_cd    = f"{glove_res['c_drift']:.4f}" if glove_res else "N/A"
        comparison_lines.append(f"{c.upper():<12} | {w2v_j:<13} | {w2v_cd:<13} | {glv_j:<15} | {glv_cd}")
    logger.info("\n".join(comparison_lines))

    logger.info("""Interpretation guide:
  High W2V / Low GloVe  → outlet-specific framing not captured by pretrained vectors
                           (corpus-idiosyncratic usage, possible domain-specific sense)
  High GloVe / Low W2V  → divergence in general associations, W2V unstable on this word
                           (likely corpus size issue — treat W2V result with caution)
  Both high             → robust divergence confirmed by both methods
  Both low              → concept used similarly across both outlets""")

    logger.info("Generating visualizations...")
    plot_polarization(marker_df, website1, website2)

    logger.info("Word2Vec semantic maps:")
    for concept in concepts:
        plot_semantic_map(concept, m1, m2, rotation_matrix, website1, website2)

    logger.info("GloVe semantic maps:")
    for concept in concepts:
        plot_glove_semantic_map(concept, s1, s2, website1, website2)

    logodds_csv = SemanticDivergenceConfig.LOGODDS_CSV_PATTERN.format(website1=website1, website2=website2)
    marker_df.to_csv(logodds_csv, index=False)
    drift_df.to_csv(f"{website1}_vs_{website2}_w2v_drift.csv", index=False)

    if glove_results:
        glove_df = pd.DataFrame([
            {"concept":           c,
             "jaccard":           r["jaccard"],
             "centroid_drift":    r["c_drift"],
             "excl_site1":        str(r["excl1"][:10]),
             "excl_site2":        str(r["excl2"][:10]),
             "shared":            str(r["shared"][:10]),
             "centroid_nn_site1": str(r["nn_c1"]),
             "centroid_nn_site2": str(r["nn_c2"]),
            }
            for c, r in glove_results.items()
        ])
        glove_df.to_csv(f"{website1}_vs_{website2}_glove_divergence.csv", index=False)

    # Sentence lists, models, rotation matrix and glove_results are saved so
    # src/semantic_divergence/visualize.py can regenerate all plots without
    # re-running the (expensive) preprocessing/training/bootstrap steps above.
    sentences_pkl = SemanticDivergenceConfig.SENTENCES_PKL_PATTERN.format(website1=website1, website2=website2)
    with open(sentences_pkl, "wb") as f:
        pickle.dump({"s1": s1, "s2": s2}, f)

    w2v_model1 = SemanticDivergenceConfig.W2V_MODEL_PATTERN.format(website=website1)
    w2v_model2 = SemanticDivergenceConfig.W2V_MODEL_PATTERN.format(website=website2)
    m1.save(w2v_model1)
    m2.save(w2v_model2)

    rotation_npy = SemanticDivergenceConfig.ROTATION_NPY_PATTERN.format(website1=website1, website2=website2)
    np.save(rotation_npy, rotation_matrix)

    glove_results_pkl = SemanticDivergenceConfig.GLOVE_RESULTS_PKL_PATTERN.format(website1=website1, website2=website2)
    with open(glove_results_pkl, "wb") as f:
        pickle.dump(glove_results, f)

    meta_pkl = SemanticDivergenceConfig.META_PKL_PATTERN.format(website1=website1, website2=website2)
    with open(meta_pkl, "wb") as f:
        pickle.dump({
            "website1":     website1,
            "website2":     website2,
            "concepts":     concepts,
            "w2v_reliable": w2v_reliable,
        }, f)

    logger.info(
        "Saved:\n"
        f"   {logodds_csv}\n"
        f"   {website1}_vs_{website2}_w2v_drift.csv\n"
        f"   {website1}_vs_{website2}_glove_divergence.csv\n"
        f"   {sentences_pkl}\n"
        f"   {w2v_model1}  |  {w2v_model2}\n"
        f"   {rotation_npy}\n"
        f"   {glove_results_pkl}\n"
        f"   {meta_pkl}"
    )


if __name__ == "__main__":
    main()
