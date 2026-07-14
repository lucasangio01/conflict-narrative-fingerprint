import functools
import numpy as np
import spacy
from collections import Counter
from scipy.spatial.distance import cosine
from src.utils.constants import PretrainedModels


@functools.lru_cache(maxsize=1)
def get_nlp():
    """
    Lazily loads en_core_web_lg on first call, then caches it -- importing
    this module (or compute.py/visualize.py, which import from it) must not
    eagerly load the model, since the CLI imports modules just to list them
    as menu options.

    En_core_web_lg ships with a parser -- do NOT add a sentencizer on top.
    Parser is needed for sentence segmentation for Word2Vec training.
    NER is disabled elsewhere (dictionary-based matching used instead).
    """
    return spacy.load(PretrainedModels.SPACY_MODEL_LG)


def get_glove_vector(word):
    """Returns the GloVe vector for a word from en_core_web_lg, or None."""
    token = get_nlp().vocab[word]
    if token.has_vector:
        return token.vector
    return None


def centroid_nearest_neighbors(centroid_vec, top_k=8, exclude=None):
    """
    Finds the top_k vocabulary words whose GloVe vectors are closest to a
    given centroid vector. This reveals the latent semantic region that a
    corpus's usage of a concept gravitates toward in GloVe space.

    For example, if one outlet's 'peace' centroid is nearest to
    {deal, agreement, negotiation} and another's is nearest to
    {rights, liberation, dignity}, that is the "smoking gun" of semantic
    divergence -- not just different neighbors, but different conceptual frames.

    exclude: set of words to suppress (e.g. the concept itself, stopwords)
    """
    exclude = exclude or set()
    centroid_norm = centroid_vec / (np.linalg.norm(centroid_vec) + 1e-12)

    scored = []
    for word in get_nlp().vocab:
        if not word.has_vector or not word.is_alpha or word.is_stop:
            continue
        if word.text.lower() in exclude or len(word.text) < 3:
            continue
        vec = word.vector / (np.linalg.norm(word.vector) + 1e-12)
        sim = float(np.dot(centroid_norm, vec))
        scored.append((word.text.lower(), sim))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [w for w, _ in scored[:top_k]]


def get_cooccurrence_neighbors(concept, sentences, top_k=20):
    """
    Finds the top_k most frequent content words that co-occur with `concept`
    in the same sentence (full sentence scope). Returns a Counter.most_common list.
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
    Both corpora share the same GloVe space (en_core_web_lg, Common Crawl 840B
    tokens), so no Procrustes alignment is needed -- the geometry is fixed and
    identical for both.

    Method: find each corpus's most frequent sentence-level co-occurrence
    neighbors of `concept`, compute the frequency-weighted GloVe centroid of
    each neighbor set, and compare via Jaccard distance (neighbor overlap) and
    cosine distance between centroids.
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

    freqs1 = {w: f for w, f in neighbors1}
    freqs2 = {w: f for w, f in neighbors2}
    total1 = sum(freqs1.get(w, 1) for w in words1)
    total2 = sum(freqs2.get(w, 1) for w in words2)

    c1 = np.sum([get_glove_vector(w) * (freqs1.get(w, 1) / total1) for w in words1], axis=0)
    c2 = np.sum([get_glove_vector(w) * (freqs2.get(w, 1) / total2) for w in words2], axis=0)

    c_drift = round(float(cosine(c1, c2)), 4)

    # Nearest neighbors of each centroid in GloVe space -- reveals the latent
    # semantic region each corpus's usage of the concept gravitates toward.
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


def analyze_neighborhood(word, m1, m2, R, top_k=15):
    """
    Compares the semantic neighborhoods of a word across two Word2Vec models
    aligned via Procrustes rotation R (see align_and_drift in compute.py).
    Jaccard distance measures neighborhood overlap in each model's own space
    (correct -- Jaccard is about which words are nearby, not cross-model
    distance). Centroid drift uses the rotation to compare neighborhood
    centers cross-model.
    """
    if word not in m1.wv or word not in m2.wv:
        return None

    n1 = [w for w, _ in m1.wv.most_similar(word, topn=top_k)]
    n2 = [w for w, _ in m2.wv.most_similar(word, topn=top_k)]
    set1, set2 = set(n1), set(n2)

    jaccard = 1 - (len(set1 & set2) / len(set1 | set2))
    c1      = np.mean([m1.wv[w] for w in n1], axis=0)
    c2      = np.mean([m2.wv[w] for w in n2], axis=0) @ R
    c_drift = cosine(c1, c2)

    return {
        "word":    word,
        "jaccard": round(jaccard, 4),
        "c_drift": round(c_drift, 4),
        "excl1":   sorted(set1 - set2),
        "excl2":   sorted(set2 - set1),
        "shared":  sorted(set1 & set2),
    }
