import pandas as pd
import pickle
from transformers import pipeline
import spacy
import networkx as nx
from networkx.algorithms.link_analysis.hits_alg import hits
import warnings
from tqdm import tqdm
from src.utils.constants import NamesDicts, Websites, NetworksConfig, PretrainedModels, PreprocessingConfig
from src.networks.common import build_graph


def main(website="rt"):
    warnings.filterwarnings('ignore')

    clean_data_file = PreprocessingConfig.STAGE_FINAL.format(website=website)
    nlp = spacy.load(PretrainedModels.SPACY_MODEL_LG)

    print("✨ Initializing RoBERTa Sentiment Transformer...")
    try:
        sentiment_task = pipeline(
            "sentiment-analysis",
            model=PretrainedModels.SENTIMENT_MODEL,
            device=0, batch_size=32,
        )
        print("   ✅ Running on GPU")
    except Exception as e:
        print(f"   ⚠️  GPU unavailable ({e}), falling back to CPU")
        sentiment_task = pipeline(
            "sentiment-analysis",
            model=PretrainedModels.SENTIMENT_MODEL,
            device=-1,
        )

    def roberta_polarity(text):
        try:
            result = sentiment_task(text[:512])[0]
            label = result['label']
            score = result['score']
            if label == 'positive':
                return score
            elif label == 'negative':
                return -score
            else:
                return 0.0
        except Exception:
            return 0.0

    is_il_pa        = website in Websites.WEBSITES_PALESTINE_ISRAEL
    active_entities = NamesDicts.IZ_PA_BASE if is_il_pa else NamesDicts.RU_UK_BASE
    theater_name    = Websites.THEATER_IL_PA if is_il_pa else Websites.THEATER_RU_UK
    search_keys     = sorted(list(set(list(NamesDicts.SYNONYM_MAP.keys()) + list(active_entities.keys()))), key=len, reverse=True)

    print(f"🌍 Theater Detected: {theater_name}")

    def get_entity_match(token, doc):
        chunk_text = next((chunk.text.lower() for chunk in doc.noun_chunks if token.i in range(chunk.start, chunk.end)), None)
        if not chunk_text:
            chunk_text = " ".join(t.text.lower() for t in doc[max(0, token.i - 2): min(len(doc), token.i + 3)])
        match = next((k for k in search_keys if k in chunk_text), None)
        if match:
            clean = NamesDicts.SYNONYM_MAP.get(match, match)
            if clean in active_entities:
                return clean
        return None

    def extract_triples(sent, doc):
        triples = []
        for token in sent:
            if token.pos_ != "VERB":
                continue
            subj, obj, is_passive = None, None, False

            active_subj = next((c for c in token.children if c.dep_ == "nsubj"), None)
            if active_subj:
                direct_obj = next((c for c in token.children if c.dep_ in ("dobj", "obj")), None)
                if direct_obj:
                    subj, obj = active_subj, direct_obj
                else:
                    for child in token.children:
                        if child.dep_ == "prep" and child.text.lower() in NetworksConfig.MEANINGFUL_PREPS:
                            pobj = next((c for c in child.children if c.dep_ == "pobj"), None)
                            if pobj:
                                subj, obj = active_subj, pobj
                                break

            if not subj:
                passive_subj = next((c for c in token.children if c.dep_ == "nsubjpass"), None)
                if passive_subj:
                    is_passive = True
                    agent_prep = next((c for c in token.children if c.dep_ == "agent"), None)
                    if agent_prep:
                        agent_noun = next((c for c in agent_prep.children if c.dep_ == "pobj"), None)
                        if agent_noun:
                            subj, obj = agent_noun, passive_subj

            if subj and obj:
                triples.append((subj, token, obj, is_passive))

        return triples

    def build_sentiment_cache(texts_list):
        print("📦 Collecting unique sentences for RoBERTa batch scoring...")
        unique_sents = set()
        for doc in nlp.pipe(texts_list, batch_size=50, disable=["ner"]):
            for sent in doc.sents:
                unique_sents.add(sent.text.strip())

        unique_sents = list(unique_sents)
        polarities = {}
        batch_size = 64

        print(f"🤖 Scoring {len(unique_sents)} unique sentences with RoBERTa...")
        for i in tqdm(range(0, len(unique_sents), batch_size)):
            batch = unique_sents[i: i + batch_size]
            truncated = [s[:512] for s in batch]
            try:
                results = sentiment_task(truncated)
                for sent_text, result in zip(batch, results):
                    label, score = result['label'], result['score']
                    if label == 'positive':
                        polarities[sent_text] = score
                    elif label == 'negative':
                        polarities[sent_text] = -score
                    else:
                        polarities[sent_text] = 0.0
            except Exception:
                for sent_text in batch:
                    polarities[sent_text] = 0.0

        return polarities

    df = pd.read_csv(clean_data_file)
    texts_list = df['text'].dropna().astype(str).tolist()

    all_edges = []

    print(f"🚀 Building Narrative Network for {website}...")

    sentiment_cache = build_sentiment_cache(texts_list)

    for doc in tqdm(nlp.pipe(texts_list, batch_size=50, disable=["ner"]), total=len(texts_list)):
        for sent in doc.sents:
            sentiment = sentiment_cache.get(sent.text.strip(), 0.0)
            for subj_tok, verb_tok, obj_tok, is_passive in extract_triples(sent, doc):
                s_clean = get_entity_match(subj_tok, doc)
                o_clean = get_entity_match(obj_tok, doc)
                if s_clean and o_clean and s_clean != o_clean:
                    all_edges.append({
                        "source":    s_clean,
                        "target":    o_clean,
                        "verb":      verb_tok.lemma_,
                        "sentiment": sentiment,
                        "passive":   is_passive,
                        "label_s":   active_entities[s_clean],
                        "label_o":   active_entities[o_clean],
                    })

    df_edges = pd.DataFrame(all_edges)
    print(f"   Extracted {len(df_edges)} directed triples ({df_edges['passive'].sum()} from passive constructions)")

    edges_csv = NetworksConfig.EDGES_CSV_PATTERN.format(website=website)
    df_edges.to_csv(edges_csv, index=False)
    print(f"✅ Saved raw edges to {edges_csv}")

    G = build_graph(df_edges)

    centrality = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
    hub_scores, authority_scores = hits(G, max_iter=1000, normalized=True)

    def sentiment_weighted_agency(G):
        agency = {}
        for n in G.nodes():
            out_weighted = sum(G[n][v]['mean_sentiment'] * G[n][v]['weight'] for v in G.successors(n))
            in_weighted = sum(G[u][n]['mean_sentiment'] * G[u][n]['weight'] for u in G.predecessors(n))
            total_weight = (sum(G[n][v]['weight'] for v in G.successors(n)) + sum(G[u][n]['weight'] for u in G.predecessors(n)))
            agency[n] = (out_weighted - in_weighted) / total_weight if total_weight > 0 else 0.0
        return agency

    agency = sentiment_weighted_agency(G)

    report_df = pd.DataFrame({
        'Centrality_Pivot': centrality,
        'Hub_Score':        hub_scores,
        'Authority_Score':  authority_scores,
        'Sentiment_Agency': agency,
    })
    report_df.index.name = 'entity'
    report_df['Label'] = report_df.index.map(active_entities)

    print(f"\n--- NARRATIVE NETWORK REPORT: {website} ---")
    print(report_df.sort_values(by='Centrality_Pivot', ascending=False).to_string())

    metrics_csv = NetworksConfig.METRICS_CSV_PATTERN.format(website=website)
    report_df.to_csv(metrics_csv)
    print(f"\n✅ Saved to {metrics_csv}")

    # Saves active_entities and theater_name so visualize.py needs no manual config
    # beyond `website` -- it never has to re-derive the theater or re-run extraction.
    meta_pkl = NetworksConfig.META_PKL_PATTERN.format(website=website)
    with open(meta_pkl, "wb") as f:
        pickle.dump({
            "website":         website,
            "active_entities": active_entities,
            "theater_name":    theater_name,
        }, f)

    print(f"✅ Saved metadata to {meta_pkl}")

    return report_df


if __name__ == "__main__":
    main()
