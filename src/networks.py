import pandas as pd
from transformers import pipeline
import spacy
import networkx as nx
from networkx.algorithms.link_analysis.hits_alg import hits
import matplotlib.pyplot as plt
import numpy as np
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')


website = "ukpravda"
clean_data_file = f"{website}_final.csv"
nlp = spacy.load("en_core_web_lg")

print("✨ Initializing RoBERTa Sentiment Transformer...")
try:
    sentiment_task = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", device=0, batch_size=32)
    print("   ✅ Running on GPU")
except Exception as e:
    print(f"   ⚠️  GPU unavailable ({e}), falling back to CPU")
    sentiment_task = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", device=-1)


def roberta_polarity(text):
    """Returns a signed float in [-1, 1]: positive → +score, negative → -score, neutral → 0."""
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


SYNONYM_MAP = {
    "the united states": "usa", "u.s.": "usa", "united states": "usa",
    "washington": "usa", "america": "usa",
    "kyiv": "kiev", "vladimir zelensky": "zelensky", "volodymyr zelensky": "zelensky",
    "vladimir putin": "putin", "russian federation": "russia", "the kremlin": "kremlin",
    "the state of israel": "israel", "jerusalem": "israel",
    "benjamin netanyahu": "netanyahu", "bibi": "netanyahu",
    "israeli defense forces": "idf", "israeli army": "idf", "occupation forces": "idf",
    "the gaza strip": "gaza", "gaza strip": "gaza", "the strip": "gaza",
    "palestinian authority": "pa", "the palestinian authority": "pa",
    "hamas movement": "hamas",
    "the resistance": "militants", "fighters": "militants",
    "the west bank": "palestine"
}


RU_UK_BASE = {
    "russia": "RU_POLITICAL", "putin": "RU_POLITICAL", "kremlin": "RU_POLITICAL",
    "ukraine": "UKR_POLITICAL", "zelensky": "UKR_POLITICAL", "kiev": "UKR_POLITICAL",
    "usa": "WEST_ACTORS", "nato": "WEST_ACTORS", "eu": "WEST_ACTORS",
    "afu": "UKR_MILITARY",
    "civilians": "CIVILIANS"
}

IZ_PA_BASE = {
    "netanyahu": "ISR_POLITICAL", "israel": "ISR_POLITICAL", "idf": "ISR_MILITARY",
    "abbas": "PAL_POLITICAL", "pa": "PAL_POLITICAL", "palestine": "PAL_POLITICAL",
    "hamas": "PAL_ORG", "militants": "PAL_RESISTANCE", "gaza": "PAL_RESISTANCE",
    "usa": "INTL_ACTORS", "un": "INTL_ACTORS",
    "civilians": "CIVILIANS"
}


is_il_pa = website in ["ynet", "alquds", "jpost", "ynet_global"]
active_entities = IZ_PA_BASE if is_il_pa else RU_UK_BASE
theater_name = "Israel-Palestine" if is_il_pa else "Russia-Ukraine"
search_keys = sorted(list(set(list(SYNONYM_MAP.keys()) + list(active_entities.keys()))), key=len, reverse=True)

print(f"🌍 Theater Detected: {theater_name}")


MEANINGFUL_PREPS = {"at", "against", "into", "on", "upon", "toward", "towards", "over"}

def get_entity_match(token, doc):
    """
    Match a token to a known entity using its full noun chunk,
    falling back to a ±2 token window if no chunk is found.
    Using noun chunks avoids missing multi-word entities like 'Israeli Defense Forces'.
    """
    chunk_text = next((chunk.text.lower() for chunk in doc.noun_chunks if token.i in range(chunk.start, chunk.end)), None)
    if not chunk_text:
        chunk_text = " ".join(t.text.lower() for t in doc[max(0, token.i - 2): min(len(doc), token.i + 3)])

    match = next((k for k in search_keys if k in chunk_text), None)
    if match:
        clean = SYNONYM_MAP.get(match, match)
        if clean in active_entities:
            return clean
    return None


def extract_triples(sent, doc):
    """
    Extract (subject, verb, object) triples from a sentence,
    handling both active and passive constructions.

    Active:  "Hamas attacked civilians"   → nsubj + dobj/obj
    Passive: "Civilians were attacked by Hamas" → nsubjpass + agent (the 'by' phrase)
    """
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
                    if child.dep_ == "prep" and child.text.lower() in MEANINGFUL_PREPS:
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
                else:
                    pass

        if subj and obj:
            triples.append((subj, token, obj, is_passive))

    return triples


def build_sentiment_cache(texts_list):
    """
    Collects all unique sentences across the corpus, scores them in one
    batched RoBERTa pass, and returns a dict {sentence_text: polarity}.
    """
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
        # Use cached sentiment; fall back to 0.0 (not a live call) if missing
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
                    "label_o":   active_entities[o_clean]
                })

df_edges = pd.DataFrame(all_edges)
print(f"   Extracted {len(df_edges)} directed triples "
      f"({df_edges['passive'].sum()} from passive constructions)")
edges_filename = f"{website}_edges.csv"
df_edges.to_csv(edges_filename, index=False)
print(f"✅ Saved raw edges to {edges_filename}")


G = nx.DiGraph()

edge_agg = (df_edges.groupby(["source", "target"]).agg(weight=("sentiment", "count"), sentiment_sum=("sentiment", "sum")).reset_index())

for _, row in edge_agg.iterrows():
    G.add_edge(row["source"], row["target"], weight=int(row["weight"]), sentiment=row["sentiment_sum"], mean_sentiment=row["sentiment_sum"] / row["weight"])


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
    'Centrality_Pivot': centrality, 'Hub_Score': hub_scores, 'Authority_Score': authority_scores, 'Sentiment_Agency': agency})
report_df.index.name = 'entity'
report_df['Label'] = report_df.index.map(active_entities)

print(f"\n--- NARRATIVE NETWORK REPORT: {website} ---")
print(report_df.sort_values(by='Centrality_Pivot', ascending=False).to_string())

report_df.to_csv(f"{website}_network_metrics.csv")
print(f"\n✅ Saved to {website}_network_metrics.csv")


fig, axes = plt.subplots(1, 2, figsize=(20, 9))
pos = nx.spring_layout(G, k=1.2, seed=42)

unique_labels = list(set(active_entities.values()))
color_map = plt.colormaps['Set3']
label_to_color = {label: color_map(i / len(unique_labels)) for i, label in enumerate(unique_labels)}
node_colors = [label_to_color[active_entities[n]] for n in G.nodes()]

edges = list(G.edges())
edge_weights = [np.log1p(G[u][v]['weight']) * 2 for u, v in edges]
edge_colors = ['#d32f2f' if G[u][v]['mean_sentiment'] < 0 else '#388e3c' for u, v in edges]


def draw_graph(ax, node_sizes, title):
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes, node_color=node_colors, alpha=0.9)
    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_weights, edge_color=edge_colors,
                           alpha=0.4, arrowsize=20, connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight='bold')
    ax.set_title(title, fontsize=13)
    ax.axis('off')

draw_graph(axes[0], node_sizes=[centrality[n] * 10000 for n in G.nodes()], title=f"Narrative Centrality (Eigenvector)\n{website}")
draw_graph(axes[1], node_sizes=[authority_scores[n] * 10000 for n in G.nodes()], title=f"Narrative Salience (HITS Authority)\n{website}")

from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color='#d32f2f', linewidth=2, label='Negative sentiment'), Line2D([0], [0], color='#388e3c', linewidth=2, label='Positive sentiment')]
fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=10, frameon=False)

plt.suptitle(f"Narrative Topology: {theater_name} — {website}\n(Node size = centrality metric | Edge color = mean sentiment)", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(f"{website}_network_graph.png", dpi=150, bbox_inches='tight')
plt.show()
print(f"✅ Graph saved to {website}_network_graph.png")