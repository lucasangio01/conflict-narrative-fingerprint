import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
from src.utils.constants import Websites
from src.semantic_divergence.common import (
    get_glove_vector, analyze_glove_neighborhood, analyze_neighborhood,
)


def main(website1="kpru", website2="ukpravda"):
    with open(f"{website1}_vs_{website2}_meta.pkl", "rb") as f:
        meta = pickle.load(f)
        concepts     = meta["concepts"]
        w2v_reliable = meta["w2v_reliable"]

    with open(f"{website1}_vs_{website2}_sentences.pkl", "rb") as f:
        sents = pickle.load(f)
        s1, s2 = sents["s1"], sents["s2"]

    with open(f"{website1}_vs_{website2}_glove_results.pkl", "rb") as f:
        glove_results = pickle.load(f)

    marker_df       = pd.read_csv(f"{website1}_vs_{website2}_logodds.csv")
    rotation_matrix = np.load(f"{website1}_vs_{website2}_rotation.npy")

    m1 = Word2Vec.load(f"{website1}_w2v.model")
    m2 = Word2Vec.load(f"{website2}_w2v.model")

    print(f"✅ Loaded all objects for {website1} vs {website2}")
    print(f"   W2V reliable: {w2v_reliable}")
    print(f"   Concepts: {concepts}")

    def plot_polarization(marker_df, label1, label2, top_n=15):
        significant = marker_df[marker_df["is_significant"]]
        plot_df = pd.concat([significant.head(top_n), significant.tail(top_n)])
        plt.figure(figsize=(10, 8))
        colors = ["skyblue" if x > 0 else "salmon" for x in plot_df["log_odds"]]
        plt.barh(plot_df["word"], plot_df["log_odds"], color=colors)
        plt.axvline(0, color="black", linewidth=0.8)
        plt.xlabel("\nLog-odds ratio (z-score significant only)")
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=11)
        plt.grid(alpha=0.1)
        plt.tight_layout()
        plt.savefig(f"{label1}_vs_{label2}_lexicons_bar.png", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

    def plot_glove_semantic_map(concept, sents1, sents2, label1, label2, top_k=10):
        res = analyze_glove_neighborhood(concept, sents1, sents2, top_k=top_k)
        if res is None:
            print(f"   '{concept}' — insufficient GloVe coverage, skipping.")
            return

        n1 = [w for w in res["excl1"][:top_k//2] + res["shared"][:top_k//2] if get_glove_vector(w) is not None][:top_k]
        n2 = [w for w in res["excl2"][:top_k//2] + res["shared"][:top_k//2] if get_glove_vector(w) is not None][:top_k]

        # Pivots at each corpus's frequency-weighted centroid in GloVe space.
        # This makes the two pivots informationally distinct and directly
        # visualizes centroid drift: the further apart the pivots, the higher
        # the centroid drift value reported in the text.
        c1 = res["centroid1"]
        c2 = res["centroid2"]

        all_vecs = np.vstack([c1, c2] + [get_glove_vector(w) for w in n1] + [get_glove_vector(w) for w in n2])
        coords = PCA(n_components=2).fit_transform(all_vecs)

        plt.figure(figsize=(10, 7))

        def draw_cloud(pts, color, lbl):
            if len(pts) >= 3:
                try:
                    hull = ConvexHull(pts)
                    plt.fill(pts[hull.vertices, 0], pts[hull.vertices, 1], color=color, alpha=0.1, label=f"{lbl} context")
                except Exception:
                    pass

        offset = 2
        draw_cloud(coords[offset:offset+len(n1)], "skyblue", Websites.DISPLAY_NAMES.get(label1, label1))
        draw_cloud(coords[offset+len(n1):],       "salmon",  Websites.DISPLAY_NAMES.get(label2, label2))
        plt.scatter(coords[offset:offset+len(n1), 0], coords[offset:offset+len(n1), 1], color="skyblue", s=40, alpha=0.6)
        plt.scatter(coords[offset+len(n1):, 0],       coords[offset+len(n1):, 1],       color="salmon",  s=40, alpha=0.6)
        # Pivots represent each corpus's centroid -- distinct points in GloVe space
        plt.scatter(coords[0, 0], coords[0, 1], color="skyblue", s=500, marker="*",
                    label=f"{Websites.DISPLAY_NAMES.get(label1, label1)} centroid", edgecolors="navy", zorder=10)
        plt.scatter(coords[1, 0], coords[1, 1], color="salmon", s=400, marker="o",
                    label=f"{Websites.DISPLAY_NAMES.get(label2, label2)} centroid", edgecolors="darkred", zorder=10)
        for i, w in enumerate(n1):
            plt.annotate(w, (coords[offset+i, 0], coords[offset+i, 1]), color="navy", fontsize=13, alpha=0.9, xytext=(3, 3), textcoords="offset points")
        for i, w in enumerate(n2):
            plt.annotate(w, (coords[offset+len(n1)+i, 0], coords[offset+len(n1)+i, 1]), color="darkred", fontsize=13, alpha=0.9, xytext=(3, 3), textcoords="offset points")
        plt.legend(fontsize=9)
        plt.grid(alpha=0.1)
        plt.tight_layout()
        plt.savefig(f"{label1}_vs_{label2}_glove_{concept}.png", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

    def plot_semantic_map(word, m1, m2, R, label1, label2, top_k=10):
        if word not in m1.wv or word not in m2.wv:
            print(f"   '{word}' not in one or both models — skipping.")
            return
        metrics = analyze_neighborhood(word, m1, m2, R, top_k=top_k)
        n1 = [w for w, _ in m1.wv.most_similar(word, topn=top_k)]
        n2 = [w for w, _ in m2.wv.most_similar(word, topn=top_k)]
        all_vecs = np.vstack([m1.wv[word], m2.wv[word] @ R] + [m1.wv[w] for w in n1] + [m2.wv[w] @ R for w in n2])
        coords = PCA(n_components=2).fit_transform(all_vecs)
        plt.figure(figsize=(10, 7))

        def draw_cloud(pts, color, lbl):
            if len(pts) >= 3:
                try:
                    hull = ConvexHull(pts)
                    plt.fill(pts[hull.vertices, 0], pts[hull.vertices, 1], color=color, alpha=0.1, label=f"{lbl} context")
                except Exception:
                    pass

        draw_cloud(coords[2:2+top_k], "skyblue", Websites.DISPLAY_NAMES.get(label1, label1))
        draw_cloud(coords[2+top_k:],  "salmon",  Websites.DISPLAY_NAMES.get(label2, label2))
        plt.scatter(coords[2:2+top_k, 0], coords[2:2+top_k, 1], color="skyblue", s=40, alpha=0.6)
        plt.scatter(coords[2+top_k:, 0],  coords[2+top_k:, 1],  color="salmon",  s=40, alpha=0.6)
        plt.scatter(coords[0, 0], coords[0, 1], color="skyblue", s=500, marker="*",
                    label=f"{Websites.DISPLAY_NAMES.get(label1, label1)} pivot", edgecolors="navy", zorder=10)
        plt.scatter(coords[1, 0], coords[1, 1], color="salmon", s=400, marker="o",
                    label=f"{Websites.DISPLAY_NAMES.get(label2, label2)} pivot", edgecolors="darkred", zorder=10)
        for i in range(top_k):
            plt.annotate(n1[i], (coords[i+2, 0], coords[i+2, 1]), color="navy", fontsize=12, alpha=0.8, xytext=(3, 3), textcoords="offset points")
            plt.annotate(n2[i], (coords[i+2+top_k, 0], coords[i+2+top_k, 1]), color="darkred", fontsize=12, alpha=0.8, xytext=(3, 3), textcoords="offset points")
        plt.legend(fontsize=9)
        plt.grid(alpha=0.1)
        plt.tight_layout()
        plt.savefig(f"{label1}_vs_{label2}_w2v_{word}.png", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

    print("\n📊 Generating polarized lexicon chart...")
    plot_polarization(marker_df, website1, website2)

    if w2v_reliable:
        print("\n  Word2Vec semantic maps:")
        for concept in concepts:
            plot_semantic_map(concept, m1, m2, rotation_matrix, website1, website2)
    else:
        print("\n  ⚠️  Skipping W2V maps — vocabulary below reliability threshold.")

    print("\n  GloVe semantic maps:")
    for concept in concepts:
        plot_glove_semantic_map(concept, s1, s2, website1, website2)

    print("\n✅ All plots saved.")


if __name__ == "__main__":
    main()
