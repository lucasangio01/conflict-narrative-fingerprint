import pandas as pd
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.patheffects as PathEffects
import warnings
from src.utils.constants import NetworksConfig, PlotConfig
from src.networks.common import build_graph
from src.utils.logging_config import get_logger

logger = get_logger("NETWORKS")


def main(website="rt"):
    warnings.filterwarnings('ignore')
    plt.rcParams.update({
        **PlotConfig.RCPARAMS_SERIF_BASE,
        "axes.titlesize": 16,
        "axes.labelsize": 12,
    })

    with open(NetworksConfig.META_PKL_PATTERN.format(website=website), "rb") as f:
        meta = pickle.load(f)
        active_entities = meta["active_entities"]
        theater_name    = meta["theater_name"]

    df_metrics = pd.read_csv(NetworksConfig.METRICS_CSV_PATTERN.format(website=website), index_col='entity')
    df_edges   = pd.read_csv(NetworksConfig.EDGES_CSV_PATTERN.format(website=website))

    logger.info(f"Loaded data for {website} ({theater_name})")

    G = build_graph(df_edges)

    pos = nx.spring_layout(G, k=1.2, seed=42)

    unique_labels  = sorted(set(active_entities.values()))
    color_map      = plt.colormaps['Set3']
    label_to_color = {label: color_map(i / max(1, len(unique_labels) - 1)) for i, label in enumerate(unique_labels)}
    node_colors    = [label_to_color[active_entities.get(n, "OTHER")] for n in G.nodes()]

    edges        = list(G.edges())
    edge_weights = [np.log1p(G[u][v]['weight']) * 2 for u, v in edges]
    edge_colors  = ['#d32f2f' if G[u][v]['mean_sentiment'] < 0 else '#388e3c' for u, v in edges]

    eigen_sizes = [df_metrics.loc[n, 'Centrality_Pivot'] * 12000 if n in df_metrics.index else 0 for n in G.nodes()]
    auth_sizes  = [df_metrics.loc[n, 'Authority_Score']  * 12000 if n in df_metrics.index else 0 for n in G.nodes()]

    def draw_network(node_sizes, output_filename):
        fig, ax = plt.subplots(figsize=(12, 10))

        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_size=node_sizes,
            node_color=node_colors,
            alpha=0.9,
            edgecolors='white',
            linewidths=1.5,
        )
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            width=edge_weights,
            edge_color=edge_colors,
            alpha=0.5,
            arrowsize=25,
            connectionstyle="arc3,rad=0.15",
        )

        texts = nx.draw_networkx_labels(G, pos, ax=ax, font_size=16, font_weight='bold')
        for _, text_obj in texts.items():
            text_obj.set_path_effects([PathEffects.withStroke(linewidth=3.5, foreground='white', alpha=0.9)])

        ax.axis('off')

        sentiment_legend = [
            Line2D([0], [0], color='#d32f2f', linewidth=4, label='Negative sentiment'),
            Line2D([0], [0], color='#388e3c', linewidth=4, label='Positive sentiment'),
        ]
        ax.legend(
            handles=sentiment_legend,
            loc='lower center', ncol=2, fontsize=20, frameon=False,
            bbox_to_anchor=(0.5, -0.05),
        )

        plt.tight_layout()
        plt.savefig(output_filename, format='png', dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {output_filename}")
        plt.show()
        plt.close()

    logger.info("Generating centrality plot...")
    draw_network(eigen_sizes, f"{website}_centrality.png")

    logger.info("Generating authority plot...")
    draw_network(auth_sizes, f"{website}_authority.png")


if __name__ == "__main__":
    main()
