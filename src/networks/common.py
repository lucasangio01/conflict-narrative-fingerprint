import networkx as nx


def build_graph(df_edges):
    """
    Aggregates raw (source, target) triples into a weighted directed graph:
    weight = number of triples between the pair, mean_sentiment = their
    average sentiment. Shared by extract.py (which builds the graph once to
    compute network metrics) and visualize.py (which rebuilds it from the
    saved edges csv so it can be redrawn without recomputing anything).
    """
    G = nx.DiGraph()
    edge_agg = (
        df_edges.groupby(["source", "target"])
        .agg(weight=("sentiment", "count"), sentiment_sum=("sentiment", "sum"))
        .reset_index()
    )
    for _, row in edge_agg.iterrows():
        G.add_edge(
            row["source"], row["target"],
            weight=int(row["weight"]),
            sentiment=row["sentiment_sum"],
            mean_sentiment=row["sentiment_sum"] / row["weight"],
        )
    return G
