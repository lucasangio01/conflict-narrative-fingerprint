import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
from src.utils.constants import CoOccurrenceConfig, PlotConfig


def plot_cooccurrence_visuals(website, pairs_csv, labels_csv, highlight_pairs=None):
    """
    Generates two co-occurrence visualizations, saved as PNGs:
      1. Label-level heatmap (mean Jaccard similarity, lower triangle)
      2. Jaccard vs. PMI scatter plot (entity-level pairs)

    Parameters
    ----------
    website         : str — outlet identifier, used for output filenames.
    pairs_csv       : str — path to *_cooccurrence_pairs.csv
    labels_csv      : str — path to *_cooccurrence_labels.csv (kept for structural
                      reference; the matrix itself is built from pairs_csv).
    highlight_pairs : list of (str, str) tuples, optional — entity pairs to
                      force-annotate on the scatter plot, in addition to the
                      automatic top-PMI / top-Jaccard selection.
    """
    print(f"Generating co-occurrence visualizations for: {website}")

    pairs_df = pd.read_csv(pairs_csv)

    # --- Label-level heatmap (mean Jaccard similarity) ---
    #
    # Raw co-occurrence counts are corpus-size-dependent and cannot be compared
    # across outlets. Mean Jaccard aggregated from entity-level pairs provides
    # a normalized, directly comparable association measure.
    # Within-label pairs (label_a == label_b) are excluded; they measure
    # intra-category co-occurrence, a distinct quantity from inter-label
    # association. Labels are reordered by ascending row-sum so the
    # least-connected labels appear at the top-left of the lower triangle and
    # the most analytically important pairs fill the visible bottom-right region.

    label_jaccard = (
        pairs_df[pairs_df['label_a'] != pairs_df['label_b']]
        .groupby(['label_a', 'label_b'])['jaccard']
        .mean()
        .reset_index()
    )

    unique_labels = sorted(set(label_jaccard['label_a']) | set(label_jaccard['label_b']))

    # Build symmetric matrix; cells with no entity pairs remain NaN (shown as blank)
    mat = pd.DataFrame(np.nan, index=unique_labels, columns=unique_labels)
    for _, row in label_jaccard.iterrows():
        mat.at[row['label_a'], row['label_b']] = row['jaccard']
        mat.at[row['label_b'], row['label_a']] = row['jaccard']

    row_sums = mat.fillna(0).sum(axis=1).sort_values(ascending=True)
    mat = mat.loc[row_sums.index, row_sums.index]

    clean_labels = [lbl.replace('_', '\n') for lbl in mat.index]
    vmax = float(np.nanmax(mat.values))

    fig1, ax1 = plt.subplots(figsize=(9, 7))

    mask_upper = np.triu(np.ones_like(mat, dtype=bool))

    sns.heatmap(
        mat,
        mask=mask_upper,
        annot=True,
        fmt='.3f',
        cmap='Reds',
        vmin=0,
        vmax=vmax,
        linewidths=0.6,
        linecolor='#e0e0e0',
        cbar_kws={"shrink": 0.90, "label": "\n\nMean Jaccard similarity"},
        ax=ax1,
        xticklabels=clean_labels,
        yticklabels=clean_labels,
        annot_kws={"size": 15},
    )
    ax1.collections[0].colorbar.ax.tick_params(labelsize=15)
    ax1.collections[0].colorbar.ax.yaxis.label.set_size(15)

    ax1.tick_params(axis='x', rotation=45, labelsize=10)
    ax1.tick_params(axis='y', rotation=0,  labelsize=10)
    ax1.set_xlabel("")
    ax1.set_ylabel("")

    plt.tight_layout()
    heatmap_out = f"{website}_label_heatmap.png"
    plt.savefig(heatmap_out, format='png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {heatmap_out}")
    plt.show()
    plt.close(fig1)

    # --- Jaccard vs. PMI scatter (entity-level pairs) ---
    #
    # Bubble sizes are normalized to [25, 475] so the largest bubbles don't
    # dominate the visual field and obscure neighbors. Alpha=0.40 reduces
    # overplotting in the dense low-Jaccard cluster. A horizontal reference
    # line at PMI=0 separates pairs that co-occur more than chance (PMI > 0)
    # from those that co-occur less (PMI < 0). Annotation candidates: top-5 by
    # PMI + top-3 by Jaccard, plus any caller-supplied highlight_pairs.
    # Directional text offsets are assigned relative to the median to
    # minimize label overlap. The size legend is built from manually scaled
    # scatter handles using the same normalization as the data points.

    fig2, ax2 = plt.subplots(figsize=(11, 7))
    ax2.tick_params(axis='x', labelsize=12)
    ax2.tick_params(axis='y', labelsize=12)

    cnt = pairs_df['count']
    sizes = 25 + 450 * (cnt - cnt.min()) / (cnt.max() - cnt.min() + 1e-9)

    ax2.axhline(0, color='#888888', lw=0.9, ls='--', zorder=1)
    ax2.text(
        pairs_df['jaccard'].max() * 0.97, 0.12,
        'PMI = 0', color='#888888', fontsize=9,
        ha='right', va='bottom', style='italic',
    )

    ax2.scatter(
        pairs_df['jaccard'], pairs_df['pmi'],
        s=sizes, alpha=0.40,
        color='#2b83ba', edgecolors='white', linewidths=0.4, zorder=2,
    )

    to_ann = pd.concat([
        pairs_df.nlargest(5, 'pmi'),
        pairs_df.nlargest(3, 'jaccard'),
    ]).drop_duplicates(subset=['entity_a', 'entity_b'])

    if highlight_pairs:
        hp_mask = pairs_df.apply(
            lambda r: (r['entity_a'], r['entity_b']) in highlight_pairs or
                      (r['entity_b'], r['entity_a']) in highlight_pairs,
            axis=1,
        )
        to_ann = pd.concat([to_ann, pairs_df[hp_mask]]).drop_duplicates(subset=['entity_a', 'entity_b'])

    x_mid = pairs_df['jaccard'].median()
    y_mid = pairs_df['pmi'].median()

    for _, row in to_ann.iterrows():
        label = f"{row['entity_a']} — {row['entity_b']}"
        dx = 15 if row['jaccard'] <= x_mid else -15
        dy = 10 if row['pmi'] >= y_mid else -14
        ax2.annotate(
            label,
            xy=(row['jaccard'], row['pmi']),
            xytext=(dx, dy),
            textcoords='offset points',
            fontsize=10,
            fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#444444', lw=0.7),
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.85),
        )

    for val, lbl in zip([50, 200, 500], ['50', '200', '500']):
        s = 25 + 450 * (val - cnt.min()) / (cnt.max() - cnt.min() + 1e-9)
        ax2.scatter([], [], s=s, color='#2b83ba', alpha=0.55, edgecolors='white', linewidths=0.4, label=lbl)

    ax2.legend(
        title="Co-occurrence\nCount",
        frameon=True, framealpha=0.85, edgecolor='#cccccc',
        loc='upper right', fontsize=12, title_fontsize=12,
    )

    ax2.set_xlabel("\nJaccard similarity (structural context overlap)", fontweight='bold')
    ax2.set_ylabel("PMI (association specificity)\n", fontweight='bold')
    ax2.grid(True, ls='--', alpha=0.3, zorder=0)

    plt.tight_layout()
    scatter_out = f"{website}_jaccard_pmi.png"
    plt.savefig(scatter_out, format='png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {scatter_out}")
    plt.show()
    plt.close(fig2)


def main(website="ynet_global"):
    warnings.filterwarnings('ignore')
    plt.rcParams.update({
        **PlotConfig.RCPARAMS_SERIF_BASE,
        "axes.labelsize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })
    plot_cooccurrence_visuals(
        website=website,
        pairs_csv=CoOccurrenceConfig.PAIRS_CSV_PATTERN.format(website=website),
        labels_csv=CoOccurrenceConfig.LABELS_CSV_PATTERN.format(website=website),
        highlight_pairs=CoOccurrenceConfig.highlight_pairs_by_website().get(website, []),
    )


if __name__ == "__main__":
    main()
