import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import matplotlib.gridspec as gridspec
import warnings
import math
from src.utils.constants import Websites, NamesDicts, CharactersConfig

warnings.filterwarnings('ignore')

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

MORALITY_CMAP = mcolors.LinearSegmentedColormap.from_list("morality", CharactersConfig.MORALITY_CMAP_COLORS)

def morality_to_color(val):
    norm = (val - CharactersConfig.MORALITY_VMIN) / (CharactersConfig.MORALITY_VMAX - CharactersConfig.MORALITY_VMIN)
    return MORALITY_CMAP(np.clip(norm, 0, 1))

def plot_adjective_bars(website, target_labels, characters_csv=None, top_n=8):
    if characters_csv is None:
        characters_csv = f"{website}_character_adjectives.csv"

    df = pd.read_csv(characters_csv)
    n_panels = len(target_labels)

    # Compute grid dimensions automatically
    if n_panels == 3:
        nrows, ncols = 2, 2
    else:
        ncols = min(n_panels, 3)
        nrows = math.ceil(n_panels / ncols)
    fig = plt.figure(figsize=(14 * ncols / 4, 5 * nrows))
    gs  = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.7, wspace=0.8)

    axes_list = []
    for idx in range(n_panels):
        row, col = divmod(idx, ncols)
        axes_list.append(fig.add_subplot(gs[row, col]))

    # Hide any unused cells in the last row
    total_cells = nrows * ncols
    for idx in range(n_panels, total_cells):
        row, col = divmod(idx, ncols)
        fig.add_subplot(gs[row, col]).set_visible(False)

    full_name = Websites.DISPLAY_NAMES.get(website, website.upper())

    for ax, label in zip(axes_list, target_labels):
        sub = df[df['label'] == label]
        sub = sub[~sub['adjective'].str.replace('not_', '').isin(CharactersConfig.ADJ_STOPWORDS)]

        top = (sub.groupby('adjective')
               .agg(count=('adjective', 'count'), competence=('competence', 'mean'), morality=('morality', 'mean'))
               .reset_index()
               .sort_values('count', ascending=False)
               .head(top_n)
               .sort_values('competence'))

        if top.empty:
            ax.set_visible(False)
            continue

        colors = [morality_to_color(m) for m in top['morality']]
        bars = ax.barh(top['adjective'], top['competence'], color=colors, edgecolor='white', linewidth=0.5, height=0.65)

        for bar, cnt in zip(bars, top['count']):
            x = bar.get_width()
            x_pos = x + 0.08 if x >= 0 else x - 0.08
            ha = 'left' if x >= 0 else 'right'
            ax.text(x_pos, bar.get_y() + bar.get_height() / 2, f'n={cnt}', va='center', ha=ha, fontsize=9, color='#444444')

        ax.axvline(0, color='#333333', lw=1.1, zorder=3)
        ax.set_xlabel("\nCompetence score", fontsize=13, fontweight='bold', labelpad=10)
        ax.set_title(f"{full_name}  ·  {label.replace('_', ' ')}\n", fontsize=12, fontweight='bold', pad=10)

        ax.tick_params(axis='y', labelsize=12)
        ax.tick_params(axis='x', labelsize=12)
        ax.grid(axis='x', ls='--', alpha=0.3, zorder=0)
        ax.spines[['top', 'right']].set_visible(False)

    sm = plt.cm.ScalarMappable(cmap=MORALITY_CMAP, norm=mcolors.Normalize(vmin=CharactersConfig.MORALITY_VMIN, vmax=CharactersConfig.MORALITY_VMAX))
    sm.set_array([])
    # Positioning the colorbar so it doesn't overlap the centered 3rd plot
    cbar = fig.colorbar(sm, ax=axes_list, orientation='vertical', fraction=0.02, pad=0.08, shrink=0.7)
    cbar.set_label("\nMorality score", fontsize=13, fontweight='bold', labelpad=15)
    cbar.ax.tick_params(labelsize=12)

    plt.savefig(f"{website}_adjective_bars.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()


def main(website="ynet_global", target_labels=None):
    if target_labels is None:
        is_il_pa = website in Websites.WEBSITES_PALESTINE_ISRAEL
        active_entities = NamesDicts.IZ_PA_BASE if is_il_pa else NamesDicts.RU_UK_BASE
        target_labels = sorted(set(active_entities.values()))

    plot_adjective_bars(website, target_labels, top_n=CharactersConfig.TOP_N)


if __name__ == "__main__":
    main()
