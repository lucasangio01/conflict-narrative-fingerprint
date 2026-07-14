import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
from src.utils.constants import Websites, AgencyConfig

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


def plot_agency_bars(theater):
    """
    Generates a grouped horizontal bar chart showing the Agency Ratio
    (proportion of verb appearances in grammatical agent position) for
    each entity label, grouped by outlet. A dashed reference line at 0.5
    marks the agent/patient parity point.

    Parameters
    ----------
    theater : str — 'ru_ua' or 'il_pa'
    """
    cfg = AgencyConfig.THEATER_CONFIG[theater]

    outlets        = cfg["outlets"]
    outlet_labels  = {o: Websites.DISPLAY_NAMES.get(o, o) for o in outlets}
    outlet_colors  = cfg["outlet_colors"]
    labels_ordered = cfg["labels"]
    csv_pattern    = cfg["agency_csv_pattern"]

    print(f"Generating grouped agency bar chart for theater: {theater}")

    frames = []
    for outlet in outlets:
        path = csv_pattern.format(outlet=outlet)
        df = pd.read_csv(path)
        summary = (
            df.groupby('Label')
              .agg(agency_ratio=('Role', lambda x: (x == 'AGENT').sum() / len(x)), n=('Role', 'count'))
              .reset_index()
        )
        summary['outlet'] = outlet
        frames.append(summary)

    combined = pd.concat(frames, ignore_index=True)

    # Suppress sparse cells
    combined.loc[combined['n'] < AgencyConfig.MIN_N, 'agency_ratio'] = np.nan

    combined = combined[combined['Label'].isin(labels_ordered)]

    n_labels  = len(labels_ordered)
    n_outlets = len(outlets)
    bar_width = 0.16
    x = np.arange(n_labels)

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, outlet in enumerate(outlets):
        sub  = combined[combined['outlet'] == outlet].set_index('Label')
        vals = [sub.loc[lbl, 'agency_ratio'] if lbl in sub.index else np.nan for lbl in labels_ordered]
        # Center the group of bars symmetrically around each x tick
        offsets = x + (i - n_outlets / 2 + 0.5) * bar_width

        bars = ax.bar(
            offsets, vals,
            width=bar_width,
            color=outlet_colors[outlet],
            label=outlet_labels[outlet],
            edgecolor='white', linewidth=0.5,
            alpha=0.88,
        )

        # Value annotations at bar tops — suppressed for NaN bars
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9.5, color='#444444',
                )

    # Agent/patient parity reference line
    ax.axhline(0.5, color='#888888', lw=1.0, ls='--', zorder=1)
    ax.text(n_labels - 0.45, 0.515, 'Agent / Patient\nparity', color='#888888', fontsize=11, va='bottom', ha='right', style='italic')

    clean_labels = [lbl.replace('_', '\n') for lbl in labels_ordered]
    ax.set_xticks(x)
    ax.set_xticklabels(clean_labels, fontsize=12)

    ax.set_ylabel("Agency Ratio\n", fontweight='bold', fontsize=14)
    ax.set_ylim(0, min(1.0, combined['agency_ratio'].max(skipna=True) + 0.15))
    ax.tick_params(axis='y', labelsize=10)

    ax.legend(frameon=True, framealpha=0.9, edgecolor='#cccccc', fontsize=11, loc='upper right')
    ax.grid(axis='y', ls='--', alpha=0.3, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    out = f"{theater}_agency_bars.png"
    plt.savefig(out, format='png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {out}")
    plt.show()
    plt.close(fig)


def main(theater="il_pa"):
    plot_agency_bars(theater=theater)


if __name__ == "__main__":
    main()
