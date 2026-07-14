import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import warnings
from src.utils.constants import AgencyConfig

warnings.filterwarnings('ignore')

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def plot_agency_violence(website, agency_csv=None, agency_threshold=AgencyConfig.AGENCY_THRESHOLD, violence_threshold=AgencyConfig.VIOLENCE_THRESHOLD):
    """Generates an Agency Ratio × Violence Rate scatter plot for one outlet."""
    if agency_csv is None:
        agency_csv = f"{website}_agency_actions.csv"

    print(f"Generating Agency-Violence scatter for: {website}")

    try:
        df = pd.read_csv(agency_csv)
    except FileNotFoundError:
        print(f"⚠️ Error: Could not find {agency_csv}")
        return

    # Exclude pronoun-level labels (the IN_GROUP/OUT_GROUP values produced by
    # AgencyConfig.PRONOUN_GROUPS in extract.py)
    df = df[~df['Label'].isin(set(AgencyConfig.PRONOUN_GROUPS.values()))]

    summary = (
        df.groupby('Label')
        .agg(
            agency_ratio  = ('Role',       lambda x: (x == 'AGENT').sum() / len(x)),
            violence_rate = ('Is_Violent', 'mean'),
            n             = ('Role',       'count'),
        )
        .reset_index()
        .round(4)
    )

    # Drop sparse labels
    sparse = summary[summary['n'] < AgencyConfig.MIN_N]['Label'].tolist()
    if sparse:
        print(f"  Dropping sparse labels (n < {AgencyConfig.MIN_N}): {sparse}")
    summary = summary[summary['n'] >= AgencyConfig.MIN_N]

    fig, ax = plt.subplots(figsize=(10, 7))

    # Dynamic axis limits
    ax.set_xlim(-0.05, 1.05)
    y_max = summary['violence_rate'].max()
    plot_y_max = max(y_max * 1.35, violence_threshold * 1.5) + 0.005
    ax.set_ylim(-0.005, plot_y_max)

    # Dynamic quadrant label positions
    x_lo = (-0.05 + agency_threshold) / 2
    x_hi = (agency_threshold + 1.05) / 2
    y_hi = plot_y_max - (plot_y_max * 0.05)  # top 5% margin
    y_lo = violence_threshold / 2

    ax.axvline(agency_threshold,   color='#bdc3c7', lw=1.0, ls='--', zorder=1)
    ax.axhline(violence_threshold, color='#bdc3c7', lw=1.0, ls='--', zorder=1)

    # va='top' for the upper labels so they hang down and never clip
    ax.text(x_lo, y_hi, 'Passive\nAggressors', color='#222222', fontsize=15, ha='center', va='top', style='italic', zorder=2)
    ax.text(x_hi, y_hi, 'Active\nAggressors',  color='#222222', fontsize=15, ha='center', va='top', style='italic', zorder=2)
    ax.text(x_lo, y_lo, 'Passive\nVictims',    color='#222222', fontsize=15, ha='center', va='center', style='italic', zorder=2)
    ax.text(x_hi, y_lo, 'Active\nDefenders',   color='#222222', fontsize=15, ha='center', va='center', style='italic', zorder=2)

    for _, row in summary.iterrows():
        color = AgencyConfig.LABEL_COLORS.get(row['Label'], AgencyConfig.DEFAULT_LABEL_COLOR)
        size  = min(150 + row['n'] * 0.05, 600)

        ax.scatter(
            row['agency_ratio'], row['violence_rate'],
            s=size, color=color,
            edgecolors='white', linewidths=0.8,
            alpha=0.88, zorder=3,
        )

        lbl = row['Label'].replace('_', '\n')
        x_offset = 0.015 if row['agency_ratio'] < 0.80 else -0.015
        ha = 'left' if row['agency_ratio'] < 0.80 else 'right'

        txt = ax.text(
            row['agency_ratio'] + x_offset, row['violence_rate'], lbl,
            fontsize=11, fontweight='bold',
            va='center', ha=ha, color=color, zorder=4,
        )
        txt.set_path_effects([PathEffects.withStroke(linewidth=2.5, foreground='white', alpha=0.9)])

    ax.set_xlabel("Agency ratio\n", fontweight='bold', fontsize=16, labelpad=25)
    ax.set_ylabel("\nViolence rate", fontweight='bold', fontsize=16, labelpad=25)

    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)
    ax.grid(True, ls='--', alpha=0.25, zorder=0)

    plt.tight_layout()
    out = f"{website}_agency_violence.png"
    plt.savefig(out, format='png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {out}")
    plt.show()
    plt.close(fig)


def main(website="alquds"):
    plot_agency_violence(
        website            = website,
        agency_threshold   = AgencyConfig.AGENCY_THRESHOLD,
        violence_threshold = AgencyConfig.VIOLENCE_THRESHOLD,
    )


if __name__ == "__main__":
    main()
