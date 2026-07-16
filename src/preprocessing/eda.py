import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
from src.utils.constants import EdaConfig, Websites, PreprocessingConfig


def safe_load(path):
    try:
        df = pd.read_csv(path, low_memory=False)
        print(f"    ✅ Loaded: {path} ({len(df)} rows)")
        return df
    except FileNotFoundError:
        print(f"    ❌ Not found: {path}")
        return None


def get_date_range(df):
    """Extract date range from a dataframe with a date column."""
    if df is None or "date" not in df.columns:
        return "N/A", "N/A"
    dates = pd.to_datetime(df["date"], dayfirst=True, errors="coerce").dropna()
    if len(dates) == 0:
        return "N/A", "N/A"
    return (dates.min().strftime("%d-%m-%Y"), dates.max().strftime("%d-%m-%Y"))


def collect_stats(outlet_dict, side_map, theater_name):
    rows = []
    final_dfs = {}

    for name, prefix in outlet_dict.items():
        print(f"\n  [{name}]")

        df_original = safe_load(PreprocessingConfig.STAGE_ORIGINAL.format(website=prefix))
        df_chunked = safe_load(PreprocessingConfig.STAGE_CHUNKED.format(website=prefix))
        df_filtered = safe_load(PreprocessingConfig.STAGE_FILTERED.format(website=prefix))
        df_english = safe_load(PreprocessingConfig.STAGE_ENGLISH.format(website=prefix))
        df_final = safe_load(PreprocessingConfig.STAGE_FINAL.format(website=prefix))

        n_raw_articles = len(df_original) if df_original is not None else np.nan

        n_chunks = len(df_chunked) if df_chunked is not None else np.nan
        n_filtered_chunks = len(df_filtered) if df_filtered is not None else np.nan
        n_final_chunks = len(df_final) if df_final is not None else np.nan

        date_min, date_max = get_date_range(df_english)

        retention_filter = (n_filtered_chunks / n_chunks * 100 if not np.isnan(n_chunks) and n_chunks > 0 else np.nan)
        retention_final = (n_final_chunks / n_chunks * 100 if not np.isnan(n_chunks) and n_chunks > 0 else np.nan)

        avg_chunk_len = np.nan
        total_words = np.nan
        avg_tox = np.nan
        avg_rel = np.nan

        if df_final is not None and "text" in df_final.columns:
            lengths = df_final["text"].dropna().apply(len)
            avg_chunk_len = lengths.mean()
            total_words = df_final["text"].dropna().apply(lambda x: len(str(x).split())).sum()

        if df_final is not None and "toxicity" in df_final.columns:
            avg_tox = df_final["toxicity"].mean()

        if df_final is not None and all(c in df_final.columns for c in ["filter1", "filter2", "filter3"]):
            avg_rel = df_final[["filter1", "filter2", "filter3"]].max(axis=1).mean()

        rows.append({
            "Outlet": name,
            "Side": side_map[name],
            "Scraped Articles": int(n_raw_articles) \
                if not np.isnan(n_raw_articles) else "N/A",
            "Total Chunks": int(n_chunks) \
                if not np.isnan(n_chunks) else "N/A",
            "Filtered Chunks": int(n_filtered_chunks) \
                if not np.isnan(n_filtered_chunks) else "N/A",
            "Final Chunks": int(n_final_chunks) \
                if not np.isnan(n_final_chunks) else "N/A",
            "Retention after filter (%)": round(retention_filter, 1) \
                if not np.isnan(retention_filter) else "N/A",
            "Retention final (%)": round(retention_final, 1) \
                if not np.isnan(retention_final) else "N/A",
            "Date Range": f"{date_min} → {date_max}",
            "Avg Chunk Length (chars)": round(avg_chunk_len, 0) \
                if not np.isnan(avg_chunk_len) else "N/A",
            "Total Words": int(total_words) \
                if not np.isnan(total_words) else "N/A",
            "Avg Toxicity": round(avg_tox, 4) \
                if not np.isnan(avg_tox) else "N/A",
            "Avg Relevance Score": round(avg_rel, 4) \
                if not np.isnan(avg_rel) else "N/A",
        })

        if df_final is not None:
            df_final = df_final.copy()
            df_final["outlet"] = name
            df_final["side"] = side_map[name]
            df_final["theater"] = theater_name
            df_final["date_parsed"] = pd.to_datetime(df_final["date"], dayfirst=True, errors="coerce") if "date" in df_final.columns else pd.NaT
            final_dfs[name] = df_final

    stats_df = pd.DataFrame(rows)
    combined_df = pd.concat(final_dfs.values(), ignore_index=True) if final_dfs else None

    return stats_df, combined_df


def print_summary(stats_df, theater_name):
    print(f"\n{'='*75}")
    print(f"  {theater_name.upper()} — CORPUS STATISTICS")
    print(f"{'='*75}")

    print("\n--- PIPELINE FUNNEL (per outlet) ---")
    cols1 = ["Outlet", "Side", "Scraped Articles", "Total Chunks", "Filtered Chunks", "Final Chunks", "Retention after filter (%)", "Retention final (%)"]
    available1 = [c for c in cols1 if c in stats_df.columns]
    print(stats_df[available1].to_string(index=False))

    print("\n--- TEXT QUALITY AND COVERAGE (per outlet) ---")
    cols2 = ["Outlet", "Side", "Date Range", "Avg Chunk Length (chars)", "Total Words", "Avg Toxicity", "Avg Relevance Score"]
    available2 = [c for c in cols2 if c in stats_df.columns]
    print(stats_df[available2].to_string(index=False))

    print("\n--- THEATER TOTALS ---")
    for col in ["Scraped Articles", "Total Chunks", "Filtered Chunks", "Final Chunks", "Total Words"]:
        if col in stats_df.columns:
            total = stats_df[col].apply(lambda x: x if isinstance(x, int) else 0).sum()
            print(f"  {col:<30}: {total:,}")

    print("\n--- BY SIDE ---")
    total_final = stats_df["Final Chunks"].apply(lambda x: x if isinstance(x, int) else 0).sum()
    for side in stats_df["Side"].unique():
        side_total = stats_df.loc[stats_df["Side"] == side, "Final Chunks"].apply(lambda x: x if isinstance(x, int) else 0).sum()
        pct = side_total / total_final * 100 if total_final > 0 else 0
        print(f"  {side:<20}: {side_total:,} final chunks ({pct:.1f}%)")


def plot_overview(combined_df, stats_df, theater_name, side_colors):
    if combined_df is None or len(stats_df) == 0:
        print(f"  No data available for {theater_name}")
        return

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f"Corpus Overview: {theater_name}", fontsize=15, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    outlets = stats_df["Outlet"].tolist()
    colors = [side_colors.get(s, "grey") for s in stats_df["Side"]]

    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(outlets))
    width = 0.25

    total_chunks = stats_df["Total Chunks"].apply(lambda x: x if isinstance(x, int) else 0).tolist()
    filtered_chunks = stats_df["Filtered Chunks"].apply(lambda x: x if isinstance(x, int) else 0).tolist()
    final_chunks = stats_df["Final Chunks"].apply(lambda x: x if isinstance(x, int) else 0).tolist()

    ax1.bar(x - width, total_chunks, width, label="Total Chunks", color="lightgrey", edgecolor="white")
    ax1.bar(x, filtered_chunks, width, label="After Filter", color="steelblue", edgecolor="white", alpha=0.8)
    ax1.bar(x + width, final_chunks, width, label="Final (+Toxicity)", color=colors, edgecolor="white", alpha=0.9)

    ax1.set_xticks(x)
    ax1.set_xticklabels(outlets, rotation=20, ha="right", fontsize=8)
    ax1.set_ylabel("Chunks")
    ax1.set_title("Pipeline Funnel per Outlet")
    ax1.legend(fontsize=7)
    ax1.spines[["top", "right"]].set_visible(False)

    ax2 = fig.add_subplot(gs[0, 1])
    if "date_parsed" in combined_df.columns:
        combined_df["month"] = combined_df["date_parsed"].dt.to_period("M")
        for outlet in outlets:
            subset = combined_df[combined_df["outlet"] == outlet]
            side = stats_df.loc[stats_df["Outlet"] == outlet, "Side"].values[0]
            monthly = subset.groupby("month").size()
            if len(monthly) > 0:
                monthly.index = monthly.index.to_timestamp()
                ax2.plot(monthly.index, monthly.values, label=outlet, color=side_colors.get(side, "grey"), alpha=0.75, linewidth=1.8)
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Chunks")
    ax2.set_title("Publication Volume Over Time")
    ax2.legend(fontsize=7)
    ax2.spines[["top", "right"]].set_visible(False)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

    ax3 = fig.add_subplot(gs[1, 0])
    if "toxicity" in combined_df.columns:
        tox_data = [combined_df.loc[combined_df["outlet"] == o, "toxicity"].dropna().tolist() for o in outlets]
        bp = ax3.boxplot(tox_data, labels=outlets, patch_artist=True, medianprops=dict(color="black", linewidth=2))
        for patch, outlet in zip(bp["boxes"], outlets):
            side = stats_df.loc[stats_df["Outlet"] == outlet, "Side"].values[0]
            patch.set_facecolor(side_colors.get(side, "grey"))
            patch.set_alpha(0.7)
        ax3.set_ylabel("Toxicity Score")
        ax3.set_title("Toxicity Distribution per Outlet")
        ax3.spines[["top", "right"]].set_visible(False)
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=20, ha="right")

    ax4 = fig.add_subplot(gs[1, 1])
    ret_filter = stats_df["Retention after filter (%)"].apply(lambda x: x if isinstance(x, float) else 0).tolist()
    ret_final = stats_df["Retention final (%)"].apply(lambda x: x if isinstance(x, float) else 0).tolist()
    x2 = np.arange(len(outlets))
    ax4.bar(x2 - 0.2, ret_filter, 0.35, label="After semantic filter", color="steelblue", edgecolor="white", alpha=0.8)
    ax4.bar(x2 + 0.2, ret_final, 0.35, label="After toxicity scoring", color=colors, edgecolor="white", alpha=0.85)
    ax4.axhline(y=100, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    ax4.set_xticks(x2)
    ax4.set_xticklabels(outlets, rotation=20, ha="right", fontsize=8)
    ax4.set_ylabel("Retention Rate (%)")
    ax4.set_title("Chunk Retention Rate per Outlet")
    ax4.legend(fontsize=7)
    ax4.spines[["top", "right"]].set_visible(False)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=s) for s, c in side_colors.items()]
    ax4.legend(handles=legend_elements, loc="lower right", fontsize=8)

    fname = (f"corpus_overview_{theater_name.replace(' ','_').replace('/','_')}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor="white")
    plt.show()
    print(f"  📊 Saved: {fname}")


def main():
    warnings.filterwarnings('ignore')

    print("🚀 Computing corpus statistics...\n")

    print(f"📂 {Websites.THEATER_RU_UK} Theater:")
    ru_uk_stats, ru_uk_df = collect_stats(EdaConfig.RU_UK_OUTLETS, EdaConfig.SIDE_MAP_RU_UK, Websites.THEATER_RU_UK)

    print(f"\n📂 {Websites.THEATER_IL_PA} Theater:")
    il_pa_stats, il_pa_df = collect_stats(EdaConfig.IL_PA_OUTLETS, EdaConfig.SIDE_MAP_IL_PA, Websites.THEATER_IL_PA)

    if ru_uk_stats is not None and len(ru_uk_stats) > 0:
        print_summary(ru_uk_stats, Websites.THEATER_RU_UK)

    if il_pa_stats is not None and len(il_pa_stats) > 0:
        print_summary(il_pa_stats, Websites.THEATER_IL_PA)

    print(f"\n{'='*75}")
    print(f"  COMBINED CORPUS — ALL THEATERS")
    print(f"{'='*75}")
    all_stats = pd.concat([s for s in [ru_uk_stats, il_pa_stats] if s is not None], ignore_index=True)
    for col in ["Scraped Articles", "Total Chunks", "Filtered Chunks", "Final Chunks", "Total Words"]:
        if col in all_stats.columns:
            total = all_stats[col].apply(lambda x: x if isinstance(x, int) else 0).sum()
            print(f"  {col:<30}: {total:,}")

    print("\n📊 Generating plots...")
    if ru_uk_df is not None:
        plot_overview(ru_uk_df, ru_uk_stats, Websites.THEATER_RU_UK, EdaConfig.RU_UK_COLORS)
    if il_pa_df is not None:
        plot_overview(il_pa_df, il_pa_stats, Websites.THEATER_IL_PA, EdaConfig.IL_PA_COLORS)

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
