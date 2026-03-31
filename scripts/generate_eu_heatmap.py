#!/usr/bin/env python3
"""Generate EU bidding zone heatmaps for business analysis.

Produces choropleth maps with:
- Colored zone polygons (country boundaries, split for DK/NO/SE)
- Interconnector lines with width proportional to capacity
- Multiple metric views: volatility, negative prices, attractiveness

Outputs:
    output/eu_zone_heatmap.png        -- 4-panel static map
    output/eu_attractiveness_map.png  -- single large attractiveness map
    output/zone_metrics.csv           -- raw metrics table

Run: uv run python scripts/generate_eu_heatmap.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, LineString, MultiPolygon

from da_forecast.config import ZONES, ZONE_LABELS, ZONE_EIC, INTERCONNECTORS
from da_forecast.data import available_zones, load_all

OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Zone centroids for labels and interconnector lines
ZONE_COORDS = {
    "DK_1": (56.2, 9.0), "DK_2": (55.5, 12.2),
    "NO_1": (60.0, 11.0), "NO_2": (58.5, 7.5), "NO_3": (63.5, 10.5),
    "NO_4": (69.0, 18.0), "NO_5": (60.5, 5.5),
    "SE_1": (66.5, 18.0), "SE_2": (63.0, 16.0),
    "SE_3": (59.5, 16.0), "SE_4": (56.5, 14.0),
    "FI": (63.0, 26.0),
    "DE_LU": (51.0, 10.0), "NL": (52.3, 5.0), "BE": (50.8, 4.5),
    "FR": (46.5, 2.5), "AT": (47.5, 14.0), "PL": (52.0, 20.0),
    "EE": (58.8, 25.5), "LV": (57.0, 24.5), "LT": (55.5, 24.0),
}

# Map zone codes to country ISO codes (for single-zone countries)
ZONE_TO_ISO = {
    "FI": "FIN", "NL": "NLD", "BE": "BEL", "FR": "FRA",
    "AT": "AUT", "PL": "POL", "EE": "EST", "LV": "LVA", "LT": "LTU",
}

# Estimated entry costs (EUR)
ZONE_ENTRY_COSTS = {
    "DK_1": 25_000, "DK_2": 25_000,
    "NO_1": 25_000, "NO_2": 25_000, "NO_3": 25_000, "NO_4": 25_000, "NO_5": 25_000,
    "SE_1": 25_000, "SE_2": 25_000, "SE_3": 25_000, "SE_4": 25_000,
    "FI": 30_000,
    "EE": 35_000, "LV": 35_000, "LT": 35_000,
    "DE_LU": 50_000, "NL": 45_000, "BE": 45_000,
    "FR": 55_000, "AT": 45_000, "PL": 40_000,
}


def load_europe_boundaries():
    """Load country boundaries and split multi-zone countries."""
    url = "https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_0_countries.zip"
    try:
        world = gpd.read_file(url)
    except Exception:
        url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
        world = gpd.read_file(url)

    # Get relevant country names
    name_map = {
        "Denmark": "DNK", "Norway": "NOR", "Sweden": "SWE",
        "Finland": "FIN", "Germany": "DEU", "Netherlands": "NLD",
        "Belgium": "BEL", "France": "FRA", "Austria": "AUT",
        "Poland": "POL", "Estonia": "EST", "Latvia": "LVA",
        "Lithuania": "LTU", "Luxembourg": "LUX",
    }

    europe = world[world["NAME"].isin(name_map.keys())].copy()
    europe["iso"] = europe["NAME"].map(name_map)

    # Build zone geometries
    zone_gdf_rows = []

    for zone in ZONES:
        if zone in ZONE_TO_ISO:
            # Single-zone country
            iso = ZONE_TO_ISO[zone]
            country_name = [k for k, v in name_map.items() if v == iso[:3] + iso[3:] or v == iso]
            row = europe[europe["iso"] == iso]
            if row.empty:
                # Try matching by name
                for cname, ciso in name_map.items():
                    if ciso == iso:
                        row = europe[europe["NAME"] == cname]
                        break
            if not row.empty:
                eu_clip = box(-12, 34, 35, 72)  # clip to Europe (drops Caribbean/overseas)
                geom = row.geometry.values[0].intersection(eu_clip)
                if not geom.is_empty:
                    zone_gdf_rows.append({"zone": zone, "geometry": geom})
            continue

        if zone == "DE_LU":
            # Merge Germany + Luxembourg
            de = europe[europe["NAME"] == "Germany"]
            lu = europe[europe["NAME"] == "Luxembourg"]
            if not de.empty:
                geom = de.geometry.values[0]
                if not lu.empty:
                    geom = geom.union(lu.geometry.values[0])
                zone_gdf_rows.append({"zone": zone, "geometry": geom})
            continue

        # Multi-zone countries: split by latitude
        if zone.startswith("DK_"):
            dk = europe[europe["NAME"] == "Denmark"]
            if dk.empty:
                continue
            geom = dk.geometry.values[0]
            # Split: DK_1 is west (Jutland), DK_2 is east (Zealand)
            split_lon = 10.8
            if zone == "DK_1":
                clip = box(-20, 50, split_lon, 62)
            else:
                clip = box(split_lon, 50, 20, 62)
            clipped = geom.intersection(clip)
            if not clipped.is_empty:
                zone_gdf_rows.append({"zone": zone, "geometry": clipped})

        elif zone.startswith("NO_"):
            no = europe[europe["NAME"] == "Norway"]
            if no.empty:
                continue
            geom = no.geometry.values[0]
            # Approximate latitude splits for Norwegian zones
            lat_splits = {
                "NO_1": (58.0, 62.0, 8.0, 16.0),   # SE Norway
                "NO_2": (56.0, 60.0, 4.0, 9.0),     # S Norway
                "NO_3": (62.0, 66.0, 4.0, 18.0),    # Mid Norway
                "NO_4": (66.0, 75.0, 4.0, 35.0),    # N Norway
                "NO_5": (58.0, 63.0, 3.0, 8.0),     # W Norway
            }
            if zone in lat_splits:
                s_lat, n_lat, w_lon, e_lon = lat_splits[zone]
                clip = box(w_lon, s_lat, e_lon, n_lat)
                clipped = geom.intersection(clip)
                if not clipped.is_empty:
                    zone_gdf_rows.append({"zone": zone, "geometry": clipped})

        elif zone.startswith("SE_"):
            se = europe[europe["NAME"] == "Sweden"]
            if se.empty:
                continue
            geom = se.geometry.values[0]
            # Approximate latitude splits for Swedish zones
            lat_splits = {
                "SE_1": (64.5, 70.0),
                "SE_2": (61.0, 64.5),
                "SE_3": (57.5, 61.0),
                "SE_4": (55.0, 57.5),
            }
            if zone in lat_splits:
                s_lat, n_lat = lat_splits[zone]
                clip = box(10, s_lat, 25, n_lat)
                clipped = geom.intersection(clip)
                if not clipped.is_empty:
                    zone_gdf_rows.append({"zone": zone, "geometry": clipped})

    if not zone_gdf_rows:
        return gpd.GeoDataFrame(), europe

    zone_gdf = gpd.GeoDataFrame(zone_gdf_rows, crs=europe.crs)
    return zone_gdf, europe


def compute_zone_metrics():
    """Compute business-relevant metrics for each zone with data."""
    zones_available = available_zones()
    metrics = []

    for zone in ZONES:
        source = zones_available.get(zone, "none")
        if source == "none":
            continue

        try:
            data = load_all(zone)
        except Exception:
            continue

        prices = data.get("prices")
        if prices is None or prices.empty:
            continue

        p = prices["price_eur_mwh"]

        metrics.append({
            "zone": zone,
            "label": ZONE_LABELS.get(zone, zone),
            "mean_price": p.mean(),
            "std_price": p.std(),
            "cv_price": p.std() / abs(p.mean()) if p.mean() != 0 else 0,
            "neg_pct": (p < 0).mean() * 100,
            "price_range": p.max() - p.min(),
            "entry_cost_eur": ZONE_ENTRY_COSTS.get(zone, 50_000),
            "has_wind_solar": data.get("wind_solar") is not None,
            "has_load": data.get("load") is not None,
        })

    df = pd.DataFrame(metrics)

    if df.empty:
        return df

    # Normalize and compute attractiveness
    for col, ascending in [("cv_price", False), ("neg_pct", False), ("entry_cost_eur", True)]:
        vals = df[col]
        rng = vals.max() - vals.min()
        if rng > 0:
            normalized = (vals - vals.min()) / rng
            if ascending:
                normalized = 1 - normalized
            df[f"{col}_norm"] = normalized
        else:
            df[f"{col}_norm"] = 0.5

    df["data_score"] = (df["has_wind_solar"].astype(float) + df["has_load"].astype(float)) / 2

    df["attractiveness"] = (
        0.35 * df["cv_price_norm"]
        + 0.25 * df["neg_pct_norm"]
        + 0.20 * df["entry_cost_eur_norm"]
        + 0.20 * df["data_score"]
    )

    return df


def draw_interconnectors(ax, metrics_df, alpha=0.7):
    """Draw interconnector lines with width proportional to capacity."""
    max_cap = max(cap for _, _, cap in INTERCONNECTORS)
    zones_with_data = set(metrics_df["zone"].values)

    for from_z, to_z, cap in INTERCONNECTORS:
        if from_z not in ZONE_COORDS or to_z not in ZONE_COORDS:
            continue
        if from_z not in zones_with_data and to_z not in zones_with_data:
            continue

        lat1, lon1 = ZONE_COORDS[from_z]
        lat2, lon2 = ZONE_COORDS[to_z]

        width = 1.0 + 3.5 * (cap / max_cap)
        # White outline for contrast against colored zones
        ax.plot([lon1, lon2], [lat1, lat2],
                color="white", linewidth=width + 1.5, alpha=0.8,
                solid_capstyle="round", zorder=4)
        # Blue line on top
        ax.plot([lon1, lon2], [lat1, lat2],
                color="#1d4ed8", linewidth=width, alpha=0.55,
                solid_capstyle="round", zorder=4)


def draw_zone_labels(ax, metrics_df, fontsize=7):
    """Add zone labels at centroids."""
    for _, row in metrics_df.iterrows():
        zone = row["zone"]
        if zone in ZONE_COORDS:
            lat, lon = ZONE_COORDS[zone]
            ax.annotate(zone.replace("_", ""), (lon, lat),
                       ha="center", va="center", fontsize=fontsize,
                       fontweight="bold", color="black",
                       bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                                alpha=0.7, edgecolor="none"),
                       zorder=5)


def generate_maps(zone_gdf, europe_gdf, metrics_df):
    """Generate the multi-panel and single-panel maps."""

    # Merge metrics into zone geometries
    merged = zone_gdf.merge(metrics_df, on="zone", how="inner")

    # Background: all of Europe in light gray
    xlim = (-5, 32)
    ylim = (44, 72)

    # --- Individual maps ---
    panels = [
        ("cv_price", "Price Volatility (Coefficient of Variation)",
         "YlOrRd", "Higher volatility = more trading opportunity", "eu_volatility_map.png"),
        ("neg_pct", "Negative Price Frequency (%)",
         "YlGnBu", "Wind surplus drives negative prices", "eu_negative_prices_map.png"),
        ("entry_cost_eur", "Estimated Entry Cost (EUR)",
         "RdYlGn_r", "Exchange membership + collateral", "eu_entry_cost_map.png"),
        ("attractiveness", "Combined Attractiveness Score",
         "RdYlGn", "Weighted composite: volatility, neg. prices, cost, data", "eu_zone_heatmap.png"),
    ]

    max_cap = max(cap for _, _, cap in INTERCONNECTORS)
    cap_examples = [500, 2500, 5000]
    legend_lines = [Line2D([0], [0], color="#1d4ed8",
                           linewidth=1.0 + 3.5 * (c / max_cap),
                           label=f"{c} MW")
                    for c in cap_examples]

    for col, title, cmap, subtitle, filename in panels:
        fig, ax = plt.subplots(figsize=(10, 10))

        europe_gdf.plot(ax=ax, color="#e8e8e8", edgecolor="#cccccc", linewidth=0.3)
        draw_interconnectors(ax, metrics_df, alpha=0.45)

        if col in merged.columns:
            merged.plot(ax=ax, column=col, cmap=cmap, edgecolor="black",
                       linewidth=0.8, legend=True, zorder=3,
                       legend_kwds={"shrink": 0.5, "label": col.replace("_", " ").title()})

        draw_zone_labels(ax, metrics_df, fontsize=8)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect(1.8)
        ax.set_title(f"{title}\n{subtitle}", fontsize=13, pad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.legend(handles=legend_lines, loc="lower left", title="Interconnector capacity",
                  fontsize=9, title_fontsize=10, framealpha=0.9)

        path = OUTPUT_DIR / filename
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")

    # --- Single large attractiveness map ---
    fig, ax = plt.subplots(figsize=(10, 10))

    europe_gdf.plot(ax=ax, color="#e8e8e8", edgecolor="#cccccc", linewidth=0.3)
    draw_interconnectors(ax, metrics_df, alpha=0.35)

    if "attractiveness" in merged.columns:
        merged.plot(ax=ax, column="attractiveness", cmap="RdYlGn",
                   edgecolor="black", linewidth=1.0, legend=True, zorder=3,
                   legend_kwds={"shrink": 0.6, "label": "Attractiveness Score",
                               "orientation": "horizontal", "pad": 0.02})

    # Labels with value
    for _, row in metrics_df.iterrows():
        zone = row["zone"]
        if zone in ZONE_COORDS:
            lat, lon = ZONE_COORDS[zone]
            score = row.get("attractiveness", 0)
            label = f"{zone.replace('_', '')}\n{score:.2f}"
            ax.annotate(label, (lon, lat),
                       ha="center", va="center", fontsize=7,
                       fontweight="bold", color="black",
                       bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                alpha=0.8, edgecolor="gray", linewidth=0.5),
                       zorder=5)

    # Interconnector legend
    legend_lines = [Line2D([0], [0], color="#2563eb",
                           linewidth=1.0 + 3.5 * (c / max_cap),
                           label=f"{c} MW")
                    for c in [500, 2500, 5000, 7300]]
    ax.legend(handles=legend_lines, loc="upper left", title="Interconnector Capacity",
              fontsize=10, title_fontsize=11, framealpha=0.9)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect(1.8)
    ax.set_title("Zone Attractiveness for Algorithmic Day-Ahead Trading\n"
                 "Score = 0.35 * volatility + 0.25 * negative prices + 0.20 * (1-cost) + 0.20 * data",
                 fontsize=12, pad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    path1 = OUTPUT_DIR / "eu_attractiveness_map.png"
    fig.savefig(path1, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path1}")


def generate_summary_table(df):
    """Print a ranked summary table."""
    ranked = df.sort_values("attractiveness", ascending=False)

    print("\n" + "=" * 90)
    print("  ZONE RANKING BY ATTRACTIVENESS")
    print("=" * 90)
    print(f"  {'Rank':>4s}  {'Zone':>6s}  {'Label':<22s}  {'Volatility':>10s}  {'Neg%':>6s}  {'Cost':>8s}  {'Score':>6s}")
    print(f"  {'-' * 84}")

    for i, (_, row) in enumerate(ranked.iterrows()):
        print(f"  {i+1:4d}  {row['zone']:>6s}  {row['label']:<22s}  "
              f"{row['cv_price']:10.3f}  {row['neg_pct']:5.1f}%  "
              f"{row['entry_cost_eur']:>7,.0f}  {row['attractiveness']:6.3f}")

    print()


def main():
    print("Loading European boundaries...")
    zone_gdf, europe_gdf = load_europe_boundaries()

    if zone_gdf.empty:
        print("Failed to load zone boundaries.")
        sys.exit(1)

    print(f"Zone polygons: {len(zone_gdf)}")

    print("Computing zone metrics...")
    metrics_df = compute_zone_metrics()

    if metrics_df.empty:
        print("No zone data found. Run fetch_entsoe_data.py first.")
        sys.exit(1)

    print(f"Zones with data: {len(metrics_df)}")
    generate_summary_table(metrics_df)
    generate_maps(zone_gdf, europe_gdf, metrics_df)

    csv_path = OUTPUT_DIR / "zone_metrics.csv"
    metrics_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
