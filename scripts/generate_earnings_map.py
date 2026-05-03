#!/usr/bin/env python3
"""Generate EU earnings map from pipeline backtest results.

Reads the pipeline output log to extract backtest P&L per zone,
applies cost haircuts, and generates a map showing estimated
annual net earnings per zone.

Must be run AFTER: uv run python scripts/run_pipeline.py (with backtest)

Outputs:
    output/eu_earnings_map.png   -- net earnings per zone
    output/earnings_summary.csv  -- tabulated earnings estimates

Run: uv run python scripts/generate_earnings_map.py
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

from da_forecast.config import ZONES, ZONE_LABELS, INTERCONNECTORS

OUTPUT_DIR = Path(__file__).parent.parent / "output"

# Reuse zone coords and boundary loading from heatmap script
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

ZONE_TO_ISO = {
    "FI": "FIN", "NL": "NLD", "BE": "BEL", "FR": "FRA",
    "AT": "AUT", "PL": "POL", "EE": "EST", "LV": "LVA", "LT": "LTU",
}

# Annual costs per zone (EUR) -- exchange fees, membership, data feeds
# Sources: Nord Pool fee schedule 2025, EPEX SPOT membership tiers
ZONE_ANNUAL_COSTS = {
    "DK_1": 15_000, "DK_2": 15_000,
    "NO_1": 15_000, "NO_2": 15_000, "NO_3": 15_000, "NO_4": 15_000, "NO_5": 15_000,
    "SE_1": 15_000, "SE_2": 15_000, "SE_3": 15_000, "SE_4": 15_000,
    "FI": 18_000,
    "EE": 20_000, "LV": 20_000, "LT": 20_000,
    "DE_LU": 35_000, "NL": 30_000, "BE": 30_000,
    "FR": 40_000, "AT": 30_000, "PL": 25_000,
}


def parse_backtest_results(output_dir: Path) -> pd.DataFrame:
    """Load backtest results from JSON (fast_backtest.py output) or pipeline log."""
    json_path = output_dir / "backtest_summary.json"
    log_path = output_dir / "pipeline_report.log"

    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        rows = []
        for r in data:
            if r.get("status") != "ok":
                continue
            rows.append({
                "zone": r["zone"],
                "backtest_pnl": r["total_pnl"],
                "sharpe": r["sharpe_ratio"],
                "win_pct": r["win_rate_pct"],
                "trades": r["n_trades"],
                "n_test_days": r["n_test_days"],
            })
        if rows:
            print(f"Loaded backtest results from {json_path}")
            return pd.DataFrame(rows)

    if log_path.exists():
        text = log_path.read_text()
        pattern = r"Backtest (\w+): P&L=([\d.-]+) EUR, Sharpe=([\d.-]+), Win=(\d+)%, Trades=(\d+)"
        matches = re.findall(pattern, text)
        if matches:
            rows = [{"zone": z, "backtest_pnl": float(p), "sharpe": float(s),
                     "win_pct": float(w), "trades": int(t)} for z, p, s, w, t in matches]
            return pd.DataFrame(rows)

    print("No backtest results found. Run fast_backtest.py or run_pipeline.py first.")
    sys.exit(1)


def compute_earnings(bt_df: pd.DataFrame, position_mwh: float = 10.0) -> pd.DataFrame:
    """Compute estimated annual earnings per zone.

    Assumptions (listed explicitly):
      A1. Position size: {position_mwh} MWh per hour
      A2. Backtest period annualized to 365 days
      A3. Model performance haircut: 50% (base case for live vs backtest)
      A4. Annual zone costs: exchange fees + membership + data
      A5. No market impact (valid for positions < 50 MW)

    Sources:
      - Live vs backtest ratio: De Prado, Advances in Financial ML (2018), Ch. 11
      - Nord Pool fees: https://www.nordpoolgroup.com/trading/fees/
      - Market impact threshold: ACER 2024 Wholesale Market Report
    """
    df = bt_df.copy()

    # The backtest runs at 1 MWh. Scale to position size and annualize.
    n_days = df["n_test_days"] if "n_test_days" in df.columns else 30
    df["gross_annual"] = df["backtest_pnl"] * position_mwh * (365 / n_days)

    # Haircuts
    df["after_haircut_50pct"] = df["gross_annual"] * 0.50
    df["after_haircut_25pct"] = df["gross_annual"] * 0.25

    # Subtract annual costs
    df["annual_cost"] = df["zone"].map(ZONE_ANNUAL_COSTS).fillna(30_000)
    df["net_base_case"] = df["after_haircut_50pct"] - df["annual_cost"]
    df["net_conservative"] = df["after_haircut_25pct"] - df["annual_cost"]

    # Add labels
    df["label"] = df["zone"].map(ZONE_LABELS)

    return df


def load_europe_boundaries():
    """Load country boundaries and split multi-zone countries."""
    try:
        world = gpd.read_file("https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_0_countries.zip")
    except Exception:
        world = gpd.read_file("https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip")

    name_map = {
        "Denmark": "DNK", "Norway": "NOR", "Sweden": "SWE",
        "Finland": "FIN", "Germany": "DEU", "Netherlands": "NLD",
        "Belgium": "BEL", "France": "FRA", "Austria": "AUT",
        "Poland": "POL", "Estonia": "EST", "Latvia": "LVA",
        "Lithuania": "LTU", "Luxembourg": "LUX",
    }

    europe = world[world["NAME"].isin(name_map.keys())].copy()
    europe["iso"] = europe["NAME"].map(name_map)

    zone_gdf_rows = []

    for zone in ZONES:
        if zone in ZONE_TO_ISO:
            iso = ZONE_TO_ISO[zone]
            eu_clip = box(-12, 34, 35, 72)  # clip to Europe (drops Caribbean/overseas)
            for cname, ciso in name_map.items():
                if ciso == iso:
                    row = europe[europe["NAME"] == cname]
                    if not row.empty:
                        geom = row.geometry.values[0].intersection(eu_clip)
                        if not geom.is_empty:
                            zone_gdf_rows.append({"zone": zone, "geometry": geom})
                    break
            continue

        if zone == "DE_LU":
            de = europe[europe["NAME"] == "Germany"]
            lu = europe[europe["NAME"] == "Luxembourg"]
            if not de.empty:
                geom = de.geometry.values[0]
                if not lu.empty:
                    geom = geom.union(lu.geometry.values[0])
                zone_gdf_rows.append({"zone": zone, "geometry": geom})
            continue

        if zone.startswith("DK_"):
            dk = europe[europe["NAME"] == "Denmark"]
            if dk.empty:
                continue
            geom = dk.geometry.values[0]
            split_lon = 10.8
            clip = box(-20, 50, split_lon, 62) if zone == "DK_1" else box(split_lon, 50, 20, 62)
            clipped = geom.intersection(clip)
            if not clipped.is_empty:
                zone_gdf_rows.append({"zone": zone, "geometry": clipped})

        elif zone.startswith("NO_"):
            no = europe[europe["NAME"] == "Norway"]
            if no.empty:
                continue
            geom = no.geometry.values[0]
            lat_splits = {
                "NO_1": (58.0, 62.0, 8.0, 16.0), "NO_2": (56.0, 60.0, 4.0, 9.0),
                "NO_3": (62.0, 66.0, 4.0, 18.0), "NO_4": (66.0, 75.0, 4.0, 35.0),
                "NO_5": (58.0, 63.0, 3.0, 8.0),
            }
            if zone in lat_splits:
                s_lat, n_lat, w_lon, e_lon = lat_splits[zone]
                clipped = geom.intersection(box(w_lon, s_lat, e_lon, n_lat))
                if not clipped.is_empty:
                    zone_gdf_rows.append({"zone": zone, "geometry": clipped})

        elif zone.startswith("SE_"):
            se = europe[europe["NAME"] == "Sweden"]
            if se.empty:
                continue
            geom = se.geometry.values[0]
            lat_splits = {"SE_1": (64.5, 70.0), "SE_2": (61.0, 64.5),
                          "SE_3": (57.5, 61.0), "SE_4": (55.0, 57.5)}
            if zone in lat_splits:
                s_lat, n_lat = lat_splits[zone]
                clipped = geom.intersection(box(10, s_lat, 25, n_lat))
                if not clipped.is_empty:
                    zone_gdf_rows.append({"zone": zone, "geometry": clipped})

    zone_gdf = gpd.GeoDataFrame(zone_gdf_rows, crs=europe.crs) if zone_gdf_rows else gpd.GeoDataFrame()
    return zone_gdf, europe


def generate_earnings_map(zone_gdf, europe_gdf, earnings_df):
    """Generate map colored by estimated net annual earnings."""
    merged = zone_gdf.merge(earnings_df, on="zone", how="inner")

    xlim = (-5, 32)
    ylim = (44, 72)
    max_cap = max(cap for _, _, cap in INTERCONNECTORS)

    fig, ax = plt.subplots(figsize=(10, 10))

    # Clip Europe to view bounds (drops overseas territories that cause rendering issues)
    eu_clip = box(-12, 34, 35, 72)
    europe_clipped = europe_gdf.copy()
    europe_clipped["geometry"] = europe_clipped.geometry.intersection(eu_clip)
    europe_clipped = europe_clipped[~europe_clipped.is_empty]
    europe_clipped.plot(ax=ax, color="#e8e8e8", edgecolor="#cccccc", linewidth=0.3, zorder=1)

    # Interconnectors
    for from_z, to_z, cap in INTERCONNECTORS:
        if from_z not in ZONE_COORDS or to_z not in ZONE_COORDS:
            continue
        lat1, lon1 = ZONE_COORDS[from_z]
        lat2, lon2 = ZONE_COORDS[to_z]
        width = 1.0 + 3.5 * (cap / max_cap)
        ax.plot([lon1, lon2], [lat1, lat2],
                color="white", linewidth=width + 1.5, alpha=0.8,
                solid_capstyle="round", zorder=4)
        ax.plot([lon1, lon2], [lat1, lat2],
                color="#1d4ed8", linewidth=width, alpha=0.45,
                solid_capstyle="round", zorder=4)

    # Colored zones by net earnings
    merged.plot(ax=ax, column="net_base_case", cmap="RdYlGn",
                edgecolor="black", linewidth=0.8, legend=True, zorder=3,
                legend_kwds={"shrink": 0.5, "label": "Annualized Backtest P&L (EUR, simulation only)"})

    # Labels with EUR value
    for _, row in earnings_df.iterrows():
        zone = row["zone"]
        if zone in ZONE_COORDS:
            lat, lon = ZONE_COORDS[zone]
            net = row["net_base_case"]
            if net >= 1_000_000:
                label = f"{zone.replace('_', '')}\n{net/1e6:.1f}M"
            elif net >= 1000:
                label = f"{zone.replace('_', '')}\n{net/1e3:.0f}K"
            else:
                label = f"{zone.replace('_', '')}\n{net:.0f}"
            color = "#006400" if net > 0 else "#8b0000"
            ax.annotate(label, (lon, lat),
                       ha="center", va="center", fontsize=7,
                       fontweight="bold", color=color,
                       bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                alpha=0.85, edgecolor="gray", linewidth=0.5),
                       zorder=5)

    # Legend
    legend_lines = [Line2D([0], [0], color="#1d4ed8",
                           linewidth=1.0 + 3.5 * (c / max_cap),
                           label=f"{c} MW")
                    for c in [500, 2500, 5000]]
    ax.legend(handles=legend_lines, loc="lower left", title="Interconnector capacity",
              fontsize=9, title_fontsize=10, framealpha=0.9)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect(1.8)
    ax.set_title("Backtest Signal Strength by Zone (Simulation Only)\n"
                 "Annualized backtest P&L at 50% haircut. Not a forecast of real returns.",
                 fontsize=12, pad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    path = OUTPUT_DIR / "eu_earnings_map.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    bt_df = parse_backtest_results(OUTPUT_DIR)
    print(f"Found backtest results for {len(bt_df)} zones")

    earnings_df = compute_earnings(bt_df, position_mwh=10.0)

    # Print summary
    ranked = earnings_df.sort_values("net_base_case", ascending=False)
    print("\n" + "=" * 95)
    print("  BACKTEST SIGNAL STRENGTH BY ZONE (10 MWh/h, simulation only)")
    print("=" * 95)
    print("  Assumptions:")
    print("    Position: 10 MWh/h | Haircut: 50% (base) / 75% (conservative)")
    print("    Sources: De Prado (2018) Ch.11, Nord Pool fees, ACER 2024")
    print()
    print(f"  {'Rank':>4s}  {'Zone':>6s}  {'Label':<22s}  {'Gross':>12s}  {'Base case':>12s}  {'Conservative':>12s}")
    print(f"  {'-' * 80}")

    for i, (_, row) in enumerate(ranked.iterrows()):
        print(f"  {i+1:4d}  {row['zone']:>6s}  {row['label']:<22s}  "
              f"{row['gross_annual']:>10,.0f}  {row['net_base_case']:>10,.0f}  {row['net_conservative']:>10,.0f}")

    total_base = ranked["net_base_case"].sum()
    total_conservative = ranked["net_conservative"].sum()
    print(f"\n  {'':4s}  {'':>6s}  {'TOTAL':<22s}  {'':>10s}  {total_base:>10,.0f}  {total_conservative:>10,.0f}")
    print(f"\n  All values in EUR. Multiply by ~7.46 for DKK.")
    print()

    # Save CSV
    csv_path = OUTPUT_DIR / "earnings_summary.csv"
    ranked.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Generate map
    print("Loading boundaries...")
    zone_gdf, europe_gdf = load_europe_boundaries()
    generate_earnings_map(zone_gdf, europe_gdf, earnings_df)


if __name__ == "__main__":
    main()
