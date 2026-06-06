# =============================================================
# DWAE Projekt – Phase 6: Visualisierung & Analyse
# Thema: Einfluss von Wetter auf Luftqualität in Zürich
# =============================================================
# Voraussetzung: Phase 5 abgeschlossen
# Ausführen:     python W06_visualisierung.py
# =============================================================

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from pathlib import Path

# Windows-Terminal: UTF-8 erzwingen damit ✓ und Umlaute korrekt ausgegeben werden
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

Path("output/plots").mkdir(parents=True, exist_ok=True)

import plot_config                                     # setzt rcParams global
from plot_config import (FARBEN, JAHRESZEIT_FARBEN,
                         who_linie, suptitel_stats,
                         FIG_KLEIN, FIG_GROSS,
                         finde_schadstoff_cols)

ERKLAERUNGEN = {
    "regen": (
        "Niederschlag wäscht Partikel durch nasse Deposition aus der Luft. "
        "Der Effekt ist bei PM10 stärker als bei NO2, da Gase weniger "
        "effizient ausgewaschen werden."
    ),
    "inversion": (
        "Inversionslagen verhindern die vertikale Durchmischung der Luft. "
        "Schadstoffe akkumulieren in der Bodenschicht, da der normale "
        "Temperaturgradient umgekehrt ist."
    ),
    "temperatur": (
        "Hohe Temperaturen begünstigen photochemische Reaktionen, die O3 aus "
        "NO2 und VOCs bilden. NO2 sinkt dabei, weil es als Vorläufersubstanz "
        "verbraucht wird."
    ),
    "rush_hour": (
        "Verkehrsbedingte Emissionen dominieren das Tagesprofil. NO2-Peaks um "
        "8h und 18h spiegeln den Pendlerverkehr wider."
    ),
}

print("=" * 60)
print("PHASE 6 – Visualisierung & Analyse")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# DATEN LADEN
# ─────────────────────────────────────────────────────────────
df = pd.read_csv("data/processed/datensatz_final.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(None)

schadstoff_cols = finde_schadstoff_cols(df)
if not schadstoff_cols:
    print("FEHLER: Keine Schadstoffspalten gefunden.")
    print("Vorhandene Spalten:", list(df.columns))
    print("Bitte Spaltennamen in plot_config.py anpassen.")
    exit(1)

wetter_cols = [c for c in [
    "temperature_2m", "precipitation", "wind_speed_10m",
    "relative_humidity_2m", "surface_pressure", "sonnenschein_h"]
    if c in df.columns]

haupt_schadstoff = schadstoff_cols[0] if schadstoff_cols else None
print(f"\n  Hauptschadstoff für Analyse: {haupt_schadstoff}")
print(f"  Wettervariablen: {wetter_cols}")
print(f"  Datensatz: {len(df):,} Stunden\n")


# ─────────────────────────────────────────────────────────────
# PLOT 1: ÜBERSICHTSDASHBOARD (Jahresverlauf)
# ─────────────────────────────────────────────────────────────
print("[1/7] Jahresübersicht Dashboard...")

if haupt_schadstoff and "temperature_2m" in df.columns:
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # Tägliche Aggregate
    daily = df.set_index("timestamp").resample("D").agg({
        haupt_schadstoff:   "median",
        "temperature_2m":   "mean",
        "precipitation":    "sum",
        "wind_speed_10m":   "mean",
        "relative_humidity_2m": "mean",
    })

    plots = [
        (gs[0, :], haupt_schadstoff,       FARBEN["gruen"],
         f"{haupt_schadstoff} (Tagesmedian)", "µg/m³"),
        (gs[1, 0], "temperature_2m",       FARBEN["rot"],
         "Temperatur (Tagesmittel)", "°C"),
        (gs[1, 1], "precipitation",        FARBEN["blau"],
         "Niederschlag (Tagessumme)", "mm"),
        (gs[2, 0], "wind_speed_10m",       FARBEN["lila"],
         "Windgeschwindigkeit (Mittel)", "km/h"),
        (gs[2, 1], "relative_humidity_2m", FARBEN["gelb"],
         "Relative Luftfeuchtigkeit", "%"),
    ]

    for spec, col, farbe, titel, unit in plots:
        ax = fig.add_subplot(spec)
        if col not in daily.columns:
            continue
        ax.plot(daily.index, daily[col], lw=0.8, color=farbe, alpha=0.4)
        ax.fill_between(daily.index, daily[col], alpha=0.08, color=farbe)
        rolling_7 = daily[col].rolling(7, center=True, min_periods=4).mean()
        ax.plot(daily.index, rolling_7, lw=2.0, color=farbe,
                label="7-Tage-Mittel")
        who_linie(ax, col)
        ax.legend(loc="upper right")
        ax.set_title(titel)
        ax.set_ylabel(unit)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        "Einfluss von Wetter auf Luftqualität in Zürich – Jahresübersicht 2023\n"
        f"{suptitel_stats(df, haupt_schadstoff)}",
        fontsize=12, y=1.02
    )
    plt.savefig("output/plots/viz_01_jahresuebersicht.png",
                bbox_inches="tight")
    plt.close()
    print("  ✓ output/plots/viz_01_jahresuebersicht.png")


# ─────────────────────────────────────────────────────────────
# PLOT 2: KORRELATIONS-HEATMAP (Pearson + Spearman nebeneinander)
# ─────────────────────────────────────────────────────────────
print("[2/7] Korrelations-Heatmap (Pearson & Spearman)...")

alle_cols = schadstoff_cols + wetter_cols
alle_cols = [c for c in alle_cols if c in df.columns]

if len(alle_cols) >= 2:
    corr_p = df[alle_cols].corr(method="pearson")
    corr_s = df[alle_cols].corr(method="spearman")

    def _heatmap(ax, corr, titel):
        im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax, shrink=0.75, label="r")
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.index)))
        ax.set_xticklabels(corr.columns, rotation=40, ha="right", fontsize=8)
        ax.set_yticklabels(corr.index, fontsize=8)
        for i in range(len(corr)):
            for j in range(len(corr.columns)):
                val = corr.iloc[i, j]
                col_txt = "white" if abs(val) > 0.55 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7.5, color=col_txt, fontweight="bold")
        ax.set_title(titel, pad=12)

    fig, axes = plt.subplots(1, 2, figsize=(22, 8))
    _heatmap(axes[0], corr_p,
             "Pearson-Korrelation\n(linear)")
    _heatmap(axes[1], corr_s,
             "Spearman-Korrelation\n(monoton, robust gg. Ausreisser)")
    fig.suptitle(
        "Korrelationsmatrizen: Schadstoffe & Wettervariablen (2023)\n"
        f"{suptitel_stats(df, haupt_schadstoff)}",
        fontsize=12
    )
    plt.tight_layout()
    plt.savefig("output/plots/viz_02_korrelation.png", bbox_inches="tight")
    plt.close()
    print("  ✓ output/plots/viz_02_korrelation.png")


# ─────────────────────────────────────────────────────────────
# PLOT 3: SCHADSTOFF NACH JAHRESZEIT (Boxplot)
# ─────────────────────────────────────────────────────────────
print("[3/7] Saisonale Boxplots...")

if haupt_schadstoff and "jahreszeit" in df.columns:
    reihenfolge = ["Winter", "Frühling", "Sommer", "Herbst"]
    reihenfolge = [j for j in reihenfolge if j in df["jahreszeit"].unique()]

    fig, axes = plt.subplots(1, len(schadstoff_cols[:3]),
                             figsize=(5 * len(schadstoff_cols[:3]), 6))
    if len(schadstoff_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, schadstoff_cols[:3]):
        data_jz = [df[df["jahreszeit"] == jz][col].dropna().values
                   for jz in reihenfolge]
        farben_jz = [JAHRESZEIT_FARBEN.get(jz, FARBEN["grau"])
                     for jz in reihenfolge]

        bp = ax.boxplot(data_jz, patch_artist=True,
                        medianprops=dict(color="white", linewidth=2),
                        flierprops=dict(marker=".", markersize=2,
                                        alpha=0.3))
        for patch, farbe in zip(bp["boxes"], farben_jz):
            patch.set_facecolor(farbe)
            patch.set_alpha(0.8)

        ax.set_xticklabels(reihenfolge)
        ax.set_title(col)
        ax.set_ylabel("µg/m³")
        ax.grid(True, alpha=0.25, axis="y")

        # Mittelwerte als Punkte
        for i, vals in enumerate(data_jz):
            if len(vals) > 0:
                ax.plot(i + 1, np.mean(vals), marker="D",
                        color="white", markersize=5, zorder=5)
        who_linie(ax, col)
        ax.legend(fontsize=8)

    fig.suptitle(
        f"Luftschadstoffe nach Jahreszeit – Zürich 2023\n"
        f"{suptitel_stats(df, haupt_schadstoff)}",
        fontsize=12
    )
    plt.tight_layout()
    plt.savefig("output/plots/viz_03_jahreszeit_boxplot.png",
                bbox_inches="tight")
    plt.close()
    print("  ✓ output/plots/viz_03_jahreszeit_boxplot.png")


# ─────────────────────────────────────────────────────────────
# PLOT 4: TEMPERATUR vs. SCHADSTOFFE (Scatter + Trend)
# ─────────────────────────────────────────────────────────────
print("[4/7] Scatter: Temperatur vs. Schadstoffe...")

if haupt_schadstoff and "temperature_2m" in df.columns \
   and "jahreszeit" in df.columns:
    n_cols = min(len(schadstoff_cols), 3)
    fig, axes = plt.subplots(1, n_cols, figsize=(5.5 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

    for ax, col in zip(axes, schadstoff_cols[:n_cols]):
        sample = df[["temperature_2m", col, "jahreszeit"]].dropna()

        for jz, gruppe in sample.groupby("jahreszeit"):
            ax.scatter(gruppe["temperature_2m"], gruppe[col],
                       alpha=0.12, s=3,
                       color=JAHRESZEIT_FARBEN.get(jz, FARBEN["grau"]),
                       label=jz)

        # Trendlinie (Polynomfit 2. Grades)
        x = sample["temperature_2m"].values
        y = sample[col].values
        if len(x) > 10:
            z  = np.polyfit(x, y, 2)
            p  = np.poly1d(z)
            xs = np.linspace(x.min(), x.max(), 200)
            ax.plot(xs, p(xs), color="black", lw=1.8,
                    linestyle="--", label="Trend (Poly2)", zorder=5)

        r = sample[["temperature_2m", col]].corr().iloc[0, 1]
        ax.set_xlabel("Temperatur (°C)")
        ax.set_ylabel(f"{col} (µg/m³)")
        ax.set_title(f"Temperatur vs. {col}\n(r = {r:.3f})")
        who_linie(ax, col)
        ax.legend(markerscale=5)
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        f"Einfluss der Temperatur auf Luftschadstoffe – Zürich 2023\n"
        f"{suptitel_stats(df, haupt_schadstoff)}",
        fontsize=12
    )
    plt.tight_layout()
    plt.savefig("output/plots/viz_04_scatter_temperatur.png",
                bbox_inches="tight")
    plt.close()
    print("  ✓ output/plots/viz_04_scatter_temperatur.png")
    print(f"\n  Physikalischer Hintergrund:\n  {ERKLAERUNGEN['temperatur']}")


# ─────────────────────────────────────────────────────────────
# PLOT 5: REGEN-EFFEKT (Vor / Während / Nach Niederschlag)
# ─────────────────────────────────────────────────────────────
print("[5/7] Regen-Effekt auf Schadstoffe...")

if haupt_schadstoff and "regen" in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=FIG_KLEIN)

    # Box: kein Regen vs. Regen
    kein_regen = df[df["regen"] == 0][haupt_schadstoff].dropna()
    mit_regen  = df[df["regen"] == 1][haupt_schadstoff].dropna()

    bp = axes[0].boxplot(
        [kein_regen, mit_regen],
        patch_artist=True,
        medianprops=dict(color="white", linewidth=2),
        flierprops=dict(marker=".", markersize=2, alpha=0.3)
    )
    bp["boxes"][0].set_facecolor(FARBEN["gelb"])
    bp["boxes"][1].set_facecolor(FARBEN["blau"])
    bp["boxes"][0].set_alpha(0.8)
    bp["boxes"][1].set_alpha(0.8)
    axes[0].set_xticklabels(["Kein Regen", "Regen"])
    axes[0].set_title(f"{haupt_schadstoff}: Regen vs. kein Regen")
    axes[0].set_ylabel("µg/m³")
    axes[0].grid(True, alpha=0.25, axis="y")
    who_linie(axes[0], haupt_schadstoff)

    # Mittelwerte
    for i, (vals, label) in enumerate([(kein_regen, "Mittel"),
                                        (mit_regen, "Mittel")]):
        axes[0].plot(i + 1, vals.mean(), "wD", markersize=6, zorder=5)

    diff = kein_regen.mean() - mit_regen.mean()
    axes[0].text(0.5, 0.92,
                 f"Differenz: {diff:.1f} µg/m³ ({diff/kein_regen.mean()*100:.1f}%)",
                 transform=axes[0].transAxes, ha="center", fontsize=9,
                 color=FARBEN["grau"])

    # Scatter: Niederschlag vs. Schadstoff
    sample = df[["precipitation", haupt_schadstoff]].dropna()
    sample = sample[sample["precipitation"] > 0]  # nur Regenstunden
    axes[1].scatter(sample["precipitation"], sample[haupt_schadstoff],
                    alpha=0.25, s=8, color=FARBEN["blau"])
    r = sample[["precipitation", haupt_schadstoff]].corr().iloc[0, 1]
    axes[1].set_xlabel("Niederschlag (mm/h)")
    axes[1].set_ylabel(f"{haupt_schadstoff} (µg/m³)")
    axes[1].set_title(f"Niederschlagsmenge vs. {haupt_schadstoff}\n(r = {r:.3f})")
    axes[1].grid(True, alpha=0.25)

    fig.suptitle(
        f"Auswascheffekt von Niederschlag auf Luftschadstoffe\n"
        f"{suptitel_stats(df, haupt_schadstoff)}",
        fontsize=12
    )
    plt.tight_layout()
    plt.savefig("output/plots/viz_05_regen_effekt.png", bbox_inches="tight")
    plt.close()
    print("  ✓ output/plots/viz_05_regen_effekt.png")
    print(f"\n  Physikalischer Hintergrund:\n  {ERKLAERUNGEN['regen']}")


# ─────────────────────────────────────────────────────────────
# PLOT 6: INVERSIONSLAGEN-ANALYSE
# ─────────────────────────────────────────────────────────────
print("[6/7] Inversionslagen-Analyse...")

if haupt_schadstoff and "inversion_indikator" in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=FIG_KLEIN)

    normal    = df[df["inversion_indikator"] == 0][haupt_schadstoff].dropna()
    inversion = df[df["inversion_indikator"] == 1][haupt_schadstoff].dropna()

    # Histogramm
    axes[0].hist(normal,    bins=50, alpha=0.65, color=FARBEN["gruen"],
                 label=f"Normal (n={len(normal):,})",    density=True)
    axes[0].hist(inversion, bins=50, alpha=0.65, color=FARBEN["rot"],
                 label=f"Inversion (n={len(inversion):,})", density=True)
    axes[0].axvline(normal.mean(),    color=FARBEN["gruen"],
                    lw=2, linestyle="--",
                    label=f"Mittel normal: {normal.mean():.1f}")
    axes[0].axvline(inversion.mean(), color=FARBEN["rot"],
                    lw=2, linestyle="--",
                    label=f"Mittel Inversion: {inversion.mean():.1f}")
    axes[0].set_xlabel(f"{haupt_schadstoff} (µg/m³)")
    axes[0].set_ylabel("Dichte")
    axes[0].set_title(f"Verteilung {haupt_schadstoff}:\nNormal vs. Inversionswetterlage")
    who_linie(axes[0], haupt_schadstoff, vertikal=True)
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    # Monatliche Inversions-Häufigkeit
    df["monat_label"] = df["timestamp"].dt.strftime("%b")
    monate_order = ["Jan","Feb","Mär","Apr","Mai","Jun",
                    "Jul","Aug","Sep","Okt","Nov","Dez"]
    inv_monthly = df.groupby("monat")["inversion_indikator"].mean() * 100
    axes[1].bar(range(1, 13), inv_monthly.values,
                color=FARBEN["blau"], alpha=0.75, edgecolor="white")
    axes[1].set_xticks(range(1, 13))
    axes[1].set_xticklabels(monate_order[:len(inv_monthly)], fontsize=8)
    axes[1].set_ylabel("Anteil Inversionsstunden (%)")
    axes[1].set_title("Monatliche Häufigkeit von Inversionslagen")
    axes[1].grid(True, alpha=0.25, axis="y")

    diff_inv = inversion.mean() - normal.mean()
    fig.suptitle(
        f"Inversionslagen erhöhen {haupt_schadstoff} "
        f"um Ø {diff_inv:.1f} µg/m³ ({diff_inv/normal.mean()*100:.0f}%)\n"
        f"{suptitel_stats(df, haupt_schadstoff)}",
        fontsize=12
    )
    plt.tight_layout()
    plt.savefig("output/plots/viz_06_inversion.png", bbox_inches="tight")
    plt.close()
    print("  ✓ output/plots/viz_06_inversion.png")
    print(f"\n  Physikalischer Hintergrund:\n  {ERKLAERUNGEN['inversion']}")


# ─────────────────────────────────────────────────────────────
# PLOT 7: TAGES- UND WOCHENPROFIL
# ─────────────────────────────────────────────────────────────
print("[7/7] Tages- und Wochenprofil...")

if haupt_schadstoff and "stunde" in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=FIG_KLEIN)

    # Tagesprofil nach Jahreszeit
    for jz, gruppe in df.groupby("jahreszeit"):
        profil = gruppe.groupby("stunde")[haupt_schadstoff].median()
        axes[0].plot(profil.index, profil.values, lw=1.8,
                     color=JAHRESZEIT_FARBEN.get(jz, FARBEN["grau"]),
                     label=jz, marker="o", markersize=3)

    axes[0].set_xlabel("Stunde des Tages")
    axes[0].set_ylabel(f"{haupt_schadstoff} (µg/m³)")
    axes[0].set_title(f"Tagesprofil {haupt_schadstoff} nach Jahreszeit")
    axes[0].set_xticks(range(0, 24, 2))
    who_linie(axes[0], haupt_schadstoff)
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    # Wochenprofil: Werktag vs. Wochenende
    if "ist_wochenende" in df.columns:
        tage_labels = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]
        profil_tag  = df.groupby("wochentag")[haupt_schadstoff].median()
        farben_tage = [FARBEN["gruen"] if i < 5 else FARBEN["rot"]
                       for i in range(7)]
        bars = axes[1].bar(range(len(profil_tag)),
                           profil_tag.values,
                           color=farben_tage, alpha=0.8, edgecolor="white")
        axes[1].set_xticks(range(len(profil_tag)))
        axes[1].set_xticklabels(tage_labels[:len(profil_tag)], fontsize=9)
        axes[1].set_ylabel(f"{haupt_schadstoff} (µg/m³)")
        axes[1].set_title(f"Wochenprofil {haupt_schadstoff}")
        axes[1].grid(True, alpha=0.25, axis="y")
        who_linie(axes[1], haupt_schadstoff)

        # Legende
        from matplotlib.patches import Patch
        axes[1].legend(handles=[
            Patch(color=FARBEN["gruen"], alpha=0.8, label="Werktag"),
            Patch(color=FARBEN["rot"],   alpha=0.8, label="Wochenende"),
        ], fontsize=8)

    fig.suptitle(
        f"Zeitliche Muster – {haupt_schadstoff} Zürich 2023\n"
        f"{suptitel_stats(df, haupt_schadstoff)}",
        fontsize=12
    )
    plt.tight_layout()
    plt.savefig("output/plots/viz_07_zeitprofile.png", bbox_inches="tight")
    plt.close()
    print("  ✓ output/plots/viz_07_zeitprofile.png")
    print(f"\n  Physikalischer Hintergrund:\n  {ERKLAERUNGEN['rush_hour']}")


# ─────────────────────────────────────────────────────────────
# PLOT 8: WINDROSE – Windrichtung vs. Schadstoffkonzentration
# ─────────────────────────────────────────────────────────────
print("[8/9] Windrose: Windrichtung vs. Schadstoff (22.5°-Sektoren)...")

if haupt_schadstoff and "wind_direction_10m" in df.columns:
    n_sek      = 16
    breite_deg = 360 / n_sek
    bins       = np.linspace(0, 360, n_sek + 1)
    mitte      = bins[:-1] + breite_deg / 2

    df_wind = df[["wind_direction_10m", haupt_schadstoff]].dropna().copy()
    df_wind["wind_direction_10m"] = df_wind["wind_direction_10m"] % 360
    df_wind["sektor"] = pd.cut(
        df_wind["wind_direction_10m"], bins=bins,
        labels=False, include_lowest=True
    )
    mittelwerte = (df_wind.groupby("sektor")[haupt_schadstoff]
                   .mean()
                   .reindex(range(n_sek), fill_value=0))

    fig, ax = plt.subplots(figsize=(8, 8),
                           subplot_kw={"projection": "polar"})
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location("N")

    theta    = np.deg2rad(mitte)
    breite_r = np.deg2rad(breite_deg) * 0.9
    vmin_w, vmax_w = mittelwerte.min(), max(mittelwerte.max(), 1e-9)
    norm_w = plt.Normalize(vmin_w, vmax_w)
    ax.bar(theta, mittelwerte.values, width=breite_r,
           bottom=0, align="center",
           color=plt.cm.YlOrRd(norm_w(mittelwerte.values)),
           alpha=0.85, edgecolor="white", linewidth=0.5)

    richtungen = ["N","NNO","NO","ONO","O","OSO","SO","SSO",
                  "S","SSW","SW","WSW","W","WNW","NW","NNW"]
    ax.set_xticks(np.deg2rad(mitte))
    ax.set_xticklabels(richtungen, fontsize=9)

    sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=norm_w)
    plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.1,
                 label=f"Mittleres {haupt_schadstoff} (µg/m³)")
    ax.set_title(
        f"Windrose: {haupt_schadstoff} nach Windrichtung\n"
        f"(22.5°-Sektoren, Zürich 2023)",
        pad=20, fontsize=11
    )
    plt.savefig("output/plots/viz_08_windrose.png",
                bbox_inches="tight", dpi=120)
    plt.close()
    print("  ✓ output/plots/viz_08_windrose.png")
else:
    print("  ⚠ wind_direction_10m fehlt – Windrose übersprungen")


# ─────────────────────────────────────────────────────────────
# PLOT 9: LAG-ANALYSE – Verzögerter Regeneffekt (0–6h)
# ─────────────────────────────────────────────────────────────
print("[9/9] Lag-Analyse: Regen → Schadstoff mit Zeitversatz 0–6h...")

if haupt_schadstoff and "precipitation" in df.columns:
    df_lag  = df[["precipitation", haupt_schadstoff]].dropna().copy()
    max_lag = 6
    pearson_lags  = []
    spearman_lags = []

    for lag in range(max_lag + 1):
        regen  = df_lag["precipitation"].shift(lag)
        ziel   = df_lag[haupt_schadstoff]
        valide = regen.notna() & ziel.notna()
        pearson_lags.append(
            regen[valide].corr(ziel[valide], method="pearson"))
        spearman_lags.append(
            regen[valide].corr(ziel[valide], method="spearman"))

    fig, ax = plt.subplots(figsize=FIG_KLEIN)
    x, w = np.arange(max_lag + 1), 0.38
    ax.bar(x - w/2, pearson_lags,  w, label="Pearson r",
           color=FARBEN["blau"],  alpha=0.85, edgecolor="white")
    ax.bar(x + w/2, spearman_lags, w, label="Spearman ρ",
           color=FARBEN["gruen"], alpha=0.85, edgecolor="white")
    ax.axhline(0, color="black", lw=0.8, linestyle="--")

    idx_min = int(np.argmin(pearson_lags))
    ax.annotate(
        f"Stärkstes Lag: {idx_min}h\n(r = {pearson_lags[idx_min]:.3f})",
        xy=(idx_min - w/2, pearson_lags[idx_min]),
        xytext=(idx_min + 0.5, pearson_lags[idx_min] - 0.025),
        fontsize=8, color=FARBEN["rot"],
        arrowprops=dict(arrowstyle="->", color=FARBEN["rot"], lw=1.2)
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"Lag {l}h" for l in range(max_lag + 1)],
                       fontsize=9)
    ax.set_xlabel("Zeitverzögerung des Niederschlags")
    ax.set_ylabel("Korrelationskoeffizient")
    ax.set_title(
        f"Lag-Analyse: Niederschlag (verzögert) → {haupt_schadstoff}\n"
        f"Wirkt Regen erst mit Verzögerung reinigend?",
        fontsize=11
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, axis="y")
    plt.tight_layout()
    plt.savefig("output/plots/viz_09_lag_analyse.png",
                bbox_inches="tight", dpi=120)
    plt.close()
    print("  ✓ output/plots/viz_09_lag_analyse.png")


# ─────────────────────────────────────────────────────────────
# ANALYSE-ZUSAMMENFASSUNG (für Bericht)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ANALYSE-ERGEBNISSE – Für den Bericht")
print("=" * 60)

if haupt_schadstoff:
    print(f"\n  Zielvariable: {haupt_schadstoff}")

    if "jahreszeit" in df.columns:
        print("\n  Saisonale Mittelwerte:")
        for jz in ["Winter", "Frühling", "Sommer", "Herbst"]:
            mask = df["jahreszeit"] == jz
            if mask.sum() > 0:
                mittel = df.loc[mask, haupt_schadstoff].mean()
                print(f"    {jz:<12}: {mittel:.1f} µg/m³")

    if "regen" in df.columns:
        m_kein = df[df["regen"]==0][haupt_schadstoff].mean()
        m_mit  = df[df["regen"]==1][haupt_schadstoff].mean()
        diff   = m_kein - m_mit
        print(f"\n  Regen-Effekt: {diff:+.1f} µg/m³ "
              f"({'Regen reduziert' if diff > 0 else 'Regen erhöht'} "
              f"den Schadstoffwert um {abs(diff)/m_kein*100:.0f}%)")

    if "inversion_indikator" in df.columns:
        m_norm = df[df["inversion_indikator"]==0][haupt_schadstoff].mean()
        m_inv  = df[df["inversion_indikator"]==1][haupt_schadstoff].mean()
        diff   = m_inv - m_norm
        print(f"  Inversionslage: +{diff:.1f} µg/m³ "
              f"(+{diff/m_norm*100:.0f}% höher als normal)")

    if wetter_cols:
        print(f"\n  Stärkste Korrelationen mit {haupt_schadstoff}:")
        corr_vals = df[[haupt_schadstoff] + wetter_cols].corr()[
            haupt_schadstoff].drop(haupt_schadstoff)
        for var, r in corr_vals.abs().sort_values(
                ascending=False).items():
            rval = corr_vals[var]
            print(f"    {var:<35} r = {rval:+.3f}")

print(f"""
  Gespeicherte Plots (9 total):
  viz_01_jahresuebersicht.png   (inkl. 7-Tage-Rolling-Mean auf allen Panels)
  viz_02_korrelation.png        (Pearson + Spearman nebeneinander)
  viz_03_jahreszeit_boxplot.png
  viz_04_scatter_temperatur.png
  viz_05_regen_effekt.png
  viz_06_inversion.png
  viz_07_zeitprofile.png
  viz_08_windrose.png           (Polar-Plot: Windrichtung vs. {haupt_schadstoff})
  viz_09_lag_analyse.png        (Pearson/Spearman bei Lag 0–6h)

  Nächster Schritt: Phase 7 – Pipeline (W07_pipeline.py)
""")

print("─" * 60)
print("PHYSIKALISCHE INTERPRETATIONEN – Für den Bericht")
print("─" * 60)

_ERKL_TITEL = {
    "temperatur": "Temperatureinfluss auf Schadstoffe (viz_04)",
    "regen":      "Auswascheffekt durch Niederschlag (viz_05)",
    "inversion":  "Inversionslagen (viz_06)",
    "rush_hour":  "Tagesprofil / Rush-Hour (viz_07)",
}
for _key, _titel in _ERKL_TITEL.items():
    print(f"\n  {_titel}:")
    print(f"  {ERKLAERUNGEN[_key]}")

print("=" * 60)


# ─────────────────────────────────────────────────────────────
# HYPOTHESENTEST – Kriterium A
# ─────────────────────────────────────────────────────────────
def teste_hypothesen(df):
    """
    Testet alle in hypothesen.py definierten Hypothesen.
    Methoden:
      korrelation    → Pearson + Spearman, Schwelle |r| ≥ 0.10
      gruppenvergleich → Welch-t-Test + Cohen's d, Schwelle |d| ≥ 0.20
    Gibt dict {H-ID: Urteil-String} zurück.
    """
    try:
        from scipy import stats as sp
    except ImportError:
        print("  ⚠ scipy fehlt – pip install scipy>=1.10")
        return {}

    from hypothesen import HYPOTHESEN

    def finde_col(muster):
        if muster in df.columns:
            return muster
        kandidaten = [
            c for c in df.columns
            if muster.upper() in c.upper()
            and not any(s in c for s in
                        ["_missing", "_std", "_norm", "kategorie"])
        ]
        if muster.upper() == "NO":          # NO nicht mit NO2 verwechseln
            kandidaten = [c for c in kandidaten
                          if "NO2" not in c.upper()]
        return kandidaten[0] if kandidaten else None

    print("\n" + "=" * 60)
    print("HYPOTHESENTEST – Kriterium A")
    print("=" * 60)

    ergebnisse = {}

    for h_id, h in HYPOTHESEN.items():
        v1, v2 = h["variablen"]
        col1, col2 = finde_col(v1), finde_col(v2)

        print(f"\n{h_id}: {h['text']}")
        print(f"     Erwartet: {h['erwartet']}e Richtung")

        if col1 is None or col2 is None:
            fehlend = v1 if col1 is None else v2
            print(f"     ⚠ '{fehlend}' nicht im Datensatz → übersprungen")
            ergebnisse[h_id] = "n/a"
            continue

        methode = h.get("methode", "korrelation")

        if methode == "korrelation":
            daten = df[[col1, col2]].dropna()
            r_p, p_p = sp.pearsonr(daten[col1], daten[col2])
            r_s, p_s = sp.spearmanr(daten[col1], daten[col2])

            dir_ok = ((h["erwartet"] == "negativ" and r_p < 0) or
                      (h["erwartet"] == "positiv" and r_p > 0))
            signif  = p_p < 0.05
            stark   = abs(r_p) >= 0.10

            print(f"     Pearson  r = {r_p:+.3f}  (p = {p_p:.2e})")
            print(f"     Spearman ρ = {r_s:+.3f}  (p = {p_s:.2e})")
            print(f"     n = {len(daten):,} Stunden")

            if dir_ok and signif and stark:
                urteil = "✓ bestätigt"
            elif dir_ok and signif:
                urteil = "~ teilweise bestätigt  (Effekt schwach, |r| < 0.10)"
            elif not dir_ok:
                urteil = "✗ nicht bestätigt  (Richtung verkehrt)"
            else:
                urteil = "~ teilweise bestätigt  (nicht signifikant)"

        elif methode == "gruppenvergleich":
            g0 = df[df[col1] == 0][col2].dropna()
            g1 = df[df[col1] == 1][col2].dropna()

            if len(g0) < 5 or len(g1) < 5:
                print(f"     ⚠ Zu wenige Datenpunkte → übersprungen")
                ergebnisse[h_id] = "n/a"
                continue

            t_stat, p_val = sp.ttest_ind(g1, g0, equal_var=False)
            diff   = g1.mean() - g0.mean()
            pooled = np.sqrt((g0.std()**2 + g1.std()**2) / 2)
            cohens_d = diff / pooled if pooled > 0 else 0.0

            dir_ok = ((h["erwartet"] == "positiv" and diff > 0) or
                      (h["erwartet"] == "negativ" and diff < 0))
            signif = p_val < 0.05
            stark  = abs(cohens_d) >= 0.20

            print(f"     Basis  (Gruppe 0): Ø {g0.mean():.2f} µg/m³"
                  f"  (n = {len(g0):,})")
            print(f"     Effekt (Gruppe 1): Ø {g1.mean():.2f} µg/m³"
                  f"  (n = {len(g1):,})")
            print(f"     Δ = {diff:+.2f} µg/m³  |  "
                  f"t = {t_stat:.2f}  |  p = {p_val:.2e}  |  "
                  f"Cohen's d = {cohens_d:.3f}")

            if dir_ok and signif and stark:
                urteil = "✓ bestätigt"
            elif dir_ok and signif:
                urteil = "~ teilweise bestätigt  (Effekt schwach, d < 0.20)"
            elif not dir_ok:
                urteil = "✗ nicht bestätigt  (Richtung verkehrt)"
            else:
                urteil = "~ teilweise bestätigt  (nicht signifikant)"

        else:
            urteil = "? unbekannte Methode"

        print(f"     → {urteil}")
        ergebnisse[h_id] = urteil

    # ── Zusammenfassung ──
    n_best = sum(1 for v in ergebnisse.values() if v.startswith("✓"))
    n_teil = sum(1 for v in ergebnisse.values() if v.startswith("~"))
    n_nein = sum(1 for v in ergebnisse.values() if v.startswith("✗"))
    n_na   = sum(1 for v in ergebnisse.values() if v == "n/a")

    print(f"\n{'─' * 60}")
    print("ERGEBNIS HYPOTHESENTEST:")
    print(f"  ✓ bestätigt:          {n_best}/{len(HYPOTHESEN)}")
    print(f"  ~ teilweise:          {n_teil}/{len(HYPOTHESEN)}")
    print(f"  ✗ nicht bestätigt:    {n_nein}/{len(HYPOTHESEN)}")
    if n_na:
        print(f"  ⚠ nicht testbar:      {n_na}/{len(HYPOTHESEN)}")
    print("=" * 60)

    return ergebnisse


teste_hypothesen(df)