# =============================================================
# DWAE Projekt – Phase 6: Visualisierung & Analyse
# Thema: Einfluss von Wetter auf Luftqualität in Zürich
# =============================================================
# Voraussetzung: Phase 5 abgeschlossen
# Ausführen:     python W06_visualisierung.py
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from pathlib import Path

Path("output/plots").mkdir(parents=True, exist_ok=True)

# Einheitliches Styling für alle Plots
plt.rcParams.update({
    "font.family":    "sans-serif",
    "font.size":      10,
    "axes.titlesize": 11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":     120,
})

FARBEN = {
    "blau":    "#185FA5",
    "gruen":   "#1D9E75",
    "rot":     "#D85A30",
    "lila":    "#534AB7",
    "gelb":    "#C89B1C",
    "grau":    "#6B7280",
}
JAHRESZEIT_FARBEN = {
    "Winter":   FARBEN["blau"],
    "Frühling": FARBEN["gruen"],
    "Sommer":   FARBEN["rot"],
    "Herbst":   FARBEN["gelb"],
}

print("=" * 60)
print("PHASE 6 – Visualisierung & Analyse")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# DATEN LADEN
# ─────────────────────────────────────────────────────────────
df = pd.read_csv("data/processed/datensatz_final.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

schadstoff_cols = [c for c in df.columns
                   if any(x in c.upper()
                          for x in ["NO2","NO","O3","PM10","PM2"])
                   and "_missing" not in c
                   and "_std" not in c
                   and "_norm" not in c
                   and "kategorie" not in c]

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

    for spec, col, farbe, titel, einheit in plots:
        ax = fig.add_subplot(spec)
        if col not in daily.columns:
            continue
        ax.plot(daily.index, daily[col], lw=0.9, color=farbe)
        ax.fill_between(daily.index, daily[col], alpha=0.12, color=farbe)
        ax.set_title(titel)
        ax.set_ylabel(einheit, fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        "Einfluss von Wetter auf Luftqualität in Zürich – Jahresübersicht 2023",
        fontsize=13, y=1.01
    )
    plt.savefig("output/plots/viz_01_jahresuebersicht.png",
                bbox_inches="tight")
    plt.close()
    print("  ✓ output/plots/viz_01_jahresuebersicht.png")


# ─────────────────────────────────────────────────────────────
# PLOT 2: KORRELATIONS-HEATMAP
# ─────────────────────────────────────────────────────────────
print("[2/7] Korrelations-Heatmap...")

alle_cols = schadstoff_cols + wetter_cols
alle_cols = [c for c in alle_cols if c in df.columns]

if len(alle_cols) >= 2:
    corr = df[alle_cols].corr()

    fig, ax = plt.subplots(figsize=(11, 8))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")

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

    ax.set_title("Korrelationsmatrix: Schadstoffe & Wettervariablen (2023)",
                 pad=14)
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
                             figsize=(5 * len(schadstoff_cols[:3]), 5))
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

        ax.set_xticklabels(reihenfolge, fontsize=9)
        ax.set_title(col)
        ax.set_ylabel("µg/m³")
        ax.grid(True, alpha=0.25, axis="y")

        # Mittelwerte als Punkte
        for i, vals in enumerate(data_jz):
            if len(vals) > 0:
                ax.plot(i + 1, np.mean(vals), marker="D",
                        color="white", markersize=5, zorder=5)

    fig.suptitle("Luftschadstoffe nach Jahreszeit – Zürich 2023",
                 fontsize=12)
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
        ax.legend(fontsize=7, markerscale=5)
        ax.grid(True, alpha=0.25)

    fig.suptitle("Einfluss der Temperatur auf Luftschadstoffe – Zürich 2023",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig("output/plots/viz_04_scatter_temperatur.png",
                bbox_inches="tight")
    plt.close()
    print("  ✓ output/plots/viz_04_scatter_temperatur.png")


# ─────────────────────────────────────────────────────────────
# PLOT 5: REGEN-EFFEKT (Vor / Während / Nach Niederschlag)
# ─────────────────────────────────────────────────────────────
print("[5/7] Regen-Effekt auf Schadstoffe...")

if haupt_schadstoff and "regen" in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

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

    fig.suptitle("Auswascheffekt von Niederschlag auf Luftschadstoffe",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig("output/plots/viz_05_regen_effekt.png", bbox_inches="tight")
    plt.close()
    print("  ✓ output/plots/viz_05_regen_effekt.png")


# ─────────────────────────────────────────────────────────────
# PLOT 6: INVERSIONSLAGEN-ANALYSE
# ─────────────────────────────────────────────────────────────
print("[6/7] Inversionslagen-Analyse...")

if haupt_schadstoff and "inversion_indikator" in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

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
    axes[0].legend(fontsize=8)
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
        f"um Ø {diff_inv:.1f} µg/m³ ({diff_inv/normal.mean()*100:.0f}%)",
        fontsize=12
    )
    plt.tight_layout()
    plt.savefig("output/plots/viz_06_inversion.png", bbox_inches="tight")
    plt.close()
    print("  ✓ output/plots/viz_06_inversion.png")


# ─────────────────────────────────────────────────────────────
# PLOT 7: TAGES- UND WOCHENPROFIL
# ─────────────────────────────────────────────────────────────
print("[7/7] Tages- und Wochenprofil...")

if haupt_schadstoff and "stunde" in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

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
    axes[0].legend(fontsize=8)
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

        # Legende
        from matplotlib.patches import Patch
        axes[1].legend(handles=[
            Patch(color=FARBEN["gruen"], alpha=0.8, label="Werktag"),
            Patch(color=FARBEN["rot"],   alpha=0.8, label="Wochenende"),
        ], fontsize=8)

    fig.suptitle(f"Zeitliche Muster – {haupt_schadstoff} Zürich 2023",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig("output/plots/viz_07_zeitprofile.png", bbox_inches="tight")
    plt.close()
    print("  ✓ output/plots/viz_07_zeitprofile.png")


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
  Gespeicherte Plots (7 total):
  output/plots/viz_01_jahresuebersicht.png
  output/plots/viz_02_korrelation.png
  output/plots/viz_03_jahreszeit_boxplot.png
  output/plots/viz_04_scatter_temperatur.png
  output/plots/viz_05_regen_effekt.png
  output/plots/viz_06_inversion.png
  output/plots/viz_07_zeitprofile.png

  Nächster Schritt: Phase 7 – Pipeline (W07_pipeline.py)
""")
print("=" * 60)