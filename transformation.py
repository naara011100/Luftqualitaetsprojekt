# =============================================================
# DWAE Projekt – Phase 4: Datentransformation & Merge
# Thema: Einfluss von Wetter auf Luftqualität in Zürich
# =============================================================
# Voraussetzung: Phase 3 abgeschlossen
# Ausführen:     python W04_transformation.py
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

Path("data/processed").mkdir(parents=True, exist_ok=True)
Path("output/plots").mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("PHASE 4 – Datentransformation & Feature Engineering")
print("=" * 60)


# ─────────────────────────────────────────────────────────────
# 1. BEREINIGTE DATEN LADEN
# ─────────────────────────────────────────────────────────────
print("\n[1/6] Bereinigte Daten laden...")

df_luft = pd.read_csv("data/processed/luftqualitaet_2023_clean.csv")
df_luft["timestamp"] = pd.to_datetime(df_luft["timestamp"])
df_luft = df_luft.sort_values("timestamp").reset_index(drop=True)

df_wetter = pd.read_csv("data/processed/wetter_2023_clean.csv")
df_wetter["Datum"] = pd.to_datetime(df_wetter["Datum"])
df_wetter = df_wetter.sort_values("Datum").reset_index(drop=True)

print(f"  Luftqualität : {df_luft.shape[0]:,} Zeilen × {df_luft.shape[1]} Spalten")
print(f"  Wetter       : {df_wetter.shape[0]:,} Zeilen × {df_wetter.shape[1]} Spalten")


# ─────────────────────────────────────────────────────────────
# 2. ZEITSTEMPEL VEREINHEITLICHEN
# ─────────────────────────────────────────────────────────────
# Beide Datensätze müssen denselben Zeitstempel-Format haben
# damit der Merge korrekt funktioniert.
print("\n[2/6] Zeitstempel vereinheitlichen...")

# Auf stündliche Präzision runden (Minuten/Sekunden entfernen)
df_luft["timestamp"] = df_luft["timestamp"].dt.floor("h")
df_wetter["Datum"]   = df_wetter["Datum"].dt.floor("h")

# Zeitzone sicherstellen (beide auf Europe/Zurich)
# Falls Zeitzone bereits gesetzt: konvertieren, sonst: lokalisieren
def setze_zeitzone(series, tz="Europe/Zurich"):
    if series.dt.tz is None:
        return series.dt.tz_localize(tz, ambiguous="NaT",
                                     nonexistent="shift_forward")
    else:
        return series.dt.tz_convert(tz)

df_luft["timestamp"] = setze_zeitzone(df_luft["timestamp"])
df_wetter["Datum"]   = setze_zeitzone(df_wetter["Datum"])

# Zeitbereich prüfen
print(f"  Luftqualität : {df_luft['timestamp'].min()} "
      f"bis {df_luft['timestamp'].max()}")
print(f"  Wetter       : {df_wetter['Datum'].min()} "
      f"bis {df_wetter['Datum'].max()}")

# Überlappenden Zeitraum bestimmen
start = max(df_luft["timestamp"].min(), df_wetter["Datum"].min())
ende  = min(df_luft["timestamp"].max(), df_wetter["Datum"].max())
print(f"  Überlappung  : {start} bis {ende}")

# Auf gemeinsamen Zeitraum filtern
df_luft   = df_luft[(df_luft["timestamp"] >= start) &
                     (df_luft["timestamp"] <= ende)]
df_wetter = df_wetter[(df_wetter["Datum"] >= start) &
                       (df_wetter["Datum"] <= ende)]


# ─────────────────────────────────────────────────────────────
# 3. DATENSÄTZE ZUSAMMENFÜHREN (MERGE)
# ─────────────────────────────────────────────────────────────
# Inner Join: nur Stunden behalten, für die BEIDE Quellen
# Daten haben. So vermeiden wir NaNs durch fehlende Joins.
print("\n[3/6] Datensätze zusammenführen (Merge)...")

# Wetter umbenennen für den Merge
df_wetter_merge = df_wetter.rename(columns={"Datum": "timestamp"})

df_merged = pd.merge(
    df_luft,
    df_wetter_merge,
    on="timestamp",
    how="inner",          # nur übereinstimmende Zeitstempel
    suffixes=("_luft", "_wetter")
)

df_merged = df_merged.sort_values("timestamp").reset_index(drop=True)

print(f"  Luftqualität : {len(df_luft):,} Zeilen")
print(f"  Wetter       : {len(df_wetter):,} Zeilen")
print(f"  Nach Merge   : {len(df_merged):,} Zeilen × "
      f"{df_merged.shape[1]} Spalten")
print(f"  Verlust      : {len(df_luft) - len(df_merged):,} Zeilen "
      f"(keine Entsprechung in beiden Quellen)")

# Merge-Qualität prüfen
merge_pct = len(df_merged) / max(len(df_luft), len(df_wetter)) * 100
print(f"  Merge-Qualität: {merge_pct:.1f}% der Stunden erfolgreich gemergt")
if merge_pct < 80:
    print("  WARNUNG: <80% gemergt – Zeitstempel-Format prüfen!")


# ─────────────────────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
# Neue Features aus bestehenden Spalten ableiten,
# die für die Analyse Wetter ↔ Luftqualität nützlich sind.
print("\n[4/6] Feature Engineering...")

# ── Zeitfeatures ──
df_merged["stunde"]       = df_merged["timestamp"].dt.hour
df_merged["wochentag"]    = df_merged["timestamp"].dt.dayofweek   # 0=Mo
df_merged["monat"]        = df_merged["timestamp"].dt.month
df_merged["jahreszeit"]   = df_merged["monat"].map({
    12: "Winter", 1: "Winter",  2: "Winter",
    3:  "Frühling", 4: "Frühling", 5: "Frühling",
    6:  "Sommer",  7: "Sommer",  8: "Sommer",
    9:  "Herbst",  10: "Herbst", 11: "Herbst"
})
df_merged["ist_wochenende"] = (df_merged["wochentag"] >= 5).astype(int)
df_merged["ist_rush_hour"]  = df_merged["stunde"].isin(
    [7, 8, 9, 17, 18, 19]).astype(int)  # Mo-Fr Pendlerzeiten

print("  Zeitfeatures erstellt: stunde, wochentag, monat, "
      "jahreszeit, ist_wochenende, ist_rush_hour")

# ── Wetterfeatures ──
# Temperaturkategorien (für Ozon-Analyse relevant: hohe Temp → mehr O3)
if "temperature_2m" in df_merged.columns:
    df_merged["temp_kategorie"] = pd.cut(
        df_merged["temperature_2m"],
        bins=[-30, 0, 10, 20, 30, 50],
        labels=["Frost", "Kalt", "Mild", "Warm", "Heiss"]
    )
    print("  temp_kategorie erstellt: Frost/Kalt/Mild/Warm/Heiss")

# Niederschlag binär (regnet / regnet nicht)
if "precipitation" in df_merged.columns:
    df_merged["regen"] = (df_merged["precipitation"] > 0.1).astype(int)
    print("  regen erstellt: 1 = Niederschlag > 0.1mm")

# Wind-Kategorien (Beaufort vereinfacht)
if "wind_speed_10m" in df_merged.columns:
    df_merged["wind_kategorie"] = pd.cut(
        df_merged["wind_speed_10m"],
        bins=[0, 5, 20, 40, 300],
        labels=["Windstill", "Leicht", "Mässig", "Stark"]
    )
    print("  wind_kategorie erstellt: Windstill/Leicht/Mässig/Stark")

# Inversions-Indikator: hoher Luftdruck + niedrige Temperatur
# → schlechte Durchmischung → Schadstoffe sammeln sich
if "surface_pressure" in df_merged.columns and \
   "temperature_2m" in df_merged.columns:
    druck_hoch = df_merged["surface_pressure"] > \
                 df_merged["surface_pressure"].quantile(0.75)
    temp_kalt  = df_merged["temperature_2m"] < \
                 df_merged["temperature_2m"].quantile(0.25)
    df_merged["inversion_indikator"] = (druck_hoch & temp_kalt).astype(int)
    n_inv = df_merged["inversion_indikator"].sum()
    print(f"  inversion_indikator erstellt: {n_inv:,} Stunden ({n_inv/len(df_merged)*100:.1f}%)")

# Sonnenschein in Stunden (statt Sekunden)
if "sunshine_duration" in df_merged.columns:
    df_merged["sonnenschein_h"] = df_merged["sunshine_duration"] / 3600
    print("  sonnenschein_h erstellt: Sonnenscheindauer in Stunden")

print(f"\n  Mergedatensatz: {df_merged.shape[0]:,} Zeilen × "
      f"{df_merged.shape[1]} Spalten")


# ─────────────────────────────────────────────────────────────
# 5. SKALIERUNG (für spätere Analyse vorbereiten)
# ─────────────────────────────────────────────────────────────
# Wir erstellen skalierte Features als separate Spalten,
# damit die Originaldaten erhalten bleiben.
print("\n[5/6] Skalierung numerischer Features...")

from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Nur numerische Spalten ohne Hilfsspalten skalieren
skip_cols = {"timestamp", "stunde", "wochentag", "monat",
             "ist_wochenende", "ist_rush_hour", "regen",
             "inversion_indikator"}
skip_suffixes = ("_missing",)

num_cols = [
    c for c in df_merged.select_dtypes(include="number").columns
    if c not in skip_cols
    and not any(c.endswith(s) for s in skip_suffixes)
]

# StandardScaler (Z-Score) für ML-Modelle
scaler_std = StandardScaler()
scaled_std = scaler_std.fit_transform(df_merged[num_cols].fillna(0))
df_scaled_std = pd.DataFrame(
    scaled_std,
    columns=[f"{c}_std" for c in num_cols],
    index=df_merged.index
)

# MinMaxScaler [0,1] für Visualisierungen
scaler_mm = MinMaxScaler()
scaled_mm = scaler_mm.fit_transform(df_merged[num_cols].fillna(0))
df_scaled_mm = pd.DataFrame(
    scaled_mm,
    columns=[f"{c}_norm" for c in num_cols],
    index=df_merged.index
)

print(f"  StandardScaler (Z-Score): {len(num_cols)} Spalten → *_std")
print(f"  MinMaxScaler  [0–1]:      {len(num_cols)} Spalten → *_norm")
print(f"  Skalierte auf Trainingsdaten (gesamter 2023er Datensatz)")


# ─────────────────────────────────────────────────────────────
# 6. KORRELATIONSANALYSE & PLOTS
# ─────────────────────────────────────────────────────────────
print("\n[6/6] Korrelationsanalyse & Plots...")

# Schadstoffe und Wettervariablen identifizieren
schadstoff_cols = [c for c in df_merged.columns
                   if any(x in c.upper()
                          for x in ["NO2", "NO", "O3", "PM10", "PM2"])
                   and "_missing" not in c
                   and "_std" not in c
                   and "_norm" not in c]

wetter_cols_plot = [c for c in [
    "temperature_2m", "precipitation", "wind_speed_10m",
    "relative_humidity_2m", "surface_pressure", "sonnenschein_h"]
    if c in df_merged.columns]

alle_analyse_cols = schadstoff_cols + wetter_cols_plot

# ── Plot 1: Korrelationsmatrix ──
if len(alle_analyse_cols) >= 2:
    corr = df_merged[alle_analyse_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 9))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Pearson-Korrelation")

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)

    # Werte in Zellen eintragen
    for i in range(len(corr)):
        for j in range(len(corr.columns)):
            val = corr.iloc[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color)

    ax.set_title("Korrelationsmatrix: Schadstoffe × Wettervariablen (2023)",
                 fontsize=11, pad=15)
    plt.tight_layout()
    plt.savefig("output/plots/transformation_korrelation.png",
                dpi=120, bbox_inches="tight")
    plt.close()
    print("  Gespeichert: output/plots/transformation_korrelation.png")

    # Top-Korrelationen ausgeben
    print("\n  Stärkste Korrelationen (Wetter ↔ Schadstoffe):")
    print(f"  {'Paar':<50} {'Korrelation':>12}")
    print(f"  {'-'*64}")
    pairs = []
    for s in schadstoff_cols:
        for w in wetter_cols_plot:
            if s in corr.index and w in corr.columns:
                pairs.append((s, w, corr.loc[s, w]))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    for s, w, r in pairs[:10]:
        richtung = "↑" if r > 0 else "↓"
        print(f"  {s:<25} × {w:<25} {r:>+.3f} {richtung}")

# ── Plot 2: Scatterplots Temperatur vs. Schadstoffe ──
if schadstoff_cols and "temperature_2m" in df_merged.columns:
    n = min(len(schadstoff_cols), 3)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, schadstoff_cols[:n]):
        sample = df_merged[[col, "temperature_2m", "jahreszeit"]].dropna()
        farben = {"Winter": "#185FA5", "Frühling": "#1D9E75",
                  "Sommer": "#D85A30", "Herbst":  "#B8860B"}
        for jz, gruppe in sample.groupby("jahreszeit"):
            ax.scatter(gruppe["temperature_2m"], gruppe[col],
                       alpha=0.15, s=2, color=farben.get(jz, "grey"),
                       label=jz)
        ax.set_xlabel("Temperatur (°C)")
        ax.set_ylabel(col)
        ax.set_title(f"Temperatur vs. {col}")
        ax.legend(fontsize=7, markerscale=4)
        ax.grid(True, alpha=0.3)
    plt.suptitle("Temperatur vs. Luftschadstoffe (farbig nach Jahreszeit)",
                 fontsize=10)
    plt.tight_layout()
    plt.savefig("output/plots/transformation_scatter_temp.png",
                dpi=120, bbox_inches="tight")
    plt.close()
    print("  Gespeichert: output/plots/transformation_scatter_temp.png")

# ── Plot 3: Tagesprofil (Stunde) ──
if schadstoff_cols:
    col = schadstoff_cols[0]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Nach Stunde
    stunden_profil = df_merged.groupby("stunde")[col].median()
    axes[0].plot(stunden_profil.index, stunden_profil.values,
                 color="#1D9E75", linewidth=2, marker="o", markersize=4)
    axes[0].fill_between(stunden_profil.index, stunden_profil.values,
                         alpha=0.15, color="#1D9E75")
    axes[0].set_title(f"Tagesprofil {col} (Stundenmedian)")
    axes[0].set_xlabel("Stunde des Tages")
    axes[0].set_ylabel(col)
    axes[0].set_xticks(range(0, 24, 2))
    axes[0].grid(True, alpha=0.3)

    # Nach Wochentag
    tage = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]
    tag_profil = df_merged.groupby("wochentag")[col].median()
    farben_tage = ["#1D9E75"] * 5 + ["#D85A30"] * 2
    axes[1].bar(tage, tag_profil.values, color=farben_tage, alpha=0.8,
                edgecolor="white")
    axes[1].set_title(f"Wochenprofil {col} (Tagesmedian)")
    axes[1].set_xlabel("Wochentag")
    axes[1].set_ylabel(col)
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.suptitle(f"Zeitliche Muster – {col}", fontsize=11)
    plt.tight_layout()
    plt.savefig("output/plots/transformation_zeitprofile.png", dpi=120)
    plt.close()
    print("  Gespeichert: output/plots/transformation_zeitprofile.png")

# ── Finalen Datensatz speichern ──
df_final = pd.concat([df_merged, df_scaled_std, df_scaled_mm], axis=1)
df_final.to_csv("data/processed/datensatz_final.csv", index=False)
print(f"\n  Finaler Datensatz gespeichert:")
print(f"  data/processed/datensatz_final.csv")
print(f"  ({df_final.shape[0]:,} Zeilen × {df_final.shape[1]} Spalten)")


# ─────────────────────────────────────────────────────────────
# ZUSAMMENFASSUNG FÜR DEN BERICHT
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ZUSAMMENFASSUNG – Transformationsschritte für den Bericht")
print("=" * 60)
print(f"""
  Dokumentierte Transformationsschritte:

  1. Zeitstempel vereinheitlicht
     - Beide Datensätze auf stündliche Präzision gerundet
     - Zeitzone: Europe/Zurich für beide Quellen
     - Überlappender Zeitraum: {start.date()} bis {ende.date()}

  2. Inner Join über Zeitstempel
     - {len(df_merged):,} Stunden erfolgreich gemergt ({merge_pct:.1f}%)
     - Methode: pd.merge(..., how='inner', on='timestamp')

  3. Feature Engineering
     - Zeitfeatures: Stunde, Wochentag, Monat, Jahreszeit
     - Binäre Features: Wochenende, Rush Hour, Regen
     - Kategorien: Temperatur, Wind (Beaufort)
     - Inversions-Indikator: hoher Druck + tiefe Temperatur

  4. Skalierung
     - StandardScaler (Z-Score): für ML-Algorithmen
     - MinMaxScaler [0–1]: für Visualisierungen
     - Original-Werte bleiben erhalten

  → Finaler Datensatz: data/processed/datensatz_final.csv
""")

print("Nächster Schritt: Phase 5+6 – Pipeline & Visualisierung")
print("(W05_pipeline.py)")
print("=" * 60)