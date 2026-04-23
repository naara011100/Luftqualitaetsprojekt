# =============================================================
# DWAE Projekt – Phase 2: Explorative Datenanalyse (EDA)
# Thema: Einfluss von Wetter auf Luftqualität in Zürich
# =============================================================
# Voraussetzung: Phase 1 abgeschlossen, Dateien in data/raw/
# Ausführen:     python W02_eda.py
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

Path("data/processed").mkdir(parents=True, exist_ok=True)
Path("output/plots").mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("PHASE 2 – Explorative Datenanalyse (EDA)")
print("=" * 60)


# ─────────────────────────────────────────────────────────────
# 1. DATEN LADEN
# ─────────────────────────────────────────────────────────────
print("\n[1/7] Daten laden...")

# Luftqualität
df_luft = pd.read_csv("data/raw/luftqualitaet_2023_roh.csv")
# Zeitstempel parsen – Spaltenname ggf. anpassen (z.B. "Datum", "time")
zeit_col = df_luft.columns[0]
df_luft[zeit_col] = pd.to_datetime(df_luft[zeit_col])
df_luft = df_luft.rename(columns={zeit_col: "timestamp"})
df_luft = df_luft.sort_values("timestamp").reset_index(drop=True)

# Wetter
df_wetter = pd.read_csv("data/raw/wetter_2023_roh.csv")
df_wetter["Datum"] = pd.to_datetime(df_wetter["Datum"])
df_wetter = df_wetter.sort_values("Datum").reset_index(drop=True)

print(f"  Luftqualität : {df_luft.shape[0]:>7,} Zeilen × {df_luft.shape[1]} Spalten")
print(f"  Wetter       : {df_wetter.shape[0]:>7,} Zeilen × {df_wetter.shape[1]} Spalten")


# ─────────────────────────────────────────────────────────────
# 2. FORM UND DATENTYPEN
# ─────────────────────────────────────────────────────────────
print("\n[2/7] Form und Datentypen...")

for name, df in [("Luftqualität", df_luft), ("Wetter", df_wetter)]:
    print(f"\n  --- {name} ---")
    print(f"  Zeitraum: {df.iloc[:, 0].min()} bis {df.iloc[:, 0].max()}")
    print(f"  {'Spalte':<35} {'Dtype':<12} {'Min':>10} {'Max':>10} {'Unique':>8}")
    print(f"  {'-'*75}")
    for col in df.columns:
        dtype = str(df[col].dtype)
        if pd.api.types.is_numeric_dtype(df[col]):
            mn = f"{df[col].min():.2f}"
            mx = f"{df[col].max():.2f}"
        else:
            mn = str(df[col].min())[:10]
            mx = str(df[col].max())[:10]
        uniq = df[col].nunique()
        print(f"  {col:<35} {dtype:<12} {mn:>10} {mx:>10} {uniq:>8}")


# ─────────────────────────────────────────────────────────────
# 3. FEHLENDE WERTE
# ─────────────────────────────────────────────────────────────
print("\n[3/7] Fehlende Werte analysieren...")

for name, df in [("Luftqualität", df_luft), ("Wetter", df_wetter)]:
    print(f"\n  --- {name} ---")
    missing = df.isnull().sum()
    missing_pct = missing / len(df) * 100
    missing_df = pd.DataFrame({
        "Fehlend": missing,
        "Prozent": missing_pct
    }).sort_values("Fehlend", ascending=False)
    missing_df = missing_df[missing_df["Fehlend"] > 0]
    if missing_df.empty:
        print("  Keine fehlenden Werte!")
    else:
        print(f"  {'Spalte':<35} {'Fehlend':>8} {'%':>8}")
        print(f"  {'-'*53}")
        for col, row in missing_df.iterrows():
            bar = "█" * int(row["Prozent"] / 2)
            print(f"  {col:<35} {int(row['Fehlend']):>8} {row['Prozent']:>7.1f}%  {bar}")

# Heatmap fehlende Werte (Luftqualität)
numeric_luft = df_luft.select_dtypes(include="number")
if not numeric_luft.empty:
    fig, ax = plt.subplots(figsize=(12, 3))
    missing_matrix = numeric_luft.isnull().astype(int)
    # Monatliche Aggregation für bessere Übersicht
    missing_matrix.index = df_luft["timestamp"]
    missing_monthly = missing_matrix.resample("ME").mean()
    im = ax.imshow(missing_monthly.T, aspect="auto", cmap="RdYlGn_r",
                   vmin=0, vmax=1)
    ax.set_yticks(range(len(missing_monthly.columns)))
    ax.set_yticklabels(missing_monthly.columns, fontsize=8)
    ax.set_xticks(range(len(missing_monthly.index)))
    ax.set_xticklabels([d.strftime("%b") for d in missing_monthly.index],
                       fontsize=8)
    ax.set_title("Fehlende Werte Luftqualität (grün = vollständig, rot = leer)",
                 fontsize=10)
    plt.colorbar(im, ax=ax, label="Anteil fehlend")
    plt.tight_layout()
    plt.savefig("output/plots/eda_fehlende_werte_heatmap.png", dpi=120)
    plt.close()
    print("\n  Gespeichert: output/plots/eda_fehlende_werte_heatmap.png")


# ─────────────────────────────────────────────────────────────
# 4. DUPLIKATE
# ─────────────────────────────────────────────────────────────
print("\n[4/7] Duplikate prüfen...")

for name, df, tcol in [("Luftqualität", df_luft, "timestamp"),
                        ("Wetter", df_wetter, "Datum")]:
    dup_total = df.duplicated().sum()
    dup_ts = df.duplicated(subset=[tcol]).sum()
    print(f"  {name}: {dup_total} vollständige Duplikate, "
          f"{dup_ts} doppelte Zeitstempel")


# ─────────────────────────────────────────────────────────────
# 5. DESKRIPTIVE STATISTIK
# ─────────────────────────────────────────────────────────────
print("\n[5/7] Deskriptive Statistik...")

print("\n  Luftqualität – Schadstoffe:")
luft_num = df_luft.select_dtypes(include="number")
stats = luft_num.describe().T[["mean", "std", "min", "50%", "max"]]
stats.columns = ["Mittelwert", "Std", "Min", "Median", "Max"]
print(stats.round(2).to_string())

print("\n  Wetter – Variablen:")
wetter_num = df_wetter.select_dtypes(include="number")
stats_w = wetter_num.describe().T[["mean", "std", "min", "50%", "max"]]
stats_w.columns = ["Mittelwert", "Std", "Min", "Median", "Max"]
print(stats_w.round(2).to_string())


# ─────────────────────────────────────────────────────────────
# 6. VERTEILUNGEN UND ZEITREIHEN
# ─────────────────────────────────────────────────────────────
print("\n[6/7] Plots erstellen...")

# --- Plot 1: Zeitreihen Schadstoffe ---
luft_cols = [c for c in df_luft.columns
             if any(x in c.upper() for x in ["NO2", "O3", "PM10", "PM2"])]
if luft_cols:
    fig, axes = plt.subplots(len(luft_cols), 1,
                             figsize=(14, 3 * len(luft_cols)), sharex=True)
    if len(luft_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, luft_cols):
        # Tagesmedian für übersichtlichere Darstellung
        daily = df_luft.set_index("timestamp")[col].resample("D").median()
        ax.plot(daily.index, daily.values, linewidth=0.8, color="#1D9E75")
        ax.fill_between(daily.index, daily.values, alpha=0.15, color="#1D9E75")
        ax.set_ylabel(col, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axes[0].set_title("Luftschadstoffe 2023 – Tagesmedian", fontsize=11)
    plt.tight_layout()
    plt.savefig("output/plots/eda_zeitreihen_schadstoffe.png", dpi=120)
    plt.close()
    print("  Gespeichert: output/plots/eda_zeitreihen_schadstoffe.png")

# --- Plot 2: Zeitreihen Wetter ---
wetter_cols = ["temperature_2m", "precipitation",
               "wind_speed_10m", "relative_humidity_2m"]
wetter_cols = [c for c in wetter_cols if c in df_wetter.columns]
colors = ["#D85A30", "#185FA5", "#534AB7", "#0F6E56"]
units  = ["°C", "mm", "km/h", "%"]

fig, axes = plt.subplots(len(wetter_cols), 1,
                         figsize=(14, 3 * len(wetter_cols)), sharex=True)
if len(wetter_cols) == 1:
    axes = [axes]
for ax, col, color, unit in zip(axes, wetter_cols, colors, units):
    daily = df_wetter.set_index("Datum")[col].resample("D").mean()
    ax.plot(daily.index, daily.values, linewidth=0.8, color=color)
    ax.fill_between(daily.index, daily.values, alpha=0.12, color=color)
    ax.set_ylabel(f"{col}\n({unit})", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
axes[0].set_title("Wettervariablen 2023 – Tagesmittel", fontsize=11)
plt.tight_layout()
plt.savefig("output/plots/eda_zeitreihen_wetter.png", dpi=120)
plt.close()
print("  Gespeichert: output/plots/eda_zeitreihen_wetter.png")

# --- Plot 3: Histogramme Schadstoffe ---
if luft_cols:
    n = len(luft_cols)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, luft_cols):
        vals = df_luft[col].dropna()
        ax.hist(vals, bins=60, color="#1D9E75", edgecolor="white",
                linewidth=0.3, alpha=0.85)
        ax.axvline(vals.mean(), color="#712B13", linewidth=1.5,
                   linestyle="--", label=f"Mittel: {vals.mean():.1f}")
        ax.axvline(vals.median(), color="#0C447C", linewidth=1.5,
                   linestyle=":", label=f"Median: {vals.median():.1f}")
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("Wert")
        ax.set_ylabel("Häufigkeit")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.suptitle("Verteilungen Luftschadstoffe", fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig("output/plots/eda_histogramme_schadstoffe.png",
                dpi=120, bbox_inches="tight")
    plt.close()
    print("  Gespeichert: output/plots/eda_histogramme_schadstoffe.png")

# --- Plot 4: Saisonaler Boxplot (Monat) ---
if luft_cols:
    col = luft_cols[0]
    df_luft["Monat"] = df_luft["timestamp"].dt.month
    monate = ["Jan","Feb","Mär","Apr","Mai","Jun",
              "Jul","Aug","Sep","Okt","Nov","Dez"]
    fig, ax = plt.subplots(figsize=(12, 4))
    data_by_month = [df_luft[df_luft["Monat"] == m][col].dropna().values
                     for m in range(1, 13)]
    bp = ax.boxplot(data_by_month, patch_artist=True, notch=False,
                    medianprops=dict(color="#712B13", linewidth=1.5))
    for patch in bp["boxes"]:
        patch.set_facecolor("#9FE1CB")
        patch.set_alpha(0.7)
    ax.set_xticklabels(monate)
    ax.set_title(f"Saisonale Verteilung – {col} (2023)", fontsize=11)
    ax.set_ylabel(col)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("output/plots/eda_saisonaler_boxplot.png", dpi=120)
    plt.close()
    print("  Gespeichert: output/plots/eda_saisonaler_boxplot.png")
    df_luft.drop(columns=["Monat"], inplace=True)


# ─────────────────────────────────────────────────────────────
# 7. AUSREISSER ERKENNEN (IQR-Methode)
# ─────────────────────────────────────────────────────────────
print("\n[7/7] Ausreisser erkennen (IQR-Methode)...")

def erkenne_ausreisser_iqr(df, cols, label=""):
    print(f"\n  --- {label} ---")
    print(f"  {'Spalte':<35} {'Q1':>8} {'Q3':>8} {'IQR':>8} "
          f"{'Untergrenze':>12} {'Obergrenze':>12} {'Ausreisser':>11}")
    print(f"  {'-'*100}")
    ergebnisse = {}
    for col in cols:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        untere = q1 - 1.5 * iqr
        obere  = q3 + 1.5 * iqr
        n_out  = ((s < untere) | (s > obere)).sum()
        pct    = n_out / len(s) * 100
        ergebnisse[col] = {
            "q1": q1, "q3": q3, "iqr": iqr,
            "untere": untere, "obere": obere,
            "n_ausreisser": n_out, "pct": pct
        }
        flag = " (!)" if pct > 5 else ""
        print(f"  {col:<35} {q1:>8.2f} {q3:>8.2f} {iqr:>8.2f} "
              f"{untere:>12.2f} {obere:>12.2f} "
              f"{n_out:>8} ({pct:.1f}%){flag}")
    return ergebnisse

if luft_cols:
    out_luft = erkenne_ausreisser_iqr(df_luft, luft_cols, "Luftqualität")

out_wetter = erkenne_ausreisser_iqr(
    df_wetter,
    ["temperature_2m", "precipitation", "wind_speed_10m",
     "relative_humidity_2m", "surface_pressure"],
    "Wetter"
)


# ─────────────────────────────────────────────────────────────
# ZUSAMMENFASSUNG & BEREINIGUNGS-STRATEGIE
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ZUSAMMENFASSUNG EDA – Bereinigungs-Strategie")
print("=" * 60)
print("""
Basierend auf der EDA empfohlene nächste Schritte (Phase 3):

  Fehlende Werte:
  - Kurze Lücken (1-3h): Lineare Interpolation (Zeitreihe)
  - Längere Lücken:      Als NaN belassen + Indikatorvariable
  - Dokumentieren:       Warum fehlen Werte? (MCAR/MAR/MNAR)

  Ausreisser:
  - Physikalisch unmögliche Werte entfernen (z.B. NO2 < 0)
  - Extremwerte dokumentieren (Ereignisse? Messfehler?)
  - Winsorisierung bei starker Rechtsschiefe erwägen

  Zeitstempel:
  - Beide Datensätze auf UTC oder Europe/Zurich vereinheitlichen
  - Merge über stündlichen Zeitstempel (Phase 4)

  Für den Bericht notieren:
  - Welche Stationen sind vorhanden?
  - Wie vollständig sind die Daten (% vorhanden)?
  - Gibt es saisonale Muster (→ relevant für Analyse)?
""")

print("Gespeicherte Plots: output/plots/")
for f in sorted(Path("output/plots").glob("eda_*.png")):
    print(f"  {f}")

print("\nNächster Schritt: Phase 3 – Datenbereinigung (W03_bereinigung.py)")
print("=" * 60)