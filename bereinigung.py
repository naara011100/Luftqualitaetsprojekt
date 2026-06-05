# =============================================================
# DWAE Projekt – Phase 3: Datenbereinigung
# Thema: Einfluss von Wetter auf Luftqualität in Zürich
# =============================================================
# Voraussetzung: Phase 2 abgeschlossen
# Ausführen:     python W03_bereinigung.py
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import plot_config                # setzt rcParams global

Path("data/processed").mkdir(parents=True, exist_ok=True)
Path("output/plots").mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("PHASE 3 – Datenbereinigung")
print("=" * 60)


# ─────────────────────────────────────────────────────────────
# 1. DATEN LADEN (Rohdaten aus Phase 1)
# ─────────────────────────────────────────────────────────────
print("\n[1/6] Rohdaten laden...")

df_luft = pd.read_csv("data/raw/luftqualitaet_2023_roh.csv")
zeit_col = df_luft.columns[0]
df_luft[zeit_col] = pd.to_datetime(df_luft[zeit_col])
df_luft = df_luft.rename(columns={zeit_col: "timestamp"})
df_luft = df_luft.sort_values("timestamp").reset_index(drop=True)

df_wetter = pd.read_csv("data/raw/wetter_2023_roh.csv")
df_wetter["Datum"] = pd.to_datetime(df_wetter["Datum"])
df_wetter = df_wetter.sort_values("Datum").reset_index(drop=True)

print(f"  Luftqualität : {df_luft.shape[0]:,} Zeilen (vorher)")
print(f"  Wetter       : {df_wetter.shape[0]:,} Zeilen (vorher)")

# Kopien für Bereinigung
df_luft_clean  = df_luft.copy()
df_wetter_clean = df_wetter.copy()


# ─────────────────────────────────────────────────────────────
# 2. DUPLIKATE ENTFERNEN
# ─────────────────────────────────────────────────────────────
print("\n[2/6] Duplikate entfernen...")

for name, df, tcol in [("Luftqualität", df_luft_clean, "timestamp"),
                        ("Wetter",       df_wetter_clean, "Datum")]:
    vorher = len(df)
    # Doppelte Zeitstempel: ersten Eintrag behalten
    df.drop_duplicates(subset=[tcol], keep="first", inplace=True)
    df.reset_index(drop=True, inplace=True)
    entfernt = vorher - len(df)
    print(f"  {name}: {entfernt} Duplikate entfernt "
          f"({vorher:,} → {len(df):,} Zeilen)")


# ─────────────────────────────────────────────────────────────
# 3. PHYSIKALISCH UNMÖGLICHE WERTE ENTFERNEN
# ─────────────────────────────────────────────────────────────
# Werte ausserhalb physikalisch plausibler Grenzen sind
# Messfehler und werden auf NaN gesetzt (nicht gelöscht,
# damit der Zeitindex vollständig bleibt).
print("\n[3/6] Physikalisch unmögliche Werte entfernen...")

# Grenzwerte basierend auf Domänenwissen Zürich / Schweiz
grenzwerte_luft = {
    # Spaltenname : (minimum, maximum, Einheit, Begründung)
    # Passe Spaltennamen an deine tatsächlichen Spalten an!
    "NO":   (0,   500,  "µg/m³", "NO kann nicht negativ sein; >500 = Messfehler"),
    "NO2":  (0,   300,  "µg/m³", "WHO-Grenzwert 40; >300 = sicher Messfehler"),
    "O3":   (0,   300,  "µg/m³", "Ozon >300 extrem unwahrscheinlich in CH"),
    "PM10": (0,   500,  "µg/m³", "PM10 >500 = grober Messfehler"),
    "PM2":  (0,   300,  "µg/m³", "PM2.5 >300 = grober Messfehler"),
}

grenzwerte_wetter = {
    "temperature_2m":       (-30, 45,   "°C",   "Extremwerte für Zürich"),
    "relative_humidity_2m": (0,   100,  "%",    "Physikalisch 0-100%"),
    "precipitation":        (0,   150,  "mm",   "Stundenmax. Schweiz ~100mm"),
    "wind_speed_10m":       (0,   200,  "km/h", "Orkan >120, >200 = Fehler"),
    "surface_pressure":     (920, 1060, "hPa",  "Normbereich Mitteleuropa"),
    "cloud_cover":          (0,   100,  "%",    "Physikalisch 0-100%"),
    "sunshine_duration":    (0,   3600, "s",    "Max. 3600s pro Stunde"),
}

def bereinige_grenzwerte(df, grenzwerte, label):
    total_ersetzt = 0
    print(f"\n  --- {label} ---")
    print(f"  {'Spalte':<30} {'Untergrenze':>12} {'Obergrenze':>12} "
          f"{'Ersetzt':>10}")
    print(f"  {'-'*68}")
    for col, (mn, mx, einheit, begruendung) in grenzwerte.items():
        # Spaltenname suchen (auch Teilübereinstimmung)
        treffer = [c for c in df.columns
                   if col.upper() in c.upper() and c != "timestamp"
                   and c != "Datum"]
        for actual_col in treffer:
            if actual_col not in df.columns:
                continue
            maske = (df[actual_col] < mn) | (df[actual_col] > mx)
            n = maske.sum()
            if n > 0:
                df.loc[maske, actual_col] = np.nan
                total_ersetzt += n
            print(f"  {actual_col:<30} {mn:>12} {mx:>12} "
                  f"{n:>8} → NaN")
            if n > 0:
                print(f"    Begründung: {begruendung}")
    print(f"  Total ersetzt: {total_ersetzt} Werte → NaN")
    return total_ersetzt

bereinige_grenzwerte(df_luft_clean,  grenzwerte_luft,   "Luftqualität")
bereinige_grenzwerte(df_wetter_clean, grenzwerte_wetter, "Wetter")


# ─────────────────────────────────────────────────────────────
# 4. FEHLENDE WERTE BEHANDELN
# ─────────────────────────────────────────────────────────────
print("\n[4/6] Fehlende Werte behandeln...")

# ── 4a. Indikatorvariablen erstellen (VOR der Imputation!) ──
# So können wir im Bericht und in der Analyse erkennen,
# wo ursprünglich Werte gefehlt haben (MCAR/MAR-Analyse)
luft_num_cols = df_luft_clean.select_dtypes(include="number").columns.tolist()
for col in luft_num_cols:
    if df_luft_clean[col].isnull().sum() > 0:
        df_luft_clean[f"{col}_missing"] = df_luft_clean[col].isnull().astype(int)

wetter_num_cols = df_wetter_clean.select_dtypes(include="number").columns.tolist()
for col in wetter_num_cols:
    if df_wetter_clean[col].isnull().sum() > 0:
        df_wetter_clean[f"{col}_missing"] = df_wetter_clean[col].isnull().astype(int)

print("  Indikatorvariablen (_missing) erstellt.")

# ── 4b. MCAR/MAR/MNAR-Analyse ──
# Vor der Imputation prüfen wir, welchem Typ die fehlenden Werte angehören:
#   MCAR – fehlen vollständig zufällig (kein Muster erkennbar)
#   MAR  – Fehlen korreliert mit anderen beobachteten Variablen
#   MNAR – Fehlen hängt vom Wert selbst ab (z.B. Sensor über Messbereich)
print("\n  MCAR/MAR/MNAR-Analyse...")

def analysiere_mcar_mar(df, num_cols, tcol, label):
    analyse_cols = [
        c for c in num_cols
        if f"{c}_missing" in df.columns
        and df[f"{c}_missing"].sum() > 0
    ]
    if not analyse_cols:
        print(f"  {label}: keine fehlenden Werte – Analyse entfällt")
        return

    print(f"\n  --- {label} ---")
    print(f"  {'Spalte':<30} {'n':>6} {'%':>5}  "
          f"{'CV mon.':>7}  {'Max-Run':>8}  Typ")
    print(f"  {'-'*78}")

    df_t = df.copy()
    df_t[tcol] = pd.to_datetime(df_t[tcol])

    for col in analyse_cols:
        miss_col = f"{col}_missing"
        n   = int(df[miss_col].sum())
        pct = n / len(df) * 100

        # Test 1: Monatlicher Variationskoeffizient
        # Hoher CV → saisonale Häufung → MAR-Hinweis
        monthly = (df_t.set_index(tcol)[miss_col]
                   .resample("ME").mean() * 100)
        cv = float(monthly.std() / monthly.mean()) if monthly.mean() > 0 else 0.0

        # Test 2: Längste zusammenhängende Fehlstrecke (Run-Length)
        # Kurze Runs (<= 3h): zufällige Aussetzer → MCAR
        # Lange Runs (> 24h): Geräteausfall/Wartung → MCAR
        # Mittlere Runs:      Kalibrierung/Saisonal → MAR
        run_ids  = (df[miss_col] != df[miss_col].shift(1)).cumsum()
        max_run  = int(
            df.groupby(run_ids)[miss_col]
              .apply(lambda x: len(x) if x.iloc[0] == 1 else 0)
              .max()
        )

        # Heuristische Klassifikation
        if cv < 0.4 and max_run <= 6:
            typ = "MCAR  (zufällige Aussetzer)"
        elif cv >= 0.6:
            typ = "MAR   (saisonal/strukturiert)"
        elif max_run > 24:
            typ = "MCAR  (längerer Geräteausfall)"
        else:
            typ = "MAR/MCAR (Blockausfälle)"

        print(f"  {col:<30} {n:>6,} {pct:>5.1f}%  "
              f"CV={cv:.2f}  run={max_run:>4}h  → {typ}")

    print("""
  Interpretationshilfe für den Bericht:
    CV (monatlicher Variationskoeffizient der Fehlrate):
      < 0.4 → gleichmässig übers Jahr verteilt → MCAR
      ≥ 0.6 → saisonale Schwankungen           → MAR

    Maximale zusammenhängende Fehlstrecke (Run):
      ≤  3h  Kurze Aussetzer: Messrauschen / Sensorwackler   → MCAR
      > 24h  Lange Blöcke:    Geplante Wartung / Ausfall      → MCAR
       4–24h  Mittlere Blöcke: Kalibrierung / saisonaler Effekt → MAR

    Folgerung für die Imputation:
      MCAR → Lineare Interpolation (≤ 3h) liefert unverzerrte Schätzungen
      MAR  → Interpolation möglich; _missing-Indikator im Modell behalten,
             da das Fehlen selbst Information trägt
      MNAR → Kritisch: Extremwerte könnten systematisch fehlen (Sensorlimit).
             Ergebnisse mit Vorsicht interpretieren.
""")

analysiere_mcar_mar(df_luft_clean,  luft_num_cols,   "timestamp", "Luftqualität")
analysiere_mcar_mar(df_wetter_clean, wetter_num_cols, "Datum",     "Wetter")

# ── Heatmap: Monatliche Fehlraten (MCAR/MAR-Visualisierung) ──
print("\n  Erstelle MCAR/MAR-Heatmap...")

def plot_mcar_heatmap(datasets, save_path):
    """
    Manuelle Heatmap der monatlichen Fehlraten ohne missingno.
    datasets : list of (df, num_cols, tcol, prefix)
    Farben   : grün = vollständig · gelb = 50 % fehlt · rot = alles fehlt
    Unten    : automatische Textinterpretation (MAR / MCAR / Mischform)
    """
    from matplotlib.colors import LinearSegmentedColormap

    MONATE = ["Jan","Feb","Mär","Apr","Mai","Jun",
              "Jul","Aug","Sep","Okt","Nov","Dez"]

    # ── Monatliche Fehlrate pro Spalte berechnen ──
    zeilen = []  # [(label, [rate_Jan, ..., rate_Dez]), ...]
    for df_c, num_c, tcol, prefix in datasets:
        df_t = df_c.copy()
        df_t[tcol] = pd.to_datetime(df_t[tcol])
        df_t = df_t.set_index(tcol)
        for col in num_c:
            if "_missing" in col:
                continue
            miss_col = f"{col}_missing"
            if miss_col not in df_c.columns or df_c[miss_col].sum() == 0:
                continue
            monthly = df_t[miss_col].resample("ME").mean()
            monthly.index = monthly.index.month
            row = [float(monthly.get(m, 0.0)) for m in range(1, 13)]
            zeilen.append((f"{prefix}: {col}", row))

    if not zeilen:
        print("  Keine fehlenden Werte – Heatmap entfällt")
        return

    labels = [z[0] for z in zeilen]
    data   = np.array([z[1] for z in zeilen])
    n_r    = len(labels)

    # ── Farbpalette: grün → gelb → rot ──
    cmap = LinearSegmentedColormap.from_list(
        "mcar", ["#1D9E75", "#F4D03F", "#C0392B"]
    )

    fig, (ax_heat, ax_text) = plt.subplots(
        2, 1,
        figsize=(14, max(4, n_r * 0.6) + 5),
        gridspec_kw={"height_ratios": [max(n_r, 3), 5]},
    )

    # ── Heatmap ──
    im = ax_heat.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax_heat, fraction=0.015, pad=0.01,
                 label="Anteil fehlend  (grün = vollständig · rot = alles fehlt)")
    ax_heat.set_xticks(range(12))
    ax_heat.set_xticklabels(MONATE)
    ax_heat.set_yticks(range(n_r))
    ax_heat.set_yticklabels(labels)
    ax_heat.set_title(
        "Monatliche Fehlraten – MCAR/MAR-Analyse\n"
        "Grün = vollständig vorhanden · Rot = vollständig fehlend",
        pad=10
    )
    ax_heat.set_xlabel("Monat")

    # Prozentwerte in die Zellen schreiben
    for i in range(n_r):
        for j in range(12):
            v = data[i, j]
            if v > 0.005:
                ax_heat.text(j, i, f"{v:.0%}",
                             ha="center", va="center", fontsize=8.5,
                             color="white" if v > 0.55 else "black")

    # ── Interpretation berechnen ──
    mar_cols  = []
    mcar_cols = []
    txt_zeilen = [
        "Interpretation der MCAR/MAR-Analyse:",
        f"  {'Spalte':<32} {'Typ':<32} Begründung",
        "  " + "─" * 82,
    ]

    for label, row in zeilen:
        arr        = np.array(row)
        cv         = float(arr.std() / arr.mean()) if arr.mean() > 0 else 0.0
        aktiv      = [MONATE[j] for j in range(12) if arr[j] > 0.05]

        if cv >= 0.5 and len(aktiv) <= 6:
            typ = "MAR  (saisonal geclustert)"
            det = f"Häufung in: {', '.join(aktiv)}"
            mar_cols.append(label)
        elif cv < 0.35:
            typ = "MCAR (gleichmässig verteilt)"
            det = f"CV={cv:.2f} – kein saisonales Muster"
            mcar_cols.append(label)
        else:
            typ = "MCAR/MAR (Mischform)"
            det = f"CV={cv:.2f} – teilw. in: {', '.join(aktiv)}"

        txt_zeilen.append(f"  {label:<32} {typ:<32} {det}")

    txt_zeilen += ["", "  Fazit:"]
    if mar_cols:
        txt_zeilen.append(
            f"  → MAR-Verdacht bei: {', '.join(mar_cols)}")
        txt_zeilen.append(
            "    Saisonale Clusterung → wahrscheinlich Wartungsrhythmen des Messnetzes")
        txt_zeilen.append(
            "    oder saisonaler Messbetrieb. _missing-Indikator im Modell behalten,")
        txt_zeilen.append(
            "    da das Fehlen selbst eine Information ist.")
    if mcar_cols:
        txt_zeilen.append(
            f"  → MCAR-Verdacht bei: {', '.join(mcar_cols)}")
        txt_zeilen.append(
            "    Gleichmässige Verteilung → zufällige Sensorausfälle.")
        txt_zeilen.append(
            "    Lineare Interpolation (≤ 3h) liefert hier unverzerrte Schätzungen.")
    if not mar_cols and not mcar_cols:
        txt_zeilen.append(
            "  → Gemischtes Muster bei allen Spalten (MCAR/MAR).")
        txt_zeilen.append(
            "    _missing-Indikatoren werden im finalen Datensatz behalten.")

    ax_text.axis("off")
    ax_text.text(0.01, 0.98, "\n".join(txt_zeilen),
                 transform=ax_text.transAxes,
                 va="top", ha="left",
                 fontsize=8.5, fontfamily="monospace",
                 linespacing=1.5)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Plot gespeichert: {save_path}")


plot_mcar_heatmap(
    [
        (df_luft_clean,   luft_num_cols,   "timestamp", "Luft"),
        (df_wetter_clean, wetter_num_cols, "Datum",     "Wett."),
    ],
    "output/plots/bereinigung_mcar_analyse.png",
)

# ── 4c. Kurze Lücken interpolieren (≤ 3 Stunden) ──
# Zeitreihen-Interpolation ist besser als Mittelwert-Imputation,
# da sie den zeitlichen Verlauf berücksichtigt.
print("\n  Interpolation kurzer Lücken (≤ 3h)...")

def interpoliere_kurze_luecken(df, num_cols, max_luecke=3, label=""):
    vorher_gesamt = df[num_cols].isnull().sum().sum()
    for col in num_cols:
        if col not in df.columns:
            continue
        # Nur Lücken ≤ max_luecke Stunden interpolieren
        df[col] = df[col].interpolate(
            method="linear",
            limit=max_luecke,          # max. 3 aufeinanderfolgende NaNs
            limit_direction="forward"
        )
    nachher_gesamt = df[num_cols].isnull().sum().sum()
    print(f"  {label}: {vorher_gesamt - nachher_gesamt} Werte interpoliert, "
          f"{nachher_gesamt} NaNs verbleiben")

interpoliere_kurze_luecken(
    df_luft_clean, luft_num_cols, max_luecke=3, label="Luftqualität")
interpoliere_kurze_luecken(
    df_wetter_clean, wetter_num_cols, max_luecke=3, label="Wetter")

# ── 4d. Verbleibende NaNs dokumentieren ──
print("\n  Verbleibende fehlende Werte nach Interpolation:")
for name, df, cols in [("Luftqualität", df_luft_clean,  luft_num_cols),
                        ("Wetter",       df_wetter_clean, wetter_num_cols)]:
    remaining = df[cols].isnull().sum()
    remaining = remaining[remaining > 0]
    if remaining.empty:
        print(f"  {name}: Keine fehlenden Werte mehr!")
    else:
        print(f"  {name}:")
        for col, n in remaining.items():
            pct = n / len(df) * 100
            print(f"    {col:<35} {n:>6} ({pct:.1f}%) → bleibt NaN")
        print(f"    → Strategie: Längere Lücken als NaN belassen.")
        print(f"      Beim Merge werden diese Stunden ausgeschlossen.")


# ─────────────────────────────────────────────────────────────
# 5. AUSREISSER BEHANDELN
# ─────────────────────────────────────────────────────────────
print("\n[5/6] Ausreisser behandeln (Winsorisierung)...")

# Strategie basierend auf EDA:
# - Precipitation: NICHT winsorisieren (physikalisch korrekte Verteilung)
# - Surface pressure: NICHT winsorisieren (Extremwetterlagen sind real)
# - Wind speed: NICHT winsorisieren (Sturmereignisse sind interessant)
# - Andere: Winsorisierung bei >5% Ausreissern erwägen

def winsorisiere(df, col, q_low=0.01, q_high=0.99, label=""):
    """Ersetzt Extremwerte mit den 1. und 99. Perzentile."""
    if col not in df.columns:
        return 0
    s = df[col].dropna()
    p_low  = s.quantile(q_low)
    p_high = s.quantile(q_high)
    maske_low  = df[col] < p_low
    maske_high = df[col] > p_high
    n = maske_low.sum() + maske_high.sum()
    df.loc[maske_low,  col] = p_low
    df.loc[maske_high, col] = p_high
    print(f"  {label} – {col}: {n} Werte winsorisiert "
          f"(Grenzen: {p_low:.2f} – {p_high:.2f})")
    return n

# Nur Schadstoffe winsorisieren wo nötig
# (Wettervariablen behalten wir als echte Extremwerte)
print("\n  Luftqualität (Winsorisierung 1%–99%):")
for col in luft_num_cols:
    if "_missing" in col:
        continue
    n_out = ((df_luft_clean[col] < df_luft_clean[col].quantile(0.01)) |
             (df_luft_clean[col] > df_luft_clean[col].quantile(0.99))).sum()
    pct = n_out / df_luft_clean[col].notna().sum() * 100
    if pct > 3:
        winsorisiere(df_luft_clean, col, label="Luftqualität")
    else:
        print(f"  Luftqualität – {col}: {pct:.1f}% Ausreisser → behalten")

print("\n  Wetter: Ausreisser werden NICHT winsorisiert.")
print("  Begründung: Sturm, Starkregen, Inversionswetterlagen")
print("  sind real und für die Analyse relevant.")


# ─────────────────────────────────────────────────────────────
# 6. BEREINIGTE DATEN SPEICHERN + VERGLEICHSPLOT
# ─────────────────────────────────────────────────────────────
print("\n[6/6] Bereinigte Daten speichern...")

df_luft_clean.to_csv("data/processed/luftqualitaet_2023_clean.csv",
                     index=False)
df_wetter_clean.to_csv("data/processed/wetter_2023_clean.csv",
                       index=False)

print("  Gespeichert:")
print("    data/processed/luftqualitaet_2023_clean.csv")
print("    data/processed/wetter_2023_clean.csv")

# Vorher/Nachher Vergleich
print("\n  Vorher/Nachher Vergleich:")
print(f"  {'Datensatz':<20} {'Zeilen vorher':>14} {'Zeilen nachher':>15} "
      f"{'NaN vorher':>11} {'NaN nachher':>12}")
print(f"  {'-'*74}")

nan_luft_vorher   = df_luft[luft_num_cols].isnull().sum().sum()
nan_luft_nachher  = df_luft_clean[luft_num_cols].isnull().sum().sum()
nan_wett_vorher   = df_wetter[wetter_num_cols].isnull().sum().sum()
nan_wett_nachher  = df_wetter_clean[wetter_num_cols].isnull().sum().sum()

print(f"  {'Luftqualität':<20} {len(df_luft):>14,} "
      f"{len(df_luft_clean):>15,} "
      f"{nan_luft_vorher:>11,} {nan_luft_nachher:>12,}")
print(f"  {'Wetter':<20} {len(df_wetter):>14,} "
      f"{len(df_wetter_clean):>15,} "
      f"{nan_wett_vorher:>11,} {nan_wett_nachher:>12,}")

# Vorher/Nachher Plot (erster Schadstoff)
plot_col = next((c for c in luft_num_cols
                 if "_missing" not in c), None)
if plot_col:
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    for ax, df_plot, titel, farbe in zip(
        axes,
        [df_luft, df_luft_clean],
        [f"{plot_col} – Rohdaten", f"{plot_col} – Bereinigt"],
        ["#D85A30", "#1D9E75"]
    ):
        daily = df_plot.set_index("timestamp")[plot_col].resample("D").median()
        ax.plot(daily.index, daily.values, linewidth=0.8, color=farbe)
        ax.fill_between(daily.index, daily.values, alpha=0.15, color=farbe)
        ax.set_title(titel, fontsize=10)
        ax.set_ylabel(plot_col)
        ax.grid(True, alpha=0.3)
    plt.suptitle("Datenbereinigung – Vorher/Nachher Vergleich", fontsize=11)
    plt.tight_layout()
    plt.savefig("output/plots/bereinigung_vergleich.png", dpi=120)
    plt.close()
    print("\n  Plot gespeichert: output/plots/bereinigung_vergleich.png")


# ─────────────────────────────────────────────────────────────
# ZUSAMMENFASSUNG FÜR DEN BERICHT
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ZUSAMMENFASSUNG – Bereinigungsschritte für den Bericht")
print("=" * 60)
print("""
  Dokumentierte Bereinigungsschritte:

  1. Duplikate entfernen
     - Doppelte Zeitstempel: ersten Eintrag behalten

  2. Physikalisch unmögliche Werte → NaN
     - Negative Schadstoffwerte (z.B. NO2 < 0)
     - Werte ausserhalb plausibler Messbereiche
     - Begründung: Sicher Messfehler, keine echten Daten

  3. Fehlende Werte
     - Indikatorvariablen (_missing) vor Imputation erstellt
     - MCAR/MAR/MNAR-Analyse: Typ basierend auf monatlichem CV
       und maximaler Run-Length bestimmt (siehe Ausgabe oben)
     - Kurze Lücken ≤ 3h: Lineare Interpolation (zeitlich, unverzerrte
       Schätzung bei MCAR; MAR-Typ über _missing-Indikator dokumentiert)
     - Längere Lücken: Als NaN belassen

  4. Ausreisser
     - Luftschadstoffe: Winsorisierung 1%–99% wo nötig
     - Wettervariablen: Keine Winsorisierung
       (Extremwetterereignisse sind für Analyse relevant)

  → Bereinigte Dateien: data/processed/
""")

print("Nächster Schritt: Phase 4 – Datentransformation (W04_transformation.py)")
print("=" * 60)