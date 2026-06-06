# =============================================================
# DWAE Projekt – Phase 5: Qualitätsprüfung
# Thema: Einfluss von Wetter auf Luftqualität in Zürich
# =============================================================
# Voraussetzung: Phase 4 abgeschlossen
# Ausführen:     python W05_pipeline.py
# =============================================================

import sys
import pandas as pd
import numpy as np
import json
import hashlib
from pathlib import Path
from datetime import datetime
from plot_config import finde_schadstoff_cols

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

Path("output/qualitaet").mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("PHASE 5 – Qualitätsprüfung")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# DATEN LADEN
# ─────────────────────────────────────────────────────────────
df = pd.read_csv("data/processed/datensatz_final.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(None)

print(f"\n  Datensatz: {df.shape[0]:,} Zeilen × {df.shape[1]} Spalten")

# Nur Original-Spalten (keine skalierten / missing-Indikatoren)
basis_cols = [c for c in df.columns
              if "_std" not in c
              and "_norm" not in c
              and "_missing" not in c
              and c != "timestamp"]

schadstoff_cols = finde_schadstoff_cols(df)

wetter_cols = [c for c in [
    "temperature_2m", "precipitation", "wind_speed_10m",
    "relative_humidity_2m", "surface_pressure", "sonnenschein_h"]
    if c in df.columns]


# ─────────────────────────────────────────────────────────────
# QUALITÄTSDIMENSIONEN (aus W01 Unterlagen)
# Geprüft werden alle 4 Dimensionen:
# System / Inhalt / Nutzung / Darstellung
# ─────────────────────────────────────────────────────────────

qualitaet_bericht = {
    "erstellt_am": datetime.now().isoformat(),
    "datensatz":   "datensatz_final.csv",
    "zeilen":      int(len(df)),
    "spalten":     int(df.shape[1]),
    "pruefungen":  {}
}

print("\n" + "─" * 60)
print("DIMENSION 1 – SYSTEM (Zugänglichkeit & Belastbarkeit)")
print("─" * 60)

# Vollständigkeit des Zeitindex
expected_hours = pd.date_range(
    start=df["timestamp"].min(),
    end=df["timestamp"].max(),
    freq="h", tz=df["timestamp"].dt.tz
)
tatsaechliche = set(df["timestamp"])
fehlende_ts   = [t for t in expected_hours if t not in tatsaechliche]
vollstaendigkeit_ts = (1 - len(fehlende_ts) / len(expected_hours)) * 100

print(f"\n  Erwartete Stunden : {len(expected_hours):,}")
print(f"  Vorhandene Stunden: {len(df):,}")
print(f"  Fehlende Stunden  : {len(fehlende_ts):,}")
print(f"  Zeitvollständigkeit: {vollstaendigkeit_ts:.2f}%")
status = "✓ OK" if vollstaendigkeit_ts >= 95 else "✗ PRÜFEN"
print(f"  Status: {status} (Schwellenwert: ≥ 95%)")

qualitaet_bericht["pruefungen"]["zeitvollstaendigkeit_pct"] = \
    round(vollstaendigkeit_ts, 2)

# Dateigrösse & Lesbarkeit
dateipfad = Path("data/processed/datensatz_final.csv")
groesse_mb = dateipfad.stat().st_size / 1e6

# MD5-Checksum für Reproduzierbarkeit
with open(dateipfad, "rb") as f:
    checksum = hashlib.md5(f.read()).hexdigest()

print(f"\n  Dateigrösse: {groesse_mb:.2f} MB")
print(f"  MD5-Checksum: {checksum}")
print(f"  (Checksum im Bericht dokumentieren → Reproduzierbarkeit)")
qualitaet_bericht["pruefungen"]["md5_checksum"] = checksum
qualitaet_bericht["pruefungen"]["groesse_mb"]   = round(groesse_mb, 2)


print("\n" + "─" * 60)
print("DIMENSION 2 – INHALT (Fehlerfrei, Objektiv, Glaubwürdig)")
print("─" * 60)

# Fehlende Werte pro Spalte
print("\n  Fehlende Werte (Originalspalten):")
print(f"  {'Spalte':<35} {'Fehlend':>8} {'%':>8} {'Status':>10}")
print(f"  {'-'*65}")

nan_bericht = {}
for col in schadstoff_cols + wetter_cols:
    if col not in df.columns:
        continue
    n    = df[col].isnull().sum()
    pct  = n / len(df) * 100
    stat = "✓" if pct < 5 else ("△" if pct < 20 else "✗")
    print(f"  {col:<35} {n:>8,} {pct:>7.1f}%  {stat:>10}")
    nan_bericht[col] = {"fehlend": int(n), "prozent": round(pct, 2)}

qualitaet_bericht["pruefungen"]["fehlende_werte"] = nan_bericht

# Plausibilitätsprüfung: Wertebereiche
print("\n  Plausibilitätsprüfung (physikalische Grenzen):")
grenzen = {
    "temperature_2m":       (-30, 45),
    "relative_humidity_2m": (0, 100),
    "precipitation":        (0, 150),
    "wind_speed_10m":       (0, 200),
    "surface_pressure":     (920, 1060),
    "cloud_cover":          (0, 100),
}
print(f"  {'Spalte':<35} {'Min IST':>10} {'Max IST':>10} "
      f"{'Min SOLL':>10} {'Max SOLL':>10} {'Status':>8}")
print(f"  {'-'*85}")

plaus_bericht = {}
for col, (soll_min, soll_max) in grenzen.items():
    if col not in df.columns:
        continue
    ist_min = df[col].min()
    ist_max = df[col].max()
    ok = ist_min >= soll_min and ist_max <= soll_max
    stat = "✓" if ok else "✗ FEHLER"
    print(f"  {col:<35} {ist_min:>10.2f} {ist_max:>10.2f} "
          f"{soll_min:>10} {soll_max:>10}  {stat}")
    plaus_bericht[col] = {
        "ist_min": round(float(ist_min), 2),
        "ist_max": round(float(ist_max), 2),
        "ok": bool(ok)
    }

qualitaet_bericht["pruefungen"]["plausibilitaet"] = plaus_bericht


print("\n" + "─" * 60)
print("DIMENSION 3 – NUTZUNG (Vollständig, Relevant, Aktuell)")
print("─" * 60)

# Vollständigkeit Merge
n_luft_roh   = len(pd.read_csv("data/raw/luftqualitaet_2023_roh.csv"))
merge_quote  = len(df) / n_luft_roh * 100
print(f"\n  Luftqualität Rohdaten : {n_luft_roh:,} Stunden")
print(f"  Finaler Datensatz     : {len(df):,} Stunden")
print(f"  Merge-Quote           : {merge_quote:.1f}%")
stat = "✓ OK" if merge_quote >= 80 else "✗ Merge-Problem prüfen"
print(f"  Status: {stat}")
qualitaet_bericht["pruefungen"]["merge_quote_pct"] = round(merge_quote, 2)

# Relevanz: Korrelationen zur Zielvariable
if schadstoff_cols and wetter_cols:
    print("\n  Relevanz der Wettervariablen (Korrelation mit Schadstoffen):")
    ziel = schadstoff_cols[0]
    print(f"  Zielvariable: {ziel}")
    korr = df[wetter_cols + [ziel]].corr()[ziel].drop(ziel)
    korr_sortiert = korr.abs().sort_values(ascending=False)
    for var in korr_sortiert.index:
        r = korr[var]
        balken = "█" * int(abs(r) * 20)
        print(f"  {var:<35} {r:>+.3f}  {balken}")

# Aktualität
print(f"\n  Zeitraum: 2023 (abgeschlossenes Kalenderjahr)")
print(f"  Datenstand Luftqualität: OGD Stadt Zürich 2023")
print(f"  Datenstand Wetter:       Open-Meteo Archive API")
qualitaet_bericht["pruefungen"]["zeitraum"] = "2023-01-01 bis 2023-12-31"


print("\n" + "─" * 60)
print("DIMENSION 4 – DARSTELLUNG (Einheitlich, Verständlich)")
print("─" * 60)

# Datentypen prüfen
print("\n  Datentypen im finalen Datensatz:")
dtype_probleme = []
for col in df.columns[:20]:  # erste 20 Spalten
    dtype = str(df[col].dtype)
    print(f"  {col:<40} {dtype}")
    if dtype == "object" and col != "timestamp" \
       and "kategorie" not in col and "jahreszeit" not in col:
        dtype_probleme.append(col)

if dtype_probleme:
    print(f"\n  WARNUNG: Folgende Spalten sind 'object' – prüfen:")
    for c in dtype_probleme:
        print(f"    {c}")
else:
    print("\n  ✓ Alle numerischen Spalten korrekt typisiert")

# Einheitliche Benennung prüfen
print("\n  Spaltenbenennungs-Konvention:")
snake_case_ok = all(
    c == c.lower() or c == "timestamp"
    for c in df.columns
    if not any(x in c for x in ["NO", "PM"])
)
print(f"  Snake_case konsistent: {'✓' if snake_case_ok else '✗'}")
qualitaet_bericht["pruefungen"]["snake_case_ok"] = snake_case_ok


# ─────────────────────────────────────────────────────────────
# QUALITÄTSBERICHT SPEICHERN
# ─────────────────────────────────────────────────────────────
bericht_pfad = "output/qualitaet/qualitaetsbericht.json"
with open(bericht_pfad, "w", encoding="utf-8") as f:
    json.dump(qualitaet_bericht, f, indent=2, ensure_ascii=False)

print(f"\n{'=' * 60}")
print(f"QUALITÄTSBERICHT GESPEICHERT")
print(f"{'=' * 60}")
print(f"  {bericht_pfad}")
print(f"\n  Gesamtbewertung:")

checks = {
    "Zeitvollständigkeit":
        qualitaet_bericht["pruefungen"]["zeitvollstaendigkeit_pct"] >= 95,
    "Plausibilität":
        all(v["ok"] for v in plaus_bericht.values()),
    "Merge-Quote":
        qualitaet_bericht["pruefungen"]["merge_quote_pct"] >= 80,
    "Datentypen":
        len(dtype_probleme) == 0,
}
for check, ok in checks.items():
    print(f"  {'✓' if ok else '✗'} {check}")

bestanden = sum(checks.values())
print(f"\n  {bestanden}/{len(checks)} Qualitätsprüfungen bestanden")
print(f"\nNächster Schritt: Phase 6 – Visualisierung (W06_visualisierung.py)")
print("=" * 60)