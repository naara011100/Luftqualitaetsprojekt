# =============================================================
# DWAE Projekt – Phase 1: Datenbeschaffung
# Thema: Einfluss von Wetter auf Luftqualität in Zürich
# =============================================================
# Datenquellen:
#   1. Luftqualität: OGD Stadt Zürich (AWEL Luftmessnetz)
#   2. Wetterdaten:  Open-Meteo API (kostenlos, kein API-Key)
# =============================================================

import requests
import pandas as pd
import json
from pathlib import Path
from io import StringIO

# Ausgabe-Ordner anlegen
Path("data/raw").mkdir(parents=True, exist_ok=True)
Path("data/processed").mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("PHASE 1 – Datenbeschaffung")
print("=" * 60)


# ─────────────────────────────────────────────────────────────
# QUELLE 1: Luftqualität – OGD Stadt Zürich (AWEL)
# ─────────────────────────────────────────────────────────────
# Stündliche Messwerte des Luftmessnetzes:
#   - Station: Zürich Stampfenbachstrasse (städtisch, verkehrsnah)
#   - Schadstoffe: NO, NO2, O3, PM10 (je nach Verfügbarkeit)
#
# Portal: https://data.stadt-zuerich.ch
# Datensatz: "Luftqualität – Stundenwerte"
# Hinweis: Jahresdatei wählen, z.B. ugz_ogd_air_h1_2023.csv
# Falls der Download gesperrt ist: Datei manuell herunterladen
# und im Ordner data/raw/ ablegen.

LUFT_URL = (
    "https://data.stadt-zuerich.ch/dataset/"
    "ugz_luftschadstoffmessung_stundenwerte/download/"
    "ugz_ogd_air_h1_2023.csv"
)

print("\n[1/2] Lade Luftqualitätsdaten (AWEL / OGD Stadt Zürich)...")

try:
    headers = {"User-Agent": "Mozilla/5.0 (research project, FHNW)"}
    response = requests.get(LUFT_URL, timeout=30, headers=headers)
    response.raise_for_status()
    df_luft_raw = pd.read_csv(StringIO(response.text))
    print(f"      OK: {df_luft_raw.shape[0]:,} Zeilen, "
          f"{df_luft_raw.shape[1]} Spalten")

except requests.exceptions.RequestException as e:
    print(f"      Direkter Download nicht möglich: {e}")
    print("      → Bitte Datei manuell herunterladen:")
    print("        https://data.stadt-zuerich.ch (Suche: Luftqualität Stundenwerte)")
    print("        Speichern als: data/raw/ugz_ogd_air_h1_2023.csv")
    # Lokale Datei laden falls vorhanden
    local_path = Path("data/raw/ugz_ogd_air_h1_2023.csv")
    if local_path.exists():
        df_luft_raw = pd.read_csv(local_path)
        print(f"      Lokale Datei geladen: {df_luft_raw.shape}")
    else:
        df_luft_raw = None

if df_luft_raw is not None:
    # Rohdaten speichern
    df_luft_raw.to_csv("data/raw/luftqualitaet_2023_roh.csv", index=False)
    print(f"      Spalten: {list(df_luft_raw.columns)}")

    # Erste Übersicht
    print(f"\n      Fehlende Werte pro Spalte:")
    for col in df_luft_raw.columns:
        n = df_luft_raw[col].isnull().sum()
        pct = n / len(df_luft_raw) * 100
        print(f"        {col:<35} {n:>6} ({pct:.1f}%)")

    print(f"\n      Erste 3 Zeilen:")
    print(df_luft_raw.head(3).to_string())


# ─────────────────────────────────────────────────────────────
# QUELLE 2: Wetterdaten – Open-Meteo API
# ─────────────────────────────────────────────────────────────
# Dokumentation: https://open-meteo.com/en/docs/historical-weather-api
# Kein API-Key nötig, kostenlos für nicht-kommerzielle Nutzung
# Koordinaten Zürich: Lat 47.3769, Lon 8.5417
# Zeitzone: Europe/Zurich (wichtig für korrekten Zeitstempel-Match)

WETTER_URL = "https://archive-api.open-meteo.com/v1/archive"

WETTER_PARAMS = {
    "latitude": 47.3769,
    "longitude": 8.5417,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "hourly": [
        "temperature_2m",        # °C      – Hauptvariable
        "relative_humidity_2m",  # %       – Einfluss auf Partikelkonzentration
        "precipitation",         # mm      – Auswascheffekte auf PM10
        "wind_speed_10m",        # km/h    – Dispersion von Schadstoffen
        "wind_direction_10m",    # °       – Herkunft der Luftmassen
        "surface_pressure",      # hPa     – Inversionslagen
        "cloud_cover",           # %       – Strahlungseinfluss auf O3-Bildung
        "sunshine_duration",     # s/h     – Photochemie (Ozonbildung)
    ],
    "timezone": "Europe/Zurich",
}

print("\n[2/2] Lade Wetterdaten (Open-Meteo API)...")
print(f"      Zeitraum: {WETTER_PARAMS['start_date']} "
      f"bis {WETTER_PARAMS['end_date']}")

try:
    response = requests.get(WETTER_URL, params=WETTER_PARAMS, timeout=60)
    response.raise_for_status()
    wetter_json = response.json()

    # JSON als Rohdaten sichern (immer gut für Reproduzierbarkeit)
    with open("data/raw/wetter_2023_roh.json", "w") as f:
        json.dump(wetter_json, f, indent=2)

    # In DataFrame umwandeln
    df_wetter_raw = pd.DataFrame(wetter_json["hourly"])
    df_wetter_raw.rename(columns={"time": "Datum"}, inplace=True)

    # Als CSV speichern
    df_wetter_raw.to_csv("data/raw/wetter_2023_roh.csv", index=False)

    print(f"      OK: {df_wetter_raw.shape[0]:,} Zeilen, "
          f"{df_wetter_raw.shape[1]} Spalten")
    print(f"      Zeitraum: {df_wetter_raw['Datum'].min()} "
          f"bis {df_wetter_raw['Datum'].max()}")

    # Einheiten dokumentieren (wichtig für Bericht)
    einheiten = {
        "temperature_2m":        "°C",
        "relative_humidity_2m":  "%",
        "precipitation":         "mm",
        "wind_speed_10m":        "km/h",
        "wind_direction_10m":    "° (meteorologisch, 0=N, 90=E)",
        "surface_pressure":      "hPa",
        "cloud_cover":           "%",
        "sunshine_duration":     "Sekunden pro Stunde (max. 3600)",
    }
    print(f"\n      Variablen und Einheiten:")
    for var, einheit in einheiten.items():
        nulls = df_wetter_raw[var].isnull().sum()
        print(f"        {var:<30} {einheit:<35} fehlend: {nulls}")

    print(f"\n      Erste 3 Zeilen:")
    print(df_wetter_raw.head(3).to_string())

except requests.exceptions.RequestException as e:
    print(f"      FEHLER: {e}")
    df_wetter_raw = None


# ─────────────────────────────────────────────────────────────
# ZUSAMMENFASSUNG FÜR DEN BERICHT
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ZUSAMMENFASSUNG – Datenbeschaffung")
print("=" * 60)

quellen = {
    "Luftqualität (AWEL)": {
        "df": df_luft_raw,
        "Quelle": "OGD Stadt Zürich",
        "Lizenz": "Open Government Data (OGD)",
        "Format": "CSV, stündlich",
        "URL": "data.stadt-zuerich.ch",
    },
    "Wetter (Open-Meteo)": {
        "df": df_wetter_raw,
        "Quelle": "Open-Meteo Historical API",
        "Lizenz": "CC BY 4.0 (nicht-kommerziell kostenlos)",
        "Format": "JSON → CSV, stündlich",
        "URL": "archive-api.open-meteo.com",
    },
}

for name, info in quellen.items():
    df = info.pop("df")
    print(f"\n  {name}:")
    for k, v in info.items():
        print(f"    {k:<12}: {v}")
    if df is not None:
        print(f"    Shape     : {df.shape[0]:,} Zeilen × {df.shape[1]} Spalten")
        print(f"    Fehlend   : {df.isnull().sum().sum()} Werte total")
    else:
        print(f"    Status    : Nicht verfügbar – manuell herunterladen")

print("\n" + "=" * 60)
print("Gespeicherte Dateien:")
for f in sorted(Path("data/raw").glob("*")):
    size = f.stat().st_size / 1024
    print(f"  {f}  ({size:.1f} KB)")
print("\nNächster Schritt: Phase 2 – EDA (W02_eda.py)")
print("=" * 60)