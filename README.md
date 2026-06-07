# Einfluss von Wetter auf Luftqualität in Zürich

**DWAE Projekt – Daten Wrangling & Analyse Engineering**

| Eigenschaft | Wert |
|-------------|------|
| Station | Zürich Stampfenbachstrasse (AWEL Luftmessnetz) |
| Zeitraum | 2023-01-01 bis 2023-12-31 (8 760 Stunden) |
| Schadstoffe | NO, NO2, O3, PM10, PM2.5 |
| Python | 3.13 · Windows 11 |

---

## Projektbeschreibung

Dieses Projekt untersucht den Zusammenhang zwischen Wettervariablen und Luftschadstoffkonzentrationen in Zürich. Die vollautomatisierte Pipeline deckt alle Phasen von der Datenbeschaffung bis zur statistischen Auswertung ab.

**Zentrale Fragestellung:** Können Wettervariablen (Niederschlag, Temperatur, Wind, Inversionslagen) die stündlichen Schadstoffkonzentrationen statistisch erklären und vorhersagen?

---

## Projektstruktur

```
Luftqualitaetsprojekt/
│
├── datenbeschaffung.py          # Phase 1 – Download + Long→Wide Pivot
├── eda.py                       # Phase 2 – Explorative Datenanalyse (EDA)
├── bereinigung.py               # Phase 3 – Bereinigung + MCAR/MAR-Analyse
├── transformation.py            # Phase 4 – Feature Engineering & Merge
├── pipeline.py                  # Phase 5 – Qualitätsprüfung (4 Dimensionen)
├── visualisierung.py            # Phase 6 – 9 Visualisierungen + Hypothesentest
├── bonus_modell.py              # Bonus   – ML-Modell (3 Algorithmen)
├── total pipeline,py            # Phase 7 – Vollautomatische Pipeline (1–6)
│
├── plot_config.py               # Globales Matplotlib-Styling & Hilfsfunktionen
├── hypothesen.py                # Forschungshypothesen H1–H4
│
├── requirements.txt             # Python-Abhängigkeiten
├── README.md                    # Diese Datei
│
├── data/
│   ├── raw/
│   │   ├── luftqualitaet_2023_roh.csv       # Wide-Format (nach Pivot)
│   │   ├── luftqualitaet_2023_roh_long.csv  # Long-Format Backup (OGD Original)
│   │   ├── wetter_2023_roh.csv              # Open-Meteo Wetterdaten
│   │   └── wetter_2023_roh.json             # Open-Meteo Rohantwort
│   └── processed/
│       ├── luftqualitaet_2023_clean.csv     # Bereinigt (Phase 3)
│       ├── wetter_2023_clean.csv            # Bereinigt (Phase 3)
│       └── datensatz_final.csv              # Fertig (Phase 4, 53 Spalten)
│
└── output/
    ├── plots/                               # 15 erzeugte Grafiken
    │   ├── bereinigung_mcar_analyse.png     # MCAR/MAR-Heatmap
    │   ├── bereinigung_vergleich.png        # Vorher/Nachher Bereinigung
    │   ├── eda_fehlende_werte_heatmap.png
    │   ├── eda_zeitreihen_wetter.png
    │   ├── transformation_korrelation.png
    │   ├── transformation_scatter_temp.png
    │   ├── transformation_zeitprofile.png
    │   ├── viz_01_jahresuebersicht.png      # Dashboard mit Rolling Mean
    │   ├── viz_02_korrelation.png           # Pearson + Spearman
    │   ├── viz_03_jahreszeit_boxplot.png
    │   ├── viz_04_scatter_temperatur.png
    │   ├── viz_05_regen_effekt.png
    │   ├── viz_06_inversion.png
    │   ├── viz_07_zeitprofile.png           # Rush-Hour-Muster
    │   ├── viz_08_windrose.png              # Polar-Plot
    │   └── viz_09_lag_analyse.png           # Lag 0–6h
    └── qualitaet/
        └── qualitaetsbericht.json
```

---

## Datenquellen

| Quelle | Beschreibung | Lizenz |
|--------|-------------|--------|
| **OGD Stadt Zürich** | Stündliche Luftqualitätsmessungen (AWEL), Station Stampfenbachstrasse, Long-Format CSV mit BOM | Open Government Data |
| **Open-Meteo Archive API** | Stündliche Wetterdaten: Temperatur, Niederschlag, Wind, Druck, Strahlung | CC BY 4.0 |

---

## Installation

```bash
pip install -r requirements.txt
```

| Paket | Version (getestet) | Zweck |
|-------|-------------------|-------|
| pandas | 2.3.3 | Datenverarbeitung, Pivot, Zeitreihen |
| numpy | 2.2.0 | Numerische Berechnungen |
| matplotlib | 3.10.7 | Visualisierungen (inkl. Polar-Plot) |
| scikit-learn | 1.7.2 | ML-Pipeline, Skalierung, Modelle |
| scipy | 1.16.2 | Statistiktests (Pearson, Spearman, t-Test) |
| requests | 2.32.5 | API-Zugriff (Open-Meteo) |

---

## Pipeline ausführen

### Einzelne Phasen (empfohlen für Entwicklung)

```bash
python datenbeschaffung.py    # Phase 1 – Download + Long→Wide Pivot
python eda.py                 # Phase 2 – Explorative Datenanalyse
python bereinigung.py         # Phase 3 – Bereinigung + MCAR/MAR-Heatmap
python transformation.py      # Phase 4 – Feature Engineering
python pipeline.py            # Phase 5 – Qualitätsprüfung
python visualisierung.py      # Phase 6 – 9 Plots + Hypothesentest
python bonus_modell.py        # Bonus   – ML-Modell
```

### Vollautomatisch (alle Phasen in einem Schritt)

```bash
python "total pipeline,py"                     # Phasen 1–6 komplett
python "total pipeline,py" --ab-phase 3        # Ab Phase 3 starten
python "total pipeline,py" --nur-phase 6       # Nur Phase 6
```

---

## Methodik & Features

### Phase 1 – Datenbeschaffung (`datenbeschaffung.py`)
- BOM-Bereinigung via `encoding="utf-8-sig"` (OGD-CSVs haben UTF-8-BOM)
- **Long→Wide Pivot:** OGD-Rohdaten kommen im Langformat (eine Zeile pro Stunde und Schadstoff). Der Pivot erzeugt das Breitformat: eine Zeile pro Stunde, je eine Spalte pro Schadstoff
- Zeitstempel in UTC flooren → Timezone entfernen (DST-sicher, kein `AmbiguousTimeError`)
- Backup des Long-Format-Originals unter `luftqualitaet_2023_roh_long.csv`

### Phase 3 – Bereinigung (`bereinigung.py`)
- Duplikate entfernen, physikalisch unmögliche Werte → NaN
- **MCAR/MAR/MNAR-Analyse** (neu):
  - Monatlicher Variationskoeffizient (CV) der Fehlrate pro Spalte
  - Maximale zusammenhängende Fehlstrecke (Run-Length)
  - Automatische Klassifikation: MCAR / MAR / Mischform
  - Speichert Heatmap: `output/plots/bereinigung_mcar_analyse.png`
- Kurze Lücken ≤ 3h: lineare Zeitreiheninterpolation
- Indikatorvariablen (`_missing`) vor der Imputation erstellt

### Phase 4 – Transformation (`transformation.py`)
- Merge Luft + Wetter über `timestamp` (Inner Join)
- **Zeitfeatures:** Stunde, Wochentag, Monat, Jahreszeit, Wochenende, Rush-Hour (7–9h, 17–19h)
- **Wetterfeatures:** Temperaturkategorien, Regen-Indikator (> 0.1 mm/h), Windkategorien (Beaufort vereinfacht)
- **Inversions-Indikator:** Luftdruck > 75. Perzentil UND Temperatur < 25. Perzentil
- Skalierung: StandardScaler (→ `_std`) + MinMaxScaler (→ `_norm`)
- Finaler Datensatz: 8 760 Zeilen × 53 Spalten

### Phase 6 – Visualisierungen (`visualisierung.py`)

| Plot | Inhalt | Besonderheit |
|------|--------|-------------|
| viz_01 | Jahresübersicht Dashboard | 7-Tage Rolling Mean auf allen Panels |
| viz_02 | Korrelationsmatrizen | Pearson + Spearman nebeneinander |
| viz_03 | Saisonale Boxplots | WHO-Grenzwerte als gestrichelte Linie |
| viz_04 | Scatter Temperatur vs. Schadstoffe | Polynomfit 2. Grades je Jahreszeit |
| viz_05 | Regen-Effekt | Boxplot + Scatter mit Korrelation |
| viz_06 | Inversionslagen-Analyse | Histogramm + monatliche Häufigkeit |
| viz_07 | Tages- und Wochenprofil | Rush-Hour-Muster sichtbar |
| viz_08 | Windrose (Polar-Plot) | 22.5°-Sektoren, Schadstoffintensität |
| viz_09 | Lag-Analyse Niederschlag | Pearson + Spearman für Lag 0–6h |

**Gemeinsames Styling (`plot_config.py`):**
- Titel 12 px, Achsenlabels 10 px, Ticks 9 px, `dpi=150`
- `FIG_KLEIN = (14, 6)`, `FIG_GROSS = (16, 10)`
- `finde_schadstoff_cols(df)` – erkennt NO, NO2, O3, PM10, PM2.5 und PM25
- `who_linie(ax, col)` – WHO-Grenzwert als gestrichelte rote Linie
- `suptitel_stats(df, col)` – befüllt Suptitle automatisch mit n, Zeitraum, Mittelwert

**Physikalische Erklärungen (`ERKLAERUNGEN`-Dict in visualisierung.py):**
Nach jedem relevanten Plot wird der physikalische Mechanismus erklärt und im Terminal ausgegeben (für direktes Copy-Paste in den Bericht).

### Hypothesentest (`hypothesen.py` + `visualisierung.py`)

| ID | Hypothese | Methode | Ergebnis |
|----|-----------|---------|----------|
| H1 | Niederschlag reduziert PM10 (Auswaschung) | Pearson + Spearman | ✓ bestätigt |
| H2 | Inversionslagen erhöhen NO-Konzentration | Welch-t-Test + Cohen's d | ✓ bestätigt |
| H3 | Hohe Temperaturen erhöhen O3 (Photochemie) | Pearson + Spearman | ✓ bestätigt |
| H4 | Rush-Hour-Stunden zeigen erhöhte NO2-Werte | Welch-t-Test + Cohen's d | ✓ bestätigt |

Entscheidungslogik:
- **✓ bestätigt**: korrekte Richtung + p < 0.05 + |r| ≥ 0.10 (Korrelation) bzw. |d| ≥ 0.20 (Gruppenvergleich)
- **~ teilweise**: korrekte Richtung + signifikant, aber schwacher Effekt
- **✗ nicht bestätigt**: falsche Richtung

### Bonus – ML-Modell (`bonus_modell.py`)

Drei Modelle im Vergleich (alle mit `sklearn.Pipeline` + `StandardScaler`):

| Modell | Beschreibung |
|--------|-------------|
| Lineare Regression | Baseline |
| Ridge Regression | Regularisierung (alpha=1.0) |
| Random Forest | 100 Trees, max_depth=10, n_jobs=-1 |

Features: 15 Wetter- und Zeitvariablen (kein Datenleck – andere Schadstoffe ausgeschlossen)
Evaluation: R², MAE, RMSE, 5-fold Cross-Validation, Feature Importance

---

## WHO-Grenzwerte (2021)

| Schadstoff | Grenzwert | Einheit | Typ |
|-----------|-----------|---------|-----|
| NO2 | 40 | µg/m³ | Jahresmittel |
| PM10 | 45 | µg/m³ | 24h-Mittel |
| O3 | 100 | µg/m³ | 8h-Mittel (max.) |
| PM2.5 | 15 | µg/m³ | Jahresmittel |

---

## Datensatz-Statistiken (2023, Stampfenbachstrasse)

| Schadstoff | Mittelwert | Min | Max | Einheit |
|-----------|-----------|-----|-----|---------|
| NO | 10.1 | 0.4 | 104.2 | µg/m³ |
| NO2 | 21.4 | 1.9 | 77.0 | µg/m³ |
| O3 | 53.6 | 0.8 | 165.2 | µg/m³ |
| PM10 | 13.4 | 2.0 | 86.1 | µg/m³ |
| PM2.5 | 9.0 | 1.1 | 53.3 | µg/m³ |

**Saisonale NO-Mittelwerte:**

| Jahreszeit | Mittelwert |
|-----------|-----------|
| Winter | 14.7 µg/m³ |
| Frühling | 8.2 µg/m³ |
| Sommer | 6.1 µg/m³ |
| Herbst | 11.6 µg/m³ |

**Stärkste Wetterkorrelation mit NO:** Windgeschwindigkeit (r ≈ −0.38) – hoher Wind dispergiert Schadstoffe.

---

## Bekannte Einschränkungen

- Nur eine Messstation (Stampfenbachstrasse, verkehrsnah) – kein räumliches Bild der Stadt
- OGD-Rohdaten: Long-Format mit BOM, muss mit `encoding="utf-8-sig"` gelesen werden
- Zeitstempel in UTC flooren (nicht in Europe/Zurich), um `AmbiguousTimeError` an DST-Übergängen zu vermeiden
- Windows-Terminal: `sys.stdout.reconfigure(encoding="utf-8")` in allen Skripten erforderlich
