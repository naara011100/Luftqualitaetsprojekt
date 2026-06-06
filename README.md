# Einfluss von Wetter auf Luftqualität in Zürich
**DWAE Projekt – Daten Wrangling & Analyse Engineering**
Station: Zürich Stampfenbachstrasse (AWEL Luftmessnetz) | Zeitraum: 2023

---

## Projektbeschreibung
Dieses Projekt untersucht den Zusammenhang zwischen Wettervariablen und Luftschadstoffkonzentrationen (NO, NO2, O3, PM10, PM2.5) in Zürich. Die vollautomatisierte Pipeline deckt alle Phasen von der Datenbeschaffung bis zur statistischen Auswertung ab.

**Zentrale Fragestellung:** Können Wettervariablen (Niederschlag, Temperatur, Windgeschwindigkeit, Inversionslagen) die stündlichen Schadstoffkonzentrationen erklären?

---

## Projektstruktur

```
Luftqualitaetsprojekt/
├── datenbeschaffung.py       # Phase 1 – Download + Long→Wide Pivot
├── eda.py                    # Phase 2 – Explorative Datenanalyse
├── bereinigung.py            # Phase 3 – Bereinigung + MCAR/MAR-Analyse
├── transformation.py         # Phase 4 – Feature Engineering & Merge
├── pipeline.py               # Phase 5 – Qualitätsprüfung
├── visualisierung.py         # Phase 6 – Visualisierung (9 Plots)
├── bonus_modell.py           # Bonus  – ML-Modell (Linear/Ridge/RandomForest)
├── total pipeline,py         # Phase 7 – Vollautomatische Pipeline (Phasen 1–6)
├── plot_config.py            # Globales Matplotlib-Styling & Hilfsfunktionen
├── hypothesen.py             # Forschungshypothesen H1–H4
├── requirements.txt          # Python-Abhängigkeiten
├── data/
│   ├── raw/                  # Rohdaten (OGD + Open-Meteo)
│   └── processed/            # Bereinigte und zusammengeführte Daten
└── output/
    ├── plots/                # Alle erzeugten Grafiken (9 Visualisierungen)
    └── qualitaet/            # Qualitätsbericht (JSON)
```

---

## Datenquellen

| Quelle | Beschreibung | Lizenz |
|--------|-------------|--------|
| **OGD Stadt Zürich** | Stündliche Luftqualitätsmessungen (AWEL), Station Stampfenbachstrasse | Open Government Data |
| **Open-Meteo Archive API** | Stündliche Wetterdaten (Temperatur, Niederschlag, Wind, Druck, Strahlung) | CC BY 4.0 |

---

## Installation

```bash
pip install -r requirements.txt
```

**Abhängigkeiten:** pandas>=2.0, numpy>=1.24, matplotlib>=3.7, scikit-learn>=1.3, scipy>=1.10, requests>=2.28

---

## Pipeline ausführen

### Einzelne Phasen (empfohlen für Entwicklung)
```bash
python datenbeschaffung.py    # Phase 1: Download + Pivot
python eda.py                 # Phase 2: EDA (optional)
python bereinigung.py         # Phase 3: Bereinigung
python transformation.py      # Phase 4: Feature Engineering
python pipeline.py            # Phase 5: Qualitätsprüfung
python visualisierung.py      # Phase 6: Visualisierung + Hypothesentest
python bonus_modell.py        # Bonus:   ML-Modell
```

### Vollautomatisch (alle Phasen in einem Schritt)
```bash
python "total pipeline,py"
python "total pipeline,py" --ab-phase 3    # ab Phase 3 starten
python "total pipeline,py" --nur-phase 6   # nur Phase 6
```

---

## Features & Methodik

### Datenaufbereitung
- **Long→Wide Pivot:** OGD-Rohdaten liegen im Langformat vor (eine Zeile pro Stunde und Schadstoff). Der Pivot erzeugt das Breitformat (eine Zeile pro Stunde, eine Spalte pro Schadstoff).
- **MCAR/MAR-Analyse:** Fehlende Werte werden anhand von monatlichem Variationskoeffizient (CV) und maximaler Run-Length klassifiziert.
- **Zeitreiheninterpolation:** Lücken ≤ 3h werden linear interpoliert; längere Lücken bleiben als NaN.

### Feature Engineering
- Zeitfeatures: Stunde, Wochentag, Monat, Jahreszeit, Wochenende, Rush-Hour (7–9h, 17–19h)
- Wetterfeatures: Temperaturkategorien, Regen-Indikator (> 0.1 mm/h), Windkategorien
- Inversions-Indikator: hoher Luftdruck + tiefe Temperatur = schlechte Durchmischung

### Visualisierungen (9 Plots)
| Plot | Inhalt |
|------|--------|
| viz_01 | Jahresübersicht Dashboard (7-Tage Rolling Mean) |
| viz_02 | Korrelationsmatrizen: Pearson + Spearman nebeneinander |
| viz_03 | Saisonale Boxplots mit WHO-Grenzwerten |
| viz_04 | Scatter Temperatur vs. Schadstoffe (Poly2-Trend) |
| viz_05 | Regen-Effekt: kein Regen vs. Regen |
| viz_06 | Inversionslagen-Analyse (Histogramm + Monatshäufigkeit) |
| viz_07 | Tages- und Wochenprofil (Rush-Hour-Muster) |
| viz_08 | Windrose (Polar-Plot, 22.5°-Sektoren) |
| viz_09 | Lag-Analyse Niederschlag → Schadstoff (0–6h) |

### Hypothesentest (Kriterium A)
| ID | Hypothese | Methode |
|----|-----------|---------|
| H1 | Niederschlag reduziert PM10 (Auswaschung) | Pearson + Spearman |
| H2 | Inversionslagen erhöhen NO-Konzentration | Welch-t-Test + Cohen's d |
| H3 | Hohe Temperaturen erhöhen O3 (Photochemie) | Pearson + Spearman |
| H4 | Rush-Hour-Stunden zeigen erhöhte NO2-Werte | Welch-t-Test + Cohen's d |

### ML-Modell (Bonus)
Drei Modelle im Vergleich (alle mit sklearn Pipeline + StandardScaler):
- Lineare Regression
- Ridge Regression (alpha=1.0)
- Random Forest (100 Trees, max_depth=10)

Ausgabe: R², MAE, RMSE, 5-fold Cross-Validation, Feature Importance

---

## WHO-Grenzwerte (2021)
| Schadstoff | Grenzwert | Einheit |
|-----------|-----------|---------|
| NO2 | 40 | µg/m³ (Jahresmittel) |
| PM10 | 45 | µg/m³ (24h-Mittel) |
| O3 | 100 | µg/m³ (8h-Mittel) |
| PM2.5 | 15 | µg/m³ (Jahresmittel) |

---

## Ergebnisse (2023, Stampfenbachstrasse)

**Saisonale NO-Mittelwerte:**
- Winter: ~14.7 µg/m³ | Frühling: ~8.2 µg/m³ | Sommer: ~6.1 µg/m³ | Herbst: ~11.6 µg/m³

**Alle 4 Hypothesen bestätigt (H1–H4: ✓)**

**Stärkste Wetterkorrelation mit NO:** Windgeschwindigkeit (r ≈ −0.38) — hoher Wind dispergiert Schadstoffe.
