# =============================================================
# plot_config.py – Globales Matplotlib-Styling für alle Phasen
# Importieren mit:  import plot_config          (setzt rcParams)
#                   from plot_config import …   (Farben, Helfer)
# =============================================================

import pandas as pd
import matplotlib.pyplot as plt

# ── Einheitliches Styling (beim Import automatisch gesetzt) ──
plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.size":          10,        # Basis / Achsenlabels
    "axes.titlesize":     12,        # Subplot-Titel
    "axes.labelsize":     10,        # Achsenbeschriftungen
    "xtick.labelsize":    9,         # Tick-Labels
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         150,       # Bericht-Qualität
    "savefig.dpi":        150,
    "savefig.bbox":       "tight",
})

# ── Standard-Figurengrössen ──
FIG_KLEIN   = (14, 6)    # Standard-Plot
FIG_GROSS   = (16, 10)   # Dashboard / mehrzeiliges Layout
FIG_QUADRAT = (10, 8)    # Quadratische Darstellungen
FIG_POLAR   = (8,  8)    # Windrose / Polar-Plots

# ── Farben ──
FARBEN = {
    "blau":  "#185FA5",
    "gruen": "#1D9E75",
    "rot":   "#D85A30",
    "lila":  "#534AB7",
    "gelb":  "#C89B1C",
    "grau":  "#6B7280",
}
JAHRESZEIT_FARBEN = {
    "Winter":   FARBEN["blau"],
    "Frühling": FARBEN["gruen"],
    "Sommer":   FARBEN["rot"],
    "Herbst":   FARBEN["gelb"],
}

# ── Spalten → Masseinheiten ──
_EINHEITEN = {
    "temperature_2m":       "°C",
    "relative_humidity_2m": "%",
    "precipitation":        "mm/h",
    "wind_speed_10m":       "km/h",
    "wind_direction_10m":   "°",
    "surface_pressure":     "hPa",
    "cloud_cover":          "%",
    "sunshine_duration":    "s/h",
    "sonnenschein_h":       "h",
    "NO2":                  "µg/m³",
    "NO":                   "µg/m³",
    "O3":                   "µg/m³",
    "PM10":                 "µg/m³",
    "PM2":                  "µg/m³",
}

# ── WHO-Luftqualitäts-Grenzwerte 2021 ──
WHO_GRENZWERTE = {
    "NO2":  40.0,    # µg/m³  Jahresmittel
    "PM10": 45.0,    # µg/m³  24h-Mittel
    "O3":   100.0,   # µg/m³  8h-Mittel (max)
    "PM2":  15.0,    # µg/m³  PM2.5 Jahresmittel
}


def einheit(col: str) -> str:
    """Masseinheit für eine Spaltenbezeichnung."""
    for key, unit in _EINHEITEN.items():
        if key.upper() in col.upper():
            return unit
    return ""


def who_linie(ax, col: str, vertikal: bool = False, **kwargs):
    """
    Zeichnet WHO-Grenzwert als gestrichelte Linie auf ax.
    vertikal=True  → axvline  (Histogramme: Schadstoff auf X-Achse)
    vertikal=False → axhline  (Standard: Zeitreihen, Boxplots, Balken)
    Gibt den Grenzwert zurück, oder None wenn kein Grenzwert bekannt.
    """
    for schadstoff, grenzwert in WHO_GRENZWERTE.items():
        if schadstoff.upper() in col.upper():
            opts = dict(
                color="#C0392B", linestyle="--", linewidth=1.3,
                alpha=0.85, zorder=3,
                label=f"WHO {grenzwert:.0f} µg/m³"
            )
            opts.update(kwargs)
            if vertikal:
                ax.axvline(grenzwert, **opts)
            else:
                ax.axhline(grenzwert, **opts)
            return grenzwert
    return None


def suptitel_stats(df, col: str = None, prefix: str = "") -> str:
    """
    Liefert Kennzahlen-Zeile für fig.suptitle().
    Format: 'n = X,XXX Std.  |  DD.MM.YYYY – DD.MM.YYYY  |  Ø XX.X Einheit'
    """
    parts = []

    n = int(df[col].notna().sum()) if (col and col in df.columns) else len(df)
    parts.append(f"n = {n:,} Std.")

    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"]).dropna()
        if len(ts) > 0:
            parts.append(
                f"{ts.min().strftime('%d.%m.%Y')} – "
                f"{ts.max().strftime('%d.%m.%Y')}"
            )

    if col and col in df.columns:
        mittel = df[col].mean()
        if pd.notna(mittel):
            parts.append(f"Ø {mittel:.1f} {einheit(col)}")

    stat_str = "  |  ".join(parts)
    return f"{prefix}  |  {stat_str}" if prefix else stat_str
