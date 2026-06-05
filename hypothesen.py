# =============================================================
# hypothesen.py – Forschungshypothesen (Kriterium A)
# Thema: Einfluss von Wetter auf Luftqualität in Zürich
# =============================================================
# Verwendet in W06_visualisierung.py::teste_hypothesen(df)
# =============================================================

HYPOTHESEN = {
    "H1": {
        "text": (
            "Niederschlag reduziert PM10 signifikant durch "
            "Auswaschung (wet deposition)"
        ),
        "erwartet":  "negativ",
        "variablen": ["precipitation", "PM10"],
        "methode":   "korrelation",
        "begruendung": (
            "Niederschlag wäscht Partikel aus der Luft (Washout-Effekt); "
            "negative Korrelation zwischen Niederschlagsmenge und PM10 erwartet."
        ),
    },
    "H2": {
        "text": (
            "Inversionslagen (hoher Druck + tiefe Temperatur) "
            "erhöhen NO-Konzentration"
        ),
        "erwartet":  "positiv",
        "variablen": ["inversion_indikator", "NO"],
        "methode":   "gruppenvergleich",
        "begruendung": (
            "Bei Inversionswetterlagen fehlt die vertikale Durchmischung; "
            "Schadstoffe aus Verkehr und Industrie akkumulieren bodennahe."
        ),
    },
    "H3": {
        "text": (
            "Hohe Temperaturen im Sommer erhöhen O3 durch "
            "photochemische Reaktionen"
        ),
        "erwartet":  "positiv",
        "variablen": ["temperature_2m", "O3"],
        "methode":   "korrelation",
        "begruendung": (
            "Ozon entsteht durch UV-Strahlung und Wärme aus NOx + VOC; "
            "positive Korrelation mit Temperatur im Sommer erwartet."
        ),
    },
    "H4": {
        "text": (
            "Rush-Hour-Stunden (7–9h, 17–19h) zeigen erhöhte "
            "NO2-Werte gegenüber Nachtstunden"
        ),
        "erwartet":  "positiv",
        "variablen": ["ist_rush_hour", "NO2"],
        "methode":   "gruppenvergleich",
        "begruendung": (
            "Verkehrsbedingte NO2-Emissionen steigen in Pendler-Stosszeiten; "
            "Gruppenunterschied zwischen Rush-Hour und Nicht-Rush-Hour erwartet."
        ),
    },
}
