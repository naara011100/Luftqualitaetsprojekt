# =============================================================
# DWAE Projekt – Bonus: ML-Modell (LinearRegression)
# Thema: Einfluss von Wetter auf Luftqualität in Zürich
# =============================================================
# Fragestellung: Kann NO₂ (oder anderer Schadstoff) basierend
# auf Wettervariablen vorhergesagt werden?
# Ausführen: python W08_bonus_modell.py
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

Path("output/plots").mkdir(parents=True, exist_ok=True)
Path("output/modell").mkdir(parents=True, exist_ok=True)

import plot_config                                    # setzt rcParams global
from plot_config import FARBEN, who_linie, suptitel_stats, FIG_GROSS

print("=" * 60)
print("BONUS – ML-Modell: Vorhersage Luftschadstoff aus Wetter")
print("=" * 60)


# ─────────────────────────────────────────────────────────────
# 1. DATEN LADEN UND VORBEREITEN
# ─────────────────────────────────────────────────────────────
print("\n[1/5] Daten laden...")

df = pd.read_csv("data/processed/datensatz_final.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Zielvariable: erster verfügbarer Schadstoff
ziel_kandidaten = [c for c in df.columns
                   if any(x in c.upper()
                          for x in ["NO2", "NO", "O3", "PM10", "PM2"])
                   and not any(s in c
                               for s in ["_missing", "_std", "_norm",
                                         "kategorie"])]

if not ziel_kandidaten:
    print("  FEHLER: Keine Schadstoffspalten gefunden.")
    exit(1)

ZIELVARIABLE = ziel_kandidaten[0]
print(f"  Zielvariable: {ZIELVARIABLE}")

# Features: nur Wettervariablen + Zeitfeatures
# (keine anderen Schadstoffe — das wäre Datenleck)
FEATURES = [c for c in [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
    "cloud_cover",
    "sonnenschein_h",
    "stunde",
    "wochentag",
    "monat",
    "ist_wochenende",
    "ist_rush_hour",
    "inversion_indikator",
    "regen",
] if c in df.columns]

print(f"  Features ({len(FEATURES)}): {FEATURES}")

# Zeilen ohne fehlende Werte in Ziel oder Features
df_model = df[[ZIELVARIABLE] + FEATURES].dropna()
print(f"  Datensatz nach dropna: {len(df_model):,} Zeilen "
      f"({len(df_model)/len(df)*100:.1f}% des Originals)")


# ─────────────────────────────────────────────────────────────
# 2. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────
# Wichtig: random_state für Reproduzierbarkeit
# test_size=0.2 → 80% Training, 20% Test
print("\n[2/5] Train/Test Split (80/20)...")

X = df_model[FEATURES]
y = df_model[ZIELVARIABLE]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"  Training: {len(X_train):,} Stunden")
print(f"  Test:     {len(X_test):,} Stunden")
print(f"  Hinweis: Scaler wird NUR auf X_train gefittet (kein Data Leakage)")


# ─────────────────────────────────────────────────────────────
# 3. MODELLE TRAINIEREN UND VERGLEICHEN
# ─────────────────────────────────────────────────────────────
print("\n[3/5] Modelle trainieren...")

# Drei Modelle: LinearRegression, Ridge, RandomForest
# Pipeline: Skalierung + Modell in einem Schritt
modelle = {
    "Lineare Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LinearRegression())
    ]),
    "Ridge Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  Ridge(alpha=1.0))
    ]),
    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ))
    ]),
}

ergebnisse = {}

print(f"\n  {'Modell':<25} {'MAE':>8} {'RMSE':>8} {'R²':>8} "
      f"{'CV R² (Ø)':>12} {'CV Std':>8}")
print(f"  {'-'*73}")

for name, pipeline in modelle.items():
    # Training
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Metriken
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    # Cross-Validation (5-fold) auf Trainingsdaten
    cv_scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=5, scoring="r2", n_jobs=-1
    )

    ergebnisse[name] = {
        "pipeline": pipeline,
        "y_pred":   y_pred,
        "mae":      mae,
        "rmse":     rmse,
        "r2":       r2,
        "cv_mean":  cv_scores.mean(),
        "cv_std":   cv_scores.std(),
    }

    stern = " ★" if name == "Random Forest" else ""
    print(f"  {name:<25} {mae:>8.2f} {rmse:>8.2f} {r2:>8.3f} "
          f"{cv_scores.mean():>12.3f} {cv_scores.std():>8.3f}{stern}")

# Bestes Modell bestimmen
bestes_name = max(ergebnisse, key=lambda k: ergebnisse[k]["r2"])
bestes       = ergebnisse[bestes_name]
print(f"\n  Bestes Modell: {bestes_name} (R² = {bestes['r2']:.3f})")


# ─────────────────────────────────────────────────────────────
# 4. FEATURE IMPORTANCE (Random Forest)
# ─────────────────────────────────────────────────────────────
print("\n[4/5] Feature Importance...")

rf_pipeline = modelle["Random Forest"]
rf_model    = rf_pipeline.named_steps["model"]
importances = rf_model.feature_importances_

feat_imp = pd.DataFrame({
    "Feature":    FEATURES,
    "Importance": importances
}).sort_values("Importance", ascending=False)

print(f"\n  Top Features für {ZIELVARIABLE}-Vorhersage:")
print(f"  {'Feature':<35} {'Importance':>12} {'Balken'}")
print(f"  {'-'*65}")
for _, row in feat_imp.iterrows():
    balken = "█" * int(row["Importance"] * 100)
    print(f"  {row['Feature']:<35} {row['Importance']:>12.4f}  {balken}")


# ─────────────────────────────────────────────────────────────
# 5. VISUALISIERUNGEN
# ─────────────────────────────────────────────────────────────
print("\n[5/5] Plots erstellen...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ── Plot 1: Modellvergleich (R²) ──
namen  = list(ergebnisse.keys())
r2_vals = [ergebnisse[n]["r2"] for n in namen]
cv_vals = [ergebnisse[n]["cv_mean"] for n in namen]
x = np.arange(len(namen))
w = 0.35
axes[0, 0].bar(x - w/2, r2_vals, w, label="Test R²",
               color=FARBEN["blau"], alpha=0.85, edgecolor="white")
axes[0, 0].bar(x + w/2, cv_vals,  w, label="CV R² (5-fold)",
               color=FARBEN["gruen"], alpha=0.85, edgecolor="white")
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(namen, fontsize=9)
axes[0, 0].set_ylabel("R² Score")
axes[0, 0].set_title("Modellvergleich: Test R² vs. Cross-Validation R²")
axes[0, 0].legend(fontsize=9)
axes[0, 0].set_ylim(0, 1)
axes[0, 0].axhline(0.5, color=FARBEN["grau"], lw=0.8,
                   linestyle="--", alpha=0.5)
axes[0, 0].grid(True, alpha=0.2, axis="y")
for i, (r, c) in enumerate(zip(r2_vals, cv_vals)):
    axes[0, 0].text(i - w/2, r + 0.01, f"{r:.2f}",
                    ha="center", fontsize=8)
    axes[0, 0].text(i + w/2, c + 0.01, f"{c:.2f}",
                    ha="center", fontsize=8)

# ── Plot 2: Vorhergesagt vs. Tatsächlich (bestes Modell) ──
y_pred_best = bestes["y_pred"]
lim = [min(y_test.min(), y_pred_best.min()) * 0.9,
       max(y_test.max(), y_pred_best.max()) * 1.1]
axes[0, 1].scatter(y_test, y_pred_best, alpha=0.15, s=4,
                   color=FARBEN["lila"])
axes[0, 1].plot(lim, lim, color=FARBEN["rot"],
                lw=1.5, linestyle="--", label="Perfekte Vorhersage")
axes[0, 1].set_xlim(lim); axes[0, 1].set_ylim(lim)
axes[0, 1].set_xlabel(f"Tatsächlicher {ZIELVARIABLE} (µg/m³)")
axes[0, 1].set_ylabel(f"Vorhergesagter {ZIELVARIABLE} (µg/m³)")
axes[0, 1].set_title(f"{bestes_name}: Vorhergesagt vs. Tatsächlich\n"
                     f"R²={bestes['r2']:.3f}, MAE={bestes['mae']:.2f}")
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(True, alpha=0.2)

# ── Plot 3: Feature Importance ──
top_n = min(10, len(feat_imp))
top   = feat_imp.head(top_n)
axes[1, 0].barh(top["Feature"][::-1], top["Importance"][::-1],
                color=FARBEN["gruen"], alpha=0.85, edgecolor="white")
axes[1, 0].set_xlabel("Importance")
axes[1, 0].set_title(f"Top {top_n} Features – Random Forest\n"
                     f"(Wichtigkeit für {ZIELVARIABLE}-Vorhersage)")
axes[1, 0].grid(True, alpha=0.2, axis="x")

# ── Plot 4: Residuen ──
residuen = y_test.values - y_pred_best
axes[1, 1].hist(residuen, bins=60, color=FARBEN["blau"],
                alpha=0.8, edgecolor="white")
axes[1, 1].axvline(0, color=FARBEN["rot"], lw=1.5,
                   linestyle="--", label="Kein Fehler")
axes[1, 1].axvline(residuen.mean(), color=FARBEN["gruen"], lw=1.5,
                   linestyle="-",
                   label=f"Mittlerer Fehler: {residuen.mean():.2f}")
axes[1, 1].set_xlabel(f"Residuen (Tatsächlich − Vorhergesagt)")
axes[1, 1].set_ylabel("Häufigkeit")
axes[1, 1].set_title(f"Residuenverteilung – {bestes_name}")
axes[1, 1].legend(fontsize=9)
axes[1, 1].grid(True, alpha=0.2)

plt.suptitle(
    f"ML-Modell: Vorhersage {ZIELVARIABLE} aus Wetterdaten – Zürich 2023",
    fontsize=13, y=1.01
)
plt.tight_layout()
plt.savefig("output/plots/bonus_modell.png",
            bbox_inches="tight", dpi=120)
plt.close()
print("  ✓ output/plots/bonus_modell.png")

# Ergebnisse speichern (für Bericht)
ergebnis_df = pd.DataFrame([{
    "Modell":   name,
    "MAE":      round(e["mae"], 3),
    "RMSE":     round(e["rmse"], 3),
    "R2_Test":  round(e["r2"], 3),
    "R2_CV":    round(e["cv_mean"], 3),
    "CV_Std":   round(e["cv_std"], 3),
} for name, e in ergebnisse.items()])
ergebnis_df.to_csv("output/modell/modell_ergebnisse.csv", index=False)
feat_imp.to_csv("output/modell/feature_importance.csv", index=False)
print("  ✓ output/modell/modell_ergebnisse.csv")
print("  ✓ output/modell/feature_importance.csv")


# ─────────────────────────────────────────────────────────────
# ZUSAMMENFASSUNG FÜR DEN BERICHT
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ERGEBNISSE FÜR DEN BERICHT")
print("=" * 60)
print(f"""
  Zielvariable: {ZIELVARIABLE}
  Features:     {len(FEATURES)} Wetter- und Zeitvariablen
  Datensatz:    {len(df_model):,} Stunden (nach Bereinigung)
  Split:        80% Training / 20% Test (random_state=42)

  Modellvergleich:""")

for name, e in ergebnisse.items():
    stern = " ← bestes Modell" if name == bestes_name else ""
    print(f"    {name:<25} R²={e['r2']:.3f}  "
          f"MAE={e['mae']:.2f}  RMSE={e['rmse']:.2f}{stern}")

print(f"""
  Feature Importance (Top 3):""")
for _, row in feat_imp.head(3).iterrows():
    print(f"    {row['Feature']:<35} {row['Importance']:.4f}")

print(f"""
  Interpretation:
  Ein R² von {bestes['r2']:.2f} bedeutet, dass das {bestes_name}-Modell
  {bestes['r2']*100:.0f}% der Varianz in {ZIELVARIABLE} durch Wettervariablen
  erklären kann. {'Das ist ein gutes Ergebnis' if bestes['r2'] > 0.5
  else 'Das zeigt, dass Wetter einen Teil erklärt, aber andere Faktoren'
       ' (z.B. Verkehr, Industrie) ebenfalls wichtig sind'}.

  Für den Bericht:
  - Tabelle: output/modell/modell_ergebnisse.csv
  - Feature Importance: output/modell/feature_importance.csv
  - Plot: output/plots/bonus_modell.png
""")
print("=" * 60)
