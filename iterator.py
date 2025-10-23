import pandas as pd
import numpy as np

# --- USTAWIENIA ---
INPUT_FILE = "wildfires_wy.csv"
OUTPUT_FILE = "fire_report.txt"

# --- WCZYTANIE DANYCH ---
df = pd.read_csv(INPUT_FILE)

# --- FILTROWANIE LAT 1992–2015 ---
df = df[(df["FIRE_YEAR"] >= 1992) & (df["FIRE_YEAR"] <= 2015)]

# --- OBLICZENIE DŁUGOŚCI POŻARU (w dniach) ---
def calc_duration(row):
    try:
        if not np.isnan(float(row["CONT_DOY"])) and not np.isnan(float(row["DISCOVERY_DOY"])):
            return float(row["CONT_DOY"]) - float(row["DISCOVERY_DOY"])
    except:
        return np.nan
    return np.nan

df["FIRE_DURATION_DAYS"] = df.apply(calc_duration, axis=1)

# --- WYBÓR KOLUMN LICZBOWYCH DOT. POŻARÓW ---
cols_to_analyze = [
    "DISCOVERY_DATE", "DISCOVERY_DOY", "DISCOVERY_TIME",
    "STAT_CAUSE_CODE", "CONT_DATE", "CONT_DOY", "CONT_TIME",
    "FIRE_SIZE", "FIRE_DURATION_DAYS"
]

# --- KONWERSJA DO NUMERYCZNYCH ---
for col in cols_to_analyze:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# --- GRUPOWANIE I OBLICZENIA ---
stats = df.groupby("FIRE_YEAR")[cols_to_analyze].agg(["mean", "min", "max"])
stats["LICZBA_POZAROW"] = df.groupby("FIRE_YEAR")["OBJECTID"].count()

# --- ZAPIS DO PLIKU ---
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("ROCZNE ŚREDNIE I STATYSTYKI POŻARÓW (1992–2015)\n")
    f.write("="*75 + "\n\n")

    for year in stats.index:
        f.write(f"Rok: {year}\n")
        f.write("-"*75 + "\n")
        f.write(f"Liczba pożarów: {int(stats.loc[year, 'LICZBA_POZAROW'])}\n")

        for col in cols_to_analyze:
            mean_val = stats.loc[year, (col, "mean")]
            min_val = stats.loc[year, (col, "min")]
            max_val = stats.loc[year, (col, "max")]

            def fmt(x):
                return "N/A" if pd.isna(x) else round(x, 2)

            f.write(f"{col} -> Śr: {fmt(mean_val)}, Min: {fmt(min_val)}, Max: {fmt(max_val)}\n")

        f.write("-"*75 + "\n\n")

print(f"✅ Raport zapisano w pliku: {OUTPUT_FILE}")
