import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- USTAWIENIA ---
INPUT_FILE = "wildfires_wy.csv"
OUTPUT_FOLDER = "plots"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- WCZYTANIE DANYCH ---
df = pd.read_csv(INPUT_FILE)

# --- FILTROWANIE LAT 1992–2015 ---
df = df[(df["FIRE_YEAR"] >= 1992) & (df["FIRE_YEAR"] <= 2015)]

# --- OBLICZENIE DŁUGOŚCI POŻARU ---
def calc_duration(row):
    try:
        if not np.isnan(float(row["CONT_DOY"])) and not np.isnan(float(row["DISCOVERY_DOY"])):
            return float(row["CONT_DOY"]) - float(row["DISCOVERY_DOY"])
    except:
        return np.nan
    return np.nan

df["FIRE_DURATION_DAYS"] = df.apply(calc_duration, axis=1)

# --- KONWERSJA KOLUMN LICZBOWYCH ---
cols_to_analyze = [
    "DISCOVERY_DATE", "DISCOVERY_DOY",
    "STAT_CAUSE_CODE", "CONT_DATE", "CONT_DOY",
    "FIRE_SIZE", "FIRE_DURATION_DAYS"
]

for col in cols_to_analyze:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# --- POPRAWNE ŚREDNIE GODZIN ---
def convert_hhmm_to_minutes(x):
    try:
        x = int(x)
        hours = x // 100
        minutes = x % 100
        return hours*60 + minutes
    except:
        return np.nan

def minutes_to_hhmm(minutes):
    if np.isnan(minutes):
        return "N/A"
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"{hours:02d}:{mins:02d}"

df["DISCOVERY_MINUTES"] = df["DISCOVERY_TIME"].apply(convert_hhmm_to_minutes)
df["CONT_MINUTES"] = df["CONT_TIME"].apply(convert_hhmm_to_minutes)

# --- GRUPOWANIE PO ROKU I OBLICZENIE ŚREDNICH ---
stats_mean = df.groupby("FIRE_YEAR")[cols_to_analyze].mean().reset_index()

# Średnia godzin w minutach
stats_mean["DISCOVERY_TIME"] = df.groupby("FIRE_YEAR")["DISCOVERY_MINUTES"].mean().values
stats_mean["CONT_TIME"] = df.groupby("FIRE_YEAR")["CONT_MINUTES"].mean().values

# --- ŚREDNIA ZE WSZYSTKICH LAT ---
overall_mean = stats_mean[cols_to_analyze + ["DISCOVERY_TIME", "CONT_TIME"]].mean()

# --- TWORZENIE WYKRESÓW ---
for col in cols_to_analyze + ["DISCOVERY_TIME", "CONT_TIME"]:
    plt.figure(figsize=(8, 5))
    plt.plot(stats_mean["FIRE_YEAR"], stats_mean[col], marker='o', linestyle='-', linewidth=2, label="Średnia roczna")
    
    # Pozioma linia ze średnią ze wszystkich lat
    plt.axhline(y=overall_mean[col], color='red', linestyle='--', linewidth=2,
                label=f"Średnia (1992-2015) = {minutes_to_hhmm(overall_mean[col])}" if col in ["DISCOVERY_TIME", "CONT_TIME"] else f"{overall_mean[col]:.2f}")
    
    plt.title(f"Średnia wartość: {col} (1992–2015)")
    plt.xlabel("Rok")
    
    # Oś Y - jeśli czas, zamiana minut na HH:MM
    if col in ["DISCOVERY_TIME", "CONT_TIME"]:
        y_ticks = plt.yticks()[0]  # pobranie obecnych ticków
        plt.yticks(y_ticks, [minutes_to_hhmm(m) for m in y_ticks])
        plt.ylabel("Godzina (HH:MM)")
    else:
        plt.ylabel(col)
    
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    filename = os.path.join(OUTPUT_FOLDER, f"{col}_trend.png")
    plt.savefig(filename, dpi=150)
    plt.close()

print(f"✅ Wykresy zapisano w folderze: {OUTPUT_FOLDER}")
