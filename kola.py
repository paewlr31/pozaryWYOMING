import pandas as pd

# --- Dane historyczne (średnie wartości 1992-2015) ---
historical = {
    "Human": {"count": 222.38, "percent": 39.09},
    "Lightning": {"count": 269.21, "percent": 46.25},
    "Other": {"count": 98.67, "percent": 14.66}
}

# --- Dane symulacji (rok 2006) ---
simulation = {
    "Human": {"count": 532*0.2744, "percent": 27.44},  # obliczone liczby z procentów
    "Lightning": {"count": 532*0.2575, "percent": 25.75},
    "Other": {"count": 532*0.0338, "percent": 3.38},
    "Spread_local": {"count": 532*0.3421, "percent": 34.21},
    "Spread_inter": {"count": 532*0.0921, "percent": 9.21}
}

# --- Wyświetlenie porównania ---
print("Porównanie średnich wyników modelu z danymi historycznymi (2006 vs 1992-2015):\n")
for group in ["Human", "Lightning", "Other"]:
    hist = historical[group]
    sim = simulation[group]
    print(f"{group}:")
    print(f"  Średnia liczba pożarów (historyczna): {hist['count']:.2f}")
    print(f"  Średnia liczba pożarów (symulacja): {sim['count']:.2f}")
    print(f"  Średni udział procentowy (historyczna): {hist['percent']:.2f}%")
    print(f"  Średni udział procentowy (symulacja): {sim['percent']:.2f}%\n")

# --- Dodatkowo średnia całkowita pożarów historycznych ---
total_historical = sum(historical[g]["count"] for g in historical)
print(f"Średnia całkowita liczba pożarów w latach 1992-2015: {total_historical:.2f}")
print(f"Całkowita liczba pożarów w symulacji 2006: 532")
