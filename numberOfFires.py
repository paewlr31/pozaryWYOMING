import matplotlib.pyplot as plt
import numpy as np
import os

# --- Dane historyczne liczby pożarów ---
years = np.arange(1992, 2016)
fires_count = np.array([
    746, 269, 919, 530, 1032, 195, 198, 305,
    571, 672, 670, 717, 409, 521, 769, 595,
    449, 474, 631, 640, 1046, 581, 595, 632
])

# --- Obliczenie średniej liczby pożarów ---
mean_fires = np.mean(fires_count)

# --- Folder wyjściowy ---
output_folder = "plots"
os.makedirs(output_folder, exist_ok=True)
output_file = os.path.join(output_folder, "numberOfFires.png")

# --- Tworzenie wykresu liniowego ---
plt.figure(figsize=(12,6))
plt.plot(years, fires_count, marker='o', linestyle='-', linewidth=2, label="Liczba pożarów")
plt.axhline(mean_fires, color='red', linestyle='--', linewidth=2, label=f"Średnia = {mean_fires:.1f}")

# --- Dodanie etykiet przy punktach ---
for x, y in zip(years, fires_count):
    plt.text(x, y + 20, f'{y}', ha='center', va='bottom', fontsize=8)

plt.title("Liczba pożarów w Wyoming (1992–2015)")
plt.xlabel("Rok")
plt.ylabel("Liczba pożarów")
plt.xticks(years, rotation=45)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()

# --- Zapis do pliku PNG ---
plt.savefig(output_file, dpi=150)
plt.close()

print(f"✅ Wykres zapisano w pliku: {output_file}")
