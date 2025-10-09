import pandas as pd
import numpy as np

# Wczytaj dane Wyoming
df = pd.read_csv('wildfires_wy.csv', low_memory=False)

# Ogranicz do kluczowych kolumn
df = df[['FIRE_YEAR', 'DISCOVERY_DOY', 'LATITUDE', 'LONGITUDE', 'FIRE_SIZE', 'STAT_CAUSE_DESCR']]

# Stwórz siatkę (1 km x 1 km)
grid_size = 0.01  # ~1 km
df['grid_x'] = np.floor(df['LONGITUDE'] / grid_size).astype(int)
df['grid_y'] = np.floor(df['LATITUDE'] / grid_size).astype(int)

# Przykład: Inicjalizacja siatki dla roku 2015
wy_2015 = df[df['FIRE_YEAR'] == 2015]
grid_x_min, grid_x_max = df['grid_x'].min(), df['grid_x'].max()
grid_y_min, grid_y_max = df['grid_y'].min(), df['grid_y'].max()
grid = np.zeros((grid_y_max - grid_y_min + 1, grid_x_max - grid_x_min + 1), dtype=int)

# Oznacz pożary na siatce
for _, row in wy_2015.iterrows():
    x = row['grid_x'] - grid_x_min
    y = row['grid_y'] - grid_y_min
    grid[y, x] = 2  # Płonie

# Zapisz siatkę jako JSON
output = {"time_step": 0, "grid": grid.tolist()}
import json
with open('wy_fires_grid.json', 'w') as f:
    json.dump(output, f)

print("Siatka początkowa zapisana do 'wy_fires_grid.json'.")