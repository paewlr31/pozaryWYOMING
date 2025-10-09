import pandas as pd

# Wczytaj plik CSV
df = pd.read_csv('wildfires.csv', low_memory=False)

# Filtruj dane dla Wyoming (STATE = 'WY')
wy_fires = df[df['STATE'] == 'WY']

# Zapisz do nowego pliku CSV
wy_fires.to_csv('wildfires_wy.csv', index=False, encoding='utf-8')

print(f"Zapisano {len(wy_fires)} rekordów dla Wyoming do pliku 'wildfires_wy.csv'.")