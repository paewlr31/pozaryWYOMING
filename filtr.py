import pandas as pd

# Odczyt pliku CSV
df = pd.read_csv('wildfires.csv')

# Filtracja tylko pożarów w stanie Wyoming (kolumna STATE == 'WY')
df_wyoming = df[df['STATE'] == 'WY']

# Zapis do nowego pliku
df_wyoming.to_csv('wildfires_wy.csv', index=False)

print(f"Zapisano {len(df_wyoming)} pożarów z Wyoming do pliku 'wildfires_wy.csv'")