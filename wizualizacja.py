import pandas as pd
import folium
from folium.plugins import TimestampedGeoJson
from datetime import datetime, timedelta

# Wczytaj dane z wildfires_wy.csv
df = pd.read_csv('wildfires_wy.csv', low_memory=False)

# Filtruj dane dla roku 2015 i usuń rekordy z brakującym FIRE_SIZE
# Usuwamy brakujące CONT_DOY w filtrze, ale dla pewności obsłużymy je poniżej
wy_2015 = df[(df['FIRE_YEAR'] == 2015) & (df['FIRE_SIZE'].notna())]

# Sprawdź dane
print("Liczba rekordów w 2015:", len(wy_2015))
print("Statystyki FIRE_SIZE:", wy_2015['FIRE_SIZE'].describe())
print("Brakujące CONT_DOY:", wy_2015['CONT_DOY'].isna().sum())
print("Pożary z DISCOVERY_DOY == CONT_DOY:", len(wy_2015[wy_2015['DISCOVERY_DOY'] == wy_2015['CONT_DOY']]))

# Przygotuj dane dla GeoJSON
features = []
for _, row in wy_2015.iterrows():
    # Konwersja DISCOVERY_DOY na datę
    discovery_date = datetime(2015, 1, 1) + timedelta(days=int(row['DISCOVERY_DOY']) - 1)
    
    # Obsługa CONT_DOY (domyślnie 5 dni, jeśli brak lub DISCOVERY_DOY == CONT_DOY)
    if pd.isna(row['CONT_DOY']) or row['DISCOVERY_DOY'] == row['CONT_DOY']:
        cont_date = discovery_date + timedelta(days=5)
    else:
        cont_date = datetime(2015, 1, 1) + timedelta(days=int(row['CONT_DOY']) - 1)
    
    # Oblicz czas trwania pożaru (minimum 5 dni dla animacji)
    duration_days = max(5, (cont_date - discovery_date).days)
    
    # Maksymalny promień (ograniczony, by uniknąć ogromnych kropek)
    max_radius = min(50, max(1, row['FIRE_SIZE'] / 20))  # Ograniczenie do 50
    
    # Generuj punkty dla każdego dnia pożaru
    for day in range(duration_days + 1):
        current_date = discovery_date + timedelta(days=day)
        timestamp = current_date.isoformat()
        
        # Oblicz dynamiczny promień (rośnie do połowy, potem maleje)
        if day <= duration_days / 2:
            radius = max_radius * (day / (duration_days / 2))  # Rośnie
        else:
            radius = max_radius * ((duration_days - day) / (duration_days / 2))  # Maleje
        
        # Stwórz punkt GeoJSON
        feature = {
            'type': 'Feature',
            'geometry': {
                'type': 'Point',
                'coordinates': [row['LONGITUDE'], row['LATITUDE']],
            },
            'properties': {
                'time': timestamp,
                'popup': f"Fire: {row.get('FIRE_NAME', 'Unknown')}<br>Size: {row['FIRE_SIZE']} acres<br>Cause: {row.get('STAT_CAUSE_DESCR', 'Unknown')}<br>Day: {day+1}/{duration_days+1}",
                'icon': 'circle',
                'iconstyle': {
                    'fillColor': 'red',
                    'color': 'red',
                    'fillOpacity': 0.6,
                    'radius': max(1, radius)  # Minimum 1 dla widoczności
                }
            }
        }
        features.append(feature)

# Stwórz mapę zablokowaną na Wyoming
m = folium.Map(
    location=[43.0, -107.5],  # Środek Wyoming
    zoom_start=7,
    tiles='OpenStreetMap',
    min_zoom=7,
    max_bounds=True,
    dragging=False,  # Bez przesuwania
    zoom_control=False  # Bez zoomu
)

# Ustaw granice Wyoming
m.fit_bounds([[41.0, -111.0], [45.0, -104.0]])

# Dodaj animację czasową
geojson = {
    'type': 'FeatureCollection',
    'features': features
}
TimestampedGeoJson(
    geojson,
    period='P1D',  # Krok: 1 dzień
    duration='P1D',  # Punkt widoczny tylko przez 1 dzień
    auto_play=True,
    loop=False,
    max_speed=5,  # Wolniejsza animacja
    transition_time=500,  # Płynne przejścia
    add_last_point=False  # Nie zostawia punktów na końcu
).add_to(m)

# Zapisz mapę
m.save('wy_fires_2015_dynamic.html')

print("Dynamiczna mapa Wyoming zapisana jako 'wy_fires_2015_dynamic.html'. Otwórz w przeglądarce!")