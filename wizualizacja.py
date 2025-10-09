import pandas as pd
import folium
from folium.plugins import TimestampedGeoJson
from datetime import datetime, timedelta

# Wczytaj dane z wildfires_wy.csv
df = pd.read_csv('wildfires_wy.csv', low_memory=False)

# Filtruj dane dla roku 2015 i usuń rekordy bez CONT_DOY
wy_2015 = df[(df['FIRE_YEAR'] == 2015) & (df['CONT_DOY'].notna())]

# Przygotuj dane dla GeoJSON
features = []
for _, row in wy_2015.iterrows():
    # Konwersja DISCOVERY_DOY i CONT_DOY na daty
    discovery_date = datetime(2015, 1, 1) + timedelta(days=int(row['DISCOVERY_DOY']) - 1)
    cont_date = datetime(2015, 1, 1) + timedelta(days=int(row['CONT_DOY']) - 1)
    
    # Oblicz czas trwania pożaru (w dniach)
    duration_days = (cont_date - discovery_date).days
    if duration_days <= 0:
        duration_days = 1  # Minimum 1 dzień, jeśli DISCOVERY_DOY == CONT_DOY
    
    # Maksymalny promień proporcjonalny do FIRE_SIZE
    max_radius = row['FIRE_SIZE'] / 100  # Skalowanie
    
    # Generuj punkty dla każdego dnia pożaru
    for day in range(duration_days + 1):
        current_date = discovery_date + timedelta(days=day)
        timestamp = current_date.isoformat()
        
        # Oblicz dynamiczny promień (rośnie do połowy, potem maleje)
        if day <= duration_days / 2:
            # Rosnąca faza
            radius = max_radius * (day / (duration_days / 2))
        else:
            # Malejąca faza
            radius = max_radius * ((duration_days - day) / (duration_days / 2))
        
        # Stwórz punkt GeoJSON
        feature = {
            'type': 'Feature',
            'geometry': {
                'type': 'Point',
                'coordinates': [row['LONGITUDE'], row['LATITUDE']],
            },
            'properties': {
                'time': timestamp,
                'popup': f"Fire: {row.get('FIRE_NAME', 'Unknown')}<br>Size: {row['FIRE_SIZE']} acres<br>Cause: {row['STAT_CAUSE_DESCR']}<br>Day: {day+1}/{duration_days+1}",
                'style': {
                    'radius': max(0.1, radius),  # Minimum 0.1, by kropki były widoczne
                    'fillColor': 'red',
                    'color': 'red',
                    'fillOpacity': 0.6
                }
            }
        }
        features.append(feature)

# Stwórz mapę zablokowaną na Wyoming
m = folium.Map(
    location=[43.0, -107.5],  # Środek Wyoming
    zoom_start=7,
    tiles='OpenStreetMap',
    min_zoom=7,  # Minimalny zoom
    max_bounds=True,  # Ogranicza przesuwanie
    dragging=False,  # Wyłącza przesuwanie
    zoom_control=False  # Wyłącza kontrolki zoomu (opcjonalne)
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
    period='P1D',  # Krok czasowy: 1 dzień
    auto_play=True,
    loop=False,
    max_speed=10,
    add_last_point=True
).add_to(m)

# Zapisz mapę do pliku HTML
m.save('wy_fires_2015_dynamic.html')

print("Dynamiczna mapa Wyoming zapisana jako 'wy_fires_2015_dynamic.html'. Otwórz w przeglądarce!")