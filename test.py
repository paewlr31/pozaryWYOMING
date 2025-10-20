import pandas as pd
import numpy as np
import folium
from folium.plugins import TimestampedGeoJson
from datetime import datetime, timedelta
import random

# ==========================
# 1. Wczytanie danych
# ==========================
df = pd.read_csv('wildfires_wy.csv', low_memory=False)
df = df[['FIRE_YEAR', 'DISCOVERY_DOY', 'CONT_DOY', 'LATITUDE', 'LONGITUDE', 'FIRE_SIZE', 'STAT_CAUSE_DESCR', 'STATE', 'COUNTY', 'FIPS_CODE', 'FIPS_NAME']]

# Filtr dla Wyoming i brakujących danych
df = df[df['STATE'] == 'WY']
df = df[df['LATITUDE'].notna() & df['LONGITUDE'].notna()]

# Poprawka dla DURATION: zapewnienie dodatnich wartości
df['DURATION'] = df['CONT_DOY'].fillna(df['DISCOVERY_DOY'] + 5) - df['DISCOVERY_DOY']
df['DURATION'] = df['DURATION'].apply(lambda x: max(1, x))  # Minimalna długość trwania: 1 dzień
avg_duration = df['DURATION'].mean()
std_duration = df['DURATION'].std()

# Debugowanie danych DURATION
print("Min DURATION:", df['DURATION'].min())
print("Max DURATION:", df['DURATION'].max())
print("Avg DURATION:", avg_duration)
print("Std DURATION:", std_duration)

# ==========================
# 2. Tworzenie siatki 1km x 1km
# ==========================
grid_size = 0.01
df['grid_x'] = np.floor(df['LONGITUDE'] / grid_size).astype(int)
df['grid_y'] = np.floor(df['LATITUDE'] / grid_size).astype(int)

grid_x_min, grid_x_max = df['grid_x'].min(), df['grid_x'].max()
grid_y_min, grid_y_max = df['grid_y'].min(), df['grid_y'].max()
grid_width = int(grid_x_max - grid_x_min + 1)
grid_height = int(grid_y_max - grid_y_min + 1)

# Inicjalizacja siatki
grid = np.zeros((grid_height, grid_width), dtype=int)  # 0 = brak, 1 = las, 2 = pożar, 3 = miasto

# ==========================
# 3. Definicja agentów
# ==========================
class CountyAgent:
    def __init__(self, fips, name, forest_area, fire_history):
        self.fips = fips
        self.name = name
        self.forest_area = forest_area
        self.fire_history = fire_history
        self.active_fires = []

class ForestAgent:
    def __init__(self, grid_x, grid_y, density=0.8):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.status = 1 if random.random() < density else 0  # 1=las, 0=pusty

class CityAgent:
    def __init__(self, grid_x, grid_y):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.status = 3  # 3=miasto

class FireIncidentAgent:
    def __init__(self, grid_x, grid_y, size, cause, start_day):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.size = random.uniform(10, 100) if size == 1 else size  # Losowy rozmiar 10-100 akrów dla nowych pożarów
        self.cause = cause
        self.active = True
        self.start_day = start_day
        self.duration = max(5, int(np.random.normal(avg_duration, std_duration)))  # Minimum 5 dni

class FirefighterAgent:
    def __init__(self, grid_x, grid_y, effectiveness=0.7, range_cells=2):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.effectiveness = effectiveness
        self.range_cells = range_cells

    def try_extinguish(self, fire):
        if abs(self.grid_x - fire.grid_x) <= self.range_cells and abs(self.grid_y - fire.grid_y) <= self.range_cells:
            if random.random() < self.effectiveness:
                fire.active = False  # Pożar ugaszony
                fire.duration = min(fire.duration, max(1, int(fire.duration * 0.5)))  # Skrócenie czasu trwania

class WeatherAgent:
    def __init__(self):
        self.temperature = np.random.normal(25, 5)
        self.humidity = np.random.normal(40, 10)
        self.wind_speed = np.random.normal(5, 2)

    def update(self):
        self.temperature += np.random.normal(0, 1)
        self.humidity += np.random.normal(0, 2)
        self.wind_speed += np.random.normal(0, 0.5)
        self.temperature = np.clip(self.temperature, 10, 40)
        self.humidity = np.clip(self.humidity, 10, 90)
        self.wind_speed = np.clip(self.wind_speed, 0, 15)

    def fire_spread_probability(self):
        temp_factor = (self.temperature - 20) / 20
        humidity_factor = (60 - self.humidity) / 60
        wind_factor = self.wind_speed / 10
        return np.clip(0.3 + temp_factor * 0.2 + humidity_factor * 0.3 + wind_factor * 0.2, 0.1, 0.6)

class RegionAgent:
    def __init__(self, name, counties):
        self.name = name
        self.counties = counties

    def monte_carlo_first_fire(self, day, weather, total_fires):
        if len(total_fires) >= 30000:  # Ograniczenie liczby pożarów
            return
        county = random.choice(self.counties)
        forest_cells = [(f.grid_x, f.grid_y) for f in forests if f.status == 1]
        if forest_cells and random.random() < 0.05:
            x, y = random.choice(forest_cells)
            if random.random() < weather.fire_spread_probability():
                fire = FireIncidentAgent(x, y, size=1, cause='Monte Carlo', start_day=day)
                county.active_fires.append(fire)
                total_fires.append(fire)

# ==========================
# 4. Inicjalizacja agentów
# ==========================
counties = {}
forests = []
cities = []
firefighters = []

for fips, group in df.groupby('FIPS_CODE'):
    county_name = group['FIPS_NAME'].iloc[0]
    forest_area = len(group)
    fire_history = group['FIRE_YEAR'].unique().tolist()
    county_agent = CountyAgent(fips, county_name, forest_area, fire_history)
    counties[fips] = county_agent
    for _, row in group.iterrows():
        f_agent = ForestAgent(row['grid_x'] - grid_x_min, row['grid_y'] - grid_y_min)
        forests.append(f_agent)

# Dodanie 500 lasów w Yellowstone (80% aktywnych lasów)
yellowstone_lon_min, yellowstone_lon_max = -111.0, -109.8
yellowstone_lat_min, yellowstone_lat_max = 44.1, 45.0
yellowstone_density = 0.8
num_yellowstone_forests = 500
for _ in range(num_yellowstone_forests):
    lon = random.uniform(yellowstone_lon_min, yellowstone_lon_max)
    lat = random.uniform(yellowstone_lat_min, yellowstone_lat_max)
    grid_x = int(np.floor(lon / grid_size) - grid_x_min)
    grid_y = int(np.floor(lat / grid_size) - grid_y_min)
    if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
        f_agent = ForestAgent(grid_x, grid_y, density=yellowstone_density)
        forests.append(f_agent)

# Debugowanie liczby lasów w Yellowstone
yellowstone_forests = [f for f in forests if (yellowstone_lon_min <= (f.grid_x + grid_x_min) * grid_size <= yellowstone_lon_max and
                                              yellowstone_lat_min <= (f.grid_y + grid_y_min) * grid_size <= yellowstone_lat_max)]
yellowstone_active_forests = [f for f in yellowstone_forests if f.status == 1]
print(f"Liczba wszystkich lasów dodanych w Yellowstone: {len(yellowstone_forests)}")
print(f"Liczba aktywnych lasów (status=1) w Yellowstone: {len(yellowstone_active_forests)}")

# Lista miast Wyoming (25 lokalizacji)
city_locations = [
    (-105.5022, 41.1399),  # Cheyenne
    (-104.8202, 41.1359),  # Casper
    (-106.3197, 42.8641),  # Laramie
    (-105.9399, 44.2910),  # Gillette
    (-109.2490, 41.6372),  # Rock Springs
    (-104.8253, 44.7970),  # Sheridan
    (-108.2023, 43.0238),  # Cody
    (-110.3326, 43.4799),  # Jackson
    (-105.5911, 44.3483),  # Buffalo
    (-107.1357, 43.6599),  # Riverton
    (-104.1827, 42.8494),  # Douglas
    (-108.7373, 42.7552),  # Green River
    (-110.9632, 41.3114),  # Evanston
    (-107.2009, 44.0802),  # Worland
    (-104.6097, 41.6322),  # Torrington
    (-108.3897, 42.8347),  # Lander
    (-104.2047, 44.9083),  # Newcastle
    (-106.4064, 44.6872),  # Powell
    (-104.0555, 42.0625),  # Wheatland
    (-110.0752, 41.7911),  # Kemmerer
    (-105.7455, 42.0978),  # Rawlins
    (-107.5437, 41.5875),  # Saratoga
    (-104.1389, 44.2720),  # Sundance
    (-108.8964, 42.0663),  # Pinedale
    (-106.6392, 42.7475)   # Hanna
]
cities = []
for lon, lat in city_locations:
    grid_x = int(np.floor(lon / grid_size) - grid_x_min)
    grid_y = int(np.floor(lat / grid_size) - grid_y_min)
    cities.append(CityAgent(grid_x, grid_y))

# Generowanie 100 remiz strażackich: 70% w pobliżu miast, 30% losowo w Wyoming
firestation_locations = []
for _ in range(70):
    city = random.choice(city_locations)
    lon = city[0] + random.uniform(-0.05, 0.05)
    lat = city[1] + random.uniform(-0.05, 0.05)
    lon = np.clip(lon, -111.0, -104.0)
    lat = np.clip(lat, 41.0, 45.0)
    firestation_locations.append((lon, lat))

for _ in range(30):
    lon = random.uniform(-111.0, -104.0)
    lat = random.uniform(41.0, 45.0)
    firestation_locations.append((lon, lat))

firefighters = []
for lon, lat in firestation_locations:
    grid_x = int(np.floor(lon / grid_size) - grid_x_min)
    grid_y = int(np.floor(lat / grid_size) - grid_y_min)
    firefighters.append(FirefighterAgent(grid_x, grid_y, range_cells=2))

# Debugowanie współrzędnych remiz
print("Współrzędne remiz strażackich:")
for i, (lon, lat) in enumerate(firestation_locations):
    print(f"Remiza {i+1}: lon={lon:.4f}, lat={lat:.4f}")

regions = [RegionAgent('Region 1', list(counties.values())[:10]),
           RegionAgent('Region 2', list(counties.values())[10:20]),
           RegionAgent('Region 3', list(counties.values())[20:])]

weather = WeatherAgent()

# ==========================
# 5. Funkcje pomocnicze
# ==========================
def spread_fire(fire, grid, weather, counties, total_fires):
    if len(total_fires) >= 30000:
        return []
    directions = [(-1,0), (1,0), (0,-1), (0,1)]
    new_fires = []
    for dx, dy in directions:
        nx, ny = fire.grid_x + dx, fire.grid_y + dy
        if 0 <= nx < grid_width and 0 <= ny < grid_height:
            if grid[ny, nx] == 1 and random.random() < weather.fire_spread_probability() * 0.5:
                new_fire = FireIncidentAgent(nx, ny, size=1, cause='Spread', start_day=fire.start_day)
                new_fires.append(new_fire)
                if len(total_fires) + len(new_fires) >= 30000:
                    break
    current_county = None
    for county in counties.values():
        for f in county.active_fires:
            if f.grid_x == fire.grid_x and f.grid_y == fire.grid_y:
                current_county = county
                break
        if current_county:
            break
    if current_county and len(total_fires) + len(new_fires) < 30000:
        neighbor_counties = random.sample(list(counties.values()), min(3, len(counties)))
        for neighbor in neighbor_counties:
            if neighbor != current_county and random.random() < 0.15:
                forest_cells = [(f.grid_x, f.grid_y) for f in forests if f.status == 1]
                if forest_cells:
                    x, y = random.choice(forest_cells)
                    new_fire = FireIncidentAgent(x, y, size=1, cause='County Spread', start_day=fire.start_day)
                    new_fires.append(new_fire)
                    if len(total_fires) + len(new_fires) >= 30000:
                        break
    return new_fires

# ==========================
# 6. Symulacja krok po kroku
# ==========================
simulation_days = 365
all_fires = []

for day in range(1, simulation_days + 1):
    print(f"Symulacja: dzień {day}, liczba pożarów: {len(all_fires)}")
    if len(all_fires) >= 30000:
        break
    weather.update()
    for region in regions:
        region.monte_carlo_first_fire(day, weather, all_fires)
    new_fires = []
    for county in counties.values():
        for fire in county.active_fires[:]:
            if fire.active and fire.start_day + fire.duration >= day:
                new_fires.extend(spread_fire(fire, grid, weather, counties, all_fires))
                if len(all_fires) + len(new_fires) >= 30000:
                    break
        if len(all_fires) + len(new_fires) >= 30000:
            break
    for nf in new_fires:
        if len(all_fires) >= 30000:
            break
        county = random.choice(list(counties.values()))
        county.active_fires.append(nf)
        all_fires.append(nf)
    for ff in firefighters:
        for county in counties.values():
            for fire in county.active_fires:
                if fire.active:
                    ff.try_extinguish(fire)
    grid[:, :] = 0
    for f_agent in forests:
        if f_agent.status == 1:
            grid[f_agent.grid_y, f_agent.grid_x] = 1
    for city in cities:
        grid[city.grid_y, city.grid_x] = 3
    for county in counties.values():
        for fire in county.active_fires:
            if fire.active:
                grid[fire.grid_y, fire.grid_x] = 2

# ==========================
# 7. Wizualizacja mapowa (Folium)
# ==========================
# Inicjalizacja mapy
m = folium.Map(
    location=[43.0, -107.5],
    zoom_start=9,
    tiles='OpenStreetMap',
    min_zoom=7,
    max_bounds=True,
    max_bounds_viscosity=1.0,
    dragging=True,
    zoom_control=True
)
bounds = [[41.0, -111.0], [45.0, -104.0]]
m.fit_bounds(bounds)
m.options['maxBounds'] = bounds
m.options['maxBoundsViscosity'] = 1.0

# Oddzielne warstwy dla statycznych obiektów
forests_group = folium.FeatureGroup(name='Forests', show=True)
cities_group = folium.FeatureGroup(name='Cities', show=True)
firestations_group = folium.FeatureGroup(name='Fire Stations', show=True)

# Lasy (statyczne, niski zIndex)
forests_subset = random.sample(forests, min(len(forests), 5500))
for forest in forests_subset:
    if forest.status == 1:
        lon = forest.grid_x * grid_size + grid_x_min * grid_size
        lat = forest.grid_y * grid_size + grid_y_min * grid_size
        if not (41.0 <= lat <= 45.0 and -111.0 <= lon <= -104.0):
            continue
        folium.CircleMarker(
            location=[lat, lon],
            radius=2,
            fill=True,
            color='green',
            fill_color='green',
            fill_opacity=0.4,
            weight=1,
            popup=f"Forest at ({forest.grid_x}, {forest.grid_y})",
            zIndexOffset=100  # Niski zIndex dla lasów
        ).add_to(forests_group)

# Miasta (statyczne, średni zIndex)
for city in cities:
    lon = city.grid_x * grid_size + grid_x_min * grid_size
    lat = city.grid_y * grid_size + grid_y_min * grid_size
    if not (41.0 <= lat <= 45.0 and -111.0 <= lon <= -104.0):
        continue
    folium.CircleMarker(
        location=[lat, lon],
        radius=5,
        fill=True,
        color='blue',
        fill_color='blue',
        fill_opacity=0.8,
        weight=1,
        popup=f"City at ({city.grid_x}, {city.grid_y})",
        zIndexOffset=300  # Średni zIndex dla miast
    ).add_to(cities_group)

# Remizy strażackie (statyczne, wyższy zIndex)
for ff in firefighters:
    lon = ff.grid_x * grid_size + grid_x_min * grid_size
    lat = ff.grid_y * grid_size + grid_y_min * grid_size
    if not (41.0 <= lat <= 45.0 and -111.0 <= lon <= -104.0):
        continue
    folium.CircleMarker(
        location=[lat, lon],
        radius=2,
        fill=True,
        color='black',
        fill_color='black',
        fill_opacity=0.8,
        weight=1,
        popup=f"Fire Station at ({ff.grid_x}, {ff.grid_y})",
        zIndexOffset=400  # Wyższy zIndex dla remiz
    ).add_to(firestations_group)

# Dodanie statycznych warstw do mapy
forests_group.add_to(m)
cities_group.add_to(m)
firestations_group.add_to(m)

# Pożary (dynamiczne, najwyższy zIndex)
features = []
for county in counties.values():
    for fire in county.active_fires:
        if fire.duration < 2:  # Pomijaj pożary trwające krócej niż 2 dni
            continue
        lon = fire.grid_x * grid_size + grid_x_min * grid_size
        lat = fire.grid_y * grid_size + grid_y_min * grid_size
        print(f"Pożar: lon={lon}, lat={lat}, start_day={fire.start_day}, duration={fire.duration}, size={fire.size}")
        if not (41.0 <= lat <= 45.0 and -111.0 <= lon <= -104.0):
            continue
        duration_days = max(5, fire.duration)  # Minimum 5 dni
        max_radius = min(50, max(2, fire.size / 3))  # Maksymalny promień, minimum 2 piksele
        for day in range(duration_days + 1):  # Liczenie od 0 do duration_days
            current_date = datetime(2015, 1, 1) + timedelta(days=fire.start_day - 1 + day)
            timestamp = current_date.isoformat()
            if day <= duration_days / 2:
                radius = max_radius * (day / (duration_days / 2))
            else:
                radius = max_radius * ((duration_days - day) / (duration_days / 2))
            radius = max(2, radius)  # Minimalny promień 2 piksele dla widoczności
            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': [lon, lat]
                },
                'properties': {
                    'time': timestamp,
                    'popup': f"Fire: {fire.cause}<br>Size: {fire.size} acres<br>Day: {day + 1}/{duration_days + 1}",
                    'icon': 'circle',
                    'iconstyle': {
                        'fillColor': 'red',
                        'color': 'red',
                        'fillOpacity': 0.6,
                        'radius': radius,
                        'weight': 1,
                        'zIndex': 2000  # Najwyższy zIndex dla pożarów
                    }
                }
            }
            features.append(feature)

# Dodanie pożarów jako dynamicznej warstwy
geojson = {'type': 'FeatureCollection', 'features': features}
TimestampedGeoJson(
    geojson,
    period='P1D',
    duration='P1D',
    auto_play=True,
    loop=False,
    max_speed=5,
    transition_time=500,
    add_last_point=False
).add_to(m)

# Dodanie kontroli warstw
folium.LayerControl().add_to(m)

# Debugowanie liczby remiz
print(f"Liczba wygenerowanych remiz: {len(firefighters)}")
print(f"Liczba punktów w features dla remiz: {sum(1 for ff in firefighters if 41.0 <= (ff.grid_y * grid_size + grid_y_min * grid_size) <= 45.0 and -111.0 <= (ff.grid_x * grid_size + grid_x_min * grid_size) <= -104.0)}")

# Zapisz mapę
m.save('wy_fires_simulation_365_dynamic_size_corrected.html')
print(f"Symulacja zapisana jako 'wy_fires_simulation_365_dynamic_size_corrected.html'. Liczba pożarów: {len(all_fires)}")