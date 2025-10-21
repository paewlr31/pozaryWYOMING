import pandas as pd
import numpy as np
import folium
from folium.plugins import TimestampedGeoJson
from datetime import datetime, timedelta
import random
from roads import road_locations
from rivers import river_locations
from mountains import mountain_locations

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
grid = np.zeros((grid_height, grid_width), dtype=int)  # 0=brak, 1=las, 2=pożar, 3=miasto, 4=rzeka, 5=góry, 6=droga

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
        self.ignition_days = []  # lista dni startu zapalenia

class CityAgent:
    def __init__(self, grid_x, grid_y):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.status = 3  # 3=miasto

class RiverAgent:
    def __init__(self, points):
        self.points = points  # Lista krotek (grid_x, grid_y)
        self.status = 4  # 4=rzeka

class MountainAgent:
    def __init__(self, min_x, max_x, min_y, max_y):
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.status = 5  # 5=góry

class RoadAgent:
    def __init__(self, points, weight, name=""):
        self.points = points  # Lista krotek (grid_x, grid_y)
        self.status = 6  # 6=droga
        self.weight = weight  # 1=autostrada międzystanowa, 2=autostrada stanowa, 3=powiatowa, 4=gminna
        self.name = name  # Nazwa drogi

class FireIncidentAgent:
    def __init__(self, grid_x, grid_y, size, cause, start_day):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.size = random.uniform(10, 100) if size == 1 else size
        self.cause = cause
        self.active = True
        self.start_day = start_day
        self.duration = max(5, int(np.random.normal(avg_duration, std_duration)))

class FirefighterAgent:
    def __init__(self, grid_x, grid_y, effectiveness=0.7, range_cells=2):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.effectiveness = effectiveness
        self.range_cells = range_cells

    def try_extinguish(self, fire):
        if abs(self.grid_x - fire.grid_x) <= self.range_cells and abs(self.grid_y - fire.grid_y) <= self.range_cells:
            if random.random() < self.effectiveness:
                fire.active = False
                fire.duration = min(fire.duration, max(1, int(fire.duration * 0.5)))

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
        if len(total_fires) >= 30000:
            return
        county = random.choice(self.counties)
        forest_cells = [(f.grid_x, f.grid_y) for f in forests if f.status == 1]
        if forest_cells and random.random() < 0.05:
            x, y = random.choice(forest_cells)
            if random.random() < weather.fire_spread_probability():
                fire = FireIncidentAgent(x, y, size=1, cause='Monte Carlo', start_day=day)
                county.active_fires.append(fire)
                total_fires.append(fire)
                forest = next((fg for fg in forests if fg.grid_x == x and fg.grid_y == y), None)
                if forest is not None:
                    forest.ignition_days.append(day)

# ==========================
# 4. Inicjalizacja agentów
# ==========================
counties = {}
forests = []
cities = []
firefighters = []
rivers = []
mountains = []
roads = []

for fips, group in df.groupby('FIPS_CODE'):
    county_name = group['FIPS_NAME'].iloc[0]
    forest_area = len(group)
    fire_history = group['FIRE_YEAR'].unique().tolist()
    county_agent = CountyAgent(fips, county_name, forest_area, fire_history)
    counties[fips] = county_agent
    for _, row in group.iterrows():
        f_agent = ForestAgent(row['grid_x'] - grid_x_min, row['grid_y'] - grid_y_min)
        forests.append(f_agent)

# Dodanie 500 lasów w Yellowstone
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

# Lista miast Wyoming
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

# Inicjalizacja rzek
rivers = []
for river_points in river_locations:
    grid_points = []
    for lon, lat in river_points:
        grid_x = int(np.floor(lon / grid_size) - grid_x_min)
        grid_y = int(np.floor(lat / grid_size) - grid_y_min)
        if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
            grid_points.append((grid_x, grid_y))
    if grid_points:
        expanded_points = []
        for x, y in grid_points:
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < grid_width and 0 <= ny < grid_height:
                        expanded_points.append((nx, ny))
        rivers.append(RiverAgent(expanded_points))

# Inicjalizacja dróg
roads = []
for road in road_locations:
    grid_points = []
    for lon, lat in road['points']:
        grid_x = int(np.floor(lon / grid_size) - grid_x_min)
        grid_y = int(np.floor(lat / grid_size) - grid_y_min)
        if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
            grid_points.append((grid_x, grid_y))
    if grid_points:
        expanded_points = []
        for x, y in grid_points:
            range_x = 2 if road['weight'] == 1 else 1
            range_y = 2 if road['weight'] == 1 else 1
            for dx in range(-range_x, range_x + 1):
                for dy in range(-range_y, range_y + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < grid_width and 0 <= ny < grid_height:
                        expanded_points.append((nx, ny))
        roads.append(RoadAgent(expanded_points, road['weight'], road['name']))

# Debugowanie liczby dróg
print(f"Liczba dodanych dróg: {len(roads)}")
for road in roads:
    print(f"Droga: {road.name}, Typ: {road.weight}, Punkty: {len(road.points)}")

# Inicjalizacja gór
mountains = []
for mountain in mountain_locations:
    min_x = int(np.floor(mountain['lon_min'] / grid_size) - grid_x_min)
    max_x = int(np.floor(mountain['lon_max'] / grid_size) - grid_x_min)
    min_y = int(np.floor(mountain['lat_min'] / grid_size) - grid_y_min)
    max_y = int(np.floor(mountain['lat_max'] / grid_size) - grid_y_min)
    min_x = max(0, min_x)
    max_x = min(grid_width - 1, max_x)
    min_y = max(0, min_y)
    max_y = min(grid_height - 1, max_y)
    mountains.append(MountainAgent(min_x, max_x, min_y, max_y))

# Generowanie remiz strażackich
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
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    new_fires = []
    for dx, dy in directions:
        nx, ny = fire.grid_x + dx, fire.grid_y + dy
        if 0 <= nx < grid_width and 0 <= ny < grid_height:
            spread_prob = weather.fire_spread_probability() * 0.5
            if grid[ny, nx] == 4:  # Rzeka
                spread_prob *= 0.3  # 70% redukcja
            elif grid[ny, nx] == 5:  # Góry
                spread_prob *= 1.2  # 20% zwiększenie
            elif grid[ny, nx] == 6:  # Droga
                spread_prob *= 0.9  # 50% redukcja
            if grid[ny, nx] == 1 and random.random() < spread_prob:
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
                    spread_prob = weather.fire_spread_probability()
                    if grid[y, x] == 4:
                        spread_prob *= 0.3
                    elif grid[y, x] == 5:
                        spread_prob *= 1.2
                    elif grid[y, x] == 6:
                        spread_prob *= 1.5
                    if random.random() < spread_prob:
                        new_fire = FireIncidentAgent(x, y, size=1, cause='County Spread', start_day=fire.start_day)
                        new_fires.append(new_fire)
                        if len(total_fires) + len(new_fires) >= 30000:
                            break
    for nf in new_fires:
        forest = next((fg for fg in forests if fg.grid_x == nf.grid_x and nf.grid_y == nf.grid_y), None)
        if forest is not None:
            forest.ignition_days.append(nf.start_day)
    return new_fires

# ==========================
# 6. Symulacja krok po kroku
# ==========================
simulation_days = 365
all_fires = []

# Ustawienie rzek, gór i dróg na siatce przed symulacją
for river in rivers:
    for x, y in river.points:
        if 0 <= x < grid_width and 0 <= y < grid_height:
            grid[y, x] = 4
for mountain in mountains:
    for x in range(mountain.min_x, mountain.max_x + 1):
        for y in range(mountain.min_y, mountain.max_y + 1):
            if 0 <= x < grid_width and 0 <= y < grid_height:
                grid[y, x] = 5
for road in roads:
    for x, y in road.points:
        if 0 <= x < grid_width and 0 <= y < grid_height:
            grid[y, x] = 6

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
    # Aktualizuj siatkę
    grid[:, :] = 0
    for river in rivers:
        for x, y in river.points:
            if 0 <= x < grid_width and 0 <= y < grid_height:
                grid[y, x] = 4
    for mountain in mountains:
        for x in range(mountain.min_x, mountain.max_x + 1):
            for y in range(mountain.min_y, mountain.max_y + 1):
                if 0 <= x < grid_width and 0 <= y < grid_height:
                    grid[y, x] = 5
    for road in roads:
        for x, y in road.points:
            if 0 <= x < grid_width and 0 <= y < grid_height:
                grid[y, x] = 6
    for f_agent in forests:
        if f_agent.status == 1:
            if 0 <= f_agent.grid_y < grid_height and 0 <= f_agent.grid_x < grid_width:
                if grid[f_agent.grid_y, f_agent.grid_x] not in [4, 5, 6]:
                    grid[f_agent.grid_y, f_agent.grid_x] = 1
    for city in cities:
        if 0 <= city.grid_y < grid_height and 0 <= city.grid_x < grid_width:
            grid[city.grid_y, city.grid_x] = 3
    for county in counties.values():
        for fire in county.active_fires:
            if fire.active:
                if 0 <= fire.grid_y < grid_height and 0 <= fire.grid_x < grid_width:
                    grid[fire.grid_y, fire.grid_x] = 2

# ==========================
# 7. Wizualizacja mapowa (Folium)
# ==========================
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
forests_group = folium.FeatureGroup(name='Forests (statyczne)', show=True)
cities_group = folium.FeatureGroup(name='Cities', show=True)
firestations_group = folium.FeatureGroup(name='Fire Stations', show=True)
rivers_group = folium.FeatureGroup(name='Rivers', show=True)
mountains_group = folium.FeatureGroup(name='Mountains', show=True)
roads_group = folium.FeatureGroup(name='Roads', show=True)

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
            zIndexOffset=100
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
        zIndexOffset=800
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
        zIndexOffset=400
    ).add_to(firestations_group)

# Rzeki (statyczne, bardzo niski zIndex)
for river in rivers:
    geo_points = [(y * grid_size + grid_y_min * grid_size, x * grid_size + grid_x_min * grid_size) for x, y in river.points]
    if geo_points:
        folium.PolyLine(
            locations=geo_points,
            color='blue',
            weight=2,
            opacity=0.5,
            popup="River",
            zIndexOffset=50
        ).add_to(rivers_group)

# Góry (statyczne, bardzo niski zIndex)
for mountain in mountains:
    bounds = [
        [mountain.min_y * grid_size + grid_y_min * grid_size, mountain.min_x * grid_size + grid_x_min * grid_size],
        [mountain.max_y * grid_size + grid_y_min * grid_size, mountain.max_x * grid_size + grid_x_min * grid_size]
    ]
    folium.Rectangle(
        bounds=bounds,
        color='gray',
        fill=True,
        fill_color='gray',
        fill_opacity=0.3,
        weight=1,
        popup="Mountains",
        zIndexOffset=50
    ).add_to(mountains_group)

# Drogi (statyczne, niski zIndex)
road_colors = {
    1: '#FFD700',  # Złoty dla autostrad międzystanowych
    2: '#FFA500',  # Pomarańczowy dla autostrad stanowych
    3: '#FFFF99',  # Jasnożółty dla dróg powiatowych
    4: '#FFFACD'   # Bardzo jasnożółty dla dróg gminnych
}
for road in roads:
    geo_points = [(y * grid_size + grid_y_min * grid_size, x * grid_size + grid_x_min * grid_size) for x, y in road.points]
    if geo_points:
        folium.PolyLine(
            locations=geo_points,
            color=road_colors[road.weight],
            weight=3 if road.weight in [1, 2] else 2,
            opacity=0.7,
            popup=f"Road: {road.name} (Type {road.weight})",
            zIndexOffset=75
        ).add_to(roads_group)

# Dodanie statycznych warstw do mapy
forests_group.add_to(m)
cities_group.add_to(m)
firestations_group.add_to(m)
rivers_group.add_to(m)
mountains_group.add_to(m)
roads_group.add_to(m)

# Pożary (dynamiczne, najwyższy zIndex)
features = []
for county in counties.values():
    for fire in county.active_fires:
        if fire.duration < 2:
            continue
        lon = fire.grid_x * grid_size + grid_x_min * grid_size
        lat = fire.grid_y * grid_size + grid_y_min * grid_size
        print(f"Pożar: lon={lon}, lat={lat}, start_day={fire.start_day}, duration={fire.duration}, size={fire.size}")
        if not (41.0 <= lat <= 45.0 and -111.0 <= lon <= -104.0):
            continue
        duration_days = max(5, fire.duration)
        max_radius = min(25, max(1, fire.size / 6))
        for day in range(duration_days + 1):
            current_date = datetime(2015, 1, 1) + timedelta(days=fire.start_day - 1 + day)
            timestamp = current_date.isoformat()
            if day <= duration_days / 2:
                radius = max_radius * (day / (duration_days / 2))
            else:
                radius = max_radius * ((duration_days - day) / (duration_days / 2))
            radius = max(1, radius)
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
                        'zIndex': 2000
                    }
                }
            }
            features.append(feature)

# Dynamiczne lasy (pomarańczowe po zapaleniu)
for forest in forests:
    if forest.status != 1:
        continue
    lon = forest.grid_x * grid_size + grid_x_min * grid_size
    lat = forest.grid_y * grid_size + grid_y_min * grid_size
    if not (41.0 <= lat <= 45.0 and -111.0 <= lon <= -104.0):
        continue
    for ign_day in forest.ignition_days:
        for d in range(0, 90):
            day_num = ign_day - 1 + d
            if day_num < 0 or day_num >= simulation_days:
                continue
            current_date = datetime(2015, 1, 1) + timedelta(days=day_num)
            timestamp = current_date.isoformat()
            feature = {
                'type': 'Feature',
                'geometry': {'type': 'Point', 'coordinates': [lon, lat]},
                'properties': {
                    'time': timestamp,
                    'popup': f"Burning forest (since day {ign_day}) - day {d+1}/90",
                    'icon': 'circle',
                    'iconstyle': {
                        'fillColor': 'orange',
                        'color': 'orange',
                        'fillOpacity': 0.8,
                        'radius': 3,
                        'weight': 1,
                        'zIndex': 1500
                    }
                }
            }
            features.append(feature)
        restore_day = ign_day + 90
        if 1 <= restore_day <= simulation_days:
            current_date = datetime(2015, 1, 1) + timedelta(days=restore_day - 1)
            timestamp = current_date.isoformat()
            feature = {
                'type': 'Feature',
                'geometry': {'type': 'Point', 'coordinates': [lon, lat]},
                'properties': {
                    'time': timestamp,
                    'popup': f"Forest restored (after ignition day {ign_day})",
                    'icon': 'circle',
                    'iconstyle': {
                        'fillColor': 'green',
                        'color': 'green',
                        'fillOpacity': 0.6,
                        'radius': 2,
                        'weight': 1,
                        'zIndex': 100
                    }
                }
            }
            features.append(feature)

# Dodanie pożarów i dynamicznych lasów jako GeoJSON
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

# Debugowanie
print(f"Liczba wygenerowanych remiz: {len(firefighters)}")
print(f"Liczba punktów w features (pożary + dynamiczne lasy): {len(features)}")
print(f"Liczba rzek: {len(rivers)}")
print(f"Liczba obszarów górskich: {len(mountains)}")
print(f"Liczba dróg: {len(roads)}")

# Zapisz mapę
m.save('wy_fires_simulation_365_with_100rivers_10mountains_150roads.html')
print(f"Symulacja zapisana jako 'wy_fires_simulation_365_with_100rivers_10mountains_150roads.html'. Liczba pożarów: {len(all_fires)}")