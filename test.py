import pandas as pd
import numpy as np
import folium
from folium.plugins import TimestampedGeoJson
from datetime import datetime, timedelta
import random
import logging
from scipy.spatial import cKDTree

# Pliki z danymi zewnętrznymi (założono, że istnieją)
from roads import road_locations
from rivers import river_locations
from mountains import mountain_locations

# ==========================
# Konfiguracja logowania
# ==========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simulation_logs.txt', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# ==========================
# 1. Wczytanie danych
# ==========================
df = pd.read_csv('wildfires_wy.csv', low_memory=False)
df = df[['FIRE_YEAR', 'DISCOVERY_DOY', 'CONT_DOY', 'LATITUDE', 'LONGITUDE', 'FIRE_SIZE', 'STAT_CAUSE_DESCR', 'STATE', 'COUNTY', 'FIPS_CODE', 'FIPS_NAME']]

# Filtr dla Wyoming i brakujących danych
df = df[df['STATE'] == 'WY']
df = df[df['LATITUDE'].notna() & df['LONGITUDE'].notna()]

# Poprawka dla DURATION
df['DURATION'] = df['CONT_DOY'].fillna(df['DISCOVERY_DOY'] + 5) - df['DISCOVERY_DOY']
df['DURATION'] = df['DURATION'].apply(lambda x: max(1, x))
avg_duration = df['DURATION'].mean()
std_duration = df['DURATION'].std()

# Średni i odchylenie rozmiaru pożaru
avg_size = df['FIRE_SIZE'].mean()
std_size = df['FIRE_SIZE'].std()

# Analiza przyczyn pożarów
human_causes = ['Arson', 'Campfire', 'Children', 'Equipment Use', 'Fireworks', 'Railroad', 'Smoking', 'Debris Burning', 'Powerline']
df['CAUSE_TYPE'] = df['STAT_CAUSE_DESCR'].apply(
    lambda x: 'Human' if x in human_causes else 'Lightning' if x == 'Lightning' else 'Other'
)
# Rozkład przyczyn - ZMIANA: Podwojenie udziału Human i Lightning
human_fire_ratio = 0.84  # ZMIANA: z 0.42 na 0.84
lightning_fire_ratio = 0.66  # ZMIANA: z 0.33 na 0.66
other_fire_ratio = 0.10  # ZMIANA: z 0.25 na 0.10
# Normalizacja proporcji
total = human_fire_ratio + lightning_fire_ratio + other_fire_ratio
human_fire_ratio /= total  # ≈ 0.525
lightning_fire_ratio /= total  # ≈ 0.4125
other_fire_ratio /= total  # ≈ 0.0625
logger.info("Założony rozkład przyczyn: Human=%.2f%%, Lightning=%.2f%%, Other=%.2f%%",
            human_fire_ratio * 100, lightning_fire_ratio * 100, other_fire_ratio * 100)

# Debugowanie danych
logger.info("Min DURATION: %s", df['DURATION'].min())
logger.info("Max DURATION: %s", df['DURATION'].max())
logger.info("Avg DURATION: %s", avg_duration)
logger.info("Std DURATION: %s", std_duration)
logger.info("Avg FIRE_SIZE: %s", avg_size)
logger.info("Std FIRE_SIZE: %s", std_size)

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
        self.human_activity = HumanActivityAgent(
            fips,
            tourism_level=random.uniform(0.2, 0.8),
            agriculture_level=random.uniform(0.1, 0.6),
            transport_level=random.uniform(0.2, 0.7),
            industry_level=random.uniform(0.1, 0.5)
        )

class HumanActivityAgent:
    def __init__(self, county_fips, tourism_level=0.5, agriculture_level=0.3, transport_level=0.4, industry_level=0.2):
        self.county_fips = county_fips
        self.tourism_level = tourism_level
        self.agriculture_level = agriculture_level
        self.transport_level = transport_level
        self.industry_level = industry_level
        self.fire_risk_multiplier = self.calculate_fire_risk()

    def calculate_fire_risk(self):
        return 1.0 + (self.tourism_level * 2.0 + self.agriculture_level * 1.2 +
                      self.transport_level * 0.8 + self.industry_level * 0.4)

    def update(self):
        self.tourism_level = np.clip(self.tourism_level + np.random.normal(0, 0.05), 0, 1)
        self.agriculture_level = np.clip(self.agriculture_level + np.random.normal(0, 0.03), 0, 1)
        self.transport_level = np.clip(self.transport_level + np.random.normal(0, 0.02), 0, 1)
        self.industry_level = np.clip(self.industry_level + np.random.normal(0, 0.02), 0, 1)
        self.fire_risk_multiplier = self.calculate_fire_risk()

class ForestAgent:
    def __init__(self, grid_x, grid_y, density=0.8):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.status = 1 if random.random() < density else 0
        self.ignition_days = []

class CityAgent:
    def __init__(self, grid_x, grid_y):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.status = 3

class RiverAgent:
    def __init__(self, points):
        self.points = points
        self.status = 4

class MountainAgent:
    def __init__(self, min_x, max_x, min_y, max_y):
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.status = 5

class RoadAgent:
    def __init__(self, points, weight, name=""):
        self.points = points
        self.status = 6
        self.weight = weight
        self.name = name

class FireIncidentAgent:
    def __init__(self, grid_x, grid_y, size, cause, start_day):
        self.grid_x = grid_x
        self.grid_y = grid_y
        if size == 1 or size == 0.5:
            base_size = np.random.lognormal(np.log(avg_size), 0.3)
            while base_size > 255:
                base_size = np.random.lognormal(np.log(avg_size), 0.3)
            if cause == 'Spread' or cause == 'County Spread':
                self.size = min(150, base_size * 0.25)
            elif cause == 'Lightning':
                self.size = min(150, base_size * 2.4)
            elif cause == 'Other':
                self.size = min(150, base_size * 2.05)
            else:  # Human
                self.size = min(150, base_size * 2.25)
        else:
            self.size = size
        self.cause = cause
        self.active = True
        self.start_day = start_day
        self.duration = max(1, int(np.random.normal(avg_duration, std_duration * 0.3)))
        self.assigned_resources = []
        self.spread_influences = []
        logger.info("Nowy pożar w (%s, %s), przyczyna: %s, rozmiar: %.2f akrów", 
                    grid_x, grid_y, cause, self.size)
        sample_sizes = [np.random.lognormal(np.log(avg_size), 0.3) for _ in range(100)]
        sample_sizes = [s for s in sample_sizes if s <= 75]
        logger.debug("Statystyki rozmiaru: bieżący=%.2f, min=%.2f, max=%.2f, avg=%.2f, avg_size=%.2f", 
                     self.size, min(sample_sizes) if sample_sizes else 0, 
                     max(sample_sizes) if sample_sizes else 0, 
                     np.mean(sample_sizes) if sample_sizes else 0, avg_size)

class ResourceAgent:
    def __init__(self, resource_type, grid_x, grid_y, home_station, speed, range_cells, effectiveness):
        self.resource_type = resource_type
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.home_station = home_station
        self.speed = speed
        self.range_cells = range_cells
        self.effectiveness = effectiveness
        self.status = 'idle'
        self.target_fire = None
        self.days_traveling = 0
        self.days_fighting = 0
        self.path = []
        self.returning = False

    def move_towards(self, target_x, target_y, day, grid):
        if self.status not in ['moving', 'returning'] or (self.returning and self.target_fire is None):
            return
        target = (self.home_station.grid_x, self.home_station.grid_y) if self.returning else (target_x, target_y)
        distance = np.sqrt((target_x - self.grid_x) ** 2 + (target_y - self.grid_y) ** 2)
        speed = self.speed
        if self.resource_type == 'truck' and 0 <= self.grid_y < grid_height and 0 <= self.grid_x < grid_width:
            if grid[self.grid_y, self.grid_x] == 6:
                speed *= 1.5
        if distance <= speed:
            self.grid_x, self.grid_y = target
            self.path.append((self.grid_x, self.grid_y))
            if self.returning:
                self.status = 'idle'
                self.target_fire = None
                self.returning = False
                self.path = []
                logger.info("%s z remizy (%s, %s) wrócił do bazy.", self.resource_type, self.home_station.grid_x, self.home_station.grid_y)
            else:
                self.status = 'fighting'
                logger.info("%s z remizy (%s, %s) dotarł do pożaru (%s, %s).", self.resource_type, self.home_station.grid_x, self.home_station.grid_y, target_x, target_y)
        else:
            angle = np.arctan2(target_y - self.grid_y, target_x - self.grid_x)
            dx = int(round(speed * np.cos(angle)))
            dy = int(round(speed * np.sin(angle)))
            new_x = np.clip(self.grid_x + dx, 0, grid_width - 1)
            new_y = np.clip(self.grid_y + dy, 0, grid_height - 1)
            self.grid_x, self.grid_y = new_x, new_y
            self.path.append((self.grid_x, self.grid_y))
            self.days_traveling += 1
            logger.info("%s z remizy (%s, %s) porusza się do (%s, %s), aktualna pozycja: (%s, %s), dystans pozostały: %.2f.", 
                        self.resource_type, self.home_station.grid_x, self.home_station.grid_y, target_x, target_y, self.grid_x, self.grid_y, distance)

    def try_extinguish(self, fire, day):
        if self.status == 'fighting' and self.target_fire == fire and fire.active:
            self.days_fighting += 1
            extinguish_prob = self.effectiveness
            if random.random() < extinguish_prob:
                fire.active = False
                fire.duration = min(fire.duration, max(1, int(fire.duration * 0.5)))
                logger.info("%s z remizy (%s, %s) ugasił pożar (%s, %s).", self.resource_type, self.home_station.grid_x, self.home_station.grid_y, fire.grid_x, fire.grid_y)
                self.status = 'returning'
                self.returning = True
            else:
                logger.info("%s z remizy (%s, %s) walczy z pożarem (%s, %s), dni gaszenia: %s.", 
                            self.resource_type, self.home_station.grid_x, self.home_station.grid_y, fire.grid_x, fire.grid_y, self.days_fighting)

class FirefighterAgent:
    def __init__(self, grid_x, grid_y, effectiveness=0.7, range_cells=2):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.effectiveness = effectiveness
        self.range_cells = range_cells
        self.resources = []
        num_resources = random.randint(2, 5)
        has_airplane = random.random() < 0.3
        for i in range(num_resources):
            if has_airplane and i == 0:
                resource_type = 'airplane'
                speed = random.uniform(10, 20)
                range_cells_resource = 50
                effectiveness_resource = 0.9
            else:
                resource_type = 'truck'
                speed = random.uniform(2, 5)
                range_cells_resource = 20
                effectiveness_resource = 0.7
            resource = ResourceAgent(resource_type, grid_x, grid_y, self, speed, range_cells_resource, effectiveness_resource)
            self.resources.append(resource)
            logger.info("[Inicjalizacja] Remiza (%s, %s) dodała %s (prędkość: %.2f, zasięg: %s, skuteczność: %s).", 
                        grid_x, grid_y, resource_type, speed, range_cells_resource, effectiveness_resource)

    def dispatch_resources(self, fire, day, grid):
        if not fire.active or fire in [r.target_fire for r in self.resources if r.target_fire]:
            return
        current_day = day - fire.start_day + 1
        max_resources = min(5, 1 + int(current_day / 3))
        if fire.size > 100:
            max_resources = min(5, max_resources + 1)
        available_resources = [r for r in self.resources if r.status == 'idle' and np.sqrt((fire.grid_x - self.grid_x) ** 2 + (fire.grid_y - self.grid_y) ** 2) <= r.range_cells]
        num_to_dispatch = min(max_resources - len(fire.assigned_resources), len(available_resources))
        for resource in random.sample(available_resources, num_to_dispatch):
            if (resource.resource_type == 'airplane' and (fire.size > 50 or current_day > 5)) or resource.resource_type == 'truck':
                resource.status = 'moving'
                resource.target_fire = fire
                resource.path = [(self.grid_x, self.grid_y)]
                fire.assigned_resources.append(resource)
                logger.info("[Dzień %s] Wysyłanie %s z remizy (%s, %s) do pożaru (%s, %s), rozmiar: %.2f, dzień pożaru: %s.", 
                            day, resource.resource_type, self.grid_x, self.grid_y, fire.grid_x, fire.grid_y, fire.size, current_day)

    def try_extinguish(self, fire, day):
        distance = np.sqrt((self.grid_x - fire.grid_x) ** 2 + (self.grid_y - fire.grid_y) ** 2)
        if distance <= self.range_cells and fire.active:
            if random.random() < self.effectiveness:
                fire.active = False
                fire.duration = min(fire.duration, max(1, int(fire.duration * 0.5)))
                logger.info("[Dzień %s] Remiza (%s, %s) ugasiła pożar (%s, %s) bezpośrednio.", 
                            day, self.grid_x, self.grid_y, fire.grid_x, fire.grid_y)
            else:
                logger.info("[Dzień %s] Remiza (%s, %s) próbuje ugasić pożar (%s, %s).", 
                            day, self.grid_x, self.grid_y, fire.grid_x, fire.grid_y)

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
        if len(total_fires) >= 2000:
            return
        county = random.choice(self.counties)
        forest_cells = [(f.grid_x, f.grid_y) for f in forests if f.status == 1]
        if forest_cells:
            base_ignition_prob = 100000.0 * county.human_activity.fire_risk_multiplier
            real_coords = [(row['grid_x'] - grid_x_min, row['grid_y'] - grid_y_min) for _, row in df[df['FIRE_YEAR'] == 2006].iterrows()]
            if real_coords:
                tree = cKDTree(real_coords)
                distances, _ = tree.query(forest_cells, k=1)
                weights = np.exp(-distances / 0.00000005)
                weights /= weights.sum()
                x, y = forest_cells[np.random.choice(len(forest_cells), p=weights)]
            else:
                x, y = random.choice(forest_cells)
            if random.random() < base_ignition_prob:
                if random.random() < weather.fire_spread_probability() and grid[y, x] == 1:
                    rand = random.random()
                    if rand < 0.525:
                        cause = 'Human'
                    elif rand < 0.9375:
                        cause = 'Lightning'
                    else:
                        cause = 'Other'
                    fire = FireIncidentAgent(x, y, size=1, cause=cause, start_day=day)
                    # Sprawdzanie bliskości rzek, gór i dróg
                    influences = ['WeatherAgent', 'HumanActivityAgent']
                    def is_near_agent(x, y, agent_type, radius=2):
                        if agent_type == 'RiverAgent':
                            for river in rivers:
                                for rx, ry in river.points:
                                    if abs(rx - x) <= radius and abs(ry - y) <= radius:
                                        return True
                        elif agent_type == 'MountainAgent':
                            for mountain in mountains:
                                if mountain.min_x <= x <= mountain.max_x and mountain.min_y <= y <= mountain.max_y:
                                    return True
                        elif agent_type == 'RoadAgent':
                            for road in roads:
                                for rx, ry in road.points:
                                    if abs(rx - x) <= radius and abs(ry - y) <= radius:
                                        return True
                        return False
                    if is_near_agent(x, y, 'RiverAgent'):
                        influences.append('RiverAgent')
                    if is_near_agent(x, y, 'MountainAgent'):
                        influences.append('MountainAgent')
                    if is_near_agent(x, y, 'RoadAgent'):
                        influences.append('RoadAgent')
                    fire.spread_influences = influences
                    county.active_fires.append(fire)
                    total_fires.append(fire)
                    forest = next((fg for fg in forests if fg.grid_x == x and fg.grid_y == y), None)
                    if forest is not None and forest.status == 1:
                        forest.ignition_days.append(day)
                        logger.info("Zapłon lasu w (%s, %s), dzień: %s, powiat: %s, przyczyna: %s, wpływy: %s", 
                                    x, y, day, county.name, cause, influences)
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

yellowstone_lon_min, yellowstone_lon_max = -111.0, -109.8
yellowstone_lat_min, yellowstone_lat_max = 44.1, 45.0
yellowstone_density = 0.8
num_yellowstone_forests = 50
for _ in range(num_yellowstone_forests):
    lon = random.uniform(yellowstone_lon_min, yellowstone_lon_max)
    lat = random.uniform(yellowstone_lat_min, yellowstone_lat_max)
    grid_x = int(np.floor(lon / grid_size) - grid_x_min)
    grid_y = int(np.floor(lat / grid_size) - grid_y_min)
    if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
        f_agent = ForestAgent(grid_x, grid_y, density=yellowstone_density)
        forests.append(f_agent)

yellowstone_forests = [f for f in forests if (yellowstone_lon_min <= (f.grid_x + grid_x_min) * grid_size <= yellowstone_lon_max and
                                              yellowstone_lat_min <= (f.grid_y + grid_y_min) * grid_size <= yellowstone_lat_max)]
yellowstone_active_forests = [f for f in yellowstone_forests if f.status == 1]
logger.info("Liczba wszystkich lasów dodanych w Yellowstone: %s", len(yellowstone_forests))
logger.info("Liczba aktywnych lasów (status=1) w Yellowstone: %s", len(yellowstone_active_forests))

city_locations = [
    (-105.5022, 41.1399), (-104.8202, 41.1359), (-106.3197, 42.8641), (-105.9399, 44.2910),
    (-109.2490, 41.6372), (-104.8253, 44.7970), (-108.2023, 43.0238), (-110.3326, 43.4799),
    (-105.5911, 44.3483), (-107.1357, 43.6599), (-104.1827, 42.8494), (-108.7373, 42.7552),
    (-110.9632, 41.3114), (-107.2009, 44.0802), (-104.6097, 41.6322), (-108.3897, 42.8347),
    (-104.2047, 44.9083), (-106.4064, 44.6872), (-104.0555, 42.0625), (-110.0752, 41.7911),
    (-105.7455, 42.0978), (-107.5437, 41.5875), (-104.1389, 44.2720), (-108.8964, 42.0663),
    (-106.6392, 42.7475)
]
cities = []
for lon, lat in city_locations:
    grid_x = int(np.floor(lon / grid_size) - grid_x_min)
    grid_y = int(np.floor(lat / grid_size) - grid_y_min)
    cities.append(CityAgent(grid_x, grid_y))

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

logger.info("Liczba dodanych dróg: %s", len(roads))
for road in roads:
    logger.info("Droga: %s, Typ: %s, Punkty: %s", road.name, road.weight, len(road.points))

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

logger.info("Współrzędne remiz strażackich:")
for i, (lon, lat) in enumerate(firestation_locations):
    logger.info("Remiza %s: lon=%.4f, lat=%.4f", i+1, lon, lat)

regions = [RegionAgent('Region 1', list(counties.values())[:10]),
           RegionAgent('Region 2', list(counties.values())[10:20]),
           RegionAgent('Region 3', list(counties.values())[20:])]

weather = WeatherAgent()

# ==========================
# 5. Funkcje pomocnicze
# ==========================
def spread_fire(fire, grid, weather, counties, total_fires):
    if len(total_fires) >= 2000:
        return []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    new_fires = []
    current_county = None
    for county in counties.values():
        if fire in county.active_fires:
            current_county = county
            break
    human_risk = current_county.human_activity.fire_risk_multiplier if current_county else 1.0

    # Funkcja pomocnicza do sprawdzania bliskości
    def is_near_agent(x, y, agent_type, radius=2):
        if agent_type == 'RiverAgent':
            for river in rivers:
                for rx, ry in river.points:
                    if abs(rx - x) <= radius and abs(ry - y) <= radius:
                        return True
        elif agent_type == 'MountainAgent':
            for mountain in mountains:
                if mountain.min_x <= x <= mountain.max_x and mountain.min_y <= y <= mountain.max_y:
                    return True
        elif agent_type == 'RoadAgent':
            for road in roads:
                for rx, ry in road.points:
                    if abs(rx - x) <= radius and abs(ry - y) <= radius:
                        return True
        return False

    # Lokalne rozprzestrzenianie
    for dx, dy in directions:
        nx, ny = fire.grid_x + dx, fire.grid_y + dy
        if 0 <= nx < grid_width and 0 <= ny < grid_height:
            spread_prob = weather.fire_spread_probability() * 0.28 * human_risk
            influences = ['WeatherAgent', 'HumanActivityAgent']
            # Sprawdzanie bliskości rzek, gór i dróg
            if is_near_agent(nx, ny, 'RiverAgent'):
                spread_prob *= 0.1  # Redukcja prawdopodobieństwa w pobliżu rzek
                influences.append('RiverAgent')
            if is_near_agent(nx, ny, 'MountainAgent'):
                spread_prob *= 3.0  # Zwiększenie prawdopodobieństwa w górach
                influences.append('MountainAgent')
            if is_near_agent(nx, ny, 'RoadAgent'):
                spread_prob *= 2.5  # Zwiększenie prawdopodobieństwa w pobliżu dróg
                influences.append('RoadAgent')
            # Sprawdzanie statusu komórki (dla logowania i zgodności)
            if grid[ny, nx] in [4, 5, 6]:
                logger.info("Sąsiednia komórka (%s, %s) ma status %s", nx, ny, grid[ny, nx])
            if grid[ny, nx] == 1 and random.random() < spread_prob:
                new_fire = FireIncidentAgent(nx, ny, size=0.5, cause='Spread', start_day=fire.start_day)
                new_fire.spread_influences = influences
                new_fires.append(new_fire)
                forest = next((fg for fg in forests if fg.grid_x == nx and fg.grid_y == ny), None)
                if forest is not None and forest.status == 1:
                    forest.ignition_days.append(fire.start_day)
                    logger.info("Rozprzestrzenianie pożaru do (%s, %s), dzień: %s, powiat: %s, wpływy: %s, prawdopodobieństwo: %.4f", 
                                nx, ny, fire.start_day, current_county.name if current_county else 'brak', influences, spread_prob)
                if len(total_fires) + len(new_fires) >= 2000:
                    break

    # Rozprzestrzenianie między powiatami
    if current_county and len(total_fires) + len(new_fires) < 2000 and random.random() < 0.072:
        neighbor_counties = random.sample(list(counties.values()), min(3, len(counties)))
        for neighbor in neighbor_counties:
            if neighbor != current_county:
                forest_cells = [(f.grid_x, f.grid_y) for f in forests if f.status == 1]
                if forest_cells:
                    x, y = random.choice(forest_cells)
                    spread_prob = weather.fire_spread_probability() * neighbor.human_activity.fire_risk_multiplier * 0.1
                    influences = ['WeatherAgent', 'HumanActivityAgent']
                    # Sprawdzanie bliskości rzek, gór i dróg
                    if is_near_agent(x, y, 'RiverAgent'):
                        spread_prob *= 0.1
                        influences.append('RiverAgent')
                    if is_near_agent(x, y, 'MountainAgent'):
                        spread_prob *= 3.0
                        influences.append('MountainAgent')
                    if is_near_agent(x, y, 'RoadAgent'):
                        spread_prob *= 2.5
                        influences.append('RoadAgent')
                    if grid[y, x] in [4, 5, 6]:
                        logger.info("Komórka między powiatami (%s, %s) ma status %s", x, y, grid[y, x])
                    if grid[y, x] == 1 and random.random() < spread_prob:
                        new_fire = FireIncidentAgent(x, y, size=0.5, cause='County Spread', start_day=fire.start_day)
                        new_fire.spread_influences = influences
                        new_fires.append(new_fire)
                        forest = next((fg for fg in forests if fg.grid_x == x and fg.grid_y == y), None)
                        if forest is not None and forest.status == 1:
                            forest.ignition_days.append(fire.start_day)
                            logger.info("Rozprzestrzenianie między powiatami do (%s, %s), dzień: %s, powiat: %s, wpływy: %s, prawdopodobieństwo: %.4f", 
                                        x, y, fire.start_day, neighbor.name, influences, spread_prob)
                        if len(total_fires) + len(new_fires) >= 2000:
                            break
    return new_fires

# ==========================
# 6. Symulacja krok po kroku
# ==========================
simulation_days = 365
all_fires = []

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
    logger.info("Symulacja: dzień %s, liczba pożarów: %s", day, len(all_fires))
    if len(all_fires) >= 2000:  # ZMIANA: z 800 na 2000
        break
    weather.update()
    for county in counties.values():
        county.human_activity.update()
    for region in regions:
        region.monte_carlo_first_fire(day, weather, all_fires)
    new_fires = []
    for county in counties.values():
        for fire in county.active_fires[:]:
            if fire.active and fire.start_day + fire.duration >= day:
                new_fires.extend(spread_fire(fire, grid, weather, counties, all_fires))
                if len(all_fires) + len(new_fires) >= 2000:  # ZMIANA: z 800 na 2000
                    break
        if len(all_fires) + len(new_fires) >= 2000:  # ZMIANA: z 800 na 2000
            break
    for nf in new_fires:
        if len(all_fires) >= 2000:  # ZMIANA: z 800 na 2000
            break
        county = random.choice(list(counties.values()))
        county.active_fires.append(nf)
        all_fires.append(nf)
    for county in counties.values():
        for fire in county.active_fires:
            if fire.active:
                sorted_firefighters = sorted(firefighters, key=lambda ff: np.sqrt((ff.grid_x - fire.grid_x) ** 2 + (ff.grid_y - fire.grid_y) ** 2))
                for ff in sorted_firefighters[:5]:
                    ff.dispatch_resources(fire, day, grid)
    for ff in firefighters:
        for resource in ff.resources:
            if resource.status == 'moving' and resource.target_fire and resource.target_fire.active:
                resource.move_towards(resource.target_fire.grid_x, resource.target_fire.grid_y, day, grid)
            elif resource.status == 'returning':
                resource.move_towards(ff.grid_x, ff.grid_y, day, grid)
            elif resource.status == 'fighting' and resource.target_fire and resource.target_fire.active:
                resource.try_extinguish(resource.target_fire, day)
            if resource.status == 'fighting' and resource.target_fire and not resource.target_fire.active:
                resource.status = 'returning'
                resource.returning = True
                logger.info("%s z remizy (%s, %s) rozpoczyna powrót, ponieważ pożar (%s, %s) został ugaszony.", 
                            resource.resource_type, ff.grid_x, ff.grid_y, resource.target_fire.grid_x, resource.target_fire.grid_y)
    for ff in firefighters:
        for county in counties.values():
            for fire in county.active_fires:
                if fire.active:
                    ff.try_extinguish(fire, day)
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
            if fire.active and 0 <= fire.grid_y < grid_height and 0 <= fire.grid_x < grid_width:
                if grid[fire.grid_y, fire.grid_x] == 1:
                    grid[fire.grid_y, fire.grid_x] = 2

# ==========================
# 7. Zapis statystyk do pliku
# ==========================
total_fires = len(all_fires)
if total_fires > 0:
    human_fires = sum(1 for fire in all_fires if fire.cause == 'Human')
    lightning_fires = sum(1 for fire in all_fires if fire.cause == 'Lightning')
    other_fires = sum(1 for fire in all_fires if fire.cause == 'Other')
    spread_fires = sum(1 for fire in all_fires if fire.cause == 'Spread')
    county_spread_fires = sum(1 for fire in all_fires if fire.cause == 'County Spread')

    human_percent = (human_fires / total_fires) * 100
    lightning_percent = (lightning_fires / total_fires) * 100
    other_percent = (other_fires / total_fires) * 100
    spread_percent = (spread_fires / total_fires) * 100
    county_spread_percent = (county_spread_fires / total_fires) * 100

    weather_influenced = sum(1 for fire in all_fires if 'WeatherAgent' in getattr(fire, 'spread_influences', []))
    human_activity_influenced = sum(1 for fire in all_fires if 'HumanActivityAgent' in getattr(fire, 'spread_influences', []))
    river_influenced = sum(1 for fire in all_fires if 'RiverAgent' in getattr(fire, 'spread_influences', []))
    mountain_influenced = sum(1 for fire in all_fires if 'MountainAgent' in getattr(fire, 'spread_influences', []))
    road_influenced = sum(1 for fire in all_fires if 'RoadAgent' in getattr(fire, 'spread_influences', []))

    weather_percent = (weather_influenced / total_fires) * 100 if total_fires > 0 else 0
    human_activity_percent = (human_activity_influenced / total_fires) * 100 if total_fires > 0 else 0
    river_percent = (river_influenced / total_fires) * 100 if total_fires > 0 else 0
    mountain_percent = (mountain_influenced / total_fires) * 100 if total_fires > 0 else 0
    road_percent = (road_influenced / total_fires) * 100 if total_fires > 0 else 0

    with open('simulation_summary.txt', 'w') as f:
        f.write("=== Podsumowanie symulacji pożarów ===\n\n")
        f.write(f"Całkowita liczba pożarów: {total_fires}\n\n")
        f.write("Procent pożarów wywołanych przez różne mechanizmy:\n")
        f.write(f"  - Wywołane przez człowieka (Human): {human_percent:.2f}%\n")
        f.write(f"  - Wywołane przez pioruny (Lightning): {lightning_percent:.2f}%\n")
        f.write(f"  - Nieustalone/inne (Other): {other_percent:.2f}%\n")
        f.write(f"  - Rozprzestrzenianie lokalne (automat komórkowy): {spread_percent:.2f}%\n")
        f.write(f"  - Rozprzestrzenianie między powiatami (automat komórkowy): {county_spread_percent:.2f}%\n\n")
        f.write("Procent pożarów podtrzymywanych przez agentów:\n")
        f.write(f"  - WeatherAgent: {weather_percent:.2f}% (pogoda wpłynęła na rozprzestrzenianie)\n")
        f.write(f"  - HumanActivityAgent: {human_activity_percent:.2f}% (aktywność ludzka wpłynęła na rozprzestrzenianie)\n")
        f.write(f"  - RiverAgent: {river_percent:.2f}% (rozprzestrzenianie w pobliżu rzek)\n")
        f.write(f"  - MountainAgent: {mountain_percent:.2f}% (rozprzestrzenianie w górach)\n")
        f.write(f"  - RoadAgent: {road_percent:.2f}% (rozprzestrzenianie w pobliżu dróg)\n")
        f.write("\nLiczba pożarów z wpływem poszczególnych agentów:\n")  # ZMIANA: Dodano statystyki liczby
        f.write(f"  - Rzeki (RiverAgent): {river_influenced} pożarów\n")
        f.write(f"  - Góry (MountainAgent): {mountain_influenced} pożarów\n")
        f.write(f"  - Drogi (RoadAgent): {road_influenced} pożarów\n")
    logger.info("Zapisano podsumowanie statystyk do pliku 'simulation_summary.txt'.")
else:
    with open('simulation_summary.txt', 'w') as f:
        f.write("=== Podsumowanie symulacji pożarów ===\n\n")
        f.write("Brak pożarów w symulacji.\n")
    logger.info("Zapisano podsumowanie statystyk do pliku 'simulation_summary.txt' (brak pożarów).")

# ==========================
# 8. Wizualizacja mapowa (Folium)
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

forests_group = folium.FeatureGroup(name='Forests (statyczne)', show=True)
cities_group = folium.FeatureGroup(name='Cities', show=True)
firestations_group = folium.FeatureGroup(name='Fire Stations', show=True)
rivers_group = folium.FeatureGroup(name='Rivers', show=True)
mountains_group = folium.FeatureGroup(name='Mountains', show=True)
roads_group = folium.FeatureGroup(name='Roads', show=True)

for forest in forests:
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

road_colors = {
    1: '#FFD700',
    2: '#FFA500',
    3: '#FFFF99',
    4: '#FFFACD'
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

forests_group.add_to(m)
cities_group.add_to(m)
firestations_group.add_to(m)
rivers_group.add_to(m)
mountains_group.add_to(m)
roads_group.add_to(m)

features = []
for county in counties.values():
    for fire in county.active_fires:
        if fire.duration < 2:
            continue
        lon = fire.grid_x * grid_size + grid_x_min * grid_size
        lat = fire.grid_y * grid_size + grid_y_min * grid_size
        logger.info("Pożar: lon=%s, lat=%s, start_day=%s, duration=%s, size=%s, cause=%s", 
                    lon, lat, fire.start_day, fire.duration, fire.size, fire.cause)
        if not (41.0 <= lat <= 45.0 and -111.0 <= lon <= -104.0):
            continue
        duration_days = max(5, fire.duration)
        max_radius = min(25, max(1, fire.size / 5))
        for day in range(duration_days + 1):
            current_date = datetime(2006, 1, 1) + timedelta(days=fire.start_day - 1 + day)  # ZMIANA: z 2005 na 2000
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

for forest in forests:
    if forest.status != 1 or not forest.ignition_days:
        continue
    lon = forest.grid_x * grid_size + grid_x_min * grid_size
    lat = forest.grid_y * grid_size + grid_y_min * grid_size
    if not (41.0 <= lat <= 45.0 and -111.0 <= lon <= -104.0):
        continue
    for ign_day in forest.ignition_days:
        logger.info("Palący się las: lon=%s, lat=%s, ignition_day=%s", lon, lat, ign_day)
        for d in range(0, 90):
            day_num = ign_day - 1 + d
            if day_num < 0 or day_num >= simulation_days:
                continue
            current_date = datetime(2006, 1, 1) + timedelta(days=day_num)  # ZMIANA: z 2005 na 2000
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
            current_date = datetime(2006, 1, 1) + timedelta(days=restore_day - 1)  # ZMIANA: z 2005 na 2000
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

folium.LayerControl().add_to(m)

logger.info("Liczba wygenerowanych remiz: %s", len(firefighters))
total_resources = sum(len(ff.resources) for ff in firefighters)
total_airplanes = sum(1 for ff in firefighters for r in ff.resources if r.resource_type == 'airplane')
logger.info("Liczba wszystkich pojazdów: %s, w tym samolotów: %s", total_resources, total_airplanes)
logger.info("Liczba punktów w features (pożary + dynamiczne lasy): %s", len(features))
logger.info("Liczba rzek: %s", len(rivers))
logger.info("Liczba obszarów górskich: %s", len(mountains))
logger.info("Liczba dróg: %s", len(roads))
logger.info("Liczba lasów z ignition_days: %s", len([f for f in forests if f.ignition_days]))

m.save('wy_fires_simulation_365_with_resources.html')
logger.info("Symulacja zapisana jako 'wy_fires_simulation_365_with_resources.html'. Liczba pożarów: %s", len(all_fires))

# ==========================
# 9. Walidacja wyników symulacji z danymi rzeczywistymi
# ==========================
def validate_simulation(real_df, sim_fires, grid_size, grid_x_min, grid_y_min, simulation_year=2006):  # ZMIANA: z 2005 na 2000
    logger.info("Rozpoczynanie walidacji wyników symulacji z danymi rzeczywistymi...")
    
    real_df = real_df[real_df['FIRE_YEAR'] == simulation_year].copy()
    if real_df.empty:
        logger.warning("Brak danych rzeczywistych dla roku %s.", simulation_year)
        with open('validation_results.txt', 'w') as f:
            f.write("=== Wyniki walidacji symulacji ===\n\n")
            f.write(f"Brak danych rzeczywistych dla roku {simulation_year}.\n")
        return
    
    real_df['CAUSE_TYPE'] = real_df['STAT_CAUSE_DESCR'].apply(
        lambda x: 'Human' if x in human_causes else 'Lightning' if x == 'Lightning' else 'Other'
    )
    real_coords = real_df[['LONGITUDE', 'LATITUDE']].values
    real_causes = real_df['CAUSE_TYPE'].values
    real_sizes = real_df['FIRE_SIZE'].values
    real_durations = real_df['DURATION'].values
    real_doys = real_df['DISCOVERY_DOY'].values

    sim_coords = []
    sim_causes = []
    sim_sizes = []
    sim_durations = []
    sim_doys = []
    for fire in sim_fires:
        lon = fire.grid_x * grid_size + grid_x_min * grid_size
        lat = fire.grid_y * grid_size + grid_y_min * grid_size
        if not (41.0 <= lat <= 45.0 and -111.0 <= lon <= -104.0):
            continue
        sim_coords.append([lon, lat])
        sim_causes.append(fire.cause if fire.cause in ['Human', 'Lightning', 'Other'] else 'Spread')
        sim_sizes.append(fire.size)
        sim_durations.append(fire.duration)
        sim_doys.append(fire.start_day)
    
    sim_coords = np.array(sim_coords)
    sim_causes = np.array(sim_causes)
    sim_sizes = np.array(sim_sizes)
    sim_durations = np.array(sim_durations)
    sim_doys = np.array(sim_doys)

    if len(sim_coords) == 0 or len(real_coords) == 0:
        logger.warning("Brak danych symulowanych lub rzeczywistych do porównania.")
        with open('validation_results.txt', 'w') as f:
            f.write("=== Wyniki walidacji symulacji ===\n\n")
            f.write("Brak danych symulowanych lub rzeczywistych do porównania.\n")
        return

    spatial_threshold = 0.01
    tree = cKDTree(real_coords)
    distances, indices = tree.query(sim_coords, k=1, distance_upper_bound=spatial_threshold)
    matched_fires = np.sum(~np.isinf(distances))
    spatial_overlap_percent = (matched_fires / len(sim_fires)) * 100 if sim_fires else 0
    logger.info("Procent pożarów symulowanych z odpowiednikiem rzeczywistym w promieniu %.2f km: %.2f%%",
                spatial_threshold * 111, spatial_overlap_percent)

    real_cause_counts = pd.Series(real_causes).value_counts(normalize=True) * 100
    sim_cause_counts = pd.Series(sim_causes).value_counts(normalize=True) * 100
    cause_comparison = {}
    for cause in ['Human', 'Lightning', 'Other', 'Spread']:
        real_percent = real_cause_counts.get(cause, 0)
        sim_percent = sim_cause_counts.get(cause, 0)
        cause_comparison[cause] = {'Real': real_percent, 'Simulated': sim_percent}
    logger.info("Porównanie rozkładu przyczyn: %s", cause_comparison)

    real_size_mean = np.mean(real_sizes)
    sim_size_mean = np.mean(sim_sizes) if sim_sizes.size > 0 else 0
    size_mae = np.mean(np.abs(real_sizes[:min(len(real_sizes), len(sim_sizes))] - sim_sizes[:min(len(real_sizes), len(sim_sizes))])) if sim_sizes.size > 0 else float('inf')
    logger.info("Średni rozmiar pożaru - Rzeczywisty: %.2f akrów, Symulowany: %.2f akrów, MAE: %.2f akrów",
                real_size_mean, sim_size_mean, size_mae)

    real_duration_mean = np.mean(real_durations)
    sim_duration_mean = np.mean(sim_durations) if sim_durations.size > 0 else 0
    duration_mae = np.mean(np.abs(real_durations[:min(len(real_durations), len(sim_durations))] - sim_durations[:min(len(real_durations), len(sim_durations))])) if sim_durations.size > 0 else float('inf')
    logger.info("Średni czas trwania pożaru - Rzeczywisty: %.2f dni, Symulowany: %.2f dni, MAE: %.2f dni",
                real_duration_mean, sim_duration_mean, duration_mae)

    temporal_threshold = 3
    temporal_matches = 0
    for sim_doy in sim_doys:
        if np.any(np.abs(real_doys - sim_doy) <= temporal_threshold):
            temporal_matches += 1
    temporal_overlap_percent = (temporal_matches / len(sim_doys)) * 100 if sim_doys.size > 0 else 0 
    logger.info("Procent pożarów symulowanych w oknie czasowym ±%s dni od rzeczywistych: %.2f%%",
                temporal_threshold, temporal_overlap_percent)

    with open('validation_results.txt', 'w') as f:
        f.write("=== Wyniki walidacji symulacji ===\n\n")
        f.write(f"Rok symulacji: {simulation_year}\n")
        f.write(f"Całkowita liczba pożarów symulowanych: {len(sim_fires)}\n")
        f.write(f"Całkowita liczba pożarów rzeczywistych (dla roku {simulation_year}): {len(real_df)}\n\n")
        
        f.write("1. Przestrzenne pokrycie:\n")
        f.write(f"  - Procent pożarów symulowanych z odpowiednikiem rzeczywistym w promieniu {spatial_threshold * 111:.2f} km: {spatial_overlap_percent:.2f}%\n")
        f.write(f"  - Liczba dopasowanych pożarów: {matched_fires}\n\n")
        
        f.write("2. Rozkład przyczyn pożarów (%):\n")
        for cause, stats in cause_comparison.items():
            f.write(f"  - {cause}:\n")
            f.write(f"      Rzeczywiste: {stats['Real']:.2f}%\n")
            f.write(f"      Symulowane: {stats['Simulated']:.2f}%\n")
        f.write("\n")
        
        f.write("3. Rozmiar pożarów:\n")
        f.write(f"  - Średni rozmiar rzeczywisty: {real_size_mean:.2f} akrów\n")
        f.write(f"  - Średni rozmiar symulowany: {sim_size_mean:.2f} akrów\n")
        f.write(f"  - Średni błąd bezwzględny (MAE): {size_mae:.2f} akrów\n\n")
        
        f.write("4. Czas trwania pożarów:\n")
        f.write(f"  - Średni czas trwania rzeczywisty: {real_duration_mean:.2f} dni\n")
        f.write(f"  - Średni czas trwania symulowany: {sim_duration_mean:.2f} dni\n")
        f.write(f"  - Średni błąd bezwzględny (MAE): {duration_mae:.2f} dni\n\n")
        
        f.write("5. Pokrycie czasowe:\n")
        f.write(f"  - Procent pożarów symulowanych w oknie ±{temporal_threshold} dni od rzeczywistych: {temporal_overlap_percent:.2f}%\n")
        f.write(f"  - Liczba dopasowanych pożarów czasowo: {temporal_matches}\n")
    
    logger.info("Zapisano wyniki walidacji do pliku 'validation_results.txt'.")

validate_simulation(df, all_fires, grid_size, grid_x_min, grid_y_min, simulation_year=2006)  # ZMIANA: z 2005 na 2000