import pandas as pd
import numpy as np
import folium
from folium.plugins import TimestampedGeoJson
from datetime import datetime, timedelta
import random
import logging
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import os

# Pliki z danymi zewnętrznymi (założono, że istnieją)
from roads import road_locations
from rivers import river_locations
from mountains import mountain_locations

IGNITION_MULTIPLIER = 50000000000000000000000000000.0
SPREAD_REDUCTION_FACTOR = 0.2
MAX_FIRES = 10000

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

# Utworzenie folderu na wykresy, jeśli nie istnieje
os.makedirs('simPlots', exist_ok=True)

# Lista do przechowywania statystyk dla wszystkich lat
all_years_stats = []

# Zakres lat do symulacji
SIMULATION_YEARS = range(1992, 2016)

for SIMULATION_FIRE_YEAR in SIMULATION_YEARS:
    logger.info(f"Rozpoczynanie symulacji dla roku {SIMULATION_FIRE_YEAR}")

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
    # Rozkład przyczyn
    human_fire_ratio = 5.0
    lightning_fire_ratio = 9.66
    other_fire_ratio = 4.10
    total = human_fire_ratio + lightning_fire_ratio + other_fire_ratio
    human_fire_ratio /= total
    lightning_fire_ratio /= total
    other_fire_ratio /= total
    logger.info("Założony rozkład przyczyn dla roku %s: Human=%.2f%%, Lightning=%.2f%%, Other=%.2f%%",
                SIMULATION_FIRE_YEAR, human_fire_ratio * 100, lightning_fire_ratio * 100, other_fire_ratio * 100)

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

    grid = np.zeros((grid_height, grid_width), dtype=int)

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
            return 3.5 + (self.tourism_level * 2.0 + self.agriculture_level * 1.2 +
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

        def try_extinguish(self, fire, day):
            distance = np.sqrt((self.grid_x - fire.grid_x) ** 2 + (self.grid_y - fire.grid_y) ** 2)
            if distance <= self.range_cells and fire.active:
                if random.random() < self.effectiveness:
                    fire.active = False
                    fire.duration = min(fire.duration, max(1, int(fire.duration * 0.5)))
                    logger.info("[Dzień %s] Remiza (%s, %s) ugasiła pożar (%s, %s bezpośrednio.",
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
            if len(total_fires) >= MAX_FIRES:
                return
            season_factor = 1.0
            simulated_month = (day % 365) // 30
            if simulated_month in [5, 6, 7, 8]:
                season_factor = 3.0
            county = random.choice(self.counties)
            forest_cells = [(f.grid_x, f.grid_y) for f in forests if f.status == 1]
            if forest_cells:
                base_ignition_prob = 0.08 * season_factor * county.human_activity.fire_risk_multiplier * IGNITION_MULTIPLIER
                real_coords = [(row['grid_x'] - grid_x_min, row['grid_y'] - grid_y_min) for _, row in df[df['FIRE_YEAR'] == SIMULATION_FIRE_YEAR].iterrows()]
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
                        if rand < 0.4217:
                            cause = 'Human'
                        elif rand < 0.8575:
                            cause = 'Lightning'
                        else:
                            cause = 'Other'
                        fire = FireIncidentAgent(x, y, size=1, cause=cause, start_day=day)
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

    regions = [RegionAgent('Region 1', list(counties.values())[:10]),
               RegionAgent('Region 2', list(counties.values())[10:20]),
               RegionAgent('Region 3', list(counties.values())[20:])]

    weather = WeatherAgent()

    # ==========================
    # 5. Funkcje pomocnicze
    # ==========================
    def spread_fire(fire, grid, weather, counties, total_fires):
        if len(total_fires) >= MAX_FIRES:
            return []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        new_fires = []
        current_county = None
        for county in counties.values():
            if fire in county.active_fires:
                current_county = county
                break
        human_risk = current_county.human_activity.fire_risk_multiplier if current_county else 1.0

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

        for dx, dy in directions:
            nx, ny = fire.grid_x + dx, fire.grid_y + dy
            if 0 <= nx < grid_width and 0 <= ny < grid_height:
                spread_prob = weather.fire_spread_probability() * 0.03 * human_risk * SPREAD_REDUCTION_FACTOR
                influences = ['WeatherAgent', 'HumanActivityAgent']
                if is_near_agent(nx, ny, 'RiverAgent'):
                    spread_prob *= 0.1
                    influences.append('RiverAgent')
                if is_near_agent(nx, ny, 'MountainAgent'):
                    spread_prob *= 3.0
                    influences.append('MountainAgent')
                if is_near_agent(nx, ny, 'RoadAgent'):
                    spread_prob *= 2.5
                    influences.append('RoadAgent')
                if grid[ny, nx] == 1 and random.random() < spread_prob:
                    new_fire = FireIncidentAgent(nx, ny, size=0.5, cause='Spread', start_day=fire.start_day)
                    new_fire.spread_influences = influences
                    new_fires.append(new_fire)
                    forest = next((fg for fg in forests if fg.grid_x == nx and fg.grid_y == ny), None)
                    if forest is not None and forest.status == 1:
                        forest.ignition_days.append(fire.start_day)
                    if len(total_fires) + len(new_fires) >= MAX_FIRES:
                        break

        if current_county and len(total_fires) + len(new_fires) < 2000 and random.random() < 0.01:
            neighbor_counties = random.sample(list(counties.values()), min(3, len(counties)))
            for neighbor in neighbor_counties:
                if neighbor != current_county:
                    forest_cells = [(f.grid_x, f.grid_y) for f in forests if f.status == 1]
                    if forest_cells:
                        x, y = random.choice(forest_cells)
                        spread_prob = weather.fire_spread_probability() * neighbor.human_activity.fire_risk_multiplier * 0.02
                        influences = ['WeatherAgent', 'HumanActivityAgent']
                        if is_near_agent(x, y, 'RiverAgent'):
                            spread_prob *= 0.1
                            influences.append('RiverAgent')
                        if is_near_agent(x, y, 'MountainAgent'):
                            spread_prob *= 3.0
                            influences.append('MountainAgent')
                        if is_near_agent(x, y, 'RoadAgent'):
                            spread_prob *= 2.5
                            influences.append('RoadAgent')
                        if grid[y, x] == 1 and random.random() < spread_prob:
                            new_fire = FireIncidentAgent(x, y, size=0.5, cause='County Spread', start_day=fire.start_day)
                            new_fire.spread_influences = influences
                            new_fires.append(new_fire)
                            forest = next((fg for fg in forests if fg.grid_x == x and fg.grid_y == y), None)
                            if forest is not None and forest.status == 1:
                                forest.ignition_days.append(fire.start_day)
                            if len(total_fires) + len(new_fires) >= MAX_FIRES:
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
        if len(all_fires) >= 2000:
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
                    if len(all_fires) + len(new_fires) >= 2000:
                        break
            if len(all_fires) + len(new_fires) >= 2000:
                break
        for nf in new_fires:
            if len(all_fires) >= 2000:
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
    year_stats = {'year': SIMULATION_FIRE_YEAR}
    if total_fires > 0:
        human_fires = sum(1 for fire in all_fires if fire.cause == 'Human')
        lightning_fires = sum(1 for fire in all_fires if fire.cause == 'Lightning')
        other_fires = sum(1 for fire in all_fires if fire.cause == 'Other')
        spread_fires = sum(1 for fire in all_fires if fire.cause == 'Spread')
        county_spread_fires = sum(1 for fire in all_fires if fire.cause == 'County Spread')

        year_stats['total_fires'] = total_fires
        year_stats['human_percent'] = (human_fires / total_fires) * 100
        year_stats['lightning_percent'] = (lightning_fires / total_fires) * 100
        year_stats['other_percent'] = (other_fires / total_fires) * 100
        year_stats['spread_percent'] = (spread_fires / total_fires) * 100
        year_stats['county_spread_percent'] = (county_spread_fires / total_fires) * 100

        weather_influenced = sum(1 for fire in all_fires if 'WeatherAgent' in getattr(fire, 'spread_influences', []))
        human_activity_influenced = sum(1 for fire in all_fires if 'HumanActivityAgent' in getattr(fire, 'spread_influences', []))
        river_influenced = sum(1 for fire in all_fires if 'RiverAgent' in getattr(fire, 'spread_influences', []))
        mountain_influenced = sum(1 for fire in all_fires if 'MountainAgent' in getattr(fire, 'spread_influences', []))
        road_influenced = sum(1 for fire in all_fires if 'RoadAgent' in getattr(fire, 'spread_influences', []))

        year_stats['weather_percent'] = (weather_influenced / total_fires) * 100
        year_stats['human_activity_percent'] = (human_activity_influenced / total_fires) * 100
        year_stats['river_percent'] = (river_influenced / total_fires) * 100
        year_stats['mountain_percent'] = (mountain_influenced / total_fires) * 100
        year_stats['road_percent'] = (road_influenced / total_fires) * 100

        real_df = df[df['FIRE_YEAR'] == SIMULATION_FIRE_YEAR].copy()
        if real_df.empty:
            logger.warning(f"Brak danych rzeczywistych dla roku {SIMULATION_FIRE_YEAR}.")
            year_stats.update({
                'spatial_overlap_percent': 0,
                'sim_size_mean': 0,
                'size_mae': float('inf'),
                'sim_duration_mean': 0,
                'duration_mae': float('inf'),
                'temporal_overlap_percent': 0
            })
        else:
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
            for fire in all_fires:
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
                logger.warning(f"Brak danych symulowanych lub rzeczywistych dla roku {SIMULATION_FIRE_YEAR}.")
                year_stats.update({
                    'spatial_overlap_percent': 0,
                    'sim_size_mean': 0,
                    'size_mae': float('inf'),
                    'sim_duration_mean': 0,
                    'duration_mae': float('inf'),
                    'temporal_overlap_percent': 0
                })
            else:
                spatial_threshold = 0.01
                tree = cKDTree(real_coords)
                distances, _ = tree.query(sim_coords, k=1, distance_upper_bound=spatial_threshold)
                matched_fires = np.sum(~np.isinf(distances))
                year_stats['spatial_overlap_percent'] = (matched_fires / len(all_fires)) * 100 if all_fires else 0

                real_size_mean = np.mean(real_sizes)
                year_stats['sim_size_mean'] = np.mean(sim_sizes) if sim_sizes.size > 0 else 0
                year_stats['size_mae'] = np.mean(np.abs(real_sizes[:min(len(real_sizes), len(sim_sizes))] - sim_sizes[:min(len(real_sizes), len(sim_sizes))])) if sim_sizes.size > 0 else float('inf')

                real_duration_mean = np.mean(real_durations)
                year_stats['sim_duration_mean'] = np.mean(sim_durations) if sim_durations.size > 0 else 0
                year_stats['duration_mae'] = np.mean(np.abs(real_durations[:min(len(real_durations), len(sim_durations))] - sim_durations[:min(len(real_durations), len(sim_durations))])) if sim_durations.size > 0 else float('inf')

                temporal_threshold = 3
                temporal_matches = 0
                for sim_doy in sim_doys:
                    if np.any(np.abs(real_doys - sim_doy) <= temporal_threshold):
                        temporal_matches += 1
                year_stats['temporal_overlap_percent'] = (temporal_matches / len(sim_doys)) * 100 if sim_doys.size > 0 else 0

        all_years_stats.append(year_stats)

# ==========================
# 8. Generowanie wykresów
# ==========================
metrics = [
    ('total_fires', 'Całkowita liczba pożarów', 'Liczba pożarów'),
    ('human_percent', 'Procent pożarów wywołanych przez człowieka', 'Procent (%)'),
    ('lightning_percent', 'Procent pożarów wywołanych przez pioruny', 'Procent (%)'),
    ('other_percent', 'Procent pożarów nieustalonych/innych', 'Procent (%)'),
    ('spread_percent', 'Procent pożarów z rozprzestrzeniania lokalnego', 'Procent (%)'),
    ('county_spread_percent', 'Procent pożarów z rozprzestrzeniania między powiatami', 'Procent (%)'),
    ('weather_percent', 'Procent pożarów z wpływem pogody', 'Procent (%)'),
    ('human_activity_percent', 'Procent pożarów z wpływem aktywności ludzkiej', 'Procent (%)'),
    ('river_percent', 'Procent pożarów w pobliżu rzek', 'Procent (%)'),
    ('mountain_percent', 'Procent pożarów w górach', 'Procent (%)'),
    ('road_percent', 'Procent pożarów w pobliżu dróg', 'Procent (%)'),
    ('spatial_overlap_percent', 'Procent przestrzennego pokrycia', 'Procent (%)'),
    ('sim_size_mean', 'Średni rozmiar pożaru (symulowany)', 'Rozmiar (akry)'),
    ('size_mae', 'Średni błąd bezwzględny rozmiaru', 'MAE (akry)'),
    ('sim_duration_mean', 'Średni czas trwania pożaru (symulowany)', 'Czas (dni)'),
    ('duration_mae', 'Średni błąd bezwzględny czasu trwania', 'MAE (dni)'),
    ('temporal_overlap_percent', 'Procent czasowego pokrycia', 'Procent (%)')
]

years = [stats['year'] for stats in all_years_stats]

for metric_key, metric_title, y_label in metrics:
    values = [stats.get(metric_key, 0) for stats in all_years_stats]
    mean_value = np.mean(values) if values else 0

    plt.figure(figsize=(10, 6))
    plt.plot(years, values, marker='o', linestyle='-', label=metric_title)
    plt.axhline(y=mean_value, color='r', linestyle='--', label=f'Średnia: {mean_value:.2f}')
    plt.title(metric_title)
    plt.xlabel('Rok')
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'simPlots/{metric_key}_plot.png')
    plt.close()

# ==========================
# 9. Zapis średnich statystyk do pliku
# ==========================
mean_stats = {}
for metric_key, metric_title, _ in metrics:
    values = [stats.get(metric_key, 0) for stats in all_years_stats]
    mean_stats[metric_key] = np.mean(values) if values else 0

with open('simPlots/mean_simulation_stats.txt', 'w') as f:
    f.write("=== Średnie statystyki symulacji dla lat 1992-2015 ===\n\n")
    f.write(f"Średnia całkowita liczba pożarów: {mean_stats['total_fires']:.2f}\n\n")
    f.write("Średni procent pożarów wywołanych przez różne mechanizmy:\n")
    f.write(f"  - Wywołane przez człowieka (Human): {mean_stats['human_percent']:.2f}%\n")
    f.write(f"  - Wywołane przez pioruny (Lightning): {mean_stats['lightning_percent']:.2f}%\n")
    f.write(f"  - Nieustalone/inne (Other): {mean_stats['other_percent']:.2f}%\n")
    f.write(f"  - Rozprzestrzenianie lokalne (automat komórkowy): {mean_stats['spread_percent']:.2f}%\n")
    f.write(f"  - Rozprzestrzenianie między powiatami (automat komórkowy): {mean_stats['county_spread_percent']:.2f}%\n\n")
    f.write("Średni procent pożarów podtrzymywanych przez agentów:\n")
    f.write(f"  - WeatherAgent: {mean_stats['weather_percent']:.2f}% (pogoda wpłynęła na rozprzestrzenianie)\n")
    f.write(f"  - HumanActivityAgent: {mean_stats['human_activity_percent']:.2f}% (aktywność ludzka wpłynęła na rozprzestrzenianie)\n")
    f.write(f"  - RiverAgent: {mean_stats['river_percent']:.2f}% (rozprzestrzenianie w pobliżu rzek)\n")
    f.write(f"  - MountainAgent: {mean_stats['mountain_percent']:.2f}% (rozprzestrzenianie w górach)\n")
    f.write(f"  - RoadAgent: {mean_stats['road_percent']:.2f}% (rozprzestrzenianie w pobliżu dróg)\n\n")
    f.write("Średnie wyniki walidacji:\n")
    f.write(f"  - Przestrzenne pokrycie: {mean_stats['spatial_overlap_percent']:.2f}%\n")
    f.write(f"  - Średni rozmiar pożaru (symulowany): {mean_stats['sim_size_mean']:.2f} akrów\n")
    f.write(f"  - Średni błąd bezwzględny rozmiaru (MAE): {mean_stats['size_mae']:.2f} akrów\n")
    f.write(f"  - Średni czas trwania pożaru (symulowany): {mean_stats['sim_duration_mean']:.2f} dni\n")
    f.write(f"  - Średni błąd bezwzględny czasu trwania (MAE): {mean_stats['duration_mae']:.2f} dni\n")
    f.write(f"  - Czasowe pokrycie: {mean_stats['temporal_overlap_percent']:.2f}%\n")

# Wyświetlenie średnich statystyk w konsoli
print("=== Średnie statystyki symulacji dla lat 1992-2015 ===")
print(f"Średnia całkowita liczba pożarów: {mean_stats['total_fires']:.2f}")
print("Średni procent pożarów wywołanych przez różne mechanizmy:")
print(f"  - Wywołane przez człowieka (Human): {mean_stats['human_percent']:.2f}%")
print(f"  - Wywołane przez pioruny (Lightning): {mean_stats['lightning_percent']:.2f}%")
print(f"  - Nieustalone/inne (Other): {mean_stats['other_percent']:.2f}%")
print(f"  - Rozprzestrzenianie lokalne (automat komórkowy): {mean_stats['spread_percent']:.2f}%")
print(f"  - Rozprzestrzenianie między powiatami (automat komórkowy): {mean_stats['county_spread_percent']:.2f}%")
print("Średni procent pożarów podtrzymywanych przez agentów:")
print(f"  - WeatherAgent: {mean_stats['weather_percent']:.2f}%")
print(f"  - HumanActivityAgent: {mean_stats['human_activity_percent']:.2f}%")
print(f"  - RiverAgent: {mean_stats['river_percent']:.2f}%")
print(f"  - MountainAgent: {mean_stats['mountain_percent']:.2f}%")
print(f"  - RoadAgent: {mean_stats['road_percent']:.2f}%")
print("Średnie wyniki walidacji:")
print(f"  - Przestrzenne pokrycie: {mean_stats['spatial_overlap_percent']:.2f}%")
print(f"  - Średni rozmiar pożaru (symulowany): {mean_stats['sim_size_mean']:.2f} akrów")
print(f"  - Średni błąd bezwzględny rozmiaru (MAE): {mean_stats['size_mae']:.2f} akrów")
print(f"  - Średni czas trwania pożaru (symulowany): {mean_stats['sim_duration_mean']:.2f} dni")
print(f"  - Średni błąd bezwzględny czasu trwania (MAE): {mean_stats['duration_mae']:.2f} dni")
print(f"  - Czasowe pokrycie: {mean_stats['temporal_overlap_percent']:.2f}%")

logger.info("Zapisano wykresy do folderu 'simPlots/' oraz średnie statystyki do pliku 'simPlots/mean_simulation_stats.txt'.")