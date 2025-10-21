import numpy as np
import random
from mountains import mountain_locations

def generate_curved_points(start, end, num_points=5, deviation=0.05, avoid_mountains=True):
    """Generuje krzywoliniowe punkty między start a end, z opcją omijania gór."""
    start_lon, start_lat = start
    end_lon, end_lat = end
    points = [(start_lon, start_lat)]
    
    for _ in range(num_points - 2):
        t = (_ + 1) / (num_points - 1)
        base_lon = start_lon + t * (end_lon - start_lon)
        base_lat = start_lat + t * (end_lat - start_lat)
        
        # Losowe odchylenie
        deviation_factor = deviation * (1 + t * 0.5)  # Większe odchylenie w środku trasy
        lon = base_lon + random.uniform(-deviation_factor, deviation_factor)
        lat = base_lat + random.uniform(-deviation_factor, deviation_factor)
        
        # Ogranicz współrzędne do Wyoming
        lon = np.clip(lon, -111.0, -104.0)
        lat = np.clip(lat, 41.0, 45.0)
        
        # Omijanie gór, jeśli włączone
        if avoid_mountains:
            for mountain in mountain_locations:
                if (mountain['lon_min'] <= lon <= mountain['lon_max'] and 
                    mountain['lat_min'] <= lat <= mountain['lat_max']):
                    # Przesuń punkt poza obszar gór
                    lon += random.choice([-0.1, 0.1])
                    lat += random.choice([-0.1, 0.1])
                    lon = np.clip(lon, -111.0, -104.0)
                    lat = np.clip(lat, 41.0, 45.0)
        
        points.append((lon, lat))
    
    points.append((end_lon, end_lat))
    return points

# Lista miast z test.py (główne miasta Wyoming)
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

# Lista dróg w Wyoming (150 dróg)
road_locations = []

# 1. Autostrady międzystanowe (10 segmentów, łączą główne miasta)
interstate_connections = [
    # I-80: Evanston -> Rock Springs -> Rawlins -> Laramie -> Cheyenne
    {'name': 'I-80 W-E (seg 1)', 'weight': 1, 'points': generate_curved_points((-110.9632, 41.3114), (-109.2490, 41.6372), 6, deviation=0.02)},
    {'name': 'I-80 W-E (seg 2)', 'weight': 1, 'points': generate_curved_points((-109.2490, 41.6372), (-105.7455, 42.0978), 6, deviation=0.02)},
    {'name': 'I-80 W-E (seg 3)', 'weight': 1, 'points': generate_curved_points((-105.7455, 42.0978), (-106.3197, 42.8641), 6, deviation=0.02)},
    {'name': 'I-80 W-E (seg 4)', 'weight': 1, 'points': generate_curved_points((-106.3197, 42.8641), (-105.5022, 41.1399), 6, deviation=0.02)},
    # I-25: Cheyenne -> Casper -> Buffalo
    {'name': 'I-25 S-N (seg 1)', 'weight': 1, 'points': generate_curved_points((-105.5022, 41.1399), (-104.8202, 41.1359), 5, deviation=0.02)},
    {'name': 'I-25 S-N (seg 2)', 'weight': 1, 'points': generate_curved_points((-104.8202, 41.1359), (-105.5911, 44.3483), 5, deviation=0.02)},
    # I-90: Buffalo -> Sheridan
    {'name': 'I-90 S-N (seg 1)', 'weight': 1, 'points': generate_curved_points((-105.5911, 44.3483), (-104.8253, 44.7970), 5, deviation=0.02)},
    {'name': 'I-90 S-N (seg 2)', 'weight': 1, 'points': generate_curved_points((-104.8253, 44.7970), (-104.1389, 44.2720), 4, deviation=0.02)},  # Sheridan -> Sundance
    # Dodatkowe segmenty I-80
    {'name': 'I-80 W-E (seg 5)', 'weight': 1, 'points': generate_curved_points((-110.9632, 41.3114), (-108.7373, 42.7552), 5, deviation=0.02)},  # Evanston -> Green River
    {'name': 'I-80 W-E (seg 6)', 'weight': 1, 'points': generate_curved_points((-108.7373, 42.7552), (-107.5437, 41.5875), 5, deviation=0.02)},  # Green River -> Saratoga
]

road_locations.extend(interstate_connections)

# 2. Autostrady stanowe (30 dróg, łączą miasta i odgałęziają się od autostrad)
state_highway_connections = [
    {'name': 'WY-789 S-N', 'weight': 2, 'points': generate_curved_points((-108.7373, 42.7552), (-108.3897, 42.8347), 6, deviation=0.03)},  # Green River -> Lander
    {'name': 'WY-59 S-N', 'weight': 2, 'points': generate_curved_points((-104.1827, 42.8494), (-105.9399, 44.2910), 6, deviation=0.03)},  # Douglas -> Gillette
    {'name': 'WY-220 E-W', 'weight': 2, 'points': generate_curved_points((-104.8202, 41.1359), (-105.7455, 42.0978), 5, deviation=0.03)},  # Casper -> Rawlins
    {'name': 'WY-28 E-W', 'weight': 2, 'points': generate_curved_points((-108.3897, 42.8347), (-108.8964, 42.0663), 5, deviation=0.03)},  # Lander -> Pinedale
    {'name': 'WY-120 S-N', 'weight': 2, 'points': generate_curved_points((-108.2023, 43.0238), (-106.4064, 44.6872), 5, deviation=0.03)},  # Cody -> Powell
    {'name': 'WY-22 E-W', 'weight': 2, 'points': generate_curved_points((-110.3326, 43.4799), (-108.2023, 43.0238), 5, deviation=0.03)},  # Jackson -> Cody
    {'name': 'WY-130 E-W', 'weight': 2, 'points': generate_curved_points((-106.3197, 42.8641), (-107.5437, 41.5875), 4, deviation=0.03)},  # Laramie -> Saratoga
    {'name': 'WY-70 E-W', 'weight': 2, 'points': generate_curved_points((-107.5437, 41.5875), (-105.7455, 42.0978), 4, deviation=0.03)},  # Saratoga -> Rawlins
    {'name': 'WY-14 E-W', 'weight': 2, 'points': generate_curved_points((-108.2023, 43.0238), (-107.2009, 44.0802), 5, deviation=0.03)},  # Cody -> Worland
    {'name': 'WY-89 S-N', 'weight': 2, 'points': generate_curved_points((-110.9632, 41.3114), (-110.3326, 43.4799), 6, deviation=0.03)},  # Evanston -> Jackson
    {'name': 'WY-26 E-W', 'weight': 2, 'points': generate_curved_points((-108.3897, 42.8347), (-107.1357, 43.6599), 4, deviation=0.03)},  # Lander -> Riverton
    {'name': 'WY-287 S-N', 'weight': 2, 'points': generate_curved_points((-108.3897, 42.8347), (-107.2009, 44.0802), 5, deviation=0.03)},  # Lander -> Worland
    {'name': 'WY-191 S-N', 'weight': 2, 'points': generate_curved_points((-109.2490, 41.6372), (-108.8964, 42.0663), 5, deviation=0.03)},  # Rock Springs -> Pinedale
    {'name': 'WY-50 S-N', 'weight': 2, 'points': generate_curved_points((-105.9399, 44.2910), (-104.2047, 44.9083), 5, deviation=0.03)},  # Gillette -> Newcastle
    {'name': 'WY-116 S-N', 'weight': 2, 'points': generate_curved_points((-104.2047, 44.9083), (-104.1389, 44.2720), 4, deviation=0.03)},  # Newcastle -> Sundance
    {'name': 'WY-34 E-W', 'weight': 2, 'points': generate_curved_points((-104.0555, 42.0625), (-104.6097, 41.6322), 4, deviation=0.03)},  # Wheatland -> Torrington
    {'name': 'WY-270 S-N', 'weight': 2, 'points': generate_curved_points((-104.6097, 41.6322), (-104.1827, 42.8494), 4, deviation=0.03)},  # Torrington -> Douglas
    {'name': 'WY-430 S-N', 'weight': 2, 'points': generate_curved_points((-109.2490, 41.6372), (-110.0752, 41.7911), 3, deviation=0.03)},  # Rock Springs -> Kemmerer
    {'name': 'WY-230 E-W', 'weight': 2, 'points': generate_curved_points((-106.3197, 42.8641), (-105.5022, 41.1399), 3, deviation=0.03)},  # Laramie -> Cheyenne
    {'name': 'WY-374 E-W', 'weight': 2, 'points': generate_curved_points((-108.7373, 42.7552), (-109.2490, 41.6372), 3, deviation=0.03)},  # Green River -> Rock Springs
    {'name': 'WY-410 S-N', 'weight': 2, 'points': generate_curved_points((-110.0752, 41.7911), (-110.9632, 41.3114), 3, deviation=0.03)},  # Kemmerer -> Evanston
    {'name': 'WY-530 S-N', 'weight': 2, 'points': generate_curved_points((-108.7373, 42.7552), (-110.0752, 41.7911), 4, deviation=0.03)},  # Green River -> Kemmerer
    {'name': 'WY-94 E-W', 'weight': 2, 'points': generate_curved_points((-104.1827, 42.8494), (-104.0555, 42.0625), 3, deviation=0.03)},  # Douglas -> Wheatland
    {'name': 'WY-24 S-N', 'weight': 2, 'points': generate_curved_points((-104.1389, 44.2720), (-104.2047, 44.9083), 3, deviation=0.03)},  # Sundance -> Newcastle
    {'name': 'WY-585 S-N', 'weight': 2, 'points': generate_curved_points((-104.2047, 44.9083), (-104.8253, 44.7970), 3, deviation=0.03)},  # Newcastle -> Sheridan
    {'name': 'WY-212 E-W', 'weight': 2, 'points': generate_curved_points((-105.9399, 44.2910), (-105.5911, 44.3483), 3, deviation=0.03)},  # Gillette -> Buffalo
    {'name': 'WY-789 S-N (2)', 'weight': 2, 'points': generate_curved_points((-107.1357, 43.6599), (-107.2009, 44.0802), 4, deviation=0.03)},  # Riverton -> Worland
    {'name': 'WY-30 E-W', 'weight': 2, 'points': generate_curved_points((-104.6097, 41.6322), (-105.5022, 41.1399), 3, deviation=0.03)},  # Torrington -> Cheyenne
    {'name': 'WY-150 S-N', 'weight': 2, 'points': generate_curved_points((-110.9632, 41.3114), (-110.3326, 43.4799), 3, deviation=0.03)},  # Evanston -> Jackson
    {'name': 'WY-16 E-W', 'weight': 2, 'points': generate_curved_points((-105.5911, 44.3483), (-106.4064, 44.6872), 3, deviation=0.03)},  # Buffalo -> Powell
]

road_locations.extend(state_highway_connections)

# 3. Drogi powiatowe (50 dróg, odgałęzienia od autostrad i miast)
county_roads = []
for i in range(50):
    # Wybierz losowe miasto lub punkt z autostrady/stanowej jako początek
    start_source = random.choice(city_locations + [random.choice(road['points']) for road in road_locations[:40]])
    start_lon, start_lat = start_source
    # Losowy kierunek i odległość (krótsze trasy)
    length = random.uniform(0.1, 0.5)  # Odległość w stopniach
    angle = random.uniform(0, 2 * np.pi)
    end_lon = start_lon + length * np.cos(angle)
    end_lat = start_lat + length * np.sin(angle)
    end_lon = np.clip(end_lon, -111.0, -104.0)
    end_lat = np.clip(end_lat, 41.0, 45.0)
    county_roads.append({
        'name': f'CR-{i+1} County',
        'weight': 3,
        'points': generate_curved_points((start_lon, start_lat), (end_lon, end_lat), 4, deviation=0.04, avoid_mountains=True)
    })

road_locations.extend(county_roads)

# 4. Drogi lokalne (60 dróg, krótkie, kręte, wychodzące z miast lub innych dróg)
local_roads = []
for i in range(60):
    # Wybierz losowe miasto lub punkt z innej drogi jako początek
    start_source = random.choice(city_locations + [random.choice(road['points']) for road in road_locations[:90]])
    start_lon, start_lat = start_source
    # Krótsze trasy lokalne
    length = random.uniform(0.05, 0.2)  # Odległość w stopniach
    angle = random.uniform(0, 2 * np.pi)
    end_lon = start_lon + length * np.cos(angle)
    end_lat = start_lat + length * np.sin(angle)
    end_lon = np.clip(end_lon, -111.0, -104.0)
    end_lat = np.clip(end_lat, 41.0, 45.0)
    local_roads.append({
        'name': f'Local-{i+1}',
        'weight': 4,
        'points': generate_curved_points((start_lon, start_lat), (end_lon, end_lat), 3, deviation=0.06, avoid_mountains=True)
    })

road_locations.extend(local_roads)