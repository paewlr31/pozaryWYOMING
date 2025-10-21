import random
import numpy as np

# Lista rzek w Wyoming (100 rzeczywistych rzek)
river_locations = [
    # 1. North Platte River
    [(-106.5, 41.1), (-106.3, 41.5), (-106.1, 42.0), (-105.9, 42.5)],
    # 2. Green River
    [(-110.0, 41.5), (-109.8, 41.7), (-109.5, 42.0), (-109.3, 42.3)],
    # 3. Yellowstone River
    [(-110.5, 44.5), (-110.3, 44.7), (-110.1, 44.9)],
    # 4. Snake River
    [(-110.8, 43.5), (-110.7, 43.7), (-110.6, 44.0)],
    # 5. Bear River
    [(-111.0, 41.5), (-110.9, 41.7), (-110.8, 42.0)],
    # 6. Powder River
    [(-105.5, 44.0), (-105.3, 44.3), (-105.1, 44.7)],
    # 7. Wind River
    [(-108.5, 43.0), (-108.3, 43.3), (-108.1, 43.6)],
    # 8. Bighorn River
    [(-108.0, 44.0), (-107.8, 44.3), (-107.6, 44.6)],
    # 9. Shoshone River
    [(-109.0, 44.5), (-108.8, 44.6), (-108.6, 44.7)],
    # 10. Sweetwater River
    [(-108.5, 42.0), (-108.3, 42.2), (-108.1, 42.4)],
    # 11. Laramie River
    [(-105.8, 41.3), (-105.6, 41.6), (-105.4, 42.0)],
    # 12. Belle Fourche River
    [(-104.5, 44.5), (-104.3, 44.6), (-104.1, 44.7)],
    # 13. Little Snake River
    [(-107.5, 41.0), (-107.3, 41.2), (-107.1, 41.4)],
    # 14. Popo Agie River
    [(-108.7, 42.8), (-108.6, 42.9), (-108.5, 43.0)],
    # 15. Tongue River
    [(-107.3, 44.8), (-107.2, 44.9), (-107.1, 45.0)],
    # 16. Greybull River
    [(-108.5, 44.5), (-108.4, 44.6), (-108.3, 44.7)],
    # 17. Clarks Fork Yellowstone River
    [(-109.2, 44.9), (-109.1, 44.8), (-109.0, 44.7)],
    # 18. Salt River
    [(-110.9, 42.7), (-110.8, 42.8), (-110.7, 42.9)],
    # 19. Hams Fork
    [(-110.7, 42.1), (-110.6, 42.2), (-110.5, 42.3)],
    # 20. Blacks Fork
    [(-110.5, 41.5), (-110.4, 41.6), (-110.3, 41.7)],
    # 21. Henrys Fork
    [(-110.8, 42.4), (-110.7, 42.5), (-110.6, 42.6)],
    # 22. Medicine Bow River
    [(-106.3, 41.2), (-106.2, 41.3), (-106.1, 41.4)],
    # 23. Encampment River
    [(-107.0, 41.2), (-106.9, 41.3), (-106.8, 41.4)],
    # 24. Niobrara River
    [(-104.5, 42.8), (-104.4, 42.9), (-104.3, 43.0)],
    # 25. Cheyenne River
    [(-104.8, 43.0), (-104.7, 43.1), (-104.6, 43.2)]
]

# Generowanie dodatkowych 75 rzek (dopływów i strumieni)
main_river_basins = river_locations[:25]
for i in range(75):
    base_river = random.choice(main_river_basins)
    base_point = random.choice(base_river)
    base_lon, base_lat = base_point
    river_points = []
    num_points = random.randint(3, 4)
    for j in range(num_points):
        lon = base_lon + random.uniform(-0.2, 0.2)
        lat = base_lat + random.uniform(-0.2, 0.2)
        lon = np.clip(lon, -111.0, -104.0)
        lat = np.clip(lat, 41.0, 45.0)
        river_points.append((lon, lat))
        base_lon, base_lat = lon, lat
    river_locations.append(river_points)