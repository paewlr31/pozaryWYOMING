import re
import os
import matplotlib.pyplot as plt
import numpy as np

# Utworzenie folderu simPlots, jeśli nie istnieje
os.makedirs('simPlots', exist_ok=True)

# Słowniki do przechowywania danych dla każdego roku
data = {
    'years': [],
    'total_fires': [],
    'causes': {'Human': [], 'Lightning': [], 'Other': [], 'Spread': [], 'County Spread': []},
    'agents_percent': {'WeatherAgent': [], 'HumanActivityAgent': [], 'RiverAgent': [], 'MountainAgent': [], 'RoadAgent': []},
    'agents_count': {'RiverAgent': [], 'MountainAgent': [], 'RoadAgent': []},
    'spatial_coverage_percent': [],
    'spatial_matched': [],
    'fire_size_mean': [],
    'fire_size_mae': [],
    'fire_duration_mean': [],
    'fire_duration_mae': [],
    'temporal_coverage_percent': [],
    'temporal_matched': []
}

# Funkcja do parsowania pliku
def parse_file(filename):
    current_year = None
    encodings = ['utf-8', 'windows-1250', 'latin-1', 'utf-16']
    lines = None
    for encoding in encodings:
        try:
            with open(filename, 'r', encoding=encoding) as file:
                lines = file.readlines()
                print(f"Użyto kodowania: {encoding}")
                break
        except UnicodeDecodeError:
            print(f"Błąd kodowania {encoding}, próbuję następnego...")
    if lines is None:
        print("Nie udało się otworzyć pliku z żadnym kodowaniem!")
        return

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        print(f"Przetwarzanie linii {i+1}: {line}")  # Debug: wypisz każdą linię
        # Normalizacja linii: usuń wielokrotne spacje, zamień na małe litery
        normalized_line = re.sub(r'\s+', ' ', line.lower()).strip()
        # Znajdź rok
        if re.match(r'^\d{4}$', line):
            current_year = int(line)
            data['years'].append(current_year)
            print(f"Znaleziono rok: {current_year}")
            i += 1
            continue
        # Podsumowanie symulacji
        if 'podsumowanie symulacji' in normalized_line:
            i += 1
            # Całkowita liczba pożarów
            if i < len(lines) and 'liczba po' in normalized_line:
                match = re.search(r'\d+', lines[i])
                if match:
                    total_fires = int(match.group())
                    data['total_fires'].append(total_fires)
                    print(f"  Liczba pożarów: {total_fires}")
                else:
                    print(f"  Błąd: Nie znaleziono liczby pożarów w linii: {lines[i].strip()}")
                i += 1
            # Procent pożarów wywołanych przez mechanizmy
            if i < len(lines) and 'procent po' in normalized_line:
                i += 1
                causes = ['cz owieka', 'pioruny', 'nieustalone', 'lokalne', 'mi dzy powiatami']
                for cause in causes:
                    if i < len(lines):
                        match = re.search(r'([\d.]+)%', lines[i])
                        if match:
                            percent = float(match.group(1))
                            key = {'cz owieka': 'Human', 'pioruny': 'Lightning', 'nieustalone': 'Other', 
                                   'lokalne': 'Spread', 'mi dzy powiatami': 'County Spread'}[cause]
                            data['causes'][key].append(percent)
                            print(f"  Przyczyna {key}: {percent}%")
                        else:
                            print(f"  Błąd: Nie znaleziono procentu dla przyczyny {cause} w linii: {lines[i].strip()}")
                        i += 1
                i += 1  # Pomiń pustą linię
            # Procent pożarów podtrzymywanych przez agentów
            if i < len(lines) and 'procent po' in normalized_line:
                i += 1
                for agent in ['WeatherAgent', 'HumanActivityAgent', 'RiverAgent', 'MountainAgent', 'RoadAgent']:
                    if i < len(lines):
                        match = re.search(r'([\d.]+)%', lines[i])
                        if match:
                            percent = float(match.group(1))
                            data['agents_percent'][agent].append(percent)
                            print(f"  Agent {agent}: {percent}%")
                        else:
                            print(f"  Błąd: Nie znaleziono procentu dla agenta {agent} w linii: {lines[i].strip()}")
                        i += 1
                i += 1  # Pomiń pustą linię
            # Liczba pożarów z wpływem agentów
            if i < len(lines) and 'liczba po' in normalized_line:
                i += 1
                for agent in ['RiverAgent', 'MountainAgent', 'RoadAgent']:
                    if i < len(lines):
                        match = re.search(r'(\d+)', lines[i])
                        if match:
                            count = int(match.group(1))
                            data['agents_count'][agent].append(count)
                            print(f"  Liczba dla {agent}: {count}")
                        else:
                            print(f"  Błąd: Nie znaleziono liczby dla agenta {agent} w linii: {lines[i].strip()}")
                        i += 1
                i += 1  # Pomiń pustą linię
        # Wyniki walidacji
        if 'wyniki walidacji' in normalized_line:
            i += 1
            # Całkowita liczba pożarów (pomiń)
            if i < len(lines) and 'liczba po' in normalized_line:
                i += 1
            # Przestrzenne pokrycie
            if i < len(lines) and 'przestrzenne pokrycie' in normalized_line:
                i += 1
                match_percent = re.search(r'([\d.]+)%', lines[i])
                if match_percent:
                    data['spatial_coverage_percent'].append(float(match_percent.group(1)))
                    print(f"  Przestrzenne pokrycie: {float(match_percent.group(1))}%")
                else:
                    print(f"  Błąd: Nie znaleziono procentu przestrzennego pokrycia w linii: {lines[i].strip()}")
                i += 1
                match_matched = re.search(r'(\d+)', lines[i])
                if match_matched:
                    data['spatial_matched'].append(int(match_matched.group(1)))
                    print(f"  Dopasowane przestrzennie: {int(match_matched.group(1))}")
                else:
                    print(f"  Błąd: Nie znaleziono liczby dopasowanych przestrzennie w linii: {lines[i].strip()}")
                i += 1
            # Rozkład przyczyn (pomiń)
            if i < len(lines) and 'rozk ad przyczyn' in normalized_line:
                i += 4
            # Rozmiar pożarów
            if i < len(lines) and 'rozmiar po' in normalized_line:
                i += 1
                match_size = re.search(r'([\d.]+)', lines[i])
                if match_size:
                    data['fire_size_mean'].append(float(match_size.group(1)))
                    print(f"  Średni rozmiar: {float(match_size.group(1))} akrów")
                else:
                    print(f"  Błąd: Nie znaleziono średniego rozmiaru w linii: {lines[i].strip()}")
                i += 1
                match_mae = re.search(r'([\d.]+)', lines[i])
                if match_mae:
                    data['fire_size_mae'].append(float(match_mae.group(1)))
                    print(f"  MAE rozmiaru: {float(match_mae.group(1))} akrów")
                else:
                    print(f"  Błąd: Nie znaleziono MAE rozmiaru w linii: {lines[i].strip()}")
                i += 1
            # Czas trwania pożarów
            if i < len(lines) and 'czas trwania' in normalized_line:
                i += 1
                match_duration = re.search(r'([\d.]+)', lines[i])
                if match_duration:
                    data['fire_duration_mean'].append(float(match_duration.group(1)))
                    print(f"  Średni czas trwania: {float(match_duration.group(1))} dni")
                else:
                    print(f"  Błąd: Nie znaleziono średniego czasu trwania w linii: {lines[i].strip()}")
                i += 1
                match_mae = re.search(r'([\d.]+)', lines[i])
                if match_mae:
                    data['fire_duration_mae'].append(float(match_mae.group(1)))
                    print(f"  MAE czasu trwania: {float(match_mae.group(1))} dni")
                else:
                    print(f"  Błąd: Nie znaleziono MAE czasu trwania w linii: {lines[i].strip()}")
                i += 1
            # Pokrycie czasowe
            if i < len(lines) and 'pokrycie czasowe' in normalized_line:
                i += 1
                match_percent = re.search(r'([\d.]+)%', lines[i])
                if match_percent:
                    data['temporal_coverage_percent'].append(float(match_percent.group(1)))
                    print(f"  Pokrycie czasowe: {float(match_percent.group(1))}%")
                else:
                    print(f"  Błąd: Nie znaleziono procentu pokrycia czasowego w linii: {lines[i].strip()}")
                i += 1
                match_matched = re.search(r'(\d+)', lines[i])
                if match_matched:
                    data['temporal_matched'].append(int(match_matched.group(1)))
                    print(f"  Dopasowane czasowo: {int(match_matched.group(1))}")
                else:
                    print(f"  Błąd: Nie znaleziono liczby dopasowanych czasowo w linii: {lines[i].strip()}")
                i += 1
        i += 1

# Funkcja do tworzenia wykresu liniowego z linią średnią
def plot_parameter(years, values, title, ylabel, filename):
    if not values or len(values) != len(years):
        print(f"Pomijam wykres dla {title}: brak danych lub niezgodna liczba danych ({len(values)} wartości, {len(years)} lat)")
        return
    plt.figure(figsize=(10, 6))
    plt.plot(years, values, marker='o', linestyle='-', color='b', label='Dane roczne')
    mean_value = np.mean(values)
    plt.axhline(y=mean_value, color='r', linestyle='--', label=f'Średnia: {mean_value:.2f}')
    plt.title(title)
    plt.xlabel('Rok')
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.xticks(years, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'simPlots/{filename}.png')
    plt.close()

# Parsowanie pliku
parse_file('siema.txt')

# Sprawdzenie, czy dane zostały wczytane
if not data['years']:
    print("Błąd: Nie wczytano żadnych danych z pliku siema.txt")
else:
    # Generowanie wykresów
    plot_parameter(data['years'], data['total_fires'], 'Całkowita liczba pożarów w latach 1992–2015', 'Liczba pożarów', 'total_fires')
    for cause in data['causes']:
        plot_parameter(data['years'], data['causes'][cause], f'Procent pożarów: {cause}', 'Procent (%)', f'cause_{cause}')
    for agent in data['agents_percent']:
        plot_parameter(data['years'], data['agents_percent'][agent], f'Procent pożarów podtrzymywanych: {agent}', 'Procent (%)', f'agent_percent_{agent}')
    for agent in data['agents_count']:
        plot_parameter(data['years'], data['agents_count'][agent], f'Liczba pożarów z wpływem: {agent}', 'Liczba pożarów', f'agent_count_{agent}')
    plot_parameter(data['years'], data['spatial_coverage_percent'], 'Procent przestrzennego pokrycia', 'Procent (%)', 'spatial_coverage_percent')
    plot_parameter(data['years'], data['spatial_matched'], 'Liczba dopasowanych pożarów przestrzennie', 'Liczba pożarów', 'spatial_matched')
    plot_parameter(data['years'], data['fire_size_mean'], 'Średni rozmiar pożaru', 'Rozmiar (akry)', 'fire_size_mean')
    plot_parameter(data['years'], data['fire_size_mae'], 'Średni błąd bezwzględny rozmiaru', 'MAE (akry)', 'fire_size_mae')
    plot_parameter(data['years'], data['fire_duration_mean'], 'Średni czas trwania pożaru', 'Czas (dni)', 'fire_duration_mean')
    plot_parameter(data['years'], data['fire_duration_mae'], 'Średni błąd bezwzględny czasu trwania', 'MAE (dni)', 'fire_duration_mae')
    plot_parameter(data['years'], data['temporal_coverage_percent'], 'Procent pokrycia czasowego', 'Procent (%)', 'temporal_coverage_percent')
    plot_parameter(data['years'], data['temporal_matched'], 'Liczba dopasowanych pożarów czasowo', 'Liczba pożarów', 'temporal_matched')

    # Obliczanie i wypisywanie średnich
    print("\n=== Podsumowanie średnich dla lat 1992–2015 ===")
    print(f"Średnia całkowita liczba pożarów: {np.mean(data['total_fires']) if data['total_fires'] else 'brak danych'}")
    print("\nŚredni procent pożarów wywołanych przez mechanizmy:")
    for cause in data['causes']:
        print(f"  - {cause}: {np.mean(data['causes'][cause]) if data['causes'][cause] else 'brak danych'}%")
    print("\nŚredni procent pożarów podtrzymywanych przez agentów:")
    for agent in data['agents_percent']:
        print(f"  - {agent}: {np.mean(data['agents_percent'][agent]) if data['agents_percent'][agent] else 'brak danych'}%")
    print("\nŚrednia liczba pożarów z wpływem agentów:")
    for agent in data['agents_count']:
        print(f"  - {agent}: {np.mean(data['agents_count'][agent]) if data['agents_count'][agent] else 'brak danych'}")
    print("\nŚrednie wyniki walidacji:")
    print(f"  - Procent przestrzennego pokrycia: {np.mean(data['spatial_coverage_percent']) if data['spatial_coverage_percent'] else 'brak danych'}%")
    print(f"  - Liczba dopasowanych pożarów przestrzennie: {np.mean(data['spatial_matched']) if data['spatial_matched'] else 'brak danych'}")
    print(f"  - Średni rozmiar pożaru: {np.mean(data['fire_size_mean']) if data['fire_size_mean'] else 'brak danych'} akrów")
    print(f"  - Średni błąd bezwzględny rozmiaru: {np.mean(data['fire_size_mae']) if data['fire_size_mae'] else 'brak danych'} akrów")
    print(f"  - Średni czas trwania pożaru: {np.mean(data['fire_duration_mean']) if data['fire_duration_mean'] else 'brak danych'} dni")
    print(f"  - Średni błąd bezwzględny czasu trwania: {np.mean(data['fire_duration_mae']) if data['fire_duration_mae'] else 'brak danych'} dni")
    print(f"  - Procent pokrycia czasowego: {np.mean(data['temporal_coverage_percent']) if data['temporal_coverage_percent'] else 'brak danych'}%")
    print(f"  - Liczba dopasowanych pożarów czasowo: {np.mean(data['temporal_matched']) if data['temporal_matched'] else 'brak danych'}")