import sqlite3
import csv

# Ścieżka do pliku SQLite
db_path = 'FPA_FOD_20170508.sqlite/FPA_FOD_20170508.sqlite'

# Połączenie z bazą danych SQLite
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Zapytanie SQL do odczytania tabeli 'Fires'
query = "SELECT * FROM Fires"
cursor.execute(query)

# Pobierz nazwy kolumn
columns = [description[0] for description in cursor.description]

# Otwórz plik CSV do zapisu
with open('wildfires.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    # Zapisz nagłówki (nazwy kolumn)
    writer.writerow(columns)

    # Przetwarzaj rekordy linia po linii
    for row in cursor:
        writer.writerow(row)

# Zamknij połączenie z bazą
conn.close()

print("Konwersja zakończona. Plik 'wildfires.csv' został utworzony.")