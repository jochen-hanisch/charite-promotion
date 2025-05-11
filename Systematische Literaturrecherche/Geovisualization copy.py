import os
import bibtexparser
import pandas as pd
from tqdm import tqdm
import folium
from folium.plugins import Fullscreen

# Terminal bereinigen
os.system('cls' if os.name == 'nt' else 'clear')

# Dateipfade
geonames_file = 'allCountries.txt'
bib_file = 'Research/Charité - Universitätsmedizin Berlin/Systematische Literaturrecherche/Literaturverzeichnis.bib'

# Laden der GeoNames-Daten
print("Laden der GeoNames-Daten...")
geonames_columns = [
    'geonameid', 'name', 'asciiname', 'alternatenames', 'latitude',
    'longitude', 'feature class', 'feature code', 'country code', 'cc2',
    'admin1 code', 'admin2 code', 'admin3 code', 'admin4 code', 'population',
    'elevation', 'dem', 'timezone', 'modification date'
]

chunksize = 10**6
geonames_data = pd.DataFrame()
for chunk in tqdm(pd.read_csv(geonames_file, sep='\t', header=None, names=geonames_columns, chunksize=chunksize, dtype=str, encoding='utf-8')):
    geonames_data = pd.concat([geonames_data, chunk], ignore_index=True)

# Laden der BibTeX-Daten
print("Laden der BibTeX-Daten...")
with open(bib_file, encoding='utf-8') as bibtex_file:
    bib_database = bibtexparser.load(bibtex_file)

# Ortsnamen extrahieren und bereinigen
print("Extrahieren und Bereinigen der Ortsnamen...")
locations = set()
for entry in bib_database.entries:
    if 'address' in entry:
        locations.update(entry['address'].replace(';', ',').replace('&', 'and').split(','))

locations = {loc.strip() for loc in locations}
print(f"Bereinigte Ortsnamen: {locations}")

# Geo-Koordinaten zuordnen
print("Zuordnen der Geo-Koordinaten...")
geo_data = []
for location in tqdm(locations):
    matching_rows = geonames_data[geonames_data['name'].str.contains(location, case=False, na=False)]
    if not matching_rows.empty:
        best_match = matching_rows.iloc[0]
        geo_data.append({
            'name': location,
            'latitude': best_match['latitude'],
            'longitude': best_match['longitude']
        })

if not geo_data:
    print("Keine gültigen Koordinaten gefunden.")
else:
    df = pd.DataFrame(geo_data)

    # Karte erstellen
    print("Erstellen der Karte...")
    m = folium.Map(location=[0, 0], zoom_start=2)

    for _, row in df.iterrows():
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=row['name']
        ).add_to(m)

    # Vollbildmodus und LayerControl hinzufügen
    Fullscreen(position='topright').add_to(m)
    folium.LayerControl().add_to(m)

    # Karte speichern
    m.save('literature_map_with_zoom.html')
    print("Karte wurde gespeichert als 'literature_map_with_zoom.html'.")
