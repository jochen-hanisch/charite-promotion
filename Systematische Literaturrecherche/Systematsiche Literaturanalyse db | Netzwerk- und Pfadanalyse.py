"""
CAVE!!!!!

Datei muss aus Zotero mit BibTeX exportiert werden!
"""

import os

# Clear the terminal
os.system('cls' if os.name == 'nt' else 'clear')

import bibtexparser
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict, Counter
from itertools import product
from wordcloud import WordCloud
from tabulate import tabulate
import plotly.express as px
import plotly.graph_objects as go
import random
import math
import re
import subprocess

# Export-Flags für Visualisierungen
export_fig_visualize_network = False
export_fig_visualize_tags = False
export_fig_visualize_index = False
export_fig_visualize_research_questions = False
export_fig_visualize_categories = False
export_fig_visualize_time_series = False
export_fig_visualize_top_authors = False
export_fig_visualize_top_publications = False
export_fig_create_path_diagram = False
export_fig_create_sankey_diagram = False
export_fig_visualize_sources_status = False
export_fig_create_wordcloud_from_titles = False
export_fig_visualize_search_term_distribution = False

# Optional: slugify-Funktion
def slugify(value):
    return re.sub(r'[^a-zA-Z0-9_-]', '', value.replace(' ', '_').lower())

# Exportfunktionen für jede Visualisierung
def export_visualize_network(fig):
    if export_fig_visualize_network:
        safe_filename = slugify("visualize_network")
        export_path = f"{safe_filename}.html"
        fig.write_html(export_path, full_html=True, include_plotlyjs="cdn")
        remote_path = "jochen-hanisch@sternenflottenakademie.local:/mnt/deep-space-nine/public/plot/promotion/"
        try:
            subprocess.run(["scp", export_path, remote_path], check=True, capture_output=True, text=True)
            print(f"✅ Datei '{export_path}' erfolgreich übertragen.")
        except subprocess.CalledProcessError as e:
            print("❌ Fehler beim Übertragen:")
            print(e.stderr)

def export_visualize_tags(fig):
    if export_fig_visualize_tags:
        safe_filename = slugify("visualize_tags")
        export_path = f"{safe_filename}.html"
        fig.write_html(export_path, full_html=True, include_plotlyjs="cdn")
        remote_path = "jochen-hanisch@sternenflottenakademie.local:/mnt/deep-space-nine/public/plot/promotion/"
        try:
            subprocess.run(["scp", export_path, remote_path], check=True, capture_output=True, text=True)
            print(f"✅ Datei '{export_path}' erfolgreich übertragen.")
        except subprocess.CalledProcessError as e:
            print("❌ Fehler beim Übertragen:")
            print(e.stderr)

def export_visualize_index(fig):
    if export_fig_visualize_index:
        safe_filename = slugify("visualize_index")
        export_path = f"{safe_filename}.html"
        fig.write_html(export_path, full_html=True, include_plotlyjs="cdn")
        remote_path = "jochen-hanisch@sternenflottenakademie.local:/mnt/deep-space-nine/public/plot/promotion/"
        try:
            subprocess.run(["scp", export_path, remote_path], check=True, capture_output=True, text=True)
            print(f"✅ Datei '{export_path}' erfolgreich übertragen.")
        except subprocess.CalledProcessError as e:
            print("❌ Fehler beim Übertragen:")
            print(e.stderr)

def export_visualize_research_questions(fig):
    if export_fig_visualize_research_questions:
        safe_filename = slugify("visualize_research_questions")
        export_path = f"{safe_filename}.html"
        fig.write_html(export_path, full_html=True, include_plotlyjs="cdn")
        remote_path = "jochen-hanisch@sternenflottenakademie.local:/mnt/deep-space-nine/public/plot/promotion/"
        try:
            subprocess.run(["scp", export_path, remote_path], check=True, capture_output=True, text=True)
            print(f"✅ Datei '{export_path}' erfolgreich übertragen.")
        except subprocess.CalledProcessError as e:
            print("❌ Fehler beim Übertragen:")
            print(e.stderr)

def export_visualize_categories(fig):
    if export_fig_visualize_categories:
        safe_filename = slugify("visualize_categories")
        export_path = f"{safe_filename}.html"
        fig.write_html(export_path, full_html=True, include_plotlyjs="cdn")
        remote_path = "jochen-hanisch@sternenflottenakademie.local:/mnt/deep-space-nine/public/plot/promotion/"
        try:
            subprocess.run(["scp", export_path, remote_path], check=True, capture_output=True, text=True)
            print(f"✅ Datei '{export_path}' erfolgreich übertragen.")
        except subprocess.CalledProcessError as e:
            print("❌ Fehler beim Übertragen:")
            print(e.stderr)

def export_visualize_time_series(fig):
    if export_fig_visualize_time_series:
        safe_filename = slugify("visualize_time_series")
        export_path = f"{safe_filename}.html"
        fig.write_html(export_path, full_html=True, include_plotlyjs="cdn")
        remote_path = "jochen-hanisch@sternenflottenakademie.local:/mnt/deep-space-nine/public/plot/promotion/"
        try:
            subprocess.run(["scp", export_path, remote_path], check=True, capture_output=True, text=True)
            print(f"✅ Datei '{export_path}' erfolgreich übertragen.")
        except subprocess.CalledProcessError as e:
            print("❌ Fehler beim Übertragen:")
            print(e.stderr)

def export_visualize_top_authors(fig):
    if export_fig_visualize_top_authors:
        safe_filename = slugify("visualize_top_authors")
        export_path = f"{safe_filename}.html"
        fig.write_html(export_path, full_html=True, include_plotlyjs="cdn")
        remote_path = "jochen-hanisch@sternenflottenakademie.local:/mnt/deep-space-nine/public/plot/promotion/"
        try:
            subprocess.run(["scp", export_path, remote_path], check=True, capture_output=True, text=True)
            print(f"✅ Datei '{export_path}' erfolgreich übertragen.")
        except subprocess.CalledProcessError as e:
            print("❌ Fehler beim Übertragen:")
            print(e.stderr)

def export_visualize_top_publications(fig):
    if export_fig_visualize_top_publications:
        safe_filename = slugify("visualize_top_publications")
        export_path = f"{safe_filename}.html"
        fig.write_html(export_path, full_html=True, include_plotlyjs="cdn")
        remote_path = "jochen-hanisch@sternenflottenakademie.local:/mnt/deep-space-nine/public/plot/promotion/"
        try:
            subprocess.run(["scp", export_path, remote_path], check=True, capture_output=True, text=True)
            print(f"✅ Datei '{export_path}' erfolgreich übertragen.")
        except subprocess.CalledProcessError as e:
            print("❌ Fehler beim Übertragen:")
            print(e.stderr)

def export_create_path_diagram(fig):
    if export_fig_create_path_diagram:
        safe_filename = slugify("create_path_diagram")
        export_path = f"{safe_filename}.html"
        fig.write_html(export_path, full_html=True, include_plotlyjs="cdn")
        remote_path = "jochen-hanisch@sternenflottenakademie.local:/mnt/deep-space-nine/public/plot/promotion/"
        try:
            subprocess.run(["scp", export_path, remote_path], check=True, capture_output=True, text=True)
            print(f"✅ Datei '{export_path}' erfolgreich übertragen.")
        except subprocess.CalledProcessError as e:
            print("❌ Fehler beim Übertragen:")
            print(e.stderr)

def export_create_sankey_diagram(fig):
    if export_fig_create_sankey_diagram:
        safe_filename = slugify("create_sankey_diagram")
        export_path = f"{safe_filename}.html"
        fig.write_html(export_path, full_html=True, include_plotlyjs="cdn")
        remote_path = "jochen-hanisch@sternenflottenakademie.local:/mnt/deep-space-nine/public/plot/promotion/"
        try:
            subprocess.run(["scp", export_path, remote_path], check=True, capture_output=True, text=True)
            print(f"✅ Datei '{export_path}' erfolgreich übertragen.")
        except subprocess.CalledProcessError as e:
            print("❌ Fehler beim Übertragen:")
            print(e.stderr)

def export_visualize_sources_status(fig):
    if export_fig_visualize_sources_status:
        safe_filename = slugify("visualize_sources_status")
        export_path = f"{safe_filename}.html"
        fig.write_html(export_path, full_html=True, include_plotlyjs="cdn")
        remote_path = "jochen-hanisch@sternenflottenakademie.local:/mnt/deep-space-nine/public/plot/promotion/"
        try:
            subprocess.run(["scp", export_path, remote_path], check=True, capture_output=True, text=True)
            print(f"✅ Datei '{export_path}' erfolgreich übertragen.")
        except subprocess.CalledProcessError as e:
            print("❌ Fehler beim Übertragen:")
            print(e.stderr)

# Farben definieren
colors = {
    "background": "#003366",            # Hintergrundfarbe
    "text": "#333333",                  # Textfarbe
    "accent": "#663300",                # Akzentfarbe
    "primaryLine": "#660066",           # Bildungswirkfaktor
    "secondaryLine": "#cc6600",         # Bildungswirkindikator
    "depthArea": "#006666",             # Kompetenzmessunsicherheit
    "brightArea": "#66CCCC",            # Kompetenzentwicklungsunsicherheit
    "positiveHighlight": "#336600",     # Positive Hervorhebung
    "negativeHighlight": "#990000",     # Negative Hervorhebung
    "white": "#ffffff"                  # Weiß
}

# Liste der Farben, die für die Wörter verwendet werden sollen
word_colors = [
    colors["white"],
    colors["brightArea"],
    colors["positiveHighlight"],
    colors["negativeHighlight"]
]

# Aktuelles Datum
current_date = datetime.now().strftime("%Y-%m-%d")


# Lade Zotero-SQLite-Datenbank und erzeuge bib_database.entries-ähnliche Struktur
import sqlite3

def load_zotero_entries(sqlite_path):
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()

    query = """
    SELECT
        items.itemID,
        COALESCE(value_title.value, '') AS title,
        COALESCE(value_year.value, '') AS year,
        COALESCE(creators.lastName || ', ' || creators.firstName, '') AS author,
        GROUP_CONCAT(DISTINCT tags.name) AS keywords,
        itemTypes.typeName AS type
    FROM items

    -- Titel
    LEFT JOIN itemData AS title_data ON items.itemID = title_data.itemID
    LEFT JOIN fields AS title_field ON title_data.fieldID = title_field.fieldID AND title_field.fieldName = 'title'
    LEFT JOIN itemDataValues AS value_title ON title_data.valueID = value_title.valueID

    -- Jahr
    LEFT JOIN itemData AS year_data ON items.itemID = year_data.itemID
    LEFT JOIN fields AS year_field ON year_data.fieldID = year_field.fieldID AND year_field.fieldName = 'date'
    LEFT JOIN itemDataValues AS value_year ON year_data.valueID = value_year.valueID

    -- Autoren
    LEFT JOIN itemCreators ON items.itemID = itemCreators.itemID
    LEFT JOIN creators ON itemCreators.creatorID = creators.creatorID

    -- Tags
    LEFT JOIN itemTags ON items.itemID = itemTags.itemID
    LEFT JOIN tags ON itemTags.tagID = tags.tagID

    -- Typ
    LEFT JOIN itemTypes ON items.itemTypeID = itemTypes.itemTypeID

    -- Sammlungen
    LEFT JOIN collectionItems ON items.itemID = collectionItems.itemID
    LEFT JOIN collections ON collectionItems.collectionID = collections.collectionID

    WHERE collections.collectionName IN (
        'S:01 Learning Management System',
        'S:02 Online-Lernplattform',
        'S:03 Online-Lernumgebung',
        'S:05 eLearning',
        'S:04 MOOC',
        'S:06 Bildungstechnologie',
        'S:07 Digitale Medien',
        'S:08 Blended Learning',
        'S:09 Digitales Lernen',
        'S:12 Digital Learning',
        'S:10 Online Lernen',
        'S:11 Online Learning',
        'S:13 Berichte',
        'S:14 Agiles Lernen',
        'S:15 Learning Analytics',
        'S:16 Dissertationen',
        'S:17 ePortfolio'
    )
    GROUP BY items.itemID
    """

    cursor.execute(query)
    rows = cursor.fetchall()

    # Umwandlung in bib_database.entries-kompatibles Format
    entries = []
    for row in rows:
        item = {
            'ID': str(row[0]),
            'title': row[1],
            'year': row[2],
            'author': row[3],
            'keywords': row[4] if row[4] else '',
            'ENTRYTYPE': row[5]
        }
        entries.append(item)

    conn.close()
    return entries

bib_database = type("BibDatabase", (object,), {})()
bib_database.entries = load_zotero_entries('/Users/jochen_hanisch-johannsen/Zotero/zotero.sqlite')

# Stopplisten laden
with open('de_complete.txt', 'r', encoding='utf-8') as file:
    stop_words_de = set(file.read().split())

with open('en_complete.txt', 'r', encoding='utf-8') as file:
    stop_words_en = set(file.read().split())

# Kombinierte Stoppliste
stop_words = stop_words_de.union(stop_words_en)

# Funktion zur Berechnung der Stichprobengröße
def calculate_sample_size(N, Z=1.96, p=0.5, e=0.05):
    n_0 = (Z**2 * p * (1 - p)) / (e**2)
    n = n_0 / (1 + ((n_0 - 1) / N))
    return math.ceil(n)

# Visualisierung 1: Netzwerkanalyse
# Visualisierung 1: Netzwerkanalyse
def visualize_network(bib_database):
    search_terms = {
        '0': 'digital:learning',
        '1': 'learning:management:system',
        '2': 'online:Lernplattform',
        '3': 'online:Lernumgebung',
        '4': 'MOOC',
        '5': 'e-learning',
        '6': 'Bildung:Technologie',
        '7': 'digital:Medien',
        '8': 'blended:learning',
        '9': 'digital:lernen',
        'a': 'online:lernen',
        'b': 'online:learning'
    }

    numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b']
    types = [
        'Zeitschriftenartikel',
        'Buch',
        'Buchteil',
        'Bericht',
        'Konferenz-Paper'
    ]
    tags_to_search = set()
    for number, type_ in product(numbers, types):
        search_term = search_terms[number]
        tag = f'#{number}:{type_}:{search_term}'
        tags_to_search.add(tag.lower())

    tag_counts = defaultdict(int)
    for entry in bib_database.entries:
        if 'keywords' in entry:
            entry_keywords = list(map(str.lower, map(str.strip, entry['keywords'].replace('\\#', '#').split(','))))
            for keyword in entry_keywords:
                for tag in tags_to_search:
                    if tag in keyword:
                        tag_counts[tag] += 1

    fundzahlen = defaultdict(int)
    for tag, count in tag_counts.items():
        search_term = tag.split(':')[-1]
        for key, value in search_terms.items():
            if search_term == value:
                fundzahlen[value] += count

    search_terms_network = {
        "Primäre Begriffe": {
            "learning:management:system": [
                "e-learning",
                "bildung:technologie",
                "online:lernplattform",
                "online:lernumgebung",
                "digital:learning",
                "digitales:lernen"
            ]
        },
        "Sekundäre Begriffe": {
            "e-learning": [
                "mooc",
                "online:lernplattform"
            ],
            "bildung:technologie": [
                "digital:learning",
                "digitales:lernen",
                "blended:learning"
            ],
            "digital:learning": [
                "digitale:medien",
                "online:learning"
            ],
            "digitales:lernen": [
                "digitale:medien",
                "online:lernen"
            ],
            "blended:learning": ["mooc"]
        },
        "Tertiäre Begriffe": {
            "online:learning": [],
            "online:lernen": []
        }
    }

    G = nx.Graph()

    hierarchy_colors = {
        "Primäre Begriffe": colors['primaryLine'],
        "Sekundäre Begriffe": colors['secondaryLine'],
        "Tertiäre Begriffe": colors['brightArea']
    }

    def add_terms_to_graph(level, terms):
        for primary_term, related_terms in terms.items():
            if primary_term not in G:
                G.add_node(primary_term, color=hierarchy_colors[level], size=fundzahlen.get(primary_term, 10))
            else:
                if level == "Tertiäre Begriffe":
                    G.nodes[primary_term]['color'] = hierarchy_colors[level]
            for related_term in related_terms:
                if related_term not in G:
                    G.add_node(related_term, color=hierarchy_colors[level], size=fundzahlen.get(related_term, 10))
                else:
                    if level == "Tertiäre Begriffe":
                        G.nodes[related_term]['color'] = hierarchy_colors[level]
                G.add_edge(primary_term, related_term)

    for level, terms in search_terms_network.items():
        add_terms_to_graph(level, terms)

    np.random.seed(42)
    pos = nx.spring_layout(G)

    x_scale_min, x_scale_max = 0, 10
    y_scale_min, y_scale_max = 0, 10

    min_x = min(pos[node][0] for node in pos)
    max_x = max(pos[node][0] for node in pos)
    min_y = min(pos[node][1] for node in pos)
    max_y = max(pos[node][1] for node in pos)

    scale_x_range = x_scale_max - x_scale_min
    scale_y_range = y_scale_max - y_scale_min

    for node in pos:
        x, y = pos[node]
        norm_x = scale_x_range * (x - min_x) / (max_x - min_x) + x_scale_min
        norm_y = scale_y_range * (y - min_y) / (max_y - min_y) + y_scale_min
        pos[node] = (norm_x, norm_y)

    for node in pos:
        x, y = pos[node]
        x = max(min(x, x_scale_max), x_scale_min)
        y = max(min(y, y_scale_max), y_scale_min)
        pos[node] = (x, y)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color=colors['white']),
        hoverinfo='none',
        mode='lines')

    # Knoten in drei Traces aufteilen: Primär, Sekundär, Tertiär
    primary_nodes = []
    secondary_nodes = []
    tertiary_nodes = []

    for node in G.nodes():
        color = G.nodes[node]['color']
        size = math.log(G.nodes[node].get('size', 10) + 1) * 10
        x, y = pos[node]
        hovertext = f"{node}<br>Anzahl Funde: {fundzahlen.get(node, 0)}"
        node_data = dict(x=x, y=y, text=node, size=size, hovertext=hovertext)
        if color == colors['primaryLine']:
            primary_nodes.append(node_data)
        elif color == colors['secondaryLine']:
            secondary_nodes.append(node_data)
        elif color == colors['brightArea']:
            tertiary_nodes.append(node_data)

    def create_node_trace(nodes, name, color):
        return go.Scatter(
            x=[n['x'] for n in nodes],
            y=[n['y'] for n in nodes],
            mode='markers+text',
            text=[n['text'] for n in nodes],
            hovertext=[n['hovertext'] for n in nodes],
            hoverinfo='text',
            marker=dict(
                size=[n['size'] for n in nodes],
                color=color,
                line_width=2
            ),
            textposition="top center",
            textfont=dict(size=12),
            name=name
        )

    primary_trace = create_node_trace(primary_nodes, "Primäre Begriffe", colors['primaryLine'])
    secondary_trace = create_node_trace(secondary_nodes, "Sekundäre Begriffe", colors['secondaryLine'])
    tertiary_trace = create_node_trace(tertiary_nodes, "Tertiäre Begriffe", colors['brightArea'])

    fig = go.Figure(data=[edge_trace, primary_trace, secondary_trace, tertiary_trace],
                    layout=go.Layout(
                        title=f'Suchbegriff-Netzwerk nach Relevanz und Semantik (n={sum(fundzahlen.values())}, Stand: {current_date})',
                        titlefont_size=16,
                        showlegend=True,
                        legend=dict(
                            bgcolor=colors['background'],
                            bordercolor=colors['white'],
                            borderwidth=1,
                            font=dict(color=colors['white']),
                            itemsizing='constant'
                        ),
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(
                            range=[x_scale_min, x_scale_max + 1],
                            showgrid=True,
                            zeroline=True,
                            tickmode='linear',
                            tick0=x_scale_min,
                            dtick=(x_scale_max - x_scale_min) / 4,
                            title='Technologische Dimension'
                        ),
                        yaxis=dict(
                            range=[y_scale_min, y_scale_max + 1],
                            showgrid=True,
                            zeroline=True,
                            tickmode='linear',
                            tick0=y_scale_min,
                            dtick=(y_scale_max - y_scale_min) / 4,
                            title='Pädagogische Dimension'
                        ),
                        plot_bgcolor=colors['background'],
                        paper_bgcolor=colors['background'],
                        font=dict(color=colors['white'])
                    ))

    fig.show()
    export_visualize_network(fig)

    # Einfache Pfadanalyse nach dem Anzeigen der Figur
    if 'e-learning' in G and 'online:lernen' in G:
        try:
            pfad = nx.shortest_path(G, source='e-learning', target='online:lernen')
            print(f"Kürzester Pfad von 'e-learning' zu 'online:lernen': {pfad}")
        except nx.NetworkXNoPath:
            print("Kein Pfad von 'e-learning' zu 'online:lernen' gefunden.")

 # Visualisierung 2: Häufigkeit spezifischer Tags
def visualize_tags(bib_database):
    # Definierte Suchbegriffe
    search_terms = {
        '0': 'digital:learning',
        '1': 'learning:management:system',
        '2': 'online:Lernplattform',
        '3': 'online:Lernumgebung',
        '4': 'MOOC',
        '5': 'e-learning',
        '6': 'Bildung:Technologie',
        '7': 'digital:Medien',
        '8': 'blended:learning',
        '9': 'digital:lernen',
        'a': 'online:lernen',
        'b': 'online:learning'
    }

    # Kombinierte Tags erzeugen
    numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b']
    types = [
        'Zeitschriftenartikel',
        'Buch',
        'Buchteil',
        'Bericht',
        'Konferenz-Paper'
    ]
    tags_to_search = set(
        f"#{number}:{type_}:{search_terms[number]}"
        for number, type_ in product(numbers, types)
    )

    # Tag-Zählungen initialisieren
    tag_counts = defaultdict(int)
    if not bib_database or not bib_database.entries:
        print("Fehler: Keine Einträge in der Datenbank gefunden.")
        return

    for entry in bib_database.entries:
        if 'keywords' in entry:
            entry_keywords = map(
                str.lower,
                map(str.strip, entry['keywords'].replace('\\#', '#').split(','))
            )
            for keyword in entry_keywords:
                for tag in tags_to_search:
                    if tag in keyword:
                        tag_counts[tag] += 1

    # Daten für Visualisierung aufbereiten
    data = [
        {'Tag': tag, 'Count': count, 'Type': tag.split(':')[1].lower()}
        for tag, count in tag_counts.items()
        if count > 0
    ]

    if not data:
        print("Warnung: Keine Tags gefunden, die den Suchkriterien entsprechen.")
        return

    # Farbzuordnung
    color_map = {
        'zeitschriftenartikel': colors['primaryLine'],
        'konferenz-paper': colors['secondaryLine'],
        'buch': colors['depthArea'],
        'buchteil': colors['brightArea'],
        'bericht': colors['accent']
    }

    # Visualisierung erstellen
    total_count = sum(tag_counts.values())
    fig = px.bar(
        data,
        x='Tag',
        y='Count',
        title=f'Häufigkeit der Suchbegriffe in der Literaturanalyse (n={total_count}, Stand: {current_date})',
        labels={'Tag': 'Tag', 'Count': 'Anzahl der Vorkommen'},
        color='Type',
        color_discrete_map=color_map,
        text_auto=True
    )

    # Layout anpassen
    fig.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font=dict(color=colors['white']),
        margin=dict(l=0, r=0, t=40, b=40),
        autosize=True
    )

    fig.update_traces(
        marker_line_color=colors['white'],
        marker_line_width=1.5
    )

    fig.show(config={"responsive": True})
    export_visualize_tags(fig)

 # Visualisierung 3: Häufigkeit Index
def visualize_index(bib_database):
    index_terms = [
        'Lernsystemarchitektur',
        'Bildungstheorien',
        'Lehr- und Lerneffektivität',
        'Kollaboratives Lernen',
        'Bewertungsmethoden',
        'Technologieintegration',
        'Datenschutz und IT-Sicherheit',
        'Systemanpassung',
        'Krisenreaktion im Bildungsbereich',
        'Forschungsansätze'
    ]

    index_counts = defaultdict(int)
    for entry in bib_database.entries:
        if 'keywords' in entry:
            entry_keywords = list(map(str.lower, map(str.strip, entry['keywords'].replace('\\#', '#').split(','))))
            for index_term in index_terms:
                if index_term.lower() in entry_keywords:
                    index_counts[index_term] += 1

    index_data = [{'Index': index, 'Count': count} for index, count in index_counts.items()]
    index_data = sorted(index_data, key=lambda x: x['Count'], reverse=True)

    total_count = sum(index_counts.values())
    print(f"Häufigkeit Indizes (Gesamtanzahl: {total_count}):")
    print(tabulate(index_data, headers="keys", tablefmt="grid"))

    fig = px.bar(index_data, x='Index', y='Count', title=f'Relevanzschlüssel nach Indexkategorien (n={total_count}, Stand: {current_date})', labels={'Index': 'Index', 'Count': 'Anzahl der Vorkommen'}, text_auto=True)

    fig.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font=dict(color=colors['white']),
        margin=dict(l=0, r=0, t=40, b=40),
        autosize=True
    )

    fig.update_traces(marker_color=colors['primaryLine'], marker_line_color=colors['white'], marker_line_width=1.5)

    fig.show(config={"responsive": True})
    export_visualize_index(fig)

 # Visualisierung 4: Häufigkeit Forschungsunterfragen
def visualize_research_questions(bib_database):
    research_questions = {
        'promotion:fu1': 'Akzeptanz und Nützlichkeit (FU1)',
        'promotion:fu2a': 'Effekt für Lernende (FU2a)',
        'promotion:fu2b': 'Effekt-Faktoren für Lehrende (FU2b)',
        'promotion:fu3': 'Konzeption und Merkmale (FU3)',
        'promotion:fu4a': 'Bildungswissenschaftliche Mechanismen (FU4a)',
        'promotion:fu4b': 'Technisch-gestalterische Mechanismen (FU4b)',
        'promotion:fu5': 'Möglichkeiten und Grenzen (FU5)',
        'promotion:fu6': 'Beurteilung als Kompetenzerwerbssystem (FU6)',
        'promotion:fu7': 'Inputs und Strategien (FU7)'
    }

    rq_counts = defaultdict(int)
    for entry in bib_database.entries:
        if 'keywords' in entry:
            entry_keywords = list(map(str.lower, map(str.strip, entry['keywords'].replace('\\#', '#').split(','))))
            for keyword in entry_keywords:
                if keyword in research_questions:
                    rq_counts[keyword] += 1

    rq_data = [{'Research_Question': research_questions[keyword], 'Count': count} for keyword, count in rq_counts.items()]
    rq_data = sorted(rq_data, key=lambda x: x['Count'], reverse=True)

    rq_data_df = pd.DataFrame(rq_data)

    total_count = rq_data_df['Count'].sum()
    print(f"Häufigkeit Forschungsunterfragen (Gesamtanzahl: {total_count}):")
    print(tabulate(rq_data, headers="keys", tablefmt="grid"))

    fig = px.bar(rq_data_df, x='Research_Question', y='Count', title=f'Zuordnung der Literatur zu Forschungsunterfragen (n={total_count}, Stand: {current_date})', labels={'Research_Question': 'Forschungsunterfrage', 'Count': 'Anzahl der Vorkommen'}, text_auto=True)

    fig.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font=dict(color=colors['white']),
        margin=dict(l=0, r=0, t=40, b=40),
        autosize=True
    )

    fig.update_traces(marker_color=colors['primaryLine'], marker_line_color=colors['white'], marker_line_width=1.5)

    fig.show(config={"responsive": True})
    export_visualize_research_questions(fig)

 # Visualisierung 5: Häufigkeit spezifischer Kategorien
def visualize_categories(bib_database):
    categories = {
        'promotion:argumentation': 'Argumentation',
        'promotion:kerngedanke': 'Kerngedanke',
        'promotion:weiterführung': 'Weiterführung',
        'promotion:schlussfolgerung': 'Schlussfolgerung'
    }

    cat_counts = defaultdict(int)
    for entry in bib_database.entries:
        if 'keywords' in entry:
            entry_keywords = list(map(str.lower, map(str.strip, entry['keywords'].replace('\\#', '#').split(','))))
            for keyword in entry_keywords:
                if keyword in categories:
                    cat_counts[keyword] += 1

    cat_data = [{'Category': categories[keyword], 'Count': count} for keyword, count in cat_counts.items()]
    cat_data = sorted(cat_data, key=lambda x: x['Count'], reverse=True)

    cat_data_df = pd.DataFrame(cat_data)

    total_count = cat_data_df['Count'].sum()
    print(f"Häufigkeit Kategorien (Gesamtanzahl: {total_count}):")
    print(tabulate(cat_data, headers="keys", tablefmt="grid"))

    fig = px.bar(cat_data_df, x='Category', y='Count', title=f'Textsortenzuordnung der analysierten Quellen (n={total_count}, Stand: {current_date})', labels={'Category': 'Kategorie', 'Count': 'Anzahl der Vorkommen'}, text_auto=True)

    fig.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font=dict(color=colors['white']),
        margin=dict(l=0, r=0, t=40, b=40),
        autosize=True
    )

    fig.update_traces(marker_color=colors['primaryLine'], marker_line_color=colors['white'], marker_line_width=1.5)

    fig.show(config={"responsive": True})
    export_visualize_categories(fig)

 # Zeitreihenanalyse der Veröffentlichungen
def extract_year_from_entry(entry):
    year_str = entry.get('year', '').strip()
    if not year_str:
        return None
    try:
        matches = re.findall(r'\b(19[0-9]{2}|20[0-9]{2})\b', year_str)
        years = [int(y) for y in matches if 1900 <= int(y) <= datetime.now().year + 1]
        return min(years) if years else None
    except Exception as e:
        print(f"⚠️ Fehler bei Jahreswert '{year_str}': {e}")
        return None

def visualize_time_series(bib_database):
    publication_years = []

    for entry in bib_database.entries:
        year = extract_year_from_entry(entry)
        if year is not None:
            publication_years.append(year)

    if publication_years:
        year_counts = Counter(publication_years)
        df = pd.DataFrame(year_counts.items(), columns=['Year', 'Count']).sort_values('Year')

        fig = px.line(
            df,
            x='Year',
            y='Count',
            title=f'Jährliche Veröffentlichungen in der Literaturanalyse (n={sum(year_counts.values())}, Stand: {current_date})',
            labels={'Year': 'Jahr', 'Count': 'Anzahl der Veröffentlichungen'}
        )

        fig.update_layout(
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            font=dict(color=colors['white']),
            xaxis=dict(
                tickmode='linear',
                dtick=2,
                tick0=min(publication_years)
            ),
            margin=dict(l=0, r=0, t=40, b=40),
            autosize=True
        )

        fig.update_traces(line=dict(color=colors['secondaryLine'], width=3))
        fig.show(config={"responsive": True})
        export_visualize_time_series(fig)
    else:
        print("Keine gültigen Veröffentlichungsjahre gefunden.")

 # Top Autoren nach Anzahl der Werke
def visualize_top_authors(bib_database):
    top_n = 25  # Anzahl der Top-Autoren, die angezeigt werden sollen
    author_counts = defaultdict(int)
    for entry in bib_database.entries:
        if 'author' in entry and entry['author'].strip():
            authors = [a.strip() for a in entry['author'].split(' and ') if a.strip()]
            for author in authors:
                author_counts[author] += 1

    top_authors = Counter(author_counts).most_common(top_n)
    if top_authors:
        df = pd.DataFrame(top_authors, columns=['Author', 'Count'])

        fig = px.bar(df, x='Author', y='Count', title=f'Meistgenannte Autor:innen in der Literaturanalyse (Top {top_n}, n={sum(author_counts.values())}, Stand: {current_date})', labels={'Author': 'Autor', 'Count': 'Anzahl der Werke'}, text_auto=True)
        fig.update_layout(
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            font=dict(color=colors['white']),
            margin=dict(l=0, r=0, t=40, b=40),
            autosize=True
        )
        fig.update_traces(marker_color=colors['primaryLine'], marker_line_color=colors['white'], marker_line_width=1.5)

        fig.show(config={"responsive": True})
        export_visualize_top_authors(fig)
    else:
        print("Keine Autoren gefunden.")


 # Top Titel nach Anzahl der Werke
def normalize_title(title):
    # Entfernen von Sonderzeichen und Standardisierung auf Kleinbuchstaben
    title = title.lower().translate(str.maketrans('', '', ",.!?\"'()[]{}:;"))
    # Zusammenführen ähnlicher Titel, die sich nur in geringfügigen Details unterscheiden
    title = " ".join(title.split())
    # Entfernen häufiger Füllwörter oder Standardphrasen, die die Unterscheidung nicht unterstützen
    common_phrases = ['eine studie', 'untersuchung der', 'analyse von']
    for phrase in common_phrases:
        title = title.replace(phrase, '')
    return title.strip()

def visualize_top_publications(bib_database):
    top_n = 25  # Anzahl der Top-Publikationen, die angezeigt werden sollen
    publication_counts = defaultdict(int)
    
    for entry in bib_database.entries:
        invalid_titles = {"pdf", "no title found", "published entry", "", None}
        title = normalize_title(entry.get('title', ''))
        if title.lower() not in invalid_titles and len(title) > 5:
            publication_counts[title] += 1

    top_publications = sorted(publication_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    publication_data = [{'Title': title[:50] + '...' if len(title) > 50 else title, 'Count': count} for title, count in top_publications]

    df = pd.DataFrame(publication_data)
    
    fig = px.bar(df, x='Title', y='Count', title=f'Häufig zitierte Publikationen in der Analyse (Top {top_n}, n={sum(publication_counts.values())}, Stand: {current_date})', labels={'Title': 'Titel', 'Count': 'Anzahl der Nennungen'})
    
    fig.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font=dict(color=colors['white']),
        xaxis_tickangle=-45,
        margin=dict(l=0, r=0, t=40, b=40),
        autosize=True
    )
    
    fig.update_traces(marker_color=colors['primaryLine'], marker_line_color=colors['white'], marker_line_width=1.5)
    
    fig.show(config={"responsive": True})
    export_visualize_top_publications(fig)



##########


# Daten vorbereiten
def prepare_path_data(bib_database):
    research_questions = {
        'promotion:fu1': 'Akzeptanz und Nützlichkeit (FU1)',
        'promotion:fu2a': 'Effekt für Lernende (FU2a)',
        'promotion:fu2b': 'Effekt-Faktoren für Lehrende (FU2b)',
        'promotion:fu3': 'Konzeption und Merkmale (FU3)',
        'promotion:fu4a': 'Bildungswissenschaftliche Mechanismen (FU4a)',
        'promotion:fu4b': 'Technisch-gestalterische Mechanismen (FU4b)',
        'promotion:fu5': 'Möglichkeiten und Grenzen (FU5)',
        'promotion:fu6': 'Beurteilung als Kompetenzerwerbssystem (FU6)',
        'promotion:fu7': 'Inputs und Strategien (FU7)'
    }

    categories = {
        'promotion:argumentation': 'Argumentation',
        'promotion:kerngedanke': 'Kerngedanke',
        'promotion:weiterführung': 'Weiterführung',
        'promotion:schlussfolgerung': 'Schlussfolgerung'
    }

    index_terms = [
        'Lernsystemarchitektur',
        'Bildungstheorien',
        'Lehr- und Lerneffektivität',
        'Kollaboratives Lernen',
        'Bewertungsmethoden',
        'Technologieintegration',
        'Datenschutz und IT-Sicherheit',
        'Systemanpassung',
        'Krisenreaktion im Bildungsbereich',
        'Forschungsansätze'
    ]

    entry_types = [
        'Zeitschriftenartikel',
        'Buch',
        'Buchteil',
        'Bericht',
        'Konferenz-Paper'
    ]

    data = []

    for entry in bib_database.entries:
        entry_data = {
            'FU': None,
            'Category': None,
            'Index': None,
            'Type': entry.get('ENTRYTYPE', '').lower()
        }

        if 'keywords' in entry:
            entry_keywords = list(map(str.lower, map(str.strip, entry['keywords'].replace('\\#', '#').split(','))))

            for key, value in research_questions.items():
                if key in entry_keywords:
                    entry_data['FU'] = value

            for key, value in categories.items():
                if key in entry_keywords:
                    entry_data['Category'] = value

            for index_term in index_terms:
                if index_term.lower() in entry_keywords:
                    entry_data['Index'] = index_term

        if all(value is not None for value in entry_data.values()):
            data.append(entry_data)

    return data

 # Pfaddiagramm erstellen
def create_path_diagram(data):
    labels = []
    sources = []
    targets = []
    values = []
    color_map = {
        'zeitschriftenartikel': colors['primaryLine'],
        'konferenz-paper': colors['secondaryLine'],
        'buch': colors['depthArea'],
        'buchteil': colors['brightArea'],
        'bericht': colors['accent']
    }

    def add_to_labels(label):
        if label not in labels:
            labels.append(label)
        return labels.index(label)

    for entry in data:
        fu_idx = add_to_labels(entry['FU'])
        category_idx = add_to_labels(entry['Category'])
        index_idx = add_to_labels(entry['Index'])
        type_idx = add_to_labels(entry['Type'])

        sources.extend([fu_idx, category_idx, index_idx])
        targets.extend([category_idx, index_idx, type_idx])
        values.extend([1, 1, 1])

    node_colors = [color_map.get(label, colors['primaryLine']) for label in labels]

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color=colors['white'], width=0.5),
            label=labels,
            color=node_colors
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values
        )
    )])

    fig.update_layout(
        title_text=f'Kategorischer Analysepfad der Literatur (n={len(data)}, Stand: {current_date})',
        font=dict(size=10, color=colors['white']),
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background']
    )

    fig.show()
    export_create_path_diagram(fig)


#############

def create_sankey_diagram(bib_database):
    def extract_year(entry):
        """
        Extrahiert ein gültiges Jahr aus dem `year`-Feld eines Eintrags.
        """
        year_str = entry.get('year', '').strip()
        try:
            # Suche nach einer 4-stelligen Jahreszahl
            year_match = re.search(r'\b\d{4}\b', year_str)
            if year_match:
                return int(year_match.group())
            else:
                raise ValueError(f"Kein gültiges Jahr gefunden: {year_str}")
        except ValueError:
            print(f"Warnung: Ungültiger Jahreswert in Eintrag übersprungen: {year_str}")
            return None

    current_year = datetime.now().year
    filtered_entries = [
        entry for entry in bib_database.entries
        if 'promotion:literaturanalyse' in entry.get('keywords', '').lower()
    ]

    initial_sources = len(filtered_entries)
    screened_sources = initial_sources  # Da bereits gefiltert
    quality_sources = sum(
        1 for entry in filtered_entries
        if entry.get('ENTRYTYPE') in ['article', 'phdthesis']
    )
    relevance_sources = sum(
        1 for entry in filtered_entries
        if entry.get('ENTRYTYPE') in ['article', 'phdthesis']
        and any(rq in entry.get('keywords', '').lower() for rq in ['promotion:fu3', 'promotion:kerngedanke'])
    )
    thematic_sources = sum(
        1 for entry in filtered_entries
        if entry.get('ENTRYTYPE') in ['article', 'phdthesis']
        and any(rq in entry.get('keywords', '').lower() for rq in ['promotion:fu3', 'promotion:kerngedanke'])
        and any(kw in entry.get('keywords', '').lower() for kw in ['digital', 'learning'])
    )
    recent_sources = sum(
        1 for entry in filtered_entries
        if entry.get('ENTRYTYPE') in ['article', 'phdthesis']
        and any(rq in entry.get('keywords', '').lower() for rq in ['promotion:fu3', 'promotion:kerngedanke'])
        and any(kw in entry.get('keywords', '').lower() for kw in ['digital', 'learning'])
        and (year := extract_year(entry)) and year >= current_year - 5
    )
    classic_sources = sum(
        1 for entry in filtered_entries
        if entry.get('ENTRYTYPE') in ['article', 'phdthesis']
        and any(rq in entry.get('keywords', '').lower() for rq in ['promotion:fu3', 'promotion:kerngedanke'])
        and any(kw in entry.get('keywords', '').lower() for kw in ['digital', 'learning'])
        and (year := extract_year(entry)) and year < current_year - 5
        and 'classic' in entry.get('keywords', '').lower()
    )
    selected_sources = recent_sources + classic_sources

    # Stichprobengröße berechnen
    sample_size = calculate_sample_size(initial_sources)

    # Phasen und Verbindungen definieren
    phases = [
        "Identifizierte Quellen",
        "Nach Screening (Literaturanalyse-Markierung)",
        "Nach Qualitätsprüfung (Artikel und Dissertationen)",
        "Nach Relevanzprüfung (FU3 und Kerngedanken)",
        "Nach thematischer Prüfung (Digital & Learning)",
        "Aktuelle Forschung (letzte 5 Jahre)",
        "Klassische Werke",
        "Ausgewählte Quellen (Endauswahl)"
    ]

    sources = [0, 1, 2, 3, 4, 4, 4]
    targets = [1, 2, 3, 4, 5, 6, 7]
    values = [
        screened_sources,
        quality_sources,
        relevance_sources,
        thematic_sources,
        recent_sources,
        classic_sources,
        selected_sources
    ]

    # Prozentsätze berechnen für die Labels
    percentages = [
        "100.0%",  # Startwert
        f"{screened_sources / initial_sources * 100:.1f}%",
        f"{quality_sources / screened_sources * 100:.1f}%" if screened_sources > 0 else "0.0%",
        f"{relevance_sources / quality_sources * 100:.1f}%" if quality_sources > 0 else "0.0%",
        f"{thematic_sources / relevance_sources * 100:.1f}%" if relevance_sources > 0 else "0.0%",
        f"{recent_sources / thematic_sources * 100:.1f}%" if thematic_sources > 0 else "0.0%",
        f"{classic_sources / thematic_sources * 100:.1f}%" if thematic_sources > 0 else "0.0%",
        f"{selected_sources / (recent_sources + classic_sources) * 100:.1f}%" if (recent_sources + classic_sources) > 0 else "0.0%"
    ]

    # Labels für Knoten anpassen, um Prozentsätze anzuzeigen
    node_labels = [f"{ph} ({pct})" for ph, pct in zip(phases, percentages)]

    # Farben für die einzelnen Phasen
    node_colors = [
        colors['primaryLine'],          # Identifizierte Quellen
        colors['secondaryLine'],        # Nach Screening
        colors['brightArea'],           # Nach Qualitätsprüfung
        colors['depthArea'],            # Nach Relevanzprüfung
        colors['positiveHighlight'],    # Nach thematischer Prüfung
        colors['negativeHighlight'],    # Aktuelle Forschung
        colors['accent'],               # Klassische Werke
        colors['positiveHighlight']     # Ausgewählte Quellen
    ]

    # Sankey-Diagramm erstellen
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels,
            color=node_colors
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            hoverinfo='all',  # Zeigt detaillierte Infos bei Mouseover an
            color=colors['accent']
        )
    ))

    # Layout anpassen
    fig.update_layout(
        title_text=f"Flussdiagramm der Literaturselektion (Stichprobe: n={sample_size}, Stand: {current_date})",
        font_size=12,  # Größere Schriftgröße für bessere Lesbarkeit
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font=dict(color=colors['white'])
    )

    fig.show()
    export_create_sankey_diagram(fig)

##########

def calculate_sample_size(N, Z=1.96, p=0.5, e=0.05):
    """
    Berechnet die Stichprobengröße basierend auf der Gesamtanzahl der Einträge (N).
    """
    if N <= 0:
        return 0
    n_0 = (Z**2 * p * (1 - p)) / (e**2)
    n = n_0 / (1 + ((n_0 - 1) / N))
    return math.ceil(n)

def visualize_sources_status(bib_database):
    """
    Visualisiert den Status der analysierten und nicht analysierten Quellen pro Suchordner.
    """
    search_folder_tags = [
        "#1:zeitschriftenartikel:learning:management:system",
        "#2:zeitschriftenartikel:online:lernplattform",
        "#3:zeitschriftenartikel:online:lernumgebung",
        "#4:zeitschriftenartikel:mooc",
        "#5:zeitschriftenartikel:e-learning",
        "#6:zeitschriftenartikel:bildung:technologie",
        "#7:zeitschriftenartikel:digital:medien",
        "#8:zeitschriftenartikel:blended:learning",
        "#9:zeitschriftenartikel:digital:lernen",
        "#a:zeitschriftenartikel:online:lernen",
        "#b:zeitschriftenartikel:online:learning",
        "#0:zeitschriftenartikel:digital:learning",
        "#1:konferenz-paper:learning:management:system",
        "#2:konferenz-paper:online:lernplattform",
        "#3:konferenz-paper:online:lernumgebung",
        "#4:konferenz-paper:mooc",
        "#5:konferenz-paper:e-learning",
        "#6:konferenz-paper:bildung:technologie",
        "#7:konferenz-paper:digital:medien",
        "#8:konferenz-paper:blended:learning",
        "#9:konferenz-paper:digital:lernen",
        "#a:konferenz-paper:online:lernen",
        "#b:konferenz-paper:online:learning",
        "#0:konferenz-paper:digital:learning"
    ]

    category_tags = {"promotion:argumentation", "promotion:kerngedanke", "promotion:weiterführung", "promotion:schlussfolgerung"}
    source_data = defaultdict(lambda: {'Identifiziert': 0, 'Analysiert': 0})

    if not bib_database or not bib_database.entries:
        print("Fehler: Die Datenbank enthält keine Einträge.")
        return

    for entry in bib_database.entries:
        keywords = entry.get('keywords', '')
        if not keywords:
            continue

        entry_keywords = set(map(str.lower, map(str.strip, keywords.replace('\\#', '#').split(','))))

        for tag in search_folder_tags:
            if tag.lower() in entry_keywords:
                source_data[tag]['Identifiziert'] += 1
                if entry_keywords & category_tags:
                    source_data[tag]['Analysiert'] += 1

    table_data = []
    analysiert_values = []
    nicht_analysiert_values = []
    analysiert_colors = []
    tags = []

    for tag, counts in sorted(source_data.items(), key=lambda item: item[1]['Identifiziert'], reverse=True):
        stichprobe = calculate_sample_size(counts['Identifiziert'])
        noch_zu_analysieren = counts['Identifiziert'] - counts['Analysiert']
        noch_benoetigt_fuer_stichprobe = max(0, stichprobe - counts['Analysiert'])

        table_data.append([
            tag,
            counts['Identifiziert'],
            counts['Analysiert'],
            noch_zu_analysieren,
            stichprobe,
            noch_benoetigt_fuer_stichprobe
        ])

        analysiert_values.append(counts['Analysiert'])
        nicht_analysiert_values.append(noch_zu_analysieren)
        tags.append(tag)

        analysiert_colors.append(colors['positiveHighlight'] if counts['Analysiert'] >= stichprobe else colors['negativeHighlight'])

    print(tabulate(
        table_data,
        headers=['Suchordner', 'Identifiziert', 'Analysiert', 'nicht-Analysiert', 'Stichprobe', 'Noch benötigt für Stichprobe'],
        tablefmt='grid'
    ))

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=tags,
        y=analysiert_values,
        name='Analysiert',
        marker=dict(color=analysiert_colors)
    ))

    fig.add_trace(go.Bar(
        x=tags,
        y=nicht_analysiert_values,
        name='Nicht-Analysiert',
        marker=dict(color=colors['primaryLine'])
    ))

    fig.update_layout(
        barmode='stack',
        title=f'Analyse- und Stichprobenstatus je Suchordner (n={sum(counts["Identifiziert"] for counts in source_data.values())}, Stand: {current_date})',
        xaxis_title='Suchbegriffsordner',
        yaxis_title='Anzahl der Quellen',
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font=dict(color=colors['white']),
        xaxis=dict(
            categoryorder='array',
            categoryarray=search_folder_tags
        )
    )

    fig.show()
    export_visualize_sources_status(fig)

#############

# Funktion zur Erstellung einer Wortwolke aus Überschriften
def create_wordcloud_from_titles(bib_database, stop_words):
    titles = [entry.get('title', '') for entry in bib_database.entries]

    # Wörter zählen
    word_counts = defaultdict(int)
    for title in titles:
        for word in title.split():
            word = word.lower().strip(",.!?\"'()[]{}:;")
            if word and word not in stop_words:
                word_counts[word] += 1

    # Wortwolke erstellen
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color=colors['background'],
        color_func=lambda *args, **kwargs: random.choice(word_colors)
    ).generate_from_frequencies(word_counts)

    # Wortwolke anzeigen
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Häufigkeitsanalyse von Titelwörtern (Stand: {current_date})', color=colors['white'])
    plt.show()

# Exportfunktion für visualize_search_term_distribution
def export_visualize_search_term_distribution(fig):
    if export_fig_visualize_search_term_distribution:
        safe_filename = slugify("visualize_search_term_distribution")
        export_path = f"{safe_filename}.html"
        fig.write_html(export_path, full_html=True, include_plotlyjs="cdn")
        remote_path = "jochen-hanisch@sternenflottenakademie.local:/mnt/deep-space-nine/public/plot/promotion/"
        try:
            subprocess.run(["scp", export_path, remote_path], check=True, capture_output=True, text=True)
            print(f"✅ Datei '{export_path}' erfolgreich übertragen.")
        except subprocess.CalledProcessError as e:
            print("❌ Fehler beim Übertragen:")
            print(e.stderr)

# Kuchengrafik zur Verteilung der Einträge auf primäre, sekundäre und tertiäre Begriffsordner
def visualize_search_term_distribution(bib_database):
    """
    Erstellt eine Kuchengrafik zur Verteilung der Einträge auf primäre, sekundäre und tertiäre Begriffsordner.
    """
    hierarchy_counts = {
        'Primär': 0,
        'Sekundär': 0,
        'Tertiär': 0
    }

    primary_folders = {
        'S:01 Learning Management System',
        'S:02 Online-Lernplattform',
        'S:03 Online-Lernumgebung',
        'S:05 eLearning',
        'S:04 MOOC',
        'S:06 Bildungstechnologie',
        'S:07 Digitale Medien',
        'S:08 Blended Learning',
        'S:09 Digitales Lernen',
        'S:12 Digital Learning',
        'S:10 Online Lernen',
        'S:11 Online Learning'
    }

    secondary_folders = {
        'S:13 Berichte',
        'S:14 Agiles Lernen',
        'S:15 Learning Analytics'
    }

    tertiary_folders = {
        'S:16 Dissertationen',
        'S:17 ePortfolio'
    }

    conn = sqlite3.connect('/Users/jochen_hanisch-johannsen/Zotero/zotero.sqlite')
    cursor = conn.cursor()

    query = """
    SELECT collections.collectionName, COUNT(DISTINCT items.itemID)
    FROM items
    JOIN collectionItems ON items.itemID = collectionItems.itemID
    JOIN collections ON collectionItems.collectionID = collections.collectionID
    WHERE collections.collectionName IN (
        'S:01 Learning Management System',
        'S:02 Online-Lernplattform',
        'S:03 Online-Lernumgebung',
        'S:05 eLearning',
        'S:04 MOOC',
        'S:06 Bildungstechnologie',
        'S:07 Digitale Medien',
        'S:08 Blended Learning',
        'S:09 Digitales Lernen',
        'S:12 Digital Learning',
        'S:10 Online Lernen',
        'S:11 Online Learning',
        'S:13 Berichte',
        'S:14 Agiles Lernen',
        'S:15 Learning Analytics',
        'S:16 Dissertationen',
        'S:17 ePortfolio'
    )
    GROUP BY collections.collectionName
    """

    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()

    for collection, count in rows:
        if collection in primary_folders:
            hierarchy_counts['Primär'] += count
        elif collection in secondary_folders:
            hierarchy_counts['Sekundär'] += count
        elif collection in tertiary_folders:
            hierarchy_counts['Tertiär'] += count

    labels = list(hierarchy_counts.keys())
    values = list(hierarchy_counts.values())
    colors_pie = [colors['primaryLine'], colors['secondaryLine'], colors['brightArea']]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors_pie),
        textinfo='label+percent',
        hoverinfo='label+value'
    )])

    fig.update_layout(
        title='Verteilung der Suchbegriffsordner (Primär, Sekundär, Tertiär)',
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font=dict(color=colors['white'])
    )

    fig.show()
    export_visualize_search_term_distribution(fig)

# Aufrufen der Visualisierungsfunktionen
visualize_network(bib_database)
visualize_tags(bib_database)
visualize_index(bib_database)
visualize_research_questions(bib_database)
visualize_categories(bib_database)
visualize_time_series(bib_database)
visualize_top_authors(bib_database)
visualize_top_publications(bib_database)
data = prepare_path_data(bib_database)
create_path_diagram(data)
create_sankey_diagram(bib_database)
visualize_sources_status(bib_database)
create_wordcloud_from_titles(bib_database, stop_words)
visualize_search_term_distribution(bib_database)
