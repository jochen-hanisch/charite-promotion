#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/Users/jochenhanisch-johannsen/Library/Mobile Documents/iCloud~md~obsidian/Documents/Jochen-Hanisch/Allgemein beruflich/Research/Promotion"
cd "$PROJECT_ROOT"

pandoc \
  dissertation.md \
  chapters/01_einleitung_und_theoretischer_rahmen.md \
  chapters/02_theorieteil.md \
  chapters/03_forschungsgegenstand.md \
  chapters/04_methodologie.md \
  chapters/05_ergebnisse.md \
  chapters/06_diskussion.md \
  chapters/07_conclusio.md \
  -o dissertation.pdf \
  --pdf-engine=xelatex \
  --citeproc

echo "Fertig: $PROJECT_ROOT/dissertation.pdf"
