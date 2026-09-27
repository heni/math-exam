#!/usr/bin/env bash
# Таблица готовности по всем 20 вопросам: bash tools/status.sh
set -euo pipefail
cd "$(dirname "$0")/.."

printf '%-3s %-34s %-7s %-7s %-7s %-6s %s\n' '№' 'каталог' 'theory' 'slides' 'nb' 'фиг.' 'вопрос'
printf '%s\n' '---------------------------------------------------------------------------------------'
while IFS=$'\t' read -r num slug title block src; do
  d="questions/$slug"
  mark() { if [ -s "$1" ]; then printf 'pdf'; elif [ -s "$2" ]; then printf 'md '; else printf ' - '; fi; }
  th=$(mark "$d/theory.pdf" "$d/theory.md")
  sl=$(mark "$d/slides.pdf" "$d/slides.md")
  if [ -s "$d/examples.ipynb" ]; then nb='ipynb'; elif [ -s "$d/examples.py" ]; then nb='py   '; else nb=' -   '; fi
  nfig=$(find "$d/figures" -type f 2>/dev/null | wc -l)
  printf '%-3s %-34s %-7s %-7s %-7s %-6s %s\n' "$num" "$slug" "$th" "$sl" "$nb" "$nfig" "${title:0:40}"
done < tools/questions.tsv
