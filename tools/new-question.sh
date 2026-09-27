#!/usr/bin/env bash
# Развернуть шаблоны в каталоге вопроса: bash tools/new-question.sh 11
# Существующие файлы НЕ перезаписываются.
set -euo pipefail
cd "$(dirname "$0")/.."

num=$(printf '%02d' "${1#0}")
row=$(awk -F'\t' -v n="$num" '$1==n {print; exit}' tools/questions.tsv)
[ -n "$row" ] || { echo "вопрос $num не найден в tools/questions.tsv"; exit 1; }

slug=$(cut -f2 <<<"$row")
title=$(cut -f3 <<<"$row")
dir="questions/$slug"
mkdir -p "$dir/figures"

for f in theory slides; do
  if [ -e "$dir/$f.md" ]; then
    echo "  пропуск: $dir/$f.md уже есть"
  else
    sed -e "s/ЗАГОЛОВОК ВОПРОСА/$title/" -e "s/Вопрос NN/Вопрос $num/" \
        -e "s/^subtitle: \"Вопрос NN\"/subtitle: \"Вопрос $num\"/" \
        "templates/$f.md" > "$dir/$f.md"
    echo "  создан: $dir/$f.md"
  fi
done

if [ -e "$dir/examples.py" ]; then
  echo "  пропуск: $dir/examples.py уже есть"
else
  sed -e "s/ЗАГОЛОВОК/$title/" -e "s/Вопрос NN/Вопрос $num/" \
      -e "s|questions\/NN-slug|questions/$slug|" \
      templates/examples.py > "$dir/examples.py"
  echo "  создан: $dir/examples.py"
fi

echo "Готово. Сборка: make q${num#0}"
