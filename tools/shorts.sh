#!/usr/bin/env bash
# Сборка единого question-shorts.pdf: склеивает short.md всех вопросов в один
# markdown с заголовком «# Вопрос NN. ...» на каждый вопрос и \newpage между
# ними. Использование: bash tools/shorts.sh > <combined.md>
set -euo pipefail
cd "$(dirname "$0")/.."

# Орнамент после содержания, перед первым билетом — визуальный разрыв.
# Стоит в теле первым элементом: pandoc ставит оглавление перед телом, так что
# орнамент оказывается внизу той же страницы, а \newpage ниже уносит билеты.
printf '\\begin{center}\\includegraphics[width=0.45\\textwidth]{assets/ornament.pdf}\\end{center}\n\n'

for md in questions/*/short.md; do
  [ -e "$md" ] || continue
  title=$(sed -n 's/^title: "\(.*\)"/\1/p' "$md" | head -1)
  # Разрыв перед КАЖДЫМ билетом, включая первый: pandoc ставит оглавление
  # перед телом документа, и без разрыва перед первым билетом содержание
  # слипается с ним.
  printf '\n\\newpage\n\n'
  printf '# %s\n\n' "$title"
  # Тело без YAML-шапки: short.md начинается с блока --- ... ---
  sed -n '/^---$/,/^---$/d; p' "$md" | sed '1{/^$/d}'
  printf '\n'
done
