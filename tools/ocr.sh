#!/usr/bin/env bash
# Распознать сканы: bash tools/ocr.sh <файл.pdf|файл.djvu> ...
#
# Кладёт рядом с источником каталог .ocr/ и пишет туда PDF с текстовым слоем и
# файл .txt. Исходники НЕ изменяются: OCR — производный артефакт, а книгу мы
# цитируем по странице, поэтому нумерация страниц должна остаться исходной.
#
# Формулы не распознаются и не должны: цель — сделать текст искомым и
# цитируемым, выкладки читаются с самого скана.
#
# LANG=eng — распознавать как англоязычный текст (по умолчанию rus). Обязательно для
# англоязычных сканов: tesseract с -l rus подбирает кириллические похожие буквы и
# выдаёт правдоподобный мусор («МАМАСЕМЕМТ ЗСЕМСЕ» вместо «MANAGEMENT SCIENCE»),
# не сообщая об ошибке. LANG=rus+eng — для смешанных.
#
# FORCE=1 — распознать заново поверх существующего текстового слоя. Нужно, когда
# слой есть, но нечитаем: у сканов с подложенным «текстом» без ToUnicode вся
# кириллица выходит знаками «?». Проверять долю кириллицы (tools/readable.sh),
# а не длину: длина в таком файле нормальная.
set -uo pipefail
command -v ocrmypdf >/dev/null || { echo "нет ocrmypdf"; exit 1; }
command -v tesseract >/dev/null || { echo "нет tesseract"; exit 1; }
LANG="${LANG_OCR:-rus}"
for l in ${LANG//+/ }; do
  tesseract --list-langs 2>/dev/null | grep -qx "$l" || { echo "нет языка $l для tesseract"; exit 1; }
done

JOBS="${JOBS:-$(( $(nproc) > 4 ? 4 : 1 ))}"
fail=0

for src in "$@"; do
  [ -e "$src" ] || { echo "нет файла: $src"; fail=1; continue; }
  dir=$(dirname "$src"); base=$(basename "$src"); stem="${base%.*}"
  out="$dir/.ocr"; mkdir -p "$out"
  txt="$out/$stem.txt"; pdf="$out/$stem.pdf"

  if [ -s "$txt" ]; then echo "== пропуск (уже есть): $stem"; continue; fi

  # djvu -> промежуточный PDF: ocrmypdf djvu не читает.
  work="$src"
  if [[ "$src" == *.djvu ]]; then
    work="$out/$stem.src.pdf"
    if [ ! -s "$work" ]; then
      echo "== djvu -> pdf: $stem"
      ddjvu -format=pdf -quality=85 "$src" "$work" 2>/dev/null || { echo "  ddjvu упал"; fail=1; continue; }
    fi
  fi

  echo "== OCR: $stem"
  mode="--skip-text"; [ "${FORCE:-0}" = "1" ] && mode="--force-ocr"
  if ocrmypdf -l "$LANG" $mode --optimize 1 --jobs "$JOBS" \
       --sidecar "$txt" "$work" "$pdf" 2>"$out/$stem.log"; then
    printf '   готово: %s знаков\n' "$(wc -c <"$txt")"
  else
    echo "   ОШИБКА, см. $out/$stem.log"; tail -3 "$out/$stem.log" | sed 's/^/     /'; fail=1
  fi
  [ "$work" != "$src" ] && rm -f "$work"
done
exit "$fail"
