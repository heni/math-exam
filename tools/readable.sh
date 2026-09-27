#!/usr/bin/env bash
# Читаем ли текстовый слой: bash tools/readable.sh <файл> ...
# Логика загрузки — tools/booktext.py (там же перечислены четыре ловушки).
set -uo pipefail
cd "$(dirname "$0")/.."
python3 - "$@" <<'PY'
import sys; sys.path.insert(0,'tools')
import booktext, os
for f in sys.argv[1:]:
    t,i=booktext.load(f)
    tag=' [OCR]' if i['from_ocr'] else (f" [перекодировано {i['recoded']}]" if i['recoded'] else '')
    print(f"{i['verdict']:46} {os.path.basename(f)[:46]:48} зн:{i['len']:>9} кир:{i['cyr']:>8} «?»:{i['q']:>6}{tag}")
PY
