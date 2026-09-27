#!/usr/bin/env bash
# Плотность доказательств на 100k знаков: bash tools/density.sh <файл> ...
# Дешёвый признак пригодности там, где нужны полные выкладки. Считает СЛОВО, а
# не выкладку: сужает круг чтения, но не заменяет его.
# Логика загрузки и шаблон с допуском разрядки — tools/booktext.py.
set -uo pipefail
cd "$(dirname "$0")/.."
python3 - "$@" <<'PY'
import sys, re, os; sys.path.insert(0,'tools')
import booktext
PROOF=re.compile(booktext.PROOF_PAT,re.I)
THM  =re.compile(booktext.THEOREM_PAT,re.I)
rows=[]
for f in sys.argv[1:]:
    t,i=booktext.load(f)
    if i['verdict'].startswith(('СКАН','БИТЫЙ')): rows.append((-1,f,0,0,i)); continue
    pr=len(PROOF.findall(t)); th=len(THM.findall(t))
    rows.append((pr/(len(t)/100000),f,pr,th,i))
for d,f,pr,th,i in sorted(rows,reverse=True):
    name=os.path.basename(f)[:54]
    tag=' [OCR]' if i['from_ocr'] else ''
    if d<0: print(f'{i["verdict"][:14]:>14}  {name}')
    else:   print(f'{d:7.1f}  {name:56} доказ:{pr:>5} теорем:{th:>5}{tag}')
PY
