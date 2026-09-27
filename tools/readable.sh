#!/usr/bin/env bash
# Проверить, читаем ли текстовый слой: bash tools/readable.sh <файл> ...
#
# Длина слоя ничего не значит: у PDF без ToUnicode для кириллического шрифта
# слой нормального размера, но все буквы выходят знаками «?». Признак — доля
# кириллицы, а не число символов.
set -uo pipefail
python3 - "$@" <<'PY'
import subprocess,re,sys
for f in sys.argv[1:]:
    cmd=['timeout','400','djvutxt',f] if f.endswith('.djvu') else ['timeout','300','pdftotext','-q',f,'-']
    try: t=subprocess.run(cmd,capture_output=True).stdout.decode('utf8','replace')
    except Exception as e: print(f'{f}: не прочитан ({e})'); continue
    n=len(t); cyr=len(re.findall(r'[А-Яа-я]',t)); lat=len(re.findall(r'[A-Za-z]',t)); q=t.count('?')
    letters=cyr+lat
    if n<5000: v='СКАН (слоя нет) -> tools/ocr.sh'
    elif letters and q>0.15*letters: v='БИТЫЙ СЛОЙ (кириллица -> «?») -> FORCE=1 tools/ocr.sh'
    elif letters<n*0.25: v='слой сомнителен — посмотреть глазами'
    else: v='читаем'
    print(f'{v:46} {f.split("/")[-1][:48]:50} зн:{n:>9} кир:{cyr:>8} «?»:{q:>7}')
PY
