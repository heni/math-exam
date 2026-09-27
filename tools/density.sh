#!/usr/bin/env bash
# Плотность доказательств: bash tools/density.sh <файл> ...
#
# Считает вхождения слова «доказательство» на 100 000 знаков. Дешёвый признак
# пригодности книги там, где нужны полные выкладки.
#
# ВАЖНО: регулярное выражение допускает пробелы между буквами. Русская
# типографская традиция набирает «Д о к а з а т е л ь с т в о» разрядкой, и
# наивный поиск подстроки её не видит: у одной книги это дало 17 вместо 220,
# то есть плотность 1,6 вместо 21,3 — ошибка в тринадцать раз, причём в
# сторону «книга не годится».
#
# Метрика считает СЛОВО, а не выкладку: она сужает круг чтения, но не заменяет его.
set -uo pipefail
python3 - "$@" <<'PY'
import subprocess,re,sys
SP=lambda w: r'\s{0,2}'.join(w)   # допускаем разрядку
PROOF=re.compile(SP('доказательство'),re.I)
THM  =re.compile(SP('теорема'),re.I)
rows=[]
import os
for f in sys.argv[1:]:
    # если рядом есть распознанный текст (.ocr/<stem>.txt) — мерить по нему:
    # иначе скан всегда покажет «СКАН», даже когда OCR давно сделан.
    d,b=os.path.split(f); stem=os.path.splitext(b)[0]
    ocr=os.path.join(d,'.ocr',stem+'.txt')
    if os.path.exists(ocr) and os.path.getsize(ocr)>5000:
        t=open(ocr,encoding='utf8',errors='replace').read().replace('ё','е')
        t=re.sub(r'\s+',' ',re.sub(r'-\s*\n\s*','',t))
        pr=len(PROOF.findall(t)); th=len(THM.findall(t))
        rows.append((pr/(len(t)/100000),f+' [OCR]',pr,th,len(t))); continue
    cmd=['timeout','400','djvutxt',f] if f.endswith('.djvu') else ['timeout','300','pdftotext','-q',f,'-']
    t=subprocess.run(cmd,capture_output=True).stdout.decode('utf8','replace').replace('ё','е')
    t=re.sub(r'\s+',' ',re.sub(r'-\s*\n\s*','',t))
    if len(t)<5000: rows.append((-1,f,0,0,0)); continue
    pr=len(PROOF.findall(t)); th=len(THM.findall(t))
    rows.append((pr/(len(t)/100000),f,pr,th,len(t)))
for d,f,pr,th,n in sorted(rows,reverse=True):
    name=f.split('/')[-1][:56]
    if d<0: print(f'{"СКАН":>7}  {name}')
    else:   print(f'{d:7.1f}  {name:58} доказ:{pr:>5} теорем:{th:>5}')
PY
