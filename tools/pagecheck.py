#!/usr/bin/env python3
"""Проверка: совпадает ли колонтитул распознанной страницы с её номером в файле.

Зачем. Ссылка вида «[22] теорема 1, с. 193» берётся из книги, а номер страницы —
из распознанного текста, и распознавание ошибается систематически: у [22]
ведущая «1» читается как «4», так что страницы 191–194 выходят как 491–494.
Один раз это уже стоило ссылки, уводившей в другую главу книги.

Что печатает: для каждой распознанной книги — сколько проб колонтитула совпало с
номером страницы файла, сколько разошлось ровно ведущей цифрой (признак подмены)
и каково смещение печатной нумерации к нумерации файла, если оно постоянно.

Чего НЕ делает: не проверяет номера теорем и не заменяет чтение изображения.
Вывод «подмена» означает «этой книге верить нельзя, смотреть страницу глазами»:

    ddjvu -format=pnm -page=193 deps/22_....djvu p.pnm

Запуск: python3 tools/pagecheck.py [номер книги ...]
"""
import os, re, sys, glob, collections

OCR_DIR = os.path.join(os.path.dirname(__file__), '..', 'deps', '.ocr')
PROBE_STEP = 17          # шаг проб: простое число, чтобы не попадать в такт вёрстки
HEAD_CHARS = 60          # колонтитул ищем в начале страницы


def probes(pages):
    """-> (совпало, подменена ведущая цифра, постоянное смещение или None)."""
    ok = bad = 0
    offsets = collections.Counter()
    for i in range(60, min(len(pages), 400), PROBE_STEP):
        head = pages[i - 1].strip().split('\n')[0][:HEAD_CHARS] if pages[i - 1].strip() else ''
        nums = re.findall(r'\b\d{2,4}\b', head)
        if not nums:
            continue
        if str(i) in nums:
            ok += 1
            offsets[0] += 1
            continue
        same_tail = [n for n in nums
                     if len(n) == len(str(i)) and n[1:] == str(i)[1:] and n[0] != str(i)[0]]
        if same_tail:
            bad += 1
            continue
        near = [int(n) - i for n in nums if abs(int(n) - i) <= 4]
        if near:
            ok += 1
            offsets[near[0]] += 1
    steady = None
    if offsets:
        top, cnt = offsets.most_common(1)[0]
        if cnt >= 3 and cnt >= 0.8 * sum(offsets.values()):
            steady = top
    return ok, bad, steady


def main(argv):
    want = set(argv[1:])
    files = sorted(glob.glob(os.path.join(OCR_DIR, '*.txt')))
    if not files:
        print('распознанных текстов нет: см. tools/ocr.sh')
        return 1
    flagged = []
    for f in files:
        num = os.path.basename(f).split('_')[0]
        if want and num not in want:
            continue
        pages = open(f, encoding='utf8', errors='replace').read().split('\f')
        if len(pages) < 50:
            print(f'[{num}] страниц меньше пятидесяти — проба не строится')
            continue
        ok, bad, steady = probes(pages)
        off = 'смещение неустойчиво' if steady is None else f'смещение {steady:+d}'
        if bad:
            flagged.append(num)
            print(f'[{num}] совпало {ok}, подменена ведущая цифра {bad} — '
                  f'НОМЕРАМ ИЗ ТЕКСТА НЕ ВЕРИТЬ, смотреть изображение')
        else:
            print(f'[{num}] совпало {ok}, подмен нет, {off}')
    if flagged:
        print('\nкниги, у которых номер страницы нельзя брать из распознанного текста: '
              + ', '.join(flagged))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
