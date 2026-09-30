#!/usr/bin/env python3
"""Совпадает ли колонтитул распознанной страницы с её номером в файле.

Зачем. Ссылка вида «теорема 1, с. 193» берётся из книги, а номер страницы — из
распознанного текста, и распознавание ошибается систематически: у одной из книг
набора ведущая «1» читается как «4», так что страницы 191-194 выходят как
491-494. Один раз это уже стоило ссылки, уводившей в другую главу книги.

Что печатает: по каждому переданному файлу распознанного текста - сколько проб
колонтитула совпало с номером страницы, сколько разошлось ровно ведущей цифрой
(признак подмены) и каково смещение печатной нумерации, если оно постоянно.

Чего НЕ делает: не проверяет номера теорем и не заменяет чтение изображения.
Вывод «подмена» означает «этой книге верить нельзя, смотреть страницу глазами»:

    ddjvu -format=pnm -page=193 <книга.djvu> p.pnm

Запуск: python3 tools/pagecheck.py <распознанный.txt> ...
Каталог с книгами передаётся аргументом: в репозиторий книги не входят, и путь
к ним знает только тот, у кого они есть.
"""
import os, re, sys, collections

PROBE_STEP = 17          # шаг проб: простое, чтобы не попадать в такт вёрстки
HEAD_CHARS = 60          # колонтитул ищем в начале страницы
FIRST, LAST = 60, 400    # первые страницы часто без колонтитула


def probes(pages):
    """-> (совпало, подменена ведущая цифра, постоянное смещение или None)."""
    ok = bad = 0
    offsets = collections.Counter()
    for i in range(FIRST, min(len(pages), LAST), PROBE_STEP):
        first_line = pages[i - 1].strip().split('\n')[0] if pages[i - 1].strip() else ''
        nums = re.findall(r'\b\d{2,4}\b', first_line[:HEAD_CHARS])
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
    if len(argv) < 2:
        print(__doc__)
        return 1
    flagged = []
    for path in argv[1:]:
        name = os.path.basename(path)
        if not os.path.exists(path):
            print(f'{name}: файла нет')
            continue
        pages = open(path, encoding='utf8', errors='replace').read().split('\f')
        if len(pages) < 50:
            print(f'{name}: страниц меньше пятидесяти — проба не строится')
            continue
        ok, bad, steady = probes(pages)
        if bad:
            flagged.append(name)
            print(f'{name}: совпало {ok}, подменена ведущая цифра {bad} — '
                  f'НОМЕРАМ ИЗ ТЕКСТА НЕ ВЕРИТЬ, смотреть изображение')
        else:
            off = 'смещение неустойчиво' if steady is None else f'смещение {steady:+d}'
            print(f'{name}: совпало {ok}, подмен нет, {off}')
    if flagged:
        print('\nномер страницы нельзя брать из распознанного текста: '
              + ', '.join(flagged))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
