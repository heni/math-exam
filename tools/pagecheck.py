#!/usr/bin/env python3
"""Совпадает ли колонтитул распознанной страницы с её номером в файле.

Зачем. Ссылка вида «теорема 1, с. 193» берётся из книги, а номер страницы — из
распознанного текста, и распознавание ошибается систематически: у одной из книг
набора ведущая «1» читается как «4», так что страницы 191-194 выходят как
491-494. Один раз это уже стоило ссылки, уводившей в другую главу книги.

Как. Смещение печатной нумерации заранее неизвестно и у разных книг разное,
поэтому инструмент его не угадывает, а калибрует: собирает разности «число в
колонтитуле минус номер страницы в файле» по всем пробам и берёт ту, которая
повторилась чаще прочих. Фиксированное окно допустимого смещения не годилось: у
одной книги набора смещение +5, и окно отбрасывало её вместе с единственной
величиной, которую инструмент существует чтобы напечатать.

Что печатает: по каждому файлу — сколько проб сделано, на сколько из них
колонтитул сошёлся с номером при выбранном смещении, каково само смещение и какая
подстановка цифры объясняет расхождения. Подстановка тоже калибруется, а не
угадывается: разовое расхождение одной цифрой даёт любое посторонне число в
колонтитуле, поэтому значение имеет повторяемость одной и той же пары. Три
повтора и больше — «НЕ ВЕРИТЬ», два — «ПОДОЗРЕНИЕ»: у книги с плохим
распознаванием проб с читаемым колонтитулом мало, и требовать трёх значило бы
пропускать её молча. Все величины — счётчики проб, а не свойства книги: книга, где
колонтитул не распознан, даёт «НЕ ПРОВЕРЕНО», и это означает именно «не
проверено», а не «нарушений нет».

Чего НЕ делает: не проверяет номера теорем и не заменяет чтение изображения.
Вывод «подмена» означает «этой книге верить нельзя, смотреть страницу глазами»:

    ddjvu -format=pnm -page=193 <книга.djvu> p.pnm
    pdftoppm -f 193 -l 193 <книга.pdf> p

Запуск: python3 tools/pagecheck.py [распознанный.txt|каталог ...]
Без аргументов берёт весь распознанный текст набора из deps/.ocr. Сами книги в
репозиторий не входят, но инструмент запускает тот, у кого они есть, и путь для
него — рабочая величина.

Код возврата: 0 — подмен не найдено, 1 — есть книги, номерам которых верить
нельзя, 2 — ошибка запуска или чтения.
"""
import glob, os, re, sys, collections

OCR_DIR = 'deps/.ocr'    # распознанный текст набора; сам набор вне git
HEAD_CHARS = 60          # колонтитул ищем в начале страницы
SKIP_FRONT = 0.2         # доля начала книги без колонтитулов (титул, оглавление)
WANT_PROBES = 20         # сколько проб пытаемся сделать, растягивая по книге
MIN_PAGES = 50           # короче — проба не строится
MIN_HITS = 3             # меньше совпадений — проба несостоятельна
SUSPECT_HITS = 2         # столько повторов подстановки — повод посмотреть глазами
MAX_OFFSET = 60          # разности больше по модулю — не смещение, а номер главы


def head_numbers(page):
    """Числа из первой непустой строки страницы — там стоит колонтитул."""
    line = next((s for s in page.split('\n') if s.strip()), '')
    return [int(n) for n in re.findall(r'\b\d{2,4}\b', line[:HEAD_CHARS])]


def calibrate(samples):
    """Смещение печатной нумерации по модальной разности, или None.

    Номера параграфов в колонтитуле («§71 ЛИНЕЙНЫЕ ОПЕРАТОРЫ») дают разности
    случайные и потому не повторяющиеся — модальный выбор отбрасывает их сам.
    """
    diffs = collections.Counter(n - i for i, nums in samples for n in nums
                                if abs(n - i) <= MAX_OFFSET)
    if not diffs:
        return None
    offset, cnt = diffs.most_common(1)[0]
    return offset if cnt >= MIN_HITS else None


def misread(nums, target):
    """Какой подстановкой цифры объясняется число в колонтитуле, или None.

    Возвращает пару ('1', '4'), если target превращается в одно из чисел заменой
    всех вхождений одной цифры на другую, либо ('+', цифра) для лишней
    вставленной. Одна такая находка ничего не значит: в колонтитуле законно
    стоят посторонние числа, и любое из них может случайно отличаться одной
    цифрой. Значение имеет ПОВТОРЯЕМОСТЬ пары по книге — её считает probes.
    """
    t = str(target)
    for n in nums:
        s = str(n)
        if len(s) == len(t) and s != t:
            pairs = {(a, b) for a, b in zip(t, s) if a != b}
            if len(pairs) == 1:
                return pairs.pop()
        if len(s) == len(t) + 1:
            for k in range(len(s)):
                if s[:k] + s[k + 1:] == t:
                    return ('+', s[k])
    return None


def probes(pages):
    """-> (проб, проб с числами, сошлось, подмен, смещение или None)."""
    first, last = int(len(pages) * SKIP_FRONT), len(pages)
    step = max(1, (last - first) // WANT_PROBES)
    samples = [(i, head_numbers(pages[i - 1]))
               for i in range(first + 1, last + 1, step)]
    tried = len(samples)
    with_nums = sum(1 for _, nums in samples if nums)
    offset = calibrate(samples)
    if offset is None:
        return tried, with_nums, 0, 0, None
    hit, misreads = 0, []
    for i, nums in samples:
        if i + offset in nums:
            hit += 1
        else:
            pair = misread(nums, i + offset)
            if pair:
                misreads.append(pair)
    # Систематической считается подстановка, повторившаяся не меньше MIN_HITS
    # раз: разовое расхождение одной цифрой даёт любое посторонне число в
    # колонтитуле, и без этого порога ложно флагуется половина набора.
    top = collections.Counter(misreads).most_common(1)
    defect = top[0] if top and top[0][1] >= SUSPECT_HITS else None
    return tried, with_nums, hit, defect, offset


def report(name, pages):
    """-> 'flagged' | 'unchecked' | 'ok'; печатает строку о книге."""
    if len(pages) < MIN_PAGES:
        print(f'{name}: страниц меньше {MIN_PAGES} — проба не строится')
        return 'unchecked'
    tried, with_nums, hit, defect, offset = probes(pages)
    if offset is None or hit < MIN_HITS:
        print(f'{name}: из {tried} проб числа в колонтитуле нашлись на {with_nums}, '
              f'с номером страницы не связалось ничего — НЕ ПРОВЕРЕНО')
        return 'unchecked'
    if defect:
        (was, now), cnt = defect
        how = (f'лишняя цифра «{now}»' if was == '+'
               else f'цифра «{was}» читается как «{now}»')
        verdict = ('НОМЕРАМ ИЗ ТЕКСТА НЕ ВЕРИТЬ' if cnt >= MIN_HITS
                   else 'ПОДОЗРЕНИЕ, проверить глазами')
        print(f'{name}: из {tried} проб сошлось {hit} при смещении {offset:+d}, '
              f'но на {cnt} {how} — {verdict}, смотреть изображение')
        return 'flagged' if cnt >= MIN_HITS else 'suspect'
    print(f'{name}: из {tried} проб сошлось {hit} '
          f'при смещении {offset:+d}, подмен нет')
    return 'ok'


def expand(paths):
    """Каталог в аргументе раскрывается в свои *.txt — так его и передают."""
    out = []
    for p in paths:
        out.extend(sorted(glob.glob(os.path.join(p, '*.txt')))
                   if os.path.isdir(p) else [p])
    return out


def main(argv):
    paths = expand(argv[1:]) or sorted(glob.glob(os.path.join(OCR_DIR, '*.txt')))
    if not paths:
        print(__doc__)
        print(f'в {OCR_DIR} нет распознанного текста — нечего проверять')
        return 2
    flagged, suspect, unchecked, failed = [], [], [], []
    for path in paths:
        name = os.path.basename(path)
        try:
            with open(path, encoding='utf8', errors='replace') as fh:
                pages = fh.read().split('\f')
        except OSError as exc:
            failed.append(name)
            print(f'{name}: не прочитан — {exc.strerror}')
            continue
        verdict = report(name, pages)
        {'flagged': flagged, 'suspect': suspect,
         'unchecked': unchecked}.get(verdict, []).append(name)
    if flagged:
        print('\nномер страницы нельзя брать из распознанного текста: '
              + ', '.join(flagged))
    if suspect:
        print('\nподстановка повторилась дважды, посмотреть страницу глазами: '
              + ', '.join(suspect))
    if unchecked:
        print('\nне проверено, колонтитул не распознан: ' + ', '.join(unchecked))
    if failed:
        return 2
    return 1 if flagged or suspect else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
