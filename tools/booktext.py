"""Загрузка и нормализация текста книги. Единственное место, где это делается.

Три инструмента (readable, density, coverage) раньше держали свою копию этой
логики, и три ошибки из четырёх случились именно в ней. Ошибки были такие:

1. `tr -d '\n'` склеивал слова, и многословные термины, разорванные переносом
   строки, становились невидимы. Лечится заменой пробельного на ОДИН пробел.
2. Читаемость проверялась по длине слоя. У книги с кириллическим шрифтом без
   ToUnicode слой нормального размера, но все буквы — знаки «?». Лечится
   проверкой доли кириллицы.
3. Поиск подстроки не видел разрядку: «Д о к а з а т е л ь с т в о» — обычный
   русский набор. Лечится шаблоном с допуском пробелов между буквами (`spaced`).
7. Англоязычные книги считались русским шаблоном и давали ноль доказательств
   (четыре тома Argyros). Лечится двуязычным шаблоном (`either`).
5. Пустой результат извлечения принимался за «слоя нет», хотя он получается и
   когда файла нет (имя в нормализации NFD против NFC) или когда извлекатель упал.
   Дало ложный диагноз «СКАН» и напрасный OCR на 285 страниц. Лечится явной
   проверкой существования файла и кода возврата.
6. Слой в cp1251, прочитанный как latin1: «ÐÎÑÑÈÉÑÊÀß» вместо «РОССИЙСКАЯ».
   Кириллицы ноль, знаков «?» ноль — прежняя проверка считала такой файл
   читаемым. Лечится попыткой перекодировки (`_fix_mojibake`); OCR не нужен.

Все они — один класс: инструмент молча возвращал осмысленно выглядящий
результат.
"""
import os, re, subprocess

def spaced(word):
    """Шаблон слова, допускающий русскую разрядку: 'Д о к а з а т е л ь с т в о'."""
    return r'\s{0,2}'.join(word)

def either(ru, en):
    """Шаблон термина на двух языках: в коллекции есть англоязычные книги, и
    русский шаблон давал у них ноль — пятый случай одной и той же ошибки
    (инструмент мерил не то, что нужно, и молчал об этом)."""
    return spaced(ru) + r'|\b' + en + r'\w{0,3}\b'

PROOF_PAT = either('доказательство', 'proof')
THEOREM_PAT = either('теорема', 'theorem')

def _raw(path):
    """Сырой текст: распознанный (.ocr/<stem>.txt) в приоритете, иначе извлечённый.

    Пустой результат НЕ означает «слоя нет»: он получается и когда файла нет, и
    когда извлекатель упал. Один раз это дало ложный диагноз «СКАН» и напрасный
    прогон OCR на 285 страниц: имя файла было в нормализации NFD, путь с тем же
    именем в NFC не совпал, pdftotext вернул пусто. Поэтому существование файла и
    код возврата проверяются явно.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f'нет файла: {path} '
                                '(проверить нормализацию имени: NFC против NFD)')
    d, b = os.path.split(path)
    stem = os.path.splitext(b)[0]
    ocr = os.path.join(d, '.ocr', stem + '.txt')
    if os.path.exists(ocr) and os.path.getsize(ocr) > 5000:
        return open(ocr, encoding='utf8', errors='replace').read(), True
    cmd = (['timeout', '400', 'djvutxt', path] if path.endswith('.djvu')
           else ['timeout', '300', 'pdftotext', '-q', path, '-'])
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0 and not r.stdout:
        raise RuntimeError(f'{cmd[2]} вернул код {r.returncode} на {path}: '
                           + r.stderr.decode('utf8', 'replace')[:200])
    return r.stdout.decode('utf8', 'replace'), False

def _fix_mojibake(t):
    """Слой в однобайтовой кодировке, прочитанный как latin1 -> вернуть кириллицу."""
    if len(re.findall(r'[А-Яа-я]', t)) > 200:
        return t, None
    for enc in ('cp1251', 'koi8-r'):
        try:
            f = t.encode('latin1', 'replace').decode(enc, 'replace')
        except Exception:
            continue
        if len(re.findall(r'[А-Яа-я]', f)) > 1000:
            return f, enc
    return t, None

def normalize(t):
    t = re.sub(r'-\s*\n\s*', '', t)       # дефисный перенос склеиваем
    t = re.sub(r'\s+', ' ', t)            # прочее пробельное -> ОДИН пробел
    return t.replace('ё', 'е').replace('Ё', 'Е')

def load(path):
    """-> (нормализованный текст, сведения). Сведения: from_ocr, recoded, verdict."""
    t, from_ocr = _raw(path)
    t, recoded = _fix_mojibake(t)
    n = len(t)
    cyr = len(re.findall(r'[А-Яа-я]', t))
    lat = len(re.findall(r'[A-Za-z]', t))
    q = t.count('?')
    letters = cyr + lat
    if n < 5000:
        verdict = 'СКАН (слоя нет) -> tools/ocr.sh'
    elif letters and q > 0.15 * letters:
        verdict = 'БИТЫЙ СЛОЙ (кириллица -> «?») -> FORCE=1 tools/ocr.sh'
    elif letters < n * 0.25:
        verdict = 'слой сомнителен — посмотреть глазами'
    else:
        verdict = 'читаем'
    return normalize(t), {'len': n, 'cyr': cyr, 'lat': lat, 'q': q,
                          'from_ocr': from_ocr, 'recoded': recoded, 'verdict': verdict}
