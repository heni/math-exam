#!/usr/bin/env python3
"""Линтер short.md: нумерация Т/Л/П, внутренние ссылки, черновые пометки,
баланс $, дословные дубли пунктов, тильды вне математики.

Использование: python3 tools/short-lint.py
Выход: 0 — чисто, 1 — есть находки. Не заменяет чтение: ловит только
машинно-проверяемые классы дефектов (дубль Т5 в №18, висячая ссылка на
лемму в №12 — именно такие случаи и собраны).

Нумерация: заголовки вида «- **Т7 ...», «- **Л1 ...», «- **П1 ...»;
допускается кириллический суффикс («Т8а» — самостоятельный номер, не дубль
Т8). Ссылки вида Т7, Л2, Т8а вне заголовков должны указывать на
существующие номера своего класса.

Тильды: «~» вне математики в исходнике печатается в PDF как видимый символ
(pandoc не трактует её как неразрывный пробел в прозе) — флаг. Дублирующе
проверяется извлечённый текст собранного PDF: если тильда реально
напечаталась, она видна там независимо от причины.
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

DRAFT = re.compile(r"TODO|FIXME|XXX|уточнить позже|вставить позже|заглушка|черновик")
HEADER = re.compile(r"\s*- \*\*([ТЛП])(\d+)([а-я]?)\b")
REF = re.compile(r"(?<![А-Яа-яA-Za-z\\])([ТЛП])(\d+)([а-я]?)")


def lint(md: Path):
    findings = []
    lines = md.read_text(encoding="utf8").split("\n")
    headers = []            # (класс, номер, суффикс, строка)
    refs = []               # (строка, класс, номер, суффикс)
    bullets = {}            # нормализованный текст пункта -> номер строки

    in_math = False
    math_start = 0
    for i, ln in enumerate(lines, 1):
        m = DRAFT.search(ln)
        if m and not ln.lstrip().startswith("<!--"):
            findings.append(f"{i}: черновая пометка «{m.group(0)}»")
        # баланс $ и тильды: сегментный проход с состоянием математики
        pos = 0
        for ch in re.finditer(r"(?<!\\)\$", ln):
            seg = ln[pos:ch.start()]
            if not in_math and re.search(r"(?<!\\)~", seg):
                findings.append(f"{i}: «~» вне математики — печатается видимой тильдой в PDF")
            in_math = not in_math
            if in_math:
                math_start = i
            pos = ch.end()
        seg = ln[pos:]
        if not in_math and re.search(r"(?<!\\)~", seg):
            findings.append(f"{i}: «~» вне математики — печатается видимой тильдой в PDF")
        hm = HEADER.match(ln)
        if hm:
            headers.append((hm.group(1), int(hm.group(2)), hm.group(3), i))
        for rm in REF.finditer(ln):
            refs.append((i, rm.group(1), int(rm.group(2)), rm.group(3)))

    if in_math:
        findings.append(f"{math_start}: незакрытая inline-математика (до конца файла)")

    defined = {(k, n, suf) for k, n, suf, _ in headers}
    header_lines = {(k, n, suf): i for k, n, suf, i in headers}
    # 1. дубли и пропуски (по каждому классу отдельно; суффикс — часть ключа)
    seen = {}
    for k, n, suf, i in headers:
        key = (k, n, suf)
        if key in seen:
            findings.append(f"{i}: повтор номера {k}{n}{suf} (уже определён на строке {seen[key]})")
        seen[key] = i
    for k in ("Т", "Л", "П"):
        nums = sorted({n for kk, n, _, _ in headers if kk == k})
        if nums:
            missing = sorted(set(range(1, max(nums) + 1)) - set(nums))
            if missing:
                findings.append(f"пропуски в нумерации {k}: {missing}")
    # 2. ссылки на несуществующие номера
    for i, k, n, suf in refs:
        if (k, n, suf) in defined and header_lines[(k, n, suf)] == i:
            continue  # собственный заголовок
        if (k, n, suf) not in defined:
            have = sorted(f"{kk}{n2}{s2}" for kk, n2, s2 in defined if kk == k)
            findings.append(f"{i}: ссылка на {k}{n}{suf}, но определены только {have}")
    # 3. дословные дубли пунктов
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.lstrip().startswith("- "):
            body = [ln]
            j = i + 1
            while j < len(lines) and lines[j].strip() and not lines[j].lstrip().startswith(("- ", "#")):
                body.append(lines[j])
                j += 1
            norm = re.sub(r"\s+", " ", " ".join(body)).strip()
            if norm in bullets:
                findings.append(f"{i + 1}: дословный дубль пункта строки {bullets[norm]}")
            else:
                bullets[norm] = i + 1
            i = j
        else:
            i += 1
    return findings


def lint_pdf_tildes(md: Path):
    """Тильды в извлечённом тексте собранного PDF — что реально напечаталось."""
    pdf = md.with_suffix(".pdf")
    if not pdf.exists():
        return ["PDF нет — тильды в печати НЕ ПРОВЕРЕНЫ"]
    if not shutil.which("pdftotext"):
        return ["pdftotext не найден — тильды в печати НЕ ПРОВЕРЕНЫ"]
    out = subprocess.run(["pdftotext", "-raw", str(pdf), "-"],
                         capture_output=True, text=True).stdout
    bad = []
    for page, chunk in enumerate(out.split("\f"), 1):
        for ln in chunk.split("\n"):
            if "~" in ln:
                bad.append(f"PDF стр. {page}: напечатанная тильда: {ln.strip()[:80]}")
    return bad


def main():
    bad = 0
    for md in sorted(Path("questions").glob("*/short.md")):
        f = lint(md)
        f += lint_pdf_tildes(md)
        if f:
            bad = 1
            print(f"== {md}")
            for x in f:
                print(f"   {x}")
    if not bad:
        print("  чисто")
    sys.exit(bad)


if __name__ == "__main__":
    main()
