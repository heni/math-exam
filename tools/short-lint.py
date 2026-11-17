#!/usr/bin/env python3
"""Линтер short.md: нумерация Т/Л, внутренние ссылки, черновые пометки,
баланс $, дословные дубли пунктов.

Использование: python3 tools/short-lint.py
Выход: 0 — чисто, 1 — есть находки. Не заменяет чтение: ловит только
машинно-проверяемые классы дефектов (дубль Т5 в №18, висячая ссылка на
лемму в №12 — именно такие случаи и собраны).
"""
import re
import sys
from pathlib import Path

DRAFT = re.compile(r"TODO|FIXME|XXX|уточнить позже|вставить позже|заглушка|черновик")

def lint(path: Path):
    findings = []
    lines = path.read_text(encoding="utf8").split("\n")
    headers_t = []          # номера определённых теорем (из заголовков «**Т7 ...»)
    headers_l = []          # нумерованные леммы (в некоторых вопросах: «**Л1 ...»)
    refs = []               # (строка, вид, номер) всех упоминаний Т/Л цифрой
    bullets = {}            # нормализованный текст пункта -> номер строки

    in_math = False         # inline-математика может переноситься между строками
    math_start = 0          # пункта внутри одного абзаца — баланс считаем по файлу
    for i, ln in enumerate(lines, 1):
        # черновые пометки
        m = DRAFT.search(ln)
        if m and not ln.lstrip().startswith("<!--"):
            findings.append(f"{i}: черновая пометка «{m.group(0)}»")
        # баланс $ (неэкранированных), состояние переносится между строками
        for ch in re.finditer(r"(?<!\\)\$", ln):
            in_math = not in_math
            if in_math:
                math_start = i
        # заголовки теорем и (где принято) лемм
        hm = re.match(r"\s*- \*\*([ТЛ])(\d+)\b", ln)
        if hm:
            (headers_t if hm.group(1) == "Т" else headers_l).append((int(hm.group(2)), i))
        # ссылки вида Т7, Л2 — только явные Т/Л+цифра
        for rm in re.finditer(r"(?<![А-Яа-яA-Za-z\\])([ТЛ])(\d+)", ln):
            refs.append((i, rm.group(1), int(rm.group(2))))
    if in_math:
        findings.append(f"{math_start}: незакрытая inline-математика (до конца файла)")

    defined = {n for n, _ in headers_t}
    # 1. дубли и пропуски в нумерации теорем
    seen = {}
    for n, i in headers_t:
        if n in seen:
            findings.append(f"{i}: повтор номера Т{n} (уже определена на строке {seen[n]})")
        seen[n] = i
    if defined:
        missing = sorted(set(range(1, max(defined) + 1)) - defined)
        if missing:
            findings.append(f"пропуски в нумерации теорем: {missing}")
    # 2. ссылки на несуществующие номера; леммы не нумеруются — любая «Лn» подозрительна
    header_lines = {i for _, i in headers_t}
    for i, kind, n in refs:
        if kind == "Т":
            if n in defined and i in header_lines:
                continue
            if n not in defined:
                findings.append(f"{i}: ссылка на Т{n}, но определены только Т{sorted(defined)}")
        else:
            defined_l = {num for num, _ in headers_l}
            if n not in defined_l:
                findings.append(f"{i}: ссылка на Л{n}, но нумерованные леммы: "
                                f"Л{sorted(defined_l)}")
    # 5. дословные дубли пунктов
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


def main():
    bad = 0
    for md in sorted(Path("questions").glob("*/short.md")):
        f = lint(md)
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
