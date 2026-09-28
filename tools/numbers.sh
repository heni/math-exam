#!/usr/bin/env bash
# Числа материалов против вывода ноутбука: bash tools/numbers.sh [NN]
#
# ЧТО ЭТО МЕРИТ. Для каждого числа вида $0{,}1234$ в theory.md и slides.md
# проверяет, встречается ли оно в выводе ячеек examples.ipynb того же вопроса
# (с точностью до округления до последнего знака). Печатает те, что не нашлись.
#
# ЧТО ЭТО НЕ МЕРИТ И ПОЧЕМУ ЭТО НЕ ГЕЙТ. По `CLAUDE.md` §5 число законно, если
# прослеживается до расчёта в examples.py, до учебника с главой ИЛИ получается
# прямым выводом из формулы, стоящей рядом. Третий сорт машина не отличает от
# числа, взятого с потолка: у вопроса 11 все пять «ненайденных» — арифметика от
# выписанной тут же формулы ($\sqrt{0{,}21}$, $\tfrac12(1+\tfrac{\sqrt3}{2})$).
# Поэтому вывод этого скрипта — СПИСОК КАНДИДАТОВ НА СВЕРКУ ГЛАЗАМИ, а не
# приговор, и в `tools/check.sh` он не включён: гейт, который валит сборку на
# законном числе, вынудит обходить себя, а обойдённый гейт хуже отсутствующего.
#
# Польза измерена: на вопросе 12 нашёл два числа, которые действительно нигде не
# печатались (дисперсия суммы очков и нецелый объём выборки), и оба были добавлены
# в вывод ноутбука.
set -uo pipefail
cd "$(dirname "$0")/.."
python3 - "${1:-}" <<'PY'
import re, json, glob, os, sys
only = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else None

def notebook_numbers(d):
    f = os.path.join(d, 'examples.ipynb')
    if not os.path.exists(f):
        return None
    nb = json.load(open(f, encoding='utf8'))
    out = ''.join(''.join(o.get('text', [])) for c in nb['cells'] for o in c.get('outputs', []))
    return [float(x) for x in re.findall(r'\d+\.\d+', out)] + \
           [float(x) for x in re.findall(r'(?<![\d.])\d+(?![\d.])', out)]

total = 0
for d in sorted(glob.glob('questions/*/')):
    if only and not os.path.basename(d.rstrip('/')).startswith(only):
        continue
    nums = notebook_numbers(d)
    if nums is None:
        continue
    for src in ('theory.md', 'slides.md'):
        p = os.path.join(d, src)
        if not os.path.exists(p):
            continue
        txt = open(p, encoding='utf8').read()
        # Уровень квантили в индексе (z_{0{,}975}) — не вычисленная величина, а
        # имя: печатать его ноутбук не обязан.
        txt = re.sub(r'z_\{\d+\{,\}\d+\}', 'z', txt)
        miss = []
        for m in re.finditer(r'(\d+)\{,\}(\d+)', txt):
            val = float(m.group(1) + '.' + m.group(2))
            tol = 0.5 * 10 ** (-len(m.group(2)))
            if not any(abs(x - val) <= tol for x in nums):
                miss.append(m.group(0).replace('{,}', ','))
        if miss:
            total += len(set(miss))
            print(f'  {p}: {sorted(set(miss))}')
print(f'  кандидатов на сверку: {total}' if total else '  все числа нашлись в выводе ноутбука')
PY
