#!/usr/bin/env bash
# Проверки согласованности материалов: bash tools/check.sh
# Не заменяет review-fix loop — ловит только машинно-проверяемое.
set -uo pipefail
cd "$(dirname "$0")/.."
fail=0
note() { printf '  %s\n' "$1"; }
bad()  { printf '  [!] %s\n' "$1"; fail=1; }

echo "== 1. Шаблонные заглушки, оставшиеся в тексте =="
if grep -rn --include='*.md' --include='*.py' -E 'ЗАГОЛОВОК ВОПРОСА|Вопрос NN|NN-slug|АВТОР' questions/ 2>/dev/null; then
  bad "в материалах остались подстановки из шаблона"
else
  note "чисто"
fi

echo "== 2. Картинки, на которые ссылаются, но которых нет =="
# Цикл while раньше стоял в конвейере, то есть в подоболочке, и присваивание
# fail=1 из bad() терялось: пункт печатал «[!]» и возвращал ноль. Гейт, который
# жалуется и пропускает, хуже отсутствующего. Подстановка процесса держит цикл
# в текущей оболочке.
missing_img=0
for md in questions/*/theory.md questions/*/slides.md; do
  [ -e "$md" ] || continue
  d=$(dirname "$md")
  # Ищем ЦЕЛЬ ссылки, а не картинку целиком: подписи бывают многострочными и
  # содержат «]» внутри математики ($[-1,1]$, \eqref{...}), поэтому шаблон
  # «!\[[^]]*\]\(...\)» обрывался на первой же скобке и не видел ни одной
  # картинки конспекта. Проверено подстановкой: удалённый figures/*.pdf теперь
  # ловится, раньше проходил молча.
  while read -r img; do
    [ -n "$img" ] || continue
    case "$img" in http*) continue;; esac
    if [ ! -e "$d/$img" ] && [ ! -e "$img" ] && [ ! -e "assets/$img" ]; then
      bad "$md -> отсутствует $img"
      missing_img=1
    fi
  done < <(grep -oE '\]\([^)]+\.(pdf|png|jpg|jpeg|svg)\)' "$md" 2>/dev/null | sed -E 's/^\]\((.*)\)$/\1/')
done
[ "$missing_img" -eq 0 ] && note "чисто"

echo "== 3. PDF старше своего источника =="
for src in questions/*/theory.md questions/*/slides.md; do
  [ -e "$src" ] || continue
  pdf="${src%.md}.pdf"
  if [ -e "$pdf" ] && [ "$src" -nt "$pdf" ]; then bad "$pdf старше $src — нужен make"; fi
done
for src in questions/*/examples.py; do
  [ -e "$src" ] || continue
  nb="${src%.py}.ipynb"
  if [ -e "$nb" ] && [ "$src" -nt "$nb" ]; then bad "$nb старше $src — нужен make"; fi
done
# Картинки строит ноутбук, а вставляют их конспект и слайды: PDF, собранный
# раньше картинки, показывает прошлую версию графика и выглядит свежим.
# Сравниваем любую картинку вопроса с любым его PDF, не разбирая, кто какую
# вставляет: лишняя пересборка дешевле пропущенной устаревшей картинки.
for pdf in questions/*/theory.pdf questions/*/slides.pdf; do
  [ -e "$pdf" ] || continue
  d=$(dirname "$pdf")
  for fig in "$d"/figures/*.pdf; do
    [ -e "$fig" ] || continue
    if [ "$fig" -nt "$pdf" ]; then bad "$pdf старше картинки $fig — нужен make"; fi
  done
done
note "проверено"

echo "== 4. Явный seed в численных примерах =="
for py in questions/*/examples.py; do
  [ -e "$py" ] || continue
  grep -q 'default_rng' "$py" || note "$py: нет np.random.default_rng — проверить, нужен ли seed"
  grep -qE 'np\.random\.(seed|rand|randn|normal|uniform|choice)\(' "$py" \
    && bad "$py: legacy np.random.* — только default_rng(SEED)"
done
note "проверено"

echo "== 5. Термины материалов против глоссария =="
if [ -s docs/glossary.md ]; then
  missing=0
  for d in questions/*/; do
    [ -s "$d/theory.md" ] || continue
    slug=$(basename "$d")
    grep -q "$slug" docs/glossary.md || { note "глоссарий не ссылается на $slug"; missing=1; }
  done
  [ "$missing" -eq 0 ] && note "чисто"
else
  bad "docs/glossary.md пуст или отсутствует"
fi

echo "== 6. Пайплайн-файлы не должны быть в git =="
if [ -d .git ]; then
  tracked=$(git ls-files -- CLAUDE.md TODO.md AI.md backlog.md 'AI/*' 2>/dev/null)
  if [ -n "$tracked" ]; then bad "в git попали пайплайн-файлы:"; echo "$tracked" | sed 's/^/      /'
  else note "чисто"; fi
  untracked=$(git status --porcelain --untracked-files=all 2>/dev/null | grep -E '^\?\? (CLAUDE\.md|TODO\.md|AI\.md|backlog\.md|AI/)' || true)
  [ -n "$untracked" ] && bad "пайплайн-файлы видны как untracked — проверить .gitignore"
else
  note "git не инициализирован — пропуск"
fi

echo "== 7. Git-версируемые файлы не ссылаются на локальные каталоги =="
# Правило аудиторий: читатель репозитория не видит deps/ и presexmpl/, значит
# путей внутрь них в закоммиченных файлах быть не должно. На книгу ссылаться
# можно и нужно — но библиографической записью: номером [N] из docs/sources.md,
# который сам версионируется и путей не содержит.
# Исключены .gitignore (пути там обязаны быть) и сам этот скрипт (его grep-шаблон
# содержит те же строки — иначе проверка вечно флагает себя).
if [ -d .git ]; then
  leak=$(git ls-files -z 2>/dev/null | xargs -0 grep -ln -E 'deps/|presexmpl/' 2>/dev/null \
         | grep -vE '^(\.gitignore|tools/check\.sh)$' || true)
  if [ -n "$leak" ]; then
    bad "ссылки на локальные каталоги в git-версируемых файлах:"
    echo "$leak" | sed 's/^/      /'
  else
    note "чисто"
  fi
else
  note "git не инициализирован — пропуск"
fi

echo "== 8. Ссылки на литературу: [N] против набора и списка =="
# Три независимые проверки, потому что рассинхронизироваться могут три вещи:
# файлы в наборе, записи в списке литературы и ссылки [N] в текстах. Пропажа
# библиографической записи однажды уже случилась молча при пересборке списка.
python3 - <<'PYCHK'
import re,glob,os,sys
fail=0
# 1) номера работ в наборе
works=set()
for f in glob.glob('deps/*.pdf')+glob.glob('deps/*.djvu'):
    works.add(os.path.basename(f).split('_')[0])
# 2) номера в списке литературы
src=open('docs/sources.md',encoding='utf8').read()
a=src.index('## Список литературы'); b=src.index('\n## ',a+5)
biblio={f'{int(m.group(1)):02d}' for m in re.finditer(r'^(\d+)\.\s', src[a:b], re.M)}
only_w=sorted(works-biblio); only_b=sorted(biblio-works)
if only_w: print(f'  [!] работы без библиографической записи: {only_w}'); fail=1
if only_b: print(f'  [!] записи без файла в наборе: {only_b}'); fail=1
if not only_w and not only_b: print(f'  набор и список согласованы: {len(works)} работ')
# 3) ссылки [N] во всех версионируемых текстах
bad={}
# Конспект и слайды тоже версионируются и тоже полны ссылок [N] — в вопросе 11
# их двадцать, больше, чем в его README, и именно конспект читает экзаменатор.
# Математика вида $[a,b]$, $[-1,1]$, $(2k-1)$ под шаблон не попадает.
for p in (['docs/sources.md','docs/glossary.md','README.md','docs/style-guide.md']
          + glob.glob('questions/*/README.md')
          + glob.glob('questions/*/theory.md') + glob.glob('questions/*/slides.md')):
    if not os.path.exists(p): continue
    for m in re.finditer(r'\[(\d{1,2})\]', open(p,encoding='utf8').read()):
        n=f'{int(m.group(1)):02d}'
        if n not in works: bad.setdefault(p,set()).add(m.group(1))
if bad:
    fail=1
    for p,ns in sorted(bad.items()): print(f'  [!] {p}: ссылки на несуществующие работы {sorted(ns)}')
else: print('  все ссылки [N] указывают на существующие работы')
sys.exit(fail)
PYCHK
[ $? -eq 0 ] || fail=1

echo "== 9. Markdown-разметка внутри LaTeX-окружений =="
# Содержимое \begin{theorem}...\end{theorem} pandoc отдаёт в LaTeX КАК ЕСТЬ,
# поэтому любая markdown-конструкция внутри печатается сырой. Уже случались все
# четыре: **жирный**, *курсив*, `код`, [текст](ссылка). Отдельно: «~\ref» в
# markdown-прозе печатается видимой тильдой. Проверяем ИСХОДНИК — так ловится
# класс, а не перечень известных случаев.
#
# Что НЕ ловится (осознанно): markdown внутри однострочной математики $...$ и
# внутри многострочных $$-блоков маскируется целиком, поэтому дефект,
# спрятанный в формулу, пройдёт. Такой ещё ни разу не случался.
python3 - <<'PYMD'
import re, glob, sys

MARKDOWN = (
    (r'\*\*', '** (жирный)'),
    (r'(?<![\w*\\])\*(?![\s*])[^*\n]*[^\s*]\*(?![\w*])', '*курсив*'),
    (r'(?<!\\)`', '` (код)'),
    # [43] (Зализняк) — ссылка на литературу, а не markdown: цифра перед ] исключена
    (r'(?<![0-9])(?<!\\ref)(?<!\\eqref)(?<!\\cite)\][ ]*\([^)\s]{1,80}\)', '[текст](ссылка)'),
)
MATH_ENV = r'equation|align|aligned|cases|array|gather|multline|split|pmatrix|bmatrix'
bad = 0
for path in sorted(glob.glob('questions/*/theory.md') + glob.glob('questions/*/slides.md')):
    depth = 0
    in_display = False
    math_depth = 0
    hits = []
    last_open = 0
    for i, line in enumerate(open(path, encoding='utf8'), 1):
        # многострочный $$-блок и математические окружения: состояние
        # переносится между строками, иначе L_n[f](x) в \begin{equation}
        # читается как markdown-ссылка
        n_dd = line.count('$$')
        was_display = in_display
        if n_dd % 2:
            in_display = not in_display
        opened_math = len(re.findall(r'\\begin\{(?:' + MATH_ENV + r')\*?\}', line))
        opened_math += len(re.findall(r'(?<!\\)\\\[', line))
        closed_math = len(re.findall(r'\\end\{(?:' + MATH_ENV + r')\*?\}', line))
        closed_math += len(re.findall(r'(?<!\\)\\\]', line))
        in_math = math_depth > 0 or opened_math > 0
        math_depth += opened_math - closed_math
        if was_display or in_display or in_math:
            probe = ''
        else:
            probe = re.sub(r'\$\$.*?\$\$|\$[^$\n]*\$', '', line)
        # заголовок окружения разбирается уже как внутренний: \begin{theorem}[...]
        opens = len(re.findall(r'\\begin\{', line))
        closes = len(re.findall(r'\\end\{', line))
        inside = depth > 0 or opens > closes
        if inside:
            for pat, what in MARKDOWN:
                if re.search(pat, probe):
                    hits.append(f'  [!] {path}:{i}: markdown {what} внутри LaTeX-окружения')
        else:
            if re.search(r'~\\(ref|eqref)\{', probe):
                print(f'  [!] {path}:{i}: ~\\ref вне окружения — печатается видимой тильдой')
                bad = 1
        if opens > closes:
            last_open = i
        depth += opens - closes
    # Непарный \begin делает «внутри окружения» весь остаток файла, и находки
    # ниже него — ложные. В этом случае печатаем только причину: иначе поиск
    # уходит на тридцать несуществующих дефектов вместо одной строки.
    if depth != 0:
        print(f'  [!] {path}: незакрытое окружение (глубина {depth} в конце файла); '
              f'последний непарный \\begin — строка {last_open}')
        bad = 1
    elif hits:
        for h in hits:
            print(h)
        bad = 1
if not bad:
    print('  чисто')
sys.exit(bad)
PYMD
[ $? -eq 0 ] || fail=1

echo "== 10. Одиночная обратная косая в конце строки внутри LaTeX-окружения =="
# Внутри \begin{env}...\end{env} pandoc разбирает сырой LaTeX сам, и «\» перед
# переводом строки сбивает ему токенизацию: окружение выводится ДВАЖДЫ — один раз
# верно, второй раз искажённым огрызком, после чего xelatex падает на «Missing $».
# Сообщение указывает на строку в сгенерированном .tex, а не в исходнике, поэтому
# без гейта причина ищется вслепую.
#
# Две границы области действия, обе проверены подстановкой:
#  - «\\» (перевод строки LaTeX) безвреден — ловим только НЕЧЁТНОЕ число косых;
#  - вне окружений, в прозе и в $$-блоках, «\» в конце строки тоже безвреден:
#    там pandoc читает математику своим разбором. Поэтому отслеживаем глубину
#    окружений, как в пункте 9, и флагаем только внутри.
python3 - <<'PYBS'
import re, glob, sys
bad = 0
for path in sorted(glob.glob('questions/*/theory.md') + glob.glob('questions/*/slides.md')
                   + glob.glob('docs/*.md')):
    depth = 0
    for i, line in enumerate(open(path, encoding='utf8'), 1):
        line = line.rstrip('\n')
        opens = len(re.findall(r'\\begin\{', line))
        closes = len(re.findall(r'\\end\{', line))
        inside = depth > 0 or opens > closes
        if inside and re.search(r'(^|[^\\])(\\\\)*\\$', line):
            print(f'  [!] {path}:{i}: одиночная «\\» в конце строки внутри окружения')
            bad = 1
        depth += opens - closes
sys.exit(bad)
PYBS
[ $? -eq 0 ] && note "чисто" || fail=1

echo "== 11. Кириллица внутри \\mathrm и родственных =="
# В конспекте (scrartcl) такой индекс печатается, в beamer — нет: metropolis берёт
# математический шрифт из Latin Modern, где кириллицы нет, и буквы ПРОПАДАЮТ, а
# сборка проходит. Ловится только предупреждением «Missing character» в логе,
# которого никто не читает. Правильная запись — \text{...}: он берёт текстовый шрифт.
if grep -rn --include='*.md' -P '\\(mathrm|mathbf|mathit|mathsf|mathtt)\{[^}]*[А-Яа-яЁё]' questions/ 2>/dev/null; then
  bad "кириллица внутри \\mathrm — в beamer буквы не печатаются; писать \\text{...}"
else
  note "чисто"
fi

echo
[ "$fail" -eq 0 ] && echo "ИТОГ: чисто" || echo "ИТОГ: есть замечания"
exit "$fail"
