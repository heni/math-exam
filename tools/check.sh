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
for md in questions/*/theory.md questions/*/slides.md; do
  [ -e "$md" ] || continue
  d=$(dirname "$md")
  grep -oE '!\[[^]]*\]\(([^)]+)\)' "$md" 2>/dev/null | sed -E 's/.*\((.*)\)/\1/' | sed 's/{.*//' | while read -r img; do
    [ -n "$img" ] || continue
    case "$img" in http*) continue;; esac
    [ -e "$d/$img" ] || [ -e "$img" ] || [ -e "assets/$img" ] && continue
    echo "  [!] $md -> отсутствует $img"
  done
done
note "проверено"

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
for p in ['docs/sources.md','docs/glossary.md','README.md','docs/style-guide.md']+glob.glob('questions/*/README.md'):
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
    (r'\][ ]*\([^)\s]*[/.][^)\s]*\)', '[текст](ссылка)'),
)
bad = 0
for path in sorted(glob.glob('questions/*/theory.md') + glob.glob('questions/*/slides.md')):
    depth = 0
    in_display = False
    for i, line in enumerate(open(path, encoding='utf8'), 1):
        # многострочный $$-блок: состояние переносится между строками
        n_dd = line.count('$$')
        was_display = in_display
        if n_dd % 2:
            in_display = not in_display
        if was_display or in_display:
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
                    print(f'  [!] {path}:{i}: markdown {what} внутри LaTeX-окружения')
                    bad = 1
        else:
            if re.search(r'~\\(ref|eqref)\{', probe):
                print(f'  [!] {path}:{i}: ~\\ref вне окружения — печатается видимой тильдой')
                bad = 1
        depth += opens - closes
    if depth != 0:
        print(f'  [!] {path}: незакрытые окружения (глубина {depth})')
        bad = 1
if not bad:
    print('  чисто')
sys.exit(bad)
PYMD
[ $? -eq 0 ] || fail=1

echo
[ "$fail" -eq 0 ] && echo "ИТОГ: чисто" || echo "ИТОГ: есть замечания"
exit "$fail"
