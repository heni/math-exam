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
  tracked=$(git ls-files -- CLAUDE.md TODO.md AI.md backlog.md 'AI/*' '.claude/*' 2>/dev/null)  # gate-pattern
  if [ -n "$tracked" ]; then bad "в git попали пайплайн-файлы:"; echo "$tracked" | sed 's/^/      /'
  else note "чисто"; fi
  untracked=$(git status --porcelain --untracked-files=all 2>/dev/null | grep -E '^\?\? (CLAUDE\.md|TODO\.md|AI\.md|backlog\.md|AI/|\.claude/)' || true)  # gate-pattern
  [ -n "$untracked" ] && bad "пайплайн-файлы видны как untracked — проверить .gitignore"
else
  note "git не инициализирован — пропуск"
fi

echo "== 7. Git-версируемые файлы не ссылаются на локальное и на пайплайн-файлы =="
# ДВА правила с РАЗНОЙ областью действия, и смешивать их нельзя — один раз уже
# смешал, и гейт потребовал выкрутить инструмент, которому путь к книгам нужен
# по существу.
#
# Первое: читатель репозитория не видит deps/ и presexmpl/, поэтому путей внутрь
# них нет в том, что адресовано ЕМУ, — а это всё закоммиченное, кроме каталога
# tools/ целиком: и материалы вопросов, и преамбулы, и Makefile он читает. На
# книгу ссылаться можно и нужно, но библиографической записью: номером [N] из
# docs/sources.md. tools/ выведен потому, что его запускает тот, у кого книги
# есть, и путь для него — рабочая величина, а не шум.
#
# Второе: отсылок к локальным служебным файлам, которых у читателя репозитория
# нет, не должно быть НИ В ОДНОМ закоммиченном файле, включая комментарии кода, —
# область шире первой, и поэтому второй проход идёт по всем текстовым файлам
# индекса (-I отбрасывает двоичные), этот скрипт включительно. Единственное
# место, где имена стоят по делу, — список пункта 6; оно помечено маркером
# gate-pattern, и фильтр принимает маркер ТОЛЬКО в строках самого этого файла.
# Так сделано потому, что маркер без такой привязки прятал нарушение в любом
# файле, куда его приписали: проверено подстановкой в Makefile до и после
# правки. Исключать файл целиком нельзя — под таким исключением здесь уже жила
# запрещённая отсылка к локальному файлу в комментарии.
# .gitignore: перечислять эти пути — его работа, поэтому первый проход исключает
# его целиком. Во втором проходе исключаются только строки-пути: комментариев в
# нём десяток, и отсылка, спрятанная в комментарий, — такое же нарушение.
if [ -d .git ]; then
  leak=$(git ls-files -z 2>/dev/null \
         | grep -zvE '^(tools/|\.gitignore$)' \
         | xargs -0 grep -lnI -E 'deps/|presexmpl/' 2>/dev/null || true)
  if [ -n "$leak" ]; then
    bad "ссылки на локальные каталоги в git-версируемых файлах:"
    echo "$leak" | sed 's/^/      /'
  else
    note "каталоги: чисто"
  fi
  # Пайплайн-файлы: пункт 6 проверяет, что они не попали в индекс, здесь — что
  # на них не ссылаются, и ловушка сработала именно в комментарии кода — в
  # tools/numbers.sh, где комментарий цитировал локальный файл по имени.
  #
  # Шаблон требует явного упоминания файла, а не подстроки. Класс предшествующих
  # символов широкий — всё, кроме букв, цифр, «_» и «-», — потому что узкий
  # пропускал форму за формой: имя после «./» и после «../», каталог в обрамлении
  # ёлочек и звёздочек, каталог с номером главы без дефиса. Следующий символ не
  # ограничивается вовсе: замер по всем закоммиченным .ipynb (1,45 МБ) дал ноль
  # ложных срабатываний на base64, так что платить за эту защиту дырами незачем.
  # NB: формы сюда не выписывать дословно — этот же проход их и поймает.
  #
  # Проход построчный, и выводятся строки, а не файлы: отбрасывается ровно
  # помеченная маркером строка, а не весь файл, — иначе этот скрипт, у которого
  # имена стоят в шаблоне, оказался бы вне проверки вместе со своими
  # комментариями.
  #
  # .gitignore идёт отдельным проходом и наоборот: перечислять эти пути — его
  # работа, поэтому строка-путь пропускается, а комментарий в нём проверяется,
  # как любой другой. Режется по первому «#», а не по наличию «#» в строке: у
  # строки-пути бывает хвостовой комментарий.
  pat='(^|[^A-Za-z0-9_-])(CLAUDE|TODO|backlog|AI)\.md([^A-Za-z0-9]|$)|(^|[^A-Za-z0-9_-])AI/|\.claude/'  # gate-pattern
  refs=$(git ls-files -z 2>/dev/null \
         | grep -zv '^\.gitignore$' \
         | xargs -0 grep -nIE "$pat" 2>/dev/null \
         | grep -vE '^tools/check\.sh:[0-9]+:.*# gate-pattern$' || true)
  gi=$(grep -n '#' .gitignore 2>/dev/null | sed 's/^\([0-9]*\):[^#]*#/\1:#/' \
       | grep -E "$pat" | sed 's|^|.gitignore:|' || true)
  refs=$(printf '%s\n%s\n' "$refs" "$gi" | grep -v '^$' || true)
  if [ -n "$refs" ]; then
    bad "ссылки на git-ignored пайплайн-файлы в git-версируемых файлах:"
    echo "$refs" | sed 's/^/      /'
  else
    note "пайплайн-файлы: чисто"
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
echo "== 12. Выключная формула \[...\] вне LaTeX-окружения =="
# Вне окружений pandoc считает обратную косую перед скобкой экранированием, и
# «\[» превращается в обычную «[»: формула попадает в LaTeX текстом, сборка
# падает на первом же \begin{aligned} или на «Missing $». Внутри окружений
# содержимое идёт сырым, и там «\[…\]» работает — поэтому гейт смотрит глубину.
# Правило записано в docs/build.md; за одну сессию ловушка сработала дважды,
# оба раза при переносе блока из окружения в прозу.
python3 - <<'PYBR'
import re, glob, sys
bad = 0
for path in sorted(glob.glob('questions/*/theory.md') + glob.glob('questions/*/slides.md')
                   + glob.glob('docs/*.md')):
    depth = 0
    for i, line in enumerate(open(path, encoding='utf8'), 1):
        opens = len(re.findall(r'\\begin\{', line))
        closes = len(re.findall(r'\\end\{', line))
        inside = depth > 0 or opens > closes
        # `\[` в обратных кавычках — это описание правила, а не разметка:
        # так оно и стоит в docs/build.md, который иначе флагает сам себя.
        probe = re.sub(r'`[^`]*`', '', line)
        if not inside and (re.search(r'(?<!\\)\\\[', probe) or re.search(r'(?<!\\)\\\]', probe)):
            print(f'  [!] {path}:{i}: \\[…\\] вне окружения — писать $$…$$')
            bad = 1
        depth += opens - closes
sys.exit(bad)
PYBR
[ $? -eq 0 ] && note "чисто" || fail=1

echo "== 13. Текст, пропавший из собранного PDF =="
# Переполненный кадр beamer не «вылезает за поля», а МОЛЧА выбрасывает
# содержимое: сборка зелёная, слово в исходнике есть, в PDF его нет. Замер
# геометрии по pdftotext -bbox это не ловит — он меряет напечатанное, а не то,
# что должно было быть напечатано. Ловится только сверкой словарей источника и
# собранного PDF. Проверено подстановкой: два кадра вопроса 08 теряли по абзацу
# при зелёной сборке и чистом замере геометрии.
#
# Сверяются только русские слова длиной от четырёх букв: короткие тонут в шуме
# переносов, а математика и команды LaTeX из исходника выброшены до сравнения.
# Слово, разорванное переносом, в pdftotext приходит целым (переносов в тексте
# нет: у нас нет \hyphenation-разрывов внутри слов в выводе pdftotext).
python3 - <<'PYLOST'
import re, glob, subprocess, sys
bad = 0
for md in sorted(glob.glob('questions/*/slides.md') + glob.glob('questions/*/theory.md')):
    pdf = md[:-3] + '.pdf'
    try:
        out = subprocess.run(['pdftotext', '-raw', pdf, '-'], capture_output=True, text=True, check=True).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        continue
    src = open(md, encoding='utf8').read()
    src = re.sub(r'\$\$.*?\$\$', ' ', src, flags=re.S)
    src = re.sub(r'\$[^$\n]*\$', ' ', src)
    src = re.sub(r'!\[[^\]]*\]\([^)]*\)', ' ', src)
    src = re.sub(r'<!--.*?-->', ' ', src, flags=re.S)
    src = re.sub(r'\\[A-Za-z]+\*?(\[[^\]]*\])?', ' ', src)
    # Две поправки, обе получены подстановкой на закрытых вопросах.
    # 1) Перенос по слогам режет слово на две строки, и pdftotext в нашей сборке
    #    теряет сам дефис: «пара-\nметры» приходит как «пара» и «метры». Поэтому
    #    слово ищется ПОДСТРОКОЙ в потоке букв PDF без пробелов — там разорванное
    #    переносом слово снова склеивается. Недобрать проверка может, выдумать —
    #    нет: выброшенный абзац теряет все свои слова сразу.
    # 2) Режим по умолчанию раскладывает страницу по координатам и вклинивает
    #    между половинами перенесённого слова то, что стоит рядом по вертикали
    #    (подстрочный индекс, заголовок соседнего абзаца). Из-за этого склейка
    #    рвалась и проверка ругалась на целые слова: замер дал четыре ложных
    #    срабатывания на трёх закрытых вопросах. Режим -raw идёт в порядке
    #    содержимого страницы и все четыре снимает.
    stream = ''.join(re.findall(r'[а-яё]+', out.lower()))
    want = {w for w in re.findall(r'[а-яё]{5,}', src.lower())}
    lost = sorted(w for w in want if w not in stream)
    if lost:
        print(f'  [!] {pdf}: в исходнике есть, в PDF нет — {len(lost)} слов: {lost[:12]}')
        bad = 1
if not bad:
    print('  чисто')
sys.exit(bad)
PYLOST
[ $? -eq 0 ] || fail=1


echo
[ "$fail" -eq 0 ] && echo "ИТОГ: чисто" || echo "ИТОГ: есть замечания"
exit "$fail"
