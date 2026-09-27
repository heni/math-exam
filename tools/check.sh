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

echo
[ "$fail" -eq 0 ] && echo "ИТОГ: чисто" || echo "ИТОГ: есть замечания"
exit "$fail"
