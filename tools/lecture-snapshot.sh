#!/usr/bin/env bash
# Снимок лекционного git-репозитория в один PDF с содержанием → deps/.
#
# Источник снимка — git-репозиторий ИЛИ уже существующий архив снимка (zip с
# PDF-разделами, md-исходниками и манифестом; см. режим архива ниже).
#
# Использование:
#   tools/lecture-snapshot.sh SOURCE OUT_BASENAME [ОПЦИИ]
# SOURCE — URL репозитория или путь к .zip-архиву.
# Опции:
#   --md SUBDIR        (только режим репо) конвертировать все .md под SUBDIR
#                      в PDF по одному на файл; санитизация грязного markdown
#                      внутри. В режиме архива PDF-разделы уже готовы — опция
#                      не нужна
#   --with-pdfs GLOB   включить нативные PDF, glob в синтаксисе find -path
#                      (например 'lectures/*/*.pdf'); в режиме архива без этой
#                      опции берутся ВСЕ PDF архива
#   --header FILE      markdown-фрагмент для шапки страницы содержания
#                      (манифест: URL, коммит, дата, оговорки). В режиме архива
#                      без этой опции шапкой служит manifest.txt архива
#
# Имя результата: deps/OUT_BASENAME_ГГГГ-ММ-ДД_<short7>.pdf
# Зависимости: git + python3-venv + cairosvg (только режим репо); pandoc+xelatex
# и poppler-utils (pdfunite, pdfinfo) — оба режима.
set -euo pipefail

SOURCE="${1:?repo url or archive}"; shift
OUT_BASENAME="${1:?out basename}"; shift
MD_SUBDIR=""
PDF_GLOB=""
HEADER_FILE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --md) MD_SUBDIR="$2"; shift 2;;
        --with-pdfs) PDF_GLOB="$2"; shift 2;;
        --header) HEADER_FILE="$2"; shift 2;;
        *) echo "неизвестная опция: $1" >&2; exit 2;;
    esac
done

TMP="$(mktemp -d /tmp/lecture-snapshot.XXXXXX)"
trap 'rm -rf "$TMP"' EXIT
SRC="$TMP/src"
SNAP_DATE="$(date +%F)"

if [ -f "$SOURCE" ]; then
    MODE=archive
    unzip -q "$SOURCE" -d "$SRC"
    HASHES="$(grep -m1 -E 'Коммит|commit' "$SRC/manifest.txt" 2>/dev/null \
              | grep -oE '[0-9a-f]{7,40}' || true)"
    COMMIT="${HASHES:0:7}"
    if [ -z "$COMMIT" ]; then
        B="$(basename "$SOURCE")"; B="${B%.zip}"; COMMIT="${B##*_}"
        case "$COMMIT" in *[!0-9a-f]*|"") COMMIT=local;; esac
    fi
    REPO_URL="архив: $(basename "$SOURCE")"
    [ -n "$HEADER_FILE" ] || HEADER_FILE="$SRC/manifest.txt"
else
    MODE=repo
    git clone -q --depth 1 "$SOURCE" "$SRC"
    COMMIT="$(git -C "$SRC" rev-parse --short=7 HEAD)"
    REPO_URL="$SOURCE"
fi
OUT="deps/${OUT_BASENAME}_${SNAP_DATE}_${COMMIT}.pdf"

# svg → pdf (картинки md-файлов); cairosvg в одноразовом venv
if [ "$MODE" = repo ]; then
python3 -m venv "$TMP/venv"
"$TMP/venv/bin/pip" -q install cairosvg
"$TMP/venv/bin/python3" - "$SRC" <<'PY'
import os, sys, cairosvg
root = sys.argv[1]; n = 0
for dp, _, fns in os.walk(root):
    for fn in fns:
        if fn.lower().endswith(".svg"):
            src = os.path.join(dp, fn)
            try:
                cairosvg.svg2pdf(url=src, write_to=src[:-4] + ".pdf"); n += 1
            except Exception:
                pass
print(f"snapshot: svg→pdf: {n}")
PY
fi

PDFS=()
if [ -n "$PDF_GLOB" ]; then
    while IFS= read -r p; do PDFS+=("$p"); done \
        < <(find "$SRC" -path "$SRC/$PDF_GLOB" -name '*.pdf' | LC_ALL=C sort)
elif [ "$MODE" = archive ]; then
    while IFS= read -r p; do PDFS+=("$p"); done \
        < <(find "$SRC" -name '*.pdf' | LC_ALL=C sort)
fi

if [ "$MODE" = repo ] && [ -n "$MD_SUBDIR" ]; then
    python3 - "$SRC" "$MD_SUBDIR" "$TMP" <<'PY'
import os, re, subprocess, sys, json
root, sub, tmp = sys.argv[1], sys.argv[2], sys.argv[3]
ENV = re.compile(r"\\begin\{(align\*?|equation\*?|gather\*?|multline\*?)\}")
SPAN = re.compile(r"\$\$(.+?)\$\$")

def sanitize(t):
    t = re.compile(r"\A---\n.*?\n---\n", re.S).sub("", t, count=1)
    def img(m):
        alt, path = m.group(1), m.group(2).strip()
        low = path.lower()
        if low.startswith("http") or low.endswith((".shtml", ".htm", ".html", ".php")):
            return ""
        if low.endswith(".svg"):
            path = path[:-4] + ".pdf"
        return f"![{alt}]({path})"
    t = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", img, t)
    t = re.sub(r"<img[^>]*>", "", t)
    t = (t.replace("\\begin{split}", "\\begin{aligned}")
           .replace("\\end{split}", "\\end{aligned}"))
    t = re.sub(r"\\tag\{[^}]*\}", "", t)
    t = t.replace("\\faPython", "Python").replace("\\faGem", "")
    t = re.sub(r"\\R(?![a-zA-Z])", r"\\mathbb{R}", t)
    t = t.replace("Customer\\Source", "Customer / Source")
    t = t.replace("best\\worst", "best/worst")
    t = t.replace("\\_{", "_{")
    t = re.sub(r"\{\#[a-zA-Z0-9_-]+\}", "", t)
    t = re.sub(r"\\lt(?![a-zA-Z])", "<", t)
    t = re.sub(r"\\gt(?![a-zA-Z])", ">", t)
    lines = t.split("\n"); out = []; in_f = in_c = False; buf = []
    def flush():
        nonlocal buf
        if buf:
            content = "\n".join(buf)
            out.append("")
            if ENV.search(content):
                out.extend(b for b in buf if b.strip())
            else:
                out.append("\\begin{displaymath}")
                out.extend(buf)
                out.append("\\end{displaymath}")
            out.append("")
            buf = []
    for ln in lines:
        s = ln.strip()
        if s.startswith("```"):
            flush(); in_c = not in_c; out.append(ln); continue
        if in_c:
            out.append(ln); continue
        s_f = re.sub(r"^(?:(?:[-*+]|\d+\.)\s+)", "", s)   # $$ с маркером списка
        if s_f == "$$":
            if in_f: in_f = False; flush()
            else: in_f = True; buf = []
            continue
        if in_f:
            if s: buf.append(ln)
            continue
        if "$$" in ln:
            sld = re.match(r"^\$\$(.+)\$\$$", s)
            if sld:
                body = sld.group(1)
                out.append("")
                out.append(body if ENV.search(body)
                           else "\\begin{displaymath} " + body + " \\end{displaymath}")
                out.append("")
                continue
            ln = SPAN.sub(lambda m: "\\begin{displaymath} " + m.group(1) +
                          " \\end{displaymath}", ln)
            if "$$" not in ln:
                out.append(ln); continue
            m = re.match(r"^(.*?\S)(\$\$)$", ln)
            if m:
                if m.group(1).strip(): out.append(m.group(1))
                in_f = True; buf = []
                continue
            m2 = re.match(r"^\$\$(.+)$", ln.strip())
            if m2:
                out.append("")
                out.append("\\begin{displaymath} " + m2.group(1))
                out.append("")
                in_f = True; buf = []
                continue
            out.append(ln); continue
        out.append(ln)
    if in_f: out.append("\\end{displaymath}")
    else: flush()
    t = "\n".join(out)
    lines = t.split("\n"); out = []; in_c = False
    for ln in lines:
        s = ln.strip()
        if s.startswith("```"):
            in_c = not in_c; out.append(ln); continue
        if not in_c:
            ln = re.sub(r"(?<!\\) \$", "$", ln)
            ln = re.sub(r"\$ ", "$", ln)
        out.append(ln)
    return "\n".join(out)

mds = []
base = os.path.join(root, sub)
for dp, dns, fns in os.walk(base):
    dns.sort()
    for fn in sorted(fns):
        if fn.endswith(".md"):
            mds.append(os.path.join(dp, fn))
os.makedirs(os.path.join(tmp, "mdpdf"), exist_ok=True)
ok, fail = [], []
for f in mds:
    rel = os.path.relpath(f, root)
    dst = os.path.join(tmp, "mdpdf", rel[:-3].replace("/", "__") + ".pdf")
    open(os.path.join(tmp, "_one.md"), "w", encoding="utf-8").write(
        sanitize(open(f, encoding="utf-8", errors="ignore").read()))
    r = subprocess.run(["pandoc", os.path.join(tmp, "_one.md"), "-o", dst,
                        "--pdf-engine=xelatex", f"--resource-path={root}",
                        "-V", "mainfont=DejaVu Serif", "-V", "geometry:margin=2cm",
                        "-V", "colorlinks=true", "-V", "urlcolor=NavyBlue",
                        "--highlight-style=tango"],
                       capture_output=True, text=True, timeout=180)
    (ok if r.returncode == 0 else fail).append(rel)
json.dump({"ok": ok, "fail": fail}, open(os.path.join(tmp, "md_convert.json"), "w"))
print(f"snapshot: md→pdf ok={len(ok)} fail={len(fail)}")
if fail:
    print("snapshot: не конвертированы:", "; ".join(fail))
PY
fi

# упорядоченный список всех PDF для содержания и склейки
: > "$TMP/pdf_list.txt"
for p in "${PDFS[@]}"; do printf '%s\n' "$p" >> "$TMP/pdf_list.txt"; done
if [ -n "$MD_SUBDIR" ]; then
    while IFS= read -r p; do printf '%s\n' "$p" >> "$TMP/pdf_list.txt"; done \
        < <(find "$TMP/mdpdf" -name '*.pdf' | LC_ALL=C sort)
fi

# содержание: шапка + таблица с накопленными номерами страниц
python3 - "$TMP" "$REPO_URL" "$COMMIT" "$SNAP_DATE" "$HEADER_FILE" <<'PY'
import re, subprocess, sys, json, os
tmp, url, commit, snap, header_file = sys.argv[1:6]
pdfs = [l.strip() for l in open(os.path.join(tmp, "pdf_list.txt")) if l.strip()]
title_of = {}
mcj = os.path.join(tmp, "md_convert.json")
if os.path.exists(mcj):
    # режим репо: заголовки из сконвертированных md (ok-список)
    for rel in json.load(open(mcj))["ok"]:
        title = None
        for ln in open(os.path.join(tmp, "src", rel), encoding="utf-8",
                       errors="ignore"):
            m = re.match(r"^#\s+(.+)", ln.strip())
            if m:
                title = re.sub(r"\{#[^}]+\}", "", m.group(1)).strip(); break
        title_of[rel[:-3].replace("/", "__") + ".pdf"] = title or rel
# общий случай (режим архива): ищем md-исходник по имени PDF с любым префиксом
for dp, _, fns in os.walk(os.path.join(tmp, "src")):
    for fn in fns:
        if not fn.endswith(".md"):
            continue
        full = os.path.join(dp, fn)
        rel = os.path.relpath(full, os.path.join(tmp, "src"))
        parts = rel[:-3].split("/")
        keys = ["__".join(parts[i:]) + ".pdf" for i in range(len(parts))]
        title = None
        for ln in open(full, encoding="utf-8", errors="ignore"):
            m = re.match(r"^#\s+(.+)", ln.strip())
            if m:
                title = re.sub(r"\{#[^}]+\}", "", m.group(1)).strip(); break
        if title:
            for k in keys:
                title_of.setdefault(k, title)
lines = []
if header_file and os.path.exists(header_file):
    lines.append(open(header_file, encoding="utf-8").read().rstrip())
lines += ["# Содержание", "",
          f"Снимок: {url}, коммит {commit}; собран {snap}.", "",
          "| Раздел | Файл | Страница |", "|---|---|---|"]
page = 1
for pdf in pdfs:
    base = os.path.basename(pdf)
    info = subprocess.run(["pdfinfo", pdf], capture_output=True, text=True).stdout
    n = int(re.search(r"Pages:\s+(\d+)", info).group(1))
    title = title_of.get(base, base[:-4])
    lines.append(f"| {title.replace('|', '/')} | {base} | с. {page} |")
    page += n
open(os.path.join(tmp, "toc.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
PY

pandoc "$TMP/toc.md" -o "$TMP/toc.pdf" --pdf-engine=xelatex \
    -V mainfont="DejaVu Serif" -V geometry:margin=1.5cm -V fontsize=10pt

mapfile -t ALL < "$TMP/pdf_list.txt"
pdfunite "$TMP/toc.pdf" "${ALL[@]}" "$TMP/book.pdf"
cp "$TMP/book.pdf" "$OUT"
PAGES="$(pdfinfo "$OUT" | awk '/^Pages:/{print $2}')"
echo "snapshot: $OUT ($PAGES страниц)"
