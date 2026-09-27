# Сборка материалов экзамена. Запускать ИЗ КОРНЯ проекта: pandoc-настройки
# в build/ ссылаются на пути относительно корня.
#
#   make                      собрать всё, что имеет исходники
#   make q11                  собрать все артефакты вопроса 11
#   make new Q=11             развернуть шаблоны в каталоге вопроса 11
#   make theory | slides | notebooks
#   make check                проверки согласованности
#   make venv                 создать .venv и поставить requirements.txt
#   make freeze               зафиксировать версии в requirements-lock.txt
#   make clean                удалить собранные артефакты
#
# .PHONY-цели не параллелятся по вопросам сами: для параллельной сборки
# использовать `make -j4`.

PANDOC   := pandoc
THEORY_YAML := build/defaults-theory.yaml
SLIDES_YAML := build/defaults-slides.yaml

# Питон берём из локального venv, если он есть, иначе системный: цели должны
# работать и на чистой машине, где venv ещё не создан (там упадут с внятным
# "not found", а не молча возьмут другой интерпретатор).
VENV     := .venv
VENV_BIN := $(CURDIR)/$(VENV)/bin
PY       := $(if $(wildcard $(VENV_BIN)/python),$(VENV_BIN)/python,python3)
PIP      := $(if $(wildcard $(VENV_BIN)/pip),$(VENV_BIN)/pip,pip3)
JUPYTEXT := $(if $(wildcard $(VENV_BIN)/jupytext),$(VENV_BIN)/jupytext,jupytext)

THEORY_SRC := $(wildcard questions/*/theory.md)
SLIDES_SRC := $(wildcard questions/*/slides.md)
NB_SRC     := $(wildcard questions/*/examples.py)

THEORY_PDF := $(THEORY_SRC:.md=.pdf)
SLIDES_PDF := $(SLIDES_SRC:.md=.pdf)
NB_OUT     := $(NB_SRC:.py=.ipynb)

.PHONY: all theory slides notebooks check clean new list help venv freeze

all: theory slides notebooks

theory:    $(THEORY_PDF)
slides:    $(SLIDES_PDF)
notebooks: $(NB_OUT)

# resource-path дополняется каталогом вопроса: в slides.md/theory.md картинки
# пишутся как figures/..., а pandoc работает из корня.
questions/%/theory.pdf: questions/%/theory.md $(THEORY_YAML) build/preamble-theory.tex
	@echo "==> theory: $*"
	@$(PANDOC) $< --defaults $(THEORY_YAML) \
		--resource-path=.:assets:questions/$* -o $@

questions/%/slides.pdf: questions/%/slides.md $(SLIDES_YAML) build/preamble-slides.tex
	@echo "==> slides: $*"
	@$(PANDOC) $< --defaults $(SLIDES_YAML) \
		--resource-path=.:assets:questions/$* -o $@

# Ноутбук — производный артефакт: источник в git — examples.py (jupytext percent).
# Исполняется при сборке, чтобы в .ipynb лежали настоящие выводы, а не пустые
# ячейки; при падении ячейки сборка падает — это гейт, а не помеха.
questions/%/examples.ipynb: questions/%/examples.py
	@echo "==> notebook: $*"
	@cd questions/$* && $(JUPYTEXT) --to ipynb --execute -o examples.ipynb examples.py

# Собрать всё по одному вопросу: make q11
q%:
	@dir=$$(ls -d questions/$**/ 2>/dev/null | head -1); \
	if [ -z "$$dir" ]; then echo "нет каталога для вопроса $*"; exit 1; fi; \
	dir=$${dir%/}; \
	for t in theory slides; do \
	  [ -f "$$dir/$$t.md" ] && $(MAKE) --no-print-directory "$$dir/$$t.pdf"; \
	done; \
	[ -f "$$dir/examples.py" ] && $(MAKE) --no-print-directory "$$dir/examples.ipynb"; \
	true

venv:
	@test -d $(VENV) || python3 -m venv $(VENV)
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt
	@# Без зарегистрированного kernel'а jupytext --execute не находит
	@# интерпретатор venv и падает ещё до запуска ячеек.
	@$(PY) -m ipykernel install --user --name math-exam --display-name "math-exam (.venv)"
	@echo "готово: $(VENV)"

freeze:
	@$(PIP) freeze > requirements-lock.txt
	@echo "записано: requirements-lock.txt"

new:
	@test -n "$(Q)" || { echo "использование: make new Q=11"; exit 1; }
	@bash tools/new-question.sh $(Q)

check:
	@bash tools/check.sh

list:
	@bash tools/status.sh

clean:
	@rm -f questions/*/theory.pdf questions/*/slides.pdf questions/*/examples.ipynb
	@echo "собранные артефакты удалены"

help:
	@sed -n '1,20p' Makefile
