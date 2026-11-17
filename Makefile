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
#   make clean                удалить временные файлы (кэши Python)
#   make cleanall             удалить собранные артефакты всех вопросов
#
# .PHONY-цели не параллелятся по вопросам сами: для параллельной сборки
# использовать `make -j4`.

PANDOC   := pandoc
THEORY_YAML := build/defaults-theory.yaml
SLIDES_YAML := build/defaults-slides.yaml
SHORT_YAML  := build/defaults-short.yaml

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
SHORT_SRC  := $(wildcard questions/*/short.md)
NB_SRC     := $(wildcard questions/*/examples.py)

THEORY_PDF := $(THEORY_SRC:.md=.pdf)
SLIDES_PDF := $(SLIDES_SRC:.md=.pdf)
SHORT_PDF  := $(SHORT_SRC:.md=.pdf)
NB_OUT     := $(NB_SRC:.py=.ipynb)

.PHONY: all theory slides short notebooks check clean cleanall new list help venv freeze

# Ноутбуки первыми: они строят картинки, которые вставляют конспект и слайды.
# При обратном порядке PDF собирается раньше своей картинки и показывает
# прошлую версию графика, выглядя при этом свежим (ловится tools/check.sh, п. 3).
all: notebooks theory slides short question-shorts.pdf

theory:    $(THEORY_PDF)
slides:    $(SLIDES_PDF)
short:     $(SHORT_PDF)
notebooks: $(NB_OUT)

# Картинки вставляет PDF, а строит их ноутбук, поэтому ноутбук — настоящая
# предпосылка, а не «идёт раньше в списке целей»: пересобрался ноутбук — могли
# измениться картинки, значит PDF надо пересобрать тоже. Order-only здесь не
# годится (проверено: PDF оставался старше свежих картинок), а голый порядок
# целей ломается при `make -j4`.
NB_DEP = $(if $(wildcard questions/$*/examples.py),questions/$*/examples.ipynb)

# resource-path дополняется каталогом вопроса: в slides.md/theory.md картинки
# пишутся как figures/..., а pandoc работает из корня.
.SECONDEXPANSION:
questions/%/theory.pdf: questions/%/theory.md $(THEORY_YAML) build/preamble-theory.tex $$(NB_DEP)
	@echo "==> theory: $*"
	@$(PANDOC) $< --defaults $(THEORY_YAML) \
		--resource-path=.:assets:questions/$* -o $@

questions/%/slides.pdf: questions/%/slides.md $(SLIDES_YAML) build/preamble-slides.tex $$(NB_DEP)
	@echo "==> slides: $*"
	@$(PANDOC) $< --defaults $(SLIDES_YAML) \
		--resource-path=.:assets:questions/$* -o $@

questions/%/short.pdf: questions/%/short.md $(SHORT_YAML) build/preamble-theory.tex build/preamble-short.tex
	@echo "==> short: $*"
	@$(PANDOC) $< --defaults $(SHORT_YAML) \
		--resource-path=.:assets:questions/$* -o $@

# Единый PDF со всеми краткими планами: склейка short.md вопросов в том
# порядке, в котором они идут в программе (каталоги zero-padded).
question-shorts.pdf: $(SHORT_SRC) $(SHORT_YAML) build/preamble-theory.tex build/preamble-short.tex build/preamble-shortscover.tex tools/shorts.sh
	@echo "==> question-shorts"
	@tmp=$$(mktemp /tmp/question-shorts-XXXX.md); \
	bash tools/shorts.sh > "$$tmp"; \
	$(PANDOC) "$$tmp" --defaults $(SHORT_YAML) --toc --toc-depth=1 \
		--metadata title="Экзаменационные билеты по специальности 1.2.2" \
		--metadata subtitle="Краткие планы ответов" \
		--metadata toc-title="Вопросы" \
		--include-in-header=build/preamble-shortscover.tex \
		-o $@; \
	rm -f "$$tmp"

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
	[ -f "$$dir/examples.py" ] && $(MAKE) --no-print-directory "$$dir/examples.ipynb"; \
	for t in theory slides short; do \
	  [ -f "$$dir/$$t.md" ] && $(MAKE) --no-print-directory "$$dir/$$t.pdf"; \
	done; \
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
	@rm -rf tools/__pycache__ .ruff_cache
	@echo "временные файлы удалены"

# Полная очистка — редкая операция (пересборка всего с нуля):
# артефакты закоммичены как доставляемые, их удаление затрагивает все вопросы.
cleanall:
	@rm -f questions/*/theory.pdf questions/*/slides.pdf questions/*/short.pdf questions/*/examples.ipynb question-shorts.pdf
	@echo "собранные артефакты удалены"

help:
	@sed -n '1,20p' Makefile
