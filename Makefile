# Сборка материалов экзамена. Запускать ИЗ КОРНЯ проекта: pandoc-настройки
# в build/ ссылаются на пути относительно корня.
#
#   make                      собрать всё, что имеет исходники
#   make q11                  собрать все артефакты вопроса 11
#   make new Q=11             развернуть шаблоны в каталоге вопроса 11
#   make theory | slides | notebooks
#   make check                проверки согласованности
#   make clean                удалить собранные артефакты
#
# .PHONY-цели не параллелятся по вопросам сами: для параллельной сборки
# использовать `make -j4`.

PANDOC   := pandoc
THEORY_YAML := build/defaults-theory.yaml
SLIDES_YAML := build/defaults-slides.yaml

THEORY_SRC := $(wildcard questions/*/theory.md)
SLIDES_SRC := $(wildcard questions/*/slides.md)
NB_SRC     := $(wildcard questions/*/examples.py)

THEORY_PDF := $(THEORY_SRC:.md=.pdf)
SLIDES_PDF := $(SLIDES_SRC:.md=.pdf)
NB_OUT     := $(NB_SRC:.py=.ipynb)

.PHONY: all theory slides notebooks check clean new list help

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
	@cd questions/$* && jupytext --to ipynb --execute -o examples.ipynb examples.py

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
