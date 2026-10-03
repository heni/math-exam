# %% [markdown]
# # Вопрос NN. ЗАГОЛОВОК — численные примеры
#
# Файл ведётся в формате jupytext (percent). Парный `.ipynb` получается
# командой `make questions/NN-slug/examples.ipynb`; в git идёт **этот** файл,
# ноутбук — производный артефакт.
#
# Правила: каждый пример отвечает на вопрос из `theory.md`; seed задан явно;
# число, попавшее в `theory.md` или `slides.md`, печатается здесь.

# %%
import numpy as np
import matplotlib.pyplot as plt

SEED = 20260927
rng = np.random.default_rng(SEED)

FIGDIR = "figures"

# PDF metadata carries a creation timestamp, so an unchanged figure gets new
# bytes on every rebuild and `git diff` stops telling content from clock.
SAVE_KW = {"metadata": {"CreationDate": None}}

# Golden pins: the seed and the expected outputs are fixed constants. Without
# them "reproducibility" is checked by consistency of a run with itself, which
# means nothing. A mismatch fails the notebook build, so it works as a gate.
GOLDEN = {
    # "имя_величины": ожидаемое_значение,
}
TOL = 1e-3


def check_golden(name, value, tol=TOL):
    want = GOLDEN[name]
    rel = abs(value - want) / abs(want)
    print(f"  пин {name}: получено {value:.12g}, ожидалось {want:.12g}")
    assert rel <= tol, f"golden-пин {name} не сошёлся: {value} vs {want}"

# %% [markdown]
# ## Пример 1. НАЗВАНИЕ
#
# **Вопрос:** что именно проверяем и какого результата ждём (предсказание
# формулируется ДО прогона).

# %%
# ... вычисление ...

# %% [markdown]
# **Вывод:** совпало / не совпало с предсказанием и почему. Каждое число вывода
# сверить с фактическим выводом ячейки, а не с соседней: подпись, измеренная
# одним способом и названная другим, — самая частая ошибка проекта.
# Не сбывшееся предсказание не переписывается — оно остаётся и объясняется.

# %%
fig, ax = plt.subplots(figsize=(6, 3.6), layout="constrained")
# ... график ...
# mathtext, не LaTeX: \geq вместо \ge (docs/build.md, «Разметка формул»)
ax.set_xlabel("...")
ax.set_ylabel("...")
fig.savefig(f"{FIGDIR}/fig-01.pdf", **SAVE_KW)
