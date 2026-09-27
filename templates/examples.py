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

# %% [markdown]
# ## Пример 1. НАЗВАНИЕ
#
# **Вопрос:** что именно проверяем и какого результата ждём (предсказание
# формулируется ДО прогона).

# %%
# ... вычисление ...

# %% [markdown]
# **Вывод:** совпало / не совпало с предсказанием и почему.
# Не сбывшееся предсказание не переписывается — оно остаётся и объясняется.

# %%
fig, ax = plt.subplots(figsize=(6, 3.6), layout="constrained")
# ... график ...
ax.set_xlabel("...")
ax.set_ylabel("...")
fig.savefig(f"{FIGDIR}/fig-01.pdf")
