# %% [markdown]
# # Вопрос 14. Интервальное оценивание. Проверка статистических гипотез — численные примеры
#
# Файл ведётся в формате jupytext (percent). Парный `.ipynb` получается
# командой `make questions/14-interval-estimation-tests/examples.ipynb`; в git
# идёт **этот** файл, ноутбук — производный артефакт.
#
# **Сквозной пример** — тот же датчик, что в вопросе 12: выборка независимых
# измерений $\xi_k = a + \text{шум}$. Пять примеров отвечают на пять вопросов
# конспекта:
#
# | № | Вопрос из theory.md | Закон шума |
# |---|---|---|
# | 1 | четыре точных интервала нормальной модели (теорема о четырёх интервалах) | нормальный |
# | 2 | что ломается без нормальности: покрытие z- и t-интервалов | пять законов из вопроса 12 |
# | 3 | функция мощности u- и t-критерия; число наблюдений для мощности | нормальный |
# | 4 | лемма Неймана — Пирсона: оптимальный критерий против «интуитивного» | нормальный |
# | 5 | критерии согласия как проверка допущения нормальности | нормальный, Парето, Коши |
#
# Каждый пример формулирует предсказание **до** прогона; предсказание
# остаётся в тексте, даже если не сбылось.

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

SEED = 20261002

# Separate generators derived from one seed: a shared stream would couple the
# examples, and changing the ensemble size of example 2 would shift every
# golden pin downstream.
rng_ex1, rng_ex2, rng_ex3, rng_ex4, rng_ex5 = (
    np.random.default_rng([SEED, k]) for k in range(5)
)

FIGDIR = "figures"

# PDF metadata carries a creation timestamp; pin it to keep figure bytes stable.
SAVE_KW = {"metadata": {"CreationDate": None}}

# Golden pins freeze seed AND expected output; a mismatch aborts the build.
# Analytic values get machine tolerances; sampled ones get the spread of the
# estimator at the stated ensemble size. Sampled pins are added from a real
# run on a second pass.
GOLDEN = {
    "t_quant_24": (2.0638985616280205, 1e-12),
    "chi2_quant_975_24": (39.36407702660391, 1e-12),
    "chi2_quant_025_24": (12.401150217444435, 1e-12),
    "power_u_d05_N30": (0.7819079987319799, 1e-12),
    "power_t_d05_N30": (0.7539647157437906, 1e-12),
    "local_limit_h2": (0.5160052739761748, 1e-12),
    "N_min_u": (43.0, 1e-12),
    "N_min_t": (44.0, 1e-12),
    "power_ump_plus": (0.4745986612454448, 1e-12),
    "power_twosided_plus": (0.35260808244470254, 1e-12),
    "power_ump_minus": (0.0006276833271278148, 1e-12),
    "np_threshold": (0.5201483878755574, 0.005),  # tolerance covers the scan grid step
    "chi2_crit_7": (14.067140449340169, 1e-12),
    "chi2_crit_5": (11.070497693516351, 1e-12),
    "kolm_lambda_095": (1.3580986393225505, 1e-12),
    # sampled (20 000 прогонов; допуск ≈ 4σ биномиального разброса)
    "cover_z_normal": (0.9486, 0.01),
    "cover_t_normal": (0.9494, 0.01),
    "cover_z_uniform": (0.9520, 0.01),
    "cover_t_uniform": (0.9513, 0.01),
    "cover_z_bernoulli": (0.9435, 0.01),
    "cover_t_bernoulli": (0.7843, 0.01),
    "cover_z_pareto": (0.9618, 0.01),
    "cover_t_pareto": (0.8731, 0.01),
    "cover_z_cauchy": (0.2163, 0.01),
    "cover_t_cauchy": (0.9782, 0.01),
    "pow_u_normal_b": (0.7823, 0.01),
    "pow_t_normal_b": (0.7540, 0.01),
    "pow_u_uniform_b": (0.7820, 0.01),
    "pow_t_uniform_b": (0.7512, 0.01),
    "pow_u_bernoulli_b": (0.7864, 0.01),
    "pow_t_bernoulli_b": (1.0, 0.01),
    "pow_u_pareto_b": (0.8105, 0.01),
    "pow_t_pareto_b": (0.9756, 0.01),
    "pow_u_cauchy_b": (0.8212, 0.01),
    "pow_t_cauchy_b": (0.0747, 0.01),
    "u_cauchy_null": (0.7812313154639832, 1e-9),
    # sampled (2000 выборок; для уровня допуск 3σ при p = 0.05)
    "level_chi2_fix": (0.0490, 0.015),
    "level_chi2_fit": (0.0535, 0.015),
    "level_ks": (0.0465, 0.015),
    "pow_par_chi2_fix": (1.0, 0.01),
    "pow_par_chi2_fit": (1.0, 0.01),
    "pow_par_ks": (1.0, 0.01),
    "pow_cau_chi2_fix": (0.9990, 0.01),
    "pow_cau_chi2_fit": (1.0, 0.01),
    "pow_cau_ks": (1.0, 0.01),
    # одна выборка датчика: значения зафиксированы семенем
    "pnorm_chi2_fix": (0.8069685035136452, 1e-9),
    "pnorm_chi2_fit": (0.7220014223914515, 1e-9),
    "pnorm_ks": (0.7819025042618662, 1e-9),
}


def check_golden(name, value):
    want, tol = GOLDEN[name]
    rel = abs(value - want) / max(abs(want), 1e-300)
    print(f"  пин {name}: получено {value:.12g}, ожидалось {want:.12g} (допуск {tol:g})")
    assert rel <= tol, f"golden-пин {name} не сошёлся: {value} vs {want}"


# %% [markdown]
# ## Пример 1. Четыре интервала нормальной модели
#
# Выборка $N = 25$ из $\mathcal N(2{,}0,\; 1{,}5^2)$, уровень доверия
# $\gamma = 0{,}95$.
#
# **Предсказание.** (1) Интервал для $a$ при неизвестном $\sigma$ (t-интервал)
# шире, чем при известном (z-интервал) — плата за оценку $\sigma$ заметна при
# $N = 25$ (множитель $t_{0{,}975;\,24} = 2{,}064$ против $z_{0{,}975} =
# 1{,}960$). (2) Интервал для $\sigma^2$ при неизвестном $a$ асимметричен
# вокруг $s_N^2$ (распределение $\chi^2$ скошено). (3) Квантили scipy совпадут
# с табличными значениями [06]: $t_{0{,}975;\,24} = 2{,}064$,
# $\chi^2_{0{,}975;\,24} = 39{,}36$, $\chi^2_{0{,}025;\,24} = 12{,}40$.

# %%
N1 = 25
A_TRUE, SIGMA_TRUE = 2.0, 1.5
GAMMA = 0.95
B_PLUS, B_MINUS = (1 + GAMMA) / 2, (1 - GAMMA) / 2

sample1 = rng_ex1.normal(A_TRUE, SIGMA_TRUE, N1)
mean1 = sample1.mean()
var_biased1 = sample1.var(ddof=0)   # \widehat\sigma_N^2 (1/N)
var_unb1 = sample1.var(ddof=1)      # s_N^2 (1/(N-1))

z_bp = stats.norm.ppf(B_PLUS)
t_bp = stats.t.ppf(B_PLUS, N1 - 1)
chi_bp = stats.chi2.ppf(B_PLUS, N1 - 1)
chi_bm = stats.chi2.ppf(B_MINUS, N1 - 1)

ci_mean_known = (mean1 - z_bp * SIGMA_TRUE / np.sqrt(N1),
                 mean1 + z_bp * SIGMA_TRUE / np.sqrt(N1))
ci_mean_unknown = (mean1 - t_bp * np.sqrt(var_unb1 / N1),
                   mean1 + t_bp * np.sqrt(var_unb1 / N1))
var_a_known = ((sample1 - A_TRUE) ** 2).mean()
chi_bp_N = stats.chi2.ppf(B_PLUS, N1)
chi_bm_N = stats.chi2.ppf(B_MINUS, N1)
ci_var_known = (N1 * var_a_known / chi_bp_N, N1 * var_a_known / chi_bm_N)
ci_var_unknown = ((N1 - 1) * var_unb1 / chi_bp, (N1 - 1) * var_unb1 / chi_bm)

print(f"выборка: mean = {mean1:.4f}, s^2 = {var_unb1:.4f}")
print(f"1) a, sigma известно:  ({ci_mean_known[0]:.4f}; {ci_mean_known[1]:.4f})")
print(f"2) a, sigma неизвестно:({ci_mean_unknown[0]:.4f}; {ci_mean_unknown[1]:.4f})")
print(f"3) sigma^2, a известно: ({ci_var_known[0]:.4f}; {ci_var_known[1]:.4f})")
print(f"4) sigma^2, a неизвестно: ({ci_var_unknown[0]:.4f}; {ci_var_unknown[1]:.4f})")
print("таблица [06] против scipy:")
print(f"  t_0.975;24   = {t_bp:.4f}  (таблица: 2.064)")
print(f"  chi2_0.975;24= {chi_bp:.4f}  (таблица: 39.36)")
print(f"  chi2_0.025;24= {chi_bm:.4f}  (таблица: 12.40)")
check_golden("t_quant_24", t_bp)
check_golden("chi2_quant_975_24", chi_bp)
check_golden("chi2_quant_025_24", chi_bm)

# %% [markdown]
# **Вывод.** Предсказание сбылось по всем пунктам. Квантили scipy совпали с
# таблицей [06] в третьем знаке: $2{,}0639$ против $2{,}064$;
# $39{,}3641$ против $39{,}36$; $12{,}4012$ против $12{,}40$. t-интервал для
# $a$ равен $(1{,}0216;\ 2{,}3971)$ и шире z-интервала $(1{,}1214;\ 2{,}2974)$
# — плата за оценку $\sigma$ при $N = 25$ заметна. Интервалы для $\sigma^2$
# асимметричны вокруг $s_N^2 = 2{,}7762$: $(1{,}6926;\ 5{,}3728)$ — правая
# граница дальше от точки, чем левая (распределение $\chi^2$ скошено вправо).

# %% [markdown]
# ## Пример 2. Покрытие z- и t-интервалов при чужом законе шума
#
# Теорема о нормальной выборке лежит в основе t-интервала; здесь шум тот же,
# что в примере 2 вопроса 12 (пять законов с нулевым средним и единичной
# дисперсией, где они есть): нормальный, равномерный на $[-\sqrt3, \sqrt3]$,
# центрированный бернуллиевский с $p = 0{,}05$ (единичная дисперсия после
# нормировки), Парето с $\alpha = 3$ (конечная дисперсия, тяжёлый правый
# хвост), Коши (среднего и дисперсии нет). $N = 30$, $\gamma = 0{,}95$,
# $20\,000$ повторений; считается доля интервалов, накрывших истинное $a = 0$.
#
# **Предсказание.** При нормальном шуме оба интервала покрывают с долей
# $\approx 0{,}95$. Равномерный шум — тоже почти $0{,}95$ (лёгкие хвосты).
# Бернуллиевский с $p = 0{,}05$: сильная асимметрия, покрытие заметно ниже
# 0.95 (см. урок примера 3 вопроса 12: асимметрия бьёт по односторонним
# квантилям). Парето и Коши: покрытие падает катастрофически, t-интервал —
# сильнее z-интервала (усиление хвостов через $s_N$).

# %%
N2, TRIALS2 = 30, 20_000


def make_noise(name, size, rng):
    if name == "normal":
        return rng.normal(0, 1, size)
    if name == "uniform":
        return rng.uniform(-np.sqrt(3), np.sqrt(3), size)
    if name == "bernoulli":
        # centered Bernoulli with Var = 1: mass 1-p at -p/s, p at (1-p)/s,
        # s = sqrt(p(1-p))
        p = 0.05
        s = np.sqrt(p * (1 - p))
        return np.where(rng.random(size) < p, (1 - p) / s, -p / s)
    if name == "pareto":
        # Pareto with alpha = 3 (finite variance), centered, unit variance:
        # Var of Pareto(a) = a / ((a-1)^2 (a-2)) = 3/4; scale sqrt(4/3)
        a = 3.0
        scale = np.sqrt(4.0 / 3.0)
        return (rng.pareto(a, size) - 1.0 / (a - 1)) * scale
    if name == "cauchy":
        return rng.standard_cauchy(size)
    raise ValueError(name)


LAWS2 = ["normal", "uniform", "bernoulli", "pareto", "cauchy"]
z_bp2 = stats.norm.ppf(B_PLUS)
t_bp2 = stats.t.ppf(B_PLUS, N2 - 1)

print(f"{'закон':<11}{'z-интервал':>12}{'t-интервал':>12}")
cover_z, cover_t = {}, {}
for law in LAWS2:
    noise = make_noise(law, (TRIALS2, N2), rng_ex2)
    means = noise.mean(axis=1)
    half_z = z_bp2 / np.sqrt(N2)
    s_unb = noise.std(axis=1, ddof=1)
    half_t = t_bp2 * s_unb / np.sqrt(N2)
    cover_z[law] = float((np.abs(means) <= half_z).mean())
    cover_t[law] = float((np.abs(means) <= half_t).mean())
    print(f"{law:<11}{cover_z[law]:>12.4f}{cover_t[law]:>12.4f}")

fig, ax = plt.subplots(figsize=(7, 3.5))
x = np.arange(len(LAWS2))
ax.bar(x - 0.2, [cover_z[k] for k in LAWS2], 0.4, label="z-интервал")
ax.bar(x + 0.2, [cover_t[k] for k in LAWS2], 0.4, label="t-интервал")
ax.axhline(GAMMA, color="black", lw=1, ls="--", label=r"$\gamma = 0.95$")
ax.set_xticks(x, LAWS2)
ax.set_ylabel("доля покрытий")
ax.set_ylim(0, 1.05)
ax.legend()
fig.tight_layout()
fig.savefig(f"{FIGDIR}/fig-01.pdf", **SAVE_KW)
plt.close(fig)
print("фигура: figures/fig-01.pdf")
check_golden("cover_z_normal", cover_z["normal"])
check_golden("cover_t_normal", cover_t["normal"])
check_golden("cover_z_uniform", cover_z["uniform"])
check_golden("cover_t_uniform", cover_t["uniform"])
check_golden("cover_z_bernoulli", cover_z["bernoulli"])
check_golden("cover_t_bernoulli", cover_t["bernoulli"])
check_golden("cover_z_pareto", cover_z["pareto"])
check_golden("cover_t_pareto", cover_t["pareto"])
check_golden("cover_z_cauchy", cover_z["cauchy"])
check_golden("cover_t_cauchy", cover_t["cauchy"])

# %% [markdown]
# **Вывод.** Предсказание сбылось для нормального и равномерного шума и не
# сбылось в тяжёлых хвостах — картина там оказалась тоньше. Точный t-интервал
# держит уровень при нормальном шуме ($0{,}9494$ против заявленных $0{,}95$)
# и почти держит при равномерном ($0{,}9513$). Дальше по законам: бернуллиевский
# с $p=0{,}05$ — z почти держит ($0{,}9435$), t падает до $0{,}7843$
# (дискретность плюс асимметрия ломают калибровку Стьюдента); Парето
# ($\alpha=3$) — z завышен ($0{,}9618$), t занижен ($0{,}8731$); Коши —
# z недопокрывает катастрофически ($0{,}2163$), а t **перекрывает**
# ($0{,}9782$): выборочное $s_N$ взрывается тяжёлыми хвостами, и интервал
# раздувается. Предсказанное «покрытие t падает сильнее z» при Коши неверно
# в знаке: у z интервал схлопывается в точку $\overline X_N$, который сам
# неустойчив. Урок: без нормальности ломается калибровка уровня, и направление
# искажения заранее неизвестно — это повторяется в примере 3 (б) и разобрано
# в одноимённом разделе конспекта.

# %% [markdown]
# ## Пример 3. Функции мощности u- и t-критерия; объём выборки для заданной мощности
#
# Нормальная модель с $\sigma = 1$: $H_0: a = a_0$ против двусторонней
# альтернативы, уровень $\alpha = 0{,}05$. u-критерий (z-критерий) знает
# $\sigma$; t-критерий оценивает её через $s_N$. Формула мощности u-критерия —
# (eq:power-u); для t-критерия статистика при альтернативе $a$ имеет
# распределение Стьюдента с $N-1$ степенями свободы и параметром смещения
# $(a-a_0)\sqrt N$ (`scipy.stats.nct`).
#
# **Предсказание.** (1) При $a = a_0$ обе мощности равны $\alpha$; при
# $a \ne a_0$ t-критерий всюду слабее u-критерия — плата за оценку $\sigma$.
# (2) Для локальной альтернативы $a_0 + h/\sqrt N$ мощность u-критерия от
# $N$ не зависит (сдвиг измерен в единицах $\sigma/\sqrt N$), а мощность
# t-критерия сходится к тому же пределу снизу. (3) Минимальный объём для
# мощности $0{,}9$ при сдвиге $0{,}5\sigma$: u-критерию достаточно
# $N \approx 43$, t-критерию — чуть больше. (4) При чужом законе шума (тот
# же набор, что в примере 2) эмпирическая мощность падает: равномерный шум —
# немного выше нормального, бернуллиевский с $p=0{,}05$ — заметно ниже
# (асимметрия), Парето ($\alpha=3$) — ещё ниже, Коши — до уровня $\alpha$:
# без конечного среднего выборочное среднее не несёт информации о сдвиге.
# Оговорка: u-критерий в блоке (б) использует $\sigma=1$, что для Парето
# и Коши неверно — он там некалиброван, сравнивать его мощность с t некорректно.

# %%
ALPHA3 = 0.05
Z23 = stats.norm.ppf(1 - ALPHA3 / 2)   # z_{1-α/2} = 1.96


def power_u(delta, N):
    # (eq:power-u), delta = (a - a0)/σ
    return (1 - stats.norm.cdf(Z23 - delta * np.sqrt(N))
            + stats.norm.cdf(-Z23 - delta * np.sqrt(N)))


def power_t(delta, N):
    tcrit = stats.t.ppf(1 - ALPHA3 / 2, N - 1)
    nc = delta * np.sqrt(N)
    return stats.nct.sf(tcrit, N - 1, nc) + stats.nct.cdf(-tcrit, N - 1, nc)


check_golden("power_u_d05_N30", power_u(0.5, 30))
check_golden("power_t_d05_N30", power_t(0.5, 30))

# локальная альтернатива a0 + hσ/√N
H_LOC = 2.0
limit_local = 1 - stats.norm.cdf(Z23 - H_LOC) + stats.norm.cdf(-Z23 - H_LOC)
check_golden("local_limit_h2", limit_local)
print(f"предел мощности при h = {H_LOC:g}: {limit_local:.6f}")
for N in (25, 100, 400, 1600):
    print(f"  N = {N:>5}:  u = {power_u(H_LOC / np.sqrt(N), N):.6f}   "
          f"t = {power_t(H_LOC / np.sqrt(N), N):.6f}")


def min_sample_size(power_fn, target=0.9, delta=0.5):
    for N in range(2, 2000):
        if power_fn(delta, N) >= target:
            return N


N_U = min_sample_size(power_u)
N_T = min_sample_size(power_t)
check_golden("N_min_u", float(N_U))
check_golden("N_min_t", float(N_T))
print(f"минимальный N для мощности 0.9 при сдвиге 0.5σ: u-критерий {N_U}, "
      f"t-критерий {N_T}")

deltas = np.linspace(-1.2, 1.2, 241)
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.plot(deltas, [power_u(d, 25) for d in deltas], label="u-критерий, $N = 25$")
ax.plot(deltas, [power_t(d, 25) for d in deltas], ls="--",
        label="t-критерий, $N = 25$")
ax.axhline(ALPHA3, color="black", lw=1, ls=":", label=r"$\alpha = 0.05$")
ax.axvline(0, color="gray", lw=0.8)
ax.set_xlabel(r"сдвиг $\Delta = (a - a_0)/\sigma$")
ax.set_ylabel("мощность")
ax.set_ylim(0, 1.02)
ax.legend()
fig.tight_layout()
fig.savefig(f"{FIGDIR}/fig-02.pdf", **SAVE_KW)
plt.close(fig)
print("фигура: figures/fig-02.pdf")

# %% [markdown]
# **Вывод.** Предсказание сбылось по всем пунктам. Мощность u-критерия при
# локальной альтернативе от $N$ не зависит в точности ($0{,}516005$ при всех
# $N$ — сдвиг измерен в единицах $\sigma/\sqrt N$), а мощность t-критерия
# подтягивается к тому же пределу снизу: $0{,}4840$ при $N=25$, $0{,}5083$
# при $N=100$, $0{,}5155$ при $N=1600$. Минимальные объёмы для мощности
# $0{,}9$ при сдвиге $0{,}5\sigma$: $N = 43$ (u) и $N = 44$ (t) — плата за
# оценку $\sigma$ одно наблюдение. Это и есть численный ответ на вопрос
# «сколько нужно измерений», рядом с асимптотическим ответом приложения 1
# вопроса 12.

# %% [markdown]
# ### (б) Мощность при чужом законе шума
#
# Тот же датчик и те же пять законов шума, что в примере 2: альтернатива —
# сдвиг среднего на $0{,}5$ при $N = 30$, $20\,000$ прогонов.

# %%
N3B, TRIALS3 = 30, 20_000
SHIFT3 = 0.5
T_CRIT_3B = stats.t.ppf(1 - ALPHA3 / 2, N3B - 1)

print(f"эмпирическая мощность при сдвиге {SHIFT3}σ, N = {N3B} "
      f"({TRIALS3} прогонов):")
print(f"{'закон':<11}{'u (σ = 1)':>12}{'t':>12}")
pow3_u, pow3_t = {}, {}
for law in LAWS2:
    noise = make_noise(law, (TRIALS3, N3B), rng_ex3) + SHIFT3
    means = noise.mean(axis=1)
    s_unb = noise.std(axis=1, ddof=1)
    pow3_u[law] = float((np.abs(means) * np.sqrt(N3B) >= Z23).mean())
    pow3_t[law] = float((np.abs(means) / s_unb * np.sqrt(N3B) >= T_CRIT_3B).mean())
    print(f"{law:<11}{pow3_u[law]:>12.4f}{pow3_t[law]:>12.4f}")

# фактический уровень u-критерия при коши-шуме, аналитически
u_cauchy_null = 2 * stats.cauchy.sf(Z23 / np.sqrt(N3B))
check_golden("u_cauchy_null", u_cauchy_null)
print(f"фактический уровень u-критерия при коши-шуме: {u_cauchy_null:.4f}")

fig, ax = plt.subplots(figsize=(7, 3.5))
x = np.arange(len(LAWS2))
ax.bar(x - 0.2, [pow3_u[k] for k in LAWS2], 0.4, label="u-критерий")
ax.bar(x + 0.2, [pow3_t[k] for k in LAWS2], 0.4, label="t-критерий")
ax.axhline(ALPHA3, color="black", lw=1, ls=":", label=r"$\alpha = 0.05$")
ax.set_xticks(x, LAWS2)
ax.set_ylabel("мощность")
ax.set_ylim(0, 1.05)
ax.legend()
fig.tight_layout()
fig.savefig(f"{FIGDIR}/fig-03.pdf", **SAVE_KW)
plt.close(fig)
print("фигура: figures/fig-03.pdf")
check_golden("pow_u_normal_b", pow3_u["normal"])
check_golden("pow_t_normal_b", pow3_t["normal"])
check_golden("pow_u_uniform_b", pow3_u["uniform"])
check_golden("pow_t_uniform_b", pow3_t["uniform"])
check_golden("pow_u_bernoulli_b", pow3_u["bernoulli"])
check_golden("pow_t_bernoulli_b", pow3_t["bernoulli"])
check_golden("pow_u_pareto_b", pow3_u["pareto"])
check_golden("pow_t_pareto_b", pow3_t["pareto"])
check_golden("pow_u_cauchy_b", pow3_u["cauchy"])
check_golden("pow_t_cauchy_b", pow3_t["cauchy"])

# %% [markdown]
# **Вывод.** Предсказание о порядке мощностей не сбылось, и причина содержательна:
# «мощность» без калибровки уровня неинформативна. Фактический уровень
# t-критерия при чужом законе — это $1 - {}$покрытие его интервала из примера 2:
# бернуллиевский $0{,}216$, Парето $0{,}127$, Коши $0{,}022$. Поэтому мощность
# $1{,}0000$ при бернуллиевском шуме означает лишь, что критическое значение
# в пересчёте на раздутое редкими выбросами $s_N$ достигается почти всегда —
# критерий «всесилен», потому что развален уже при $H_0$. У Коши наоборот:
# $s_N$ взрывается настолько, что сдвиг $0{,}5\sigma$ тонет, и мощность
# $0{,}0747$ лишь немного выше фактического уровня $0{,}022$ — сдвиг почти
# не детектируется. Закономерность одна: при чужом законе калибровка ломается,
# и мощность изолированно смотреть нельзя. u-колонка подтверждает ту же мысль
# сильнее: при Коши $\sigma = 1$ ложна, и «критерий» отклоняет верную $H_0$
# уже в $78\,\%$ случаев (точное значение $0{,}7812$ напечатано выше пином
# u_cauchy_null; выборочное среднее Коши — снова Коши), так что
# число $0{,}8212$ при сдвиге $0{,}5$ — не мощность, а артефакт.

# %% [markdown]
# ## Пример 4. Лемма Неймана — Пирсона: оптимальный критерий и его границы
#
# Выборка $N = 10$ из $\mathcal N(\theta, 1)$; $H_0: \theta = 0$ против
# $H_1: \theta = \delta$ с $\delta = 0{,}5$; уровень $\varepsilon = 0{,}05$.
# Область $S_\lambda = \{x : p_1(x) - \lambda p_0(x) \ge 0\}$ есть
# $\overline X_N \ge \frac{\delta}{2} + \frac{\ln\lambda}{N\delta}$ —
# подуровенное множество отношения правдоподобия, которое для нормального
# сдвига совпадает с «интуитивным» порогом по выборочному среднему.
#
# **Предсказание.** (1) Сканирование порогов $c$ в семействе
# $\{\overline X_N \ge c\}$ даст максимум мощности при уровне не выше
# $\varepsilon$ ровно в $c = z_{1-\varepsilon}/\sqrt N$ — лемма
# подтверждается численно. (2) Односторонний критерий с этим порогом —
# р.н.м.к. для $\theta > 0$: при $\theta = +0{,}5$ его мощность $0{,}4746$
# против $0{,}3526$ у двустороннего критерия (двойственность по интервалу).
# (3) Симметрия двустороннего критерия оплачивается мощностью при
# односторонней альтернативе — и наоборот: при $\theta = -0{,}5$ верхний
# односторонний критерий имеет мощность $0{,}0006$. Р.н.м.к. для
# двусторонней альтернативы не существует.

# %%
N4 = 10
DELTA4 = 0.5
EPS4 = 0.05
Z95 = stats.norm.ppf(1 - EPS4)          # z_{0.95} = 1.645
C_NP = Z95 / np.sqrt(N4)


def power_upper(theta):
    return 1 - stats.norm.cdf(Z95 - theta * np.sqrt(N4))


def power_two_sided(theta):
    z2 = stats.norm.ppf(1 - EPS4 / 2)
    return (1 - stats.norm.cdf(z2 - theta * np.sqrt(N4))
            + stats.norm.cdf(-z2 - theta * np.sqrt(N4)))


check_golden("power_ump_plus", power_upper(DELTA4))
check_golden("power_twosided_plus", power_two_sided(DELTA4))
check_golden("power_ump_minus", power_upper(-DELTA4))

# скан порогов c: размер и мощность семейства {X̄ ≥ c}
grid = np.linspace(0.25, 0.8, 551)
sizes = 1 - stats.norm.cdf(grid * np.sqrt(N4))
powers4 = 1 - stats.norm.cdf((grid - DELTA4) * np.sqrt(N4))
ok = sizes <= EPS4
c_opt = float(grid[ok][np.argmax(powers4[ok])])
check_golden("np_threshold", c_opt)
print(f"порог максимальной мощности: {c_opt:.4f} "
      f"(теория: z_0.95/√N = {C_NP:.4f})")

thetas = np.linspace(-1.0, 1.2, 221)
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.plot(thetas, [power_upper(t) for t in thetas],
        label="односторонний (р.н.м.к. при $\\theta > 0$)")
ax.plot(thetas, [power_two_sided(t) for t in thetas], ls="--",
        label="двусторонний (двойственность)")
ax.axhline(EPS4, color="black", lw=1, ls=":")
ax.axvline(0, color="gray", lw=0.8)
ax.set_xlabel(r"$\theta$")
ax.set_ylabel("мощность")
ax.set_ylim(0, 1.02)
ax.legend()
fig.tight_layout()
fig.savefig(f"{FIGDIR}/fig-04.pdf", **SAVE_KW)
plt.close(fig)
print("фигура: figures/fig-04.pdf")

# %% [markdown]
# **Вывод.** Предсказание сбылось по всем пунктам. Скан порогов дал максимум
# мощности при $c = 0{,}5210$ — в пределах шага сетки от теоретического
# $z_{0{,}95}/\sqrt{10} = 0{,}5201$: «интуитивный» порог по среднему совпал с
# оптимумом Неймана — Пирсона, что и утверждает лемма. Мощности при
# $\theta = +0{,}5$: односторонний $0{,}4746$ против двустороннего $0{,}3526$ —
# плата двустороннего за симметрию. При $\theta = -0{,}5$ картина зеркальна:
# верхний односторонний имеет мощность $0{,}0006$ (уровень почти весь ушёл
# в другую сторону), двусторонний — те же $0{,}3526$. Ни один порог по $T$
# не доминирует при всех $\theta \ne \theta_0$ сразу — р.н.м.к. для
# двусторонней альтернативы не существует, и двойственные критерии из
# предложения о двойственности — рабочая замена.

# %% [markdown]
# ## Пример 5. Критерии согласия: нормальность шума датчика
#
# $H_0$: выборка объёма $N = 200$ из $\mathcal N(0,1)$. Критерий хи-квадрат
# Пирсона: $r = 8$ равновероятных классов (границы — квантили $\Phi$),
# статистика (eq:chi2-stat); по теореме Пирсона при верной $H_0$ она
# сходится к $\chi^2_{r-1}$. Если параметры оценены по выборке
# ($\widehat a = \overline X_N$, $\widehat\sigma^2 = \widehat\sigma_N^2$ —
# результат Фишера из расширенной части конспекта), предел — $\chi^2_{r-3}$.
# Критерий Колмогорова: $\sqrt N\, D_N$ против функции $K$; здесь
# `scipy.special.kolmogorov` — функция выживания $1 - K$, так что p-value
# равен `kolmogorov(√N·D_N)`.
#
# **Предсказание.** (1) Уровень: на нормальных выборках доля отклонений
# каждого из трёх критериев близка к $\alpha = 0{,}05$. (2) Мощность: на
# выборках из Парето ($\alpha = 1{,}5$) и Коши все три критерии отклоняют
# почти всегда (доля > 0{,}99). (3) Для нормальной выборки p-value варианта
# с оценёнными параметрами больше, чем с фиксированными: оценка подгоняет
# распределение под данные, потеря двух степеней свободы это компенсирует.

# %%
from scipy.optimize import brentq
from scipy.special import kolmogorov   # survival: kolmogorov(x) = 1 - K(x)

N5 = 200
R5 = 8
EDGES5 = np.concatenate([stats.norm.ppf(np.arange(1, R5) / R5), [np.inf]])
EDGES5 = np.concatenate([[-np.inf], EDGES5])
CHI2_CRIT_FIX = stats.chi2.ppf(0.95, R5 - 1)   # G задана полностью
CHI2_CRIT_FIT = stats.chi2.ppf(0.95, R5 - 3)   # оценены 2 параметра
LAM_CRIT = brentq(lambda x: kolmogorov(x) - 0.05, 0.5, 3)
check_golden("chi2_crit_7", CHI2_CRIT_FIX)
check_golden("chi2_crit_5", CHI2_CRIT_FIT)
check_golden("kolm_lambda_095", LAM_CRIT)
print(f"критические значения: χ²_0.95;7 = {CHI2_CRIT_FIX:.4f}, "
      f"χ²_0.95;5 = {CHI2_CRIT_FIT:.4f}, λ_0.95 = {LAM_CRIT:.4f}")


def chi2_stat(sample, probs):
    obs = np.histogram(sample, bins=EDGES5)[0]
    exp = N5 * probs
    return float(((obs - exp) ** 2 / exp).sum())


def fitted_probs(sample):
    mu, s = sample.mean(), sample.std(ddof=0)
    return np.diff(stats.norm.cdf(EDGES5, mu, s))


def ks_dn(sample):
    xs = np.sort(sample)
    phi = stats.norm.cdf(xs)
    up = np.arange(1, N5 + 1) / N5
    lo = np.arange(0, N5) / N5
    return float(max(np.max(up - phi), np.max(phi - lo)))


sample_norm5 = rng_ex5.normal(0, 1, N5)
sample_par5 = rng_ex5.pareto(1.5, N5)
sample_cau5 = rng_ex5.standard_cauchy(N5)

print("одна выборка датчика:")
print(f"{'закон':<9}{'χ² p (df 7)':>13}{'χ² p (df 5)':>13}{'KS p':>12}")
pvals5 = {}
for name, smp in [("normal", sample_norm5), ("pareto", sample_par5),
                  ("cauchy", sample_cau5)]:
    p_fix = stats.chi2.sf(chi2_stat(smp, np.full(R5, 1 / R5)), R5 - 1)
    p_fit = stats.chi2.sf(chi2_stat(smp, fitted_probs(smp)), R5 - 3)
    p_ks = kolmogorov(np.sqrt(N5) * ks_dn(smp))
    pvals5[name] = (p_fix, p_fit, p_ks)
    print(f"{name:<9}{p_fix:>13.4g}{p_fit:>13.4g}{p_ks:>12.4g}")

# %% [markdown]
# ### Уровень и мощность критериев согласия
#
# $2000$ выборок каждого закона: доля отклонений при $\alpha = 0{,}05$.

# %%
TRIALS5 = 2000


def chi2_batch(arr, fitted):
    idx = np.digitize(arr, EDGES5[1:-1])
    obs = np.apply_along_axis(lambda r: np.bincount(r, minlength=R5), 1, idx)
    if fitted:
        mu = arr.mean(axis=1, keepdims=True)
        s = arr.std(axis=1, keepdims=True)
        probs = np.diff(stats.norm.cdf(EDGES5[None, :], mu, s), axis=1)
    else:
        probs = np.full((1, R5), 1 / R5)
    exp = N5 * probs
    return ((obs - exp) ** 2 / exp).sum(axis=1)


def ks_batch(arr):
    xs = np.sort(arr, axis=1)
    phi = stats.norm.cdf(xs)
    up = np.arange(1, N5 + 1) / N5
    lo = np.arange(0, N5) / N5
    return np.maximum((up - phi).max(axis=1), (phi - lo).max(axis=1))


def draw_batch(name, size):
    if name == "normal":
        return rng_ex5.normal(0, 1, size)
    if name == "pareto":
        return rng_ex5.pareto(1.5, size)
    return rng_ex5.standard_cauchy(size)


rates5 = {}
print(f"доля отклонений ({TRIALS5} выборок, α = 0.05):")
print(f"{'закон':<9}{'χ² df 7':>9}{'χ² df 5':>9}{'KS':>9}")
for name in ("normal", "pareto", "cauchy"):
    arr = draw_batch(name, (TRIALS5, N5))
    r_fix = float((chi2_batch(arr, fitted=False) > CHI2_CRIT_FIX).mean())
    r_fit = float((chi2_batch(arr, fitted=True) > CHI2_CRIT_FIT).mean())
    r_ks = float((np.sqrt(N5) * ks_batch(arr) > LAM_CRIT).mean())
    rates5[name] = (r_fix, r_fit, r_ks)
    print(f"{name:<9}{r_fix:>9.4f}{r_fit:>9.4f}{r_ks:>9.4f}")

fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
xs = np.linspace(-4, 6, 601)
for ax, (name, smp) in zip(axes, [("normal", sample_norm5),
                                  ("pareto", sample_par5),
                                  ("cauchy", sample_cau5)]):
    ax.plot(xs, stats.norm.cdf(xs), color="black", label="$\\Phi(x)$")
    ax.step(np.sort(smp), np.arange(1, N5 + 1) / N5, where="post",
            label="эмпирическая")
    ax.set_title(name)
    ax.set_xlim(-4, 6)
    ax.set_xlabel("$x$")
axes[0].set_ylabel("$F$")
axes[0].legend()
fig.tight_layout()
fig.savefig(f"{FIGDIR}/fig-05.pdf", **SAVE_KW)
plt.close(fig)
print("фигура: figures/fig-05.pdf")
check_golden("pnorm_chi2_fix", pvals5["normal"][0])
check_golden("pnorm_chi2_fit", pvals5["normal"][1])
check_golden("pnorm_ks", pvals5["normal"][2])
check_golden("level_chi2_fix", rates5["normal"][0])
check_golden("level_chi2_fit", rates5["normal"][1])
check_golden("level_ks", rates5["normal"][2])
check_golden("pow_par_chi2_fix", rates5["pareto"][0])
check_golden("pow_par_chi2_fit", rates5["pareto"][1])
check_golden("pow_par_ks", rates5["pareto"][2])
check_golden("pow_cau_chi2_fix", rates5["cauchy"][0])
check_golden("pow_cau_chi2_fit", rates5["cauchy"][1])
check_golden("pow_cau_ks", rates5["cauchy"][2])

# %% [markdown]
# **Вывод.** Пункты (1) и (2) предсказания сбылись, пункт (3) — нет, и это
# поучительно. Уровень: на нормальных выборках доли отклонений равны $0{,}0490$
# ($\chi^2$, фиксированные параметры), $0{,}0535$ ($\chi^2$, оценённые
# параметры) и $0{,}0465$ (Колмогоров) — все в пределах разброса от
# $\alpha = 0{,}05$; предельные распределения ($\chi^2_7$, $\chi^2_5$, $K$)
# работают уже при $N = 200$. Мощность: Парето и Коши отклоняются практически
# всегда (доли $\ge 0{,}999$). А вот знак разницы p-value на одной нормальной
# выборке предсказан неверно: оценённые параметры дали $0{,}7220$ — меньше,
# чем $0{,}8070$ у фиксированных. Оба варианта асимптотически калиброваны, и
# знак разницы на одной выборке случаен: подгонка снижает статистику, но
# потеря двух степеней свободы сдвигает распределение в другую сторону. Из
# стабильного: p-value Парето и Коши практически нулевые у всех критериев
# (у Коши с оценёнными параметрами $10^{-211}$), то есть отклонение $H_0$
# однозначное. Критерии согласия справляются со своей задачей: заметить
# ненормальность шума датчика.
