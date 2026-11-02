# %% [markdown]
# # Вопрос 17. Метод Монте-Карло — численные примеры
#
# Файл ведётся в формате jupytext (percent). Парный `.ipynb` получается
# командой `make questions/17-monte-carlo/examples.ipynb`; в git идёт **этот**
# файл, ноутбук — производный артефакт.
#
# **Сквозной пример** — тот же нормальный датчик, что в вопросах 12 и 14:
# выборка $\xi_k = \theta + \text{шум}$. Четыре примера отвечают на вопросы
# конспекта:
#
# | № | Вопрос из theory.md | Объект счёта |
# |---|---|---|
# | 1 | сходимость оценки и вероятная ошибка (теорема о погрешности) | вероятность накрытия u-интервала |
# | 2 | сравнение с квадратурными формулами по размерности | объём шара в $\mathbb{R}^n$ |
# | 3 | способы уменьшения дисперсии | интеграл $\int_0^1 e^x dx$ |
# | 4 | квазислучайные последовательности | гладкий интеграл в кубе |
#
# Каждый пример формулирует предсказание **до** прогона; предсказание
# остаётся в тексте, даже если не сбылось.

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

SEED = 20261102

# Separate generators derived from one seed: a shared stream would couple the
# examples, and changing the ensemble size of example 1 would shift every
# golden pin downstream.
rng_ex1, rng_ex2, rng_ex3, rng_ex4 = (
    np.random.default_rng([SEED, k]) for k in range(4)
)

FIGDIR = "figures"

# PDF metadata carries a creation timestamp; pin it to keep figure bytes stable.
SAVE_KW = {"metadata": {"CreationDate": None}}

# Golden pins freeze seed AND expected output; a mismatch aborts the build.
# Analytic values get machine tolerances; sampled ones get the spread of the
# estimator at the stated ensemble size.
GOLDEN = {
    "u_z975": (1.959963984540054, 1e-12),
    # sampled pin: tolerance ~6.5 sigma of the binomial spread
    "cover_u_N20000": (0.9488, 0.01),
    "V3_analytic": (4.1887902047863905, 1e-12),
    "V10_analytic": (2.550164039877345, 1e-12),
    # analytic values from the theory propositions (repr from a run of this file)
    "var_plain_exp": (0.24203560745276542, 1e-12),
    "var_antithetic_exp": (0.003912496949625588, 1e-12),
    "var_ratio_antithetic": (30.93109216045873, 1e-12),
    # theory problem 2: importance-sampling variances for n=1 and n=2
    "var_is_n1": (0.18518518518518512, 1e-12),
    "var_is_n2": (1.5987199999999984, 1e-12),
    "I_2d_analytic": (2.9524924420125593, 1e-12),
    # deterministic LP-tau prefix mean at N=2^18: repr pinned
    "qmc_err_N2_18": (3.8142137245237155e-06, 1e-9),
    "mc_err_N2_18": (0.000671759903571503, 0.05),
    # analytic coefficients of the variance-reduction schemes (closed forms)
    "coef_main_analytic": (0.04365076924514407, 1e-12),
    "coef_strat_analytic": (0.0026594472922521818, 1e-12),
    # sampled slopes and coefficients (repr from a run of this file)
    "coef_is_sobol": (0.026908428183502142, 1e-9),
    "slope_mc_2": (-0.45348685749449413, 0.15),
    "slope_mc_8": (-0.43404084695667067, 0.15),
    "slope_qmc_unscrambled": (-0.99655141583403, 0.1),
    # control MC estimates at N=2^17 (repr from a run of this file)
    "mc_relerr_V3": (0.002158539623161312, 0.2),
    "mc_relerr_V10": (0.03547260439254167, 0.2),
}


def check_golden(name, value):
    want, tol = GOLDEN[name]
    rel = abs(value - want) / abs(want)
    print(f"  пин {name}: получено {value:.12g}, ожидалось {want:.12g}")
    assert rel <= tol, f"golden-пин {name} не сошёлся: {value} vs {want}"


# %% [markdown]
# ## Пример 1. Оценка вероятности накрытия доверительного интервала
#
# **Вопрос:** как метод Монте-Карло оценивает вероятность накрытия и какова
# погрешность этой оценки (теорема о погрешности, вероятная ошибка из
# конспекта)?
#
# Постановка — из вопроса 14 (интервальное оценивание и проверка гипотез):
# выборка $\xi_1,\dots,\xi_n$ из $\mathcal N(\theta,\sigma^2)$ с $n = 25$,
# $\sigma = 1$, истинное $\theta = 0$; u-интервал
# $\overline X \pm z_{0{,}975}\,\sigma/\sqrt n$ с известной $\sigma$.
# Вероятность накрытия
# $J = \Prob\{\theta \in CI\}$ известна аналитически: $J = 0{,}95$.
# Методом Монте-Карло оцениваем $J_N = \frac{1}{N}\sum_k \zeta_k$, где
# $\zeta_k$ — индикатор накрытия в $k$-м прогоне эксперимента.
#
# **Предсказание (до прогона).** (1) Оценка $J_N$ при $N = 20\,000$
# лежит в пределах $\pm 0{,}01$ от $0{,}95$: разброс оценки равен
# $\sqrt{J(1-J)/N} \approx 0{,}00154$, то есть допуск — более $6$ сигм.
# (2) Эмпирическая вероятная ошибка $0{,}6745\sqrt{J_N(1-J_N)/N}$
# совпадает с фактическим отклонением $|J_N - J|$ по порядку величины
# (обе величины $\sim 0{,}001$). (3) Вероятная ошибка пропорциональна
# $1/\sqrt N$, поэтому на графике «траектория против $1/\sqrt N$»
# траектория $J_N$ должна быть зажата **клином** $J \pm r_N$ с вершиной в
# пределе $J = 0{,}95$: обе границы клина — прямые через точку $(0, J)$.
# Полоса постоянной ширины получилась бы в других координатах —
# $\sqrt N\,(J_N - J)$ против $N$.

# %%
N_EXPERIMENTS = 20_000
N_SAMPLE = 25
SIGMA = 1.0
THETA = 0.0
Z975 = stats.norm.ppf(0.975)
check_golden("u_z975", Z975)


# Vectorized trials: each row is one experiment (a sample of N_SAMPLE).
# Drawing the whole batch at once consumes the same PCG64 stream as a loop.
samples = rng_ex1.normal(THETA, SIGMA, size=(N_EXPERIMENTS, N_SAMPLE))
half_width = Z975 * SIGMA / np.sqrt(N_SAMPLE)
hits = (np.abs(samples.mean(axis=1) - THETA) <= half_width).astype(float)

# Trajectory of the running estimate
N_GRID = np.unique(
    np.concatenate(
        [np.arange(100, 1000, 100), np.arange(1000, N_EXPERIMENTS + 1, 1000)]
    )
)
traj = np.cumsum(hits)[N_GRID - 1] / N_GRID

J_hat = hits.mean()
var_emp = J_hat * (1 - J_hat)
prob_err_emp = 0.6745 * np.sqrt(var_emp / N_EXPERIMENTS)
abs_dev = abs(J_hat - 0.95)

print(f"оценка J_N при N={N_EXPERIMENTS}: {J_hat:.5f} (истина 0.95)")
print(f"эмпирическая вероятная ошибка: {prob_err_emp:.5f}")
print(f"фактическое отклонение |J_N - J|: {abs_dev:.5f}")
print(
    f"отношение отклонения к вероятной ошибке: {abs_dev / prob_err_emp:.2f} "
    "(ожидается порядка 1-2)"

)
check_golden("cover_u_N20000", J_hat)

# %% [markdown]
# Проверка скорости сходимости: границы клина $J \pm r_N$ в координатах
# $1/\sqrt N$ — прямые $J \pm 0{,}6745\sqrt{J(1-J)}\,x$, то есть линейные
# функции от $x = 1/\sqrt N$; траектория должна лежать внутри клина
# (за пределами — примерно половина случайных реализаций).

# %%
fig, ax = plt.subplots(figsize=(6, 3.6), layout="constrained")
xs = 1 / np.sqrt(N_GRID)
ax.axhline(0.95, color="black", linewidth=1, linestyle="--", label=r"$J = 0.95$")
k_wedge = 0.6745 * np.sqrt(0.95 * 0.05)
x_line = np.array([0.0, xs.max()])
ax.plot(x_line, 0.95 + k_wedge * x_line, color="tab:red", linewidth=1,
        linestyle=":", label=r"границы клина $J \pm r_N$")
ax.plot(x_line, 0.95 - k_wedge * x_line, color="tab:red", linewidth=1,
        linestyle=":")
ax.plot(xs, traj, ".-", markersize=3, linewidth=0.8, label=r"оценка $J_N$")
ax.set_xlabel(r"$1/\sqrt{N}$")
ax.set_ylabel(r"оценка вероятности накрытия $J_N$")
ax.legend(loc="lower right", fontsize=8)
fig.savefig(f"{FIGDIR}/fig-01.pdf", **SAVE_KW)

# %% [markdown]
# **Вывод:** предсказания подтвердились: оценка близка к $0{,}95$, эмпирическая
# вероятная ошибка и фактическое отклонение одного порядка
# (отношение напечатано выше). На рисунке виден клин: в координатах
# $1/\sqrt N$ вероятная ошибка растёт линейно от абсциссы, и траектория
# сходится к вершине клина $J = 0{,}95$ — именно так выглядит сходимость
# со скоростью $N^{-1/2}$. (Полоса постоянной ширины получилась бы в
# координатах $\sqrt N\,(J_N - J)$ против $N$.) Это и есть сквозной сюжет
# блока: интервал из вопроса 14 проверяется тем самым методом, который
# изучаем здесь, — оценкой математического ожидания индикатора.

# %% [markdown]
# ## Пример 2. Проклятие размерности: метод Монте-Карло против сеточной квадратуры
#
# **Вопрос:** как сравниваются метод Монте-Карло и сеточные квадратурные
# формулы при росте размерности (раздел «Сравнение с квадратурными
# формулами» конспекта)?
#
# Объект счёта — объём единичного шара $V_n = \pi^{n/2}/\Gamma(n/2+1)$ в
# $\mathbb{R}^n$ (аналитическое значение известно, источник — свойство
# гамма-функции; для нас это просто контрольное значение). Интеграл:
# $V_n = \int_{[-1,1]^n} \mathbb 1\{\lVert x\rVert \leq 1\}\,dx$ —
# математическое ожидание индикатора при равномерной точке в кубе, оценка
# $V_n \approx 2^n \cdot (\text{доля попаданий})$.
#
# **Предсказание (до прогона).** (1) На логарифмической шкале ошибка метода
# Монте-Карло падает с наклоном $\approx -1/2$; для $n = 12$ наклон
# измерим лишь по хвосту, где на реплику в среднем больше единицы попаданий
# ($N \gtrsim 2^{14}$: $V_{12}/2^{12} \approx 3{,}3\cdot10^{-4}$), а при
# меньших $N$ (в среднем меньше одного попадания) ошибка $100\%$ у нулевых
# реплик и наклон не определён.
# (2) Сеточная квадратура (произведение средних прямоугольников с $m$ узлами
# на координату, $N = m^n$): подынтегральная функция — индикатор, то есть
# разрывная, и теория даёт для неё эффективный порядок единицы (погрешность
# сносят пограничные ячейки), то есть тренд $\approx -1/n$, а не $-4/n$
# формулы Симпсона. Точки ошибки при этом скачут от сетки к сетке (зависит
# от того, как узлы ложатся на границу шара), поэтому наклон квадратуры
# устойчиво не измерить — показываем точки и эталонные наклоны без претензии
# на подгонку. Устойчиво измеримые факты: при $n = 2$ сетка заметно точнее
# МК при малом $N$; при $n = 8$ и $n = 12$ квадратура застревает на ошибке
# порядка единицы до недостижимых объёмов ($m^n$). (3) Контрольные значения:
# $V_3 = 4\pi/3 \approx
# 4{,}18879$, $V_{10} = \pi^5/120 \approx 2{,}55016$ — оценки метода
# Монте-Карло при $N = 2^{17}$ попадают в эти значения с точностью порядка
# $N^{-1/2}$ (для $V_{10}$ относительная ошибка $\sim$ нескольких процентов,
# поскольку доля попаданий мала: $V_{10}/2^{10} \approx 0{,}0025$).

# %%
from scipy.special import gamma


def ball_volume(d):
    return np.pi ** (d / 2) / gamma(d / 2 + 1)


def mc_ball_volume(d, n, rng):
    pts = rng.uniform(-1.0, 1.0, size=(n, d))
    hits = np.sum(pts * pts, axis=1) <= 1.0
    return (2.0 ** d) * hits.mean(), int(hits.sum())


def mc_ball_curve(d, n_replicates=16):
    """Mean |error| curve over prefix estimates, averaged over independent
    replicates — a single stream bounces too much for a slope estimate."""
    errs = np.zeros(len(POWERS))
    total_hits = np.zeros(len(POWERS), dtype=int)
    for rep in range(n_replicates):
        stream = np.random.default_rng([SEED, 9, d, rep])
        pts = stream.uniform(-1.0, 1.0, size=(2 ** POWERS[-1], d))
        hits_cum = np.cumsum(np.sum(pts * pts, axis=1) <= 1.0)
        ests = (2.0 ** d) * hits_cum[2 ** POWERS - 1] / (2.0 ** POWERS)
        errs += np.abs(ests - ball_volume(d)) / ball_volume(d)
        total_hits += hits_cum[2 ** POWERS - 1]
    return errs / n_replicates, total_hits / n_replicates


def grid_ball_volume(d, m):
    """Midpoint-product rule: m^n evaluations on a uniform grid of the cube."""
    g = (np.arange(m) + 0.5) / m * 2.0 - 1.0
    coords = np.meshgrid(*([g] * d), indexing="ij")
    r2 = np.zeros((m,) * d)
    for c in coords:
        r2 = r2 + c * c
    frac = np.mean(r2 <= 1.0)
    return (2.0 ** d) * frac


def slope_fit(ns, errs, lo, hi):
    m = (ns >= lo) & (ns <= hi)
    return np.polyfit(np.log2(ns[m]), np.log2(errs[m]), 1)[0]


DIMS_MC = [2, 8, 12]
POWERS = np.arange(8, 18)  # N = 256 .. 131072
mc_err = {}
mc_hits = {}
for d in DIMS_MC:
    mc_err[d], mc_hits[d] = mc_ball_curve(d)

# Наклоны МК: полный диапазон для n=2, 8; для n=12 — только где есть
# попадания (иначе ошибка тривиально равна 1 и наклон не определён).
slope_mc = {
    2: slope_fit(2.0 ** POWERS, mc_err[2], 2 ** 8, 2 ** 17),
    8: slope_fit(2.0 ** POWERS, mc_err[8], 2 ** 8, 2 ** 17),
    12: slope_fit(2.0 ** POWERS, mc_err[12], 2 ** 14, 2 ** 17),
}
print("наклоны МК (по среднему из 16 реплик):",
      {d: f"{s:.2f}" for d, s in slope_mc.items()})
print(f"среднее число попаданий при n=12 (16 реплик): "
      f"{dict(zip([f'2^{p}' for p in POWERS], mc_hits[12]))}")

DIMS_GRID = [2, 8, 12]
MS = {2: np.arange(2, 300, 6), 8: np.arange(2, 7), 12: np.arange(2, 4)}
grid_pts = {}
grid_err = {}
for d in DIMS_GRID:
    true_v = ball_volume(d)
    ns, es = [], []
    for m in MS[d]:
        est = grid_ball_volume(d, m)
        ns.append(m ** d)
        es.append(abs(est - true_v) / true_v)
    grid_pts[d] = np.array(ns)
    grid_err[d] = np.array(es)

# Наклон квадратуры по скачущим точкам не измеряем (см. предсказание):
# печатаем МНК-тренд как справку, без претензии на измерение порядка.
slope_grid_trend = {
    d: slope_fit(grid_pts[d], grid_err[d], grid_pts[d].min(), grid_pts[d].max())
    for d in DIMS_GRID
}
print("МНК-тренд сетки (точки скачут, справочно):",
      {d: f"{s:.2f}" for d, s in slope_grid_trend.items()})
check_golden("slope_mc_2", slope_mc[2])
check_golden("slope_mc_8", slope_mc[8])

for d in [3, 10]:
    print(f"V_{d} = {ball_volume(d):.10f}")
check_golden("V3_analytic", ball_volume(3))
check_golden("V10_analytic", ball_volume(10))

# control announced in the prediction: plain MC estimates of V_3 and V_10
for d in [3, 10]:
    est_v, _ = mc_ball_volume(d, 2 ** 17, rng_ex2)
    rel = abs(est_v - ball_volume(d)) / ball_volume(d)
    print(f"MC-оценка V_{d} при N=2^17: {est_v:.6f} "
          f"(истина {ball_volume(d):.6f}, отн. ошибка {rel:.2%})")
    check_golden(f"mc_relerr_V{d}", rel)

# %% [markdown]
# График: относительная ошибка против числа вычислений подынтегральной
# функции, обе оси логарифмические. Пунктир — эталонные наклоны $-1/2$
# (метод Монте-Карло) и $-1/n$ (сеточная квадратура с разрывным подынтегральным
# выражением). Кресты на уровне $10^0$ для МК при $n = 12$ — объёмы, где в
# среднем меньше одного попадания на реплику (ошибка $100\%$ у нулевых
# реплик); наклон там не определён.

# %%
fig, ax = plt.subplots(figsize=(6.4, 4.0), layout="constrained")
markers = {2: "o", 8: "s", 12: "^"}
for d in DIMS_MC:
    ax.plot(
        2.0 ** POWERS,
        mc_err[d],
        marker=markers[d],
        markersize=4,
        linewidth=0.9,
        linestyle="",
        label=f"МК, $n={d}$",
    )
    ax.plot(
        grid_pts[d],
        grid_err[d],
        marker=markers[d],
        markersize=6,
        linestyle="",
        mfc="none",
        label=f"сетка, $n={d}$",
    )
ns = np.array([1e3, 3e5])
ax.plot(ns, 3e-1 * (ns / ns[0]) ** (-0.5), "k--", linewidth=1,
        label=r"наклон $-1/2$")
ax.plot(ns, 1e-2 * (ns / ns[0]) ** (-1 / 12), "k:", linewidth=1,
        label=r"наклон $-1/12$")
# zero-hit region for n=12 (mean hits < 1 per replicate): x-marks
z = [p for p, h in zip(2.0 ** POWERS, mc_hits[12]) if h < 1.0]
if z:
    ax.plot(z, [1.0] * len(z), "x", color="tab:red", markersize=7,
            label="МК $n=12$: нет попаданий")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"число вычислений функции $N$")
ax.set_ylabel("относительная ошибка")
ax.legend(loc="lower left", fontsize=8)
fig.savefig(f"{FIGDIR}/fig-02.pdf", **SAVE_KW)

# %% [markdown]
# **Вывод.** Наклоны напечатаны выше. Средняя по 16 репликам кривая метода
# Монте-Карло идёт с наклоном, близким к $-1/2$ (измерено $-0{,}43\ldots-0{,}45$
# при $n = 2, 8$), при $n = 12$ до $N \approx 2^{11}$ в среднем меньше одного
# попадания на реплику (кресты на уровне $100\%$ — наклон там не
# определён), а по хвосту $N \gtrsim 2^{14}$ кривая встаёт в общий наклон:
# независимость порядка от размерности подтверждена на измеримой части.
# Точки сетки скачут, и наклон квадратуры устойчиво не измеряется; устойчивый
# измеренный контраст — другой: при $n = 8$ и $n = 12$ квадратура держит
# ошибку порядка единицы до недостижимых объёмов (при $n = 12$ это
# $N \gtrsim 5\cdot10^{5}$), а метод Монте-Карло сходится с тем же порядком
# $-1/2$ при любой размерности. Для гладких подынтегральных функций картина
# та же по структуре: порядок $-r/n$ квадратур против независимого от $n$
# порядка $-1/2$ метода Монте-Карло.


# %% [markdown]
# ## Пример 3. Уменьшение дисперсии: три приёма на одном интеграле
#
# **Вопрос:** как работают способы уменьшения дисперсии из конспекта
# (выделение главной части, существенная выборка, симметризация, расслоение)
# и каков измеренный выигрыш?
#
# Объект — тот же интеграл $I = \int_0^1 e^x\,dx = e - 1 \approx 1{,}71828$,
# у Соболя он же (гл. 3 §§ 2–3). Считаем пять оценок при $N = 10^4$ и
# измеряем их дисперсии методом $m$ вариантов (п. 1.5 [10], то же, что
# в конспекте): $M = 400$ повторов расчёта оценки, дисперсия оценки —
# выборочная дисперсия повторов, делённая на $M$.
#
# **Предсказание (до прогона).** Дисперсии совпадут с аналитическими
# значениями конспекта (коэффициенты при $1/N$): простейшая
# $\tfrac{e^2-1}{2}-(e-1)^2 \approx 0{,}2420$; выделение главной части
# с $g_1 = 1+x$: $\Var(e^\xi - 1 - \xi) \approx 0{,}0437$ (выигрыш в
# $5{,}5$ раза); симметризованная $\approx 0{,}0039$ (выигрыш $\approx 31$
# раз на равное число вычислений $g$); существенная выборка с реальной
# плотностью $p_1 = \tfrac23(1+x)$ из примера Соболя [10, гл. 3 § 3.2]:
# коэффициент $\tfrac32\int_0^1 \tfrac{e^{2x}}{1+x}\,dx - I^2 \approx 0{,}0269$
# (выигрыш около $9$ раз); расслоение на $L = 10$ пропорциональных слоёв:
# коэффициент $\tfrac{1}{L}\sum_h\sigma_h^2 \approx 0{,}0027$ (выигрыш в
# $\sim 90$ раз) — межслоевая дисперсия устраняется полностью, а в узком слое
# $e^x$ почти линейна, так что внутрислоевой остаток мал. Предельный случай
# $p^* \propto e^x$ (дисперсия нуль, оценка тождественно $e-1$) на графике не
# показан: он обнулил бы логарифмическую шкалу; проверяется отдельной строкой
# печати ниже.

# %%
I_EXP = np.e - 1
VAR_PLAIN = (np.e ** 2 - 1) / 2 - I_EXP ** 2  # ∫e^{2x}dx - I^2
COV_ANT = np.e - I_EXP ** 2  # E[e^ξ e^{1-ξ}] - I^2
VAR_ANT = (VAR_PLAIN + COV_ANT) / 2
print(f"аналитика: Var простейшей = {VAR_PLAIN:.10f}")
print(f"аналитика: ковариация антитетической пары = {COV_ANT:.10f}")
print(f"аналитика: Var симметризованной = {VAR_ANT:.10f}")
print(f"аналитика: выигрыш на равное число вычислений = "
      f"{VAR_PLAIN / (2 * VAR_ANT):.2f}")
check_golden("var_plain_exp", VAR_PLAIN)
check_golden("var_antithetic_exp", VAR_ANT)
check_golden("var_ratio_antithetic", VAR_PLAIN / (2 * VAR_ANT))

# theory problem 2: importance-sampling variances for ∫ x^n e^{-x} dx
# (k = A = 1) with density a·e^{-ax}; optimum a* = 1/(n+1)
from scipy.special import factorial


def var_importance(n):
    a_opt = 1.0 / (n + 1)
    second = (1.0 / a_opt) * factorial(2 * n) / (2.0 - a_opt) ** (2 * n + 1)
    mean = factorial(n)  # I = n! for k = A = 1
    return second - mean ** 2


for n in (1, 2):
    print(f"задача 2, n={n}: Var существенной выборки = "
          f"{var_importance(n):.10f} (простейший: "
          f"{factorial(2 * n) - factorial(n) ** 2:.1f})")
check_golden("var_is_n1", var_importance(1))
check_golden("var_is_n2", var_importance(2))

# %% [markdown]
# Пять оценок одного интеграла при $N = 10^4$, $M = 400$ повторов для
# измерения дисперсии оценки:

# %%
N_POINTS = 10_000
M_REPEATS = 400
L_STRATA = 10


def estimate_plain(n, rng):
    return np.exp(rng.uniform(size=n)).mean()


def estimate_antithetic(n, rng):
    u = rng.uniform(size=n // 2)
    return ((np.exp(u) + np.exp(1.0 - u)) / 2).mean()


def estimate_importance_real(n, rng):
    # Sobol's practical density p1 = (2/3)(1+x): xi = sqrt(1+3u) - 1
    u = rng.uniform(size=n)
    xi = np.sqrt(1.0 + 3.0 * u) - 1.0
    return np.mean(1.5 * np.exp(xi) / (1.0 + xi))


def estimate_importance_star(n, rng):
    # limiting case p* ∝ e^x: xi = ln(1 + (e-1)u); the estimator is e - 1
    u = rng.uniform(size=n)
    xi = np.log1p((np.e - 1.0) * u)
    return np.mean(np.exp(xi) * 1.0 / (np.exp(xi) / I_EXP))


def estimate_main_part(n, rng):
    # g1 = 1 + x, J1 = 3/2; remainder e^x - 1 - x
    u = rng.uniform(size=n)
    return 1.5 + np.mean(np.exp(u) - 1.0 - u)


def estimate_stratified(n, rng):
    per = n // L_STRATA
    total = 0.0
    for h in range(L_STRATA):
        u = (h + rng.uniform(size=per)) / L_STRATA
        total += np.exp(u).mean()
    return total / L_STRATA


SCHEMES = {
    "простейшая": estimate_plain,
    "выделение главной части": estimate_main_part,
    "существенная выборка, p1=(2/3)(1+x)": estimate_importance_real,
    "симметризация": estimate_antithetic,
    f"расслоение, L={L_STRATA}": estimate_stratified,
}

# The p* limiting case breaks the log scale of fig-03; verify it separately:
star_reps = np.array([estimate_importance_star(N_POINTS, rng_ex3)
                      for _ in range(M_REPEATS)])
print(f"предельный случай p*∝e^x: среднее = {star_reps.mean():.10f} "
      f"(= e - 1), разброс = {star_reps.std(ddof=1):.2e} (машинный нуль)")

measured = {}
for name, est in SCHEMES.items():
    reps = np.array([est(N_POINTS, rng_ex3) for _ in range(M_REPEATS)])
    measured[name] = reps.var(ddof=1) / M_REPEATS
    print(f"{name:34s}: Var оценки = {measured[name]:.3e} "
          f"(разброс повторов σ = {reps.std(ddof=1):.2e}), "
          f"среднее = {reps.mean():.6f}")
    # the estimate must approach e - 1: pin catches defects like a lost factor
    rel = abs(reps.mean() - I_EXP) / I_EXP
    assert rel < 1e-3, f"оценка '{name}' смещена: среднее {reps.mean()}"

# coefficients of 1/N: variance of one estimate (xM) at volume N;
# estimate_antithetic(n) spends exactly n evaluations of g (n//2 pairs), so
# for every scheme the coefficient is per one evaluation of g.
coef_meas = {k: v * M_REPEATS * N_POINTS for k, v in measured.items()}


def analytic_main_part():
    # h(x) = e^x - 1 - x: Var h(ξ), ξ ~ U(0,1), closed-form integrals
    e2 = (np.e ** 2 - 1) / 2 - 2 * np.e + 7 / 3  # E h^2
    return e2 - (I_EXP - 1.5) ** 2


def analytic_stratified():
    # (1/L) sum_h sigma_h^2 for proportional allocation, equal strata
    total = 0.0
    for h in range(L_STRATA):
        a, b = h / L_STRATA, (h + 1) / L_STRATA
        m1 = (np.e ** b - np.e ** a) / (b - a)
        m2 = (np.e ** (2 * b) - np.e ** (2 * a)) / (2 * (b - a))
        total += m2 - m1 ** 2
    return total / L_STRATA


def analytic_importance_sobol():
    # Var of 1.5*e^ξ/(1+ξ), ξ ~ p1 = (2/3)(1+x); Sobol [10], § 3.2.1: 0.0269
    from scipy.integrate import quad
    second, _ = quad(lambda x: 1.5 * np.exp(2 * x) / (1.0 + x), 0.0, 1.0,
                     epsabs=1e-14)
    return second - I_EXP ** 2


COEF_ANALYTIC = {
    "простейшая": VAR_PLAIN,
    "выделение главной части": analytic_main_part(),
    "существенная выборка, p1=(2/3)(1+x)": analytic_importance_sobol(),
    "симметризация": 2 * VAR_ANT,
    f"расслоение, L={L_STRATA}": analytic_stratified(),
}
check_golden("coef_main_analytic", COEF_ANALYTIC["выделение главной части"])
check_golden("coef_strat_analytic", COEF_ANALYTIC[f"расслоение, L={L_STRATA}"])
check_golden("coef_is_sobol", COEF_ANALYTIC["существенная выборка, p1=(2/3)(1+x)"])
print("\nкоэффициенты дисперсии на вычисление g (измерено vs аналитика):")
for name in SCHEMES:
    cm = max(coef_meas[name], 1e-30)
    if name == "простейшая":
        print(f"  {name:32s}: {cm:.5f} vs {COEF_ANALYTIC[name]:.5f}")
    else:
        print(f"  {name:32s}: {cm:.5f} vs {COEF_ANALYTIC[name]:.5f}"
              f"  (выигрыш {VAR_PLAIN / cm:.1f}x)")

fig, ax = plt.subplots(figsize=(6, 3.6), layout="constrained")
names = list(SCHEMES)
coefs = [max(coef_meas[n], 1e-30) for n in names]
ax.bar(range(len(names)), coefs, color="steelblue")
ax.set_yscale("log")
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
ax.set_ylabel(r"коэффициент дисперсии $\mathsf{D}\zeta$")
ax.axhline(VAR_PLAIN, color="black", linewidth=1, linestyle="--")
fig.savefig(f"{FIGDIR}/fig-03.pdf", **SAVE_KW)

# %% [markdown]
# **Вывод:** предсказание подтвердилось по всем пяти схемам (числа
# напечатаны в блоке выше). Существенная выборка с реальной плотностью
# $p_1 = \tfrac23(1+x)$ даёт выигрыш около $9$ раз — это и есть практический
# режим приёма; предельный случай $p^* \propto e^x$ оценка тождественно равна
# $e - 1$ (разброс повторов — машинный нуль, порядка $10^{-16}$ относительно,
# а не «измеренная нулевая дисперсия»). Симметризация выигрывает в
# $\approx 31$ раз на равное число вычислений $g$, расслоение на десяти слоях
# — в $\sim 90$ раз: межслоевая дисперсия устранена полностью, а внутри узких
# слоёв $e^x$ почти линейна. Заметим, что выигрыш расслоения здесь особенно
# велик, потому что $e^x$ гладкая; для функции с внутрислоевым изломом
# выигрыш был бы скромнее — выбор приёма опирается на трудоёмкость и
# структуру $g$, а не на дисперсию одну.


# %% [markdown]
# ## Пример 4. Квазислучайные последовательности против псевдослучайных
#
# **Вопрос:** каков порядок сходимости квазислучайной квадратуры (раздел
# «Квазислучайные последовательности» конспекта) и когда она выигрывает у
# случайного метода?
#
# Объект — гладкий интеграл с известным значением:
# $I = \int_{[0,1]^2} e^{x_1 + x_2}\,dx = (e - 1)^2 \approx 2{,}95249$
# (та же подынтегральная функция, что в примере 3, теперь двумерная).
# Сравниваем: псевдослучайный метод Монте-Карло и **детерминированную**
# ЛП$\tau$-последовательность (`scipy.stats.qmc.Sobol` без скремблирования,
# `bits=64`: точки scipy по умолчанию квантуются 30 битами и рисуют
# фальшивое плато ошибки $\sim 2^{-30}$).
#
# **Предсказание (до прогона).** (1) Ошибка псевдослучайного метода (в среднем
# по репликам) идёт с наклоном $\approx -1/2$. (2) Для детерминированной
# ЛП$\tau$-последовательности теоретическая граница $O(N^{-1}\ln^2 N)$ в
# размерности $n = 2$ даёт чистый наклон около $-1$ на доступных $N$, без
# усреднения по реализациям. (3) При $N = 2^{18}$ квазислучайная ошибка
# ожидается порядка нескольких $10^{-6}$ — на два с лишним порядка меньше
# псевдослучайной. Доверительный интервал детерминированная схема не даёт
# (только оценку через отклонение); **рандомизированный** вариант
# (скремблирование с осреднением по повторам) интервал возвращает — его
# характеристики напечатаны ниже отдельно.

# %%
D = 2
I_2D = I_EXP ** D
check_golden("I_2d_analytic", I_2D)

POW2 = np.arange(4, 19)  # N = 16 .. 262144
NS = 2 ** POW2


def mc_2d_errors(n_replicates=16):
    """Mean |error| of the plain MC estimate over prefix means, averaged
    over replicates — smooths the N^{-1/2} noise of one stream."""
    errs = np.zeros(len(NS))
    for rep in range(n_replicates):
        stream = np.random.default_rng([SEED, 6, rep])
        pts = stream.uniform(size=(NS[-1], D))
        vals = np.exp(pts.sum(axis=1))
        ests = np.array([vals[:n].mean() for n in NS])
        errs += np.abs(ests - I_2D) / I_2D
    return errs / n_replicates


def qmc_2d_errors():
    """Deterministic LP-tau sequence (no scrambling), prefix means.
    bits=64: the scipy default (bits=30) quantizes coordinates to a
    2^{-30} grid and fakes an error plateau at ~1e-9."""
    from scipy.stats import qmc

    engine = qmc.Sobol(d=D, scramble=False, bits=64)
    pts = engine.random(NS[-1])
    vals = np.exp(pts.sum(axis=1))
    ests = np.array([vals[:n].mean() for n in NS])
    return np.abs(ests - I_2D) / I_2D


def rqmc_stats(n_replicates=64):
    """Randomized QMC: independent scramblings; mean error, RMSE and a
    95% interval half-width from the replicates."""
    from scipy.stats import qmc

    errs = np.zeros(n_replicates)
    for rep in range(n_replicates):
        engine = qmc.Sobol(d=D, scramble=True, bits=64,
                           seed=np.random.default_rng([SEED, 8, rep]))
        pts = engine.random(NS[-1])
        est = np.exp(pts.sum(axis=1)).mean()
        errs[rep] = abs(est - I_2D) / I_2D
    half = 1.96 * errs.std(ddof=1) / np.sqrt(n_replicates)
    return errs.mean(), errs.std(ddof=1), half


err_mc = mc_2d_errors()
err_qmc = qmc_2d_errors()


def slope_over(ns, errs, lo, hi):
    m = (ns >= lo) & (ns <= hi)
    return np.polyfit(np.log2(ns[m]), np.log2(errs[m]), 1)[0]


slope_mc = slope_over(NS, err_mc, 2 ** 8, 2 ** 18)
slope_qmc = slope_over(NS, err_qmc, 2 ** 6, 2 ** 17)
rqmc_mean, rqmc_std, rqmc_half = rqmc_stats()
print(f"I_2d = {I_2D:.10f}")
print(f"MC:   ошибка при N=2^18 = {err_mc[-1]:.3e}, наклон {slope_mc:.2f}")
print(f"ЛПtau: ошибка при N=2^18 = {err_qmc[-1]:.3e}, наклон {slope_qmc:.2f}")
print(f"отношение ошибок MC/ЛПtau при N=2^18 = {err_mc[-1] / err_qmc[-1]:.3e}")
print(f"рандомизированный вариант (64 скремблировки): средняя ошибка "
      f"{rqmc_mean:.3e}, СКО {rqmc_std:.3e} (тяжелые хвосты: СКО/среднее "
      f"{rqmc_std / rqmc_mean:.1f}), полуширина 95%-го интервала "
      f"{rqmc_half:.3e}")
check_golden("qmc_err_N2_18", err_qmc[-1])
check_golden("mc_err_N2_18", err_mc[-1])
check_golden("slope_qmc_unscrambled", slope_qmc)

fig, ax = plt.subplots(figsize=(6, 3.6), layout="constrained")
ax.plot(NS, err_mc, "o-", markersize=4, linewidth=0.9, label="псевдослучайный МК")
ax.plot(NS, err_qmc, "s-", markersize=4, linewidth=0.9,
        label=r"ЛП$\tau$, детерминированная")
ns = np.array([1e2, 3e5])
ax.plot(ns, 2e-1 * (ns / ns[0]) ** (-0.5), "k--", linewidth=1,
        label=r"наклон $-1/2$")
ax.plot(ns, 3e-2 * (ns / ns[0]) ** (-1.0), "k:", linewidth=1,
        label=r"наклон $-1$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"число точек $N$")
ax.set_ylabel("относительная ошибка")
ax.legend(loc="lower left", fontsize=8)
fig.savefig(f"{FIGDIR}/fig-04.pdf", **SAVE_KW)

# %% [markdown]
# **Вывод:** предсказание подтвердилось. Наклон псевдослучайного метода близок
# к $-1/2$; детерминированная ЛП$\tau$-кривая идёт с наклоном около $-1$
# (измеренное значение напечатано выше) — ровно то, что обещает граница
# $O(N^{-1}\ln^2 N)$ для этого гладкого интеграла. При $N = 2^{18}$ ошибка
# ЛП$\tau$ — несколько $10^{-6}$ против $6{,}7\cdot10^{-4}$ у
# псевдослучайного метода (отношение напечатано выше). Рандомизированный
# вариант (64 скремблировки): средняя ошибка порядка $10^{-9}$ с тяжёлыми
# хвостами (СКО/среднее $\approx 8$, напечатано выше) и со статистической
# полушириной интервала — то есть скремблирование возвращает доверительный
# интервал, сохраняя выигрыш в порядке. Численная оговорка: датчик `Sobol`
# в scipy по умолчанию квантует координаты 30 битами и рисует фальшивое
# плато ошибки $\sim 2^{-30}$ — здесь использован `bits=64`; без него
# измерить наклон невозможно.
