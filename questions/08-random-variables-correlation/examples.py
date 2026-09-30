# %% [markdown]
# # Вопрос 08. Случайные величины и векторы. Корреляционная теория — численные примеры
#
# Пять примеров, каждый отвечает на вопрос, заданный в `theory.md`. Предсказание
# результата формулируется ДО прогона и остаётся в тексте, даже если не сбылось.
#
# 1. Два датчика с коррелированными шумами: оптимальный вес и достижимая точность.
# 2. Некоррелированность против независимости: что видит и чего не видит $r$.
# 3. Эллипс рассеяния, главные оси и грубость неравенства Чебышёва.
# 4. Численно неустойчивый счёт дисперсии.
# 5. Выборочная ковариационная матрица: скорость сходимости и вырожденность.

# %%
import numpy as np
import matplotlib.pyplot as plt

SEED = 20260930

# Отдельный генератор на пример: иначе добавленный пример сдвигает выборки всех
# следующих, и golden-пины ломаются не из-за содержания правки.
rng1, rng2, rng3, rng4, rng5 = (np.random.default_rng([SEED, k]) for k in range(1, 6))

FIGDIR = "figures"

# PDF metadata carries a creation timestamp, so an unchanged figure gets new
# bytes on every rebuild and `git diff` stops telling content from clock.
SAVE_KW = {"metadata": {"CreationDate": None}}

# Golden-пины: сид И ожидаемый выход зафиксированы константами. Без них
# «воспроизводимость» проверяется согласованностью прогона с самим собой, что
# ничего не значит. Расхождение роняет сборку ноутбука, то есть работает как гейт.
GOLDEN = {
    "w_opt_2_1_09": -0.5714285714285715,
    "v_opt_2_1_09": 0.5428571428571427,
    "v_sample_2_1_09": 0.5444266130872027,
    "r_abs_uniform_square": 0.00030647804809945736,
    "mse_lin_square": 0.0893174162753037,
    "r_abs_sign_example": 0.0030955023203494266,
    "lambda_max": 4.843074902771996,
    "frac_outside_c10": 0.0068625,
    "var_ref_nodrift": 1.0001692820071044,
    "var_naive_relerr_1e9": 1.0,
    "var_naive_f32_1e6": -131072.0,
    "cov_slope": -0.4922736655235022,
    "w_opt_equal_sigma": 0.5,
    "v_opt_r_09999": 0.0001999100359855335,
    "v_sample_rel_dev": 0.0028911293711632396,
    "nu2_binned": 0.9998757157056548,
    "cauchy_var_ratio": 5192.757000403138,
    "gauss_var_ratio": 1.03904702708531,
    "frac_inside_c95": 0.949825,
    "c95_level": 5.991464547107982,
    "sigma_frac_c10": 0.9629161576112543,
    "cov_offdiag_before": 1.8047048731722592,
}
TOL = 1e-3


def check_golden(name, value, tol=TOL):
    want = GOLDEN[name]
    rel = abs(value - want) / max(abs(want), 1e-300)
    print(f"  пин {name}: получено {value:.12g}, ожидалось {want:.12g}")
    assert rel <= tol, f"golden-пин {name} не сошёлся: {value} vs {want}"


# %% [markdown]
# ## Пример 1. Два датчика с коррелированными шумами
#
# **Вопрос.** Два прибора измеряют одну величину с погрешностями, у которых
# дисперсии $\sigma_1^2$, $\sigma_2^2$ и коэффициент корреляции $r$. Ищем
# несмещённую оценку $w\eta_1+(1-w)\eta_2$ с наименьшей дисперсией.
#
# **Предсказание** (предложение о двух датчиках в `theory.md`):
#
# * при $\sigma_1=\sigma_2$ оптимальный вес равен $1/2$ при любом $r$, а
#   дисперсия равна $\sigma^2(1+r)/2$ — то есть при $r\to1$ второй прибор не
#   даёт ничего;
# * при $\sigma_1\ne\sigma_2$ и $r\to1$ дисперсия оптимальной комбинации
#   стремится к **нулю**;
# * при $r>\sigma_2/\sigma_1$ оптимальный вес худшего прибора **отрицателен**.
#
# Проверяем формулы прямым перебором веса и моделированием.

# %%
def w_opt(s1, s2, r):
    """Optimal weight on the first sensor (theory.md, eq. wopt)."""
    return (s2**2 - r * s1 * s2) / (s1**2 + s2**2 - 2 * r * s1 * s2)


def v_opt(s1, s2, r):
    """Variance of the optimal combination (theory.md, eq. vopt)."""
    return s1**2 * s2**2 * (1 - r**2) / (s1**2 + s2**2 - 2 * r * s1 * s2)


def v_of_w(w, s1, s2, r):
    return w**2 * s1**2 + (1 - w) ** 2 * s2**2 + 2 * w * (1 - w) * r * s1 * s2


print("sigma1=2, sigma2=1 — таблица из разобранной задачи:")
for r in (0.0, 0.5, 0.9):
    w, v = w_opt(2.0, 1.0, r), v_opt(2.0, 1.0, r)
    grid = np.linspace(-3, 3, 600001)
    w_num = grid[np.argmin(v_of_w(grid, 2.0, 1.0, r))]
    print(f"  r={r:<4} w*={w:+.6f} (перебор {w_num:+.6f})  V*={v:.6f}")

check_golden("w_opt_2_1_09", w_opt(2.0, 1.0, 0.9))
check_golden("v_opt_2_1_09", v_opt(2.0, 1.0, 0.9))

# Пункты 1 и 2 предсказания тоже должны быть напечатанными числами, а не
# читаться с графика: картинка не является источником значения.
print("\nравные дисперсии (sigma1=sigma2=1): вес не должен зависеть от r")
for r in (-0.9, 0.0, 0.9):
    print(f"  r={r:+4} w*={w_opt(1.0, 1.0, r):.12f}  V*={v_opt(1.0, 1.0, r):.6f}"
          f"  против sigma^2(1+r)/2 = {(1 + r) / 2:.6f}")
check_golden("w_opt_equal_sigma", w_opt(1.0, 1.0, -0.9))

print("\nразные дисперсии (sigma1=1, sigma2=0.5): V* при r -> 1")
for r in (0.9, 0.99, 0.999, 0.9999):
    print(f"  r={r:<8} V*={v_opt(1.0, 0.5, r):.8e}")
check_golden("v_opt_r_09999", v_opt(1.0, 0.5, 0.9999))

# %%
# Моделирование: та же дисперсия должна получиться на выборке.
N1 = 400_000
s1, s2, r = 2.0, 1.0, 0.9
cov = np.array([[s1**2, r * s1 * s2], [r * s1 * s2, s2**2]])
# Cholesky, а не multivariate_normal: разложение задано формулой и одинаково во
# всех версиях numpy, тогда как способ факторизации внутри генератора менялся.
eps = rng1.standard_normal((N1, 2)) @ np.linalg.cholesky(cov).T
w = w_opt(s1, s2, r)
combo = w * eps[:, 0] + (1 - w) * eps[:, 1]
v_hat = combo.var(ddof=1)
v_true = v_opt(s1, s2, r)
rel_dev = abs(v_hat - v_true) / v_true
sd_var = np.sqrt(2.0 / (N1 - 1))  # СКО оценки дисперсии по N1 нормальным наблюдениям
print(f"дисперсия комбинации по выборке: {v_hat:.6f}")
print(f"дисперсия по формуле:            {v_true:.6f}")
print(f"относительное уклонение:         {rel_dev:.6%}")
print(f"СКО оценки дисперсии sqrt(2/(N-1)): {sd_var:.6%}  ->  уклонение "
      f"{rel_dev / sd_var:.2f} сигмы")
print(f"для сравнения, лучший прибор в одиночку: {s2**2:.6f}")
check_golden("v_sample_2_1_09", v_hat)
check_golden("v_sample_rel_dev", rel_dev)

# %% [markdown]
# **Вывод.** Выборочная дисперсия комбинации и дисперсия по формуле совпали:
# относительное уклонение, стандартное уклонение оценки дисперсии
# $\sqrt{2/(N-1)}$ и их отношение в сигмах напечатаны ячейкой выше — уклонение
# в пределах статистической погрешности. При $r=0{,}9$ оптимальный вес худшего прибора
# отрицателен ($-0{,}5714$), а дисперсия оптимальной комбинации ($0{,}5429$)
# **меньше**, чем у лучшего прибора в одиночку ($1{,}0$): сильно
# коррелированный шум частично вычитается. Предсказание сбылось.

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.4), layout="constrained")
rs = np.linspace(-0.98, 0.98, 400)
for ratio, style in ((1.0, "-"), (0.5, "--"), (0.25, ":")):
    ss1, ss2 = 1.0, ratio
    ax1.plot(rs, [w_opt(ss1, ss2, rr) for rr in rs], style,
             label=rf"$\sigma_2/\sigma_1={ratio}$")
    ax2.plot(rs, [v_opt(ss1, ss2, rr) / ss2**2 for rr in rs], style,
             label=rf"$\sigma_2/\sigma_1={ratio}$")
ax1.axhline(0.0, color="0.6", lw=0.8)
ax1.set_xlabel("коэффициент корреляции r")
ax1.set_ylabel("оптимальный вес $w^*$")
ax1.legend(fontsize=8)
ax2.axhline(1.0, color="0.6", lw=0.8)
ax2.set_xlabel("коэффициент корреляции r")
ax2.set_ylabel(r"$V^*/\sigma_2^2$")
ax2.set_yscale("log")
ax2.legend(fontsize=8)
fig.savefig(f"{FIGDIR}/fig-01-sensors.pdf", **SAVE_KW)
plt.show()

# %% [markdown]
# На левом графике видно, что при равных дисперсиях вес не зависит от $r$ и
# равен $1/2$, а при $\sigma_2<\sigma_1$ он уходит в отрицательную область,
# как только $r$ превосходит $\sigma_2/\sigma_1$. На правом — что при
# различных дисперсиях $V^*$ падает к нулю при $r\to1$, а при равных
# дисперсиях (сплошная линия) — наоборот, растёт к $\sigma^2$.

# %% [markdown]
# ## Пример 2. Некоррелированность против независимости
#
# **Вопрос.** Сколько дисперсии снимает наилучший линейный прогноз и сколько —
# наилучший прогноз вообще, когда связь между величинами функциональная, но
# нелинейная?
#
# **Предсказание.** Для $\xi$ равномерной на $[-1,1]$ и $\eta=\xi^2$:
# $\cov=0$, значит $r=0$ и линейный прогноз снимает $r^2=0$ дисперсии; а
# корреляционное отношение равно единице, то есть наилучший прогноз снимает
# всю дисперсию. Теоретическая дисперсия $\eta$ равна
# $\mathsf E\xi^4-(\mathsf E\xi^2)^2 = 1/5-1/9 = 4/45$.

# %%
N2 = 400_000
xi = rng2.uniform(-1.0, 1.0, size=N2)
eta = xi**2

r_sample = np.corrcoef(xi, eta)[0, 1]
b = np.cov(xi, eta, ddof=1)[0, 1] / xi.var(ddof=1)
a = eta.mean() - b * xi.mean()
mse_lin = np.mean((eta - a - b * xi) ** 2)

# Условное среднее оцениваем по бинам, а не подставляем xi**2: подстановка дала бы
# тождественный нуль и не отличила бы верный расчёт от неверного.
NBINS = 200
edges = np.linspace(-1.0, 1.0, NBINS + 1)
idx = np.clip(np.digitize(xi, edges) - 1, 0, NBINS - 1)
cnt = np.bincount(idx, minlength=NBINS)
ssum = np.bincount(idx, weights=eta, minlength=NBINS)
binmean = np.divide(ssum, cnt, out=np.zeros(NBINS), where=cnt > 0)
mse_cond = np.mean((eta - binmean[idx]) ** 2)

var_eta = eta.var(ddof=1)
print(f"выборочный r(xi, eta)        = {r_sample:+.6f}")
print(f"дисперсия eta (выборочная)   = {var_eta:.6f}  (теория 4/45 = {4/45:.6f})")
print(f"ошибка наилучшего линейного  = {mse_lin:.6f}")
print(f"ошибка по условным средним   = {mse_cond:.3e}  ({NBINS} бинов)")
print(f"доля, снятая линейным        = {1 - mse_lin / var_eta:.3e}   (r^2 = {r_sample**2:.3e})")
print(f"доля, снятая условным (nu^2) = {1 - mse_cond / var_eta:.6f}")
check_golden("r_abs_uniform_square", abs(r_sample))
check_golden("mse_lin_square", mse_lin)
check_golden("nu2_binned", 1 - mse_cond / var_eta)

# %% [markdown]
# **Вывод.** Выборочный коэффициент корреляции порядка $10^{-4}$, то есть
# линейный прогноз снимает порядка $10^{-8}$ дисперсии — ничего: остаточная
# ошибка линейного прогноза совпала с полной выборочной дисперсией, а та — с
# теоретической $4/45=0{,}088889$. Корреляционное отношение оценено по условным
# средним в $200$ бинах, а не подстановкой $\xi^2$: так единица получается
# **измеренной**, а остаточная ошибка по бинам равна разбросу $\eta$ внутри
# бина, то есть величине порядка квадрата ширины бина, а не нулю по
# построению. Разрыв между $r^2$ и $\nu^2$ здесь максимален. Предсказание
# сбылось.

# %%
# Второй контрпример: гауссовская пара со случайным знаком.
N2b = 400_000
z = rng2.standard_normal(N2b)
s = rng2.choice([-1.0, 1.0], size=N2b)
y = s * z
print(f"r(z, y)      = {np.corrcoef(z, y)[0, 1]:+.6f}   (ожидается ~0)")
print(f"r(z^2, y^2)  = {np.corrcoef(z**2, y**2)[0, 1]:+.6f}   (ожидается ровно 1)")
print(f"доля |z|=|y| = {np.mean(np.isclose(np.abs(z), np.abs(y))):.6f}")
# Обе компоненты нормальны: сверим выборочные моменты с N(0,1).
print(f"среднее y = {y.mean():+.5f}, дисперсия y = {y.var(ddof=1):.5f}")
check_golden("r_abs_sign_example", abs(np.corrcoef(z, y)[0, 1]))

# %% [markdown]
# **Вывод.** Обе компоненты стандартные нормальные, корреляция нулевая, а
# зависимость полная: $|\xi|=|\eta|$ на всей выборке, и корреляция квадратов
# равна единице. Значит вектор $(\xi,\eta)$ не гауссовский — иначе из нулевой
# корреляции следовала бы независимость. Нормальности каждой компоненты для
# теоремы о равносильности недостаточно.

# %%
# Посылка о конечных вторых моментах: что бывает без неё (ссылка из theory.md,
# раздел о конечности моментов).
#
# Предсказание: у гауссовой выборки выборочная дисперсия при росте N
# устаканивается около единицы; у коши-выборки предела нет — величина
# определяется несколькими наибольшими по модулю наблюдениями и потому прыгает
# на порядки при каждом их появлении.
cauchy = rng2.standard_cauchy(2_000_000)
gauss0 = rng2.standard_normal(2_000_000)
sizes = (10**3, 10**4, 10**5, 10**6, 2 * 10**6)
print("выборочная дисперсия при росте N:")
print("        N        Коши          гаусс")
for n_ in sizes:
    print(f"  {n_:9d}  {cauchy[:n_].var(ddof=1):12.4e}  {gauss0[:n_].var(ddof=1):9.6f}")
c_var = np.array([cauchy[:n_].var(ddof=1) for n_ in sizes])
g_var = np.array([gauss0[:n_].var(ddof=1) for n_ in sizes])
c_ratio = c_var.max() / c_var.min()
g_ratio = g_var.max() / g_var.min()
print(f"\nотношение наибольшей выборочной дисперсии к наименьшей по этим пяти N:")
print(f"  Коши:  {c_ratio:.2f}")
print(f"  гаусс: {g_ratio:.4f}")
check_golden("cauchy_var_ratio", c_ratio)
check_golden("gauss_var_ratio", g_ratio)

# %% [markdown]
# **Вывод.** У гауссовой выборки отношение наибольшей выборочной дисперсии к
# наименьшей по пяти объёмам равно $1{,}039$, то есть разброс в пределах четырёх
# процентов; у коши-выборки то же отношение равно $5{,}2\cdot10^{3}$, почти
# четыре порядка, и значения не сглаживаются с ростом $N$, а прыгают
# (в напечатанной таблице $7{,}2\cdot10^{3}$, $8{,}6\cdot10^{3}$,
# $3{,}7\cdot10^{7}$, $1{,}8\cdot10^{7}$, $9{,}1\cdot10^{6}$).
# Предсказание сбылось. Формула выборочной дисперсии (и выборочной ковариации)
# считается в обоих случаях и в обоих случаях выдаёт число — но во втором
# случае утверждать про это число нечего: сходиться ему не к чему, потому что
# второго момента не существует.

# %% [markdown]
# ## Пример 3. Эллипс рассеяния, главные оси и грубость неравенства Чебышёва
#
# **Вопрос.** Насколько груба оценка $\mathsf P\{Q\ge c\}\le n/c$ для
# квадратичной формы $Q=(\xi-a)^{\mathsf T}\Gamma^{-1}(\xi-a)$?
#
# **Предсказание.** Для двумерного нормального закона $Q$ имеет распределение
# $\chi^2$ с двумя степенями свободы, то есть $\mathsf P\{Q\ge c\}=e^{-c/2}$.
# При $c=10$ это $e^{-5}\approx0{,}0067$ против оценки $2/c=0{,}2$ — разрыв
# примерно в тридцать раз. Отдельно печатается уровень $c=-2\ln0{,}05$, вне
# которого остаётся ровно $5\,\%$ массы: он же стоит на легенде картинки.

# %%
N3 = 400_000
Gamma = np.array([[4.0, 1.8], [1.8, 1.0]])
a_vec = np.array([1.0, -0.5])
lam, Q = np.linalg.eigh(Gamma)
order = np.argsort(lam)[::-1]
lam, Q = lam[order], Q[:, order]
print(f"собственные значения: {lam[0]:.6f}, {lam[1]:.6f}")
print(f"след = {np.trace(Gamma):.6f} = сумма собственных = {lam.sum():.6f}")
print(f"r(компонент) = {Gamma[0,1]/np.sqrt(Gamma[0,0]*Gamma[1,1]):.4f}")
check_golden("lambda_max", lam[0])

sample = a_vec + rng3.standard_normal((N3, 2)) @ np.linalg.cholesky(Gamma).T
d = sample - a_vec
Qform = np.einsum("ij,jk,ik->i", d, np.linalg.inv(Gamma), d)

C95 = -2.0 * np.log(0.05)  # уровень, вне которого остаётся ровно 5 % массы
print(f"\nуровень 95 %: c = -2 ln 0.05 = {C95:.4f}")
print("\n     c   доля вне эллипса   точно exp(-c/2)   Чебышёв 2/c")
for c in (1.0, 2.0, 5.0, C95, 10.0):
    frac = np.mean(Qform >= c)
    print(f"{c:6.4f}   {frac:.6f}          {np.exp(-c/2):.6f}        {2/c:.6f}")
frac10 = np.mean(Qform >= 10.0)
check_golden("frac_outside_c10", frac10)
check_golden("frac_inside_c95", 1.0 - np.mean(Qform >= C95))
check_golden("c95_level", C95)

p10 = np.exp(-5.0)
sd_frac = np.sqrt(p10 * (1 - p10) / N3)  # СКО выборочной доли при c = 10
print(f"\nпри c=10: уклонение доли от точного {abs(frac10 - p10):.3e}, "
      f"СКО доли sqrt(p(1-p)/N) = {sd_frac:.3e}  ->  {abs(frac10 - p10)/sd_frac:.2f} сигмы")
print(f"отношение оценки Чебышёва к точному значению при c=10: {0.2/p10:.2f}")
check_golden("sigma_frac_c10", abs(frac10 - p10) / sd_frac)

# %%
# Некоррелированность в главных осях: поворот Q^T убирает ковариацию.
zeta = d @ Q
cov_before = np.cov(d.T, ddof=1)
cov_after = np.cov(zeta.T, ddof=1)
print(f"выборочная ковариационная матрица до поворота:\n{cov_before}")
print(f"она же в главных осях:\n{cov_after}")
print(f"внедиагональный элемент: {cov_before[0,1]:.6f} -> {cov_after[0,1]:.3e}")
print(f"ожидались дисперсии {lam[0]:.4f} и {lam[1]:.4f} при нулевой ковариации")
check_golden("cov_offdiag_before", cov_before[0, 1])

# %%
fig, ax = plt.subplots(figsize=(5.4, 4.2), layout="constrained")
sub = sample[:3000]
ax.plot(sub[:, 0], sub[:, 1], ".", ms=1.6, alpha=0.35, color="0.35")
t = np.linspace(0, 2 * np.pi, 400)
for c, style, lab in ((1.0, "-", "c = 1"), (C95, "--", f"c = {C95:.2f} (95 %)")):
    pts = a_vec[:, None] + Q @ (np.sqrt(c * lam)[:, None] * np.vstack([np.cos(t), np.sin(t)]))
    ax.plot(pts[0], pts[1], style, lw=1.6, label=lab)
for k in (0, 1):
    v = Q[:, k] * np.sqrt(lam[k])
    ax.annotate("", xy=a_vec + v, xytext=a_vec,
                arrowprops=dict(arrowstyle="->", lw=1.4, color="C3"))
ax.set_aspect("equal")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.legend(fontsize=8, loc="upper left")
fig.savefig(f"{FIGDIR}/fig-03-ellipse.pdf", **SAVE_KW)
plt.show()

# %% [markdown]
# **Вывод.** Измеренная доля точек вне эллипса совпала с $e^{-c/2}$ во всех
# пяти строках, включая уровень $95\,\%$. Уклонение при $c=10$, стандартное
# уклонение доли $\sqrt{p(1-p)/N}$ и их отношение в сигмах напечатаны ячейкой
# выше; оценка Чебышёва при том же $c$ даёт $0{,}2$, и отношение её к точному
# значению тоже напечатано. Предсказание сбылось.
# Стрелки на картинке — главные оси длиной $\sqrt{\lambda_k}$; поворот в эти
# оси обнулил выборочную ковариацию: её значения до и после поворота
# напечатаны выше.

# %% [markdown]
# ## Пример 4. Численно неустойчивый счёт дисперсии
#
# **Вопрос.** Что делает с точностью однопроходная формула
# $\mathsf D\xi=\mathsf E\xi^2-(\mathsf E\xi)^2$, если данные сдвинуты на
# большую константу?
#
# **Предсказание.** Оба слагаемых имеют порядок $c^2$, а разность — порядок
# дисперсии. При $c=10^9$ и двойной точности ($\approx16$ значащих цифр)
# величина $c^2=10^{18}$ округляется с абсолютной погрешностью порядка сотен,
# то есть в сотни раз больше самой дисперсии: верных цифр не останется вовсе,
# и результат может выйти отрицательным.

# %%
def var_naive(x):
    """One-pass formula: catastrophic cancellation when the data are shifted.

    The accumulator dtype is pinned to the data dtype on purpose: with a float64
    accumulator over float32 data the same expression yields a different answer,
    so leaving it to the numpy default would make the printed numbers depend on
    an implementation detail rather than on the precision under discussion.
    """
    dt = x.dtype
    n = x.size
    mean_sq = np.add.reduce(x * x, dtype=dt) / dt.type(n)
    mean = np.add.reduce(x, dtype=dt) / dt.type(n)
    return float(mean_sq - mean * mean)


def var_twopass(x):
    m = np.mean(x)
    return float(np.mean((x - m) ** 2))


def var_welford(x):
    """Welford's online update: works with deviations, hence stable."""
    mean = 0.0
    m2 = 0.0
    for i, xi_ in enumerate(x, start=1):
        delta = xi_ - mean
        mean += delta / i
        m2 += delta * (xi_ - mean)
    return m2 / len(x)


N4 = 200_000
base = rng4.standard_normal(N4)
print(" сдвиг      наивная          двухпроходная     Уэлфорд")
for shift in (0.0, 1e4, 1e8, 1e9):
    x = base + shift
    print(f"{shift:8.0e}   {var_naive(x):+16.8f}   {var_twopass(x):.8f}   {var_welford(x):.8f}")

ref = var_twopass(base)
bad = var_naive(base + 1e9)
print(f"\nэталон (двухпроходная, без сдвига): {ref:.8f}")
print(f"наивная при сдвиге 1e9:             {bad:+.8f}")
print(f"относительная ошибка наивной:       {abs(bad - ref) / ref:.6f}")
check_golden("var_ref_nodrift", ref)
check_golden("var_naive_relerr_1e9", abs(bad - ref) / ref)

# Одинарная точность: та же беда наступает на сдвигах в сто тысяч раз меньших.
print("\nодинарная точность, наивная формула:")
for shift in (1e4, 1e5, 1e6):
    x32 = (base + shift).astype(np.float32)
    print(f"  сдвиг {shift:8.0e}: {var_naive(x32):+14.4f}")
check_golden("var_naive_f32_1e6", var_naive((base + 1e6).astype(np.float32)))

# %% [markdown]
# **Вывод.** Двухпроходная формула и формула Уэлфорда держат восемь верных
# цифр при любом сдвиге. Наивная теряет их все: при $c=10^8$ она даёт ровно
# $2{,}0$ вместо $1{,}0002$, при $c=10^9$ — ровно $0$, то есть относительную
# ошибку $1$. Предсказание сбылось. Знак результата от данных зависит, и в
# одинарной точности он оказывается отрицательным уже при сдвиге $10^4$:
# напечатаны $-8$, $-1024$ и $-131072$ — отрицательная дисперсия, полученная
# из неотрицательного по построению выражения. Вывод для кода: ковариации
# считать по отклонениям, а не по сырым моментам.

# %% [markdown]
# ## Пример 5. Выборочная ковариационная матрица
#
# **Вопрос.** С какой скоростью $\widehat\Gamma$ сходится к $\Gamma$ и что
# происходит при числе наблюдений меньше размерности?
#
# **Предсказание.** Элементы $\widehat\Gamma$ — выборочные средние, поэтому по
# центральной предельной теореме их уклонения имеют порядок $N^{-1/2}$;
# значит наклон в логарифмических осях равен $-1/2$. При $N\le n$ матрица
# обязана быть вырожденной: её ранг не превосходит $N-1$.

# %%
n5 = 8
A5 = rng5.standard_normal((n5, n5))
Gamma5 = A5 @ A5.T + n5 * np.eye(n5)
chol5 = np.linalg.cholesky(Gamma5)
norm_G = np.linalg.norm(Gamma5, "fro")
print(f"число обусловленности Gamma: {np.linalg.cond(Gamma5):.2f}")


def sample_cov(N, gen):
    z = gen.standard_normal((N, n5))
    x = z @ chol5.T
    return np.cov(x.T, ddof=1)


Ns = np.array([16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
REP = 24
errs, sds = [], []
for N in Ns:
    e = [np.linalg.norm(sample_cov(int(N), rng5) - Gamma5, "fro") / norm_G for _ in range(REP)]
    errs.append(np.mean(e))
    sds.append(np.std(e, ddof=1))
errs, sds = np.array(errs), np.array(sds)
slope = np.polyfit(np.log(Ns), np.log(errs), 1)[0]
for N, e, sd in zip(Ns, errs, sds):
    print(f"  N={N:5d}   средняя по {REP} повторениям относительная ошибка "
          f"{e:.5f} (СКО по повторениям {sd:.5f})")
print(f"\nнаклон в логарифмических осях: {slope:.4f} (предсказано -0.5); "
      f"уклонение {abs(slope + 0.5)/0.5:.2%} от предсказанного")
check_golden("cov_slope", slope)

# %%
for N in (4, 8, 9, 20):
    G_hat = sample_cov(N, rng5)
    print(f"  N={N:3d}, n={n5}: ранг {np.linalg.matrix_rank(G_hat)}, "
          f"наименьшее собственное значение {np.linalg.eigvalsh(G_hat)[0]:+.3e}")

# %%
fig, ax = plt.subplots(figsize=(5.6, 3.6), layout="constrained")
ax.errorbar(Ns, errs, yerr=sds, fmt="o-", capsize=3,
            label=f"средняя по {REP} повторениям")
ax.set_xscale("log")
ax.set_yscale("log")
ax.loglog(Ns, errs[0] * (Ns / Ns[0]) ** -0.5, "--", label=r"наклон $-1/2$")
ax.set_xlabel("объём выборки N")
ax.set_ylabel(r"средняя $\|\widehat\Gamma-\Gamma\|_F/\|\Gamma\|_F$")
ax.legend(fontsize=8)
fig.savefig(f"{FIGDIR}/fig-05-samplecov.pdf", **SAVE_KW)
plt.show()

# %% [markdown]
# **Вывод.** Измеренный наклон против предсказанного $-1/2$: уклонение
# напечатано ячейкой выше вместе с разбросом по повторениям, так что видно, на
# сколько оно значимо. «Совпало до второго знака» было бы завышением —
# округление до двух знаков даёт $-0{,}49$ против $-0{,}50$. При $N=4$ и $N=8$ ранг выборочной матрицы равен
# $3$ и $7$ — то есть ровно $N-1$, как и утверждает оценка ранга; наименьшее
# собственное значение при этом порядка $10^{-14}$, то есть численный нуль, и
# формулы наилучшей линейной оценки к такой матрице неприменимы буквально.
# При $N=9$ матрица уже невырождена. Предсказание сбылось.
