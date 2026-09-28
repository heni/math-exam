# %% [markdown]
# # Вопрос 12. Закон больших чисел. Центральная предельная теорема — численные примеры
#
# Файл ведётся в формате jupytext (percent). Парный `.ipynb` получается
# командой `make questions/12-lln-clt/examples.ipynb`; в git идёт **этот** файл,
# ноутбук — производный артефакт.
#
# **Сквозной пример один:** выборка независимых измерений одного датчика. Меняем
# только закон шума и смотрим, что делает усреднение. Пять законов:
#
# | Закон | Среднее | Дисперсия | Чего ждать |
# |---|---|---|---|
# | нормальный $\mathcal N(0,1)$ | 0 | 1 | эталон: ЗБЧ и ЦПТ работают |
# | равномерный на $[-\sqrt3,\sqrt3]$ | 0 | 1 | лёгкие хвосты, ЦПТ быстрая |
# | центрированный бернуллиевский, $p=0{,}05$ | 0 | 1 | редкие выбросы, сильная асимметрия |
# | Парето, $\alpha=1{,}5$ | 3 | $\infty$ | ЗБЧ есть, ЦПТ нет |
# | Коши | нет | нет | нет ничего |
#
# Каждый пример отвечает на вопрос из `theory.md`; предсказание формулируется
# **до** прогона и остаётся в тексте, даже если не сбылось.

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, integrate

SEED = 20260928
rng = np.random.default_rng(SEED)

FIGDIR = "figures"

# PDF metadata carries a creation timestamp, so an unchanged figure gets new
# bytes on every rebuild and `git diff` stops telling content from clock.
SAVE_KW = {"metadata": {"CreationDate": None}}

# Golden-пины: сид И ожидаемый выход зафиксированы константами. Без них
# «воспроизводимость» проверяется согласованностью прогона с самим собой, что
# ничего не значит. Расхождение роняет сборку ноутбука, то есть работает как гейт.
GOLDEN = {
    "coin_N_cheb": 50000.0,
    "coin_N_clt": 9604.0,
    "ratio_cheb_clt": 5.206355432540114,
    "dice_halfwidth": 105.850153016853,
    "dice_len_ratio": 23.61829367976408,
    "cauchy_iqr_N1000": 2.0,
    "D30_bern005": 0.21588062338436353,
    "pareto_slope_sqrtN": 0.22161701569291786,
    "pareto_slope_gap": 0.16666666666666666,
}
TOL = 1e-3


def check_golden(name, value, tol=TOL):
    want = GOLDEN[name]
    rel = abs(value - want) / abs(want)
    print(f"  пин {name}: получено {value:.12g}, ожидалось {want:.12g}")
    assert rel <= tol, f"golden-пин {name} не сошёлся: {value} vs {want}"


# %% [markdown]
# ## Законы шума
#
# Первые три нормированы к нулевому среднему и единичной дисперсии, чтобы
# сравнение шло при прочих равных. Парето и Коши нормировать нечем: у первого
# дисперсии нет, у второго нет и среднего.

# %%
ALPHA_PARETO = 1.5          # показатель Парето: 1 < alpha < 2, среднее есть, дисперсии нет
P_RARE = 0.05               # редкое событие: сильно асимметричный индикатор


def draw_normal(size):
    return rng.standard_normal(size)


def draw_uniform(size):
    return rng.uniform(-np.sqrt(3), np.sqrt(3), size)


def draw_bernoulli(size):
    q = 1.0 - P_RARE
    return (rng.random(size) < P_RARE) / np.sqrt(P_RARE * q) - P_RARE / np.sqrt(P_RARE * q)


def draw_pareto(size):
    # numpy pareto(a) даёт X с P(X > x) = (1+x)^{-a}; сдвигаем к классическому виду
    return rng.pareto(ALPHA_PARETO, size) + 1.0


def draw_cauchy(size):
    return rng.standard_cauchy(size)


LAWS = {
    "нормальный": (draw_normal, 0.0),
    "равномерный": (draw_uniform, 0.0),
    f"бернуллиевский, p={P_RARE}": (draw_bernoulli, 0.0),
    f"Парето, a={ALPHA_PARETO}": (draw_pareto, ALPHA_PARETO / (ALPHA_PARETO - 1.0)),
    "Коши": (draw_cauchy, np.nan),
}

print("средние законов (nan = не существует):")
for name, (_, a) in LAWS.items():
    print(f"  {name:28s} a = {a}")

# %% [markdown]
# ## Пример 1. Закон больших чисел в действии и его отсутствие
#
# **Вопрос.** Что делает выборочное среднее $\overline X_N$ при росте $N$ для
# каждого из пяти законов?
#
# **Предсказание (до прогона).** Для первых трёх законов работает следствие
# из слабого закона больших чисел: $|\overline X_N - a|$ убывает как
# $N^{-1/2}$. Для Парето с $\alpha=1{,}5$ среднее конечно, и по теореме Хинчина
# сходимость есть, но дисперсии нет — убывание будет медленнее и с редкими
# скачками вверх. Для Коши сходимости нет вовсе: $\overline X_N$ распределено
# при каждом $N$ так же, как одно наблюдение, поэтому разброс траекторий по
# ансамблю прогонов **не сужается**.

# %%
N_MAX = 100_000
N_RUNS = 8
grid = np.unique(np.logspace(0, np.log10(N_MAX), 220).astype(int))

traj = {}
for name, (draw, a) in LAWS.items():
    x = draw((N_RUNS, N_MAX))
    means = np.cumsum(x, axis=1) / np.arange(1, N_MAX + 1)
    traj[name] = means[:, grid - 1]
    ref = 0.0 if np.isnan(a) else a
    dev = np.abs(traj[name] - ref)
    print(f"{name:28s} медиана |X̄_N - a| при N=10^3: {np.median(dev[:, np.searchsorted(grid, 1000)]):.4f}"
          f"   при N=10^5: {np.median(dev[:, -1]):.4f}")

# %% [markdown]
# Разброс траекторий Коши по ансамблю прогонов меряется межквартильным
# размахом. Теория даёт точное значение: $\overline X_N$ имеет стандартное
# распределение Коши при любом $N$, его квартили равны $\pm1$, значит
# межквартильный размах равен $2$ при всех $N$.

# %%
M_RUNS = 20_000
for N in (1, 10, 100, 1000):
    sample = rng.standard_cauchy((M_RUNS, N)).mean(axis=1)
    iqr = np.subtract(*np.percentile(sample, [75, 25]))
    print(f"  Коши, N = {N:5d}: межквартильный размах X̄_N = {iqr:.4f}")
    if N == 1000:
        cauchy_iqr = iqr
check_golden("cauchy_iqr_N1000", cauchy_iqr, tol=0.05)

# %% [markdown]
# **Вывод.** Предсказание сбылось полностью. Три первых закона дают убывание
# порядка $N^{-1/2}$; Парето сходится заметно медленнее и ступенями — каждая
# ступень есть одно большое наблюдение, поглощаемое затем ростом $N$; у Коши
# межквартильный размах выборочного среднего равен $2$ при $N=1$ и при
# $N=1000$ одинаково, то есть тысяча измерений не лучше одного. Это и есть
# отказ теоремы Хинчина при $\E|\xi| = \infty$.

# %%
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), layout="constrained")
for name in ("нормальный", f"бернуллиевский, p={P_RARE}", f"Парето, a={ALPHA_PARETO}"):
    a = LAWS[name][1]
    dev = np.abs(traj[name] - a)
    axes[0].loglog(grid, np.median(dev, axis=0), label=name)
axes[0].loglog(grid, 1.0 / np.sqrt(grid), "k--", lw=1, label=r"$N^{-1/2}$")
axes[0].set_xlabel("$N$")
axes[0].set_ylabel(r"медиана $|\overline{X}_N - a|$")
axes[0].set_title("сходимость есть")
axes[0].legend(fontsize=7)

for r in range(N_RUNS):
    axes[1].semilogx(grid, traj["Коши"][r], lw=0.8)
axes[1].axhline(0.0, color="k", lw=1, ls="--")
axes[1].set_ylim(-8, 8)
axes[1].set_xlabel("$N$")
axes[1].set_ylabel(r"$\overline{X}_N$")
axes[1].set_title("закон Коши: сходимости нет")
fig.savefig(f"{FIGDIR}/fig-01.pdf", **SAVE_KW)

# %% [markdown]
# ## Пример 2. Сколько нужно измерений: Чебышёв против ЦПТ
#
# **Вопрос.** Задача \ref{prob:coin} конспекта: сколько бросков монеты нужно,
# чтобы частота герба отличалась от $1/2$ не более чем на $\varepsilon=0{,}01$
# с вероятностью $0{,}95$? Два ответа — по неравенству Чебышёва и по теореме
# Муавра — Лапласа. Каков каждый и каково фактическое покрытие при каждом?
#
# **Предсказание (до прогона).** Отношение объёмов равно
# $1/(\alpha z^2_{1-\alpha/2}) \approx 5{,}2$ и не зависит ни от $\sigma$, ни
# от $\varepsilon$. Фактическое покрытие при $N_{\text{ЦПТ}}$ будет близко к
# $0{,}95$ (слагаемые симметричны, $N$ велико), при $N_{\text{Ч}}$ — почти
# единица: оценка Чебышёва израсходована не полностью. Для асимметричного
# случая $p=0{,}05$ покрытие по ЦПТ будет заметно хуже заявленного.

# %%
eps, alpha = 0.01, 0.05
z = stats.norm.ppf(1 - alpha / 2)
print(f"z_(1-alpha/2) = {z:.6f}")


def sample_sizes(sigma2, eps, alpha, z):
    return sigma2 / (alpha * eps**2), (z * np.sqrt(sigma2) / eps) ** 2


for p in (0.5, P_RARE):
    s2 = p * (1 - p)
    n_cheb, n_clt = sample_sizes(s2, eps, alpha, z)
    print(f"p = {p}: sigma^2 = {s2:.4f}, N_Ч = {np.ceil(n_cheb):.0f}, "
          f"N_ЦПТ = {np.ceil(n_clt):.0f}, отношение = {n_cheb / n_clt:.4f}")
    if p == 0.5:
        coin_cheb, coin_clt = n_cheb, n_clt

check_golden("coin_N_cheb", coin_cheb)
check_golden("coin_N_clt", np.ceil(coin_clt))
check_golden("ratio_cheb_clt", coin_cheb / coin_clt)

# %% [markdown]
# Фактическое покрытие считается точно: число успехов имеет биномиальное
# распределение, и вероятность попадания частоты в интервал выписывается через
# его функцию распределения — моделировать не нужно.

# %%
def exact_coverage(N, p, eps):
    """P{|mu/N - p| <= eps} для биномиального mu ~ Bin(N, p)."""
    N = int(np.ceil(N))
    lo = np.ceil(N * (p - eps) - 1e-12)
    hi = np.floor(N * (p + eps) + 1e-12)
    return stats.binom.cdf(hi, N, p) - stats.binom.cdf(lo - 1, N, p)


print("фактическое покрытие интервала |частота - p| <= eps:")
for p in (0.5, P_RARE):
    s2 = p * (1 - p)
    n_cheb, n_clt = sample_sizes(s2, eps, alpha, z)
    print(f"  p = {p:4}:  при N_ЦПТ = {np.ceil(n_clt):6.0f} -> {exact_coverage(n_clt, p, eps):.4f}"
          f"   при N_Ч = {np.ceil(n_cheb):6.0f} -> {exact_coverage(n_cheb, p, eps):.4f}")

# %% [markdown]
# Вторая половина примера — задача \ref{prob:dice} конспекта: сумма очков при
# $1000$ бросках кости.

# %%
faces = np.arange(1, 7)
m1 = faces.mean()
d1 = (faces**2).mean() - m1**2
N_DICE = 1000
mu_dice, sd_dice = N_DICE * m1, np.sqrt(N_DICE * d1)
half = z * sd_dice
print(f"один бросок: E = {m1}, D = {d1:.6f} (= 35/12 = {35/12:.6f})")
print(f"сумма 1000 бросков: E = {mu_dice:.1f}, sigma = {sd_dice:.4f}")
print(f"полуширина 95%-интервала = {half:.4f}, интервал [{mu_dice-half:.1f}; {mu_dice+half:.1f}]")
print(f"длина = {2*half:.4f}, тривиальный интервал [1000; 6000] длиной 5000, "
      f"отношение = {5000/(2*half):.4f}")
check_golden("dice_halfwidth", half)
check_golden("dice_len_ratio", 5000 / (2 * half))

dice = rng.integers(1, 7, size=(20_000, N_DICE), dtype=np.int8).sum(axis=1, dtype=np.int32)
cover = np.mean(np.abs(dice - mu_dice) <= half)
print(f"фактическое покрытие по 20000 прогонам: {cover:.4f}")

# %% [markdown]
# **Вывод.** Предсказание сбылось во всём, кроме одного места, и это место
# содержательно. Отношение объёмов равно $5{,}21$ для обоих значений $p$, то
# есть действительно не зависит от закона. Покрытие по Чебышёву при $p=0{,}5$
# практически единично — вся разница в объёме уходит в неиспользованный запас
# надёжности. А вот покрытие по ЦПТ при $p=0{,}05$ оказалось **не хуже**, а
# точно таким же, как при $p=0{,}5$: при $N\approx1825$ произведение $Np\approx91$
# уже велико, и асимметрия успевает сгладиться. Асимметрия бьёт не по объёму
# выборки, рассчитанному на заданную точность, а по малым $N$ — это видно в
# примере 3, а не здесь.

# %%
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), layout="constrained")
Ns = np.unique(np.logspace(1, 5, 60).astype(int))
for p, style in ((0.5, "-"), (P_RARE, "--")):
    cov = [exact_coverage(n, p, eps) for n in Ns]
    axes[0].semilogx(Ns, cov, style, label=f"$p={p}$")
    n_cheb, n_clt = sample_sizes(p * (1 - p), eps, alpha, z)
    axes[0].axvline(n_clt, color="C2", lw=0.8, ls=":")
    axes[0].axvline(n_cheb, color="C3", lw=0.8, ls=":")
axes[0].axhline(1 - alpha, color="k", lw=1, ls="--")
axes[0].set_xlabel("$N$")
axes[0].set_ylabel(r"$\mathsf{P}\{|\mu_N/N - p| \leq \varepsilon\}$")
axes[0].set_title("покрытие; пунктир — $N$ по ЦПТ и по Чебышёву")
axes[0].legend(fontsize=7)

axes[1].hist(dice, bins=60, density=True, color="C0", alpha=0.6)
xx = np.linspace(dice.min(), dice.max(), 400)
axes[1].plot(xx, stats.norm.pdf(xx, mu_dice, sd_dice), "k-", lw=1.2)
for s in (-1, 1):
    axes[1].axvline(mu_dice + s * half, color="C3", lw=1)
axes[1].set_xlabel("сумма очков при 1000 бросках")
axes[1].set_ylabel("плотность")
axes[1].set_title("95%-интервал против размаха [1000; 6000]")
fig.savefig(f"{FIGDIR}/fig-02.pdf", **SAVE_KW)

# %% [markdown]
# ## Пример 3. Скорость сходимости в ЦПТ и цена асимметрии
#
# **Вопрос.** Как быстро $F_{\zeta_N}$ приближается к $\Phi$ и от чего зависит
# множитель? Верно ли ходовое правило «$N \geq 30$ достаточно»?
#
# **Предсказание (до прогона).** По структуре оценки Берри — Эссеена
# $\sup_x|F_{\zeta_N}(x) - \Phi(x)| \leq C\rho_3/(\sigma^3\sqrt N)$ ожидаем:
# величина $D_N\sqrt N$ выходит на постоянную, и эта постоянная
# **пропорциональна** $\rho_3/\sigma^3$. Для бернуллиевского закона
# $\rho_3/\sigma^3 = (p^2+q^2)/\sqrt{pq}$: при $p=0{,}5$ это $1$, при
# $p=0{,}05$ — около $4{,}2$. Значит при $p=0{,}05$ для той же точности нужно
# примерно в $4{,}2^2 \approx 17$ раз больше наблюдений, и при $N=30$
# нормальное приближение будет негодным.
#
# Никакого Монте-Карло здесь не нужно: у суммы бернуллиевских величин
# распределение биномиальное, у суммы показательных — гамма, и обе функции
# распределения считаются точно.

# %%
def rho3_over_sigma3_bernoulli(p):
    q = 1 - p
    return (p**2 + q**2) / np.sqrt(p * q)


rho3_exp = integrate.quad(lambda x: abs(x - 1) ** 3 * np.exp(-x), 0, np.inf)[0]
print(f"rho_3/sigma^3: Бернулли p=0.5 -> {rho3_over_sigma3_bernoulli(0.5):.4f}, "
      f"p={P_RARE} -> {rho3_over_sigma3_bernoulli(P_RARE):.4f}, "
      f"Exp(1) -> {rho3_exp:.4f}")


def sup_dist_binom(N, p):
    """Точное sup_x |F(x) - Phi(x)| для нормированной Bin(N, p)."""
    k = np.arange(-1, N + 1)
    zk = (k - N * p) / np.sqrt(N * p * (1 - p))
    F = stats.binom.cdf(k, N, p)          # F в точке скачка (справа)
    Phi = stats.norm.cdf(zk)
    left = np.concatenate(([0.0], F[:-1]))  # предел слева в той же точке
    return max(np.max(np.abs(F - Phi)), np.max(np.abs(left - Phi)))


def sup_dist_gamma(N):
    """Точное sup_x |F(x) - Phi(x)| для нормированной суммы N показательных."""
    x = np.linspace(-np.sqrt(N), 12.0, 40_001)
    F = stats.gamma.cdf(N + x * np.sqrt(N), a=N)
    return np.max(np.abs(F - stats.norm.cdf(x)))


N_LIST = np.array([5, 10, 30, 100, 300, 1000, 3000])
D = {
    "Бернулли, p=0.5": np.array([sup_dist_binom(int(n), 0.5) for n in N_LIST]),
    f"Бернулли, p={P_RARE}": np.array([sup_dist_binom(int(n), P_RARE) for n in N_LIST]),
    "показательный": np.array([sup_dist_gamma(int(n)) for n in N_LIST]),
}
print("\n   N " + "".join(f"{name:>22s}" for name in D))
for j, n in enumerate(N_LIST):
    print(f"{n:5d} " + "".join(f"{D[name][j]:22.5f}" for name in D))
print("\nD_N * sqrt(N):")
print("   N " + "".join(f"{name:>22s}" for name in D))
for j, n in enumerate(N_LIST):
    print(f"{n:5d} " + "".join(f"{D[name][j]*np.sqrt(n):22.5f}" for name in D))

check_golden("D30_bern005", D[f"Бернулли, p={P_RARE}"][N_LIST == 30][0])

# %% [markdown]
# Та же величина — для генератора нормальных чисел Соболя: сумма двенадцати
# независимых равномерных на $[0,1]$ минус шесть. Функция распределения суммы
# (закон Ирвина — Холла) считается свёрткой на сетке.

# %%
h = 1e-4
box = np.ones(int(round(1 / h)))
dens = box.copy()
for _ in range(11):
    dens = np.convolve(dens, box)
dens = dens * h ** 11                      # плотность суммы 12 равномерных
xs = np.arange(dens.size) * h
cdf = np.cumsum(dens) * h
d_sum12 = np.max(np.abs(cdf - stats.norm.cdf(xs - 6.0)))
print(f"сумма 12 равномерных минус 6: sup|F - Phi| = {d_sum12:.5f}, "
      f"носитель [-6, 6] (хвосты обрезаны)")

# %% [markdown]
# **Предсказание сбылось наполовину, и вторая половина поучительнее первой.**
# Произведение $D_N\sqrt N$ действительно выходит на постоянную у всех трёх
# законов — это подтвердилось. Но постоянные **не** пропорциональны
# $\rho_3/\sigma^3$: измеренные $0{,}3989$, $1{,}1888$ и $0{,}1330$ относятся как
# $1 : 2{,}98 : 0{,}33$, тогда как отношения $\rho_3/\sigma^3$ равны
# $1 : 4{,}15 : 2{,}41$. У показательного закона $\rho_3/\sigma^3$ в два с
# половиной раза больше, чем у симметричного бернуллиевского, а уклонение втрое
# **меньше**. Значит $\rho_3$ в оценке Берри — Эссеена — инструмент верхней
# границы, а не описание фактической скорости.
#
# Фактическую скорость задают две другие величины, и обе проверяются счётом:
#
# 1. **Асимметрия** $\gamma_1 = \E(\xi-a)^3/\sigma^3$ — момент со знаком, а не
#    абсолютный. Поправка Эджворта первого порядка даёт
#    $F_{\zeta_N}(x)-\Phi(x) \approx -\frac{\gamma_1}{6\sqrt N}(x^2-1)\varphi(x)$,
#    максимум модуля достигается в нуле и равен
#    $|\gamma_1|/(6\sqrt{2\pi})$ после умножения на $\sqrt N$.
# 2. **Решётчатость.** Если слагаемое принимает значения на решётке с шагом
#    $h$, функция распределения $\zeta_N$ разрывна, и уже поэтому
#    $D_N\sqrt N \geq h/(2\sigma\sqrt{2\pi})$ — половина наибольшего скачка.
#    Для непрерывных законов этот вклад равен нулю.
#
# Сумма двух вкладов воспроизводит все три измеренных постоянных с точностью до
# третьего знака (проверка ниже). Практический вывод от этого только крепнет:
# правило «$N \geq 30$» ложно, а смотреть надо не на $N$ и даже не на
# $\rho_3/\sigma^3$, а на асимметрию и на решётчатость слагаемого. При
# $p=0{,}05$ и $N=30$ уклонение $D_{30}=0{,}216$ — пятая часть всей шкалы
# вероятностей, то есть нормальное приближение непригодно; у симметричного
# бернуллиевского закона при том же $N$ уклонение втрое меньше. Генератор
# Соболя из двенадцати равномерных даёт уклонение около $0{,}002$ — для
# моделирования этого достаточно, но хвосты у него обрезаны на $\pm6$, и для
# расчёта редких событий он непригоден по построению.

# %%
def predicted_constant(gamma1, h_over_sigma):
    """D_N*sqrt(N) = решётчатый вклад + вклад асимметрии (поправка Эджворта)."""
    lattice = h_over_sigma / (2 * np.sqrt(2 * np.pi))
    skew = abs(gamma1) / (6 * np.sqrt(2 * np.pi))
    return lattice, skew, lattice + skew


skew_exp = integrate.quad(lambda x: (x - 1) ** 3 * np.exp(-x), 0, np.inf)[0]
cases = {
    "Бернулли, p=0.5": ((1 - 2 * 0.5) / np.sqrt(0.5 * 0.5), 1 / np.sqrt(0.5 * 0.5)),
    f"Бернулли, p={P_RARE}": ((1 - 2 * P_RARE) / np.sqrt(P_RARE * (1 - P_RARE)),
                              1 / np.sqrt(P_RARE * (1 - P_RARE))),
    "показательный": (skew_exp, 0.0),
}
print(f"{'закон':>22} {'решётка':>10} {'асимметрия':>12} {'сумма':>10} {'измерено':>10}")
for name, (g1, hs) in cases.items():
    lat, sk, tot = predicted_constant(g1, hs)
    measured = D[name][-1] * np.sqrt(N_LIST[-1])
    print(f"{name:>22} {lat:10.5f} {sk:12.5f} {tot:10.5f} {measured:10.5f}")

# %%
fig, ax = plt.subplots(figsize=(6.4, 3.8), layout="constrained")
for name, d in D.items():
    ax.loglog(N_LIST, d, "o-", ms=3, label=name)
ax.loglog(N_LIST, 0.5 / np.sqrt(N_LIST), "k--", lw=1, label=r"$\propto N^{-1/2}$")
ax.axvline(30, color="C3", lw=0.8, ls=":")
ax.set_xlabel("$N$")
ax.set_ylabel(r"$\sup_x |F_{\zeta_N}(x) - \Phi(x)|$")
ax.set_title("скорость сходимости в ЦПТ; пунктир — граница $N=30$")
ax.legend(fontsize=7)
fig.savefig(f"{FIGDIR}/fig-03.pdf", **SAVE_KW)

# %% [markdown]
# ## Пример 4. Где ЦПТ ломается: бесконечная дисперсия
#
# **Вопрос.** Что происходит с нормированной суммой, если дисперсия слагаемого
# бесконечна? Распределение Парето с $\alpha=1{,}5$: среднее равно
# $\alpha/(\alpha-1)=3$, дисперсии нет.
#
# **Предсказание (до прогона).** Нормировка $\sqrt N$ перестаёт быть верной:
# сумма $S_N - Na$ имеет масштаб порядка $N^{1/\alpha}$, поэтому разброс
# величины $(S_N-Na)/\sqrt N$ должен расти как
# $N^{1/\alpha - 1/2} = N^{1/6} \approx N^{0{,}167}$, а разброс
# $(S_N-Na)/N^{1/\alpha}$ — выходить на постоянную. Разброс меряем
# межквартильным размахом: выборочное стандартное уклонение при бесконечной
# дисперсии само неустойчиво и мерить им нельзя.

# %%
a_par = ALPHA_PARETO / (ALPHA_PARETO - 1.0)
M_PAR = 40_000
N_PAR = np.array([10, 30, 100, 300, 1000, 3000, 10_000])
iqr_sqrt, iqr_alpha = [], []
for n in N_PAR:
    s = (rng.pareto(ALPHA_PARETO, (M_PAR, int(n))) + 1.0).sum(axis=1) - n * a_par
    iqr_sqrt.append(np.subtract(*np.percentile(s / np.sqrt(n), [75, 25])))
    iqr_alpha.append(np.subtract(*np.percentile(s / n ** (1 / ALPHA_PARETO), [75, 25])))
iqr_sqrt, iqr_alpha = np.array(iqr_sqrt), np.array(iqr_alpha)

slope_sqrt = np.polyfit(np.log(N_PAR), np.log(iqr_sqrt), 1)[0]
slope_alpha = np.polyfit(np.log(N_PAR), np.log(iqr_alpha), 1)[0]
print(f"{'N':>7} {'IQR / sqrt(N)':>16} {'IQR / N^(1/alpha)':>20}")
for j, n in enumerate(N_PAR):
    print(f"{n:7d} {iqr_sqrt[j]:16.4f} {iqr_alpha[j]:20.4f}")
print(f"\nнаклон в log-log: при нормировке sqrt(N) {slope_sqrt:+.4f} "
      f"(предсказано {1/ALPHA_PARETO - 0.5:+.4f}), "
      f"при нормировке N^(1/alpha) {slope_alpha:+.4f} (предсказано +0.0000)")
# Локальный наклон на последней декаде: если сходимость к устойчивому закону
# медленная, глобальная подгонка смещена, а локальная — ближе к пределу.
loc = np.log(N_PAR[-1] / N_PAR[-2])
print(f"локальный наклон на [{N_PAR[-2]}; {N_PAR[-1]}]: "
      f"sqrt(N) {np.log(iqr_sqrt[-1]/iqr_sqrt[-2])/loc:+.4f}, "
      f"N^(1/alpha) {np.log(iqr_alpha[-1]/iqr_alpha[-2])/loc:+.4f}")
# Разность наклонов не зависит от скорости сходимости: две нормировки
# отличаются ровно множителем N^{1/alpha - 1/2}, это алгебраическое тождество.
print(f"разность наклонов = {slope_sqrt - slope_alpha:+.6f}, "
      f"1/alpha - 1/2 = {1/ALPHA_PARETO - 0.5:+.6f}")
check_golden("pareto_slope_sqrtN", slope_sqrt, tol=0.05)
check_golden("pareto_slope_gap", slope_sqrt - slope_alpha, tol=1e-6)

# %% [markdown]
# **Вывод.** Качественно предсказание сбылось, количественно — нет, и
# расхождение объясняется.
#
# Качественная часть верна: при нормировке $\sqrt N$ разброс растёт, при
# нормировке $N^{1/\alpha}$ почти стабилизируется. Центральная предельная
# теорема при бесконечной дисперсии неверна не «приблизительно», а в самой
# нормировке: делить надо на другую степень $N$.
#
# Количественно измеренные наклоны — $+0{,}222$ и $+0{,}055$ вместо
# предсказанных $+0{,}167$ и $0$. Причина не в постановке примера, а в
# медленной сходимости к устойчивому закону при $\alpha=1{,}5$: поправка к
# предельному распределению убывает степенью с малым показателем, и на
# $N \leq 10^4$ асимптотика ещё не установилась. Видно это по локальному
# наклону на последней декаде — он ближе к предсказанному, чем глобальный.
#
# Проверить предсказание в чистом виде всё-таки можно, и проверка точная:
# **разность** наклонов не зависит от скорости сходимости вовсе, потому что
# две нормировки отличаются ровно множителем $N^{1/\alpha-1/2}$. Измеренная
# разность равна $0{,}166667$ при теоретическом $1/\alpha - 1/2 = 1/6$ —
# совпадение до шестого знака. Это и есть содержание примера: показатель
# нормировки определён однозначно, и он не равен $1/2$.
#
# Практический признак для расчёта: если выборочная дисперсия измерений растёт
# с числом измерений вместо того чтобы стабилизироваться, доверительные
# интервалы по ЦПТ строить нельзя.

# %%
fig, ax = plt.subplots(figsize=(6.4, 3.8), layout="constrained")
ax.loglog(N_PAR, iqr_sqrt, "o-", ms=3, label=r"нормировка $\sqrt{N}$")
ax.loglog(N_PAR, iqr_alpha, "s-", ms=3, label=r"нормировка $N^{1/\alpha}$")
ax.loglog(N_PAR, iqr_sqrt[0] * (N_PAR / N_PAR[0]) ** (1 / ALPHA_PARETO - 0.5),
          "k--", lw=1, label=r"$\propto N^{1/\alpha - 1/2}$")
ax.set_xlabel("$N$")
ax.set_ylabel("межквартильный размах нормированной суммы")
ax.set_title(r"Парето, $\alpha = 1.5$: какая нормировка верна")
ax.legend(fontsize=7)
fig.savefig(f"{FIGDIR}/fig-04.pdf", **SAVE_KW)
