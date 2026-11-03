# %% [markdown]
# # Вопрос 16. PCA и ICA — численные примеры
#
# Файл ведётся в формате jupytext (percent). Парный `.ipynb` получается
# командой `make questions/16-pca-ica/examples.ipynb`.
#
# Каждый пример отвечает на вопрос из `theory.md`. Сквозная постановка —
# один на весь файл: многоканальная регистрация. Три независимых источника
# сигнала (синусоидальный, прямоугольный и пилообразный) регистрируются
# тремя датчиками, каждый из которых принимает линейную смесь источников
# с фиксированной матрицей смешения $A$. На одних и тех же данных: PCA
# (пример 1), ICA / FastICA (пример 2), гауссовский контрпример (пример 3),
# меры негауссовости (пример 4).

# %%
import matplotlib.pyplot as plt
import numpy as np

SEED = 20261103

FIGDIR = "figures"

# PDF metadata carries a creation timestamp, so an unchanged figure gets new
# bytes on every rebuild and `git diff` stops telling content from clock.
SAVE_KW = {"metadata": {"CreationDate": None}}

# Golden pins: the seed and the expected outputs are fixed constants. Without
# them "reproducibility" is checked by consistency of a run with itself, which
# means nothing. A mismatch fails the notebook build, so it works as a gate.
GOLDEN = {
    # example 1 (PCA and Eckart-Young)
    "recon_err_q2": 75.1401967916,
    "rho_1": 0.96397676118,
    # example 2 (FastICA vs sklearn reference)
    "corr_own_min": 0.99999764028,
    "corr_sklearn_min": 0.999999234516,
    "iters_own_total": 10,
    # example 3 (Gaussian counterexample)
    "angle_gauss_deg": 15.5172173741,
    # example 4 (non-Gaussianity measures)
    "kurt_laplace": 2.97548679519,
    "kurt_unif": -1.19728574426,
}
TOL = 1e-3


def check_golden(name, value, tol=TOL):
    if GOLDEN[name] is None:
        print(f"  пин {name}: получено {value:.12g} (не задан — заполнить)")
        return
    want = GOLDEN[name]
    rel = abs(value - want) / abs(want)
    print(f"  пин {name}: получено {value:.12g}, ожидалось {want:.12g}")
    assert rel <= tol, f"golden-пин {name} не сошёлся: {value} vs {want}"


# %% [markdown]
# ## Сквозная постановка: три источника, три датчика
#
# Источники: $s_1(t) = \sin(2\pi f_1 t)$ (синусоида),
# $s_2(t) = \operatorname{sign}\sin(2\pi f_2 t)$ (меандр),
# $s_3(t) = 2\,(t f_3 \bmod 1) - 1$ (пила). Частоты разнесены, чтобы спектры
# не перекрывались. Датчики регистрируют смеси $x_i = \sum_j a_{ij} s_j$ с
# фиксированной невырожденной матрицей $A$ умеренной
# обусловленности (по факту прогона cond$(A) \approx 15{,}6$ — см. печать ниже). Данные
# центрируются (см. замечание к теореме о главных компонентах в theory.md).

# %%
N = 1000
T = np.linspace(0, 1, N, endpoint=False)

S = np.column_stack(
    [
        np.sin(2 * np.pi * 5 * T),
        np.sign(np.sin(2 * np.pi * 11 * T + 0.3)),
        2 * ((8 * T) % 1) - 1,
    ]
)

rng_mix = np.random.default_rng([SEED, 0])
A_true = rng_mix.normal(0.0, 1.0, (3, 3))
A_true = A_true + 2.0 * np.eye(3)  # dominant diagonal, full mixing
X = S @ A_true.T

Xc = X - X.mean(axis=0)
Sc = S - S.mean(axis=0)

print(f"cond(A_true) = {np.linalg.cond(A_true):.3f}")
print(f"max |corr источников| = {np.abs(np.corrcoef(Sc.T) - np.eye(3)).max():.3f}")


# %% [markdown]
# ## Пример 1. PCA: теорема Эккарта — Янга на данных и слепое место PCA
#
# **Вопрос (theory.md, теорема Эккарта — Янга в норме Фробениуса и
# теорема о главных компонентах через SVD):** равна ли
# фактическая ошибка $q$-мерной реконструкции центрированных смесей хвосту
# сингулярного спектра $\sum_{j>q}\sigma_j^2$, и действительно ли главные
# направления не совпадают с направлениями источников (некоррелированность
# не различает вращений, лемма о неопределённости вращения)?
#
# **Предсказание до прогона.** Сингулярный спектр смесей имеет три выраженных
# значения; при $q = 2$ ошибка реконструкции должна совпасть с $\sigma_3^2$
# до машинного нуля. Главные направления — собственные векторы выборочной
# ковариации смесей; источники между собой почти ортогональны, но матрица $A$
# не ортогональна, поэтому ковариация смесей анизотропна и главные направления
# систематически отклоняются от направлений источников: углы заметны, и ни
# одна главная компонента не восстанавливает отдельный источник.

# %%
U_x, d_x, Vt_x = np.linalg.svd(Xc, full_matrices=False)


def recon_err(Xc, Vt_q):
    """Frobenius error of rank-q PCA reconstruction (theory.md, thm:pca-svd)."""
    H = Vt_q.T @ Vt_q
    return float(np.linalg.norm(Xc - Xc @ H, "fro") ** 2)


err_q2 = recon_err(Xc, Vt_x[:2])
tail_q2 = float(np.sum(d_x[2:] ** 2))
print(f"q=2: ошибка реконструкции {err_q2:.10f}, хвост спектра {tail_q2:.10f}")
check_golden("recon_err_q2", err_q2)

rho_1 = float(d_x[0] ** 2 / np.sum(d_x**2))
print(f"доля дисперсии первой компоненты rho_1 = {rho_1:.6f}")
check_golden("rho_1", rho_1)

# principal directions vs source directions (angles between lines)
V_pc = Vt_x.T
A_dir = A_true / np.linalg.norm(A_true, axis=0, keepdims=True)
cos_angles = np.clip(np.abs(V_pc.T @ A_dir), 0, 1)
angles_deg = np.degrees(np.arccos(cos_angles))
print("углы между главными направлениями и направлениями источников (град):")
print(np.array2string(angles_deg, precision=1))

# %% [markdown]
# Сингулярный спектр структурированных смесей сравним со спектром тех же
# данных со случайно перемешанными временными метками по каждому столбцу:
# перемешивание уничтожает временную структуру, и спектр схлопывается к
# «шумовому» (тот же приём сравнения со шумовым эталоном — [16]).
#
# **Предсказание:** у перемешанных данных сингулярные числа сблизятся, доля
# $\rho_1$ упадёт, «локоть» спектра исчезнет.

# %%
rng_perm = np.random.default_rng([SEED, 1])
X_perm = Xc.copy()
for j in range(X_perm.shape[1]):
    X_perm[:, j] = X_perm[rng_perm.permutation(N), j]
d_perm = np.linalg.svd(X_perm, compute_uv=False)

print(f"sigma(Xc)   = {np.array2string(d_x, precision=2)}")
print(f"sigma(perm) = {np.array2string(d_perm, precision=2)}")

fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))

axes[0].plot(T, S[:, 0], lw=0.8)
axes[0].set_title("источник 1 (синус)")
axes[0].set_xlim(0, 0.5)
axes[1].plot(T, X[:, 1], lw=0.8, color="darkred")
axes[1].set_title("смесь на датчике 2")
axes[1].set_xlim(0, 0.5)
axes[2].plot(d_x, "o-", label="смеси")
axes[2].plot(d_perm, "s--", label="перемешанные")
axes[2].set_title("сингулярный спектр")
axes[2].set_xlabel("индекс")
axes[2].legend()

for ax in axes[:2]:
    ax.set_xlabel("t")
fig.tight_layout()
fig.savefig(f"{FIGDIR}/fig-01.pdf", **SAVE_KW)
plt.close(fig)


# %% [markdown]
# ## Пример 2. FastICA: разделение источников
#
# **Вопрос (theory.md, раздел «Алгоритм FastICA»):** восстанавливает ли
# итерация фиксированной точки $w^{+} = \E[Z g(w^{\T}Z)] -
# \E[g'(w^{\T}Z)]\,w$ с $g = \tanh$ источники по центрированным смесям, и
# совпадает ли результат с эталонной реализацией scikit-learn?
#
# **Предсказание до прогона.** После отбеливания матрица смешения становится
# ортогональной (предложение об отбеливании), и независимые проекции
# возможны только вдоль осей источников (теорема идентифицируемости ICA);
# значит, FastICA сойдётся к $W = A^{-1}$ с точностью до перестановки и знака
# (нормировка дисперсий снимает свободу масштаба). Меры качества: (1) для каждого восстановленного сигнала максимум
# $|\operatorname{corr}(\hat s_j, s_k)|$ по $k$ близок к единице; (2) матрица
# $W M A$ (разделение · отбеливание · смешение) близка к матрице перестановки
# со знаками; (3) своя реализация и sklearn дают одни и те же компоненты с
# точностью до знака и перестановки. Итераций ожидаем несколько десятков.

# %%
def whiten(Xc):
    """Whitening via SVD of the centered data (prop:whiten).

    Xc = U diag(d) V^T, sample covariance V diag(d^2/N) V^T, so
    M = diag(sqrt(N)/d) V^T whitens: rows of Xc @ M.T = sqrt(N) U.
    """
    U, d, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = np.sqrt(N) * U
    M = (np.sqrt(N) / d)[:, None] * Vt
    return Z, M


def fastica_one(Z, w0, W_prev=None, max_iter=200, tol=1e-10):
    """One-unit FastICA fixed-point iteration with tanh nonlinearity.

    With W_prev given, every iterate is orthogonalized against the already
    found rows (deflation inside the iteration), otherwise the unit can
    converge to a recovered direction and leave nothing for the orthogonal
    complement.
    """
    w = w0 / np.linalg.norm(w0)
    if W_prev is not None and len(W_prev):
        w = w - W_prev.T @ (W_prev @ w)
        w /= np.linalg.norm(w)
    it = 0
    for it in range(1, max_iter + 1):
        g = np.tanh(Z @ w)
        gp = 1.0 - g**2
        w_new = (Z * g[:, None]).mean(axis=0) - gp.mean() * w
        if W_prev is not None and len(W_prev):
            w_new = w_new - W_prev.T @ (W_prev @ w_new)
        nrm = np.linalg.norm(w_new)
        if nrm < 1e-12:
            break
        w_new /= nrm
        if abs(abs(w @ w_new) - 1.0) < tol:
            w = w_new
            break
        w = w_new
    return w, it


def fastica(Z, n_comp, rng):
    """Sequential (deflation) FastICA with Gram-Schmidt orthogonalization."""
    W = np.zeros((n_comp, Z.shape[1]))
    total_iters = 0
    for j in range(n_comp):
        w0 = rng.normal(0.0, 1.0, Z.shape[1])
        w, iters = fastica_one(Z, w0, W_prev=W[:j])
        W[j] = w
        total_iters += iters
        print(f"  компонента {j + 1}: сошлась за {iters} итераций")
    return W, total_iters


Z, M_white = whiten(Xc)

rng_ica = np.random.default_rng([SEED, 2])
W_own, iters_total = fastica(Z, 3, rng_ica)
S_hat_own = Z @ W_own.T
print(f"суммарное число итераций: {iters_total}")
check_golden("iters_own_total", float(iters_total))

# recovered mixing: W M A must be close to a signed permutation
WA = W_own @ M_white @ A_true
print("W_own · M · A_true (должна быть ≈ матрица перестановки со знаками):")
print(np.array2string(WA, precision=3))


def best_source_corr(S_hat, S):
    """For each recovered row: max_k |corr| with the sources (ICA quality)."""
    C = np.corrcoef(np.hstack([S_hat, S]).T)
    n = S_hat.shape[1]
    return np.abs(C[:n, n:]).max(axis=1)


corr_own = best_source_corr(S_hat_own, Sc)
print(f"max-corr по источникам (своя реализация): {np.array2string(corr_own, precision=6)}")
check_golden("corr_own_min", corr_own.min())

# %% [markdown]
# Тот же эксперимент эталонной реализацией — для сверки.

# %%
from sklearn.decomposition import FastICA as SkFastICA

ica_sk = SkFastICA(n_components=3, whiten="arbitrary-variance", fun="logcosh",
                   max_iter=2000, tol=1e-10, random_state=SEED)
S_hat_sk = ica_sk.fit_transform(Xc)

corr_sk = best_source_corr(S_hat_sk, Sc)
print(f"max-corr по источникам (sklearn): {np.array2string(corr_sk, precision=6)}")
check_golden("corr_sklearn_min", corr_sk.min())

C = np.corrcoef(np.hstack([S_hat_own, S_hat_sk]).T)[:3, 3:]
for j in range(3):
    k = int(np.argmax(np.abs(C[j])))
    print(f"  own[{j}] ↔ sk[{k}]: corr {C[j, k]:+.9f}")

fig, axes = plt.subplots(3, 1, figsize=(9, 5.5), sharex=True)
for j, ax in enumerate(axes):
    ax.plot(T, Sc[:, j], lw=0.9, label="источник")
    k = int(np.argmax(np.abs(C[:, j])))
    est = S_hat_own[:, k]
    ax.plot(T, est / np.std(est), lw=0.9, alpha=0.75, ls="--", label="ICA-оценка")
    ax.set_ylabel(f"s_{j + 1}")
    ax.set_xlim(0, 0.5)
axes[0].legend(loc="upper right")
axes[-1].set_xlabel("t")
fig.suptitle("FastICA: источники и восстановленные компоненты")
fig.tight_layout()
fig.savefig(f"{FIGDIR}/fig-02.pdf", **SAVE_KW)
plt.close(fig)

# %% [markdown]
# ## Пример 3. Гауссовский контрпример: почему требуется негауссовость
#
# **Вопрос (theory.md, лемма о неопределённости вращения и теорема
# идентифицируемости ICA):** если источники гауссовы, разделение принципиально
# невозможно — любой поворот отбеленных данных снова даёт независимые
# гауссовы компоненты. Проверим: прогоним ту же схему на трёх независимых
# гауссовских источниках с той же матрицей смешения.
#
# **Предсказание до прогона.** Матрица $W M A$ не приблизится к перестановочной:
# в пространстве источников строки $W M A$ — произвольные ортогональные
# направления, угол между ними и осями источников останется большим (порядка
# десятков градусов). Это не дефект алгоритма, а следствие
# неидентифицируемости: по ковариации $I$ и условию независимости гауссовых
# компонентов повороты неразличимы.

# %%
rng_g = np.random.default_rng([SEED, 3])
S_g = rng_g.normal(0.0, 1.0, (N, 3))
X_g = S_g @ A_true.T
X_g_c = X_g - X_g.mean(axis=0)

Z_g, M_g = whiten(X_g_c)
rng_ica_g = np.random.default_rng([SEED, 4])
W_g, _ = fastica(Z_g, 3, rng_ica_g)
S_hat_g = Z_g @ W_g.T

WA_g = W_g @ M_g @ A_true
print("W · M · A_true для гауссовых источников (перестановочной не будет):")
print(np.array2string(WA_g, precision=3))

# rows of W M A live in the source space; angle of each row to the nearest axis
WA_g_dir = WA_g / np.linalg.norm(WA_g, axis=1, keepdims=True)
cos_g = np.clip(np.abs(WA_g_dir), 0, 1)
angles_g = np.degrees(np.arccos(cos_g))
row_min = angles_g.min(axis=1)
print(f"углы строк W·M·A до ближайшей оси источника (град): "
      f"{np.array2string(row_min, precision=1)}")
check_golden("angle_gauss_deg", row_min.min())

# %% [markdown]
# ## Пример 4. Меры негауссовости
#
# **Вопрос (theory.md, определение дифференциальной энтропии и задача
# о куртозисе суммы):** куртозис и приближение негэнтропии
# $(\E G(Y) - \E G(\nu))^2$ с $G(u) = -\log\cosh(u)$ различают распределения
# по негауссовости так, как предсказывает теория: куртозис гаусса равен нулю,
# равномерного распределения отрицателен ($-1{,}2$), лапласова — положителен
# ($+3$); обе меры обращаются в нуль на гауссе.
#
# **Предсказание до прогона.** Выборочные оценки на $n = 200000$ точках
# должны попасть в теоретические значения с погрешностью порядка $10^{-2}$:
# куртозис $\approx -1{,}2$ (равномерное), $\approx +3$ (лапласово),
# $\approx -2$ (двухточечное $\pm 1$); прокси-негэнтропия строго положительна
# вне гаусса и $\approx 0$ на гауссе.

# %%
n_mc = 200_000
rng_m = np.random.default_rng([SEED, 5])


def standardize(v):
    return (v - v.mean()) / v.std(ddof=0)


samples = {
    "gauss": rng_m.normal(0.0, 1.0, n_mc),
    "unif": rng_m.uniform(-np.sqrt(3), np.sqrt(3), n_mc),
    # laplace(0, 1/sqrt(2)) has unit variance and excess kurtosis +3
    "laplace": rng_m.laplace(0.0, 1.0, n_mc) / np.sqrt(2),
    "bimodal": rng_m.choice([-1.0, 1.0], n_mc) + 0.05 * rng_m.normal(0, 1, n_mc),
}
samples = {k: standardize(v) for k, v in samples.items()}

g_nu = float(-np.log(np.cosh(rng_m.normal(0.0, 1.0, n_mc))).mean())


def neg_entropy_proxy(y):
    """(E G(y) - E G(nu))^2 with G(u) = -log cosh(u), nu ~ N(0,1)."""
    g_y = float(-np.log(np.cosh(y)).mean())
    return (g_y - g_nu) ** 2


labels = list(samples)
kurt = {k: float((v**4).mean() - 3.0) for k, v in samples.items()}
jproxy = {k: float(neg_entropy_proxy(v)) for k, v in samples.items()}
for k in labels:
    print(f"{k:8s}: kurt = {kurt[k]:+.4f}, J-прокси = {jproxy[k]:.6f}")

check_golden("kurt_laplace", kurt["laplace"])
check_golden("kurt_unif", kurt["unif"])

fig, ax = plt.subplots(figsize=(7, 3.6))
xpos = np.arange(len(labels))
ax.bar(xpos - 0.2, [kurt[k] for k in labels], width=0.4, label="куртозис")
ax.bar(xpos + 0.2, [10 * jproxy[k] for k in labels], width=0.4,
       label=r"негэнтропия $\times 10$")
ax.axhline(0.0, color="k", lw=0.6)
ax.set_xticks(xpos)
ax.set_xticklabels(labels)
ax.set_title("Меры негауссовости стандартизованных распределений")
ax.legend()
fig.tight_layout()
fig.savefig(f"{FIGDIR}/fig-03.pdf", **SAVE_KW)
plt.close(fig)
