# %% [markdown]
# # Вопрос 16. PCA и ICA — численные примеры
#
# Файл ведётся в формате jupytext (percent). Парный `.ipynb` получается
# командой `make questions/16-pca-ica/examples.ipynb`.
#
# Каждый пример отвечает на вопрос из `theory.md`. Сквозная постановка —
# один на весь файл: многоканальная регистрация. Три независимых источника
# сигнала (синусоидальный, прямоугольный и пилообразный) с единичной
# дисперсией (посылка модели ICA, определение в theory.md) записываются
# тремя датчиками полных смесей с фиксированной матрицей смешения $B$
# (сингулярные числа $2{,}2, 1{,}6, 1{,}1$). Для PCA (пример 1) каждая смесь
# пишется тремя локальными датчиками с шумом (девять каналов) плюс один
# фоновый; для ICA (примеры 2–3) — те же три смеси без повторов; далее
# гауссовский контрпример (пример 3) и меры негауссовости (пример 4).

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
    # пример 1 (PCA, p=10: 9 локальных датчиков + 1 фоновый, смеси троекратны)
    "recon_err_q3": 1052.91821747,
    "rho_first3": 0.894708178253,
    "n_above_perm": 3,
    # пример 2 (FastICA против эталона sklearn)
    "corr_own_min": 0.99999764028,
    "corr_sklearn_min": 0.999999234516,
    "iters_own_total": 9,
    # пример 3 (гауссовский контрпример)
    "angle_gauss_deg": 15.5171334609,
    # пример 4 (меры негауссовости)
    "kurt_laplace": 2.97548679519,
    "kurt_unif": -1.19728574426,
}
TOL = 1e-3


def check_golden(name, value, tol=TOL):
    if GOLDEN[name] is None:
        print(f"  pin {name}: got {value:.12g} (unset — fill in)")
        return
    want = GOLDEN[name]
    rel = abs(value - want) / abs(want)
    print(f"  pin {name}: got {value:.12g}, expected {want:.12g}")
    assert rel <= tol, f"golden pin {name} mismatch: {value} vs {want}"


# %% [markdown]
# ## Сквозная постановка: три источника, десять датчиков
#
# Источники: $s_1(t) = \sin(2\pi f_1 t)$ (синусоида),
# $s_2(t) = \operatorname{sign}\sin(2\pi f_2 t)$ (меандр),
# $s_3(t) = 2\,(t f_3 \bmod 1) - 1$ (пила); частоты разнесены. Каждый источник
# стандартизован делением на выборочное стандартное отклонение: модель ICA
# (определение в theory.md) требует $\Var S_j = 1$, без этого соглашения
# разделение определено лишь с точностью до масштаба. Три датчика регистрируют
# полные смеси $x_i = \sum_j b_{ij} s_j$ с матрицей смешения $B$
# (ортогональные сомножители из того же seed, сингулярные числа
# $2{,}2, 1{,}6, 1{,}1$ — сильные и различные, чтобы главные направления были
# определены однозначно). Каждая смесь к тому же пишется тремя локальными
# датчиками с шумом $0{,}1$ (каналы 1–9: три тройки копий, сильно
# коррелированные внутри себя); датчик 10 — фоновый шум без сигнала. Данные
# центрируются (см. замечание к теореме о главных компонентах в theory.md), а
# перед параллельным анализом стандартизуются по столбцам (критерий на
# корреляционной матрице, theory.md).

# %%
N = 1000
T = np.linspace(0, 1, N, endpoint=False)

S_raw = np.column_stack(
    [
        np.sin(2 * np.pi * 5 * T),
        np.sign(np.sin(2 * np.pi * 11 * T + 0.3)),
        2 * ((8 * T) % 1) - 1,
    ]
)
S = S_raw / S_raw.std(axis=0, ddof=0)  # unit-variance sources (model assumption)

rng_mix = np.random.default_rng([SEED, 9])
U_mix, _ = np.linalg.qr(rng_mix.normal(0.0, 1.0, (3, 3)))
V_mix, _ = np.linalg.qr(rng_mix.normal(0.0, 1.0, (3, 3)))
B_true = U_mix @ np.diag([2.2, 1.6, 1.1]) @ V_mix.T  # controlled spectrum

X3c = S @ B_true.T  # clean 3-channel mixtures, used in examples 2-3
X3n = X3c / X3c.std(axis=0, ddof=0)  # column-standardized mixtures
local = np.repeat(X3n, 3, axis=1) + 0.1 * rng_mix.normal(0.0, 1.0, (N, 9))
X10 = np.hstack([local, 0.5 * rng_mix.normal(0.0, 1.0, (N, 1))])  # p = 10

X10c = X10 - X10.mean(axis=0)
sd = X10c.std(axis=0, ddof=0)
Z10 = X10c / sd  # standardized columns: parallel analysis on correlations
Sc = S - S.mean(axis=0)

print(f"singular values of B_true = {np.linalg.svd(B_true, compute_uv=False)}")
print(f"source kurtoses = {np.array([(v**4).mean() / (v**2).mean()**2 - 3.0 for v in S.T])}")


# %% [markdown]
# ## Пример 1. PCA: Эккарт — Янг и параллельный анализ на стандартизованных данных
#
# **Вопрос (theory.md, теорема Эккарта — Янга в норме Фробениуса, теорема о
# главных компонентах через SVD и замечание о выборе числа компонент):**
# равна ли ошибка $q$-мерной реконструкции хвосту сингулярного спектра, и
# выделяет ли параллельный анализ ровно $q = 3$ сигнальные компоненты среди
# $p = 10$ каналов?
#
# **Предсказание до прогона.** Сигнальная часть — девять каналов, записывающих
# три стандартизованные смеси троекратно: корреляционная матрица состоит из
# трёх блоков «смесь + копии», её крупные собственные значения кратны
# собственным значениям корреляции смесей и различны (сингулярные числа $B$
# различны), поэтому главные направления определены однозначно. После
# стандартизации три сигнальные $\sigma$ превосходят спектральный край
# случайного эталона ($\sqrt{N} + \sqrt{p} \approx 35$), шумовые — нет.
# Ожидаем спектр с локтем после $q = 3$, ошибку реконструкции при $q = 3$,
# равную хвосту $\sum_{j>3} \sigma_j^2$ до машинной точности (теорема
# Эккарта — Янга — про любую матрицу, в том числе шумную), долю дисперсии
# первых трёх компонент около $0{,}9$ и ровно три компоненты выше
# эталона.
# Главные направления не совпадают с профилями смесей (столбцы $B$,
# размноженные по тройкам копий) —
# честная геометрия: PCA находит сигнальное подпространство смесей, не
# разделяя источников.

# %%
U_x, d_x, Vt_x = np.linalg.svd(Z10, full_matrices=False)


def recon_err(Xc, Vt_q):
    """Frobenius error of rank-q PCA reconstruction (theory.md, thm:pca-svd)."""
    H = Vt_q.T @ Vt_q
    return float(np.linalg.norm(Xc - Xc @ H, "fro") ** 2)


err_q3 = recon_err(Z10, Vt_x[:3])
tail_q3 = float(np.sum(d_x[3:] ** 2))
print(f"q=3: recon error {err_q3:.10f}, spectral tail {tail_q3:.10f}")
check_golden("recon_err_q3", err_q3)

rho_first3 = float(np.sum(d_x[:3] ** 2) / np.sum(d_x**2))
print(f"variance share of first 3 components rho = {rho_first3:.6f}")
check_golden("rho_first3", rho_first3)

# %% [markdown]
# Эталон «без структуры» — те же стандартизованные данные с независимо
# перемешанным каждым столбцом: перестановки не меняют дисперсий каналов, но
# разрушают корреляции между ними, и спектр эталона выравнивается около
# $\sqrt{N}$ (теорема о выборе числа компонент в theory.md). Критерий:
# компонента оставляется, если $\sigma_j$ данных больше $\sigma_j$ эталона.
#
# **Предсказание:** эталон выравнивается ($\sigma_j \approx \sqrt{N}$ для
# всех $j$), а данные сохраняют три сигнальных значения выше эталона — критерий
# даёт ровно $q = 3$.

# %%
rng_perm = np.random.default_rng([SEED, 1])
Z_perm = Z10.copy()
for j in range(Z_perm.shape[1]):
    Z_perm[:, j] = Z_perm[rng_perm.permutation(N), j]
d_perm = np.linalg.svd(Z_perm, compute_uv=False)
n_above = int(np.sum(d_x > d_perm))

print(f"sigma(data) = {np.array2string(d_x, precision=1)}")
print(f"sigma(perm) = {np.array2string(d_perm, precision=1)}")
print(f"components above permutation baseline: {n_above}")
check_golden("n_above_perm", float(n_above))

# angles of principal directions to source directions: sources sit in the
# coordinate subspaces of their sensor triples
V_pc = Vt_x[:3].T  # 10 x 3
src_dirs = np.zeros((10, 3))
for k in range(3):
    src_dirs[3 * k:3 * k + 3, k] = 1.0 / np.sqrt(3.0)
cos_angles = np.clip(np.abs(V_pc.T @ src_dirs), 0, 1)
angles_deg = np.degrees(np.arccos(cos_angles))
print("angles PCA vs source directions (deg):")
print(np.array2string(angles_deg, precision=1))

fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))

axes[0].plot(T, S[:, 0], lw=0.8)
axes[0].set_title("источник 1 (синус)")
axes[0].set_xlim(0, 0.5)
axes[1].plot(T, X10[:, 0], lw=0.8, color="darkred")
axes[1].set_title("смесь, датчик 1")
axes[1].set_xlim(0, 0.5)
axes[2].plot(d_x, "o-", label="данные")
axes[2].plot(d_perm, "s--", label="перемешанные")
axes[2].axvline(2.5, color="gray", ls=":", lw=0.8)
axes[2].set_title("сингулярный спектр (параллельный анализ)")
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
# \E[g'(w^{\T}Z)]\,w$ с $g = \tanh$ (в theory $g = -\tanh$ — замена
# $g \to -g$ сокращается нормировкой $w$) источники по трём центрированным
# смесям, и совпадает ли результат с эталонной реализацией scikit-learn?
#
# **Предсказание до прогона.** После отбеливания матрица смешения становится
# ортогональной (предложение об отбеливании), и независимые проекции возможны
# только вдоль осей источников (теорема идентифицируемости ICA); значит,
# FastICA сойдётся к $W = B^{-1}$ с точностью до перестановки и знака
# (нормировка дисперсий снимает свободу масштаба). Все три источника
# субгауссовы (куртозис синуса $-1{,}5$, меандра $-2$, пилы $-1{,}2$), поэтому
# постановка через экстремумы $(\E G - \E G(\nu))^2$ важна: итерация ищет
# стационарные точки, а не максимум $\E G$ (максимумы отвечали бы только
# супергауссовым компонентам). Меры качества: (1) для каждого восстановленного
# сигнала максимум $|\operatorname{corr}(\hat s_j, s_k)|$ по $k$ близок к
# единице; (2) матрица $W M B$ близка к перестановочной матрице со знаками;
# (3) своя реализация и sklearn дают одни и те же компоненты. Итераций
# ожидаем несколько десятков.

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
        print(f"  component {j + 1}: converged in {iters} iterations")
    return W, total_iters


Z, M_white = whiten(X3c)

rng_ica = np.random.default_rng([SEED, 2])
W_own, iters_total = fastica(Z, 3, rng_ica)
S_hat_own = Z @ W_own.T
print(f"total iterations: {iters_total}")
check_golden("iters_own_total", float(iters_total))

# recovered mixing: W M B must be close to a signed permutation
WB = W_own @ M_white @ B_true
print("W_own · M · B_true (should be ~ signed permutation):")
print(np.array2string(WB, precision=3))


def best_source_corr(S_hat, S):
    """For each recovered row: max_k |corr| with the sources (ICA quality)."""
    C = np.corrcoef(np.hstack([S_hat, S]).T)
    n = S_hat.shape[1]
    return np.abs(C[:n, n:]).max(axis=1)


corr_own = best_source_corr(S_hat_own, Sc)
print(f"max-corr per source (own): {np.array2string(corr_own, precision=6)}")
check_golden("corr_own_min", corr_own.min())

# %% [markdown]
# Тот же эксперимент эталонной реализацией — для сверки.

# %%
from sklearn.decomposition import FastICA as SkFastICA

ica_sk = SkFastICA(n_components=3, whiten="arbitrary-variance", fun="logcosh",
                   max_iter=2000, tol=1e-10, random_state=SEED)
S_hat_sk = ica_sk.fit_transform(X3c)

corr_sk = best_source_corr(S_hat_sk, Sc)
print(f"max-corr per source (sklearn): {np.array2string(corr_sk, precision=6)}")
check_golden("corr_sklearn_min", corr_sk.min())

C = np.corrcoef(np.hstack([S_hat_own, S_hat_sk]).T)[:3, 3:]
for j in range(3):
    k = int(np.argmax(np.abs(C[j])))
    print(f"  own[{j}] <-> sk[{k}]: corr {C[j, k]:+.9f}")

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
# **Предсказание до прогона.** Матрица $W M B$ не приблизится к перестановочной:
# в пространстве источников строки $W M B$ — произвольные ортогональные
# направления, угол между ними и осями источников останется большим (порядка
# десятков градусов). Это не дефект алгоритма, а следствие
# неидентифицируемости: по ковариации $I$ и условию независимости гауссовых
# компонентов повороты неразличимы.

# %%
rng_g = np.random.default_rng([SEED, 3])
S_g = rng_g.normal(0.0, 1.0, (N, 3))
X_g = S_g @ B_true.T
X_g_c = X_g - X_g.mean(axis=0)

Z_g, M_g = whiten(X_g_c)
rng_ica_g = np.random.default_rng([SEED, 4])
W_g, _ = fastica(Z_g, 3, rng_ica_g)
S_hat_g = Z_g @ W_g.T

WB_g = W_g @ M_g @ B_true
print("W · M · B_true для гауссовых источников (перестановочной не будет):")
print(np.array2string(WB_g, precision=3))

# rows of W M B live in the source space; angle of each row to the nearest axis
WB_g_dir = WB_g / np.linalg.norm(WB_g, axis=1, keepdims=True)
cos_g = np.clip(np.abs(WB_g_dir), 0, 1)
angles_g = np.degrees(np.arccos(cos_g))
row_min = angles_g.min(axis=1)
print(f"angles of W·M·B rows to nearest source axis (deg): "
      f"{np.array2string(row_min, precision=1)}")
check_golden("angle_gauss_deg", row_min.min())

# %% [markdown]
# ## Пример 4. Меры негауссовости
#
# **Вопрос (theory.md, определение дифференциальной энтропии и задача о
# куртозисе суммы):** куртозис $\E Y^4/(\E Y^2)^2 - 3$ и приближение
# негэнтропии $(\E G(Y) - \E G(\nu))^2$ с $G(u) = -\log\cosh(u)$ различают
# распределения по негауссовости так, как предсказывает теория: куртозис
# гаусса равен нулю, равномерного распределения отрицателен ($-1{,}2$),
# лапласова — положителен ($+3$); обе меры обращаются в нуль на гауссе.
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


def kurtosis(v):
    return float((v**4).mean() / (v**2).mean() ** 2 - 3.0)


labels = list(samples)
kurt = {k: kurtosis(v) for k, v in samples.items()}
jproxy = {k: float(neg_entropy_proxy(v)) for k, v in samples.items()}
for k in labels:
    print(f"{k:8s}: kurt = {kurt[k]:+.4f}, J-proxy = {jproxy[k]:.6f}")

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
