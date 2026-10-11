# %% [markdown]
# # Вопрос 07. Преобразования Фурье, Лапласа, Хаара — численные примеры
#
# Файл ведётся в формате jupytext (percent). Парный `.ipynb` получается командой
# `make questions/07-fourier-laplace-haar/examples.ipynb`; в git идёт **этот**
# файл, ноутбук — производный артефакт.
#
# Каждый пример отвечает на вопрос, поставленный в `theory.md`, и предсказание
# формулируется **до** прогона. Числа, попавшие в `theory.md` и `slides.md`,
# печатаются здесь.
#
# Сквозной сигнал вопроса — экспоненциальный импульс $u(t) = e^{-t}\mathbb
# I_{[0,\infty)}$: его преобразования считаются аналитически в задачах
# конспекта (образы Фурье, изображение Лапласа, коэффициенты Хаара), а
# «изломанная» версия семейства возвращается в примере 3. Все примеры
# детерминированы (случайные числа не используются); seed зафиксирован по
# правилам проекта.

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.special import sici

SEED = 20261011
rng = np.random.default_rng(SEED)   # no random draws in this notebook

FIGDIR = "figures"

# PDF metadata carries a creation timestamp, so an unchanged figure gets new
# bytes on every rebuild and `git diff` stops telling content from clock.
SAVE_KW = {"metadata": {"CreationDate": None}}

# Golden pins: seed AND expected output are frozen constants. Each tolerance is
# set per pin. For a pin with expected value 0.0 the tolerance is absolute
# (relative tolerance of a zero is meaningless).
GOLDEN = {
}


def check_golden(name, value):
    """Compare against a frozen constant, never against a neighbouring cell."""
    want, tol = GOLDEN[name]
    if want == 0.0:
        err = abs(value)
        rel = err
    else:
        err = abs(value - want)
        rel = err / abs(want)
    status = "ok" if rel <= tol else "РАСХОЖДЕНИЕ"
    print(f"  пин {name}: получено {value:.12g}, ожидалось {want:.12g} "
          f"(допуск {tol:g}, {status})")
    assert rel <= tol, f"golden-пин {name} не сошёлся: {value} vs {want}"


# %% [markdown]
# ## Пример 1. Образы Фурье и эффект Гиббса
#
# **Вопрос (а).** Конспект (задача об образах импульсов) вычисляет
# аналитически $\widehat{\mathbb I_{[-1,1]}}(\omega) = \sqrt{2/\pi}\,
# \sin\omega/\omega$ и $\hat u(\omega) = 1/(\sqrt{2\pi}(1+i\omega))$ и
# проверяет для $u$ равенство Парсеваля $\|u\|_2^2 = \|\hat u\|_2^2 = 1/2$.
# Сверяем аналитические образы с двумя численными путями: прямым
# интегрированием трапециями по конечному отрезку (хвост у $u$ при $t > T$
# оценивается: $|\int_T^{\infty} e^{-(1+i\omega)t}dt| \leq e^{-T}$) и БПФ
# сигнала, дополненного нулями. Golden-пин — численное значение
# $\|\hat u\|_2^2$, посчитанное по частотной сетке.
#
# **Предсказание.** (1) Прямое интегрирование попадает в аналитический образ с
# точностью не хуже $10^{-5}$ на окне $|\omega| \leq 20$ (ошибка трапеций
# $O(h^2\omega^2)$). (2) БПФ гладкого сигнала $u$ совпадает с аналитическим
# образом почти до машинного эпсилон, а БПФ разрывного импульса отклоняется на
# уровне $10^{-3}$–$10^{-2}$: дискретизация скачка и периодизация окна
# наследуются образом. (3) $\|\hat u\|_2^2$ по частотной сетке БПФ совпадёт с
# точным $1/2$ в пределах $e^{-2T}$ и погрешности прямоугольной суммы.

# %%
print("1a. Образы Фурье: аналитика против двух численных путей")

OM = np.linspace(0.0, 20.0, 241)


def rect_hat(w):
    """sqrt(2/pi) * sin(w)/w with the continuous value sqrt(2/pi) at 0."""
    return np.sqrt(2.0 / np.pi) * np.sinc(w / np.pi)


def u_hat(w):
    """1/(sqrt(2 pi) (1 + i w)) — образ u(t) = e^{-t} на [0, inf)."""
    return 1.0 / (np.sqrt(2.0 * np.pi) * (1.0 + 1j * w))


# path 1: trapezoids on a finite interval; for u the tail beyond T is <= e^{-T}
T_CUT = 20.0
t_rect = np.linspace(-1.0, 1.0, 8001)
t_u = np.linspace(0.0, T_CUT, 12001)


def direct_image(f_vals, t, w):
    """Trapezoid estimate of 1/sqrt(2pi) * int f(t) e^{-i w t} dt."""
    kernel = np.exp(-1j * t[:, None] * w[None, :])
    return np.trapezoid(f_vals[:, None] * kernel, t, axis=0) / np.sqrt(2.0 * np.pi)


rect_dir = direct_image(np.ones_like(t_rect), t_rect, OM)
u_dir = direct_image(np.exp(-t_u), t_u, OM)
err_rect_dir = float(np.max(np.abs(rect_dir - rect_hat(OM))))
err_u_dir = float(np.max(np.abs(u_dir - u_hat(OM))))
print(f"  прямое интегрирование: max |числ - ан| импульс = {err_rect_dir:.3e}, "
      f"u = {err_u_dir:.3e}")
print(f"    хвост u за t > {T_CUT:g} по модулю <= e^{-T_CUT:g} = "
      f"{np.exp(-T_CUT):.1e}")


# path 2: FFT of the zero-padded sampled signal (unitary normalization)
def fft_image(f_vals, dt):
    """Unitary Fourier image from FFT samples: dt * fft / sqrt(2 pi)."""
    return np.fft.fft(f_vals) * dt / np.sqrt(2.0 * np.pi)


N1 = 1 << 16
# rectangular pulse on [-16, 16)
dt_r = 32.0 / N1
tg_r = np.arange(N1) * dt_r - 16.0
fg_r = (np.abs(tg_r) <= 1.0).astype(float)
w_r = 2.0 * np.pi * np.fft.fftfreq(N1, d=dt_r)
F_rect = fft_image(fg_r, dt_r)
# u on [0, 40): e^{-40} ~ 4e-18, so the windowing jump at the edge is invisible
dt_u = 40.0 / N1
tg_u = np.arange(N1) * dt_u
fg_u = np.exp(-tg_u)
w_u = 2.0 * np.pi * np.fft.fftfreq(N1, d=dt_u)
F_u = fft_image(fg_u, dt_u)

sel_r = np.abs(w_r) <= 20.0
sel_u = np.abs(w_u) <= 20.0
err_rect_fft = float(np.max(np.abs(F_rect[sel_r] - rect_hat(w_r[sel_r]))))
err_u_fft = float(np.max(np.abs(F_u[sel_u] - u_hat(w_u[sel_u]))))
print(f"  БПФ (окно |w| <= 20):   max |числ - ан| импульс = {err_rect_fft:.3e}, "
      f"u = {err_u_fft:.3e}")

# Parseval for u: energy from the frequency grid against the exact value 1/2
order = np.argsort(w_u)
energy_w = float(np.trapezoid(np.abs(F_u[order]) ** 2, w_u[order]))
energy_t = float(np.trapezoid(np.abs(fg_u) ** 2, tg_u))
print(f"  Парсеваль для u: ||u||_2^2 = {energy_t:.12f} (по t), "
      f"||hat u||_2^2 = {energy_w:.12f} (по w), точное 1/2")

# %% [markdown]
# **Вывод (а).** — заполняется по факту прогона (второй заход).

# %% [markdown]
# **Вопрос (б).** Частичные суммы ряда Фурье скачка. Берём $f(t) =
# \mathrm{sgn}(t - 1/2)$ на $[0,1]$ с периодом $1$ — тот же сигнал, что и в
# примере 2 вопроса 09 (преемственность). Коэффициенты выписываются в явном
# виде: $c_k = \frac{(-1)^k - 1}{\pi i k}$ ($k \neq 0$), то есть $c_k =
# \frac{2i}{\pi k}$ при нечётных $k$ и $0$ при чётных; $c_0 = 0$. Частичная
# сумма $S_n$ — сумма по $|k| \leq n$, что даёт $S_n(t) = -\frac{4}{\pi}
# \sum_{k\ \mathrm{odd} \leq n} \frac{\sin 2\pi k t}{k}$. Гипотеза конспекта
# (замечание о локализации к теореме о Парсевале для Хаара): выброс около
# скачка не исчезает с ростом $n$ и стремится к константе Уилбрахама
# $\frac{2}{\pi}\mathrm{Si}(\pi) - 1 \approx 8{,}95\%$ от величины скачка.
# Golden-пины: измеренный выброс при $n = 64$ и доля энергии в первых $n$
# гармониках.
#
# **Предсказание.** (1) $\max_{t > 1/2} S_n(t)$ убывает к $1 + 0{,}0895\cdot 2$
# сверху: выброс около $9\%$ от скачка уже при $n = 64$. (2) Доля энергии
# $\sum_{|k| \leq n}|c_k|^2$ растёт к $1$ медленно, как $1 - O(1/n)$: энергии
# гармоник $|c_k|^2 \sim 1/k^2$, и хвост ряда энергий убывает как $1/n$.

# %%
print("1b. Эффект Гиббса у скачка: sgn(t - 1/2) на [0, 1], период 1")
tg = np.linspace(0.0, 1.0, 100001)


def fourier_partial(n, t):
    """S_n(t) = -(4/pi) * sum_{k odd <= n} sin(2 pi k t)/k."""
    acc = np.zeros_like(t)
    for k in range(1, n + 1, 2):
        acc += np.sin(2.0 * np.pi * k * t) / k
    return -(4.0 / np.pi) * acc


print("  n    max S_n справа     выброс, % скачка    доля энергии |k| <= n")
right = tg >= 0.5 + 5e-4
gibbs_table = {}
for n in (4, 8, 16, 32, 64, 128):
    mx = float(fourier_partial(n, tg)[right].max())
    kk = np.arange(1, n + 1, 2)
    e_frac = float(np.sum(8.0 / (np.pi * kk) ** 2))
    gibbs_table[n] = (mx, (mx - 1.0) / 2.0, e_frac)
    print(f"{n:4d}    {mx:.8f}        {100 * gibbs_table[n][1]:9.4f}      "
          f"{e_frac:.8f}")
si_pi, _ = sici(np.pi)
print(f"\n  константа Уилбрахама (2/pi) Si(pi) - 1 = {2 / np.pi * si_pi - 1:.6f}"
      f" = {100 * (2 / np.pi * si_pi - 1):.2f} % скачка")

# %%
fig, ax = plt.subplots(figsize=(6, 3.6), layout="constrained")
step_t = np.array([0.0, 0.5, 0.5, 1.0])
step_v = np.array([-1.0, -1.0, 1.0, 1.0])
ax.plot(step_t, step_v, ":", color="black", linewidth=1.4,
        label=r"$\mathrm{sgn}(t - 1/2)$")
for n, style in ((4, "--"), (16, "-."), (64, "-")):
    ax.plot(tg, fourier_partial(n, tg), style, linewidth=1.1,
            label=f"$S_{{{n}}}$")
ax.set_xlabel("$t$")
ax.set_ylabel("$S_n(t)$")
ax.set_ylim(-1.5, 1.5)
ax.set_title("эффект Гиббса: выброс у скачка не исчезает с ростом $n$",
             fontsize=9)
ax.legend(fontsize=7, loc="lower left")
fig.savefig(f"{FIGDIR}/fig-02.pdf", **SAVE_KW)

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.6), layout="constrained")

ax1.plot(OM, rect_hat(OM), "-", linewidth=1.2, color="black",
         label=r"$\sqrt{2/\pi}\,\sin\omega/\omega$")
ax1.plot(w_r[sel_r], np.real(F_rect[sel_r]), "o", markersize=3,
         color="tab:blue", label="БПФ с нулями")
ax1.plot(OM, np.real(rect_dir), ".", markersize=2, color="tab:red",
         label="прямые трапеции")
ax1.set_xlabel(r"$\omega$")
ax1.set_ylabel(r"$\hat{I}_{[-1,1]}(\omega)$")
ax1.set_title("прямоугольный импульс (образ вещественный)", fontsize=9)
ax1.legend(fontsize=7)

ax2.plot(OM, np.abs(u_hat(OM)), "-", linewidth=1.2, color="black",
         label=r"$1/(\sqrt{2\pi}\,|1+i\omega|)$")
ax2.plot(w_u[sel_u], np.abs(F_u[sel_u]), "o", markersize=3,
         color="tab:blue", label="БПФ с нулями")
ax2.plot(OM, np.abs(u_dir), ".", markersize=2, color="tab:red",
         label="прямые трапеции")
ax2.set_xlabel(r"$\omega$")
ax2.set_ylabel(r"$|\hat u(\omega)|$")
ax2.set_title(r"экспоненциальный импульс $u(t) = e^{-t}$, $t \geq 0$",
              fontsize=9)
ax2.legend(fontsize=7)
fig.savefig(f"{FIGDIR}/fig-01.pdf", **SAVE_KW)

# %% [markdown]
# **Вывод (б).** — заполняется по факту прогона (второй заход).

# %% [markdown]
# ## Пример 2. Преобразование Лапласа и задача Коши
#
# **Вопрос (а).** Конспект (задача о колебательном контуре) решает $y'' + y =
# 0$, $y(0) = 0$, $y'(0) = 1$ операторным методом: $(s^2+1)Y(s) = 1$, $Y(s) =
# 1/(s^2+1) \dashv \sin t$. Сверяем аналитическое решение с численным
# интегрированием системы (`solve_ivp`, жёсткие допуски) на $[0, 2\pi]$.
#
# **Вопрос (б).** Затухающий контур $y'' + 2\beta y' + y = 0$ с теми же
# начальными условиями, $\beta = 1/5$. Изображение решает $(s^2 + 2\beta s +
# 1)Y(s) = 1$. Выделение полного квадрата $s^2 + 2\beta s + 1 = (s+\beta)^2 +
# (1-\beta^2)$ и сдвиг по $s$ дают оригинал $y(t) = e^{-\beta t}
# \sin(\omega_d t)/\omega_d$, $\omega_d = \sqrt{1-\beta^2}$ — это
# предсказание, которое сверяется с `solve_ivp`.
#
# **Вопрос (в).** Формула Меллина (теорема конспекта) восстанавливает оригинал
# интегралом по прямой $\operatorname{Re} s = \sigma_1$; берём $\sigma_1 = 1
# > 0 = a_0$ и обращаем интеграл трапециями по $\omega \in [-\Omega, \Omega]$.
# Два источника ошибки разведены по построению: обрезание по $\Omega$ (хвост
# $|F(\sigma_1+i\omega)| \sim 1/\omega^2$, интегральный хвост $\sim 1/\Omega$)
# и квадратурная ошибка трапеций. Golden-пины: квадратурная ошибка при
# $N_\omega = 2^{12}$ и отношение ошибок при удвоении сетки.
#
# **Предсказание.** (а), (б) расхождение аналитики и `solve_ivp` — не выше
# $10^{-8}$. (в) Квадратурная ошибка при фиксированном $\Omega$ убывает как
# $O(h^2)$: удвоение $N_\omega$ уменьшает её примерно в $4$ раза, а ошибка
# обрезания стоит на месте, поэтому сходимость к $\sin t$ упирается в плато
# уровня $e^{\sigma_1 t}/\Omega$ — растущее по $t$.

# %%
print("2ab. Задача Коши через изображения против solve_ivp")


def osc_rhs(t, y, beta):
    return [y[1], -2.0 * beta * y[1] - y[0]]


BETA = 0.2
WD = np.sqrt(1.0 - BETA ** 2)


def y_exact(t, beta):
    """sin t при beta = 0; e^{-beta t} sin(wd t)/wd иначе."""
    if beta == 0.0:
        return np.sin(t)
    wd = np.sqrt(1.0 - beta ** 2)
    return np.exp(-beta * t) * np.sin(wd * t) / wd


tt = np.linspace(0.0, 2.0 * np.pi, 2001)
for beta, name in ((0.0, "y'' + y = 0,      Y(s) = 1/(s^2+1)"),
                   (BETA, f"y'' + 0.4 y' + y = 0, Y(s) = 1/(s^2+0.4s+1)")):
    sol = solve_ivp(osc_rhs, (0.0, 2.0 * np.pi), [0.0, 1.0], args=(beta,),
                    t_eval=tt, rtol=1e-12, atol=1e-14)
    err = float(np.max(np.abs(sol.y[0] - y_exact(tt, beta))))
    print(f"  {name}: max |аналитика - solve_ivp| на [0, 2pi] = {err:.3e}")
print(f"\n  затухающий контур: beta = 1/5, wd = sqrt(1-beta^2) = {WD:.10f}")
print(f"    y(2pi) = e^{{-2 pi beta}} sin(2 pi wd)/wd = "
      f"{np.exp(-BETA * 2 * np.pi) * np.sin(WD * 2 * np.pi) / WD:.6e}")

# %%
print("2c. Численное обращение Меллина: f(t) = 1/(2pi) int F(sigma1 + i w)"
      " e^{(sigma1 + i w) t} dw, Y(s) = 1/(s^2+1), sigma1 = 1")

SIG1 = 1.0
OMEGA_M = 4096.0
TGRID = np.linspace(0.0, 2.0 * np.pi, 121)


def mellin_trap(nw, sigma1=SIG1, omega=OMEGA_M, tgrid=TGRID):
    """Trapezoid inversion of the Bromwich integral on [-omega, omega]."""
    w = np.linspace(-omega, omega, nw + 1)
    f_vals = 1.0 / ((sigma1 + 1j * w) ** 2 + 1.0)
    e = np.exp((sigma1 + 1j * w)[None, :] * tgrid[:, None])
    return np.trapezoid(f_vals[None, :] * e, w, axis=1) / (2.0 * np.pi)


# reference: trapezoid on a very fine grid approximates the TRUNCATED integral
# to ~1e-12 (aliasing e^{-2 pi/h} from the pole a distance 1 away), so the
# quadrature error is measured against it, not against a library solver
N_REF = 1 << 20
f_ref = mellin_trap(N_REF)
f_rec_12 = mellin_trap(1 << 12)

print("  N_w       max |f_N - sin t|     max |f_N - f_ref|     квадрат. отношение")
prev_quad = None
mellin_table = {}
for nw in (1 << 10, 1 << 11, 1 << 12, 1 << 13, 1 << 14):
    f_rec = mellin_trap(nw)
    e_sin = float(np.max(np.abs(f_rec - np.sin(TGRID))))
    e_quad = float(np.max(np.abs(f_rec - f_ref)))
    ratio = (prev_quad / e_quad) if prev_quad is not None else float("nan")
    prev_quad = e_quad
    mellin_table[nw] = (e_sin, e_quad)
    print(f"  2^{int(np.log2(nw)):2d}      {e_sin:.6e}          {e_quad:.6e}          "
          f"{ratio:8.3f}")
diff_sigma = float(np.max(np.abs(mellin_trap(1 << 12, sigma1=2.0) - f_rec_12)))
print(f"\n  независимость от прямой: max |f_{{sigma1=2}} - f_{{sigma1=1}}| "
      f"при N = 2^12: {diff_sigma:.3e}")

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.6), layout="constrained")

ax1.plot(tt, y_exact(tt, 0.0), "-", linewidth=1.2, label=r"$y''+y=0:\ \sin t$")
ax1.plot(tt, y_exact(tt, BETA), "-", linewidth=1.2,
         label=r"$y''+0.4y'+y=0:\ e^{-t/5}\sin(\omega_d t)/\omega_d$")
ax1.plot(tt, np.exp(-BETA * tt) / WD, "--", linewidth=0.9, color="grey",
         label=r"огибающая $\pm e^{-t/5}/\omega_d$")
ax1.plot(tt, -np.exp(-BETA * tt) / WD, "--", linewidth=0.9, color="grey")
ax1.set_xlabel("$t$")
ax1.set_ylabel("$y(t)$")
ax1.set_title(r"контур через изображения: без затухания и с $\beta = 1/5$",
              fontsize=9)
ax1.legend(fontsize=7, loc="lower left")

ax2.semilogy(TGRID, np.abs(f_rec_12 - np.sin(TGRID)), "-", linewidth=1.1,
             label="ошибка к $\\sin t$ (с обрезанием)")
ax2.semilogy(TGRID, np.abs(f_rec_12 - f_ref), "-", linewidth=1.1,
             label="квадратурная ошибка (к $f_{\\mathrm{ref}}$)")
ax2.set_xlabel("$t$")
ax2.set_ylabel("абсолютная ошибка")
ax2.set_title("обращение Меллина, $N_\\omega = 2^{12}$, $\\Omega = 4096$",
              fontsize=9)
ax2.legend(fontsize=7, loc="lower left")
fig.savefig(f"{FIGDIR}/fig-03.pdf", **SAVE_KW)

# %% [markdown]
# **Вывод (2).** — заполняется по факту прогона (второй заход).

# %% [markdown]
# ## Пример 3. Хаар против Фурье при сжатии сигнала с изломом
#
# **Сигнал.** Конспект называет «изломанной» версией сквозного сигнала $t
# e^{-t}$ на $[0,1]$, но она гладка внутри отрезка: разрыв у её
# периодического продолжения живёт только на стыке $1 \to 0$. Разложения
# ниже — по периодическому базису, поэтому фиксируем сигнал с изломом внутри
# отрезка, оставаясь в семействе сквозного примера:
# $$
# f(t) = \min(t, 1-t)\,e^{-t}, \qquad f(t) = t e^{-t}\ (t < 1/2), \quad
# f(t) = (1-t)e^{-t}\ (t \geq 1/2).
# $$
# Он непрерывен на окружности (периодизация без скачка), а производная
# разрывна в $t = 1/2$: слева $f'(1/2) = \frac12 e^{-1/2}$, справа $f'(1/2) =
# -\frac32 e^{-1/2}$. Дискретизация $N = 2^{12}$ по центрам ячеек.
#
# **Вопрос (а).** Разложение по периодическому базису: ДПФ (`numpy.fft`, без
# нормирующего множителя, как в конспекте) и дискретный Хаар — пирамида средних
# и деталей из определения конспекта, реализованная своим кодом с подсчётом
# операций: теорема о БПХ обещает $2(N-1)$ сложений. Golden-пин — посчитанное
# число сложений.
#
# **Предсказание.** Пирамида обратима до машинного эпсилон, энергия
# сохраняется (ортогональность), число сложений сойдётся ровно с оценкой
# $2(N-1) = 8190$.

# %%
print("3a. ДПФ и дискретный Хаар (пирамида средних и деталей, единичная"
      " нормировка)")

N3 = 1 << 12
t3 = (np.arange(N3) + 0.5) / N3
f3 = np.minimum(t3, 1.0 - t3) * np.exp(-t3)
half = 0.5 * np.exp(-0.5)
print(f"  излом в t = 1/2: f' слева = +{half:.6f}, справа = -{3 * half:.6f}")


def haar_forward(x):
    """Pyramid of means/details with 1/sqrt(2) scaling, counting operations."""
    lev = np.asarray(x, dtype=float).copy()
    details = []
    adds = 0
    mults = 0
    n = len(lev)
    while n > 1:
        s = (lev[0::2] + lev[1::2]) / np.sqrt(2.0)
        d = (lev[0::2] - lev[1::2]) / np.sqrt(2.0)
        adds += n            # one addition and one subtraction per pair
        mults += n           # two scalings by 1/sqrt(2) per pair
        details.append(d)
        lev = s
        n //= 2
    return lev, details, adds, mults


def haar_inverse(s_top, details):
    """Exact inverse of haar_forward (same butterflies, transposed)."""
    lev = np.asarray(s_top, dtype=float)
    for d in reversed(details):
        n2 = len(d)
        up = np.empty(2 * n2)
        up[0::2] = (lev + d) / np.sqrt(2.0)
        up[1::2] = (lev - d) / np.sqrt(2.0)
        lev = up
    return lev


s_top, details, adds, mults = haar_forward(f3)
f_back = haar_inverse(s_top, details)
energy_x = float(np.sum(f3 ** 2))
energy_c = float(s_top[0] ** 2 + sum(np.sum(d ** 2) for d in details))
print(f"  N = {N3}: сложений {adds} (теорема БПХ: 2(N-1) = {2 * (N3 - 1)}), "
      f"умножений {mults}")
print(f"  обратимость: max |f - f_back| = {np.max(np.abs(f3 - f_back)):.3e}")
print(f"  энергия: ||f||^2 = {energy_x:.12f}, ||s_top||^2 + sum ||d_nu||^2 = "
      f"{energy_c:.12f}")
X3 = np.fft.fft(f3)
print(f"  ДПФ Парсеваль: ||f||^2 = {energy_x:.10f} = ||X||^2/N = "
      f"{float(np.sum(np.abs(X3) ** 2)) / N3:.10f}")


# independent check on a small n: pyramid == orthonormal Haar matrix
def haar_matrix(n):
    """Rows: n^{-1/2} 1, then per level nu (coarsest to finest) the
    vectors with +-2^{-nu/2} on the dyadic blocks."""
    lev_count = int(round(np.log2(n)))
    rows = [np.full(n, 1.0 / np.sqrt(n))]
    for nu in range(lev_count, 0, -1):
        width = 1 << nu
        amp = 2.0 ** (-nu / 2.0)
        for k in range(n // width):
            v = np.zeros(n)
            v[k * width:(k + 1) * width] = amp
            v[(k + 1) * width - width // 2:(k + 1) * width] = -amp
            rows.append(v)
    return np.array(rows)


n_check = 64
x_check = f3[:n_check]
s_c, det_c, _, _ = haar_forward(x_check)
coeffs_pyramid = np.concatenate([s_c] + det_c[::-1])
coeffs_matrix = haar_matrix(n_check) @ x_check
print(f"  сверка с матрицей Хаара (n = {n_check}): max |пирамида - матрица| = "
      f"{np.max(np.abs(coeffs_pyramid - coeffs_matrix)):.3e}")

# %% [markdown]
# **Вывод (а).** — заполняется по факту прогона (второй заход).

# %% [markdown]
# **Вопрос (б).** Пороговое сжатие: для доли сохранённых коэффициентов $\rho
# \in \{1/256, 1/64, 1/16, 1/4\}$ обнуляем мелкие по модулю коэффициенты и
# измеряем относительную ошибку $\|f - \tilde f\|_2/\|f\|_2$. Для Фурье —
# усечение ДПФ, для Хаара — усечение вектора коэффициентов пирамиды. В (в)
# смотрим восстановленные сигналы при $\rho = 1/64$. Golden-пины: ошибки
# Хаара и Фурье при $\rho = 1/64$.
#
# **Предсказание.** Излом непрерывен, но производная разрывна, поэтому
# $|c_k| \sim 1/k^2$ и энергетический хвост Фурье $\sim m^{-3}$, тогда как
# детали Хаара уровня $j$ имеют порядок $O(2^{-3j/2})$ и несутся из окрестности
# излома: ждём, что Хаар точнее Фурье на всех $\rho$, с ростом разрыва к малым
# $\rho$; наклоны на log-log примерно $3/2$ (Фурье) против $1$ (Хаар). У Фурье
# возле $t = 1/2$ будут осцилляции (подавленный Гиббс — скачка значения нет),
# у Хаара — ступеньки без выбросов.

# %%
print("3b. Пороговое сжатие: относительная ошибка против доли коэффициентов")
RHOS = (1.0 / 256, 1.0 / 64, 1.0 / 16, 1.0 / 4)


def keep_largest(coeffs, rho):
    """Zero all but round(rho * n) largest-by-modulus coefficients."""
    m = max(1, int(round(rho * len(coeffs))))
    idx = np.argpartition(np.abs(coeffs), -m)[-m:]
    out = np.zeros_like(coeffs)
    out[idx] = coeffs[idx]
    return out


def haar_compress(x, rho):
    s, det, _, _ = haar_forward(x)
    blocks = [s] + det[::-1]
    lens = [len(b) for b in blocks]
    kept = keep_largest(np.concatenate(blocks), rho)
    parts = np.split(kept, np.cumsum(lens)[:-1])
    return haar_inverse(parts[0], list(parts[1:][::-1]))


def fft_compress(x, rho):
    return np.real(np.fft.ifft(keep_largest(np.fft.fft(x), rho)))


norm_f = float(np.linalg.norm(f3))
print("  rho       m       Хаар err      Фурье err     Хаар/Фурье")
comp = {}
for rho in RHOS:
    e_h = float(np.linalg.norm(f3 - haar_compress(f3, rho)) / norm_f)
    e_f = float(np.linalg.norm(f3 - fft_compress(f3, rho)) / norm_f)
    comp[rho] = (e_h, e_f)
    m = max(1, int(round(rho * N3)))
    print(f"  {rho:6.4f}  {m:5d}    {e_h:.6e}    {e_f:.6e}    {e_h / e_f:8.4f}")
slopes_h = [np.log2(comp[a][0] / comp[b][0]) for a, b in zip(RHOS, RHOS[1:])]
slopes_f = [np.log2(comp[a][1] / comp[b][1]) for a, b in zip(RHOS, RHOS[1:])]
print(f"  скользящие наклоны log2 err против log2 rho: Хаар "
      f"{np.round(slopes_h, 2)}, Фурье {np.round(slopes_f, 2)}")

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.6), layout="constrained")

ax1.loglog(RHOS, [comp[r][0] for r in RHOS], "o-", label="Хаар")
ax1.loglog(RHOS, [comp[r][1] for r in RHOS], "s-", label="Фурье (ДПФ)")
ax1.loglog(RHOS, [0.35 * r for r in RHOS], ":", color="grey",
           label=r"референс $\sim\rho$")
ax1.loglog(RHOS, [1.4 * r ** 1.5 for r in RHOS], "--", color="grey",
           label=r"референс $\sim\rho^{3/2}$")
ax1.set_xlabel(r"доля сохранённых коэффициентов $\rho$")
ax1.set_ylabel(r"относительная ошибка $\|f-\tilde f\|_2/\|f\|_2$")
ax1.set_title("сжатие сигнала с изломом: $f(t)=\min(t,1-t)e^{-t}$", fontsize=9)
ax1.legend(fontsize=7)

RHO_SHOW = 1.0 / 64
ax2.plot(t3, f3, "-", linewidth=1.4, color="black", label="исходный $f$")
ax2.plot(t3, haar_compress(f3, RHO_SHOW), "-", linewidth=1.0,
         label="Хаар, $\\rho = 1/64$")
ax2.plot(t3, fft_compress(f3, RHO_SHOW), "-", linewidth=1.0,
         label="Фурье, $\\rho = 1/64$")
ax2.set_xlabel("$t$")
ax2.set_ylabel("$f(t)$")
ax2.set_title("восстановление при $\\rho = 1/64$: ступеньки против"
              " осцилляций", fontsize=9)
ax2.legend(fontsize=7, loc="upper left")
fig.savefig(f"{FIGDIR}/fig-04.pdf", **SAVE_KW)

# %%
fig, ax = plt.subplots(figsize=(6, 3.6), layout="constrained")
zoom = (t3 > 0.30) & (t3 < 0.70)
ax.plot(t3[zoom], f3[zoom], "-", linewidth=1.6, color="black",
        label="исходный $f$")
ax.plot(t3[zoom], haar_compress(f3, RHO_SHOW)[zoom], "-", linewidth=1.1,
        drawstyle="steps-post", label="Хаар, $\\rho = 1/64$")
ax.plot(t3[zoom], fft_compress(f3, RHO_SHOW)[zoom], "-", linewidth=1.1,
        label="Фурье, $\\rho = 1/64$")
ax.set_xlabel("$t$")
ax.set_ylabel("$f(t)$")
ax.set_title("излом крупно: у Фурье — осцилляции, у Хаара — ступеньки",
             fontsize=9)
ax.legend(fontsize=7)
fig.savefig(f"{FIGDIR}/fig-05.pdf", **SAVE_KW)

# %% [markdown]
# **Вывод (3).** — заполняется по факту прогона (второй заход).

# %% [markdown]
# ## Пример 4. Теорема отсчётов
#
# **Вопрос (а).** Полосовой сигнал $x(t) = \sin 2\pi t + \sin 6\pi t$ ($f_1 =
# 1$ Гц, $f_2 = 3$ Гц, полоса $[-3, 3]$ Гц, $\omega_c = 2\pi\cdot 3$).
# Отсчёты с $f_s = 8$ Гц $> 2f_2 = 6$ Гц восстанавливают сигнал sinc-рядом
# Котельникова—Шеннона (теорема конспекта): $x_{\mathrm{rec}}(t) = \sum_n
# x(nT_s)\,\mathrm{sinc}\bigl(\frac{t-nT_s}{T_s}\bigr)$; сумму обрезаем $40$
# отсчётами. Golden-пин — максимальная ошибка восстановления на отрезке.
#
# **Вопрос (б).** Та же процедура при $f_s = 5$ Гц $< 6$ Гц: условие теоремы
# нарушено, тон $f_2 = 3$ Гц отражается в $|f_s - f_2| = 2$ Гц (aliasing).
# Golden-пин — измеренная частота пика спектра восстановленного сигнала.
#
# **Предсказание.** (а) Максимум ошибки на отрезке — порядка $10^{-2}$:
# обрезание sinc-ряда на $M$ отсчётах запаса даёт ошибку $\sim 1/M$.
# (б) Спектр восстановленного сигнала несёт пики на $1$ Гц (переживший тон)
# и на $2$ Гц (alias тона $3$ Гц); пика на $3$ Гц нет.

# %%
print("4. Теорема отсчётов: sinc-восстановление и aliasing")

F1, F2 = 1.0, 3.0
FS_OK, FS_ALIAS = 8.0, 5.0
N_S = 40


def two_tone(t):
    return np.sin(2.0 * np.pi * F1 * t) + np.sin(2.0 * np.pi * F2 * t)


def sinc_rec(t_eval, ts, xs):
    """Truncated Shannon series from samples (ts, xs), Ts = ts[1] - ts[0]."""
    ts_gap = ts[1] - ts[0]
    return (xs[None, :] * np.sinc((t_eval[:, None] - ts[None, :]) / ts_gap)
            ).sum(axis=1)


ts_a = np.arange(N_S) / FS_OK
xs_a = two_tone(ts_a)
t_ev = np.linspace(1.0, 3.0, 2401)
x_rec_a = sinc_rec(t_ev, ts_a, xs_a)
err_a = float(np.max(np.abs(x_rec_a - two_tone(t_ev))))
print(f"  (а) f_s = 8 Гц > 2 f_2 = 6 Гц: max |x_rec - x| на [1, 3] = {err_a:.3e}")

ts_b = np.arange(N_S) / FS_ALIAS
xs_b = two_tone(ts_b)
x_rec_b = sinc_rec(t_ev, ts_b, xs_b)
err_b = float(np.max(np.abs(x_rec_b - two_tone(t_ev))))
print(f"  (б) f_s = 5 Гц < 2 f_2 = 6 Гц: max |x_rec - x| на [1, 3] = {err_b:.3e}")
print("      ожидание alias: sin(6 pi n/5) = -sin(4 pi n/5), тон 3 Гц -> 2 Гц")


# spectra of original vs reconstructed on exactly 3 s (integer periods of
# 1 Hz and 2 Hz tones, so a rectangular window leaks nothing)
def top_peaks(sig, n_pts=6000, rate=2000.0, n_peaks=4):
    """Local maxima of |rfft| below the Nyquist frequency, by amplitude."""
    spec = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(n_pts, d=1.0 / rate)
    cand = []
    for k in range(1, len(spec) - 1):
        if spec[k] >= spec[k - 1] and spec[k] >= spec[k + 1]:
            cand.append(k)
    cand = sorted(cand, key=lambda k: -spec[k])[:n_peaks]
    return sorted((float(freqs[k]), float(spec[k])) for k in cand)


t_sp = 0.5 + np.arange(6000) / 2000.0
for fs, ts, xs, rec, tag in ((FS_OK, ts_a, xs_a, x_rec_a, "8 Гц"),
                             (FS_ALIAS, ts_b, xs_b, x_rec_b, "5 Гц")):
    peaks_ref = top_peaks(two_tone(t_sp))
    peaks_rec = top_peaks(sinc_rec(t_sp, ts, xs))
    print(f"  спектры при f_s = {tag}: исходный пики "
          + ", ".join(f"{f:.3f} Гц" for f, _ in peaks_ref[:3]))
    print(f"      восстановленный пики "
          + ", ".join(f"{f:.3f} Гц (A={a:.0f})" for f, a in peaks_rec[:3]))

# %%
fig, axes = plt.subplots(2, 2, figsize=(9.6, 5.6), layout="constrained")

axes[0, 0].plot(t_ev, two_tone(t_ev), "-", linewidth=1.2, color="black",
                label="исходный $x$")
axes[0, 0].plot(t_ev, x_rec_a, "--", linewidth=1.1,
                label="восстановленный")
axes[0, 0].set_title("(а) $f_s = 8$ Гц $> 2 f_2$: восстановление", fontsize=9)
axes[0, 0].set_xlabel("$t$, с")
axes[0, 0].set_ylabel("$x(t)$")
axes[0, 0].legend(fontsize=7, loc="upper right")

freqs_sp = np.fft.rfftfreq(6000, d=1.0 / 2000.0)
sel_band = freqs_sp <= 4.5
spec_a = np.abs(np.fft.rfft(sinc_rec(t_sp, ts_a, xs_a)))
spec_ref = np.abs(np.fft.rfft(two_tone(t_sp)))
axes[0, 1].plot(freqs_sp[sel_band], spec_ref[sel_band],
                "-", linewidth=1.0, color="black", label="исходный")
axes[0, 1].plot(freqs_sp[sel_band], spec_a[sel_band],
                "--", linewidth=1.0, label="восстановленный")
axes[0, 1].set_title("(а) спектры: пики на 1 и 3 Гц", fontsize=9)
axes[0, 1].set_xlabel("$f$, Гц")
axes[0, 1].set_ylabel("$|X(f)|$")
axes[0, 1].legend(fontsize=7)

axes[1, 0].plot(t_ev, two_tone(t_ev), "-", linewidth=1.2, color="black",
                label="исходный $x$")
axes[1, 0].plot(t_ev, x_rec_b, "--", linewidth=1.1,
                label="восстановленный")
axes[1, 0].set_title("(б) $f_s = 5$ Гц $< 2 f_2$: alias", fontsize=9)
axes[1, 0].set_xlabel("$t$, с")
axes[1, 0].set_ylabel("$x(t)$")
axes[1, 0].legend(fontsize=7, loc="upper right")

spec_rec_b = np.abs(np.fft.rfft(sinc_rec(t_sp, ts_b, xs_b)))
axes[1, 1].plot(freqs_sp[sel_band], spec_ref[sel_band],
                "-", linewidth=1.0, color="black", label="исходный")
axes[1, 1].plot(freqs_sp[sel_band], spec_rec_b[sel_band],
                "--", linewidth=1.0, label="восстановленный")
axes[1, 1].axvline(2.0, color="grey", linestyle=":", linewidth=1.0)
axes[1, 1].set_title("(б) спектры: пик ушёл с 3 Гц на 2 Гц", fontsize=9)
axes[1, 1].set_xlabel("$f$, Гц")
axes[1, 1].set_ylabel("$|X(f)|$")
axes[1, 1].legend(fontsize=7)
fig.savefig(f"{FIGDIR}/fig-06.pdf", **SAVE_KW)

# %% [markdown]
# **Вывод (4).** — заполняется по факту прогона (второй заход).
