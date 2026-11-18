---
title: "Вопрос 12. Закон больших чисел. Центральная предельная теорема"
---

**Основной результат.** Слабый закон больших чисел: для независимых
одинаково распределённых величин с конечным математическим ожиданием
$\overline X_N\xrightarrow{\Prob}a$ (Хинчин); усиленный: при
$\E\abs{\xi_1}<\infty$ $\overline X_N\to a$ почти наверное (Колмогоров). Центральная
предельная теорема: $(S_N-Na)/(\sigma\sqrt N)\xrightarrow{d}\mathcal N(0,1)$
— нормированная сумма асимптотически нормальна; главное приложение —
асимптотический доверительный интервал $\overline X_N\pm
z_{1-\alpha/2}\widehat\sigma_N/\sqrt N$.

## Определения

- **Виды сходимости:** почти наверное $\Prob\{\zeta_N\to\zeta\}=1$; по
вероятности $\Prob\{\abs{\zeta_N-\zeta}\ge\varepsilon\}\to0$; в среднем
порядка $r$: $\E\abs{\zeta_N-\zeta}^r\to0$ (нам нужен $L_2$); по
распределению $F_{\zeta_N}(x)\to F_\zeta(x)$ в точках непрерывности $F_\zeta$.
- **Оценка** $\widehat\theta_N$: состоятельна ($\xrightarrow{\Prob}$), сильно
состоятельна ($\xrightarrow{\text{п.н.}}$), асимптотически нормальна:
$\sqrt N(\widehat\theta_N-\theta)\xrightarrow{d}\mathcal N(0,\sigma^2(\theta))$.
- **Характеристическая функция:** $\varphi_\xi(t)=\E e^{it\xi}$.

## Леммы

- **Бореля — Кантелли.** $\sum_N\Prob(A_N)<\infty$ $\Rightarrow$ почти наверное
наступает лишь конечное число $A_N$.
- **О сходимости распределений (Леви, наша форма).** Если
$\E\zeta_N=0$, $\E\zeta_N^2=1$ и $\varphi_{\zeta_N}(t)\to e^{-t^2/2}$ при каждом
$t$, то $F_{\zeta_N}\to\Phi$ поточечно.
- **Гауссовское сглаживание [A]–[R].** $p_\delta(y)=\frac{1}{\delta\sqrt{2\pi}}e^{-y^2/(2\delta^2)}
=\frac{1}{2\pi}\int e^{-\delta^2t^2/2}e^{-ity}\,dt$.

## Теоремы

- **Т1 (Марков — Чебышёв).** $\Prob\{\abs{\xi}\ge a\}\le\E f(\abs{\xi})/f(a)$
для неубывающей $f\ge0$; частные случаи: Маркова $\E\abs{\xi}/a$ и Чебышёва
$\Prob\{\abs{\xi-\E\xi}\ge a\}\le\Var\xi/a^2$ (множитель $1/a^2$ точен).
- **Т2 (Йенсен).** $\psi$ выпукла $\Rightarrow$ $\psi(\E\xi)\le\E\psi(\xi)$;
следствие Ляпунова: $(\E\abs{\xi}^s)^{1/s}\le(\E\abs{\xi}^t)^{1/t}$ при $s<t$.
- **Т3 (соотношения сходимостей).** п.н. $\Rightarrow$ $\xrightarrow{\Prob}$
$\Rightarrow$ $\xrightarrow{d}$; $L_2$ $\Rightarrow$ $\xrightarrow{\Prob}$;
обратные импликации неверны.
- **Т4 (слабый ЗБЧ, Чебышёв).** Попарно некоррелированные $\xi_k$,
$\Var\xi_k\le C$: $\Prob\{\abs{\overline X_N-\E\overline X_N}\ge\varepsilon\}
\le\frac{C}{N\varepsilon^2}\to0$. Следствия: для независимых
одинаково распределённых $\overline X_N\xrightarrow{\Prob}a$ с оценкой
$\frac{\sigma^2}{N\varepsilon^2}$; теорема Бернулли: $\mu_N/N\xrightarrow{\Prob}p$.
- **Т5 (ЗБЧ Хинчина).** Независимые одинаково распределённые с
$\E\abs{\xi_1}<\infty$: $\overline X_N\xrightarrow{\Prob}a$ — без конечной
дисперсии.
- **Т6 (неравенство Колмогорова).** Независимые центрированные $\xi_k$:
$\Prob\{\max_{k\le N}\abs{S_k}\ge\varepsilon\}\le\Var S_N/\varepsilon^2$.
- **Т7 (усиленный ЗБЧ, Колмогоров).** Независимые одинаково распределённые
с $\E\abs{\xi_1}<\infty$: $\overline X_N\xrightarrow{\text{п.н.}}a$
(усечение $+$ лемма Колмогорова $\sum_k\Var\zeta_k/k^2<\infty$
$\Rightarrow$ $(S_N-\E S_N)/N\to0$ п.н. $+$ лемма Бореля — Кантелли).
- **Т8 (свойства характеристической функции).** $\abs{\varphi}\le1$;
$\varphi_{c\xi+b}(t)=e^{itb}\varphi_\xi(ct)$; на независимых слагаемых
$\varphi_{S_N}=\prod\varphi_{\xi_k}$; при $\E\xi^2<\infty$
$\varphi(t)=1+it\E\xi-\frac{t^2}{2}\E\xi^2+o(t^2)$; для
$\zeta\sim\mathcal N(0,\sigma^2)$: $\varphi_\zeta=e^{-\sigma^2t^2/2}$.
- **Т9 (ЦПТ).** Независимые одинаково распределённые, $0<\sigma^2<\infty$:
$\Prob\bigl\{\frac{S_N-Na}{\sigma\sqrt N}\le x\bigr\}\to\Phi(x)$ равномерно по
$x$. Следствия: Муавр — Лаплас
$\frac{\mu_N-Np}{\sqrt{Npq}}\xrightarrow{d}\mathcal N(0,1)$;
$\Prob\{\abs{\zeta_N}\le c\}\to2\Phi(c)-1$.
- **Т10 (объём выборки и доверительный интервал).** Безусловная оценка по
Чебышёву: $N\ge N_{\text{Ч}}=\frac{\sigma^2}{\alpha\varepsilon^2}$;
асимптотическая по ЦПТ: $N\ge N_{\text{ЦПТ}}=\bigl(\frac{z_{1-\alpha/2}\sigma}{\varepsilon}\bigr)^2$;
$N_{\text{Ч}}/N_{\text{ЦПТ}}=1/(\alpha z_{1-\alpha/2}^2)$; выборочная
дисперсия $\widehat\sigma_N^2\xrightarrow{\Prob}\sigma^2$ (по Т5), поэтому
$\Prob\{\abs{\overline X_N-a}\le z_{1-\alpha/2}\widehat\sigma_N/\sqrt N\}\to1-\alpha$.

## Методы и алгоритмы

- **Метод Монте-Карло [A].** $u_k$ независимы, $U[0,1]$,
$I=\int_0^1g\,dx$: $I_N=\frac1N\sum_kg(u_k)\xrightarrow{\Prob}I$ (ЗБЧ),
контроль погрешности по ЦПТ — подробно в вопросе 17 (метод Монте-Карло).

*Приложения [A]–[R]:* **Теорема (Линдеберг).** Независимые центрированные $\xi_k$,
$s_N^2=\Var S_N$: условие $\frac{1}{s_N^2}\sum_k\E[\xi_k^2\mathbb 1_{\{\abs{\xi_k}\ge\tau s_N\}}]\to0$
влечёт $S_N/s_N\xrightarrow{d}\mathcal N(0,1)$; достаточное условие Ляпунова:
$\frac{1}{s_N^{2+\delta}}\sum_k\E\abs{\xi_k}^{2+\delta}\to0$; для одинаково
распределённых условие автоматически. **Эмпирическая функция распределения:**
$F_N^{*}(x)\to F(x)$ почти наверное,
$\sqrt N(F_N^{*}(x)-F(x))\xrightarrow{d}\mathcal N(0,F(x)(1-F(x)))$.
