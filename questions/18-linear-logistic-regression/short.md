---
title: "Вопрос 18. Линейная и логистическая регрессии"
---

**Основной результат.** МНК: $\hat\theta$ — решение нормальных уравнений
$A^{\T}A\hat\theta=A^{\T}y$, при полном ранге $\hat\theta=(A^{\T}A)^{-1}A^{\T}y$;
при гауссовских ошибках $\hat\theta\sim\mathcal N(\theta,\sigma^2(A^{\T}A)^{-1})$
и МНК — наилучшая линейная несмещённая оценка (Гаусс — Марков). Гребневая
регрессия $\hat\theta_\alpha=(A^{\T}A+\alpha I)^{-1}A^{\T}y$ — единственное
решение при $\alpha>0$, спектральный фильтр $d_j^2/(d_j^2+\alpha)$ сжимает
малые сингулярные направления. Логистическая регрессия: оценка МП для
log-loss — выпуклая задача, решаемая методом Ньютона = IRLS.

## Определения

- **Линейная модель:** $y=A\theta+\varepsilon$, $\theta\in\R^n$ неизвестно.
- **МНК:** $\hat\theta\in\argmin_\theta\norm{A\theta-y}_2^2$; $Q(\hat\theta)$ —
остаточная сумма квадратов.
- **Псевдообратная Мура — Пенроуза:** $A^{+}=V\Sigma^{+}U^{\T}$ по SVD;
решение минимальной нормы $\hat\theta=A^{+}y$ (при полном ранге
$A^{+}=(A^{\T}A)^{-1}A^{\T}$).
- **Гребневая регрессия:** $\hat\theta_\alpha=\argmin_\theta\norm{A\theta-y}_2^2+\alpha\norm{\theta}_2^2$;
$\mathrm{df}(\alpha)=\tr[A(A^{\T}A+\alpha I)^{-1}A^{\T}]=\sum_j\frac{d_j^2}{d_j^2+\alpha}$
— эффективное число степеней свободы, от $r$ до $0$.
- **Логистическая модель:** $\Prob(y_i=1\mid a_i)=s(a_i^{\T}\theta)$,
$s(z)=1/(1+e^{-z})$; классификация $\hat y_i=\mathbb 1\{a_i^{\T}\theta>0\}$,
граница $a^{\T}\theta=0$.
- **Log-loss (бинарная перекрёстная энтропия):**
$Q(\theta)=\sum_i\bigl[\ln(1+e^{a_i^{\T}\theta})-y_ia_i^{\T}\theta\bigr]=-\ln L(\theta)$;
ОМП $=\argmin Q$.

## Теоремы

- **Т1 (нормальные уравнения).** $\hat\theta$ минимизирует $Q$
$\Leftrightarrow$ $A^{\T}(A\hat\theta-y)=0$; при $\rank A=n$ решение
единственно: $(A^{\T}A)^{-1}A^{\T}y$. При неполном ранге решения непусты,
выпуклы, замкнуты; минимальная норма — $\sum_{j\le r}\frac{u_j^{\T}y}{d_j}v_j$.
- **Т2 (несмещённость и ковариация).** При посылках (G):
$\E\hat\theta=\theta$, $\Var\hat\theta=\sigma^2(A^{\T}A)^{-1}$;
$\hat\sigma^2=\frac{\norm{y-A\hat\theta}_2^2}{m-n}$ несмещена
($\E\norm{P\varepsilon}^2=\sigma^2k$ для симметричной идемпотентной $P$
ранга $k$).
- **Т3 (Гаусс — Марков).** При (G) $\hat\theta$ — BLUE: для любой линейной
несмещённой $\tilde\theta=Cy$ ($CA=I$) выполнено
$\Var\hat\theta\preceq\Var\tilde\theta$, равенство только при
$C=(A^{\T}A)^{-1}A^{\T}$.
- **Т4 (МНК = максимальное правдоподобие).** При
$\varepsilon\sim\mathcal N(0,\sigma^2I)$: ОМП $\theta$ совпадает с МНК;
$\hat\theta\sim\mathcal N(\theta,\sigma^2(A^{\T}A)^{-1})$;
$(m-n)\hat\sigma^2/\sigma^2\sim\chi^2_{m-n}$; $\hat\theta$ и $\hat\sigma^2$
независимы; ОМП дисперсии смещена: $\E\hat\sigma^2_{\mathrm{ML}}=\frac{m-n}{m}\sigma^2$.
- **Т5 (численная сторона) [A].** Нормальные уравнения в машинной арифметике:
относительная ошибка $\approx u\,\kappa_2(A)^2$. QR-разложение ($A=QR$,
тонкое): $R\hat\theta=Q^{\T}y$, невязка $\norm{(I-QQ^{\T})y}^2$; обратная
устойчивость ($\norm{\delta A}\lesssim mu\norm{A}$), ошибка
$O(u(\kappa_2(A)+\tan\varphi\,\kappa_2(A)^2))$. Чувствительность:
$\frac{\norm{\hat\theta-\theta_{LS}}}{\norm{\theta_{LS}}}\le
\delta\bigl\{\frac{2\kappa_2(A)}{\cos\varphi}+\tan\varphi\,\kappa_2(A)^2\bigr\}+O(\delta^2)$.
- **Т6 (гребень).** При $\alpha>0$ решение единственно и линейно по $y$;
$\hat\theta_\alpha=\sum_j\frac{d_j}{d_j^2+\alpha}(u_j^{\T}y)v_j$,
$A\hat\theta_\alpha=\sum_j\frac{d_j^2}{d_j^2+\alpha}(u_j^{\T}y)u_j$ — фильтр
сжимает малые $d_j$; $\mathrm{df}(\alpha)$ непрерывно убывает от $r$ до $0$
(мост к вопросу о регуляризации).
- **Т7 (логистическая регрессия).** $Q$ выпукла, гладкая:
$\nabla Q=A^{\T}(p-y)$, $\nabla^2Q=A^{\T}WA\succeq0$,
$W=\diag(p_i(1-p_i))$; при $\rank A=n$ и $0<p_i<1$ строго выпукла; минимум
существует $\Leftrightarrow$ условие перекрытия (для любого $z$ с $Az\ne0$
есть $i$ с $a_i^{\T}z>0$, $y_i=0$, или $a_i^{\T}z<0$, $y_i=1$).
## Методы и алгоритмы

- **Решение задачи МНК.** Нормальные уравнения $A^{\T}A\hat\theta=A^{\T}y$:
относительная ошибка в машинной арифметике $\approx u\,\kappa_2(A)^2$.
Тонкое QR-разложение $A=QR$: $\hat\theta$ из $R\hat\theta=Q^{\T}y$,
минимум $\norm{(I-QQ^{\T})y}^2$; обратная устойчивость
($\norm{\delta A}\lesssim mu\norm{A}$, $\norm{\delta y}\lesssim mu\norm{y}$),
ошибка $O(u(\kappa_2(A)+\tan\varphi\,\kappa_2(A)^2))$ — выбор при плохой
обусловленности.
- **Повторно взвешенные наименьшие квадраты (IRLS, Ньютон для log-loss).**
Шаг из точки $\theta$:
$\theta^{+}=\theta+(A^{\T}WA)^{-1}A^{\T}(y-p)$ совпадает с решением
взвешенной МНК с весами $W=\diag(p_i(1-p_i))$ и рабочим откликом
$z=A\theta+W^{-1}(y-p)$: $\theta^{+}=\arg\min_\vartheta(z-A\vartheta)^{\T}W(z-A\vartheta)$;
вес максимален у объектов у границы ($p_i\approx1/2$).
- **R², t- и F-критерии, softmax [A].** $R^2=1-\mathrm{RSS}/\mathrm{TSS}$
— доля объяснённой дисперсии; значимость модели — $F_{n-1,m-n}$, отдельных
коэффициентов — t-критерии; мультикласс ($K$ классов, $K-1$ логитов) —
softmax: $\Prob(G=k\mid a)\propto e^{a^{\T}\theta_k}$.
