---
title: "Методы минимизации функций многих переменных"
subtitle: "Вопрос 10"
author: "Крохалев Е. М."
date: "2026"
---

# Постановка

## Зачем это нужно

- Оптимальное управление: функционал качества $\sum (x_k^{\mathsf T} Q x_k + u_k^{\mathsf T} R u_k)$ — квадратичная функция параметров.
- Машинное обучение: минимизация эмпирического риска по параметрам модели.
- Обратные задачи: минимизация функционала невязки.

## Формальная постановка

Дана $f\colon \mathbb{R}^n \to \mathbb{R}$. Требуется найти $x^* \in \arg\min f$ итерационным методом с заданной точностью.

$$\text{сквозной пример: квадратичная } \varphi(x) = \tfrac12 x^{\mathsf T} A x - b^{\mathsf T} x, \quad \varkappa = \mu_{\max}/\mu_{\min} \gg 1$$

# Основные результаты

## Определения

\begin{definition}
$f$ \textbf{гладкая с константой $L$}, если $\norm{\nabla f(y) - \nabla f(x)} \le L \norm{y - x}$; эквивалентно $f(y) \le f(x) + \scal{\nabla f(x)}{y-x} + \frac{L}{2}\norm{y-x}^2$.
\end{definition}

\begin{definition}
$f$ \textbf{сильно выпукла с константой $\mu$}, если $f(y) \ge f(x) + \scal{\nabla f(x)}{y-x} + \frac{\mu}{2}\norm{y-x}^2$.
\end{definition}

\textbf{Число обусловленности:} $\varkappa = L/\mu$.

## Условия оптимальности

\begin{theorem}[необходимые и достаточные условия экстремума]
Пусть $f$ дифференцируема в точке $x^*$.

\textbf{(а)} Если $x^*$ — локальный минимум, то $\nabla f(x^*) = 0$.

\textbf{(б)} Если $f$ дважды дифференцируема в $x^*$ и $x^*$ — локальный минимум, то $\nabla^2 f(x^*) \succeq 0$.

\textbf{(в)} Если $f$ дважды дифференцируема в окрестности $x^*$, $\nabla f(x^*) = 0$ и $\nabla^2 f(x^*) \succ 0$, то $x^*$ — строгий локальный минимум.
\end{theorem}

## Градиентный спуск

$$x^{k+1} = x^k - \gamma_k \nabla f(x^k), \qquad \gamma_k = \tfrac1L$$

\begin{lemma}[спуск]
Пусть $f$ $L$-гладка. Тогда $f\bigl(x - \tfrac1L \nabla f(x)\bigr) \le f(x) - \frac{1}{2L} \norm{\nabla f(x)}^2$.
\end{lemma}

## Теорема о сходимости

\begin{theorem}[сходимость градиентного спуска]
Пусть $f$ $L$-гладкая, $x^{k+1} = x^k - \frac1L \nabla f(x^k)$, $R = \norm{x^0 - x^*}$.

\textbf{(а) Невыпуклый случай.} $f$ ограничена снизу: $f \ge f_* > -\infty$. Тогда $\min_{0 \le k \le N-1} \norm{\nabla f(x^k)} \le \sqrt{\frac{2L(f(x^0) - f_*)}{N}}$.

\textbf{(б) Выпуклый случай.} $f$ выпукла. Тогда $f\bigl(\bar x^N\bigr) - f^* \le \frac{L R^2}{2N}$.

\textbf{(в) Сильно выпуклый случай.} $f$ $\mu$-сильно выпукла. Тогда $f(x^N) - f^* \le \frac{L R^2}{2} \min\bigl\{ \frac{1}{N}, (1 - \frac{\mu}{L})^{N} \bigr\}$.
\end{theorem}

## Где ломается

\begin{remark}[существенность посылок]
Каждая посылка существенна: без липшицевости градиента метод расходится ($\abs{x}^{2+\varepsilon}$); без ограниченности снизу градиент не обязан стремиться к нулю (линейная $f$); при $\gamma \ge 2/L$ — расходимость на $\frac12\norm{x}^2$; без сильной выпуклости скорость может быть сколь угодно малой.
\end{remark}

## Точная линейная оценка

\begin{theorem}[точная оценка для сильно выпуклых гладких функций]
Пусть $f$ дважды дифференцируема и $\mu I \preceq \nabla^2 f(x) \preceq L I$ для всех $x$. Тогда градиентный спуск с постоянным шагом $\gamma$ сходится: $\norm{x^k - x^*} \le \norm{x^0 - x^*} q^k$, $q(\gamma) = \max\{\abs{1 - \gamma\mu}, \abs{1 - \gamma L}\}$; минимум $q^* = \frac{\varkappa - 1}{\varkappa + 1}$ достигается при $\gamma^* = \frac{2}{L+\mu}$. Оценка неулучшаема.
\end{theorem}

$$N \approx \frac{\varkappa}{2} \ln \frac{1}{\varepsilon} \quad \text{итераций} \qquad (\varkappa = 10^3: \approx 3\,500)$$

## Нижние оценки

\begin{theorem}[нижние оценки]
\textbf{(а)} Существует $L$-гладкая выпуклая $f$ с $\norm{x^0-x^*} \le R$ такая, что для любого метода из $\operatorname{span}$ градиентов $f(x^N) - f^* \ge \frac{3 L R^2}{32 (N+1)^2}$.

\textbf{(б)} Для любых $L \ge \mu > 0$ существует $L$-гладкая $\mu$-сильно выпуклая $f$ такая, что $f(x^N) - f^* \ge \frac{\mu R^2}{2}\bigl(\frac{\sqrt\varkappa - 1}{\sqrt\varkappa + 1}\bigr)^{2N}$.
\end{theorem}

\textbf{Вывод:} градиентный спуск оптимален в классе гладких выпуклых; разрыв $O(\varkappa)$ против $O(\sqrt\varkappa)$ закрывается ускорением.

# Чебышёвское ускорение

## Идея

$$e^k = p_k(A) e^0, \quad p_k(0) = 1; \qquad \min \max_{\lambda \in [\mu, L]} \abs{p_k(\lambda)} \Rightarrow p_k^* = \frac{T_k(z(\lambda))}{T_k(z(0))}$$

$$\norm{e^k} \le q_k \norm{e^0}, \qquad q_k = \frac{2\sigma^k}{1 + \sigma^{2k}}, \qquad \sigma = \frac{\sqrt\varkappa - 1}{\sqrt\varkappa + 1}$$

(аппарат многочленов Чебышёва — вопрос 11)

## Устойчивая рекуррентность

$$y^{(k+1)} = \omega_{k+1}\bigl(y^{(k)} - y^{(k-1)} + \gamma z^{(k)}\bigr) + y^{(k-1)}, \quad \omega_{k+1} = 2\mu \frac{T_k(\mu)}{T_{k+1}(\mu)}$$

\begin{theorem}[сходимость чебышёвских полуитераций]
Итерации удовлетворяют $y^{(k)} - x^* = p_k(A)(y^{(0)} - x^*)$ с чебышёвским $p_k$, и $\norm{y^{(k)} - x^*} \le q_k \norm{y^{(0)} - x^*}$. Для точности $\varepsilon$ достаточно $k \ge \frac{\ln(2/\varepsilon)}{2\sqrt\xi}$ итераций, $\xi = \mu_{\min}/\mu_{\max}$.
\end{theorem}

\textbf{Сравнение:} $\frac{\varkappa}{2}\ln\frac1\varepsilon$ (стационарный шаг) против $\frac{\sqrt\varkappa}{2}\ln\frac2\varepsilon$ (чебышёв).

## Метод сопряжённых градиентов

$$x^{k+1} = x^k + \gamma_k h^k, \qquad h^{k+1} = -\nabla\varphi(x^{k+1}) + \beta_k h^k, \qquad \beta_k = \frac{\norm{\nabla\varphi(x^{k+1})}^2}{\norm{\nabla\varphi(x^k)}^2}$$

\begin{theorem}[конечность сопряжённых градиентов]
Итерации для квадратичной функции с положительно определённой $A$ находят точный минимум за не более чем $n$ шагов; $x^k$ минимизирует $\varphi$ на крыловском подпространстве.
\end{theorem}

\begin{theorem}[оценка сходимости CG]
$\norm{x^k - x^*}_A \le 2\sqrt{L/\mu}\, q^k \norm{x^0 - x^*}_A$, $q = \frac{\sqrt\varkappa - 1}{\sqrt\varkappa + 1}$.
\end{theorem}

\textbf{CG оптимален среди методов первого порядка и не требует знания спектра.}

## Что нужно знать

| Метод | Итерации | Требует |
|---|---|---|
| Градиентный $\gamma^*$ | $O(\varkappa \ln \frac1\varepsilon)$ | $\mu, L$ |
| Наискорейший спуск | не лучше стационарного | ничего |
| Чебышёвские полуитерации | $O(\sqrt\varkappa \ln \frac2\varepsilon)$ | $\mu, L$ |
| Сопряжённые градиенты | $O(\sqrt\varkappa \ln \frac1\varepsilon)$ | ничего |

# Методы второго порядка и ускорение

## Метод тяжёлого шарика

$$x^{k+1} = x^k - \alpha \nabla f(x^k) + \beta (x^k - x^{k-1})$$

\begin{theorem}[локальная сходимость тяжёлого шарика]
Пусть $\mu I \preceq \nabla^2 f(x^*) \preceq L I$, $0 \le \beta < 1$, $0 < \alpha < \frac{2(1+\beta)}{L}$. Тогда локально $\norm{x^k - x^*} \le c (q + \delta)^k$; при $\alpha^* = \frac{4}{(\sqrt{L}+\sqrt{\mu})^2}$, $\beta^* = \bigl(\frac{\sqrt{L}-\sqrt{\mu}}{\sqrt{L}+\sqrt{\mu}}\bigr)^2$ знаменатель $q^* = \frac{\sqrt\varkappa - 1}{\sqrt\varkappa + 1}$.
\end{theorem}

## Метод Нестерова

$$x^{k+1} = y^k - \tfrac1L \nabla f(y^k), \qquad y^{k+1} = x^{k+1} + \tfrac{k}{k+3}(x^{k+1} - x^k)$$

\begin{theorem}[сходимость метода Нестерова]
Для $L$-гладкой выпуклой $f$: $f(x^N) - f^* \le \frac{2 L R^2}{(N+1)^2}$. Для $\mu$-сильно выпуклой с постоянным коэффициентом $\frac{\sqrt\varkappa - 1}{\sqrt\varkappa + 1}$: $f(x^N) - f^* \le \frac{L R^2}{2}\bigl(\frac{\sqrt\varkappa - 1}{\sqrt\varkappa + 1}\bigr)^N$.
\end{theorem}

\textbf{Тяжёлый шарик vs Нестеров:} градиент в $x^k$ (локально) против градиента в $y^k$ (глобально по нижним оценкам).

## Метод Ньютона

$$x^{k+1} = x^k - [\nabla^2 f(x^k)]^{-1} \nabla f(x^k)$$

\begin{theorem}[локальная квадратичная сходимость]
Пусть $f$ $\mu$-сильно выпукла, $\norm{\nabla^2 f(x) - \nabla^2 f(y)} \le M\norm{x-y}$, $q_0 = \frac{M}{2\mu^2}\norm{\nabla f(x^0)} < 1$. Тогда $\norm{x^k - x^*} \le \frac{2\mu}{M} q_0^{2^k}$.
\end{theorem}

## Теорема Ньютона—Канторовича

\begin{theorem}[Ньютона—Канторовича]
Пусть $\norm{\Gamma_0 \nabla f(x^0)} \le \eta$, $\norm{\Gamma_0 \nabla^2 f(x)} \le K$ в шаре радиуса $r$, $\Gamma_0 = [\nabla^2 f(x^0)]^{-1}$, $h = K\eta \le \frac12$, $r \ge r_0 = \frac{1-\sqrt{1-2h}}{h}\eta$. Тогда в шаре существует решение $\nabla f(x) = 0$, к которому сходится метод Ньютона, и $\norm{x^k - x^*} \le \frac{1}{2^k}(2h)^{2^k}\frac{\eta}{h}$; решение единственно при $r < \frac{1+\sqrt{1-2h}}{h}\eta$.
\end{theorem}

# Условная оптимизация

## Проекция

$$\proj_Q(y) = \argmin_{x \in Q} \tfrac12 \norm{x-y}^2$$

\begin{theorem}[критерий проекции]
$x = \proj_Q(y)$ тогда и только тогда, когда $\scal{y-x}{z-x} \le 0$ для всех $z \in Q$; проекция нерасширяющая: $\norm{\proj_Q(y) - \proj_Q(y')} \le \norm{y - y'}$.
\end{theorem}

## Метод проекции градиента

$$x^{k+1} = \proj_Q\bigl(x^k - \tfrac1L \nabla f(x^k)\bigr)$$

\begin{theorem}[сходимость PGD]
$f$ выпукла, $L$-гладка на выпуклом замкнутом $Q$ $\Rightarrow$ $f(x^N) - f^* \le \frac{L R^2}{2N}$.
\end{theorem}

## Метод Франка—Вулфа

$$y^k = \argmin_{y \in Q} \scal{\nabla f(x^k)}{y}, \qquad x^{k+1} = x^k + \gamma_k (y^k - x^k), \quad \gamma_k = \tfrac{2}{k+2}$$

\begin{theorem}[сходимость Франка—Вулфа]
$f$ выпукла, $L$-гладка на выпуклом компакте $Q$ диаметра $R$ $\Rightarrow$ $f(x^N) - f^* \le \frac{2LR^2}{N+2}$.
\end{theorem}

\textbf{Оговорка:} оценка $O(1/k^2)$ для сильно выпуклых функций, встречающаяся в изложениях, неверна — доказательство отбрасывает полезный член с $\mu$; контрпример $f(x) = \norm{x}^2$ на симплексе с tie-breaking «свежая вершина» сходится как $\Theta(1/k)$ (измерено).

## Где это работает

| Постановка | Что минимизируется | Метод |
|---|---|---|
| ЛК-регулятор | $\sum (x^{\mathsf T}Qx + u^{\mathsf T}Ru)$ | CG, Ньютон |
| Обратная задача | $\norm{Aq - f}^2$ (+ регуляризатор) | градиентный через сопряжённую задачу |
| Обучение модели | эмпирический риск | SGD + импульс (= тяжёлый шарик) |
| Задача на симплексе | гладкая выпуклая | Франк—Вулф (линейный оракул) |

# Численная иллюстрация

## Что считаем

:::::: columns
::: {.column width="52%"}

- Квадратичная задача, $\varkappa = 10^3$: измеренный знаменатель градиентного спуска совпал с $q^* = (\varkappa-1)/(\varkappa+1)$.
- Тяжёлый шарик и Нестеров: $q \approx 0{,}94$ против $0{,}998$ — выигрыш $\approx 30$ раз.
- Стресс-тест: трёхчленная чебышёвская форма устойчива, множительная взрывается.
- FW на симплексе: $\Theta(1/k)$, наклон $-1{,}07$.

:::
::: {.column width="48%"}

![](figures/fig-02.pdf){width=100%}

:::
::::::

## Итог

## Что нужно помнить

- Лемма спуска $\Rightarrow$ три режима сходимости градиентного спуска; все оценки точны по порядку.
- Плохая обусловленность — враг: $O(\varkappa)$ итераций; ускорение покупается чебышёвским набором, тяжёлым шариком, Нестеровым или CG — $O(\sqrt\varkappa)$.
- Чебышёвская трёхчленная рекуррентность устойчива, множительная форма — нет.
- Ньютон: квадратичная сходимость локально; теорема Канторовича — полулокальная гарантия.
- Условная оптимизация: PGD (проекция) и Франк—Вулф (линейный оракул) — оба $O(1/k)$.
