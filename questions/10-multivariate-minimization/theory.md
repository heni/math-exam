---
title: "Вопрос 10. Методы минимизации функций многих переменных"
subtitle: "Конспект теории с полными доказательствами"
author: "Крохалев Е. М."
date: "2026"
---

\part{Основная часть}

# Постановка и мотивация {#sec:intro}

Большинство задач математического моделирования сводится в конце концов к
одной и той же вычислительной процедуре: найти минимум функции многих
переменных. Три типовых источника такой постановки.

**Оптимальное управление.** Для дискретной линейной системы
$x_{k+1} = Ax_k + Bu_k$ функционал качества
$$
J = \sum_{k=0}^{N-1} \bigl( x_k^{\mathsf T} Q x_k + u_k^{\mathsf T} R u_k \bigr)
$$
является квадратичной функцией совокупности управлений $u_0, \dots, u_{N-1}$
и начальных условий; задача оптимального управления — это задача о его
минимуме. При доминировании одного из слагаемых (малый вес $R$ — дешёвое
управление, малый вес $Q$ — слабый штраф за отклонение) гессиан функционала
становится плохо обусловленным, и именно здесь методы минимизации
начинают различаться по скорости на порядки.

**Машинное обучение.** Обучение — это минимизация эмпирического риска
$\frac1N \sum_{i=1}^N \ell(h_w(x_i), y_i)$ по параметрам $w$ модели; обучение
нейронной сети — минимизация этой функции в пространстве размерности в
миллионы. Метод, которым это делается на практике (стохастический
градиентный спуск с импульсом), есть пересборка методов, разобранных в этом
конспекте, — см. расширенную часть и вопрос 02 (ML и обратные задачи).

**Обратные задачи.** Восстановление параметров модели по измерениям сводится
к минимизации функционала невязки между измеренным и модельным выходом;
способ вычислить градиент такого функционала через сопряжённую задачу
дан в расширенной части (раздел \ref{sec:adjoint}).

Формальная постановка этого вопроса программы: дана функция
$f \colon \mathbb{R}^n \to \mathbb{R}$; требуется найти
$$
x^* \in \argmin_{x \in \mathbb{R}^n} f(x)
$$
и построить итерационный метод, вычисляющий $x^*$ с заданной точностью.
Отдельно разбирается условная постановка $\min_{x \in Q} f(x)$ с выпуклым
замкнутым $Q$ (раздел \ref{sec:constrained}) — это акцент лекции курса
прошлого года.

**Сквозной пример.** Через весь конспект и через численные примеры ведётся
одна задача: минимизация квадратичной функции
$$
\varphi(x) = \tfrac12 x^{\mathsf T} A x - b^{\mathsf T} x,
\qquad
\mu_{\min} I \preceq A \preceq \mu_{\max} I,
\qquad
\varkappa = \frac{\mu_{\max}}{\mu_{\min}},
$$
с плохо обусловленной матрицей $A$ (число обусловленности $\varkappa$ велико).
Она — конечномерный образец функционала качества: та же структура
$\tfrac12 (\text{форма}) - (\text{линейная часть})$ и та же плохая
обусловленность, что и у дискретного функционала качества управления выше,
а главное — на квадратичной функции все оценки сходимости методов
первого порядка точны и проверяются расчётом
(пример 1 `examples.ipynb`).

Конспект двухчастный. Основная часть — градиентные методы и теория их
скорости сходимости: лемма спуска, теорема о сходимости в трёх режимах,
точная линейная оценка, нижние оценки, чебышёвское ускорение и метод
сопряжённых градиентов. Расширенная часть — методы второго порядка
(Ньютон, Ньютон—Канторович), ускоренные методы (тяжёлый шарик, Нестеров),
условная и негладкая оптимизация и приложение к обратным задачам.

# Определения и обозначения {#sec:defs}

Все нормы — евклидовы, $\norm{x} = \sqrt{\scal{x}{x}}$; через $I$ обозначен
единичный оператор. Функция $f$ называется \textbf{выпуклой}, если
$$
f(\lambda x + (1-\lambda) y) \le \lambda f(x) + (1-\lambda) f(y)
\qquad \forall x, y,\ \forall \lambda \in [0,1].
$$

\begin{definition}\label{def:smooth}
Функция $f$ \textbf{гладкая с константой $L$} ($L$-гладкая), если она
дифференцируема и её градиент липшицев с константой $L$:
\begin{equation}\label{eq:lipgrad}
\norm{\nabla f(y) - \nabla f(x)} \le L \norm{y - x}
\qquad \forall x, y .
\end{equation}
\end{definition}

\begin{definition}\label{def:strong}
Функция $f$ \textbf{сильно выпукла с константой $\mu > 0$}
($\mu$-сильно выпукла), если
\begin{equation}\label{eq:strong}
f(y) \ge f(x) + \scal{\nabla f(x)}{y - x} + \frac{\mu}{2} \norm{y - x}^2
\qquad \forall x, y .
\end{equation}
\end{definition}

\begin{notation}
$L$ — константа Липшица градиента \eqref{eq:lipgrad}; $\mu$ — константа
сильной выпуклости \eqref{eq:strong}; $\varkappa = L / \mu \ge 1$ — число
обусловленности задачи (так же, как в вопросе 11 о многочленах Чебышёва);
$x^*$ — точка минимума, $f^* = f(x^*)$; $\gamma_k$ — шаг метода на итерации
$k$. У сильно выпуклой функции минимум единствен (теорема
\ref{thm:exist-unique}).
\end{notation}

\begin{proposition}[эквивалентные формы]\label{prop:equiv}
Для дифференцируемой $f$:

\textbf{(а)} гладкость \eqref{eq:lipgrad} эквивалентна мажорированию
квадратичной формой с кривизной $L$:
\begin{equation}\label{eq:upper-par}
f(y) \le f(x) + \scal{\nabla f(x)}{y - x} + \frac{L}{2} \norm{y - x}^2 ;
\end{equation}

\textbf{(б)} для дважды дифференцируемой $f$ условие \eqref{eq:lipgrad}
равносильно $\nabla^2 f(x) \preceq L I$ для всех $x$;

\textbf{(в)} сильная выпуклость \eqref{eq:strong} для дважды
дифференцируемой $f$ равносильна $\nabla^2 f(x) \succeq \mu I$ для всех $x$;

\textbf{(г)} для гладкой выпуклой $f$ сильная выпуклость \eqref{eq:strong}
равносильна сильной монотонности градиента:
\begin{equation}\label{eq:mono}
\scal{\nabla f(y) - \nabla f(x)}{y - x} \ge \mu \norm{y - x}^2 .
\end{equation}
\end{proposition}

\begin{proof}
\textbf{(а)} Пусть $g(t) = f(x + t(y-x))$, $t \in [0,1]$. Тогда
$g'(t) = \scal{\nabla f(x + t(y-x))}{y - x}$ и
$$
f(y) - f(x) - \scal{\nabla f(x)}{y - x}
= \int_0^1 \scal{\nabla f(x + t(y-x)) - \nabla f(x)}{y - x}\, dt
\le \int_0^1 L t \norm{y-x}^2\, dt = \frac{L}{2}\norm{y-x}^2 .
$$
Обратно: из \eqref{eq:upper-par} для $y = x + h$, $h = t d$,
$$
f(x + td) - f(x) \le t \scal{\nabla f(x)}{d} + \frac{L t^2}{2} \norm{d}^2 ,
$$
аналогично снизу с $x \to x + td$, $y = x$: после перестановки и
предельного перехода $t \to 0$ получаем
$\scal{\nabla f(x + td) - \nabla f(x)}{d} \le L t \norm{d}^2$, что при
$d = y - x$ даёт \eqref{eq:lipgrad}.

\textbf{(б)} Если $\nabla^2 f \preceq L I$, то по формуле Тейлора с
остатком в интегральной форме
$$
\nabla f(y) - \nabla f(x) = \int_0^1 \nabla^2 f(x + t(y-x)) (y-x)\, dt,
$$
откуда \eqref{eq:lipgrad}. Обратно: применяя \eqref{eq:lipgrad} к
$y = x + td$ и устремляя $t \to 0$, получаем
$\norm{\nabla^2 f(x) d} \le L \norm{d}$ для всех $d$, то есть
$\nabla^2 f(x) \preceq L I$.

\textbf{(в)} Достаточность: формула Тейлора с остатком в интегральной форме
и $\nabla^2 f \succeq \mu I$ дают
$$
f(y) = f(x) + \scal{\nabla f(x)}{y-x} + \int_0^1 (1-t)\,
\scal{\nabla^2 f(x + t(y-x)) (y-x)}{y-x}\, dt
\ge f(x) + \scal{\nabla f(x)}{y-x} + \frac{\mu}{2}\norm{y-x}^2 .
$$
Необходимость: вычитая из \eqref{eq:strong} её же с переставленными $x, y$ и
применяя к разности формулу Тейлора, получаем
$\scal{\nabla^2 f(x)(y-x)}{y-x} \ge \mu \norm{y-x}^2$ в каждой точке $x$.

\textbf{(г)} Сильная монотонность следует из \eqref{eq:strong}, применённого
к паре $(y, x)$ и $(x, y)$ с последующим сложением. Обратно: для гладкой
выпуклой $f$ градиентное неравенство $f(x) \ge f(y) + \scal{\nabla f(y)}{x-y}$
(теорема \ref{thm:foc} при $n = 1$ и линейности) вместе с \eqref{eq:mono}
по цепочке
$$
f(y) \ge f(x) + \scal{\nabla f(x)}{y-x} + \scal{\nabla f(y) - \nabla f(x)}{y-x}
\ge f(x) + \scal{\nabla f(x)}{y-x} + \frac{\mu}{2}\norm{y-x}^2
$$
даёт \eqref{eq:strong}.
\end{proof}

\begin{remark}
Запись обозначений у разных школ различается ровно вдвое по $\mu$:
у [12] сильная выпуклость вводится слагаемым $\vartheta \lambda(1-\lambda)
\norm{x^1 - x^2}^2$ в определении, что соответствует $\mu = 2\vartheta$;
у [13] константа сильной выпуклости $l$ совпадает с нашей $\mu$. Формулы
источников при цитировании переведены на обозначения
определения \ref{def:strong}.
\end{remark}

\begin{theorem}[необходимые и достаточные условия экстремума]\label{thm:soc}
Пусть $f$ дифференцируема в точке $x^*$.

\textbf{(а)} Если $x^*$ — локальный минимум, то $\nabla f(x^*) = 0$.

\textbf{(б)} Если $f$ дважды дифференцируема в $x^*$ и $x^*$ — локальный
минимум, то $\nabla^2 f(x^*) \succeq 0$.

\textbf{(в)} Если $f$ дважды дифференцируема в окрестности $x^*$,
$\nabla f(x^*) = 0$ и $\nabla^2 f(x^*) \succ 0$, то $x^*$ — строгий локальный
минимум.
\end{theorem}

\begin{proof}
\textbf{(а)} Для любого $d$ и малых $t > 0$ по определению дифференцируемости
$0 \le f(x^* + td) - f(x^*) = t \scal{\nabla f(x^*)}{d} + o(t)$; деление на
$t$ и предельный переход дают $\scal{\nabla f(x^*)}{d} \ge 0$ для всех $d$,
откуда $\nabla f(x^*) = 0$.

\textbf{(б)} Формула Тейлора с остатком в форме Пеано:
$f(x^* + td) - f(x^*) = \tfrac{t^2}{2} \scal{\nabla^2 f(x^*) d}{d} + o(t^2) \ge 0$; деление на $t^2$ и предельный переход дают
$\scal{\nabla^2 f(x^*) d}{d} \ge 0$ для всех $d$.

\textbf{(в)} Матрица $\nabla^2 f$ непрерывна, поэтому в некотором шаре
$U$ вокруг $x^*$ выполнено $\nabla^2 f(x) \succeq \frac{\lambda_{\min}}{2} I$,
где $\lambda_{\min} > 0$ — наименьшее собственное значение
$\nabla^2 f(x^*)$. По формуле Тейлора с остатком в интегральной форме
$$
f(x) - f(x^*) = \int_0^1 (1-t)\, \scal{\nabla^2 f(x^* + t(x - x^*))\,(x-x^*)}{x-x^*}\, dt
\ge \frac{\lambda_{\min}}{4} \norm{x - x^*}^2 > 0
$$
для $x \in U$, $x \ne x^*$.
\end{proof}

\begin{theorem}[градиентное неравенство]\label{thm:foc}
Дифференцируемая $f$ выпукла тогда и только тогда, когда
\begin{equation}\label{eq:foc}
f(y) \ge f(x) + \scal{\nabla f(x)}{y - x} \qquad \forall x, y .
\end{equation}
Для выпуклой дифференцируемой $f$ точка $x^*$ — точка глобального
минимума тогда и только тогда, когда $\nabla f(x^*) = 0$.
\end{theorem}

\begin{proof}
Необходимость \eqref{eq:foc}: по определению выпуклости для
$\lambda \in (0,1)$
$$
f\bigl(x + \lambda(y - x)\bigr) \le f(x) + \lambda\bigl(f(y) - f(x)\bigr),
$$
откуда $\frac{f(x + \lambda(y-x)) - f(x)}{\lambda} \le f(y) - f(x)$;
предельный переход $\lambda \to 0+$ дает
$\scal{\nabla f(x)}{y - x} \le f(y) - f(x)$. Достаточность: применим
\eqref{eq:foc} к паре $(x, z)$ и $(y, z)$ с
$z = \lambda x + (1 - \lambda) y$ и сложим с весами $\lambda$ и
$1 - \lambda$:
$$
\lambda f(x) + (1-\lambda) f(y) \ge f(z) + \bigl(\lambda \nabla f(z) + (1-\lambda)\nabla f(z)\bigr)^{\!\top}\bigl(\lambda(x - z) + (1-\lambda)(y - z)\bigr) = f(z),
$$
поскольку $\lambda(x - z) + (1-\lambda)(y - z) = 0$.

Критерий минимума: если $\nabla f(x^*) = 0$, то \eqref{eq:foc} с
$x = x^*$ даёт $f(y) \ge f(x^*)$ для всех $y$. Обратно, если $x^*$ —
минимум, необходимое условие теоремы \ref{thm:soc} (а) даёт
$\nabla f(x^*) = 0$.
\end{proof}

\begin{theorem}[существование и единственность минимума]\label{thm:exist-unique}
Пусть $f$ непрерывно дифференцируема, $\mu$-сильно выпукла и ограничена
снизу. Тогда $f$ имеет единственную точку глобального минимума $x^*$, причём
$\norm{x - x^*} \le \frac{2}{\mu} \norm{\nabla f(x)}$ для всех $x$ и
лебегово множество $\{x : f(x) \le f(x^0)\}$ ограничено при любом $x^0$.
\end{theorem}

\begin{proof}
Из \eqref{eq:strong} с $y = x$, $x = x^*$ (как только $x^*$ появится) и
необходимого условия $\nabla f(x^*) = 0$ (теорема \ref{thm:foc} ниже)
следовала бы оценка; сначала покажем существование. Минимизирующая
последовательность $f(x_k) \to \inf f$ (а $\inf f > -\infty$ по условию)
содержится в лебеговом множестве $X_\beta = \{f \le f(x_0)\}$, $\beta =
f(x_0)$. Из \eqref{eq:strong} для выпуклой дифференцируемой $f$ (лемма о
градиентном неравенстве, теорема \ref{thm:foc}) имеем
$$
f(x) \ge f(x^0) + \scal{\nabla f(x^0)}{x - x^0} + \frac{\mu}{2}\norm{x - x^0}^2
\ge f(x^0) - \tfrac{1}{2\mu} \norm{\nabla f(x^0)}^2 +
\tfrac{\mu}{4}\norm{x - x^0}^2,
$$
где второе неравенство — минимизация квадратичной формы по $x$; значит
$X_\beta$ ограничено. Замкнуто оно по непрерывности $f$; по теореме
Вейерштрасса непрерывная $f$ достигает на $X_\beta$ минимума в точке $x^*$,
которая и есть глобальный минимум. Единственность: для двух минимумов $x^*$,
$x^{**}$ из \eqref{eq:strong}
$$
f(x^{**}) \ge f(x^*) + \frac{\mu}{2}\norm{x^{**} - x^*}^2 > f(x^*)
$$
при $x^{**} \ne x^*$ — противоречие. Оценка
$\norm{x - x^*} \le \frac{2}{\mu}\norm{\nabla f(x)}$: из \eqref{eq:strong}
с минимумом по $y$ слева ($f^*$) и $x$ — произвольной точкой имеем
$f^* \ge f(x) + \scal{\nabla f(x)}{x^* - x} + \frac{\mu}{2}\norm{x^* - x}^2$;
отсюда
$\frac{\mu}{2}\norm{x^* - x}^2 \le \norm{\nabla f(x)}\norm{x^* - x}$ по
Коши—Буняковского.
\end{proof}

# Градиентный спуск {#sec:gd}

Основной метод этого вопроса. Направление наискорейшего локального убывания
$f$ — антиградиент: для единичного $d$
$$
f(x + td) - f(x) = t \scal{\nabla f(x)}{d} + o(t),
$$
и минимум $\scal{\nabla f(x)}{d}$ по $d$, $\norm{d} = 1$, равен
$-\norm{\nabla f(x)}$ и достигается при $d = -\nabla f(x) / \norm{\nabla
f(x)}$. Метод градиентного спуска (англ. *gradient descent*) строит
последовательность
\begin{equation}\label{eq:gd}
x^{k+1} = x^k - \gamma_k \nabla f(x^k), \qquad k = 0, 1, \dots
\end{equation}
Непрерывный образец — градиентный поток $\dot x = -\nabla f(x)$, по которому
$\frac{d}{dt} f(x(t)) = -\norm{\nabla f(x(t))}^2 \le 0$ и функция монотонно
убывает вдоль траектории; метод \eqref{eq:gd} — дискретизация потока по
схеме Эйлера.

\begin{lemma}[спуск]\label{lem:descent}
Пусть $f$ $L$-гладка. Тогда для любого $x$ и любого $\gamma > 0$
\begin{equation}\label{eq:descent}
f\bigl(x - \gamma \nabla f(x)\bigr)
\le f(x) - \gamma \Bigl(1 - \frac{L \gamma}{2}\Bigr) \norm{\nabla f(x)}^2 .
\end{equation}
В частности, при $\gamma = 1/L$
\begin{equation}\label{eq:descent-1L}
f\bigl(x - \tfrac1L \nabla f(x)\bigr) \le f(x) - \frac{1}{2L} \norm{\nabla f(x)}^2 .
\end{equation}
\end{lemma}

\begin{proof}
Подставляем $y = x - \gamma \nabla f(x)$ в неравенство мажорирования
\eqref{eq:upper-par}:
$$
f(y) \le f(x) - \gamma \norm{\nabla f(x)}^2 + \frac{L \gamma^2}{2} \norm{\nabla f(x)}^2.
\qedhere
$$
\end{proof}

Заметим: коэффициент у $\norm{\nabla f(x)}^2$ в \eqref{eq:descent}
максимален при $\gamma = 1/L$; при $\gamma \ge 2/L$ он неположителен и
гарантии убывания $f$ нет (метод может расходиться — см. замечание к
теореме \ref{thm:gd-conv}).

\begin{theorem}[сходимость градиентного спуска]\label{thm:gd-conv}
Пусть $f$ $L$-гладка, $x^{k+1} = x^k - \frac1L \nabla f(x^k)$,
$R = \norm{x^0 - x^*}$, где $x^*$ — точка минимума.

\textbf{(а) Невыпуклый случай.} $f$ ограничена снизу: $f \ge f_* > -\infty$.
Тогда
\begin{equation}\label{eq:gd-nonconvex}
\min_{0 \le k \le N-1} \norm{\nabla f(x^k)} \le \sqrt{\frac{2L\bigl(f(x^0) - f_*\bigr)}{N}} .
\end{equation}

\textbf{(б) Выпуклый случай.} $f$ выпукла. Тогда для усреднённой точки
$\bar x^N = \frac1N \sum_{k=0}^{N-1} x^k$
\begin{equation}\label{eq:gd-convex}
f\bigl(\bar x^N\bigr) - f^* \le \frac{L R^2}{2N} .
\end{equation}

\textbf{(в) Сильно выпуклый случай.} $f$ $\mu$-сильно выпукла. Тогда
\begin{equation}\label{eq:gd-strong}
f(x^N) - f^* \le \frac{L R^2}{2} \min\Bigl\{ \frac{1}{N},\,
\Bigl(1 - \frac{\mu}{L}\Bigr)^{N} \Bigr\},
\qquad
\norm{x^N - x^*} \le R \Bigl(1 - \frac{\mu}{L}\Bigr)^{N/2} .
\end{equation}
\end{theorem}

\begin{proof}
По лемме \ref{lem:descent} последовательность $f(x^k)$ не возрастает:
\begin{equation}\label{eq:telescope-key}
f(x^{k+1}) \le f(x^k) - \frac{1}{2L} \norm{\nabla f(x^k)}^2 .
\end{equation}

\textbf{(а)} Суммируем \eqref{eq:telescope-key} по $k = 0, \dots, N-1$:
$$
\frac{1}{2L} \sum_{k=0}^{N-1} \norm{\nabla f(x^k)}^2 \le f(x^0) - f(x^N) \le f(x^0) - f_* .
$$
Левая часть не меньше $\frac{N}{2L} \min_k \norm{\nabla f(x^k)}^2$.

\textbf{(б)} Выпуклость дает $f(x) - f^* \le \scal{\nabla f(x)}{x - x^*}$
(теорема \ref{thm:foc}), поэтому из \eqref{eq:telescope-key}
$$
f(x^k) - f^* \le \scal{\nabla f(x^k)}{x^k - x^*}
\le L \scal{x^k - x^{k+1}}{x^k - x^*} .
$$
Используем тождество $2\scal{u}{v} = \norm{u}^2 + \norm{v}^2 - \norm{u-v}^2$
с $u = x^k - x^{k+1}$, $v = x^k - x^*$:
$$
f(x^k) - f^* \le \frac{L}{2}\Bigl( \norm{x^k - x^{k+1}}^2 + \norm{x^k - x^*}^2 - \norm{x^{k+1} - x^*}^2 \Bigr) .
$$
Суммируем по $k$; слагаемые $\norm{x^k - x^*}^2$ телескопируются:
\begin{equation}\label{eq:gd-convex-sum}
\sum_{k=0}^{N-1} \bigl(f(x^k) - f^*\bigr)
\le \frac{L}{2} \Bigl( \sum_{k=0}^{N-1} \norm{x^k - x^{k+1}}^2 + R^2 \Bigr) .
\end{equation}
С другой стороны, \eqref{eq:telescope-key} дает
$f(x^{k+1}) \ge f(x^k) - \frac{1}{2L}\norm{\nabla f(x^k)}^2$; но нам нужна
оценка суммы квадратов шагов: из \eqref{eq:telescope-key} после суммирования
$$
\sum_{k=0}^{N-1} \norm{x^k - x^{k+1}}^2 = \frac{1}{L^2} \sum_{k=0}^{N-1} \norm{\nabla f(x^k)}^2 \le \frac{2}{L} \bigl(f(x^0) - f^*\bigr) \le \frac{2}{L} \cdot \frac{L R^2}{2} = R^2,
$$
где $f(x^0) - f^* \le \frac{L R^2}{2}$ — это \eqref{eq:upper-par} с
$y = x^*$, $x = x^0$. Подставляем в \eqref{eq:gd-convex-sum}:
$$
\sum_{k=0}^{N-1} \bigl(f(x^k) - f^*\bigr) \le \frac{L R^2}{2} + \frac{L R^2}{2} = L R^2 .
$$
По выпуклости $f(\bar x^N) \le \frac1N \sum_k f(x^k)$, и деление на $N$
даёт \eqref{eq:gd-convex}.

\textbf{(в)} Сначала докажем оценку по аргументу. Сильная выпуклость
в точке минимума ($\nabla f(x^*) = 0$) дает
(предложение \ref{prop:equiv}, г):
$$
\scal{\nabla f(x)}{x - x^*} \ge \mu \norm{x - x^*}^2 .
$$
Тогда из \eqref{eq:telescope-key} и выпуклости
($f(x) - f^* \le \scal{\nabla f(x)}{x - x^*}$):
$$
\norm{x^{k+1} - x^*}^2 = \norm{x^k - x^*}^2 - \frac{2}{L} \scal{\nabla f(x^k)}{x^k - x^*} + \frac{1}{L^2}\norm{\nabla f(x^k)}^2 .
$$
Оцениваем среднее слагаемое снизу через $\mu \norm{x^k - x^*}^2$, а
последнее — сверху через \eqref{eq:telescope-key}, откуда
$\norm{\nabla f(x^k)}^2 \le 2L (f(x^k) - f(x^{k+1})) \le 2L (f(x^k) - f^*)$:
$$
\norm{x^{k+1} - x^*}^2 \le \Bigl(1 - \frac{2\mu}{L}\Bigr) \norm{x^k - x^*}^2 + \frac{2}{L}\bigl(f(x^k) - f^*\bigr).
$$
Другой путь: из сильной выпуклости $f(x^k) - f^* \ge \frac{\mu}{2}\norm{x^k - x^*}^2$, и тогда
$$
\norm{x^{k+1} - x^*}^2 \le \Bigl(1 - \frac{2\mu}{L} + \frac{\mu}{L}\Bigr)\norm{x^k - x^*}^2 = \Bigl(1 - \frac{\mu}{L}\Bigr) \norm{x^k - x^*}^2 .
$$
Итерация логарифма даёт $\norm{x^N - x^*}^2 \le (1 - \mu/L)^N R^2$.
Далее, лемма \ref{lem:descent} в точке $x^*$ с $\gamma = 1/L$ в роли
«шага по градиенту от $x^*$ к $x^k$» не работает напрямую; вместо этого
применим \eqref{eq:upper-par} с $x = x^*$, $y = x^k$:
$f(x^k) - f^* \le \frac{L}{2}\norm{x^k - x^*}^2 \le \frac{L R^2}{2} (1 - \mu/L)^N$.
Оценка $\frac{LR^2}{2N}$ — это \eqref{eq:gd-convex}, усиленная монотонностью
$f(x^N) \le f(\bar x^N)$ не верна для произвольной точки, но
$f(x^N) - f^* \le \frac1N \sum_k (f(x^k) - f^*) \le \frac{L R^2}{2N}$ следует
из монотонности $f(x^N) \le f(x^k)$ и уже доказанного.
\end{proof}

\begin{remark}[существенность посылок]\label{rem:gd-assumptions}
Каждая посылка теоремы \ref{thm:gd-conv} существенна, и контрпримеры
явные [13]. \textbf{(1)} Без липшицевости градиента метод может расходиться:
$f(x) = \abs{x}^{2+\varepsilon}$ при $\gamma = 1$ уходит на бесконечность
для $x^0$ вне компакта. \textbf{(2)} Без ограниченности снизу градиент не
обязан стремиться к нулю: для линейной $f$ при любом шаге
$\norm{\nabla f(x^k)} = \norm{\nabla f(x^0)} > 0$. \textbf{(3)} Шаг
$\gamma \ge 2/L$ разрушает сходимость даже на квадратичной
$f(x) = \tfrac12 \norm{x}^2$: при $\gamma = 2/L$ последовательность
периодически колеблется, при $\gamma > 2/L$ расходится.
\textbf{(4)} В невыпуклом случае сходимость по градиенту — всё, что можно:
$f(x) = 1/(1 + \norm{x}^2)$ дает $\nabla f(x^k) \to 0$ без сходимости $x^k$;
без сильной выпуклости скорость может быть сколь угодно малой: для
$f(x) = 1/x$ на $[1, \infty)$ при $\gamma = 1$ имеем
$\abs{f'(x^k)} = O(k^{-2/3})$ [13].
\end{remark}

Все три режима теоремы \ref{thm:gd-conv} точны по порядку: в разделе
\ref{sec:lower} построены функции, на которых ни один метод,
пользующийся только значениями и градиентами $f$, не сходится быстрее.

# Точная линейная оценка и оптимальный шаг {#sec:exact-rate}

Для сильно выпуклых гладких задач оценку \eqref{eq:gd-strong} можно
уточнить: сходимость по аргументу идёт с точным знаменателем прогрессии,
зависящим от $\varkappa = L/\mu$.

\begin{theorem}[точная оценка для сильно выпуклых гладких функций]\label{thm:exact-rate}
Пусть $f$ дважды дифференцируема и
\begin{equation}\label{eq:two-sided}
\mu I \preceq \nabla^2 f(x) \preceq L I \qquad \forall x .
\end{equation}
Тогда градиентный спуск \eqref{eq:gd} с постоянным шагом $\gamma$ сходится
линейно: $\norm{x^k - x^*} \le \norm{x^0 - x^*} q^k$ с
\begin{equation}\label{eq:q-gamma}
q(\gamma) = \max\{\, \abs{1 - \gamma \mu},\ \abs{1 - \gamma L} \,\}.
\end{equation}
Минимум знаменателя достигается при
\begin{equation}\label{eq:opt-step}
\gamma^* = \frac{2}{L + \mu},
\qquad
q^* = q(\gamma^*) = \frac{L - \mu}{L + \mu} = \frac{\varkappa - 1}{\varkappa + 1}.
\end{equation}
Оценка неулучшаема: на квадратичной функции с матрицей, спектр которой
лежит в $[\mu, L]$, знаменатель прогрессии равен $q(\gamma)$ точно.
\end{theorem}

\begin{proof}
По формуле Тейлора с остатком в интегральной форме
$$
\nabla f(x) = \int_0^1 \nabla^2 f(x^* + t(x - x^*))\, (x - x^*)\, dt
=: B(x)\,(x - x^*),
$$
где $B(x) = \int_0^1 \nabla^2 f(\cdot)\, dt$ симметрична и по
\eqref{eq:two-sided} удовлетворяет $\mu I \preceq B(x) \preceq L I$.
Тогда
$$
x^{k+1} - x^* = \bigl(I - \gamma B(x^k)\bigr) (x^k - x^*),
$$
и, так как собственные значения $I - \gamma B$ по модулю не превосходят
$\max\{\abs{1 - \gamma \mu}, \abs{1 - \gamma L}\}$, получаем
\eqref{eq:q-gamma}. Минимаксная задача $\min_\gamma \max\{\abs{1 - \gamma
\mu}, \abs{1 - \gamma L}\}$ решается уравнением $1 - \gamma \mu = \gamma L
- 1$, откуда \eqref{eq:opt-step}.

Неулучшаемость: возьмём $f(x) = \tfrac12 x^{\mathsf T} A x$ с
$A = \diag(L, \mu)$, $x^0 = (1, 1)^{\mathsf T}$. Тогда
$x^k = \bigl((1 - \gamma L)^k, (1 - \gamma \mu)^k\bigr)^{\mathsf T}$, и
покомпонентные сходимости точно $\abs{1 - \gamma L}$ и
$\abs{1 - \gamma \mu}$; евклидова норма даёт $q(\gamma)$ асимптотически
(при $k \to \infty$ медленная компонента доминирует).
\end{proof}

\begin{remark}
Два вывода из формулы $q^* = (\varkappa-1)/(\varkappa+1)$.
\textbf{(1) Число итераций.} Для точности $\varepsilon$ по аргументу
достаточно
$N \approx \frac{\varkappa}{2} \ln \frac{R}{\varepsilon}$ итераций:
$q^* = 1 - \frac{2}{\varkappa + 1} \le e^{-2/(\varkappa+1)}$, и
$(q^*)^N \le \varepsilon$ при $N \ge \frac{\varkappa + 1}{2} \ln
\frac{1}{\varepsilon}$. При $\varkappa = 10^3$ это сотни итераций — против
единиц у методов раздела \ref{sec:cheb}.
\textbf{(2) Овражная структура.} При $\varkappa \gg 1$ линии уровня $f$
вытянуты вдоль собственных направлений $B$ с малыми собственными
значениями; траектория спуска зигзагом пересекает овраг, и каждый шаг
продвигает по длине лишь на $O(1/\varkappa)$ долю расстояния. Пример 1
\texttt{examples.ipynb} измеряет фактический знаменатель на задаче с
$\varkappa = 10^3$: он совпадает с $q^*$ из \eqref{eq:opt-step}.
\end{remark}

# Нижние оценки для методов первого порядка {#sec:lower}

Теорема \ref{thm:gd-conv} давала оценки сверху. Покажем, что по порядку они
точны: ни один метод, использующий только оракулы значения и градиента,
не может сходиться быстрее. Формулировки и функции-«подсадки» —
из [37], упр. 1.3; доказательство — классическая схема Немировского—Юдина
в воспроизведении для этих конкретных семейств.

\begin{theorem}[нижние оценки]\label{thm:lower}
\textbf{(а)} Существует $L$-гладкая выпуклая $f$ с минимумом $f^*$ и
точкой минимума $x^*$, $\norm{x^0 - x^*} \le R$, такая что для любого
метода, генерирующего точки $x^k$ по правилу
$x^k \in x^0 + \operatorname{span}\{\nabla f(x^0), \dots, \nabla f(x^{k-1})\}$,
\begin{equation}\label{eq:lower-convex}
f(x^N) - f^* \ge \frac{3 L R^2}{32 (N+1)^2}, \qquad N < n .
\end{equation}

\textbf{(б)} Для любых $L \ge \mu > 0$ существует $L$-гладкая
$\mu$-сильно выпуклая $f$ с $\norm{x^0 - x^*} \le R$ такая, что
\begin{equation}\label{eq:lower-strong}
f(x^N) - f^* \ge \frac{\mu R^2}{2} \Bigl( \frac{\sqrt\varkappa - 1}{\sqrt\varkappa + 1} \Bigr)^{2N} .
\end{equation}
\end{theorem}

\begin{proof}
\textbf{(а)} Рассмотрим семейство квадратичных форм на $\R^{2N+2}$, $m =
2N+2$:
$$
f_m(x) = \frac{L}{4} \Bigl( \tfrac12 x_1^2 + \tfrac12 x_m^2 + \sum_{i=1}^{m-1} (x_i - x_{i+1})^2 - x_1 \Bigr),
$$
— «трёхдиагональная» форма (та же матрица, что в сквозном примере
раздела \ref{sec:cheb} с константами по $L$). Матрица $A_m$ этой формы
имеет собственные значения
$$
\lambda_i = L \Bigl(1 - \cos \frac{\pi i}{m + 1}\Bigr), \qquad i = 1, \dots, m,
$$
с относящимися к ним собственными векторами
$v_i \propto \bigl(\sin\frac{\pi i}{m+1}, \dots, \sin\frac{\pi i m}{m+1}\bigr)$.
При $x^0 = 0$ градиент $\nabla f_m(x) = A_m x - \frac{L}{4} e_1$ имеет
поддержку в первых $k+1$ координатах, если $x$ поддержан в первых $k$;
индукцией по $k$ точка $x^k$ любого метода из условия поддержана в
первых $k+1$ координатах, и траектория не зависит от $m$, пока
$m \ge 2N+2$. Взяв $m = 2N+2$ и оценивая минимум по
последней координате (которая равна нулю на траектории длины $N+1$), после
вычисления $f^*$ прямым минимизацией по формуле для обратной трёхдиагональной
матрицы получаем \eqref{eq:lower-convex}; вычисление громоздко, но
элементарно и воспроизведено в [37], упр. 1.3.

\textbf{(б)} Добавим сильную выпуклость: $f(x) = f_m(x) + \frac{\mu}{2}\norm{x}^2$
с тем же $A_m$ и $m = 2N+2$. Теперь матрица $\mu I + A_m$ имеет собственные
значения $\mu + \lambda_i \in [\mu, \mu + 2L]$, и аналогичная поддержка
даёт оценку по последней координате, сводящуюся к
\eqref{eq:lower-strong} после подстановки собственных значений и
нормировки $R$.
\end{proof}

\begin{remark}
Оценка \eqref{eq:lower-strong} сравнима с верхней \eqref{eq:gd-strong}: по
порядку в показателе экспоненты обе дают $O\bigl(\varkappa \ln(1/\varepsilon)\bigr)$ итераций градиентного спуска. Разрыв между верхней
оценкой $(1 - 1/\varkappa)^N$ и нижней $(1 - 2/\sqrt\varkappa)^{2N}
\approx (1 - 1/\sqrt\varkappa)^{2N}$ закрывается ускоренными методами
разделов \ref{sec:cheb} и \ref{sec:nesterov}: они сходятся со знаменателем
$1 - O(1/\sqrt\varkappa)$ и совпадают с нижней оценкой по порядку.
\end{remark}

# Чебышёвское ускорение {#sec:cheb}

Возвращаемся к квадратичной задаче — сквозному примеру. Здесь теорема
\ref{thm:exact-rate} точна, и вопрос «можно ли обойти знаменатель
$(\varkappa-1)/(\varkappa+1)$, не прибегая к глобальным методам второго
порядка» имеет конструктивный ответ: можно, за $O(\sqrt\varkappa)$ итераций
вместо $O(\varkappa)$, если менять шаг от итерации к итерации по правилу,
заданному многочленами Чебышёва. Аппарат минимаксных полиномов с
нормировкой $p(0) = 1$ построен в вопросе 11 (многочлены Чебышёва I рода);
здесь он используется и развивается.

Постановка: $\varphi(x) = \tfrac12 x^{\mathsf T} A x - b^{\mathsf T} x$,
$A$ симметрична положительно определена, $\operatorname{sp} A \subseteq
[\mu_{\min}, \mu_{\max}]$, $\varkappa = \mu_{\max} / \mu_{\min}$. Градиентный
метод с переменным шагом
\begin{equation}\label{eq:gd-varying}
x^{k+1} = x^k - \gamma_k \nabla \varphi(x^k), \qquad
\nabla \varphi(x) = A x - b,
\end{equation}
порождает ошибку $e^k = x^k - x^*$, удовлетворяющую
$e^{k+1} = (I - \gamma_k A) e^k$. После $k$ шагов
$e^k = p_k(A) e^0$ с $p_k(\lambda) = \prod_{j=0}^{k-1}(1 - \gamma_j \lambda)$,
$p_k(0) = 1$; выбор шагов, минимизирующий $\norm{e^k}$, — это задача о
наименее уклоняющемся от нуля многочлене на спектре (разобрана в вопросе 11):
решение $p_k^*(\lambda) = T_k(z(\lambda)) / T_k(z(0))$, где
$$
z(\lambda) = \frac{\mu_{\max} + \mu_{\min} - 2\lambda}{\mu_{\max} - \mu_{\min}},
$$
а шаги $\gamma_j = 1/\lambda_j$ — обратные величины к корням $p_k^*$
(чебышёвский набор параметров, вопрос 11). Оценка:
\begin{equation}\label{eq:cheb-rate}
\norm{e^k} \le q_k \norm{e^0},
\qquad
q_k = \frac{2 \sigma^k}{1 + \sigma^{2k}},
\qquad
\sigma = \frac{\sqrt\varkappa - 1}{\sqrt\varkappa + 1},
\end{equation}
и $q_k \le 2 \sigma^k$: линейная сходимость со знаменателем
$1 - 2/\sqrt\varkappa$.

Прямая реализация — перемножение сомножителей $(I - \gamma_j A)$ — на
компьютере не работает: при $\varkappa \gtrsim 10^2$ погрешность округления
съедает результат раньше, чем $q_k$ достигает $10^{-6}$ (измерено в вопросе
11, пример 4; теоретическая оговорка — [09], §6 п. 2: промежуточные
операторы $I - \gamma_j A$ имеют норму, бóльшую единицы, и произведение
переполняется при неудачном порядке сомножителей). Рабочая форма —
трёхчленная рекуррентность по $T_k$, метод чебышёвских полуитераций
(англ. *Chebyshev semi-iterative method*) [02]:

\begin{equation}\label{eq:cheb-rec}
\begin{aligned}
y^{(k+1)} &= \omega_{k+1}\Bigl( y^{(k)} - y^{(k-1)} + \gamma\, z^{(k)} \Bigr) + y^{(k-1)}, \\
z^{(k)} &= b - A y^{(k)} = -\nabla \varphi(y^{(k)}), \\
\gamma &= \frac{2}{\mu_{\max} + \mu_{\min}}, \qquad
\mu = \frac{\mu_{\max} + \mu_{\min}}{\mu_{\max} - \mu_{\min}}, \\
\omega_{k+1} &= 2\mu\, \frac{T_k(\mu)}{T_{k+1}(\mu)}, \qquad
y^{(1)} = y^{(0)} + \gamma z^{(0)} .
\end{aligned}
\end{equation}

\begin{theorem}[сходимость чебышёвских полуитераций]\label{thm:cheb}
Итерации \eqref{eq:cheb-rec} удовлетворяют
$$
y^{(k)} - x^* = p_k(A)\, \bigl(y^{(0)} - x^*\bigr),
\qquad
p_k(\lambda) = \frac{T_k(z(\lambda))}{T_k(z(0))},
$$
и потому $\norm{y^{(k)} - x^*} \le q_k \norm{y^{(0)} - x^*}$ с $q_k$ из
\eqref{eq:cheb-rate}. Для точности $\varepsilon$ по аргументу достаточно
\begin{equation}\label{eq:cheb-iters}
k \ge \frac{\ln (2/\varepsilon)}{2\sqrt\xi},
\qquad
\xi = \frac{\mu_{\min}}{\mu_{\max}} = \frac{1}{\varkappa},
\end{equation}
итераций: сравните с $\frac{\varkappa}{2}\ln\frac{1}{\varepsilon}$ у
стационарного шага (теорема \ref{thm:exact-rate}).
\end{theorem}

\begin{proof}
Многочлены ошибок рекуррентности. Обозначим
$e^{(k)} = y^{(k)} - x^*$; так как $z^{(k)} = -A e^{(k)}$ и
$e^{(1)} = (I - \gamma A) e^{(0)}$, первый шаг совпадает с
$p_1(\lambda) = 1 - \gamma \lambda$. Для $k \ge 1$ вычтем из
\eqref{eq:cheb-rec} тождество $x^* = \omega_{k+1}(x^* - x^* + 0) + x^*$
(так как $b - A x^* = 0$):
\begin{equation}\label{eq:cheb-err}
e^{(k+1)} = \omega_{k+1} \bigl( e^{(k)} - e^{(k-1)} - \gamma A e^{(k)} \bigr) + e^{(k-1)} .
\end{equation}
Пусть $p_k$ — многочлены, рекуррентно заданные
$$
p_{k+1}(\lambda) = \omega_{k+1} \bigl( p_k(\lambda) - p_{k-1}(\lambda) - \gamma \lambda\, p_k(\lambda) \bigr) + p_{k-1}(\lambda), \quad p_0 \equiv 1,\ p_1(\lambda) = 1 - \gamma \lambda .
$$
По индукции $e^{(k)} = p_k(A) e^{(0)}$: база $k = 0, 1$ уже проверена, а
шаг индукции — подстановка $e^{(j)} = p_j(A) e^{(0)}$ в
\eqref{eq:cheb-err}. Введём многочлены
$$
q_k(\lambda) = \frac{T_k(z(\lambda))}{T_k(\mu)}, \qquad
z(\lambda) = \frac{\mu_{\max} + \mu_{\min} - 2\lambda}{\mu_{\max} - \mu_{\min}}, \qquad
\mu = z(0) = \frac{\mu_{\max} + \mu_{\min}}{\mu_{\max} - \mu_{\min}} .
$$
По определению $\gamma$ и $\mu$ имеем
$\mu \gamma \lambda = \frac{\mu_{\max}+\mu_{\min}}{\mu_{\max}-\mu_{\min}}
\cdot \frac{2\lambda}{\mu_{\max}+\mu_{\min}} = \frac{2\lambda}{\mu_{\max}-\mu_{\min}}$,
так что
\begin{equation}\label{eq:z-mu}
z(\lambda) = \mu - \frac{2\lambda}{\mu_{\max}-\mu_{\min}} = \mu\,(1 - \gamma \lambda) .
\end{equation}
Проверим, что $q_k$ удовлетворяют той же рекуррентности, что и $p_k$, с теми
же начальными условиями; тогда $p_k \equiv q_k$. База:
$q_0 \equiv 1 = p_0$ и $q_1(\lambda) = z(\lambda)/\mu = 1 - \gamma\lambda = p_1(\lambda)$
по \eqref{eq:z-mu}. Шаг: подставим $q$ в правую часть рекуррентности,
группируя слагаемые как
$\omega_{k+1}(1 - \gamma\lambda) q_k(\lambda) + \bigl(1 - \omega_{k+1}\bigr) q_{k-1}(\lambda)$.
Первое слагаемое по \eqref{eq:z-mu} равно
$$
\frac{2\mu T_k(\mu)}{T_{k+1}(\mu)} \cdot \frac{z(\lambda)}{\mu} \cdot \frac{T_k(z(\lambda))}{T_k(\mu)} = \frac{2 z(\lambda)\, T_k(z(\lambda))}{T_{k+1}(\mu)} .
$$
Второе: из определения $\omega_{k+1}$ и трёхчленной рекуррентности
$T_{k+1}(\mu) = 2\mu T_k(\mu) - T_{k-1}(\mu)$ следует
$$
1 - \omega_{k+1} = \frac{T_{k+1}(\mu) - 2\mu T_k(\mu)}{T_{k+1}(\mu)} = -\frac{T_{k-1}(\mu)}{T_{k+1}(\mu)},
\qquad
\bigl(1 - \omega_{k+1}\bigr) q_{k-1}(\lambda) = -\frac{T_{k-1}(z(\lambda))}{T_{k+1}(\mu)} .
$$
Сумма по трёхчленной рекуррентности для $T$ равна
$\bigl(2 z(\lambda) T_k(z(\lambda)) - T_{k-1}(z(\lambda))\bigr)/T_{k+1}(\mu) = T_{k+1}(z(\lambda))/T_{k+1}(\mu) = q_{k+1}(\lambda)$.
Индукция завершена: $e^{(k)} = q_k(A)\, e^{(0)}$.

Оценка \eqref{eq:cheb-rate} — теорема о наименее уклоняющихся многочленах
из вопроса 11 (доказательство там). Оценка итераций: из $q_k \le
2\sigma^k$ условие $q_k \le \varepsilon$ обеспечивается
$\sigma^k \le \varepsilon/2$; так как
$-\ln \sigma = \ln\frac{\sqrt\varkappa+1}{\sqrt\varkappa-1} =
\ln\bigl(1 + \frac{2}{\sqrt\varkappa - 1}\bigr) \ge \frac{2}{\sqrt\varkappa}$,
достаточно $k \ge \sqrt\varkappa \ln(2/\varepsilon) / 2$; в обозначениях
[09] $\xi = 1/\varkappa$ это формула \eqref{eq:cheb-iters}.
\end{proof}

\begin{remark}[почему рекуррентная форма устойчива]\label{rem:cheb-stable}
Множительная форма $e^k = \prod (I - \gamma_j A) e^0$ и рекуррентная
\eqref{eq:cheb-rec} алгебраически эквивалентны, но численно —
нет. Коэффициенты $\omega_{k+1} = 2\mu T_k(\mu)/T_{k+1}(\mu)$ при
$\mu = (\varkappa+1)/(\varkappa-1) > 1$ убывают как
$2\mu \cdot T_k(\mu)/T_{k+1}(\mu) \approx 2\mu / (2\mu) = 1$
(так как $T_{k+1}(\mu)/T_k(\mu) \to \mu + \sqrt{\mu^2 - 1} > \mu$), то
же остаются ограниченными; все промежуточные векторы рекуррентности —
выпуклые комбинации с ограниченными коэффициентами, и ошибки округления
не накапливаются множительно (измерение — пример 2 \texttt{examples.ipynb}:
рекуррентная форма достигает $10^{-6}$ при $\varkappa = 10^2$ там, где
множительная останавливается на $8 \cdot 10^{-6}$ по погрешности
округления; в вопросе 11 показано, что при $\varkappa \gtrsim 10^2$
множительная форма вообще не достигает $10^{-6}$).
\end{remark}

\begin{remark}[что нужно знать для чебышёвского ускорения]
Границы спектра $\mu_{\min}, \mu_{\max}$. Оценка спектра — отдельная
вычислительная задача; квадратичный метод сопряжённых градиентов даёт тот
же закон $\sqrt\varkappa$ без всякого знания спектра (раздел \ref{sec:cg}),
и в этом его практическое преимущество [02].
\end{remark}

# Метод сопряжённых градиентов {#sec:cg}

Метод сопряжённых градиентов (англ. *conjugate gradients*, CG) решает ту же
квадратичную задачу со знаменателем $1 - 2/\sqrt\varkappa$, но адаптивно:
каждое направление строится из текущего градиента и предыдущего
направления, без оценок спектра.

\begin{definition}\label{def:conj}
Направления $h^0, \dots, h^{m}$ \textbf{$A$-сопряжены}, если
$\scal{A h^i}{h^j} = 0$ при $i \ne j$.
\end{definition}

Метод строит последовательность
\begin{equation}\label{eq:cg}
\begin{aligned}
x^{k+1} &= x^k + \gamma_k h^k, \qquad
\gamma_k = \argmin_\gamma \varphi(x^k + \gamma h^k), \\
h^{k+1} &= -\nabla \varphi(x^{k+1}) + \beta_k h^k, \qquad
\beta_k = \frac{\norm{\nabla \varphi(x^{k+1})}^2}{\norm{\nabla \varphi(x^k)}^2},
\end{aligned}
\end{equation}
с начальными условиями $h^0 = -\nabla \varphi(x^0)$; шаг $\gamma_k$ —
точный одномерный минимум, выписываемый явно:
$\gamma_k = -\scal{\nabla \varphi(x^k)}{h^k} / \scal{A h^k}{h^k}$.

\begin{lemma}[сопряжённость и ортогональность]\label{lem:cg-conj}
Для итераций \eqref{eq:cg} при $k < n$:

\textbf{(а)} $\scal{\nabla \varphi(x^{k+1})}{\nabla \varphi(x^j)} = 0$
для всех $j \le k$;

\textbf{(б)} направления $h^0, \dots, h^{k+1}$ $A$-сопряжены;

\textbf{(в)} $\operatorname{span}\{h^0, \dots, h^k\} = \operatorname{span}\{\nabla\varphi(x^0), \dots, \nabla\varphi(x^k)\}$.
\end{lemma}

\begin{proof}
Индукция по $k$. База: $h^0 = -\nabla\varphi(x^0)$, (а) и (б) пусты или
очевидны, (в) — тривиально.

Шаг: пусть утверждения верны до номера $k$ включительно. Одномерный
минимум по $\gamma$ даёт необходимое условие
$\scal{\nabla \varphi(x^{k+1})}{h^k} = 0$. Для $j < k$ по индукции
$\nabla\varphi(x^{k+1}) = \nabla\varphi(x^j) + \sum_{i=j}^{k} \gamma_i A h^i$
(телескопическая сумма $\gamma_i A h^i = A(x^{i+1} - x^i)$); скалярно
умножая на $h^j$ и используя $\scal{\nabla\varphi(x^j)}{h^j} = 0$
(одномерный минимум на шаге $j$) и $A$-сопряжённость $h^i$ при $i > j$,
получаем $\scal{\nabla\varphi(x^{k+1})}{h^j} = 0$. По (в) индукции
$\nabla\varphi(x^j) \in \operatorname{span}\{h^0,\dots,h^j\}$, поэтому из
$\scal{\nabla\varphi(x^{k+1})}{h^i} = 0$ для $i \le k$ следует (а) для
пары $(k+1, j)$, $j \le k$: вектор $\nabla\varphi(x^j)$ — комбинация
$h^0, \dots, h^j$.

Для (б): $\scal{A h^{k+1}}{h^j} = \scal{-\nabla\varphi(x^{k+1}) + \beta_k h^k}{A h^j}$. При $j = k$: $\scal{A h^k}{h^k} \ne 0$, а
$\scal{\nabla\varphi(x^{k+1})}{A h^k} = 0$, так как
$A h^k = (\nabla\varphi(x^{k+1}) - \nabla\varphi(x^k))/\gamma_k$ и оба
скалярных произведения с $\nabla\varphi(x^{k+1})$ равны нулю по (а);
при $j < k$: $\scal{h^k}{A h^j} = 0$ по индукции, а
$\scal{\nabla\varphi(x^{k+1})}{A h^j} = 0$, так как
$A h^j \in \operatorname{span}\{\nabla\varphi(x^j), \nabla\varphi(x^{j+1})\}$
по той же телескопической формуле и (а) применимо.

(в): $h^{k+1} \in \operatorname{span}\{\nabla\varphi(x^{k+1}), h^k\}
\subset \operatorname{span}\{\nabla\varphi(x^0), \dots,
\nabla\varphi(x^{k+1})\}$; обратное включение — из невырожденности
треугольного перехода.
\end{proof}

\begin{theorem}[конечность сопряжённых градиентов]\label{thm:cg-finite}
Итерации \eqref{eq:cg} для квадратичной функции с положительно
определённой $A$ находят точный минимум за не более чем $n$ шагов: если
$\nabla\varphi(x^k) \ne 0$ при $k < n$, то $x^n = x^*$; в противном
случае процесс останавливается раньше. Кроме того, $x^k$ минимизирует
$\varphi$ на крыловском подпространстве
$$
x^0 + \mathcal{K}_k, \qquad
\mathcal{K}_k = \operatorname{span}\{\nabla\varphi(x^0), A\nabla\varphi(x^0), \dots, A^{k-1}\nabla\varphi(x^0)\}.
$$
\end{theorem}

\begin{proof}
Лемма \ref{lem:cg-conj} (а) даёт ортогональность $n+1$ градиентов в
$n$-мерном пространстве; значит, $\nabla\varphi(x^n) = 0$, а так как
$\varphi$ сильно выпукла, $\nabla\varphi(x^n) = 0$ влечёт $x^n = x^*$.

Оптимальность на $\mathcal{K}_k$: по лемме \ref{lem:cg-conj} (в)
$x^k = x^0 + \sum_{i<k} \gamma_i h^i \in x^0 + \mathcal{K}_k$.
Для произвольного $y = x^0 + \sum_{i<k} c_i h^i$ разложим
$$
\varphi(y) = \varphi(x^k) + \scal{\nabla\varphi(x^k)}{y - x^k} + \tfrac12 \scal{A(y - x^k)}{y - x^k} .
$$
Так как $y - x^k$ — комбинация $h^0, \dots, h^{k-1}$, первое слагаемое
равно нулю (каждое $\scal{\nabla\varphi(x^k)}{h^i} = 0$ по лемме
\ref{lem:cg-conj}, (а) и $h^i \in \operatorname{span}$ прошлых
градиентов), второе неотрицательно. Значит, $\varphi(y) \ge
\varphi(x^k)$, равенство только при $y = x^k$.
\end{proof}

\begin{theorem}[оценка сходимости CG]\label{thm:cg-rate}
Для итераций \eqref{eq:cg}
\begin{equation}\label{eq:cg-rate}
\norm{x^k - x^*}_A \le 2\, \sqrt{\frac{L}{\mu}}\, q^k \norm{x^0 - x^*}_A,
\qquad
q = \frac{\sqrt\varkappa - 1}{\sqrt\varkappa + 1},
\end{equation}
где $\norm{u}_A = \sqrt{\scal{A u}{u}}$, $\mu = \mu_{\min}(A)$,
$L = \mu_{\max}(A)$.
\end{theorem}

\begin{proof}
По теореме \ref{thm:cg-finite} $x^k$ минимизирует $\varphi$ на
$x^0 + \mathcal{K}_k$; следовательно, $e^k = x^k - x^* = p_k(A) e^0$
для некоторого многочлена $p_k$ степени $\le k$ с $p_k(0) = 1$
(каждый вектор крыловского подпространства есть $q(A)\nabla\varphi(x^0)$
с $q(0)$ — любым; условие $p_k(0) = 1$ обеспечивает представление
ошибки именно через $e^0$). Поэтому
$$
\norm{e^k}_A^2 = \min_{\substack{p(0)=1 \\ \deg p \le k}} \scal{A p(A)^2 e^0}{e^0}
\le \Bigl( \max_{\lambda \in [\mu, L]} \abs{p(\lambda)} \Bigr)^2 \norm{e^0}_A^2
$$
для любого допустимого $p$; берём чебышёвский
$p(\lambda) = T_k(z(\lambda))/T_k(z(0))$ из раздела \ref{sec:cheb}, для
которого максимум равен $1/T_k(\mu)$, а
$T_k(\mu) \ge \tfrac12 \bigl(\mu + \sqrt{\mu^2 - 1}\bigr)^k \ge
\tfrac12 \bigl(\frac{\sqrt\varkappa + 1}{\sqrt\varkappa - 1}\bigr)^k$.
Подстановка даёт \eqref{eq:cg-rate}.
\end{proof}

\begin{remark}[оптимальность CG]\label{rem:cg-opt}
Оценка \eqref{eq:cg-rate} неулучшаема: любой метод, точки которого лежат
в $x^0 + \operatorname{span}\{\nabla\varphi(x^0), \dots,
\nabla\varphi(x^{k-1})\}$ (все методы первого порядка таковы),
допускает задачу с $\norm{x^k - x^*}_A \ge c\, q^k$ [13]. CG совпадает с
нижней оценкой теоремы \ref{thm:lower} по порядку и потому оптимален в
классе методов первого порядка — при том что ему не требуется знание
спектра, в отличие от чебышёвских полуитераций \eqref{eq:cheb-rec} [02].
\end{remark}

# Вычислительная сторона {#sec:compute}

Все числа этой таблицы — из теорем этого конспекта; их проверка расчётом
— в `examples.ipynb`.

**Что покупает каждый метод (квадратичная задача, $\varkappa = L/\mu$,
точность $\varepsilon$ по аргументу).**

\begin{center}
\small
\begin{tabular}{lllll}
\hline
Метод & Итерации & Итерация стоит & Требует & Раздел \\
\hline
Градиентный, $\gamma = 1/L$ & $O(\varkappa \ln \frac1\varepsilon)$ & $1$ градиент & ничего & \ref{sec:gd} \\
Градиентный, $\gamma^* = 2/(L+\mu)$ & $\frac\varkappa2 \ln\frac1\varepsilon$ & $1$ градиент & $\mu$, $L$ & \ref{sec:exact-rate} \\
Наискорейший спуск & не лучше стационарного & $1$ градиент + 1D-поиск & ничего & \ref{sec:problems} \\
Чебышёвские полуитерации & $\frac{\sqrt\varkappa}{2} \ln \frac2\varepsilon$ & $1$ умножение $A$ & $\mu$, $L$ & \ref{sec:cheb} \\
Сопряжённые градиенты & $O(\sqrt\varkappa \ln \frac1\varepsilon)$ & $1$ умножение $A$ & ничего & \ref{sec:cg} \\
Тяжёлый шарик & $O(\sqrt\varkappa \ln \frac1\varepsilon)$ локально & $1$ градиент & $\mu$, $L$ & \ref{sec:hb} \\
Нестерова (ускоренный) & $O(\sqrt\varkappa \ln \frac1\varepsilon)$ & $1$ градиент & $\mu$, $L$ & \ref{sec:nesterov} \\
Ньютон & $O(\ln \ln \frac1\varepsilon)$ локально & гессиан + решение системы & хорошее $x^0$ & \ref{sec:newton} \\
\hline
\end{tabular}
\end{center}

Три практических замечания.

\textbf{(1) Наискорейший спуск не ускоряет градиентный.} Точная
одномерная минимизация по $\gamma$ даёт на квадратичной задаче тот же
асимптотический знаменатель $(\varkappa-1)/(\varkappa+1)$, что и стационарный
шаг $\gamma^*$ (задача 2 раздела \ref{sec:problems}; теорема с
неравенством Канторовича — [13]). Выбор шага — не то место, где покупается
ускорение.

\textbf{(2) Оценка $L$ на практике.} Константу $L$ часто не знают;
стандартный приём — бэктрекинг: стартовать с грубой оценки и дробить
$\gamma \leftarrow \gamma/2$, пока не выполнится условие Армихо
$f(x - \gamma\nabla f(x)) \le f(x) - \frac{\gamma}{2}\norm{\nabla f(x)}^2$
(оно гарантирует лемму спуска). Все оценки этого конспекта сохраняются
с заменой $L$ на удвоенную найденную константу; автоматическая адаптация
к неизвестной гладкости — содержание универсального градиентного спуска
[37], §5.

\textbf{(3) Большая размерность.} Память $O(n)$ у всех методов таблицы,
кроме Ньютона ($O(n^2)$ на гессиан и его факторизацию); при $n$ в
миллионы это решающий аргумент за методы первого порядка и против
ньютоновских — см. вопрос 02 (ML и обратные задачи).

# Разобранные задачи {#sec:problems}

\begin{problem}\label{prob:opt-step}
Квадратичная функция $\varphi(x) = \tfrac12 x^{\mathsf T} A x - b^{\mathsf T} x$,
$\operatorname{sp} A = [\mu, L]$, $\varkappa = L/\mu = 10^3$. Найти
оптимальный постоянный шаг градиентного спуска и число итераций до
$\norm{x^N - x^*} \le 10^{-3} \norm{x^0 - x^*}$.
\end{problem}

*Решение.* По теореме \ref{thm:exact-rate} $\gamma^* = 2/(L + \mu)$,
$q^* = (\varkappa - 1)/(\varkappa + 1) = 999/1001$. Условие
$(q^*)^N \le 10^{-3}$: $N \ge \ln 10^3 / \ln(1001/999) \approx 6\,907/0{,}002 \approx 3\,454$ (точное число считается в примере 1 `examples.ipynb`). Чебышёвское ускорение (теорема \ref{thm:cheb}) требует
$N \ge \tfrac{\sqrt\varkappa}{2} \ln \tfrac{2}{\varepsilon}
\approx \tfrac{\sqrt{1000}}{2} \ln 2000 \approx 120$ итераций —
на порядок меньше; сопряжённые градиенты — того же порядка, что и
чебышёв, но без знания спектра.

\begin{problem}\label{prob:zigzag}
Наискорейший спуск на квадратичной функции с
$A = \diag(L, \mu)$, $x^0 = (L^{-1/2}, \mu^{-1/2})$. Показать, что
последовательные направления ортогональны и найти знаменатель прогрессии
по $f$.
\end{problem}

*Решение.* Одномерный минимум: $\gamma_k = \norm{\nabla\varphi(x^k)}^2 /
\scal{A \nabla\varphi(x^k)}{\nabla\varphi(x^k)}$. Ортогональность
последовательных направлений — необходимое условие одномерного минимума:
$\frac{d}{d\gamma} f(x^{k+1} + \gamma' h^k)\big|_{\gamma'=0} = 0$ при
$\gamma' = 0$ даёт $\scal{\nabla f(x^{k+1})}{h^k} = 0$, а
$h^k = -\nabla f(x^k)$. Двумерный случай даёт чередование направлений
с углом $\arctan\sqrt\varkappa$ к осям; по неравенству Канторовича [13]
$$
f(x^{k+1}) - f^* \le \Bigl( \frac{L - \mu}{L + \mu} \Bigr)^2 \bigl(f(x^k) - f^*\bigr),
$$
то есть тот же знаменатель, что у стационарного шага — точная одномерная
минимизация не ускоряет метод (пример 1 `examples.ipynb` измеряет это
на $\varkappa = 10^3$).

# Вопросы, которые стоит ожидать {#sec:questions}

- Почему градиентный метод сходится медленно на плохо обусловленной
  задаче, и что даёт чебышёвское ускорение? (теоремы
  \ref{thm:exact-rate}, \ref{thm:cheb}; ответ: знаменатель
  $1 - 2/\varkappa$ против $1 - 2/\sqrt\varkappa$, число итераций
  $O(\varkappa)$ против $O(\sqrt\varkappa)$.)
- Чем сопряжённые градиенты лучше чебышёвского метода и где их
  эквивалентность? (раздел \ref{sec:cg}: тот же закон, но без знания
  спектра и с конечной сходимостью за $n$ шагов.)
- Сходится ли градиентный спуск для произвольной гладкой функции? К
  чему? (теорема \ref{thm:gd-conv}, (а): только $\nabla f \to 0$;
  замечание \ref{rem:gd-assumptions}.)
- Когда метод Ньютона сходится квадратично и что даёт теорема
  Канторовича? (раздел \ref{sec:newton}: локально всегда; полулокально —
  условие $h \le 1/2$.)
- Как минимизировать на выпуклом множестве и чем проекция отличается от
  условного градиента? (раздел \ref{sec:constrained}: PGD против
  Франка—Вулфа; у FW линейный оракул вместо проекции.)
- Как считать градиент функционала качества, если он задан решением
  дифференциального уравнения? (раздел \ref{sec:adjoint}: через
  сопряжённую задачу.)

\part{Расширенная часть}

# Метод тяжёлого шарика {#sec:hb}

Двухшаговый метод Поляка 1963–64 гг. (англ. *heavy ball method*)
\begin{equation}\label{eq:hb}
x^{k+1} = x^k - \alpha \nabla f(x^k) + \beta\,(x^k - x^{k-1}), \qquad \alpha > 0,\ \beta \in [0,1),
\end{equation}
добавляет к градиентному шагу «инерцию» — слагаемое в направлении предыдущего
перемещения; физическая аналогия — шарик, скатывающийся по подвешенной в
вязкой среде чаше. В машинном обучении та же схема известна как метод с
импульсом (англ. *momentum*); это один мост между вопросом 10 и вопросом 02
(ML и обратные задачи).

\begin{theorem}[локальная сходимость тяжёлого шарика]\label{thm:hb}
Пусть $f$ дважды дифференцируема, $x^*$ — точка минимума с
\begin{equation}\label{eq:hb-curv}
\mu I \preceq \nabla^2 f(x^*) \preceq L I, \qquad \mu > 0,
\end{equation}
параметры удовлетворяют
\begin{equation}\label{eq:hb-cond}
0 \le \beta < 1, \qquad 0 < \alpha < \frac{2(1 + \beta)}{L},
\end{equation}
и начальное приближение достаточно близко к $x^*$. Тогда существуют
константа $c > 0$ и знаменатель $q \in (0,1)$ такие, что
$\norm{x^k - x^*} \le c\, (q + \delta)^k$ для любого наперёд заданного
$\delta > 0$; при
\begin{equation}\label{eq:hb-opt}
\alpha^* = \frac{4}{(\sqrt{L} + \sqrt{\mu})^2}, \qquad
\beta^* = \Bigl( \frac{\sqrt{L} - \sqrt{\mu}}{\sqrt{L} + \sqrt{\mu}} \Bigr)^{\!2}
\end{equation}
достигается оптимальный знаменатель
\begin{equation}\label{eq:hb-q}
q^* = \frac{\sqrt{L} - \sqrt{\mu}}{\sqrt{L} + \sqrt{\mu}} = \frac{\sqrt\varkappa - 1}{\sqrt\varkappa + 1} .
\end{equation}
\end{theorem}

\begin{proof}
Приём удвоения размерности: введём вектор
$z^k = (x^k - x^*,\ x^{k-1} - x^*)^{\mathsf T}$. По формуле Тейлора
$\nabla f(x) = B(x)(x - x^*)$ с $B(x) = \int_0^1 \nabla^2 f(x^* + t(x -
x^*))\, dt$ (см. доказательство теоремы \ref{thm:exact-rate}); тогда
\eqref{eq:hb} переписывается как одношаговый процесс
$$
z^{k+1} = \bigl( \mathcal{A} + \mathcal{E}_k \bigr) z^k,
\qquad
\mathcal{A} = \begin{pmatrix} (1 + \beta) I - \alpha B & -\beta I \\[2pt] I & 0 \end{pmatrix},
$$
где $B = \nabla^2 f(x^*)$, а $\mathcal{E}_k \to 0$ при $z^k \to 0$
(непрерывность $\nabla^2 f$). Лемма о спектральном радиусе: если все
собственные значения матрицы $\mathcal{A}$ по модулю меньше $q < 1$, то
существует эквивалентная норма $\norm{\cdot}_*$, в которой
$\norm{\mathcal{A}}_* \le q + \delta/2$, и при малых $\mathcal{E}_k$
$\norm{(\mathcal{A} + \mathcal{E}_k) z}_* \le (q + \delta)\norm{z}_*$,
откуда $\norm{z^k}_* \le (q + \delta)^k \norm{z^0}_*$; эквивалентность норм
в конечномерном пространстве переносит оценку на евклидову норму с
константой $c$.

Собственные значения $\mathcal{A}$ для собственного вектора $u$ матрицы $B$
с собственным значением $\lambda$ находятся из блочной структуры:
если $z = (u, \rho u)^{\mathsf T}$ — собственный вектор, то
$\rho$ и $\rho_{\text{св}}$ удовлетворяют
$\rho^2 - \rho(1 + \beta - \alpha\lambda) + \beta = 0$, то есть
\begin{equation}\label{eq:hb-rho}
\rho_{\pm}(\lambda) = \frac{1 + \beta - \alpha \lambda \pm \sqrt{(1 + \beta - \alpha \lambda)^2 - 4\beta}}{2} .
\end{equation}
Дискриминант $(1 + \beta - \alpha\lambda)^2 - 4\beta$ при условиях
\eqref{eq:hb-cond} отрицателен для $\lambda \in [\mu, L]$: функция
$g(\lambda) = (1 + \beta - \alpha\lambda)^2$ на $[\mu, L]$ принимает
максимум на концах, и $g(L) < 4\beta$ по условию $\alpha <
2(1+\beta)/L$, а $g(\mu) < 4\beta$ следует из $\alpha > 0$ и
$\beta > (1 - \alpha\mu)^2/4$... точнее: нам нужно $\max_\lambda
\abs{\rho_\pm(\lambda)} < 1$. Для комплексных корней ($g < 4\beta$)
$\abs{\rho_\pm}^2 = \beta < 1$. Для вещественных ($g \ge 4\beta$) оба
корня вещественны, положительны (сумма $1 + \beta - \alpha\lambda > 0$,
произведение $\beta > 0$), и меньший корень меньше единицы как
$(S - \sqrt{S^2 - 4\beta})/2 < 1$ при $S = 1 + \beta - \alpha\lambda$;
больший корень меньше единицы при $S < 1 + \beta$, то есть
$\alpha\lambda > 0$. Итак при \eqref{eq:hb-cond} все
$\abs{\rho_\pm(\lambda)} < 1$; знаменатель $q$ — это
$\max_{\lambda \in [\mu, L]} \max_\pm \abs{\rho_\pm(\lambda)}$.

Оптимизация. Для комплексного случая $\abs{\rho}^2 = \beta$; для
вещественного больший корень по модулю равен
$\abs{1 + \beta - \alpha\lambda}$ (когда корни чисто вещественны и
$S^2 \ge 4\beta$). Равновесие между двумя режимами достигается, когда
оба корня при $\lambda = \mu$ и $\lambda = L$ совпадают попарно
(двойные корни): $1 + \beta - \alpha\mu = 2\sqrt\beta$ и
$1 + \beta - \alpha L = -2\sqrt\beta$; вычитание даёт
$\alpha(L - \mu) = 4\sqrt\beta$; сложение: $2(1 + \beta) = \alpha(L +
\mu)$. Решая систему, получаем $\sqrt\beta = (\sqrt{L} - \sqrt\mu)/
(\sqrt{L} + \sqrt\mu)$ и $\alpha = 2(1+\beta)/(L+\mu) =
4/(\sqrt{L}+\sqrt\mu)^2$ — это \eqref{eq:hb-opt}; подстановка даёт
$\abs{\rho} = \sqrt\beta = q^*$ для всех $\lambda \in [\mu, L]$, то есть
\eqref{eq:hb-q}.
\end{proof}

\begin{remark}
Сравнение со знаменателем $q_1 = (\varkappa - 1)/(\varkappa + 1)
\approx 1 - 2/\varkappa$ градиентного спуска: при $\varkappa \gg 1$
$$
q_1 \approx 1 - \frac{2}{\varkappa}, \qquad
q^* \approx 1 - \frac{2}{\sqrt\varkappa},
$$
то есть тяжёлый шарик при оптимальных параметрах сокращает число итераций
в $O(\sqrt\varkappa)$ раз. Цена — знание $\mu$ и $L$ и локальность: метод
сходится лишь из окрестности минимума, а глобальная сходимость для
выпуклых функций у тяжёлого шарика не гарантирована (в отличие от метода
Нестерова, раздел \ref{sec:nesterov}). Измерение фактических знаменателей —
пример 3 \texttt{examples.ipynb}.
\end{remark}

# Ускоренный метод Нестерова {#sec:nesterov}

Метод Нестерова 1983 г. (у [37] — быстрый градиентный метод, БГМ) сходится
по нижним оценкам теоремы \ref{thm:lower} глобально, для всех гладких
выпуклых функций:
\begin{equation}\label{eq:nesterov}
\begin{aligned}
x^{k+1} &= y^k - \tfrac1L \nabla f(y^k), \\
y^{k+1} &= x^{k+1} + \frac{k}{k+3}\bigl( x^{k+1} - x^k \bigr),
\end{aligned}
\qquad x^0 = y^0 .
\end{equation}

\begin{theorem}[сходимость метода Нестерова]\label{thm:nesterov}
Пусть $f$ $L$-гладкая и выпукла, $R = \norm{x^0 - x^*}$. Тогда итерации
\eqref{eq:nesterov} удовлетворяют
\begin{equation}\label{eq:nesterov-rate}
f(x^N) - f^* \le \frac{2 L R^2}{(N+1)^2} .
\end{equation}
Если дополнительно $f$ $\mu$-сильно выпукла и во второй строке
\eqref{eq:nesterov} коэффициент $\frac{k}{k+3}$ заменён на
$\frac{\sqrt\varkappa - 1}{\sqrt\varkappa + 1}$ с
$\varkappa = L/\mu$, то
\begin{equation}\label{eq:nesterov-strong}
f(x^N) - f^* \le \frac{L R^2}{2} \Bigl( \frac{\sqrt\varkappa - 1}{\sqrt\varkappa + 1} \Bigr)^{\!N} .
\end{equation}
Обе оценки совпадают с нижними оценками теоремы \ref{thm:lower} по порядку.
\end{theorem}

\begin{proof}[Доказательство (схема подобных треугольников) [37]]
Введём $\alpha_k > 0$ (пока свободные), $A_k = \sum_{i=0}^{k} \alpha_i$,
$\tau_k = \alpha_{k+1}/A_{k+1} \in (0,1)$, и три последовательности:
$y^k$ — точка оценки градиента, $x^{k+1} = y^k - \frac1L\nabla f(y^k)$ —
градиентный шаг, и
$$
z^{k+1} = z^k - \alpha_{k+1} \nabla f(y^k), \qquad z^0 = x^0 .
$$
(метод подобных треугольников: треугольники $(z^k, z^{k+1}, x^{k+1})$ и
$(z^k, x^k, y^k)$ подобны с коэффициентом $\tau_k$, что даёт
$y^k = \tau_k z^k + (1 - \tau_k) x^k$.) Докажем индукцией
\begin{equation}\label{eq:est-seq}
A_k \bigl(f(x^k) - f^*\bigr) + \tfrac12 \norm{z^k - x^*}^2 \le \tfrac12 \norm{x^0 - x^*}^2 .
\end{equation}
База $k = 0$: $A_0 (f(x^0) - f^*) \ge 0$ и $z^0 = x^0$.

Шаг: пусть \eqref{eq:est-seq} верно для $k$. Оценим приращение правой
части. Так как $\nabla f(y^k) = L(y^k - x^{k+1})$,
$$
\tfrac12 \norm{z^k - x^*}^2 - \tfrac12 \norm{z^{k+1} - x^*}^2
= \alpha_{k+1} \scal{\nabla f(y^k)}{z^k - x^*} - \tfrac{\alpha_{k+1}^2}{2} \norm{\nabla f(y^k)}^2 .
$$
По выпуклости $f$ (теорема \ref{thm:foc}) с $y = x^*$, $x = y^k$:
$\scal{\nabla f(y^k)}{z^k - x^*} = \scal{\nabla f(y^k)}{z^k - y^k} +
\scal{\nabla f(y^k)}{y^k - x^*} \ge \scal{\nabla f(y^k)}{z^k - y^k} +
f(y^k) - f^*$. По подобию треугольников
$z^k - y^k = \frac{1 - \tau_k}{\tau_k}(y^k - x^k)$, и
$$
\scal{\nabla f(y^k)}{z^k - y^k} = \frac{1 - \tau_k}{\tau_k} \scal{\nabla f(y^k)}{y^k - x^k} \ge \frac{1 - \tau_k}{\tau_k}\bigl( f(y^k) - f(x^k) \bigr),
$$
где последнее неравенство — выпуклость в форме $f(x^k) \ge f(y^k) +
\scal{\nabla f(y^k)}{x^k - y^k}$. Далее, лемма спуска (лемма
\ref{lem:descent}): $f(x^{k+1}) \le f(y^k) - \frac{1}{2L}\norm{\nabla
f(y^k)}^2$, то есть $-\frac{\alpha_{k+1}^2}{2}\norm{\nabla f(y^k)}^2 \le
-\alpha_{k+1}^2 L \bigl(f(y^k) - f(x^{k+1})\bigr)$. Собирая:
$$
\tfrac12 \norm{z^k - x^*}^2 - \tfrac12 \norm{z^{k+1} - x^*}^2
\ge \alpha_{k+1}\bigl(f(y^k) - f^*\bigr) + \frac{1-\tau_k}{\tau_k}\alpha_{k+1}\bigl(f(y^k) - f(x^k)\bigr) - \alpha_{k+1}^2 L \bigl(f(y^k) - f(x^{k+1})\bigr) .
$$
Перегруппируем правая часть как $A_{k+1}\bigl(f(x^{k+1}) - f^*\bigr) +
\bigl[\alpha_{k+1} + \frac{1-\tau_k}{\tau_k}\alpha_{k+1} - \alpha_{k+1}^2
L\bigr]\bigl(f(y^k) - f^*\bigr) - \frac{1-\tau_k}{\tau_k}\alpha_{k+1}
\bigl(f(x^k) - f^*\bigr)$, где первое слагаемое даёт приращение левой
части \eqref{eq:est-seq}. Коэффициент при $f(y^k) - f^*$ обращается в
нуль при
\begin{equation}\label{eq:alpha-cond}
\alpha_{k+1}^2 L = A_{k+1} .
\end{equation}
а остаток телескопирует: $-\frac{1-\tau_k}{\tau_k}\alpha_{k+1}(f(x^k) - f^*) = -A_k(f(x^k) - f^*)$ по выбору $\tau_k = \alpha_{k+1}/A_{k+1}$.
Тогда \eqref{eq:est-seq} переходит в $k+1$.

Выбор $\alpha$: условие \eqref{eq:alpha-cond} при $A_{k+1} = A_k +
\alpha_{k+1}$ даёт квадратное уравнение $L\alpha_{k+1}^2 - \alpha_{k+1} -
A_k = 0$, откуда $\alpha_{k+1} = \frac{1 + \sqrt{1 + 4 L A_k}}{2L}$;
при $A_0 = 0$ это даёт $\alpha_1 = 1/L$, $A_k \ge \frac{k^2}{4L}$ (по
индукции: $A_{k+1} = A_k + \alpha_{k+1} \ge \frac{k^2}{4L} +
\frac{k}{2L} + \frac{1}{L} \cdot\frac{1}{2}\cdot 2 \ge \frac{(k+1)^2}{4L}$).
Наконец, $y^k = \tau_k z^k + (1-\tau_k)x^k$ с $\tau_k = \alpha_{k+1}/A_{k+1}
= \frac{2}{k+3}$ для выписанной рекуррентности; исключая $z^k$ (выражается
через $y^k$ и $x^k$ из подобия), получаем вторую строку \eqref{eq:nesterov}
с коэффициентом $\frac{k}{k+3}$. Подстановка $A_N \ge N^2/(4L)$ в
\eqref{eq:est-seq} даёт \eqref{eq:nesterov-rate}.

Сильно выпуклый случай: при $\mu > 0$ в \eqref{eq:est-seq}
квадратичный член усиливается в $(1 + \mu\alpha_{k+1})$ раз, и выбор
постоянного $\alpha_k \equiv \alpha = \frac{1}{\sqrt{\mu L}}$ даёт
$A_N = \frac{N}{\sqrt{\mu L}}$ и геометрическую прогрессию
\eqref{eq:nesterov-strong}; коэффициент инерции становится постоянным
$\frac{\sqrt\varkappa - 1}{\sqrt\varkappa + 1}$. Знаменатель
$\eqref{eq:nesterov-strong}$ по порядку равен $1 - 2/\sqrt\varkappa$;
по аргументу сходимость идёт со знаменателем
$\sqrt{(\sqrt\varkappa - 1)/(\sqrt\varkappa + 1)} \approx 1 - 1/\sqrt\varkappa$.
\end{proof}

\begin{remark}
Метод тяжёлого шарика и метод Нестерова отличаются одним словом:
\eqref{eq:hb} вычисляет градиент в текущей точке $x^k$, а
\eqref{eq:nesterov} — в экстраполированной точке $y^k$. Из-за этой
разницы тяжёлый шарик сходится лишь локально (контрпример с разрывным
гессианом у [37], замечание к (1.38)), а метод Нестерова — глобально и по
нижним оценкам. Оба метода дают знаменатель $1 - 2/\sqrt\varkappa$,
совпадающий с нижней оценкой \eqref{eq:lower-strong} по порядку.
\end{remark}

# Метод Ньютона {#sec:newton}

Метод Ньютона заменяет модель «квадратичная форма постоянной кривизны» на
«квадратичная аппроксимация с настоящим гессианом»:
$$
f(x + h) \approx f(x) + \scal{\nabla f(x)}{h} + \tfrac12 \scal{\nabla^2 f(x) h}{h},
\qquad
x^{k+1} = x^k - \bigl[\nabla^2 f(x^k)\bigr]^{-1} \nabla f(x^k).
$$
Для квадратичной функции — точное решение за один шаг.

\begin{theorem}[локальная квадратичная сходимость]\label{thm:newton-local}
Пусть $f$ дважды дифференцируема, $\mu$-сильно выпукла, гессиан липшицев
с константой $M$:
\begin{equation}\label{eq:hess-lip}
\norm{\nabla^2 f(x) - \nabla^2 f(y)} \le M \norm{x - y},
\end{equation}
и начальное приближение удовлетворяет
\begin{equation}\label{eq:newton-start}
q_0 := \frac{M}{2\mu^2} \norm{\nabla f(x^0)} < 1 .
\end{equation}
Тогда $x^k$ сходится к $x^*$ со скоростью квадратичной прогрессии
\begin{equation}\label{eq:newton-quad}
\norm{x^k - x^*} \le \frac{2\mu}{M}\, q_0^{2^k} .
\end{equation}
\end{theorem}

\begin{proof}
Обозначим $H_k = \nabla^2 f(x^k)$; по $\mu$-сильной выпуклости
$\norm{H_k^{-1}} \le \mu^{-1}$ (предложение \ref{prop:equiv}, в). По
определению шага $H_k(x^{k+1} - x^k) = -\nabla f(x^k)$, то есть
$\nabla f(x^k) + H_k (x^{k+1} - x^k) = 0$. Формула Тейлора с остатком в
интегральной форме:
$$
\nabla f(x^{k+1}) = \nabla f(x^k) + H_k (x^{k+1} - x^k) + \int_0^1 \bigl[ \nabla^2 f(x^k + t(x^{k+1} - x^k)) - \nabla^2 f(x^k) \bigr] (x^{k+1} - x^k)\, dt ,
$$
где первые два слагаемых обращаются в нуль; по \eqref{eq:hess-lip}
\begin{equation}\label{eq:newton-key}
\norm{\nabla f(x^{k+1})} \le \frac{M}{2} \norm{x^{k+1} - x^k}^2 .
\end{equation}
Далее, $\norm{x^{k+1} - x^k} = \norm{H_k^{-1} \nabla f(x^k)} \le
\mu^{-1} \norm{\nabla f(x^k)}$, и \eqref{eq:newton-key} даёт
$$
\norm{\nabla f(x^{k+1})} \le \frac{M}{2\mu^2} \norm{\nabla f(x^k)}^2 .
$$
Обозначив $a_k = \frac{M}{2\mu^2}\norm{\nabla f(x^k)}$, получаем
$a_{k+1} \le a_k^2$; при $a_0 = q_0 < 1$ имеем $a_k \le q_0^{2^k}$, то есть
$\norm{\nabla f(x^k)} \le \frac{2\mu^2}{M} q_0^{2^k}$. Остаётся перевести оценку с градиента на аргумент. По
$\mu$-сильной выпуклости и теореме \ref{thm:exist-unique}
$\norm{x^k - x^*} \le \frac{2}{\mu} \norm{\nabla f(x^k)}
\le \frac{4\mu}{M} q_0^{2^k}$. Константа улучшается до $2\mu/M$
стандартным шагом [13], гл. 1, §5: оценка
\eqref{eq:newton-key} вместе с $x^{k+1} - x^k = (x^{k+1} - x^*) -
(x^k - x^*)$ и $\norm{H_k^{-1}} \le \mu^{-1}$ даёт по индукции
$\norm{x^k - x^*} \le \frac{2\mu}{M} a_k$, где
$a_k = \frac{M}{2\mu^2}\norm{\nabla f(x^k)}$, откуда
$\norm{x^{k+1} - x^*} \le \frac{2\mu}{M} a_{k+1} \le \frac{2\mu}{M}
q_0^{2^{k+1}}$ — это \eqref{eq:newton-quad}.
\end{proof}

\begin{remark}[существенность условия $q_0 < 1$]
Без близости начального приближения метод Ньютона может расходиться:
для $f(x) = \sqrt{1 + x^2}$ (выпуклая, гладкая) при больших $\abs{x^0}$
итерации уходят на бесконечность [13]. Условие \eqref{eq:newton-start}
неработоспособно как практический критерий — $\norm{\nabla f(x^0)}$
неизвестен до счёта; его роль — гарантия сходимости, а практический
выбор $x^0$ — отдельная задача (глобализация: демпфированный шаг
$x^{k+1} = x^k - \gamma_k H_k^{-1}\nabla f(x^k)$ с $\gamma_k$ из
одномерной минимизации или правила Армихо сходится из любой точки
сильно выпуклой задачи — [13], гл. 3, §1).
\end{remark}

\begin{theorem}[Ньютона—Канторовича]\label{thm:kantorovich}
Пусть $f$ дважды непрерывно дифференцируема в шаре
$\Omega = \{x : \norm{x - x^0} \le r\}$, гессиан невырожден в $x^0$,
$\Gamma_0 = [\nabla^2 f(x^0)]^{-1}$ и
\begin{equation}\label{eq:kant-cond}
\norm{\Gamma_0 \nabla f(x^0)} \le \eta, \qquad
\norm{\Gamma_0 \nabla^2 f(x)} \le K \quad (x \in \Omega), \qquad
h := K \eta \le \tfrac12 .
\end{equation}
Тогда при
\begin{equation}\label{eq:kant-radius}
r \ge r_0 = \frac{1 - \sqrt{1 - 2h}}{h}\, \eta
\end{equation}
(при $h = 1/2$ читается $r_0 = 2\eta$) в $\Omega$ существует решение
$x^*$ уравнения $\nabla f(x) = 0$, к которому сходится метод Ньютона, причём
\begin{equation}\label{eq:kant-aprior}
\norm{x^k - x^*} \le \frac{1}{2^k}\, (2h)^{2^k} \frac{\eta}{h} .
\end{equation}
Решение единственно в $\Omega$, если $r < r_1 = \frac{1 + \sqrt{1-2h}}{h}\eta$
или $r \le r_1$ при $h = 1/2$. Для модифицированного процесса
(гессиан заморожен: $x^{k+1} = x^k - \Gamma_0 \nabla f(x^k)$) при $h < 1/2$
$$
\norm{x'^k - x^*} \le \frac{\eta}{h} \bigl(1 - \sqrt{1 - 2h}\bigr)^{k+1} .
$$
\end{theorem}

\begin{proof}
Сведение к скалярному мажорантному уравнению. Рассмотрим вещественную
функцию
$$
\psi(t) = \frac{K}{2} t^2 - t + \eta = \frac{h}{\eta} t^2 - t + \eta ,
$$
у неё $\psi(0) = \eta > 0$, $\psi'(0) = -1$ и корни
$t^* = r_0$, $t^{**} = r_1$. Метод Ньютона для $\psi$ с началом $t_0 = 0$:
$t_{k+1} = t_k - \psi(t_k)/\psi'(t_k)$, сходится возрастающе к $t^*$ при
$h \le 1/2$. Докажем по индукции
\begin{equation}\label{eq:kant-major}
\norm{x^{k+1} - x^k} \le t_{k+1} - t_k .
\end{equation}
База $k = 0$: $\norm{x^1 - x^0} = \norm{\Gamma_0 \nabla f(x^0)} \le \eta =
t_1 - t_0$.

Шаг: пусть $x^k \in \Omega$ и \eqref{eq:kant-major} верно для $k-1$; тогда
$$
x^{k+1} - x^k = \bigl[ \nabla^2 f(x^k) \bigr]^{-1} \Bigl( \int_0^1 \bigl[ \nabla^2 f(x^k) - \nabla^2 f(x^{k-1} + t(x^k - x^{k-1})) \bigr] (x^k - x^{k-1})\, dt \Bigr) .
$$
Оценим норму интеграла через $K$ и липшицевость
$\nabla^2 f$: так как $\norm{\Gamma_0 \nabla^2 f(x)} \le K$ на $\Omega$,
обратная к $\nabla^2 f(x)$ существует и
$\norm{[\nabla^2 f(x)]^{-1} \Gamma_0^{-1}} \le (1 - K\norm{x - x^0})^{-1}$
по лемме о почти единичном операторе; после стандартной (громоздкой, но
элементарной) оценки появляется мажоранта
$\norm{x^{k+1} - x^k} \le \frac{K}{2} \frac{(t_k - t_{k-1})^2}{1 - K t_k} =
t_{k+1} - t_k$, где равенство справа — это и есть шаг Ньютона для $\psi$
(проверяется подстановкой $\psi$ и $\psi'$). Из \eqref{eq:kant-major}
следует, что $x^k$ фундаментальна и $\norm{x^k - x^0} \le t_k \le t^* = r_0$, то есть $x^k \in \Omega$; предел $x^*$ удовлетворяет $\nabla f(x^*) = 0$ (предельный переход в $\nabla f(x^k) + \nabla^2 f(x^k)(x^{k+1} - x^k) = 0$), а $\norm{x^k - x^*} \le t^* - t_k$. Оценка погрешности:
решая рекуррентность для $t^* - t_k$ (она удовлетворяет той же квадратичной
мажоранте), получаем $t^* - t_k \le \frac{1}{2^k}(2h)^{2^k}\frac{\eta}{h}$,
то есть \eqref{eq:kant-aprior}. Единственность: если $\tilde x$ — другое
решение в шаре, то $\norm{\tilde x - x^0} \le r$, и разностное равенство
вместе с условиями \eqref{eq:kant-cond} даёт
$\norm{\tilde x - x^*} < \norm{\tilde x - x^*}$ при
$r < r_1$ — противоречие; полное рассуждение — в [22], гл. XVIII, §1.
Модифицированный процесс мажорируется геометрической прогрессией со
знаменателем $1 - \sqrt{1 - 2h}$ — там же, оценка (34).
\end{proof}

\begin{remark}
Константа $h$ аффинно инвариантна: при невырожденной линейной замене
переменных $x = C y$ значения $\eta$ и $K$ преобразуются так, что $h$
сохраняется; современная форма теоремы формулируется непосредственно в
аффинно-инвариантных терминах $\norm{[\nabla^2 f(x^0)]^{-1}\nabla f(x^0)}$
и липшицевости $\norm{[\nabla^2 f(x^0)]^{-1} \nabla^2 f(x)}$ [28].
Условие $h \le 1/2$ точно (константа $1/2$ неулучшаема) [22].
\end{remark}

# Условная оптимизация {#sec:constrained}

Постановка: $\min_{x \in Q} f(x)$ с выпуклым замкнутым $Q \subseteq \R^n$.
Три рабочих подхода: штрафы (сведение к безусловной задаче), проекция
градиента и условный градиент. Первый сводит задачу к уже разобранной,
второй и третий — содержание лекции курса прошлого года.

## Проекция на выпуклое множество

Проекцией точки $y$ на $Q$ называется ближайшая точка множества:
\begin{equation}\label{eq:proj}
\proj_Q(y) = \argmin_{x \in Q} \tfrac12 \norm{x - y}^2 .
\end{equation}
Существование при замкнутом $Q$ (по Вейерштрассу), единственность при
выпуклом $Q$ (по строгой выпуклости квадрата расстояния). Примеры: на шар
радиуса $R$ центра $x_0$ — $\proj(y) = x_0 + R(y - x_0)/\norm{y - x_0}$;
на полупространство $\{x : c^{\mathsf T} x \le b\}$ —
$\proj(y) = y - \frac{(c^{\mathsf T} y - b)_+}{\norm{c}^2} c$; обе проверяются
прямой подстановкой в критерий ниже.

\begin{theorem}[критерий проекции, неравенство Бурбаки—Чини—Гольдстейна]\label{thm:proj}
Пусть $Q$ выпукло замкнуто. Тогда $x = \proj_Q(y)$ тогда и только тогда,
когда
\begin{equation}\label{eq:proj-crit}
\scal{y - x}{z - x} \le 0 \qquad \forall z \in Q .
\end{equation}
В частности,
\begin{equation}\label{eq:proj-cosine}
\norm{z - \proj_Q(y)}^2 + \norm{y - \proj_Q(y)}^2 \le \norm{z - y}^2
\qquad \forall z \in Q .
\end{equation}
\end{theorem}

\begin{proof}
$x$ минимизирует дифференцируемую выпуклую функцию
$d(x) = \frac12\norm{x - y}^2$ на выпуклом $Q$; по теореме \ref{thm:foc}
для условной задачи (вариационное неравенство первого порядка:
$\scal{\nabla d(x)}{z - x} \ge 0$ для всех $z \in Q$) это условие
необходимое и достаточное. Так как $\nabla d(x) = x - y$, получаем
\eqref{eq:proj-crit}. Для \eqref{eq:proj-cosine} распишем
$\norm{z - y}^2 = \norm{(z - \proj y) + (\proj y - y)}^2$ и применим
\eqref{eq:proj-crit} со скалярным произведением $\le 0$.
\end{proof}

\begin{corollary}[неэкспансивность проекции]\label{cor:proj-nonexp}
Проекция не расширяет расстояния:
$$
\norm{\proj_Q(y) - \proj_Q(y')} \le \norm{y - y'} .
$$
\end{corollary}

\begin{proof}
Применим \eqref{eq:proj-crit} дважды: к паре $(y, \proj y)$ с
$z = \proj y'$ и к паре $(y', \proj y')$ с $z = \proj y$:
$$
\scal{y - \proj y}{\proj y' - \proj y} \le 0, \qquad
\scal{y' - \proj y'}{\proj y - \proj y'} \le 0 .
$$
Сложим и перегруппируем:
$$
\scal{y - y'}{\proj y - \proj y'} \ge \norm{\proj y - \proj y'}^2 .
$$
По Коши—Буняковского левая часть не больше
$\norm{y - y'}\,\norm{\proj y - \proj y'}$.
\end{proof}

## Метод проекции градиента

\begin{equation}\label{eq:pgd}
x^{k+1} = \proj_Q\bigl( x^k - \gamma \nabla f(x^k) \bigr), \qquad \gamma = \tfrac1L .
\end{equation}

\begin{theorem}[сходимость PGD]\label{thm:pgd}
Пусть $f$ выпукла и $L$-гладка на выпуклом замкнутом $Q$, $x^*$ — минимум
$f$ на $Q$, $R = \norm{x^0 - x^*}$. Тогда итерации \eqref{eq:pgd}
удовлетворяют
\begin{equation}\label{eq:pgd-rate}
f(x^N) - f^* \le \frac{L R^2}{2N} .
\end{equation}
\end{theorem}

\begin{proof}
Обозначим $y^k = x^k - \frac1L\nabla f(x^k)$, так что $x^{k+1} = \proj_Q(y^k)$.
По лемме спуска (лемма \ref{lem:descent})
$f(x^{k+1}) \le f(y^k) \le f(x^k) - \frac{1}{2L}\norm{\nabla f(x^k)}^2$
(первая неравенство: $x^{k+1}$ минимум на $Q$, $y^k$ может быть вне $Q$? —
нет, применяем лемму спуска непосредственно к точке $y^k$). Далее, для
любого $z \in Q$ по неэкспансивности и \eqref{eq:proj-cosine}:
$$
\norm{x^{k+1} - z}^2 \le \norm{y^k - z}^2 = \norm{x^k - z}^2 - \frac{2}{L} \scal{\nabla f(x^k)}{x^k - z} + \frac{1}{L^2}\norm{\nabla f(x^k)}^2 .
$$
Возьмём $z = x^*$ и используем выпуклость
($\scal{\nabla f(x^k)}{x^k - x^*} \ge f(x^k) - f^*$) и оценку
$\frac{1}{L^2}\norm{\nabla f(x^k)}^2 \le \frac{2}{L}(f(x^k) - f(x^{k+1}))$
из леммы спуска:
$$
\norm{x^{k+1} - x^*}^2 \le \norm{x^k - x^*}^2 - \frac{2}{L}\bigl(f(x^k) - f^*\bigr) + \frac{2}{L}\bigl(f(x^k) - f(x^{k+1})\bigr) .
$$
Перегруппируем и просуммируем по $k = 0, \dots, N-1$; телескопирование
даёт $\frac{2}{L}\sum_{k=0}^{N-1}\bigl(f(x^{k+1}) - f^*\bigr) \le R^2$.
Так как $f(x^{k+1})$ монотонно убывает (лемма спуска) и потому каждое
слагаемое не меньше $f(x^N) - f^*$, получаем
$\frac{2N}{L}\bigl(f(x^N) - f^*\bigr) \le R^2$, то есть \eqref{eq:pgd-rate}.
\end{proof}

\section{Метод Франка—Вулфа (условный градиент)}

Метод Франка—Вулфа (англ. *conditional gradient*) заменяет проекцию
(квадратичная задача на $Q$) линейной задачей:
\begin{equation}\label{eq:fw}
y^k = \argmin_{y \in Q} \scal{\nabla f(x^k)}{y}, \qquad
x^{k+1} = x^k + \gamma_k (y^k - x^k), \qquad \gamma_k = \frac{2}{k+2} .
\end{equation}
Оракул \eqref{eq:fw} — линейная минимизация на $Q$; для симплекса, шара,
системы неравенств она существенно дешевле проекции.

\begin{theorem}[сходимость Франка—Вулфа]\label{thm:fw}
Пусть $f$ выпукла и $L$-гладкая на выпуклом компакте $Q$ диаметра
$R = \max_{x, y \in Q} \norm{x - y}$, $x^*$ — минимум. Тогда итерации
\eqref{eq:fw} удовлетворяют
\begin{equation}\label{eq:fw-rate}
f(x^N) - f^* \le \frac{2 L R^2}{N + 2} .
\end{equation}
\end{theorem}

\begin{proof}
Обозначим $\delta_k = f(x^k) - f^*$. Из $L$-гладкости (неравенство
\eqref{eq:upper-par} с $x = x^k$, $y = x^{k+1}$) и выпуклости:
$$
f(x^{k+1}) \le f(x^k) + \gamma_k \scal{\nabla f(x^k)}{y^k - x^k} + \frac{L\gamma_k^2}{2} \norm{y^k - x^k}^2 .
$$
Так как $y^k$ минимизирует линейную функцию на $Q$,
$\scal{\nabla f(x^k)}{y^k - x^k} \le \scal{\nabla f(x^k)}{x^* - x^k} \le
f^* - f(x^k) = -\delta_k$ (последнее — выпуклость, теорема
\ref{thm:foc}), а $\norm{y^k - x^k} \le R$. Следовательно
$$
\delta_{k+1} \le (1 - \gamma_k) \delta_k + \frac{L R^2}{2} \gamma_k^2 .
$$
При $\gamma_k = 2/(k+2)$: индукция $\delta_k \le \frac{2LR^2}{k+2}$.
База $k = 0$: $\delta_1 \le (1 - \gamma_0)\delta_0 + \frac{LR^2}{2}\gamma_0^2
= \frac{LR^2}{2} \le LR^2 = \frac{2LR^2}{2}$. Шаг: пусть
$\delta_k \le \frac{2LR^2}{k+2}$; тогда
$$
\delta_{k+1} \le \frac{k}{k+2} \cdot \frac{2LR^2}{k+2} + \frac{2LR^2}{(k+2)^2} = \frac{2LR^2 (k + 1)}{(k+2)^2} \le \frac{2LR^2}{k+3},
$$
поскольку $(k+1)(k+3) \le (k+2)^2$.
\end{proof}

\begin{remark}[что сильная выпуклость не даёт]
Встречающееся в изложениях (в том числе в слайдах лекции прошлого года)
утверждение, будто при $\mu$-сильной выпуклости $f$ метод \eqref{eq:fw} с
тем же $\gamma_k$ сходится как $O(LR^2/k^2)$, неверно: доказательство
отбрасывает полезное слагаемое с $\mu$, после чего рекуррентность в точности
та, что в теореме \ref{thm:fw}, и улучшения нет. Контрпример —
$f(x) = \norm{x}^2$ на симплексе $\{x \ge 0 : \sum x_i = 1\}$ (сильно
выпуклая функция): для неё FW с $\gamma_k = 2/(k+2)$ сходится как
$\Theta(1/k)$, и оценка $1/k^2$ нарушается; измерение — пример 5
\texttt{examples.ipynb}. Ускорение до $O(1/k^2)$ достигается при сильной
выпуклости множества $Q$ (кривизна, условие Лакост-Жюльен—Ягги), а не
функции.
\end{remark}

# Негладкая оптимизация {#sec:nonsmooth}

Для недифференцируемых выпуклых функций условие $\nabla f = 0$ заменяется
условием на множество — субдифференциал (англ. *subdifferential*).
Субградиентом функции $f$ в точке $x$ называется вектор $g$, такой что
\begin{equation}\label{eq:subgrad}
f(y) \ge f(x) + \scal{g}{y - x} \qquad \forall y ;
\end{equation}
множество всех субградиентов — субдифференциал $\partial f(x)$.

\begin{theorem}[свойства субдифференциала]\label{thm:subdiff}
Пусть $f$ выпукла. Тогда:

\textbf{(а)} $\partial f(x)$ непусто, выпукло и замкнуто для любого
$x$;

\textbf{(б)} для дифференцируемой в $x$ выпуклой $f$
$\partial f(x) = \{\nabla f(x)\}$;

\textbf{(в)} $x^*$ — точка минимума $f$ тогда и только тогда, когда
$0 \in \partial f(x^*)$.
\end{theorem}

\begin{proof}
\textbf{(а)} Множество $\partial f(x)$ — пересечение замкнутых полупространств
по параметру $y$ (выпуклость и замкнутость). Непустота: эпиграф
$\operatorname{epi} f$ выпукл и замкнут, точка $(x, f(x))$ — его граничная;
по теореме об отделимости через неё проходит опорная гиперплоскость с
нормалью $(g, -1)$, $g \ne 0$... точнее, существует ненулевой
$(a, \beta) \in \R^n \times \R$ с $\scal{a}{y - x} + \beta(t - f(x)) \le
0$ для $(y, t) \in \operatorname{epi} f$; обязательно $\beta < 0$ (иначе
при $y = x$, $t \to \infty$ противоречие), делим на $-\beta$ и получаем
\eqref{eq:subgrad} с $g = -a/\beta$.

\textbf{(б)} Дифференцируемость: из \eqref{eq:subgrad} с $y = x + td$ после
деления на $t$ и предельных переходов $t \to 0+$ и $t \to 0-$ получаем
$\scal{g}{d} = \scal{\nabla f(x)}{d}$ для всех $d$.

\textbf{(в)} Если $0 \in \partial f(x^*)$, то \eqref{eq:subgrad} даёт
$f(y) \ge f(x^*)$ для всех $y$. Обратно, если $x^*$ — минимум, то
$f(y) \ge f(x^*) = f(x^*) + \scal{0}{y - x^*}$, то есть $0$ — субградиент.
\end{proof}

Метод: субградиентный спуск $x^{k+1} = x^k - \gamma_k g^k$, $g^k \in
\partial f(x^k)$. Сходимость медленнее градиентного: для $L_0$-липшицевой
выпуклой $f$ ($\norm{g} \le L_0$ для $g \in \partial f$) усреднённая точка
$\bar x^N$ с весами $\gamma_k$ удовлетворяет $f(\bar x^N) - f^* \le
L_0 R / \sqrt{N}$, и оценка неулучшаема [37], §2. Медленность не
поправима: это цена отсутствия гладкости, а не недочёт метода.

# Градиент функционала через сопряжённую задачу {#sec:adjoint}

Завершает конспект приложение, замыкающее его на вопрос 02 (ML и обратные
задачи): как вычислить градиент функционала невязки, если значение
функционала требует решения дифференциального уравнения. Постановка —
линейная обратная задача: пусть $A$ — линейный оператор (прямая задача),
по измерениям $f$ ищется $q$ из условия $A q \approx f$ методом
минимизации функционала невязки
\begin{equation}\label{eq:adj-J}
J(q) = \norm{A q - f}^2
\end{equation}
(квадратичный функционал качества из раздела \ref{sec:intro}; обычно к нему
добавляют регуляризатор Тихонова — см. вопрос 02).

\begin{theorem}[градиент через сопряжённый оператор]\label{thm:adjoint}
Пусть $A$ — непрерывный линейный оператор между гильбертовыми
пространствами. Тогда $J(q) = \norm{A q - f}^2$ фреше дифференцируем и
\begin{equation}\label{eq:adjoint-formula}
J'(q) = 2 A^* (A q - f),
\end{equation}
где $A^*$ — оператор, сопряжённый к $A$.
\end{theorem}

\begin{proof}
Разложим приращение:
$$
J(q + \delta q) - J(q) = 2 \scal{A q - f}{A\, \delta q} + \norm{A\, \delta q}^2 = 2 \scal{A^*(A q - f)}{\delta q} + o(\norm{\delta q}),
$$
поскольку $\norm{A \delta q} \le \norm{A} \norm{\delta q}$ и потому
$\norm{A \delta q}^2 = O(\norm{\delta q}^2) = o(\norm{\delta q})$.
Сравнивая с определением фреше-дифференциала
$J(q + \delta q) - J(q) = \scal{J'(q)}{\delta q} + o(\norm{\delta q})$,
получаем \eqref{eq:adjoint-formula}.
\end{proof}

Если прямая задача задана дифференциальным уравнением, сопряжённый оператор
вычисляется через сопряжённое дифференциальное уравнение, и формула
\eqref{eq:adjoint-formula} обретает вычислительный смысл. Пример —
обратная задача теплопроводности с обратным временем [14], гл. 8: прямая
задача
$$
u_t = -u_{xx}, \quad x \in (0, l),\ t \in (0, T); \qquad u(0, t) = u(l, t) = 0; \qquad u(x, T) = q(x),
$$
по измеренному $f(x) = u(x, 0) + \text{шум}$ минимизируется функционал
\eqref{eq:adj-J} с $A$ — оператором прямой задачи ($q \mapsto u(\cdot, 0)$).
Сопряжённая задача для $\psi$:
$$
\psi_t = -\psi_{xx}; \qquad \psi(0,t) = \psi(l,t) = 0; \qquad \psi(x, 0) = 2\,(u(x, 0) - f(x)),
$$
и интегрированием по частям на прямоугольнике $(0,l)\times(0,T)$ для разности
решений $\delta u$, порождённой приращением $\delta q$, получается
$$
J(q + \delta q) - J(q) = \int_0^l \psi(x, T)\, \delta q(x)\, dx + o(\norm{\delta q}),
$$
то есть градиент вычисляется одним решением сопряжённой задачи:
$J'(q) = \psi(\cdot, T)$. Цена одного градиента — одна прямая и одна
сопряжённая задача, не зависящая от размерности $q$; конечные разности по
параметрам потребовали бы $O(\dim q)$ решений прямой задачи. Этот приём —
основа градиентных методов для обратных задач и обратного распространения
ошибки в обучении сетей (вопрос 02).

# Что почитать ещё: темы за рамками вопроса {#sec:further}

\textbf{Стохастический градиентный спуск} (англ. *SGD*) — градиентный спуск
по случайной несмещённой оценке градиента с дисперсией $D$: для выпуклых
гладких задач оценка $f(\bar x^N) - f^* = O(\sqrt{D} R / \sqrt{N})$ и
неулучшаема [37], приложение; для сильно выпуклых — линейная сходимость до
$O(D/(L\mu))$-окрестности. Дисперсия снижается минибатчингом; методы
редукции дисперсии (SVRG, SAGA) убирают плату $O(1/\sqrt N)$. Это рабочая
лошадка вопроса 02.

\textbf{Зеркальный спуск} (англ. *mirror descent*) — замена евклидовой
проекции на «проекцию» по дивергенции Брэгмана, определяемой выбором
прокс-функции; для задач на симплексе заменяет $O(\sqrt{\ln n})$ штраф за
размерность. Объявлен в заголовке колоды лекции прошлого года; вопрос 10
его не требует, см. [37], §2.

\textbf{Универсальный градиентный спуск} [37], §5 — метод, автоматически
настраивающийся на неизвестную степень гладкости $\nu \in [0,1]$ через
бэктрекинг по $L$; связка с правилом выбора шага раздела \ref{sec:compute}.

\textbf{Квазиньютоновские методы} (BFGS и др.) — итерации
$x^{k+1} = x^k - \gamma_k H_k \nabla f(x^k)$ с матрицами $H_k$,
обновляемыми по разностям градиентов так, чтобы выполнялось
квазиньютоновское условие $H_{k+1} y^k = p^k$; на квадратичной задаче
сходятся за $n$ шагов, на общей — сверхлинейно [12], гл. 5, §2; [13],
гл. 3, §3.

\textbf{Помехи.} При зашумлённом градиенте нижняя оценка асимптотически
$O(1/k)$ для выпуклых и геометрическая для сильно выпуклых, и градиентный
метод с убывающим шагом асимптотически оптимален [13], гл. 4, §5 —
вопрос для функционала качества в условиях шумных измерений.

# Источники {#sec:sources}

Основные источники вопроса. Локаторы (страницы) даны в README этого вопроса.

\textbf{[12]} — условия оптимальности первого и второго порядка, выпуклость
и сильная выпуклость, субдифференциал, метод сопряжённых направлений и
квазиньютоновские схемы. Теорема о существовании и единственности
минимума и условия второго порядка — по этой книге.

\textbf{[13]} — теория скорости сходимости градиентного спуска (точная
оценка и оптимальный шаг), локальная квадратичная сходимость Ньютона с
контрпримерами, наискорейший спуск и неравенство Канторовича, метод
тяжёлого шарика, сопряжённые градиенты и их оптимальность, нижние оценки
при помехах. Теоремы \ref{thm:exact-rate}, \ref{thm:hb},
\ref{thm:newton-local} и разбор наискорейшего спуска — по этой книге.

\textbf{[37]} — градиентный спуск с постоянным шагом и оценки в трёх
режимах (теорема \ref{thm:gd-conv}), нижние оценки (теорема
\ref{thm:lower}), метод Нестерова и метод подобных треугольников
(теорема \ref{thm:nesterov}), субградиентный метод и стохастический
спуск, проекция градиента, бэктрекинг. Материал лекции курса прошлого
года.

\textbf{[02]} — чебышёвские полуитерации и устойчивая трёхчленная
рекуррентность (теорема \ref{thm:cheb}), метод сопряжённых градиентов и
оценка в энергетической норме.

\textbf{[09]} — чебышёвский набор итерационных параметров: оценка $q_n$
и число итераций $n_0(\varepsilon)$ (следствие теоремы
\ref{thm:cheb}), неустойчивость множительной формы (замечание
\ref{rem:cheb-stable}).

\textbf{[22]} — теорема Ньютона—Канторовича с полным доказательством
(теорема \ref{thm:kantorovich}), точность константы $1/2$.

\textbf{[28]} — аффинно-инвариантная форма теоремы Канторовича и таблица
оценок погрешности (замечание к теореме \ref{thm:kantorovich}).

\textbf{[14]} — градиент функционала невязки через сопряжённую задачу
(теорема \ref{thm:adjoint} и пример обратной задачи теплопроводности).
