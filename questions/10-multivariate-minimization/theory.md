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

**Машинное обучение.** Обучение — это минимизация по параметрам $w$ модели
эмпирического риска — средней ошибки $\ell$ модели $h_w$ на выборке
$(x_i, y_i)$:
$$
\frac1N \sum_{i=1}^N \ell\bigl(h_w(x_i), y_i\bigr) ;
$$
обучение нейронной сети — минимизация этой функции в пространстве
размерности в миллионы. Метод, которым это делается на практике (стохастический
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

\textbf{(а)} для выпуклой $f$ гладкость \eqref{eq:lipgrad} эквивалентна
мажорированию квадратичной формой с кривизной $L$:
\begin{equation}\label{eq:upper-par}
f(y) \le f(x) + \scal{\nabla f(x)}{y - x} + \frac{L}{2} \norm{y - x}^2 ;
\end{equation}
прямая сторона верна и без выпуклости; обратная без выпуклости неверна:
$f(x) = -x^4$ на прямой удовлетворяет \eqref{eq:upper-par} при любом
$L \ge 0$, но её производная не липшицева ни при каком $L$;

\textbf{(б)} для дважды дифференцируемой $f$ условие \eqref{eq:lipgrad}
равносильно $\nabla^2 f(x) \preceq L I$ для всех $x$;

\textbf{(в)} сильная выпуклость \eqref{eq:strong} для дважды
дифференцируемой $f$ равносильна $\nabla^2 f(x) \succeq \mu I$ для всех $x$;

\textbf{(г)} для гладкой выпуклой $f$ сильная выпуклость \eqref{eq:strong}
равносильна сильной монотонности градиента:
\begin{equation}\label{eq:mono}
\scal{\nabla f(y) - \nabla f(x)}{y - x} \ge \mu \norm{y - x}^2 ;
\end{equation}

\textbf{(д)} для гладкой выпуклой $f$ градиент коэрцитивен (англ.
\emph{co-coercive}):
\begin{equation}\label{eq:cocoerc}
\scal{\nabla f(y) - \nabla f(x)}{y - x} \ge \frac{1}{L} \norm{\nabla f(y) - \nabla f(x)}^2 .
\end{equation}
\end{proposition}

\begin{proof}
\textbf{(а)} $g(t) = f(x + t(y-x))$, $t \in [0,1]$; тогда
$g'(t) = \scal{\nabla f(x + t(y-x))}{y - x}$ и
$$
f(y) - f(x) - \scal{\nabla f(x)}{y - x}
= \int_0^1 \scal{\nabla f(x + t(y-x)) - \nabla f(x)}{y - x}\, dt .
$$
Оценивая подынтегральное выражение через липшицевость градиента,
$$
\int_0^1 \scal{\nabla f(x + t(y-x)) - \nabla f(x)}{y - x}\, dt
\le \int_0^1 L t \norm{y-x}^2\, dt = \frac{L}{2}\norm{y-x}^2 .
$$

Обратно (для выпуклой $f$): зафиксируем $y$ и применим прямую сторону к
выпуклой $L$-гладкой функции $g(x) = f(x) - \scal{\nabla f(y)}{x}$,
минимализируемой в точке $y$ ($\nabla g(y) = 0$). Подставляя в
\eqref{eq:upper-par} вместо $y$ точку $x - \frac1L \nabla g(x)$:
$$
g(y) \le g\bigl(x - \tfrac1L \nabla g(x)\bigr)
\le g(x) - \tfrac{1}{2L} \norm{\nabla g(x)}^2 ,
$$
то есть $f(x) - f(y) - \scal{\nabla f(y)}{x - y} \ge
\frac{1}{2L}\norm{\nabla f(x) - \nabla f(y)}^2$. Выписывая то же
неравенство с переставленными $x$ и $y$ и складывая, получаем
кокоэрцитивность \eqref{eq:cocoerc} из пункта (д); по
Коши—Буняковскому она даёт \eqref{eq:lipgrad}:
$\norm{\nabla f(y) - \nabla f(x)} \cdot \norm{y - x} \ge
\frac1L \norm{\nabla f(y) - \nabla f(x)}^2$.

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
\scal{\nabla^2 f(x + t(y-x)) (y-x)}{y-x}\, dt .
$$
Подынтегральное выражение не меньше $(1-t)\,\mu \norm{y-x}^2$, поэтому
$$
f(y) \ge f(x) + \scal{\nabla f(x)}{y-x} + \frac{\mu}{2}\norm{y-x}^2 .
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

\textbf{(д)} Доказано в ходе обратного направления (а): неравенство
\eqref{eq:cocoerc} получено там для произвольной выпуклой $L$-гладкой $f$.
\end{proof}

\begin{remark}
Запись обозначений у разных школ различается ровно вдвое по $\mu$:
у [12] (Сухарев, Тимохов, Федоров) сильная выпуклость вводится слагаемым $\vartheta \lambda(1-\lambda)
\norm{x^1 - x^2}^2$ в определении, что соответствует $\mu = 2\vartheta$;
у [13] (Поляк) константа сильной выпуклости $l$ совпадает с нашей $\mu$. Формулы
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
и минимум $\scal{\nabla f(x)}{d}$ по таким $d$ равен
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
$\bar x^N = \frac1N \sum_{k=1}^{N} x^k$
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

\textbf{(б)} Выпуклость даёт $f(x) - f^* \le \scal{\nabla f(x)}{x - x^*}$
(теорема \ref{thm:foc}), поэтому
$$
\norm{x^{k+1} - x^*}^2 = \norm{x^k - x^*}^2 - \frac{2}{L} \scal{\nabla f(x^k)}{x^k - x^*} + \frac{1}{L^2}\norm{\nabla f(x^k)}^2 ,
$$
где $\frac{1}{L^2}\norm{\nabla f(x^k)}^2 \le \frac{2}{L}\bigl(f(x^k) - f(x^{k+1})\bigr)$
по \eqref{eq:telescope-key}. Перегруппировка даёт
$$
\frac{2}{L}\bigl(f(x^{k+1}) - f^*\bigr) \le \norm{x^k - x^*}^2 - \norm{x^{k+1} - x^*}^2 .
$$
Суммируем по $k = 0, \dots, N-1$; правая часть телескопируется:
$$
\frac{2}{L} \sum_{k=1}^{N} \bigl(f(x^k) - f^*\bigr) \le R^2 .
$$
По выпуклости $f(\bar x^N) \le \frac1N \sum_{k=1}^N f(x^k)$, и деление на $N$
даёт \eqref{eq:gd-convex}.

\textbf{(в)} Сначала оценка по аргументу. Функция $g = f - \frac{\mu}{2}\norm{\cdot}^2$
выпукла (это \eqref{eq:strong} в точке $x = y$) и $(L-\mu)$-гладка: по
пункту (а) достаточно выпуклости $\frac{L}{2}\norm{\cdot}^2 - f$, а она
имеется — это прямая сторона (а), применённая к $L$-гладкой $f$. Применяем
к $g$ кокоэрцитивность (предложение \ref{prop:equiv}, д) и переносим члены:
$$
L \scal{\nabla f(x) - \nabla f(y)}{x - y} \ge \norm{\nabla f(x) - \nabla f(y)}^2 + \mu L \norm{x - y}^2 .
$$
Подставляем $x = x^k$, $y = x^*$ (где $\nabla f(x^*) = 0$) в разложение
$$
\norm{x^{k+1} - x^*}^2 = \norm{x^k - x^*}^2 - \frac{2}{L} \scal{\nabla f(x^k)}{x^k - x^*} + \frac{1}{L^2}\norm{\nabla f(x^k)}^2 :
$$
$$
\norm{x^{k+1} - x^*}^2 \le \frac{L - \mu}{L + \mu}\, \norm{x^k - x^*}^2 - \frac{L - \mu}{L^2(\mu + L)}\, \norm{\nabla f(x^k)}^2 \le \Bigl(1 - \frac{\mu}{L}\Bigr) \norm{x^k - x^*}^2 ,
$$
так как $\frac{L-\mu}{L+\mu} \le 1 - \frac{\mu}{L}$. Итерация логарифма
даёт $\norm{x^N - x^*}^2 \le (1 - \mu/L)^N R^2$. Далее, применяя
\eqref{eq:upper-par} с $x = x^*$, $y = x^N$:
$f(x^N) - f^* \le \frac{L}{2}\norm{x^N - x^*}^2 \le \frac{L R^2}{2} (1 - \mu/L)^N$.
Наконец, $f(x^N) - f^* \le \frac1N \sum_{k=1}^N \bigl(f(x^k) - f^*\bigr) \le
\frac{L R^2}{2N}$ — это пункт (б) и монотонность $f(x^N) \le f(x^k)$.
\end{proof}

\begin{remark}[существенность посылок]\label{rem:gd-assumptions}
Каждая посылка теоремы \ref{thm:gd-conv} существенна, и контрпримеры
явные [13] (Поляк). \textbf{(1)} Без липшицевости градиента метод может расходиться:
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
$\abs{f'(x^k)} = O(k^{-2/3})$ [13] (Поляк).
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
продвигает по длине лишь на $O(1/\varkappa)$ долю расстояния. Фактический
знаменатель на задаче с $\varkappa = 10^3$ измеряет пример 1
\texttt{examples.ipynb}: он совпадает с $q^*$ из \eqref{eq:opt-step}.
\end{remark}

# Нижние оценки для методов первого порядка {#sec:lower}

Теорема \ref{thm:gd-conv} давала оценки сверху. Покажем, что по порядку они
точны: ни один метод, использующий только оракулы значения и градиента,
не может сходиться быстрее. Функции-«подсадки» и постановка — из [37] (Гасников),
упр. 1.3; выкладки ниже выписаны полностью.

\begin{theorem}[нижние оценки]\label{thm:lower}
\textbf{(а)} Для любых $L > 0$, $N \ge 1$ существует $L$-гладкая выпуклая
$f$ на $\R^n$ ($2N+1 \le n$) с точкой минимума $x^*$ такая, что при
$x^0 = 0$ для любого метода, генерирующего точки по правилу
$x^k \in x^0 + \operatorname{span}\{\nabla f(x^0), \dots, \nabla f(x^{k-1})\}$,
\begin{equation}\label{eq:lower-convex}
\min_{k=1,\dots,N} f(x^k) - f^* \ge \frac{3 L \norm{x^0 - x^*}^2}{32 (N+1)^2} \, .
\end{equation}

\textbf{(б)} Для любых $L \ge \mu > 0$, $N \ge 1$ существует $L$-гладкая
$\mu$-сильно выпуклая $f$ на $\ell_2$ с точкой минимума $x^*$ такая, что
при $x^0 = 0$ для любого такого метода
\begin{equation}\label{eq:lower-strong}
f(x^N) - f^* \ge \frac{\mu}{2} \Bigl( \frac{\sqrt\varkappa - 1}{\sqrt\varkappa + 1} \Bigr)^{2N} \norm{x^0 - x^*}^2 , \qquad \varkappa = \frac{L}{\mu} \, .
\end{equation}
\end{theorem}

\begin{proof}
Обозначим через $S$ матрицу смежности пути (единицы на двух побочных
диагоналях, в бесконечномерном случае — ограниченный оператор в $\ell_2$,
$\norm{S} \le 2$). Ключевое наблюдение для обоих пунктов — про опор
траектории: градиент рассматриваемых функций имеет вид
$\nabla f(x) = c\,(2I - S)x - c\,e_1$ ($c > 0$ — константа), а если $x$
поддержан на первых $j$ координатах, то $(2I - S)x$ поддержан на первых
$j+1$ (новая координата $j+1$ возникает из члена $-x_j$), поэтому
$\nabla f(x)$ поддержан на первых $j+1$ координатах. Так как $x^0 = 0$,
индукция по $k$ даёт: $x^k$ любого метода из условия теоремы поддержан
на первых $k$ координатах, $x^k_i = 0$ при $i > k$.

\textbf{(а)} Пусть $m = 2N+1$ и
$$
f(x) = F_m(x) = \frac{L}{8}\Bigl[ x_1^2 + \sum_{i=1}^{m-1}(x_i - x_{i+1})^2 + x_m^2 - 2 x_1 \Bigr]
$$
(координаты $i > m$ в $f$ не входят). Раскрыв скобки,
$F_m(x) = \frac{L}{4}\bigl[ \norm{x}^2 - \sum_{i=1}^{m-1} x_i x_{i+1} \bigr] - \frac{L}{4}x_1$,
то есть $\nabla^2 F_m = \frac{L}{4}(2I - S)$. Собственные значения $2I - S$
равны $2 - 2\cos\frac{\pi i}{m+1} \in (0, 4)$, поэтому $F_m$ выпукла и
$L$-гладка. Уравнение минимума $\nabla F_m = 0$ — это $(2I - S)x = e_1$;
обратная к $2I - S$ трёхдиагональна,
$\bigl[(2I-S)^{-1}\bigr]_{ij} = \frac{\min(i,j)\,(m+1-\max(i,j))}{m+1}$
(проверяется прямым умножением: определитель углового минора $k \times k$
матрицы $2I - S$ равен $k+1$), поэтому
$$
x^*_i = \frac{m+1-i}{m+1} \ (i \le m), \qquad x^*_i = 0 \ (i > m) .
$$
Домножив уравнение минимума на $(x^*)^{\mathsf T}$, получаем
$\norm{x^*}^2 - \sum_{i=1}^{m-1} x^*_i x^*_{i+1} = \frac{x^*_1}{2}$, откуда
$$
f^* = -\frac{L}{8}\, x^*_1 = -\frac{L}{8}\, \frac{m}{m+1},
$$
$$
\norm{x^0 - x^*}^2 = \norm{x^*}^2 = \sum_{i=1}^m \frac{(m+1-i)^2}{(m+1)^2} = \frac{m(2m+1)}{6(m+1)} \le \frac{2(N+1)}{3}
$$
(подстановка $m = 2N+1$).

На подпространстве $\{x : x_i = 0,\ i > k\}$ функция $F_m$ совпадает с
$F_k$ той же формы, и по выкладке выше её минимум равен
$-\frac{L}{8}\frac{k}{k+1}$. Поэтому для $k \le N$
$$
f(x^k) \ge -\frac{L}{8}\, \frac{k}{k+1} \ge -\frac{L}{8}\, \frac{N}{N+1},
\qquad
f(x^k) - f^* \ge \frac{L}{8}\Bigl[ \frac{2N+1}{2N+2} - \frac{N}{N+1} \Bigr] = \frac{L}{16(N+1)} \, .
$$
Сравнивая с нормой, получаем $\frac{L}{16(N+1)} \ge \frac{3L \norm{x^0-x^*}^2}{32(N+1)^2}$ — это и есть \eqref{eq:lower-convex}.

\textbf{(б)} Положим $\chi = \varkappa$ и
$$
f(x) = \frac{\mu(\chi-1)}{8}\Bigl[ x_1^2 + \sum_{i=1}^{\infty}(x_i - x_{i+1})^2 - 2 x_1 \Bigr] + \frac{\mu}{2}\norm{x}^2 , \qquad x \in \ell_2 .
$$
Сумма сходится: $\sum_{i\ge1}(x_i-x_{i+1})^2 \le 2\norm{x}^2 + 2\norm{x}^2 = 4\norm{x}^2$, то же верно и для $\scal{Sx}{x}$, так что квадратичная форма в скобках равна $\scal{(2I-S)x}{x} \in [0, 4\norm{x}^2]$ и $f$ корректно определена на $\ell_2$, выпукла, $\mu$-сильно выпукла (член $\frac{\mu}{2}\norm{x}^2$) и $L$-гладка: $\mu I \preceq \nabla^2 f \preceq \mu I + \frac{\mu(\chi-1)}{4}\cdot 4I = \mu\chi\, I$.

Пусть $q = \frac{\sqrt\chi - 1}{\sqrt\chi + 1} \in [0,1)$; тогда
$\chi - 1 = \frac{4q}{(1-q)^2}$ и
\begin{equation}\label{eq:q-ident}
q + \frac1q = \frac{2(\chi+1)}{\chi-1} = 2 + \frac{4}{\chi-1} \, .
\end{equation}
Уравнение минимума $\nabla f(x) = \frac{\mu(\chi-1)}{4}\bigl[(2I - S)x - e_1\bigr] + \mu x = 0$ после деления на $\frac{\mu(\chi-1)}{4}$ и подстановки \eqref{eq:q-ident} принимает вид $T x = e_1$, где $T = (q + q^{-1})I - S$. Вектор $x^*_i = q^i$ принадлежит $\ell_2$ и решает его: при $i \ge 2$ строка равна $q^{i-1}\bigl[(q+q^{-1})q - 1 - q^2\bigr] = 0$, а при $i = 1$: $(q+q^{-1})q - q^2 = 1$. По сильной выпуклости это и есть точка минимума, и
$$
f^* = -\frac{\mu(\chi-1)}{8}\, q , \qquad \norm{x^0 - x^*}^2 = \norm{x^*}^2 = \sum_{i\ge1} q^{2i} = \frac{q^2}{1-q^2} \, .
$$

На подпространстве $\{x : x_i = 0,\ i > k\}$ минимум достигается на векторе
$y_i = \frac{q^i - q^{2k+2-i}}{1 - q^{2k+2}}$ ($i \le k$): для $2 \le i \le k$
строка $T$ обнуляется как выше, а для $i = 1$ получаем
$\bigl[(q+q^{-1})(q - q^{2k+1}) - (q^2 - q^{2k})\bigr] = 1 - q^{2k+2}$
(снова \eqref{eq:q-ident}), то есть $T_k y = e_1$. Значит,
$$
\min_{\{x_i = 0,\, i > k\}} f = -\frac{\mu(\chi-1)}{8}\, y_1 = -\frac{\mu(\chi-1)}{8}\, \frac{q\,(1 - q^{2k})}{1 - q^{2k+2}} \, .
$$
Для $k \le N$ отсюда
$$
f(x^k) - f^* \ge \frac{\mu(\chi-1)}{8}\Bigl[ q - \frac{q\,(1 - q^{2N})}{1 - q^{2N+2}} \Bigr] = \frac{\mu(\chi-1)}{8}\, \frac{q^{2N+1}(1 - q^2)}{1 - q^{2N+2}} \ge \frac{\mu(\chi-1)}{8}\, q^{2N+1}(1 - q^2) .
$$
Осталось сравнить с целевой величиной
$\frac{\mu}{2} q^{2N} \norm{x^0-x^*}^2 = \frac{\mu}{2}\, \frac{q^{2N+2}}{1 - q^2}$:
$$
\frac{\mu(\chi-1)}{8}\, q^{2N+1}(1-q^2) \,\Big/\, \frac{\mu}{2}\, \frac{q^{2N+2}}{1-q^2} = \frac{(\chi-1)(1-q^2)^2}{4q} = \frac{(1-q^2)^2}{(1-q)^2} = (1+q)^2 \ge 1
$$
(в середине — подстановка $\chi - 1 = \frac{4q}{(1-q)^2}$). Значит,
$f(x^k) - f^* \ge \frac{\mu}{2} q^{2N} \norm{x^0-x^*}^2$ — это
\eqref{eq:lower-strong}.
\end{proof}

\begin{remark}
Оценка \eqref{eq:lower-strong} сравнима с верхней \eqref{eq:gd-strong}: по
порядку в показателе экспоненты обе дают $O\bigl(\varkappa \ln(1/\varepsilon)\bigr)$ итераций градиентного спуска. Разрыв между верхней
оценкой $(1 - 1/\varkappa)^N$ и нижней $(1 - 2/\sqrt\varkappa)^{2N}
\approx (1 - 1/\sqrt\varkappa)^{2N}$ закрывается ускоренными методами
разделов \ref{sec:cheb} и \ref{sec:nesterov}: они сходятся со знаменателем
$1 - O(1/\sqrt\varkappa)$ и совпадают с нижней оценкой по порядку. В п.(б)
тот же опор траектории даёт и оценку по аргументу:
$\norm{x^N - x^*}^2 \ge \sum_{i > N} q^{2i} = q^{2N} \norm{x^0 - x^*}^2$.
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
11, пример 4; теоретическая оговорка — [9] (Самарский, Гулин), §6 п. 2: промежуточные
операторы $I - \gamma_j A$ имеют норму, бóльшую единицы, и произведение
переполняется при неудачном порядке сомножителей). Рабочая форма —
трёхчленная рекуррентность по $T_k$, метод чебышёвских полуитераций
(англ. *Chebyshev semi-iterative method*) [2] (Голуб, Ван Лоун):

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
[9] (Самарский, Гулин) $\xi = 1/\varkappa$ это формула \eqref{eq:cheb-iters}.
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
не накапливаются множительно (измерение — пример 2
\texttt{examples.ipynb}: при $\varkappa = 10^2$ за 60 итераций трёхчленная
форма даёт относительную ошибку $8{,}9 \cdot 10^{-6}$ — как теоретическая
граница $q_{60} \approx 1{,}2 \cdot 10^{-5}$ по порядку, — тогда как
множительная форма расходится из-за округления: её относительная ошибка
возрастает до $1{,}3 \cdot 10^{8}$; в вопросе 11, пример 4, показано, что
при $\varkappa \gtrsim 10^2$ множительная форма вообще не достигает
точности $10^{-6}$).
\end{remark}

\begin{remark}[что нужно знать для чебышёвского ускорения]
Границы спектра $\mu_{\min}, \mu_{\max}$. Оценка спектра — отдельная
вычислительная задача; квадратичный метод сопряжённых градиентов даёт тот
же закон $\sqrt\varkappa$ без всякого знания спектра (раздел \ref{sec:cg}),
и в этом его практическое преимущество [2] (Голуб, Ван Лоун).
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
минимум по $\gamma$ на шаге $k$ даёт необходимое условие
$\scal{\nabla \varphi(x^{k+1})}{h^k} = 0$. Для $j \le k$ по индукции
$\nabla\varphi(x^{k+1}) = \nabla\varphi(x^j) + \sum_{i=j}^{k} \gamma_i A h^i$
(телескопическая сумма $\gamma_i A h^i = A(x^{i+1} - x^i)$); скалярно
умножая на $h^j$, получаем
$$
\scal{\nabla\varphi(x^{k+1})}{h^j} = \scal{\nabla\varphi(x^j)}{h^j} + \gamma_j \scal{A h^j}{h^j} + \sum_{i>j} \gamma_i \scal{A h^i}{h^j} .
$$
Сумма равна нулю по $A$-сопряжённости (индукция),
$\scal{\nabla\varphi(x^j)}{h^j} = -\norm{\nabla\varphi(x^j)}^2$ (для
$j \ge 1$ — потому что $h^j = -\nabla\varphi(x^j) + \beta_{j-1} h^{j-1}$ и
$\scal{\nabla\varphi(x^j)}{h^{j-1}} = 0$ по одномерному минимуму шага
$j-1$; для $j = 0$ — сразу из $h^0 = -\nabla\varphi(x^0)$), и по явной
формуле $\gamma_j = -\scal{\nabla\varphi(x^j)}{h^j} / \scal{A h^j}{h^j}$
имеем $\gamma_j \scal{A h^j}{h^j} = \norm{\nabla\varphi(x^j)}^2$. Итак
$\scal{\nabla\varphi(x^{k+1})}{h^j} = 0$ для всех $j \le k$. По (в) индукции
$\nabla\varphi(x^j) \in \operatorname{span}\{h^0,\dots,h^j\}$ для $j \le k$,
поэтому это и есть (а) для пары $(k+1, j)$.

Для (б): $\scal{A h^{k+1}}{h^j} = \scal{-\nabla\varphi(x^{k+1}) + \beta_k h^k}{A h^j}$. При $j < k$: $\scal{h^k}{A h^j} = 0$ по индукции, а
$\scal{\nabla\varphi(x^{k+1})}{A h^j} = 0$, так как
$A h^j = (\nabla\varphi(x^{j+1}) - \nabla\varphi(x^j))/\gamma_j$ — комбинация
прошлых градиентов и (а) применимо. При $j = k$:
$\scal{\nabla\varphi(x^{k+1})}{A h^k} = \norm{\nabla\varphi(x^{k+1})}^2/\gamma_k$, поэтому
$$
\scal{A h^{k+1}}{h^k} = -\tfrac{\norm{\nabla\varphi(x^{k+1})}^2}{\gamma_k} + \beta_k \scal{A h^k}{h^k} = 0,
$$
так как $\gamma_k \scal{A h^k}{h^k} = \norm{\nabla\varphi(x^k)}^2$ (это
вычисление из (а) при $j = k$), а $\beta_k$ по \eqref{eq:cg} — отношение
тех же квадратов норм:
$$
\beta_k = \norm{\nabla\varphi(x^{k+1})}^2 / \norm{\nabla\varphi(x^k)}^2 .
$$

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
\norm{x^k - x^*}_A \le 2\, q^k \norm{x^0 - x^*}_A,
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
$p(\lambda) = T_k(z(\lambda))/T_k(z(0))$ из раздела \ref{sec:cheb}. Его
максимум на $[\mu, L]$ равен $1/T_k(z(0))$, где $z(0) = \frac{L + \mu}{L - \mu}$, а
$$
T_k(z(0)) \ge \tfrac12 \Bigl( z(0) + \sqrt{z(0)^2 - 1} \Bigr)^{\!k} = \tfrac12 \Bigl( \frac{\sqrt\varkappa + 1}{\sqrt\varkappa - 1} \Bigr)^{\!k} .
$$
Подстановка даёт \eqref{eq:cg-rate}.
\end{proof}

\begin{remark}[оптимальность CG]\label{rem:cg-opt}
Оценка \eqref{eq:cg-rate} неулучшаема: любой метод, точки которого лежат
в $x^0 + \operatorname{span}\{\nabla\varphi(x^0), \dots,
\nabla\varphi(x^{k-1})\}$ (все методы первого порядка таковы),
допускает задачу с $\norm{x^k - x^*}_A \ge c\, q^k$ [13] (Поляк). CG совпадает с
нижней оценкой теоремы \ref{thm:lower} по порядку и потому оптимален в
классе методов первого порядка — при том что ему не требуется знание
спектра, в отличие от чебышёвских полуитераций \eqref{eq:cheb-rec} [2] (Голуб, Ван Лоун).
\end{remark}

# Вычислительная сторона {#sec:compute}

Все числа этой таблицы — из теорем этого конспекта; их проверка расчётом
— в `examples.ipynb`.

**Что покупает каждый метод (квадратичная задача, $\varkappa = L/\mu$,
точность $\varepsilon$ по аргументу).**

\begin{center}
\footnotesize
\begin{tabular}{lllll}
\hline
Метод & Итерации & Итерация стоит & Требует & Раздел \\
\hline
Градиентный, $\gamma = 1/L$ & $O(\varkappa \ln \frac1\varepsilon)$ & $1$ градиент & ничего & \ref{sec:gd} \\
Градиентный, $\gamma^* = 2/(L+\mu)$ & $\frac\varkappa2 \ln\frac1\varepsilon$ & $1$ градиент & $\mu$, $L$ & \ref{sec:exact-rate} \\
Наискорейший спуск & как стационарный & $1$ градиент + 1D-поиск & ничего & \ref{sec:problems} \\
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
неравенством Канторовича — [13] (Поляк)). Выбор шага — не то место, где покупается
ускорение.

\textbf{(2) Оценка $L$ на практике.} Константу $L$ часто не знают;
стандартный приём — бэктрекинг: стартовать с грубой оценки и дробить
$\gamma \leftarrow \gamma/2$, пока не выполнится условие Армихо
$f(x - \gamma\nabla f(x)) \le f(x) - \frac{\gamma}{2}\norm{\nabla f(x)}^2$
(оно гарантирует лемму спуска). Все оценки этого конспекта сохраняются
с заменой $L$ на удвоенную найденную константу; автоматическая адаптация
к неизвестной гладкости — содержание универсального градиентного спуска
[37] (Гасников), §5.

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
с углом $\arctan\sqrt\varkappa$ к осям; по неравенству Канторовича [13] (Поляк)
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
Обозначим $S(\lambda) = 1 + \beta - \alpha\lambda$ и
$D(\lambda) = S(\lambda)^2 - 4\beta$; тогда
$\rho_\pm(\lambda) = \bigl(S(\lambda) \pm \sqrt{D(\lambda)}\bigr)/2$. По
\eqref{eq:hb-cond} для всех $\lambda \in [\mu, L]$ выполнено
$0 < \alpha\lambda < 2(1+\beta)$, то есть $\abs{S(\lambda)} < 1 + \beta$.
Два случая.

(1) $D(\lambda) < 0$: корни комплексны и
$\abs{\rho_\pm}^2 = \rho_+\rho_- = \beta < 1$.

(2) $D(\lambda) \ge 0$: корни вещественны и одного знака (произведение
$\beta > 0$), причём
$\max_\pm \abs{\rho_\pm} = \frac{\abs{S} + \sqrt{S^2 - 4\beta}}{2} \le \abs{S} < 1$,
так как $\sqrt{S^2 - 4\beta} \le \abs{S}$.

Итак при \eqref{eq:hb-cond} все $\abs{\rho_\pm(\lambda)} < 1$; знаменатель
$q$ — это $\max_{\lambda \in [\mu, L]} \max_\pm \abs{\rho_\pm(\lambda)}$.

Оптимизация. Определим $\Phi_\beta(s) = \sqrt\beta$ при
$s \le 2\sqrt\beta$ и $\Phi_\beta(s) = \frac{s + \sqrt{s^2 - 4\beta}}{2}$
при $s \ge 2\sqrt\beta$ ($\Phi_\beta$ непрерывна и неубывает); тогда
$\max_\pm \abs{\rho_\pm(\lambda)} = \Phi_\beta\bigl(\abs{S(\lambda)}\bigr)$.
Так как $\abs{S(\lambda)}$ выпукла по $\lambda$, а $\Phi_\beta$ неубывающая
и выпуклая, максимум по $[\mu, L]$ достигается на концах отрезка, и
$$
q(\alpha, \beta) = \max\Bigl\{ \Phi_\beta\bigl(\abs{1 + \beta - \alpha\mu}\bigr),\ \Phi_\beta\bigl(\abs{1 + \beta - \alpha L}\bigr) \Bigr\}.
$$
При фиксированном $\beta$ минимум по $\alpha$ достигается при
выравнивании величин на концах: $\abs{S(\mu)} = \abs{S(L)}$, то есть
$\alpha = \frac{2(1+\beta)}{L+\mu}$ (одна из величин убывает по $\alpha$,
другая возрастает). Тогда $S(\mu) = -S(L) = \frac{(1+\beta)(L-\mu)}{L+\mu}$
и $q = \Phi_\beta\Bigl(\frac{(1+\beta)(L-\mu)}{L+\mu}\Bigr)$. Если
$\frac{(1+\beta)(L-\mu)}{L+\mu} \le 2\sqrt\beta$, то $q = \sqrt\beta$ и
$q$ растёт по $\beta$; если $\ge 2\sqrt\beta$, то, обозначив
$c = \frac{L-\mu}{L+\mu} < 1$ и $s = c(1+\beta)$, имеем
$q = \frac{1}{2}\bigl(s + \sqrt{s^2 - 4\beta}\bigr)$ и
$2q'(\beta) = c + \frac{sc - 2}{\sqrt{s^2 - 4\beta}} < 0$, так как
$(2 - sc)^2 - c^2(s^2 - 4\beta) = 4(1 - c^2) > 0$ — то есть $q$ убывает
по $\beta$. Значит минимум достигается на стыке режимов:
$\frac{(1+\beta)(L-\mu)}{L+\mu} = 2\sqrt\beta$. Решая квадратное
уравнение относительно $\sqrt\beta$ и выбирая корень из $(0,1)$, получаем
$\sqrt\beta = (\sqrt{L} - \sqrt\mu)/(\sqrt{L} + \sqrt\mu)$ и
$\alpha = 2(1+\beta)/(L+\mu) = 4/(\sqrt{L}+\sqrt\mu)^2$ —
это \eqref{eq:hb-opt}; подстановка даёт
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

Метод Нестерова 1983 г. (у [37] (Гасников) — быстрый градиентный метод, БГМ) сходится
по нижним оценкам теоремы \ref{thm:lower} глобально, для всех гладких
выпуклых функций. Запишем его через вспомогательную последовательность
$t_k$ ($t_0 = 0$, $t_1 = 1$):
\begin{equation}\label{eq:nesterov}
\begin{aligned}
x^{k+1} &= y^k - \tfrac1L \nabla f(y^k), \\
y^{k+1} &= x^{k+1} + \frac{t_{k+1} - 1}{t_{k+2}}\bigl( x^{k+1} - x^k \bigr),
\end{aligned}
\qquad y^0 = x^0 ,
\end{equation}
\begin{equation}\label{eq:t-rec}
t_{k+2} = \frac{1 + \sqrt{1 + 4 t_{k+1}^2}}{2}
\qquad \bigl(\Longleftrightarrow\ t_{k+2}^2 - t_{k+2} = t_{k+1}^2\bigr) .
\end{equation}
(при $t_1 = 1$ первый шаг экстраполяции нулевой: $y^1 = x^1$; далее
$t_k \ge \frac{k+1}{2}$ и коэффициент $\frac{t_{k+1}-1}{t_{k+2}}$ растёт
к $1$, а у [37] (Гасников) в той же роли — практическая форма $k/(k+3)$, формула
(1.39)).

\begin{theorem}[сходимость метода Нестерова]\label{thm:nesterov}
Пусть $f$ $L$-гладкая и выпукла, $R = \norm{x^0 - x^*}$. Тогда итерации
\eqref{eq:nesterov} удовлетворяют
\begin{equation}\label{eq:nesterov-rate}
f(x^N) - f^* \le \frac{2 L R^2}{(N+1)^2} .
\end{equation}
Если дополнительно $f$ $\mu$-сильно выпукла, то перезапуск каждые
$m = \lceil 4\sqrt\varkappa \rceil$ итераций (сброс экстраполяции:
$y \leftarrow x$) даёт
\begin{equation}\label{eq:nesterov-strong}
f(x^N) - f^* \le 2 L R^2 \Bigl( \frac12 \Bigr)^{\!2 \lfloor N/m \rfloor} ,
\end{equation}
то есть точность $\varepsilon$ достигается за
$N = O\bigl(\sqrt\varkappa\,\ln \tfrac{L R^2}{\varepsilon}\bigr)$ итераций.
Обе оценки совпадают с нижними оценками теоремы \ref{thm:lower} по порядку.
\end{theorem}

\begin{proof}[Доказательство (схема подобных треугольников) [37] (Гасников)]
Введём $A_k = t_k^2 / L$, $\alpha_{k+1} = A_{k+1} - A_k = t_{k+1} / L$
(равенство $t_{k+1}^2 - t_k^2 = t_{k+1}$ — это соотношение
\eqref{eq:t-rec}) и $\tau_k = \alpha_{k+1} / A_{k+1} = 1 / t_{k+1}$, а
также последовательность $z^{k+1} = z^k - \alpha_{k+1} \nabla f(y^k)$,
$z^0 = x^0$. Точку оценки градиента задаём подобием треугольников:
$y^k = \tau_k z^k + (1 - \tau_k) x^k$. Докажем индукцией
\begin{equation}\label{eq:est-seq}
A_k \bigl(f(x^k) - f^*\bigr) + \tfrac12 \norm{z^k - x^*}^2 \le \tfrac12 \norm{x^0 - x^*}^2 .
\end{equation}
База $k = 0$: $A_0 = t_0^2/L = 0$ и $z^0 = x^0$.

Шаг. Так как $\nabla f(y^k) = L (y^k - x^{k+1})$,
$$
\tfrac12 \norm{z^k - x^*}^2 - \tfrac12 \norm{z^{k+1} - x^*}^2
= \alpha_{k+1} \scal{\nabla f(y^k)}{z^k - x^*} - \tfrac{\alpha_{k+1}^2}{2} \norm{\nabla f(y^k)}^2 .
$$
По выпуклости $f$ (теорема \ref{thm:foc}) с $x = y^k$, $y = x^*$:
$\scal{\nabla f(y^k)}{z^k - x^*} = \scal{\nabla f(y^k)}{z^k - y^k} +
\scal{\nabla f(y^k)}{y^k - x^*} \ge \scal{\nabla f(y^k)}{z^k - y^k} +
f(y^k) - f^*$. По подобию треугольников
$z^k - y^k = \frac{1 - \tau_k}{\tau_k}(y^k - x^k)$, и
$$
\scal{\nabla f(y^k)}{z^k - y^k} = \frac{1 - \tau_k}{\tau_k} \scal{\nabla f(y^k)}{y^k - x^k} \ge \frac{1 - \tau_k}{\tau_k}\bigl( f(y^k) - f(x^k) \bigr),
$$
последнее — выпуклость в форме $f(x^k) \ge f(y^k) +
\scal{\nabla f(y^k)}{x^k - y^k}$. Лемма спуска (лемма
\ref{lem:descent}): $f(x^{k+1}) \le f(y^k) - \frac{1}{2L}\norm{\nabla
f(y^k)}^2$, то есть $-\frac{\alpha_{k+1}^2}{2}\norm{\nabla f(y^k)}^2 \le
-\alpha_{k+1}^2 L \bigl(f(y^k) - f(x^{k+1})\bigr)$. Собирая, получаем
$$
\begin{aligned}
\tfrac12 \norm{z^k - x^*}^2 - \tfrac12 \norm{z^{k+1} - x^*}^2
\ge {} & \alpha_{k+1}\bigl(f(y^k) - f^*\bigr) + \frac{1-\tau_k}{\tau_k}\alpha_{k+1}\bigl(f(y^k) - f(x^k)\bigr) \\
& - \alpha_{k+1}^2 L \bigl(f(y^k) - f(x^{k+1})\bigr) .
\end{aligned}
$$
Здесь $\alpha_{k+1}^2 L = \frac{t_{k+1}^2}{L} = A_{k+1}$ и
$\frac{1-\tau_k}{\tau_k}\alpha_{k+1} = (t_{k+1} - 1)\frac{t_{k+1}}{L} =
\frac{t_{k+1}^2 - t_{k+1}}{L} = \frac{t_k^2}{L} = A_k$ (оба раза —
\eqref{eq:t-rec}), поэтому коэффициент при $f(y^k) - f^*$ сокращается до
нуля и неравенство сворачивается в
$$
\tfrac12 \norm{z^k - x^*}^2 - \tfrac12 \norm{z^{k+1} - x^*}^2
\ge A_{k+1} \bigl(f(x^{k+1}) - f^*\bigr) - A_k \bigl(f(x^k) - f^*\bigr),
$$
что и есть шаг индукции для \eqref{eq:est-seq}.

Сверка с \eqref{eq:nesterov}: исключая
$z^k = \bigl(y^k - (1-\tau_k)x^k\bigr)/\tau_k$ из определения
$y^{k+1} = \tau_{k+1} z^{k+1} + (1-\tau_{k+1})x^{k+1}$ и пользуясь
$L\alpha_{k+1} = 1/\tau_k$ (снова \eqref{eq:t-rec}), получаем
$y^{k+1} = x^{k+1} + \beta_k (x^{k+1} - x^k)$ с
$\beta_k = \frac{\tau_{k+1}}{\tau_k}(1 - \tau_k) = \frac{t_{k+1}-1}{t_{k+2}}$.

Оценка скорости: из \eqref{eq:est-seq} верно
$f(x^N) - f^* \le \frac{L R^2}{2 t_N^2}$, а по индукции
$t_k \ge \frac{k+1}{2}$ (база $t_1 = 1$; шаг: $\sqrt{1 + 4t^2} \ge 2t$,
поэтому $t_{k+2} \ge \frac{1 + 2t_{k+1}}{2} \ge \frac{k+3}{2}$), что и
даёт \eqref{eq:nesterov-rate}.

Сильно выпуклый случай — рестарты. Сильно выпуклая $f$ тем более выпукла,
поэтому \eqref{eq:nesterov-rate} работает: за $r$ итераций из точки на
расстоянии $\rho$ от $x^*$ невязка по функции не превосходит
$\frac{2 L \rho^2}{(r+1)^2}$. По \eqref{eq:strong} с $x = x^*$ новое
расстояние удовлетворяет $\rho_{\text{нов}}^2 \le \frac{2}{\mu} \cdot
\frac{2 L \rho^2}{(r+1)^2} = \frac{4 \varkappa \,\rho^2}{(r+1)^2}$, то есть
при $r + 1 \ge 4\sqrt\varkappa$ квадрат расстояния четвертуется. Значит,
каждая фаза из $m = \lceil 4\sqrt\varkappa \rceil$ итераций даёт
$f - f^* \le \frac{L}{2}\rho^2$, а квадрат расстояния уменьшается вчетверо;
после $j = \lfloor N/m \rfloor$ фаз $f(x^N) - f^* \le \frac{L}{2} R^2
4^{-j}$, что даёт \eqref{eq:nesterov-strong}.
\end{proof}

\begin{remark}
Метод тяжёлого шарика и метод Нестерова отличаются одним словом:
\eqref{eq:hb} вычисляет градиент в текущей точке $x^k$, а
\eqref{eq:nesterov} — в экстраполированной точке $y^k$. Из-за этой
разницы тяжёлый шарик сходится лишь локально (контрпример с разрывным
гессианом у [37] (Гасников), замечание к (1.38)), а метод Нестерова — глобально и по
нижним оценкам. По числу итераций оба метода дают знаменатель
$1 - O(1/\sqrt\varkappa)$, совпадающий с нижней оценкой
\eqref{eq:lower-strong} по порядку. У [37] (Гасников) нет оценки с явной константой
для сильно выпуклого случая («с точностью до числовых множителей»):
вместо рестартов в коде чаще ставят постоянный коэффициент
$\frac{\sqrt\varkappa - 1}{\sqrt\varkappa + 1}$ во второй строке
\eqref{eq:nesterov} — его поведение против теоретической кривой
\eqref{eq:nesterov-strong} измеряет пример 3 \texttt{examples.ipynb}.
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
стандартным шагом [13] (Поляк), гл. 1, §5: оценка
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
итерации уходят на бесконечность [13] (Поляк). Условие \eqref{eq:newton-start}
неработоспособно как практический критерий — $\norm{\nabla f(x^0)}$
неизвестен до счёта; его роль — гарантия сходимости, а практический
выбор $x^0$ — отдельная задача (глобализация: демпфированный шаг
$x^{k+1} = x^k - \gamma_k H_k^{-1}\nabla f(x^k)$ с $\gamma_k$ из
одномерной минимизации или правила Армихо сходится из любой точки
сильно выпуклой задачи — [13] (Поляк), гл. 3, §1).
\end{remark}

\begin{theorem}[Ньютона—Канторовича]\label{thm:kantorovich}
Пусть $f$ дважды непрерывно дифференцируема в шаре
$\Omega = \{x : \norm{x - x^0} \le r\}$, гессиан невырожден в $x^0$,
$\Gamma_0 = [\nabla^2 f(x^0)]^{-1}$ и
\begin{equation}\label{eq:kant-cond}
\norm{\Gamma_0 \nabla f(x^0)} \le \eta, \qquad
\norm{\Gamma_0 \bigl(\nabla^2 f(x) - \nabla^2 f(y)\bigr)} \le K \norm{x - y} \quad (x, y \in \Omega), \qquad
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
Решение единственно в открытом шаре радиуса
$r_1 = \frac{1 + \sqrt{1-2h}}{h}\eta$. Для модифицированного процесса
(гессиан заморожен: $x^{k+1} = x^k - \Gamma_0 \nabla f(x^k)$) при $h < 1/2$
$$
\norm{x'^k - x^*} \le \frac{\eta}{h} \bigl(1 - \sqrt{1 - 2h}\bigr)^{k+1} .
$$
\end{theorem}

\begin{proof}
Сведение к скалярному мажорантному уравнению. Рассмотрим вещественную
функцию
$$
\psi(t) = \frac{K}{2} t^2 - t + \eta
$$
— половину мажоранты $K t^2 - 2t + 2\eta$ из [22] (шаг Ньютона
инвариантен к масштабу, поэтому множитель $1/2$ ни на что не влияет).
У неё $\psi(0) = \eta > 0$, $\psi'(0) = -1$ и, так как $h = K\eta$,
корни $t^* = r_0$, $t^{**} = r_1$. Метод Ньютона для $\psi$ с началом $t_0 = 0$:
$t_{k+1} = t_k - \psi(t_k)/\psi'(t_k)$, сходится возрастающе к $t^*$ при
$h \le 1/2$. Докажем по индукции
\begin{equation}\label{eq:kant-major}
\norm{x^{k+1} - x^k} \le t_{k+1} - t_k
\quad\text{и}\quad
\norm{x^k - x^0} \le t_k .
\end{equation}
База $k = 0$: $\norm{x^1 - x^0} = \norm{\Gamma_0 \nabla f(x^0)} \le \eta =
t_1 - t_0$, а $\norm{x^0 - x^0} = 0 = t_0$.

Шаг: пусть \eqref{eq:kant-major} верно для $k-1$; тогда $x^k \in \Omega$.
По определению шага Ньютона с номером $k-1$ скобка
$\nabla f(x^{k-1}) + \nabla^2 f(x^{k-1})(x^k - x^{k-1})$ равна нулю,
поэтому по теореме Ньютона—Лейбница для $\nabla f$ вдоль отрезка
$[x^{k-1}, x^k] \subset \Omega$
$$
\Gamma_0 \nabla f(x^k) = \Gamma_0 \int_0^1 \bigl[ \nabla^2 f(x^{k-1} + t(x^k - x^{k-1})) - \nabla^2 f(x^{k-1}) \bigr] (x^k - x^{k-1})\, dt .
$$
Оцениваем норму через \eqref{eq:kant-cond}:
\begin{equation}\label{eq:kant-key}
\norm{\Gamma_0 \nabla f(x^k)} \le \int_0^1 K t \norm{x^k - x^{k-1}}^2\, dt = \frac{K}{2} \norm{x^k - x^{k-1}}^2 .
\end{equation}
Далее, $\norm{\Gamma_0 \nabla^2 f(x^k) - I} = \norm{\Gamma_0 \bigl(\nabla^2 f(x^k) - \nabla^2 f(x^0)\bigr)} \le K \norm{x^k - x^0} \le K t_k < 1$ (по индукции $t_k \le t^*$, а $K t^* = 1 - \sqrt{1 - 2h} \le 2h \le 1$, причём для конечных $k$ неравенство строгое), значит $\nabla^2 f(x^k)$ обратим и по лемме о почти единичном операторе $\norm{[\nabla^2 f(x^k)]^{-1} \Gamma_0^{-1}} \le (1 - K t_k)^{-1}$. Следовательно
$$
\norm{x^{k+1} - x^k} \le \norm{[\nabla^2 f(x^k)]^{-1} \Gamma_0^{-1}}\, \norm{\Gamma_0 \nabla f(x^k)} \le \frac{1}{1 - K t_k}\, \frac{K}{2} \norm{x^k - x^{k-1}}^2 .
$$
Для мажорантной последовательности то же вычисление точное: формула
Тейлора для квадратичной $\psi$ с центром в $t_{k-1}$ вместе с определением
шага Ньютона даёт $\psi(t_k) = \frac{K}{2}(t_k - t_{k-1})^2$, и так как
$\psi'(t_k) = K t_k - 1 < 0$,
$$
t_{k+1} - t_k = \frac{\psi(t_k)}{1 - K t_k} = \frac{1}{1 - K t_k}\, \frac{K}{2} (t_k - t_{k-1})^2 .
$$
По индукции $\norm{x^{k+1} - x^k} \le t_{k+1} - t_k$, и тогда
$\norm{x^{k+1} - x^0} \le t_{k+1}$, что замыкает \eqref{eq:kant-major}.
Из \eqref{eq:kant-major}
следует, что $x^k$ фундаментальна и $\norm{x^k - x^0} \le t_k \le t^* = r_0$, то есть $x^k \in \Omega$; предел $x^*$ удовлетворяет $\nabla f(x^*) = 0$ (предельный переход в $\nabla f(x^k) + \nabla^2 f(x^k)(x^{k+1} - x^k) = 0$), а $\norm{x^k - x^*} \le t^* - t_k$. Оценка погрешности:
решая рекуррентность для $t^* - t_k$ (она удовлетворяет той же квадратичной
мажоранте), получаем $t^* - t_k \le \frac{1}{2^k}(2h)^{2^k}\frac{\eta}{h}$,
то есть \eqref{eq:kant-aprior}. Единственность: пусть $\tilde x$ — другое
решение в шаре, $s = \norm{\tilde x - x^0} < r_1$. Из
$0 = \Gamma_0 \nabla f(\tilde x) = \Gamma_0 \nabla f(x^0) + (\tilde x - x^0)
+ \int_0^1 R_t (\tilde x - x^0)\,dt$, где
$R_t = \Gamma_0\bigl(\nabla^2 f\bigl(x^0 + t(\tilde x - x^0)\bigr) -
\nabla^2 f(x^0)\bigr)$, $\norm{R_t} \le K t s$ по \eqref{eq:kant-cond},
следует $s \le \eta + \frac{K}{2} s^2$, то есть $\psi(s) \ge 0$; корни
$\psi$ — $r_0$ и $r_1$, поэтому $s \le r_0$. С другой стороны, для любых
$x, y$ в шаре радиуса $\rho \le r_0$ тот же приём даёт
$\scal{\Gamma_0 \nabla f(x) - \Gamma_0 \nabla f(y)}{x - y}
\ge \bigl(1 - K \rho\bigr) \norm{x - y}^2$; так как
$K r_0 = 1 - \sqrt{1 - 2h} \le 1$, оператор $\Gamma_0 \nabla f$ строго
монотонен на открытом шаре радиуса $r_1$ (для $h = \tfrac12$, когда
$K r_0 = 1$, — на любом меньшем концентрическом шаре). Применяя
неравенство к паре $x^*, \tilde x$, получаем $\tilde x = x^*$.
Модифицированный процесс мажорируется тем же приёмом со скалярной
рекуррентностью $s^{k+1} = \eta + \frac{K}{2} (s^k)^2$, $s^0 = 0$, которая
сходится к $t^*$ со знаменателем $1 - \sqrt{1 - 2h}$; перенос оценки на
$\norm{x'^k - x^*}$ повторяет индукцию \eqref{eq:kant-major} (подробно —
[22] (Канторович, Акилов), гл. XVIII, §1, оценка (34)).
\end{proof}

\begin{remark}
Константа $h$ аффинно инвариантна: при невырожденной линейной замене
переменных $x = C y$ значения $\eta$ и $K$ преобразуются так, что $h$
сохраняется; современная форма теоремы формулируется непосредственно в
аффинно-инвариантных терминах $\norm{[\nabla^2 f(x^0)]^{-1}\nabla f(x^0)}$
и липшицевости $\norm{[\nabla^2 f(x^0)]^{-1}\bigl(\nabla^2 f(x) - \nabla^2 f(y)\bigr)} \le K \norm{x - y}$ [28] (Argyros, Regmi, Argyros, George).
Условие $h \le 1/2$ точно (константа $1/2$ неулучшаема) [22] (Канторович, Акилов).
Запись \eqref{eq:kant-cond} — перенос посылки Канторовича
$\norm{\Gamma_0 P''(x)} \le K$ на случай $P = \nabla f$: там ограничена
третья производная, отсюда следует липшицевость $\nabla^2 f$, которая и
используется в доказательстве; сама теорема в [22] (Канторович, Акилов) формулируется для
произвольного нелинейного оператора $P$.
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
Лемма о градиентном отображении: для любого $z \in Q$
\begin{equation}\label{eq:grad-map}
f(x^{k+1}) \le f(z) + \frac{L}{2}\bigl( \norm{x^k - z}^2 - \norm{x^{k+1} - z}^2 \bigr) .
\end{equation}
Действительно, подставляя $y^k = x^k - \frac1L \nabla f(x^k)$ в критерий
проекции \eqref{eq:proj-crit}, получаем для любого $z \in Q$
$$
\scal{\nabla f(x^k)}{x^{k+1} - z} \le L \scal{x^k - x^{k+1}}{x^{k+1} - z} .
$$
Складываем оценку снизу $f(x^k) \le f(z) + \scal{\nabla f(x^k)}{x^k - z}$
(выпуклость, теорема \ref{thm:foc}) и оценку сверху
$f(x^{k+1}) \le f(x^k) + \scal{\nabla f(x^k)}{x^{k+1} - x^k} + \frac{L}{2}\norm{x^{k+1} - x^k}^2$
(неравенство \eqref{eq:upper-par}):
$$
f(x^{k+1}) \le f(z) + \scal{\nabla f(x^k)}{x^{k+1} - z} + \frac{L}{2}\norm{x^{k+1} - x^k}^2 ,
$$
$$
f(x^{k+1}) \le f(z) + \frac{L}{2}\Bigl( 2\scal{x^k - x^{k+1}}{x^{k+1} - z} + \norm{x^{k+1} - x^k}^2 \Bigr) ,
$$
а выражение в скобках равно $\norm{x^k - z}^2 - \norm{x^{k+1} - z}^2$ по
тождеству $2\scal{a}{b} + \norm{a}^2 = \norm{a+b}^2 - \norm{b}^2$.

При $z = x^*$ лемма даёт телескоп
$\frac{2}{L}\bigl(f(x^{k+1}) - f^*\bigr) \le \norm{x^k - x^*}^2 - \norm{x^{k+1} - x^*}^2$;
суммирование по $k = 0, \dots, N-1$ даёт
$$
\frac{2}{L} \sum_{k=1}^{N} \bigl(f(x^k) - f^*\bigr) \le R^2 .
$$
При $z = x^k \in Q$ лемма даёт $f(x^{k+1}) \le f(x^k)$, то есть $f(x^k)$
монотонно убывает, и потому
$f(x^N) - f^* \le \frac1N \sum_{k=1}^N \bigl(f(x^k) - f^*\bigr) \le \frac{L R^2}{2N}$.
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
L_0 R / \sqrt{N}$, и оценка неулучшаема [37] (Гасников), §2. Медленность не
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
обратная задача теплопроводности с обратным временем [14] (Кабанихин), гл. 8: прямая
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
неулучшаема [37] (Гасников), приложение; для сильно выпуклых — линейная сходимость до
$O(D/(L\mu))$-окрестности. Дисперсия снижается минибатчингом; методы
редукции дисперсии (SVRG, SAGA) убирают плату $O(1/\sqrt N)$. Это рабочая
лошадка вопроса 02.

\textbf{Зеркальный спуск} (англ. *mirror descent*) — замена евклидовой
проекции на «проекцию» по дивергенции Брэгмана, определяемой выбором
прокс-функции; для задач на симплексе заменяет $O(\sqrt{\ln n})$ штраф за
размерность. Объявлен в заголовке колоды лекции прошлого года; вопрос 10
его не требует, см. [37] (Гасников), §2.

\textbf{Универсальный градиентный спуск} [37] (Гасников), §5 — метод, автоматически
настраивающийся на неизвестную степень гладкости $\nu \in [0,1]$ через
бэктрекинг по $L$; связка с правилом выбора шага раздела \ref{sec:compute}.

\textbf{Квазиньютоновские методы} (BFGS и др.) — итерации
$x^{k+1} = x^k - \gamma_k H_k \nabla f(x^k)$ с матрицами $H_k$,
обновляемыми по разностям градиентов так, чтобы выполнялось
квазиньютоновское условие $H_{k+1} y^k = p^k$; на квадратичной задаче
сходятся за $n$ шагов, на общей — сверхлинейно [12] (Сухарев, Тимохов, Федоров), гл. 5, §2; [13] (Поляк),
гл. 3, §3.

\textbf{Помехи.} При зашумлённом градиенте нижняя оценка асимптотически
$O(1/k)$ для выпуклых и геометрическая для сильно выпуклых, и градиентный
метод с убывающим шагом асимптотически оптимален [13] (Поляк), гл. 4, §5 —
вопрос для функционала качества в условиях шумных измерений.

# Источники {#sec:sources}

Основные источники вопроса. Локаторы (страницы) даны в README этого вопроса.

\textbf{[12]} (Сухарев, Тимохов, Федоров) — условия оптимальности первого и второго порядка, выпуклость
и сильная выпуклость, субдифференциал, метод сопряжённых направлений и
квазиньютоновские схемы. Теорема о существовании и единственности
минимума и условия второго порядка — по этой книге.

\textbf{[13]} (Поляк) — теория скорости сходимости градиентного спуска (точная
оценка и оптимальный шаг), локальная квадратичная сходимость Ньютона с
контрпримерами, наискорейший спуск и неравенство Канторовича, метод
тяжёлого шарика, сопряжённые градиенты и их оптимальность, нижние оценки
при помехах. Теоремы \ref{thm:exact-rate}, \ref{thm:hb},
\ref{thm:newton-local} и разбор наискорейшего спуска — по этой книге.

\textbf{[37]} (Гасников) — градиентный спуск с постоянным шагом и оценки в трёх
режимах (теорема \ref{thm:gd-conv}), нижние оценки (теорема
\ref{thm:lower}), метод Нестерова и метод подобных треугольников
(теорема \ref{thm:nesterov}), субградиентный метод и стохастический
спуск, проекция градиента, бэктрекинг. Материал лекции курса прошлого
года.

\textbf{[02]} (Голуб, Ван Лоун) — чебышёвские полуитерации и устойчивая трёхчленная
рекуррентность (теорема \ref{thm:cheb}), метод сопряжённых градиентов и
оценка в энергетической норме.

\textbf{[09]} (Самарский, Гулин) — чебышёвский набор итерационных параметров: оценка $q_n$
и число итераций $n_0(\varepsilon)$ (следствие теоремы
\ref{thm:cheb}), неустойчивость множительной формы (замечание
\ref{rem:cheb-stable}).

\textbf{[22]} (Канторович, Акилов) — теорема Ньютона—Канторовича с полным доказательством
(теорема \ref{thm:kantorovich}), точность константы $1/2$.

\textbf{[28]} (Argyros, Regmi, Argyros, George) — аффинно-инвариантная форма теоремы Канторовича и таблица
оценок погрешности (замечание к теореме \ref{thm:kantorovich}).

\textbf{[14]} (Кабанихин) — градиент функционала невязки через сопряжённую задачу
(теорема \ref{thm:adjoint} и пример обратной задачи теплопроводности).
