---
title: "Вопрос 07. Преобразования Фурье, Лапласа, Хаара"
subtitle: "Конспект теории с полными доказательствами"
author: "Крохалев Е. М."
date: "2026"
---

\part{Основная часть}

# Постановка и мотивация {#sec:intro}

Математическая модель почти всегда живёт в бесконечномерном пространстве
функций, а считать приходится с конечным набором чисел. Между этими двумя
мирами стоят интегральные преобразования: они переводят функцию в функцию
другой переменной, в которой искомая информация читается напрямую. Спектр
сигнала виден из преобразования Фурье; решение линейного дифференциального
уравнения с постоянными коэффициентами превращается в преобразовании Лапласа
в алгебраическое уравнение; положение излома сигнала — из разложения по
системе Хаара. Все три конструкции укладываются в одну рамку.

\begin{definition}\label{def:integral-transform}
Пусть $(T,\mu)$ — пространство с мерой, $K(s,t)$ — функция двух переменных,
называемая \textbf{ядром}. \textbf{Интегральным преобразованием с ядром $K$ по
мере $\mu$} называется отображение, ставящее функции $f$ функцию
\begin{equation}\label{eq:kernel-transform}
F(s) = \int_T K(s,t)\, f(t)\, d\mu(t)
\end{equation}
при условии, что интеграл существует. Функция $f$ называется
\textbf{оригиналом}, функция $F$ — \textbf{образом} (или \textbf{изображением}).
\end{definition}

Значение преобразования определяется выбором ядра. Комплексные экспоненты
$e^{-i\omega t}$ — собственные функции всех операторов свёртки, поэтому ядро
Фурье диагонализирует линейные инвариантные во времени системы [34]. Экспонента
$e^{-st}$ с комплексным $s$ той же диагонализацией обладает для сигналов,
начинающихся в нуле, и вдобавок сводит дифференцирование к умножению.
Система Хаара идёт другим путём: это счётное семейство кусочно-постоянных
функций, и «преобразование» — не интеграл с ядром, а разложение по
ортонормированному базису, в котором коэффициенты локализованы во времени.

\begin{center}\small
\begin{tabular}{p{28mm}p{36mm}p{34mm}p{46mm}}
\hline
Преобразование & Ядро & Мера по $t$ & Область применимости \\
\hline
Фурье & $e^{-i\omega t}$ & $dt$ на $\R$ & $L_1(\R)$, продолжение на $L_2(\R)$ \\
Лапласа & $e^{-st}$, $s = \sigma + i\omega \in \C$ & $dt$ на $[0,\infty)$ &
  оригиналы, $\operatorname{Re} s > a_0$ \\
Хаара & нет: семейство $\{\psi_{j,k}\}$ & считающая по $(j,k) \in \Z^2$ &
  $L_2(\R)$, разложение по ортонормированному базису \\
\hline
\end{tabular}
\end{center}

Через весь конспект проходит один сигнал — экспоненциальный импульс
\begin{equation}\label{eq:u}
u(t) = e^{-t}\, \mathbb I_{[0,\infty)}(t),
\end{equation}
проведённый через все три теории. Его преобразование Фурье вычисляется
напрямую и служит проверкой теорем обращения и Парсеваля (задача
\ref{prob:fourier-images}); его изображение по Лапласу $U(s) = 1/(s+1)$
решает через теорему о свёртке задачу о колебательном контуре (задача
\ref{prob:ode}); его коэффициенты по системе Хаара выписываются в явном виде
и показывают характерную скорость затухания (задача \ref{prob:haar-coeffs}).
«Изломанная» версия того же сигнала $t\,e^{-t}$ разделяет Фурье и Хаар на
частотно-временной плоскости: ряд Фурье ведёт себя у излома заметно хуже,
чем ряд Хаара (сравнение — в разделе \ref{sec:haar} и в примерах ноутбука).

Прикладные источники рамки задают сами [42]: колебательный контур с
сопротивлением, ёмкостью и индуктивностью — это задача Коши для линейного
уравнения второго порядка, которую глава 10 решает целиком в изображениях
Лапласа. В обратных задачах та же связь работает в обе стороны: показания
линейного прибора связаны с входным сигналом уравнением свёртки первого
рода, и переход к изображениям по Лапласу превращает его в умножение — но
восстановление оригинала по образу на практике оказывается некорректной
задачей [14].

Вопрос опирается на вопрос 09 (метрические и гильбертовы пространства):
пространства $L_1$, $L_2$, их полнота, ортонормированные системы и критерий
их полноты берутся там и здесь только цитируются. Логический мост вперёд —
вопрос 05 (методы вейвлет-анализа): система Хаара — простейший ортогональный
вейвлет, а пирамида «сглаживание–детализация» — зачаток кратномасштабного
анализа, который и есть предмет вопроса 05.

# Что берётся готовым {#sec:given}

Доказательства этого конспекта элементарны, но опираются на интеграл Лебега
и теорию гильбертовых пространств. Список того, что не доказывается:

1. **Предельные переходы под знаком интеграла.** Теорема Лебега о
   мажорированной сходимости, теорема Леви о монотонной сходимости, лемма
   Фату и теорема Фубини для интеграла Лебега. Все применения ниже имеют
   суммируемую мажоранту, и мы не оговариваем это каждый раз.
2. **Гауссов интеграл** $\displaystyle\int_{-\infty}^{\infty} e^{-x^2}\,dx
   = \sqrt{\pi}$ — единственный «взрослый» интеграл, встречающийся в
   выкладках.
3. **Полнота пространств $L_p$ и непрерывность сдвига.** $L_p(\R)$, $1\le
   p<\infty$, полно; $\norm{f(\cdot - h) - f}_p \to 0$ при $h\to 0$; непрерывные
   функции с компактным носителем плотны в $L_p$. Отсюда же: сходимость
   аппроксимативной единицы $f * \theta_\varepsilon \to f$ в $L_p$.
4. **Теория ортонормированных систем** из вопроса 09: неравенство Бесселя,
   критерий полности системы через равенство Парсеваля, теорема
   Рисса—Фишера о сумме ряда с суммируемым квадратом коэффициентов, а также
   единственность продолжения линейного ограниченного оператора с плотного
   подмножества (для продолжения преобразования Фурье с $L_1\cap L_2$ на
   $L_2$).
5. **Ряды Фурье периодических функций.** Если $T$-периодическая непрерывная
   функция имеет абсолютно суммируемые коэффициенты Фурье, её ряд Фурье
   сходится к ней равномерно.
6. **Теорема о дифференцировании интеграла по комплексному параметру.**
   Если $\Phi(s) = \int_0^\infty \varphi(t,s)\,dt$, функции $\varphi$ и
   $\partial\varphi/\partial s$ непрерывны по $s$ и локально мажорируются
   суммируемой функцией от $t$, то $\Phi$ голоморфна и $\Phi'(s) = \int
   \partial\varphi(t,s)/\partial s\,dt$. Для преобразования Лапласа в
   теореме \ref{thm:laplace-conv} это проверяется вручную.
7. **Формула Ньютона—Лейбница и интегрирование по частям** для функций,
   кусочно гладких на каждом конечном отрезке.
8. **Факт из теории меры:** если $f \in L_1^{\mathrm{loc}}$ и
   $\int_a^b f\,dt = 0$ для всех интервалов $(a,b)$, то $f = 0$ почти всюду.

Всё остальное — определения, формулировки и доказательства — выводится
внутри конспекта.

# Определения и обозначения {#sec:defs}

\begin{notation}
Индикатор множества $E$ пишется $\mathbb I_E$; скалярное произведение в
$L_2$ и в $\C^N$ линейно по первому аргументу: $\scal{f}{g} = \int f\,
\overline{g}\,dt$. Корень из единицы $\varepsilon_N = e^{2\pi i / N}$.
\end{notation}

\begin{definition}\label{def:lp}
Пространство $L_1(\R)$ — классы измеримых функций с конечной нормой
$\norm{f}_1 = \int_{-\infty}^{\infty} |f(t)|\,dt$; пространство $L_2(\R)$ —
с конечной нормой $\norm{f}_2 = \bigl(\int |f(t)|^2\,dt\bigr)^{1/2}$ и
скалярным произведением $\scal{f}{g} = \int f\,\overline{g}\,dt$. Равенство
понимается почти всюду.
\end{definition}

\begin{definition}\label{def:convolution}
\textbf{Свёрткой} функций $f, g \in L_1(\R)$ называется функция
\begin{equation}\label{eq:convolution}
(f * g)(t) = \int_{-\infty}^{\infty} f(\tau)\, g(t - \tau)\, d\tau .
\end{equation}
Интеграл существует почти для всех $t$, и $f*g \in L_1(\R)$
(лемма \ref{lem:conv-l1} ниже).
\end{definition}

\begin{definition}\label{def:fourier}
\textbf{Преобразованием Фурье} (в единичной нормировке) функции $f \in
L_1(\R)$ называется функция
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{\infty}
f(t)\, e^{-i\omega t}\, dt , \qquad \omega \in \R .
\end{equation}
\textbf{Обратным преобразованием} называется оператор
\begin{equation}\label{eq:fourier-inverse}
\check g(t) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{\infty}
g(\omega)\, e^{i\omega t}\, d\omega
\end{equation}
(для $g \in L_1(\R)$ по частоте).
\end{definition}

\begin{definition}\label{def:original}
\textbf{Оригиналом} называется функция $f$ действительного аргумента со
значениями в $\C$, удовлетворяющая трём условиям:

1. $f$ интегрируема на любом конечном интервале (локально интегрируема);
2. $f(t) = 0$ при $t < 0$;
3. существуют постоянные $M > 0$ и $a$ такие, что $|f(t)| \le M e^{a t}$
   при $t \ge 0$.

Нижняя грань $a_0$ всех показателей $a$, для которых выполнено условие 3,
называется \textbf{показателем роста} функции $f$.
\end{definition}

\begin{definition}\label{def:laplace}
\textbf{Изображением по Лапласу} (преобразованием Лапласа) оригинала $f$
называется функция комплексной переменной $s$,
\begin{equation}\label{eq:laplace}
F(s) = \int_0^{\infty} f(t)\, e^{-s t}\, dt , \qquad
s = \sigma + i\omega .
\end{equation}
Соответствие оригинала и изображения записывается $f(t) \dashv F(s)$;
две функции с одним изображением совпадают в точках непрерывности (теорема
\ref{thm:laplace-uniq}).
\end{definition}

\begin{definition}\label{def:haar}
\textbf{Функцией Хаара} называется
\begin{equation}\label{eq:haar-function}
\psi(t) = \mathbb I_{[0,\tfrac12)}(t) - \mathbb I_{[\tfrac12, 1)}(t) .
\end{equation}
Её сжатия и сдвиги $\psi_{j,k}(t) = 2^{j/2}\,\psi\bigl(2^j t - k\bigr)$,
$j, k \in \Z$, имеют носитель
\begin{equation}\label{eq:dyadic}
I_{j,k} = \bigl[k\,2^{-j},\ (k+1)\,2^{-j}\bigr)
\end{equation}
(мелкие масштабы отвечают $j \to +\infty$), значения $\pm 2^{j/2}$ на половинах
$I_{j,k}$ и нуль вне $I_{j,k}$. Интервалы \eqref{eq:dyadic} называются
\textbf{диадическими}. Система $\{\psi_{j,k}\}_{j,k\in\Z}$ называется
\textbf{системой Хаара}.
\end{definition}

\begin{definition}\label{def:signal-zn}
\textbf{Сигналом на $\Z_N$} называется $N$-периодическая последовательность
$x\colon \Z \to \C$; все сигналы образуют пространство $\C^N$ со скалярным
произведением $\scal{x}{y} = \sum_{j=0}^{N-1} x(j)\,\overline{y(j)}$.
\textbf{Дискретным преобразованием Фурье} (ДПФ) называется сигнал $X =
\mathcal F_N x$ с отсчётами
\begin{equation}\label{eq:dft}
X(k) = \sum_{j=0}^{N-1} x(j)\, \varepsilon_N^{-k j}, \qquad
\varepsilon_N = e^{2\pi i / N}.
\end{equation}
\textbf{Циклической свёрткой} сигналов $x$ и $y$ называется сигнал
\begin{equation}\label{eq:cyclic-conv}
(x * y)(j) = \sum_{k=0}^{N-1} x(k)\, y(j - k).
\end{equation}
\end{definition}

\begin{definition}\label{def:pyramid}
Пусть $s_0 = (s_{0,0}, \dots, s_{0,N-1}) \in \C^N$, $N = 2^s$.
\textbf{Пирамидой средних и деталей} (в единичной нормировке) называется
семейство $\{s_\nu, d_\nu\}_{\nu=1}^{s}$ с отсчётами
\begin{equation}\label{eq:pyramid}
s_{\nu,k} = \tfrac{1}{\sqrt2}\,\bigl(s_{\nu-1,\,2k} + s_{\nu-1,\,2k+1}\bigr),
\qquad
d_{\nu,k} = \tfrac{1}{\sqrt2}\,\bigl(s_{\nu-1,\,2k} - s_{\nu-1,\,2k+1}\bigr),
\end{equation}
где $\nu = 1, \dots, s$ — номер уровня, $k \in 0 : N_\nu - 1$,
$N_\nu = N/2^\nu$.
\end{definition}

\begin{definition}\label{def:bandlimited}
Сигнал $f \in L_2(\R)$ называется \textbf{полосовым с ограниченным спектром},
если $\supp \hat f \subset [-\Omega, \Omega]$ для некоторого $\Omega > 0$.
Для сигнала на $\Z_N$ с чётным $N$ частотой Найквиста называется $N/2$ — верхняя
граница частот, различимых по спектру: значения $X(k)$ и $X(N-k)$
описывают одну и ту же гармонику.
\end{definition}

# Преобразование Фурье {#sec:fourier}

Первые свойства преобразования \eqref{eq:fourier} получаются непосредственно
из определения. Все они собраны в одну теорему; единственный шаг, требующий
отдельной леммы, — стремление образа к нулю (лемма Римана—Лебега).

\begin{lemma}\label{lem:indicator-transform}
Образ индикатора интервала:
\begin{equation}\label{eq:indicator-transform}
\widehat{\mathbb I_{[a,b]}}(\omega) =
\frac{1}{\sqrt{2\pi}}\cdot\frac{e^{-i\omega b} - e^{-i\omega a}}{-i\omega},
\qquad
\bigl|\widehat{\mathbb I_{[a,b]}}(\omega)\bigr|
\le \frac{2}{\sqrt{2\pi}\,|\omega|}\ (\omega \ne 0).
\end{equation}
\end{lemma}

\begin{proof}
Непосредственное интегрирование:
$\int_a^b e^{-i\omega t}\,dt = \frac{e^{-i\omega b} - e^{-i\omega a}}{-i\omega}$;
модуль числителя не превосходит 2.
\end{proof}

\begin{lemma}[Римана—Лебега]\label{lem:riemann-lebesgue}
Если $f \in L_1(\R)$, то $\hat f$ непрерывна на $\R$ и
\begin{equation}\label{eq:riemann-lebesgue}
\hat f(\omega) \to 0 \qquad \text{при } |\omega| \to \infty .
\end{equation}
\end{lemma}

\begin{proof}
Непрерывность. При $h \to 0$ имеем
$$
|\hat f(\omega + h) - \hat f(\omega)|
\le \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{\infty}
|f(t)|\, |e^{-i h t} - 1|\, dt .
$$
Подынтегральная функция не превосходит $2|f(t)| \in L_1$ и стремится к нулю
при каждом $t$; по теореме Лебега о мажорированной сходимости интеграл
стремится к нулю.

Стремление к нулю. Шаг 1: для $f = \mathbb I_{[a,b]}$
утверждение — это лемма \ref{lem:indicator-transform}, по линейности оно
верно для любой ступенчатой функции. Шаг 2: ступенчатые функции плотны в
$L_1$: для любого $\varepsilon > 0$ найдётся ступенчатая $\varphi$ с
$\norm{f - \varphi}_1 < \varepsilon$. Тогда
$$
|\hat f(\omega)| \le |\widehat{f - \varphi}(\omega)| + |\hat\varphi(\omega)|
\le \frac{\varepsilon}{\sqrt{2\pi}} + |\hat\varphi(\omega)| .
$$
Устремив $|\omega| \to \infty$ (при фиксированном $\varphi$), получаем
$\limsup_{|\omega|\to\infty} |\hat f(\omega)| \le
\varepsilon/\sqrt{2\pi}$; в силу произвольности $\varepsilon$ предел равен
нулю.
\end{proof}

\begin{theorem}[Свойства преобразования Фурье на $L_1$]\label{thm:ft-props}
Пусть $f, g \in L_1(\R)$, $\alpha, \beta, \tau \in \R$, $\lambda > 0$.
Тогда:

\begin{enumerate}
\item \textbf{линейность:} $\widehat{\alpha f + \beta g} =
\alpha \hat f + \beta \hat g$;
\item \textbf{сдвиг:} $\widehat{f(\cdot - \tau)}(\omega) =
e^{-i\omega\tau}\, \hat f(\omega)$;
\item \textbf{модуляция:} $\widehat{e^{i\lambda\cdot} f}(\omega) =
\hat f(\omega - \lambda)$;
\item \textbf{растяжение:} $\widehat{f(\lambda\cdot)}(\omega) =
\frac{1}{\lambda}\, \hat f\bigl(\frac{\omega}{\lambda}\bigr)$;
\item \textbf{дифференцирование оригинала:} если $f$ непрерывно
дифференцируема, $f' \in L_1(\R)$ и $f(t) \to 0$ при $|t| \to \infty$, то
$\widehat{f'}(\omega) = i\omega\, \hat f(\omega)$;
\item \textbf{дифференцирование образа:} если $t\,f(t) \in L_1(\R)$, то
$\hat f \in C^1(\R)$ и $\widehat{t f}(\omega) = i\, \frac{d}{d\omega}\hat
f(\omega)$;
\item \textbf{комплексное сопряжение:}
$\widehat{\overline{f}}(\omega) = \overline{\hat f(-\omega)}$; в частности,
для вещественной $f$ образ эрмитов: $\hat f(-\omega) =
\overline{\hat f(\omega)}$;
\item \textbf{мажоранта и непрерывность:} $|\hat f(\omega)| \le
\frac{1}{\sqrt{2\pi}} \norm{f}_1$, образ непрерывен и $\hat f(\omega) \to 0$
при $|\omega| \to \infty$ (лемма Римана—Лебега).
\end{enumerate}
\end{theorem}

\begin{proof}
Пункты 1–4 и 7 — замены переменных под знаком интеграла. Например, в пункте
4 после подстановки $u = \lambda t$ получаем
$\frac{1}{\sqrt{2\pi}} \int f(u) e^{-i\omega u/\lambda}\, \frac{du}{\lambda}$;
пункт 2: подстановка $u = t - \tau$ выносит за скобку множитель
$e^{-i\omega\tau}$.

Пункт 5. Интегрируем по частям на $[-A, A]$ с последующим предельным
переходом:
$$
\int_{-A}^{A} f(t)\, e^{-i\omega t} dt
= \frac{f(A)e^{-i\omega A} - f(-A)e^{i\omega A}}{-i\omega}
+ \frac{1}{i\omega} \int_{-A}^{A} f'(t)\, e^{-i\omega t}\, dt .
$$
Граничные члены стремятся к нулю по условию $f(t)\to 0$, а интеграл сходится
к $\sqrt{2\pi}\,\widehat{f'}(\omega)$; домножая на $-i\omega$ и делая
$\frac{1}{\sqrt{2\pi}}$, получаем $\widehat{f'}(\omega) = i\omega\hat
f(\omega)$.

Пункт 6. Рассмотрим разностное отношение
$$
\frac{\hat f(\omega + h) - \hat f(\omega)}{h}
= \frac{1}{\sqrt{2\pi}} \int f(t)\, e^{-i\omega t}\,
\frac{e^{-i h t} - 1}{h}\, dt .
$$
Так как $|e^{-ix} - 1| \le |x|$ для действительного $x$, подынтегральное
выражение мажорируется $|f(t)|\,|t| \in L_1$ и стремится к $-i t f(t)
e^{-i\omega t}$ при $h \to 0$; по теореме Лебега
$\frac{d}{d\omega}\hat f(\omega) = \frac{-i}{\sqrt{2\pi}} \int t f(t)
e^{-i\omega t}dt = -i\,\widehat{t f}(\omega)$, что и есть пункт 6.
Непрерывность производной следует из уже доказанной непрерывности образа
функции $t f(t)$.

Пункт 8. Оценка прямо из определения; остальное — лемма
\ref{lem:riemann-lebesgue}.
\end{proof}

\begin{remark}
Условия пунктов 5 и 6 существенны. Дифференцирование оригинала требует
интегрируемости $f'$ и убывания $f$ на бесконечности: без них интегрирование
по частям оставляет неопределённый граничный член. В пункте 6 условие $t
f(t) \in L_1$ — единственное, что позволяет проносить производную под
знак интеграла; оно же даёт лишнюю гладкость образа, а это предвещает
принцип неопределённости (раздел \ref{sec:heisenberg}): локализация оригинала
покупается потерей гладкости образа.
\end{remark}

Обращение преобразования Фурье — первая теорема, для которой нужна
техника, а не только замена переменных. Доказательство [34] (теорема 2.1)
идёт через гауссову регуляризацию; ниже оно воспроизведено в единичной
нормировке.

\begin{theorem}[обращения Фурье]\label{thm:ft-inverse}
Пусть $f \in L_1(\R)$ и $\hat f \in L_1(\R)$. Тогда $f$ совпадает почти
всюду с непрерывной функцией, и в каждой точке $t$ непрерывности $f$
\begin{equation}\label{eq:ft-inverse}
f(t) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{\infty}
\hat f(\omega)\, e^{i\omega t}\, d\omega .
\end{equation}
\end{theorem}

\begin{proof}
Рассмотрим гауссово ядро
$\theta_\varepsilon(x) = \frac{1}{\sqrt{\pi}\,\varepsilon}\,
e^{-x^2/\varepsilon^2}$ с параметром $\varepsilon > 0$. Оно удовлетворяет
условиям аппроксимативной единицы:
\begin{equation}\label{eq:gauss-kernel}
\theta_\varepsilon \ge 0, \qquad
\int_{-\infty}^{\infty} \theta_\varepsilon(x)\,dx = 1, \qquad
\int_{|x| \ge \delta} \theta_\varepsilon(x)\,dx \to 0
\quad (\varepsilon \to 0)\ \ \forall \delta > 0 ;
\end{equation}
первое непосредственно, второе следует из гауссова интеграла подстановкой
$u = x/\varepsilon$, третье — из оценки
$\int_\delta^\infty \theta_\varepsilon(x)\,dx \le
\frac{1}{\sqrt{\pi}\,\varepsilon} \int_\delta^\infty
\frac{x}{\delta} e^{-x^2/\varepsilon^2} dx =
\frac{\varepsilon}{2\sqrt{\pi}\,\delta}\, e^{-\delta^2/\varepsilon^2}$.

Введём регуляризованный интеграл обращения
$$
I_\varepsilon(t) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{\infty}
\hat f(\omega)\, e^{-\varepsilon^2\omega^2/4}\, e^{i\omega t}\, d\omega
$$
и вычислим его двумя способами. Так как $|\hat f(\omega)
e^{-\varepsilon^2\omega^2/4} e^{i\omega t}| \le |\hat f(\omega)|$, по
теореме Лебега
\begin{equation}\label{eq:inv-first}
I_\varepsilon(t) \xrightarrow[\varepsilon \to 0]{}
\frac{1}{\sqrt{2\pi}} \int_{-\infty}^{\infty}
\hat f(\omega)\, e^{i\omega t}\, d\omega =: g(t),
\end{equation}
причём $g$ непрерывна (та же мажорантная оценка для разности
$g(t+h)-g(t)$).

С другой стороны, подставляя $\hat f$ и меняя порядок интегрирования
(Фубини применим: $\frac{1}{2\pi}\iint |f(u)| e^{-\varepsilon^2\omega^2/4}
\,du\,d\omega = \frac{1}{2\pi}\norm{f}_1 \cdot \frac{2\sqrt\pi}{\varepsilon}
< \infty$), получаем
$$
I_\varepsilon(t) = \frac{1}{2\pi} \int_{-\infty}^{\infty} f(u)
\Bigl( \int_{-\infty}^{\infty}
e^{-\varepsilon^2\omega^2/4}\, e^{i\omega(t-u)}\, d\omega \Bigr) du .
$$
Внутренний интеграл вычисляется элементарно. Обозначим его $\Phi(x)$ при
$x = t - u$; дифференцирование по параметру (мажоранта
$\omega e^{-\varepsilon^2\omega^2/4}$ суммируема) даёт
$\Phi'(x) = \int i\omega e^{-\varepsilon^2\omega^2/4} e^{i\omega x}d\omega$;
так как $\frac{d}{d\omega}e^{-\varepsilon^2\omega^2/4} =
-\frac{\varepsilon^2}{2}\,\omega\, e^{-\varepsilon^2\omega^2/4}$, интегрирование
по частям по $\omega$ (граничные члены равны нулю) даёт
$\Phi'(x) = -\frac{2x}{\varepsilon^2}\,\Phi(x)$. Следовательно,
$\Phi(x) = \Phi(0)\,e^{-x^2/\varepsilon^2}$, а
$\Phi(0) = \int e^{-\varepsilon^2\omega^2/4}d\omega =
\frac{2\sqrt\pi}{\varepsilon}$ по гауссову интегралу. Итого
$$
I_\varepsilon(t) = \int_{-\infty}^{\infty} f(u)\,
\frac{1}{\sqrt{\pi}\,\varepsilon}\, e^{-(t-u)^2/\varepsilon^2}\, du
= (f * \theta_\varepsilon)(t).
$$

Покажем, что $(f*\theta_\varepsilon)(t) \to f(t)$ в каждой точке
непрерывности $f$. По второму условию \eqref{eq:gauss-kernel}
$$
|(f*\theta_\varepsilon)(t) - f(t)|
\le \int_{-\infty}^{\infty} |f(t - x) - f(t)|\,
\theta_\varepsilon(x)\, dx .
$$
Выберем $\delta > 0$ так, чтобы $|f(t-x) - f(t)| < \varepsilon'$ при
$|x| \le \delta$ (непрерывность в $t$; $\varepsilon'$ — произвольное
положительное число). Тогда
$$
|(f*\theta_\varepsilon)(t) - f(t)|
\le \varepsilon' \int_{|x|\le\delta}\theta_\varepsilon +
2 \sup_{|x|\le\delta}|f| \cdot \int_{|x|>\delta}\theta_\varepsilon
+ \int_{|x|>\delta} |f(t-x)|\,\theta_\varepsilon(x)\,dx .
$$
Первый член не превосходит $\varepsilon'$; второй стремится к нулю по
третьему условию \eqref{eq:gauss-kernel}; третий оценивается через
$\norm{f}_1 \cdot \sup_{|x|>\delta}\theta_\varepsilon(x) =
\frac{e^{-\delta^2/\varepsilon^2}}{\sqrt\pi\,\varepsilon}\norm{f}_1 \to 0$.
Значит, $\limsup_\varepsilon |(f*\theta_\varepsilon)(t) - f(t)| \le
\varepsilon'$, а по произвольности $\varepsilon'$ предел есть $f(t)$.

Итак, $I_\varepsilon(t) \to g(t)$ при всех $t$ и $I_\varepsilon(t) \to f(t)$
в точках непрерывности; по совместности пределов $f = g$ в этих точках.
Осталось заметить, что $f*\theta_\varepsilon \to f$ в $L_1$ (по непрерывности
сдвига, готовый пункт 3 раздела \ref{sec:given}), поэтому подпоследовательность
$(f*\theta_{\varepsilon_k})$ сходится к $f$ почти всюду, и $f = g$ почти
всюду.
\end{proof}

\begin{remark}
Посылки $f \in L_1$ и $\hat f \in L_1$ существенны по отдельности. Индикатор
$[a,b]$ принадлежит $L_1$, но его образ (лемма
\ref{lem:indicator-transform}) не интегрируем, и равенство
\eqref{eq:ft-inverse} в классическом смысле невозможно — ряд и интеграл
Фурье разрывной функции требуют обобщённых функций [34]. Непрерывность в
точке применения формулы нужна: в точке скачка интеграл обращения сходится к
среднему односторонних пределов, а не к значению функции.
\end{remark}

Теперь главная теорема о квадратичной теории — изометрия в $L_2$.

\begin{theorem}[Планшерель и Парсеваль]\label{thm:plancherel}
\textbf{(а)} Если $f, g \in L_1(\R) \cap L_2(\R)$, то $\hat f, \hat g \in
L_2(\R)$ и
\begin{equation}\label{eq:parseval}
\scal{f}{g} = \scal{\hat f}{\hat g}, \qquad\text{в частности}\qquad
\norm{f}_2 = \norm{\hat f}_2 .
\end{equation}
\textbf{(б)} Оператор $f \mapsto \hat f$ продолжается с плотного
подпространства $L_1\cap L_2$ до унитарного оператора на $L_2(\R)$;
обратный оператор есть продолжение $\check{\phantom{f}}$ с
\eqref{eq:fourier-inverse}. Для $f \in L_2(\R)$ образ совпадает с пределом
по норме $L_2$ усечённых интегралов:
\begin{equation}\label{eq:l2-limit}
\hat f = \lim_{A\to\infty} \frac{1}{\sqrt{2\pi}}
\int_{-A}^{A} f(t)\, e^{-i\omega t}\, dt .
\end{equation}
\end{theorem}

\begin{proof}
(а) Шаг 1: пусть дополнительно $\hat f \in L_1$. Положим
$$
h(t) = \int_{-\infty}^{\infty} f(u)\,\overline{f(u-t)}\,du .
$$
По неравенству Коши—Буняковского $|h(t)| \le
\norm{f}_2^2$, по теореме Фубини—Тонелли $\int|h(t)|\,dt \le \norm{f}_1^2$
(двойной интеграл от $|f(u)|\,|f(u-t)|$ разбивается на произведение норм),
и $h$ непрерывна (сдвиг непрерывен в $L_2$). Вычислим образ $h$ напрямую,
меняя порядок интегрирования (Фубини оправдан оценкой $\norm{f}_1^2$) и
подставляя $v = u - t$:
$$
\hat h(\omega) = \frac{1}{\sqrt{2\pi}} \iint f(u)\,
\overline{f(v)}\, e^{-i\omega(u-v)}\,du\,dv
= \sqrt{2\pi}\; \hat f(\omega)\, \overline{\hat f(\omega)}
= \sqrt{2\pi}\,|\hat f(\omega)|^2 .
$$
Так как $\hat f \in L_1\cap L_\infty$, то $|\hat f|^2 \le
\norm{\hat f}_\infty |\hat f| \in L_1$, то есть $\hat h \in L_1$; обе
посылки теоремы \ref{thm:ft-inverse} выполнены. Применяем её в точке
$t = 0$:
$$
\norm{f}_2^2 = h(0) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{\infty}
\hat h(\omega)\,d\omega = \int_{-\infty}^{\infty} |\hat f(\omega)|^2
\,d\omega = \norm{\hat f}_2^2 .
$$
Шаг 2: уберём дополнительное предположение. Для произвольного $f \in L_1
\cap L_2$ рассмотрим $f_\varepsilon = f * \theta_\varepsilon$ с гауссовым
ядром из доказательства теоремы \ref{thm:ft-inverse}. Тем же приёмом
(Фубини и замена $x = t - u$) получаем $\widehat{f_\varepsilon}(\omega) =
\hat f(\omega) \int \theta_\varepsilon(x) e^{-i\omega x}dx$; внутренний
интеграл уже вычислен при доказательстве теоремы
\ref{thm:ft-inverse} и равен $e^{-\varepsilon^2\omega^2/4}$. Таким
образом $\widehat{f_\varepsilon} = \hat f \cdot
e^{-\varepsilon^2\omega^2/4} \in L_1$ (произведение ограниченной функции
на гауссиану), и по шагу 1 $\norm{f_\varepsilon}_2 =
\norm{\widehat{f_\varepsilon}}_2$. Слева
$f_\varepsilon \to f$ в $L_2$ (аппроксимативная единица). Справа
$|\widehat{f_\varepsilon}(\omega)|^2 = |\hat f(\omega)|^2
e^{-\varepsilon^2\omega^2/2} \uparrow |\hat f(\omega)|^2$ при
$\varepsilon \downarrow 0$, поэтому по теореме Леви
$\norm{\widehat{f_\varepsilon}}_2^2 \to \int |\hat f|^2$ (предел конечен,
поскольку слева сходимость к $\norm{f}_2^2$). Итак, $\hat f \in L_2$ и
$\norm{f}_2 = \norm{\hat f}_2$ для всех $f \in L_1 \cap L_2$.

Равенство скалярных произведений получается поляризацией: равенство норм,
применённое к $f + g$ и $f + ig$, после вычитания тождеств параллелограмма
даёт $\scal{f}{g} = \scal{\hat f}{\hat g}$.

(б) По пункту (а) преобразование Фурье — изометрия плотного (по готовому
пункту 3 раздела \ref{sec:given}) подмножества $L_1\cap L_2 \subset L_2$;
по единственности продолжения она продолжается до изометрии всего $L_2$, а
по полноте образа (обратный оператор строится по той же схеме с ядром
$e^{i\omega t}$) — до унитарного. Представление \eqref{eq:l2-limit} —
сходимость того же семейства усечений, которым в пункте (а) аппроксимировали
$f$ ядром $\theta_\varepsilon$.
\end{proof}

\begin{remark}
Равенство \eqref{eq:parseval} — теорема Планшереля; равенство норм —
равенство Парсеваля. Посылка $f \in L_1 \cap L_2$ для вывода
\eqref{eq:parseval} неизбежна: для $f \in L_2 \setminus L_1$ интеграл
\eqref{eq:fourier} может не сходиться ни в одной точке, и образ
определяется только пределом \eqref{eq:l2-limit}. Для $f \in L_1 \setminus
L_2$ образ определён поточечно, но не обязан принадлежать $L_2$ — равенство
норм нарушено уже на $\mathbb I_{[a,b]}$.
\end{remark}

Последняя теорема непрерывной части — о свёртке, тождественная оператору,
который породил всю конструкцию (инвариантные во времени системы — это
свёртки [34]).

\begin{lemma}\label{lem:conv-l1}
Если $f, g \in L_1(\R)$, то $f * g$ определена почти всюду, $f*g \in
L_1(\R)$ и $\norm{f * g}_1 \le \norm{f}_1\, \norm{g}_1$.
\end{lemma}

\begin{proof}
Функция $F(t,\tau) = f(\tau)g(t-\tau)$ измерима на $\R^2$, и по теореме
Фубини—Тонелли
$$
\iint |F(t,\tau)|\,dt\,d\tau = \int |f(\tau)| \Bigl( \int
|g(t-\tau)|\,dt \Bigr) d\tau = \norm{f}_1 \norm{g}_1 < \infty .
$$
Значит, при почти каждом $t$ сечение $\tau \mapsto F(t,\tau)$
суммируемо (то есть $(f*g)(t)$ существует почти всюду) и $\int|(f*g)(t)|dt
\le \iint|F| = \norm{f}_1\norm{g}_1$.
\end{proof}

\begin{theorem}[о свёртке]\label{thm:convolution}
Если $f, g \in L_1(\R)$, то
\begin{equation}\label{eq:convolution-transform}
\widehat{f * g}(\omega) = \sqrt{2\pi}\; \hat f(\omega)\, \hat g(\omega).
\end{equation}
\end{theorem}

\begin{proof}
По лемме \ref{lem:conv-l1} $f*g \in L_1$, и
$$
\widehat{f*g}(\omega) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{\infty}
\Bigl( \int_{-\infty}^{\infty} f(\tau) g(t-\tau)\,d\tau \Bigr)
e^{-i\omega t}\,dt .
$$
Меняем порядок интегрирования (Фубини: $\frac{1}{\sqrt{2\pi}} \iint
|f(\tau)g(t-\tau)|d\tau dt = \frac{1}{\sqrt{2\pi}}\norm{f}_1\norm{g}_1 <
\infty$) и подставляем $t = u + \tau$:
$$
\widehat{f*g}(\omega) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{\infty}
f(\tau) e^{-i\omega\tau} \Bigl( \int_{-\infty}^{\infty}
g(u)\, e^{-i\omega u}\,du \Bigr) d\tau = \sqrt{2\pi}\,\hat f(\omega)\,
\hat g(\omega) .
$$
Множитель $\sqrt{2\pi}$ — плата единичной нормировки: обе части
\eqref{eq:convolution-transform} содержат его ровно один раз.
\end{proof}

\begin{remark}
Множитель $\sqrt{2\pi}$ зависит от нормировки и легко теряется при переносе
формул из источников с безмножительной нормировкой [34]; все перенесённые
ниже формулы пересчитаны, и проверка каждой — подстановка
$\hat f_{\text{без}} = \sqrt{2\pi}\,\hat f$. Теорема верна и для $f \in L_2$,
$g \in L_1$ (тогда $f*g \in L_2$ и $\norm{f*g}_2 \le \norm{f}_2\norm{g}_1$);
вывод тот же с заменой теоремы Фубини на оценку через $|f|^2$.
\end{remark}

# Преобразование Лапласа {#sec:laplace}

Преобразование Лапласа — преобразование Фурье, повёрнутое на четверть
плоскости: подстановка $s = \sigma_1 + i\omega$ с фиксированным $\sigma_1$
превращает \eqref{eq:laplace} в интеграл Фурье от $f(t)e^{-\sigma_1 t}$.
Эта связь даёт все доказательства раздела: теория оригиналов элементарна,
и набор не содержит источника, доказывающего её целиком ([30] заявляет
справочный характер и даёт постановки и свойства без доказательств), —
поэтому все доказательства ниже собственные.

\begin{theorem}[сходимость и аналитичность изображения]\label{thm:laplace-conv}
Пусть $f$ — оригинал с показателем роста $a_0$ и изображением $F(s)$.
Тогда:

\begin{enumerate}
\item интеграл \eqref{eq:laplace} сходится абсолютно при $\operatorname{Re}
s > a_0$ и равномерно по $s$ в каждой полуплоскости $\operatorname{Re} s
\ge a_0 + \delta$, $\delta > 0$;
\item при каждом $a > a_0$ выполнена оценка
\begin{equation}\label{eq:laplace-estimate}
|F(s)| \le \frac{M}{\operatorname{Re} s - a}, \qquad
\operatorname{Re} s > a,
\end{equation}
в частности $F(s) \to 0$ при $\operatorname{Re} s \to +\infty$ равномерно
по $\operatorname{Im} s$;
\item $F$ голоморфна в полуплоскости $\operatorname{Re} s > a_0$ и
\begin{equation}\label{eq:laplace-deriv}
F^{(n)}(s) = (-1)^n \int_0^{\infty} t^n f(t)\, e^{-s t}\, dt,
\qquad n = 1, 2, \dots
\end{equation}
\end{enumerate}
\end{theorem}

\begin{proof}
По определению показателя роста для каждого $a > a_0$ найдётся $M > 0$:
$|f(t)| \le M e^{a t}$ при $t \ge 0$.

1. При $\operatorname{Re} s = \sigma > a$ имеем $|f(t)e^{-st}| \le M
e^{-(\sigma - a)t}$ — суммируемая мажоранта; отсюда абсолютная сходимость.
Если же $\sigma \ge a_0 + \delta$, выберем $a \in (a_0, a_0 + \delta)$:
мажоранта $M e^{-(a_0 + \delta - a)t}$ не зависит от $s$, и по признаку
Вейерштрасса сходимость равномерна (и потому $F$ непрерывна в полуплоскости
$\operatorname{Re} s > a_0$).

2. Следует из мажорантной оценки:
$|F(s)| \le \int_0^\infty M e^{-(\sigma-a)t}dt = M/(\sigma - a)$.

3. Достаточно проверить случай $n = 1$, дальше — индукция. Зафиксируем
$s$ с $\operatorname{Re} s = \sigma > a_0$, выберем $a \in (a_0,
\sigma)$ и рассмотрим приращение $h$ с $|h| < \sigma - a$:
$$
\frac{F(s+h) - F(s)}{h} = \int_0^{\infty} f(t)\, e^{-s t}\,
\frac{e^{-h t} - 1}{h}\, dt .
$$
Из $|e^{-z} - 1| \le |z| e^{|z|}$ при $\operatorname{Re} z \ge 0$ следует
$\bigl|\frac{e^{-ht}-1}{h}\bigr| \le t\, e^{|h|t}$, и подынтегральное
выражение мажорируется $M t\, e^{-(\sigma - a - |h|)t} \in L_1(0,\infty)$;
при $h \to 0$ оно стремится к $-t f(t) e^{-st}$ почти всюду. По теореме
Лебега $F'(s) = -\int_0^\infty t f(t) e^{-st}dt$. Показатель сходимости
у производного интеграла тот же, так что процесс можно повторять.
\end{proof}

\begin{remark}
Все три посылки определения \ref{def:original} работают. Локальная
интегрируемость даёт существование интеграла на каждом конечном отрезке;
обнуление при $t<0$ делает преобразование односторонним и позволяет
сводить его к Фурье; условие роста определяет полуплоскость
сходимости. Оценка \eqref{eq:laplace-estimate} точна по порядку: для
$f(t) = e^{a_0 t}$ имеем $F(s) = 1/(s - a_0)$ и $|F(s)| = 1/(\sigma -
a_0)$. Посылка $\operatorname{Re} s > a_0$ существенна: при $\sigma <
a_0$ сходимость не гарантируется никакими свойствами $f$ — уже
$f(t) = e^{a_0 t}$ расходится при $\sigma \le a_0$.
\end{remark}

\begin{theorem}[единственности]\label{thm:laplace-uniq}
Пусть $f_1, f_2$ — оригиналы и их изображения совпадают в полуплоскости
$\operatorname{Re} s > \max(a_{0,1}, a_{0,2})$. Тогда $f_1(t) = f_2(t)$
в каждой точке $t > 0$ непрерывности обеих функций (и почти всюду на
$(0,\infty)$).
\end{theorem}

\begin{proof}
$f = f_1 - f_2$ — оригинал с показателем роста $a_0 =
\max(a_{0,1}, a_{0,2})$ и нулевым изображением. Зафиксируем $\sigma_1 >
a_0$ и положим $g(t) = f(t) e^{-\sigma_1 t}$; по пункту 2 теоремы
\ref{thm:laplace-conv} $g$ суммируема на $\R$, так как $\int |g| \le
\int M e^{-( \sigma_1 - a)t}dt < \infty$ при $a \in (a_0, \sigma_1)$.
Образ Фурье функции $g$ есть
$$
\hat g(\omega) = \frac{1}{\sqrt{2\pi}} \int_0^{\infty}
f(t)\, e^{-(\sigma_1 + i\omega)t}\,dt
= \frac{1}{\sqrt{2\pi}}\, F(\sigma_1 + i\omega) = 0
\quad \text{при всех } \omega \in \R .
$$
Единственность преобразования Фурье на $L_1$ следует из теоремы
\ref{thm:ft-inverse}: непрерывный представитель, восстанавливаемый по
образу нулевой интегралом обращения, тождественно равен нулю, то есть $g
= 0$ почти всюду и $f = 0$ почти всюду. В точках непрерывности $f_1$ и
$f_2$ из совпадения почти всюду следует совпадение в точности.
\end{proof}

Свойства преобразования Лапласа собраны в одну теорему; все они —
прямые следствия определения и теоремы \ref{thm:laplace-conv}.

\begin{lemma}\label{lem:derivative-original}
Пусть $f$ — оригинал с показателем роста $a_0$, кусочно гладкая на
$[0,\infty)$, и её производная $f'$ удовлетворяет условию роста
$|f'(t)| \le M_1 e^{a_1 t}$. Тогда $f'$ — оригинал и
\begin{equation}\label{eq:derivative-original}
\mathcal L\{f'\}(s) = s\, F(s) - f(+0), \qquad
\operatorname{Re} s > \max(a_0, a_1).
\end{equation}
\end{lemma}

\begin{proof}
Первое условие определения \ref{def:original} для $f'$ следует из кусочной
гладкости, второе — из обнуления $f$ при $t<0$ (при $t<0$ в окрестности
нуля $f'\equiv0$ справа от $0$ и равна нулю слева), третье — из условия
леммы. Формула Ньютона—Лейбница на отрезке $[0,t]$ даёт
$f(t) = f(+0) + \int_0^t f'(\tau)\,d\tau$, откуда видно, что и $f(+0)$
определено конечным. Проинтегрируем по частям на $[0, A]$, не забывая
точки излома (на каждом гладком куске формула верна, суммарные граничные
члены внутри сокращаются):
$$
\int_0^{A} f'(t)\, e^{-s t}\,dt
= f(A)\,e^{-sA} - f(+0) + s \int_0^{A} f(t)\, e^{-s t}\,dt .
$$
При $\operatorname{Re} s > a_0$ граничный член $f(A)e^{-sA} \to 0$
(выберем $a \in (a_0, \operatorname{Re} s)$: $|f(A)e^{-sA}| \le
M e^{(a - \operatorname{Re}s)A} \to 0$); при $\operatorname{Re} s > a_1$
интеграл слева сходится. Переходя к пределу $A \to \infty$, получаем
\eqref{eq:derivative-original}.
\end{proof}

\begin{theorem}[свойства преобразования Лапласа]\label{thm:laplace-props}
Пусть $f, g$ — оригиналы с показателями роста $a_0, b_0$ и изображениями
$F, G$. Тогда при указанных полуплоскостях:

\begin{enumerate}
\item \textbf{линейность:} $\alpha f + \beta g \dashv \alpha F + \beta G$,
$\operatorname{Re} s > \max(a_0, b_0)$;
\item \textbf{подобие:} $f(\lambda t) \dashv \frac{1}{\lambda}
F\bigl(\frac{s}{\lambda}\bigr)$ при $\lambda > 0$,
$\operatorname{Re} s > \lambda a_0$;
\item \textbf{дифференцирование оригинала:} если $f, f', \dots, f^{(n)}$
— оригиналы, то
$$
f^{(n)}(t) \dashv s^n F(s) - s^{n-1} f(+0) - s^{n-2} f'(+0) - \dots -
f^{(n-1)}(+0);
$$
\item \textbf{дифференцирование изображения:} $t^n f(t) \dashv
(-1)^n F^{(n)}(s)$, $\operatorname{Re} s > a_0$;
\item \textbf{интегрирование оригинала:} $\displaystyle\int_0^t f(\tau)\,
d\tau \dashv \frac{F(s)}{s}$, $\operatorname{Re} s > \max(a_0, 0)$;
\item \textbf{запаздывание:} для $\tau > 0$ функция $f(t - \tau)$
(обращённая в нуль при $t < \tau$) есть оригинал и $f(t-\tau) \dashv
e^{-s\tau} F(s)$, $\operatorname{Re} s > a_0$;
\item \textbf{сдвиг по $s$:} $e^{\lambda t} f(t) \dashv F(s - \lambda)$
для любого $\lambda \in \C$, $\operatorname{Re} s > a_0 +
\operatorname{Re}\lambda$;
\item \textbf{свёртка:} $(f * g)(t) = \int_0^t f(\tau) g(t-\tau)\,d\tau
\dashv F(s)\,G(s)$, $\operatorname{Re} s > \max(a_0, b_0)$.
\end{enumerate}
\end{theorem}

\begin{proof}
Все пункты, кроме 3, 5 и 8, — замены переменных и линейность под знаком
интеграла; разберем их коротко.

2. Подстановка $u = \lambda t$:
$\int_0^\infty f(\lambda t)e^{-st}dt = \frac1\lambda\int_0^\infty
f(u)e^{-(s/\lambda)u}du$.

4. Это формула \eqref{eq:laplace-deriv} из теоремы
\ref{thm:laplace-conv}.

6. Функция $\tilde f(t) = f(t-\tau)\mathbb I_{[\tau,\infty)}$ — оригинал
с тем же показателем роста; подстановка $u = t - \tau$:
$\int_\tau^\infty f(t-\tau)e^{-st}dt = e^{-s\tau}\int_0^\infty
f(u)e^{-su}du$.

7. $\int_0^\infty e^{\lambda t}f(t)e^{-st}dt = F(s-\lambda)$; показатель
сходимости $s - \lambda$ есть $\operatorname{Re} s - \operatorname{Re}
\lambda > a_0$.

3. Последовательное применение леммы \ref{lem:derivative-original}: из
$f \dashv F$ и $f, f'$ — оригиналы следует $f' \dashv sF(s) - f(+0)$;
применяя лемму к $f'$ и т.д., на $n$-м шаге получаем формулу пункта.

5. Функция $\varphi(t) = \int_0^t f(\tau)d\tau$ удовлетворяет условию
роста: при $a_0 < 0$ имеем $|\varphi(t)| \le M\int_0^t e^{a_0\tau}d\tau
\le M/|a_0|$, при $a_0 = 0$ — $|\varphi(t)| \le Mt \le M e^{\delta t}$,
при $a_0 > 0$ — $|\varphi(t)| \le \frac{M}{a_0}e^{a_0 t}$. Так как
$\varphi' = f$ почти всюду и $\varphi(+0) = 0$, по лемме
\ref{lem:derivative-original} $\mathcal L\{f\}(s) = s\,\mathcal
L\{\varphi\}(s)$, откуда $\mathcal L\{\varphi\}(s) = F(s)/s$ при
$\operatorname{Re} s > \max(a_0, 0)$ (условие $\operatorname{Re} s > 0$
нужно, чтобы $\varphi$ удовлетворяла условию роста с показателем
$\max(a_0, 0)$).

8. Оценим свёртку: при $a \ne b$
$$
|(f*g)(t)| \le M_f M_g \int_0^t e^{a\tau} e^{b(t-\tau)}d\tau =
M_f M_g\,\frac{e^{at} - e^{bt}}{a - b} \le C e^{\max(a,b)t},
$$
при $a = b$ появляется множитель $t$, и $t e^{at} \le C_\delta
e^{(a+\delta)t}$; в обоих случаях $f*g$ — оригинал с показателем не выше
$\max(a_0, b_0)$. Далее меняем порядок интегрирования по теореме
Фубини—Тонелли (при $\operatorname{Re} s > \max(a_0,b_0)$ всё абсолютно
сходится):
$$
\int_0^\infty e^{-st} \int_0^t f(\tau)g(t-\tau)\,d\tau\,dt
= \int_0^\infty f(\tau) e^{-s\tau} \Bigl( \int_\tau^\infty
g(t-\tau) e^{-s(t-\tau)}\,dt \Bigr) d\tau = F(s)\,G(s) .
$$
\end{proof}

\begin{remark}
Посылки о росте существенны уже в пункте 3: существование изображения у
$f$ не влечёт его существования у $f'$ (оригинал может осциллировать с
растущей частотой). Даже формально равенство $\mathcal L\{f'\} = sF(s) -
f(+0)$ показывает главное для приложений: начальные условия входят в
формулу автоматически, и дифференцирование заменяется умножением. Пункт 8
— причина, по которой преобразование Лапласа диагонализирует линейные
инвариантные системы [42]: отклик системы на входной сигнал есть свёртка
с импульсной переходной функцией, и в изображениях она становится
произведением.
\end{remark}

Формула обращения восстанавливает оригинал по изображению интегралом по
вертикальной прямой; это формула Меллина [30].

\begin{theorem}[Меллина]\label{thm:mellin}
Пусть $f$ — оригинал с показателем роста $a_0$, непрерывная в точке $t >
0$, и пусть при некотором $\sigma_1 > a_0$
\begin{equation}\label{eq:mellin-cond}
\int_{-\infty}^{\infty} \bigl|F(\sigma_1 + i\omega)\bigr|\,d\omega
< \infty .
\end{equation}
Тогда
\begin{equation}\label{eq:mellin}
f(t) = \frac{1}{2\pi i} \int_{\sigma_1 - i\infty}^{\sigma_1 + i\infty}
F(s)\, e^{s t}\, ds
= \frac{1}{2\pi i}\, \mathrm{v.p.}\!\!
\int_{\sigma_1 - i\infty}^{\sigma_1 + i\infty} F(s)\, e^{s t}\, ds ,
\end{equation}
где несобственный интеграл понимается как симметрический предел по
$\operatorname{Im} s$.
\end{theorem}

\begin{proof}
Зафиксируем $\sigma_1 > a_0$ и положим $g(u) = f(u)\,e^{-\sigma_1 u}$
(продолженную нулём при $u<0$). По оценке \eqref{eq:laplace-estimate}
$|F(\sigma_1 + i\omega)| \le M/(\sigma_1 - a)$ равномерно по $\omega$, а
вместе с условием \eqref{eq:mellin-cond} это даёт $g \in L_1(\R)$ и $\hat g
\in L_1(\R)$: действительно, $\hat g(\omega) = \frac{1}{\sqrt{2\pi}}
F(\sigma_1 + i\omega)$ по определению изображения. Функция $g$ непрерывна
в точке $u = t$ (так как $t > 0$ и $f$ непрерывна в $t$). Все посылки
теоремы \ref{thm:ft-inverse} выполнены, и она даёт
$$
f(t)\,e^{-\sigma_1 t} = g(t) = \frac{1}{\sqrt{2\pi}}
\int_{-\infty}^{\infty} \hat g(\omega)\, e^{i\omega t}\,d\omega
= \frac{1}{2\pi} \int_{-\infty}^{\infty}
F(\sigma_1 + i\omega)\, e^{i\omega t}\, d\omega .
$$
Умножая на $e^{\sigma_1 t}$ и подставляя $s = \sigma_1 + i\omega$,
$ds = i\,d\omega$, получаем \eqref{eq:mellin}; симметричные пределы по
$\omega$ переходят в симметричные пределы по $\operatorname{Im} s$.
\end{proof}

\begin{remark}
Условие \eqref{eq:mellin-cond} выполнено, например, если $|F(s)| \le
C(1+|s|)^{-1-\delta}$ на прямой $\operatorname{Re} s = \sigma_1$. Без
него формула \eqref{eq:mellin} остаётся верной в смысле главного значения
при более слабых условиях на $f$ (например, конечных односторонних
производных в точке [30]), но доказательство тогда требует отдельной
регуляризации образа; в приложениях изображение обычно рационально и
\eqref{eq:mellin-cond} проверяется непосредственно. Принципиально, что
прямую $\operatorname{Re} s = \sigma_1$ можно брать любую правее $a_0$:
интеграл не зависит от $\sigma_1$, так как подынтегральная функция
голоморфна между любыми двумя такими прямыми и оценка
\eqref{eq:laplace-estimate} позволяет деформировать контур.
\end{remark}

Из формулы Меллина получается вторая пара
преобразования — для произведений.

\begin{corollary}[о произведении оригиналов]\label{cor:product}
Пусть $f, g$ — оригиналы, $F, G$ — их изображения и
$\int_{-\infty}^{\infty} |F(\sigma_1 + i\omega)|\,d\omega < \infty$ при
некотором $\sigma_1 > a_0$. Тогда при $\operatorname{Re} s > a_0 + b_0$
\begin{equation}\label{eq:product}
f(t)\,g(t) \dashv \frac{1}{2\pi i}
\int_{\sigma_1 - i\infty}^{\sigma_1 + i\infty} F(q)\,G(s - q)\,dq .
\end{equation}
\end{corollary}

\begin{proof}
По формуле Меллина \eqref{eq:mellin} $f(t) = \frac{1}{2\pi i}\int F(q)
e^{qt}dq$; подставляя это в интеграл $\int_0^\infty f(t)g(t)e^{-st}dt$ и
меняя порядок интегрирования (Фубини: всё абсолютно сходится, так как
$\operatorname{Re} q = \sigma_1 > a_0$ и $\operatorname{Re}(s - q) >
b_0$), получаем
$$
\int_0^\infty f(t)g(t)e^{-st}dt = \frac{1}{2\pi i}\int_{\sigma_1 -
i\infty}^{\sigma_1 + i\infty} F(q) \Bigl( \int_0^\infty g(t)
e^{-(s-q)t}\,dt \Bigr) dq = \frac{1}{2\pi i}\int F(q)\,G(s-q)\,dq .
$$
\end{proof}

Теперь вся машина собирается в схему решения задачи Коши для линейного
уравнения с постоянными коэффициентами — постановка и рамка из [30]
(§1.3), общая формула выводится из пунктов 1 и 3 теоремы
\ref{thm:laplace-props}.

\begin{corollary}[операторный метод для задачи Коши]\label{cor:ode-solution}
Рассмотрим задачу
\begin{equation}\label{eq:cauchy-problem}
y^{(n)} + a_1 y^{(n-1)} + \dots + a_{n-1} y' + a_n y = f(t), \qquad
y(0) = y_0,\ y'(0) = y_1,\ \dots,\ y^{(n-1)}(0) = y_{n-1},
\end{equation}
где $f$ — оригинал. Если $y(t)$ — решение-оригинал с изображением $Y(s)$,
то
\begin{equation}\label{eq:operator-equation}
A(s)\,Y(s) = B(s) + F(s),
\end{equation}
где
$$
A(s) = s^n + a_1 s^{n-1} + \dots + a_{n-1} s + a_n, \qquad
B(s) = \sum_{k=0}^{n-1} y_k \sum_{j=0}^{n-1-k} a_j\, s^{\,n-1-k-j},
\quad a_0 := 1
$$
(внутренний многочлен при $k$ — хвост $A(s)$, начинающийся со степени
$s^{n-1-k}$). Следовательно,
\begin{equation}\label{eq:ode-solution}
Y(s) = \frac{B(s) + F(s)}{A(s)},
\end{equation}
и решение задачи \eqref{eq:cauchy-problem} восстанавливается из $Y(s)$
по формуле Меллина или разложением на простейшие дроби.
\end{corollary}

\begin{proof}
Применим преобразование Лапласа к обеим частям уравнения. По линейности
(пункт 1 теоремы \ref{thm:laplace-props}) и формуле дифференцирования
оригинала (пункт 3)
$$
\mathcal L\{y^{(j)}\}(s) = s^j Y(s) - s^{j-1}y_0 - s^{j-2}y_1 - \dots -
y_{j-1} .
$$
Подставляя в уравнение и собирая слагаемые при $y_k$, получаем
$A(s)Y(s) - B(s) = F(s)$: вклад $y_k$ с одной стороны вынесен из
производных со степенями $s$, с другой — вычтен, и обе части складываются
в $y_k \sum_{j=0}^{n-1-k} a_j s^{n-1-k-j}$ — хвост характеристического
многочлена. Равенство \eqref{eq:operator-equation} — алгебраическое
уравнение, единственное решение которого \eqref{eq:ode-solution}; по
теореме \ref{thm:laplace-uniq} оригинал $y(t)$ определяется по $Y(s)$
однозначно в точках непрерывности.
\end{proof}

\begin{remark}
Правая часть $F(s)$ входит в \eqref{eq:ode-solution} аддитивно, а полином
$B(s)$ кодирует начальные условия: их вклад в решение есть отклик на
«начальный импульс». Формула \eqref{eq:ode-solution} работает и для
систем уравнений: тогда $A(s)$ — характеристическая матрица, и $X =
A^{-1}(B + F)$ [30]. Пример расчёта для колебательного контура (RLC-цепи)
разобран в задаче \ref{prob:ode}; прикладная мотивация метода — [42].
\end{remark}

# Система Хаара {#sec:haar}

Система Хаара — старейший пример ортонормированного вейвлет-базиса (известен
с 1910 года [11]). В отличие от экспонент Фурье, её функции имеют
компактный носитель и нулевое среднее, и эти два свойства определяют всю
теорию.

\begin{theorem}[ортонормированность системы Хаара]\label{thm:haar-on}
Система $\{\psi_{j,k}\}_{j,k\in\Z}$ ортонормирована в $L_2(\R)$:
\begin{equation}\label{eq:haar-on}
\scal{\psi_{j,k}}{\psi_{j',k'}} = \delta_{j j'}\,\delta_{k k'} .
\end{equation}
\end{theorem}

\begin{proof}
Норма: подстановкой $u = 2^j t - k$
$$
\norm{\psi_{j,k}}_2^2 = 2^j \int_{-\infty}^{\infty} |\psi(2^j t - k)|^2
dt = \int_{-\infty}^{\infty} |\psi(u)|^2 du = 1 ,
$$
так как $\psi^2 = \mathbb I_{[0,1)}$.

Ортогональность. При $j = j'$, $k \ne k'$ носители $I_{j,k}$ и $I_{j,k'}$
не пересекаются, и скалярное произведение равно нулю.

При $j < j'$ носитель $I_{j',k'}$ либо не пересекается с $I_{j,k}$
(снова ноль), либо содержится в одной из половин $I_{j,k}$, на которой
$\psi_{j,k}$ постоянна и равна $\pm 2^{j/2}$. На всём $I_{j',k'}$
функция $\psi_{j',k'}$ имеет нулевой интеграл (она принимает значения
$\pm 2^{j'/2}$ на равных половинах), поэтому
$$
\scal{\psi_{j,k}}{\psi_{j',k'}} = \pm 2^{j/2} \int_{I_{j',k'}}
\psi_{j',k'}(t)\,dt = 0 .
$$
Случай $j > j'$ симметричен (скалярное произведение сопряжено).
\end{proof}

Полнота — содержательная часть; доказательство [11] (§1.3.3) идёт через
усреднение по диадическим интервалам, и в нашей индексации $\psi_{j,k} =
2^{j/2}\psi(2^j t - k)$ оно выглядит так.

\begin{theorem}[полнота системы Хаара]\label{thm:haar-complete}
Если $f \in L_2(\R)$ и $\scal{f}{\psi_{j,k}} = 0$ при всех $j, k \in \Z$,
то $f = 0$ почти всюду.
\end{theorem}

\begin{proof}
Шаг 1: усреднение по диадическим интервалам. Условие
$\scal{f}{\psi_{j,k}} = 0$ означает равенство интегралов $f$ по половинам
$I_{j,k}$:
$$
\int_{k 2^{-j}}^{(k+\frac12)2^{-j}} f\,dt =
\int_{(k+\frac12)2^{-j}}^{(k+1)2^{-j}} f\,dt
= \frac12 \int_{I_{j,k}} f\,dt .
$$
Значит, среднее значение $f$ по диадическому интервалу совпадает со
средним по каждому его диадическому подинтервалу, полученному
последовательным делением пополам.

Шаг 2: затухание среднего на больших интервалах. Возьмём произвольный
диадический интервал $I_{j,k} = [k2^{-j}, (k+1)2^{-j})$ и больший
диадический интервал $J = [0, 2^m) \supset I_{j,k}$ с достаточно
большим $m$ при $k \ge 0$ (если $k < 0$, берём $J = [-2^m, 0)$;
рассуждение то же).
Последовательно деля $J$ пополам и пользуясь шагом 1, получаем
$$
\frac{1}{|I_{j,k}|} \int_{I_{j,k}} f\,dt = \frac{1}{|J|} \int_J f\,dt .
$$
По неравенству Коши—Буняковского
$$
\Bigl| \frac{1}{|J|}\int_J f\,dt \Bigr| \le
\frac{1}{|J|}\, \norm{f}_2\, \sqrt{|J|} =
\frac{\norm{f}_2}{\sqrt{|J|}} \xrightarrow[m \to \infty]{} 0 ,
$$
поэтому $\int_{I_{j,k}} f\,dt = 0$ для всех $j, k \in \Z$.

Шаг 3: от диадических интервалов к произвольным. Каждый интервал $(a,b)$
представляется счётным объединением попарно непересекающихся диадических
интервалов $I_n$ (максимальные диадические интервалы, содержащиеся в
$(a,b)$). Интеграл $f$ по объединению равен сумме интегралов по кускам:
ряд сходится абсолютно, так как $\sum_n \bigl|\int_{I_n} f\,dt\bigr| \le
\sum_n \int_{I_n} |f|\,dt = \int_a^b |f|\,dt < \infty$ (на конечном
интервале $f \in L_2(a,b) \subset L_1(a,b)$). Итак, $\int_a^b f\,dt = 0$
для всех интервалов. По готовому факту из теории меры (раздел
\ref{sec:given}) $f = 0$ почти всюду.
\end{proof}

\begin{remark}
Доказательство [11] формулируется через плотность: любую $L_2$-функцию
аппроксимируют кусочно-постоянными на диадической сетке, а каждую такую
— конечной комбинацией $\psi_{j,k}$ и почти константы на большом
интервале, чья норма мала. Это тот же усреднительный механизм, что и
выше, и он одновременно доказывает теорему \ref{thm:haar-parseval}
(равномерную сходимость). Обе версии эквивалентны; полнота существенно
опирается на бесконечность прямой — среднее уходит на диадические
интервалы неограниченной длины. На отрезке этого механизма нет: в
$L_2[0,1)$ чистая вейвлет-система $\{\psi_{j,k}\}_{j \ge 0}$
ортонормирована, но не полна (она ортогональна константе
$\mathbb I_{[0,1)}$), и для базиса нужен ещё индикатор отрезка —
масштабирующая функция; дискретный аналог — вектор $N^{-1/2}\mathbb 1$
в теореме \ref{thm:fwt}.
\end{remark}

\begin{theorem}[Парсеваль и сходимость ряда Хаара]\label{thm:haar-parseval}
\textbf{(а)} Для любой $f \in L_2(\R)$ её ряд Хаара сходится к $f$ по
норме $L_2$:
\begin{equation}\label{eq:haar-series}
f = \sum_{j,k \in \Z} \scal{f}{\psi_{j,k}}\, \psi_{j,k},
\end{equation}
и выполнено равенство Парсеваля
\begin{equation}\label{eq:haar-parseval}
\norm{f}_2^2 = \sum_{j,k\in\Z} \bigl|\scal{f}{\psi_{j,k}}\bigr|^2 .
\end{equation}
\textbf{(б)} Частичная сумма по уровням $S_J f = \sum_{j \le J}
\sum_{k\in\Z} \scal{f}{\psi_{j,k}}\psi_{j,k}$ есть ортогональная
проекция $f$ на подпространство $V_{J+1}$ функций, кусочно-постоянных
на диадических интервалах уровня $J+1$; для равномерно непрерывной $f$
\begin{equation}\label{eq:haar-uniform}
\norm{S_J f - f}_\infty \le \omega\bigl(f, 2^{-J-1}\bigr),
\end{equation}
где $\omega(f,\delta) = \sup_{|t-s|\le\delta} |f(t) - f(s)|$ — модуль
непрерывности.
\end{theorem}

\begin{proof}
(а) По теоремам \ref{thm:haar-on} и \ref{thm:haar-complete} система Хаара
— ортонормированный базис $L_2(\R)$. Равенство \eqref{eq:haar-parseval}
— это критерий полноты ортонормированной системы из вопроса 09
(равенство Парсеваля эквивалентно полноте), равенство
\eqref{eq:haar-series} — сходимость ряда Фурье по полному
ортонормированному базису (теорема Рисса—Фишера даёт сходимость
частичных сумм к элементу, совпадающему с $f$ по полноте системы).

(б) Обозначим через $V_j \subset L_2(\R)$ множество функций,
кусочно-постоянных на интервалах $I_{j,k}$, $k \in \Z$; это замкнутое
подпространство, изоморфное $\ell_2$, и ортогональная проекция на него
есть усреднение:
$$
(P_{V_j} f)(t) = \frac{1}{|I_{j,k}|} \int_{I_{j,k}} f\,d\tau ,
\qquad t \in I_{j,k},
$$
(невязка ортогональна каждому $\mathbb I_{I_{j,k}}$ по выбору постоянной,
а индикаторы порождают $V_j$). Далее, при фиксированном $j$ функции
$\psi_{j,k}$, $k \in \Z$, ортонормированы (теорема \ref{thm:haar-on}) и
порождают дополнение $W_j = V_{j+1} \ominus V_j$: всякий $h \in V_{j+1}$,
ортогональный $V_j$, имеет нулевое среднее на каждом $I_{j,k}$, а значит,
равен $\sum_k c_k \psi_{j,k}$ с $c_k = \pm 2^{j/2}h\big|_{I_{j,k}}$.
Поэтому $P_{W_j} f = \sum_k \scal{f}{\psi_{j,k}}\psi_{j,k}$ и, так как
проекции на взаимно ортогональные дополнения складываются,
$$
S_J f = \sum_{j\le J} P_{W_j} f = P_{V_{J+1}} f .
$$
(равномерная оценка) Если $f$ равномерно непрерывна, то на каждом
$I_{J+1,k}$ из определения проекции
$$
|S_J f(t) - f(t)| = \Bigl| \frac{1}{|I_{J+1,k}|}\int_{I_{J+1,k}}
\bigl(f(\tau) - f(t)\bigr)\,d\tau \Bigr| \le \omega\bigl(f, |I_{J+1,k}|
\bigr) = \omega\bigl(f, 2^{-J-1}\bigr),
$$
что и даёт \eqref{eq:haar-uniform} (при $J \to \infty$ правая часть
стремится к нулю по определению равномерной непрерывности).
\end{proof}

\begin{remark}[локализация и эффект Гиббса]
Оценка \eqref{eq:haar-uniform} носит локальный характер: сходимость
одинаково быстрая у гладких кусков и у скачков, никаких осцилляций у
разрыва не возникает — проекция на $V_{J+1}$ вблизи скачка просто
принимает среднее значение по интервалу. У тригонометрического ряда
картина иная: частичные суммы Фурье разрывной функции колеблются в
окрестности скачка с амплитудой, не зависящей от числа гармоник
(эффект Гиббса [34]). Разложение по Хаару точно представляет
кусочно-постоянные сигналы конечным числом коэффициентов, а для
кусочно-гладких коэффициенты уровня $j$ вдали от изломов имеют порядок
$O(2^{-3j/2})$ (задача \ref{prob:haar-coeffs}); численное сравнение
сжатия сигнала $t\,e^{-t}$ в базисах Фурье и Хаара — в ноутбуке вопроса.
Цена локализации — негладкость базисных функций: теорема
\ref{thm:heisenberg} не даёт хорошо локализованной системы с гладкими
элементами, и вопрос 05 (методы вейвлет-анализа) посвящён именно
покупке гладкости ценой усложнения конструкции.
\end{remark}

# Вычислительная сторона {#sec:compute}

На практике сигнал задан отсчётами, и все три преобразования превращаются
в линейную алгебру над $\C^N$. ДПФ — матричное умножение за $O(N^2)$,
быстрое преобразование Фурье — та же матрица за $O(N\log_2 N)$; ряд
Хаара — ортогональное преобразование, вычисляемое вообще без умножений.

\begin{lemma}[об ортогональности экспонент]\label{lem:dft-orth}
При любом $j \in \Z$
\begin{equation}\label{eq:dft-orth}
\frac{1}{N} \sum_{k=0}^{N-1} \varepsilon_N^{k j} = \delta_N(j) =
\begin{cases} 1, & j \equiv 0 \pmod N, \\ 0, & \text{иначе}. \end{cases}
\end{equation}
\end{lemma}

\begin{proof}
При $j \equiv 0 \pmod N$ все слагаемые равны 1. Иначе это сумма
геометрической прогрессии со знаменателем $\varepsilon_N^j \ne 1$:
$\frac{\varepsilon_N^{jN} - 1}{\varepsilon_N^j - 1} = 0$, так как
$\varepsilon_N^{jN} = 1$.
\end{proof}

\begin{theorem}[ДПФ: обращение, Парсеваль, свёртка]\label{thm:dft}
Пусть $X = \mathcal F_N x$, $Y = \mathcal F_N y$. Тогда:

\begin{enumerate}
\item \textbf{обращение:}
$\displaystyle x(j) = \frac{1}{N} \sum_{k=0}^{N-1} X(k)\,
\varepsilon_N^{k j}$;
\item \textbf{Парсеваль:} $\scal{x}{y} = \frac{1}{N}\scal{X}{Y}$, в
частности $\norm{x}_2^2 = \frac1N\norm{X}_2^2$;
\item \textbf{свёртка:} $\mathcal F_N(x * y) = X\,Y$ (покомпонентное
произведение); следовательно $x * y = \mathcal F_N^{-1}(XY)$.
\end{enumerate}
\end{theorem}

\begin{proof}
1. Подставим определение в правую часть и воспользуемся леммой
\ref{lem:dft-orth}:
$$
\frac{1}{N}\sum_{k=0}^{N-1} X(k)\,\varepsilon_N^{kj}
= \sum_{l=0}^{N-1} x(l)\, \frac1N\sum_{k=0}^{N-1}\varepsilon_N^{k(j-l)}
= \sum_l x(l)\,\delta_N(j-l) = x(j) .
$$

2. По определению и лемме \ref{lem:dft-orth}
$$
\scal{X}{Y} = \sum_{k=0}^{N-1} \sum_{j=0}^{N-1}\sum_{l=0}^{N-1}
x(j)\,\overline{y(l)}\,\varepsilon_N^{-kj}\varepsilon_N^{kl}
= N \sum_{j=0}^{N-1} x(j)\,\overline{y(j)} = N\scal{x}{y}.
$$

3. Аналогично, меняя порядок суммирования,
$$
[\mathcal F_N(x*y)](k) = \sum_{j=0}^{N-1}\sum_{l=0}^{N-1}
x(l)\,y(j-l)\,\varepsilon_N^{-kj}
= \sum_{l=0}^{N-1} x(l)\,\varepsilon_N^{-kl}
\sum_{j=0}^{N-1} y(j-l)\,\varepsilon_N^{-k(j-l)} = X(k)\,Y(k),
$$
внутренняя сумма равна $Y(k)$ по $N$-периодичности $y$.
\end{proof}

\begin{remark}
Формула обращения с множителем $1/N$ (а не $1/\sqrt N$) и Парсеваль с
$1/N$ — сознательное отступление от унитарной нормировки ДПФ, за
источником [66]. Система $\{u_k(j) = \varepsilon_N^{kj}\}_{k=0}^{N-1}$
ортогональна с $\norm{u_k}^2 = N$ (лемма \ref{lem:dft-orth}), то есть
экспоненциальный базис не нормирован, и множители в формулах
обращения и Парсеваля — компенсация этой нормы. При чётном $N$ частота
Найквиста $N/2$ — граница различимости: значения $X(k)$ и $X(N-k)$
совпадают с точностью до комплексного сопряжения для вещественного
сигнала, поэтому «высшие» половины спектра описывают те же гармоники.
\end{remark}

Прямое вычисление ДПФ — матричное умножение с $N^2$ умножениями.
Следующая теорема показывает, что при $N = 2^s$ оно разваливается на
повторяющиеся бабочки.

\begin{theorem}[Кули—Тьюки]\label{thm:fft}
При $N = 2^s$ ДПФ вычисляется за
\begin{equation}\label{eq:fft-count}
\tfrac12\, N \log_2 N \ \text{умножений} \qquad\text{и}\qquad
N \log_2 N \ \text{сложений},
\end{equation}
то есть за $O(N\log_2 N)$ операций.
\end{theorem}

\begin{proof}
Разобьём сумму \eqref{eq:dft} по чётности индекса, $W_N^k =
\varepsilon_N^{-k}$:
$$
X(k) = \sum_{j=0}^{N/2-1} x(2j)\,\varepsilon_{N/2}^{-kj}
+ W_N^k \sum_{j=0}^{N/2-1} x(2j+1)\,\varepsilon_{N/2}^{-kj}
= X_e(k) + W_N^k X_o(k),
$$
где $X_e, X_o$ — ДПФ половинной длины, а равенство верно при всех $k$
по периодичности $X_e, X_o$ с периодом $N/2$ и тождеству
$\varepsilon_{N/2}^{-kj} = W_N^{2kj}$. Так как $W_N^{k+N/2} =
-W_N^k$ и $X_e(k + N/2) = X_e(k)$, для пары частот $(k, k+N/2)$,
$k \in 0 : N/2 - 1$, имеем
$$
X(k) = X_e(k) + W_N^k X_o(k), \qquad
X(k + N/2) = X_e(k) - W_N^k X_o(k) .
$$
Пара значений вычисляется из пары значений половинных ДПФ одним
комплексным умножением ($W_N^k X_o(k)$) и двумя сложениями. Если $T(N)$ —
число умножений, то $T(N) = 2T(N/2) + N/2$, $T(1) = 0$, откуда $T(N) =
\frac12 N \log_2 N$; сложения считаются так же, их два на каждую бабочку
из $N/2$ на уровне, итого $N \log_2 N$. Рекурсия опускается до ДПФ
длины 1, которые тривиальны; схема с прореживанием по времени и точный
подсчёт операций — [66].
\end{proof}

\begin{remark}
Оценка \eqref{eq:fft-count} точна для базовой схемы; на практике учитывают
тривиальные повороты ($W_N^0 = 1$) и считают настоящую стоимость в
flop-ах, но порядок $N\log_2 N$ не меняется. Устойчивость схемы следует
из унитарности нормированного ДПФ: $\frac{1}{\sqrt N}\mathcal F_N$ —
унитарная матрица, её норма равна 1, и погрешность входа не
усиливается. Обратное преобразование — та же схема с комплексным
сопряжением и множителем $1/N$.
\end{remark}

Третья конструкция — дискретный аналог системы Хаара. Пирамида
\eqref{eq:pyramid} с нормировкой $1/\sqrt2$ есть именно разложение по
ортонормированному дискретному базису Хаара, и её обратимость видна
напрямую.

\begin{theorem}[БПХ]\label{thm:fwt}
Пусть $N = 2^s$, $x \in \C^N$, и $\{s_\nu, d_\nu\}$ — его пирамида
средних и деталей \eqref{eq:pyramid}. Тогда:

\begin{enumerate}
\item каждый уровень обратим: $s_{\nu-1,2k} =
\frac{1}{\sqrt2}(s_{\nu,k} + d_{\nu,k})$, $s_{\nu-1,2k+1} =
\frac{1}{\sqrt2}(s_{\nu,k} - d_{\nu,k})$;
\item весь набор $\{s_s\} \cup \{d_\nu\}_{\nu=1}^s$ — образ $x$ при
ортогональном (в единичной нормировке) преобразовании, то есть
$x$ восстанавливается по нему единственным образом;
\item все коэффициенты вычисляются за $2(N-1)$ сложений и $N-1$
умножений на $1/\sqrt2$ (в схеме с немасштабированными суммами и
разностями — вообще без умножений [66]);
\item при движении вперёд средние «осредняют» сигнал, а детали несут
разности соседних отсчётов; сигнал, кусочно-постоянный на блоках
длины $2^\nu$, имеет нулевые детали всех уровней $\le \nu$.
\end{enumerate}
\end{theorem}

\begin{proof}
1. Прямая подстановка:
$\frac{1}{\sqrt2}(s_{\nu,k}+d_{\nu,k}) =
\frac{1}{\sqrt2}\cdot\frac{1}{\sqrt2}\bigl((s_{\nu-1,2k}+s_{\nu-1,2k+1}) +
(s_{\nu-1,2k}-s_{\nu-1,2k+1})\bigr) = s_{\nu-1,2k}$ и, с тем же
сокращением,
$\frac{1}{\sqrt2}(s_{\nu,k}-d_{\nu,k}) = s_{\nu-1,2k+1}$.

2. Каждый шаг \eqref{eq:pyramid} — умножение вектора на ортогональную
матрицу из блоков $\frac1{\sqrt2}\left(\begin{smallmatrix} 1 & 1 \\ 1 &
-1 \end{smallmatrix}\right)$ (бабочка), поэтому вся пирамида — ортогональное
преобразование $\C^N \to \C^N$, обратное — произведение транспонированных
бабочек, что и описывает пункт 1.

3. На уровне $\nu$ вычисляется $N_\nu = N/2^\nu$ пар «сумма–разность»,
каждая ценой одного сложения и одного вычитания: итого
$2\sum_{\nu=1}^s N_\nu = 2(N-1)$ сложения. Умножений на $1/\sqrt2$
ровно $N-1$ (по одному на каждую пару); если хранить немасштабированные
суммы и разности и компенсировать множители $2^{-\nu}$ при
восстановлении, как в схеме \mbox{[66]}, умножения исчезают вовсе.

4. Первое утверждение — определение пирамиды; второе следует из того,
что $\psi$ имеет нулевой интеграл: разность значений кусочно-постоянной
на блоке функции внутри блока равна нулю.
\end{proof}

\begin{remark}
Связь с непрерывной системой Хаара прямая: детали $d_{\nu,k}$ —
коэффициенты $\scal{x}{h^{(\nu)}_k}$ по ортонормированным векторам
$h^{(\nu)}_k$ со значениями $\pm 2^{-\nu/2}$ на диадическом блоке длины
$2^\nu$ (первый полублок «$+$», второй «$-$»), а среднее $s_s$ —
коэффициент по нормированному индикатору $N^{-1/2}\mathbb 1$ всей сетки
— дискретный аналог масштабирующей функции. Вместе $\{N^{-1/2}\mathbb
1\} \cup \{h^{(\nu)}_k\}$ образуют ортонормированный базис $\C^N$:
взаимная ортогональность проверяется тем же рассуждением о вложенных
диадических блоках, что в теореме \ref{thm:haar-on}, а число векторов
$1 + \sum_{\nu=1}^s N_\nu = N$. В нормировке [66] те же векторы имеют
норму $2^{\nu/2}$, и компенсирующие множители $2^{-\nu}$ стоят в формуле
разложения; в нашей единичной нормировке компенсация встроена в
$1/\sqrt2$. Обе схемы совершают одни и те же $2(N-1)$ сложения.
Преобразование ортогонально, поэтому абсолютно устойчиво: погрешность
входа не усиливается ни на одном уровне.
\end{remark}

Теорема обратной стороны принципа неопределённости работает и на $\Z_N$:
носитель сигнала и носитель его спектра не могут быть одновременно малы.

\begin{remark}[принцип неопределённости на $\Z_N$]\label{rem:uncertainty-zn}
Для любого ненулевого сигнала $x \in \C^N$
\begin{equation}\label{eq:uncertainty-zn}
|\operatorname{supp} x| \cdot |\operatorname{supp} X| \ge N .
\end{equation}
\end{remark}

\begin{proof}
Это теорема 10.1 из [66]. Если $m = |\operatorname{supp}x|$, то среди
любых $m$ последовательных значений спектра $X(q+1), \dots, X(q+m)$
найдётся ненулевое: иначе система $\sum_k z_k^{q+l}\, x(j_k) = 0$,
$l = 1..m$, с матрицей Вандермонда из различных $z_k =
\varepsilon_N^{-j_k}$ имела бы ненулевое решение $(x(j_k))$, что
невозможно. Значит, $|\operatorname{supp}X| \ge N/m$.
\end{proof}

Оговорки о сложности и устойчивости собраны: БПФ — $O(N\log_2 N)$ с
усилением ошибки в норме $\sqrt N$ у ненормированного ДПФ (единица у
унитарной версии); БПХ — $2(N-1)$ сложений и ортогональность; обращение
Меллина в численной постановке — интегрирование быстроосциллирующей
функции, чувствительное к шуму: задача восстановления оригинала по
изображению некорректна по Адамару [14], и это мост к вопросу 02 (ML и
некорректные задачи).

\textbf{Где работает конструкция.}

\begin{center}\small
\begin{tabular}{p{36mm}p{38mm}p{44mm}p{24mm}}
\hline
Постановка & Что вычисляется & Ответ & Что покупается \\
\hline
Спектральный анализ записи & $X = \mathcal F_N x$ & частотный состав,
доля энергии на гармониках & БПФ: $O(N\log_2N)$ вместо $O(N^2)$ \\
Линейное ОДУ с постоянными коэффициентами & $Y(s) = (B(s)+F(s))/A(s)$ &
$y(t)$ по формуле Меллина или таблице & дифферен\-цирование заменено
умножением; начальные условия входят автоматически \\
Отклик линейной системы & $\mathcal F_N^{-1}(X Y)$ или $f*g$ во
временной области & свёртка за $O(N\log_2N)$ & та же БПФ-машина \\
Сжатие сигнала с изломами & коэффициенты Хаара, пороговое обнуление &
приближение малой долей коэффициентов & нет осцилляций Гиббса у скачков \\
\hline
\end{tabular}
\end{center}

# Разобранные задачи {#sec:problems}

\begin{problem}\label{prob:fourier-images}
Найти преобразования Фурье прямоугольного импульса $\mathbb I_{[-1,1]}$ и
сигнала $u(t) = e^{-t}\mathbb I_{[0,\infty)}(t)$; проверить теорему
\ref{thm:convolution} на свёртке $u * u$ и равенство Парсеваля для $u$.
\end{problem}

*Решение.* Для прямоугольного импульса по лемме \ref{lem:indicator-transform}
$$
\widehat{\mathbb I_{[-1,1]}}(\omega) = \frac{1}{\sqrt{2\pi}}\cdot
\frac{e^{i\omega} - e^{-i\omega}}{i\omega} =
\sqrt{\frac{2}{\pi}}\cdot\frac{\sin\omega}{\omega}
$$
(в нуле — непрерывное продолжение $\sqrt{2/\pi}$). Для экспоненциального
импульса
$$
\hat u(\omega) = \frac{1}{\sqrt{2\pi}} \int_0^{\infty}
e^{-(1+i\omega)t}\,dt = \frac{1}{\sqrt{2\pi}}\cdot\frac{1}{1+i\omega} .
$$

Свёртка двух экспонент вычисляется непосредственно: при $t \ge 0$
$$
(u*u)(t) = \int_0^t e^{-\tau} e^{-(t-\tau)}\,d\tau = t\,e^{-t},
$$
и $(u*u)(t) = 0$ при $t < 0$. С другой стороны, по теореме
\ref{thm:convolution}
$$
\widehat{u*u}(\omega) = \sqrt{2\pi}\,\hat u(\omega)^2 =
\frac{1}{\sqrt{2\pi}}\cdot\frac{1}{(1+i\omega)^2} ,
$$
что совпадает с прямым вычислением
$\frac{1}{\sqrt{2\pi}}\int_0^\infty t\,e^{-(1+i\omega)t}dt =
\frac{1}{\sqrt{2\pi}}(1+i\omega)^{-2}$ (интеграл — вторая производная
экспоненты по параметру или повторное интегрирование по частям).

Парсеваль для $u$: $\norm{u}_2^2 = \int_0^\infty e^{-2t}dt = \frac12$;
$$
\norm{\hat u}_2^2 = \frac{1}{2\pi}\int_{-\infty}^{\infty}
\frac{d\omega}{1+\omega^2} = \frac{1}{2\pi}\cdot\pi = \frac12 .
$$
Равенство \eqref{eq:parseval} выполнено, и оно же гарантирует, что $\hat
u \in L_2 \setminus L_1$ (интеграл $\int|1+i\omega|^{-1}d\omega$
расходится), так что для $u$ теорема обращения \ref{thm:ft-inverse} не
применима — её регуляризованная версия из доказательства восстанавливает
$u$ как предел по норме $L_2$.

\begin{problem}\label{prob:ode}
Решить задачу Коши $y'' + y = 0$, $y(0) = 0$, $y'(0) = 1$ операторным
методом и проверить общую формулу \eqref{eq:ode-solution}.
\end{problem}

*Решение.* Правая часть здесь нулевая: $f \equiv 0$, $F(s) = 0$. По формуле
дифференцирования оригинала $\mathcal L\{y''\}(s) = s^2 Y(s) - s\,y(0) -
y'(0) = s^2 Y(s) - 1$, поэтому операторное уравнение
\eqref{eq:operator-equation}: $(s^2 + 1)Y(s) = 1$, то есть $Y(s) =
1/(s^2+1)$. По таблице изображений (или формуле Меллина) $1/(s^2+1)
\dashv \sin t$: действительно, $\int_0^\infty e^{-st}\sin t\,dt =
\operatorname{Im}\int_0^\infty e^{-(s-i)t}dt = \operatorname{Im}
\frac{1}{s-i} = \frac{1}{s^2+1}$ при $\operatorname{Re} s > 0$. По
теореме единственности \ref{thm:laplace-uniq} другого решения-оригинала
нет; проверка тривиальна: $(\sin t)'' + \sin t = 0$, $\sin 0 = 0$,
$(\sin)'(0) = 1$.

Общая формула: $n = 2$, $a_1 = 0$, $a_2 = 1$, $y_0 = 0$, $y_1 = 1$;
$A(s) = s^2 + 1$, $B(s) = y_0(s + a_1) + y_1 a_0 = 1$ (с $a_0 := 1$);
$Y(s) = (B(s)+F(s))/A(s) = 1/(s^2+1)$ — совпадает. Тот же ответ даёт и
разложение на простейшие дроби: $1/(s^2+1) = \frac{1}{2i}\bigl(
\frac{1}{s-i} - \frac{1}{s+i}\bigr)$, а каждая дробь есть изображение
экспоненты — это содержание второй теоремы разложения [30].

\begin{problem}\label{prob:haar-coeffs}
Найти коэффициенты системы Хаара: (а) кусочно-постоянной на диадической
сетке функции; (б) $f(t) = t\,e^{-t}$ на $[0,1)$; (в) сигнала $u(t) =
e^{-t}\mathbb I_{[0,\infty)}(t)$ на всей прямой.
\end{problem}

*Решение.* (а) Пусть $f$ кусочно-постоянна на интервалах уровня $J$.
Если носитель $\psi_{j,k}$ лежит внутри одного интервала постоянства,
то $\scal{f}{\psi_{j,k}} = 0$ (нулевой интеграл $\psi$). Отличны от нуля
лишь коэффициенты с носителем, пересекающим границы постоянства, — при
$j < J$ их конечное число (не более $J$ на каждую границу). Итог:
кусочно-постоянный сигнал представляется конечным рядом Хаара — это
точность на скачках, о которой идёт речь в замечании к теореме
\ref{thm:haar-parseval}.

(б) Для $f(t) = t e^{-t}$ на $[0,1)$ (продолженной нулём вне отрезка)
$\scal{f}{\psi_{j,k}} = 2^{j/2}\Bigl(\int_{k2^{-j}}^{(k+\frac12)2^{-j}} f -
\int_{(k+\frac12)2^{-j}}^{(k+1)2^{-j}} f\Bigr)$. Если $I_{j,k}$ целиком
лежит внутри $(0,1)$, вычитая значение в середине $c_{j,k} = (k +
\frac12)2^{-j}$ и пользуясь $\int\psi = 0$, получаем
$$
\bigl|\scal{f}{\psi_{j,k}}\bigr| \le 2^{j/2} \sup_{I_{j,k}}|f'|\cdot
2^{-j-1} \cdot 2^{-j} = O\bigl(2^{-3j/2}\bigr),
$$
так как $|f'| = |(1-t)e^{-t}|$ ограничена. У конца отрезка $t = 1$
(точки разрыва) оценка хуже — только $O(2^{-j/2})$: величина скачка
умножается на площадь носителя. Ряд Парсеваля \eqref{eq:haar-parseval}
сходится, поскольку $f \in L_2(\R)$; равенство можно проверить численно
в ноутбуке вопроса, его аналог для Фурье расходится равномерно у
излома — эффект Гиббса [34].

(в) Носитель $\psi_{j,k}$ при $k < 0$ лежит в $t < 0$, где $u = 0$;
при $k \ge 0$ прямая подстановка $u = e^{-t}$ даёт
$$
\scal{u}{\psi_{j,k}} = 2^{j/2}\Bigl(
\int_{k2^{-j}}^{(k+\frac12)2^{-j}} e^{-t}dt -
\int_{(k+\frac12)2^{-j}}^{(k+1)2^{-j}} e^{-t}dt \Bigr)
= 2^{j/2}\, e^{-k 2^{-j}} \bigl(1 - e^{-2^{-j-1}}\bigr)^2 .
$$
При $j \to +\infty$ выражение $\bigl(1 - e^{-2^{-j-1}}\bigr)^2 \sim
2^{-2j-2}$, так что коэффициенты затухают как $O(2^{-3j/2})$ — та же
скорость, что в пункте (б): сигнал гладок на полуоси, и затухание
определяется локальной гладкостью, а не скачком в нуле.

\part{Расширенная часть}

# Формула Пуассона и теорема Котельникова—Шеннона {#sec:sampling}

Две теоремы, замыкающие связь между непрерывным и дискретным миром.
Первая — тождество между суммой по сетке и суммой по спектру.

\begin{theorem}[Пуассона]\label{thm:poisson}
Пусть $f \in L_1(\R)$ непрерывна, а ряды
\begin{equation}\label{eq:poisson-cond}
\begin{aligned}
&\sum_{n\in\Z} |f(t + nT)| \ \text{сходится равномерно по } t\in[0,T],\\
&\sum_{k\in\Z} \Bigl|\hat f\Bigl(\omega + \tfrac{2\pi k}{T}\Bigr)\Bigr|
\ \text{сходится равномерно по } \omega .
\end{aligned}
\end{equation}
Тогда для всех $t$
\begin{equation}\label{eq:poisson}
\sum_{n\in\Z} f(t + nT) = \frac{\sqrt{2\pi}}{T} \sum_{k\in\Z}
\hat f\Bigl(\frac{2\pi k}{T}\Bigr)\, e^{2\pi i k t / T} .
\end{equation}
\end{theorem}

\begin{proof}
Периодизация $F(t) = \sum_n f(t+nT)$ корректно определена (равномерная
сходимость), $T$-периодична и непрерывна. Вычислим её коэффициенты Фурье
$c_k = \frac1T\int_0^T F(t)e^{-2\pi ikt/T}dt$; подставляя ряд и меняя
порядок интегрирования и суммирования (равномерная сходимость
оправдывает), получаем
$$
c_k = \frac1T \sum_n \int_0^T f(t+nT) e^{-2\pi ikt/T}dt =
\frac1T \int_{-\infty}^{\infty} f(\tau) e^{-2\pi ik\tau/T}d\tau =
\frac{\sqrt{2\pi}}{T}\,\hat f\Bigl(\frac{2\pi k}{T}\Bigr),
$$
после чего ряд $\sum_k|c_k|$ сходится по второму условию
\eqref{eq:poisson-cond}. По готовому факту из раздела \ref{sec:given}
(равномерная сходимость ряда Фурье при абсолютной суммируемости
коэффициентов) $F(t) = \sum_k c_k e^{2\pi ikt/T}$, что и есть
\eqref{eq:poisson}.
\end{proof}

\begin{remark}
Множитель $\sqrt{2\pi}/T$ — результат пересчёта нормировки [34] (там
$1/T$ при безмножительном образе): замена $\hat f_{\text{без}} =
\sqrt{2\pi}\,\hat f$ даёт $\frac1T\sqrt{2\pi}$. Достаточное условие для
\eqref{eq:poisson-cond} — оценки $|f(t)| \le C(1+|t|)^{-1-\delta}$ и
$|\hat f(\omega)| \le C(1+|\omega|)^{-1-\delta}$; первая обеспечивает
равномерную сходимость периодизации, вторая — суммируемость коэффициентов.
Формула Пуассона — инструментальное тождество: оно стоит за теоремой
отсчётов, за суммированием рядов и за дискретизацией в обработке сигналов
[34].
\end{remark}

\begin{theorem}[Котельникова—Шеннона]\label{thm:shannon}
Пусть $f \in L_2(\R)$, $\supp \hat f \subset [-\frac{\pi}{T},
\frac{\pi}{T}]$ и $\hat f \in L_1(\R)$. Тогда $f$ непрерывна, ограничена,
и для всех $t$
\begin{equation}\label{eq:shannon}
f(t) = \sum_{n\in\Z} f(nT)\,
\frac{\sin \frac{\pi}{T}(t - nT)}{\frac{\pi}{T}(t - nT)} ,
\end{equation}
причём ряд сходится равномерно (и по норме $L_2$).
\end{theorem}

\begin{proof}
По условию $\hat f$ компактно поддержана, поэтому $\hat f \in L_1\cap
L_2$ и теорема \ref{thm:ft-inverse} даёт непрерывное ограниченное
представление
$$
f(t) = \frac{1}{\sqrt{2\pi}} \int_{-\pi/T}^{\pi/T} \hat f(\omega)
e^{i\omega t}\,d\omega .
$$
Тот же факт доказывает \eqref{eq:poisson} в нашем частном случае: при
периоде $2\pi/T$ периодизация $\hat f$ есть $\hat f$ на полосе и нуль вне
её, так что сумма по сетке частот равна просто $\hat f$. Раскладываем
$\hat f$ на отрезке $[-\frac{\pi}{T}, \frac{\pi}{T}]$ в ряд Фурье по
ортонормированной системе $e_n(\omega) = \sqrt{\frac{T}{2\pi}}\,
e^{i n T \omega}$, $n \in \Z$ (ортонормированность — прямой счёт
интегралов). Коэффициенты определяются формулой обращения:
$$
\gamma_n = \scal{\hat f}{e_n} = \sqrt{\tfrac{T}{2\pi}}
\int_{-\pi/T}^{\pi/T} \hat f(\omega)\, e^{-i n T\omega}\,d\omega
= \sqrt{T}\; f(-nT),
$$
и по Парсевалю на отрезке $\sum_n T |f(nT)|^2 = \int |\hat f|^2 =
\norm{f}_2^2 < \infty$, то есть $\{f(nT)\} \in \ell_2$. Ряд Фурье
сходится к $\hat f$ по норме $L_2$ отрезка; подставляя частичную сумму в
интеграл обращения и оценивая разность через неравенство
Коши—Буняковского на отрезке, получаем сходимость равномерную по $t$.
Почленное интегрирование даёт
$$
f(t) = \frac{1}{\sqrt{2\pi}} \sum_{n\in\Z} \sqrt{T}\, f(-nT)
\sqrt{\tfrac{T}{2\pi}} \int_{-\pi/T}^{\pi/T} e^{i\omega(t + nT)}\,d\omega
= \sum_{n\in\Z} f(-nT)\, \frac{T}{\pi}\,
\frac{\sin \frac{\pi}{T}(t + nT)}{t + nT} ,
$$
так как $\int_{-\pi/T}^{\pi/T} e^{i\omega x}d\omega =
\frac{2\sin(\pi x / T)}{x}$ и $\sin\frac{\pi}{T}(t+nT) =
(-1)^n \sin\frac{\pi t}{T}$. Замена $n$ на $-n$ переписывает сумму в виде
\eqref{eq:shannon}: $\frac{T}{\pi}\cdot\frac{(-1)^n\sin(\pi t/T)}{t-nT} =
\frac{\sin\frac{\pi}{T}(t-nT)}{\frac{\pi}{T}(t-nT)}$. Сходимость ряда
равномерна по $t$: при фиксированном $t$ оценка
Коши—Буняковского
$$
\sum_n |f(nT)|\, \bigl|\mathrm{sinc}_T(t - nT)\bigr|
\le \Bigl(\sum_n |f(nT)|^2\Bigr)^{1/2}
\Bigl(\sum_n \mathrm{sinc}_T^2(t - nT)\Bigr)^{1/2}
$$
завершает дело, так как $\sum_n \mathrm{sinc}_T^2(t-nT) \le
\frac{T^2}{\pi^2}\sum_n \frac{1}{(t-nT)^2}$ равномерно ограничена по $t$
(это $T$-периодическая функция с суммируемым рядом по периоду). Предел
непрерывен и совпадает с $f$ в смысле $L_2$, а значит, всюду.
\end{proof}

\begin{remark}
Интерполяция \eqref{eq:shannon} работает, только если носитель образа
укладывается в полосу $[-\pi/T, \pi/T]$: при нарушении условия вклад
сдвинутых копий спектра (наложение, aliasing) искажает низкочастотную
часть, и восстановление невозможно никакой линейной интерполяцией [34].
Шаг $T$ соответствует частоте Найквиста $\pi/T$: теорема говорит, что
отсчётов с частотой не ниже удвоенной максимальной частоты сигнала
достаточно для точного восстановления. Для сигнала на $\Z_N$ аналогом
служит теорема об отсчётах [66]: спектр, сосредоточенный в полосе ширины
$m$, восстанавливается по $m$ отсчётам на периоде.
\end{remark}

# Принцип неопределённости Гейзенберга {#sec:heisenberg}

Теорема Планшереля говорит, что энергия сосредоточена одинаково во времени
и по частоте; принцип неопределённости — что она не может быть сосредоточена
в обоих представлениях одновременно. Для $f \in L_2(\R)$, $\norm{f}_2 =
1$, с конечными моментами
$$
\mu_t = \int t\,|f(t)|^2dt, \qquad \mu_\omega = \int \omega\,
|\hat f(\omega)|^2d\omega
$$
определим дисперсии $\sigma_t^2 = \int (t-\mu_t)^2|f(t)|^2dt$,
$\sigma_\omega^2 = \int(\omega - \mu_\omega)^2|\hat f(\omega)|^2d\omega$.

\begin{theorem}[Гейзенберг]\label{thm:heisenberg}
Если $f \in L_2(\R)$, $\norm{f}_2 = 1$, обе дисперсии конечны и
$\sqrt{|t|}\,f(t) \to 0$ при $|t| \to \infty$, то
\begin{equation}\label{eq:heisenberg}
\sigma_t \cdot \sigma_\omega \ge \frac12 .
\end{equation}
Равенство достигается только на модулированных гауссианах
\begin{equation}\label{eq:heisenberg-equality}
f(t) = e^{i\varphi}\, e^{i b t}\, \Bigl(\frac{c}{\pi}\Bigr)^{1/4}\,
e^{-c(t-\mu)^2/2}, \qquad c > 0,\ b, \mu, \varphi \in \R .
\end{equation}
\end{theorem}

\begin{proof}
Шаг 1: центрирование. Сдвиг по времени $f(t) \mapsto f(t-\mu_t)$ и
модуляция $f \mapsto e^{-i\mu_\omega t}f$ переводят $f$ в функцию с
нулевыми средними и не меняют ни дисперсий, ни $\norm{f}_2$ (пункты 2, 3,
7 теоремы \ref{thm:ft-props} для $L_2$-версии — по продолжению,
теорема \ref{thm:plancherel}). Поэтому достаточно доказать
\eqref{eq:heisenberg} при $\mu_t = \mu_\omega = 0$.

Шаг 2: ключевое тождество. Так как $f \to 0$ на бесконечности,
интегрирование по частям даёт
$$
\int_{-\infty}^{\infty} t\,\frac{d}{dt}|f(t)|^2\,dt
= \Bigl[t\,|f(t)|^2\Bigr]_{-\infty}^{\infty} -
\int_{-\infty}^{\infty}|f(t)|^2\,dt = -1
$$
(граничный член равен нулю по условию $\sqrt{|t|}\,f \to 0$). С другой
стороны, $\frac{d}{dt}|f|^2 = f'\overline{f} + f\overline{f'}$, поэтому
$$
1 = -\int t\,\frac{d}{dt}|f|^2\,dt = -2\,\operatorname{Re}
\int t f'(t)\,\overline{f(t)}\,dt
\le 2 \int |t f'(t)|\,|f(t)|\,dt .
$$

Шаг 3: оценка Коши—Буняковского. Применяем неравенство
Коши—Буняковского к паре $|f'|$ и $|t f|$:
$$
1 \le 2\,\norm{f'}_2\,\norm{t f}_2 .
$$
Осталось заметить, что $\norm{t f}_2 = \sigma_t$ (после центрирования),
а $\norm{f'}_2 = \sigma_\omega$: по пункту 5 теоремы
\ref{thm:ft-props} $\widehat{f'}(\omega) = i\omega\hat f(\omega)$, и
по Парсевалю (теорема \ref{thm:plancherel}) $\norm{f'}_2 =
\norm{\widehat{f'}}_2 = \bigl(\int\omega^2|\hat f|^2\bigr)^{1/2} =
\sigma_\omega$. Подставляя, получаем $1 \le 2\sigma_t\sigma_\omega$.

Шаг 4: случай равенства. Равенство в неравенстве Коши—Буняковского
возможно только при $|f'| = \lambda |t f|$ почти всюду, равенство в
переходе $-2\operatorname{Re}\int t f'\bar f \le 2\bigl|\int t f'\bar
f\bigr| \le 2\int|t f'|\,|f|$ — только при постоянном аргументе
подынтегральной функции и вещественном неположительном $\int t f'\bar f$
(после общего выбора фазы). Вместе это даёт $f'(t) = -c\,t f(t)$ с $c > 0$
действительным, откуда $f(t) = A e^{-ct^2/2}$. Нормировка $\norm{f}_2 = 1$
даёт $|A| = (c/\pi)^{1/4}$; возвращая сдвиг и модуляцию, получаем
\eqref{eq:heisenberg-equality}. Обратная подстановка показывает, что для
таких $f$ все неравенства обращаются в равенства.
\end{proof}

\begin{remark}
Константа $1/2$ относится к единичной нормировке \eqref{eq:fourier} и
стандартным отклонениям; у [34] (безмножительная нормировка) записано
то же неравенство для произведения дисперсий $\sigma_t^2\sigma_\omega^2
\ge 1/4$. Пересчёт здесь тривиален: частотная переменная в обеих
нормировках одна и та же, а $\sigma_\omega$ по $|\hat f|^2$ не зависит от
постоянного множителя перед образом, так что нижняя граница $1/2$ для
произведения стандартных отклонений сохраняется. Проверка на гауссиане
$c=1$: $f(t) = \pi^{-1/4}e^{-t^2/2}$ есть собственная функция
преобразования Фурье ($\hat f = f$ — вычисляется тем же дифференциальным
уравнением, что в теореме \ref{thm:ft-inverse}), $\sigma_t = \sigma_\omega
= 1/\sqrt2$ и $\sigma_t\sigma_\omega = 1/2$: равенство достигается.
Гауссианы — единственные экстремали: всякая система, претендующая на
одновременную локализацию во времени и частоте, упирается в
\eqref{eq:heisenberg} — это фундаментальное ограничение и для фильтров
[34], и для вейвлетов (вопрос 05).
\end{remark}

# Что дальше {#sec:next}

Система Хаара — простейший ортогональный вейвлет: функция $\psi$ с
компактным носителем, нулевым интегралом, и сдвиги с растяжениями которой
образуют базис $L_2(\R)$. Всё, чего у Хаара нет — гладкости и
частотной локализации, — покупается усложнением конструкции, и это
содержание вопроса 05 (методы вейвлет-анализа): условие допустимости
материнского вейвлета, непрерывное преобразование, кратномасштабный анализ
и его масштабирующая функция, конструкции Добеши с компактным носителем и
заданной гладкостью. Цепочка вложенных пространств $V_j$ уже встречалась
здесь в доказательстве теоремы \ref{thm:haar-parseval}: проекции на $V_j$
складывались в сходящийся ряд, а дополнения $W_j$ несли детали — в вопросе
05 та же схема становится определяющей.

В обратную сторону ведут два моста. Преобразование Лапласа восстанавливает
оригинал по изображению некорректно: малые возмущения $F$ на прямой
$\operatorname{Re} s = \sigma_1$ дают большие отклонения $f$ [14], и
регуляризация таких задач — предмет вопроса 02 (ML и некорректные задачи).
Дискретные преобразования работают как машина умножения матриц, и
устойчивость её обусловлена нормами — снова вопрос 09; вопрос 13
(численное дифференцирование и интегрирование) пользуется теми же ДПФ и
БПФ для сглаживания и быстрых свёрток.

# Вопросы, которые стоит ожидать {#sec:exam}

- Дайте определение оригинала и показателя роста. Где гарантирована
  абсолютная сходимость интеграла Лапласа? (Определение
  \ref{def:original}, теорема \ref{thm:laplace-conv}, замечание к ней.)
- Сформулируйте и докажите теорему обращения Фурье. Зачем в
  доказательстве гауссовская регуляризация? (Теорема
  \ref{thm:ft-inverse}; прямая подмена порядка интегрирования запрещена
  теоремой Фубини — множитель $e^{-\varepsilon^2\omega^2/4}$ делает
  подынтегральную функцию суммируемой.)
- Сформулируйте равенство Парсеваля в единичной нормировке. Что меняется
  при другой нормировке? (Теорема \ref{thm:plancherel}, замечание к ней;
  см. также замечания к теоремам \ref{thm:dft} и \ref{thm:poisson}.)
- Докажите теорему о свёртке и объясните, откуда берётся множитель
  $\sqrt{2\pi}$. (Теорема \ref{thm:convolution}.)
- Почему изображение Лапласа единственно? Что можно сказать о равенстве
  оригиналов почти всюду? (Теорема \ref{thm:laplace-uniq}.)
- Выведите формулу дифференцирования оригинала. Как решается задача Коши
  для $y^{(n)} + a_1 y^{(n-1)} + \dots + a_n y = f$? (Лемма
  \ref{lem:derivative-original}, следствие \ref{cor:ode-solution}.)
- Докажите, что система Хаара ортонормирована и полна. Где используется
  нулевой интеграл $\psi$? (Теоремы \ref{thm:haar-on},
  \ref{thm:haar-complete}; нулевой интеграл — в ортогональности разных
  масштабов и в затухании коэффициентов для гладких функций.)
- Чем сходимость ряда Хаара у скачка отличается от сходимости ряда Фурье?
  (Теорема \ref{thm:haar-parseval}, замечание о локализации и эффекте
  Гиббса к ней.)
- Сформулируйте теорему Котельникова—Шеннона. Что такое наложение спектров
  и частота Найквиста? (Теорема \ref{thm:shannon}, замечание к ней,
  определение \ref{def:bandlimited}.)
- Сформулируйте и докажите принцип неопределённости Гейзенберга. На чём
  достигается равенство? (Теорема \ref{thm:heisenberg} и замечание с
  проверкой на гауссиане.)
- Сколько операций требуют ДПФ, БПФ и БПХ? Почему БПХ устойчиво? (Теоремы
  \ref{thm:fft}, \ref{thm:fwt}.)
- Сформулируйте принцип неопределённости на $\Z_N$ и соотнесите его с
  непрерывной версией. (Замечание \ref{rem:uncertainty-zn} и теорема
  \ref{thm:heisenberg}.)

# Источники {#sec:sources}

Номера отвечают общему списку литературы; локаторы (главы, страницы)
живут в README вопроса.

1. [34] (Малла) — основной источник по непрерывному Фурье: обращение
   (теорема 2.1, доказательство через гауссову регуляризацию), свёртка
   (2.2), Парсеваль и Планшерель (2.3), формула Пуассона (2.4), принцип
   неопределённости Гейзенберга (2.5, доказательство Вейля), эффект
   Гиббса (утверждение 2.2), теорема выборки Уиттекера—Котельникова—Шеннона
   (3.1). Нормировка Фурье у [34] без множителя — все цитируемые формулы
   пересчитаны на единичную (замечание к теореме
   \ref{thm:convolution}).
2. [66] (Малоземов, Машарский) — основной источник по дискретной части:
   ДПФ (глава 1 §§2–5: лемма об ортогональности, обращение, Парсеваль,
   свёртка), теорема об отсчётах, схема Кули—Тьюки с подсчётом операций
   (глава 2 §§2–3), дискретный базис Хаара и быстрое преобразование Хаара
   (глава 2 §§4–5, схема (5.4) и подсчёт $2(N-1)$ сложений), принцип
   неопределённости на $\Z_N$ (глава 1 §10). Скалярное произведение у
   [66] линейно по первому аргументу — совпадает с нашей конвенцией;
   корень из единицы переименован из $\omega_N$ в $\varepsilon_N$,
   нормировка ДПФ без множителя сохранена.
3. [11] (Добеши) — §1.3.3: полные доказательства ортонормированности и
   полноты системы Хаара усреднением по диадическим интервалам;
   полнота у [11] сформулирована через плотность кусочно-постоянных
   функций, в конспекте использован эквивалентный усреднительный вариант
   (теорема \ref{thm:haar-complete}). Индексация у [11] обратная
   ($\psi_{m,n} = 2^{-m/2}\psi(2^{-m}x - n)$), пересчитана на нашу
   $\psi_{j,k} = 2^{j/2}\psi(2^j t - k)$; нормировка Фурье у [11]
   единичная, как у нас.
4. [30] (Гребенникова) — источник постановок части «Лаплас»: определение
   оригинала, показатель роста, теоремы об аналитичности и единственности,
   свойства 1–9, формула обращения Меллина, теоремы разложения,
   операторный метод для ОДУ и систем. Книга справочная и доказательств
   не даёт — все доказательства раздела \ref{sec:laplace} собственные;
   переменная $p$ у [30] заменена на $s$, формула обращения доказана при
   дополнительном условии интегрируемости образа на прямой (замечание к
   теореме \ref{thm:mellin}).
5. [42] (Галанин и др.) — глава 10: рамка интегральных преобразований с
   ядром и прикладная иллюстрация операторного метода (колебания в
   RLC-цепи как задача Коши для уравнения второго порядка). Использовано
   в мотивации.
6. [14] (Кабанихин) — замечания о роли преобразования Лапласа в обратных
   задачах: интерпретация показаний линейного прибора как уравнения
   свёртки первого рода и сведение нестационарных задач к спектральным в
   изображениях; цена обращения — некорректность. Одно-два предложения в
   мотивации и разделе «Что дальше».
