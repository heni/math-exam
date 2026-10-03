---
title: "Вопрос 14. Интервальное оценивание. Проверка статистических гипотез"
subtitle: "Конспект теории с полными доказательствами"
author: "Крохалев Е. М."
date: "2026"
---

# Постановка и мотивация

Сюжет ведёт тот же пример, что вопросы 08 и 12 (случайные величины и векторы;
закон больших чисел и центральная предельная теорема): измеряется величина $a$,
результат каждого измерения есть $\xi_k = a + \text{шум}$, измерения независимы и
одинаково распределены. Вопрос 12 ответил, **чему** сходится выборочное среднее и
превратил это в асимптотический доверительный интервал и формулу для числа
наблюдений. Остались два вопроса, и это предмет настоящего вопроса.

**Точность при конечном $N$.** Асимптотический интервал
$\overline X_N \pm z_{(1+\gamma)/2}\,\widehat\sigma_N/\sqrt N$ держит заявленный
уровень доверия $\gamma$ только в пределе; при конечном $N$ его фактическая
надёжность неизвестна. Хочется интервала, для которого вероятность накрыть $a$
равна $\gamma$ **при любом $N$** — по крайней мере в модели, где это вообще
возможно. Такая модель есть: нормальный закон шума.

**Согласие с гипотезой.** Практика ставит вопрос иначе: «верно ли, что среднее
показаний равно норме $a_0$?» — то есть требуется не число, а решение «да/нет»
со **заданной вероятностью ошибки**. Это задача проверки статистических
гипотез, и обе задачи оказываются двойственными: умея строить доверительный
интервал, получаешь критерий бесплатно, и наоборот.

Обе постановки — частные случаи общей проблемы принятия решения (вопрос 03,
принятие решения по функции потерь): оценка и критерий — решающие правила, и их
сравнивают по риску, здесь — по вероятности ошибки. Аппарат опирается на
предельные теоремы вопроса 12 и на гауссовские векторы вопроса 08: нормальная
выборка есть гауссов вектор с ковариацией $\sigma^2 I$, и главная теорема
раздела о доверительном оценивании — теорема о его проекциях.

# Выборка, оценки и их свойства

\begin{definition}\label{def:sample}
Пусть $\xi_1, \xi_2, \dots, \xi_N$ — независимые одинаково распределённые
случайные величины с распределением $P_\theta$, $\theta \in \Theta$ (англ.
\emph{random sample}). Вектор $\xi = (\xi_1, \dots, \xi_N)$ называется выборкой,
$N$ — объёмом выборки; любая борелевская функция $T(\xi)$ называется
статистикой (англ. \emph{statistic}), а статистика $\widehat\theta(\xi)$,
предназначенная для приближения $\theta$, — оценкой (англ. \emph{estimator}).
\end{definition}

Основные примеры оценок — выборочное среднее $\overline X_N = \frac1N\sum_k
\xi_k$, выборочная дисперсия $\widehat\sigma_N^{\,2} = \frac1N\sum_k (\xi_k -
\overline X_N)^2$ и исправленная (смещение снято делением на $N-1$) выборочная
дисперсия $s_N^2 = \frac{1}{N-1}\sum_k (\xi_k - \overline X_N)^2$.

Как сравнивать оценки? Естественный функционал качества — среднеквадратичная
ошибка $\E_\theta(\widehat\theta - \theta)^2$, и она раскладывается на две
разнородные части:
\begin{equation}\label{eq:mse}
\E_\theta(\widehat\theta - \theta)^2 = \Var_\theta \widehat\theta +
\bigl(\E_\theta \widehat\theta - \theta\bigr)^2 .
\end{equation}
Первое слагаемое — разброс, второе — квадрат смещения. Из \eqref{eq:mse}
вытекают три стандартных свойства.

\begin{definition}\label{def:unbiased}
Оценка $\widehat\theta$ называется несмещённой (англ. \emph{unbiased}), если
$\E_\theta \widehat\theta = \theta$ при всех $\theta \in \Theta$;
асимптотически несмещённой, если $\E_\theta \widehat\theta \to \theta$ при
$N \to \infty$.
\end{definition}

\begin{definition}\label{def:consistent}
Оценка $\widehat\theta_N$ называется состоятельной (англ. \emph{consistent}), если
$\widehat\theta_N \xrightarrow{\Prob} \theta$ при $N \to \infty$ при всех
$\theta \in \Theta$. Это статистическое имя сходимости по вероятности из
вопроса 12.
\end{definition}

\begin{definition}\label{def:efficient}
Несмещённая оценка $\widehat\theta^{\,*}$ называется эффективной (англ.
\emph{efficient}, наилучшей несмещённой), если
$\Var_\theta \widehat\theta^{\,*} = \inf_{\widehat\theta: \E_\theta
\widehat\theta = \theta} \Var_\theta \widehat\theta$ при всех
$\theta \in \Theta$. Если отношение дисперсии эффективной оценки к дисперсии
$\widehat\theta$ стремится к единице, оценка асимптотически эффективна.
\end{definition}

\begin{example}\label{ex:unbiased-s2}
Исправленная выборочная дисперсия $s_N^2$ несмещённа: так как $\E_\theta(\xi_1 -
\overline X_N)^2 = \sigma^2 \frac{N-1}{N}$ (разложение суммы квадратов отклонений
относительно $a$ в сумму квадратов относительно $\overline X_N$ и перекрёстный
член с нулевым средним), получаем $\E_\theta s_N^2 = \sigma^2$, тогда как
$\E_\theta \widehat\sigma_N^{\,2} = \sigma^2 \frac{N-1}{N}$: выборочная
дисперсия асимптотически несмещённа.
\end{example}

Граница, ниже которой дисперсия несмещённой оценки опуститься не может (при
условиях регулярности), задаётся информацией Фишера.

\begin{definition}\label{def:fisher}
Пусть семейство $\{P_\theta\}$ доминируемо с плотностью $p_\theta(x)$ по мере
$\mu$. Скалярная функция
\begin{equation}\label{eq:fisher}
I_N(\theta) = \Var_\theta\Bigl(\frac{\partial}{\partial\theta} \ln
p_\theta(\xi)\Bigr) = -\E_\theta\Bigl(\frac{\partial^2}{\partial\theta^2}
\ln p_\theta(\xi)\Bigr)
\end{equation}
называется информацией Фишера (англ. \emph{Fisher information}) выборки объёма $N$;
случайную величину $\frac{\partial}{\partial\theta}\ln p_\theta(\xi)$
называют вкладом (англ. \emph{score}).
\end{definition}

Равенство двух форм в \eqref{eq:fisher} — тождество, оно доказано ниже в
первом шаге доказательства теоремы \ref{thm:rao-cramer}. Для выборки из
одномерной плотности $g$ по независимости слагаемых
$I_N(\theta) = N \cdot \Var_\theta \frac{\partial}{\partial\theta} \ln
g(\xi_1;\theta)$: информация аддитивна по выборке, и граница дисперсий имеет
порядок $1/N$.

\begin{theorem}[неравенство Рао — Крамера]\label{thm:rao-cramer}
Пусть $\xi$ — выборка объёма $N$ с плотностью $p_\theta(x)$, $\theta \in \Theta
\subseteq \R$, функции $p_\theta(x)$ и $\widehat\theta(x)\,p_\theta(x)$
дифференцируемы по $\theta$ под знаком интеграла, носитель $\{p_\theta > 0\}$
не зависит от $\theta$, и $0 < I_N(\theta) < \infty$. Тогда для всякой
несмещённой оценки $\tau(\theta)$ с равномерно ограниченным вторым моментом
\begin{equation}\label{eq:rao-cramer}
\Var_\theta \widehat\theta \ \ge\ \frac{\bigl(\tau'(\theta)\bigr)^2}{I_N(\theta)}
\qquad \text{при всех } \theta \in \Theta,
\end{equation}
причём равенство достигается тогда и только тогда, когда
\begin{equation}\label{eq:equality-rc}
\frac{\partial}{\partial\theta} \ln p_\theta(x) = C(\theta)\,
\bigl(\widehat\theta(x) - \tau(\theta)\bigr) \quad
\text{почти наверное по мере } P_\theta .
\end{equation}
\end{theorem}

У [06] доказан случай $\tau(\theta) = \theta$; у
[07] — общий, для $\tau(\theta)$, — его и воспроизводим.

\begin{proof}
Шаг 1: среднее вклада равно нулю, а вторая форма информации совпадает с
первой. Дифференцируя тождество $\int p_\theta(x)\,\mu(dx) = 1$ под знаком
интеграла, получаем
\begin{equation}\label{eq:score-zero}
\E_\theta \frac{\partial}{\partial\theta} \ln p_\theta(\xi) = \int
\frac{\partial p_\theta(x)}{\partial\theta}\,\mu(dx) = \frac{\partial}{\partial
\theta} \int p_\theta(x)\,\mu(dx) = 0 .
\end{equation}
Продифференцируем ещё раз и умножим на $-1$:
$$
-\E_\theta \frac{\partial^2}{\partial\theta^2} \ln p_\theta(\xi)
= \int \biggl( \frac{(p'_\theta(x))^2}{p_\theta(x)} -
\frac{p''_{\theta\theta}(x)}{1} \cdot \frac{p_\theta(x)}{p_\theta(x)} \biggr)
\mu(dx) = \E_\theta \Bigl(\frac{\partial}{\partial\theta}\ln
p_\theta(\xi)\Bigr)^2 ,
$$
так как $\int p''_{\theta\theta}(x)\,\mu(dx) = \frac{\partial^2}{\partial
\theta^2}\int p_\theta\, \mu(dx) = 0$. Вместе с \eqref{eq:score-zero} это и
есть равенство двух форм в \eqref{eq:fisher}.

Шаг 2: скалярное произведение оценки с вкладом. Несмещённость $\E_\theta
\widehat\theta = \tau(\theta)$ есть тождество $\int \widehat\theta(x)\,
p_\theta(x)\,\mu(dx) = \tau(\theta)$. Дифференцируя его по $\theta$ под знаком
интеграла (условия теоремы это разрешают), получаем
$$
\tau'(\theta) = \int \widehat\theta(x) \, \frac{\partial
p_\theta(x)}{\partial\theta}\,\mu(dx) = \E_\theta \Bigl( \widehat\theta \,
\frac{\partial}{\partial\theta}\ln p_\theta(\xi) \Bigr) .
$$
Вычитая из этого равенство $\tau(\theta) \cdot 0 = 0$ с множителем
\eqref{eq:score-zero}, приходим к ключевому тождеству:
\begin{equation}\label{eq:key-rc}
\E_\theta \Bigl( (\widehat\theta - \tau(\theta)) \,
\frac{\partial}{\partial\theta}\ln p_\theta(\xi) \Bigr) = \tau'(\theta) .
\end{equation}

Шаг 3: неравенство Коши — Буняковского. Применяем его к паре
\eqref{eq:key-rc}:
$$
\bigl(\tau'(\theta)\bigr)^2 \le \Var_\theta \widehat\theta \cdot
\E_\theta \Bigl(\frac{\partial}{\partial\theta}\ln p_\theta(\xi)\Bigr)^2 =
\Var_\theta \widehat\theta \cdot I_N(\theta),
$$
что и есть \eqref{eq:rao-cramer}. Равенство в неравенстве Коши — Буняковского
достигается тогда и только тогда, когда векторы линейно зависимы:
$\frac{\partial}{\partial\theta}\ln p_\theta = C(\theta)\,(\widehat\theta -
\tau(\theta))$ почти наверное, что и есть \eqref{eq:equality-rc}.
\end{proof}

\begin{corollary}[критерий эффективности]\label{cor:efficient}
Несмещённая оценка $\widehat\theta$ эффективна для $\tau(\theta)$ тогда и
только тогда, когда выполнено \eqref{eq:equality-rc}; при этом
$C(\theta) = I_N(\theta)/\tau'(\theta)$ и $\Var_\theta \widehat\theta =
(\tau'(\theta))^2 / I_N(\theta)$ [07].
\end{corollary}

\begin{example}\label{ex:rao-bernoulli}
Пусть $\xi_k \sim \mathrm{Bin}(1, \theta)$. Тогда $\ln p_\theta(x) =
\sum_k \bigl( x_k \ln\theta + (1-x_k)\ln(1-\theta) \bigr)$, вклад равен
$\frac{\sum_k x_k}{\theta} - \frac{N - \sum_k x_k}{1-\theta}$, и прямая
подстановка даёт $I_N(\theta) = N/(\theta(1-\theta))$. Оценка $\overline X_N$
несмещённа для $\theta$, и $\Var_\theta \overline X_N = \theta(1-\theta)/N =
1/I_N(\theta)$: неравенство обращается в равенство, $\overline X_N$ эффективна.
В силу критерия \ref{cor:efficient} она единственная эффективная (с точностью
до совпадения почти наверное).
\end{example}

\begin{remark}\label{rem:rao-conditions}
Посылки существенны. Без независимости носителя от $\theta$ тождество
\eqref{eq:score-zero} ломается: для равномерного распределения $U(0,\theta)$
оценка $\frac{N+1}{N} X_{(N)}$ несмещённа и имеет дисперсию порядка
$1/N^2$ — быстрее любой границы $c(\theta)/N$. Без дифференцируемости под
знаком интеграла шаг 2 невозможен. Зато в доказательстве нигде не
использовалась непрерывность $\xi_k$: всё верно и для дискретных выборок с
суммами вместо интегралов [06].
\end{remark}

## Два метода построения оценок

Метод моментов (англ. \emph{method of moments}, Пирсон) приравнивает эмпирические моменты теоретическим:
для модели $\E_\theta \xi_1 = m(\theta)$ берёт решение уравнения $m(\theta) =
\overline X_N$. Максимальное правдоподобие (англ. \emph{maximum likelihood}, Фишер) берёт точку максимума
функции правдоподобия $L(\theta) = p_\theta(\xi)$, то есть решение уравнения
правдоподобия $\frac{\partial}{\partial\theta} \ln L(\theta) = 0$ [06]. Оба метода состоятельны под стандартными условиями; для оценки
максимального правдоподобия (ОМП) верен предельный результат, который делает
её рабочей лошадкой интервального оценивания.

\begin{theorem}[асимптотическая нормальность ОМП]\label{thm:mle-an}
Пусть выполнены условия регулярности теоремы \ref{thm:rao-cramer} плюс
условия второго порядка (существование и ограниченность третьих
производных $\ln p_\theta$, единственность решения уравнения правдоподобия).
Тогда ОМП $\widehat\theta^{\,\text{МП}}$ состоятельна, асимптотически
эффективна и
\begin{equation}\label{eq:mle-an}
\sqrt{N}\,\bigl(\widehat\theta^{\,\text{МП}} - \theta\bigr)
\ \xrightarrow{d}\ \mathcal N\bigl(0,\; 1/i(\theta)\bigr),
\qquad i(\theta) = \Var_\theta \tfrac{\partial}{\partial\theta}\ln
g(\xi_1;\theta),
\end{equation}
где $g$ — одномерная плотность слагаемого выборки [06].
\end{theorem}

Идея доказательства — разложение уравнения правдоподобия в точке $\theta$
по формуле Тейлора с остатком в средней точке и предельный переход по закону
больших чисел в знаменателе; полное доказательство — в расширенной части,
раздел \ref{sec:mle-an-proof}. Следствие для интервалов: заменяя в
\eqref{eq:mle-an} неизвестную $i(\theta)$ состоятельной оценкой и применяя
лемму Слуцкого, получаем асимптотический доверительный интервал
$\widehat\theta^{\,\text{МП}} \pm z_{(1+\gamma)/2}\,/\sqrt{N\,
\widehat i_N}$ — это и есть метод построения асимптотических интервалов из
следующего раздела в общей форме.

# Доверительное оценивание

\begin{notation}
Далее $\gamma \in (0,1)$ — уровень доверия (англ. \emph{confidence level}); буква
$\alpha$ зарезервирована за уровнем значимости и в доверительном оценивании
встречается только как $\alpha = 1 - \gamma$. Символом $z_\beta$ обозначаем
$\beta$-квантиль распределения $\mathcal N(0,1)$ (соглашение вопроса 12);
символами $\chi^2_{\beta;\,k}$, $t_{\beta;\,k}$, $F_{\beta;\,k,m}$ —
$\beta$-квантили распределений хи-квадрат, Стьюдента и Фишера с указанными
числами степеней свободы. Двусторонний интервал уровня $\gamma$ строится по
квантилям $\beta_{\pm} = (1 \pm \gamma)/2$.
\end{notation}

\begin{remark}\label{rem:quantile-convention}
Квантиль $z_\beta$ — точка, слева от которой масса $\beta$. У [06]
табулированы «верхние $\alpha$-пределы» — точки, \textbf{справа} от которых
масса $\alpha$; это наши квантили с $\beta = 1 - \alpha$, и переход между
соглашениями формальный: $\chi^2_{\alpha;\,k}$ у [06] есть $\chi^2_{1-\alpha;\,k}$
здесь. У [24] «надёжность $\gamma$» — это уровень доверия. Объём выборки мы
обозначаем $N$ (как в вопросе 12), тогда как все книги блока пишут $n$;
перенос формул однозначен.
\end{remark}

Точечная оценка даёт число, но не отвечает, насколько ему можно доверять.
Ответ — интервал со случайными границами, накрывающий параметр с
гарантированной вероятностью.

\begin{definition}\label{def:ci}
Пусть $\{P_\theta, \theta \in \Theta\}$ — параметрическое семейство, $\Theta
\subseteq \R$. Пара статистик $(T_1(\xi), T_2(\xi))$ называется доверительным
интервалом уровня доверия $\gamma$ для $\theta$, если при всех $\theta \in
\Theta$
\begin{equation}\label{eq:ci-def}
\Prob_\theta \bigl( T_1(\xi) < \theta < T_2(\xi) \bigr) \ge \gamma ;
\end{equation}
если вероятность равна $\gamma$ при всех $\theta$, интервал называется точным.
Последовательность пар статистик $(T_{1N}, T_{2N})$ называется асимптотическим
доверительным интервалом уровня $\gamma$, если $\liminf_N
\Prob_\theta (T_{1N} < \theta < T_{2N}) \ge \gamma$ при всех $\theta$; при
равенстве предела $\gamma$ — точным асимптотическим [07]. Подмножество $S(\xi) \subseteq \Theta$ называется
доверительной областью уровня $\gamma$, если $\Prob_\theta(\theta \in S(\xi))
\ge \gamma$ [07].
\end{definition}

В неравенстве \eqref{eq:ci-def} случайными являются границы, а не параметр:
«интервал накрывает $\theta$» — краткая запись события $T_1 < \theta < T_2$.
Отсюда же ясно, почему требуют верность при **всех** $\theta$: конспект
обязан гарантировать уровень, не зная истинного значения.

## Метод центральной статистики

Общий способ построения даёт центральная статистика (англ. *pivot*) —
случайная величина, распределение которой известно и не зависит от параметра,
сама же она параметром зависит.

\begin{proposition}[метод центральной статистики]\label{thm:pivot}
Пусть $G(\xi, \theta)$ — такая функция, что её распределение при каждом
$\theta$ одно и то же, известное, и пусть $g_1, g_2$ — квантили её
распределения с $g_1 < g_2$. Тогда множество
\begin{equation}\label{eq:pivot-region}
S(\xi) = \{ \theta \in \Theta : g_1 \le G(\xi, \theta) \le g_2 \}
\end{equation}
является доверительной областью уровня $\gamma = \Prob(g_1 \le G \le g_2)$
[07].
\end{proposition}

\begin{proof}
Так как распределение $G(\xi, \theta)$ при $\theta$ совпадает с опорным,
$\Prob_\theta(g_1 \le G(\xi,\theta) \le g_2) = \Prob(g_1 \le G \le g_2) = \gamma$,
а событие $g_1 \le G(\xi,\theta) \le g_2$ есть в точности $\theta \in S(\xi)$.
\end{proof}

Если $G(\xi, \cdot)$ непрерывна и монотонна по $\theta$, множество
\eqref{eq:pivot-region} — интервал $(T_1, T_2)$ с явными границами. Два
примера — один точный, один асимптотический.

\begin{example}[$U(0,\theta)$, точный]\label{ex:ci-uniform}
Пусть $\xi_k \sim U(0,\theta)$, $\theta > 0$, и $X_{(N)} = \max_k \xi_k$.
Статистика $G(\xi,\theta) = X_{(N)}/\theta$ имеет распределение с плотностью
$N x^{N-1}$ на $[0,1]$ — распределение не зависит от $\theta$. Взяв $g_2 = 1$
и $g_1 = (1-\gamma)^{1/N}$ (квантили этого распределения), получаем точный
интервал
$$
X_{(N)} \ <\ \theta \ <\ X_{(N)} \,(1-\gamma)^{-1/N},
$$
его длина имеет порядок $1/N$ — быстрее, чем $1/\sqrt N$ у средних [07].
\end{example}

\begin{example}[доля, асимптотический и «точный по нормальному счёту»]\label{ex:ci-share}
Пусть $\xi_k \sim \mathrm{Bin}(1, p)$ и $\widehat p = \overline X_N$. По
центральной предельной теореме (вопрос 12)
$\sqrt N (\widehat p - p)/\sqrt{p(1-p)} \xrightarrow{d} \mathcal N(0,1)$,
и замена $p(1-p)$ состоятельной оценкой $\widehat p(1-\widehat p)$ легальна
по лемме Слуцкого. Отсюда асимптотический интервал
\begin{equation}\label{eq:ci-share-wald}
\widehat p \ \pm\ z_{(1+\gamma)/2}\,\sqrt{\frac{\widehat p(1-\widehat
p)}{N}} .
\end{equation}
Интервал \eqref{eq:ci-share-wald} называют вальдовым; он может выйти за
границы $[0,1]$ при экстремальных $\widehat p$. Решая же неравенство
$\abs{\widehat p - p} \le z\,\sqrt{p(1-p)/N}$ как квадратное относительно
$p$, получают интервал [24]
\begin{equation}\label{eq:ci-share-exact}
p_{1,2} = \frac{N}{z^2 + N}\left( \widehat p + \frac{z^2}{2N} \mp z
\sqrt{\frac{\widehat p(1-\widehat p)}{N} + \frac{z^2}{4N^2}} \right), \qquad z
= z_{(1+\gamma)/2},
\end{equation}
который всегда лежит в $(0,1)$; при $N \to \infty$ оба интервала
асимптотически совпадают.
\end{example}

## Нормальная модель: точные интервалы при любом объёме

Точные интервалы возможны, когда распределение выборки известно с точностью
до конечномерного параметра. Рабочий случай — нормальный закон шума датчика:
$\xi_k \sim \mathcal N(a, \sigma^2)$, оба параметра неизвестны. Всё
интервальное оценивание и все критерии в этой модели опираются на одну
теорему о совместном распределении выборочного среднего и выборочной
дисперсии.

\begin{theorem}[о нормальной выборке]\label{thm:normal-sample}
Пусть $\xi_1, \dots, \xi_N$ — выборка из $\mathcal N(a, \sigma^2)$,
$\overline X_N$ и $s_N^2$ — выборочное среднее и исправленная выборочная
дисперсия. Тогда

1. $\overline X_N$ и $s_N^2$ независимы;
2. $\dfrac{(N-1)\,s_N^2}{\sigma^2} \sim \chi^2_{N-1}$;
3. $\overline X_N \sim \mathcal N\bigl(a, \sigma^2/N\bigr)$.

[06].
\end{theorem}

\begin{proof}
Центрируем и нормируем: $\eta_k = (\xi_k - a)/\sigma$ — независимы и имеют
распределение $\mathcal N(0,1)$, то есть $\eta = (\eta_1, \dots, \eta_N)$ —
стандартный гауссов вектор. Выборочные характеристики выражаются через него:
$$
\overline X_N = \sigma \overline\eta + a, \qquad
s_N^2 = \frac{\sigma^2}{N-1} \sum_{k=1}^N (\eta_k - \overline\eta)^2
= \frac{\sigma^2}{N-1} \Bigl( \sum_{k=1}^N \eta_k^2 - N\,\overline\eta^{\,2}
\Bigr).
$$
Возьмём ортогональную матрицу $C$, первая строка которой
$(1/\sqrt N, \dots, 1/\sqrt N)$, и повернём вектор: $\zeta = C\eta$. По
теореме о сохранении гауссовости при ортогональных преобразованиях (вопрос
08) $\zeta$ — снова стандартный гауссов вектор. Первая компонента
$\zeta_1 = \sqrt N\,\overline\eta$, а из ортогональности
$\sum_k \eta_k^2 = \sum_k \zeta_k^2$, так что
$$
s_N^2 = \frac{\sigma^2}{N-1} \sum_{k=2}^N \zeta_k^2, \qquad
\overline X_N = \frac{\sigma}{\sqrt N}\,\zeta_1 + a .
$$
Первое представление — функция от $(\zeta_2, \dots, \zeta_N)$, второе —
функция от $\zeta_1$; по независимости компонент гауссового вектора
(некоррелированность компонент $\zeta$ следует из $C C\T = I$) $\overline X_N$
и $s_N^2$ независимы — пункт 1. Пункт 2: сумма квадратов $N-1$ независимых
$\mathcal N(0,1)$ по определению имеет распределение $\chi^2_{N-1}$.
Пункт 3: $\overline X_N$ — линейная комбинация независимых нормальных
величин с коэффициентами $1/N$, а $\E \overline X_N = a$, $\Var \overline X_N
= \sigma^2/N$.
\end{proof}

Поворот с первой строкой $(1/\sqrt N, \dots, 1/\sqrt N)$ — классическое
преобразование Хелмерта; оно «отделяет» среднее от остальной суммы квадратов.
Именно пункт 1 — содержательный: из независимости среднего и дисперсии
следует, что замена неизвестной $\sigma$ выборочной $s_N$ в статистике
Стьюдента законна.

\begin{definition}\label{def:student-fisher}
Случайная величина $\zeta = \xi/\sqrt{\eta/k}$ с независимыми $\xi \sim
\mathcal N(0,1)$ и $\eta \sim \chi^2_k$ имеет распределение Стьюдента с $k$
степенями свободы; величина $\zeta = (\xi/k)/(\eta/m)$ с независимыми $\xi
\sim \chi^2_k$, $\eta \sim \chi^2_m$ — распределение Фишера — Снедекора с
$(k, m)$ степенями свободы [07].
Плотность распределения $\chi^2_k$:
$p(x) = x^{k/2-1} e^{-x/2} \big/ \bigl(2^{k/2} \Gamma(k/2)\bigr)$ при $x \ge
0$ [07].
\end{definition}

\begin{corollary}[статистика Стьюдента]\label{cor:student}
При выборке из $\mathcal N(a, \sigma^2)$
\begin{equation}\label{eq:student}
T = \frac{(\overline X_N - a)}{s_N}\,\sqrt{N} \ \sim\ t_{N-1}.
\end{equation}
[06].
\end{corollary}

\begin{proof}
Разделим числитель и знаменатель на $\sigma$:
$T = \bigl((\overline X_N - a)\sqrt N/\sigma\bigr)\big/\sqrt{s_N^2/\sigma^2}$.
Числитель распределён $\mathcal N(0,1)$, подкоренное выражение $s_N^2/\sigma^2
= \chi^2_{N-1}/(N-1)$, и они независимы по теореме
\ref{thm:normal-sample} — это в точности представление Стьюдента из
определения \ref{def:student-fisher}.
\end{proof}

Теперь четыре классических интервала — примеры метода
\ref{thm:pivot} с центральными статистиками из теоремы
\ref{thm:normal-sample}. Формы записи совпадают с [06]
и с рецептами [24], откуда они переходят в задачники.

\begin{theorem}[четыре доверительных интервала нормальной модели]\label{thm:four-intervals}
Пусть $\xi_1, \dots, \xi_N$ — выборка из $\mathcal N(a, \sigma^2)$,
$\beta_{\pm} = (1 \pm \gamma)/2$. Тогда точные доверительные интервалы уровня
$\gamma$:

1. \textbf{среднее $a$, $\sigma$ известно} (центральная статистика — из п. 3
   теоремы \ref{thm:normal-sample}):
   $$
   \overline X_N \ \pm\ z_{\beta_{+}}\,\frac{\sigma}{\sqrt N};
   $$
2. \textbf{среднее $a$, $\sigma$ неизвестно} (статистика \eqref{eq:student}):
   $$
   \overline X_N \ \pm\ t_{\beta_{+};\,N-1}\,\frac{s_N}{\sqrt N};
   $$
3. \textbf{дисперсия $\sigma^2$, $a$ известно}: с $\widehat\sigma_a^2 = \frac1N
   \sum_k (\xi_k - a)^2$ величина $N\widehat\sigma_a^2/\sigma^2 \sim
   \chi^2_N$, откуда
   $$
   \Bigl( \tfrac{N\widehat\sigma_a^2}{\chi^2_{\beta_{+};\,N}},
         \tfrac{N\widehat\sigma_a^2}{\chi^2_{\beta_{-};\,N}} \Bigr);
   $$
4. \textbf{дисперсия $\sigma^2$, $a$ неизвестно} (центральная статистика — из п. 2
   теоремы \ref{thm:normal-sample}):
   $$
   \Bigl( \tfrac{(N-1)\,s_N^2}{\chi^2_{\beta_{+};\,N-1}},
         \tfrac{(N-1)\,s_N^2}{\chi^2_{\beta_{-};\,N-1}} \Bigr).
   $$
\end{theorem}

\begin{proof}
Пункт 1: $\sqrt N(\overline X_N - a)/\sigma \sim \mathcal N(0,1)$, а
$\Prob(\abs{\mathcal N(0,1)} \le z_{\beta_{+}}) = \gamma$. Пункт 2: по следствию
\ref{cor:student} $\abs{T} \le t_{\beta_{+};\,N-1}$ с вероятностью $\gamma$,
а неравенство $\abs{\overline X_N - a}\,\sqrt N/s_N \le t_{\beta_{+};\,N-1}$
равносильно записанному интервалу. Пункт 3: $(\xi_k - a)/\sigma$ независимы
и $\mathcal N(0,1)$, сумма их квадратов по определению $\sim \chi^2_N$.
Пункт 4: $\Prob\bigl(\chi^2_{\beta_{-};\,N-1} \le (N-1)s_N^2/\sigma^2 \le
\chi^2_{\beta_{+};\,N-1}\bigr) = \gamma$ по пункту 2 теоремы
\ref{thm:normal-sample}, и остаётся разрешить двойное неравенство
относительно $\sigma^2$.
\end{proof}

\begin{remark}\label{rem:width-price}
Цена надёжности — ширина. Пример [06] с выборкой $N = 12$
показывает это в числах: интервал для $a$ при $\gamma = 0{,}90$ равен
$(1{,}68;\ 3{,}00)$, а при $\gamma = 0{,}99$ — уже $(1{,}19;\ 3{,}49)$.
t-интервал пункта 2 шире z-интервала пункта 1 при том же $\gamma$ — плата
за неизвестность $\sigma$; разница заметна при малых $N$ и тает по мере
роста (квантили $t_{\beta_{+};\,N-1}$ убывают к $z_{\beta_{+}}$).
\end{remark}

## Две выборки

Пусть независимые выборки $\xi_1, \dots, \xi_N$ и $\eta_1, \dots, \eta_M$ взяты
из $\mathcal N(a_1, \sigma_1^2)$ и $\mathcal N(a_2, \sigma_2^2)$. Теорема
\ref{thm:normal-sample}, применённая к каждой выборке, и независимость пар
$(\overline X_N, s_{1N}^2) \perp (\overline Y_M, s_{2M}^2)$ дают три стандартных
интервала [06]:

\begin{proposition}[интервалы для сравнения двух нормальных выборок]\label{thm:two-sample}
При равных дисперсиях $\sigma_1 = \sigma_2 = \sigma$ (неизвестной) интервал
для разности средних $a_1 - a_2$:
$$
(\overline X_N - \overline Y_M) \ \pm\ t_{\beta_{+};\,N+M-2}\; s_{\text{обш}}
\sqrt{\tfrac1N + \tfrac1M}, \qquad
s_{\text{обш}}^2 = \frac{(N-1)s_{1N}^2 + (M-1)s_{2M}^2}{N+M-2},
$$
где $s_{\text{обш}}^2$ — объединённая оценка $\sigma^2$; если $\sigma$
известно, множитель $t_{\beta_{+};\,N+M-2}\,s_{\text{обш}}$ заменяется на
$z_{\beta_{+}}\,\sigma$. Интервал для отношения дисперсий:
$$
\Bigl( \tfrac{s_{1N}^2}{s_{2M}^2}\,F_{\beta_{-};\,M-1,\,N-1},
      \tfrac{s_{1N}^2}{s_{2M}^2}\,F_{\beta_{+};\,M-1,\,N-1} \Bigr),
$$
так как $\dfrac{s_{1N}^2/\sigma_1^2}{s_{2M}^2/\sigma_2^2} \sim F_{N-1,\,M-1}$.
\end{proposition}

Выводы повторяют доказательство теоремы \ref{thm:four-intervals}:
нормированные разности имеют распределение Стьюдента с объединённой оценкой в
знаменателе [06], а отношение исправленных дисперсий
— представление Фишера.

## Двойственность интервалов и критериев

Построение доверительной области и проверка гипотезы — взаимно обратные
операции; это наблюдение связывает обе половины вопроса.

\begin{proposition}[двойственность]\label{thm:duality}
Пусть $\{P_\theta, \theta \in \Theta\}$ — параметрическое семейство.

1. Если $S(\xi)$ — доверительная область уровня $\gamma = 1 - \varepsilon$
   для $\theta$, то для всякой точечной гипотезы $H_0: \theta = \theta_0$
   множество $\widetilde S_{\theta_0} = \{ \xi : \theta_0 \notin S(\xi) \}$
   является критерием уровня значимости $\varepsilon$.
2. Если для всякого $\theta_0$ построен критерий $\widetilde S_{\theta_0}$
   уровня $\varepsilon$ для $H_0: \theta = \theta_0$, то $S(\xi) = \{ \theta
   \in \Theta : \xi \notin \widetilde S_\theta \}$ — доверительная область
   уровня $\gamma = 1 - \varepsilon$.

[07].
\end{proposition}

\begin{proof}
1. $\Prob_{\theta_0}(\xi \in \widetilde S_{\theta_0}) = \Prob_{\theta_0}
(\theta_0 \notin S(\xi)) = 1 - \Prob_{\theta_0}(\theta_0 \in S(\xi)) \le 1 -
\gamma = \varepsilon$.
2. Для любого $\theta_0$: $\Prob_{\theta_0}(\theta_0 \in S(\xi)) =
\Prob_{\theta_0}(\xi \notin \widetilde S_{\theta_0}) = 1 -
\Prob_{\theta_0}(\xi \in \widetilde S_{\theta_0}) \ge 1 - \varepsilon$.
\end{proof}

Двойственность объясняет, почему двусторонний критерий для среднего имеет
вид «отклонить, если $\theta_0$ вне интервала»: это критерий пункта 1 по
интервалу пункта 2 теоремы \ref{thm:four-intervals}. Она же — причина, по
которой нет единого «наилучшего» двустороннего критерия: двусторонние
интервалы можно строить разными парами квантилей, и у двойственных критериев
разная мощность (см. замечание \ref{rem:two-sided}).

# Проверка статистических гипотез

## Постановка и базовые понятия

\begin{definition}\label{def:hypothesis}
Суждение о неизвестном распределении наблюдения $\xi$ называется статистической
гипотезой (англ. \emph{statistical hypothesis}). В параметрической модели
$\{P_\theta, \theta \in \Theta\}$ гипотеза $H_0: \theta \in \Theta_0$
называется простой, если $\Theta_0$ — одна точка (гипотеза однозначно
восстанавливает распределение), и сложной в противном случае; гипотеза
$H_1: \theta \in \Theta_1$ с $\Theta_0 \cap \Theta_1 = \varnothing$ —
конкурирующей (альтернативой). Наблюдение $\xi$ — выборка объёма $N$
(определение \ref{def:sample}).
\end{definition}

\begin{definition}\label{def:test}
Критерием (англ. \emph{test}) называется измеримое множество $S$ выборочного
пространства (множество $S$ называют также критической областью): наблюдение
$x \in S$ влечёт отклонение $H_0$ в пользу $H_1$, $x \notin S$ — неотклонение. Критерий — решающее правило из вопроса 03,
действие два. Ошибкой первого рода (англ. \emph{type I error}) называется
отклонение верной $H_0$; ошибкой второго рода (англ. \emph{type II error}) —
неотклонение неверной $H_0$ [06].
\end{definition}

\begin{definition}\label{def:level-power}
Функция $\beta(\theta, S) = \Prob_\theta(\xi \in S)$ называется функцией
мощности критерия $S$ (англ. \emph{power function}); её значения $\beta(\theta,
S)$ при $\theta \in \Theta_1$ — мощностью, а
\begin{equation}\label{eq:size}
\alpha(S) = \sup_{\theta \in \Theta_0} \beta(\theta, S)
\end{equation}
— размером критерия. Критерий имеет уровень значимости $\alpha$ (англ.
\emph{significance level}), если $\alpha(S) \le \alpha$ [07].
\end{definition}

Парадокс постановки: при фиксированном объёме выборки уменьшение вероятности
одной ошибки оборачивается ростом другой, поэтому правило строят так: жёстко
контролируют ошибку первого рода (уровень $\alpha$, обычно $0{,}05$ или
$0{,}01$), а в этом классе ищут критерий с максимальной мощностью.

\begin{definition}\label{def:ump}
Критерий $S$ уровня значимости $\varepsilon$ называется равномерно наиболее
мощным (р.н.м.к., англ. \emph{uniformly most powerful test}), если
$\alpha(S) \le \varepsilon$ и $\beta(\theta, S) \ge \beta(\theta, R)$ при
всех $\theta \in \Theta_1$ для любого другого критерия $R$ уровня
$\varepsilon$ [07].
\end{definition}

\begin{definition}\label{def:pvalue}
Если критерий строится по статистике $T$ большим значениям ($S = \{T \ge
c\}$), то p-value (англ. \emph{p-value}) наблюдения $x$ — масса распределения
$T$ при $\theta_0$ на хвосте $[T(x), +\infty)$: $p = \Prob_{\theta_0}(T \ge
T(x))$ [06]. Гипотезу с уровнем $\alpha$ отклоняют ровно тогда,
когда $p \le \alpha$; p-value — наименьший уровень, при котором наблюдение
ещё отклоняет $H_0$.
\end{definition}

## Лемма Неймана — Пирсона

Для простой гипотезы против простой альтернативы р.н.м.к. строится явно —
это и есть лемма Неймана — Пирсона. В наборе она есть только у [07];
у [06] слова «Нейман» нет вовсе.

\begin{theorem}[Нейман — Пирсон]\label{thm:np}
Пусть $H_0: P = P_0$ против $H_1: P = P_1$, распределения имеют плотности
$p_0, p_1$ по общей мере $\mu$. Для $\lambda > 0$ положим
\begin{equation}\label{eq:np-region}
S_\lambda = \{ x : p_1(x) - \lambda p_0(x) \ge 0 \} .
\end{equation}
Тогда для любого критерия $R$ с $\Prob_0(\xi \in R) \le \Prob_0(\xi \in
S_\lambda)$

1. $\Prob_1(\xi \in R) \le \Prob_1(\xi \in S_\lambda)$;
2. $\Prob_0(\xi \in S_\lambda) \le \Prob_1(\xi \in S_\lambda)$ — критерий
   $S_\lambda$ несмещён.

[07].
\end{theorem}

\begin{proof}
1. Для всякой точки $x$ верно
$$
\mathbb I_R(x)\,\bigl(p_1(x) - \lambda p_0(x)\bigr) \le
\mathbb I_{S_\lambda}(x)\,\bigl(p_1(x) - \lambda p_0(x)\bigr):
$$
если $x \in S_\lambda$, то $\mathbb I_{S_\lambda} = 1$, $\mathbb I_R \le 1$,
а разность неотрицательна; если $x \notin S_\lambda$, то $\mathbb
I_{S_\lambda} = 0$, $\mathbb I_R \ge 0$, а разность неположительна.
Интегрируем по мере $\mu$:
$$
\Prob_1(\xi \in R) - \lambda \Prob_0(\xi \in R) \le
\Prob_1(\xi \in S_\lambda) - \lambda \Prob_0(\xi \in S_\lambda),
$$
откуда
$$
\Prob_1(\xi \in R) - \Prob_1(\xi \in S_\lambda) \le \lambda \bigl(
\Prob_0(\xi \in R) - \Prob_0(\xi \in S_\lambda) \bigr) \le 0 .
$$
2. Если $\lambda \ge 1$, то на $S_\lambda$ выполнено $p_1 \ge \lambda p_0
\ge p_0$, и $\Prob_0(\xi \in S_\lambda) = \int_{S_\lambda} p_0\,d\mu \le
\int_{S_\lambda} p_1\,d\mu = \Prob_1(\xi \in S_\lambda)$. Если $\lambda \in
(0,1)$, то вне $S_\lambda$ выполнено $p_1 < \lambda p_0 < p_0$, так что
$\Prob_1(\xi \notin S_\lambda) < \Prob_0(\xi \notin S_\lambda)$, что равносильно
нужному.
\end{proof}

\begin{corollary}[р.н.м.к. для простых гипотез]\label{cor:np-ump}
Если $\lambda > 0$ удовлетворяет $\Prob_0(\xi \in S_\lambda) =
\varepsilon$, то $S_\lambda$ — равномерно наиболее мощный критерий уровня
$\varepsilon$ для простой $H_0$ против простой $H_1$ [07].
\end{corollary}

\begin{proof}
Любой другой критерий $R$ уровня $\varepsilon$ подпадает под пункт 1 теоремы
\ref{thm:np}: $\Prob_0(\xi \in R) \le \varepsilon = \Prob_0(\xi \in
S_\lambda)$.
\end{proof}

В терминах отношения правдоподобия (англ. *likelihood ratio*)
$p_1(x)/p_0(x)$ область \eqref{eq:np-region} — подуровневое множество
$\{p_1/p_0 \ge \lambda\}$: критерий отклоняет $H_0$ там, где данные в
$\lambda$ и более раз правдоподобнее при альтернативе, чем при гипотезе.
Уравнение $\Prob_0(S_\lambda) = \varepsilon$ в $\lambda$ почти всегда
разрешимо при абсолютно непрерывных $P_0$; в дискретном случае размер критерия
принимает скачками значения $\sum_{x \in S_\lambda} p_0(x)$, и точное
равенство $\varepsilon$ недостижимо.

\begin{remark}[рандомизация]\label{rem:randomization}
Чтобы достичь уровня в точности, в дискретном случае граничную точку
области отклоняют с вероятностью $\rho \in (0,1)$: критерий — функция
$\psi(x) \in [0,1]$ (вероятность отклонения при наблюдении $x$); при
$\psi \in \{0,1\}$ это обычный критерий, а внутренняя точка — «подброс
монетки» — рандомизированный критерий (англ. \emph{randomized test}) [07].
На практике рандомизацию почти не применяют:
обходятся уровнем «не выше $\varepsilon$», как в следствии
\ref{cor:np-ump}.
\end{remark}

## Монотонное отношение правдоподобия и р.н.м.к. для сложных гипотез

Лемма \ref{thm:np} строит р.н.м.к. лишь для пары точек. Для односторонних
гипотез в однопараметрических семействах его обобщает теорема о монотонном
отношении правдоподобия.

\begin{definition}\label{def:mlr}
Доминируемое семейство $\{P_\theta, \theta \in \Theta \subseteq \R\}$ с
плотностями $p_\theta$ имеет монотонное отношение правдоподобия по статистике
$T(\xi)$ (англ. \emph{monotone likelihood ratio}), если для любых $\theta_1 <
\theta_2$ отношение $p_{\theta_2}(x)/p_{\theta_1}(x) =
\psi_{\theta_1,\theta_2}(T(x))$, где $\psi$ неубывает (монотонность одна и
та же для всех пар) [07]. При этом $T$ —
достаточная статистика (англ. \emph{sufficient statistic}).
\end{definition}

\begin{lemma}\label{lem:mlr-monotone}
Пусть семейство имеет монотонное отношение правдоподобия по $T$ с
неубывающей $\psi$. Тогда функция $\Prob_\theta(T \ge c)$ неубывает по
$\theta$ [07].
\end{lemma}

\begin{proof}
Положим $D = \{x : T(x) \ge c\}$. Если $\psi(c) \ge 1$, то на $D$
выполнено $p_{\theta_2} = \psi(T)\,p_{\theta_1} \ge \psi(c)\,p_{\theta_1}
\ge p_{\theta_1}$, и интегрирование по $D$ даёт $\Prob_{\theta_2}(T \ge c)
\ge \Prob_{\theta_1}(T \ge c)$. Если $\psi(c) \in [0,1]$, то вне $D$, то
есть при $T(x) < c$, наоборот $p_{\theta_2} = \psi(T)\,p_{\theta_1} \le
\psi(c)\,p_{\theta_1} \le p_{\theta_1}$, и интегрирование по дополнению $D$
даёт $\Prob_{\theta_2}(T < c) \le \Prob_{\theta_1}(T < c)$, что равносильно
нужному.
\end{proof}

\begin{theorem}[о монотонном отношении правдоподобия]\label{thm:mlr}
Пусть семейство $\{P_\theta\}$ имеет монотонное отношение правдоподобия по
$T$ с неубывающей непрерывной $\psi$, распределение $T$ при $\theta_0$
непрерывно, и $c$ удовлетворяет
\begin{equation}\label{eq:mlr-c}
\Prob_{\theta_0}(T(\xi) \ge c) = \varepsilon .
\end{equation}
Тогда критерий $S = \{x : T(x) \ge c\}$ — равномерно наиболее мощный
уровня значимости $\varepsilon$ для гипотезы $H_0: \theta \le \theta_0$
против альтернативы $H_1: \theta > \theta_0$ [07] (непрерывность $T$
добавлена, чтобы порог достигал уровня в точности,
— в дискретном случае без рандомизации равенство \eqref{eq:mlr-c}
достижимо не при всех $\varepsilon$, см. замечание \ref{rem:randomization}).
\end{theorem}

\begin{proof}
Уровень: для $\theta \le \theta_0$ по лемме \ref{lem:mlr-monotone}
$\Prob_\theta(T \ge c) \le \Prob_{\theta_0}(T \ge c) = \varepsilon$, так
что $\alpha(S) = \varepsilon$.

Равномерная наибольшая мощность: фиксируем $\theta_1 > \theta_0$ и любой
критерий $R$ уровня $\varepsilon$. По следствию \ref{cor:np-ump} область
$S_\lambda = \{ x : p_{\theta_1}(x) - \lambda p_{\theta_0}(x) \ge 0 \}$ с
$\Prob_{\theta_0}(S_\lambda) = \varepsilon$ максимизирует мощность для пары
$(\theta_0, \theta_1)$. В силу неубывания и непрерывности $\psi$ для
некоторого $\tilde c$
$$
S_\lambda = \{ x : \psi_{\theta_0,\theta_1}(T(x)) \ge \lambda \}
= \{ x : T(x) \ge \tilde c \},
$$
где равенство — с точностью до множества меры ноль: граница
$\{T = \tilde c\}$ невесома по непрерывности распределения $T$. Поэтому
$\Prob_{\theta_0}(T \ge \tilde c) = \Prob_{\theta_0}(S_\lambda) =
\varepsilon$. Пороги $c$ и $\tilde c$ могут не совпадать, но тогда
$\Prob_{\theta_0}\bigl( \min(c,\tilde c) \le T < \max(c,\tilde c) \bigr) = 0$,
а на этой полосе $p_{\theta_1} = \psi(T)\,p_{\theta_0}$ с ограниченным
$\psi(T)$, так что и $\Prob_{\theta_1}$-мера полосы равна нулю: пороги
равномощны. Значит $\Prob_{\theta_1}(\xi \in R) \le
\Prob_{\theta_1}(T \ge c) = \Prob_{\theta_1}(\xi \in S)$ для всех
$\theta_1 > \theta_0$.
\end{proof}

\begin{remark}\label{rem:mlr-notes}
Тот же критерий р.н.м.к. для $H_0: \theta = \theta_0$ против $H_1: \theta >
\theta_0$ [07]. Если $\psi$ убывает, р.н.м.к. имеет
вид $T \le c$ — и оптимален для $H_0: \theta \ge \theta_0$ против
$H_1: \theta < \theta_0$. Двусторонняя альтернатива $H_1:
\theta \ne \theta_0$ принципиально другая: р.н.м.к. здесь не существует —
ни один порог по $T$ не мощнее всех конкурентов одновременно при
$\theta > \theta_0$ и $\theta < \theta_0$ [07]; там
работают двойственные критерии из предложения \ref{thm:duality}.
\end{remark}

\begin{example}[биномиальный р.н.м.к.]\label{ex:mop-binom}
Пусть $\xi_k \sim \mathrm{Bin}(1, \theta)$, $H_0: \theta \le \theta_0$
против $H_1: \theta > \theta_0$. При $\theta_1 > \theta_0$
$$
\frac{p_{\theta_1}(x)}{p_{\theta_0}(x)} = \Bigl( \frac{\theta_1 (1 -
\theta_0)}{\theta_0 (1 - \theta_1)} \Bigr)^{\sum x_k} \Bigl( \frac{1 -
\theta_1}{1 - \theta_0} \Bigr)^{N},
$$
первый множитель растёт по $T = \sum_k \xi_k$, и теорема \ref{thm:mlr}
даёт критерий $S = \bigl\{ \sum_k \xi_k \ge c_1 \bigr\}$; $T$ дискретна, равенство
\eqref{eq:mlr-c} достижимо не при всех $\varepsilon$ (см. замечание
\ref{rem:randomization}), порог берётся по $\Prob_{\theta_0}(T \ge c_1) \le \varepsilon$, и то же
рассуждение даёт р.н.м.к. среди нерандомизированных критериев уровня
$\varepsilon$ [07].
При больших $N$ квантиль нормируется:
$c_1 \approx N\theta_0 + z_{1-\varepsilon}\sqrt{N\theta_0(1-\theta_0)}$.
\end{example}

\begin{example}[нормальное среднее, одностороннее]\label{ex:mop-normal}
Пусть $\xi_k \sim \mathcal N(\theta, 1)$, $H_0: \theta \ge \theta_0$
против $H_1: \theta < \theta_0$. Отношение правдоподобия растёт по $T =
-\overline X_N$, поэтому р.н.м.к. есть $S = \{ \sqrt N\,(\overline X_N -
\theta_0) \le z_\varepsilon \}$ [07].
\end{example}

# Классические критерии нормальной модели

Двойственность \ref{thm:duality} по интервалам теоремы
\ref{thm:four-intervals} даёт критерии для четырёх стандартных пар гипотез;
вывод статистик — тот же, что у интервалов, и подробно разобран у [06].
Сводка ниже — готовые правила; уровень $\alpha$, $\beta_{+} = 1 - \alpha/2$ для
двусторонних альтернатив.

\begin{theorem}[критерии нормальной модели]\label{thm:normal-tests}
Пусть $\xi_1, \dots, \xi_N$ — выборка из $\mathcal N(a, \sigma^2)$.

1. \textbf{$H_0: a = a_0$, $\sigma$ известно (u-критерий, он же
   $z$-критерий)}: статистика $U = \sqrt N\,(\overline X_N - a_0)/\sigma
   \sim \mathcal N(0,1)$ при $H_0$;
   двусторонняя критическая область $\abs{U} \ge z_{\beta_{+}}$;
   односторонняя $U \ge z_{1-\alpha}$ для $H_1: a > a_0$.
2. \textbf{$H_0: a = a_0$, $\sigma$ неизвестно (критерий Стьюдента)}: $T =
   \sqrt N\,(\overline X_N - a_0)/s_N \sim t_{N-1}$ при $H_0$ (следствие
   \ref{cor:student}); область $\abs{T} \ge t_{\beta_{+};\,N-1}$.
3. \textbf{$H_0: \sigma^2 = \sigma_0^2$}: $\chi^2 = (N-1)s_N^2/\sigma_0^2 \sim
   \chi^2_{N-1}$ при $H_0$; область двусторонняя $\chi^2 \notin
   \bigl[\chi^2_{\alpha/2;\,N-1},\ \chi^2_{\beta_{+};\,N-1}\bigr]$.
4. \textbf{Две выборки, $H_0: a_1 = a_2$ при равных неизвестных дисперсиях}:
   $T = (\overline X_N - \overline Y_M)\big/\bigl(s_{\text{обш}}
   \sqrt{1/N + 1/M}\bigr) \sim t_{N+M-2}$ [06];
   область $\abs{T} \ge t_{\beta_{+};\,N+M-2}$.
5. \textbf{$H_0: \sigma_1^2 = \sigma_2^2$ (критерий Фишера)}: $F =
   s_{1N}^2/s_{2M}^2 \sim F_{N-1,\,M-1}$ при $H_0$; область $F \notin
   \bigl[F_{\alpha/2;\,N-1,\,M-1},\ F_{\beta_{+};\,N-1,\,M-1}\bigr]$ —
   эквивалентно отношению дисперсий вне интервала предложения
   \ref{thm:two-sample}.

Односторонний вариант критерия 1 — р.н.м.к.: при известном $\sigma$
семейство однопараметрическое, теорема \ref{thm:mlr} по $T = \overline X_N$
применима, и критерий совпадает с примером \ref{ex:mop-normal}. Для критерия
2 то же рассуждение не проходит: при неизвестном $\sigma$ семейство
двухпараметрическое, монотонного отношения правдоподобия по $\overline X_N$
нет, и единого р.н.м.к. здесь нет — односторонний t-критерий оптимален лишь
среди несмещённых критериев (за рамками вопроса).
\end{theorem}

Функции мощности выписываются явно и показывают, что оптимальность — не
пустое слово. Для критерия 1 при альтернативе $a$ мощность равна
\begin{equation}\label{eq:power-u}
\pi(a) = 1 - \Phi\bigl( z_{1-\alpha/2} - \tfrac{(a - a_0)\sqrt N}{\sigma}
\bigr) + \Phi\bigl( z_{\alpha/2} - \tfrac{(a - a_0)\sqrt N}{\sigma} \bigr),
\end{equation}
она симметрична вокруг $a_0$, возрастает с $\abs{a - a_0}$ и с $N$; при
фиксированной альтернативе $a \ne a_0$ мощность стремится к единице, при
стремящейся к $a_0$ альтернативе $a_0 + h\sigma/\sqrt N$ — к нетривиальному
пределу $1 - \Phi(z_{1-\alpha/2} - h) + \Phi(z_{\alpha/2} - h)$. Последнее
— локальная асимптотика, на ней же строится понятие асимптотической
относительной эффективности двух критериев.

\begin{remark}\label{rem:two-sided}
Двусторонние критические области пунктов 1–3 — критерии двойственности, а не
р.н.м.к.: их мощность при $\theta > \theta_0$ уступает одностороннему
критерию той же стороны. Это плата за симметрию. Пункты 2 и 4 при малых $N$
чувствительны к нарушению нормальности — см. пример 2 ноутбука; устойчивые
альтернативы — предмет робастной статистики, за рамками вопроса.
\end{remark}

# Критерии согласия

Все критерии раздела выше параметрические: вид распределения известен,
неизвестен параметр. Критерии согласия (англ. *goodness-of-fit tests*)
проверяют гипотезу о **самом виде** распределения: $H_0: F = G$ с полностью
определённой $G$. Строятся по уклонению эмпирического распределения от
гипотетического; подход общий (уклонение мало при $H_0$, велико при
альтернативе), различаются — сама мера уклонения [06].

## Критерий хи-квадрат Пирсона

Разобьём выборочное пространство на $r$ классов $X_j$; $p_j = G(X_j)$ —
гипотетические вероятности, $\nu_j$ — число попаданий выборки в $X_j$.
Уклонение К. Пирсона:
\begin{equation}\label{eq:chi2-stat}
\widehat\chi^2_N = \sum_{j=1}^r \frac{(\nu_j - N p_j)^2}{N p_j}.
\end{equation}

\begin{theorem}[Пирсон]\label{thm:pearson}
Если $H_0$ верна, то $\widehat\chi^2_N \xrightarrow{d} \chi^2_{r-1}$ при
$N \to \infty$ [06].
\end{theorem}

Идея доказательства — две строчки, полное изложение — в расширенной части
(раздел \ref{sec:pearson-proof}). Вектор $\zeta = \bigl( (\nu_j - Np_j)/
\sqrt{Np_j} \bigr)$ есть сумма независимых одинаково распределённых векторов
индикаторов, по многомерной центральной предельной теореме
$\zeta \xrightarrow{d} \mathcal N(0, \Sigma)$ с
$\Sigma = I - z z\T$, $z = (\sqrt{p_1}, \dots, \sqrt{p_r})\T$; ортогональный
поворот с первой строкой $z\T$ переводит $\Sigma$ в $\diag(0, 1, \dots, 1)$,
так что $\lVert \zeta \rVert^2 = \widehat\chi^2_N \xrightarrow{d}
\lVert \mathcal N(0, I_{r-1}) \rVert^2 = \chi^2_{r-1}$. Матрица $\Sigma$
вырождена (ранг $r-1$): классов $r$, а независимых «направлений уклонения»
$r - 1$ — одна линейная связь $\sum_j (\nu_j - Np_j) = 0$ съедает степень
свободы.

Критерий: отклонить $H_0$ при $\widehat\chi^2_N > \chi^2_{1-\alpha;\,r-1}$.
При неверной $H_0$ статистика $\xrightarrow{\Prob} +\infty$, критерий
состоятелен. Правило применимости: ожидаемые частоты не мельче
$N p_j \ge 5$ — иначе соседние классы объединяют [24]; у
[06] рекомендация строже: $N p_j \ge 10$. Если $G$ зависит от $s$ неизвестных параметров, оцененных по
выборке (методом максимального правдоподобия), предельное распределение —
$\chi^2_{r-1-s}$ (результат Фишера; формулировка в расширенной части,
раздел \ref{sec:pearson-param}).

## Критерий Колмогорова

Для непрерывной $G$ уклонением служит статистика Колмогорова
\begin{equation}\label{eq:ks-stat}
D_N = \sup_{x \in \R} \bigl| F^*_N(x) - G(x) \bigr|,
\end{equation}
где $F^*_N(x) = \frac1N \sum_{k=1}^N \mathbb I_{\{\xi_k \le x\}}$ —
эмпирическая функция распределения выборки (вопрос 12:
теорема Гливенко — Кантелли утверждает $D_N \to 0$ почти наверное).

\begin{proposition}[инвариантность]\label{thm:ks-invariance}
Если $G$ непрерывна, то распределение $D_N$ при $H_0$ не зависит от $G$:
$D_N \overset{d}{=} \sup_t \lvert U(t) - U^*_N(t) \rvert$ для выборки из
$U(0,1)$ [06].
\end{proposition}

\begin{proof}
Докажем для строго возрастающей непрерывной $G$ (к несовпадающим точкам
роста сведение — в [06]). Положим $\eta_i = G(\xi_i)$. Величины $\eta_i$
независимы как функции от независимых; далее, $F_{\eta_i}(t) =
\Prob(G(\xi_i) < t) = \Prob(\xi_i < G^{-1}(t)) = G(G^{-1}(t)) = t$ при
$t \in (0,1)$, то есть $\eta_i \sim U(0,1)$. Эмпирическая функция выборки
$\eta$ в точке $t$ равна $\frac1N \sum_i \mathbb I_{(0,t)}(\eta_i) =
\frac1N \sum_i \mathbb I_{(-\infty, G^{-1}(t))}(\xi_i) =
F^*_N(G^{-1}(t))$. Поэтому
$$
\sup_{x} \lvert F^*_N(x) - G(x) \rvert = \sup_t \bigl| F^*_N(G^{-1}(t)) -
G(G^{-1}(t)) \bigr| = \sup_t \lvert U^*_N(t) - U(t) \rvert ,
$$
и распределение последнего выражения зависит только от $N$.
\end{proof}

\begin{theorem}[А. Н. Колмогоров]\label{thm:kolmogorov}
При непрерывной $F$ для каждого $\lambda > 0$
\begin{equation}\label{eq:ks-limit}
\Prob\Bigl( \sqrt N\, D_N < \lambda \Bigr) \ \xrightarrow[N \to \infty]{}\ K(\lambda)
= \sum_{k=-\infty}^{+\infty} (-1)^k\, e^{-2k^2\lambda^2},
\end{equation}
где $K$ — функция распределения Колмогорова [06].
\end{theorem}

Доказательства у [06] нет; полный вывод (броуновский мост, вычисление
распределения его максимума) — у [07]; идея —
в расширенной части, раздел \ref{sec:ks-proof}. Критерий: отклонить $H_0$
при $\sqrt N D_N \ge \lambda_{1-\alpha}$, где $\lambda_{1-\alpha}$ —
квантиль $K$ (табулирована у [06]). Важная оговорка: критерий
неприменим, если параметры $G$ оценены по той же выборке [06] —
там предельный закон другой; это — «параметрический Колмогоров», за рамками
вопроса.

# Прикладная таблица

Маршрут выбора критерия для нормальных выборок и согласий (по структуре
таблиц [24]; статистики — теорема \ref{thm:normal-tests}):

| Задача | Статистика | Распределение при $H_0$ | Область отклонения |
|---|---|---|---|
| среднее, $\sigma$ известно | $\sqrt N(\overline X_N - a_0)/\sigma$ | $\mathcal N(0,1)$ | $\abs{\cdot} \ge z_{1-\alpha/2}$ |
| среднее, $\sigma$ неизвестно | $\sqrt N(\overline X_N - a_0)/s_N$ | $t_{N-1}$ | $\abs{\cdot} \ge t_{1-\alpha/2;\,N-1}$ |
| дисперсия (среднее неизвестно) | $(N-1)s_N^2/\sigma_0^2$ | $\chi^2_{N-1}$ | $\cdot \notin [\chi^2_{\alpha/2}, \chi^2_{1-\alpha/2}]$ |
| разность средних (равные $\sigma$) | $(\overline X - \overline Y)/(s_{\text{обш}}\sqrt{1/N+1/M})$ | $t_{N+M-2}$ | $\abs{\cdot} \ge t_{1-\alpha/2;\,N+M-2}$ |
| отношение дисперсий | $s_1^2/s_2^2$ | $F_{N-1,\,M-1}$ | $\cdot$ вне двустороннего интервала |
| вид распределения (классы) | $\sum (\nu_j - Np_j)^2/(Np_j)$ | $\chi^2_{r-1}$ | $\cdot > \chi^2_{1-\alpha;\,r-1}$ |
| вид распределения (непрерывное) | $\sqrt N \sup_x \lvert F^*_N - G \rvert$ | Колмогорова $K$ | $\cdot > \lambda_{1-\alpha}$ |

# Разобранная задача

\begin{problem}\label{prob:shiryaev}
Пусть $\xi_1, \dots, \xi_N$ — выборка из $\mathcal N(m, \sigma_0^2)$ с
\textbf{известной} дисперсией $\sigma_0^2$. Найти информацию Фишера, границу
Рао — Крамера и доверительный интервал для $m$; показать, что $\overline X_N$
эффективна [17].
\end{problem}

\begin{proof}
Плотность выборки $p_m(x) = (2\pi\sigma_0^2)^{-N/2} \exp\bigl( -\sum_k (x_k
- m)^2 / (2\sigma_0^2) \bigr)$. Вклад:
$$
\frac{\partial}{\partial m} \ln p_m(\xi) = \frac{1}{\sigma_0^2} \sum_k
(\xi_k - m) = \frac{N}{\sigma_0^2}\,(\overline X_N - m),
$$
то есть линейная связь вида \eqref{eq:equality-rc} с $C(m) =
N/\sigma_0^2$ уже достигнута. Поэтому $I_N(m) = \Var(\text{вклад}) =
N/\sigma_0^2$, граница Рао — Крамера равна $\sigma_0^2/N$, и так как
$\Var \overline X_N = \sigma_0^2/N$, оценка $\overline X_N$ эффективна —
по критерию \ref{cor:efficient} это единственная эффективная оценка.
Доверительный интервал (теорема \ref{thm:four-intervals}, пункт 1):
$\overline X_N \pm z_{(1+\gamma)/2}\,\sigma_0/\sqrt N$.
\end{proof}

# Что спросят

- **Точный или асимптотический?** t-интервал точен при нормальном законе при
  любом $N$; z-интервал — асимптотический, но не требует знания закона.
  Разница максимальна при малых $N$ (замечание \ref{rem:width-price}).
- **Почему $N$, а не $N - 1$?** Статистика Стьюдента строится на
  исправленной дисперсии: с $N$ в знаменателе она смещена, и распределение
  уже не $t$ (пример \ref{ex:unbiased-s2}).
- **Почему нельзя «принять $H_0$»?** Критерий контролирует только ошибку
  первого рода; неотклонение — слабое утверждение (ошибка второго рода не
  контролируется без заданной альтернативы). Поэтому говорят «не отклоняем».
- **Как выбрать $\alpha$?** Это цена ошибки первого рода; $\alpha = 0{,}05$
  — соглашение, не теорема. Рост мощности — ростом $N$ (формула
  \eqref{eq:power-u}).
- **Что без нормальности?** Точные интервалы и критерии ломаются: ломается
  калибровка уровня, и направление искажения зависит от закона (пример 2
  ноутбука: t-интервал недопокрывает при Парето, раздувается при Коши;
  z-интервал при Коши бесполезен). t-интервал остаётся асимптотическим
  (лемма Слуцкого). Критерии согласия как раз и проверяют допущение.
- **Откуда отношение правдоподобия?** Из леммы \ref{thm:np}: это р.н.м.к. для
  простых гипотез; оптимизационный взгляд (детекторы как вершины парето-
  границы ошибок двух родов) — у [39].

# Расширенная часть

Здесь — полные доказательства аппарата, вынесенные из основной части, и
приложения. Что использовано без воспроизведения: закон больших чисел и
центральная предельная теорема (в том числе многомерная), лемма Слуцкого и
теорема о наследовании сходимости (вопрос 12); сохранение гауссовости при
линейных преобразованиях и независимость компонент стандартного гауссова
вектора (вопрос 08); неравенство Коши — Буняковского; интегрирование по
образу меры (переход $\E g(\xi) = \int g\,dP_\xi$).

## Полное доказательство теоремы Пирсона {#sec:pearson-proof}

\begin{proof}
Воспроизводим доказательство [06] в обозначениях конспекта. Пусть $H_0$ верна,
$p_j = F(X_j)$, $\nu_j$ — частоты классов, $r \ge 2$.

Шаг 1: характеристическая функция вектора частот. Представим $\nu = (\nu_1,
\dots, \nu_r)$ как сумму независимых одинаково распределённых векторов
$\mu^{(k)} = (\mathbb I_{X_1}(\xi_k), \dots, \mathbb I_{X_r}(\xi_k))$:
$\nu = \sum_k \mu^{(k)}$. У вектора $\mu^{(1)}$ ровно одна компонента равна
единице: $\Prob(\mu^{(1)} = e_j) = p_j$, где $e_j$ — $j$-й орт. Поэтому
$\E \exp\{i(t, \mu^{(1)})\} = \sum_j p_j e^{i t_j}$, и как характеристическая
функция суммы независимых слагаемых
\begin{equation}\label{eq:pearson-char-nu}
\varphi_\nu(t) = \Bigl( \sum_{j=1}^r p_j e^{i t_j} \Bigr)^{N}.
\end{equation}

Шаг 2: характеристическая функция нормированного вектора. Вектор
$\zeta = \zeta(\nu)$ с компонентами $\zeta_j = (\nu_j - Np_j)/\sqrt{Np_j}$
(статистика \eqref{eq:chi2-stat} есть $\lVert \zeta \rVert^2$) имеет
характеристическую функцию
$$
\varphi_\zeta(t) = \exp\Bigl\{ -i\sqrt N \sum_{k=1}^r t_k \sqrt{p_k} \Bigr\}
\Bigl( \sum_{j=1}^r p_j \exp\Bigl\{ i\,\frac{t_j}{\sqrt{Np_j}} \Bigr\} \Bigr)^{N}.
$$
Берём логарифм и разлагаем экспоненту:
$e^{i t_j/\sqrt{Np_j}} = 1 + i t_j/\sqrt{Np_j} - t_j^2/(2Np_j) + o(1/N)$.
Тогда
$$
\sum_j p_j e^{i t_j/\sqrt{Np_j}} = 1 + \frac{i}{\sqrt N} \sum_j t_j \sqrt{p_j}
- \frac{1}{2N} \sum_j t_j^2 + o\Bigl(\frac1N\Bigr) = 1 + z_N,
$$
где $z_N = O(1/\sqrt N)$. С учётом $\ln(1 + z) = z - z^2/2 + o(z^2)$
$$
\ln \varphi_\zeta(t) = -i\sqrt N \sum_k t_k \sqrt{p_k} + N \Bigl( z_N -
\frac{z_N^2}{2} \Bigr) + o(1).
$$
Квадрат главной части $z_N$: $N z_N^2/2 = \frac12 \bigl( i \sum_j t_j
\sqrt{p_j} \bigr)^2 + o(1)$, члены с $i\sqrt N$ сокращаются, и
\begin{equation}\label{eq:pearson-limit-char}
\ln \varphi_\zeta(t) \to -\frac12 \Bigl( \sum_{j=1}^r t_j^2 - \Bigl(
\sum_{j=1}^r t_j \sqrt{p_j} \Bigr)^2 \Bigr) .
\end{equation}
Это логарифм характеристической функции гауссовского вектора $\alpha$ с
нулевым средним и ковариационной матрицей
\begin{equation}\label{eq:pearson-sigma}
\Sigma = I_r - z z\T, \qquad z = (\sqrt{p_1}, \dots, \sqrt{p_r})\T ,
\end{equation}
что видно из квадратичной формы $(\Sigma t, t) = \sum_j t_j^2 - (\sum_j t_j
\sqrt{p_j})^2$; она неотрицательно определена по неравенству Коши —
Буняковского. По теореме Леви (вопрос 12) $\zeta \xrightarrow{d} \alpha$.

Шаг 3: сходимость квадратичной функции. По теореме о слабой сходимости
(непрерывное отображение $y \mapsto \lVert y \rVert^2$) $\lVert \zeta
\rVert^2 \xrightarrow{d} \lVert \alpha \rVert^2$. Осталось узнать
распределение $\lVert \alpha \rVert^2$.

Шаг 4: поворот. Возьмём ортогональную матрицу $C$ с первой строкой
$(\sqrt{p_1}, \dots, \sqrt{p_r})$ и положим $\beta = C\alpha$. По линейности
$\beta$ — гауссов вектор с нулевым средним; его ковариация:
$C\Sigma C\T = C C\T - (Cz)(Cz)\T = I_r - e_1 e_1\T = \diag(0, 1, \dots, 1)$,
так как $Cz = e_1$ по выбору первой строки. Компоненты $\beta_2, \dots,
\beta_r$ некоррелированы, значит независимы и стандартно нормальны, а
$\beta_1 = 0$ почти наверное. Из ортогональности $C$:
$$
\lVert \alpha \rVert^2 = \lVert C\alpha \rVert^2 = \lVert \beta \rVert^2 =
\sum_{k=2}^r \beta_k^2 \sim \chi^2_{r-1}
$$
по определению хи-квадрат. Теорема доказана.
\end{proof}

Заметим общий механизм: многомерная ЦПТ даёт гауссов предел с вырожденной
ковариацией, а ортогональный поворот снимает вырождение. Тот же приём —
в доказательстве теоремы \ref{thm:normal-sample}, и он в третий раз
вернётся в идее доказательства Колмогорова.

## Асимптотическая нормальность ОМП: идея {#sec:mle-an-proof}

Полное доказательство — у [06]. Схема: пусть $\widehat\theta_N$ —
решение уравнения правдоподобия $\ell'_N(\widehat\theta_N) = 0$, где
$\ell_N(\theta) = \ln p_\theta(\xi)$. Разложение по Тейлору в точке $\theta$
с остатком в средней точке $\theta^*$ между $\theta$ и $\widehat\theta_N$:
$$
0 = \ell'_N(\widehat\theta_N) = \ell'_N(\theta) + \ell''_N(\theta^*)\,
(\widehat\theta_N - \theta) .
$$
Перепишем:
$$
\sqrt N (\widehat\theta_N - \theta) = \frac{\ell'_N(\theta)/\sqrt N}{-
\ell''_N(\theta^*)/N}.
$$
Числитель по ЦПТ сходится к $\mathcal N(0, i(\theta))$ (вклад — сумма
независимых одинаково распределённых величин с нулевым средним и дисперсией
$i(\theta)$). Знаменатель: $\ell''_N(\theta^*)/N \to -i(\theta)$ по закону
больших чисел Хинчина (вторая форма информации, шаг 1 теоремы
\ref{thm:rao-cramer}) и состоятельности $\widehat\theta_N$. Отсюда
\eqref{eq:mle-an}; асимптотическая эффективность — совпадение предельной
дисперсии $1/i(\theta)$ с границей Рао — Крамера.

## Параметрический χ² и критерий однородности {#sec:pearson-param}

Если гипотетическое распределение $G = G_\theta$ зависит от $s$ неизвестных
параметров и они оценены методом максимального правдоподобия по той же
выборке, то под $H_0$ статистика Пирсона \eqref{eq:chi2-stat} с подстановкой
$\widehat\theta$ сходится к $\chi^2_{r-1-s}$ (результат Р. Фишера; у [06]
— в тексте без доказательства, у [07] — с
формулировкой условий регулярности). Та же асимптотика у критерия
независимости двух признаков ($\chi^2_{(s-1)(l-1)}$) и критерия однородности
$k$ выборок ($\chi^2_{(k-1)(m-1)}$; у [06] критерия однородности нет, изложен
по [07]). Степени свободы везде устроены одинаково: число
классов минус число линейных связей минус число оценённых параметров.

## Идея доказательства предельного закона Колмогорова {#sec:ks-proof}

Полный вывод — у [07]; здесь — план. После сведения
\ref{thm:ks-invariance} считаем $F = U(0,1)$. Процесс
$\gamma_N(t) = \sqrt N\,(U^*_N(t) - t)$ на $[0,1]$ — это эмпирический
процесс; по центральной предельной теореме его конечномерные распределения
сходятся к гауссовскому процессу с нулевым средним и ковариацией
$\E \gamma(s)\gamma(t) = \min(s,t) - st$ — броуновскому мосту (англ.
*Brownian bridge*). Теорема Донскера (слабая сходимость в метрическом
пространстве непрерывных функций, у [07] — через критерий Александрова)
поднимает сходимость с конечномерных на весь процесс, а функционал
$D_N = \sup_t \lvert \gamma_N(t) \rvert/\sqrt N$ непрерывен, так что
$\sqrt N D_N \xrightarrow{d} \sup_t \lvert \gamma(t) \rvert$. Остаток —
вычисление распределения максимума моста: $K(\lambda) = \Prob(\sup_t
\lvert \gamma(t) \rvert < \lambda) = \sum_k (-1)^k e^{-2k^2\lambda^2}$
(разложение отражениями). Теорема Смирнова о двухвыборочном уклонении
$\widehat D_{N,M} = \sup_x \lvert F^*_N(x) - G^*_M(x) \rvert$:
$\sqrt{\tfrac{NM}{N+M}}\,\widehat D_{N,M} \xrightarrow{d} K$ при непрерывном
общем распределении [07].

## Конспект без нормальности: что ломается и что остаётся

Точные результаты нормальной модели опираются на теорему
\ref{thm:normal-sample}, и без нормального закона она ложна: выборочное
среднее и выборочная дисперсия становятся зависимыми, статистика
\eqref{eq:student} уже не имеет распределения Стьюдента. Численная иллюстрация
на сквозном примере — пример 2 ноутбука ($N = 30$, $\gamma = 0{,}95$,
$20\,000$ прогонов): при равномерном шуме t-интервал сохраняет уровень
(покрытие $0{,}951$), при дискретном асимметричном (бернуллиевский с
$p = 0{,}05$) и при Парето с $\alpha = 3$ заметно недопокрывает ($0{,}784$
и $0{,}873$), а при Коши — наоборот завышенно консервативен ($0{,}978$):
тяжёлые хвосты взрывают $s_N$, и интервал раздувается. Асимптотический
z-интервал при Коши недопокрывает катастрофически ($0{,}216$): выборочное
среднее при тяжёлых хвостах неустойчиво. Общий урок: без нормальности ломается
калибровка уровня, и направление искажения заранее неизвестно. Выходы —
асимптотические интервалы (метод \ref{thm:pivot} по оценкам с доказанной
асимптотической нормальностью), непараметрические критерии предыдущего раздела
и робастные оценки; все три — за пределами вопроса, а критерии согласия —
способ проверить само допущение нормальности на данных (пример 5 ноутбука).

Вторая посылка всего аппарата — независимость измерений (определение
\ref{def:sample}). Без неё выборочное среднее остаётся несмещённым, но
$\Var_\theta \overline X_N = \frac{1}{N^2} \sum_{k,j} \cov_\theta(\xi_k,
\xi_j)$: при положительной корреляции соседних отсчётов выигрыш $1/N$
съедается средней парной ковариацией, и ни интервалы, ни критерии заявленного
уровня не держатся. Предельные теоремы меняются целиком: вместо закона
больших чисел для независимых слагаемых работает эргодическая теорема, и
среднее по времени может сходиться не к $a$, а к случайной величине (вопрос
12, раздел о зависимых слагаемых). Практический признак неприменимости —
отсчёты, снятые слишком часто относительно времени корреляции датчика.
