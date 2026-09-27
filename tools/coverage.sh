#!/usr/bin/env bash
# Матрица «книга x вопрос»: bash tools/coverage.sh <файл> ...
#
# По каждой книге считает вхождения характерных терминов всех двадцати вопросов.
# Берёт распознанный текст из .ocr/<stem>.txt, если он есть.
#
# Шаблоны ДВУЯЗЫЧНЫЕ: в коллекции есть англоязычные книги, и русский шаблон давал
# у них пустую строку - неотличимо от «тема не покрыта».
#
# ЧТО ЭТО МЕРИТ: частоту термина, а не наличие доказательства. Колонка «сильно»
# (>=5 вхождений) означает «тема в книге обсуждается», и не более. Решение о
# пригодности принимается чтением главы; плотность доказательств
# (tools/density.sh) сужает круг чтения, но тоже не заменяет его.
#
# Нормализация: дефисные переносы склеиваются, всё пробельное сводится к ОДНОМУ
# пробелу (не удаляется - иначе многословные термины, разорванные переносом
# строки, становятся невидимы), «ё» приводится к «е», между буквами шаблона
# допускаются пробелы (русская разрядка).
set -uo pipefail
cd "$(dirname "$0")/.."
python3 - "$@" <<'PY'
import re,sys,os,json; sys.path.insert(0,'tools')
import booktext
sp=booktext.spaced
TOPICS={
 '01':r'сплайн|\bspline',
 '02':r'регуляриз|некорректн\w* задач|тихонов|\bregulariz|ill\W?posed|tikhonov|neural net|нейронн\w* сет',
 '03':r'минимаксн|байесовск|'+sp('функция потерь')+r'|\bminimax|bayes\w*|loss function',
 '04':r'люенберг|наблюдаемост|идентифицируемост|идентификатор|luenberger|observabilit|state observer',
 '05':r'вейвлет|всплеск|\bwavelet',
 '06':r'принцип\w* максимума|понтрягин|гамильтона\W{0,3}якоби|pontryagin|maximum principle|hamilton\W?jacobi',
 '07':r'фурье|преобразовани\w* лаплас|хаара|\bfourier|laplace transform|\bhaar',
 '08':r'корреляционн|ковариацион|случайн\w* вектор|covarianc|correlation function|random vector',
 '09':r'гильбертов|нормированн\w* пространств|метрическ\w* пространств|hilbert space|normed space|metric space|banach space',
 '10':r'градиентн\w* (спуск|метод)|метод\w* ньютона|gradient (descent|method)|newton\W{0,3}method|steepest descent',
 '11':r'(полином|многочлен)\w* чебышев|чебышевск\w* (полином|многочлен)|chebyshev polynomial',
 '12':r'центральн\w* предельн|закон\w* больших чисел|характеристическ\w* функци|central limit|law of large numbers|characteristic function',
 '13':r'квадратурн|формул\w* симпсон|численн\w* интегрирован|численн\w* дифференцирован|quadratur|numerical integration|numerical differentiation',
 '14':r'доверительн\w* интервал|проверк\w* гипотез|неймана?\W{0,3}пирсон|отношени\w* правдоподоби|confidence interval|hypothesis test|neyman\W{0,3}pearson|likelihood ratio',
 '15':r'хана\W{0,3}банах|банаха\W{0,3}хана|продолжени\w* функционал|линейн\w* функционал|hahn\W{0,3}banach|linear functional',
 '16':r'главн\w* компонент|независим\w* компонент|principal component|independent component',
 '17':r'монте\W{0,3}карло|monte\W{0,3}carlo',
 '18':r'логистическ\w* регресс|линейн\w* регресс|наименьших квадратов|гаусса?\W{0,3}маркова|logistic regression|linear regression|least squares|gauss\W{0,3}markov',
 '19':r'равномерн\w* приближени|наилучш\w* приближени|альтернанс|чебышевск\w* приближени|uniform approximation|best approximation|alternation|chebyshev approximation',
 '20':r'марковск|динамическ\w* программирован|уравнени\w* беллман|markov (decision|chain|process)|dynamic programming|bellman equation',
}
PROOF=re.compile(booktext.PROOF_PAT,re.I)
out={}
print('книга'.ljust(52), ' '.join(f'{q}' for q in sorted(TOPICS)), '  плотн.')
for f in sys.argv[1:]:
    stem=os.path.splitext(os.path.basename(f))[0]
    t,info=booktext.load(f)
    if info['verdict'].startswith(('СКАН','БИТЫЙ')):
        print(f'{stem[:50]:52} ' + ' '.join(' ?' for _ in TOPICS) + f'    {info["verdict"][:14]}'); continue
    hits={q:len(re.findall(p,t,re.I)) for q,p in TOPICS.items()}
    dens=len(PROOF.findall(t))/(len(t)/100000)
    out[stem]={'hits':hits,'density':round(dens,1),'len':len(t)}
    cells=[]
    for q in sorted(TOPICS):
        n=hits[q]
        cells.append(' .' if n==0 else (f'{min(n,99):2d}' if n>=5 else ' ~'))
    print(f'{stem[:50]:52} ' + ' '.join(cells) + f'  {dens:6.1f}')
# Матрица в JSON — по желанию, путём в COVERAGE_JSON. Абсолютного пути здесь быть не
# должно: файл версионируется, а каталог сессии у каждого запуска свой.
out_path=os.environ.get('COVERAGE_JSON')
if out_path: json.dump(out,open(out_path,'w'),ensure_ascii=False,indent=1)
print('\nлегенда:  ".": нет вхождений   "~": 1-4 (упоминание)   число: >=5 (тема обсуждается)')
print('плотн. — доказательств на 100k знаков; <1 = книга без выкладок')
PY
