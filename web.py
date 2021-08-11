import streamlit as st
from streamlit_ace import st_ace
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import norm, binom
from io import BytesIO
from exbook import book as eb


def main():
    st.title('Programming for Business Analytics')
    st.markdown('### Department of Analytics and Operations, NUS Business School')

    topics = ['About',
              'Functions, Modules, and Packages',
              'Review of Probability Theory']#,
             # 'Sampling Distribution']

    topic = st.selectbox('Select a topic: ', topics)

    if topic == topics[0]:
        about()
    elif topic == topics[1]:
        exbook_web()
    elif topic == topics[2]:
        prob_review()


def about():

    st.components.v1.html("""<a href="https://github.com/XiongPengNUS/learn_dao" target="_blank"><img src="https://img.shields.io/static/v1?label=XiongPengNUS&message=learn_dao&color=blue&logo=github" alt="XiongPengNUS - learn_dao"></a>
<a href="https://github.com/XiongPengNUS/learn_dao" target="_blank"><img src="https://img.shields.io/github/stars/XiongPengNUS/learn_dao?style=social" alt="stars - learn_dao"></a>""", height=28)

    st.markdown("---")
    st.markdown("This web application is a learning tool used for NUS modules: ")
    st.markdown("- [**DAO2702 Programming for Business Analytics**](https://nusmods.com/modules/DAO2702/programming-for-business-analytics)")
    st.markdown("- [**BMK2502 Python Programming for Business Analytics**](https://nusmods.com/modules/BMK5202/python-programming-for-business-analytics) ")

    st.markdown("""You may use the app to access interactive coding practice questions and
    visualizations that illustrat the concepts of statistics and regression analysis. """)

    st.markdown("**Author**: [Xiong Peng](https://bizfaculty.nus.edu.sg/faculty-details/?profId=543)")


def exbook_web():

    def check(question, func, cheat):
        correct = []
        keys = ['Input '+str(i+1) + ': ' + type(question._Question__inputs[0][i]).__name__
                for i in range(len(question._Question__inputs[0]))] + \
               ['Your output'] + \
               ['Correct output: ' + type(question._Question__outputs[0][0]).__name__] + \
               ['Correct']
        cheat_dict = {key: [] for key in keys}
        cheat_table = pd.DataFrame(cheat_dict)
        for each_input, each_output in zip(question._Question__inputs, question._Question__outputs):
            arg_string = ''
            cheat_list = []
            for n in range(len(each_input)):
                if isinstance(each_input[n], (list, dict)):
                    arg_string += 'deepcopy(each_input[{0:d}])'.format(n)
                else:
                    arg_string += 'each_input[{0:d}]'.format(n)
                if n < len(each_input) - 1:
                    arg_string = arg_string + ', '
                cheat_list.append(str(each_input[n]))
            this_output = eval('func(' + arg_string + ')')
            if question.compset:
                if set(this_output) == set(each_output[0]):
                    correct.append(True)
                else:
                    correct.append(False)
            else:
                if this_output == each_output[0]:
                    correct.append(True)
                else:
                    correct.append(False)
            cheat_list = cheat_list + [str(this_output),
                                       str(each_output[0]),
                                       str(correct[-1])]
            test_dict = {keys[i]: cheat_list[i]
                         for i in range(len(keys))}
            cheat_table = cheat_table.append(test_dict, ignore_index=True)

        if cheat:
            cheat_table.rename(index={i: 'Test {0}:'.format(i+1)
                                      for i in range(cheat_table.shape[0])},
                                      inplace=True)
            st.write(cheat_table)

        correctness = 'incorrect' if False in correct else 'correct'
        right_ones = sum(correct)
        st.markdown('### Test results: ')
        st.markdown('You passed {0} of the {1} tests. \n'
                    'The solution is {2}'.format(right_ones,
                                                 len(correct), correctness))

    st.markdown('---')
    st.markdown("""This is the web version of the `exbook` package. It provides a set of coding practice questions, and your solutions
    are marked automatically by a number of tests.""")
    st.components.v1.html("""<a href="https://github.com/XiongPengNUS/exbook" target="_blank"><img src="https://img.shields.io/static/v1?label=XiongPengNUS&message=exbook&color=blue&logo=github" alt="XiongPengNUS - exbook"></a>
<a href="https://github.com/XiongPengNUS/exbook" target="_blank"><img src="https://img.shields.io/github/stars/XiongPengNUS/exbook?style=social" alt="stars - exbook"></a>""", height=28)
    st.markdown('---')

    ids = ['{}. {} ({})'.format(index, ex.id, ex.level) for index, ex in zip(range(1, len(eb)+1), eb)]

    label = st.selectbox('Select a question', ids)
    index = int(label[:label.find('.')])
    question = eb[index-1]

    all_inputs = question._Question__inputs
    keys = ['Input' + str(i+1) +
            ': ' + type(all_inputs[0][i]).__name__
            for i in range(len(all_inputs[0]))] + \
            ['Output: ' + type(question._Question__outputs[0][0]).__name__]
    cheat_dict = {key: [] for key in keys}
    cheat_table = pd.DataFrame(cheat_dict)
    for i in question.index:
        cheat_list = []
        each_input = question._Question__inputs[i]
        each_output = question._Question__outputs[i]
        for n in range(len(each_input)):
            cheat_list.append(str(each_input[n]))
        cheat_list = cheat_list + [str(each_output[0])]
        sample_dict = {keys[j]: cheat_list[j]
                       for j in range(len(keys))}
        cheat_table = cheat_table.append(sample_dict, ignore_index=True)

    cheat_table.rename(index={i: 'Test {0}:'.format(i+1)
                              for i in range(cheat_table.shape[0])},
                              inplace=True)

    st.write(question._Question__readme)
    st.write(cheat_table)

    fun_name = st.text_input('Name of the function to be tested: ')
    cheat = st.checkbox('Cheat table')
    st.markdown('<p style="font-size: 13px;">Definition of the function:</p>', unsafe_allow_html=True)
    solution = st_ace(language='python')

    if solution != '' and fun_name != '':
        try:
            exec(solution)
            exec('check(question, {}, cheat)'.format(fun_name))
        except Exception as e:
            st.markdown('### Error!')
            st.markdown(str(e))

def prob_review():

    st.markdown('---')
    st.header('Discrete Random Variable')
    st.markdown("""A random variable $X$ is defined to be **discrete** if its possible
    outcomes are finite or countable. Examples of distributions of discrete random
    variables are discrete uniform distribution (*i.e.*, outcome of rolling an even die),
    Bernouli distribution (*i.e.*, the preference of a randomly selected customer for
    Coke or Pepsi), Binomial distribution (*i.e.*, the number of customers who prefer
    Coke over Pepsi among 10 randomly selected customers), and Poisson distribution
    (*i.e.*, The number of patients arriving in an emergency room within a fixed time
    interval) etc. """)
    st.error("""**Notes**: For a discrete random variable $X$ with $k$ possible outcomes
    $x_j$, \n- the **probability mass function (PMF)** is given by: $P(X=x_j) = p_j$, for
    each $j=1, 2, ..., k$, where $p_j$ is the probability of the outcome $x_j$, and all
    $p_i$ must satisfy \n$$\\begin{cases} 0\\leq p_i \leq 1 \\\\
    \\sum_{j=1}^kp_j = 1
    \\end{cases}$$\n - The **cumulative distribution function (CDF)** of a random variable
    $X$ is defined as $F(x) = P(X\leq x)$.""")
    st.markdown("""Suppose that in Singapore, the proportion of customers who prefer Coke
    is $p$, and the remaining $p-1$ of all customers prefer Pepsi. Now we randomly survey
    $n$ customers, among which the number of customers who prefer Coke is denoted by a
    discrete random variable $X$. The PMF of $X$ and its CDF are illustrated by the graph
    below.""")

    binom_visual()

    st.markdown('---')
    st.header('Continuous Random Variables')
    st.markdown("""A variable $X$ is a **continuous random variable** if it takes on
    any real value with *zero* probability. Random variables following uniform, normal
    (Gaussian) and exponential distributions are all continuous variables.""")
    st.markdown("""For continuous random variables, there is no PMF as the discrete
    random variables, because $P(X=x)=0$ for all values of $x$. The CDF for a continuous
    random variable has the same definition as the discrete case, which is $F(x)=P(X\leq x)$.
    Based on the CDF, we have other definitions listed as follows.""")

    st.error("""**Notes**: Let $F(x)$ be the CDF of a continuous random variable $X$,
    then \n- The derivative $f(x) = \\frac{\\text{d} F(x)}{\\text{d}x}$ of the CDF $F(x)$
    is called the **probability density function (PDF)** of $X$. This definition also
    implies that $F(x) = \int_{-\infty}^{x}f(t)dt$. \n- The inverse of CDF $F(x)$, denoted
    by $F^{-1}(q)$, is called the **percent point function (PPF)**, where $q$ is the given
    cumulative probability. This function is sometimes referred to as the **inverse
    distribution function** or the **quantile function**.""")

    st.markdown("""The distribution diagram below is used to illustrate the concepts of PDF,
    CDF, and PPF. """)

    normal_visual()


def binom_visual():

    p = st.slider('Proportion of all customers who prefer Coke: ',
                  min_value=0.1, max_value=0.9, value=0.4, step=0.05)
    n = st.slider('Number of surveyed customers: ',
                  min_value=5, max_value=100, value=15, step=5)
    m = st.slider('Among the surveyed customers, the number of them who prefer Coke: ',
                  min_value=0.0, max_value=float(n), value=float(n//2),
                  step=0.1 if n <= 10 else 0.2 if n <= 25 else 0.5)

    xi = np.arange(n+1)
    pi = binom.pmf(xi, n, p)
    pc = binom.cdf(xi, n, p)
    xc = list(np.array([xi[:-1], xi[1:]]).T.reshape(2*n)) + [n]
    yc = list(np.array([pc[:-1], pc[:-1]]).T.reshape(2*n)) + [1]
    ym = binom.pmf(m, n, p)
    ycm = binom.cdf(m, n, p)
    fig, ax = plt.subplots(2, 1, figsize=(7.5, 7.5))
    ax[0].vlines(xi[xi <= m], ymin=0, ymax=pi[xi <= m],
                 linewidth=3, alpha=0.5, color='b', label='Probability counted for CDF')
    ax[0].vlines(xi[xi > m], ymin=0, ymax=pi[xi > m],
                 linewidth=3, alpha=0.5, color='k', label='Probability not counted for CDF')
    ax[0].vlines(m, ymin=0, ymax=ym,
                 linewidth=3, alpha=0.5, color='r', linestyle='--')
    ax[0].scatter(m, ym, s=60, color='r', alpha=0.5)
    if p <= 0.5:
        xt, yt = n*0.7, binom.pmf(round(n*p), n, p)*0.7
        ax[0].text(xt*1.02, yt*0.99, '$P(X={0})={1:0.4f}$'.format(m, ym),
                   color='r', fontsize=12)
        if m < xt:
            ax[0].plot([m, xt], [ym, yt], linewidth=2, color='r', alpha=0.5)
        else:
            ax[0].plot([m, m], [ym, yt*0.95], linewidth=2, color='r', alpha=0.5)

        ax[0].legend(loc='upper right')
    else:
        xt, yt = n*0.3, binom.pmf(round(n*p), n, p)*0.7
        ax[0].text(-0.1*xt, yt*0.99, '$P(X={0})={1:0.4f}$'.format(m, ym),
                   color='r', fontsize=12)
        if m > xt:
            ax[0].plot([m, xt], [ym, yt], linewidth=2, color='r', alpha=0.5)
        else:
            ax[0].plot([m, m], [ym, yt*0.95], linewidth=2, color='r', alpha=0.5)

        ax[0].legend(loc='upper left')
    ax[0].set_ylabel('PMF', fontsize=12)
    ax[0].grid(True)

    ax[1].plot(xc, yc, linewidth=3, alpha=0.5, color='b', label='CDF')
    ax[1].vlines(m, ymin=0, ymax=ycm,
                 linewidth=3, alpha=0.5, color='r', linestyle='--')
    ax[1].scatter(m, ycm, s=60, color='r', alpha=0.5)
    if p <= 0.5:
        xt, yt = n*0.7, 0.7
        ax[1].text(xt*1.02, yt*0.99, '$P(X\leq{0})={1:0.4f}$'.format(m, ycm),
                   color='r', fontsize=12)
        if m < xt:
            ax[1].plot([m, xt], [ycm, yt], linewidth=2, color='r', alpha=0.5)
        else:
            ax[1].plot([m, m], [ym, yt*1.07], linewidth=2, color='r', alpha=0.5)
    else:
        xt, yt = n*0.3, 0.7
        ax[1].text(-0.1*xt, yt*0.99, '$P(X\leq{0})={1:0.4f}$'.format(m, ycm),
                   color='r', fontsize=12)
        if m > xt:
            ax[1].plot([m, xt], [ym, yt], linewidth=2, color='r', alpha=0.5)
        else:
            ax[1].plot([m, m], [ym, yt*0.95], linewidth=2, color='r', alpha=0.5)
    ax[1].set_ylabel('CDF', fontsize=12)
    ax[1].set_xlabel('Value $x$', fontsize=12)
    ax[1].grid(True)

    buf = BytesIO()
    fig.savefig(buf, format="png")
    st.image(buf)

    tx = '- `binom.pmf({}, {}, {})'.format(m, n, p)
    tx += ' = {0:6.3f}` \n'.format(binom.pmf(m, n, p))
    tx += '- `binom.cdf({}, {}, {})'.format(m, n, p)
    tx += ' = {0:6.3f}` \n'.format(binom.cdf(m, n, p))

    st.info(tx)

def normal_visual():

    st.markdown('---')
    x = st.slider('Value of the random variable:', min_value=-3.5, max_value=3.5, value=-1.0, step=0.1)
    step = 0.01
    xs = np.arange(-3.6, 3.6+step, step)
    ys = norm.pdf(xs)
    xc = np.arange(-3.5, x+step, step)
    yc = norm.pdf(xc)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(xs, ys, color='k', linewidth=3, alpha=0.5, label='PDF curve')
    ax.fill_between(xc, y1=yc, y2=0, color='b', alpha=0.5, label='CDF')
    ax.scatter(x, 0, c='r', s=60, alpha=0.5, label='Random varaible value')

    ax.plot([x, 3], [0.5*norm.pdf(x), 0.35],
            c='b', linewidth=2.5, alpha=0.4)
    ax.plot([max(-3.6, x-0.8), x-0.01],
            [max(0.04, norm.pdf(x)), norm.pdf(x)],
            c='k', linewidth=2.5, alpha=0.5)

    ax.text(x+0.2, -0.01, '$x=$' + '{0:4.2f}'.format(x), c='r', fontsize=11)
    ax.text(1.8, 0.38, '$P(X\leq$x$)=$' + '{0:0.3f}'.format(norm.cdf(x)),
            c='b', fontsize=11)
    ax.text(max(-4.5, x-1.8), max(0.04, norm.pdf(x)-0.01), '{0:0.3f}'.format(norm.pdf(x)),
             fontsize=12)

    ax.set_xticks(np.arange(-4, 5, 1))
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True)
    ax.set_xlabel('Value $x$', fontsize=12)
    ax.set_ylabel('Probability density function', fontsize=12)
    ax.set_ylim([-0.04, 0.62])
    ax.set_xlim([-5.2, 5.2])

    buf = BytesIO()
    fig.savefig(buf, format="png")
    st.image(buf)

    tx = '- `norm.pdf({0:6.3f}, loc=0, scale=1)'.format(x)
    tx += ' = {0:6.3f}` \n'.format(norm.pdf(x))
    tx += '- `norm.cdf({0:6.3f}, loc=0, scale=1)'.format(x)
    tx += ' = {0:6.3f}` \n'.format(norm.cdf(x))
    tx += '- `norm.ppf({0:6.3f}, loc=0, scale=1)'.format(norm.cdf(x))
    tx += ' = {0:6.3f}`'.format(x)

    st.info(tx)

if __name__ == "__main__":
    main()
