import streamlit as st
from streamlit_ace import st_ace
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import norm
from io import BytesIO
from exbook import book as eb



def main():
    st.title('Programming for Business Analytics')
    st.markdown('### Department of Analytics and Operations, NUS Business School')
    
    topics = ['', 
              'Functions, Modules, and Packages', 
              'Review of Probability Theory', 
              'Sampling Distribution']
    
    topic = st.selectbox('Select a topic: ', topics)
    
    if topic == topics[1]:
        exbook_web()
    elif topic == topics[2]:
        prob_review()

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
    
    
    
    
    if solution != '':
        try:
            exec(solution)
            exec('check(question, {}, cheat)'.format(fun_name))
        except Exception as e:
            st.markdown('### Error!')
            st.markdown(str(e))
    
def prob_review():
    
    st.markdown('---')
    st.header('Continuous Random Variables')
    st.markdown("A variable $X$ is a **continuous random variable** if it takes on any real value with *zero* probability. Random variables following uniform, normal (Gaussian) and exponential distributions are all continuous variables.")
    st.markdown("For continuous random variables, there is no PMF as the discrete random variables, because $P(X=x)=0$ for all values of $x$. The CDF for a continuous random variable has the same definition as the discrete case, which is $F(x)=P(X\leq x)$. Based on the CDF, we have other definitions listed as follows.")

    st.error("""**Notes**: Let $F(x)$ be the CDF of a continuous random variable $X$, then \n- The derivative $f(x) = \\frac{\\text{d} F(x)}{\\text{d}x}$ of the CDF $F(x)$ is called the **probability density function (PDF)** of $X$. This definition also implies that $F(x) = \int_{-\infty}^{x}f(t)dt$. \n- The inverse of CDF $F(x)$, denoted by $F^{-1}(q)$, is called the **percent point function (PPF)**, where $q$ is the given cumulative probability. This function is sometimes referred to as the **inverse distribution function** or the **quantile function**.""")

    st.markdown("The distribution diagram below is used to illustrate the concepts of PDF, CDF, and PPF. ")
    
    normal_visual()
    

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
    
    ax.text(x+0.2, -0.01, '$X=$' + '{0:4.2f}'.format(x), c='r', fontsize=11)
    ax.text(1.8, 0.38, '$P(X\leq$' + '{0:5.2f}'.format(x) + '$)=$' + '{0:0.3f}'.format(norm.cdf(x)), 
            c='b', fontsize=11)
    ax.text(max(-4.5, x-1.8), max(0.04, norm.pdf(x)-0.01), '{0:0.3f}'.format(norm.pdf(x)), 
             fontsize=12)
    
    ax.set_xticks(np.arange(-4, 5, 1))
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True)
    ax.set_xlabel('Random variable $X$', fontsize=12)
    ax.set_ylabel('Probability density function', fontsize=12)
    ax.set_ylim([-0.04, 0.62])
    ax.set_xlim([-5.2, 5.2])
    
    buf = BytesIO()
    fig.savefig(buf, format="png")
    st.image(buf)
    
    tx = '- norm.pdf({0:6.3f}, loc=0, scale=1)'.format(x)
    tx += ' = {0:6.3f}\n'.format(norm.pdf(x))
    tx += '- norm.cdf({0:6.3f}, loc=0, scale=1)'.format(x)
    tx += ' = {0:6.3f}\n'.format(norm.cdf(x))
    tx += '- norm.ppf({0:6.3f}, loc=0, scale=1)'.format(norm.cdf(x))
    tx += ' = {0:6.3f}'.format(x)
    
    st.info(tx)
    st.markdown('---')

if __name__ == "__main__":
    main()