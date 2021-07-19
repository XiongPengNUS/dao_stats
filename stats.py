import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from io import BytesIO



def main():
    st.title('Basics of Statistics')
    st.markdown('### Programming for Business Analytics, NUS Business School')
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
    ax.plot([max(-3.8, x-0.8), x-0.01], 
            [norm.pdf(x), norm.pdf(x)], 
            c='k', linewidth=2.5, alpha=0.5)
    
    ax.text(x+0.2, -0.01, '$X=$' + '{0:4.2f}'.format(x), c='r', fontsize=11)
    ax.text(1.8, 0.38, '$P(X\leq$' + '{0:5.2f}'.format(x) + '$)=$' + '{0:0.3f}'.format(norm.cdf(x)), 
            c='b', fontsize=11)
    plt.text(max(-4, x-1.5), norm.pdf(x)-0.01, '{0:0.3f}'.format(norm.pdf(x)), 
             fontsize=12)
    
    ax.set_xticks(np.arange(-3.5, 4, 0.5))
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True)
    ax.set_xlabel('Random variable $X$', fontsize=12)
    ax.set_ylabel('Probability density function', fontsize=12)
    ax.set_ylim([-0.04, 0.62])
    
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