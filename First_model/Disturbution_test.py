import pymc3 as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

np.set_printoptions(precision=0, suppress=True)
# 测试分布形状用

x = np.linspace(0.01, 0.99, 100)
pdf = stats.weibull_max.pdf(x, 2)
plt.figure()
plt.plot(x, pdf)
plt.show()

c = 1.79
x = np.linspace(stats.weibull_min.ppf(0.01, c), stats.weibull_min.ppf(0.99, c), 100)
plt.plot(x, stats.weibull_min.pdf(x, c), 'r-', lw=5, alpha=0.6, label='weibull_min pdf')
plt.show()
