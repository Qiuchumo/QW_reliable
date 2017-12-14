import numpy as np
import pandas as pd
import pymc3 as pm3
from theano import shared
import theano.tensor as tt
import matplotlib.pylab as plt


new_data = pd.read_csv('new_data.txt', sep='\t')

idx = np.asarray(new_data.idx)
num_items = len(np.unique(idx))

train = new_data[:-5]
test = new_data[-5:]

masked_x1 = shared(np.asarray(train.x1))
masked_x2 = shared(np.asarray(train.x2))
masked_idx = shared(np.asarray(train.idx))

with pm3.Model() as model:
    intercept_prior = pm3.Normal('intercept_prior', mu=20, sd=100, shape=num_items)

    mu_x1 = pm3.Normal('mu_x1', mu=0, sd=20)
    std_x1 = pm3.Uniform('std_x1', lower=0, upper=100)
    x1_prior = pm3.Normal('x1_prior', mu=mu_x1, sd=std_x1, shape=num_items)

    std_x2 = pm3.Uniform('std_x2', lower=0, upper=100)
    x2_prior = pm3.HalfNormal('x2_prior', sd=std_x2, shape=num_items)

    y_est = pm3.Deterministic('y_est',
        intercept_prior[masked_idx] + x1_prior[masked_idx] * masked_x1 + x2_prior[masked_idx] * masked_x2
    )

    pm3.Normal('y_pred', mu=y_est, sd=100, observed=train.y)
    trace = pm3.sample(6000)

varnames = ['mu_x1', 'std_x1', 'std_x2']
pm3.traceplot(trace[2000:], varnames)
plt.show()
pm3.plot_posterior(trace[2000:], varnames)
plt.show()
#%%
masked_x1.set_value(np.asarray(test.x1))
masked_x2.set_value(np.asarray(test.x2))
masked_idx.set_value(np.asarray(test.idx))
ppc = pm3.sample_ppc(trace, model=model, samples=500)
abc = ppc['y_pred']
print(abc)
print(abc)
# masked_x1.set_value(test.x1)
# masked_x2.set_value(test.x2)
# ppc = pm3.sample_ppc(trace, model=model, samples=500)

