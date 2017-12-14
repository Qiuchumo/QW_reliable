import pymc3 as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

np.set_printoptions(precision=0, suppress=True)

# ======================================================================
# 数据导入，用来做27的实验 加dtype='int',变成整数型数据
# ======================================================================
dag_data = np.genfromtxt("E:/Code/Bayescode/QW_reliable/First_model/XZ.csv",
                         dtype='int', skip_header=1, usecols=[2, 3, 4, 5, 6, 7, 8],
                         missing_values="NULL", delimiter=",")
dag_sum = dag_data[:21, 1] # 测量的总数
dag_fault = dag_data[:21, 0] # 故障数目
dag_time = dag_data[:21, 2] # 时间因子
dag_faultpro = dag_fault/dag_sum
print("value"); print(dag_fault); print(len(dag_fault))
print("fault value"); print(dag_sum); print(len(dag_sum))
print("time value"); print(dag_time); print(len(dag_time))
# print("value %2.6f"%dag_faultpro); print(len(dag_faultpro))

z1 = dag_data[:21, 3]
z2 = dag_data[:21, 4]
z3 = dag_data[:21, 5]
z4 = dag_data[:21, 6]
print(z4)
print(z3)
# ======================================================================
# 模型建立：
# ======================================================================

data = dict(x=dag_time, z1=z1, z2=z2, z3=z3, z4=z4, y=dag_faultpro)
# using pymc3 GLM method
with pm.Model() as mdl_ols_glm:
    pm.glm.GLM.from_formula('y ~ 1+x + z1 +z2+z3+z4', data, family=pm.glm.families.Normal())

    traces_ols_glm = pm.sample(2000)
pm.traceplot(traces_ols_glm)
plt.show()

# def invlogit(x):
#     return pm.exp(x) / (1 + pm.exp(x))

with pm.Model() as XZ_model:
    # define priors
    gamma1 = pm.Normal('gamma1', 0, 1000)
    gamma2 = pm.Normal('gamma2', 0, 1000)
    gamma3 = pm.Normal('gamma3', 0, 1000)
    gamma4 = pm.Normal('gamma4', 0, 1000)

    mu = pm.Normal('mu', 0, 1000)
    beta = pm.Normal('beta', 0, 1000)
    # theta = pm.InverseGamma('theta', 0.01, 0.01)
    # alpha = pm.Normal('alpha', 0, theta)

    # 建立与时间相关的函数
    # define likelihood
    out_pai = pm.Deterministic('out_pai', np.exp(mu  + beta * dag_time + gamma1 * z1 + gamma2 * z2 + gamma3 * z3 + gamma4 * z4) /
                               (1 + np.exp(mu  + beta * dag_time + gamma1 * z1 + gamma2 * z2 + gamma3 * z3 + gamma4 * z4)))

    # out_pai = pm.math.invlogit(mu + alpha + beta * dag_time + gamma1 * z1 + gamma2 * z2 + gamma3 * z3 + gamma4 * z4)
    # out_pai = pm.Deterministic('out_pai', mu + beta * dag_time + gamma1 * z1 + gamma2 * z2 + gamma3 * z3 + gamma4 * z4)

    Observed = pm.Weibull("Observed", 1, out_pai, observed=dag_faultpro)  # 观测值

    start = pm.find_MAP()
    # step = pm.Metropolis()
    trace = pm.sample(10000, start=start)
chain = trace
# logistic(chain, locals())
varnames = [ 'gamma1', 'mu', 'beta', 'out_pai']
varnames1 = ['out_pai']
pm.traceplot(chain, varnames)
plt.show()



# 画出自相关曲线
pm.autocorrplot(chain)
plt.show()


print(pm.dic(trace, XZ_model))

# 画出A公司的产品曲线
sig0 = pm.hpd(trace['out_pai'], alpha=0.6)[0]
sig = pm.hpd(trace['out_pai'], alpha=0.6)[1]
sig1 = pm.hpd(trace['out_pai'], alpha=0.6)[2]
sig2 = pm.hpd(trace['out_pai'], alpha=0.6)[3]
sig3 = pm.hpd(trace['out_pai'], alpha=0.6)[4]
sig4 = pm.hpd(trace['out_pai'], alpha=0.6)[5]

plt.figure()
ax = sns.distplot(sig0)
ax = sns.distplot(sig1)
ax = sns.distplot(sig2)
ax = sns.distplot(sig3)
ax = sns.distplot(sig4)
ax = sns.distplot(sig)
plt.show()


