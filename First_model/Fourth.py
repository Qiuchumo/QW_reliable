import pymc3 as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import theano.tensor as T

# from pymc3 import get_data

np.set_printoptions(precision=0, suppress=True)
# 2017.11.10编辑
# ======================================================================
# 数据导入,
# ======================================================================
dag_data = np.genfromtxt("E:/Code/Bayescode/QW_reliable/First_model/XZsingal.csv",
                         skip_header=1, usecols=[1, 2, 3, 4, 5, 6, 7, 8], delimiter=",")
elec_data = pd.read_csv('XZsingal.csv')

# 计算同一公司产品测试地点数目
companies_num = elec_data.counts.unique()
companies = len(companies_num) # companies=7， 共7个测试地点
company_lookup = dict(zip(companies_num, range(len(companies_num))))
company = elec_data['company_code'] = elec_data.counts.replace(company_lookup).values # 加一行数据在XZsingal文件中
# companys = elec_data.counts.values - 1 # 这一句以上面两行功能相同

# elec_count = elec_data.counts.values

elec_year = elec_data.Year.values # 观测时间值X1
elec_year1 = (elec_year-np.mean(elec_year))/np.std(elec_year)
elec_tem = elec_data.Tem.values   # 观测温度值X2
elec_tem1 = (elec_tem-np.mean(elec_tem))/np.std(elec_tem)
# 计算故障率大小：故障数目/总测量数，作为模型Y值，放大1000倍以增加实际效果，结果中要缩小1000倍
# elec_fault = elec_data.Fault / elec_data.Nums
elec_faults = 1000*(elec_data.Fault.values / elec_data.Nums.values) # 数组形式
elec_faults1 = (elec_faults-np.mean(elec_faults))/np.std(elec_faults)

# ======================================================================
# 模型建立：
# 模型1：using pymc3 GLM自建立模型，Normal分布更优
# 模型2: 自己模型
# ======================================================================
data = dict(x=elec_year, z1=elec_tem, y=elec_faults)

with pm.Model() as mdl_ols_glm:
    # family = pm.glm.families.StudentT()
    pm.glm.GLM.from_formula('y ~ 1+x + z1', data, family=pm.glm.families.Normal())
    # pm.glm.GLM.from_formula('y ~ 1 + x + z1', data, family=family)

    traces_ols_glm = pm.sample(3000)
pm.traceplot(traces_ols_glm)
plt.show()


with pm.Model() as pooled_model:
    # define priors
    sigma = pm.HalfCauchy('sigma', 5)
    beta = pm.Normal('beta', 0, 1000)
    beta1 = pm.Normal('beta1', 0, 10000)
    beta2 = pm.Normal('beta2', 0, 1000)

    # define likelihood 建立与时间相关的函数
    # out_pai = pm.Deterministic('out_pai',)
    theta = beta + beta1*elec_year + beta2*elec_tem1
    Observed = pm.Normal("Observed", theta, sd=sigma,  observed=elec_faults1)  # 观测值

    # start = pm.find_MAP()
    # step = pm.Metropolis()
    trace1 = pm.sample(4000, tune=1000)
chain1 = trace1
varnames1 = ['sigma', 'beta', 'beta1', 'beta2']
pm.traceplot(chain1, varnames1)
plt.show()

# ======================================================================
# unpooled_model
# ======================================================================
with pm.Model() as unpooled_model:
    # define priors
    sigma = pm.HalfCauchy('sigma', 5)

    beta = pm.Normal('beta', 0, 1000, shape=companies)
    beta1 = pm.Normal('beta1', 0, 10000, shape=companies)
    beta2 = pm.Normal('beta2', 0, 1000)

    # define likelihood 建立与时间相关的函数
    theta = beta[company] + beta1[company]*elec_year + beta2*elec_tem1

    Observed = pm.Normal("Observed", theta, sd=sigma,  observed=elec_faults1)  # 观测值

    # start = pm.find_MAP()
    # step = pm.Metropolis()
    trace2 = pm.sample(4000, tune=500)
chain2 = trace2
varnames2 = ['sigma', 'beta', 'beta1', 'beta2']
pm.traceplot(chain2, varnames2)
plt.show()

# 画出自相关曲线
pm.autocorrplot(chain2)
plt.show()

print(pm.dic(trace2, unpooled_model))

# ======================================================================
# partial_model
# ======================================================================
with pm.Model() as partial_model:
    # define priors
    mu_a = pm.Normal('mu_a', mu=0., tau=0.0001)
    sigma_a = pm.HalfCauchy('sigma_a', 100)

    beta = pm.Normal('beta', 0, 1000, shape=companies)
    beta1 = pm.Normal('beta1', mu=mu_a, sd=sigma_a, shape=companies)
    beta2 = pm.Normal('beta2', 0, 1000)

    gamma = pm.Normal('gamma', 0, 100) # 误差项
    sigma = pm.HalfCauchy('sigma', 5)  # Model error
    # define likelihood 建立与时间相关的函数
    # out_pai = pm.Deterministic('out_pai',)
    theta = beta[company] + beta1[company]*elec_year + beta2*elec_tem1 + gamma

    Observed = pm.Normal("Observed", theta, sd=sigma,  observed=elec_faults1)  # 观测值

    # start = pm.find_MAP()
    # step = pm.Metropolis()
    trace3 = pm.sample(6000, tune=1000)
chain3 = trace3
varnames3 = ['mu_a', 'sigma_a', 'beta', 'beta1', 'beta2']
varnames4 = ['gamma', 'sigma']
pm.traceplot(chain3, varnames3)
plt.show()
pm.traceplot(chain3, varnames4)
plt.show()

plt.figure(figsize=(6, 14))
pm.forestplot(chain3, varnames=['beta'])
plt.show()
pm.forestplot(chain3, varnames=['beta1'])
plt.show()
# 画出自相关曲线
pm.autocorrplot(chain3)
plt.show()

print(pm.dic(trace3, partial_model))

# ======================================================================
# 模型对比
# ======================================================================
Waic = pm.compare([traces_ols_glm, trace1, trace2, trace3], [mdl_ols_glm, pooled_model, unpooled_model, partial_model], ic='WAIC')
print(Waic)



# # 画出A公司的产品曲线
# sig0 = pm.hpd(trace['theta'], alpha=0.6)[0]
#
# plt.figure()
# ax = sns.distplot(sig0)



