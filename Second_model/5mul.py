import pymc3 as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import theano.tensor as T

# from pymc3 import get_data

np.set_printoptions(precision=0, suppress=True)
# 2017.11.13编辑  可靠性分析项目
# 撰写人：邱楚陌
# ======================================================================
# 数据导入
# companies：代表统一产品的测试地点类别    company：测试地点的搜索索引
# companiesABC：代表不同公司类别           companyABC：公司的搜索索引
# ======================================================================
dag_data = np.genfromtxt("E:/Code/Bayescode/QW_reliable/Second_model/XZmulti_3.csv",
                         skip_header=1, usecols=[1, 2, 3, 4, 5, 6, 7, 8], delimiter=",")
elec_data = pd.read_csv('XZmulti_3.csv')

# 计算同一公司产品测试地点数目：
companies_num = elec_data.counts.unique()
companies = len(companies_num) # companies=7， 共7个测试地点
company_lookup = dict(zip(companies_num, range(len(companies_num))))
company = elec_data['company_code'] = elec_data.counts.replace(company_lookup).values # 加一行数据在XZsingal文件中
# companys = elec_data.counts.values - 1 # 这一句以上面两行功能相同

# 计算不同公司数目
company_ABC = elec_data.company.unique()
companiesABC = len(company_ABC) # companies=7， 共7个测试地点
company_lookup_ABC = dict(zip(company_ABC, range(len(company_ABC))))
companyABC = elec_data['company_ABC'] = elec_data.company.replace(company_lookup_ABC).values # 加一行数据在XZsingal文件中
# companys = elec_data.counts.values - 1 # 这一句以上面两行功能相同
# elec_count = elec_data.counts.values


# 计算观测时间，温度，光照等环境条件
elec_year = elec_data.Year.values # 观测时间值x1
elec_year1 = (elec_year-np.mean(elec_year))/np.std(elec_year)
elec_tem = elec_data.Tem.values   # 观测温度值x2
elec_tem1 = (elec_tem-np.mean(elec_tem))/np.std(elec_tem)
elec_hPa = elec_data.hPa.values  # 观测压强x3
elec_hPa1 = (elec_hPa-np.mean(elec_hPa))/np.std(elec_hPa)
elec_RH = elec_data.RH.values  # 观测湿度x3
elec_RH1 = (elec_RH-np.mean(elec_RH))/np.std(elec_RH)
# 计算故障率大小：故障数目/总测量数，作为模型Y值，放大1000倍以增加实际效果，结果中要缩小1000倍
# elec_fault = elec_data.Fault / elec_data.Nums
elec_faults = 1000*(elec_data.Fault.values / elec_data.Nums.values) # 数组形式
elec_faults1 = (elec_faults-np.mean(elec_faults))/np.std(elec_faults)

error0 = np.max(elec_data.Nums.values[:41]/np.max(elec_data.Nums.values)) # 误差项
error1 = np.max(elec_data.Nums.values[43:90]/np.max(elec_data.Nums.values)) # 误差项
error2 = np.max(elec_data.Nums.values[93:]/np.max(elec_data.Nums.values)) # 误差项
error = np.array([error0, error1, error2])
# ======================================================================
# 模型建立：
# 模型1：using pymc3 GLM自建立模型，Normal分布更优
# 模型2: 自己模型
# ======================================================================
data = dict(x=elec_year1, z1=elec_tem1, z2=elec_hPa1, z3=elec_RH1, y=elec_faults1)
# with pm.Model() as mdl_ols_glm:
#     # family = pm.glm.families.StudentT()
#     pm.glm.GLM.from_formula('y ~ 1 + x + z1 + z2 + z3', data, family=pm.glm.families.Normal())
#     # pm.glm.GLM.from_formula('y ~ 1 + x + z1', data, family=family)
#
#     traces_ols_glm = pm.sample(3000)
# pm.traceplot(traces_ols_glm)
# plt.show()
'''
# #pooled_model集中模型
with pm.Model() as pooled_model:
    # define priors
    sigma = pm.HalfCauchy('sigma', 5)
    nu = pm.Exponential('nu', 1/30)
    # nu = pm.Uniform('nu', lower=1, upper=100)
    # nu = pm.Gamma('nu', 2, 0.1)

    beta = pm.Normal('beta', 0, 10)
    beta1 = pm.Normal('beta1', 0, 10)
    beta2 = pm.Normal('beta2', 0, 10)
    beta3 = pm.Normal('beta3', 0, 10)
    beta4 = pm.Normal('beta4', 0, 10)

    # define likelihood 建立与时间相关的函数
    theta = beta + beta1*elec_year + beta2*elec_tem1 + beta3*elec_hPa1 + beta4*elec_RH1
    Observed = pm.StudentT("Observed", mu=theta, sd=sigma, nu=nu, observed=elec_faults1)  # 观测值

    start = pm.find_MAP()
    # step = pm.Metropolis()
    trace1 = pm.sample(4000, start=start)
chain1 = trace1
varnames1 = ['beta', 'beta1', 'beta2', 'beta3', 'beta4']
pm.traceplot(chain1, varnames1)
plt.show()

# 画出自相关曲线
pm.autocorrplot(chain1)
plt.show()
'''
#
# # #partial_model 部分集中模型
# with pm.Model() as partial_model:
#     # define priors
#     sigma = pm.HalfCauchy('sigma', 5)
#     # nu = pm.Exponential('nu', 1/30)
#
#     beta = pm.Normal('beta', 0, 10, shape=companiesABC)
#     beta1 = pm.Normal('beta1', 0, 10)
#     beta2 = pm.Normal('beta2', 0, 10, shape=companiesABC)
#     beta3 = pm.Normal('beta3', 0, 10)
#     beta4 = pm.Normal('beta4', 0, 10)
#
#     # define likelihood 建立与时间相关的函数
#     theta = beta[companyABC] + beta1*elec_year + beta2[companyABC]*elec_tem1 + beta3*elec_hPa1 + beta4*elec_RH1
#     Observed = pm.Normal("Observed", mu=theta, sd=sigma, observed=elec_faults1)  # 观测值
#
#     start = pm.find_MAP()
#     # step = pm.Metropolis()
#     trace2 = pm.sample(4000, start=start)
# chain2 = trace2
# varnames1 = ['beta', 'beta1', 'beta2', 'beta3', 'beta4']
# pm.traceplot(chain2, varnames1)
# plt.show()
#
# # 画出自相关曲线
# pm.autocorrplot(chain2)
# plt.show()

# ======================================================================
# student分布有较好的效果，但是部分参数的收敛性不是很好
# 加了误差项
with pm.Model() as mulpartial_model:
    # define priors
    sigma = pm.HalfCauchy('sigma', 10)
    nu = pm.Exponential('nu', 1/10)
    mu_a = pm.Uniform('mu_a', -10, 10)
    sigma_a = pm.HalfNormal('sigma_a', sd=20)
    sigma_a1 = pm.HalfCauchy('sigma_a1', 10)

    beta = pm.Normal('beta', mu=mu_a, sd=sigma_a, shape=companiesABC)
    beta1 = pm.Normal('beta1', 0, 5)
    beta2 = pm.Normal('beta2', 0, 12)
    beta3 = pm.Normal('beta3', 0, 20)
    beta4 = pm.Normal('beta4', 0, sd=sigma_a1)

    # define likelihood 建立与时间相关的函数
    theta = beta[companyABC] + beta1*elec_year1 + beta2*elec_tem1 + beta3*elec_RH1 + beta4*elec_tem1*elec_RH1
    Observed = pm.StudentT("Observed", mu=theta, sd=sigma, nu=nu, observed=elec_faults1)  # 观测值


    start = pm.find_MAP()
    # step = pm.Metropolis()
    trace3 = pm.sample(5000, start=start, tune=1000)
chain3 = trace3
varnames1 = ['beta', 'beta1', 'beta2', 'beta3', 'beta4']
pm.traceplot(chain3, varnames1)
plt.show()
varnames1 = ['sigma', 'mu_a', 'sigma_a']
pm.traceplot(chain3, varnames1)
plt.show()
# 画出自相关曲线
pm.autocorrplot(chain3)
plt.show()

tracedf = pm.trace_to_dataframe(trace3, varnames=['beta1', 'beta2', 'beta3', 'beta4'])
sns.pairplot(tracedf)
plt.show()
print(pm.dic(trace3, mulpartial_model))
# ======================================================================
# 模型对比与后验分析
# ======================================================================
# Waic = pm.compare([traces_ols_glm, trace1], [mdl_ols_glm, pooled_model], ic='WAIC')
# Waic = pm.compare([trace2, trace3], [partial_model, mulpartial_model], ic='WAIC')
# print(Waic)




