import pymc3 as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import theano.tensor as T


np.set_printoptions(precision=0, suppress=True)
# 2017.11.27编辑  可靠性分析项目
# 撰写人：邱楚陌
# ======================================================================
# 数据导入
# companies：代表统一产品的测试地点类别    company：测试地点的搜索索引
# companiesABC：代表不同公司类别           companyABC：公司的搜索索引
# ======================================================================
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
elec_RH = elec_data.RH.values  # 观测压强x3
elec_RH1 = (elec_RH-np.mean(elec_RH))/np.std(elec_RH)
# 计算故障率大小：故障数目/总测量数，作为模型Y值，放大1000倍以增加实际效果，结果中要缩小1000倍
# elec_fault = elec_data.Fault / elec_data.Nums
elec_faults = 1000*(elec_data.Fault.values / elec_data.Nums.values) # 数组形式
elec_faults1 = (elec_faults-np.mean(elec_faults))/np.std(elec_faults)

# ======================================================================
# 模型建立：
# 模型1：using pymc3 GLM自建立模型，Normal分布更优
# ======================================================================
with pm.Model() as nonliner:
    # define priors
    sigma = pm.HalfCauchy('sigma', 20)

    beta = pm.Normal('beta',  0, 1000, shape=companiesABC)
    beta1 = pm.Normal('beta1', 0, 20)
    # beta2 = pm.Normal('beta2', 0, 100)
    # beta3 = pm.Normal('beta3', 0, 5)
    # beta4 = pm.Bound(pm.Normal, lower=0.0)('beta4', mu=0, sd=1000)
    # beta5 = pm.Normal('beta5', 0, 100)

    # define likelihood 建立与时间相关的函数
    theta = beta[companyABC] + beta1*elec_year1
    # theta1 = pm.Deterministic('theta1', theta)
    Observed = pm.Normal("Observed", mu=theta, sd=sigma, observed=elec_faults1)  # 观测值
    pred = pm.Normal("pred", mu=theta, sd=sigma, shape=elec_faults1.shape)  # 预测值

    start = pm.find_MAP()
    step = pm.Metropolis()
    trace = pm.sample(7000, start=start, step=step)

map_estimate = pm.find_MAP(model=nonliner)
print(map_estimate)
chain = trace

com_pred = chain.get_values('pred')[::100].ravel()
plt.hist(com_pred,  histtype='stepfilled')
plt.show()
varnames1 = ['beta', 'beta1',  'beta4', 'sigma']
# pm.traceplot(chain3, varnames1)
# plt.show()
pm.traceplot(chain)
plt.show()
# 画出自相关曲线
pm.autocorrplot(chain)
plt.show()
print(pm.dic(trace, nonliner))
# ======================================================================
# 模型对比与后验分析
# ======================================================================
# Waic = pm.compare([traces_ols_glm, trace1], [mdl_ols_glm, pooled_model], ic='WAIC')
# Waic = pm.compare([trace2, trace3], [partial_model, mulpartial_model], ic='WAIC')
# print(Waic)




