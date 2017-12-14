import pymc3 as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import theano.tensor as tt

from matplotlib.font_manager import FontProperties
# from pymc3 import get_data
font = FontProperties(fname=r"C:\\WINDOWS\\Fonts\\simsun.ttc", size=14)
np.set_printoptions(precision=0, suppress=True)
# 2017.11.11编辑  可靠性分析项目
# 撰写人：邱楚陌
# ======================================================================
# 数据导入
# companies：代表统一产品的测试地点类别    company：测试地点的搜索索引
# companiesABC：代表不同公司类别           companyABC：公司的搜索索引
# ======================================================================
elec_data = pd.read_csv('XZmulti_3.csv')

# 计算同一公司产品测试地点数目：
companies_num = elec_data.counts.unique()
companies = len(companies_num)  # companies=7， 共7个测试地点
company_lookup = dict(zip(companies_num, range(len(companies_num))))
company = elec_data['company_code'] = elec_data.counts.replace(company_lookup).values  # 加一行数据在XZsingal文件中
# companys = elec_data.counts.values - 1 # 这一句以上面两行功能相同

# 计算不同公司数目
company_ABC = elec_data.company.unique()
companiesABC = len(company_ABC)  # companies=7， 共7个测试地点
company_lookup_ABC = dict(zip(company_ABC, range(len(company_ABC))))
companyABC = elec_data['company_ABC'] = elec_data.company.replace(company_lookup_ABC).values  # 加一行数据在XZsingal文件中
# companys = elec_data.counts.values - 1 # 这一句以上面两行功能相同
# elec_count = elec_data.counts.values


# 计算观测时间，温度，光照等环境条件
elec_year = elec_data.Year.values  # 观测时间值x1
elec_year1 = (elec_year - np.mean(elec_year)) / np.std(elec_year)
elec_tem = elec_data.Tem.values  # 观测温度值x2
elec_tem1 = (elec_tem - np.mean(elec_tem)) / np.std(elec_tem)
elec_hPa = elec_data.hPa.values  # 观测压强x3
elec_hPa1 = (elec_hPa - np.mean(elec_hPa)) / np.std(elec_hPa)
elec_RH = elec_data.RH.values  # 观测压强x3
elec_RH1 = (elec_RH - np.mean(elec_RH)) / np.std(elec_RH)
# 计算故障率大小：故障数目/总测量数，作为模型Y值，放大1000倍以增加实际效果，结果中要缩小1000倍
# elec_fault = elec_data.Fault / elec_data.Nums
elec_faults = 100 * (elec_data.Fault.values / elec_data.Nums.values)  # 数组形式
elec_faults1 = (elec_faults - np.mean(elec_faults)) / np.std(elec_faults)

plt.hist(elec_faults, range=[-1, 5], bins=130, histtype='stepfilled', color='#6495ED')
plt.axvline(elec_faults.mean(), color='r', ls='--', label='True mean')
plt.show()

# 调好的基于weibull的模型，但是后面分析有点问题，拟合效果不明显
with pm.Model() as unpooled_model:
    # define priors
    sigma = pm.Gamma('sigma', 1, 1.6)
    # mu = pm.Gamma('mu', 2, 5)

    beta = pm.Gamma('beta', 20, 2000, shape=companiesABC)
    beta1 = pm.Gamma('beta1', 20, 2000, shape=companiesABC)
    beta2 = pm.Gamma('beta2', 4, 80000)
    mu = pm.Deterministic('mu', tt.exp(beta[companyABC] + beta1[companyABC] *elec_year + beta2*elec_tem))

    Observed_pred = pm.Weibull("Observed_pred",  alpha=mu, beta=sigma, shape=elec_faults.shape)  # 观测值
    Observed = pm.Weibull("Observed", alpha=sigma, beta=mu, observed=elec_faults)  # 观测值beta为尺度参数

    start = pm.find_MAP()
    # step = pm.Metropolis()
    trace2 = pm.sample(2000,  start=start)
chain2 = trace2
# varnames1 = ['beta', 'beta1', 'sigma', 'mu']
pm.traceplot(chain2)
plt.show()

com_pred = chain2.get_values('Observed_pred')[:].ravel()
plt.hist(com_pred,  range=[0, 2], bins=130, histtype='stepfilled')
plt.axvline(elec_faults.mean(), color='r', ls='--', label='True mean')
plt.show()
# # 画出自相关曲线
# pm.autocorrplot(chain2, varnames2)
# plt.show()
#
with unpooled_model:
    post_pred = pm.sample_ppc(trace2, samples=1000)
plt.figure()
ax = sns.distplot(post_pred['Observed'].mean(axis=1), label='Posterior predictive means')
ax.axvline(elec_faults.mean(), color='r', ls='--', label='True mean')
ax.legend()
plt.show()
# print(pm.dic(trace2, unpooled_model))
