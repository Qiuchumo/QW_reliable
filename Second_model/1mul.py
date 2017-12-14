import pymc3 as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import theano.tensor as T
from matplotlib.font_manager import FontProperties
# from pymc3 import get_data
font = FontProperties(fname=r"C:\\WINDOWS\\Fonts\\simsun.ttc", size=14)
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']
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
elec_faults = 100*(elec_data.Fault.values / elec_data.Nums.values) # 数组形式
elec_faults1 = (elec_faults-np.mean(elec_faults))/np.std(elec_faults)


# # 画出原始图
# Company_names = ['XiZang', 'XinJiang', 'HeiLongJiang']
# k = np.array([0, 41, 90, 132])
# j, k1 = 0, 6
# plt.figure(figsize=(6, 8), facecolor=(1,1,1))
# for ix in range(3):
#     ax = plt.subplot(3, 1, ix+1)
#     if ix == 1:
#         k1 = 7
#     else:
#         k1 = 6
#     for jx in range(7):
#         ax.plot(elec_year[j:(j+k1)], elec_faults[j:(j+k1)], 'ko--', markersize=4, linewidth=1)
#         j = j+k1
#     plt.xlabel(u"时间t/年", fontsize=14, fontproperties=font)
#     plt.ylabel(u"故障率/%", fontsize=14, fontproperties=font)
#     ax.legend([u'故障率曲线'], loc='upper left', prop=font)
#     # ax.text(1, 0.1, u"故障率", fontsize=15)
#     plt.grid()
#     # plt.title('%s' % (Company_names[ix]))
#     k[ix+1] = k[ix+1]+1
# plt.tight_layout()
# plt.show()




faults_m = np.mean(elec_faults)
faults_sd = np.std(elec_faults)
year_m = np.mean(elec_year)
year_std = np.std(elec_year)
tem_m = np.mean(elec_tem)
tem_std = np.std(elec_tem)
# ======================================================================
# 模型建立：
# 模型1：using pymc3 GLM自建立模型，Normal分布更优
# 模型2: 自己模型
# ======================================================================
# data = dict(x1=elec_year, x2=elec_tem, y=elec_faults)

'''
with pm.Model() as pooled_model:
    # define priors
    sigma = pm.HalfCauchy('sigma', 5)
    beta = pm.Normal('beta', 0, 20)
    beta1 = pm.Normal('beta1', 0, 20)
    beta2 = pm.Normal('beta2', 0, 20)

    # define likelihood 建立与时间相关的函数
    theta = beta + beta1*elec_year + beta2*elec_tem1
    Observed = pm.Normal("Observed", theta, sd=sigma, observed=elec_faults1)  # 观测值
    # start = pm.find_MAP()
    trace1 = pm.sample(2000, chain=2)

chain1 = trace1
varnames1 = ['sigma', 'beta', 'beta1', 'beta2']
pm.traceplot(chain1, varnames1)
plt.show()
# pm.gelman_rubin(chain1)

# ======================================================================
# partial_model
# ======================================================================
with pm.Model() as partial_model:
    # define priors
    mu_a = pm.Normal('mu_a', mu=0., tau=0.0001)
    sigma_a = pm.HalfCauchy('sigma_a', beta=10, testval=1.)
    # sigma_a = pm.HalfNormal('sigma_a', 0, 20)

    beta = pm.Normal('beta', 0, 20, shape=companiesABC)
    beta1 = pm.Normal('beta1', mu=mu_a, sd=sigma_a)
    beta2 = pm.Normal('beta2', 0, 20)

    sigma = pm.HalfCauchy('sigma', 10, testval=1.)  # Model error
    # define likelihood 建立与时间相关的函数
    # theta = pm.Deterministic('theta', beta + beta1*elec_year + beta2*elec_tem1)
    theta = beta[companyABC] + beta1 * elec_year1 + beta2 * elec_tem1

    Observed = pm.Normal("Observed", theta, sd=sigma,  observed=elec_faults1)  # 观测值
    # pm.StudentT('likelihood', mu=yest, sd=sigma_y, nu=nu, observed=dfhoggs['y'])

    # start = pm.find_MAP()
    # step = pm.Metropolis()
    trace3 = pm.sample(3000)
chain3 = trace3
varnames3 = ['mu_a', 'sigma_a', 'beta', 'beta1', 'beta2', 'sigma_a']
pm.traceplot(chain3, varnames3)
plt.show()

map_estimate = pm.find_MAP(model=partial_model)
print(map_estimate)

# 画出自相关曲线
pm.autocorrplot(chain3)
plt.show()

# 画出参数间的自相关
tracedf = pm.trace_to_dataframe(trace3, varnames=['beta1', 'beta2'])
sns.pairplot(tracedf)
plt.show()

print(pm.dic(trace3, partial_model))
'''



# ======================================================================
# unpooled_model
# ======================================================================
with pm.Model() as unpooled_model:
    # define priors
    sigma = pm.HalfCauchy('sigma', 5)

    beta = pm.Normal('beta', 0, 20, shape=companiesABC)
    beta1 = pm.Normal('beta1', 0, 20, shape=companiesABC)
    beta2 = pm.Normal('beta2', 0, 20)

    # define likelihood 建立与时间相关的函数
    # theta = pm.math.logit(beta[companyABC] + beta1[companyABC]*elec_year1 + beta2*elec_tem1)
    # theta = pm.Deterministic('theta', beta[companyABC] + beta1[companyABC]*elec_year1 + beta2*elec_tem1)
    theta = beta[companyABC] + beta1[companyABC] * elec_year1 + beta2 * elec_tem1
    Observed = pm.Normal("Observed", theta, sd=sigma, observed=elec_faults1)  # 观测值
    # Observed = pm.Gamma("Observed", theta, sigma,  observed=elec_faults1)  # 观测值
    # Observed_pred = pm.Bound(pm.Normal, lower=0.0)('Observed_pred', mu=theta, sd=sigma, shape=elec_faults1.shape)  # 观测值
    # start = pm.find_MAP()
    # step = pm.Metropolis()
    trace2 = pm.sample(1000)

chain2 = trace2
varnames2 = ['beta', 'beta1', 'beta2']
# pm.traceplot(chain2, varnames2, kde_plot=True, text_size=14, color='#6CA6CD')
pm.traceplot(chain2, varnames2)
plt.show()

# 画出自相关曲线
pm.autocorrplot(chain2)
plt.show()

with unpooled_model:
    post_pred = pm.sample_ppc(trace2, samples=1000)
plt.figure()
ax = sns.distplot(post_pred['Observed'].mean(axis=1), label='Posterior predictive means')
ax.axvline(elec_faults.mean(), color='r', ls='--', label='True mean')
ax.legend()
plt.show()
# map_estimate = pm.find_MAP(model=unpooled_model)
# print(map_estimate)
# x_lim = 60
# com_pred = chain2.get_values('Observed_pred')[::10].ravel()
# plt.hist(com_pred,  range=[-2, 5], bins=90, histtype='stepfilled', color='#6CA6CD')
# plt.xlabel(u"时间t/年", fontsize=14, fontproperties=font)
# plt.ylabel(u"频率/%", fontsize=14, fontproperties=font)
# plt.show()

# 数据
post_beta = np.mean(chain2['beta'][:, 0])
post_beta1 = np.mean(chain2['beta1'][:, 0])
post_beta0 = np.mean(chain2['beta'][:, 1])
post_beta11 = np.mean(chain2['beta1'][:, 1])
post_beta00 = np.mean(chain2['beta'][:, 2])
post_beta111 = np.mean(chain2['beta1'][:, 2])

post_beta2 = np.mean(chain2['beta2'])
# 数据还原
org_beta1 = post_beta1*faults_sd/year_std
org_beta2 = post_beta2*faults_sd/tem_std
org_beta = post_beta*faults_sd + faults_m - (post_beta1*faults_sd*year_m/year_std) - (post_beta2*faults_sd*tem_m/tem_std)

org_beta11 = post_beta11*faults_sd/year_std
org_beta0 = post_beta0*faults_sd + faults_m - (post_beta11*faults_sd*year_m/year_std) - (post_beta2*faults_sd*tem_m/tem_std)

org_beta111 = post_beta111*faults_sd/year_std
org_beta00 = post_beta00*faults_sd + faults_m - (post_beta111*faults_sd*year_m/year_std) - (post_beta2*faults_sd*tem_m/tem_std)

# 后验

idx = np.argsort(elec_year)
x_ord = elec_year[idx]

ppc = pm.sample_ppc(chain2, samples=1000, model=unpooled_model)
sig_y = pm.hpd(ppc['Observed'][0:42], alpha=0.05)[idx]
sig_y1 = pm.hpd(ppc['Observed'][42:91], alpha=0.05)[idx]
sig_y11 = pm.hpd(ppc['Observed'][91:], alpha=0.05)[idx]

plt.figure(figsize=(10, 10))
# idd = range(0, len(chain2['beta2']), 10)
# for i in idd:
#     plt.plot(elec_year[0:6], chain2['beta'][:, 0][i] + chain2['beta1'][:, 0][i]*elec_year[0:6] + chain2['beta2'][i]*elec_tem[0:6], color='gray', alpha=0.5)
# plt.plot(elec_year[0:6], org_beta + org_beta1 * elec_year[0:6] + org_beta2 * elec_tem[0:6], 'b-', linewidth=4)
# plt.show()


plt.figure(figsize=(5, 3), facecolor=(1,1,1))
ax = plt.subplot(1, 1, 1)
j, k1 = 0, 6
for jx in range(7):
    k1 = 6
    if jx==1:
        ax.plot(elec_year[j:(j + k1)], elec_faults[j:(j + k1)], 'k--',  linewidth=0.9)
    else:
        plt.plot(elec_year[j:(j + k1)], elec_faults[j:(j + k1)], 'k--', linewidth=0.9)
    j = j + k1
ax.plot(elec_year[0:6], org_beta + org_beta1 * elec_year[0:6] + org_beta2 * elec_tem[0:6], 'b-', linewidth=4)
# plt.fill_between(x_ord[:125], sig_y[:125, 0], sig_y[:125, 1], color='gray', alpha=0.5)

plt.xlabel(u"时间t/年", fontsize=14, fontproperties=font)
plt.ylabel(u"故障率/%", fontsize=14, fontproperties=font)
ax.legend([u'故障率曲线', u'拟合曲线'], loc='upper left', prop=font)
plt.grid()
plt.show()


plt.figure(figsize=(5, 3), facecolor=(1,1,1))
ax = plt.subplot(1, 1, 1)
for jx in range(7, 14, 1):
    k1 = 7
    if jx==7:
        ax.plot(elec_year[j:(j + k1)], elec_faults[j:(j + k1)], 'k--',  linewidth=0.9)
    else:
        plt.plot(elec_year[j:(j + k1)], elec_faults[j:(j + k1)], 'k--', linewidth=0.9)
    j = j + k1
ax.plot(elec_year[42:49], org_beta0 + org_beta11 * elec_year[42:49] + org_beta2 * elec_tem[42:49], 'b-', linewidth=4)
# plt.fill_between(x_ord, sig_y1[:, 0], sig_y1[:, 1], color='gray', alpha=0.5)
plt.xlabel(u"时间t/年", fontsize=14, fontproperties=font)
plt.ylabel(u"故障率/%", fontsize=14, fontproperties=font)
ax.legend([u'故障率曲线', u'拟合曲线'], loc='upper left', prop=font)
plt.grid()
plt.show()


plt.figure(figsize=(5, 3), facecolor=(1,1,1))
ax = plt.subplot(1, 1, 1)
for jx in range(14, 21, 1):
    k1 = 6
    if jx==14:
        ax.plot(elec_year[j:(j + k1)], elec_faults[j:(j + k1)], 'k--',  linewidth=0.9)
    else:
        plt.plot(elec_year[j:(j + k1)], elec_faults[j:(j + k1)], 'k--', linewidth=0.9)
    j = j + k1
ax.plot(elec_year[91:97], org_beta00 + org_beta111 * elec_year[91:97] + org_beta2 * elec_tem[91:97], 'b-', linewidth=4)
# plt.fill_between(x_ord[:125], sig_y11[:125, 0], sig_y11[:125, 1], color='gray', alpha=0.5)
plt.xlabel(u"时间t/年", fontsize=14, fontproperties=font)
plt.ylabel(u"故障率/%", fontsize=14, fontproperties=font)
ax.legend((u'故障率曲线', u'拟合曲线'), loc='upper left', prop=font)
plt.grid()
plt.show()
print(pm.dic(trace2, unpooled_model))
# Waic = pm.compare([trace1, trace3, trace2], [pooled_model, partial_model, unpooled_model,], ic='WAIC')
# print(Waic)

