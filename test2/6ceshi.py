import pymc3 as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import theano.tensor as tt
from theano import shared
import pandas as pd
from scipy.special import gamma
from theano.compile.ops import as_op

from matplotlib.font_manager import FontProperties
# from pymc3 import get_data
font = FontProperties(fname=r"C:\\WINDOWS\\Fonts\\simsun.ttc", size=14)
np.set_printoptions(precision=0, suppress=True)
# 2017.12.3编辑  可靠性分析项目，测试代码
# 撰写人：邱楚陌
# ======================================================================
# 数据导入
# companies：代表统一产品的测试地点类别    company：测试地点的搜索索引
# companiesABC：代表不同公司类别           companyABC：公司的搜索索引
# ======================================================================
elec_data = pd.read_csv('XZmulti_6.csv')

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


x_shared = shared(elec_year)
x_shared1 = shared(elec_tem)

y_shared = shared(elec_faults)


elec_year1 = np.delete(elec_year, np.s_[:6])
elec_year1 = np.append([1,2,3,4,5,7], elec_year1)
elec_year1 = np.delete(elec_year1, np.s_[-1])
elec_year1 = np.append(elec_year1, 7)
elec_year1[83] = 7

# plt.hist(elec_faults, range=[0, 5], bins=130, histtype='stepfilled', color='#6495ED')
# plt.axvline(elec_faults.mean(), color='r', ls='--', label='True mean')
# plt.show()

# 画出原始图
# Company_names = ['XiZang', 'XinJiang', 'HeiLongJiang']
# k = np.array([0, 41, 90, 132])
# j, k1 = 0, 6
# plt.figure(figsize=(6, 8), facecolor='w')
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


# 撰写自己的函数 tt.dscalar：表一个值  dvector:表向量
# @as_op(itypes=[tt.lvector, tt.dvector, tt.dvector], otypes=[tt.dvector])
# def rate_(x_imput, eary, late):
#
#     if x_imput.any < 6:
#         out = eary
#     else:
#         out = late
#     return out
#     early_rate = pm.Normal('early_rate', 0, 100, shape=companiesABC)
#     late_rate = pm.Normal('late_rate', 0, 100, shape=companiesABC)
#     # beta1 = pm.math.switch(x_shared <= 5, early_rate[:], late_rate[:])
#     beta1 = rate_(x_shared, early_rate, late_rate)
# 这里将威布尔参数的alpha 与 beta交换
# with pm.Model() as pooled_model:
#     # define priors
#     alpha1 = pm.HalfCauchy('alpha1', 10)
#
#     b0eta = pm.Normal('b0eta', 0, 1000)
#     b0eta1 = pm.Normal('b0eta1', 0, 1000)
#     b0eta2 = pm.Normal('b0eta2', 0, 1000)
#     b0eta3 = pm.Normal('b0eta3', 0, 1000)
#     # mu = tt.exp(beta[companyABC] + beta1[companyABC]*elec_year + beta2*elec_tem)
#     beta_mu0 = pm.Deterministic('beta_mu0', tt.exp(b0eta + b0eta1 * x_shared + b0eta2 * x_shared1 + b0eta3* x_shared * x_shared))
#
#     # Observed_pred = pm.Weibull("Observed_pred",  alpha=mu, beta=sigma, shape=elec_faults.shape)  # 观测值
#     Observed = pm.Weibull("Observed", alpha=alpha1, beta=beta_mu0, observed=y_shared)  # 观测值
#
#     start = pm.find_MAP()
#     trace1 = pm.sample(4000,  start=start)
# pm.traceplot(trace1)
# plt.show()
with pm.Model() as unpooled_model:
    # define priors
    alpha = pm.HalfCauchy('alpha', 10, testval=.9)

    beta = pm.Normal('beta', 0, 100, shape=companiesABC, testval=-3.)
    # beta1 = pm.Normal('beta1', 0, 10, shape=companiesABC, testval=.3)
    # beta2 = pm.Normal('beta2', 0, 100, testval=0.01)
    # beta3 = pm.Normal('beta3', 0, 100)

    theta = pm.Normal('theta', 0, 100, shape=companiesABC)
    theta1 = pm.Normal('theta1', 0, 20, shape=companiesABC)
    beta1 = theta[companyABC] + theta1[companyABC] * x_shared1
    # mu = tt.exp(beta[companyABC] + beta1[companyABC]*elec_year + beta2*elec_tem)
    beta_mu = pm.Deterministic('beta_mu', tt.exp(beta[companyABC] + beta1[companyABC] * x_shared))

    # Observed_pred = pm.Weibull("Observed_pred",  alpha=mu, beta=sigma, shape=elec_faults.shape)  # 观测值
    Observed = pm.Weibull("Observed", alpha=alpha, beta=beta_mu, observed=y_shared)  # 观测值

    start = pm.find_MAP()
    # step = pm.Slice([beta1, u])
    trace2 = pm.sample(2000,  start=start)
chain2 = trace2[1000:]
varnames1 = ['alpha', 'beta_mu']
# varnames2 = ['beta', 'beta1', 'beta2', 'alpha', 'beta3']
# pm.plot_posterior(chain2, varnames2, ref_val=0)
pm.traceplot(chain2)
plt.show()
pm.energyplot(trace2)
plt.show()

# elec_year1 = np.delete(elec_year, -1)
# elec_year1 = np.append(elec_year1, 7)
x_shared.set_value(elec_year1)
# x_shared1.set_value([40, 40, 40, 60, 60, 60])
# y_shared.set_value([0, 0, 0, 0, 0, 0, 0])
with unpooled_model:
    post_pred = pm.sample_ppc(trace2, samples=1000)
abc = post_pred['Observed'].mean(axis=0)
print(abc)
# 画出自相关曲线
pm.autocorrplot(chain2, varnames2)
plt.show()
print(pm.waic(trace2, unpooled_model))

#
with unpooled_model:
    post_pred = pm.sample_ppc(trace2, samples=1000)
plt.figure()
ax = sns.distplot(post_pred['Observed'].mean(axis=1), label='Posterior predictive means')
# ax = sns.distplot(y_shared.mean(axis=1), label='Posterior predictive means')
ax.axvline(elec_faults.mean(), color='r', ls='--', label='True mean')
ax.legend()
plt.show()
# 自相关
tracedf = pm.trace_to_dataframe(trace2, varnames=['beta1', 'beta2'])
sns.pairplot(tracedf)
plt.show()

# print(pm.df_summary(trace2, varnames1))
print(pm.df_summary(trace2, varnames2))

# 读取后验区间，加.mean()是为了转换为np型数据便于计算
aaa = pm.df_summary(trace2, varnames2)
bbb = pd.DataFrame(aaa)
hpd2_5 = bbb['hpd_2.5']
hpd97_5 = bbb['hpd_97.5']
hpd25_beta = hpd2_5[:1].mean()
hpd975_beta = hpd97_5[:1].mean()

hpd25_beta1_0 = hpd2_5[1:2].mean()
hpd975_beta1_0 = hpd97_5[1:2].mean()
hpd25_beta1_1 = hpd2_5[2:3].mean()
hpd975_beta1_1 = hpd97_5[2:3].mean()
hpd25_beta1_2 = hpd2_5[3:4].mean()
hpd975_beta1_2 = hpd97_5[3:4].mean()

hpd25_beta2 = hpd2_5[4:5].mean()
hpd975_beta2 = hpd97_5[4:5].mean()

hpd25_alpha = hpd2_5[5:6].mean()
hpd975_alpha = hpd97_5[5:6].mean()

hpd25_beta3_0 = hpd2_5[6:7].mean()
hpd975_beta3_0 = hpd97_5[6:7].mean()
# hpd25_beta3_1 = hpd2_5[10:11].mean()
# hpd975_beta3_1 = hpd97_5[10:11].mean()
# hpd25_beta3_2 = hpd2_5[11].mean()
# hpd975_beta3_2 = hpd97_5[11].mean()


hpdmean = bbb['mean']
post_beta = hpdmean[0]
post_beta1 = hpdmean[1]
post_beta11 = hpdmean[2]
post_beta111 = hpdmean[3]
post_beta2 = hpdmean[4]

post_alpha = hpdmean[5]
post_beta30 = hpdmean[6]
# post_beta31 = hpdmean[10]
# post_beta32 = hpdmean[11]

# 计算平均故障率，即在时间与温度双重作用下的频率故障率变化
betaa = np.exp(post_beta + post_beta1 * elec_year[0:6] + post_beta2 * elec_tem[0:6] + post_beta30*elec_year[0:6]*elec_year[0:6])
betaa1 = np.exp(post_beta + post_beta11 * elec_year[42:48] + post_beta2 * elec_tem[42:48] + post_beta30*elec_year[0:6]*elec_year[0:6])
betaa2 = np.exp(post_beta + post_beta111 * elec_year[84:90] + post_beta2 * elec_tem[84:90] + post_beta30*elec_year[0:6]*elec_year[0:6])
Mean_gamma = betaa * gamma((1 + 1 / post_alpha))
Mean_gamma1 = betaa1 * gamma((1 + 1 / post_alpha))
Mean_gamma2 = betaa2 * gamma((1 + 1 / post_alpha))

# 计算后验均值区 间97.5
betaa975  = np.exp(hpd975_beta + hpd975_beta1_0 * elec_year[0:6] + hpd975_beta2 * elec_tem[0:6] + hpd975_beta3_0*elec_year[0:6]*elec_year[0:6])
betaa9751 = np.exp(hpd975_beta + hpd975_beta1_1 * elec_year[42:48] + hpd975_beta2 * elec_tem[42:48] + hpd975_beta3_0*elec_year[0:6]*elec_year[0:6])
betaa9752 = np.exp(hpd975_beta + hpd975_beta1_2 * elec_year[84:90] + hpd975_beta2 * elec_tem[84:90] + hpd975_beta3_0*elec_year[0:6]*elec_year[0:6])
Mean_gamma975 = betaa975 * gamma((1 + 1 / hpd975_alpha))
Mean_gamma9751 = betaa9751 * gamma((1 + 1 / hpd975_alpha))
Mean_gamma9752 = betaa9752 * gamma((1 + 1 / hpd975_alpha))


# 计算后验均值区间2.5
betaa25  = np.exp(hpd25_beta + hpd25_beta1_0 * elec_year[0:6] + hpd25_beta2 * elec_tem[0:6] + hpd25_beta3_0*elec_year[0:6]*elec_year[0:6])
betaa251 = np.exp(hpd25_beta + hpd25_beta1_1 * elec_year[42:48] + hpd25_beta2 * elec_tem[42:48] + hpd25_beta3_0*elec_year[0:6]*elec_year[0:6])
betaa252 = np.exp(hpd25_beta + hpd25_beta1_2 * elec_year[84:90] + hpd25_beta2 * elec_tem[84:90] + hpd25_beta3_0*elec_year[0:6]*elec_year[0:6])
Mean_gamma25 = betaa25 * gamma((1 + 1 / hpd25_alpha))
Mean_gamma251 = betaa251 * gamma((1 + 1 / hpd25_alpha))
Mean_gamma252 = betaa252 * gamma((1 + 1 / hpd25_alpha))


# 画图
plt.figure(figsize=(5, 3), facecolor=(1,1,1))
ax = plt.subplot(1, 1, 1)
j, k1 = 0, 6
for jx in range(7):
    k1 = 6
    plt.plot(elec_year[j:(j + k1)], elec_faults[j:(j + k1)], 'k--', linewidth=0.9)
    j = j + k1
plt.plot(elec_year[0:6], betaa, linewidth=4)
# plt.plot(elec_year[0:6], betaa25, linewidth=4)
# plt.plot(elec_year[0:6], betaa975, linewidth=4)
# plt.plot(elec_year[0:6], Mean_gamma975, linewidth=2, color='#006400')
plt.plot(elec_year[0:6], Mean_gamma, linewidth=3, color='#0000FF')
# plt.plot(elec_year[0:6], Mean_gamma25, linewidth=4, color='#B22222')
plt.xlabel(u"时间t/年", fontsize=14, fontproperties=font)
plt.ylabel(u"故障率/%", fontsize=14, fontproperties=font)
ax.legend([ u'平均故障', u'故障2.5', u'故障97.5'], loc='upper left', prop=font)
plt.grid()
plt.show()



plt.figure(figsize=(5, 3), facecolor=(1,1,1))
ax = plt.subplot(1, 1, 1)
for jx in range(7, 14, 1):
    k1 = 6
    plt.plot(elec_year[j:(j + k1)], elec_faults[j:(j + k1)], 'k--', linewidth=0.9)
    j = j + k1
plt.plot(elec_year[42:48], betaa1, linewidth=4)
plt.plot(elec_year[0:6], betaa251, linewidth=4)
# plt.plot(elec_year[0:6], betaa9751, linewidth=4)
# plt.plot(elec_year[0:6], Mean_gamma9751, linewidth=2, color='#006400')
plt.plot(elec_year[0:6], Mean_gamma1, linewidth=3, color='#0000FF')
# plt.plot(elec_year[42:48], Mean_gamma251, linewidth=4, color='#B22222')
plt.xlabel(u"时间t/年", fontsize=14, fontproperties=font)
plt.ylabel(u"故障率/%", fontsize=14, fontproperties=font)
ax.legend([u'平均故障', u'故障2.5', u'故障97.5'], loc='upper left', prop=font)
plt.grid()
plt.show()

plt.figure(figsize=(5, 3), facecolor=(1,1,1))
ax = plt.subplot(1, 1, 1)
for jx in range(14, 21, 1):
    k1 = 6
    plt.plot(elec_year[j:(j + k1)], elec_faults[j:(j + k1)], 'k--', linewidth=0.9)
    j = j + k1
plt.plot(elec_year[84:90], betaa2, linewidth=4)
plt.plot(elec_year[0:6], betaa252, linewidth=4)
# plt.plot(elec_year[0:6], betaa9752, linewidth=4)
# plt.plot(elec_year[0:6], Mean_gamma9752, linewidth=2, color='#006400')
plt.plot(elec_year[0:6], Mean_gamma2, linewidth=3, color='#0000FF')
# plt.plot(elec_year[0:6], Mean_gamma252, linewidth=4, color='#B22222')
plt.xlabel(u"时间t/年", fontsize=14, fontproperties=font)
plt.ylabel(u"故障率/%", fontsize=14, fontproperties=font)
ax.legend([u'平均故障', u'故障2.5', u'故障97.5'], loc='upper left', prop=font)
plt.grid()
plt.show()


# 可靠度计算
post_alpha1 = np.mean(chain2['alpha'])
post_beta_mu1 = np.mean(chain2['beta_mu'])

varnames1 = ['alpha', 'beta_mu']
aaa1 = pm.df_summary(trace2, varnames1)
bbb1 = pd.DataFrame(aaa1)

hpdd2_5 = bbb1['hpd_2.5']
hpdd97_5 = bbb1['hpd_97.5']
hpd2_5_alpha = hpd2_5[:1].mean()
hpd97_5_alpha = hpd97_5[:1].mean()
hpd25_beta_mu = hpd2_5[1:].mean()
hpd975_beta_mu = hpd97_5[1:].mean()

# 可靠度函数
ax = plt.subplot(1, 1, 1)
t = np.arange(1, 7, 1)
R1 = np.exp(-((t/post_beta_mu1)**post_alpha1))
R2 = np.exp(-((t/hpd25_beta_mu)**hpd2_5_alpha))
R3 = np.exp(-((t/hpd975_beta_mu)**hpd97_5_alpha))
# plt.plot(t, R2, 'k-', t, R1, 'bo--', t, R3, 'r')
plt.plot(t, R2, 'k-', t, R1, 'bo--', t, R3, 'r')
ax.legend([u'可靠度区间2.5', u'可靠度均值', u'可靠度区间97.5'], prop=font)
plt.show()

print(pm.dic(trace2, unpooled_model))
A = pm.compare([trace1,trace2], [pooled_model, unpooled_model], ic='WAIC')
print(A)
pm.compareplot(A)
plt.show()
