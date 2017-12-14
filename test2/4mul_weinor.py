import pymc3 as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import theano.tensor as tt
import pandas as pd
from scipy.special import gamma

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

# plt.hist(elec_faults, range=[0, 5], bins=130, histtype='stepfilled', color='#6495ED')
# plt.axvline(elec_faults.mean(), color='r', ls='--', label='True mean')
# plt.show()

# 画出原始图
Company_names = ['XiZang', 'XinJiang', 'HeiLongJiang']
k = np.array([0, 41, 90, 132])
j, k1 = 0, 6
plt.figure(figsize=(6, 8), facecolor='w')
for ix in range(3):
    ax = plt.subplot(3, 1, ix+1)
    if ix == 1:
        k1 = 7
    else:
        k1 = 6
    for jx in range(7):
        ax.plot(elec_year[j:(j+k1)], elec_faults[j:(j+k1)], 'ko--', markersize=4, linewidth=1)
        j = j+k1
    plt.xlabel(u"时间t/年", fontsize=14, fontproperties=font)
    plt.ylabel(u"故障率/%", fontsize=14, fontproperties=font)
    ax.legend([u'故障率曲线'], loc='upper left', prop=font)
    # ax.text(1, 0.1, u"故障率", fontsize=15)
    plt.grid()
    # plt.title('%s' % (Company_names[ix]))
    k[ix+1] = k[ix+1]+1
plt.tight_layout()
plt.show()


with pm.Model() as unpooled_model:
    # define priors
    # sigma = pm.Gamma('sigma', 1, 3.5)
    sigma = pm.HalfCauchy('sigma', 10, testval=1.)

    beta = pm.Normal('beta', 0, 100000)
    beta1 = pm.Normal('beta1', 0, 10000, shape=companiesABC)
    beta2 = pm.Normal('beta2', 0, 100000)
    θ = pm.Gamma('θ', 1, 100)
    u = pm.Normal('u', 0, θ*θ, shape=companiesABC)
    # mu = tt.exp(beta[companyABC] + beta1[companyABC]*elec_year + beta2*elec_tem)
    mu = pm.Deterministic('mu', tt.exp(beta + beta1[companyABC] * elec_year + beta2 * elec_tem + u[companyABC]))

    # Observed_pred = pm.Weibull("Observed_pred",  alpha=mu, beta=sigma, shape=elec_faults.shape)  # 观测值
    Observed = pm.Weibull("Observed", alpha=mu, beta=sigma, observed=elec_faults)  # 观测值

    start = pm.find_MAP()
    # step = pm.Slice([beta1, u])
    trace2 = pm.sample(6000,  start=start)
chain2 = trace2[4000:]
varnames1 = ['sigma', 'mu']
varnames2 = ['beta', 'beta1', 'beta2', 'u', 'sigma']
# pm.plot_posterior(chain2, varnames2, ref_val=0)
pm.traceplot(chain2)
plt.show()
pm.energyplot(trace2)
plt.show()
# 画出自相关曲线
pm.autocorrplot(chain2, varnames2)
plt.show()
print(pm.dic(trace2, unpooled_model))
# com_pred = chain2.get_values('Observed_pred')[:].ravel()
# plt.hist(com_pred,  range=[0, 2], bins=130, histtype='stepfilled')
# plt.axvline(elec_faults.mean(), color='r', ls='--', label='True mean')
# plt.show()

#
with unpooled_model:
    post_pred = pm.sample_ppc(trace2, samples=1000)
plt.figure()
ax = sns.distplot(post_pred['Observed'].mean(axis=1), label='Posterior predictive means')
ax.axvline(elec_faults.mean(), color='r', ls='--', label='True mean')
ax.legend()
plt.show()
# 自相关
tracedf = pm.trace_to_dataframe(trace2, varnames=['beta1', 'beta2'])
sns.pairplot(tracedf)
plt.show()


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

hpd25_u_0 = hpd2_5[5:6].mean()
hpd975_u_0 = hpd97_5[5:6].mean()
hpd25_u_1 = hpd2_5[6:7].mean()
hpd975_u_1 = hpd97_5[6:7].mean()
hpd25_u_2 = hpd2_5[7:8].mean()
hpd975_u_2 = hpd97_5[7:8].mean()


hpd25_sigma = hpd2_5[8:9].mean()
hpd975_sigma = hpd97_5[8:9].mean()

# mu数据
# post_beta1 = np.mean(chain2['beta1'][:, 0])
# post_beta11 = np.mean(chain2['beta1'][:, 1])
# post_beta111 = np.mean(chain2['beta1'][:, 2])
#
# post_beta = np.mean(chain2['beta'])

#
# #
# post_beta2 = np.mean(chain2['beta2'])
# post_sigma_beta = np.mean(chain2['sigma'])
# post_u0 = np.mean(chain2['u'][:, 0])
# post_u1 = np.mean(chain2['u'][:, 1])
# post_u2 = np.mean(chain2['u'][:, 2])

hpdmean = bbb['mean']
post_beta = hpdmean[0]
post_beta1 = hpdmean[1]
post_beta11 = hpdmean[2]
post_beta111 = hpdmean[3]
post_beta2 = hpdmean[4]
post_u0 = hpdmean[5]
post_u1 = hpdmean[6]
post_u2 = hpdmean[7]
post_sigma_beta = hpdmean[8]

post_beta0 = post_beta
post_beta00 = post_beta
# 计算平均故障率，即在时间与温度双重作用下的频率故障率变化
alphaa = (post_beta + post_beta1 * elec_year[0:6] + post_beta2 * elec_tem[0:6] + post_u0)
alphaa1 = (post_beta0 + post_beta11 * elec_year[42:48] + post_beta2 * elec_tem[42:48] + post_u1)
alphaa2 = (post_beta00 + post_beta111 * elec_year[84:90] + post_beta2 * elec_tem[84:90] + post_u2)
Mean_gamma = post_sigma_beta * gamma((1 + 1 / alphaa))
Mean_gamma1 = post_sigma_beta * gamma((1 + 1 / alphaa1))
Mean_gamma2 = post_sigma_beta * gamma((1 + 1 / alphaa2))

# 计算后验均值区 间97.5
alphaa975  = (hpd975_beta + hpd975_beta1_0 * elec_year[0:6] + hpd975_beta2 * elec_tem[0:6] + hpd975_u_0)
alphaa9751 = (hpd975_beta + hpd975_beta1_1 * elec_year[42:48] + hpd975_beta2 * elec_tem[42:48] + hpd975_u_1)
alphaa9752 = (hpd975_beta + hpd975_beta1_2 * elec_year[84:90] + hpd975_beta2 * elec_tem[84:90] + hpd975_u_2)
Mean_gamma975 = hpd975_sigma * gamma((1 + 1 / alphaa975))
Mean_gamma9751 = hpd975_sigma * gamma((1 + 1 / alphaa9751))
Mean_gamma9752 = hpd975_sigma * gamma((1 + 1 / alphaa9752))


# 计算后验均值区间2.5
alphaa25  = (hpd25_beta + hpd25_beta1_0 * elec_year[0:6] + hpd25_beta2 * elec_tem[0:6] + hpd25_u_0)
alphaa251 = (hpd25_beta + hpd25_beta1_1 * elec_year[42:48] + hpd25_beta2 * elec_tem[42:48] + hpd25_u_1)
alphaa252 = (hpd25_beta + hpd25_beta1_2 * elec_year[84:90] + hpd25_beta2 * elec_tem[84:90] + hpd25_u_2)
Mean_gamma25 = hpd25_sigma * gamma((1 + 1 / alphaa25))
Mean_gamma251 = hpd25_sigma * gamma((1 + 1 / alphaa251))
Mean_gamma252 = hpd25_sigma * gamma((1 + 1 / alphaa252))


# 画图
plt.figure(figsize=(5, 3), facecolor=(1,1,1))
ax = plt.subplot(1, 1, 1)
j, k1 = 0, 6
for jx in range(7):
    k1 = 6
    plt.plot(elec_year[j:(j + k1)], elec_faults[j:(j + k1)], 'k--', linewidth=0.9)
    j = j + k1
plt.plot(elec_year[0:6], alphaa, linewidth=4)
plt.plot(elec_year[0:6], alphaa25, linewidth=4)
plt.plot(elec_year[0:6], alphaa975, linewidth=4)
# plt.plot(elec_year[0:6], Mean_gamma975, linewidth=4, color='#006400')
# plt.plot(elec_year[0:6], Mean_gamma, linewidth=4, color='#0000FF')
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
plt.plot(elec_year[42:48], alphaa1, linewidth=4)
plt.plot(elec_year[0:6], alphaa251, linewidth=4)
plt.plot(elec_year[0:6], alphaa9751, linewidth=4)
# plt.plot(elec_year[0:6], Mean_gamma9751, linewidth=4, color='#006400')
# plt.plot(elec_year[0:6], Mean_gamma1, linewidth=4, color='#0000FF')
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
plt.plot(elec_year[84:90], alphaa2, linewidth=4)
plt.plot(elec_year[0:6], alphaa252, linewidth=4)
plt.plot(elec_year[0:6], alphaa9752, linewidth=4)
# plt.plot(elec_year[0:6], Mean_gamma9752, linewidth=4, color='#006400')
# plt.plot(elec_year[0:6], Mean_gamma2, linewidth=4, color='#0000FF')
# plt.plot(elec_year[0:6], Mean_gamma252, linewidth=4, color='#B22222')
plt.xlabel(u"时间t/年", fontsize=14, fontproperties=font)
plt.ylabel(u"故障率/%", fontsize=14, fontproperties=font)
ax.legend([u'平均故障', u'故障2.5', u'故障97.5'], loc='upper left', prop=font)
plt.grid()
plt.show()


# 可靠度计算
post_alpha = np.mean(chain2['mu'])
post_sigma1= np.mean(chain2['sigma'])

aaa = pm.df_summary(trace2, varnames1)
bbb = pd.DataFrame(aaa)

hpd2_5 = bbb['hpd_2.5']
hpd97_5 = bbb['hpd_97.5']
hpd2_5_muvalue = hpd2_5[1:].mean()
hpd97_5_muvalue = hpd97_5[1:].mean()
hpd25_sigmavalue = hpd2_5[:1].mean()
hpd975_sigmavalue = hpd97_5[:1].mean()

# 可靠度函数
t = np.arange(1, 7, 1)
R1 = np.exp(-((t/post_sigma1)**post_alpha))
R2 = np.exp(-((t/hpd25_sigmavalue)**hpd2_5_muvalue))
R3 = np.exp(-((t/hpd975_sigmavalue)**hpd97_5_muvalue))
plt.plot(t, R1, 'ko--', t, R2, 'b-', t, R3, 'r' )
plt.show()

print(pm.dic(trace2, unpooled_model))
