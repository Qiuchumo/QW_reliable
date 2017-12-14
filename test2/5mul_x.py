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

from matplotlib.font_manager import FontProperties
# from pymc3 import get_data
font = FontProperties(fname=r"C:\\WINDOWS\\Fonts\\simsun.ttc", size=14)
np.set_printoptions(precision=0, suppress=True)
# 2017.11.11编辑  可靠性分析项目
# 撰写人：邱楚陌
# 单省数据分析，用于调试，weibull
# ======================================================================
# 数据导入
# companies：代表统一产品的测试地点类别    company：测试地点的搜索索引
# companiesABC：代表不同公司类别           companyABC：公司的搜索索引
# ======================================================================
elec_data = pd.read_csv('XZA.csv')

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


# plt.hist(elec_faults, range=[0, 5], bins=130, histtype='stepfilled', color='#6495ED')
# plt.axvline(elec_faults.mean(), color='r', ls='--', label='True mean')
# plt.show()

# 这个为单个模型结构示意，且对同一个分布含有两种选择
# 这里将威布尔参数的alpha 与 beta交换
with pm.Model() as unpooled_model:
    sigma = pm.HalfCauchy('sigma', 10, testval=1.)

    # BoundedNormal = pm.Bound(pm.Normal, lower=3)
    # switchpoint = BoundedNormal('switchpoint', 5, 5)
    # switchpoint = pm.HalfNormal('switchpoint', 2)
    early_rate = pm.Normal('early_rate', 0, 100)
    late_rate = pm.Normal('late_rate', 0, 100)
    beta1 = pm.math.switch(x_shared <= 5, early_rate, late_rate)
    beta = pm.Normal('beta', 0, 100)
    u = pm.Normal('u', 0, 0.0001)

    mu = pm.Deterministic('mu', tt.exp(beta + beta1*x_shared + u))
    # Observed_pred = pm.Weibull("Observed_pred",  alpha=mu, beta=sigma, shape=elec_faults.shape)  # 观测值
    Observed = pm.Weibull("Observed", alpha=sigma, beta=mu, observed=y_shared)  # 观测值

    start = pm.find_MAP()
    # step = pm.Metropolis([switchpoint])
    trace2 = pm.sample(3000,  start=start)
chain2 = trace2[1000:]
varnames2 = ['beta', 'early_rate', 'late_rate','sigma','u']


pm.traceplot(chain2)
plt.show()
pm.energyplot(trace2)
plt.show()
# # 画出自相关曲线
# pm.autocorrplot(chain2, varnames2)
# plt.show()
print(pm.dic(trace2, unpooled_model))

# x_shared.set_value([6, 6, 7])
# x_shared1.set_value([20, 40, 40])
# y_shared.set_value([0, 0, 0])
elec_year1 = np.delete(elec_year, np.s_[:6])
elec_year1 = np.append([2,3,4,5,6,7], elec_year1)
x_shared.set_value(elec_year1)
with unpooled_model:
    trace3 = pm.sample(3000)
    post_pred = pm.sample_ppc(trace3)
abc = post_pred['Observed'].mean(axis=0)
print(abc)

print(pm.df_summary(trace2, varnames2))
# 读取后验区间，加.mean()是为了转换为np型数据便于计算
aaa = pm.df_summary(trace2, varnames2)
bbb = pd.DataFrame(aaa)
hpd2_5 = bbb['hpd_2.5']
hpd97_5 = bbb['hpd_97.5']

hpd25_beta = hpd2_5[:1].mean()
hpd975_beta = hpd97_5[:1].mean()

hpd25_early_rate = hpd2_5[1:2].mean()
hpd975_early_rate = hpd97_5[1:2].mean()

hpd25_late_rate = hpd2_5[2:3].mean()
hpd975_late_rate = hpd97_5[2:3].mean()

hpd25_sigma = hpd2_5[3:4].mean()
hpd975_sigma = hpd97_5[3:4].mean()
#
with unpooled_model:
    post_pred = pm.sample_ppc(trace2, samples=1000)
plt.figure()
ax = sns.distplot(post_pred['Observed'].mean(axis=1), label='Posterior predictive means')
ax.axvline(elec_faults.mean(), color='r', ls='--', label='True mean')
ax.legend()
plt.show()

hpdmean = bbb['mean']
post_beta = hpdmean[0].mean()
post_early_rate = hpdmean[1].mean()
post_late_rate = hpdmean[2].mean()
post_sigma = hpdmean[3].mean()

# 计算平均故障率，即在时间与温度双重作用下的频率故障率变化
alphaa = np.exp(post_beta + elec_year[0:5]*post_early_rate)
aaaa = np.exp(elec_year[5]*post_late_rate+post_beta)
alphaa = np.append(alphaa, aaaa)
alphaa00 = (post_beta + elec_year[0:5]*post_early_rate)
aaaa = (elec_year[5]*post_late_rate+post_beta)
alphaa00 = np.append(alphaa00, aaaa)
Mean_gamma = alphaa * gamma((1 + 1 / post_sigma))

# # 计算后验均值区 间97.5
# alphaa975  = np.exp(hpd975_beta + hpd975_beta1_0 * elec_year[0:6])
# Mean_gamma975 = hpd975_sigma * gamma((1 + 1 / alphaa975))
#
# # 计算后验均值区间2.5
# alphaa25  = np.exp(hpd25_beta + hpd25_beta1_0 * elec_year[0:6])
# Mean_gamma25 = hpd25_sigma * gamma((1 + 1 / alphaa25))

# 画图
plt.figure(figsize=(5, 3), facecolor=(1,1,1))
ax = plt.subplot(1, 1, 1)
j, k1 = 0, 6
for jx in range(7):
    k1 = 6
    plt.plot(elec_year[j:(j + k1)], elec_faults[j:(j + k1)], 'k--', linewidth=0.9)
    j = j + k1
plt.plot(elec_year[0:6], alphaa, linewidth=4)
# plt.plot(elec_year[0:6], alphaa25, linewidth=4)
plt.plot(elec_year[0:6], alphaa00, linewidth=4)
# plt.plot(elec_year[0:6], alphaa975, linewidth=4)
# plt.plot(elec_year[0:6], Mean_gamma975, linewidth=4, color='#006400')
plt.plot(elec_year[0:6], Mean_gamma, linewidth=6, color='#0000FF')
# plt.plot(elec_year[0:6], Mean_gamma25, linewidth=4, color='#B22222')
plt.xlabel(u"时间t/年", fontsize=14, fontproperties=font)
plt.ylabel(u"故障率/%", fontsize=14, fontproperties=font)
ax.legend([ u'平均故障', u'故障2.5', u'故障97.5'], loc='upper left', prop=font)
plt.grid()
plt.show()


