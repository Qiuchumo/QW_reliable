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

# 以下三行用于中文显示图形
from matplotlib.font_manager import FontProperties
# from pymc3 import get_data
font = FontProperties(fname=r"C:\\WINDOWS\\Fonts\\simsun.ttc", size=14)
np.set_printoptions(precision=0, suppress=True)
# 2017.12.3编辑  可靠性分析项目，两省数据分析，用于北大核心小论文
# 撰写人：邱楚陌
# ======================================================================
# 数据导入
# companies：代表统一产品的测试地点类别    company：测试地点的搜索索引
# companiesABC：代表不同公司类别           companyABC：公司的搜索索引
# ======================================================================
elec_data = pd.read_csv('XZAB.csv')

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

# 共享变量设置
train_year = elec_year[2:]
train_abc = companyABC[2:]
train_faults = elec_faults[2:]

test_year = elec_year[:2]
test_abc = companyABC[:2]
# test_faults = elec_faults[-2:]


x_shared = shared(np.asarray(train_year))
# x_shared1 = shared(elec_tem[:-2])
Num_shared = shared(np.asarray(train_abc))
y_shared = shared(np.asarray(train_faults))




elec_year1[0:84] = 7
plt.style.use('default')
plt.hist(elec_faults, range=[0, 5], bins=130, histtype='stepfilled', color='#6495ED')
plt.axvline(elec_faults.mean(), color='r', ls='--', label='True mean')
plt.show()

# # 画出原始图
# Company_names = ['XiZang', 'XinJiang']
# k = np.array([0, 41, 83])
# j, k1 = 0, 6
# plt.figure(figsize=(6, 4.5), facecolor='w')
#
# ax = plt.subplot(1, 1, 1)
#
# for jx in range(7):
#     # ax.imshow(elec_faults[j:(j+k1)])
#     ax.plot(elec_year[j:(j+k1)], elec_faults[j:(j+k1)], 'ko--', markersize=4, linewidth=1)
#     j = j+k1
# ax.set_xticklabels(['2016','2010', '2011', '2012', '2013', '2014', '2015'], fontsize='small')
# ax.set_xlabel(u"时间t/年", fontsize=14, fontproperties=font)
# plt.ylabel(u"故障率/%", fontsize=14, fontproperties=font)
# plt.legend([u'故障率曲线'], loc='upper left', prop=font)
# plt.grid()
# # plt.savefig('2.png', dpi=400)
# plt.savefig('1.svg', format='svg')
# plt.show()
# plt.figure(figsize=(6, 4.5), facecolor='w')
# ax = plt.subplot(1, 1, 1)
# for jx in range(7, 14, 1):
#     ax.plot(elec_year[j:(j+k1)], elec_faults[j:(j+k1)], 'ko--', markersize=4, linewidth=1)
#     j = j+k1
# ax.set_xticklabels(['2016','2010', '2011', '2012', '2013', '2014', '2015'], fontsize='small')
# ax.set_xlabel(u"时间t/年", fontsize=14, fontproperties=font)
# plt.ylabel(u"故障率/%", fontsize=14, fontproperties=font)
# plt.legend([u'故障率曲线'], loc='upper left', prop=font)
# plt.grid()
# #plt.savefig('2.svg', format='svg')
# plt.show()
#


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
#     alpha = pm.HalfCauchy('alpha', 10, testval=.9)
#
#     early_rate = pm.Normal('early_rate', 0, 100)
#     late_rate = pm.Normal('late_rate', 0, 100)
#     beta1 = pm.math.switch(x_shared <= 5, early_rate, late_rate)
#     beta = pm.Normal('beta', 0, 100)
#
#     # mu = tt.exp(beta[companyABC] + beta1[companyABC]*elec_year + beta2*elec_tem)
#     beta_mu = pm.Deterministic('beta_mu', tt.exp(beta + beta1 * x_shared))
#
#     # Observed_pred = pm.Weibull("Observed_pred",  alpha=mu, beta=sigma, shape=elec_faults.shape)  # 观测值
#     Observed = pm.Weibull("Observed", alpha=alpha, beta=beta_mu, observed=y_shared)  # 观测值
#
#     start = pm.find_MAP()
#     # step = pm.Slice([beta1, u])
#     trace1 = pm.sample(2000,  start=start)

with pm.Model() as unpooled_model:
    # define priors
    alpha = pm.HalfCauchy('alpha', 10, testval=.9)

    switch = pm.DiscreteUniform('swich', lower=x_shared.min()+3, upper=x_shared.max()-0.5)
    early_rate = pm.Normal('early_rate', 0, 100)
    late_rate = pm.Normal('late_rate', 0, 100)
    beta1 = pm.math.switch(x_shared <= switch, early_rate, late_rate)
    beta = pm.Normal('beta', 0, 100, shape=companiesABC)
    u = pm.Normal('u', 0, 0.0001)

    # mu = tt.exp(beta[companyABC] + beta1[companyABC]*elec_year + beta2*elec_tem)
    beta_mu = pm.Deterministic('beta_mu', tt.exp(beta[Num_shared] + beta1 * x_shared + u))

    # Observed_pred = pm.Weibull("Observed_pred",  alpha=mu, beta=sigma, shape=elec_faults.shape)  # 观测值
    Observed = pm.Weibull("Observed", alpha=alpha, beta=beta_mu, observed=train_faults)  # 观测值

    start = pm.find_MAP()
    # step = pm.Slice([beta1, u])
    trace2 = pm.sample(3000,  start=start, tune=1000)
chain2 = trace2[1000:]
varnames1 = ['alpha', 'beta_mu', 'swich']
print(pm.df_summary(trace2, varnames1))
varnames2 = ['beta', 'early_rate', 'late_rate', 'alpha', 'u']
# pm.plot_posterior(chain2, varnames2, ref_val=0)
pm.traceplot(chain2)
plt.show()





# 两种能量图
energy = trace2['energy']
energy_diff = np.diff(energy)
sns.distplot(energy - energy.mean(), label='energy')
sns.distplot(energy_diff, label='energy diff')
plt.legend()
plt.show()
pm.energyplot(trace2)
plt.show()
map_estimate = pm.find_MAP(model=unpooled_model)
print(map_estimate)
# 画出自相关曲线
pm.autocorrplot(chain2, varnames2)
plt.show()
print(pm.waic(trace2, unpooled_model))

#
with unpooled_model:
    post_pred = pm.sample_ppc(trace2)
plt.figure(figsize=(6, 4.5), facecolor=(1,1,1))
plt.figure()
# ppc = post_pred['Observed'] # 更改数据排列即可以画出分类图
# ax = sns.violinplot(data=ppc)
# plt.show()

ax = sns.distplot(post_pred['Observed'].mean(axis=1))
# ax = sns.distplot(y_shared.mean(axis=1), label='Posterior predictive means')
# ax.axvline(post_pred['Observed'].mean(), color='b', ls='--', label='Post mean')
ax.axvline(elec_faults.mean(), color='r', ls='--')
ax.set_xlabel(u"故障率均值", fontsize=14, fontproperties=font)
plt.ylabel(u"密度", fontsize=14, fontproperties=font)
ax.legend([u'故障率后验预测均值', u'实际故障率均值'], prop=font)
plt.grid()
# plt.savefig('3.svg', format='svg')
plt.show()
# 自相关
tracedf = pm.trace_to_dataframe(trace2, varnames=['beta', 'early_rate'])
sns.pairplot(tracedf)
plt.show()


# print(pm.df_summary(trace2, varnames1))
print(pm.df_summary(trace2, varnames2))

# 读取后验区间，加.mean()是为了转换为np型数据便于计算
aaa = pm.df_summary(trace2, varnames2)
bbb = pd.DataFrame(aaa)
hpd2_5 = bbb['hpd_2.5']
hpd97_5 = bbb['hpd_97.5']
hpd25_beta0 = hpd2_5[0].mean()
hpd975_beta0 = hpd97_5[0].mean()
hpd25_beta0_1 = hpd2_5[1].mean()
hpd975_beta0_1 = hpd97_5[1].mean()

hpd25_early_rate = hpd2_5[2].mean()
hpd975_early_rate = hpd97_5[2].mean()

hpd25_late_rate = hpd2_5[3].mean()
hpd975_late_rate = hpd97_5[3].mean()

hpd25_alpha = hpd2_5[4].mean()
hpd975_alpha = hpd97_5[4].mean()

hpd25_u = hpd2_5[5].mean()
hpd975_u = hpd97_5[5].mean()
# hpd25_alpha = hpd2_5[4].mean()
# hpd975_alpha = hpd97_5[4].mean()

hpdmean = bbb['mean']
post_beta0 = hpdmean[0]
post_beta01 = hpdmean[1]
post_early_rate = hpdmean[2]
post_late_rate = hpdmean[3]
post_alpha = hpdmean[4]
post_u = hpdmean[5]

# print(post_early_rate)
# print(post_late_rate)
# print(post_beta0)
# 计算平均故障率，即在时间与温度双重作用下的频率故障率变化
betaa = np.exp(post_beta0 + elec_year[0:5]*post_early_rate + post_u)
aaaa = np.exp(elec_year[5]*post_late_rate+post_beta0 +post_u)
betaa = np.append(betaa, aaaa)

betaa1 = np.exp(post_beta01 + elec_year[0:5]*post_early_rate + post_u)
aaaa1 = np.exp(elec_year[5]*post_late_rate+post_beta01 + post_u)
betaa1 = np.append(betaa1, aaaa1)
Mean_gamma = betaa * gamma((1 + 1 / post_alpha))
Mean_gamma1 = betaa1 * gamma((1 + 1 / post_alpha))

# 计算后验均值区 间97.5
betaa975 = np.exp(hpd975_beta0 + elec_year[0:5]*hpd975_early_rate + hpd975_u)
aaaa = np.exp(elec_year[5]*hpd975_late_rate + hpd975_beta0 + hpd975_u)
betaa975 = np.append(betaa975, aaaa)

betaa9751 = np.exp(hpd975_beta0_1 + elec_year[0:5]*hpd975_early_rate + hpd975_u)
aaaa = np.exp(elec_year[5]*hpd975_late_rate + hpd975_beta0_1 + hpd975_u)
betaa9751 = np.append(betaa9751, aaaa)
Mean_gamma975 = betaa975 * gamma((1 + 1 / hpd975_alpha))
Mean_gamma9751 = betaa9751 * gamma((1 + 1 / hpd975_alpha))


# 计算后验均值区间2.5
betaa25 = np.exp(hpd25_beta0 + elec_year[0:5]*hpd25_early_rate + hpd25_u)
aaaa = np.exp(elec_year[5]*hpd25_late_rate + hpd25_beta0 + hpd25_u)
betaa25 = np.append(betaa25, aaaa)

betaa251 = np.exp(hpd25_beta0_1 + elec_year[0:5]*hpd25_early_rate + hpd25_u)
aaaa = np.exp(elec_year[5]*hpd25_late_rate + hpd25_beta0_1 + hpd25_u)
betaa251 = np.append(betaa251, aaaa)

Mean_gamma25 = betaa25 * gamma((1 + 1 / hpd25_alpha))
Mean_gamma251 = betaa251 * gamma((1 + 1 / hpd25_alpha))



# 画出拟合曲线图
plt.figure(figsize=(6, 4.5), facecolor=(1,1,1))
plt.subplot(1, 1, 1)
j, k1 = 0, 6
for jx in range(6):
    k1 = 6
    plt.plot(elec_year[j:(j + k1)], elec_faults[j:(j + k1)], 'k--', linewidth=0.9)
    j = j + k1
ax1, = plt.plot(elec_year[j:(j + k1)], elec_faults[j:(j + k1)], 'k--', linewidth=0.9)
j = j + k1
ax2, = plt.plot(elec_year[0:6], betaa, linewidth=4) # 必须要加逗号才行
ax3, = plt.plot(elec_year[0:6], betaa25, 'ko-', markersize=4, linewidth=2)
ax4, = plt.plot(elec_year[0:6], betaa975, 'k-.', linewidth=2)
# plt.plot(elec_year[0:6], Mean_gamma975, linewidth=2, color='#006400')
# plt.plot(elec_year[0:6], Mean_gamma, linewidth=3, color='#0000FF')
# plt.plot(elec_year[0:6], Mean_gamma25, linewidth=4, color='#B22222')
plt.xlabel(u"时间t/年", fontsize=14, fontproperties=font)
plt.ylabel(u"故障率/%", fontsize=14, fontproperties=font)
plt.legend([ax1, ax2, ax3, ax4], [u'故障率', u'拟和曲线均值', u'故障2.5%', u'故障97.5%'], loc='upper left', prop=font)
plt.grid()
# plt.savefig('4.svg', format='svg')
plt.show()



plt.figure(figsize=(6, 4.5), facecolor=(1,1,1))
ax = plt.subplot(1, 1, 1)
for jx in range(7, 14, 1):
    k1 = 6
    plt.plot(elec_year[j:(j + k1)], elec_faults[j:(j + k1)], 'k--', linewidth=0.9)
    j = j + k1
ax1, = plt.plot(elec_year[j:(j + k1)], elec_faults[j:(j + k1)], 'k--', linewidth=0.9)
j = j + k1
ax2, = plt.plot(elec_year[42:48], betaa1, linewidth=4)
ax3, = plt.plot(elec_year[0:6], betaa251, 'ko-', markersize=4, linewidth=2)
ax4, = plt.plot(elec_year[0:6], betaa9751,'k-.', linewidth=2)
# plt.plot(elec_year[0:6], Mean_gamma9751, linewidth=2, color='#006400')
# plt.plot(elec_year[0:6], Mean_gamma1, linewidth=3, color='#0000FF')
# plt.plot(elec_year[42:48], Mean_gamma251, linewidth=4, color='#B22222')
plt.xlabel(u"时间t/年", fontsize=14, fontproperties=font)
plt.ylabel(u"故障率/%", fontsize=14, fontproperties=font)
plt.legend([ax1, ax2, ax3, ax4], [u'故障率', u'拟和曲线均值', u'故障2.5%', u'故障97.5%'], loc='upper left', prop=font)
plt.grid()
# plt.savefig('5.svg', format='svg')
plt.show()


# 可靠度计算，这里不太对，代码换了就没推了
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
plt.plot(t, R2, 'k-', t, R3, 'r')
ax.legend([u'可靠度区间2.5', u'可靠度均值', u'可靠度区间97.5'], prop=font)
plt.show()



print(pm.dic(trace2, unpooled_model))
A = pm.compare([trace1,trace2], [pooled_model, unpooled_model], ic='WAIC')
print(A)
pm.compareplot(A)
plt.show()





# 进行预测
# elec_year1 = elec_year
# elec_year1[0:84] = 7
# elec_year1[5:42:6] = 7
# elec_year1 = int(np.ones(len(elec_faults))*7)
print(elec_faults.mean())
# elec_faults2 = np.zeros(len(elec_faults))
x_shared.set_value(np.asarray(test_year))
# y_shared.set_value(elec_faults2)
Num_shared.set_value(np.asarray(test_abc))
# print(elec_faults.mean())
with unpooled_model:
    post_pred = pm.sample_ppc(trace2)
predictx = post_pred['Observed']
plt.hist(predictx, normed=1, bins=80, alpha=.8, label='Posterior')
plt.show()
abc = post_pred['Observed'].mean(axis=0) # axis=0以列方式计算
# abcd = abc[5:42:6].mean(axis=0)

print(pm.df_summary(trace2, varnames1))

plt.figure()
ax = sns.distplot(post_pred['Observed'].mean(axis=1), label='Posterior predictive means') # axis=1以行方式计算
ax.axvline(elec_faults.mean(), color='r', ls='--', label='True mean')
ax.axvline(post_pred['Observed'].mean(), color='b', ls='--', label='Post mean')
ax.legend()
plt.show()
print(abc)
print(abc)







