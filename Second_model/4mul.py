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
elec_RH = elec_data.RH.values  # 观测压强x3
elec_RH1 = (elec_RH-np.mean(elec_RH))/np.std(elec_RH)
# 计算故障率大小：故障数目/总测量数，作为模型Y值，放大1000倍以增加实际效果，结果中要缩小1000倍
# elec_fault = elec_data.Fault / elec_data.Nums
elec_faults = 1000*(elec_data.Fault.values / elec_data.Nums.values) # 数组形式
elec_faults1 = (elec_faults-np.mean(elec_faults))/np.std(elec_faults)

# 画出原始图
Company_names = ['XiZang', 'XinJiang', 'HeiLongJiang']
k = np.array([0, 41, 90, 132])
j, k1 = 0, 6
plt.figure(figsize=(10, 10))
for ix in range(3):
    plt.subplot(2, 2, ix+1)
    if ix == 1:
        k1 = 7
    else:
        k1 = 6
    for jx in range(7):
        plt.plot(elec_year[j:(j+k1)], elec_faults[j:(j+k1)], 'ko--')
        j = j+k1
    plt.xlabel('$x_{}$'.format(ix), fontsize=16)
    plt.ylabel('$y_{}$'.format(ix), rotation=0, fontsize=16)
    plt.title('%s' % (Company_names[ix]))
    k[ix+1] = k[ix+1]+1
plt.tight_layout()
plt.show()

# # 所有图显示在一张图上面
# plt.figure(figsize=(10, 10))
# j, k1 = 0, 6
# for jx in range(21):
#     if (jx > 6) & (jx < 14):
#         k1 = 7
#     else:
#         k1 = 6
#     plt.plot(elec_year[j:(j + k1)], elec_faults[j:(j + k1)], 'ko--')
#     j = j + k1
# plt.show()
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
    # beta3 = pm.Normal('beta3', 0, 10)
    # beta4 = pm.Normal('beta4', 0, 10)

    # define likelihood 建立与时间相关的函数
    # theta = beta + beta1 * elec_year + beta2 * elec_tem1 + beta3 * elec_hPa1 + beta4 * elec_RH1
    theta = beta + beta1*elec_year1 + beta2*elec_tem1
    Observed = pm.StudentT("Observed", mu=theta, sd=sigma, nu=nu, observed=elec_faults1)  # 观测值

    start = pm.find_MAP()
    # step = pm.Metropolis()
    trace1 = pm.sample(4000, start=start)
chain1 = trace1[1000:]
varnames1 = ['beta', 'beta1', 'beta2']
pm.traceplot(chain1, varnames1)
plt.show()
print(pm.df_summary(trace1, varnames1))
# 画出自相关曲线
pm.autocorrplot(chain1)
plt.show()



faults_m = np.mean(elec_faults)
faults_sd = np.std(elec_faults)
year_m = np.mean(elec_year)
year_std = np.std(elec_year)
tem_m = np.mean(elec_tem)
tem_std = np.std(elec_tem)
hPa_m = np.mean(elec_hPa)
hPa_std = np.std(elec_hPa)
RH_m = np.mean(elec_RH)
RH_std = np.std(elec_RH)

# 数据
post_beta = chain1['beta']
post_beta1 = chain1['beta1']
post_beta2 = chain1['beta2']
# post_beta3 = chain1['beta3']
# post_beta4 = chain1['beta4']
# post_sigma = chain1['sigma']

# 数据还原
org_beta1 = (post_beta1*faults_sd/year_std).mean()
org_beta2 = (post_beta2*faults_sd/tem_std).mean()
org_beta = (post_beta*faults_sd + faults_m - post_beta1*faults_sd*year_m/year_std - post_beta2*faults_sd*tem_m/tem_std).mean()

# 后验
plt.figure(figsize=(10, 10))
idx = np.argsort(elec_year)
x_ord = elec_year[idx]

idx1 = np.argsort(elec_tem1)
x_ord1 = [idx1]

ppc = pm.sample_ppc(chain1, samples=2000, model=pooled_model)
sig_y = pm.hpd(ppc['Observed'], alpha=0.5)[idx]
sig_y1 = pm.hpd(ppc['Observed'], alpha=0.05)[idx]
plt.fill_between(x_ord, sig_y[:, 0], sig_y[:, 1], color='gray', alpha=1)
plt.fill_between(x_ord, sig_y1[:, 0], sig_y1[:, 1], color='gray', alpha=0.5)

j, k1 = 0, 6
for jx in range(21):
    if (jx > 6) & (jx < 14):
        k1 = 7
    else:
        k1 = 6
    plt.plot(elec_year[j:(j + k1)], elec_faults[j:(j + k1)], 'k--', linewidth=0.8)
    plt.plot(elec_year[j:(j + k1)], org_beta + org_beta1 * elec_year[j:(j + k1)] + org_beta2 * elec_tem[j:(j + k1)], linewidth=5)
    j = j + k1

plt.show()



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
# # chain2 = trace2
# # varnames1 = ['beta', 'beta1', 'beta2', 'beta3', 'beta4']
# # pm.traceplot(chain2, varnames1)
# # plt.show()
# #
# # # 画出自相关曲线
# # pm.autocorrplot(chain2)
# # plt.show()


# ======================================================================
# # 模型表明，添加x1*x2这种参数后，有较好表现
# #partial_model 部分集中模型
with pm.Model() as mulpartial_model:
    # define priors InverseGamma
    sigma = pm.HalfCauchy('sigma', 5)
    # nu = pm.Exponential('nu', 1/30)
    # mu_a = pm.Uniform('mu_a', -10, 10)
    # sigma_a = pm.HalfNormal('sigma_a', sd=10)
    mu_a = pm.Uniform('mu_a', -20, 20)
    sigma_a = pm.HalfNormal('sigma_a', sd=5)

    beta = pm.Normal('beta', mu=mu_a, sd=sigma_a, shape=companiesABC)
    beta1 = pm.Normal('beta1', 0, 10)
    beta2 = pm.Normal('beta2', 0, 10)
    beta3 = pm.Normal('beta3', 0, 10)
    beta4 = pm.Normal('beta4', 0, 10)
    beta5 = pm.Normal('beta5', 0, 10)

    # define likelihood 建立与时间相关的函数
    theta = beta[companyABC] + beta1*elec_year + beta2*elec_tem1 + beta3*elec_hPa1 + beta4*elec_RH1 + beta5*elec_tem1*elec_RH1
    # theta = beta[companyABC] + beta1 * elec_year + beta2 * elec_tem1+ beta4 * elec_RH1 + beta5 * elec_tem1 * elec_RH1
    theta1 = pm.Deterministic('theta1', theta)
    beta12 = pm.Deterministic('beta12', beta1-beta2)
    Observed = pm.Normal("Observed", mu=theta, sd=sigma, observed=elec_faults1)  # 观测值

    start = pm.find_MAP()
    # step = pm.Metropolis()
    trace3 = pm.sample(5000, start=start, tune=2000)
chain3 = trace3[2000:]
varnames1 = ['beta', 'beta1', 'beta2', 'beta3', 'beta4', 'beta5']
pm.traceplot(chain3, varnames1)
plt.show()
print(pm.df_summary(trace3, varnames1))
varnames1 = ['sigma', 'mu_a', 'sigma_a', 'theta1', 'beta12']
pm.traceplot(chain3, varnames1)
plt.show()
print(pm.df_summary(trace3, varnames1))
# # 画出自相关曲线
pm.autocorrplot(chain3)
plt.show()

pm.energyplot(chain3)
plt.show()

post_beta = chain3['beta'] # y有三行数据
post_beta1 = chain3['beta'][:, 2] # 采用这种读法即可
post_beta2 = chain3['beta'][1] # 这种读法应该也可以
#
#
# # 画出参数间的自相关
# tracedf = pm.trace_to_dataframe(trace3, varnames=['beta1', 'beta2', 'beta3', 'beta4', 'beta5'])
# sns.pairplot(tracedf)
# plt.show()
# ======================================================================
# 模型对比与后验分析
# ======================================================================
# Waic = pm.compare([traces_ols_glm, trace1], [mdl_ols_glm, pooled_model], ic='WAIC')
# Waic = pm.compare([trace2, trace3], [partial_model, mulpartial_model], ic='WAIC')
# print(Waic)




