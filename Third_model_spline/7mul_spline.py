import pymc3 as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import theano.tensor as T
import scipy as sp
from theano import shared

# from pymc3 import get_data
blue, green, red, purple, gold, teal = sns.color_palette()
np.set_printoptions(precision=0, suppress=True)
# 2017.11.20编辑  可靠性分析项目
# 撰写人：邱楚陌
# ======================================================================
# 数据导入
# companies：代表统一产品的测试地点类别    company：测试地点的搜索索引
# companiesABC：代表不同公司类别           companyABC：公司的搜索索引
# ======================================================================
dag_data = np.genfromtxt("E:/Code/Bayescode/QW_reliable/Third_model_spline/XZA.csv",
                         skip_header=1, usecols=[1, 2, 3, 4, 5, 6, 7, 8], delimiter=",")
elec_data = pd.read_csv('XZA.csv')
elec_dataB = pd.read_csv('XZC.csv')

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

elec_yearB = elec_dataB.Year.values # 观测时间值x1
elec_faultsB = 1000*(elec_dataB.Fault.values / elec_dataB.Nums.values) # 数组形式
# plt.plot(elec_year, elec_faults, 'ko--')
# plt.show()
# j, k1 = 0, 6
# for jx in range(21):
#     plt.plot(elec_year[j:(j + k1)], elec_faults[j:(j + k1)], 'ko--')
#     j = j + k1
# plt.show()
# ======================================================================
# 模型建立：
# 模型1：using pymc3 GLM自建立模型，Normal分布更优
# 模型2: 自己模型
# ======================================================================
Num = len(elec_faults1)
knots = np.linspace(1, 6, Num)

NumB = len(elec_faultsB)
knotsB = np.linspace(1, 6, NumB)

Num_5 = 5 * len(elec_faults1)
model_knots = np.linspace(1, 6, Num_5)
Num_5B = 5 * len(elec_faultsB)
model_knotsB = np.linspace(1, 6, Num_5B)

basis_funcs = sp.interpolate.BSpline(knots, np.eye(Num_5), k=3)
Bx = basis_funcs(elec_year) # 表示在取值为x时的插值函数值
basis_funcsB = sp.interpolate.BSpline(knotsB, np.eye(Num_5B), k=3)
BxB = basis_funcs(elec_yearB) # 表示在取值为x时的插值函数值
# shared:符号变量（symbolic variable），a之所以叫shared variable是因为a的赋值在不同的函数中都是一致的搜索，即a是被shared的
Bx_ = shared(Bx)
Bx_B = shared(BxB)
# #将两个数据一起计算，其中可以共用部分数据，但如何将环境变量加进去
with pm.Model() as partial_model:
    # define priors
    sigma = pm.HalfCauchy('sigma', 10)

    σ_a = pm.HalfCauchy('σ_a', 5.)
    σ_aB = pm.HalfCauchy('σ_aB', 5.)
    a0 = pm.Normal('a0', 0., 20.)

    Δ_a = pm.Normal('Δ_a', 0., 10., shape=Num_5)
    Δ_aB = pm.Normal('Δ_aB', 0., 10., shape=Num_5B)
    # δ_1 = pm.Gamma('δ_1', alpha=0.000001, beta=0.000001)
    # δ = pm.Normal('δ', 0, sd = (δ_1*δ_1))
    δ = pm.Normal('δ', 0, sd=100)
    δB = pm.Normal('δB', 0, sd=100)
    theta1 = pm.Deterministic('theta1', a0 + (σ_a * Δ_a).cumsum())
    theta1B = pm.Deterministic('theta1B', a0 + (σ_aB * Δ_aB).cumsum())
    # theta1 = a0 + (σ_a * Δ_a).cumsum()

    theta = Bx_.dot(theta1) + δ
    thetaB = Bx_B.dot(theta1B) + δB
    ObservedA = pm.Normal('ObservedA', mu=theta, sd=sigma, observed=elec_faults)  # 观测值
    ObservedB = pm.Normal('ObservedB', mu=thetaB, sd=sigma, observed=elec_faultsB)  # 观测值

    start = pm.find_MAP()
    # step = pm.Metropolis()
    # trace2 = pm.sample(nuts_kwargs={'target_accept': 0.95})
    trace2 = pm.sample(3000, tune=1000)
chain2 = trace2
varnames1 = ['σ_a', 'σ_aB',  'theta1', 'theta1B']
pm.traceplot(chain2, varnames1)
plt.show()
varnames1 = ['a0', 'sigma', 'δ', 'δB', 'Δ_a', 'Δ_aB']
pm.traceplot(chain2, varnames1)
plt.show()

plt.plot(trace2['step_size_bar'])
plt.show()

pm.energyplot(chain2)
plt.show()
# 画出自相关曲线
varnames1 = ['σ_a', 'a0', 'δ', 'σ_aB', 'Δ_a', 'δB']
pm.autocorrplot(chain2, varnames1)
plt.show()


# 后验分析
with partial_model:
    pp_trace = pm.sample_ppc(trace2, 1000)

fig, ax = plt.subplots(figsize=(8, 6))

j, k1 = 0, 6
x_plot = np.linspace(1, k1, Num/7)

for jx in range(8):
    plt.plot(elec_year[j:(j + k1)], elec_faults[j:(j + k1)], 'ko--')
    j = j + k1

low, high = np.percentile(pp_trace['ObservedA'], [25, 75], axis=0)
low1 = low[:k1]
high1 = high[:k1]
ax.fill_between(x_plot, low1, high1, color=red, alpha=0.5)
ax.plot(x_plot, pp_trace['ObservedA'].mean(axis=0)[:k1], c=red, label="Spline estimate")

# ax.scatter(x, y, alpha=0.75, zorder=5)
ax.set_xlim(0, 8)
ax.legend()
plt.show()

fig, ax = plt.subplots(figsize=(10, 8))
j, k1 = 0, 6
x_plotB = np.linspace(1, k1, NumB/7)
for jx in range(8):
    plt.plot(elec_yearB[j:(j + k1)], elec_faultsB[j:(j + k1)], 'ko--')
    j = j + k1

lowB, highB = np.percentile(pp_trace['ObservedB'], [25, 75], axis=0)
lowB = lowB[:k1]
highB = highB[:k1]
ax.fill_between(x_plotB, lowB, highB, color=gold, alpha=0.5)
ax.plot(x_plotB, pp_trace['ObservedB'].mean(axis=0)[:k1], c=red, label="Spline estimate")

ax.set_xlim(0, 8)
ax.legend()
plt.show()

