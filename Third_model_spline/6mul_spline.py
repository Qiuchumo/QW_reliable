import pymc3 as pm
import numpy as np
import seaborn as sns
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import theano.tensor as T
import scipy as sp
from theano import shared

# from pymc3 import get_data
blue, green, red, purple, gold, teal = sns.color_palette()
np.set_printoptions(precision=0, suppress=True)
# 2017.11.19编辑  可靠性分析项目
# 撰写人：邱楚陌
# =============================================================================
# 数据导入
# companies：代表统一产品的测试地点类别    company：测试地点的搜索索引
# companiesABC：代表不同公司类别           companyABC：公司的搜索索引
# =============================================================================
dag_data = np.genfromtxt("E:/Code/Bayescode/QW_reliable/Third_model_spline/XZA.csv",
                         skip_header=1, usecols=[1, 2, 3, 4, 5, 6, 7, 8], delimiter=",")
elec_data = pd.read_csv('XZA.csv')

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

# 假设模型中有丢失的数据，将模型中部分数据设定为丢失状态
elec_faults[10] = -999; elec_faults[20] = -999
elec_faults_miss = np.ma.masked_values([elec_faults], value=-999)
# plt.plot(elec_year, elec_faults, 'ko--')
# plt.show()
j, k1 = 0, 6
for jx in range(21):
    plt.plot(elec_year[j:(j + k1)], elec_faults[j:(j + k1)], 'ko--')
    j = j + k1
plt.show()
# ======================================================================
# 模型建立：
# using pymc3 GLM自建立模型，Normal分布更优
# 采用三次样条基函数进行拟合
# ======================================================================
Num = len(elec_faults1)
knots = np.linspace(1, 6, Num)

Num_5 = 5 * len(elec_faults1)
model_knots = np.linspace(1, 6, Num_5)

basis_funcs = sp.interpolate.BSpline(knots, np.eye(Num_5), k=3)
Bx = basis_funcs(elec_year) # 表示在取值为x时的插值函数值
# shared:符号变量（symbolic variable），a之所以叫shared variable是因为a的赋值在不同的函数中都是一致的搜索，即a是被shared的
Bx_ = shared(Bx)


# #样条模型
with pm.Model() as partial_model:
    # define priors
    sigma = pm.HalfCauchy('sigma', 5)

    σ_a = pm.HalfCauchy('σ_a', 5.)
    a0 = pm.Normal('a0', 0., 10.)
    Δ_a = pm.Normal('Δ_a', 0., 10., shape=Num_5)
    δ_1 = pm.Gamma('δ_1', alpha=5, beta=1)
    δ = pm.Normal('δ', 0, sd = (δ_1*δ_1))
    # δ = pm.Normal('δ', 0, sd=100) # 若模型收敛差则δ改用这个语句
    theta1 = pm.Deterministic('theta1', a0 + (σ_a * Δ_a).cumsum())
    # theta1 = a0 + (σ_a * Δ_a).cumsum()

    theta = Bx_.dot(theta1) + δ
    Observed = pm.Normal('Observed', mu=theta, sd=sigma, observed=elec_faults_miss)  # 观测值

    start = pm.find_MAP()
    # step = pm.Metropolis()
    # trace2 = pm.sample(nuts_kwargs={'target_accept': 0.95})
    trace2 = pm.sample(3000, tune=1000, start=start)
chain2 = trace2
varnames1 = ['σ_a', 'a0', 'Δ_a', 'δ', 'theta1']
pm.traceplot(chain2, varnames1)
plt.show()
plt.plot(trace2['step_size_bar'])
plt.show()

pm.energyplot(chain2)  # 能量图对比，重合度越高表示模型越优
plt.show()
print(pm.waic(trace=trace2, model=partial_model))
# 画出自相关曲线
varnames1 = ['σ_a', 'a0', 'δ']
pm.autocorrplot(chain2, varnames1)
plt.show()

# ======================================================================
# 后验分析：
# 画出后验与原始图形对比图
#
# ======================================================================
with partial_model:
    pp_trace = pm.sample_ppc(trace2, 1000)

fig, ax = plt.subplots(figsize=(8, 6))
# ax.plot(x_plot, spline(x_plot), c='k', label="True function")
# 原始图形
# j, k1 = 0, 6
# for jx in range(21):
#     plt.plot(elec_year[j:(j + k1)], elec_faults[j:(j + k1)], 'ko--')
#     j = j + k1
j, k1 = 0, 6
x_plot = np.linspace(1, k1, Num/7)
# x_plot.sort()
# plt.plot(elec_year, elec_faults, 'ko--', linewidth=0.5)

for jx in range(21):
    plt.plot(elec_year[j:(j + k1)], elec_faults[j:(j + k1)], 'ko--')
    j = j + k1

low, high = np.percentile(pp_trace['Observed'], [25, 75], axis=0)
low1 = low[:k1]
high1 = high[:k1]
aaa = pp_trace['Observed'].mean(axis=0)[:k1]
ax.fill_between(x_plot, low1, high1, color=red, alpha=0.5)
ax.plot(x_plot, pp_trace['Observed'].mean(axis=0)[:k1], c=red, label="Spline estimate")

# ax.scatter(x, y, alpha=0.75, zorder=5)
ax.set_xlim(0, 8)
ax.legend()
plt.show()



