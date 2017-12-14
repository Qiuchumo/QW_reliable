import pymc3 as pm
import numpy as np
import seaborn as sns
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import theano.tensor as T
import scipy as sp
from theano import shared
from theano.compile.ops import as_op
import theano.tensor as tt
from numpy import arange, array, empty

# from pymc3 import get_data
blue, green, red, purple, gold, teal = sns.color_palette()
np.set_printoptions(precision=0, suppress=True)
# 2017.11.20编辑  可靠性分析项目
# 撰写人：邱楚陌
# =============================================================================
# 数据导入
# companies：代表统一产品的测试地点类别    company：测试地点的搜索索引
# companiesABC：代表不同公司类别           companyABC：公司的搜索索引
# =============================================================================
dag_data = np.genfromtxt("E:/Code/Bayescode/QW_reliable/Third_model_spline/XZA.csv",
                         skip_header=1, usecols=[1, 2, 3, 4, 5, 6, 7, 8], delimiter=",")
elec_data = pd.read_csv('XZB.csv')

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
elec_RH = elec_data.RH.values  # 观测湿度x3
elec_RH1 = (elec_RH - np.mean(elec_RH)) / np.std(elec_RH)
# 计算故障率大小：故障数目/总测量数，作为模型Y值，放大1000倍以增加实际效果，结果中要缩小1000倍
# elec_fault = elec_data.Fault / elec_data.Nums
elec_faults = 1000 * (elec_data.Fault.values / elec_data.Nums.values)  # 数组形式
elec_faults1 = (elec_faults - np.mean(elec_faults)) / np.std(elec_faults)

# 假设模型中有丢失的数据，将模型中部分数据设定为丢失状态
# elec_faults[10] = -999
# elec_faults[20] = -999
# elec_faults_miss = np.ma.masked_values([elec_faults], value=-999)
# plt.plot(elec_year, elec_faults, 'ko--')
# plt.show()
# j, k1 = 0, 6
# for jx in range(21):
#     plt.plot(elec_year[j:(j + k1)], elec_faults[j:(j + k1)], 'ko--')
#     j = j + k1
# plt.show()
# ======================================================================
# 模型建立：
# using pymc3 GLM自建立模型，Normal分布更优
# 采用三次样条基函数进行拟合
# ======================================================================
x_zhou = 7
Num = len(elec_faults1)
knots = np.linspace(1, x_zhou, Num)

Num_5 = 5 * len(elec_faults1)
model_knots = np.linspace(1, x_zhou, Num_5)

# 能否将这里代码改为@as_op的形式，来让x的值得以调用
basis_funcs = sp.interpolate.BSpline(knots, np.eye(Num_5), k=3) # eye()生成对角矩阵
Bx = basis_funcs(elec_year)  # 表示在取值为x时的插值函数值
# shared:符号变量（symbolic variable），a之所以叫shared variable是因为a的赋值在不同的函数中都是一致的搜索，即a是被shared的
Bx_ = shared(Bx)

# elec_faults_shared = shared(elec_faults)
# 撰写自己的函数
@as_op(itypes=[tt.dvector, tt.dscalar], otypes=[tt.dvector])
def rate_(a00, tau):
    out = empty(Num_5)
    out[0] = a00[0]
    # for i in pp:
    ii = 1
    while ii < Num_5:
        out[ii] = out[ii-1] + a00[ii] * tau
        ii = ii + 1
    return out

# #样条模型，加上平滑先验，稳定版本
with pm.Model() as partial_model:
    # define priors
    sigma = pm.HalfCauchy('sigma', 10)

    tau1 = pm.Normal('tau1', 0, 1)
    a_0 = pm.Normal('a_0', 0, 10, shape=Num_5, testval=.2)
    Δ_a = rate_(a_0, tau1)
    # σ_a = pm.HalfCauchy('σ_a', 5.)
    a0 = pm.Normal('a0', 0., 20.)

    δ_1 = pm.Gamma('δ_1', alpha=5, beta=1)
    δ = pm.Normal('δ', 0, sd=(δ_1 * δ_1))
    # δ = pm.Normal('δ', 0, sd=20) # 若模型收敛差则δ改用这个语句
    theta1 = pm.Deterministic('theta1', a0 + (Δ_a).cumsum())

    theta = Bx_.dot(theta1) + δ
    Observed = pm.Normal('Observed', mu=theta, sd=sigma, observed=elec_faults)  # 观测值

    # start = pm.find_MAP()
    step1 = pm.Slice([tau1, a_0])
    trace2 = pm.sample(1000, tune=500, step=step1)
chain2 = trace2
varnames1 = [ 'a0', 'δ', 'sigma', 'tau1']
pm.plot_posterior(chain2, varnames1, kde_plot=True)
plt.show()

pm.energyplot(chain2)  # 能量图对比，重合度越高表示模型越优
plt.show()
# 画出自相关曲线
varnames1 = [ 'a0', 'δ', 'sigma', 'tau1']
pm.autocorrplot(chain2, varnames1)
plt.show()
print(pm.df_summary(chain2, varnames1))

print(pm.waic(trace=trace2, model=partial_model))
# ======================================================================
# 后验分析：
# 画出后验与原始图形对比图
#
# ======================================================================
# Bx_.set_value([7,8] , [5,6])
with partial_model:
    pp_trace = pm.sample_ppc(trace2, 1000)

# pp_trace['Observed'].mean(axis=0)

fig, ax = plt.subplots(figsize=(8, 6))
# ax.plot(x_plot, spline(x_plot), c='k', label="True function")
# 原始图形
# j, k1 = 0, 6
# for jx in range(21):
#     plt.plot(elec_year[j:(j + k1)], elec_faults[j:(j + k1)], 'ko--')
#     j = j + k1
j, k1 = 0, 7
x_plot = np.linspace(1, k1, Num / 7)
# x_plot.sort()
# plt.plot(elec_year, elec_faults, 'ko--', linewidth=0.5)

for jx in range(21):
    plt.plot(elec_year[j:(j + k1)], elec_faults[j:(j + k1)], 'ko--')
    j = j + k1

low, high = np.percentile(pp_trace['Observed'], [5, 95], axis=0)
low1 = low[:k1]
high1 = high[:k1]
aaa = pp_trace['Observed'].mean(axis=0)[:k1]
ax.fill_between(x_plot, low1, high1, color=red, alpha=0.5)
ax.plot(x_plot, pp_trace['Observed'].mean(axis=0)[:k1], c=red, label="Spline estimate")

# ax.scatter(x, y, alpha=0.75, zorder=5)
ax.set_xlim(0, 8)
ax.legend()
plt.show()



