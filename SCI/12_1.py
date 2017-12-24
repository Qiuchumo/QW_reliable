import pymc3 as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import theano.tensor as tt
from theano import shared
import pandas as pd
from sklearn.decomposition import PCA, KernelPCA
from Plot_XZ import *
from PCA import *
from scipy.special import gamma
from theano.compile.ops import as_op

# 以下三行用于中文显示图形
# from matplotlib.font_manager import FontProperties
# from pymc3 import get_data
# font = FontProperties(fname=r"C:\\WINDOWS\\Fonts\\simsun.ttc", size=14)
np.set_printoptions(precision=0, suppress=True)
# 2017.12.19编辑  可靠性分析项目，三省数据分析，用于SCI论文
# 撰写人：邱楚陌
# ======================================================================
# 数据导入
# companies：代表统一产品的测试地点类别    company：测试地点的搜索索引
# companiesABC：代表不同公司类别           companyABC：公司的搜索索引
# ======================================================================
Savefig = 0 # 控制图形显示存储

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

# 给所有特征因素加上高斯噪声
SNR = np.random.normal(0, 2, size=[len(elec_data.Year.values), 4])

# #特征因素分析
elec_tem = elec_data.Tem.values + SNR[:, 0] # 观测温度值x2
elec_tem1 = (elec_tem - np.mean(elec_tem)) / np.std(elec_tem)
elec_hPa = elec_data.hPa.values + SNR[:, 1]  # 观测压强x3
elec_hPa1 = (elec_hPa - np.mean(elec_hPa)) / np.std(elec_hPa)
elec_RH = elec_data.RH.values + SNR[:, 2] # 观测压强x3
elec_RH1 = (elec_RH - np.mean(elec_RH)) / np.std(elec_RH)
elec_Lux = elec_data.Lux.values + SNR[:, 3] # 观测压强x3
elec_Lux1 = (elec_Lux - np.mean(elec_Lux)) / np.std(elec_Lux)

elec_Pca = np.vstack((elec_tem1, elec_hPa1, elec_RH1, elec_Lux1)).T   # 特征数据合并为一个数组
# np.savetxt('new.csv', elec_Pca, delimiter = ',')
# # PCA特征降维，减少相关性
# #白化，使得每个特征具有相同的方差。
pca = PCA(n_components=3)
pca.fit(elec_Pca)
# 将数据X转换成降维后的数据。当模型训练好后，对于新输入的数据，都可以用transform方法来降维。
elec_Pca1 = pca.transform(elec_Pca)
elec_Pca1 = np.array(elec_Pca1)
x, z= pcaa(elec_Pca)
XX = np.array(x)
ZZ = np.array(z)
print(x)
print('CCCCCCCCCCCCCCCCC\n\n')
print(z)

# 计算观测时间，温度，光照等环境条件
elec_year = elec_data.Year.values  # 观测时间值x1
elec_year1 = (elec_year - np.mean(elec_year)) / np.std(elec_year)
# 计算故障率大小：故障数目/总测量数，作为模型Y值，放大1000倍以增加实际效果，结果中要缩小1000倍
# elec_fault = elec_data.Fault / elec_data.Nums
elec_faults = 100 * (elec_data.Fault.values / elec_data.Nums.values)  # 数组形式
elec_faults1 = (elec_faults - np.mean(elec_faults)) / np.std(elec_faults)

# 将故障率以6组一行形式组成数组,变成：21*6
elec_faults2 = np.array([elec_faults[i*6:(i+1)*6] for i in np.arange(21)])
elec_year2 = np.array([elec_year[i*6:(i+1)*6] for i in np.arange(21)])
elec_tem2 = np.array([elec_tem[i*6:(i+1)*6] for i in np.arange(21)])
elec_hPa2 = np.array([elec_hPa[i*6:(i+1)*6] for i in np.arange(21)])
elec_RH2 = np.array([elec_RH[i*6:(i+1)*6] for i in np.arange(21)])
companyABC2 = np.array([companyABC[i*6:(i+1)*6] for i in np.arange(21)])

# 共享变量设置
xs_year = shared(np.asarray(elec_year2))
xs_tem = shared(np.asarray(elec_tem2))
xs_hPa = shared(np.asarray(elec_hPa2))
xs_RH = shared(np.asarray(elec_RH2))
Num_shared = shared(np.asarray(companyABC2))
ys_faults = shared(np.asarray(elec_faults2))

# plt.style.use('default')
# plt.hist(elec_faults, range=[0, 5], bins=130, histtype='stepfilled', color='#6495ED')
# plt.axvline(elec_faults.mean(), color='r', ls='--', label='True mean')
# plt.show()
# 画图
# Plot_XZ(elec_year2, elec_faults2, Savefig)

# 建模
with pm.Model() as unpooled_model:
    # define priors
    alpha = pm.HalfCauchy('alpha', 10, testval=.9)

    # switch = pm.DiscreteUniform('swich', lower=xs_year.min()+3, upper=xs_year.max()-0.5)
    # early_rate = pm.Normal('early_rate', 0, 100)
    # late_rate = pm.Normal('late_rate', 0, 100)
    beta1 = pm.Normal('beta1', 0, 100, shape=companiesABC)
    # beta1 = pm.math.switch(xs_year <= switch, early_rate, late_rate)
    beta = pm.Normal('beta', 0, 100, shape=companiesABC)
    # u = pm.Normal('u', 0, 0.0001)

    # mu = tt.exp(beta[companyABC] + beta1[companyABC]*elec_year + beta2*elec_tem)
    # beta_mu = pm.Deterministic('beta_mu', tt.exp(beta[Num_shared] + beta1 * xs_year + u))
    beta_mu = pm.Deterministic('beta_mu', tt.exp(beta[Num_shared] + beta1[Num_shared] * xs_year))
    # Observed_pred = pm.Weibull("Observed_pred",  alpha=mu, beta=sigma, shape=elec_faults.shape)  # 观测值
    Observed = pm.Weibull("Observed", alpha=alpha, beta=beta_mu, observed=ys_faults)  # 观测值

    # start = pm.find_MAP()
    # step = pm.Slice([beta1, u])
    trace2 = pm.sample(3000, init='advi', tune=1000)
chain2 = trace2[1000:]
# varnames1 = ['alpha', 'beta_mu', 'swich']
# print(pm.df_summary(trace2, varnames1))
varnames2 = ['beta', 'beta1', 'beta_mu']
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
# pm.autocorrplot(chain2, varnames2)
# plt.show()
# print(pm.waic(trace2, unpooled_model))


# new values from x=0 to x=20
X_new = np.linspace(0, 20, 600)[:,None]
with pm.Model() as model:
    ℓ = pm.Gamma("ℓ", alpha=2, beta=1)
    η = pm.HalfCauchy("η", beta=5)

    cov = η**2 * pm.gp.cov.Matern52(1, ℓ)
    gp = pm.gp.Marginal(cov_func=cov)

    σ = pm.HalfCauchy("σ", beta=5)
    y_ = gp.marginal_likelihood("y", X=X, y=y, noise=σ)

    mp = pm.find_MAP()
# add the GP conditional to the model, given the new X values
with model:
    f_pred = gp.conditional("f_pred", X_new)
# To use the MAP values, you can just replace the trace with a length-1 list with `mp`
with model:
    pred_samples = pm.sample_ppc([mp], vars=[f_pred], samples=2000)

