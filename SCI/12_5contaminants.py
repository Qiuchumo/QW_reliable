import pymc3 as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import theano.tensor as tt
from theano import shared
import pandas as pd
from matplotlib import gridspec
from sklearn.decomposition import PCA, KernelPCA
from Plot_XZ import *
from PCA import *

# 2017.12.28编辑  可靠性分析项目，两省数据分析，用于SCI论文
# 撰写人：邱楚陌
np.set_printoptions(precision=0, suppress=True)
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


# 计算观测时间，温度，光照等环境条件
elec_year = elec_data.Year.values  # 观测时间值x1
elec_year1 = (elec_year - np.mean(elec_year)) / np.std(elec_year)
data_cs_year = elec_year
# data_cs_year[42:45] = 12
# print(data_cs_year)

elec_Pca = np.vstack((elec_tem1, elec_hPa1, elec_RH1, elec_Lux1)).T   # 特征数据合并为一个数组
# elec_Pca2 = np.vstack((elec_tem, elec_hPa, elec_RH, elec_Lux)).T   # 特征数据合并为一个数组
# np.savetxt('XZ_nomean.csv', elec_Pca2, delimiter = ',')
# =============================================================================================
# # PCA特征降维，减少相关性，有两种方法，一种是自带函数，一种是网上程序，下面注释为网上程序
# x, z= pcaa(elec_Pca);  XX = np.array(x); ZZ = np.array(z)
# 将温度等4个特征降维变成2个特征，贡献率为99%以上，满足信息要求; 转换后的特征经过模型后能否还原
# =============================================================================================
# #白化，使得每个特征具有相同的方差，减少数据相关性，n_components：控制特征量个数
pca = PCA(n_components=2)
pca.fit(elec_Pca)
# 将数据X转换成降维后的数据。当模型训练好后，对于新输入的数据，都可以用transform方法来降维。
elec_Pca1 = pca.transform(elec_Pca)
elec_Pca1 = np.array(elec_Pca1)

elec_Pca_char1 = elec_Pca1[:, 0] # 降维特征1
elec_Pca_char2 = elec_Pca1[:, 1] # 降维特征2
# elec_Pca_char3 = elec_Pca1[:, 2] # 降维特征2
# print(elec_Pca_char1)

# 计算故障率大小：故障数目/总测量数，作为模型Y值，放大100倍以增加实际效果，结果中要缩小100倍
elec_faults = 100 * (elec_data.Fault.values / elec_data.Nums.values)  # 数组形式,计算故障率大小
# elec_faults1 = (elec_faults - np.mean(elec_faults)) / np.std(elec_faults)
elec_faults[42] = 2

# print(elec_faults)
# 将故障率以6组一行形式组成数组,变成：21*6
elec_faults2 = np.array([elec_faults[i*6:(i+1)*6] for i in np.arange(21)])
elec_year2 = np.array([elec_year[i*6:(i+1)*6] for i in np.arange(21)])
elec_char1 = np.array([elec_Pca_char1[i*6:(i+1)*6] for i in np.arange(21)])
elec_char2 = np.array([elec_Pca_char2[i*6:(i+1)*6] for i in np.arange(21)])
companyABC2 = np.array([companyABC[i*6:(i+1)*6] for i in np.arange(21)])

# 共享变量设置
xs_char1 = shared(np.asarray(elec_Pca_char1))
xs_char2 = shared(np.asarray(elec_Pca_char2))

ys_faults = shared(np.asarray(elec_faults))
xs_year = shared(np.asarray(data_cs_year))
Num_shared = shared(np.asarray(companyABC))
# 画图
# Plot_XZ(elec_year2, elec_faults2, Savefig)

def logit(x):
    return 1/(1+np.exp(-x))
def tlogit(x):
    return 1/(1+tt.exp(-x))
def Phi(x):
    # probit transform
    return 0.5 + 0.5 * pm.math.erf(x/pm.math.sqrt(2))




# 建模，模型,原始模型，不加污染数据时候的模型
with pm.Model() as model_1:
    # define priors
    alpha = pm.HalfCauchy('alpha', 10, testval=.6)

    #     sd_5 = pm.HalfNormal('sd_5', 0.5)

    mu_4 = pm.Normal('mu_4', mu=0, tau=.001)
    sd_4 = pm.HalfCauchy('sd_4', 10)
    mu_3 = pm.Normal('mu_3', mu=0, tau=.001)
    sd_3 = pm.HalfCauchy('sd_3', 10)
    mu_2 = pm.Normal('mu_2', mu=0, tau=.001)
    sd_2 = pm.HalfCauchy('sd_2', 10)
    mu_1 = pm.Normal('mu_1', mu=0, tau=.001)
    sd_1 = pm.HalfCauchy('sd_1', 10)
    # mu_0 = pm.Normal('mu_0', mu=0, tau=.001)
    # sd_0 = pm.HalfCauchy('sd_0', 20)

    beta4 = pm.Normal('beta4', mu_4, sd_4, shape=companiesABC)
    beta3 = pm.Normal('beta3', mu_3, sd_3, shape=companiesABC)
    beta2 = pm.Normal('beta2', mu_2, sd_2, shape=companiesABC)
    beta1 = pm.Normal('beta1', mu_1, sd_1, shape=companiesABC)
    beta = pm.Normal('beta', 0, 100)
    u = pm.Normal('u', 0, 0.01)

    beta_mu = pm.Deterministic('beta_mu', tt.exp(u + beta + \
                                             (beta1[Num_shared] * xs_year + beta2[Num_shared] * xs_char1 +\
                                              beta3[Num_shared] * xs_char2 + beta4[Num_shared] * xs_year * xs_year)))

    Observed = pm.Weibull("Observed", alpha=alpha, beta=beta_mu, observed=ys_faults)  # 观测值
    trace_1 = pm.sample(10000,  init='advi+adapt_diag')


pm.traceplot(trace_1, varnames=['beta', 'beta1', 'beta2', 'beta3', 'beta4', 'u'])
plt.show()




burnin = 9000
chain = trace_1[burnin:]
# get MAP estimate
varnames2 = ['beta', 'beta1', 'beta2', 'beta3','beta4', 'u']
tmp = pm.df_summary(chain, varnames2)
betaMAP = tmp['mean'][0]
beta1MAP = tmp['mean'][np.arange(companiesABC) + 1]
beta2MAP = tmp['mean'][np.arange(companiesABC) + 1*companiesABC+1]
beta3MAP = tmp['mean'][np.arange(companiesABC) + 2*companiesABC+1]
beta4MAP = tmp['mean'][np.arange(companiesABC) + 3*companiesABC+1]
uMAP = tmp['mean'][4*companiesABC+1]
# am0MAP = tmp['mean'][4*companiesABC+2]
# am1MAP = tmp['mean'][4*companiesABC+3]
# print(am0MAP)
# print(beta1MAP)
# print(tmp)
# print(beta2MAP)
# print(beta3MAP)
# 模型拟合效果图
ppcsamples = 500
ppcsize = 100
# ppc = defaultdict(list)
burnin = 2000
fig = plt.figure(figsize=(16, 8))
fig.text(0.5, -0.02, 'Test Interval (ms)', ha='center', fontsize=20)
fig.text(-0.02, 0.5, 'Proportion of Long Responses', va='center', rotation='vertical', fontsize=20)
gs = gridspec.GridSpec(1, 3)
ppcsamples = 100

for ip in np.arange(companiesABC):
    ax = plt.subplot(gs[ip])
    xp = elec_year2[ip * 7:(ip + 1) * 7, :]
    yp = elec_faults2[ip * 7:(ip + 1) * 7, :]

    xl = np.linspace(0.5, 6.5, 40)
    yl = np.exp(uMAP + betaMAP + (beta1MAP[ip] * xl + beta2MAP[ip] * elec_Pca_char1[ip * 42:(ip * 42 + 40)] + \
                                  beta3MAP[ip] * elec_Pca_char2[ip * 42:(ip * 42 + 40)] + beta4MAP[ip] * xl * xl))

    # Posterior sample from the trace
    for ips in np.random.randint(burnin, 3000, ppcsamples):
        param = trace_1[ips]
        yl2 = np.exp(param['u'] + param['beta'] + (param['beta1'][ip] * (xl) + \
                                                   param['beta2'][ip] * elec_Pca_char1[ip * 42:(ip * 42 + 40)] + \
                                                   param['beta3'][ip] * elec_Pca_char2[ip * 42:(ip * 42 + 40)] + \
                                                   + param['beta4'][ip] * xl * xl)
                     )
        ax.plot(xl, yl2, 'k', linewidth=2, alpha=.05)

    ax = sns.violinplot(data=elec_faults2[ip * 7:(ip + 1) * 7])
    ax.plot(xp, yp, marker='o', alpha=.8)
    plt.plot(xl, yl, 'k', linewidth=2)
    plt.axis([0.5, 7, -.1, 4.5])
    plt.title('Subject %s' % (ip + 1))

plt.tight_layout()
plt.show()

# 建模，加上含污染模型对比，第一个模型model_2用来求解model_2b
with pm.Model() as model_2:
    # define priors
    alpha = pm.HalfCauchy('alpha', 10, testval=.6)

    mu_4 = pm.Normal('mu_4', mu=0, tau=.001)
    sd_4 = pm.HalfCauchy('sd_4', 10)
    mu_3 = pm.Normal('mu_3', mu=0, tau=.001)
    sd_3 = pm.HalfCauchy('sd_3', 10)
    mu_2 = pm.Normal('mu_2', mu=0, tau=.001)
    sd_2 = pm.HalfCauchy('sd_2', 10)
    mu_1 = pm.Normal('mu_1', mu=0, tau=.001)
    sd_1 = pm.HalfCauchy('sd_1', 10)
    mu_0 = pm.Normal('mu_0', mu=0, tau=.001)
    sd_0 = pm.HalfCauchy('sd_0', 20)

    beta4 = pm.Normal('beta4', mu_4, sd_4, shape=companiesABC)
    beta3 = pm.Normal('beta3', mu_3, sd_3, shape=companiesABC)
    beta2 = pm.Normal('beta2', mu_2, sd_2, shape=companiesABC)
    beta1 = pm.Normal('beta1', mu_1, sd_1, shape=companiesABC)
    beta = pm.Normal('beta', mu_0, sd_0)
    u = pm.Normal('u', 0, 0.01)

    liner = pm.Deterministic('liner', tt.exp(u + beta + \
                                             (beta1[Num_shared] * xs_year + beta2[Num_shared] * xs_char1 + \
                                              beta3[Num_shared] * xs_char2 + beta4[Num_shared] * xs_year * xs_year)))

    # latent model for contamination
    sigma_p = pm.Uniform('sigma_p', lower=0, upper=3)
    mu_p = pm.Normal('mu_p', mu=0, tau=.001)

    probitphi = pm.Normal('probitphi', mu=mu_p, sd=sigma_p, shape=companiesABC, testval=np.ones(companiesABC))
    phii = pm.Deterministic('phii', Phi(probitphi))

    pi_ij = pm.Uniform('pi_ij', lower=0, upper=1, shape=len(Num_shared.get_value()))

    zij_ = pm.theanof.tt_rng().uniform(size=companyABC.shape)
    zij = pm.Deterministic('zij', tt.lt(zij_, phii[Num_shared]))
    #     phic = pm.Uniform('phic', lower=0, upper=1, testval=.3)
    #     zij = pm.Bernoulli('zij', p=phic, shape=len(Num_shared.get_value()))
    #     line = tt.constant(np.ones(len(Num_shared.get_value())) * .5)
    #     beta_mu = pm.Deterministic('beta_mu', tt.squeeze(tt.switch(tt.eq(zij, 0), liner, line)))
    beta_mu = pm.Deterministic('beta_mu', tt.switch(zij, liner, pi_ij))

    Observed = pm.Weibull("Observed", alpha=alpha, beta=beta_mu, observed=ys_faults)  # 观测值
    #     start = pm.find_MAP()
#     step = pm.Metropolis([zij])
# #     step1 = pm.NUTS(scaling=cov, is_cov=True)
#     #     step1 = pm.Slice([am0, am1])
#     trace = pm.sample(4000, step=[step], init='advi+adapt_diag', tune=1000)
import theano

with model_2:
    s = theano.shared(pm.floatX(1))
    inference = pm.ADVI(cost_part_grad_scale=s)
    # ADVI has nearly converged
    inference.fit(n=20000)
    # It is time to set `s` to zero
    s.set_value(0)
    approx = inference.fit(n=10000)
    trace_2 = approx.sample(3000, include_transformed=True)
    elbos1 = -inference.hist



chain_2 = trace_2[2000:]
# varnames2 = ['beta', 'beta1', 'beta2', 'beta3', 'u', 'beta4']
pm.traceplot(chain_2)
plt.show()

njob = 1
burn = 5000
start = trace_2[0]
start['zij'] = start['zij'].astype(int)
stds = approx.bij.rmap(approx.std.eval())
cov = model_2.dict_to_array(stds) ** 2
# 建模，加上含污染模型对比,这是真正的模型
with pm.Model() as model_2b:
    # define priors
    alpha = pm.HalfCauchy('alpha', 10, testval=.6)

    mu_4 = pm.Normal('mu_4', mu=0, tau=.001)
    sd_4 = pm.HalfCauchy('sd_4', 10)
    mu_3 = pm.Normal('mu_3', mu=0, tau=.001)
    sd_3 = pm.HalfCauchy('sd_3', 10)
    mu_2 = pm.Normal('mu_2', mu=0, tau=.001)
    sd_2 = pm.HalfCauchy('sd_2', 10)
    mu_1 = pm.Normal('mu_1', mu=0, tau=.001)
    sd_1 = pm.HalfCauchy('sd_1', 10)
    mu_0 = pm.Normal('mu_0', mu=0, tau=.001)
    sd_0 = pm.HalfCauchy('sd_0', 20)

    beta4 = pm.Normal('beta4', mu_4, sd_4, shape=companiesABC)
    beta3 = pm.Normal('beta3', mu_3, sd_3, shape=companiesABC)
    beta2 = pm.Normal('beta2', mu_2, sd_2, shape=companiesABC)
    beta1 = pm.Normal('beta1', mu_1, sd_1, shape=companiesABC)
    beta = pm.Normal('beta', mu_0, sd_0)
    u = pm.Normal('u', 0, 0.01)

    liner = pm.Deterministic('liner', tt.exp(u + beta + \
                                             (beta1[Num_shared] * xs_year + beta2[Num_shared] * xs_char1 + \
                                              beta3[Num_shared] * xs_char2 + beta4[Num_shared] * xs_year * xs_year)))

    # latent model for contamination
    sigma_p = pm.Uniform('sigma_p', lower=0, upper=3)
    mu_p = pm.Normal('mu_p', mu=0, tau=.001)

    probitphi = pm.Normal('probitphi', mu=mu_p, sd=sigma_p, shape=companiesABC, testval=np.ones(companiesABC))
    phii = pm.Deterministic('phii', Phi(probitphi))

    pi_ij = pm.Uniform('pi_ij', lower=0, upper=1, shape=len(Num_shared.get_value()))

    zij = pm.Bernoulli('zij', p=phii[Num_shared], shape=len(Num_shared.get_value()))
    #     phic = pm.Uniform('phic', lower=0, upper=1, testval=.3)
    #     zij = pm.Bernoulli('zij', p=phic, shape=len(Num_shared.get_value()))
    #     line = tt.constant(np.ones(len(Num_shared.get_value())) * .5)
    #     beta_mu = pm.Deterministic('beta_mu', tt.squeeze(tt.switch(tt.eq(zij, 0), liner, line)))
    beta_mu = pm.Deterministic('beta_mu', tt.switch(tt.eq(zij, 0), liner, pi_ij))

    Observed = pm.Weibull("Observed", alpha=alpha, beta=beta_mu, observed=ys_faults)  # 观测值
    #     start = pm.find_MAP()
    step = pm.NUTS(scaling=cov, is_cov=True)
    trace_2b = pm.sample(burn, step=[step], start=start, njobs=njob)

burnin = burn - 1000
chain_2b = trace_2b[burnin:]
# varnames2 = ['beta', 'beta1', 'beta2', 'beta3', 'u', 'beta4']
pm.traceplot(chain_2b)
plt.show()


varnames2b = ['beta', 'beta1', 'beta2', 'beta3','beta4', 'u']
tmp2 = pm.df_summary(chain_2b, varnames2b)
betaMAP2 = tmp2['mean'][0]
beta1MAP2 = tmp2['mean'][np.arange(companiesABC) + 1]
beta2MAP2 = tmp2['mean'][np.arange(companiesABC) + 1*companiesABC+1]
beta3MAP2 = tmp2['mean'][np.arange(companiesABC) + 2*companiesABC+1]
beta4MAP2 = tmp2['mean'][np.arange(companiesABC) + 3*companiesABC+1]
uMAP2 = tmp2['mean'][4*companiesABC+1]
# am0MAP = tmp['mean'][4*companiesABC+2]
# am1MAP = tmp['mean'][4*companiesABC+3]
# print(am0MAP)
# print(beta1MAP)
# print(tmp)
# print(beta2MAP)
# print(beta3MAP)

# 模型拟合效果图
ppcsamples = 500
ppcsize = 100
# ppc = defaultdict(list)
# burnin = 2000
fig = plt.figure(figsize=(16, 8))
fig.text(0.5, -0.02, 'Test Interval (ms)', ha='center', fontsize=20)
fig.text(-0.02, 0.5, 'Proportion of Long Responses', va='center', rotation='vertical', fontsize=20)
gs = gridspec.GridSpec(1, 3)
ppcsamples = 100

for ip in np.arange(companiesABC):
    ax = plt.subplot(gs[ip])
    xp = elec_year2[ip * 7:(ip + 1) * 7, :]
    yp = elec_faults2[ip * 7:(ip + 1) * 7, :]

    xl = np.linspace(0.5, 6.5, 40)
    yl = np.exp(uMAP + betaMAP + (beta1MAP[ip] * xl + beta2MAP[ip] * elec_Pca_char1[ip * 42:(ip * 42 + 40)] + \
                                  beta3MAP[ip] * elec_Pca_char2[ip * 42:(ip * 42 + 40)] + beta4MAP[ip] * xl * xl))

    y2 = np.exp(uMAP2 + betaMAP2 + (beta1MAP2[ip] * xl + beta2MAP2[ip] * elec_Pca_char1[ip * 42:(ip * 42 + 40)] + \
                                    beta3MAP2[ip] * elec_Pca_char2[ip * 42:(ip * 42 + 40)] + beta4MAP2[ip] * xl * xl))
    # Posterior sample from the trace
    #     for ips in np.random.randint(burnin, 3000, ppcsamples):
    #         param = trace[ips]
    #         yl2 = np.exp(param['u'] + param['beta'] + (param['beta1'][ip] * (xl) + \
    #                      param['beta2'][ip]*elec_Pca_char1[ip*42:(ip*42+40)] + \
    #                      param['beta3'][ip]*elec_Pca_char2[ip*42:(ip*42+40)] + \
    #                       + param['beta4'][ip] *xl*xl)
    #                     )
    #         ax.plot(xl, yl2, 'k', linewidth=2, alpha=.05)

    #     ax = sns.violinplot(data=elec_faults2[ip*7:(ip+1)*7])
    ax.plot(xp, yp, marker='o', alpha=.8)
    plt.plot(xl, yl, 'k', linewidth=2)
    plt.plot(xl, y2, 'r--', linewidth=2)
    plt.axis([0.5, 7, -.1, 4.5])
    plt.title('Subject %s' % (ip + 1))

plt.tight_layout()
plt.show()


