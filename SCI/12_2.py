import pymc3 as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import theano.tensor as tt
from theano import shared
import pandas as pd
from sklearn.decomposition import PCA, KernelPCA
from Plot_XZ import *
from PCA import *
from matplotlib import gridspec
from scipy.stats.kde import gaussian_kde
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
# elec_Pca2 = np.vstack((elec_tem, elec_hPa, elec_RH, elec_Lux)).T   # 特征数据合并为一个数组
# np.savetxt('XZ_nomean.csv', elec_Pca2, delimiter = ',')
# =============================================================================================
# # PCA特征降维，减少相关性，有两种方法，一种是自带函数，一种是网上程序，下面注释为网上程序
# x, z= pcaa(elec_Pca);  XX = np.array(x); ZZ = np.array(z)
# 将温度等4个特征降维变成2个特征，贡献率为99%以上，满足信息要求; 转换后的特征经过模型后能否还原
# =============================================================================================
# #白化，使得每个特征具有相同的方差，减少数据相关性，n_components：控制特征量个数
pca = PCA(n_components=2, whiten=True)
pca.fit(elec_Pca)
# 将数据X转换成降维后的数据。当模型训练好后，对于新输入的数据，都可以用transform方法来降维。
elec_Pca1 = pca.transform(elec_Pca)
elec_Pca1 = np.array(elec_Pca1)
elec_Pca_char1 = elec_Pca1[:, 0] # 降维特征1
elec_Pca_char2 = elec_Pca1[:, 1] # 降维特征2

# 计算观测时间，温度，光照等环境条件
elec_year = elec_data.Year.values  # 观测时间值x1
elec_year1 = (elec_year - np.mean(elec_year)) / np.std(elec_year)
# 计算故障率大小：故障数目/总测量数，作为模型Y值，放大100倍以增加实际效果，结果中要缩小100倍
elec_faults = 100 * (elec_data.Fault.values / elec_data.Nums.values)  # 数组形式,计算故障率大小
# elec_faults1 = (elec_faults - np.mean(elec_faults)) / np.std(elec_faults)

# 将故障率以6组一行形式组成数组,变成：21*6
elec_faults2 = np.array([elec_faults[i*6:(i+1)*6] for i in np.arange(21)])
elec_year2 = np.array([elec_year[i*6:(i+1)*6] for i in np.arange(21)])
elec_char1 = np.array([elec_Pca_char1[i*6:(i+1)*6] for i in np.arange(21)])
elec_char2 = np.array([elec_Pca_char2[i*6:(i+1)*6] for i in np.arange(21)])
companyABC2 = np.array([companyABC[i*6:(i+1)*6] for i in np.arange(21)])

# 共享变量设置
xs_char1 = shared(np.asarray(elec_char1))
xs_char2 = shared(np.asarray(elec_char2))

ys_faults = shared(np.asarray(elec_faults2))
xs_year = shared(np.asarray(elec_year2))
Num_shared = shared(np.asarray(companyABC2))


shape=companyABC2.shape
# plt.style.use('default')
# plt.hist(elec_faults, range=[0, 5], bins=130, histtype='stepfilled', color='#6495ED')
# plt.axvline(elec_faults.mean(), color='r', ls='--', label='True mean')
# plt.show()
# 画图
# Plot_XZ(elec_year2, elec_faults2, Savefig)


def logit(x):
    return 1/(1+np.exp(-x))
def tlogit(x):
    return 1/(1+tt.exp(-x))
def Phi(x):
    # probit transform
    return 0.5 + 0.5 * pm.math.erf(x/pm.math.sqrt(2))

# 建模，模型1，该模型为unpooling模型，可用于检测因素影响分析
# 建模，模型1

with pm.Model() as model1:
    # define priors
    alpha = pm.HalfCauchy('alpha', 10, testval=.6)

    mu_3 = pm.Normal('mu_3', mu=0, tau=.001)
    sd_3 = pm.HalfCauchy('sd_3', 10)
    mu_2 = pm.Normal('mu_2', mu=0, tau=.001)
    sd_2 = pm.HalfCauchy('sd_2', 10)
    mu_1 = pm.Normal('mu_1', mu=0, tau=.001)
    sd_1 = pm.HalfCauchy('sd_1', 10)
    mu_0 = pm.Normal('mu_0', mu=0, tau=.001)
    sd_0 = pm.HalfCauchy('sd_0', 10)

    beta3 = pm.Normal('beta3', mu_3, sd_3, shape=companiesABC)
    beta2 = pm.Normal('beta2', mu_2, sd_2, shape=companiesABC)
    beta1 = pm.Normal('beta1', mu_1, sd_1, shape=companiesABC)
    beta = pm.Normal('beta', mu_0, sd_0)
    u = pm.Normal('u', 0, 0.01)

    # beta_mu = pm.Deterministic('beta_mu', tt.exp(beta[Num_shared] + beta1[Num_shared] * xs_year + beta2 * xs_char1 + beta3 * xs_char2))
    linerpredi = tt.exp(u + beta + beta1[companyABC] * elec_year + \
                        beta2[companyABC] * elec_Pca_char1 + beta3[companyABC] * elec_Pca_char2)

    # latent model for contamination
    sigma_p = pm.Uniform('sigma_p', lower=0, upper=3)
    mu_p = pm.Normal('mu_p', mu=0, tau=.001)
    probitphi = pm.Normal('probitphi', mu=mu_p, sd=sigma_p, shape=companiesABC, testval=np.ones(companiesABC))
    phii = pm.Deterministic('phii', Phi(probitphi))

    pi_ij = pm.Uniform('pi_ij', lower=0, upper=1, shape=companyABC.shape)

    # Zij:判断条件，theanof.tt_rng()：Get the package-level random number generator or new with specified seed
    zij_ = pm.theanof.tt_rng().uniform(size=companyABC.shape)
    zij = pm.Deterministic('zij', tt.lt(zij_, phii[companyABC]))

    beta_mu = pm.Deterministic('beta_mu', tt.switch(zij, linerpredi, pi_ij))
    # Observed_pred = pm.Weibull("Observed_pred",  alpha=mu, beta=sigma, shape=elec_faults.shape)  # 观测值
    Observed = pm.Weibull("Observed", alpha=alpha, beta=beta_mu, observed=elec_faults)  # 观测值

    # start = pm.find_MAP()
    # step = pm.Slice([beta1, u])
    # step = pm.NUTS(scaling=cov, is_cov=True)
    # trace = pm.sample(3000, init='advi', tune=1000)

with model1:
    s = shared(pm.floatX(1))
    inference = pm.ADVI(cost_part_grad_scale=s)
    # ADVI has nearly converged
    inference.fit(n=20000)
    # It is time to set `s` to zero
    s.set_value(0)
    approx = inference.fit(n=10000)
    trace = approx.sample(3000, include_transformed=True)
    elbos1 = -inference.hist

chain = trace[2000:]
# varnames2 = ['beta', 'beta1', 'beta2', 'beta3', 'u', 'beta4']
pm.traceplot(chain)
plt.show()

# 预测，此时这种格式是每行列数len(elec_faults)个，有很多行数据
ppc = pm.sample_ppc(trace, samples=500, model=model1)
yipred = ppc['Observed']
xipred = {}
yipred_mean = yipred.mean(axis=0)

fig = plt.figure(figsize=(12, 4))
gs = gridspec.GridSpec(1, 3)
for ip in np.arange(companiesABC):
    ax = plt.subplot(gs[ip])
    xp = elec_year2[ip * 7:(ip + 1) * 7, :]  # 原始数据
    yp = elec_faults2[ip * 7:(ip + 1) * 7, :]
    ax.plot(xp, yp, marker='o', alpha=.8)

    yipred_yplot = np.array([yipred_mean[i * 6:(i + 1) * 6] for i in np.arange(7 * ip, (ip + 1) * 7)])
    xipred = np.array([np.arange(6) + 1 for i in np.arange(7)])
    ax.plot(xipred, yipred_yplot[:], 'k+-', color='r')

plt.tight_layout()
plt.show()

varnames2 = ['beta', 'beta1', 'beta2', 'beta3', 'u']
tmp = pm.df_summary(chain, varnames2)
betaMAP = tmp['mean'][0]
beta1MAP = tmp['mean'][np.arange(companiesABC) + 1]
beta2MAP = tmp['mean'][np.arange(companiesABC) + 1*companiesABC+1]
beta3MAP = tmp['mean'][np.arange(companiesABC) + 2*companiesABC+1]
# beta4MAP = tmp['mean'][np.arange(companiesABC) + 4*companiesABC]
uMAP = tmp['mean'][3*companiesABC+1]
# print(uMAP)
# print(beta1MAP)
# print(tmp)
# print(beta2MAP)
# print(beta3MAP)


from matplotlib import gridspec

plt.figure(figsize=(12, 5))
gs = gridspec.GridSpec(1, 3)
for ip in np.arange(companiesABC):
    ax = plt.subplot(gs[ip])
    xp = elec_year2[ip * 7:(ip + 1) * 7, :]  # 原始数据
    yp = elec_faults2[ip * 7:(ip + 1) * 7, :]

    xl = np.linspace(0.5, 6.5, 40)
    yl = np.exp(uMAP + betaMAP + beta1MAP[ip] * xl + beta2MAP[ip] * elec_Pca_char1[ip * 42:(ip * 42 + 40)] + \
                beta3MAP[ip] * elec_Pca_char2[ip * 42:(ip * 42 + 40)])

    ax.plot(xp, yp, marker='o', alpha=.8)
    ax = sns.violinplot(data=elec_faults2[ip * 7:(ip + 1) * 7])
    ax.plot(xl, yl, 'k', linewidth=2)
    plt.axis([0.5, 7, -.1, 4.5])
    plt.title('Subject %s' % (ip + 1))
plt.tight_layout()
plt.show()
# print(yl)



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
    yl = np.exp(uMAP + betaMAP + beta1MAP[ip] * xl + beta2MAP[ip] * elec_Pca_char1[ip * 42:(ip * 42 + 40)] + \
                beta3MAP[ip] * elec_Pca_char2[ip * 42:(ip * 42 + 40)])

    # Posterior sample from the trace
    for ips in np.random.randint(burnin, 3000, ppcsamples):
        param = trace[ips]
        yl2 = np.exp(param['u'] + param['beta'] + param['beta1'][ip] * (xl) + \
                     param['beta2'][ip] * elec_Pca_char1[ip * 42:(ip * 42 + 40)] + \
                     param['beta3'][ip] * elec_Pca_char2[ip * 42:(ip * 42 + 40)]
                     )
        ax.plot(xl, yl2, 'k', linewidth=2, alpha=.05)

    ax = sns.violinplot(data=elec_faults2[ip * 7:(ip + 1) * 7])
    ax.plot(xp, yp, marker='o', alpha=.8)
    plt.plot(xl, yl, 'k', linewidth=2)
    plt.axis([0.5, 7, -.1, 4.5])
    plt.title('Subject %s' % (ip + 1))

plt.tight_layout()
plt.show()


from scipy.stats.kde import gaussian_kde
fig = plt.figure(figsize=(16, 8))
fig.text(0.5, -0.02, 'JND (ms)', ha='center', fontsize=20)
fig.text(-0.02, 0.5, 'Posterior Density', va='center', rotation='vertical', fontsize=20)
gs = gridspec.GridSpec(1, 3)

ppcsamples = 500

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


# 两个特征值的联合后验
trace_beta2 = trace['beta2'][burnin:]
trace_beta3 = trace['beta3'][burnin:]
# datamap = np.vstack((trace_beta2[:, 1], trace_beta3[:,1]))
datamap = np.vstack((trace_beta2[1], trace_beta3[1]))
df = pd.DataFrame(datamap.transpose(), columns=["x", "y"])
g = sns.jointplot(x="x", y="y", data=df, kind="kde", color="g", stat_func=None, xlime=(-2,2), ylime=(-2,2))
g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker=".")
plt.show()


fig = plt.figure(figsize=(4, 2))
a = trace['beta1'][1000:]
plt.boxplot(a)
plt.show()

# 估计密度函数
fig = plt.figure(figsize=(8, 4))


kde_beta2 = trace['beta2'][burnin:]
kde_beta3 = trace['beta3'][burnin:]

gs = gridspec.GridSpec(1, 3)
for ip in np.arange(companiesABC):
    ax = plt.subplot(gs[ip])

    my_pdf = gaussian_kde(kde_beta2[:, ip])
    my_pdf1 = gaussian_kde(kde_beta3[:, ip])
    x = np.linspace(-6, 6, 500)

    plt.plot(x, my_pdf(x))
    plt.plot(x, my_pdf1(x))

plt.tight_layout()
plt.show()

# 跟随特征变化分析
fig = plt.figure(figsize=(15, 12))
fig.text(0.5, -0.02, 'Test Interval (ms)', ha='center', fontsize=20)
fig.text(-0.02, 0.5, 'Proportion of Long Responses', va='center', rotation='vertical', fontsize=20)
gs = gridspec.GridSpec(3, 3)
ppcsamples = 100
burnin = 2000

ylabes = ['XiZang', 'XinJiang', 'HeiLongJiang']

for ip in np.arange(companiesABC):
    ax0 = plt.subplot(gs[ip * 3])
    xp = elec_year2[ip * 7:(ip + 1) * 7, :]
    yp = elec_faults2[ip * 7:(ip + 1) * 7, :]
    ax0.plot(xp, yp, marker='o', alpha=.8)
    plt.ylabel(ylabes[ip], fontsize=12)
    plt.xlabel('Years', fontsize=12)

    x0 = np.linspace(0.5, 6, 40)
    y0 = np.exp(uMAP + betaMAP + beta1MAP[ip] * xl + beta2MAP[ip] * elec_Pca_char1[ip * 42:(ip * 42 + 40)] + beta3MAP[
        ip] * elec_Pca_char2[ip * 42:(ip * 42 + 40)])

    # Posterior sample from the trace
    #     for ips in np.random.randint(burnin, 3000, ppcsamples):
    #         param = trace[ips]
    #         yl2 = np.exp(param['beta'][ip] + param['beta1'][ip] * (xl) + param['beta2'][ip]*elec_Pca_char1[ip*42:(ip*42+40)] + \
    #                      param['beta3'][ip]*elec_Pca_char2[ip*42:(ip*42+40)])
    #         ax0.plot(xl, yl2, 'k', linewidth=2, alpha=.05)

    ax1 = plt.subplot(gs[1 + ip * 3])
    my_pdf1 = gaussian_kde(kde_beta2[:, ip])
    x1 = np.linspace(-2, 3, 300)
    ax1.plot(x1, my_pdf1(x1), 'k', lw=2.5, alpha=0.6)
    plt.xlim((-2, 3))
    plt.xlabel(r'$\beta2$', fontsize=15)
    plt.ylabel('Posterior Density', fontsize=12)

    ax2 = plt.subplot(gs[2 + ip * 3])
    my_pdf2 = gaussian_kde(kde_beta3[:, ip])
    x2 = np.linspace(-2, 3, 300)
    ax2.plot(x2, my_pdf2(x2), 'k', lw=2.5, alpha=0.6)
    plt.xlim((-2, 3))
    plt.xlabel(r'$\beta3$', fontsize=15)
    plt.ylabel('Posterior Density', fontsize=12)

    #     ax0.plot(x0, y0, linewidth=2)
    plt.title('Subject %s' % (ip + 1))

plt.tight_layout()
plt.show()



gs = gridspec.GridSpec(1, 3)
fig = plt.figure(figsize=(12, 4))
for ip in np.arange(companiesABC):
    ax = plt.subplot(gs[ip])
    ax = sns.violinplot(data=elec_faults2[ip*7:(ip+1)*7])
# plt.plot(np.float32(x).T*40, color='gray', alpha=.1)
# plt.plot(np.mean(np.float32(x).T*40, axis=1), color='r', alpha=1)
# plt.xticks(np.arange(8), ''.join(map(str, np.arange(1, 9))))
# plt.yticks([0, t], ('B', 'A'))
plt.xlabel('Stimulus')
plt.ylabel('Category Decision')
plt.show()



# 后验分析
ppc = pm.sample_ppc(trace, samples=1000, model=model1)
yipred = ppc['Observed']
# plt.hist(yipred, normed=1, bins=80, alpha=.8, label='Posterior')
# plt.show()
plt.figure()
ax = sns.distplot(ppc['Observed'].mean(axis=1), label='Posterior predictive means') # axis=1以行方式计算
ax.axvline(elec_faults.mean(), color='r', ls='--', label='True mean')
ax.legend()
plt.show()

# accept = trace.get_sampler_stats('mean_tree_accept', burn=burnin)
# print('The accept rate is: %.5f' % (accept.mean()))
