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
elec_data.Fault.values[48] =2000
# 计算故障率大小：故障数目/总测量数，作为模型Y值，放大100倍以增加实际效果，结果中要缩小100倍
elec_faults = 100 * (elec_data.Fault.values / elec_data.Nums.values)  # 数组形式,计算故障率大小
# elec_faults1 = (elec_faults - np.mean(elec_faults)) / np.std(elec_faults)
# elec_faults[25] = 3
# elec_faults[39] = 5
# elec_faults[53] = 3.8
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
def invlogit(x):
    return tt.exp(x)/(1+tt.exp(x))
def Phi(x):
    # probit transform
    return 0.5 + 0.5 * pm.math.erf(x/pm.math.sqrt(2))




import scipy.linalg
def alltrue_elemwise(vals):
    ret = 1
    for c in vals:
        ret = ret * (1 * c)
    return ret

def alltrue_scalar(vals):
    return tt.all([tt.all(1 * val) for val in vals])
def bound(logp, *conditions, **kwargs):
    broadcast_conditions = kwargs.get('broadcast_conditions', True)

    if broadcast_conditions:
        alltrue = alltrue_elemwise
    else:
        alltrue = alltrue_scalar

    return tt.switch(alltrue(conditions), logp, -np.inf)


def draw_values(params, point=None):
    """
    Draw (fix) parameter values. Handles a number of cases:

        1) The parameter is a scalar
        2) The parameter is an *RV

            a) parameter can be fixed to the value in the point
            b) parameter can be fixed by sampling from the *RV
            c) parameter can be fixed using tag.test_value (last resort)

        3) The parameter is a tensor variable/constant. Can be evaluated using
        theano.function, but a variable may contain nodes which

            a) are named parameters in the point
            b) are *RVs with a random method

    """
    # Distribution parameters may be nodes which have named node-inputs
    # specified in the point. Need to find the node-inputs to replace them.
    givens = {}
    for param in params:
        if hasattr(param, 'name'):
            named_nodes = get_named_nodes(param)
            if param.name in named_nodes:
                named_nodes.pop(param.name)
            for name, node in named_nodes.items():
                if not isinstance(node, (tt.sharedvar.SharedVariable,
                                         tt.TensorConstant)):
                    givens[name] = (node, pm.distribution._draw_value(node, point=point))
    values = []
    for param in params:
        values.append(_draw_value(param, point=point, givens=givens.values()))
    return values

class ZeroInflatedWeibull(pm.Continuous):
    '''
    Zero-inflated Poisson log-likelihood.
    Parameters
    ----------
    psi : float
        Expected proportion of Poisson variates (0 < psi < 1)
    theta : float
        Expected number of occurrences during the given interval
        (theta >= 0).
    '''

    def __init__(self, pi, alpha, beta, *args, **kwargs):
        super(ZeroInflatedWeibull, self).__init__(*args, **kwargs)
        self.alpha = alpha = tt.as_tensor_variable(alpha)
        self.beta = beta = tt.as_tensor_variable(beta)
        self.pi = pi = tt.as_tensor_variable(pi)
        self.weib =  pm.Weibull.dist(alpha, beta)
#         self.mode = self.weib.mode

    def random(self, point=None, size=None, repeat=None):
        alpha, beta, pi = draw_values([self.alpha, self.beta, self.pi],
                                  point=point)
        def _random(a, b, size=None):
            return b * (-np.log(np.random.uniform(size=size)))**(1 / a)
        g = generate_samples(_random, alpha, beta,
                                dist_shape=self.shape,
                                size=size)
        sampled = g * (np.random.random(np.squeeze(g.shape)) < pi)
        return reshape_sampled(sampled, size, self.shape)

    def logp(self, value):
        pi = self.pi
        alpha = self.alpha
        beta = self.beta

        logp_val = tt.switch(
            tt.gt(value, 0),
            tt.log(-pi) + self.weib.logp(value),
            tt.log(pi))

        return bound(
            logp_val,
            0 <= value,
            0 <= pi, pi <= 1,
            0 < alpha,
            0 < beta)

    # def _repr_latex_(self, name=None, dist=None):
    #     if dist is None:
    #         dist = self
    #     alpha = dist.alpha
    #     beta = dist.beta
    #     pi = dist.pi
    #     return r'${} \sim \text{{ZeroInflatedWeibull}}(\mathit{{alpha}}={}, \mathit{{beta}}={}, \mathit{{pi}}={})$'.format(name,
    #                                             get_variable_name(alpha),
    #                                             get_variable_name(beta),
    #                                             get_variable_name(pi))


# 建模，模型
with pm.Model() as model_1:
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
    #     mu_0 = pm.Normal('mu_0', mu=0, tau=.001)
    #     sd_0 = pm.HalfCauchy('sd_0', 20)
    beta4 = pm.Normal('beta4', mu_4, sd_4, shape=companiesABC)
    beta3 = pm.Normal('beta3', mu_3, sd_3, shape=companiesABC)
    beta2 = pm.Normal('beta2', mu_2, sd_2, shape=companiesABC)
    beta1 = pm.Normal('beta1', mu_1, sd_1, shape=companiesABC)
    beta = pm.Normal('beta', 0, 100)
    u = pm.Normal('u', 0, 0.01)

    pi_beta = pm.Normal('pi_beta', 0, 100, shape=companiesABC)
    pi = pm.Deterministic('pi', invlogit(pi_beta[Num_shared]))

    theta = pm.Deterministic('theta', tt.exp(u + beta + \
                                             (beta1[Num_shared] * xs_year + beta2[Num_shared] * xs_char1 + \
                                              beta3[Num_shared] * xs_char2 + beta4[Num_shared] * xs_year * xs_year)))

    Observed = ZeroInflatedWeibull("Observed", pi=pi, alpha=alpha, beta=theta, observed=ys_faults)  # 观测值
    trace_1 = pm.sample(3000)

pm.traceplot(trace_1)
plt.show()
