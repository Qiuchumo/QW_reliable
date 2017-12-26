
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












# #=============== 建模，模型2 ===========================================
start = trace[0]
start['zij'] = start['zij'].astype(int)
stds = approx.bij.rmap(approx.std.eval())
cov = model1.dict_to_array(stds) ** 2
with pm.Model() as model2:
    # define priors
    alpha = pm.HalfCauchy('alpha', 10, testval=.6)

    beta3 = pm.Normal('beta3', 0, 100)
    beta2 = pm.Normal('beta2', 0, 100)
    beta1 = pm.Normal('beta1', 0, 100, shape=companiesABC)
    beta = pm.Normal('beta', 0, 100, shape=companiesABC)
    # u = pm.Normal('u', 0, 0.0001)

    # beta_mu = pm.Deterministic('beta_mu', tt.exp(beta[Num_shared] + beta1[Num_shared] * xs_year + beta2 * xs_char1 + beta3 * xs_char2))
    linerpredi = tt.exp(beta[companyABC] + beta1[companyABC] * elec_year + beta2 * elec_Pca_char1 + beta3 * elec_Pca_char2)

    # latent model for contamination
    sigma_p = pm.Uniform('sigma_p', lower=0, upper=3)
    mu_p = pm.Normal('mu_p', mu=0, tau=.001)
    probitphi = pm.Normal('probitphi', mu=mu_p, sd=sigma_p, shape=companiesABC, testval=np.ones(companiesABC))
    phii = pm.Deterministic('phii', Phi(probitphi))

    pi_ij = pm.Uniform('pi_ij', lower=0, upper=1, shape=companyABC.shape)

    # Zij:判断条件，theanof.tt_rng()：Get the package-level random number generator or new with specified seed
    zij = pm.Bernoulli('zij', p=phii[companyABC], shape=companyABC.shape)
    beta_mu = pm.Deterministic('beta_mu', tt.switch(tt.eq(zij, 0), linerpredi, pi_ij))

    # Observed_pred = pm.Weibull("Observed_pred",  alpha=mu, beta=sigma, shape=elec_faults.shape)  # 观测值
    Observed = pm.Weibull("Observed", alpha=alpha, beta=beta_mu, observed=elec_faults)  # 观测值

    # start = pm.find_MAP()
    # step = pm.Slice([beta1, u])
    step = pm.NUTS(scaling=cov, is_cov=True)
    trace2 = pm.sample(3e3, step=step, start=start)

chain2 = trace2[1000:]
varnames2 = ['beta', 'beta1', 'beta2', 'beta3', 'beta_mu']
# # pm.plot_posterior(chain2, varnames2, ref_val=0)
pm.traceplot(chain2)
plt.show()
pm.traceplot(chain2, varnames2)
plt.show()


# 两种能量图
# energy = trace['energy']
# energy_diff = np.diff(energy)
# sns.distplot(energy - energy.mean(), label='energy')
# sns.distplot(energy_diff, label='energy diff')
# plt.legend()
# plt.show()
pm.energyplot(trace)
plt.show()
# map_estimate = pm.find_MAP(model=model1)
# print(map_estimate)
# # 画出自相关曲线
# pm.autocorrplot(chain, varnames1)
# plt.show()
# print(pm.waic(trace2, model1))


