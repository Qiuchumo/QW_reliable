XZmulti_12.csv：原始数据，每组12个点，不含0，连续
XZmulti_6.csv：原始数据，每组6个点，不含0，连续
XZ_CS_5：用于预测（每省取5组）
XZ_CS：用于预测（每省取7组），对比验证
XZ_CS_8：用于预测（每省取7组），预测到第8年
XZmulti_6_7Test：原始数据，用于最后的对比验证测试，增加第7年数据
XZmulti_6_Only7Test：只含有第7年的数据，用于验证与测试

XZmulti_6_A/B/C：为每省单数数据，用于SVM算法对比

BN_Boost.ipynb：为集成贝叶斯代码
BN_7Year.ipynb：为加噪声识别与不加噪声识别的代码，内含有可靠度计算代码，可靠度部分用于论文
BN_Outliers_3_30.ipynb：为尝试12个点的代码
BN_Boost_Weight：将每个模型分别放在一个For循环里面，用于增加运行次数


PLSR_SVM_Compare.ipynb：为SVM与PLSR对比代码



DataSplit.py:为将输入原始数据随机抽样放回的代码（分为训练与验证集）
DataSplitOne.py:为将每个省的输入原始数据随机抽样放回的代码（分为训练与验证集）
RMSE.py：计算RMSE代码


以下为保存的数据：
elec_Pca_char1：模型PCA之后的 属性值1
elec_Pca_char2：模型PCA之后的 属性值2


Y_PLSpred_MEAN:   PLSR 对比模型的 拟合 均值输出
Pred_Y_PLSRpred： PLSR 对比模型的 预测 均值输出
Y_SVMpred:	  SVM  对比模型的 拟合 均值输出
Pred_Y_SVMpred:	  SVM  对比模型的 预测 均值输出


Mean_output：	      三个子模型的 拟合 均值输出
M0_Mean_output：      BN对比模型的  拟合 均值输出
M2_Pred_yplot_Mean_C：子模型M2的  预测 均值输出
M3_Pred_yplot_Mean_C：子模型M3的  预测 均值输出
M4_Pred_yplot_Mean_C：子模型M4的  预测 均值输出
Mean_Cumul_M2：	      子模型M2的  预测 3省分别运行多次输出，还未累计，用于计算累计均方误差

Mean_Train_M2_A：     子模型M2的  拟合 A省分别运行多次累计输出，还未累计，用于计算累计后均方误差
Mean_Train_M2_B
Mean_Train_M2_C

三省预测值的95%区间,其中,分别给出了三个子模型的置信区间值，需要搜索以确定最小交集作为实际区间值。
前三行为三个子模型的5%，后三行为三个子模型的95%
A_PhdMaxMin0595：A省 预测 值的95%置信区间
B_PhdMaxMin0595：B省 预测 值的95%置信区间
C_PhdMaxMin0595：C省 预测 值的95%置信区间

R112233：为可靠度的图
Reliabily_AAA、BBB/CCC为可靠度的图


