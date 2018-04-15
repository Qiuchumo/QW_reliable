XZnozero_12_stop：为断断续续的数据，不含0,标记为-999
XZnozero_12_stop_stor：为断断续续的数据，不含0,删除了标记数据
XZnozero_12_stop_stor_pred：为断断续续的数据，不含0,删除了标记数据.用于预测13，14个月数据
XZnozero_12：为连续的数据，不含0

XZmulti_12：为连续数据，含0
XZmulti_12sametem：为连续数据，含0，但是每个省的温度相同
Plot_raw：画原始数据图用

outline_fault_A\B\C：用作噪声处理

1_15spline_weibull_ALL：威布尔条件下，所有数据一个似然函数，数据连续

1_17weibull_spline_SEP：威布尔条件下，各省数据分别对应一个似然函数，数据连续

1_17weibull_spline_stop1：威布尔条件下，各省数据分别对应一个似然函数，数据断续

Spline_punish：惩罚B样条模型，用于论文



MAP_tmp0\2\3：用于拟合均值，画图
sig0、1、2：用于画出拟合的置信区间

Pig_EI2：用于论文画图