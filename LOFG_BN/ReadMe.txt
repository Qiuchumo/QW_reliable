XZnozero_12.：连续数据，不含0
XZnozero_12_sum.：连续数据，不含0,累计数据，即每一年是前几年的和
XZnozero_12_longB：B省数据，因为长一点单独列出来
XZnozero_12_Pred：用于预测用
XZnozero_12_Pred_14：用于预测用12-14的
XZnozero_Source：迁移学习部分数据


DataSplitOne:用于交叉验证，只针对单省数据使用
lof:LOF的源文件

LOF_BN_for_SCI.ipynb：用于 LOF+BN的代码

LofWithBN：LOF处理代码，保存数据用于LOF_BN_for_SCI.ipynb，保存的文件名为Weight_Fault_ABC_All
Weight_Fault_ABC_All：为LofWithBN生成的LOF之后加权的文件，用于LOF_BN_for_SCI处理

LOF_Grubbs：LOF+Grubbs处理代码


Svm_Compare：对比代码



程序先在E:\Code\Bayescode\QW_reliable\LOF文件中的Reliable_LOF生成因子数据，进行计算后，再返回本文件进行计算

LOF_BN_for_SCI.ipynb
LOF_Grubbs