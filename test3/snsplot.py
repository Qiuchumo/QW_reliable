import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)


elec_data = pd.read_csv('XZAB.csv')

sns.jointplot(x="Year", y="Fault", data=elec_data)
plt.show()



# sns.stripplot(x="Year", y="Fault", data=elec_data, jitter=True)
# plt.show()
# 它使用避免重叠点的算法将分类轴上的每个散点图点定位：
# 也可以传入hue参数添加多个嵌套的分类变量
sns.swarmplot(x="Year", y="Fault", hue_order="sex", data=elec_data)
plt.show()

# 这种图形显示了分布的三个四分位值与极值
sns.boxplot(x="Year", y="Fault", hue_order="time", data=elec_data)
plt.show()

# 它结合了箱体图和分布教程中描述的核心密度估计过程
sns.violinplot(x="Year", y="Fault", hue_order="time", data=elec_data, split=True)
plt.show()

# 包括显示每个人的观察结果而不是总结框图值的方法：
sns.violinplot(x="Year", y="Fault", hue_order="time", data=elec_data, split=True, inner="stick", palette="Set3")
plt.show()



