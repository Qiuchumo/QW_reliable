"""
演示二维插值。
"""
import numpy as np
from scipy import interpolate
import pylab as pl
import matplotlib as mpl

def func(x, y):
    return (x + y) * np.exp(-5.0 * (x ** 2 + y ** 2))

# X-Y轴分为15*15的网格
y, x = np.mgrid[-1:1:15j, -1:1:15j]

fvals = func(x, y)  # 计算每个网格点上的函数值  15*15的值
print
len(fvals[0])
# 三次样条二维插值
newfunc = interpolate.interp2d(x, y, fvals, kind='cubic')

# 计算100*100的网格上的插值
xnew = np.linspace(-1, 1, 100)  # x
ynew = np.linspace(-1, 1, 100)  # y
fnew = newfunc(xnew, ynew)  # 仅仅是y值   100*100的值

# 绘图
# 为了更明显地比较插值前后的区别，使用关键字参数interpolation='nearest'
# 关闭imshow()内置的插值运算。
pl.subplot(121)
im1 = pl.imshow(fvals, extent=[-1, 1, -1, 1], cmap=mpl.cm.hot, interpolation='nearest', origin="lower")  # pl.cm.jet
# extent=[-1,1,-1,1]为x,y范围  favals为
pl.colorbar(im1)

pl.subplot(122)
im2 = pl.imshow(fnew, extent=[-1, 1, -1, 1], cmap=mpl.cm.hot, interpolation='nearest', origin="lower")
pl.colorbar(im2)

pl.show()
