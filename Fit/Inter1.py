# -*- coding:utf-8 -*-
from scipy.interpolate import lagrange
import numpy as np
import matplotlib.pyplot as plt


def interp_lagrange(x, y, xx):
    # 调用拉格朗日插值，得到插值函数p
    p = lagrange(x, y)
    yy = p(xx)
    plt.plot(x, y, "b*")
    plt.plot(xx, yy, "ro")
    plt.show()


if __name__ == '__main__':
    NUMBER = 20
    eps = np.random.rand(NUMBER) * 2

    # 构造样本数据
    x = np.linspace(0, 20, NUMBER)
    y = np.linspace(2, 14, NUMBER) + eps

    # 兴趣点数据
    xx = np.linspace(12, 15, 10)
    interp_lagrange(x, y, xx)

