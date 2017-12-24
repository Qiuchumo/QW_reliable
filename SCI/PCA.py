import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as ax3
from sklearn.decomposition import PCA, KernelPCA


# 零均值化
def zeroMean(dataMat):
  meanVal = np.mean(dataMat, axis=0)  # 按列求均值，即求各个特征的均值
  newData = dataMat - meanVal
  return newData, meanVal

def percentage2n(eigVals,percentage):
    sortArray = np.sort(eigVals)   #升序
    sortArray = sortArray[-1::-1]  #逆转，即降序
    arraySum = sum(sortArray)
    tmpSum = 0
    num = 0
    for i in sortArray:
        tmpSum += i
        num += 1
        if tmpSum >= arraySum*percentage:
            return num

def pcaa(dataMat,percentage=0.99):
    newData, meanVal = zeroMean(dataMat)

    covMat = np.cov(newData, rowvar=0)    #求协方差矩阵,return ndarray；若rowvar非0，一列代表一个样本，为0，一行代表一个样本

    eigVals, eigVects = np.linalg.eig(np.mat(covMat))#求特征值和特征向量,特征向量是按列放的，即一列代表一个特征向量
    n = percentage2n(eigVals, percentage)                 #要达到percent的方差百分比，需要前n个特征向量

    eigValIndice = np.argsort(eigVals)            #对特征值从小到大排序
    n_eigValIndice = eigValIndice[-1:-(n+1):-1]   #最大的n个特征值的下标

    n_eigVect = eigVects[:, n_eigValIndice]        #最大的n个特征值对应的特征向量
    # lowDDataMat = newData * n_eigVect               #低维特征空间的数据
    lowDDataMat = np.dot(newData, n_eigVect)
    reconMat = (np.dot(lowDDataMat, n_eigVect.T)) + np.array(meanVal)  #重构数据
    return lowDDataMat, reconMat



if __name__ == '__main__':
  elec_data = pd.read_csv('XZZZ.csv')

  # X = np.array([[-1, 1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
  X = np.array(elec_data)
  # figure1 = plt.figure(1)
  # ax = ax3(figure1)
  # ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], alpha=0.7)
  # plt.show()
  # #白化，使得每个特征具有相同的方差。
  pca = PCA(n_components=3, whiten=True)
  pca.fit(X)
  # 将数据X转换成降维后的数据。当模型训练好后，对于新输入的数据，都可以用transform方法来降维。
  x1 = pca.transform(X)
  XX11 = np.array(x1)
  # 用X来训练PCA模型，同时返回降维后的数据。这两句pca.transform、fit_transform都可以
  # x2 = pca.fit_transform(X)
  print(x1)
  print('0000000000000000000000000000000\n\n')
  # print(x2)
  print('AAAAAAAAAAAAAAAAA\n\n')

  # # KPCA
  # kpca = KernelPCA(n_components=3, kernel='rbf', gamma=15)
  # x_kpca = kpca.fit_transform(X)
  # print(x_kpca)
  # figure(1)
  # figure1 = plt.figure(1)
  # ax = ax3(figure1)
  # ax.scatter3D(x_kpca[:, 0], x_kpca[:, 1], x_kpca[:, 2], alpha=0.7)
  # plt.show()
  # figure2 = plt.figure(1)
  # ax2 = ax3(figure2)
  # ax2.scatter3D(x1[:, 0], x1[:, 1], x1[:, 2], alpha=0.7)
  # plt.show()

  print('BBBBBBBBBBBBBBBBB\n\n')
  x, z= pcaa(X)
  XX = np.array(x)
  ZZ = np.array(z)
  print(x)
  print('CCCCCCCCCCCCCCCCC\n\n')
  print(z)
  print('DDDDDDDDDDDDDDDDD\n\n')

