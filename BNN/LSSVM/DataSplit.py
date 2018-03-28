import numpy as np
import math
import theano
import random

# ====================================================================================================
#   DataSplit(Input_X, Input_Y, test_size, GroupSize_Init=7, SubSize=12):
#   用于将三个省电能表数据以组为依据(12行为一组)分为测试集与训练集
#   Input_X：输入特征数据，多维或单维
#   Input_Y：特征数据对应输出，单维
#   test_size：人为设置的测试集在原始数据Input_X比率，输入大小有限：(0, 0.7], 建议输入小于0.4
#   函数返回值：X_test， Y_test，X_train， Y_train
#
#   GroupSize_Init = 7  # 每个省的初始数据组数为 7，后期可能更改
#   SubSize = 12        # 每个省中每组的数据数为 12，后期可能更改
#   邱楚陌，2018.3.26。编程思想为：随机设置每组不同整数--随机数排序--赋值给X_test--删除后变为(X_train)
# =====================================================================================================
def DataSplit(Input_X, Input_Y, test_size, GroupSize_Init=7, SubSize=12):
    # 参数定义部分
    # X_test， Y_test，X_train， Y_train将数据分开后存储，含义如符号如何
    X_test = {}
    Y_test = {}
    X_train = Input_X # 赋值后便于删除
    Y_train = Input_Y

    # NumSize_A/B/C：随机选中A/B/C省需要删除的数据组的序号(12行为一组)
    NumSize_A = []; NumSize_B = []; NumSize_C = []
    # 如果设置的测试集太大或为负数，则直接判定为0.1比率
    if test_size > 0.7 or test_size <= .0:
        test_size = 0.1

    # 随机生成‘不同’的 需要删除的数据组编号(12行为一组), test_size*GroupSize_Init为每组输出抽出来的数据数量
    LenNum = math.ceil(test_size*GroupSize_Init) # math.ceil向上取整数
    while(len(NumSize_A) < LenNum):
        random_A = random.randint(0, GroupSize_Init-1) # GroupSize_Init-1:防止取到边界的数字
        if random_A not in NumSize_A:                  # 为了取不同随机数
            NumSize_A.append(random_A)
    while (len(NumSize_B) < LenNum):
        random_B = random.randint(GroupSize_Init, 2*GroupSize_Init-1)
        if random_B not in NumSize_B:
            NumSize_B.append(random_B)
    while (len(NumSize_C) < LenNum):
        random_C = random.randint(2*GroupSize_Init, 3*GroupSize_Init-1)
        if random_C not in NumSize_C:
            NumSize_C.append(random_C)

    # 分别将随机数排序排列好
    NumSize_A = np.sort(NumSize_A)
    NumSize_B = np.sort(NumSize_B)
    NumSize_C = np.sort(NumSize_C)
    NumSize = np.hstack((NumSize_A, NumSize_B, NumSize_C)) # 将三者结合，并排序好
    NumSize_Inv = sorted(NumSize, reverse=True)           # 取逆排序，用于删除数据
    # print(NumSize)
    # print(NumSize_Inv)

    # 按照顺序添加数组中数据，用于测试，由于空数组无法连接，故需要赋初值后再循环
    X_test = Input_X[SubSize * NumSize[0]: (SubSize * NumSize[0] + SubSize), ]
    Y_test = Input_Y[SubSize * NumSize[0]: (SubSize * NumSize[0] + SubSize)]
    for i in NumSize[1:]:
        TempX = Input_X[SubSize * i: (SubSize * i + SubSize), ]
        X_test = np.concatenate((X_test, TempX), axis=0)    # 多维：连接数组X_test
        # 保存Y的数据用于测试， SubSize=12
        TempY = Input_Y[SubSize*i: (SubSize * i + SubSize)] # 一维：：连接数组Y_test
        Y_test = np.concatenate((Y_test, TempY), axis=0)

    # 按照倒序删除数组中数据，用于训练
    for i in NumSize_Inv:
        aa = sorted(np.arange(SubSize * i, (SubSize * i + SubSize)), reverse=True) # 按照倒序排列需要删除的数组序号
        for ii in aa:
            Y_train = np.delete(Y_train, ii, ) # 删除数据
            X_train = np.delete(X_train, ii, axis=0)

    return X_train, X_test, Y_train, Y_test

# if __name__ == "__main__":
#     X_train, X_test, Y_train, Y_test = DataSplit(X, Y, 0.2)
#
#     print(np.shape(X_test))
#     print(len(Y_test))
#     print(np.shape(Y_train))
#     print(np.shape(X_train))
#     print(X_test)
