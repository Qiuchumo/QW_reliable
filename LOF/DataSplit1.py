import numpy as np
import math
import pandas as pd
import theano
floatX = theano.config.floatX
from sklearn.preprocessing import scale
import random

elec_data = pd.read_csv('XZnozero_12.csv')

# 计算同一公司产品测试地点数目：
companies_num = elec_data.counts.unique()
companies = len(companies_num)  # companies=7， 共7个测试地点
company_lookup = dict(zip(companies_num, range(len(companies_num))))
company = elec_data['company_code'] = elec_data.counts.replace(company_lookup).values  # 加一行数据在XZsingal文件中

# 计算不同公司数目
company_ABC = elec_data.company.unique()
companiesABC = len(company_ABC)  # companies=7， 共7个测试地点
company_lookup_ABC = dict(zip(company_ABC, range(len(company_ABC))))
companyABC = elec_data['company_ABC'] = elec_data.company.replace(company_lookup_ABC).values  # 加一行数据在XZsingal文件中
# elec_count = elec_data.counts.values
SNR = np.random.normal(0, 1, size=[len(elec_data.Year.values), 3])

# #特征因素分析
elec_tem = elec_data.Tem.values # 观测温度值x2，温度不加噪声
elec_tem1 = (elec_tem - np.mean(elec_tem)) / np.std(elec_tem)
elec_hPa = elec_data.hPa.values + SNR[:, 0]  # 观测压强x3
elec_hPa1 = (elec_hPa - np.mean(elec_hPa)) / np.std(elec_hPa)
elec_RH = elec_data.RH.values + SNR[:, 1] # 观测压强x3
elec_RH1 = (elec_RH - np.mean(elec_RH)) / np.std(elec_RH)
elec_Lux = elec_data.Lux.values + SNR[:, 2] # 观测压强x3
elec_Lux1 = (elec_Lux - np.mean(elec_Lux)) / np.std(elec_Lux)

# 计算观测时间，温度，光照等环境条件
elec_year = elec_data.Year.values  # 观测时间值x1
elec_year1 = (elec_year - np.mean(elec_year)) / np.std(elec_year)

# 计算故障率大小：故障数目/总测量数，作为模型Y值，放大100倍以增加实际效果，结果中要缩小100倍
elec_faults = 1*(elec_data.Fault.values / elec_data.Nums.values)  # 数组形式,计算故障率大小
elec_faults_Max = np.max(elec_faults)
elec_faults = elec_faults/elec_faults_Max #最大值限定到[0,1]之间

X = scale(np.vstack((elec_year, elec_tem, elec_RH)).T)
# X = scale(elec_year) #scale用于归一化
X = X.astype(floatX)
# Y = scale(Y)
Y = elec_faults.astype(floatX)

Input_X = X
Input_Y = Y
# X_train, X_test, Y_train, Y_test

# def DataSplit(Input_X, Input_Y, test_size):
# test_size = 0.1
# if test_size > 0.7 or test_size <= .0:
#     test_size = 0.1
# print((Input_X)[:][2:23])
# def DataSplit(X, Y, test_size):
#     test_size=3
def DataSplit(Input_X, Input_Y, test_size):
    total=7
    li= range(total)
    print(li)
    NumSizeA = random.sample(li, test_size)
    NumSizeB = random.sample(li, test_size)
    NumSizeC = random.sample(li, test_size)
    print(NumSizeA)
    print(NumSizeB)
    print(NumSizeC)
    # NumSizeA = [random.randint(0, 6) for _ in range(test_size)]
    # NumSizeB = [random.randint(0, 6) for _ in range(test_size)]
    # NumSizeC = [random.randint(0, 6) for _ in range(test_size)]
    XA=X[0:84, ]
    XB=X[84:168,]
    XC=X[168:,]
    YA=Y[0:84]
    YB=Y[84:168]
    YC=Y[168:]
    xA = []
    yA=[]
    ii_a = 1
    for i in NumSizeA:
        A_number = range(12*i,12*(i+1))
        yA_number = range(12 * i, 12 * (i + 1))
        XA_Split = XA[A_number, ]
        YA_Split = YA[yA_number ]
        if ii_a== 1:
            xA  =  XA_Split
            yA  =  YA_Split
        else:
            xA = np.vstack((xA,  XA_Split))
            yA = np.vstack((yA, YA_Split))
        ii_a = ii_a + 1
    A_ret=[]
    for j in range(7):
        if j not in NumSizeA:
            A_ret.append(j)
    yAA=[]
    xAA=[]
    ii_A=1
    for i in A_ret:
        A_number = range(12*i,12*(i+1))
        yA_number = range(12*i, 12*(i + 1))
        A_text= XA[A_number, ]
        yA_text = YA[yA_number]
        if ii_A == 1:
            xAA = A_text
            yAA = yA_text
        else:
            xAA = np.vstack((xAA, A_text))
            yAA = np.concatenate((yAA, yA_text),axis=0)
        ii_A = ii_A + 1

    xB = []
    yB=[]
    ii_b = 1
    for i in NumSizeA:
        B_number = range(12*i,12*(i+1))
        yB_number = range(12 * i, 12 * (i + 1))
        XB_Split = XB[B_number, ]
        YB_Split = YB[yB_number, ]
        if ii_b == 1:
            xB  =  XB_Split
            yB =  YB_Split
        else:
            xB = np.vstack((xB,  XB_Split))
            yB = np.concatenate((yB, YB_Split),axis=0)
        ii_b = ii_b + 1
    B_ret=[]
    for j in range(7):
        if j not in NumSizeB:
            B_ret.append(j)
    yBB=[]
    xBB=[]
    ii_B=1
    for i in B_ret:
        B_number = range(12*i,12*(i+1))
        yB_number = range(12*i, 12*(i + 1))
        B_text = XB[B_number, ]
        yB_text = YB[yB_number]
        if ii_B == 1:
            xBB = B_text
            yBB = yB_text
        else:
            xBB = np.vstack((xBB ,B_text))
            yBB = np.concatenate((yBB ,yB_text),axis=0)
        ii_B = ii_B + 1

    xC = []
    yC=[]
    ii_c= 1
    for i in NumSizeC:
        C_number = range(12*i,12*(i+1))
        yC_number = range(12 * i, 12 * (i + 1))
        XC_Split = XC[C_number, ]
        YC_Split = YC[yC_number, ]
        if ii_c == 1:
            xC  =  XC_Split
            yC =  YC_Split
        else:
            xC = np.vstack((xC,  XC_Split))
            yC = np.concatenate((yC, YC_Split),axis=0)
        ii_c = ii_c + 1
    C_ret=[]
    for j in range(7):
        if j not in NumSizeC:
            C_ret.append(j)
    yCC=[]
    xCC=[]
    ii_C=1
    for i in C_ret:
        C_number = range(12*i,12*(i+1))
        yC_number = range(12*i, 12*(i + 1))
        C_text = XC[C_number, ]
        yC_text = YC[yC_number]
        # A.append(XA_Split)
        if ii_C == 1:
            xCC = C_text
            yCC = yC_text
        else:
            xCC = np.vstack((xCC , C_text))
            yC_text = np.concatenate((yCC ,yC_text),axis=0)
        ii_C = ii_C + 1
    # print(NumSizeA)
    # print(NumSizeB)
    # print(NumSizeC)
    # print(np.shape(xCC))
    return xA,yA,xAA,yAA,xB,yB,xBB,yBB,xC,yC,xCC,yCC
if __name__ == "__main__":
    xA,yA,xAA,yAA,xB,yB,xBB,yBB,xC,yC,xCC,yCC = DataSplit(X, Y, 6)
    # print(xA)
    # print(yA)
    # print(xAA)
    # print(yAA)
    # print(xB)
    # print(yB)
    # print(xBB)
    # print(yBB)
    # print(xC)
    # print(yC)
    # print(xCC)
    # print(yCC)

