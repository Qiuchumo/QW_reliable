import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
plt.style.use('default')

def Plot_XZ(elec_year, elec_faults, Savefig):
# 画出原始图
    Company_names = ['XiZang', 'XinJiang', 'Heilongjiang']
    k = np.array([0, 41, 83])
    j= 0
    # j, k1 = 0, 6
    plt.figure(figsize=(6, 4.5), facecolor='w')

    ax = plt.subplot(1, 1, 1)
    for jx in range(7):
        ax.plot(elec_year[jx], elec_faults[jx], 'ko--', markersize=4, linewidth=1)
        # j = j+k1
    ax.set_xticklabels(['2016', '2010', '2011', '2012', '2013', '2014', '2015'], fontsize='small')
    ax.set_xlabel("time/year", fontsize=14)
    plt.ylabel("Failure rate/%", fontsize=14)
    plt.legend([Company_names[0]], loc='upper left')
    plt.grid()
    if Savefig == 1:
        plt.savefig('1.svg', format='svg')
    plt.show()
    plt.figure(figsize=(6, 4.5), facecolor='w')

    ax = plt.subplot(1, 1, 1)
    for jx in range(7, 14, 1):
        ax.plot(elec_year[jx], elec_faults[jx], 'ko--', markersize=4, linewidth=1)
        # j = j+k1
    ax.set_xticklabels(['2016','2010', '2011', '2012', '2013', '2014', '2015'], fontsize='small')
    ax.set_xlabel("time/year", fontsize=14)
    plt.ylabel("Failure rate/%", fontsize=14)
    plt.legend([Company_names[1]], loc='upper left')
    plt.grid()
    if Savefig == 1:
        plt.savefig('2.svg', format='svg')
    plt.show()

    ax = plt.subplot(1, 1, 1)
    for jx in range(14, 21, 1):
        ax.plot(elec_year[jx], elec_faults[jx], 'ko--', markersize=4, linewidth=1)
        # j = j+k1
    ax.set_xticklabels(['2016','2010', '2011', '2012', '2013', '2014', '2015'], fontsize='small')
    ax.set_xlabel("time/year", fontsize=14)
    plt.ylabel("Failure rate/%", fontsize=14)
    plt.legend([Company_names[2]], loc='upper left')
    plt.grid()
    if Savefig == 1:
        plt.savefig('3.svg', format='svg')
    plt.show()
    return 0
