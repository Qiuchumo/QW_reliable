import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
plt.style.use('default')
# 以下三行用于中文显示图形
from matplotlib.font_manager import FontProperties
# from pymc3 import get_data
font = FontProperties(fname=r"C:\\WINDOWS\\Fonts\\simsun.ttc", size=10)
np.set_printoptions(precision=0, suppress=True)

def Plot_raw(elec_year, elec_faults, Savefig):
# 画出原始图
    Company_names = ['Xizang', 'Xinjiang', 'Heilongjiang']
    k = np.array([0, 82, 166])
    j= 0
    # j, k1 = 0, 6
    plt.figure(figsize=(3.5, 2.5), facecolor='w')
    ax = plt.subplot(1, 1, 1)
    for jx in range(7):
        ax.plot(elec_year[jx], elec_faults[jx], 'ko--', markersize=3, linewidth=1)
        # j = j+k1
    ax.set_xlim(0.5, 12.5)
    ax.set_xticks(np.linspace(1,12,6)) 
    ax.set_xticklabels(['2010.3',  '2011.3', '2012.3', '2013.3',  '2014.3',  '2015.3'],rotation=45, fontsize='small')
    plt.yticks(fontsize='small')
    
    plt.xlabel(u"时间t/年", fontsize=14, fontproperties=font)
    plt.ylabel(u"故障率/%", fontsize=14, fontproperties=font)
    plt.legend([Company_names[0]], loc='upper left', frameon=False, fontsize='small', prop=font)
    # plt.grid()
    if Savefig == 1:
        plt.savefig('E:\\Code\\Bayescode\\QW_reliable\\EI\\Picture\\New1.png', dpi = 200, bbox_inches='tight')
    plt.show()
    
    plt.figure(figsize=(3.5, 2.5), facecolor='w')
    ax = plt.subplot(1, 1, 1)
    for jx in range(7, 14, 1):
        ax.plot(elec_year[jx], elec_faults[jx], 'ko--', markersize=3, linewidth=1)
        # j = j+k1
    ax.set_xlim(0.5, 12.5)
    ax.set_xticks(np.linspace(1,12,6)) 
    ax.set_xticklabels(['2010.3',  '2011.3', '2012.3', '2013.3',  '2014.3',  '2015.3'],rotation=45, fontsize='small')
    plt.yticks(fontsize='small')
    
    plt.xlabel(u"时间t/年", fontsize=14, fontproperties=font)
    plt.ylabel(u"故障率/%", fontsize=14, fontproperties=font)
    plt.legend([Company_names[1]], loc='upper left', frameon=False, fontsize='small', prop=font)
    # plt.grid()
    if Savefig == 1:
        plt.savefig('E:\\Code\\Bayescode\\QW_reliable\\EI\\Picture\\New2.png', dpi = 200, bbox_inches='tight')
    plt.show()
    
    plt.figure(figsize=(3.5, 2.5), facecolor='w')
    ax = plt.subplot(1, 1, 1)
    for jx in range(14, 21, 1):
        ax.plot(elec_year[jx], elec_faults[jx], 'ko--', markersize=3, linewidth=1)
        # j = j+k1
    ax.set_xlim(0.5, 12.5)
    ax.set_xticks(np.linspace(1,12,6)) 
    ax.set_xticklabels(['2010.3',  '2011.3', '2012.3', '2013.3',  '2014.3',  '2015.3'],rotation=45, fontsize='small')
    
    plt.xlabel(u"时间t/年", fontsize=14, fontproperties=font)
    plt.ylabel(u"故障率/%", fontsize=14, fontproperties=font)
    plt.legend([u'本文算法'], loc='upper left', frameon=False, fontsize='small', prop=font)

    ax.set_xlim(0.5, 12.5)
    ax.set_xticks(np.linspace(1,12,6)) 
    ax.set_xticklabels(['2010.3',  '2011.3', '2012.3', '2013.3',  '2014.3',  '2015.3'],rotation=45, fontsize='small')
    plt.yticks(fontsize='small')
    
    # plt.legend([ax1], [u'本文算法'], frameon=False, fontsize='small',loc='upper left', prop=font)

    if Savefig == 1:
        plt.savefig('E:\\Code\\Bayescode\\QW_reliable\\EI\\Picture\\New3.png', dpi = 200, bbox_inches='tight')
    plt.show()
    return 0
