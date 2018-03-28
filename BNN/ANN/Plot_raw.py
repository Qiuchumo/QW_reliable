import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
plt.style.use('default')

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
    ax.set_xticklabels(['2016', '2010.3', '2010.9', '2011.3', '2011.9', '2012.3', '2012.9', '2013.3', '2013.9', '2014.3', '2014.9', '2015.3', '2015.9'], fontsize='small')
    ax.set_xlabel("time(year)", fontsize=11)
    plt.ylabel("Failure rate(%)", fontsize=11)
    plt.legend([Company_names[0]], loc='upper left', frameon=False, fontsize='small')
    # plt.grid()
    if Savefig == 1:
        plt.savefig('E:\\Code\\Bayescode\\QW_reliable\\BNN\\Picture\\New1.png', dpi = 200, bbox_inches='tight')
    plt.show()
    
    plt.figure(figsize=(3.5, 2.5), facecolor='w')
    ax = plt.subplot(1, 1, 1)
    for jx in range(7, 14, 1):
        ax.plot(elec_year[jx], elec_faults[jx], 'ko--', markersize=3, linewidth=1)
        # j = j+k1
    ax.set_xticklabels(['2016','2010.3', '2010.9', '2011.3', '2011.9', '2012.3', '2012.9', '2013.3', '2013.9', '2014.3', '2014.9', '2015.3', '2015.9'], fontsize='small')
    ax.set_xlabel("time(year)", fontsize=11)
    plt.ylabel("Failure rate(%)", fontsize=11)
    plt.legend([Company_names[1]], loc='upper left', frameon=False, fontsize='small')
    # plt.grid()
    if Savefig == 1:
        plt.savefig('E:\\Code\\Bayescode\\QW_reliable\\BNN\\Picture\\New2.png', dpi = 200, bbox_inches='tight')
    plt.show()
    
    plt.figure(figsize=(3.5, 2.5), facecolor='w')
    ax = plt.subplot(1, 1, 1)
    for jx in range(14, 21, 1):
        ax.plot(elec_year[jx], elec_faults[jx], 'ko--', markersize=3, linewidth=1)
        # j = j+k1
    ax.set_xticklabels(['2016','2010.3', '2010.9', '2011.3', '2011.9', '2012.3', '2012.9', '2013.3', '2013.9', '2014.3', '2014.9', '2015.3', '2015.9'], fontsize='small')
    ax.set_xlabel("time(year)", fontsize=11)
    plt.ylabel("Failure rate(%)", fontsize=11)
    plt.legend([Company_names[2]], loc='upper left', frameon=False, fontsize='small')
#     leg = plt.legend()
#     leg.get_frame().set_linewidth(0.0)
    
    # plt.grid()
    if Savefig == 1:
        plt.savefig('E:\\Code\\Bayescode\\QW_reliable\\BNN\\Picture\\New3.png', dpi = 200, bbox_inches='tight')
    plt.show()
    return 0
