import numpy as np
import pickle
from sklearn.cluster import AgglomerativeClustering
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)
import pandas as pd
import os
import math

def mean_confidence_interval(data):
    confidence = 0.95
    # https://www.mathsisfun.com/data/confidence-interval-calculator.html
    if confidence==0.95:
        z = 1.960

    a = 1.0 * np.array(data)
    n = len(a)
    a_mean = np.mean(a)
    a_std = np.std(a)

    CI_h = z * a_std / math.sqrt(n)
    # CI_low = a_mean - CI_h
    # CI_high = a_mean + CI_h
    return CI_h


data_path = '/Users/xuzhenxing/Documents/Sepsis/sepsis_manuscript/LDH/'
clusterResult_folder = '/Users/xuzhenxing/Documents/Sepsis/sepsis_manuscript/LDH/result/'

cluster_s = [0,1,2,3]

# plot SOFA score
plot_individual = 'average'

title = 'SOFA'
score_data = pd.read_csv(data_path+'SOFA_sample.csv',index_col=0)

fig = plt.figure()
ax = fig.add_subplot(111)
# colors = ['red','blue','green','grey','purple'] # ,'purple'
colors = ['purple','darkgreen','blue','red'] # ,'purple'

pic_name_s = ['Subphenotype_0','Subphenotype_1','Subphenotype_2','Subphenotype_3'] # ,'cluster_4'
# pic_name_s = ['DI','RI','DW','RW']  # ['cluster_0','cluster_1','cluster_2','cluster_3']
# DI: Delayed Improving. RI: Rapidly Improving. DW: Delayed Worsening. RW: Rapidly Worsening.

for k in cluster_s:
    with open(clusterResult_folder + 'cluster_id_'+str(k)+'.pkl','rb') as f:  #
        cluster_id = pickle.load(f) # obtain ICUstay_ID in each cluster
        # pic_name = 'cluster_0'
        pic_name = pic_name_s[k]
        clusterScore = score_data.loc[score_data.index.isin(cluster_id)]
        print('ss',len(clusterScore))

        average_cluster_score = np.array(list(clusterScore.mean(axis = 0,skipna = True))) # compute mean
        # median_cluster_score = np.array(list(clusterScore.median(axis=0, skipna=True)))  # compute median
        # average_cluster_score = median_cluster_score

        std_cluster_score = np.array(list(clusterScore.std(axis = 0,skipna = True))) # compute std
        print('cluster:',k)
        print('average_cluster_score:',average_cluster_score)
        print('std_cluster_score',std_cluster_score)
        duration = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72]

        # plot
        ax.plot(range(len(duration)), average_cluster_score, color=colors[k], linewidth=1, label=pic_name + ' (N='+str(len(cluster_id))+')')

        # add confidence interval
        ci = 1.96 * np.std(average_cluster_score) / math.sqrt(len(average_cluster_score)) # math.sqrt(len(average_cluster_score))#
        plt.fill_between(range(len(duration)), average_cluster_score-ci, average_cluster_score+ci, alpha=0.5, color=colors[k]) # plot with confidence interval

ylim_min = 1.0
ylim_max = 8.01
step = 1
ax.set(title='Samples Cohort', ylim=[ylim_min, ylim_max], ylabel=title, xlabel='Time After ICU Admission (hours)')

plt.xlim(0, 11)
ax.set_xticks(range(len(duration)))
ax.set_xticklabels(duration)
plt.yticks(np.arange(ylim_min, ylim_max, step=step))
plt.legend(fontsize='medium', loc='lower left') # large,medium

plt.show()
fig_name = clusterResult_folder + title +'trajectory.pdf'

fig.savefig(fig_name)
plt.close('all')



